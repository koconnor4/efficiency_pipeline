import sys
import glob
import os
from optparse import OptionParser
parser = OptionParser()
(options,args)=parser.parse_args()

import copy
import pickle
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Circle
from matplotlib.colors import BoundaryNorm
import numpy as np
import itertools
import collections 
from scipy.optimize import curve_fit

import astropy
from astropy.io import ascii,fits
from astropy.table import vstack,Table,Column,Row,setdiff,join
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.units import Quantity
from astroquery.gaia import Gaia
from astropy.convolution import Gaussian2DKernel
from astropy.visualization import ZScaleInterval,simple_norm
zscale = ZScaleInterval()
from astropy.nddata import Cutout2D,NDData
from astropy.stats import sigma_clipped_stats,gaussian_fwhm_to_sigma,gaussian_sigma_to_fwhm
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel

import photutils
from photutils import find_peaks
from photutils.psf import extract_stars
from photutils import EPSFBuilder
from photutils import detect_threshold
from photutils import detect_sources
from photutils import deblend_sources
from photutils import source_properties, EllipticalAperture
from photutils import BoundingBox
from photutils import Background2D, MedianBackground

from .lco_figures import *

# Suppress warnings. Relevant for astroquery. Comment this out if you wish to see the warning messages
import warnings
warnings.filterwarnings('ignore')

def get_data(path):
    my_data = {} 
    """
    # shouldn't be any images directly in the folder path to field this is used for
    # should be tucked in dirs like dia_out, dia_trim, source_im
    ims=glob.glob('*fits')
    for i in range(len(ims)):
        filename = ims[i].split('/')[-1]
        my_data[filename] = fits.open(ims[i])[0]
    """
    # this source im is a dir that you should make for each field with the image you want to use
    # ie will be the one that psf measured on and planting into
    SOURCE_IM=glob.glob(os.path.join(path,'source_im/*fits'))
    for i in range(len(SOURCE_IM)):
        filename=SOURCE_IM[i].split('/')[-1]
        my_data[filename] = fits.open(SOURCE_IM[i])[0]
    DIA_OUT=glob.glob(os.path.join(path,'dia_out/*'))
    for i in range(len(DIA_OUT)):
        filename = DIA_OUT[i].split('/')[-1]
        my_data[filename] = fits.open(DIA_OUT[i])[0]
    DIA_TRIM=glob.glob(os.path.join(path,'dia_trim/*'))
    for i in range(len(DIA_TRIM)):
        filename = DIA_TRIM[i].split('/')[-1]
        my_data[filename] = fits.open(DIA_TRIM[i])[0]
    #print(my_data)
    return my_data

def gaia_results(image):
    # image wcs and frame, for conversions pixels/skycoord
    wcs,frame=WCS(image.header),image.header['RADESYS'].lower()
    # coord of strong lensing galaxy
    ra=image.header['CAT-RA']
    dec=image.header['CAT-DEC']
    coord = SkyCoord(ra,dec,unit=(u.hourangle,u.deg))

    # the pixscale is same along x&y and rotated to default (N up, E left) cdi_j ~ delta_ij
    cdi_i = image.header['CD1_1'] # deg/pixel
    naxis = image.header['NAXIS1'] # naxis1=naxis2
    radius = 3600*cdi_i*naxis/2 # approx 800 arcsec entire image
    radius*=.75 # do 3/4 of that
    # do the search
    r = Gaia.query_object_async(coordinate=coord, radius=radius*u.arcsec)
    return r,image

def stars(results,Nbrightest=None):
    # stars are extracted from image to be ready for use in determine ePSF
    # note ref.fits doesn't have saturate and maxlin available the image should be just one of the trims

    # unpack gaia_results into the gaia catalog and image
    r,image = results
    # need to give the results a column name x and y (pixel locations on image) for extract stars fcn which am going to apply
    wcs,frame=WCS(image.header),image.header['RADESYS'].lower()
    positions,pixels=[],[]
    for i in range(len(r)):
        position=SkyCoord(ra=r[i]['ra'],dec=r[i]['dec'],unit=u.deg,frame=frame)
        positions.append(position)
        pixel=skycoord_to_pixel(position,wcs)
        pixels.append(pixel)
    x,y=[i[0] for i in pixels],[i[1] for i in pixels]
    x,y=Column(x),Column(y)
    r.add_column(x,name='x')
    r.add_column(y,name='y')
    print('there are {} stars available within fov from gaia results queried'.format(len(r)))

    # I am finding bboxes of the extractions I will do so I can remove any stars with overlaps 
    # I want to extract all the stars wo overlaps before I start to remove any using photometry constraints to get 'good' ones for psf
    bboxes = []
    for i in r:
        x = i['x']
        y = i['y']
        size = 25
        ixmin,ixmax = int(x - size/2), int(x + size/2)
        iymin, iymax = int(y - size/2), int(y + size/2)
        
        bbox = BoundingBox(ixmin=ixmin, ixmax=ixmax, iymin=iymin, iymax=iymax)
        bboxes.append(bbox)
    bboxes = Column(bboxes)
    r.add_column(bboxes,name='bbox')
    # using the bbox of each star from results to determine intersections, dont want confusion of multi-stars for ePSF
    intersections = []
    for i,obj1 in enumerate(bboxes):
        for j in range(i+1,len(bboxes)):
            obj2 = bboxes[j]
            if obj1.intersection(obj2):
                #print(obj1,obj2)
                # these are the ones to remove 
                intersections.append(obj1) 
                intersections.append(obj2)
    # use the intersections found to remove stars
    j=0
    rows=[]
    for i in r:
        if i['bbox'] in intersections:
            #tmp.remove(i)
            row=j
            rows.append(row)
        j+=1
    r.remove_rows(rows)
    print('{} stars, after removing intersections'.format(len(r)))

    # I am going to extract stars with strong signal in rp filter (the one lco is looking in)
    r = r[r['phot_rp_mean_flux_over_error']>100]
    print('restricting extractions to stars w rp flux/error > 100 we have {} to consider'.format(len(r)))

    # sort by the strongest signal/noise in r' filter
    # r.sort('phot_rp_mean_flux_over_error')
    """
    # don't think it will be necessary to limit to some N stars, might as well take all that will give good data for building psf
    if Nbrightest == None:
        Nbrightest = len(r)
    brightest_results = r[:Nbrightest]
    """

    data = image.data
    hdr = image.header
    # the header has L1 bkg values; should be the same as sigma clipped stats 
    L1mean,L1med,L1sigma,L1fwhm = hdr['L1MEAN'],hdr['L1MEDIAN'],hdr['L1SIGMA'],hdr['L1FWHM'] # counts, fwhm in arcsec 
    mean_val, median_val, std_val = sigma_clipped_stats(data, sigma=2.)  
    WMSSKYBR = hdr['WMSSKYBR'] # mag/arcsec^2 of sky bkg measured
    # AGGMAG the guide star magnitude header value would be simpler but it is given as unknown, ra/dec are provided for it though
    # grab some other useful header values now
    pixscale,saturate,maxlin = hdr['PIXSCALE'],hdr['SATURATE'],hdr['MAXLIN'] # arcsec/pixel, counts for saturation and non-linearity levels
    
    # need bkg subtracted to extract stars, want to build ePSF using just star brightness 
    data -= median_val # L1med
    nddata = NDData(data=data)
    stars = extract_stars(nddata,catalogs=r, size=25)
    # using the bbox of each star from results to determine intersections, dont want confusion of multi-stars for ePSF
    # this was done with all stars not just those extracted, this is an optional sanity check but don't need it
    intersections = []
    for i,obj1 in enumerate(stars.bbox):
        for j in range(i+1,len(stars.bbox)):
            obj2 = stars.bbox[j]
            if obj1.intersection(obj2):
                #print(obj1,obj2)
                # these are the ones to remove 
                intersections.append(obj1) 
                intersections.append(obj2)
    # use the intersections found to remove stars
    tmp = [i for i in stars] # get a list of stars rather than single photutils obj with all of them 
    for i in tmp:
        if i.bbox in intersections:
            tmp.remove(i)
    #print('{} stars, after removing intersections'.format(len(tmp)))
    

    # note ref.fits doesn't have saturate and maxlin available the image should be just one of the trims
    for i in tmp:
        if np.max(i.data) > saturate:
            tmp.remove(i)
        elif np.max(i.data) > maxlin:
            tmp.remove(i)

    print('removed stars above saturation or non-linearity level ~ {}, {} ADU; now have {}'.format(saturate,maxlin,len(tmp)))
    good_stars = photutils.psf.EPSFStars(tmp)
    
    """
    # you should look at the images to make sure these are good stars
    nrows = 4
    ncols = 4
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20),
                            squeeze=True)
    ax = ax.ravel()
    for i in range(len(brightest_results)):
        norm = simple_norm(stars[i], 'log', percent=99.)
        ax[i].imshow(stars[i], norm=norm, origin='lower', cmap='viridis')
    """
    #return stars
    return good_stars,image

def ePSF(stars,name='psf.fits',oversampling=2):
    # using all the available Gaia results which are below non-linearity/saturation to build an effective PSF 

    # unpack the stars results into the good_stars and image 
    good_stars, image = stars

    hdr = image.header
    L1mean,L1med,L1sigma,L1fwhm = hdr['L1MEAN'],hdr['L1MEDIAN'],hdr['L1SIGMA'],hdr['L1FWHM'] # counts, fwhm in arcsec 
    pixscale,saturate,maxlin = hdr['PIXSCALE'],hdr['SATURATE'],hdr['MAXLIN'] # arcsec/pixel, counts for saturation and non-linearity levels

    # oversampling chops pixels of each star up further to get better fit
    # this is okay since stacking multiple ...
    # however more oversampled the ePSF is, the more stars you need to get smooth result
    # LCO is already oversampling the PSFs, the fwhm ~ 2 arcsec while pixscale ~ 0.4 arcsec; should be able to get good ePSF measurement without any oversampling
    # ePSF basic x,y,sigma 3 param model should be easily obtained if consider that 3*pixscale < fwhm
    epsf_builder = EPSFBuilder(oversampling=oversampling, maxiters=10,
                                progress_bar=True)  
    epsf, fitted_stars = epsf_builder(good_stars)  
    """
    # take a look at the ePSF image 
    norm = simple_norm(epsf.data, 'log', percent=99.)
    plt.imshow(epsf.data, norm=norm, origin='lower', cmap='viridis')
    plt.colorbar()
    """
    #fits.writeto(name,epsf.data,hdr,overwrite=True)
    #         fits.writeto(plantname,image.data,hdr,overwrite=True)

    return epsf, fitted_stars
from photutils.datasets import make_gaussian_sources_image

def gaussian2d(epsf,hdr=None):
    # use photutils 2d gaussian fit on the epsf
    gaussian = photutils.centroids.fit_2dgaussian(epsf.data)
    print('gaussian fit to epsf:')
    print(gaussian.param_names,gaussian.parameters)
    # unpack the parameters of fit
    constant,amplitude,x_mean,y_mean,x_stddev,y_stddev,theta=gaussian.parameters
    
    # going to print approx resolution determined by epsf (according to average of the fits along x,y)
    # we have some rough idea of what the fwhm is predicted so is good check
    sigma = (abs(x_stddev)+abs(y_stddev))/2 # the average of x and y 
    sigma*=1/epsf.oversampling[0] # the model fit was to oversampled pixels need to correct for that for true image pix res
    fwhm = gaussian_sigma_to_fwhm*sigma
    print('gaussian fwhm ~ {} pixels (an avg of the fit sigma_x sigma_y w sigma_to_fwhm)'.format(fwhm))
    if hdr:
        L1fwhm,pixscale = hdr['L1fwhm'],hdr['pixscale']
        print('frame L1fwhm stated in the hdr ~ {} arcsec, pixscale {} arcsec/pixel --> fwhm ~ {} pixel'.format(L1fwhm,pixscale,L1fwhm/pixscale))
        print('hopefully we are near consistent as sanity check')
    
    # here I take values of evaluated model fit along center of image
    # these might be useful to show
    xctr_vals = []
    y=0
    for i in range(epsf.shape[1]):
        gaussval = gaussian.evaluate(x_mean,y,constant,amplitude,x_mean,y_mean,x_stddev,y_stddev,theta)
        xctr_vals.append(gaussval)
        y+=1
    yctr_vals = []
    x=0
    for i in range(epsf.shape[0]):
        gaussval = gaussian.evaluate(x,y_mean,constant,amplitude,x_mean,y_mean,x_stddev,y_stddev,theta)
        yctr_vals.append(gaussval)
        x+=1
    
    # here I am using the stddev in epsf to define levels n*sigma below the amplitude of the fit
    # is useful for contours
    #np.mean(psf.data),np.max(psf.data),np.min(psf.data),med=np.median(psf.data)
    std=np.std(epsf.data)
    levels=[amplitude-3*std,amplitude-2*std,amplitude-std]
    #plt.contour(psf.data,levels=levels)
    
    table = Table()
    table['constant'] = [constant]
    table['amplitude'] = [amplitude]
    table['x_mean'] = [x_mean]
    table['y_mean'] = [y_mean]
    table['x_stddev'] = [x_stddev]
    table['y_stddev'] = [y_stddev]
    table['theta'] = np.radians(np.array([theta]))
    
    shape=epsf.shape
    # making numpy array of model values in shape of epsf
    image1 = make_gaussian_sources_image(shape, table)
    # turning model array into epsf obj for easy manipulation with the epsf
    img_epsf = photutils.psf.EPSFStar(image1.data,cutout_center=(x_mean,y_mean))
    # for example the residual of gaussian model with the epsf...
    resid = img_epsf.compute_residual_image(epsf)
    # idk what's happening with compute_residual_image but it isn't straight-forward subtraction of img_epsf - epsf
    # some parameter about the scale for registered epsf is being used, where it assumes img_epsf is a star, I really just can use a straight sub of the gauss model fit - epsf
    resid = img_epsf.data - epsf.data 
    return gaussian,levels,xctr_vals,yctr_vals,image1,img_epsf,resid
    #return image1
    """
    # we don't want noise for our planted psf or gaussian model, in general though could add like below
    image2 = image1 + make_noise_image(shape, distribution='gaussian',
                                   mean=5., stddev=5.)
    image3 = image1 + make_noise_image(shape, distribution='poisson',
                                   mean=5.)
    """

def source_cat(image,nsigma=2,kernel_size=(3,3),npixels=5,deblend=False,contrast=.001,targ_coord=None):
    """
    the image should be fits.open('trim.fits'), is trimmed/aligned properly w reference/differences
    for some reason reference doesn't have the catalog ra of the target strong lensing galaxy in header
    will get a cat of properties for detected sources
    """
    # to be able to translate from ra/dec <--> pixels on image
    wcs,frame = WCS(image.header),image.header['RADESYS'].lower()
    hdr = image.header
    #L1mean,L1med,L1sigma,L1fwhm = hdr['L1MEAN'],hdr['L1MEDIAN'],hdr['L1SIGMA'],hdr['L1FWHM'] # counts, fwhm in arcsec 
    #pixscale,saturate,maxlin = hdr['PIXSCALE'],hdr['SATURATE'],hdr['MAXLIN'] # arcsec/pixel, counts for saturation and non-linearity levels

    # detect threshold uses sigma clipped statistics to get bkg flux and set a threshold for detected sources as objs above nsigma*bkg
    # bkg also available in the hdr of file, either way is fine  
    threshold = detect_threshold(image.data, nsigma=nsigma)
    sigma = 3.0 * gaussian_fwhm_to_sigma  # FWHM = 3. pixels for kernel smoothing
    # optional ~ kernel smooths the image, using gaussian weighting with pixel size of 3
    kernel = Gaussian2DKernel(sigma, x_size=kernel_size[0], y_size=kernel_size[1])
    kernel.normalize()
    # make a segmentation map, id sources defined as n connected pixels above threshold (n*sigma + bkg)
    segm = detect_sources(image.data,
                          threshold, npixels=npixels, filter_kernel=kernel)
    # deblend useful for very crowded image with many overlapping objects...
    # uses multi-level threshold and watershed segmentation to sep local peaks as ind obj
    # use the same number of pixels and filter as was used on original segmentation
    # contrast is fraction of source flux local pk has to be consider its own obj
    if deblend:
        segm = deblend_sources(image.data, 
                                       segm, npixels=5,filter_kernel=kernel, 
                                       nlevels=32,contrast=contrast)

    # need bkg subtracted to do photometry using source properties
    boxsize=100
    bkg = Background2D(image.data,boxsize) # sigma-clip stats for background est over image on boxsize, regions interpolated to give final map 
    data_bkgsub = image.data - bkg.background
    cat = source_properties(data_bkgsub, segm,background=bkg.background,
                            error=None,filter_kernel=kernel)
    
    # going to id the target lensing galaxy from source catalog
    # since this is ideal detection location where strong lens could provide multi-im
    # this is going to be area where we will most want to plant and study 
    
    #CAT-RA  = 'blah'       / [HH:MM:SS.sss] Catalog RA of the object        
    #CAT-DEC = 'blah'       / [sDD:MM:SS.ss] Catalog Dec of the object
    if targ_coord == None:
        # the source images all have cat-ra cat-dec, will default grab target galaxy location from hdr
        ra = image.header['CAT-RA']
        dec = image.header['CAT-DEC']
    else:
        # if using the ref to detect source objs the target stuff isn't in there will need to provide tuple taken from source hdr 
        ra,dec = targ_coord # unpack

    lensing_gal = SkyCoord(ra,dec,unit=(u.hourangle,u.deg))
    pix_gal = astropy.wcs.utils.skycoord_to_pixel(lensing_gal,wcs)

    # TODO all sources of error including poisson from sources
    tbl = cat.to_table()
    tbl['xcentroid'].info.format = '.2f'  # optional format
    tbl['ycentroid'].info.format = '.2f'
    tbl['cxx'].info.format = '.2f'
    tbl['cxy'].info.format = '.2f'
    tbl['cyy'].info.format = '.2f'
    tbl['gini'].info.format = '.2f'

    # going to add a column of surface brightness so we can plant into the obj shapes according to those
    surf_brightnesses = []
    for obj in tbl:
        unit = 1/obj['area'].unit
        surf_bright = obj['source_sum']/obj['area'].value # flux/pix^2
        surf_brightnesses.append(surf_bright) 
    surf_brightnesses = Column(surf_brightnesses,name='surface_brightness',unit=unit)
    tbl.add_column(surf_brightnesses)

    # take a look at the brightest or most elliptical objs from phot on segm objs detected
    tbl.sort('ellipticity') #
    elliptical=tbl[-10:]
    #tbl.sort('source_sum') ('surface_brightness') 

    # there is definitely a neater/cuter way to index table than this using loc to find obj of gal 
    tmp = tbl[tbl['xcentroid'].value > pix_gal[0]-10]
    tmp = tmp[tmp['xcentroid'].value < pix_gal[0]+10]
    tmp = tmp[tmp['ycentroid'].value > pix_gal[1]-10]
    targ_obj = tmp[tmp['ycentroid'].value < pix_gal[1]+10] 
    targ_sb = targ_obj['source_sum']/targ_obj['area']
    
    return cat,image,threshold,segm,targ_obj

def target(image,targ_obj,ref=None,diff=None):
    if len(targ_obj) == 0:
        # the target isnt detected in image by my source_cat (photutils)...likely bad skymag bkg
        # going to use ref.fits to cut around the target and extract the parameters from these
        print('the target obj wasnt detected in source image, using ref image to get the target photutil params')
        #ref = my_data['ref.fits']
        wcs,frame = WCS(image.header),image.header['RADESYS'].lower()
        # the target strong lensing galaxy position
        ra=image.header['CAT-RA']
        dec=image.header['CAT-DEC']
        targ_coord = (ra,dec)
        # not cutting the ref on target to save computation time, get hdr error in source cat if do
        #coord = SkyCoord(ra,dec,unit=(u.hourangle,u.deg))        
        #pix=astropy.wcs.utils.skycoord_to_pixel(coord,wcs) # x,y pixel location
        #cut_ref = Cutout2D(ref.data,pix,25) # is 25 pixels big enough for any of the strong lens objects but small enough to avoid other objs?
        # photutils source properties to detect objs in image
        #cut_ref_catalog = source_cat(cut_ref)
        #cut_ref_cat,cut_ref_image,threshold,segm,targ_obj = source_catalog # unpacked to make a little clearer
        ref_catalog = source_cat(ref,targ_coord=targ_coord)
        ref_cat,ref_image,threshold,segm,targ_obj = ref_catalog # unpacked to make a little clearer
        # take useful photutil params for strong lensing galaxy target 
        # pixels and deg, sums ~ brightness in adu ~ for lco is straight counts (ie not yet rate isn't /exptime)
        equivalent_radius = targ_obj['equivalent_radius'][0].value
        xy = (targ_obj['xcentroid'][0].value,targ_obj['ycentroid'][0].value) 
        semimajor_axis, semiminor_axis = targ_obj['semimajor_axis_sigma'][0].value,targ_obj['semiminor_axis_sigma'][0].value
        orientation = targ_obj['orientation'][0].value 
    else:
        # the source image detected the targ_obj so take useful values already available (no need to get ref involved)
        # pixels and deg, sums ~ brightness in adu ~ for lco is straight counts (ie not yet rate isn't /exptime)
        equivalent_radius = targ_obj['equivalent_radius'][0].value
        xy = (targ_obj['xcentroid'][0].value,targ_obj['ycentroid'][0].value) 
        semimajor_axis, semiminor_axis = targ_obj['semimajor_axis_sigma'][0].value,targ_obj['semiminor_axis_sigma'][0].value
        orientation = targ_obj['orientation'][0].value 
    
    # cut around the image on target
    cut_targ = Cutout2D(image.data,xy,equivalent_radius*5)
    cuts = [cut_targ]
    if diff:
        cut_diff = Cutout2D(diff.data,xy,equivalent_radius*5)
        cuts.append(cut_diff)
    if ref:
        cut_ref = Cutout2D(ref.data,xy,equivalent_radius*5)
        cuts.append(cut_ref)

    # now going to grab (cutouts/patches) of boxes on galaxy 
    cut_xy = cut_targ.center_cutout
    shift_x = equivalent_radius*np.cos(orientation*np.pi/180)
    shift_y = equivalent_radius*np.sin(orientation*np.pi/180)

    # lets do a box on the ctr with length=width=radius 
    # the patch anchors on sw so shift the cut_xy 
    anchor_core = (cut_xy[0] - equivalent_radius/2, cut_xy[1] - equivalent_radius/2)
    # the patch (show in figures)
    box_core = matplotlib.patches.Rectangle(anchor_core,equivalent_radius,equivalent_radius,fill=None)
    # the cut (does sum for bkg)
    xy_core = xy # the center of galaxy in image
    cut_core = Cutout2D(image.data,xy_core,equivalent_radius)
    
    # shift box an equivalent radius along orientation from photutils creating next box 
    # assuming orientation ccw from x (east)
    # yes the boxes will overlap slightly unless orientation fully along x or y
    shift_x = equivalent_radius*np.cos(orientation*np.pi/180)
    shift_y = equivalent_radius*np.sin(orientation*np.pi/180)
    anchor_1 = (anchor_core[0]+shift_x,anchor_core[1]+shift_y)
    box_1 = matplotlib.patches.Rectangle(anchor_1,equivalent_radius,equivalent_radius,fill=None)
    # the cut (does sum for bkg)
    xy_1 = (xy[0]+shift_y,xy[1]+shift_y) 
    cut_1 = Cutout2D(image.data,xy_1,equivalent_radius)
    
    # similar shift one more time 
    anchor_2 = (anchor_core[0]+2*shift_x,anchor_core[1]+2*shift_y)
    box_2 = matplotlib.patches.Rectangle(anchor_2,equivalent_radius,equivalent_radius,fill=None)
    # the cut (does sum for bkg)
    xy_2 = (xy[0]+2*shift_y,xy[1]+2*shift_y) 
    cut_2 = Cutout2D(image.data,xy_2,equivalent_radius)
    
    bkg_core,bkg_1,bkg_2 = (cut_core,box_core),(cut_1,box_1),(cut_2,box_2)
    
    if diff or ref:
        # default diff None and ref None but if provided will return list of cuts order like [source,diff,ref]
        return targ_obj,cuts,bkg_core,bkg_1,bkg_2
    else:
        return targ_obj,cut_targ,bkg_core,bkg_1,bkg_2

def plant(image,psf,source_cat,hdr=None,mag=None,location=None,zp=None,plantname='planted.fits'):
    """
    the image should be the fits.open('difference.fits'), will add SN directly to here
    psf should be the epsf_builder(stars), ie a previous step is needed to check that have epsf which looks good
    source_cat is the catalog,targ_obj (strong lens galaxy), and segmentation image from photutils on image
    location should be a Skycoord(ra,dec) or if left as None will use the targ_obj strong lensing gal to place
    mag,zp (TODO need to know how to get proper zp so that scaling ePSF to correct mag; ePSF just means flux=1)
    """
    # unpack the source_cat so can use the targ_obj to place SN later if not given a location explicitly
    cat,orig_image,threshold,segm,targ_obj=source_cat # orig_image ~ meaning that pointing which source cat was run on not a diff
    
    # if image is the diff, (or any of the cutouts) none of these are available, provide it explicitly from source hdr
    skymag=hdr['SKYMAG'] # computed [mag/arcsec^2]
    skybr=hdr['WMSSKYBR'] # meas
    pixscale=hdr['PIXSCALE'] # arcsec/pixel
    mean=hdr['L1MEAN'] # sigma-clipped mean bkg [counts]
    med=hdr['L1MEDIAN']
    sig=hdr['L1SIGMA']
    fwhm=hdr['L1FWHM']
    exptime=hdr['EXPTIME']

    #print('L1: mean,exptime,pixscale,fwhm',mean,exptime,pixscale,fwhm)
    # the sigma-clipped stats I think should be same as L1 
    mean_val, median_val, std_val = sigma_clipped_stats(image.data, sigma=2.)  
    #print('scs: mean_val,median_val,std_val',mean_val,median_val,std_val)
    if zp==None:
        # there should be an L1ZP, since there isn't I'm doing what I think makes sense to calculate it
        # I know it should be zp ~ 23, so hopefully that is about what we get
        # /exptime ~ data is in counts but skymag measured seems to be doing count rate, /pixscale since want /arcsec
        # ... I thought /pixscale^2 was correct but single pixscale is closer to 'expected': https://arxiv.org/pdf/1805.12220.pdf
        zp=skybr+2.5*np.log10(med/exptime/pixscale)
        # these are in the sdss system (r' filter is what is used in our observations) https://www.sdss.org/dr14/algorithms/fluxcal/
        # effectively same as AB system, u zp is off by .04 mag, z zp might be off by .02 mag but close enough for govt work
    
    if mag==None:
        # if don't tell it what mag we want SN, I'll make it 5 mags brighter than bkg sky
        mag = skybr-5
    
    # copying image and psf so can leave original data untouched
    cpim = copy.copy(image.data)
    mu = 10**((mag-zp)/-2.5)*exptime*pixscale # the factor to multiply psf flux, to achieve given mag 
    #print('mu ',mu)
    cppsf = copy.copy(psf.data*mu) 
    
    wcs,frame = WCS(hdr),hdr['RADESYS'].lower()
    lattice,boxes = False,False 
    if location==None:
        # use the targ obj to place SN
        x = [targ_obj['xcentroid']-targ_obj['equivalent_radius'],targ_obj['xcentroid']+targ_obj['equivalent_radius']]
        x = [i[0].value for i in x]
        x = np.linspace(x[0],x[1],100)
        x = np.random.choice(x)
        y = [targ_obj['ycentroid']-targ_obj['equivalent_radius'],targ_obj['ycentroid']+targ_obj['equivalent_radius']]
        y = [i[0].value for i in y]
        y = np.linspace(y[0],y[1],100)
        y = np.random.choice(y)
        pix = [x,y]
        revpix = copy.copy(pix)
        revpix.reverse()
        location=astropy.wcs.utils.pixel_to_skycoord(pix[0],pix[1],wcs)
    elif type(location)==tuple:
        # lattice was used to generate tuple w lists (skycoords,pixels), we want to plant many SNe across the image
        lattice = location
        # unpack the lists of lattice
        locations,pixels = lattice
        lattice = True 
    elif type(location)==list:
        # 3 boxes of lxw = req^2, starting ctr on target core and then shifted by req along orientation 
        boxes = True
    else:
        # give arb skycoord loc (ra/dec) and translate to pixels for plant
        pix=astropy.wcs.utils.skycoord_to_pixel(location,wcs) # x,y pixel location
        revpix = copy.copy(list(pix)) # row,col location for adding to data... y,x
        revpix.reverse()
    
    if boxes:
        # location is list of 3 pixel locations to plant to, one ctr on core of target, 2 more shifted req along orientation of target
        # the location list is unpacked from target fcn into the pixels in pipeline 
        for pix in location:
            pix = list(pix)
            revpix = copy.copy(pix)
            revpix.reverse()
            # indexes to add the psf to
            row,col=revpix
            nrows,ncols=cppsf.shape
            # +2 in these to grab a couple more than needed, the correct shapes for broadcasting taken using actual psf.shapes
            rows=np.arange(int(np.round(row-nrows/2)),int(np.round(row+nrows/2))+2) 
            cols=np.arange(int(np.round(col-ncols/2)),int(np.round(col+ncols/2))+2) 
            rows = rows[:cppsf.shape[0]]
            cols = cols[:cppsf.shape[1]]
            cpim[rows[:, None], cols] += cppsf
            np.float64(cpim)
            
        # inserting True fakeSN into hdr w the pix location
        cphdr = copy.copy(hdr)
        cphdr['fakeSN']=True 
        cphdr['fakeSN_loc']='boxes' 
        cphdr['NfakeSNe']=str(len(location))
        cphdr['fakeSNmag']=str(mag)
        cphdr['fakeZP']=str(zp)
        fits.writeto(plantname,cpim,cphdr,overwrite=True)
        print('{} SNe mag ~ {} (epsf*=mu ~ {}) planted in boxes by targ; zp ~ {}'.format(len(location),mag,mu,zp))
        plant_im = fits.open(plantname)[0]  
        return plant_im,location
    if lattice:
        # many locations to plant to
        for pix in pixels:
            pix = list(pix)
            revpix = copy.copy(pix)
            revpix.reverse()
            # indexes to add the psf to
            row,col=revpix
            nrows,ncols=cppsf.shape
            # +2 in these to grab a couple more than needed, the correct shapes for broadcasting taken using actual psf.shapes
            rows=np.arange(int(np.round(row-nrows/2)),int(np.round(row+nrows/2))+2) 
            cols=np.arange(int(np.round(col-ncols/2)),int(np.round(col+ncols/2))+2) 
            rows = rows[:cppsf.shape[0]]
            cols = cols[:cppsf.shape[1]]
            cpim[rows[:, None], cols] += cppsf
            np.float64(cpim)
            
        # inserting True fakeSN into hdr w the pix location
        cphdr = copy.copy(hdr)
        cphdr['fakeSN']=True 
        cphdr['fakeSN_loc']='lattice' 
        cphdr['NfakeSNe']=str(len(pixels))
        cphdr['fakeSNmag']=str(mag)
        cphdr['fakeZP']=str(zp)
        fits.writeto(plantname,cpim,cphdr,overwrite=True)
        print('{} SNe mag ~ {} (epsf*=mu ~ {}) planted in lattice across image; zp ~ {}'.format(len(pixels),mag,mu,zp))
        plant_im = fits.open(plantname)[0]  
        return plant_im,pixels
    else:
        # single location for plant either using targ obj or provided skycoord
        # indexes for the lco data file to add the psf to
        row,col=revpix
        nrows,ncols=cppsf.shape
        # +2 in these to grab a couple more than needed, the correct shapes for broadcasting taken using actual psf.shapes
        rows=np.arange(int(np.round(row-nrows/2)),int(np.round(row+nrows/2))+2) 
        cols=np.arange(int(np.round(col-ncols/2)),int(np.round(col+ncols/2))+2) 
        rows = rows[:cppsf.shape[0]]
        cols = cols[:cppsf.shape[1]]
        cpim[rows[:, None], cols] += cppsf
        np.float64(cpim)
        # write the image with planted SN added to a new fits file (inserting True fakeSN into hdr)
        cphdr = copy.copy(hdr)
        cphdr['fakeSN']=True 
        cphdr['fakeSN_loc']=str(pix)
        plant_im = fits.writeto(plantname,cpim,cphdr,overwrite=True)
        print('SN mag ~ {} planted in image w bkg mag ~ {} at {} written to {}; zp ~ {}'.format(mag,skybr,location,plantname,zp))
        return plant_im 

def lattice(image):
    hdr = image.header
    wcs,frame=WCS(hdr),hdr['RADESYS'].lower()
    # have single 4kx4k chip from wide field instrument
    NX = hdr['naxis1']
    NY = hdr['naxis2']
    edge = 100 # pixels away from edge
    spacing = 100 # pixels between each location on lattice
    x = list(range(0+edge,NX-edge+1,spacing)) # +1 to make inclusive
    y = list(range(0+edge,NY-edge+1,spacing))
    pixels = list(itertools.product(x, y))
    locations = [] # skycoord locations that I will use to plant SNe across image  
    for i in range(len(pixels)):
        pix = pixels[i]
        location=astropy.wcs.utils.pixel_to_skycoord(pix[0],pix[1],wcs)
        locations.append(location)
    return locations,pixels

def detection_efficiency(plant,cat):
    # provide the plant and detection cat run to find efficiency
    # unpack the plant (the image and locations)
    plant_im,pixels=plant 
    # unpack the detection catalog objs (cat,image,threshold,segm)
    catalog,image,threshold,segm,targ_obj = cat

    hdr=image.header
    Nfakes=hdr['NfakeSNe']
    magfakes=hdr['fakeSNmag']
    #print('Nfakes ~ {} (= {} quick sanity check) planted in this image'.format(Nfakes,len(pixels)))
    #print('Nsources ~ {} detected in the image'.format(len(catalog)))
    
    # use locations and a search radius on detections and plant locations to get true positives
    tbl = catalog.to_table()
    tbl_x,tbl_y = [i.value for i in tbl['xcentroid']], [i.value for i in tbl['ycentroid']]
    tbl_pixels = list(zip(tbl_x,tbl_y))
    tbl.add_column(Column(tbl_pixels),name='pix') # adding this for easier use indexing tbl later
    search = 5 # fwhm*n might be better criteria
    truths = []
    for pixel in tbl_pixels:
        for i in pixels:
            if pixel[0] > i[0] - search  and pixel[0] < i[0] + search and pixel[1] > i[1] - search and pixel[1] < i[1] + search:
                truths.append([i,pixel])
                #print(i,pixel)
            else:
                continue
    #print('{} source detections within search radius criteria'.format(len(truths)))
    # TODO: get the tbl_pixels which were outside the search radius criteria and return them as false positives
    
    # break truths into the plant pixels and det src pixel lists; easier to work w
    plant_pixels = []
    det_src_pixels = []
    for i in truths:
        plant_pix = i[0]
        det_src_pix = i[1]
        plant_pixels.append(plant_pix)
        det_src_pixels.append(det_src_pix)
    # the plant pixels which had multiple sources detected around it
    repeat_plant = [item for item, count in collections.Counter(plant_pixels).items() if count > 1]
    # the plant pixels which only had one source detected 
    single_plant = [item for item, count in collections.Counter(plant_pixels).items() if count == 1]
    N_plants_detected = len(single_plant) + len(repeat_plant)
    # adding nearby_plantpix col to src table; using None if source wasnt within the search radius of plant
    plant_col = []
    for i in tbl:
        tbl_x,tbl_y = i['xcentroid'].value,i['ycentroid'].value
        if (tbl_x,tbl_y) in det_src_pixels:
            idx = det_src_pixels.index((tbl_x,tbl_y))
            plant_col.append(plant_pixels[idx])
        else:
            plant_col.append(None)
    tbl.add_column(Column(plant_col),name='nearby_plantpix')
    
    # index table to grab false source detections
    false_tbl = tbl[tbl['nearby_plantpix']==None]
    truth_tbl = tbl[tbl['nearby_plantpix']!=None]
    
    single_truth_tbl,repeat_truth_tbl = [],[]
    for i in truth_tbl:
        if i['nearby_plantpix'] in repeat_plant:
            repeat_truth_tbl.append(i)
        else:
            single_truth_tbl.append(i)
    # should use a check on length rather than try/except below here
    # try/excepting is to avoid error for empty lists
    # mainly an issue on repeat truth tbl 
    try:
        single_truth_tbl = vstack(single_truth_tbl)
    except:
        pass
    try:
        repeat_truth_tbl = vstack(repeat_truth_tbl)
    except:
        pass            
    #print('Final: {} planted SNe, {} clean single detections, {} as multi-sources near a plant, {} false detections'.format(Nfakes,len(single_truth_tbl),len(repeat_truth_tbl),len(false_tbl)))
    print('{} planted SNe had single clean source detected, {} planted SNe had multiple sources detected nearby, {} false detections'.format(len(single_plant),len(repeat_plant),len(false_tbl)))

    efficiency = N_plants_detected/len(pixels)

    print('Detection efficiency (N_plants_detected/N_plants) ~ {} on mag ~ {} SNe'.format(efficiency,magfakes))
    return efficiency,magfakes,tbl,single_truth_tbl,repeat_truth_tbl,false_tbl

def lco_pipe():
    #print(sys.argv[0]) # the name of this command script  
    date_key = str(sys.argv[1]) # easy enough to change sbatch to get xx.xx date
    field_key = int(sys.argv[2]) # slurm array idx in sbatch that will be used to do the different fields 
    # lco_path ~ current working dir with scripts and sub-dirs of data  
    lco_path = '/work/oconnorf/efficiency_pipeline/lco/'
    # all the dates w lco data in the lco_path 
    all_dates = [x for x in glob.glob(lco_path+'/*') if os.path.isdir(x)]
    # your batch should have xx.xx date given so script knows which set of fields you want to do 
    date_path = os.path.join(lco_path,date_key+'/*')
    all_fields = [x for x in glob.glob(date_path) if os.path.isdir(x)]
    field = all_fields[field_key]
    # which date folder are we in and which field was this slurm idx job 

    # all the fits images needed, the trims, diffs, and ref 
    my_data = get_data(field)
    # a table that has the galaxy-galaxy strong lens system: id, magnification, lens_z, source_z, peakIa mag
    glsn = ascii.read('peakGLSN.csv')

    # each field should have a folder source_im (along w dia_out and dia_trim) 
    # in source_im is the image you want to do this for
    # ie the one that psf is measured on trim and SNe planted to diff of
    source_im = glob.glob(field+'/source_im/*fits')[0]
    source_output = os.path.join(field,'source_im/output') # where will be stick results of pipeline 
    filename=source_im.split('/')[-1]
    image = my_data[filename]
    diff_image = my_data['d_'+filename]
    ref_image = my_data['ref.fits']
    hdr = image.header
    groupid,L1fwhm,pixscale,skybr = hdr['GROUPID'],hdr['L1fwhm'],hdr['pixscale'],hdr['WMSSKYBR'] # pixels, arcsec/pixels,mag/arcsec^2
    med,exptime = hdr['L1MEDIAN'],hdr['EXPTIME']
    zp=skybr+2.5*np.log10(med/exptime/pixscale)
    glsnID = glsn[glsn['Source ID'] == groupid] # idx the glsn table using id 
    print('filename ~ {} (groupid {}) has L1fwhm ~ {} pixels, pixscale ~ {} arcsec/pixel, and skybr {} mag/arcsec^2; zp ~ {}'.format(filename,groupid,L1fwhm,pixscale,skybr,zp))
    print('glsn ~ {}'.format(glsnID))
    print('\n')

    pickle_to = source_output + '/' + filename[:-5] # -5 get rid of .fits
    
    # photutils source properties to detect objs in image
    nsigma,kernel_size,npixels,deblend,contrast,targ_coord = 5,(3,3),int(np.round(L1fwhm/pixscale)),False,.001,None
    print('Source Catalog is a photutils source_properties using nsigma ~ {} (detection threshold above img bkg), gaussian kernel sized ~ {} pix, npixels ~ {} (connected pixels needed to be considered source), deblend ~ {} w contrast {}'.format(nsigma,kernel_size,npixels,deblend,contrast))
    source_catalog = source_cat(image,nsigma=nsigma,kernel_size=kernel_size,npixels=npixels,deblend=deblend,contrast=contrast,targ_coord=None)
    cat,image,threshold,segm,targ_obj = source_catalog # unpacked to make a little clearer
    pickle.dump(cat,open(pickle_to + '_source_cat.pkl','wb'))

    # get stars from the astroquery on gaia
    results = gaia_results(image)
    gaia,image = results # unpacked 
    # extract good gaia stars from image for psf
    extracted_stars = stars(results)
    good_stars,image = extracted_stars # unpacked
    # use extracted stars to build epsf
    EPSF = ePSF(extracted_stars,oversampling=2)
    epsf,fitted_stars = EPSF # unpacked
    pickle.dump(EPSF,open(pickle_to+'_epsf.pkl','wb'))
    # fit 2d gaussian to the epsf, see how 'non-gaussian' the actual psf is
    epsf_gaussian = gaussian2d(epsf)
    fit_gaussian,levels,xctr_vals,yctr_vals,image1,img_epsf,resid = epsf_gaussian # unpacked... levels list amplitude - sigma, ctr vals are gauss model sliced, image1 is array of values from gaussian fit in shape of epsf, img_epsf is epsf instance of it, resid is gauss - epsf 
    # make figures
    psf_and_gauss(epsf,epsf_gaussian,saveas=pickle_to+'_psf.pdf')
    used_stars(fitted_stars,saveas=pickle_to+'_stars.pdf')

    # target galaxy work, tuples cutting boxes around target (data,patch), how/if planting on cores might lower detection efficiency
    # also returns targ_obj again account for updates using ref (in the cases where empty targ_obj ie not detected in source)
    target_boxes = target(image,targ_obj,ref=ref_image,diff=diff_image) 
    targ_obj,cuts,bkg_core,bkg_1,bkg_2 = target_boxes # unpacked
    cut_targ,cut_diff,cut_ref = cuts # unpack cuts around target source,diff,and ref

    # measured psf is now going to be scaled to different magnitudes and planted in the difference image
    mags = np.arange(skybr-4.5,skybr+3,0.5) #zp ~ 23.5 # rough zp 

    # this loop (repeatedly) plants near target galaxy; repeated so efficiency can be determined
    # the box plant wants a list of pixel coordinates, accessing from the target boxes here
    target_locations_orig = [bkg_core[0].center_original,bkg_1[0].center_original,bkg_2[0].center_original] # [0] so cut not patch
    target_locations_cutout = [bkg_core[0].center_cutout,bkg_1[0].center_cutout,bkg_2[0].center_cutout]
    # planting on target centroid at many magnitudes, I am not using the two shifts really only interested to see core
    # running detection repeatedly to determine efficiency
    efficiencies = []
    j = 0
    for mag in mags:
        efficiencies.append([])
        # planting into difference image
        box_plantname = '{}_planted_targ_mag{}.fits'.format(pickle_to,str(mag))
        box_planted_orig = plant(diff_image,epsf,source_catalog,hdr=hdr,mag=mag,location=[target_locations_orig[0]],zp=None,plantname=box_plantname)
        # unpack
        box_plant_im,box_pixels = box_planted_orig 
        # make target figures (similar to prev w source but now diff and plants)
        #target_image(box_plant_im,targ,saveas=pickle_to+'_target_plantdiff_mag{}.pdf'.format(str(mag)))
        # get a pdf showing image,ref,diff,fakeplant for SN w this mag
        view_targetplant(image,ref_image,diff_image,box_plant_im,target_boxes,zp,saveas=pickle_to+'targetplant_mag{}.pdf'.format(str(mag)))

        j += 1
        # The detection either works or doesn't for a plant there is no randomness in algorithm 
        # i.e. the catalog returned for an image is the same for a given detection
        # Therefor don't really need to do detection in range(0,N) but I do it twice anyways
        for i in range(0,2):
            # source properties of detected objs in fake image
            print(j,i)
            fakesource_cat = source_cat(box_plant_im,nsigma=nsigma,kernel_size=kernel_size,npixels=npixels,deblend=deblend,contrast=contrast,targ_coord=None)
            fakecat,fakeimage,fakethreshold,fakesegm,faketarg_obj = fakesource_cat # unpacked to make a little clearer
            pickle.dump(fakecat,open(pickle_to+'_fakesource_cat{}.pkl'.format(str(i)),'wb'))

            # detection efficiency  
            tmp = detection_efficiency(box_planted_orig,fakesource_cat)
            efficiency,magfakes,tbl,single_truth_tbl,repeat_truth_tbl,false_tbl = tmp
            efficiencies[j-1].append(efficiency)
            pickle.dump(tmp,open(pickle_to+'_detection_efficiency{}_mag{}.pkl'.format(str(i),str(mag)),'wb'))
            print(efficiency,magfakes)
            print('--------------------------------------------------------------')
    avg_efficiencies=[]
    for i in efficiencies:
        efficiency = np.average(i)
        print(efficiency)
        avg_efficiencies.append(efficiency)
    avg_efficiencies,mags=list(avg_efficiencies),list(mags)
    avg_efficiencies.reverse()
    mags.reverse()
    m50 = np.interp(0.5,avg_efficiencies,mags)
    print('m50 ~ {}'.format(m50))  
    # make figures
    detection_efficiency(mags,avg_efficiencies,m50,target_boxes,skybr,zp,glsn=glsnID,saveas=pickle_to+'_target_detection_efficiency.pdf')

    # lattice plant into the difference, a second way to do detection efficiency
    mags = np.arange(skybr-4.5,skybr+3,0.5) # get mags back in order, zp ~ 23.5 # rough zp 
    locations = lattice(image)
    efficiencies = []
    for mag in mags:
        # create plant image; zp None, using measure sky mag/arcsec^2 from L1 hdr to set
        plantname = '{}_planted_lattice_mag{}.fits'.format(pickle_to,str(mag))
        planted = plant(diff_image,epsf,source_catalog,hdr=hdr,mag=mag,location=locations,zp=None,plantname=plantname)
        plant_im,pixels = planted # unpack

        # source properties of detected objs in fake image
        fakesource_cat = source_cat(plant_im,nsigma=nsigma,kernel_size=kernel_size,npixels=npixels,deblend=deblend,contrast=contrast,targ_coord=None)
        fakecat,fakeimage,fakethreshold,fakesegm,faketarg_obj = fakesource_cat # unpacked to make a little clearer
        pickle.dump(fakecat,open(pickle_to+'_fakesource_cat.pkl','wb'))

        # detection efficiency  
        tmp = detection_efficiency(planted,fakesource_cat)
        efficiency,magfakes,tbl,single_truth_tbl,repeat_truth_tbl,false_tbl = tmp
        efficiencies.append(efficiency)
        pickle.dump(tmp,open(pickle_to+'_detection_efficiency_mag{}.pkl'.format(str(mag)),'wb'))
        print(efficiency,magfakes)
        print('--------------------------------------------------------------')

    print(filename)
    print('efficiencies: {}'.format(efficiencies))
    print('mags: {}'.format(mags))
    # use interp to get magnitude at which we have 50% detection efficiency 
    # need the values increasing along x for interp to work properly
    efficiencies,mags=list(efficiencies),list(mags)
    efficiencies.reverse()
    mags.reverse()
    m50 = np.interp(0.5,efficiencies,mags)
    print('m50 ~ {}'.format(m50))

    # make figures
    detection_efficiency(mags,efficiencies,m50,target_boxes,skybr,zp,glsn=glsnID,saveas=pickle_to+'_detection_efficiency.pdf')
    lattice_planted(mags,m50,pickle_to=pickle_to,saveas=pickle_to+'_plants.pdf')

if __name__=="__main__":
    #print('lco pipe coming at ya')
    lco_pipe()

