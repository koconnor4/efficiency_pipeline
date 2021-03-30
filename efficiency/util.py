import os 
import glob
import pickle
import numpy as np
import collections
import astropy
from astropy.io import ascii,fits
from astropy.wcs import WCS, utils as wcsutils
from astropy.coordinates import SkyCoord
from astropy.convolution import Gaussian2DKernel
from astropy.nddata import Cutout2D,NDData
from astropy.wcs.utils import skycoord_to_pixel, pixel_to_skycoord
from astropy.stats import sigma_clipped_stats,gaussian_fwhm_to_sigma,gaussian_sigma_to_fwhm
from astropy.table import Table,Column,Row,vstack,setdiff,join
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.visualization import ZScaleInterval,simple_norm
zscale = ZScaleInterval()

import photutils
from photutils.datasets import make_gaussian_sources_image
from photutils import find_peaks
from photutils.psf import extract_stars
from photutils import EPSFBuilder
from photutils import detect_threshold
from photutils import detect_sources
from photutils import deblend_sources
from photutils import source_properties, EllipticalAperture
from photutils import BoundingBox
from photutils import Background2D, MedianBackground
from photutils.psf import IntegratedGaussianPRF, DAOGroup, BasicPSFPhotometry
from photutils.background import MMMBackground, MADStdBackgroundRMS

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.text as mpl_text

import pandas as pd
import itertools

class AnyObject(object):
    def __init__(self, text, color):
        self.my_text = text
        self.my_color = color

class AnyObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        #print orig_handle
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = mpl_text.Text(x=0, y=0, text=orig_handle.my_text, color=orig_handle.my_color, verticalalignment=u'baseline', 
                                horizontalalignment=u'left', multialignment=None, 
                                fontproperties=None, rotation=0, linespacing=None, 
                                rotation_mode=None)
        handlebox.add_artist(patch)
        return patch


def detection(loc,sources,radius):
    """
    Parameters
    -----------
    loc ~ tuple [pixels]
        (xp,yp) location of the object you want to see if is detected
    sources ~ table 
        the detected sources (DAOStarFinder or other)
    radius ~ float [pixels]
        search in circle around loc for sources
    """
    if sources == None:
        detection = 0
    else:
        distances = [] 
        for i in range(len(sources)):
            xsource,ysource=sources['xcentroid'][i],sources['ycentroid'][i]
            xloc,yloc = loc
            distance=np.sqrt( (xloc - xsource)**2 + (yloc - ysource)**2 ) 
            distances.append(distance)
        nearest = np.min(distances)
        if nearest < radius:
            detection = 1
        else:
            detection = 0
    
    return detection

def search_sdss(sdss,rs):
    """
    Given a list of r mag values take closest values from sdss['r'] table 
    """
    idxs = []
    for r in rs:
        idx = np.argmin(abs(sdss['r'] - r))
        idxs.append(idx)
    return sdss[idxs]

def pixtosky(hdu,pixel):
    """
    Given a pixel location returns the skycoord
    """
    hdr = hdu.header
    wcs,frame = WCS(hdr),hdr['RADESYS'].lower()
    xp,yp = pixel
    sky = wcsutils.pixel_to_skycoord(xp,yp,wcs)
    return sky

def skytopix(hdu,sky):
    """
    Given a skycoord (or list of skycoords) returns the pixel locations
    """
    hdr = hdu.header
    wcs,frame = WCS(hdr),hdr['RADESYS'].lower()
    pixel = wcsutils.skycoord_to_pixel(sky,wcs)
    try:
        pixel = list(zip(pixel[0],pixel[1]))
    except:
        pass
    return pixel

def cut_hdu(hdu,location,size,writetodisk=False,saveas=None):
    """
    cutout size lxw ~ (dy x dx) box on fits file centered at a pixel or skycoord location
    if size is scalar gives a square dy=dx 
    updates hdr wcs keeps other info from original
    """
    cphdu = hdu.copy()
    dat = cphdu.data
    hdr = cphdu.header
    wcs,frame = WCS(hdr),hdr['RADESYS'].lower()
        
    cut = Cutout2D(dat,location,size,wcs=wcs) 
    cutwcs = cut.wcs
    cphdu.data = cut.data
    cphdu.header.update(cut.wcs.to_header())  
    
    if writetodisk:  
        cphdu.writeto(saveas,overwrite=True)

    hdu.postage_stamp = cphdu
    
    return cphdu


def plants_MEF(hdu,positions,cutoutsize=50,writetodisk=False,saveas='mef.fits'):
    """
    Create MEF Fits file with extensions to cutouts at positions with lxw of cutoutsize [pixels] in hdu
    
    For our purpose in efficiency_pipe this is used to get single fits with the planted fake sources...
    # positions are locations of fakes  
    # saveas like f'mag{m}fakes.fits'
    
    primary data empty, primary header has magnitude that the plants were scaled to
    
    Parameters
    ----------
    hdu : astropy.io.fits
        The difference image being planted onto
    positions : List-like 
        n pixel positions to plant at in the diff. [(x1,y1),...(xn,yn)]
    cutoutsize : int
        number of pixels on a side for the image to be shown. We cut it in
        half and use the integer component, so if an odd number or float is
        provided it is rounded down to the preceding integer.
    Returns
    -----------
    MEFs ~ astropy.io.fits 
        an MEF with plants at positions on hdu 
    """
    
    nfakes = len(positions)
    diffim = hdu.data

    # init the MEF
    primary = fits.PrimaryHDU(data=None,header=None)
    primary.header["Author"] = "Kyle OConnor"
    primary.header["MEF"] = f'NFK{nfakes:03d}'
    new_hdul = [primary] # fits.HDUList(new_hdul) after append all the fakes into new_hdul
    for i in positions:
        # grab cutouts
        #print("position:",i)
        diff_location = i # (xp,yp)
        cutdiff = cut_hdu(hdu,diff_location,cutoutsize) # also accessible as hdu.postage_stamp
        #cutsearch = cut_hdu(self.searchim,search_location,cutoutsize)
        #cuttemp = cut_hdu(self.templateim,template_location,cutoutsize
        try:
            assert(fits.CompImageHDU == type(cutdiff))
        except:
            cutdiff = fits.CompImageHDU(data=hdu.postage_stamp.data,header=hdu.postage_stamp.header)            
        new_hdul.append(hdu.postage_stamp)
    
    new_hdul = fits.HDUList(new_hdul)
    if writetodisk:
        new_hdul.writeto(saveas, overwrite=True)
    return new_hdul


def get_lattice_positions(hdu, edge=500, spacing=500):
    """Function for constructing list of pixels in a grid over the image
    Parameters
    :edge : int
    "Gutter" pixels away from the edge of the image for the start/end of grid
    :spacing : int
    Number of pixels in x and y directions between each fake
    """
    #reads number of rows/columns from header and creates a grid of locations for planting
    hdr = hdu.header
    wcs,frame=WCS(hdr),hdr['RADESYS'].lower()
    
    NX = hdr['naxis1']
    NY = hdr['naxis2']
    x = list(range(0+edge,NX-edge+1,spacing)) # +1 to make inclusive
    y = list(range(0+edge,NY-edge+1,spacing))
    pixels = list(itertools.product(x, y))
    skycoords = [] # skycoord locations corresponding to the pixels  
    for i in range(len(pixels)):
        pix = pixels[i]
        skycoord=astropy.wcs.utils.pixel_to_skycoord(pix[0],pix[1],wcs)
        skycoords.append(skycoord)

    hdu.has_lattice = True # if makes it through lattice update has_lattice

    return np.array(pixels), np.array(skycoords)


def weighted_average(weights,values,plot=False,saveas=None):
    # zp ~ need to filter out nan/masked vals
    weights = np.array(weights)
    values = np.array(values)
    indices = ~np.isnan(values)
    true_count = np.sum(indices)
    values,weights = values[indices],weights[indices]
    avg = np.average(values, weights=weights)
    if plot:
        matplotlib.rcParams.update({'font.size': 20,'xtick.labelsize':15,'ytick.labelsize':15})
        fig,ax = plt.subplots(figsize=(16,8))
        spacing = np.arange(0,true_count,1)
        uncertainties = [1/i for i in weights]
        ax.errorbar(spacing,values,yerr=uncertainties,marker='x',ls='',color='red',label='zp')
        ax.hlines(avg,0,true_count,linestyle='--',label='avg~{:.1f}'.format(avg),color='red')
        plt.legend()
        plt.show()
        if saveas:
            plt.savefig(saveas,bbox_inches='tight')
        plt.close()
    return avg

def LCO_PSF_PHOT(hdu,init_guesses):
    # im ~ np array dat, pixel [x0,y0] ~ float pixel position, sigma_psf ~ LCO-PSF sigma 
    x0,y0=init_guesses
    im = hdu.data
    hdr = hdu.header
    
    fwhm = hdr['L1FWHM']/hdr['PIXSCALE'] # PSF FWHM in pixels, roughly ~ 5 pixels, ~ 2 arcsec 
    sigma_psf = fwhm*gaussian_fwhm_to_sigma # PSF sigma in pixels
    
    psf_model = IntegratedGaussianPRF(sigma=sigma_psf)
    daogroup = DAOGroup(2.0*sigma_psf*gaussian_sigma_to_fwhm)
    mmm_bkg = MMMBackground()
    fitter = LevMarLSQFitter()

    psf_model.x_0.fixed = True
    psf_model.y_0.fixed = True
    pos = Table(names=['x_0', 'y_0'], data=[[x0],[y0]]) # optionally give flux_0 has good aperture method for guessing though

    photometry = BasicPSFPhotometry(group_maker=daogroup,
                                     bkg_estimator=mmm_bkg,
                                     psf_model=psf_model,
                                     fitter=LevMarLSQFitter(),
                                     fitshape=(11,11))
    result_tab = photometry(image=im, init_guesses=pos)
    residual_image = photometry.get_residual_image()
    
    return result_tab

def LCO_AP_PHOT(hdu,init_guesses):
    """Takes in a source catalog for stars in the image. Will perform
    aperture photometry on the sources listed in this catalog.
    Parameters
    ----------
    """
    
    ##TODO: Add something to handle overstaturated sources
    
    ##Set up the apertures
    from photutils import CircularAperture , aperture_photometry , CircularAnnulus
    #[(x0,y0)]=init_guesses
    pixscale = hdu.header["PIXSCALE"]
    FWHM = hdu.header["L1FWHM"]
    aperture_radius = 2 * FWHM / pixscale
    pos = init_guesses #Table(names=['x_0', 'y_0'], data=[[x0],[y0]]) 
    apertures = CircularAperture(pos, r= aperture_radius)
    annulus_aperture = CircularAnnulus(pos, r_in = aperture_radius + 5 , r_out = aperture_radius + 10)
    annulus_masks = annulus_aperture.to_mask(method='center')
    
    ##Background subtraction using sigma clipped stats.
    ##Uses a median value from the annulus
    bkg_median = []
    for mask in annulus_masks:
        annulus_data = mask.multiply(hdu.data)
        annulus_data_1d = annulus_data[mask.data > 0]
        _ , median_sigclip, _ = sigma_clipped_stats(annulus_data_1d)
        bkg_median.append(median_sigclip)
    ##Perform photometry and subtract out background
    bkg_median = np.array(bkg_median)
    ##The pixel-wise Gaussian 1-sigma errors of the input data using sigma clipped stats.
    ##Uses a std value from the hdu, needs to be the same shape as the input data
    _, _, std_sigclip = sigma_clipped_stats(hdu.data)
    error = np.zeros(hdu.data.shape) + std_sigclip
    phot = aperture_photometry(hdu.data, apertures,error=error)
    phot['annulus_median'] = bkg_median
    phot['aper_bkg'] = bkg_median * apertures.area
    
    phot['aper_sum_bkgsub'] = phot['aperture_sum'] - phot['aper_bkg']
    
    phot['mag'] = -2.5 * np.log10( phot['aper_sum_bkgsub'] )
    
    #self.stellar_phot_table = phot
    return phot

def f_efficiency(m,m50,alpha):
    #https://arxiv.org/pdf/1509.06574.pdf, strolger 
    return (1+np.exp(alpha*(m-m50)))**-1 

def threshold_detect_sources(hdu,nsigma=3,kfwhm=5,npixels=5,deblend=False,contrast=.001, **kwargs):
    """Detect sources (transient candidates) in the diff image using
    the astropy.photutils threshold-based source detection algorithm.
    Parameters
    ----------
    nsgima : float
        SNR required for pixel to be considered detected
    kfwhm : float
        FWHM of Circular Gaussian Kernel convolved on data to smooth noise. 5 pixels default is typical L1FWHM for LCO.
    npixels : int
        Number of connected pixels which are detected to give source
    deblend : bool
        Will use multiple levels/iterations to deblend single sources into multiple
    contrast : float
        If deblending the flux ratio required for local peak to be considered its own object
    Returns
    -------
    self.sourcecatalog: :class:`~photutils.segmentation.properties.SourceCatalog`
    """
    # TODO

    # record the locations and fluxes of candidate sources in an
    # external source catalog file (or a FITS extension)

    # if a fake is detected, mark it as such in the source catalog

    # if a fake is not detected, add it to the source catalog
    # (as a false negative)

    # maybe separate?: run aperture photometry on each fake source
    # maybe separate?: run PSF fitting photometry on each fake source
    # to be able to translate from ra/dec <--> pixels on image
    hdr = hdu.header

    wcs,frame = WCS(hdr),hdr['RADESYS'].lower()
    #L1mean,L1med,L1sigma,L1fwhm = hdr['L1MEAN'],hdr['L1MEDIAN'],hdr['L1SIGMA'],hdr['L1FWHM'] # counts, fwhm in arcsec 
    #pixscale,saturate,maxlin = hdr['PIXSCALE'],hdr['SATURATE'],hdr['MAXLIN'] # arcsec/pixel, counts for saturation and non-linearity levels
    # if bkg None: detect threshold uses sigma clipped statistics to get bkg flux and set a threshold for detected sources
    # bkg also available in the hdr of file, either way is fine  
    # threshold = detect_threshold(hdu.data, nsigma=nsigma)
    # or you can provide a bkg of the same shape as data and this will be used
    boxsize=500
    bkg = Background2D(hdu.data,boxsize) # sigma-clip stats for background est over image on boxsize, regions interpolated to give final map 
    threshold = detect_threshold(hdu.data, nsigma=nsigma,background=bkg.background)
    ksigma = kfwhm * gaussian_fwhm_to_sigma  # FWHM pixels for kernel smoothing
    # optional ~ kernel smooths the image, using gaussian weighting
    kernel = Gaussian2DKernel(ksigma)
    kernel.normalize()
    # make a segmentation map, id sources defined as n connected pixels above threshold 
    segm = detect_sources(hdu.data,
                          threshold, npixels=npixels, filter_kernel=kernel)
    # deblend useful for very crowded image with many overlapping objects...
    # uses multi-level threshold and watershed segmentation to sep local peaks as ind obj
    # use the same number of pixels and filter as was used on original segmentation
    # contrast is fraction of source flux local pk has to be consider its own obj
    if deblend:
        segm = deblend_sources(hdu.data, 
                                       segm, npixels=5,filter_kernel=kernel, 
                                       nlevels=32,contrast=contrast)
    # need bkg subtracted to do photometry using source properties
    data_bkgsub = hdu.data - bkg.background
    cat = source_properties(data_bkgsub, segm,background=bkg.background,
                            error=None,filter_kernel=kernel)

    # TODO the detection parameters into meta of table
    meta = {'detect_params':{"nsigma":nsigma,"kfwhm":kfwhm,"npixels":npixels,
                                            "deblend":deblend,"contrast":contrast}}

    hdu.sourcecatalog = cat

    # TODO : identify indicies of extended sources and make a property
    #  of the class that just gives an index into the source catalog
    #for i in self.sourcecatalog:
    #    if i.ellipticity > 0.35: ##Identifies Galaxies
    ##        if i.area.value < 8 and cut_cr: ##Removes cosmic rays
    #            continue
    #        xcol.append(i.centroid[1])
    #        ycol.append(i.centroid[0])
    #        source_propertiescol.append(i)
    # hostgalaxies = Table([xcol , ycol , source_propertiescol] , names = ("x" , "y" , "Source Properties"))
    # self.hostgalaxies = hostgalaxies
    # return self.hostgalaxies

    return hdu.sourcecatalog

def detection_efficiency(plantpixels,detection_cat,search=5,plot=False,hdu=None,saveas="_detections.png",r=None):
    """
    Efficiency of detections using fake plants in image. 
    Search radius constituting true detection is default 5 pixels for plant location to detected obj location. 

    Parameters
    ____________
    plantpixels : list 
        [(xp_i,yp_i),...(xp_n,y_n)] plantpixels for the n fake planted objects
    detection_cat : `~photutils.segmentation.properties.SourceCatalog`
        the detect_sources() catalog on the planted hdu
    search : float
        Search radius around plantpixel location for detection_cat obj
    Returns
    ____________
    efficiency : float
        Ndetectedplants/Nplants
    """
    plantpixels = [tuple(i) for i in plantpixels] # makesure tuples [(xi,yi),...(xn,yn)] needs to be hashable
    # use locations and a search radius on detections and plant locations to get true positives
    try:
        tbl = detection_cat.to_table()
        tbl_x,tbl_y = [i.value for i in tbl['xcentroid']], [i.value for i in tbl['ycentroid']]
    except:
        tbl = detection_cat
        tbl_x,tbl_y = [i for i in tbl['xcentroid']], [i for i in tbl['ycentroid']]
        
    tbl_pixels = list(zip(tbl_x,tbl_y))
    tbl.add_column(Column(tbl_pixels),name='pix') # adding this for easier use indexing tbl later
    truths = [] # better name would be detections
    for pixel in tbl_pixels:
        for i in plantpixels:
            if pixel[0] > i[0] - search  and pixel[0] < i[0] + search and pixel[1] > i[1] - search and pixel[1] < i[1] + search:
                truths.append([i,pixel])
            else:
                continue

    # break truths into the plantpixels and det src pixel lists; easier to work w
    plant_pixels = []
    det_src_pixels = []
    for i in truths:
        plant_pix = i[0]
        det_src_pix = i[1]
        plant_pixels.append(plant_pix)
        det_src_pixels.append(det_src_pix)
    # non detections
    nondetections = [i for i in plantpixels if i not in plant_pixels]

    # plantpixels which had multiple sources detected around it
    repeat_plant = [item for item, count in collections.Counter(plant_pixels).items() if count > 1]
    # plantpixels which only had one source detected 
    single_plant = [item for item, count in collections.Counter(plant_pixels).items() if count == 1]
    N_plants_detected = len(single_plant) + len(repeat_plant)
    detected_plants = repeat_plant + single_plant
    efficiency = N_plants_detected/len(plantpixels)
    print("eff ~ {}, {} detected_plants, {} non detected plants".format(efficiency,len(detected_plants),len(nondetections)))
    print(detected_plants,nondetections)
    if plot:
        assert(hdu != None)
        
        fig,ax = plt.subplots(figsize=(8,8))
        ax.imshow(zscale(hdu.data),cmap='Greys')
        ax.set_xticks([])
        ax.set_yticks([])

        det_patches = []
        for i in detected_plants:
            xy = i
            patch = matplotlib.patches.Circle(xy,radius=search*5,lw=None,fill=None,color='red')
            ax.add_patch(patch)
        nondet_patches = []
        for i in nondetections:
            xy = i
            patch = matplotlib.patches.Circle(xy,radius=search*5,lw=None,fill=None,color='green')
            ax.add_patch(patch)
        plt.title("$r = {:.2f}$".format(r))
        plt.savefig(saveas,bbox_inches='tight')
        plt.close()

    """
    Have the efficiency at this point commenting out the remainder...
    Dont think I need to worry so much about false source detections in this function
    Can do a detect_sources on a clean diff to get straightforward artifacts/fp if ends up necessary

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
    #print('{} planted SNe had single clean source detected, {} planted SNe had multiple sources detected nearby, {} false detections'.format(len(single_plant),len(repeat_plant),len(false_tbl)))
    """

    #print('Detection efficiency (N_plants_detected/N_plants) ~ {} on mag ~ {} SNe'.format(efficiency,magfakes))
    return efficiency #,tbl,single_truth_tbl,repeat_truth_tbl,false_tbl


def _extract_psf_fitting_names(psf):
    """
    Determine the names of the x coordinate, y coordinate, and flux from
    a model.  Returns (xname, yname, fluxname)
    """

    if hasattr(psf, 'psf_xname'):
        xname = psf.psf_xname
    elif 'x_0' in psf.param_names:
        xname = 'x_0'
    else:
        raise ValueError('Could not determine x coordinate name for '
                         'psf_photometry.')

    if hasattr(psf, 'psf_yname'):
        yname = psf.psf_yname
    elif 'y_0' in psf.param_names:
        yname = 'y_0'
    else:
        raise ValueError('Could not determine y coordinate name for '
                         'psf_photometry.')

    if hasattr(psf, 'psf_fluxname'):
        fluxname = psf.psf_fluxname
    elif 'flux' in psf.param_names:
        fluxname = 'flux'
    else:
        raise ValueError('Could not determine flux name for psf_photometry.')

    return xname, yname, fluxname

def add_psf(hdu, psf, posflux, subshape=None,writetodisk=False,saveas="planted.fits"):
    """
    Add (or Subtract) PSF/PRFs from an image.
    Parameters
    ----------
    data : `~astropy.nddata.NDData` or array (must be 2D)
        Image data.
    psf : `astropy.modeling.Fittable2DModel` instance
        PSF/PRF model to be substracted from the data.
    posflux : Array-like of shape (3, N) or `~astropy.table.Table`
        Positions and fluxes for the objects to subtract.  If an array,
        it is interpreted as ``(x, y, flux)``  If a table, the columns
        'x_fit', 'y_fit', and 'flux_fit' must be present.
    subshape : length-2 or None
        The shape of the region around the center of the location to
        subtract the PSF from.  If None, subtract from the whole image.
    Returns
    -------
    subdata : same shape and type as ``data``
        The image with the PSF subtracted
    """

    # copying so can leave original data untouched
    cphdu = hdu.copy()
    data = cphdu.data
    cphdr = cphdu.header

    wcs,frame = WCS(cphdr),cphdr['RADESYS'].lower()

    if data.ndim != 2:
        raise ValueError(f'{data.ndim}-d array not supported. Only 2-d '
                         'arrays can be passed to subtract_psf.')

    #  translate array input into table
    if hasattr(posflux, 'colnames'):
        if 'x_fit' not in posflux.colnames:
            raise ValueError('Input table does not have x_fit')
        if 'y_fit' not in posflux.colnames:
            raise ValueError('Input table does not have y_fit')
        if 'flux_fit' not in posflux.colnames:
            raise ValueError('Input table does not have flux_fit')
    else:
        posflux = Table(names=['x_fit', 'y_fit', 'flux_fit'], data=posflux)

    # Set up contstants across the loop
    psf = psf.copy()
    xname, yname, fluxname = _extract_psf_fitting_names(psf)
    indices = np.indices(data.shape)
    subbeddata = data.copy()
    addeddata = data.copy()
    
    n = 0
    if subshape is None:
        indicies_reversed = indices[::-1]

        for row in posflux:
            getattr(psf, xname).value = row['x_fit']
            getattr(psf, yname).value = row['y_fit']
            getattr(psf, fluxname).value = row['flux_fit']

            xp,yp,flux_fit = row['x_fit'],row['y_fit'],row['flux_fit']
            sky = wcsutils.pixel_to_skycoord(xp,yp,wcs)
            idx = str(n).zfill(3) 
            cphdr['FK{}X'.format(idx)] = xp
            cphdr['FK{}Y'.format(idx)] = yp
            cphdr['FK{}RA'.format(idx)] = str(sky.ra.hms)
            cphdr['FK{}DEC'.format(idx)] = str(sky.dec.dms)
            cphdr['FK{}F'.format(idx)] = flux_fit
            # TO-DO, once have actual epsf classes will be clearer to fill the model
            cphdr['FK{}MOD'.format(idx)] = "NA"
            n += 1

            subbeddata -= psf(*indicies_reversed)
            addeddata += psf(*indicies_reversed)
    else:
        for row in posflux:
            x_0, y_0 = row['x_fit'], row['y_fit']

            # float dtype needed for fill_value=np.nan
            y = extract_array(indices[0].astype(float), subshape, (y_0, x_0))
            x = extract_array(indices[1].astype(float), subshape, (y_0, x_0))

            getattr(psf, xname).value = x_0
            getattr(psf, yname).value = y_0
            getattr(psf, fluxname).value = row['flux_fit']

            xp,yp,flux_fit = row['x_fit'],row['y_fit'],row['flux_fit']
            sky = wcsutils.pixel_to_skycoord(xp,yp,wcs)
            idx = str(n).zfill(3) 
            cphdr['FK{}X'.format(idx)] = xp
            cphdr['FK{}Y'.format(idx)] = yp
            cphdr['FK{}RA'.format(idx)] = str(sky.ra.hms)
            cphdr['FK{}DEC'.format(idx)] = str(sky.dec.dms)
            cphdr['FK{}F'.format(idx)] = flux_fit
            # TO-DO, once have actual epsf classes will be clearer to fill the model
            cphdr['FK{}MOD'.format(idx)] = "NA"
            n += 1
            
            subbeddata = add_array(subbeddata, -psf(x, y), (y_0, x_0))
            addeddata = add_array(addeddata, psf(x, y), (y_0, x_0))
    
    # the copied hdu written/returned should have data with the added psfs 
    cphdu.data = addeddata
    # inserting some new header values
    cphdr['fakeSN']=True 
    cphdr['N_fake']=str(len(posflux))
    cphdr['F_epsf']=str(psf.flux)
    cphdu.header = cphdr
    
    if writetodisk:
        fits.writeto(saveas,cphdu.data,cphdr,overwrite=True)
    
    hdu.plants = [cphdu,posflux]
    hdu.has_fakes = True # if makes it through this plant_fakes update has_fakes

    return cphdu

def lco_epsf(hdu):
    """
    Another ePSF option besides building just use circular gaussian from
    lco header on the static sky search im
    """

    shape = (51,51)
    oversample = 2

    # LCO measures PSF stored in header
    # L1FWHM ~ Frame FWHM in arcsec, PIXSCALE ~ arcsec/pixel
    hdr = hdu.header
    l1fwhm = hdr['L1FWHM']
    pixscale = hdr['PIXSCALE']

    sigma = gaussian_fwhm_to_sigma*l1fwhm
    sigma *= 1/pixscale # to pixels
    
    constant,amplitude,xmean,ymean,xstd,ystd=0,1,shape[0]/2,shape[1]/2,sigma,sigma
    flux = 10**5 # if flux and amplitude present flux is ignored
    table = Table()
    table['constant'] = [constant]
    table['flux'] = [flux]
    table['x_mean'] = [xmean]
    table['y_mean'] = [ymean]
    table['x_stddev'] = [sigma]
    table['y_stddev'] = [sigma]
    epsfdata = photutils.datasets.make_gaussian_sources_image(shape, table,oversample=oversample)

    # make this produce a fittable2d psf model
    epsfmodel = photutils.psf.FittableImageModel(epsfdata, normalize=True)

    # TODO : we should include a normalization_correction to account for
    #  an "aperture correction" due to data outside the model

    hdu.has_lco_epsf = True # update bool if makes it through this function
    hdu.lco_epsf = epsfmodel

    return epsfmodel


def table_header(hdu,idx=0):
    """
    Make a pandas data frame out of the fits header object
    """
    hdr = hdu.header
    d = {}
    for i in hdr:
        name = str(i)
        try:
            dat = float(hdr[i])
        except:
            dat = str(hdr[i])
        d[name] = dat
    df = pd.DataFrame(data=d,index=[idx])
    return df

if __name__ == "__main__":
    phots = glob.glob("*_phot.pkl")
    sdss_matches = glob.glob("*match_sdss*")
    lco_matches = glob.glob("*match_lco*")
    assert(len(phots) == len(sdss_matches) == len(lco_matches))
    print(len(phots))
    p_stacks,s_stacks,l_stacks = [],[],[]
    for i in range(len(phots)):
        phot = phots[i]
        origname = phot.split("_phot")[0]
        sdss = [i for i in sdss_matches if origname in i]
        lco = [i for i in lco_matches if origname in i]
        p,s,l = pickle.load(open(phot,"rb")),pickle.load(open(sdss[0],"rb")),pickle.load(open(lco[0],"rb"))
        p['origname'] = origname
        s['origname'] = origname
        l['origname'] = origname
        p_stacks.append(p)
        s_stacks.append(s)
        l_stacks.append(l)
    phots=vstack(p_stacks)
    sdss_matches=vstack(s_stacks)
    lco_matches=vstack(l_stacks)
    pickle.dump(phots,open("phot.pkl","wb"))
    pickle.dump(sdss_matches,open("sdss_matches.pkl","wb"))
    pickle.dump(lco_matches,open("lco_matches.pkl","wb"))

    
    batch = glob.glob("batch*pkl")
    print(len(batch))
    dfs,idxs = [],[]
    for i in batch:
        batchidx = i.split('_')[0]
        bi = pickle.load(open(i,"rb"))
        dfs.append(bi)
        idxs.append(batchidx)
    df = pd.concat(dfs)
    df.meta = {"batchidxs":idxs}
    df.meta = {"ZPcalibrated":"sdss"}
    print(df)
    print(len(df))
    pickle.dump(df,open("efficiencies.pkl","wb"))
    
