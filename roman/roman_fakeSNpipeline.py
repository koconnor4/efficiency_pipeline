import glob
import copy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 30})
from matplotlib.patches import Circle
import numpy as np
import itertools
import collections

import astropy
from astropy.io import ascii,fits
from astropy.table import vstack,Table,Column,Row,setdiff,join
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.units import Quantity
from astropy.convolution import Gaussian2DKernel
from astropy.visualization import ZScaleInterval,simple_norm
zscale = ZScaleInterval()
from astropy.nddata import Cutout2D,NDData
from astropy.stats import sigma_clipped_stats,gaussian_fwhm_to_sigma
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
from photutils import Background2D, MedianBackground
bkg_estimator = MedianBackground()
from photutils import BoundingBox



def get_data():
	ims=glob.glob('*fits')
	my_data = {} 
	for i in range(len(ims)):
		filename = ims[i].split('/')[-1]
		my_data[filename] = fits.open(ims[i])[0]
	print(my_data)
	return my_data

# source cat, provide a clean fits image good for detections
# needs to be drz to same pixscale as whatever image eventually plant SN into
# shouldn't need to be aligned with the image eventually place SN into, rotated psf doesn't really matter   
# for consistency I'd suggest the drz reg image of whichever epoch you plant into
def source_cat(image):
	"""
	the image should be fits.open('image.fits')
	will get a cat of properties for detected sources, also returns image, threshold, and segmentation image
	"""
	# to be able to translate from ra/dec <--> pixels on image
	wcs,frame = WCS(image.header),image.header['RADESYS'].lower()

	# bkg
	bkg = Background2D(image.data, (50, 50), filter_size=(3, 3),bkg_estimator=bkg_estimator)
	#threshold = bkg.background + (2. * bkg.background_rms) # I re-calculate this below, either is fine
	# detect threshold makes image array that sets bar for identifying pixels which might be sources
	# uses sigma clipped statistics to get bkg flux and std
	# thresholding needs sources to be above bkg + nsigma  
	threshold = detect_threshold(image.data, nsigma=2.) # returns as an image array
	sigma = 3.0 * gaussian_fwhm_to_sigma  # FWHM = 3.
	# optional ~ kernel smooths the image, using gaussian weighting with pixel size of 3
	kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
	kernel.normalize()
	# make a segmentation map, id sources defined as n connected pixels above threshold (nsigma*bkg)
	segm = detect_sources(image.data,
						  threshold, npixels=5, filter_kernel=kernel)
	# deblend useful for very crowded image with many overlapping objects...
	# uses multi-level threshold and watershed segmentation to sep local peaks as ind obj
	# use the same number of pixels and filter as was used on original segmentation
	# contrast is fraction of source flux local pk has to be consider its own obj
	segm_deblend = deblend_sources(image.data, 
								   segm, npixels=5,filter_kernel=kernel, 
								   nlevels=32,contrast=0.001)

	# need bkg subtracted to do photometry using source properties
	data_bkgsub = image.data - bkg.background # wfirst has no significant bkg, this isn't really important
	# negative counts commonly occur because sky is so dark, they come from readout bias
	# if get an issue here I had background as threshold before and it worked
	cat = source_properties(data_bkgsub, segm_deblend,background=bkg.background,
							error=None,filter_kernel=kernel)

	"""
	# if you want to id a specific target in the source catalog
	# since this is ideal detection location where strong lens could provide multi-im
	# this is going to be area where we will most want to plant and study 
	
	#CAT-RA  = 'blah'       / [HH:MM:SS.sss] Catalog RA of the object        
	#CAT-DEC = 'blah'       / [sDD:MM:SS.ss] Catalog Dec of the object
	ra = image.header['CAT-RA']
	dec = image.header['CAT-DEC']
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
	#print(targ_obj)
	targ_sb = targ_obj['source_sum']/targ_obj['area']
	"""
	print('{} sources detected in image {} threshold ~ {} [{}]; (2sigma above bkg)'.format(len(cat),image.header['rootname'],np.median(threshold),image.header['BUNIT']))

	return cat,image,threshold,segm_deblend

 
def stars(image,cat):
	# provide the drz reg fits image of epoch you want to plant into and catalog of detected sources from above  
	# bright unsaturated stars are determined and extracted from image to be ready for use in PSF
	# human checking the stars chosen here to make sure got good sample would be starting point if something off with planted SN ie ePSF looks strange
	
	tbl = cat.to_table()
	print('there are {} sources available within the cat'.format(len(tbl)))

	# extract stars wants x and y column, rather than rename will just add copies
	x = Column([i.value for i in tbl['xcentroid']])
	y = Column([i.value for i in tbl['ycentroid']])
	tbl.add_column(x,name='x')
	tbl.add_column(y,name='y')
	# going to also include a column of surface brightness 
	surf_brightnesses = []
	for obj in tbl:
		unit = 1/obj['area'].unit
		surf_bright = obj['source_sum']/obj['area'].value # flux/pix^2
		surf_brightnesses.append(surf_bright) 
	surf_brightnesses = Column(surf_brightnesses,name='surface_brightness',unit=unit)
	tbl.add_column(surf_brightnesses)

	# finding bboxes around all detected objects of the size I am going to use in extractions
	# I want to remove any stars with overlap on any obj detected in image 
	# Isolated is first key to good stars for epsf before remove using photometry constraints 
	bboxes = []
	for i in tbl:
		x = i['xcentroid'].value
		y = i['ycentroid'].value
		size = 25
		ixmin,ixmax = int(x - size/2), int(x + size/2)
		iymin, iymax = int(y - size/2), int(y + size/2)

		bbox = BoundingBox(ixmin=ixmin, ixmax=ixmax, iymin=iymin, iymax=iymax)
		bboxes.append(bbox)
	bboxes = Column(bboxes)
	tbl.add_column(bboxes,name='bbox')
	# using the bbox of extraction size around each obj detected to determine intersections, dont want confusion of multi-stars for ePSF
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
	for i in tbl:
		if i['bbox'] in intersections:
			#tmp.remove(i)
			row=j
			rows.append(row)
		j+=1
	tbl.remove_rows(rows)
	print('{} isolated sources, after removing intersections'.format(len(tbl)))

	# use ellipticity to get star-like objs 
	stars_tbl = tbl[tbl['ellipticity']<0.05]
	print('{} stars, after using ellipticity'.format(len(stars_tbl)))

	mean,med,bkgsigma = sigma_clipped_stats(image.data)
	hiSNstars = stars_tbl[stars_tbl['max_value']>bkgsigma*100] #need obj max value > 100*bkgsigma ~ the standard deviation in bkg values
	print('{} stars, after using high signal/noise constraint this step can be improved once have saturation and non-lin levels')

	# recall bkg here is insignificant ~ 0
	data = image.data # - bkg.background
	# in general though want bkg subtracted to extract stars, build ePSF using just star brightness 
	bkg = Background2D(data, (50, 50), filter_size=(3, 3),bkg_estimator=bkg_estimator)
	# basically a fancy subregion application of mean_val, median_val, std_val = sigma_clipped_stats(data, sigma=2.) 
	nddata = NDData(data=data)
	stars = extract_stars(nddata,catalogs=hiSNstars, size=25)
	"""
	# you should look at the images to make sure these are good stars
	nrows = 4
	ncols = 4
	fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20),
							squeeze=True)
	ax = ax.ravel()
	for i in range(len(brightest_results)):
		norm = simple_norm(stars[i], 'log', percent=99.)
		ax[i].imshow(stars[i], norm=norm, origin='lower', cmap='viridis'
	"""
	return stars

"""
Best paper to read about EPSF that these photutil functions are based on: https://iopscience.iop.org/article/10.1086/316632/pdf
Was written to do astrometry on undersampled HST images (wf camera and planetary camera) whereas I'm using here on oversampled
principle is the same though the positions (and/or shape) can be improved through iteration between epsf model and image
the paramspace (ctr_location and/or pixfraction of flux) is sampled (starting around some base initial location/smooth model) and residual taken w sources in image
if params improved the model (smaller residual), update and iterate...

much quicker/easier on already oversampled image like we have here since have solid foundation for ePSF from the get go
not relying on statistics from number of sources the whole time ~ ie the grid oversampling and kernel smoothing getting small tail overlaps to neighbor pix 
"""
def ePSF(stars,oversampling=2):
	# provide the good_stars determined in image to build an effective point spread function, this should be pretty consistent for wfirst images
	# effective means that the flux of the psf is scaled to 1, each pixel value indicates fraction of the flux 
	# oversampling chops pixels of each star up further to get better fit
	# more oversampled the ePSF is, the more stars you need to get smooth result
	epsf_builder = EPSFBuilder(oversampling=oversampling, maxiters=10,
								progress_bar=True)  
	epsf, fitted_stars = epsf_builder(stars)  
	"""
	# take a look at the ePSF image 
	norm = simple_norm(epsf.data, 'log', percent=99.)
	plt.imshow(epsf.data, norm=norm, origin='lower', cmap='viridis')
	plt.colorbar()
	# or take a look at the residual of the ePSF registered to the individual stars ~ good to id issue source if seems to be a problem
	# for i in stars...
	res_im = stars[i].compute_residual_image(epsf)
	plt.imshow(zscale(res_im))
	percentdiff = np.sum(res_im)/np.sum(good_stars[i].data) 
	"""
	return epsf, fitted_stars

from photutils.datasets import make_gaussian_sources_image

def gaussian2d(epsf):
	# use photutils 2d gaussian fit on the epsf
	gaussian = photutils.centroids.fit_2dgaussian(epsf.data)
	print('gaussian fit to epsf:',gaussian)
	print(gaussian.param_names,gaussian.parameters)
	# unpack the parameters of fit
	constant,amplitude,x_mean,y_mean,x_stddev,y_stddev,theta=gaussian.parameters
	
	# going to print approx resolution determined by epsf (according to average of the fits along x,y)
	# we have some rough idea of what the fwhm is predicted so is good check
	sigma = (abs(x_stddev)+abs(y_stddev))/2 # the average of x and y 
	sigma*=1/epsf.oversampling[0] # the model fit was to oversampled pixels need to correct for that for true image pix res
	fwhm = gaussian_sigma_to_fwhm*sigma
	print('fwhm ~ {} pixels'.format(fwhm))
	print('fwhm predicted ~ 1.86 pixels in Hounsell 19 https://arxiv.org/pdf/1702.01747.pdf (assumed 110mas/pix on wfc)')
	print('hopefully we are consistent or slightly improved resolution since the drz from 100mas sim -> 90mas Caleb made')
	
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
	return levels,xctr_vals,yctr_vals,image1,img_epsf,resid
	#return image1
	"""
	# we don't want noise for our planted psf or gaussian model, in general though could add like below
	image2 = image1 + make_noise_image(shape, distribution='gaussian',
								   mean=5., stddev=5.)
	image3 = image1 + make_noise_image(shape, distribution='poisson',
								   mean=5.)
	"""

def plant(image,psf,source_cat,mag=None,location=None,zp=26.41,plantname='planted.fits'):
	"""
	image should be fits.open('image.fits'), will add SN directly to here
	psf should be the epsf we made above
	source_cat is the catalog,image,threshold,segmentation, use the threshold to define a dim barely detectable obj
	location should be a Skycoord(ra,dec) or if left as None will use hdr crval to place (near ctr of image)
	mag,zp (TODO get phot zp), if none provided will use instrumental mag (zp ~ 0) and plant an obj 2 mags brighter than dim obj
	"""

	# unpack source_cat into the catalog of objs, reg ep im searched, threshold unit for source det, and segmentation image 
	cat,reg,threshold,segm=source_cat

	exptime=image.header['EXPTIME']
	# don't have pixscale in hdr, these images maybe rotated
	cd1_1=image.header['CD1_1']
	cd2_2=image.header['CD2_2']
	cd1_2=image.header['CD1_2']
	cd2_1=image.header['CD2_1']
	pixscale=np.sqrt(cd1_1**2 + cd2_2**2)/np.sqrt(2) # deg/pixel
	data = image.data # BUNIT e-/s 
	
	# image bkg values from sigma clipped stats
	mean_val, median_val, std_val = sigma_clipped_stats(data, sigma=2.)  
	dim = np.median(threshold) # the dimmest thing we called detected source (2sigma above bkg)
	# bkg mag ~ limit of detection around the threshold value, ie 2sigma above bkg
	dim_mag = -2.5*np.log10(dim)
	print('the threshold ~ {} [e-/s], set as dim mag (barely detectable)'.format(dim))
	
	# DEFAULT 26.41 ~ Y106 AB Zero-point Table 1: Hounsell et al 19 https://arxiv.org/pdf/1702.01747.pdf
	if zp==None:
		# The PHOT keywords for physical magnitude (Vega, ST, AB) not available in hdr
		# 28.1 ~ sensitivity f106 (5sigma ABmag in 1h) https://www.stsci.edu/roman/observatory
		# or use instrumental mag, calling zp ~ 1 e-/s
		zp = 2.5*np.log10(1) # ie zp = 0
	
	if mag==None:
		# if don't tell it what mag we want SN, I'll make it 2 mags brighter than the magnitude of threshold for detection
		# remember rn using instrumental mags so should be seeing around -2 mag range for barely visible (the 1 e-/s zp is close to real bkg of dim mag)
		mag = dim_mag+zp-2
	
	# copying image and psf so can leave original data untouched
	cpim = copy.copy(image.data)
	mu = 10**((mag-zp)/-2.5) # the factor to multiply psf flux, to achieve given mag 
	print('mu ',mu)
	cppsf = copy.copy(psf.data*mu) 

	wcs,frame = WCS(image.header),image.header['RADESYS'].lower()
	hdr = image.header
	lattice = False 
	if location==None:
		# use the hdr to place SN
		x = hdr['CRPIX1']
		y = hdr['CRPIX2']
		pix = [x,y]
		revpix = copy.copy(pix)
		revpix.reverse()
		location=astropy.wcs.utils.pixel_to_skycoord(pix[0],pix[1],wcs)
	elif type(location)==tuple:
		# lattice was used to generate tuple w lists (skycoords,pixels), we want to plant many SNe across the image
		lattice =location
		# unpack the lists of lattice
		locations,pixels = lattice
		lattice = True 
	else:
		# give arb skycoord loc (ra/dec) and translate to pixels for plant
		pix=astropy.wcs.utils.skycoord_to_pixel(location,wcs) # x,y pixel location
		revpix = copy.copy(list(pix)) # row,col location for adding to data... y,x
		revpix.reverse()
	
	if lattice:
		# many locations planting SNe all across image
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
			image.data[rows[:, None], cols] += cppsf
			np.float64(image.data)
		# inserting True fakeSN into hdr w the pix location
		hdr = copy.copy(image.header)
		hdr['fakeSN']=True 
		hdr['fakeSN_loc']='lattice'	
		hdr['NfakeSNe']=str(len(pixels))
		hdr['fakeSNmag']=str(mag)
		hdr['fakeZP']=str(zp)
		fits.writeto(plantname,image.data,hdr,overwrite=True)
		print('{} SNe mag ~ {} planted in lattice across image {} written to {}; zp ~ {}'.format(len(pixels),mag,image.header['rootname'],plantname,zp))
		plant_im = fits.open(plantname)[0]  
		return plant_im,pixels

	else:
		# single location either provided in skycoord or assigned using crval of hdr (ctr)
		# indexes to add the psf to
		row,col=revpix
		nrows,ncols=cppsf.shape
		# +2 in these to grab a couple more than needed, the correct shapes for broadcasting taken using actual psf.shapes
		rows=np.arange(int(np.round(row-nrows/2)),int(np.round(row+nrows/2))+2) 
		cols=np.arange(int(np.round(col-ncols/2)),int(np.round(col+ncols/2))+2) 
		rows = rows[:cppsf.shape[0]]
		cols = cols[:cppsf.shape[1]]
		image.data[rows[:, None], cols] += cppsf
		np.float64(image.data)
		# write the image with planted SN added to a new fits file (inserting True fakeSN into hdr)
		hdr = copy.copy(image.header)
		hdr['fakeSN']=True 
		hdr['fakeSN_loc']=str(pix)
		hdr['NfakeSNe'] = str(1)
		hdr['fakeSNmag']=str(mag)
		hdr['fakeZP']=str(zp)
		fits.writeto(plantname,image.data,hdr,overwrite=True)
		print('SN mag ~ {} planted in image {} at {} [{}] written to {}; zp ~ {}'.format(mag,image.header['rootname'],location,pix,plantname,zp))
		plant_im = fits.open(plantname)[0]  
		return plant_im


def lattice(image):
	hdr = image.header
	wcs,frame=WCS(hdr),hdr['RADESYS'].lower()
	# have single 4kx4k chip from wide field instrument
	NX = hdr['NAXIS1']
	NY = hdr['NAXIS2']
	edge = 100 # pixels away from edge
	spacing = 50 # pixels between each location on lattice
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
    catalog,image,threshold,segm = cat
    hdr=image.header
    Nfakes=hdr['NfakeSNe']
    magfakes=hdr['fakeSNmag']
    print('Nfakes ~ {} (= {} quick sanity check) planted in this image'.format(Nfakes,len(pixels)))
    print('Nsources ~ {} detected in the image'.format(len(catalog)))
    
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
    print('{} source detections within search radius criteria'.format(len(truths)))
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
    print('{} planted SNe had single clean source detected, {} planted SNe had multiple sources detected nearby'.format(len(single_plant),len(repeat_plant)))
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
    single_truth_tbl = vstack(single_truth_tbl)
    repeat_truth_tbl = vstack(repeat_truth_tbl)
                    
    print('Final: {} planted SNe, {} clean single detections, {} as multi-sources near a plant, {} false detections'.format(Nfakes,len(single_truth_tbl),len(repeat_truth_tbl),len(false_tbl)))
    
    efficiency = N_plants_detected/len(pixels)

    print('Detection efficiency (N_plants_detected/N_plants including multiple detections) ~ {} on mag ~ {} SNe'.format(efficiency,magfakes))
    return efficiency,magfakes,tbl,single_truth_tbl,repeat_truth_tbl,false_tbl

def roman_pipe():
	my_data = get_data()
	image = my_data['Caleb Duff - bright_f105w_e12_reg_drz_sci.fits']
	source_catalog = source_cat(image)
	# unpack to make a little clearer
	cat,image,threshold,segm = source_catalog 
	good_stars = stars(image,cat,Nbrightest=50)
	epsf,fitted_stars = ePSF(good_stars)
	# switch up image so that plant into the difference, don't need to run it through sndriz processing 
	image = my_data['Caleb Duff - bright_f105w_e12-e11_sub_masked.fits']
	locations = lattice(image)
	zp = 26.41 # Y106 AB Zero-point Table 1: Hounsell et al 19 https://arxiv.org/pdf/1702.01747.pdf
	mags = np.arange(zp-6,zp+2,0.5)
	efficiencies = []
	for mag in mags:
		my_data = get_data()
		image = my_data['Caleb Duff - bright_f105w_e12-e11_sub_masked.fits']
		plantname = 'planted_lattice_mag{}.fits'.format(str(mag))
		planted = plant(image,epsf,source_catalog,mag=mag,location=locations,zp=26.41,plantname=plantname)
		plant_im,pixels = planted # unpack
		fakesource_cat = source_cat(plant_im)
		tmp = detection_efficiency(planted,fakesource_cat)
		efficiencies.append(tmp)
		print(tmp)
	print('efficiencies: {}'.format(efficiencies))
	# use interp to get magnitude at which we have 50% detection efficiency 
	# need the values increasing along x for interp to work properly
	efficiencies.reverse()
	mags.reverse()
	m50 = np.interp(0.5,efficiencies,mags)
	print('m50 ~ {}'.format(m50))

if __name__=="__main__":
	print('roman pipe')
	roman_pipe()