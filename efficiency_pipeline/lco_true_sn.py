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

from lco_figures import *
from lco_fakeSNpipeline import *

# Suppress warnings. Relevant for astroquery. Comment this out if you wish to see the warning messages
import warnings
warnings.filterwarnings('ignore')

# lco image on ptf alert to real SN 
ztf20abqgzhh_data,path = {}, '/work/oconnorf/efficiency_pipeline/lco/ZTF20abqgzhh' 
output = os.path.join(path,'output')

DIA_IN=glob.glob(os.path.join(path,'DIA_IN/*fits'))
for i in range(len(DIA_IN)):
	filename=DIA_IN[i].split('/')[-1]
	ztf20abqgzhh_data[filename] = fits.open(DIA_IN[i])[0]


# lets measure the epsf and extract SN from each of images
for file in DIA_IN:
	filename=file.split('/')[-1]
	image = ztf20abqgzhh_data[filename]
	hdr = image.header
	groupid,L1fwhm,pixscale,skybr = hdr['GROUPID'],hdr['L1fwhm'],hdr['pixscale'],hdr['WMSSKYBR'] # pixels, arcsec/pixels,mag/arcsec^2
	med,exptime = hdr['L1MEDIAN'],hdr['EXPTIME']
	zp=skybr+2.5*np.log10(med/exptime/pixscale)
	print('filename ~ {} (groupid {}) has L1fwhm ~ {} pixels, pixscale ~ {} arcsec/pixel, and skybr {} mag/arcsec^2; zp ~ {}'.format(filename,groupid,L1fwhm,pixscale,skybr,zp))
	pickle_to = output + '/' + filename[:-5] # -5 get rid of .fits

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

	# target obj work, tuples cutting boxes around target (data,patch)
	# also returns targ_obj again account for updates using ref (in the cases where empty targ_obj ie not detected in source)
	target_boxes = target(image,targ_obj) 
	targ_obj,cut_targ,bkg_core,bkg_1,bkg_2 = target_boxes # unpacked

	# grab target parameters
	equivalent_radius = targ_obj['equivalent_radius'][0].value
	xy = (targ_obj['xcentroid'][0].value,targ_obj['ycentroid'][0].value) 
	semimajor_axis, semiminor_axis = targ_obj['semimajor_axis_sigma'][0].value,targ_obj['semiminor_axis_sigma'][0].value
	orientation = targ_obj['orientation'][0].value 
	
	# cut around the image on target (already available/should be same as the cuts provided but doing using image provided so easy to understand in script)
	cut_im = Cutout2D(image.data,xy,equivalent_radius*5) 
	fig, ax = plt.subplots(1,1,figsize=(7.5, 7.5))
	norm = simple_norm(cut_im.data, 'log')
	im1 = ax.imshow(cut_im.data,norm=norm,vmin=np.min(cut_im.data),cmap='viridis')
	ticks = [norm.vmax,norm.vmax/10,norm.vmax/100,norm.vmin]
	#ticks.append(norm.vmin)
	#print(ticks)
	cb = plt.colorbar(im1,ticks=ticks,format='%.1e')
	plt.savefig(pickle_to+'_SN.pdf',bbox_inches='tight')


