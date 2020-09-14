import glob
import copy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 30})
from matplotlib.patches import Circle
import numpy as np
import collections

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
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

import sncosmo

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

# Suppress warnings. Relevant for astroquery. Comment this out if you wish to see the warning messages
import warnings
warnings.filterwarnings('ignore')

import roman_fakeSNpipeline
my_data = roman_fakeSNpipeline.get_data()

def IaModel(source='salt2-extended',z=None,cosmo = FlatLambdaCDM(H0=70, Om0=0.3),MR=-19.37,
                time=np.linspace(-20,100,1000),param_dict=None,
            printt=True,is_rf_time=True,do_dust=True,):
    # making a base Ia salt2-extended SN lightcurve using a rf time 20 days before peak to 100 days past peak
    # for highest z ~ 2 I think we would consider that corresponds to 60 days before peak 300 days past peak 
    # the magnitude is set using table 3 https://ui.adsabs.harvard.edu/abs/2014ApJ...783...28G/abstract
    
    # this is mainly meant to give quick lightcurve for different z you want to set
    # however if have full param dict of values you want to set can do that 
    
    # TODO dust, sncosmo sets a Rv ~ 3.1 but how do I pick a sensible ebv for mw &/or host
    # I know there are dustmap tools available for mw given an ra/dec
    # ebv = dustmap.ebv(ra, dec) ... dustmap = sfdmap.SFDMap('/Users/kyleoconnor/Documents/GitHub/sfdmap/sfddata-master/')
    # not sure if avg on this makes sense
    if z==None:
        z=1.95
    if do_dust:
        dust = sncosmo.CCM89Dust()
        model = sncosmo.Model(source=source,
                effects=[dust, dust],
                effect_names=['host', 'mw'],
                effect_frames=['rest', 'obs'])
        model.set(z=z) # optionally include a full **param_dict
    else:
        model = sncosmo.Model(source=source)
        model.set(z=z)
    
    if MR != None:
        model.set_source_peakabsmag( MR, 'bessellr', 'vega', cosmo=cosmo)

    if printt:
        print(model.parameters,model.param_names)
        
    """
    if is_rf_time:
        time*=(1+z)
    """ 
    # roman wfi is going to have 0.5 - 2.0 micron in FZYJHFW (see table 1 https://arxiv.org/pdf/1702.01747.pdf)
    # right now just going to concern myself with Y106
    Y106 = model.bandmag('f105w', 'ab', time) # assuming wfc3 f105w is similar enough bandpass
    
    return [Y106,time,z,model]


def m50():
	matplotlib.rcParams.update({'font.size': 30})
	"""
	# reminder of how m50 is determined in pipeline
	# use interp to get magnitude at which we have 50% detection efficiency 
	# need the values increasing along x for interp to work properly
	efficiencies.reverse()
	mags.reverse()
	m50 = np.interp(0.5,efficiencies,mags)
	print('m50 ~ {}'.format(m50))
	"""
	m50 = 21.65
	zs = [0.1,0.25,0.5] # even deeper? ,0.75,1,1.5]
	models = []
	for i in zs:
	    models.append(IaModel(z=i))
	time=np.linspace(-20,100,1000) # times that went in
	plt.hlines(m50,min(time),max(time),color='black',linestyle='--',label='m50 ~ {:.2f}'.format(m50))
	mintimes,maxtimes = [],[]
	for model in models:
	    Y106,time,z,model = model # unpack
	    maxtimes.append(model.maxtime()),mintimes.append(model.mintime()) # to set time axis bounds
	    plt.plot(time,Y106,label='z = {}'.format(z))
	plt.legend(loc=(1,0),fontsize=25)
	plt.ylim(25,19.5)
	plt.xlim(max(mintimes),min(maxtimes))
	plt.ylabel('mag (AB)')
	plt.xlabel('time (obs)')
	plt.savefig('m50_Ialc.pdf',bbox_inches='tight')

def psf_and_gauss():
	# take a look at the ePSF image built from stack
	fig, ax = plt.subplots(2,2,figsize=(7.5, 7.5),gridspec_kw={'width_ratios': [3, 1],'height_ratios':[3,1]})
	matplotlib.rcParams.update({'font.size': 30})
	fig.add_subplot()
	#im1 = ax[0][0].imshow(zscale(epsf.data),cmap='gray')
	# works better with a lognormalization stretch to data 
	norm = simple_norm(epsf.data, 'log')
	im1 = ax[0][0].imshow(epsf.data,norm=norm,vmin=0,cmap='viridis')
	# Adding the colorbar
	cbaxes = fig.add_axes([-0.3, 0.1, 0.03, 0.8])  # This is the position for the colorbar
	ticks = [norm.vmax,norm.vmax/10,norm.vmax/100]
	ticks.append(norm.vmin)
	#print(ticks)
	cb = plt.colorbar(im1, cax = cbaxes,ticks=ticks,format='%.1e')
	#plt.colorbar(im1,ax=ax[0][0])


	#ax.set_xlabel('',fontsize=45)
	#ax[0][0].set_xticks([])
	#ax[0][0].set_yticks([])
	ax[1][0].set_xticks([])
	ax[1][0].set_yticks([])
	ax[0][1].set_xticks([])
	ax[0][1].set_yticks([])
	#ax2 = ax[0].twinx()

	tmp=np.arange(0,epsf.shape[0])
	# vertical slice along ctr ... the x=0 gaussian fit values of epsf
	ax[1][0].plot(tmp,yctr_vals)
	ax[1][0].text(0.01, .01, r'$\sigma_x\sim{:.1f}$'.format(abs(x_stddev)), fontsize=25,rotation=0)
	# horizontal slice ... y=0 gaussian fit vals
	ax[0][1].plot(xctr_vals,tmp)
	ax[0][1].text(0.01, 45, r'$\sigma_y\sim{:.1f}$'.format(abs(y_stddev)), fontsize=25,rotation=-90)

	"""
	# contours of the epsf for levels 1,2,and3 sigma (from np.std(data)) below gaussian fit amplitude
	ax[1][1].contour(epsf.data,levels=levels)
	ax[1][1].set_xlim(23,27)
	ax[1][1].set_ylim(23,27)
	#ax[1][1].set_xticks([15,35])
	#ax[1][1].set_yticks([15,35])
	ax[1][1].yaxis.tick_right()
	"""
	# residual, gaussian model minus the effective psf it was fitting
	med=np.median(resid)
	std=np.std(resid)
	shift=np.median(resid)+3*np.std(resid)
	resid += shift # translate everything up by 3sigma above median
	norm=simple_norm(resid,'log')
	print(med,std,shift,norm.vmin,norm.vmax)
	# use vmin of 0, the 3 sigma shift upwards should have the meaningful negative values above zero
	vmin=0
	print(vmin,vmax)
	im2 = ax[1][1].imshow(resid,origin='lower',cmap='viridis',norm=norm,vmin=vmin,vmax=vmax)
	ticks = [norm.vmin,norm.vmax,shift]

	cbaxes = fig.add_axes([1, 0.1, 0.03, 0.2])  # This is the position for the colorbar
	cb = plt.colorbar(im2,cax=cbaxes,ticks=ticks)

	plt.savefig('Roman_psf.pdf',bbox_inches='tight')

def used_stars():
	# can either show the 'good' stars ie those used to build the epsf, or using i.compute_residual_image(epsf) to show how well the epsf fit each
	matplotlib.rcParams.update({'font.size': 30})
	tmp = fitted_stars.all_good_stars
	print(len(tmp))
	nrows,ncols=int(np.sqrt(len(tmp)))+1,int(np.sqrt(len(tmp)))
	fig, ax = plt.subplots(nrows,ncols,figsize=(7.5, 7.5))
	ax=ax.ravel()
	for i in range(len(tmp)):
	    ax[i].imshow(zscale(tmp[i]))
	plt.savefig('Roman_stars.pdf',bbox_inches='tight')

def detection_efficiency():
	matplotlib.rcParams.update({'font.size': 30})
	# prelim results
	mags,efficiencies=[20.41,20.91,21.41,21.91,22.41,22.91,23.41,23.91,24.41],[1,1,0.849,0.135,0.022,.0128,.0125,.0126,.0126]
	
	fig, ax = plt.subplots(1,1,figsize=(5, 5))
	fig.add_subplot()
	ax.plot(mags,efficiencies,marker='o')
	ax.title.set_text('Detection Efficiency')
	ax.set_xlabel('mag',fontsize=45)
	ax.set_xticks([20,22,24])
	ax.set_yticks([0,0.25,0.5,0.75,1])
	plt.savefig('Roman_detection_efficiency.pdf',bbox_inches='tight')

def lattice_planted():
	matplotlib.rcParams.update({'font.size': 30})
	# get a look at grid of SNe from clearly visible high detection rate to un-detected
	position,size=(2000,2000),300 # need to zoom in on the figures to get clean look at the planted SNe
	#Cutout2D(data,position,size)
	cutmag1=Cutout2D(fits.open('planted_lattice_mag20.41.fits')[0].data,position,size)
	cutmag2=Cutout2D(fits.open('planted_lattice_mag20.91.fits')[0].data,position,size)
	cutmag3=Cutout2D(fits.open('planted_lattice_mag21.41.fits')[0].data,position,size)
	cutmag4=Cutout2D(fits.open('planted_lattice_mag21.91.fits')[0].data,position,size)

	fig, ax = plt.subplots(2,2,figsize=(10, 10))
	ax[0][0].imshow(zscale(cutmag1.data),cmap='gray')
	ax[0][1].imshow(zscale(cutmag2.data),cmap='gray')
	ax[1][0].imshow(zscale(cutmag3.data),cmap='gray')
	ax[1][1].imshow(zscale(cutmag4.data),cmap='gray')
	ax[0][0].title.set_text('mag=20.41')
	ax[0][1].title.set_text('20.91')
	ax[1][0].title.set_text('21.41')
	ax[1][1].title.set_text('21.91')
	ax[0][0].set_xticks([])
	ax[0][0].set_yticks([])
	ax[1][0].set_xticks([])
	ax[1][0].set_yticks([])
	ax[0][1].set_xticks([])
	ax[0][1].set_yticks([])
	ax[1][1].set_xticks([])
	ax[1][1].set_yticks([])

	plt.savefig('Roman_plants.pdf',bbox_inches='tight')
