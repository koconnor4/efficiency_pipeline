import glob
import copy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 30})
from matplotlib.patches import Circle
import numpy as np

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
from photutils import Background2D, MedianBackground
bkg_estimator = MedianBackground()

# Suppress warnings. Relevant for astroquery. Comment this out if you wish to see the warning messages
import warnings
warnings.filterwarnings('ignore')

import lco_fakeSNpipeline

def psf_and_gauss(epsf,epsf_gaussian,saveas='lco_psf.pdf'):
	# take a look at the ePSF image built from stack and a fitted gaussian 
	fit_gaussian,levels,xctr_vals,yctr_vals,image1,img_epsf,resid = epsf_gaussian # unpacked... levels list amplitude - sigma, ctr vals are gauss model sliced, image1 is array of values from gaussian fit in shape of epsf, img_epsf is epsf instance of it, resid is gauss - epsf 
	constant,amplitude,x_mean,y_mean,x_stddev,y_stddev,theta=fit_gaussian.parameters
	matplotlib.rcParams.update({'font.size': 30})

	fig, ax = plt.subplots(2,2,figsize=(7.5, 7.5),gridspec_kw={'width_ratios': [3, 1],'height_ratios':[3,1]})
	#fig.add_subplot()
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

	tmp=np.arange(0,epsf.shape[0])
	# vertical slice along ctr ... the x=0 gaussian fit values of epsf
	ax[1][0].plot(tmp,yctr_vals)
	ax[1][0].text(0.01, .01, r'$\sigma_x\sim{:.1f}$'.format(abs(x_stddev)), fontsize=25,rotation=0)
	# horizontal slice ... y=0 gaussian fit vals
	ax[0][1].plot(xctr_vals,tmp)
	ax[0][1].text(0.01, 45, r'$\sigma_y\sim{:.1f}$'.format(abs(y_stddev)), fontsize=25,rotation=-90)

	#ax.set_xlabel('',fontsize=45)
	#ax[0][0].set_xticks([])
	#ax[0][0].set_yticks([])
	ax[1][0].set_xticks([])
	ax[1][0].set_yticks([])
	ax[0][1].set_xticks([])
	ax[0][1].set_yticks([])
	#ax2 = ax[0].twinx()
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
	print('Residual median {}, std {} , shift {}'.format(med,std,shift))
	print('lognorm on shifted residual: norm.vmin {}, norm.vmax {}'.format(norm.vmin,norm.vmax))
	# use vmin of 0, the 3 sigma shift upwards should have the meaningful negative values above zero
	vmin,vmax=0,norm.vmax
	print('vmin,vmax used in resid lognorm imshow',vmin,vmax)
	im2 = ax[1][1].imshow(resid,origin='lower',cmap='viridis',norm=norm,vmin=vmin,vmax=vmax)
	ticks = [norm.vmin,norm.vmax,shift]

	cbaxes = fig.add_axes([1, 0.1, 0.03, 0.2])  # This is the position for the colorbar
	cb = plt.colorbar(im2,cax=cbaxes,ticks=ticks)

	plt.savefig(saveas,bbox_inches='tight')


def used_stars(fitted_stars,saveas='lco_stars.pdf'):
	# can either show the 'good' stars ie those used to build the epsf, or using i.compute_residual_image(epsf) to show how well the epsf fit each
	matplotlib.rcParams.update({'font.size': 15})
	tmp = fitted_stars.all_good_stars
	print(len(tmp))
	nrows,ncols=int(np.sqrt(len(tmp)))+1,int(np.sqrt(len(tmp)))+1
	fig, ax = plt.subplots(nrows,ncols,figsize=(7.5, 7.5))
	ax=ax.ravel()
	for i in range(len(tmp)):
		ax[i].imshow(zscale(tmp[i]))
	plt.savefig(saveas,bbox_inches='tight')

def detection_efficiency(mags,efficiencies,m50,saveas='lco_detection_efficiency.pdf'):
	# prelim results
	#mags,efficiencies=[20.41,20.91,21.41,21.91,22.41,22.91,23.41,23.91,24.41],[1,1,0.849,0.135,0.022,.0128,.0125,.0126,.0126]
	matplotlib.rcParams.update({'font.size': 30})
	fig, ax = plt.subplots(1,1,figsize=(5, 5))
	#fig.add_subplot()
	ax.plot(mags,efficiencies,marker='o')
	ax.title.set_text('Detection Efficiency')
	ax.set_xlabel('mag',fontsize=45)
	tmp = [m50-2,m50,m50+2]
	xticks = [float("{:.1f}".format(i)) for i in tmp]
	ax.set_xticks(xticks)
	ax.set_yticks([0,0.25,0.5,0.75,1])
	plt.vlines(m50,0,1,linestyle='--',color='black',label='m50 ~ {:.2f}'.format(m50))
	plt.legend(bbox_to_anchor=(1,-0.25))
	plt.savefig(saveas,bbox_inches='tight')

def lattice_planted(mags,m50,pickle_to,saveas='lco_plants.pdf'):
	# get a look at grid of SNe from clearly visible high detection rate to un-detected
	position,size=(1200,1200),450 # need to zoom in on the figures to get clean look at the planted SNe
	#Cutout2D(data,position,size)
	print(mags,m50)
	# get idx of planted mags that is nearest to the m50
	idx = min(range(len(mags)), key=lambda i: abs(mags[i]-m50))
	cutmag1=Cutout2D(fits.open(pickle_to+'_planted_lattice_mag{}.fits'.format(str(mags[idx+1])))[0].data,position,size)
	cutmag2=Cutout2D(fits.open(pickle_to+'_planted_lattice_mag{}.fits'.format(str(mags[idx])))[0].data,position,size)
	cutmag3=Cutout2D(fits.open(pickle_to+'_planted_lattice_mag{}.fits'.format(str(mags[idx-1])))[0].data,position,size)
	cutmag4=Cutout2D(fits.open(pickle_to+'_planted_lattice_mag{}.fits'.format(str(mags[idx-2])))[0].data,position,size)

	matplotlib.rcParams.update({'font.size': 30})
	fig, ax = plt.subplots(2,2,figsize=(10, 10))
	ax[0][0].imshow(zscale(cutmag1.data),cmap='gray')
	ax[0][1].imshow(zscale(cutmag2.data),cmap='gray')
	ax[1][0].imshow(zscale(cutmag3.data),cmap='gray')
	ax[1][1].imshow(zscale(cutmag4.data),cmap='gray')
	ax[0][0].title.set_text(str(mags[idx+1]))
	ax[0][1].title.set_text(str(mags[idx]))
	ax[1][0].title.set_text(str(mags[idx-1]))
	ax[1][1].title.set_text(str(mags[idx-2]))
	ax[0][0].set_xticks([])
	ax[0][0].set_yticks([])
	ax[1][0].set_xticks([])
	ax[1][0].set_yticks([])
	ax[0][1].set_xticks([])
	ax[0][1].set_yticks([])
	ax[1][1].set_xticks([])
	ax[1][1].set_yticks([])

	plt.savefig(saveas,bbox_inches='tight')

def target_image(image,target,saveas='target_image.pdf'):
    # take useful targ_obj values; comes from source_cat, is the photutils for the galaxy object
    # pixels and deg, sums ~ brightness in adu ~ for lco is straight counts (ie not yet rate isn't /exptime)
    targ_obj,cuts,bkg_core,bkg_1,bkg_2 = target # unpack
    cut_targ,cut_diff,cut_ref = cuts # assume target was provided diff and ref 
    
    (cut_core,box_core),(cut_1,box_1),(cut_2,box_2)=bkg_core,bkg_1,bkg_2 # unpack again

    # grab target parameters
    equivalent_radius = targ_obj['equivalent_radius'][0].value
    xy = (targ_obj['xcentroid'][0].value,targ_obj['ycentroid'][0].value) 
    semimajor_axis, semiminor_axis = targ_obj['semimajor_axis_sigma'][0].value,targ_obj['semiminor_axis_sigma'][0].value
    orientation = targ_obj['orientation'][0].value 
    
    # cut around the image on target (already available/should be same as the cuts provided but doing using image provided so easy to understand in script)
    cut_im = Cutout2D(image.data,xy,equivalent_radius*5) 
    cut_xy = cut_im.center_cutout

    ellipse = matplotlib.patches.Ellipse(cut_xy,semimajor_axis,semiminor_axis,angle=orientation,fill=None)

    # bkg_i need to be re-calculated there is an error otherwise (I think due to passing mpl patch as kwarg)

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
    
    """
    # to take a look at the ellipse that photutils found
    fig,ax=plt.subplots()
    ellipse = matplotlib.patches.Ellipse(cut_xy,semimajor_axis,semiminor_axis,angle=orientation,fill=None)
    ax.imshow(zscale(cut_targ.data))
    ax.add_patch(patch)
    """
    fig,ax=plt.subplots(2,2,figsize=(10,10))
    ax[0][0].imshow(zscale(cut_im.data))
    ax[0][0].add_patch(box_core)
    ax[0][0].add_patch(box_1)
    ax[0][0].add_patch(box_2)
    # do some markers showing core ctr and shift along orientation 
    xrange,yrange=np.linspace(cut_xy[0],cut_xy[0]+shift_x),np.linspace(cut_xy[1],cut_xy[1]+shift_y)
    shift0 = ax[0][0].scatter(xrange,yrange,s=equivalent_radius**2,marker='.',color='black')
    core0 = ax[0][0].scatter(cut_xy[0], cut_xy[1],s=2*equivalent_radius**2, marker="*",color='white')
    # the zoom in on on the cuts
    fcore = ax[0][1].imshow(zscale(cut_core.data),label='fcore')
    f1 = ax[1][0].imshow(zscale(cut_1.data),label='f1')
    f2 = ax[1][1].imshow(zscale(cut_2.data),label='f2')
    # legend detailing that the boxes are rectangular lxw = req and have three which start at ctr and shift req along theta 
    ax[0][0].legend([shift0,core0],['$r_{eq}(\Theta$)','core'])
    # text that shows the boxes pixel sum, area, and flux ~ sum/area/exptime ... todo eventually get to mag/arcsec^2 and/or nsigma above sky bkg
    area = equivalent_radius**2 
    try:
    	exptime = image.header['exptime']
    except:
    	# ref doesnt have the good stuff in hdr
    	exptime = 300
    ax[0][1].text(0.1, 0.8, 'fcore ~ {:.1e} adu/s/pix^2'.format(np.sum(cut_core.data)/exptime/area),
        bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10},transform=ax[0][1].transAxes,
                 verticalalignment='bottom', horizontalalignment='left',
        color='black', fontsize=15)
    ax[1][0].text(0.1, 0.8, 'f1 ~ {:.1e} adu/s/pix^2'.format(np.sum(cut_1.data)/exptime/area),
        bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10},transform=ax[1][0].transAxes,
                 verticalalignment='bottom', horizontalalignment='left',
        color='black', fontsize=15)
    ax[1][1].text(0.1, 0.8, 'f2 ~ {:.1e} adu/s/pix^2'.format(np.sum(cut_2.data)/exptime/area),
        bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10},transform=ax[1][1].transAxes,
                 verticalalignment='bottom', horizontalalignment='left',
        color='black', fontsize=15)
    ax[0][1].text(1.1,1.01,'fcore ~ $\sum_i p_i/exptime/area $ \n area ~ $r_{eq}^2 $',
                 bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10},transform=ax[1][1].transAxes,
                 verticalalignment='top', horizontalalignment='left',
        color='black', fontsize=15)
    
    # todo get r_eq formatted to show getting key error because of {eq}, similar include theta
    #plt.show()
    print('saveas has multiple arguments?',saveas)
    plt.savefig(saveas,bbox_inches='tight')
    plt.close('all')


