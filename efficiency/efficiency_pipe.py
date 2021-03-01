import os 
import glob
import sys
from optparse import OptionParser
parser = OptionParser()
(options,args)=parser.parse_args()

from astropy.table import Table,Column,Row,vstack,setdiff,join
from astropy.io import ascii,fits
import pandas as pd
import numpy as np

from photutils import DAOStarFinder
from astropy.stats import sigma_clipped_stats

from astroquery.sdss import SDSS
from astropy.coordinates import SkyCoord, match_coordinates_sky
import util
import phot
from astropy import units
import pickle

from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.text as mpl_text

#LCO = pd.read_pickle('LCO_HDRS_DF.pkl')
visibility = ascii.read('Visibility.csv')

def lco_sdss_pipeline(hdu,diffhdu,threshold=10,threshold_synthetic=3,search=5,writetodisk=False):
    """
    A pipeline to measure detection efficiency of fake planted point sources in LCO difference images. 
    The PSF-flux calibrated to AB magnitudes using ZP from sdss data. 

    Note assumes lco_xid_query has been run. i.e. that sdss data is stored locally ~ sdss_queries/target_SDSS_CleanStar.csv

    Parameters
    ---------------
    hdu : ~astropy.io.fits
        The LCO search image
    diffhdu : ~astropy.io.fits 
        The LCO difference image  
    threshold : float
        The S/N used in DAOStarFinder. Default 10

    Returns
    _________________
    df : pandas DataFrame
        Has columns with the header values and m50,alpha of efficiency.
        If pipeline fails df flag column returns idx corresponding to step of failure. 
    matched_lco : Astropy Table
        The stars in LCO image matched to known SDSS
    matched_sdss : Astropy Table
        The stars in SDSS matched to detected in LCO   
    lco_phot_tab : Astropy Table
        The LCO image photometry on the stars in matched_lco

    Steps in pipeline:
    1. Use DAOFIND to detect stars in the LCO images, https://photutils.readthedocs.io/en/stable/detection.html
    2. Use the hdu wcs to determine ra/dec of xcentroid&ycentroids for stars found
    3. Read the sdss_queries/target_SDSS_CleanStar.csv using hdu target (has sdss rmag and ra/dec, trimmed to good mags for ZP)
    4. Find DAO sources within 5 arcsec of SDSS sources
    5. Do Basic PSF-Photometry on stars in LCO matched to a SDSS, using L1FHWM and IntegratedGaussianPRF
    6. The ZP is the weighted average. weights of each ZP measurement using SDSS-rmags and LCO-fluxes fits 
    7. Add PSF to data at different mags and measure detection efficiencies
    8. Fit model of m50,alpha to efficiencies 
    """
    origname = hdu.header['ORIGNAME'].split("-e00")[0]
    print(origname)
    lco_epsf = util.lco_epsf(hdu)
    
    # 1. DAO (for the stars)
    print("\n")
    print("1. DAOStarFinder on exposure for stars")
    data,hdr = hdu.data, hdu.header 
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)  
    print("mean {:.2f}, median {:.2f}, std {:.2f}".format(mean, median, std))  
    fwhm = hdr["L1FWHM"]
    print("threshold {:.1f},fwhm {:.2f} arcsec".format(threshold,fwhm))
    daofind = DAOStarFinder(fwhm=fwhm/0.389, threshold=threshold*std)  
    sources = daofind(data - median)

    print("{} DAO sources".format(len(sources)))
    print(sources.columns)
    sources.sort('flux')

    """
    try:
        assert(fwhm >= 1.0 and fwhm <= 5.0)
    except:
        print("Image_Error 1. FWHM {:.2f} , typical is [1.5,3.5] arcsec strong peak at 2.".format(fwhm))
        print("ZP=m50=alpha=None")
        # make pd DF out of the header and store m50,alpha
        df = util.table_header(diffhdu,idx=origname)
        df['m50'] = None
        df['alpha'] = None
        df['sig_m50'] = None
        df['sig_alpha'] = None 
        df['ZP'] = None
        df['sigZP'] = None
        df['chisqZP'] = None
        df['flag'] = 1
        print(df)
        if writetodisk:
            pickle.dump(df,open(f"{origname}_df.pkl","wb"))
        matched_lco,matched_sdss,lco_phot_tab = None,None,None
        return df,matched_lco,matched_sdss,lco_phot_tab
    """

    # 2. LCO Skycoords
    print("\n")
    print("2. Ra/Dec of stars found in exposure using hdr wcs")
    lco_skycoords = []
    for i in range(len(sources)):
        pixel = [sources[i]['xcentroid'],sources[i]['ycentroid']]
        sky = util.pixtosky(hdu,pixel)
        lco_skycoords.append(sky)
    lco_skycoords = SkyCoord(lco_skycoords)

    # 3. Read-in SDSS
    print("\n")
    print("3. Reading in sdss star catalog for hdr object")
    obj = hdu.header['OBJECT']
    sdss = ascii.read(f"sdss_queries/{obj}_SDSS_CleanStar.csv")
    print("{} sdss".format(len(sdss)))
    sdss = sdss[sdss['r']>16]
    sdss = sdss[sdss['r']<20]
    print("{} sdss, after restricting to r ~ [16,20] (non-saturated with good S/N)".format(len(sdss)))
    sdss_skycoords = SkyCoord(ra=sdss['ra'],dec=sdss['dec'],unit=units.deg)
    print(sdss.columns)


    """
    try:
        assert(len(sources) >= 0.1*len(sdss))
    except:
        print("Image_Error 3. DAO detected {} stars, < 10 percent of stars in sdss {}, something wrong with image",len(sources),len(sdss))
        print("ZP=m50=alpha=None")
        # make pd DF out of the header and store m50,alpha
        df = util.table_header(diffhdu,idx=origname)
        df['m50'] = None
        df['alpha'] = None
        df['sig_m50'] = None
        df['sig_alpha'] = None 
        df['ZP'] = None
        df['sigZP'] = None
        df['chisqZP'] = None
        df['flag'] = 3
        print(df)
        if writetodisk:
            pickle.dump(df,open(f"{origname}_df.pkl","wb"))
        matched_lco,matched_sdss,lco_phot_tab = None,None,None
        return df,matched_lco,matched_sdss,lco_phot_tab
    """

    # 4. Match SkyCoords
    print("\n")
    print("4. Using match_coordinates_sky for stars in both lco DAO and sdss, within 5 arcsec ~ 12.8 pixel")
    matchcoord,catalogcoord = lco_skycoords,sdss_skycoords
    # shapes match matchcoord: idx into cat, min angle sep, unit-sphere distance 
    idx,sep2d,dist3d=match_coordinates_sky(matchcoord,catalogcoord)
    good_lcoidx,good_sdssidx,good_sep2d = [],[],[]
    matched_lco,matched_sdss = [],[]
    for i in range(len(sources)):
        if sep2d[i] < 5*units.arcsec:
            matched_lco.append(sources[i])
            matched_sdss.append(sdss[idx[i]])
            good_lcoidx.append(i)
            good_sdssidx.append(idx[i])
            good_sep2d.append(sep2d[i])
        else:
            pass

    try:
        assert(len(matched_lco) > 10)
        matched_lco,matched_sdss = vstack(matched_lco),vstack(matched_sdss)
        print("After matching (<5 arcsec separation), {} DAO sources, {} sdss".format(len(matched_lco),len(matched_sdss)))
        print("{:.2f} arcsec median separation".format(np.median([i.value*3600 for i in good_sep2d]))) 
    except:
        print("Image_Error 4. Matched < 10 stars, something wrong, not enough to do photometry and calibrate ZP.")
        print("ZP=m50=alpha=None")
        # make pd DF out of the header and store m50,alpha
        df = util.table_header(diffhdu,idx=origname)
        df['m50'] = None
        df['alpha'] = None
        df['sig_m50'] = None
        df['sig_alpha'] = None 
        df['ZP'] = None
        df['sigZP'] = None
        df['chisqZP'] = None
        df['flag'] = 4
        print(df)
        if writetodisk:
            pickle.dump(df,open(f"{origname}_df.pkl","wb"))
        lco_phot_tab = None
        return df,matched_lco,matched_sdss,lco_phot_tab   

    # 5. Photometry 
    print("\n")
    print("5. Doing Basic PSF-Photometry on the lco stars")
    lco_phots = []
    for i in range(len(matched_lco)):
        location,size = [matched_lco[i]['xcentroid'],matched_lco[i]['ycentroid']],50
        postage_stamp = util.cut_hdu(hdu,location,size)
        init_guess = postage_stamp.data.shape[0]/2,postage_stamp.data.shape[1]/2 # should be at center
        lco_psf_phot = util.LCO_PSF_PHOT(postage_stamp,init_guess)
        lco_phots.append(lco_psf_phot)
    try:
        assert(len(lco_phots) == len(matched_lco))
        lco_phot_tab = vstack(lco_phots)
        print("{} LCO PSF-Photometry".format(len(lco_phot_tab)))
        print(lco_phot_tab.columns)
        # write the successful matched stars & photometry into pkl
        pickle.dump(lco_phot_tab,open(f"{origname}_phot.pkl","wb"))
        pickle.dump(matched_lco,open(f"{origname}_match_lco.pkl","wb"))
        pickle.dump(matched_sdss,open(f"{origname}_match_sdss","wb"))
    except:
        print("Image Error 5. Photometry failed.")
        print("ZP=m50=alpha=None")
        # make pd DF out of the header and store m50,alpha
        df = util.table_header(diffhdu,idx=origname)
        df['m50'] = None
        df['alpha'] = None
        df['sig_m50'] = None
        df['sig_alpha'] = None 
        df['ZP'] = None
        df['sigZP'] = None
        df['chisqZP'] = None
        df['flag'] = 5
        print(df)
        if writetodisk:
            pickle.dump(df,open(f"{origname}_df.pkl","wb"))
        lco_phot_tab = None
        return df,matched_lco,matched_sdss,lco_phot_tab 

    # 6. ZP as weighted average or linear intercept 
    # m from sdss, f from psf-fit on lco  
    print("\n")
    print("6. Getting ZP from weighted average or linear-intercept using sdss-rmags and lco-psf flux_fits")
    ZP,b,sigZP,chisqZP = phot.ZPimage(lco_phot_tab,matched_sdss,scs=True,plot=True,saveas=f"{origname}_zp.png")
    try:
        assert(ZP != None and b != None)
        print("ZP = {:.2f}, b = {:.2f}, sigZP = {:.2f}, chisqZP = {:.2f}".format(ZP,b,sigZP,chisqZP))
        ZP = b
        print("Using the linear intercept value as true ZP")
    except:
        print("Image Error 6. ZP calibration failed.")
        print("ZP=m50=alpha=None")
        # make pd DF out of the header and store m50,alpha
        df = util.table_header(diffhdu,idx=origname)
        df['m50'] = None
        df['alpha'] = None
        df['sig_m50'] = None
        df['sig_alpha'] = None 
        df['ZP'] = None
        df['sigZP'] = None
        df['chisqZP'] = None
        df['flag'] = 6
        print(df)
        if writetodisk:
            pickle.dump(df,open(f"{origname}_df.pkl","wb"))
        return df,matched_lco,matched_sdss,lco_phot_tab 

    # 7. Add PSF to difference and measure efficiencies 
    print("\n")
    print("7. Adding PSF to difference")
    data,hdr = diffhdu.data, diffhdu.header 
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)  
    print("mean {:.2f}, median {:.2f}, std {:.2f}".format(mean, median, std))  
    lattice_pixels,lattice_sky = util.get_lattice_positions(diffhdu,edge=500,spacing=300)
    print("{} locations to plant synthetics".format(len(lattice_pixels)))
    mags = np.arange(19.5,24.1,0.5)
    fluxes = [10**( (i-ZP)/(-2.5) ) for i in mags]
    print("Using mags {}".format(mags))
    x_fit,y_fit = [i[0] for i in lattice_pixels],[i[1] for i in lattice_pixels]
    planted_hdus,efficiencies,mags = [],[],[]
    print("DAOStarFinder on differences for synthetics, search ~ {} pix".format(search))
    for flux in fluxes:
        mag = -2.5*np.log10(flux) + ZP
        flux_fit = [flux for i in range(len(lattice_pixels))]
        posflux = Table([x_fit,y_fit,flux_fit],names=["x_fit","y_fit","flux_fit"])
        cphdu = util.add_psf(diffhdu,lco_epsf,posflux)
        planted_hdus.append(cphdu)
        #detection_catalog = util.threshold_detect_sources(cphdu,nsigma=3,kfwhm=5,npixels=5)
        # S/N DAOStarFinder for the synthetic plants
        data,hdr = cphdu.data, cphdu.header 
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)  
        print("mean {:.2f}, median {:.2f}, std {:.2f}".format(mean, median, std))  
        fwhm = hdr["L1FWHM"]
        print("threshold_synthetic {:.1f},fwhm {:.2f} arcsec".format(threshold_synthetic,fwhm))
        daofind = DAOStarFinder(fwhm=fwhm/0.389, threshold=threshold_synthetic*std)  
        sources = daofind(data-median)
        print("{} DAO sources".format(len(sources)))
        print(sources.columns)
        detection_catalog = sources 
        efficiency = util.detection_efficiency(lattice_pixels,detection_catalog,search=search,plot=True,hdu=cphdu,saveas="{}_{:.2f}_detections.png".format(origname,mag),r=mag)
        efficiencies.append(efficiency)
        print("rmag {:.2f} plants recovered with efficiency {:.2f}".format(mag,efficiency))
        mags.append(mag)
        if efficiency == 0:
            break

    # ETC worst and best case estimates like limiting = 21 and 23.3 @ 3 S/N (or 20.5 and 22.7 @ 5 S/N)
    try:
        assert(efficiencies[0] >= 0.6 and efficiency <= 0.4)
    except:
        print("Image_Error 7. mag 19.5 didn't return high efficiency or 24.0 didn't return low efficiency on detections something wrong",mags,efficiencies)
        print("m50=alpha=None")
        # make pd DF out of the header and store m50,alpha
        df = util.table_header(diffhdu,idx=origname)
        df['m50'] = None
        df['alpha'] = None
        df['sig_m50'] = None
        df['sig_alpha'] = None         
        df['ZP'] = ZP
        df['sigZP'] = sigZP
        df['chisqZP'] = chisqZP
        df['flag'] = 7
        print(df)
        if writetodisk:
            pickle.dump(df,open(f"{origname}_df.pkl","wb"))
        return df,matched_lco,matched_sdss,lco_phot_tab

    # 8. Model the efficiency curve using measured
    # interp needs increasing along x ~ efficiency 
    print("\n")
    print("8. Model efficiency curve, m50 and alpha")
    mags = list(mags)
    mags.reverse()
    efficiencies = list(efficiencies)
    efficiencies.reverse()
    m50 = np.interp(0.5,efficiencies,mags)
    print("init m50 {:.2f}".format(m50))
    bounds=((m50-0.5,1), (m50+0.5, 100))

    """
    # need to do smaller steps around magnitude where it drops off to be able to fit for alpha
    mags = [m50-0.75] + list(np.arange(m50-0.4,m50+0.4,0.1)) + [m50+0.75]
    fluxes = [10**( (i-ZP)/(-2.5) ) for i in mags]
    print("Zoomed in on mags {}".format(mags))
    x_fit,y_fit = [i[0] for i in lattice_pixels],[i[1] for i in lattice_pixels]
    planted_hdus,efficiencies = [],[]
    for flux in fluxes:
        mag = -2.5*np.log10(flux) + ZP
        flux_fit = [flux for i in range(len(lattice_pixels))]
        posflux = Table([x_fit,y_fit,flux_fit],names=["x_fit","y_fit","flux_fit"])
        cphdu = util.add_psf(diffhdu,lco_epsf,posflux)
        planted_hdus.append(cphdu)
        #detection_catalog = util.threshold_detect_sources(cphdu,nsigma=3,kfwhm=5,npixels=5)
        # S/N DAOStarFinder for the synthetic plants
        data,hdr = cphdu.data, cphdu.header 
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)  
        print("mean {:.2f}, median {:.2f}, std {:.2f}".format(mean, median, std))  
        fwhm = hdr["L1FWHM"]
        print("threshold_synthetic {:.1f},fwhm {:.2f} arcsec".format(threshold_synthetic,fwhm))
        daofind = DAOStarFinder(fwhm=fwhm/0.389, threshold=threshold_synthetic*std)  
        sources = daofind(data)
        print("{} DAO sources".format(len(sources)))
        print(sources.columns)
        detection_catalog = sources 
        efficiency = util.detection_efficiency(lattice_pixels,detection_catalog,search=search,plot=True,hdu=cphdu,saveas="{}_{:.2f}_detections.png".format(origname,mag),r=mag)
        efficiencies.append(efficiency)
        print("rmag {:.2f} plants recovered with efficiency {:.2f}".format(mag,efficiency))
    """

    alpha = 10
    init_vals = [m50,alpha]  
    print("init vals (m50, alpha): {},{}".format(m50,alpha))
    print("bounds: ",bounds)
    best_vals,covar = curve_fit(util.f_efficiency, mags, efficiencies, p0=init_vals, bounds=bounds, maxfev=600) 
    
    print('fitted vals (m50, alpha): {}'.format(best_vals))
    print('covariance',covar)
    m50,alpha = best_vals

    # make pd DF out of the header and store m50,alpha
    df = util.table_header(diffhdu,idx=origname)
    df['m50'] = m50
    df['alpha'] = alpha
    """
    param_cov matrix ~ goodness of the efficiency_func fit 
    The estimated covariance of popt. 
    The diagonals provide the variance of the parameter estimate. 
    To compute one standard deviation errors on the parameters use perr = np.sqrt(np.diag(pcov)).
    """
    sigm50,sigalpha = psig = np.sqrt(np.diag(covar))
    print('psig',psig)

    # plot the fit to efficiencies 
    matplotlib.rcParams.update({'font.size': 20,'xtick.labelsize':15,'ytick.labelsize':15})
    fig,ax = plt.subplots(figsize=(8,8))
    ax.scatter(mags,efficiencies,color='black')
    m = np.linspace(np.min(mags),np.max(mags),100)
    eps = [util.f_efficiency(mi,m50,alpha) for mi in m]
    ax.plot(m,eps,color='black')
    obj_0 = util.AnyObject("$m_{50} =$", "black")
    obj_1 = util.AnyObject(r"$\alpha =$", "black")
    #obj_2 = util.AnyObject(r"$\chi^2$ =", "black")
    plt.legend([obj_0,obj_1], ['{:.1f} $\pm$ {:.1}'.format(m50,sigm50),'{:.1f} $\pm$ {:.1}'.format(alpha,sigalpha)],
           handler_map={obj_0:util.AnyObjectHandler(),obj_1:util.AnyObjectHandler()}) 
    plt.xlabel("r")
    plt.ylabel(r"$\epsilon$")
    plt.savefig(f"{origname}_fit_detections.png",bbox_inches='tight')  
    plt.close()

    df['sig_m50'] = sigm50
    df['sig_alpha'] = sigalpha 
    #df['chisqEps'] = chisq()
    df['ZP'] = ZP
    df['sigZP'] = sigZP
    df['chisqZP'] = chisqZP
    df['flag'] = 0
    print(df)
    if writetodisk:
        pickle.dump(df,open(f"{origname}_df.pkl","wb"))

    return df,matched_lco,matched_sdss,lco_phot_tab

def lco_xid_sdss_query():
    """
    Get sdss star properties, rmag and wcs for 112 target fields in our LCOLSS program

    Note requires Visiblity.csv stored locally. Matches ra/dec of lco targets from Visibility.csv 
    to stars from sdss database
    """
        
        """
        full_radius ~ pixscale * 2048 is arcsec from center of an LCO exposure image
        go to 90% of that radius to account for target ra/dec dithers i.e. not being perfectly centered and edge effects
        """
        full_radius = 0.389*(4096/2)    
        radius = 0.85*full_radius
        strradius = str(radius) + ' arcsec'
        print(radius,'ra ~ [{:.2f},{:.2f}], dec ~ [{:.2f},{:.2f}]'.format(float(ra)-radius/3600,float(ra)+radius/3600,float(dec)-radius/3600,float(dec)+radius/3600))
        fields = ['ra','dec','objid','run','rerun','camcol','field','r','mode','nChild','type','clean','probPSF',
                 'psfMag_r','psfMagErr_r'] 
        pos = SkyCoord(ra,dec,unit="deg",frame='icrs')
        xid = SDSS.query_region(pos,radius=strradius,fields='PhotoObj',photoobj_fields=fields) 
        Star = xid[xid['probPSF'] == 1]
        Gal = xid[xid['probPSF'] == 0]
        print(len(xid),len(Star),len(Gal))
        Star = Star[Star['clean']==1]
        print(len(Star))
        ascii.write(Star,f"{obj}_SDSS_CleanStar.csv")
        
        idx+=1
if __name__ == "__main__":
    obj_0 = util.AnyObject("$m_{50} =$", "black")
    print("\n")
    # load all the difference data
    fz = glob.glob("differences/*/*/_dithers/*/*")
    print("{} fz".format(len(fz)))
    diffs = [i for i in fz if os.path.basename(i)[0] == 'd']
    exps = [i for i in fz if os.path.basename(i)[0] != 'd']
    print("{} exps, {} diffs".format(len(exps),len(diffs))) # diffs > exps ... sometimes hotpants and pydia diffs 
    print("\n")

    test = False
    if test:
        batch_start = 0
        batch_end = 1
    else:
        # batch
        print("\n")
        batch_idx = int(sys.argv[1])
        print("{} batch_idx".format(batch_idx))
        batch_start,batch_end = int(batch_idx*10),int(batch_idx*10+9)
        if batch_start != 0:
            batch_start += -1
        print("batch_start,batch end ",batch_start,batch_end)
        print("\n")

    frames,batch_i = [],0
    for i in exps[batch_start:batch_end]:
        print("---------------------------------------------------")
        print("batch_i",batch_i)
        batch_i += 1
        hdu = fits.open(i)[1]
        # get the diff that matches the exposure, might be two pydia and hotpants, take 0th
        origname = hdu.header['ORIGNAME'].split('-e00')[0]
        diff = [j for j in diffs if origname in j]
        diff = diff[0]
        diffhdu = fits.open(diff)[1]
        try:
            assert(diffhdu.header['ORIGNAME'] == hdu.header['ORIGNAME'])
        except:
            print("Exposure and Difference have different origname")
            continue
        df,matched_lco,matched_sdss,lco_phot_tab = lco_sdss_pipeline(hdu,diffhdu,threshold=10,threshold_synthetic=3,search=3)
        frames.append(df)
    batch_result = pd.concat(frames)
    print("{} results".format(len(batch_result)))
    pickle.dump(batch_result,open(f"batch{batch_idx}_df.pkl","wb"))


    
    #lco_xid_sdss_query()
