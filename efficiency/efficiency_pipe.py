import os 
import glob
import sys
import shutil
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


def group_batch():
    """
    Makes tables out of all the batched results from pipeline

    efficiencies.pkl ~ headers for all the data along with results m50,alpha,ZP, and any flags
    sdss_matches.pkl ~ sdss data which had match with lco data
    lco_matches.pkl ~ lco data matched to sdss
    psf_phot.pkl ~ the psf photometry on lco_matches objects
    ap_phot.pkl ~ the aperture photometry on lco_matches objects
    search_star_detections.pkl ~ detections of stars from sdss in search image

    these along with all constituents are put into folder structure at cwd ...
    efficiency_jobs/jobsIDX (where IDX is the lowest positive integer not already a folder)
    the 4 grouped main pkls are here along with other folders
    eps/ detections.png have red/green (det/non) circles on planted sources in the diffs   
    output/ the batch efficiency results, phot, and match pkls (what went into the main pkls)              
    pyjobs/ the .out can read the print statements from pipeline batches     
    zps/  pngs of weighted avgs and linfits for ZP
    """

    # organize folder structure to stick the results 
    cwd = os.getcwd()
    if not os.path.exists('efficiency_jobs'):
        os.mkdir('efficiency_jobs')
    if not os.path.exists('efficiency_jobs/jobs0'):
        os.mkdir('efficiency_jobs/jobs0')
    previous_jobs = glob.glob('efficiency_jobs/jobs*')
    print("previous_jobs",previous_jobs)
    IDX = int(previous_jobs[-1].split('/jobs')[1]) + 1
    IDX = str(IDX)
    job = f'efficiency_jobs/jobs{IDX}'
    os.mkdir(job)
    print('this job', job)
    tmp = ['eps','output','pyjobs','zps']
    for i in tmp:
        os.mkdir(os.path.join(job,i))

    # group the results which are still in the cwd
    ap_phots = glob.glob("*ap_phot.pkl")
    psf_phots = glob.glob("*psf_phot.pkl")
    sdss_matches = glob.glob("*match_sdss*")
    lco_matches = glob.glob("*match_lco*")
    search_star_detections = glob.glob("*search_star_detections.pkl")
    assert(len(ap_phots) == len(psf_phots) == len(sdss_matches) == len(lco_matches))
    ap_stacks,psf_stacks,sdss_stacks,lco_stacks,search_stacks = [],[],[],[],[]
    for i in range(len(ap_phots)):
        ap = ap_phots[i]
        psf = psf_phots[i]
        origname = ap.split("_ap_phot")[0]
        sdss = [i for i in sdss_matches if origname in i]
        lco = [i for i in lco_matches if origname in i]
        star = [i for i in search_star_detections if origname in i]
        a,p,s,l,s2 = pickle.load(open(ap,"rb")),pickle.load(open(psf,"rb")),pickle.load(open(sdss[0],"rb")),pickle.load(open(lco[0],"rb")),pickle.load(open(star[0],"rb"))
        a['origname'] = origname
        p['origname'] = origname
        s['origname'] = origname
        l['origname'] = origname
        ap_stacks.append(a)
        psf_stacks.append(p)
        sdss_stacks.append(s)
        lco_stacks.append(l)
        search_stacks.append(s2)

    ap_phots=vstack(ap_stacks)
    psf_phots=vstack(psf_stacks)
    sdss_matches=vstack(sdss_stacks)
    lco_matches=vstack(lco_stacks)
    search_star_detections=vstack(search_stacks)
    pickle.dump(ap_phots,open("aperture_phot.pkl","wb"))
    pickle.dump(psf_phots,open("psf_phot.pkl","wb"))
    pickle.dump(sdss_matches,open("sdss_matches.pkl","wb"))
    pickle.dump(lco_matches,open("lco_matches.pkl","wb"))
    pickle.dump(search_star_detections,open("search_star_detections.pkl","wb"))

    batch = glob.glob("batch*pkl")
    dfs,idxs = [],[]
    for i in batch:
        batchidx = i.split('_')[0]
        bi = pickle.load(open(i,"rb"))
        dfs.append(bi)
        idxs.append(batchidx)
    df = pd.concat(dfs)
    df.meta = {"ZPcalibrated":"sdss-r"}
    pickle.dump(df,open("efficiencies.pkl","wb"))

    # now begin the shuffle
    sources = ['efficiencies.pkl','aperture_phot.pkl','psf_phot.pkl','sdss_matches.pkl','lco_matches.pkl','search_star_detections.pkl']
    destination = job
    for src in sources:
        shutil.copy(src,destination)

    phots = glob.glob("*_phot.pkl")
    sdss_matches = glob.glob("*match_sdss*")
    lco_matches = glob.glob("*match_lco*")
    batch = glob.glob("batch*pkl")
    searches = glob.glob("*search_star*")
    sources = phots + sdss_matches + lco_matches + batch + searches
    destination = os.path.join(job,'output')
    for src in sources:
        shutil.move(src,destination)
    mags = glob.glob("*mag*fits")
    destination = os.path.join(job,'cutouts_fakes')
    for src in mags:
        shutil.move(src,destination)
    zps = glob.glob("*zp.png")
    destination = os.path.join(job,'zps')
    for src in zps:
        shutil.move(src,destination)
    eps = glob.glob("*detections.png")
    destination = os.path.join(job,'eps')
    for src in eps:
        shutil.move(src,destination)
    pyjobs = glob.glob("pyjob_efficiency*")
    destination = os.path.join(job,'pyjobs')
    for src in pyjobs:
        shutil.move(src,destination)

def search_efficiency(hdu,diffhdu,templatehdu,match_radius=1,threshold_calibrate=10,threshold_limit=3,detection_radius=3):
    """
    A pipeline to measure detection efficiency of point sources in LCO images. 
    The PSF-flux with L1FWHM spread calibrated to AB magnitudes using ZP from sdss r-band data. 

    Note assumes lco_xid_query has been run. i.e. that sdss data is stored locally ~ sdss_queries/target_SDSS_CleanStar.csv

    Parameters
    ---------------
    hdu : ~astropy.io.fits
        The LCO search image
    diffhdu : ~astropy.io.fits 
        The LCO difference image  
    templatehdu : ~astropy.io.fits
        The LCO template image
    match_radius : float
        Radius [arcsec] required for match of SDSS position with LCO DAO detected source location (wrt threshold calibrate) 
        Default 1 arcsec
    threshold_calibrate : float
        Threshold used in DAOStarFinder to find stars for flux measurement -> ZP calibration. Default 10
    threshold_limit : float
        Threshold used in DAOStarfinder to find limit ot detectability. Default 3 
    detection_radius : float
        Radius [pixels] to search around point source positions for DAO detected source locations (wrt threshold limiting) 
        Default 3 pixels
    Returns
    _________________
    df : pandas DataFrame
        Has columns with the header values and m50,alpha of efficiency.
        If pipeline fails df flag column returns idx corresponding to step of failure. 
    matched_lco : Astropy Table
        The stars in LCO image matched to known SDSS
    matched_sdss : Astropy Table
        The stars in SDSS matched to detected in LCO   
    ap_phot_tab : Astropy Table
        Photutils aperture photometry on the stars in matched_lco
    psf_phot_tab : Astropy Table
        Photutils psf photometry on the stars in matched_lco
    stars : Astropy Table
        SDSS-stars with added DAO detection data

    Steps in pipeline:
    1. Use DAOFIND to detect stars suitable for ZP calibration in the LCO images, https://photutils.readthedocs.io/en/stable/detection.html
    2. Use the hdu wcs to determine ra/dec of xcentroid&ycentroids for stars found
    3. Read the sdss_queries/target_SDSS_CleanStar.csv using hdu target (has sdss rmag and ra/dec, trimmed to good mags for ZP)
    4. Find/match DAO sources within arcsec of SDSS sources
    5. Do Photometry on stars in LCO matched to a SDSS
    6. The ZP as weighted average and fit. Weights of each ZP measurement using SDSS-rmags and LCO-fluxes
    7. Detect Search Image Stars and fit m50/alpha efficiency function
    8. Detect Difference Image fake sources and fit m50/alpha efficiency function
    """

    hdr = hdu.header
    origname = hdr['ORIGNAME'].split("-e00")[0]
    df = util.table_header(diffhdu,idx=origname)    
    print(origname)

    lco_epsf = util.lco_epsf(hdu) 
    fwhm = hdr["L1FWHM"] # arcsec
    l1sigma = hdu.header['L1SIGMA']
    search_mean, search_median, search_std = sigma_clipped_stats(hdu.data, sigma=3.0)  
    diff_mean, diff_median, diff_std = sigma_clipped_stats(diffhdu.data, sigma=3.0)
    template_mean, template_median, template_std = sigma_clipped_stats(templatehdu.data, sigma=3.0)

    print("Test. Detection DAOStarFinder on clean difference") 
    daofind3sig = DAOStarFinder(fwhm=fwhm/0.389, threshold=3*diff_std)  
    daofind5sig = DAOStarFinder(fwhm=fwhm/0.389, threshold=5*diff_std)
    sources3sig = daofind3sig(diffhdu.data-diff_median)
    sources5sig = daofind5sig(diffhdu.data-diff_median)
    try:
        Nfp_3sig,Nfp_5sig = len(sources3sig),len(sources5sig)
    except:
        # returns None type if no sources found
        Nfp_3sig,Nfp_5sig = None,None
    print("{} 3sig DAO sources from clean diff".format(Nfp_3sig))
    print("{} 5sig DAO sources from clean diff".format(Nfp_5sig))    
    fpsources = [sources3sig,sources5sig]

    df['Nfp_3sig'] = Nfp_3sig
    df['Nfp_5sig'] = Nfp_5sig
    df['match_radius'] = match_radius
    df['threshold_calibrate'] = threshold_calibrate
    df['threshold_limit'] = threshold_limit
    df['detection_radius'] = detection_radius
    df['search_mean'] = search_mean
    df['search_median'] = search_median
    df['search_std'] = search_std
    df['diff_mean'] = diff_mean
    df['diff_median'] = diff_median
    df['diff_std'] = diff_std
    df['template_mean'] = template_mean
    df['template_median'] = template_median
    df['template_std'] = template_std
    # the values we want to determine and fill in
    matched_lco,matched_sdss,ap_phot_tab,psf_phot_tab,stars = None,None,None,None,None
    # has it failed/where?
    df['flag'] = None
    # matched stars data
    df['Nmatch'] = None
    df['Nmatch_median_sep'] = None
    df['brightest_match'] = None
    df['dimmest_match'] = None
    # ZP calibration data
    # from the psf phot
    df['ZPtrue_count_PSF'] = None
    df['ZPdof_PSF'] = None
    df['ZPweightedavg_PSF'] = None
    df['ZPstd_PSF'] = None
    df['ZPslopefit_PSF'] = None
    df['ZPfit_PSF'] = None
    df['ZP_chisq_PSF'] = None
    df['ZP_chisq_dof_PSF'] = None
    df['ZP_covxx_PSF'] = None
    df['ZP_covyy_PSF'] = None
    df['ZP_covxy_PSF'] = None
    # from the aperture phot
    df['ZPtrue_count_AP'] = None
    df['ZPdof_AP'] = None
    df['ZPweightedavg_AP'] = None
    df['ZPstd_AP'] = None
    df['ZPslopefit_AP'] = None
    df['ZPfit_AP'] = None
    df['ZP_chisq_AP'] = None
    df['ZP_chisq_dof_AP'] = None
    df['ZP_covxx_AP'] = None
    df['ZP_covyy_AP'] = None
    df['ZP_covxy_AP'] = None
    # Search image efficiency data
    df['first_nondetect'] = None
    df['last_detect'] = None
    df['search_m50'] = None
    df['search_alpha'] = None
    df['search_m50_std'] = None
    df['search_alpha_std'] = None
    df['search_efficiency_chisq'] = None
    df['search_efficiency_dof'] = None
    df['search_efficiency_chisq_dof'] = None
    # Difference image efficiency data 
    df['difference_m50'] = None
    df['difference_alpha'] = None
    df['difference_m50_std'] = None
    df['difference_alpha_std'] = None
    df['difference_efficiency_chisq'] = None
    df['difference_efficiency_dof'] = None
    df['difference_efficiency_chisq_dof'] = None

    # 1. DAO (for the stars)
    print("\n")
    print("1. DAOStarFinder on exposure for stars")
    print("DAO threshold_calibrate = {:.1f}-search_std".format(threshold_calibrate))
    daofind = DAOStarFinder(fwhm=fwhm/0.389, threshold=threshold_calibrate*search_std)  
    sources = daofind(hdu.data - search_median)
    Nlco = len(sources)
    print("{} stars available for ZP calibration".format(Nlco))
    sources.sort('flux')

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
    sdss0 = ascii.read(f"sdss_queries/{obj}_SDSS_CleanStar.csv")
    Nsdss0 = len(sdss0)
    print("{} sdss0".format(Nsdss0))
    sdss = sdss0[(sdss0['r']>16) & (sdss0['r']<20)]
    Nsdss = len(sdss)
    print("{} sdss, after restricting to r ~ [16,20] (non-saturated with good S/N)".format(Nsdss))
    sdss_skycoords = SkyCoord(ra=sdss['ra'],dec=sdss['dec'],unit=units.deg)
    print(sdss.columns)

    # 4. Match SkyCoords
    print("\n")
    print("4. Using match_coordinates_sky for stars in both lco DAO and sdss, within {} arcsec".format(match_radius))
    matchcoord,catalogcoord = lco_skycoords,sdss_skycoords
    # shapes match matchcoord: idx into cat, min angle sep, unit-sphere distance 
    idx,sep2d,dist3d=match_coordinates_sky(matchcoord,catalogcoord)
    good_lcoidx,good_sdssidx,good_sep2d = [],[],[]
    matched_lco,matched_sdss = [],[]
    for i in range(len(sources)):
        if sep2d[i] < match_radius*units.arcsec:
            matched_lco.append(sources[i])
            matched_sdss.append(sdss[idx[i]])
            good_lcoidx.append(i)
            good_sdssidx.append(idx[i])
            good_sep2d.append(sep2d[i])
        else:
            pass
    Nmatch = len(matched_lco)
    print("After matching (<{} arcsec separation), {} DAO sources, {} sdss".format(match_radius,len(matched_lco),len(matched_sdss)))
    try:
        arcsec_seps = [i.value*3600 for i in good_sep2d]
        med_sep = np.median(arcsec_seps)
        print("{:.2f} arcsec median separation".format(med_sep))         
        assert(len(matched_lco) > 5)
        matched_lco,matched_sdss = vstack(matched_lco),vstack(matched_sdss)
    except:
        print("Error: Matched < 5 stars, won't be reliable enough to do photometry and calibrate ZP.")
        df['flag'] = 'matching'
        return df,matched_lco,matched_sdss,ap_phot_tab,psf_phot_tab   

    # Matched stars data
    df['Nmatch'] = Nmatch
    df['Nmatch_median_sep'] = med_sep
    df['brightest_match'] = np.min(matched_sdss['r'])
    df['dimmest_match'] = np.max(matched_sdss['r'])

    # 5. Photometry 
    print("\n")
    print("5. Photutils BasicPSFPhotometry & aperture_photometry on the lco stars")
    psf_phots,ap_phots,objids = [],[],[]
    for i in range(len(matched_lco)):
        objid = matched_sdss[i]['objid']
        location,size = [matched_lco[i]['xcentroid'],matched_lco[i]['ycentroid']],100
        postage_stamp = util.cut_hdu(hdu,location,size)
        init_guess = postage_stamp.data.shape[0]/2,postage_stamp.data.shape[1]/2 # should be at center
        lco_psf_phot = util.LCO_PSF_PHOT(postage_stamp,init_guess)
        lco_ap_phot = util.LCO_AP_PHOT(postage_stamp, [init_guess])
        objids.append(objid)
        psf_phots.append(lco_psf_phot)
        ap_phots.append(lco_ap_phot)
    try:
        assert(len(psf_phots) == len(ap_phots) == len(matched_lco))
        psf_phot_tab = vstack(psf_phots)
        ap_phot_tab = vstack(ap_phots)
        psf_phot_tab['objid'] = objids
        ap_phot_tab['objid'] = objids
        assert(len(psf_phot_tab) == len(ap_phot_tab))
        print("{} LCO Photometry".format(len(psf_phot_tab)))
        # write the successful matched stars & photometry into pkl
        pickle.dump(psf_phot_tab,open(f"{origname}_psf_phot.pkl","wb"))
        pickle.dump(ap_phot_tab,open(f"{origname}_ap_phot.pkl","wb"))        
        pickle.dump(matched_lco,open(f"{origname}_match_lco.pkl","wb"))
        pickle.dump(matched_sdss,open(f"{origname}_match_sdss.pkl","wb"))
    except:
        print("Error: Photometry failed.")
        df['flag'] = 'phot'
        return df,matched_lco,matched_sdss,ap_phot_tab,psf_phot_tab,stars

    # 6. ZP as weighted average or linear intercept 
    # m from sdss, f from psf-fit on lco  
    print("\n")
    print("6. Calibrating ZP as weighted average and linear-intercept using sdss-rmags and lco photometry")
    print("psf phot calibration")
    PSF_true_count,PSF_ZPweightedavg,PSF_ZPstd,PSF_m,PSF_b,PSF_ZP_chisq,PSF_ZPcov = phot.ZPimage(psf_phot_tab,matched_sdss,scs=True,plot=True,saveas=f"{origname}_psf_zp.png")
    print("aperture phot calibration")
    AP_true_count,AP_ZPweightedavg,AP_ZPstd,AP_m,AP_b,AP_ZP_chisq,AP_ZPcov = phot.ZPimage(ap_phot_tab,matched_sdss,scs=True,plot=True,saveas=f"{origname}_ap_zp.png")
    try:
        assert(PSF_ZPweightedavg != None and PSF_b != None)
        assert(AP_ZPweightedavg != None and AP_b != None)
    except:
        print("Error: ZP calibration failed.")
        df['flag'] = 'ZPcalibration'
        return df,matched_lco,matched_sdss,ap_phot_tab,psf_phot_tab,stars

    # ZP calibration data
    # from the psf phot
    df['ZPtrue_count_PSF'] = PSF_true_count
    PSF_ZPdof = PSF_true_count - 2 # fitting slope and intercept
    df['ZPdof_PSF'] = PSF_ZPdof 
    df['ZPweightedavg_PSF'] = PSF_ZPweightedavg
    df['ZPstd_PSF'] = PSF_ZPstd  
    df['ZPslopefit_PSF'] = PSF_m
    df['ZPfit_PSF'] = PSF_b
    df['ZP_chisq_PSF'] = PSF_ZP_chisq
    df['ZP_chisq_dof_PSF'] = PSF_ZP_chisq/PSF_ZPdof
    PSFcovxx = PSF_ZPcov[0][0]
    PSFcovyy = PSF_ZPcov[1][1]
    PSFcovxy = PSF_ZPcov[0][1]
    df['ZP_covxx_PSF'] = PSFcovxx
    df['ZP_covyy_PSF'] = PSFcovyy
    df['ZP_covxy_PSF'] = PSFcovxy
    # from the aperture phot
    df['ZPtrue_count_AP'] = AP_true_count
    AP_ZPdof = AP_true_count - 2 # fitting slope and intercept
    df['ZPdof_AP'] = AP_ZPdof 
    df['ZPweightedavg_AP'] = AP_ZPweightedavg
    df['ZPstd_AP'] = AP_ZPstd  
    df['ZPslopefit_AP'] = AP_m
    df['ZPfit_AP'] = AP_b
    df['ZP_chisq_AP'] = AP_ZP_chisq
    df['ZP_chisq_dof_AP'] = AP_ZP_chisq/AP_ZPdof
    APcovxx = AP_ZPcov[0][0]
    APcovyy = AP_ZPcov[1][1]
    APcovxy = AP_ZPcov[0][1]
    df['ZP_covxx_AP'] = APcovxx
    df['ZP_covyy_AP'] = APcovyy
    df['ZP_covxy_AP'] = APcovxy

    ZP = PSF_b 
    dZP = np.sqrt(PSFcovyy)

    # 7. Detect Search Image Stars and fit m50/alpha efficiency function
    rs = [16,17,18,19,20,21,21.5,22,22.5,23,23.5,24.0,24.5]
    stars = util.search_sdss(sdss0,rs)
    stars_skycoords = SkyCoord(ra=stars['ra'],dec=stars['dec'],unit=units.deg)
    stars_pixels = util.skytopix(hdu,stars_skycoords)
    
    daofind = DAOStarFinder(fwhm=fwhm/0.389, threshold=threshold_limit*search_std)  
    daosources = daofind(hdu.data-search_median)
    star_detections = []
    for i in range(len(stars)):
        x,y = pix = stars_pixels[i]
        detected = util.detection(pix,daosources,detection_radius)
        star_detections.append(detected)
    stars['detected'] = star_detections
    stars['origname'] = origname
    # scipy.optimize.curve_fit
    x = np.linspace(min(stars['psfMag_r']),max(stars['psfMag_r']),100)
    search_efficiency_vals,search_efficiency_covar = curve_fit(util.f_efficiency, stars['psfMag_r'], stars['detected'], maxfev=600) 
    search_m50,search_alpha = search_efficiency_vals
    search_m50_std,search_alpha_std = psig = np.sqrt(np.diag(search_efficiency_covar))
    search_efficiency_chisq = np.sum([(util.f_efficiency(i['psfMag_r'],search_m50,search_alpha) - i['detected'])**2 for i in stars])
    search_efficiency_dof = len(stars) - 2 # fitting m50/alpha
    search_efficiency_chisq_dof = search_efficiency_chisq/search_efficiency_dof    
    detects = stars[stars['detected']==1]
    nondetects = stars[stars['detected']==0]
    try:
        first_nondetect = nondetects[0]['psfMag_r']
    except:
        # detected all the stars
        first_nondetect = -99.0
    try:
        last_detect = detects[-1]['psfMag_r']
    except:
        # didn't detect any of the stars
        last_detect = -99.0
    # Search image efficiency data (into the sdss stars table)
    stars['first_nondetect'] = first_nondetect
    stars['last_detect'] = last_detect
    stars['search_m50'] = search_m50
    stars['search_alpha'] = search_alpha
    stars['search_m50_std'] = search_m50_std
    stars['search_alpha_std'] = search_alpha_std
    stars['search_efficiency_chisq'] = search_efficiency_chisq
    stars['search_efficiency_dof'] = search_efficiency_dof
    stars['search_efficiency_chisq_dof'] = search_efficiency_chisq_dof
    pickle.dump(stars,open(f"{origname}_search_star_detections.pkl","wb"))
    # Search image efficiency data (into the header table)
    df['first_nondetect'] = first_nondetect
    df['last_detect'] = last_detect
    df['search_m50'] = search_m50
    df['search_alpha'] = search_alpha
    df['search_m50_std'] = search_m50_std
    df['search_alpha_std'] = search_alpha_std
    df['search_efficiency_chisq'] = search_efficiency_chisq
    df['search_efficiency_dof'] = search_efficiency_dof
    df['search_efficiency_chisq_dof'] = search_efficiency_chisq_dof

    print('search efficiency ~ m50,alpha',search_efficiency_vals)

    # 7. Detect fake sources in the difference iamge and fit m50/alpha efficiency function
    print("\n")
    print("7. Adding PSF to difference in a grid")
    lattice_pixels,lattice_sky = util.get_lattice_positions(diffhdu,edge=500,spacing=300)
    print("{} locations to plant synthetics".format(len(lattice_pixels)))
    mags = np.arange(search_m50-1,search_m50+1,0.2)
    fluxes = [10**( (i-ZP)/(-2.5) ) for i in mags]
    print("ZP {:.2f} +- {:.2f}, from the fit with PSF-fluxes".format(ZP,dZP))
    print("Using mags sampled around m50 of search image star detections, {}".format(mags))
    x_fit,y_fit = [i[0] for i in lattice_pixels],[i[1] for i in lattice_pixels]
    planted_diffhdus,efficiencies = [],[]
    print("DAOStarFinder on differences for synthetics")
    for flux in fluxes:
        mag = -2.5*np.log10(flux) + ZP
        flux_fit = [flux for i in range(len(lattice_pixels))]
        posflux = Table([x_fit,y_fit,flux_fit],names=["x_fit","y_fit","flux_fit"])
        cphdu = util.add_psf(diffhdu,lco_epsf,posflux)
        # HDUList object ~ MEF of cutouts on the fakes
        mef = util.plants_MEF(cphdu,lattice_pixels,cutoutsize=100,writetodisk=True,saveas=f'{origname}mag{mag}fakes.fits')
        planted_diffhdus.append(cphdu)
        mef_effs = []
        primary = mef[0]
        for i in range(1,len(mef)): # 0 is primary hdu doesn't have data 
            cutout = mef[i]
            pix = (cutout.data.shape[0]/2,cutout.data.shape[1]/2) # its at the center of cutout 
            cutout_mean, cutout_median, cutout_std = sigma_clipped_stats(cutout.data, sigma=3.0)  
            daofind = DAOStarFinder(fwhm=fwhm/0.389, threshold=threshold_limit*cutout_std)  
            daosources = daofind(cutout.data-cutout_median)
            detected = util.detection(pix,daosources,detection_radius)
            mef_effs.append(detected)
            mef[i].header['detection'] = f'{detected}' # 0 or 1
        
        efficiency = np.sum(mef_effs)/len(mef_effs) # fractional efficiency from all SN in grid at mag
        efficiencies.append(efficiency)
        primary.header['efficiency'] = f'{efficiency}'
        primary.header['ZP'] = f'{ZP}'
        primary.header['dZP'] = f'{dZP}'
        primary.header['flux'] = f'{flux}'
        primary.header['mag'] = f'{mag}'
        mef.writeto(f'{origname}mag{mag}fakes.fits', overwrite=True)
    # interp init_guess, needs increasing along x ~ efficiency 
    mags = list(mags)
    mags.reverse()
    efficiencies = list(efficiencies)
    efficiencies.reverse()
    m50_init = np.interp(0.5,efficiencies,mags)
    alpha_init = 3
    init_vals = [m50,alpha]  
    bounds=((m50_init-0.5,1), (m50_init+0.5, 100)) 
    print("init_guesses (m50, alpha): {},{}".format(m50,alpha))
    print("bounds: ",bounds)
    difference_efficiency_vals,difference_efficiency_covar = curve_fit(util.f_efficiency, mags, efficiencies, maxfev=600) 
    difference_m50,difference_alpha = difference_efficiency_vals
    difference_m50_std,difference_alpha_std = psig = np.sqrt(np.diag(difference_efficiency_covar))
    difference_efficiency_chisq = np.sum([(util.f_efficiency(mags[i],difference_m50,difference_alpha) - efficiencies[i])**2 for i in range(len(mags))])
    difference_efficiency_dof = len(mags) - 2 # fitting m50/alpha
    difference_efficiency_chisq_dof = difference_efficiency_chisq/difference_efficiency_dof 
    # Difference image efficiency data 
    df['difference_m50'] = difference_m50
    df['difference_alpha'] = difference_alpha
    df['difference_m50_std'] = difference_m50_std
    df['difference_alpha_std'] = difference_alpha_std
    df['difference_efficiency_chisq'] = difference_efficiency_chisq
    df['difference_efficiency_dof'] = difference_efficiency_dof
    df['difference_efficiency_chisq_dof'] = difference_efficiency_chisq_dof

    print('difference efficiency ~ m50,alpha',difference_efficiency_vals)

    return df,matched_lco,matched_sdss,ap_phot_tab,psf_phot_tab,stars


if __name__ == "__main__":
    fz = glob.glob("differences/*/*/_dithers/*/*")
    diffs = [i for i in fz if os.path.basename(i)[0] == 'd']
    exps = [i for i in fz if os.path.basename(i)[0] != 'd']
    print("{} exps, {} diffs".format(len(exps),len(diffs))) # diffs > exps ... sometimes hotpants and pydia diffs 
    templates = glob.glob("templates/*/template.fits")
    print("{} templates".format(len(templates)))
    print("\n")
    

    group = False
    if group:
        group_batch()

    test = True
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

    dataframes,batch_i = [],0
    for i in exps[batch_start:batch_end]:
        print("---------------------------------------------------")
        print("batch_i",batch_i)
        batch_i += 1
        try:
            hdu = fits.open(i)[1]
            # get the diff that matches the exposure, might be two pydia and hotpants, take one with lower noise (std of diff)
            origname = hdu.header['ORIGNAME'].split('-e00')[0]
            diff = [j for j in diffs if origname in j]
            diffdat = [fits.open(j)[1].data for j in diff]
            diffnoise = [np.std(j) for j in diffdat]
            diffidx = np.argmin(diffnoise)
            diff = diff[diffidx]
            diffhdu = fits.open(diff)[1]
        except:
            print("The hdu and/or diffhdu didn't open properly")
            continue
        try:
            target = hdu.header['OBJECT']
            template = [i for i in templates if target in i]
            template = template[0]
            templatehdu = fits.open(template)[0]
        except:
            print("The template didn't open properly")
            continue
        try:
            assert(diffhdu.header['ORIGNAME'] == hdu.header['ORIGNAME'])
        except:
            print("Exposure and Difference have different origname")
            continue
        
        df,matched_lco,matched_sdss,ap_phot_tab,psf_phot_tab,stars = search_efficiency(hdu,diffhdu,templatehdu,match_radius=1,threshold_calibrate=10,threshold_limit=3,detection_radius=3)
        dataframes.append(df)

    batch_result = pd.concat(dataframes)
    print("{} results".format(len(batch_result)))
    pickle.dump(batch_result,open(f"batch{batch_idx}_df.pkl","wb"))
