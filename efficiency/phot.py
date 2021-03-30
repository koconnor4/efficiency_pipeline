import glob
import os
import pickle
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt 
import util
import matplotlib
from astropy.stats import sigma_clip,sigma_clipped_stats
import matplotlib.text as mpl_text

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
"""
obj_0 = AnyObject("A", "purple")
obj_1 = AnyObject("B", "green")

plt.legend([obj_0, obj_1], ['Model Name A', 'Model Name B'],
           handler_map={obj_0:AnyObjectHandler(), obj_1:AnyObjectHandler()})
"""

def ZPimage(lco_phot_tab,matched_sdss,plot=True,saveas="zp.png",scs=True):
    """
    Make png images showing the determination of ZP.

    Parameters
    ___________
    lco_phot_tab ~ Astropy Table
        flux_fit, flux_unc from PSF-Phot on LCO image stars
    matched_sdss ~ Astropy Table
        psfMag_r, psfMagErr_r from matching sdss data

    Returns
    __________
    Weighted Average ~ ZP(star)
        ZP from each star using rmag and flux. 
    
    Linear Fit ~ rmag(-2.5log10flux)
        ZP from intercept.
    """
    try:
        assert(len(lco_phot_tab) == len(matched_sdss))
        print("{} data".format(len(lco_phot_tab)))
    except:
        print("Photometry and SDSS table aren't the same shape.")
        ZP,b = None,None
        return ZP,b

    try:
        # psf-phot table values
        flux_lco,flux_unc_lco = np.array(lco_phot_tab['flux_fit']),np.array(lco_phot_tab['flux_unc'])
    except:
        # ap-phot table values
        flux_lco,flux_unc_lco = np.array(lco_phot_tab['aper_sum_bkgsub']),np.array(lco_phot_tab['aperture_sum_err'])
    
    rmag,rmagerr = np.array(matched_sdss['psfMag_r']),np.array(matched_sdss['psfMagErr_r'])

    # clip negative fluxes 
    flux_clip = np.log10(flux_lco)
    indices = ~np.isnan(flux_clip)
    true_count = np.sum(indices)
    
    flux_lco,flux_unc_lco = flux_lco[indices],flux_unc_lco[indices]
    rmag,rmagerr = rmag[indices],rmagerr[indices]

    print("{} data after clipping bad fluxes".format(true_count))

    # uncertainties and weights
    lco_uncertainty = flux_unc_lco/flux_lco
    sdss_uncertainty = rmagerr
    uncertainties = []
    for i in range(true_count):
        uncertainties.append( np.sqrt(lco_uncertainty[i]**2 + sdss_uncertainty[i]**2) )
    print("median uncertainties {:.2f}, median sdss uncertainties {:.2f}, median lco uncertainties {:.2f}".format(np.median(uncertainties),np.median(sdss_uncertainty),np.median(lco_uncertainty)))
    weights = [1/i for i in uncertainties] 

    # ZP as weighted average of zp = m + 2.5 log10(f), for each star
    # m from sdss, f from psf-fit on lco  
    values = rmag + 2.5*np.log10(flux_lco)
    values = [i for i in values]
    ZP = util.weighted_average(weights,values)
    sigZP = np.std(values)
    # clip remaining bad data using scs around weighted avg ZP
    if scs:
        zp_clip = sigma_clip(values,sigma=5)
        indices = ~zp_clip.mask
        badindices = zp_clip.mask
        true_count = np.sum(indices)
        bad_count = np.sum(badindices)
        badvalues = np.array(values)[badindices]
        badweights = np.array(weights)[badindices]
        values = np.array(values)[indices]
        weights = np.array(weights)[indices]
        ZP = util.weighted_average(weights,values)
        sigZP = np.std(values)
        print("{} data after clipping around ZP from weighted average".format(true_count))
        print("{} indices, {} bad indices".format(true_count,bad_count))
    print("ZP from weighted_average = {:.2f} +- {:.2f}".format(ZP,sigZP))
    badflux_lco,badflux_unc_lco = flux_lco[badindices],flux_unc_lco[badindices]
    badrmag,badrmagerr = rmag[badindices],rmagerr[badindices]
    badlco_uncertainty,badsdss_uncertainty=lco_uncertainty[badindices],sdss_uncertainty[badindices]
    flux_lco,flux_unc_lco = flux_lco[indices],flux_unc_lco[indices]
    rmag,rmagerr = rmag[indices],rmagerr[indices]
    lco_uncertainty,sdss_uncertainty=lco_uncertainty[indices],sdss_uncertainty[indices]

    # weighted average plot
    if plot:
        # definitions for the axes
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.85
        spacing = 0.005
        rect_scatter = [left, bottom, width, height]
        #rect_histx = [left, bottom + height + spacing, width, 0.2]
        rect_histy = [left + width + spacing, bottom, 0.2, height]
        matplotlib.rcParams.update({'font.size': 20,'xtick.labelsize':15,'ytick.labelsize':15})
        # start with a rectangular Figure
        fig = plt.figure(figsize=(16,8))
        #ax_scatter = fig.add_subplot(111)
        #ax_histy = ax_scatter.twinx(sharey=ax_scatter)
        ax_scatter = plt.axes(rect_scatter)
        ax_scatter.tick_params(direction='in', top=True, right=True)
        ax_histy = plt.axes(rect_histy,sharey=ax_scatter)
        ax_histy.tick_params(direction='in', labelleft=False)

        #fig,ax = plt.subplots(figsize=(16,8))
        spacing = np.arange(0,true_count,1)
        uncertainties = [1/i for i in weights]
        ax_scatter.errorbar(spacing,values,yerr=uncertainties,marker='x',ls='',color='red',label='zp')
        ax_scatter.hlines(ZP,0,true_count,linestyle='--',label='ZP={:.1f}'.format(ZP),color='black')
        ax_scatter.set_xlabel("Stars (dim -> bright)")
        ax_scatter.set_ylabel("ZP") # $rmag_{sdss}$
        ax_scatter.set_ylim(ZP-5*sigZP,ZP+5*sigZP)

        ax_histy.hist(values,fill=None,color='black',orientation='horizontal')

        obj_0 = AnyObject("ZP =", "black")
        plt.legend([obj_0], ['{:.1f} $\pm$ {:.1}'.format(ZP,sigZP)],
           handler_map={obj_0:AnyObjectHandler()})
        #plt.legend()
        plt.savefig("weighted_average_"+saveas,bbox_inches='tight')
        plt.close()

    # ZP as intercept in linear fit 
    xi = -2.5*np.log10(flux_lco)
    xerr = lco_uncertainty
    badxi = -2.5*np.log10(badflux_lco)
    badxerr = badlco_uncertainty
    fig,ax = plt.subplots(figsize=(16,8))
    coeffs,residuals,rank,singular_values,conditioning_threshold = np.polyfit(xi,rmag,1,w=weights,full=True)
    _,cov = np.polyfit(xi,rmag,1,w=weights,cov=True)
    m,b =coeffs 
    #residual value returned is the sum of the squares of the fit errors
    chisq = residuals[0]
    print("chisq",chisq)
    if plot:
        ax.errorbar(xi,rmag,xerr=xerr,yerr=rmagerr,marker='x',ls='',color='red',label='')
        ax.errorbar(badxi,badrmag,xerr=badxerr,yerr=badrmagerr,marker='x',ls='',color='grey',label='')
        x=np.linspace(np.min(xi),np.max(xi),100)
        y=m*x + b
        ax.plot(x,y,ls='--',color='black',label='ZP={:.1f}'.format(b))
        plt.show()
        plt.xlabel("$-2.5log10(f_{lco})$")
        plt.ylabel("$r_{sdss}$")
        plt.xlim(np.min(xi),np.max(xi))
        plt.ylim(np.min(rmag),np.max(rmag))
        obj_0 = AnyObject("ZP =", "black")
        obj_1 = AnyObject(r"$\chi^2$ =", "black")
        plt.legend([obj_0,obj_1], ['{:.1f} $\pm$ {:.1}'.format(b,sigZP),'{:.1f}'.format(chisq)],
           handler_map={obj_0:AnyObjectHandler(),obj_1:AnyObjectHandler()})        
        plt.savefig("lin_fit_"+saveas,bbox_inches='tight')
        plt.close()
    print("ZP as interecept of linear fit = {:.2f}".format(b))

    """
    try:
        assert(np.abs(ZP-b) < 0.2)
    except:
        # things that make you go hmmm
        print("weighted average and intercept ZPs disagree by more than 0.2 mag",ZP,b)
        ZP,b,sigZP,full = None,None,None,None
        return ZP,b,sigZP,chisq
    """
    
    return true_count,ZP,sigZP,m,b,chisq,cov

if __name__ == "__main__":
    efficiency_sdss_flags = pickle.load(open("efficiency_sdss_flags.pkl","rb"))
    lco_matches = pickle.load(open("lco_matches.pkl","rb"))
    sdss_matches = pickle.load(open("sdss_matches.pkl","rb"))
    phot = pickle.load(open("phot.pkl","rb"))

    print(len(efficiency_sdss_flags),len(lco_matches),len(sdss_matches),len(phot))
    # sort tables by origname
    res = [i for i in phot['origname']]
    orignames = list(set(res))
    print(len(orignames))

    for i in range(len(orignames)): #
        origname = orignames[i]
        lco_phot_tab = phot[phot['origname']==origname]
        matched_sdss = sdss_matches[sdss_matches['origname']==origname]
        assert(len(lco_phot_tab) == len(matched_sdss))

        print(origname)
        ZPimage(lco_phot_tab,matched_sdss,saveas=f"{origname}_zp.png")
        