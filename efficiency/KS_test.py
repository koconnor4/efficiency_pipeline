from astropy.table import Table
from astropy.io import ascii,fits
import pandas as pd
import numpy as np
import pickle5 as p5
import pickle
from astropy.coordinates import SkyCoord
import scipy
from scipy import stats
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip
import matplotlib
from scipy import stats
from scipy.stats import pearsonr
import seaborn as sns 
import efficiency_func
import util
import glob
import os


def dataclipping(t):
    """
    To clip data to remove nans, and/or meaningless outliers
    t ~ astropy.table
    (note df = Table.to_pandas(t) or t = Table.from_pandas(df)
    """

    print("starting with",len(t))
    # first off removing any which didn't come in set of at least 3 frames and/or had less than 150s of exposure
    t['FRMTOTAL'] = t['FRMTOTAL'].astype(int)
    t['EXPTIME'] = t['EXPTIME'].astype(float)
    t = t[t['FRMTOTAL'] >= 3]
    print("frametot >= 3 leaves",len(t))
    t = t[t['EXPTIME'] >= 149.0]
    print("exptime >= 149.0 leaves",len(t))

    # set of parameters
    params = ["AIRMASS","SCHEDSEE","L1FWHM",
        "L1MEAN","L1MEDIAN","L1SIGMA","L1ELLIP",
        "EXPTIME","GAIN","RDNOISE","DARKCURR","SATURATE","MAXLIN","RDSPEED",
        "AGFWHM","AGLCKFRC",
        "CCDSTEMP","CCDATEMP",
        "WMSHUMID","WMSTEMP","WMSPRES","WINDSPEE","WMSMOIST","WMSDEWPT","WMSCLOUD",
        "SKYMAG","WMSSKYBR",
        "MOONFRAC","MOONDIST","MOONALT","SUNDIST","SUNALT"] 

    # take only floats
    for param in params:
        rvs = []
        try:
            # catch any masked remove those off the top
            t = t[t[param] != np.ma.masked]
        except:
            pass
        # will use t.remove_rows(list) to get anything now which cant be a float
        good,bad = [],[] 
        idx=0
        for i in t[param]:
            try:
                rvs.append(np.float64(i))
                good.append(idx)
            except:
                bad.append(idx)
            idx+=1
        nanmin=np.nanmin(rvs)
        nanmax=np.nanmax(rvs)
        print(param,nanmin,nanmax)
        t.remove_rows(bad)
        try:
            t[param] = t[param].astype(np.float64)
        except:
            pass

    print("removing non floats in params leaves",len(t))

    return t


def KStest(param,rvs=None,cdf=None,clip=True,writetodisk=True,plot=True,limits=None,label=None,bins=None):
    """
    Determine KS-test statistic and pvalue for random variables (rvs) with cumulative distribution (cdf)
    The Kolmogorov–Smirnov statistic quantifies a distance between the empirical distribution function of the sample 
    and the cumulative distribution function of the reference distribution, or between the empirical distribution functions 
    of two samples. 

    Parameters
    --------------------
    param : str 
        Should be available in LCO Header, examples: L1FWHM, WMSSKYBR
    rvs : str or None
        Default None loads data and indexes for param. Can instead provide str to load a pickled 1d-array like rvs.
        Note if provide str needs to match param, i.e. rvs = param_rvs.pkl
    cdf : Default None loads data and indexes for param. Can instead provide str to load a pickled 1d-array like cdf.
        Note if provide str needs to match param, i.e. cdf = param_cdf.pkl
    writetodisk: bool
        default True will pickle the 1d-array like rvs and cdf. Saves time avoiding load/index data for distributions.

    Returns 
    --------------------
    stat : float
        sup_x | Fn(x) - F(x) | the supremum is the set of distances, converges to zero if sample Fn(x) comes from F(x) 
    pval : float 
        Reject the null hypothesis that the two samples were drawn from the same distribution 
        if the p-value is less than your significance level. i.e. p ~ .95 is 95% confident they came from same. 
    
    
    Reminder of statistics 101
    The significance level, also denoted as alpha or α, is the probability of rejecting the null hypothesis when it is true. 
    For example, a significance level of 0.05 indicates a 5% risk of concluding that a difference exists when there is no actual difference.
    """

    print("Doing KS test of the differences data we have against full survey data")
    if rvs != None and cdf != None:
        rvs_split,cdf_split = rvs.split("_")[1],cdf.split("_")[1]
        rvs_split,cdf_split = rvs_split.split(".pkl")[0],cdf_split.split(".pkl")[0]
        try:
            assert(rvs_split == cdf_split == param)
            # asserted that they match so read the pickles
            rvs,cdf = pickle.load(open(rvs,"rb")),pickle.load(open(cdf,"rb"))
            print("Using {}".format(param))
            print("{} rvs, {} cdf".format(len(rvs),len(cdf)))
            if clip:
                qclip = [0.05,0.95] # take 90% of the data, removing bad outliers (upper/lower 10%)
                rvsq = np.nanquantile(np.array(rvs,dtype=float),qclip)
                cdfq = np.nanquantile(np.array(cdf,dtype=float),qclip)
                rvs = [i for i in rvs if i > rvsq[0] and i < rvsq[1]]
                cdf = [i for i in cdf if i > cdfq[0] and i < cdfq[1]]
                print("clipped outliers/nans now {} rvs, {} cdf".format(len(rvs),len(cdf)))
                print("rvs limits",rvsq)
                print("cdf limits",cdfq)
                #rvs=sigma_clip(rvs,sigma=5)
                #cdf=sigma_clip(cdf,sigma=5)
            q = [0.25,0.5,0.75] # quartiles 
            rvsq = np.quantile(np.array(rvs,dtype=float),q)
            cdfq = np.quantile(np.array(cdf,dtype=float),q)
            rvsq = [np.round(i,2) for i in rvsq]
            cdfq = [np.round(i,2) for i in cdfq]
            q1 = [rvsq[0],cdfq[0]]
            q2 = [rvsq[1],cdfq[1]]
            q3 = [rvsq[2],cdfq[2]]
            stat,pval=stats.kstest(rvs,cdf)
            print(stat,pval)

            #print(rvsq)
            #print(cdfq)
            if plot:
                rvs_color,cdf_color = "red","black"
                plt.hist(rvs,density=True,label="rvs",color=rvs_color,histtype='step')#,bins=bins)
                plt.hist(cdf,density=True,label="cdf",color=cdf_color,histtype='step')#,bins=bins)
                if label == None:
                    plt.xlabel(param)
                else:
                    plt.xlabel(label)
                if limits == None:
                    xmin,xmax = np.nanmin(cdf)-np.std(cdf),np.nanmax(cdf)+np.std(cdf)
                else:
                    xmin,xmax=limits

                plt.xlim(xmin,xmax)
                obj_0 = util.AnyObject(r"$p$ =", "black")
                obj_1 = util.AnyObject("quartiles =", "red")
                obj_2 = util.AnyObject("quartiles =", "black")
                plt.legend([obj_0,obj_1,obj_2], ['{:.1}'.format(pval),'\n \n {}'.format(rvsq),'\n \n {}'.format(cdfq)],
                    handler_map={obj_0:util.AnyObjectHandler(),obj_1:util.AnyObjectHandler(),obj_2:util.AnyObjectHandler()})
                #plt.legend()
                plt.show()
                if writetodisk:
                    plt.savefig("dist_"+param+".png")
                plt.close()
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html
            # returns statistic,pvalue
            return stat,pval
        except:
            print("rvs and cdf provided dont match {}".format(param))
            return
    
    # load the difference data
    fz = glob.glob("differences/*/*/_dithers/*/*fits*")
    print("Total fits in _dithers",len(fz))
    exps = [i for i in fz if os.path.basename(i)[0] != 'd']
    diffs = [i for i in fz if os.path.basename(i)[0] == 'd']
    print("Total exps, diffs",len(exps),len(diffs))
    hdrs = []
    for i in exps:
        #os.system(f"fitscheck {i}")
        hdu = fits.open(i)
        hdr = hdu[1].header
        hdrs.append(hdr)

    print("{} image data, {} survey data".format(len(hdrs),len(LCO)))

    # make sure the param is available
    try: 
        assert(param in LCO.columns)
        print("Using {}".format(param))
    except:
        print("{} not in {}".format(param,LCO.columns))

    # read it to 1d-array floats ... a little ugly since have strings/nans 
    rvs = []
    for i in hdrs:
        try:
            rvs.append(float(i[param]))
        except:
            pass
    cdf = []
    for i in LCO[param]:
        try:
            cdf.append(float(i))
        except:
            pass

    print("{} rvs, {} cdf".format(len(rvs),len(cdf)))


    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html
    # returns statistic,pvalue
    stat,pval=stats.kstest(rvs,cdf)

    print(stat,pval)

    if writetodisk:
        rvs_name,cdf_name = "rvs_" + param + ".pkl", "cdf_" + param + ".pkl"
        print("pickling {} and {}".format(rvs_name,cdf_name))
        pickle.dump(rvs, open(rvs_name,"wb"))
        pickle.dump(cdf, open(cdf_name,"wb"))

    return stat,pval

if __name__ == "__main__":
    difftab = False
    if difftab:
        # load the difference data
        fz = glob.glob("differences/*/*/_dithers/*/*fits*")
        print("Total fits in _dithers",len(fz))
        exps = [i for i in fz if os.path.basename(i)[0] != 'd']
        diffs = [i for i in fz if os.path.basename(i)[0] == 'd']
        print("Total exps, diffs",len(exps),len(diffs))
        hdrs = []
        idx = 0
        dfs = []
        for i in exps:
            #os.system(f"fitscheck {i}")
            hdu = fits.open(i)
            hdr = hdu[1].header
            hdrs.append(hdr)
            dfi = util.table_header(hdu[1],idx)
            dfs.append(dfi)
            idx+=1
        df = pd.concat(dfs)
        print(len(df))
        pickle.dump(df,open("DIFFS_HDRS_DF.pkl","wb"))

    dc = False
    if dc:
        # LCO ~ all of the headers from entire survey
        # DIFFS ~ all of the headers for images we have differences for (to measure efficiency on)
        LCO = p5.load(open("survey/LCO_HDRS_DF.pkl","rb"))
        DIFFS = pickle.load(open("differences/DIFFS_HDRS_DF.pkl","rb"))
        print("LCO headers from all proposals",len(LCO))
        print("DIFF headers available for efficiency measurements",len(DIFFS))
        t = Table.from_pandas(LCO)
        LCO = dataclipping(t)
        t = Table.from_pandas(DIFFS)
        DIFFS = dataclipping(t)

    LCO = p5.load(open("survey/LCO_HDRS_DF.pkl","rb"))
    DIFFS = pickle.load(open("differences/DIFFS_HDRS_DF.pkl","rb"))
    print("LCO headers from all proposals",len(LCO))
    print("DIFF headers available for efficiency measurements",len(DIFFS))
    cov=True
    if cov:
        pairs = [["MOONDIST","AIRMASS"],["MOONALT","MOONFRAC"],["AIRMASS","L1FWHM"],["SKYMAG","WMSSKYBR"]]
        pair = pairs[0]
        param1 = pair[0] 
        param2 = pair[1]

        data1 = LCO[param1]
        data2 = LCO[param2]
        data1 = DIFFS[param1]
        data2 = DIFFS[param2]

        print("{} data1, {} data2".format(len(data1),len(data2)))
        clip = False
        if clip:
            qclip = [0.025,0.975] # take 95% of the data, removing edges (upper/lower 2.5%)
            data1q = np.nanquantile(np.array(data1,dtype=float),qclip)
            data2q = np.nanquantile(np.array(data2,dtype=float),qclip)
            data1 = [i for i in data1 if i > data1q[0] and i < data1q[1]]
            data2 = [i for i in data2 if i > data2q[0] and i < data2q[1]]
            print("clipped outliers/nans now {} rvs, {} cdf".format(len(data1),len(data2)))
            print("data1 limits",data1q)
            print("data2 limits",data2q)

        efficiency_func.plot_cov(data1,data2,xlabel=param1,ylabel=param2,verbose=True,saveas=f"cov_{param1}_{param2}.png")
        
    KS = False
    if KS:
        params = ["AIRMASS","SCHEDSEE","L1FWHM",
        "L1MEAN","L1MEDIAN","L1SIGMA","L1ELLIP",
        "EXPTIME","GAIN","RDNOISE","DARKCURR","SATURATE","MAXLIN","RDSPEED",
        "AGFWHM","AGLCKFRC",
        "CCDSTEMP","CCDATEMP",
        "WMSHUMID","WMSTEMP","WMSPRES","WINDSPEE","WMSMOIST","WMSDEWPT","WMSCLOUD",
        "SKYMAG","WMSSKYBR",
        "MOONFRAC","MOONDIST","MOONALT","SUNDIST","SUNALT"] 
        
        
        #limits = [(0.75,2.25),(0,10),(0,5.5),(13.5,23.5),(0,1),(0,180)]
        #bins = [np.arange(0.75,2.25,0.15),np.arange(0,10,0.5),np.arange(0,5.5,0.5),np.arange(14.5,25.5,1),np.arange(0,1,0.1),np.arange(40,160,15)]
        #labels = ["AIRMASS","SEEING [arcsec]","FWHM [arcsec]", "WMSSKYBR [mag/arcsec^2]",None,None]
        #limit,label,bini=limits[i],labels[i],bins[i]
        #KStest(param,rvs=rvs,cdf=cdf,limits=limit,label=label,bins=bini)
        i=0
        for i in range(len(params)): #
            param = params[i]
            if param in ["AIRMASS","SCHEDSEE","L1FWHM","L1MEAN","L1MEDIAN","L1SIGMA","L1ELLIP",
            "WMSHUMID","WMSTEMP","WMSPRES","WINDSPEE","WMSMOIST","WMSDEWPT","WMSCLOUD","SKYMAG","WMSSKYBR"]:
                clip = True
            else:
                clip = False
            #KStest(param)
            rvs,cdf = "rvs_" + param + ".pkl", "cdf_" + param + ".pkl"
            stat,pval=KStest(param,rvs=rvs,cdf=cdf,label=param,clip=clip)
            print(param,stat,pval)
            i+=1
    