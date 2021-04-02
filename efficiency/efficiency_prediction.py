import glob
import os
import pickle
import numpy as np
from scipy import interpolate
from scipy.spatial import ConvexHull
from astropy.table import Table
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import ticker, cm
import seaborn as sns
from astropy.io import ascii,fits
import util
from scipy import stats
from scipy.stats import pearsonr
from astropy.stats import sigma_clip
from astropy.stats import sigma_clipped_stats


def data_efficiency(df,params,coord,val='m50',
    parambounds=None,
    rescale=True,method='linear',fill_value=21,verbose=True):
    """
    Using scipy griddata: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html
    Interpolate unstructured D-D data.

    Provide an efficiency m50,alpha given params and coordinate in those params.

    Note df taken from the locally pickled pandas DataFrame ~ efficiencies.pkl 
    
    Parameters
    ------------
    df ~ pandas DataFrame
        has header parameters and m50,alpha, the data interpolated on
    params ~ list of strings
        header parameters you want to interpolate over (columns in the df)
    coord ~ tuple of floats
        header parameter points you want to know the efficiency values at (should match the params)
    vals ~ list of strings for efficiency values 
        ["m50","alpha"] the values we want to use for interpolation (columns in the df)


    Returns
    ___________
    m50 ~ float
        value at params-coord on scipy.interpolate.griddata 
    alpha ~ float
        value at params-coord on scipy.interpolate.griddata
    """


    if verbose:
        # print length of the df and the params,values phase-space
        print("{} df".format(len(df)))
        mins,maxes = [],[]
        print("params:")
        for param in params:
            mins.append(np.min(df[param]))
            maxes.append(np.max(df[param]))
            print("{} ~ [{:.2f},{:.2f}]".format(param,np.min(df[param]),np.max(df[param])))
        print("value:")
        print("{} ~ [{:.2f},{:.2f}]".format(val,np.min(df[val]),np.max(df[val])))


    if parambounds:
        # clip df if param falls outside of bound
        # easier to index using astropy table
        t = Table.from_pandas(df)
        assert(len(parambounds) == len(params))
        for i in range(len(params)):
            lo,hi = parambounds[i][0],parambounds[i][1]
            param = params[i]
            t = t[t[param] > lo]
            t = t[t[param] < hi]
        df = Table.to_pandas(t)
        if verbose:
            print("{} df after clipping params outside bounds".format(len(df)))    

    # put parameters and values into lists for scipy.griddata
    points,values=[],[]
    for i in range(len(df)):
        point = [df.iloc[i][param] for param in params]
        value = df.iloc[i][val] 
        points.append(point)
        values.append(value)
    hull = ConvexHull(points)
    # interpolate to get vals at params-coord 
    coord_val = interpolate.griddata(points,values,tuple(coord),method=method,rescale=rescale,fill_value=fill_value)

    return coord_val,hull

def etc_efficiency(coord,exptime=300,method='linear',rescale=True,fill_value=21):
    """
    Interpolate grid to determine 3 S/N magnitude predicted by ETC
    https://exposure-time-calculator.lco.global/

    Note assumes have etc.txt locally to read

    Parameters
    ___________
    coord : tuple
        (Moon,Airmass) Moon float [0,1] Airmass float [1,3]
    exptime : float
        300, 600, or 150 are accepted values
    Returns
    ___________
    limiting magnitude : float
        3 S/N rp limiting mag value at (Moon,Airmass) coord 
    """

    etc=ascii.read("etc.txt")
    etc = etc[etc['Exptime'] == exptime]
    x = [i for i in etc['Moon']]
    y = [i for i in etc['Airmass']]
    values = [i for i in etc['Magnitude']]
    points = []
    for i in range(len(x)):
        point = [x[i],y[i]]
        points.append(tuple(point))

    lim_mag_coord = interpolate.griddata(points, values, tuple(coord), method=method,rescale=rescale,fill_value=fill_value)

    return lim_mag_coord

if __name__ == "__main__":
    survey = pickle.load(open("lco_headers_df.pkl","rb"))
    efficiencies0 = pickle.load(open("efficiencies.pkl","rb"))

    metc = False
    if metc:
        # add column of metc to the headers using interp on the ETC 
        coords = [] # want tuple (MOONFRAC,AIRMASS) as coord for etc_efficiency
        metcs = []
        survey['MOONFRAC'] = survey['MOONFRAC'].astype(float)
        survey['AIRMASS'] = survey['AIRMASS'].astype(float)
        survey['EXPTIME'] = survey['EXPTIME'].astype(float)
        exptimes = [300,150,600]
        for i in range(len(survey)):
            moon = survey.iloc[i]['MOONFRAC']
            airmass = survey.iloc[i]['AIRMASS']
            exp = survey.iloc[i]['EXPTIME']
            idx = np.argmin([abs(i - exp) for i in exptimes])
            exptime = exptimes[idx]
            coord = tuple([moon,airmass])
            metc = etc_efficiency(coord,exptime=exptime,method='linear',rescale=True,fill_value=21)
            coords.append(coord)
            metcs.append(metc)
        survey['metc'] = metcs
        pickle.dump(survey,open("lco_headers_df.pkl","wb"))


    correlations = True
    if correlations:
        params = ['search_m50','search_alpha','ZPfit_PSF','EXPTIME','MOONFRAC','MOONDIST','MOONALT','WINDSPEE','WMSCLOUD','AIRMASS','L1SIGMA']
        for param in params:
            try:
                efficiencies0[param] = efficiencies0[param].astype(float)
            except:
                pass
        corrparam = 'ZPfit_PSF'
        tst = efficiencies0[params]
        tst = tst[tst[corrparam]>0] # filter None values
        tst[corrparam] = tst[corrparam].astype(float)
        pearsonR = tst.corr()
        print(pearsonR[corrparam].sort_values())

    interp = True
    if interp:
        # Pearson R, strong [0.5,1], medium [0.3,0.5], weak [0.1,0.3]         
        params = ['EXPTIME','MOONFRAC','MOONALT','MOONDIST','AIRMASS']
        val = 'ZPfit_PSF' #['search_m50','search_alpha']
        fill_value = -99.0 
        efficiencies = efficiencies0[efficiencies0[val]>0] # filter None values
        efficiencies[val] = efficiencies[val].astype(float)
        # reminder of the defaults
        rescale=True
        parambounds=None
        method='nearest'
        # for a demo
        single_coord = False
        if single_coord:
            # example demonstrating value at single params-coord
            coord = [300,0.5,45,20,1.5]
            coordval,_ = data_efficiency(efficiencies,params,coord,val=val,fill_value=fill_value,method=method,parambounds=None,verbose=True)
            print("single coord example")
            print("{} = {:.2f} at {} ~ {}".format(val,coordval,coord,params))
            print("----------------------------")
    
    test_interp = True
    if test_interp:
        # check vals from interp against the actual from pipeline 
        pipe_vals = []
        coord_vals = []
        deltas = [] 
        count_fills = 0
        for i in range(len(efficiencies)):
            # actual value from pipeline
            name = efficiencies.iloc[i].name
            pipe_val = efficiencies.iloc[i][val]
            coord = [efficiencies.iloc[i][parami] for parami in params]
            # I have the efficiencies rows indexed by the image names, use that in drop to remove that row from the interp
            data = efficiencies.drop(labels=name,axis=0)
            pkl = f'{val}_{method}_interp_droptest.pkl'
            coord_val,_ = data_efficiency(data,params,coord,val=val,fill_value=fill_value,method=method,parambounds=None,rescale=rescale,verbose=False)
            # (pipe - prediction) ~ deltas
            delta = pipe_val - coord_val
            # stick everything in lists
            pipe_vals.append(pipe_val)
            coord_vals.append(coord_val)
            deltas.append(delta)
            print("{}: {} = {}".format(name,coord,params))
            print("pipe_val - coord_val = {} = {} - {}".format(delta,pipe_val,coord_val))
            if delta > 100:
                hull = _
                print("Fill {}".format(count_fills))
                count_fills += 1
        efficiencies['ZP_interped'] = coord_vals
        test = [pipe_vals,coord_vals,deltas]
        plt.hist(deltas,density=True,histtype='step',label=r"$\delta$")
        plt.xlabel('measured - interped')
        plt.savefig(f"delta_{val}.png")
        pickle.dump(test,open(pkl,"wb"))
        pickle.dump(efficiencies,open("efficiencies_"+pkl,"wb"))
        

    