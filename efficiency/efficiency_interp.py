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
from astropy.io import ascii
import util
from scipy import stats
from scipy.stats import pearsonr
from astropy.stats import sigma_clip

def data_efficiency(df,params,coord,vals=["m50","alpha"],
    parambounds=None,
    rescale=True,method='linear',fill_m50=21,fill_alpha=2,verbose=True):
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
        print("values:")
        for val in vals:
            mins.append(np.min(df[val]))
            maxes.append(np.max(df[val]))
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
    points,values,m50_values,alpha_values=[],[],[],[]
    for i in range(len(df)):
        point = [df.iloc[i][param] for param in params]
        value = [df.iloc[i][val] for val in vals]
        m50_values.append(df.iloc[i]['m50'])
        alpha_values.append(df.iloc[i]['alpha'])
        points.append(point)
        values.append(value)
    hull = ConvexHull(points)

    # interpolate to get m50 and alpha at params-coord 
    m50_coord = interpolate.griddata(points,m50_values,tuple(coord),method=method,rescale=rescale,fill_value=fill_m50)
    alpha_coord = interpolate.griddata(points,alpha_values,tuple(coord),method=method,rescale=rescale,fill_value=fill_alpha) 

    return m50_coord,alpha_coord

def etc_efficiency(coord,method='linear',rescale=True,fill_value=21):
    """
    Interpolate grid to determine 3 S/N magnitude predicted by ETC
    https://exposure-time-calculator.lco.global/

    Note assumes have etc.txt locally to read

    Parameters
    ___________
    coord : tuple
        (Moon,Airmass) Moon float [0,1] Airmass float [1,3]
    Returns
    ___________
    limiting magnitude : float
        3 S/N rp limiting mag value at (Moon,Airmass) coord 
    """

    etc=ascii.read("etc.txt")
    x = [i for i in etc['Moon']]
    y = [i for i in etc['Airmass']]
    values = [i for i in etc['Magnitude']]
    points = []
    for i in range(len(x)):
        point = [x[i],y[i]]
        points.append(tuple(point))

    lim_mag_coord = interpolate.griddata(points, values, tuple(coord), method=method,rescale=rescale,fill_value=fill_value)

    return lim_mag_coord

def df_efficiencies(verbose=False):
    """
    Read in the locally pickled pandas DataFrame with m50,alpha from pipeline along with all the header parameters
    Removing any of the rows without an m50,alpha value (i.e. any that failed in pipeline and were flagged)
    """

    df = pickle.load(open("efficiencies.pkl","rb"))
    # easier to index as astropy table
    t = Table.from_pandas(df)
    t = t[t['m50'] != np.ma.masked]
    if verbose:
        print("{} df".format(len(df)))
        print("{} df after removing those without efficiency (m50/alpha) values".format(len(t)))
    df = Table.to_pandas(t)    
    return df

if __name__ == "__main__":
    df = df_efficiencies(verbose=True)
    print('------------------------------')

    interp = True
    if interp:
        params,vals = ['MOONDIST','AIRMASS'],['m50','alpha']
        parambounds = [[0,180],[1,2]] 
        # reminder of the defaults
        rescale=True
        method='linear'
        fill_m50=21
        fill_alpha=2

    single_coord = True
    if single_coord:
        # example demonstrating the m50, alpha efficiency values at single params-coord
        coord = [90,1.5]
        m50,alpha = data_efficiency(df,params,coord,parambounds=parambounds,verbose=True)
        print("single coord example")
        print("m50={:.2f},alpha={:.2f} at {} ~ {}".format(m50,alpha,coord,params))
        print("----------------------------")
        
    test_interp = True
    if test_interp:
        # leave out one row of df at a time, interp over the rest
        # check m50 and alpha from that interp against the actual from pipeline 
        pipe_m50s, pipe_alphas = [],[]
        interp_m50s, interp_alphas = [],[]
        delta_m50s, delta_alphas = [],[] 
        for i in range(len(df)):
            # actual value from pipeline
            name,pipe_m50,pipe_alpha = df.iloc[i].name,df.iloc[i]['m50'],df.iloc[i]['alpha']
            coord = [df.iloc[i]['MOONDIST'],df.iloc[i]['AIRMASS']]
            # I have the df rows indexed by the image names, use that in drop to remove that row from the interp
            data = df.drop(labels=name,axis=0)
            m50,alpha = data_efficiency(data,params,coord,parambounds=parambounds,verbose=False)
            # (pipe - data) ~ deltas
            dm50 = pipe_m50 - m50
            dalpha = pipe_alpha - alpha

            # stick everything in lists
            pipe_m50s.append(pipe_m50)
            pipe_alphas.append(pipe_alpha)
            interp_m50s.append(m50)
            interp_alphas.append(alpha)
            delta_m50s.append(dm50)
            delta_alphas.append(dalpha)

            print("{} dm50 = {} = {} - {}".format(name,dm50,pipe_m50,m50))

        test = [delta_m50s,delta_alphas,pipe_m50s,pipe_alphas,interp_m50s,interp_alphas]
        pickle.dump(test,open("interp_test.pkl","wb"))


    grid_coord = False
    if grid_coord:
        # the m50, alpha efficiency values at many params-coords (for visualizing using colormaps) 
        x = [i for i in df[params[0]]]
        y = [i for i in df[params[1]]]
        Cm50 = [i for i in df['m50']]
        Calpha = [i for i in df['alpha']]
        
        # check if already have the grid of values run and pickled
        try:
            tmp = pickle.load(open(f"efficiencygrid100x100_{params[0].lower()}_{params[1].lower()}.pkl","rb"))
            xis,yis,ci_m50s,ci_alphas,x,y,Cm50,Calpha = tmp
            print("already have the efficiency_grid loaded pickle")
        # make 
        except:
            print("running and pickling for an efficiency_grid")
            xlin = np.linspace(parambounds[0][0],parambounds[0][1],100)
            ylin = np.linspace(parambounds[1][0],parambounds[1][1],100)
            xis,yis,ci_m50s,ci_alphas = [],[],[],[]
            j=0
            for xi in xlin:
                for yi in ylin:
                    coord = [xi,yi]
                    print(j)
                    j+=1
                    ci_m50,ci_alpha,df0 = data_efficiency(df,params,coord,parambounds=parambounds,verbose=False) 
                    xis.append(xi)
                    yis.append(yi)
                    ci_m50s.append(ci_m50)
                    ci_alphas.append(ci_alpha)

            tmp = [xis,yis,ci_m50s,ci_alphas,x,y,Cm50,Calpha]
            pickle.dump(tmp,open("efficiencygrid100x100_{}_{}.pkl".format(params[0].lower(),params[1].lower()),"wb"))
            

        gridsize = (10,10)
        # plot the hb colormap to m50 and alpha values over the param-coords grid
        title,saveas = "$m_{50}$",f"m50_{params[0]}_{params[1]}"
        plot_interp.plot_efficiency(xis,yis,ci_m50s,gridsize=gridsize,x=x,y=y,C=Cm50,saveas=saveas,xlim=parambounds[0],ylim=parambounds[1],xlabel=params[0],ylabel=params[1],ticks=None,title=title)
        title,saveas = r"$\alpha$",f"alpha_{params[0]}_{params[1]}"
        plot_interp.plot_efficiency(xis,yis,ci_alphas,gridsize=gridsize,x=x,y=y,C=Calpha,saveas=saveas,xlim=parambounds[0],ylim=parambounds[1],xlabel=params[0],ylabel=params[1],ticks=None,title=title)
    
        
            
        

    