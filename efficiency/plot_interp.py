def plot_f_efficiency(df,dist=True):
    """
    make curves showing efficiency function using quantiles of m50,alpha values
    distribution covariance of m50,alpha with hists attached
    distribution of uncertainties m50,alpha as estimated optimal parameter covariance (from the fits)
    """
    matplotlib.rcParams.update({'font.size': 20,'xtick.labelsize':15,'ytick.labelsize':15})
    print("{} data".format(len(df)))
    # make sure we have m50,alpha, indexing using astropy table 
    t = Table.from_pandas(df)
    t = t[t['m50'] != np.ma.masked]
    df = Table.to_pandas(t)
    print("{} data after clipping empty efficiencies".format(len(df)))
    
    m50 = np.array(df['m50'],dtype='float')
    rounded = np.round(m50,1) # for mode round m50 vals to .1 mag
    iqr = stats.iqr(rounded) # 75th and 25th percentile diff
    q = [0.25,0.5,0.75] # quantiles 
    m50_quantiles = np.quantile(rounded,q)
    
    mags = np.linspace(19,23.5,100)
    mode,median,mean = stats.mode(rounded),np.median(df['m50']),np.mean(df['m50'])
    sigma = np.std(df['m50'])
    print("m50:")
    print("quantiles",m50_quantiles,q)
    print("iqr",iqr)
    print("mode,median,mean",mode,median,mean)
    print("sigma",sigma)
    mode = mode[0][0]

    alpha = np.array(df['alpha'],dtype='float')
    alpha = np.round(alpha,0) # round alpha to an int
    alpha_quantiles = np.quantile(alpha,q)
    iqr = stats.iqr(alpha)
    alpha_mode = stats.mode(alpha)
    print("alpha:")
    print("quantiles",alpha_quantiles,q)
    print("iqr",iqr)
    print("mode",alpha_mode)
    alpha_mode = alpha_mode[0][0]

    q25_fracs = [util.f_efficiency(mi,m50_quantiles[0],alpha_quantiles[0]) for mi in mags]
    q50_fracs = [util.f_efficiency(mi,m50_quantiles[1],alpha_quantiles[1]) for mi in mags]
    q75_fracs = [util.f_efficiency(mi,m50_quantiles[2],alpha_quantiles[2]) for mi in mags]

    fig = plt.figure(figsize=(16,8))
    plt.plot(mags,q25_fracs,label='',color='black',ls='--')
    plt.plot(mags,q50_fracs,label=r'$m50 = $ {0:.1f}, $\alpha = $ {1:.1f}'.format(m50_quantiles[1],alpha_quantiles[1]),color='black')
    plt.plot(mags,q75_fracs,label='',color='black',ls='--')

    plt.xlim(20.0,23.25)
    plt.xlabel(r"$r$")
    plt.ylabel(r"$\epsilon$")
    plt.legend()
    plt.show()
    plt.savefig("efficiency_curve.png")

    plt.close()

    if dist:
        sigm50 = np.array(df['sig_m50'],dtype='float')
        sigalpha = np.array(df['sig_alpha'],dtype='float')
        sigZP = np.array(df['sigZP'],dtype='float')
        medm50,medalpha,medZP = np.median(sigm50),np.median(sigalpha),np.median(sigZP)
        binsm50 = np.arange(0,0.3,0.02)
        binsalpha = np.arange(0,3,0.3)
        binsZP =np.arange(0,0.2,0.02)

        # uncertainties zp,m50,alpha
        fig,(ax1,ax2,ax3) = plt.subplots(3)
        ax1.hist(sigZP,bins=binsZP,density=True,histtype='step',label=r"$\delta ZP = $ {0:.2g}".format(np.median(sigZP)))
        ax2.hist(sigm50,bins=binsm50,density=True,histtype='step',label=r"$\delta m50 = $  {0:.2g}".format(np.median(sigm50)))
        ax3.hist(sigalpha,bins=binsalpha,density=True,histtype='step',label=r"$\delta \alpha = $ {0:.2g}".format(np.median(sigalpha)))
        ax1.legend()
        ax2.legend()
        ax3.legend()
        plt.savefig("efficiency_uncertainties.png")
        plt.close()
        # now the m50,alpha distributions
        # definitions for the axes
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        spacing = 0.005
        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom + height + spacing, width, 0.2]
        rect_histy = [left + width + spacing, bottom, 0.2, height]
        matplotlib.rcParams.update({'font.size': 20,'xtick.labelsize':15,'ytick.labelsize':15})
        # start with a rectangular Figure
        fig = plt.figure(figsize=(16,8))
        ax_scatter = plt.axes(rect_scatter)
        ax_scatter.tick_params(direction='in', top=True, right=True)
        ax_histx = plt.axes(rect_histx,sharex=ax_scatter)
        ax_histx.tick_params(direction='in',labelbottom=False)
        ax_histy = plt.axes(rect_histy,sharey=ax_scatter)
        ax_histy.tick_params(direction='in', labelleft=False)

        # the covariance 
        x,y = m50,alpha
        m = x
        cov = np.cov(m,y=y)
        #ax_scatter.scatter(x,y,marker='o',c='black',alpha=0.5) # .errorbar?
        levels = [0.33,0.67,0.95]
        sns.kdeplot(x,y=y,levels=levels,fill=False,label="",color='grey',ax=ax_scatter)#,cmap='mako',cbar=False
        xmin,xmax=ymin,ymax=-100,100
        ax_scatter.hlines(alpha_quantiles[0],xmin,xmax,label="q25",ls='--',color='black')
        ax_scatter.hlines(alpha_quantiles[1],xmin,xmax,label="q50",ls='-',color='black')
        ax_scatter.hlines(alpha_quantiles[2],xmin,xmax,label="q75",ls='--',color='black')
        ax_scatter.vlines(m50_quantiles[0],ymin,ymax,label="q25",ls='--',color='black')
        ax_scatter.vlines(m50_quantiles[1],ymin,ymax,label="q50",ls='-',color='black')
        ax_scatter.vlines(m50_quantiles[2],ymin,ymax,label="q75",ls='--',color='black')
        # alpha hist
        bins = np.arange(0.5,15,0.75)
        alpha = np.array(df['alpha'],dtype=float)
        ax_histy.hist(alpha,density=True,label=r"$\alpha$",color='black',histtype='step',bins=bins,orientation='horizontal')
        # m50 hist
        bins = np.arange(19,24,0.3)
        ax_histx.hist(m50,density=True,label="$m_{50}$",color='black',histtype='step',bins=bins)
        ymin,ymax=0,1

        
        ax_scatter.set_xlabel("$m_{50}$")
        ax_scatter.set_ylabel(r"$\alpha$")
        ax_scatter.set_xlim(20.5,24)
        ax_scatter.set_ylim(-1,10)
        ax_scatter.set_xticks([21,22,23])
        ax_scatter.set_yticks([2,4,6,8])

        obj_0 = util.AnyObject(r"$cov$ =", "black")
        obj_1 = util.AnyObject("quartiles =", "black")
        cov = np.round(cov,1)
        q1,q2,q3 = [m50_quantiles[0],alpha_quantiles[0]],[m50_quantiles[1],alpha_quantiles[1]],[m50_quantiles[2],alpha_quantiles[2]]
        plt.legend([obj_0,obj_1], ['{}'.format(cov),'\n \n {},{},{}'.format(q1,q2,q3)],
           handler_map={obj_0:util.AnyObjectHandler(),obj_1:util.AnyObjectHandler()})

        plt.savefig("cov_m50alpha.png")
        plt.close()

    return

def plot_cov(data1,data2,saveas="cov_xy.png",xlabel=None,ylabel=None,verbose=False):
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]
    matplotlib.rcParams.update({'font.size': 20,'xtick.labelsize':15,'ytick.labelsize':15})
    # start with a rectangular Figure
    fig = plt.figure(figsize=(8,8))
    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.tick_params(direction='in', top=True, right=True)
    ax_histx = plt.axes(rect_histx,sharex=ax_scatter)
    ax_histx.tick_params(direction='in',labelbottom=False)
    ax_histy = plt.axes(rect_histy,sharey=ax_scatter)
    ax_histy.tick_params(direction='in', labelleft=False)

    # the covariance 
    x,y = data1,data2
    m = x
    cov = np.cov(m,y=y)
    r = pearsonr(data1,data2) # correlation coefficient ~ varxy/(sigmax*sigmay)
    if verbose:
        print("cov",cov)
        print("pearsonr",r)

    ax_scatter.scatter(x,y,marker='o',c='black',alpha=0.5) # .errorbar?
    levels = [0.33,0.67,0.95]
    sns.kdeplot(x,y=y,levels=levels,fill=False,label="",color='grey',ax=ax_scatter)
    
    ax_histy.hist(data2,density=True,label="",color='black',histtype='step',orientation='horizontal')
    ax_histx.hist(data1,density=True,label="",color='black',histtype='step')

    if True:
        ax_scatter.set_xlabel(xlabel)
        ax_scatter.set_ylabel(ylabel)
        yscs = sigma_clip(data2,sigma=5)
        xscs = sigma_clip(data1,sigma=5)
        ylim = [np.nanmin(yscs),np.nanmax(yscs)]
        xlim = [np.nanmin(xscs),np.nanmax(xscs)]
        ax_scatter.set_xlim(xlim)
        ax_scatter.set_ylim(ylim)
        #ax_scatter.set_xticks(xticks)
        #ax_scatter.set_yticks(yticks)

    obj_0 = util.AnyObject(r"$cov$ =", "black")
    obj_1 = util.AnyObject(r"$R$ =", "black")
    #cov = np.round(cov,1)
    bcov = ["{0:.2g}".format(i) for i in cov.flatten()]
    bcov = [float(i) for i in bcov]
    bcov =np.array([[bcov[0],bcov[1]],[bcov[2],bcov[3]]])
    br = ["{0:.2g}".format(i) for i in r]
    br = [float(i) for i in br]
    print(bcov,br)

    plt.legend([obj_0,obj_1], ['{}'.format(bcov),'{}'.format(br)],
       handler_map={obj_0:util.AnyObjectHandler(),obj_1:util.AnyObjectHandler()})

    if saveas:
        plt.savefig(saveas)
    plt.close()



def plot_efficiency(xi,yi,Ci,gridsize=None,x=None,y=None,C=None,levels=None,etc=False,saveas=True,
    xlim=None,ylim=None,ticks=None,xlabel=None,ylabel=None,title=""):
    """
    Visualize the efficiency grid

    x,y,C: seaborn kde density plot shows contours/scattered data distribution of parameters x,y that efficiency grid is made on
        x,y ~ WMSSKYBR,L1FWHM
    
    xi,yi,Ci: mpl hexbin shows colorbar representing efficiency value interpolated over the x,y-grid
        C ~ m50 or alpha

    Parameters
    __________
    x : list
        x-data
    y : list
        y-data
    C : list
        efficiency-data (m50,alpha,or etc val)

    xi : list
        x-interpolated data
    yi : list
        y-interpolated data
    Ci : list
        m50 or alpha interpolated value from interpolate.griddata at xi,yi point

    levels : list 
        for kdeplot which contours to show distribution of x,y data
    gridsize : int
        number of bins for hexdata in x-direction. y is chosen to make regular hexagons. 
    ticks : list
        show ticks on colorbar for hb
    xlim : list 
        set xlimits on the plots
    ylim : list
        set ylimits on the plots
    Returns
    _______
    if etc = False:
        efficiency_hb_saveas.png    (the hexbin with colorbars for m50,alpha on data interpolated over x,y)
        efficiency_data_saveas.png (the contours/scatter of x,y data)
    if etc = True:
        ETC_hb.png (hexbin with colorbar for 3 S/N limiting magnitude over moonfrac,airmass)
    """

    # to-do labels
    if gridsize == None:
        gridsize = 50
    if levels == None:
        levels=[0.33,0.67,0.95]
    if xlim == None:
        # assume Moonfrac
        xlim = [0,1]
    if ylim == None:
        # assume Moonalt
        ylim = [-90,90]
    if ticks == None:
        # assume colorbar for m50
        ticks = [23,22.5,22,21.5,21,19.5,19]



    if etc:
        # efficiency from the ETC
        xlim,ylim=[0,1],[1,3] # moon,airmass
        hb = plt.hexbin(xi,yi,C=Ci,gridsize=gridsize,cmap='viridis')
        ticks = [23,22.5,22,21.5,21,20.5]
        cb = plt.colorbar(hb,ticks=ticks,format='%.2f') 
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel("MOONFRAC")
        plt.ylabel("MOONALT")
        plt.title("$m_{ETC}$")
        if saveas:
            plt.savefig("ETC_hb.png")
        plt.close()

    else:
        # efficiency from the data 
        hb = plt.hexbin(xi,yi,C=Ci,gridsize=gridsize,cmap='viridis')
        cb = plt.colorbar(hb,format='%.2f') #ticks=ticks
        plt.xlim(xlim) 
        plt.ylim(ylim)
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        plt.title(title)
        if saveas:
            if type(saveas) == str:
                plt.savefig(f"efficiency_hb_{saveas}.png")
            else:
                plt.savefig("efficiency_hb.png")
        plt.close()

        try:
            assert(x!=None and y!=None and C!=None)
            # plot contours of the data if provided
            sns.kdeplot(x,y=y,levels=levels,fill=False,label="",cmap='mako',cbar=False)
            plt.scatter(x,y,marker='o',c='b',s=5)
            plt.xlim(xlim)
            plt.ylim(ylim)
            if saveas:
                if type(saveas) == str:
                    plt.savefig(f"efficiency_data_{saveas}.png",bbox_inches='tight')
                else:
                    plt.savefig("efficiency_data.png",bbox_inches='tight')                    
            plt.close()
        except:
            pass
    
    return 

if __name__ == "__main__":
    covar = False
    if covar:
        df = pickle.load(open("efficiency_sdss_flags.pkl","rb"))
        t = Table.from_pandas(df)
        t = t[t['m50'] != np.ma.masked]
        df = Table.to_pandas(t)        
        data1 = np.array(df['m50'],dtype=float)
        data2s = ["MOONFRAC","MOONALT","MOONDIST","WMSSKYBR","AIRMASS","L1FWHM"] #"SCHEDSEE"
        for d2 in data2s:
            data2 = np.array(df[d2],dtype=float)
            print("m50,{}".format(d2),len(data1),len(data2))
            plot_cov(data1,data2,saveas=f"cov_m50_{d2}.png",xlabel=r"$m_{50}$",ylabel=d2,verbose=True)
        data1 = np.array(df['alpha'],dtype=float)
        for d2 in data2s:
            data2 = np.array(df[d2],dtype=float)
            print("alpha,{}".format(d2),len(data1),len(data2))
            plot_cov(data1,data2,saveas=f"cov_alpha_{d2}.png",xlabel=r"$\alpha$",ylabel=d2,verbose=True)

    corr = False
    if corr:
        tmp = pickle.load(open("corr_efficiency.pkl","rb"))
        print(len(tmp))
        print(tmp['m50'])
        
    f_efficiency = False
    if f_efficiency:
        df = pickle.load(open("efficiency_sdss_flags.pkl","rb"))
        print(len(df))
        print(len(df[df['flag'] == 0]))
        
        # corr, use pandas builtin correlation coefficient (pearson-default) to find strongest correlations 
        corr = df.corr()
        print(len(corr))
        print(corr)
        pickle.dump(corr,open("corr_efficiency.pkl","wb"))

        tmp = df[df['ZP'] > 0]
        zps = [i for i in tmp['ZP']]
        chisq = [i for i in tmp['chisqZP']]
        print(np.nanmedian(zps),np.nanstd(zps))
        print(np.nanmedian(chisq))
        print('--------------------')
        tmp = df[df['m50'] <= 21]
        print(len(tmp))
        zps = [i for i in tmp['ZP']]
        chisq = [i for i in tmp['chisqZP']]
        print(np.nanmedian(zps),np.nanstd(zps))
        print(np.nanmedian(chisq))
        #print(zps)
        print('-------------------')
        tmp = df[df['m50'] >= 23.2]
        print(len(tmp))
        zps = [i for i in tmp['ZP']]
        chisq = [i for i in tmp['chisqZP']]
        print(np.nanmedian(zps),np.nanstd(zps))
        print(np.nanmedian(chisq))
        #print(zps)


        #plot_f_efficiency(df)

    etc=False
    if etc:
        # provide Moonfrac,Airmass coord to get ETC mag
        etc=ascii.read("etc.txt")
        coord = [0.5,1.5] 
        metc=etc_efficiency(coord)
        print(metc)
        xlin,ylin = np.linspace(0,1,100),np.linspace(1,3,100)
        xis,yis,cis = [],[],[]
        for xi in xlin:
            for yi in ylin:
                coord = [xi,yi]
                ci = etc_efficiency(coord) 
                xis.append(xi)
                yis.append(yi)
                cis.append(ci)
        plot_efficiency(xis,yis,cis,etc=True)



