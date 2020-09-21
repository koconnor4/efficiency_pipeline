********
Examples
********

*******************
Using Your Own Data
*******************
In order to measure the detection efficiency for your image data, you must open the fits files, :py:class:`astropy.io.fits.open`. There is an LCO example provided for reference. In this example, we have a strongly lensed galaxy-galaxy system with image dirs (in the ``efficiency_pipeline/lco_pipe_example/sdssj2309-0039/`` folder)
``dia_out``, ``dia_trim``, ``source_im``, and a table of predicted peak lensed SNIa mags ``peakGLSN.csv`` . The directories contain the differences, single visit exposures, and the file which you'd like to do the analysis for. First we can read in these images (and that table) taking/printing the necessary values from source file header:

.. code-block:: python
	
	image,diff_image,ref_image,glsnID = lco_fakeSNpipeline.lco_pipe_ex()

Out:: 

	filename ~ cpt1m010-fa16-20200816-0229-e91_trim.fits (groupid SDSSJ2309-0039) has L1fwhm ~ 1.491417758955098 pixels, pixscale ~ 0.389 arcsec/pixel, and skybr 21.9799995 mag/arcsec^2; zp ~ 23.60843226612176
	
	glsn ~   
	Source ID    Magnification Lens Z Source Z Peak Apparent Magnitude
	-------------- ------------- ------ -------- -----------------------
	SDSSJ2309-0039             4   0.29      1.0             23.22508768

*******************************
Effective Point Spread Function
*******************************
The first step is to find the stars in the image... 

All sources can be found applying several photutil helpers, detection threshold is set using :py:class:`photutils.detection.detect_threshold`, sources above the threshold value return a segmentation image object using :py:class:`photutils.segmentation.detect_sources`, the image background is determined using :py:class:`photutils.background.Background2D`, and finally photometric and morphological properties of the background subtracted image are determined using :py:class:`photutils.segmentation.source_properties`, these all occur inside the lco_fakeSNpipeline.source_cat function, but first need to define the detection parameters

.. code-block:: python
	
    nsigma,kernel_size,npixels,deblend,contrast,targ_coord = 3,(3,3),int(np.round(L1fwhm/pixscale)),False,.001,None
    print('Source Catalog using nsigma ~ {} (detection threshold above img bkg), gaussian kernel sized ~ {} pix, npixels ~ {} (connected pixels needed to be considered source), deblend ~ {} w contrast {}'.format(nsigma,kernel_size,npixels,deblend,contrast))

Out:: 

	Source Catalog is a photutils source_properties using nsigma ~ 3 (detection threshold above img bkg), gaussian kernel sized ~ (3, 3) pix, npixels ~ 4 (connected pixels needed to be considered source), deblend ~ False w contrast 0.001

Now run lco_fakeSNpipeline.source_cat to find the objects in image meeting criteria for detection

.. code-block:: python

    source_catalog = lco_fakeSNpipeline.source_cat(image,nsigma=nsigma,kernel_size=kernel_size,npixels=npixels,deblend=deblend,contrast=contrast,targ_coord=None)
    cat,image,threshold,segm,targ_obj = source_catalog # unpacked to make a little clearer
    tbl=cat.to_table()
    print('Sources detected ~ ', len(tbl))
    print(tbl.colnames)
    print(tbl[:3])

Out::

	Sources detected ~ 535

	['id', 'xcentroid', 'ycentroid', 'sky_centroid', 'sky_centroid_icrs', 'source_sum', 'source_sum_err', 'background_sum', 'background_mean', 'background_at_centroid', 'bbox_xmin', 'bbox_xmax', 'bbox_ymin', 'bbox_ymax', 'min_value', 'max_value', 'minval_xpos', 'minval_ypos', 'maxval_xpos', 'maxval_ypos', 'area', 'equivalent_radius', 'perimeter', 'semimajor_axis_sigma', 'semiminor_axis_sigma', 'orientation', 'eccentricity', 'ellipticity', 'elongation', 'covar_sigx2', 'covar_sigxy', 'covar_sigy2', 'cxx', 'cxy', 'cyy', 'gini']

	  id      xcentroid      ...         cyy                gini       
             pix         ...       1 / pix2                        
             int64      float64       ...       float64            float64      
             ----- ------------------ ... ------------------- ------------------
    1  91.56201534245015 ...  0.5351457857358272 0.1629734606310599
    2   1949.18898760756 ...   0.769125864754405 0.1569260376108942
    3 1189.4932560102652 ... 0.47597065439640546 0.1332119485911448


In a general case the properties of these objects from the catalog could be restricted to provide point sources, however in the case of this LCO survey done in rp, the positions of stars measured in similar filter are readily available :py:class:`Gaia.query_object_async`, the results of the Gaia query within the image fov are taken     

.. code-block:: python

	results = lco_fakeSNpipeline.gaia_results(image)
	gaia,image = results # unpacked  

Out:: 

	INFO: Query finished. [astroquery.utils.tap.core]
	there are 50 stars available within fov from gaia results queried

Cutouts around these stars are extracted from the image after passing criteria including that they are not overlapping with other sources and have a flux which is below the LCO detector saturation/non-linearity but still gave a strong signal in Gaia. 

.. code-block:: python

	extracted_stars = lco_fakeSNpipeline.stars(results)
	good_stars,image = extracted_stars # unpacked

Out::
	
	46 stars, after removing intersections
	restricting extractions to stars w rp flux/error > 100 we have 13 to consider
	removed stars above saturation or non-linearity level ~ 149150.0, 125600.0 ADU; now have 13

Finally, the effective point spread function is constructed for these stars using :py:class:`photutils.psf.EPSFBuilder`. A 2D-Gaussian is fit to the resulting epsf using :py:class:`photutils.centroids.GaussianConst2D`

.. code-block:: python

	EPSF = lco_fakeSNpipeline.ePSF(extracted_stars,oversampling=2)
	epsf,fitted_stars = EPSF # unpacked
	epsf_gaussian = lco_fakeSNpipeline.gaussian2d(epsf)
    fit_gaussian,levels,xctr_vals,yctr_vals,image1,img_epsf,resid = epsf_gaussian # unpacked... levels list amplitude - sigma, ctr vals are gauss model sliced, image1 is array of values from gaussian fit in shape of epsf, img_epsf is epsf instance of it, resid is gauss - epsf

Out::

	PROGRESS: iteration 1 (of max 10) [? s/iter]
	PROGRESS: iteration 2 (of max 10) [0.4 s/iter]
	PROGRESS: iteration 3 (of max 10) [0.3 s/iter]
	PROGRESS: iteration 4 (of max 10) [0.3 s/iter]
	PROGRESS: iteration 5 (of max 10) [0.3 s/iter]
	PROGRESS: iteration 6 (of max 10) [0.3 s/iter]
	PROGRESS: iteration 7 (of max 10) [0.3 s/iter]
	PROGRESS: iteration 8 (of max 10) [0.3 s/iter]
	PROGRESS: iteration 9 (of max 10) [0.4 s/iter]
	PROGRESS: iteration 10 (of max 10) [0.4 s/iter]
	gaussian fit to epsf:
	('constant', 'amplitude', 'x_mean', 'y_mean', 'x_stddev', 'y_stddev', 'theta') [ 3.31007887e-04  3.84764457e-02  2.45832228e+01  2.52068898e+01
	  3.83262067e+00  4.11401422e+00 -2.33683005e+00]
	gaussian fwhm ~ 4.67822378372141 pixels (an avg of the fit sigma_x sigma_y w sigma_to_fwhm)
	13

Images showing the epsf against fitted gaussian, as well as the extracted stars that went into the epsf are made.


.. code-block:: python

	# make figures
    lco_figures.psf_and_gauss(epsf,epsf_gaussian,saveas=pickle_to+'_psf.pdf')
    lco_figures.used_stars(fitted_stars,saveas=pickle_to+'_stars.pdf')

.. image:: _static/cpt1m010-fa16-20200816-0229-e91_trim_psf.png
    :width: 600px
    :align: center
    :height: 600px
    :alt: alternate text

.. image:: _static/cpt1m010-fa16-20200816-0229-e91_trim_stars.png
    :width: 600px
    :align: center
    :height: 600px
    :alt: alternate text

*******************************
Detection Efficiency
*******************************
A pixel level detection efficiency measurement is performed by treating the epsf as a fake SN, it is scaled to different magnitudes, planted into the difference image, and recovered; the fraction of total plants recovered defining the efficiency.

The first step is to define magnitudes to measure efficiency for 

.. code-block:: python

    # measured psf is now going to be scaled to different magnitudes and planted in the difference image
    mags = np.arange(skybr-4.5,skybr+3,0.5) #zp ~ 23.5 # rough zp 

There are different planting locations applied, one straight-forward method is to plant in a grid across a full image, another is to zoom in and do a local measurement. There are helper functions lco_fakeSNpipeline.lattice and lco_fakeSNpipeline.target which find these coordinates. (In the case of LCO, the target of interest is the strong lens galaxy-galaxy system, so the cutout is made around this region)

.. code-block:: python
    
    locations = lattice(image) # the full img grid coords
    # target galaxy work, tuples cutting boxes around target (data,patch), how/if planting on cores might change detection efficiency
    # also returns targ_obj again account for updates using ref (in the cases where empty targ_obj ie not detected in source)
    target_boxes = target(image,targ_obj,ref=ref_image,diff=diff_image) 
    targ_obj,cuts,bkg_core,bkg_1,bkg_2 = target_boxes # unpacked
    cut_targ,cut_diff,cut_ref = cuts # unpack cuts around target source,diff,and ref

Here is an example of looping through the defined magnitudes to plant, recover, and determine efficiencies in the grid method

.. code-block:: python
	
	for mag in mags:
        # create plant image
        plantname = '{}_planted_lattice_mag{}.fits'.format(pickle_to,str(mag))
        planted = plant(diff_image,epsf,source_catalog,hdr=hdr,mag=mag,location=locations,zp=None,plantname=plantname)
        plant_im,pixels = planted # unpack
        # source properties ~ detecting objs in fake image
        fakesource_cat = source_cat(plant_im,nsigma=nsigma,kernel_size=kernel_size,npixels=npixels,deblend=deblend,contrast=contrast,targ_coord=None)
        fakecat,fakeimage,fakethreshold,fakesegm,faketarg_obj = fakesource_cat # unpacked to make a little clearer
        pickle.dump(fakecat.to_table(),open(pickle_to+'_fakesource_cat.pkl','wb'))
        # detection efficiency  
        tmp = detection_efficiency(planted,fakesource_cat)
        efficiency,magfakes,tbl,single_truth_tbl,repeat_truth_tbl,false_tbl = tmp
        efficiencies.append(efficiency)
        pickle.dump(tmp,open(pickle_to+'_detection_efficiency_mag{}.pkl'.format(str(mag)),'wb'))
        print(efficiency,magfakes)
        print('--------------------------------------------------------------')

Out ::

	529 SNe mag ~ 19.9799995 (epsf*=mu ~ 3300.1942005590113) planted in lattice across image; zp ~ 23.60843226612176
	519 planted SNe had single clean source detected, 5 planted SNe had multiple sources detected nearby, 234 false detections
	Detection efficiency (N_plants_detected/N_plants) ~ 0.9905482041587902 on mag ~ 19.9799995 SNe
	0.9905482041587902 19.9799995
	--------------------------------------------------------------
	529 SNe mag ~ 20.4799995 (epsf*=mu ~ 2082.281769053648) planted in lattice across image; zp ~ 23.60843226612176
	48 planted SNe had single clean source detected, 0 planted SNe had multiple sources detected nearby, 236 false detections
	Detection efficiency (N_plants_detected/N_plants) ~ 0.09073724007561437 on mag ~ 20.4799995 SNe
	0.09073724007561437 20.4799995
	--------------------------------------------------------------
	529 SNe mag ~ 20.9799995 (epsf*=mu ~ 1313.8309754616091) planted in lattice across image; zp ~ 23.60843226612176
	3 planted SNe had single clean source detected, 0 planted SNe had multiple sources detected nearby, 237 false detections
	Detection efficiency (N_plants_detected/N_plants) ~ 0.005671077504725898 on mag ~ 20.9799995 SNe
	0.005671077504725898 20.9799995
	--------------------------------------------------------------

A quick summary of the efficiencies measured can be provided and a parameter m50 at which there is a fifty percent chance of detection defined

.. code-block:: python
	
	print(filename)
    print('efficiencies: {}'.format(efficiencies))
    print('mags: {}'.format(mags))
    # use interp to get magnitude at which we have 50% detection efficiency 
    # need the values increasing along x for interp to work properly
    efficiencies,mags=list(efficiencies),list(mags)
    efficiencies.reverse()
    mags.reverse()
    m50 = np.interp(0.5,efficiencies,mags)
    print('m50 ~ {}'.format(m50))


Out :: 
	
	cpt1m010-fa16-20200816-0229-e91_trim.fits
	efficiencies: [0.994328922495274, 0.996219281663516, 0.994328922495274, 0.994328922495274, 0.996219281663516, 0.9905482041587902, 0.09073724007561437, 0.005671077504725898, 0.003780718336483932, 0.003780718336483932, 0.003780718336483932, 0.003780718336483932, 0.003780718336483932, 0.003780718336483932, 0.003780718336483932]
	mags: [17.4799995 17.9799995 18.4799995 18.9799995 19.4799995 19.9799995 20.4799995 20.9799995 21.4799995 21.9799995 22.4799995 22.9799995 23.4799995 23.9799995 24.4799995]
	m50 ~ 20.252583533613446

An exponential curve with two parameters alpha and m50 is fit to the data :math:`\epsilon(m) = \big(1+e^{\alpha(m-m50)}\big)^{-1}`  

Figures are made showing the plants and the efficiency

.. code-block:: python

	# make figures
    lco_figures.detection_efficiency(mags,efficiencies,m50,target_boxes,skybr,zp,glsn=glsnID,saveas=pickle_to+'_detection_efficiency.pdf')
    lco_figures.lattice_planted(mags,m50,pickle_to=pickle_to,saveas=pickle_to+'_plants.pdf')

.. image:: _static/cpt1m010-fa16-20200816-0229-e91_trim_detection_efficiency.png
    :width: 600px
    :align: center
    :height: 600px
    :alt: alternate text

.. image:: _static/cpt1m010-fa16-20200816-0229-e91_trim_plants.png
    :width: 600px
    :align: center
    :height: 600px
    :alt: alternate text