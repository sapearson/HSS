
==================================

The Hough Stream Spotter (HSS)
==================================

This is the Hough Stream Spotter (HSS), described in Pearson et al. 2021 (arXiv: TBA)

The HSS is a  stream finding code which transforms individual positions of stars to search for linear structure in discrete data sets. The HSS requires only the two-dimensional plane of galactic longitude and latitude as input. The HSS was used to search for globular cluster streams in the PAndAS M31 stellar halo photometric data set in Pearson et al. 2021, in prep. 

This code is written and maintained by Sarah Pearson, with contributions from Susan E. Clark. Please feel free to get in touch with any questions, and please create pull requests to suggest improvements.

A example of how to use th code is available in the iPython notebook “HSS_example.ipynb" with a place to put in your own code.

==================================

Basic structure of code
==================================

-Hough Transforms each star in input region

-Makes 2d histogram with drho, delta_t spacings of Hough Transform (peaks correspond to overlapping sinusoids, i.e., straight lines)

-Calculates bibomial probability of certain bins in the rho-theta 2D histogram having k or more stars by chance

-Finds minima in probability distribution at certain rho and theta values

-Translates these minima into a "stripe"/stream in (x,y) position space through an Inverse Hough Transform

-Plots the input data, the detected stripe and the probability distribution of the detection


==================================

Requirements
==================================

To run the HSS you'll need  matplotlib, numpy, scipy, and astropy. 

If you include masks in your dataset, you need the astropy regions module. 

Install through:
pip install git+https://github.com/astropy/regions

==================================

Instructions For Use
==================================

	1.	Ensure you are able to import the package:
		import HSS 
		
	2.	Run the HSS for individual files, given a number of parameters. 
		HSS.RT_plot(filename[:-4], pos, unit, kpc_conversion, delta_t, drho, outlier, pointsize, mask, path_plot, verbose, rho_edge)
		
	3.	Files needed to run
		⁃	filename = 'fakestream.txt'  #some data in which you’re search for a stream. Should be a circular region in either deg or unitless. 
		⁃	masks_pos.txt’, masks_sizes.txt' #if you include masks, load in their position and sizes in Ra/dec [deg]
		
	4.	Explanation of input
		⁃	filename #see files needed to run
		⁃	pos = filename[:,0], filename[:,1]
		⁃	unit = "deg" #unit of your input data "deg" for observations or "unitless" for e.g. simulations                                        
		⁃	kpc_conversion = np.pi * d_galaxy / 180. #from deg to kpc  #this is only relevant if unit = "deg", otherwise set kpc_conversion = 1       
		⁃	delta_t = 0.1 # this is the theta spacing deg                                                                                               
		⁃	drho = 0.4	# spacing in rho (search width) 
		⁃	outlier = 20.   # -log10Pr = outlier  how large of an outlier are you searching for (binomial probability)
		⁃	pointsize = 1 #for plots
		⁃	mask = False #are you including masks in your data set? 
				⁃can switch to True, and then read in your mask pos and sizes in Ra/Dec [deg] to masks_calc.py
				- for mask = True, you need to set unit = "deg" and your input needs to be in RA/dec [deg]
		⁃      	path_plot = '/Users/…’ # location to save plots   
		⁃	verbose= True #read out plots and updates in run
		⁃	rho_edge = False #only searches 60% central part of regions, but set to False if not using overlapping regions
		
	5.	Output
			⁃ 2D histogram of Hough Transform and the binomial probability distribution
			- Input region along with possible detection
			     - Output filenames are either "Empty_", "Blob_, or "Stream_", depending on what the HSS found
	6.	Run file: run_HSS.py
	
		⁃	See file for inputs and how to run the HSS
		⁃	to run from terminal: python run_HSS.py

==================================

Copyright
==================================

If use of the HSS results in a publication, please cite Pearson et al. 2021.

The HSS maintained by Sarah Pearson (sapearson)
