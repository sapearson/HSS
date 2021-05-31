
==================================

The Hough Stream Spotter (HSS)
==================================

This is the Hough Stream Spotter (HSS), described in Pearson et al. 2021 (arXiv: TBA)

The HSS is a  stream finding code which transforms individual positions of stars to search for linear structure in discrete data sets. The HSS requires only the two-dimensional plane of galactic longitude and latitude as input. The HSS was used to search for globular cluster streams in the PAndAS M31 stellar halo photometric data set in Pearson et al. 2021, in prep. 

This code is written and maintained by Sarah Pearson, with contributions from Susan E. Clark. Please feel free to get in touch with any questions, and please create pull requests to suggest improvements.

A example of how to use th code is available in the Jupyter Notebook “HSS_example.ipynb" with a place to put in your own discrete data region.

==================================

Requirements
==================================

To run the HSS you'll need the following packages:

matplotlib, numpy, scipy, and astropy. 

If you include masks in your dataset, you need the astropy regions module. 

Install through:
pip install git+https://github.com/astropy/regions


==================================

Basic Structure of Code
==================================

The code consists of five main functions in HSS.py

Function 1: HT_starpos

	-Hough Transforms each star position (x,y) in input region to (rho,theta)

Function 2: rho_theta_grid

	-Makes 2D histogram of (rho,theta) with drho, delta_t bins (peaks in this histogram correspond to overlapping sinusoids, i.e., straight lines in (x,y)-space)

	-Calculates bibomial probability of certain bins in the rho-theta 2D histogram having k or more overlapping sinusoids (stars) by chance

Function 3: rho_theta_peaks

	-Finds minima in probability distribution at certain rho and theta values

Function 4: HT_linear_feature

	-Translates these minima into a "stripe"/stream in (x,y) position space through an Inverse Hough Transform

Function 5: RT_plot

	-Plots the input data, the detected stripe and the probability distribution of the detection.

	-Outputs the (rho,theta)-values associated with the minima in (rho,theta) space, the probability of detection, and the number if stars in the recovered linear feature


The code also has a seperate HSS_maks_calc.py file, which handles the location and probability distribution if the dataset includes masks.

The code also has a separate HSS_coordinate_calc.py file, which handles the coordinate transformations to spherical skycoordinates if your data set is in degrees.


==================================

Instructions For Use
==================================

	1.	Ensure you are able to import the package:
		import HSS
		
	2.	Run the HSS for individual files, given a number of parameters. 
		HSS.RT_plot(pos, unit, kpc_conversion, delta_t, drho, outlier, pointsize, mask, filename[:-4], path_plot, verbose, rho_edge)
		
	3.	Files needed to run:
		        - filename = 'fakestream.txt'  #discrete input data (n,2), should be a circular region in either deg or unitless. 
			- masks_pos.txt, masks_sizes.txt #if you include masks, load in their position and sizes in Ra/dec [deg]
		
	4.	HSS input
			-See HSS_example.ipynb
		
	5.	Output
			⁃ Plot of 2D histogram of Hough Transform and the binomial probability distribution
			- Plot of input region along with possible detection
			     - Output filenames are either "Empty_", "Blob_, or "Stream_", depending on what the HSS found
			     
	6.	Run file: run_HSS.py
	
		⁃	See file for inputs and how to run the HSS
		⁃	to run from terminal: python run_HSS.py

==================================

Code Use
==================================

If use of the HSS results in a publication, please cite Pearson et al. 2021.

The HSS maintained by Sarah Pearson
