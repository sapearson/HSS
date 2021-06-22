#Sarah Pearson, Susan E. Clark, June 2021

#-----------------------------------------------------------------------------
#Imports
#-----------------------------------------------------------------------------
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.path as mpltPath
from matplotlib.colors import LogNorm
from matplotlib.cm import get_cmap

import astropy.coordinates as coord
import astropy.units as u

import scipy as sc
from scipy import signal


import math
from scipy.stats import norm #could remove
from scipy.stats import binom

import time

from regions import CircleSkyRegion, EllipseSkyRegion

#-----------------------------------------------------------------------------
#Separate routine imports by SP
#----------------------------------------------------------------------------- 
import HSS_coordinate_calc as cc

# If masks = True:
#regions module is needed
#pip install git+https://github.com/astropy/regions

#-----------------------------------------------------------------------------
# To run: 
# import HSS
# HSS.RT_plot(pos, unit, kpc_conversion, delta_t, drho, outlier, pointsize, mask, filename[:-4], path_plot, verbose, rho_edge)




def HT_starpos(pos_rht, delta_t):

    """
    Calculates the Hough Transform based on the position of some points in a region                                                                                                                                               
         rho = x * cos(theta) + y * sin(theta)                                                                                                            
         rho and theta will have the same spacing.  
         each point in x,y space will be represented as a full sinusoid in rho/theta space

    Inputs: 
       pos_rht: position coordinates of each star in region in any coordinates, the shape of pos should be (n,2). Data should be read in as circular region and centered on 0,0
       delta_t: spacing for the theta steps in deg.     

    Returns: 
       rho: array of rhos, dimensions are  (len(x),len(theta_temp)), where len(x) is the number of stars in data, and len(theta_temp) depends in theta spacing
            the unit of rho will be whatever unit your pos_rht input data were in
       theta: array of thetas, dimensions are  (len(x),len(theta_temp)), unit is in degrees
    """
    
    # Positions of stars (can be in any units)
    x = pos_rht[0] 
    y = pos_rht[1] 
    
    # Array of theta 

    theta_temp = np.arange(0, (180), delta_t)
    theta = np.ones([len(x),len(theta_temp)]) * theta_temp  

    # Generate an array of rho values with shape (nstars, nthetas)
    rho = x[:, np.newaxis] * np.cos(theta_temp*u.deg.to(u.rad)) + y[:, np.newaxis] * np.sin(theta_temp*u.deg.to(u.rad))

    return rho, theta 





def rho_theta_grid(pos, drho, delta_t, kpc_conversion, mask, unit,verbose, rho_edge):
    """
    Makes 2d histogram of Hough Transform, i.e. grid of rho and theta where we bin in the rho-direction in spacings of drho, but theta remains spaced as delta_t.
    Peaks in this space correspond to overlapping sinusoids in the Hough Transform space and are linear features.

    input:
        pos: x,y position coordinates of each star in regio in any coordinates, the shape os pos should be (n,2). Data should be read in as circular region 
        drho: how large your drho bin is (should have a size similar to the width of the feature you're searching for), if unit= "deg" drho should be in kpc
        delta_t: initial spacing of angles for hough transform
        Nsampels: how many scattered backgrounds to use for overdensity detection
        kpc_conversion: go from deg to kpc for your galaxy of interest, if unit = "unitless" set to 1
        mask: do we include masks in the data (True/False)
        unit: "deg" your input data is in ra/dec [deg] or "unitless": your data could be any unit (e.g. simulation), but mask calculation won't apply here (set to "no")

    return: 
        pos_rht: the postions you feed into HT_starpos for the Hough Transform. This is centered on (0,0) and if unit = "deg" it's converted to spherical skycoords then to kpc (used for final plots too)
        rho: output of rho's from HT_starpos
        theta: output of angles from HT_starpos in [deg]
        rho_grid: 2d histogram values of rho/theta grid with drho and delta_t spaced bins
        edgex: span the rho_grid in angle direction [theta] 
        edgey: span the rho_grid in rho direction 
        rho_grid_norm: 2d histogram values of rho/theta grid with drho and delta_t spaced bins but now normalized with respect to Nsamples of backgrounds (data - mean_i/std_i)
                       From each bin in 2d histogram we subtract the mean of 100 realizations for than bin and divide by the std from 100 realizations of that bin (see background)
        mu: the mean of the rho_grid_norm distribution (should be 0 per definition)
        std: the full width half maximum of the distributions of bins from rho_grid_norm
        skew_standardized: the deviation from a gaussian that the rho_grid_norm distribution of bins exhibits
        rho_grid_norm_standardized: rho_grid_norm divided by the std of rho_grid_norm (i.e. its full width half maximum), such that this new standardized distribution has a std/fwhm = 1
    """

    if unit == "deg": 
        alpha_0, delta_0, X_rht, Y_rht, data_ang_radius, data_ra_center, data_dec_center, c_data, a, b = cc.data_skycoords(pos, kpc_conversion)
        #the pos_rht below is the data region converted to kpc after spherical skycoord transformation

    if unit == "unitless": #centers input data on 0,0 for input to HT_starpos function
        X_rht, Y_rht, a, b = cc.data_unitless(pos)

    # Star positions, centered on (0,0)
    pos_rht = X_rht, Y_rht 

    # Hough Transform of data region after it's centered on (0,0) and, if unit == "deg", converted to spherical skycoords and to kpc
    rho, theta = HT_starpos(pos_rht, delta_t)

    #np.save('rho.npy', rho)                                                                                                                                       
    #np.save('theta.npy', theta)                                                                                                                                  
    #np.save('pos_rht.npy',pos_rht)

    # Number of bins of theta direction (x) and rho direction (y) in 2d histogram
    grid_x = np.int(((np.max(theta)+delta_t) - np.min(theta))/delta_t)
    grid_y = np.int(np.round((np.max(rho) - np.min(rho))/drho)) 

    
    # Create grid based on theta and rho values and spacing based on drho and delta_t from above
    rho_grid, edgex, edgey = np.histogram2d(np.array(theta).flatten(), np.array(rho).flatten(), bins = [grid_x, grid_y])


    if mask:
        #only import if masks uses, so people don't need regions module
        import HSS_masks_calc  as masks
        try:
            mask_pos = np.genfromtxt('masks_pos.txt') #
            mask_size = np.genfromtxt('masks_sizes.txt')# 
            
        except OSError:# IOError:#FileNotFoundError:
            print("WARNING: mask files not found")
            print('If no masks, set  mask = False as input')
        if verbose:
            print("checking if masks overlap")
        #go into da/A calc for masks.
        dA = masks.mask_calc(mask, pos, kpc_conversion , alpha_0, delta_0,X_rht, Y_rht,  data_ang_radius, \
                             data_ra_center, data_dec_center, c_data,verbose, mask_pos,mask_size, delta_t,drho, rho, grid_x,grid_y, edgex,edgey)

        r = np.max(np.round(rho)) #rho_max
        
    else:
        #numerical dA/A
        r = np.max(np.round(rho)) #rho_max
        rho_centers = np.linspace( -r +drho- drho/2, r - drho/2,grid_y)
        #due to nummerical issues depending on how many bins there are in y direciton i round this array
        # the drho2 can be larger than r if I don't round the rho_centers and swrt below could be negative
        rho_centers = np.round(rho_centers,3)
        dA = np.zeros([grid_y,grid_x])# (20,1800)
        for i in range(grid_y):#for each rho
            #print(i)
            for j in range(grid_x): #same for all theta, but need to store them.
               #print(i)
                rho1 =np.abs(rho_centers[i]) - drho/2
                
                rho2 =np.abs(rho_centers[i]) + drho/2
                
                dA[i,j] = r**2 * np.arccos(rho1/r) - rho1 * np.sqrt(r**2 - rho1**2)\
                        -(  r**2 * np.arccos(rho2/r) - rho2 * np.sqrt(r**2 - rho2**2))

        
   
    # Area of region
    A =(np.pi*r**2)
    Nstars = len(pos[0])
    p = dA/A 
    k= rho_grid.T

    # log of Equation 4 in paper 
    sf = binom.logsf(k-1, Nstars, p, loc=0) #same as (1-cdf) but can handle machine precision
    #I need k - 1 in order to add in the kth term when I do (1- cdf) which is the sf function above    
    #Pr_k = 1- cdf + pmf
    Pr_k = sf


    #saving transpose to have same shape as data
    return pos_rht, rho, theta, rho_grid, edgex, edgey, A, dA.T, Pr_k.T 




def rho_theta_peaks(pos, drho, delta_t, outlier,  kpc_conversion, mask,unit,verbose, rho_edge):
    """
    Function to find peaks/outliers in the normalized rho/theta 2D histogram above at a specified threshold (sigma)
    These peaks correspond to overlapping sinusoid in the HT space and are likely linear structures/streams
    If these structures are spherical objects, they will have a full sinusoid of peaks in this space (multiple angles)
    In addition to having a signal above sigma in the rho/theta grid, there are two additional detection criteria:
          1) skew_limit: you need to deviate from a gaussian (skew=0) and have a skew towards positive values (over densities)
          

    Input: 
        - pos:  x,y position coordinates of each star in regio in any coordinates, the shape os pos should be (n,2). Data should be read in as circular region
        - drho: the spacing in rho (how wide is the structure you're searching for)
        - delta_t (theta spacing)
        - sigma: what is your threshold of detection (i.e. how much of an outlier are you in the standardized histograms of bins), store rho's and theta's associated with above values
        - skew_limit: how much do you diviate from a gaussian where skew=0
        - Nsamples: how many fake uniform backgrounds 
        - kpc_conversion: from deg -> kpc at the distance of your galaxy, set to 1 if unit = "unitless"
        - mask: are there masks in your data or not True/False
        - unit: "deg" your input data is in ra/dec [deg] or "unitless": your data could be any unit (e.g. simulation), but mask calculation won't apply here (set to "no")  
 
    Returns:
        - pos_rht: input for HT_starpos and final plots. Centered on (0,0) and in kpc if ra/dec [deg] units input
        - rho_grid, edgex, edgey, rho_grid_norm: rho/theta grid values also the normalized one
        - theta_max: angles in rho/theta grid detections in theta that meet threshold criteria 
        - rho_max: rho's in rho/theta grid detections in theta that meet threshold criteria   
        - sigma_max: in this data region what is the maximum signal detected in the standardized rho/theta space 
        - no_peaks_unique: how many unique structures did we detect
        - mu: the mean of the rho_grid_norm distribution (should be 0 per definition)                                                                                                                                 
        - std: the full width half maximum of the distributions of bins from rho_grid_norm                                                                                                                             
        - skew_standardized: the deviation from a gaussian that the rho_grid_norm distribution of bins exhibits                                                                                                        
        - rho_grid_norm_standardized: rho_grid_norm divided by the std of rho_grid_norm (i.e. its full width half maximum), such that this new standardized distribution has a std/fwhm = 1   
        - theta_smear_arr: how many degrees do each detection peaks span in theta-direction (for each peak) 
        - theta_smear_limit: what is the minimum smear in theta-direction we need to claim a peak: theta_smear_min > drho/(2rho_max)
    """

    # Compute rho theta grid from data 
    pos_rht,rho, theta,rho_grid, edgex,edgey, A, dA, Pr_k= rho_theta_grid(pos, drho, delta_t,  kpc_conversion, mask,unit,verbose, rho_edge)


    # We are actually looking for minima, so I am taking the inverse of sigma
    outlier =outlier #see below for finding minima instead of maxima for Pr_k(X>k)
    theta_smear =( drho/(2*np.max(pos_rht)))*u.rad.to(u.deg)#theta_smear_limit
    rho_edge_crit = np.round(np.max(rho),1) #set as radius of region,so we find all peaks at all rho
   # if verbose:
    #    print('theta_smear_limit = ' + str(np.round(theta_smear,2))+ ' deg')
    if rho_edge: #only search for peaks 20% away from edge!
        rho_edge_lim = 0.2*2*np.max(rho) #20% of diameter in kpc
        rho_edge_crit = np.round(np.max(rho)-rho_edge_lim,1) #radius - this new crit
        if verbose:
            print('only searching for peaks where rho-peak < ' + str(rho_edge_crit) + ' kpc')


    # Find peaks and peak indices in rho/theta grid
    theta_diff = 10/delta_t #the angle separation between peaks in rho/theta grid in deg. If peaks are separated by > theta_diff at same rho value, we store
    theta_max = [] #empty arrays to store our peak locations
    rho_max = []
    Pr_min = []
    theta_range =[]  #store all thre theta smear values for each peak found above sigma threshold
    theta_smear_arr = [] #stores only the smear theta values for peaks that mmet theta_smear crit.
    for i in range(Pr_k[0,:].shape[0]): #go through each rho bin in 2d histogram rho/theta grid
        rho_ind = i #going throuch each rho to search for peaks with height sigma (some might not have any!)
        #print(rho_ind)
        # For each drho bin, we use this scipy function to search for peaks in the histogram above sigma threshold and separated by more than theta_diff in degrees 
        
        peaks_ind, peaks = sc.signal.find_peaks(x =-(Pr_k[:,rho_ind]), height = outlier,distance = theta_diff)
        peaks_val = peaks.get('peak_heights') # grab values from dictionary, these are the "sigmas" of each peak for each rho
        peaks_ind_corr = peaks_ind * delta_t# convert this to angle based on our theta spacing to store at which angle peaks occur
        
        # instead search for ALL peaks in theta (hence use delta_t as spacing, but need to take into account indexing so distance =1, all 1800 indices)
        peaks_ind_all, peaks_all = sc.signal.find_peaks(x = -(Pr_k[:,rho_ind]), height = outlier,distance = delta_t/delta_t)
        
        peaks_val_all = peaks_all.get('peak_heights')
        peaks_ind_corr_all = peaks_ind_all * delta_t

        c = ["purple","steelblue","lightsalmon","teal","skyblue","hotpink" ,"plum","red","orange", "maroon", "indigo", "blueviolet", "steelblue","lightcoral","teal", "skyblue","lightsalmon","hotpink" ,"plum","red", "orange", "maroon", "indigo", "blueviolet"]

        #Visualize logPr values
#        plt.plot(-(Pr_k[:,rho_ind]), linestyle = '-', linewidth = 2)#,color=c[rho_ind])
#        plt.plot(peaks_ind_all, peaks_val_all, linestyle='none', marker = '*' , c='black')
#        plt.plot(peaks_ind, peaks_val, linestyle='none', marker = '*' , c='purple')
#        plt.ylabel('-(logPr(X>=k)', fontsize=18)
#        plt.xlabel('theta/delta_t', fontsize=18)
                    
 
        if len(peaks_val) != 0: #if there is a peak, store the values of rho and theta and sigma at these peaks
            theta_temp = np.linspace(0,180,np.int(180/delta_t)) #changes depending on our delta_t spacing 
            rho_i = np.round(rho_ind * drho + np.min(edgey) + drho/2.,2) # our drho bins should be centered in the middle of drho bin
            Pr_min_i = peaks_val[0] #grab the value not the array
         
            #Below: how many degrees does the peak span in theta
            theta_range_i = np.max(peaks_ind_all*delta_t) - np.min(peaks_ind_all*delta_t)

            
            if len(peaks_ind_corr) > 1: # if there are multiple peaks for one rho with diff theta values, the length of the peaks_ind_corr arr is >1 
                for i in range(len(peaks_ind_corr)): #this will store peaks with same rho but thetas separated by more than 10 deg (theta_diff above)

                    theta_i = peaks_ind_corr[i]
                    Pr_min_i = peaks_val[i]

                    #for theta_smear calculation
                    peak = peaks_ind_corr[i]
                    peak_min = peak - theta_diff*delta_t #search for peaks around your significant peak (how smewared is feature in theta)                        
                    peak_max = peak + theta_diff*delta_t
                    smear_ind_min = np.where((peaks_ind_corr_all > peak_min))
                    smear_ind_max = np.where(peaks_ind_corr_all[smear_ind_min] < peak_max)

                    theta_range_i= np.max(peaks_ind_corr_all[smear_ind_min][smear_ind_max]) -\
                                           np.min(peaks_ind_corr_all[smear_ind_min][smear_ind_max])

                    # Only store any of these value if the peak is spread out over theta_smear in theta     
                    if theta_range_i> theta_smear and np.abs(rho_i) < rho_edge_crit:
                        theta_max.append(theta_i)   
                        rho_max.append(rho_i) #keep track of the rho for each of these thetas and store both theta and associated rho 
                        Pr_min.append(Pr_min_i)
                        theta_smear_arr.append(theta_range_i)

      
                        

            else: #if there's only one peak for each given rho
                #(sc.signal.find_peaks already finds peaks for different rhos) store these rhos and thetas  
                # Only store any of these value if the peak is spread out over theta_smear in theta  
                
                if theta_range_i> theta_smear and np.abs(rho_i) < rho_edge_crit: #add rho max criterion here. 
                    rho_max.append(rho_i)       #we store the values from the first if statement (not zero peaks)
                    theta_i = peaks_ind_corr[0]
                    theta_max.append(theta_i)
                    Pr_min.append(Pr_min_i)
                    theta_smear_arr.append(theta_range_i) #output only the feature with sufficient theta smear range


    # Sort the maxima, such that the most significant peak (peaks_val/sigma) is the first one
    # Sort such that we only plot the 10 most significant peaks in plotting function (ie. largest peaks_val)                                                      
    ind = np.array(Pr_min).argsort()[::-1] # indices of peaks values that are now sorted by highest first                                                     

    rho_max = np.array(rho_max)[ind]
    theta_max = np.array(theta_max)[ind]
    Pr_min = np.array(Pr_min)[ind]
    theta_smear_arr = np.array(theta_smear_arr)[ind]
    no_peaks_unique = len(rho_max) #How many peaks did we find

    if no_peaks_unique == 0:
        if verbose:
            print("No streams detected")
    if no_peaks_unique >= 10:
       if verbose:
           print('Likely spherical object or high log10Pr-threshold')
                  
    if no_peaks_unique < 10 and no_peaks_unique > 0:
       if verbose:
           print('-------------------------------')
           print('')
           print('Stream detected at an angle of:')
           print('theta_peak = '+ str(np.round(theta_max,2)) + '[deg]')
           print('')
           print('Stream detected at a minimum Euclidian distance of:')
           print('rho_peak = '+ str(rho_max))
#           print('theta_smear = ' + str(np.round(theta_smear_arr,2)))
           print('')
           print('Stream detected at a significance of:')
           print('-(logPr(X>=k)) = ' + str(np.round(Pr_min,2)))
           print('')
           print('-------------------------------')
           print('')
            
    return pos_rht,rho_grid, edgex, edgey, theta_max, rho_max, Pr_min, no_peaks_unique, A, dA, Pr_k,theta_smear_arr, theta_smear





        

def HT_linear_feature(pos_rht,rho_max,theta_max,drho,delta_t):
    """
    This function draws polygons based on the inverse hough transform (physical space of data)
        
        rho = x * cos(theta) + y * sin(theta)  
        x = (rho -  y * sin(theta)) / cos(theta) 
    
    Thus for the max theta and rho, we use the inverse hough transform to recover the position and angle of the detected linear structure
    The polygon spans the extent of the data region and thus we do not take into account the "length of the structure"

    Inputs: 
        - pos_rht: data centered at (0,0) and if unit = "deg" converted to spherical skyycords and kpc unit centered on 0
        - rho_max: the value of rho in the 2D rho-theta grid corresponding to the sigma peaks
        - theta_max: the value of theta in the 2D rho-theta grid corresponding to the igma peaks
        - drho: rho bin size
        - delta_t: spacing in theta array for initial hough transform

    Returns: 
        - y_line_idx, x_line_idx: the recovered liniar structures based on reveres hough transform for peaks in rho/theta 2D histogram
        - poly_coords_idx: the coordinates that span the structure based on the searched drho "tube" size
        - ext_min, ext_max: which specifies the limit of the extent of the data/region size - used for spanning plots

    """

    # Define the space in which to draw linear features based on the inputted centered data positions:
    ext_min = np.min(pos_rht[1]) # we want the line to only extend through the data region
    ext_max = np.max(pos_rht[1]) 

    y_line = np.zeros([len(rho_max), 100])
    x_line = np.zeros([len(rho_max), 100]) # we need a line for each detected structure (len(rho_max)) 
    poly_coords_idx = []
    x_line_idx = []                                                                                                                                                 
    y_line_idx = []
    for i in range(len(rho_max)): #for each individual peak in 2D histogram
        # inverse hough transfor divides by 0 if theta = 90 deg, problem for "drawing" structure
        # if the structure is close to 90 deg we span a horizontal polygon instead of doing the invers HT:
        dthe = theta_max[i] - 90 #how close is our theta_max to 90 deg. 
        if np.abs(dthe) < 2: #draw a horizontal line instead of dividing by cos(90) = 0
            print('stream '+str(i+1)+': theta close to 90 deg (theta = ' + str(theta_max[i])+' deg), drew polygon as horizontal tube')
            x_line[i,:] = np.linspace(ext_min,ext_max,100)
            y_line[i,:] = np.ones(100)*rho_max[i] # at the rho_max location
        else: #do the inverse hough transform 
            y_line[i,:] = np.linspace(ext_min,ext_max,100)
            x_line[i,:] = (((rho_max[i] )- (y_line[i,:]*np.sin((theta_max[i] + delta_t/2) * u.deg.to(u.rad)))) / np.cos((theta_max[i] + delta_t/2)*u.deg.to(u.rad)))
       
        #Select only stars within the x-extent of the data
        x_l = x_line[i,:]
        y_l = y_line[i,:]

        #below can be changed if we don't have a circular data region
        idx_circ = np.where(x_l**2 + y_l**2 < ((ext_max-ext_min)/2)**2) #making sure the tube only gets drawn within region area of data

        x_line_idx.append(x_l[idx_circ]) #save these for plotting routine
        y_line_idx.append(y_l[idx_circ])

        try:
            x1 = x_l[idx_circ][0] #grab first and last x for each of the lines  
        except IndexError:
            print('')                                                                                                                                               
            print("Warning: Check shape of data array or units of data (plot pos as check)")
            print('')
        x2 = x_l[idx_circ][-1]
        y1= y_l[idx_circ][0]
        y2 = y_l[idx_circ][-1]

        
        # We now have the lines for each different linear structure to plot on the scatter plot of the data
        # Span polygon to plot on detected linear structure in data
        
        if np.abs(dthe) < 2: # if our structure is close to 90 deg, we instead use a horizontal line
            y1_mdt = y1 - drho/2 #centered on line so only half of tube size
            y2_mdt = y2 - drho/2
            y1_pdt = y1 + drho/2
            y2_pdt = y2 + drho/2

            x1_mdt = x1# - dx
            x2_mdt = x2 #- dx
            x1_pdt = x1 #+ dx
            x2_pdt = x2 #+ dx

            poly_coords = np.array([[x1_mdt, y1_mdt], [x1_pdt,y1_pdt], [x2_pdt, y2_pdt], [x2_mdt, y2_mdt]])

            #store polycoords for each of the detected structure                                                                                                        
            poly_coords_idx.append(poly_coords)
            
        else: # if structure is not close to 90 deg
            a = (y1-y2)/(x1-x2) #slope of structure             
            # Slope of perpendicular line is -1/a (need that to span polygon)                                                                                                                     
            dx = np.sqrt((drho/2)**2/((1/a**2)+1)) # shift we need to make to draw point and perpendicular width of structure (drho)
            dy = -1/a*dx #shift in y-direction 

            #Coordinates to span polygon with the width of drho
            y1_mdt = y1 - dy #centered on line so only half of tube size
            y2_mdt = y2 - dy
            y1_pdt = y1 + dy
            y2_pdt = y2 + dy

            x1_mdt = x1 - dx
            x2_mdt = x2 - dx
            x1_pdt = x1 + dx
            x2_pdt = x2 + dx

            poly_coords = np.array([[x1_mdt, y1_mdt], [x1_pdt,y1_pdt], [x2_pdt, y2_pdt], [x2_mdt, y2_mdt]])

            #store polycoords for each of the detected structure
            poly_coords_idx.append(poly_coords) 

        if i > 10:
            break #don't need polygons for the many structures apparent if it's a spherical object
       
        ### Area of poly srturctures, not using this right now but might want to later
        A_poly = drho/2 * np.sqrt((np.max(x_l[idx_circ]) - np.min(x_l[idx_circ]))**2 + (np.max(y_l[idx_circ]) - np.min(y_l[idx_circ]))**2)#area of tube

        Nstars_tot = len(pos_rht[0]) # no stars in field.
        A_reg = 2*np.pi*np.max(pos_rht[0]) #kpc^2 area of reguion
        no_dens = Nstars_tot/A_reg #stars/kpc^2 total region
        no_dens_bg_tube = np.round(np.sqrt(no_dens * A_poly),1) #sqrt number of stars in average tube for background
        

    return y_line_idx, x_line_idx, poly_coords_idx, ext_min, ext_max


def RT_plot(pos,unit,kpc_conversion,delta_t,drho,outlier, pointsize,mask, filename,path_plot,verbose, rho_edge):
    """
    Main function that reads in all above functions and plots the star position in each region as well as the recovered structures
    Also outputs 2d histogram and angles/rhos of peaks. Can chose to store various values in txt files too.
    If there are recovered structures, these are drawn as polygons, and if there are less than 10 we also save the 2D histogram with corresponding peaks

    Inputs: 
         - pos:  x,y position coordinates of each star in regio in any coordinates, the shape os pos should be (n,2). Data should be read in as circular region    
         - unit: "deg" your input data is in ra/dec [deg] or "unitless": your data could be any unit (e.g. simulation), but mask calculation won't apply here (set to "no")  
         - kpc_conversion: from deg->kpc at your object of interest, if unit = "unitless" set to 1
         - drho: bin size in rho direction for 2d rho/theta histogram
         - delta_t: theta spacing for calculating n: rho_n = xcos(theta_n) + ysin(theta_n)
         - sigma: the significance of structure with respect to background in order to flag detection
         - skew_limit: how much do you diviate from a gaussian where skew=0                                                                                    
         - pointsize: how large points to use for scatter plots of data
         - Nsamples: how many backgrounds to used for overdensity detection
         - mask: are there masks in your data ("yes"/"no")
         - filename: plotname to be saved (could be def. by region used)
         - path_plot: folder with data for regions, positions of stars

        
        

    Calculates:
    Hough transform rho = xcos(theta) + ysin(theta) for each star in the list pos
    Creates 2d grid of rho theta values with theta bin sizes defined by dthet along with no-sample fake uniform background regions with same extent number of stars as data region
    Find the maximum pixel in the 2d grid which corresponds to where most sinusoids (rhos) overlap
    Computes straight lines from these max. pixels to illustrate the linear fatures

    Returns: 
        - Saves: scatter plot of stars in region  and overplotted linear features found by Hough Transform, diff names if blob, empty or stream
        - Saves: rho/theta 2D histogram grid with overdensities IF there are detections (less than 10)
        - prints where plots are stored 
       
    """
      
    #Call all functions above to make final plots of detections   

    pos_rht,rho_grid, edgex, edgey, theta_max, rho_max, Pr_min, no_peaks_unique, A, dA, Pr_k,theta_smear_arr, theta_smear_limit =\
                                                    rho_theta_peaks(pos,drho,delta_t, outlier,kpc_conversion,mask,unit,verbose, rho_edge) 
    y_line, x_line, poly_coords, ext_min, ext_max = HT_linear_feature(pos_rht, rho_max,theta_max,drho, delta_t) 

    
#    np.save('rho_grid.npy', rho_grid)  
    mpl.rcParams.update({'font.size': 24})
    label_size = 24
    mpl.rcParams['xtick.labelsize'] = 24#label_size 
    mpl.rcParams['ytick.labelsize'] = 24#label_size 
    cm =plt.cm.get_cmap('Greys')##Blues')
    #colors used to draw recovered linear structures (only need 10)
    c = ["purple","steelblue","teal","hotpink" ,"plum","skyblue","maroon","orange", "maroon", "indigo", "blueviolet", "steelblue","lightcoral","teal",\
         "skyblue","lightsalmon","hotpink" ,"plum","red", "orange", "indigo", "blueviolet"]

    if unit == "deg":
        axislabel = "[kpc]"
    if unit == "unitless": #could change the axes to whatever unit you're using
        axislabel = "[input unit]"


    
    fig,axes = plt.subplots(1,2 ,figsize=(12,6))#, sharex=True, sharey = True)                                                \
                                                                                                                               
    axes[0].scatter(pos_rht[0],pos_rht[1], s = pointsize, color ='grey')
    axes[0].set_xlabel(axislabel)
    axes[0].set_ylabel(axislabel)

    #Scatter plot but now adding linear structures where Hough Transform found maxima on rho/theta grid                       \
                                                                                                                               
    axes[1].scatter(pos_rht[0],pos_rht[1], s = pointsize, color='grey')#, label='stream stars: '+ str(len(feature_ra)))



 
    #to store stars inside for connecting features
    feature_ra_notpeak =[]
    feature_dec_notpeak =[]
    blob_ra_notpeak =[]
    blob_dec_notpeak =[]
    Pr_notpeak = []
    stars_inside =[]
    for i in range(len(x_line)): 
    #Draw polygon imported from HT_linear_feature function
        path_poly = mpltPath.Path(poly_coords[i])
        # How many stars in data are inside the polygon
        inside = path_poly.contains_points(np.transpose([pos_rht[0],pos_rht[1]]))
        poly = mpl.patches.Polygon(poly_coords[i],alpha=0.3, color=c[i])
        stars_inside_poly = len(pos_rht[0][inside]) #str(stars_inside_poly)+' stars,
        stars_inside.append(stars_inside_poly)
        # Option to store all features... 
        # also store the inside stars in ra/dec so we can see connecting features of most significant peak
        # To use the np.save features below, you need to create a "features" and "inside" directory in your plot_path 
        if len(x_line)<10 and i ==0:#for now only the first value = the most significant 
            feature_ra = pos[0][inside]
            feature_dec = pos[1][inside]
           # np.save(path_plot+'features/stripe_ra_' +str(filename)+'.npy',feature_ra)
           # np.save(path_plot+'features/stripe_dec_' +str(filename)+'.npy',feature_dec)
           # np.save(path_plot+'features/stripe_Pr_' +str(filename)+'.npy',np.round(Pr_min[0],2))
           # np.save(path_plot+'inside/stars_inside_' +str(filename)+'.npy',inside)#for now I am only storing the most prominent
            
        if len(x_line) < 10 and i > 0: #only store if less than 10 features total!
            feature_ra = pos[0][inside]
            feature_dec = pos[1][inside]
            feature_ra_notpeak.append(feature_ra)
            feature_dec_notpeak.append(feature_dec)
            Pr_notpeak.append(np.round(Pr_min[i],2))


        #### Also store all "blobs"
        if len(x_line)>10 and i ==0:#for now only the first value = the most significant                                                                                                            
            blob_ra = pos[0][inside]#.append(pos[0][inside])                                                                                                                                    
            blob_dec = pos[1][inside]##].append(pos[1][inside])                                                                                                                                  
           # np.save(path_plot+'features/blob_ra_' +str(filename)+'.npy',blob_ra)
           # np.save(path_plot+'features/blob_dec_' +str(filename)+'.npy',blob_dec)
           # np.save(path_plot+'features/blob_Pr_' +str(filename)+'.npy',np.round(Pr_min[0],2))

        if len(x_line) > 10 and i > 0: #only store if less than 10 features total!                                                                                                                  
            blob_ra = pos[0][inside]
            blob_dec = pos[1][inside]
            blob_ra_notpeak.append(blob_ra)
            blob_dec_notpeak.append(blob_dec)
            Pr_notpeak.append(np.round(Pr_min[i],2))

        axes[1].add_patch(poly)
        axes[1].plot(x_line[i], y_line[i], linestyle='--',  \
                             label= 'Stream stars: ' + str(stars_inside[i]), color=c[i])

        axes[1].legend(fontsize=16)



            
        if i > 10: #likely spherical object                                                                                    
            break #I only want to draw the first 10 polygrams...                                                               
            #note that the 10 most significant streams will be the ones plotted (due to argsort in the peak finding algorithm) 


#    store the stars inside the less significant peaks and their Pr values
#    if len(feature_ra_notpeak)>0:# i.e. there are not only one structures
        #np.save(path_plot+'features/stripe_ra_notpeak_' +str(filename)+'.npy',feature_ra_notpeak)
        #np.save(path_plot+'features/stripe_dec_notpeak_' +str(filename)+'.npy',feature_dec_notpeak)
        #np.save(path_plot+'features/stripe_Pr_notpeak_' +str(filename)+'.npy',Pr_notpeak)
        
#   if len(blob_ra_notpeak)>0:# i.e. there are not only one structures                                                                                                       
#      # np.save(path_plot+'features/blob_ra_notpeak_' +str(filename)+'.npy',blob_ra_notpeak)
       # np.save(path_plot+'features/blob_dec_notpeak_' +str(filename)+'.npy',blob_dec_notpeak)
       # np.save(path_plot+'features/blob_Pr_notpeak_' +str(filename)+'.npy',Pr_notpeak)
        
        
    axes[1].set_xlabel(axislabel)
    axes[1].set_ylabel(axislabel)
  
    fig.tight_layout()

    
    # different output figure names are used when running on multiple region to quickly sort detections from non-detections
    if no_peaks_unique == 0: #save plot but named "Empty" if no structure found
        fig.savefig(path_plot + 'Empty_' + filename + '_drho_' + str(drho)+'_dts_' + str(delta_t)+'_outlier_'+str(outlier)+  '.png')

    if no_peaks_unique > 10: #save under diff name if likely a blob 
        fig.savefig(path_plot + 'Blob_' + filename + '_drho_' + str(drho)+'_dts_' + str(delta_t)+ '_outlier_'+str(outlier)+ '.png')

        
    if no_peaks_unique < 10 and no_peaks_unique > 0: #these are the ones that will be interesting and have less than 10 structures
        #axes[1].legend(loc='upper left', fontsize=16) #save as "stream"
        fig.savefig(path_plot + 'Stream_' + filename + '_drho_' + str(drho)+ '_outlier_'+str(outlier)+ '.png')



        
    # plot 2d histograms of data/backgrounds     

    fig2 = plt.figure(figsize=(14,11))
    ax1 = fig2.add_subplot(211)
  
    Nstars = len(pos[0])
  
    aspect_hist = (180./(ext_max-ext_min)) * 0.4 # scales based on rho range. kind of arbitrariryly chosen 
    im = ax1.imshow(rho_grid.T, origin='lower', cmap=cm, extent = (np.min(edgex), np.max(edgex), np.min(edgey), np.max(edgey)), aspect=aspect_hist, vmin=0,vmax=np.max(rho_grid))
             

    for i in range(len(x_line)):
        #Indicate where peaks are in histogream, uses same colors as the linear features on scatter plot                                                                                                  
        ax1.axvline(theta_max[i] , linestyle='--', label =  r'$\theta = $' + str(np.round(theta_max[i],2)) + r'$, \rho =$ ' + str(np.round(rho_max[i],2))+ r', ${\rm logPr}(X \geq k)$ = -' + str(np.round((Pr_min[i]),2)), color=c[i])# , color=c[i])#+ r', $\Delta$$\theta$$_{smear}$ = ' +str(np.round(theta_smear_arr[i],2)) +' deg', color=c[i])
        ax1.axhline(rho_max[i] , linestyle='--', color=c[i])
        if i > 10:
            break

    #ax1.set_title( filename + ', histogram of sinusoids') 
    ax1.set_xlabel(r'$\theta$ [deg]')
    ax1.set_ylabel(r'$\rho$ ' + axislabel)
    ax1.set_ylim([ext_min,ext_max])
    ax1.set_xlim([0,np.int(np.max(edgex))])

      
    ax3 = fig2.add_subplot(212)    
  
    if len(x_line)>0:
        vmin_pr = np.round((-Pr_min[0]),2)
    inf_test = np.min(Pr_k.T) 
    if inf_test == float('-inf'):
        vmin_pr = -800
        print('-inf in log10Pr grid')
    else:
        vmin_pr = np.min(Pr_k.T) 

    im3 = ax3.imshow((Pr_k.T), origin='lower', cmap=cm, extent = (np.min(edgex), np.max(edgex),  np.min(edgey), np.max(edgey)), aspect=aspect_hist,vmin=vmin_pr, vmax=0)#np.min(Pr_k.T), vmax=0)#,vmax=np.max(bin_data))#, \
    for i in range(len(x_line)):
        #Indicate where peaks are in histogream, uses same colors as the linear features on scatter plot                                                                                
        ax3.axvline(theta_max[i] , linestyle='--', label =  r'$\theta = $' + str(np.round(theta_max[i],2)) + r'$, \rho =$ ' + str(np.round(rho_max[i],2)) + r', ${\rm logPr}(X \geq k)$ = -' + str(np.round((Pr_min[i]),2)) , color=c[i])
        
        ax3.axhline(rho_max[i] , linestyle='--', color=c[i])

        if i > 10:
            break

    ax3.set_xlabel(r'$\theta$ [deg]')
    ax3.set_ylabel(r'$\rho$ ' + '[kpc]')
    ax3.set_ylim([ext_min,ext_max])
    ax3.set_xlim([0,np.int(np.max(edgex))])

    fig2.colorbar( im,ax=ax1 ,shrink=1, label=r'number of stars, $k$')
    fig2.colorbar( im3,ax=ax3, shrink=1,label=r'${\rm logPr}(X \geq k)$')

    fig2.tight_layout()
    

    if no_peaks_unique == 0: #save plot but named "Empty" if no structure found
        fig2.savefig(path_plot + 'Empty_' + filename + '_drho_' + str(drho)+'_dts_' + str(delta_t)+ '_outlier_'+str(outlier)+ '_2Dhist.png')

    if no_peaks_unique > 10: #save plot but named "blob" if likely spherical objects                                                                         
        ax1.legend(fontsize=12) #save as "stream"                                                                                                                       
        ax3.legend(fontsize=12) 
        fig2.savefig(path_plot + 'Blob_' + filename + '_drho_' + str(drho)+'_dts_' + str(delta_t)+ '_outlier_'+str(outlier)+ '_2Dhist.png')

    if no_peaks_unique < 10 and no_peaks_unique > 0: #these are the ones that will be interesting and have less than 10 structures                      
        ax1.legend(fontsize=16) #save as "stream"
        ax3.legend(fontsize=16)
        fig2.savefig(path_plot + 'Stream_' + filename + '_drho_' + str(drho)+ '_dts_' + str(delta_t)+'_outlier_'+str(outlier)+  '_2Dhist.png')


     
    Pr_min = np.round(np.min(Pr_k.T),4)


    return print("plots saved in " + path_plot)

 # can return any values and save them in txt files as
       # with open("test.txt", "a") as f:
        #    f.write(str(value)+ '\n')
