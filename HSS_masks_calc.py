import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import astropy.coordinates as coord
import astropy.units as u
from astropy.wcs import WCS
from astropy.coordinates import Angle, SkyCoord

from regions import CircleSkyRegion, EllipseSkyRegion
import time
#-----------------------------------------------------------------------------                                                                                          
#Read in spherical transform
#-----------------------------------------------------------------------------   
import HSS_coordinate_calc as cc
import HSS as HSS



def mask_calc(mask, pos, kpc_conversion , alpha_0, delta_0,X_sphere_rht, Y_sphere_rht,  data_ang_radius, data_ra_center, data_dec_center, c_data,verbose, mask_pos,mask_size,theta_s,drho, rho, grid_x,grid_y,edgex,edgey):    
    """
    Function that checks if your data region intersects any of the masks you have read in as txt files above. 
    Your data needs to be in ra/dec [deg] for this function to work

    inputs: 
        - mask: is either True/False
        - pos: x,y position coordinates of each star in regio in any coordinates, the shape os pos should be (n,2). Data should be read in as circular region and in ra/dec [deg]   
        - kpc_conversion: conversion from deg to kpc for your galaxy/data of interest
        - Nsamples: how many backgrounds to used for overdensity detection           
        - alpha_0: RA center of data region in radians                                                                                                                                 
        - delta_0: Dec center of data region in radius                                                                                                                                   
        - X_sphere_rht: RA in spherical skycoords centered on 0 in kpc                                                                                                                  
        - Y_sphere_rht: dec in spherical skyycords  centered on 0 in kpc                                                                                                               
        - data_ang_radius: extent of data in spherical skycoords in deg                                                                                                                  
        - data_ra_center: RA center of data region in degrees (maybe redundant when we have alpha_0)                                                                                   
        - data_dec_center: Dec center of data region in degrees (maybe redundant when we have alpha_0)                                                                                  
        -  c_data: all data from the region as an astropy skycoord objects  
    
    return:
        -  X_sphere_scattered_rht, Y_sphere_scattered_rht:  coordinates spanning same extent as data and with the same number of points as data             
                                                            centered on 0,0 so they can be used for input to HT_starpos                                                             
                                                            this is in [kpc] converted from spherical skycoords.                                                                    
         
    """
    
    
    data_region = CircleSkyRegion(center = coord.SkyCoord(data_ra_center, data_dec_center, unit='deg'), radius=data_ang_radius.value * u.deg)                              
    mask_regions= []
    
    for i in range(len(mask_pos)): # not taking the MW value positions of all masks [x,y]
        center_sky_i = SkyCoord(mask_pos[i,0], mask_pos[i,1], unit='deg', frame='icrs')
        circle_sky_i = CircleSkyRegion(center = center_sky_i, radius = mask_size[i,0] * u.deg)
        mask_regions.append(circle_sky_i)

    #  need to work in pixeled imaged for regions to work                                                                                                   
    wcs = WCS(naxis=2)
    pix_scale = 60 * u.arcsec / u.pixel #SP: make small doesn't really matter                                                                              
    wcs.wcs.crval = [10.68470833, 41.26875]  # M31                                                                                                          
    wcs.wcs.cdelt = [pix_scale.to_value(u.deg / u.pixel) for i in range(2)]
    wcs.wcs.crpix = [0, 0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    #Does the mask overlap with data_ region of interest                                                                                                   
    mask_idx = np.ones(len(mask_regions), dtype=bool) #first make array of TRUE for all read in masks
   
    for i in range(len(mask_regions)): #loop over how many masks there are
         intersect = mask_regions[i].to_pixel(wcs).intersection(data_region.to_pixel(wcs))
         mask = intersect.to_mask()
         do_they_interset = mask.data.sum() > 0 #do the masks and any of data_region defined as a circel skyregion overlap?                                 
         if do_they_interset == False: #if they to not overlap
             mask_idx[i] = False #set the mask_idx to false....                                                                                              

    #Now we should be left with only indices that are inside the mask                                                                                       
    mask_regions_overlap = []
    for i in range(len(mask_regions)):
         if mask_idx[i] == True: #storing all the masks that overlapped                                                                                  
              mask_regions_overlap.append(mask_regions[i])

    #now only store the RA/dec of the masks that intersected                                                                            
    mask_intersect_ra = mask_pos[mask_idx,0]
    mask_intersect_dec = mask_pos[mask_idx,1]

    # mask_regions_overlap is the improtant values 
    if verbose:
        print('there are '+str(len(mask_regions_overlap)) + ' masks intersecting the region')
        

    if len(mask_regions_overlap)==0: #do the analytic step, the masks don't intersect here
        if verbose:
            print('Analytic dA calc')
            print('')
        r = np.max(np.round(rho)) #rho_max
        rho_centers = np.linspace( -r + drho - drho/2,r - drho/2,grid_y)
        #I round the below to avoid rho2 > r -> netive square root
        rho_centers = np.round(rho_centers,3)
        dA = np.zeros([grid_y,grid_x])# (20,1800)                   
        for i in range(grid_y):#for each rho
            for j in range(grid_x): #sam
                rho1 = np.abs((rho_centers[i])) - drho/2
                rho2 = np.abs(rho_centers[i]) + drho/2
                dA[i,j] = r**2 * np.arccos(rho1/r) - rho1 * np.sqrt(r**2 - rho1**2) \
                    -(  r**2 * np.arccos(rho2/r) - rho2 * np.sqrt(r**2 - rho2**2))
    else: #numerically calculate dA
        if verbose:
            print('numerical dA calc')
            print('')
    # we now know if masks overlap or not. 
    
        a_list = np.abs(c_data.ra.wrap_at(180*u.deg).deg - data_ra_center)
        a = np.max(a_list)#+0.01
        #adding small number instead of finding bug, my mask iterastion won't work if I'm in spherical coords                           

        b_list = np.abs(c_data.dec.deg - data_dec_center)
        b = np.max(b_list)#+0.01

        Nstars = len(c_data)
        Nstars_times10 = Nstars*10

        #We now just make one background with 10 times as many stars (Nstars
        #for j in range(Nsamples):
        r_scramble = np.random.uniform(low=0, high=(a)**2, size = (Nstars_times10)) #radius of scramble data, high=radius data **2                      
        angle_scramble = np.random.uniform(low=0, high=2.*np.pi, size = (Nstars_times10 ))
        x_times10 = np.sqrt(r_scramble) * np.cos(angle_scramble)  
        y_times10 = np.sqrt(r_scramble) * np.sin(angle_scramble) * b/a


        c_times10 = coord.SkyCoord(ra = x_times10 + data_ra_center, dec = y_times10 + data_dec_center, unit='deg') #all data in regions  


        #what stars are contained within mask
        mask_contains = []
        for i in range(len(mask_regions_overlap)):
            mask_contains_i = mask_regions_overlap[i].contains(skycoord = c_times10, wcs=wcs)
            mask_contains.append(mask_contains_i)

            #Also store all stars that ARE NOT within the masks                                                                                                 
            idx_removed_masks = np.ones([len(c_times10.ra)], dtype=bool)
            #   for j in range(Nsamples): #for each scattered background                                                                                        

        for i in range(len(mask_regions_overlap)): #for all different intersecting masks                                                        
            for k in range(len(c_times10.ra)): #for all stars in one of the scattered regions                                                        
                if mask_contains[i][k] == True: #if that mask                                                                                    
                    idx_removed_masks[k] = False

                    #store how many missing stars we have when we remove masks from fake region                                                                         
                    #This should be for eac scattered region (10)                                                                                                       
                    #    extra_stars = np.zeros([Nsamples])
                    #   for j in range(Nsamples):
        extra_stars = len(c_times10.ra) - len(c_times10.ra[idx_removed_masks]) #no of stars missing in scat region                          
#        print(extra_stars)

         #make iteration so I fill the empty masks again with stars...                                                                                       
        iterations = 50 #can be small this algorithm converges quikcly                                                                                      

        times10_stars_allsamples = []

        #    for j in range(Nsamples): #different scattered regions                                                                                            

        extra_stars_corr = 0 #need to start iteration with having no extra stars to remove, need to iterate and use less and less stars to populate the\rest of the region                                                                                                                                         
        times10_stars_it = [c_times10[idx_removed_masks]] #use this in the end to produce scattered background with masks taken out...          
        for n in range(iterations):
            x_times10_extra = []
            y_times10_extra = []
            
            r_times10_extra = np.random.uniform(low=0, high=(a)**2., size = (np.int(extra_stars - extra_stars_corr ))) #radius of scramble data, hi\gh=radius data **2                                                                                                                                          
            angle_times10_extra = np.random.uniform(low=0, high=2.*np.pi, size = (np.int(extra_stars- extra_stars_corr)))
            x_times10_extra.append(np.sqrt(r_times10_extra) * np.cos(angle_times10_extra))
            y_times10_extra.append(np.sqrt(r_times10_extra) * np.sin(angle_times10_extra)*b/a)

            #the ones that are now in masks should be re-cointed and distrributred again until the number og stars outside                                      
            #the mask is the same number as in the read in region                                                                                               

            c_times10_extra = coord.SkyCoord(ra = x_times10_extra + data_ra_center, dec = y_times10_extra + data_dec_center, unit='deg') #all data in\ regions                                                                                                                                                    
            mask_contains_extra = []
            idx_removed_masks_it1 = np.ones([len(r_times10_extra)], dtype=bool)

            for i in range(len(mask_regions_overlap)): # for the three masks....                                                                        
                mask_contains_extra_i = mask_regions_overlap[i].contains(skycoord = c_times10_extra, wcs=wcs)
                mask_contains_extra.append(mask_contains_extra_i)

                #Also store all stars that ARE NOT within the masks                                                                                     
            for i in range(len(mask_regions_overlap)): #for all different intersecting masks                                                            
                for k in range(len(r_times10_extra)): #for all stars in one of the scattered regions                                                   
                    if mask_contains_extra[i][0][k] == True: #if that mask  #the zero is to grab list;..                                                
                        idx_removed_masks_it1[k] = False

            extra_stars_corr = extra_stars_corr + len(c_times10_extra[0][idx_removed_masks_it1])#first this was all extra stars, but need to use less   
            times10_stars_it.append(c_times10_extra[0][idx_removed_masks_it1])
            if extra_stars - extra_stars_corr == 0:
                break
                 #count all stars in the n iterations of scattered regions                                                                             

        times10_stars_allsamples.append(times10_stars_it) #there are no_samples of these (100)                                                      



        #plt.figure(figsize=(5,5))
       # for n in range(len(times10_stars_allsamples)):  
       #     plt.scatter(times10_stars_allsamples[n][0].ra.wrap_at(180*u.deg).deg, times10_stars_allsamples[n][0].dec.deg, s=1, c='grey')   
        #    plt.scatter(c_data.ra.wrap_at(180 * u.deg).deg,c_data.dec.deg,s=1, c='purple')
         #   plt.show()
    

        X_sphere_times10 = []
        Y_sphere_times10 = []
        for n in range(len(times10_stars_allsamples)):
#        print(times10_stars_allsamples[n][0])
            alpha = times10_stars_allsamples[n][0].ra.rad
        
            delta = times10_stars_allsamples[n][0].dec.rad
            X_sphere_times10_i, Y_sphere_times10_i = cc.spherical_skycoords(alpha_0, delta_0, alpha, delta)
            X_sphere_times10.append(X_sphere_times10_i*u.rad.to(u.deg))
            Y_sphere_times10.append(Y_sphere_times10_i*u.rad.to(u.deg))

   

        X_sphere_times10_rht = np.concatenate(X_sphere_times10[:]) * kpc_conversion
        Y_sphere_times10_rht = np.concatenate(Y_sphere_times10[:]) * kpc_conversion

        
        #Now I need to Hough transform and then compute dA/A = (n_bin/10timesNstar)

        nx10_tot = len(Y_sphere_times10_rht)
        #Divide into 19 different subsets of stars                                                                                                                                                  
        n_range = np.linspace(0,nx10_tot,20)
        if verbose:
            print("Hough Transforming " +str(nx10_tot) + " stars in groups of "+  str(np.int(nx10_tot/len(n_range))))

        rho_grid =  np.zeros([grid_x,grid_y]) #empty rho_grid file that span the rho/theta space (can
        for k in range(len(n_range)-1): #iterate over the 9 groups of stars
            print('iteration number: ' + str(k+1) +' out of ' + str(len(n_range)-1))
            k_i = np.int(n_range[k]) #run from k to k_p1  stars which are the groups in the n_range
            k_ip1 = np.int(n_range[k+1])
 #           print('selecting stars from ' + str(k_i) + ' to ' + str(k_ip1))
            pos_rht_i = X_sphere_times10_rht[k_i:k_ip1],Y_sphere_times10_rht[k_i:k_ip1]
            rho_sub, theta_sub = HSS.HT_starpos(pos_rht_i, theta_s)
            #at the end we have filled all of rho and theta with the 10xstars but divided into 9 different groups
            rho_grid_i, edgex, edgey = np.histogram2d(np.array(theta_sub).flatten(), np.array(rho_sub).flatten(), bins = [grid_x, grid_y])
            #add each grid from each subset of stars to the previous subset rho/theta grid
            rho_grid += rho_grid_i


    
        r = np.max(np.round(rho))
    #    print('do we get here?')
        A =(np.pi*(r)**2) #area of region
 
        #this rho grid now has dn_ran stars in each region
        dA_temp = (rho_grid/Nstars_times10) * A # should be in kpc^2 like Susan's analytic



        dA = dA_temp.T
        
    
    return dA# X_sphere_scattered_rht, Y_sphere_scattered_rht
