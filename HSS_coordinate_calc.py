import numpy as np
import astropy.coordinates as coord
import astropy.units as u



def spherical_skycoords(alpha_0, delta_0, alpha, delta):

    """                                                                                                                                                         
    Transforms from RA/DEC to coordinates spherical skycoordinates                                                                                              
                                                                                                                                                                
    Inputs:                                                                                                                                                     
        alhpa_0: center of your region in RA [radians]                                                                                                          
        delta_0: center of your region in dec [radians]                                                                                                         
        alpha: position of all stars in region in RA [radians]                                                                                                  
        delta: positions of all stars in region in Dec [radians]                                                                                                
                                                                                                                                                                
    Return:                                                                                                                                                     
        X_sphere in spherical skyycoordinates centered on 0 in [radians]                                                                                        
        Y_sphere in spherical skyycoordinates centered on 0 in [radians]                                                                                        
    """

    X_sphere = (np.cos(delta) * np.sin(alpha-alpha_0) )/\
        (np.cos(delta_0) * np.cos(delta) * np.cos(alpha-alpha_0) + np.sin(delta_0)*np.sin(delta))

    Y_sphere =  (np.cos(delta_0)*np.sin(delta) - np.cos(delta)*np.sin(delta_0)*np.cos(alpha-alpha_0)) / \
                (np.sin(delta_0)*np.sin(delta) + np.cos(delta_0)*np.cos(delta) *np.cos(alpha-alpha_0)   )

    return X_sphere, Y_sphere


def data_skycoords(pos, kpc_conversion):
    """                                                                                                                                                         
    Function reads in region of data in Ra/dec in units of deg, transforms to skycord object, finds center of data region,                                      
    transforms data to spherical coords, converts to kpc at distance of object galaxy of interest, finds ang. radius of region                                  
                                                                                                                                                                
    Inputs:                                                                                                                                                     
         pos: array of star positions pos = [x,y] in ra/dec degrees shape (n,2)                                                                                                    
         kpc_conversion: conversion from deg to kpc for your galaxy/data of interest                                                                            
                                                                                                                                                                
    return:                                                                                                                                                     
         alpha_0: RA center of data region in radians                                                                                                           
         delta_0: Dec center of data region in radius                                                                                                           
         X_sphere_rht: RA in spherical skycoords centered on 0 in kpc                                                                                           
         Y_sphere_rht: dec in spherical skyycords  centered on 0 in kpc                                                                                         
         data_ang_radius: extent of data in spherical skycoords in deg                                                                                          
         data_ra_center: RA center of data region in degrees (maybe redundant when we have alpha_0)                                                             
         data_dec_center: Dec center of data region in degrees (maybe redundant when we have alpha_0)                                                           
         c_data: all data from the region as an astropy skycoord objects                                           
         a, b: extent of data region for generating backgrounds
                                                                                                                                             
    """

    #Read in pos from region and transform to astropy skycoords                                                                                                 
    c_data = coord.SkyCoord(ra = pos[0], dec = pos[1], unit='deg')
    #find the centrer of the loaded in region in degrees                                                                                                        
    data_ra_center = np.max(c_data.ra.wrap_at(180*u.deg).deg) - (np.max(c_data.ra.wrap_at(180*u.deg).deg) - np.min(c_data.ra.wrap_at(180*u.deg).deg))/2
    data_dec_center = np.max(c_data.dec.deg) - (np.max(c_data.dec.deg) - np.min(c_data.dec.deg))/2
    #region center for tangent point projections                                                                                                                
    c_data_center = coord.SkyCoord(data_ra_center, data_dec_center, unit='deg') #just the center of the region!                                                 

    #inputs for transform to spherical skycoords                                                                                                                
    alpha_0 = c_data_center.ra.rad #center of region                                                                                                            
    alpha = c_data.ra.wrap_at(180*u.deg).rad  #all stars in region                                                                                              
    delta_0 = c_data_center.dec.rad #center of region                                                                                                           
    delta = c_data.dec.rad #all stars in region                                                                                                                 

    #Call Spherical SkyyCoord function for transformation                                                                                                       
    X_sphere_data, Y_sphere_data = spherical_skycoords(alpha_0, delta_0, alpha, delta)

    #Below is the data to feed to the Hough transform and rho/theta grid, centered on zero, in kpc                                                              
    X_sphere_rht = X_sphere_data*u.rad.to(u.deg) * kpc_conversion #actual data for RHT analysis                                                                
    Y_sphere_rht = Y_sphere_data*u.rad.to(u.deg) * kpc_conversion

    # find the angular radius of the data region in the spherical coord system (circle in this projection)                                                      
    # I need this for generating fake scattered backgrounds  - do I?                                                                                            
    # Compute the separation of each point from its center
    ang_radius = c_data.separation(c_data_center).value
    data_ang_radius = np.max(ang_radius)*u.deg #the maximum is the radius of this regions in skycoords [deg]                                                    

    #define the extent of your circle/ellipse that your data region spans 
    a_list = np.abs(c_data.ra.wrap_at(180*u.deg).deg - data_ra_center)
    
    a = np.max(a_list)#+0.01 #find the maximum extent to span your background region                                                                                                      
    # adding small number instead of finding bug, the maximum distance isn't always in major minor axis ra/dec direction                 
    # maybe the "inverse spherical skycoord trans." would work (from data_ang_radius to projected).
    
    # The data region is not necessarily a circle, but if it is a=b and the below still works                                                                                       
    b_list = np.abs(c_data.dec.deg - data_dec_center)
    b = np.max(b_list)#+0.01
    
    return alpha_0, delta_0, X_sphere_rht, Y_sphere_rht, data_ang_radius, data_ra_center, data_dec_center, c_data, a, b


def data_unitless(pos):
    """
    Function outputs data x,y positions centered on 0 in any units and the extent of that data region
 
    input:
         pos: x,y position coordinates of each star in regio in any coordinates, the shape os pos should be (n,2). Data should be read in as circular region   

    return: 
         X_rht, Y_rht: coordinates  with the same number of points as data              
                        centered on 0,0 so they can be used for input to HT_starpos                                                                                  
                   
         a,b: extent of data region for background calculation
    """
    
    x_center = np.max(pos[0]) - (np.max(pos[0]) - np.min(pos[0]))/2
    y_center = np.max(pos[1]) - (np.max(pos[1]) - np.min(pos[1]))/2

    X_rht = pos[0] - x_center
    Y_rht = pos[1] - y_center
    a = np.abs(np.max(pos[0]) - np.min(pos[0]))/2 #dividing by 2 to only get half of semi-major axis (see ellipse definition)
    b = np.abs(np.max(pos[1]) - np.min(pos[1]))/2 
    

    return X_rht, Y_rht, a, b
