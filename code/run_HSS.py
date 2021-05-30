import numpy as np
import matplotlib.pyplot as plt
import HSS 

# From terminal:
# python run_HSS.py

path_plot = '/Users/spearson/Desktop/'
#import test data
filename = 'fakestream.txt' #your data file name, in kpc in this example
# load in stream
fakestream = np.genfromtxt(filename) # (n,2) array n is number of stars in region 
pos = fakestream[:,0], fakestream[:,1] # in deg
# specify unit
unit = "deg" #unit of your input data "deg" for observations or "unitless" for e.g. simulations  
# define distance to your dataset of interest
# this is only relevant if unit = "deg", otherwise set kpc_conversion = 1         
d_galaxy = 785 # [kpc] update for other galaxy than M31                                                                                          
kpc_conversion = np.pi * d_galaxy / 180. #from deg to kpc    
#theta spacing    
delta_t = 0.1 #[deg] 
#spacing in rho (same as search width of interest)
drho = 0.4 #[kpc]
#threshold of statistical significance, -log10Pr(X>k)
outlier = 20 #i.e. log10Pr(X>k)< -20 in this example
#for plots
pointsize = 1
#Are you removing masks from your dataset (False/True)
mask = False #
#if yes, load in masks_pos.txt and masks_size.txt
#read out plots and updates in run
verbose= True 
#If overlapping regions only search 60% central parts
rho_edge = False #set to False if not using overlapping regions


HSS.RT_plot(pos, unit, kpc_conversion, delta_t, drho, outlier, pointsize, mask, filename[:-4], path_plot, verbose, rho_edge)

