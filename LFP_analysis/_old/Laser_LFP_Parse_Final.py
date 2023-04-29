# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 10:06:32 2017

@author: Bradly
"""
#Import necessary tools
import numpy as np
import easygui
import tables
import os

#Get name of directory where the data files and hdf5 file sits, and change to that directory for processing
dir_name = easygui.diropenbox()
os.chdir(dir_name)

#Look for the hdf5 file in the directory
file_list = os.listdir('./')
hdf5_name = ''
for files in file_list:
	if files[-2:] == 'h5':
		hdf5_name = files

#Open the hdf5 file
hf5 = tables.open_file(hdf5_name, 'r+')

# Create a LFP_laser group in the hdf5 file (if one exists, remove and create new)
try:
    hf5.remove_node('/LFP_Lasers', recursive = True)
except:
    pass
hf5.create_group('/', 'LFP_Lasers')

#Get laser conditions from hdf5 file (after ancillary analysis.py)
laser_conditions = hf5.root.ancillary_analysis.laser_combination_d_l[:]
laser_trials = hf5.root.ancillary_analysis.trials[:]

#Get taste and LFP information
num_tastes = len(hf5.list_nodes('/spike_trains'))
lfp_nodes = hf5.list_nodes('/Parsed_LFP')

#Get trial information based on tastants (assumes that each taste has equal number of trials)
num_trials = hf5.root.spike_trains.dig_in_0.spike_array[:].shape[0]

# Run through the tastes and laser conditions, and build arrays with respective data
for x in range(laser_conditions.shape[0]):
    
    #Create identifier based on laser combination information and create group within "LFP_Lasers" node
    las_type = 'laser_combos_d_l_'+str(int(laser_conditions[x,0]))+'_'+str(int(laser_conditions[x,1]))         
    hf5.create_group('/LFP_Lasers', las_type)
    
    #Loop through taste arrays (dig_in files), identify trial number pertaining to laser condition, and built array with respective LFP data
    for y in range(num_tastes):
        taste = 'taste_' + str(y)
        
        #Collapse data across electrodes to obtain LFPS by tastes and laser conditions
        lfp_coll = np.mean(lfp_nodes[y][:],axis=0)
        # Pick the trials of taste y in laser condition x
        trial_group = np.where((laser_trials[x, :] >= y*num_trials)*(laser_trials[x, :] < (y+1)*num_trials) == True)[0] #Create array of trials to index based on the number of trials per taste and laser combinations (assumes equal trial numbers per taste)
        # The trials picked above are on the absolute scale of trial numbers (so taste 3, for instance, will have trials 90 to 119 if there were a total of 120 trials). We need to convert this into a 0-29 scale (for 30 trials of each taste) to be able to use these numbers on the LFP arrays
        trial_group = laser_trials[x, trial_group] - int(y*num_trials)
        laser_trial_LFPs = lfp_coll[trial_group]
               
        # Create arrays (based on tastants) to build under corresonding laser condition
        hf5.create_array('/LFP_Lasers/%s' % las_type, '%s' %taste, laser_trial_LFPs)
        hf5.flush()

print("If you want to compress the file to release disk space, run 'blech_hdf5_repack.py' upon completion.")        
hf5.close()
