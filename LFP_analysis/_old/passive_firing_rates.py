#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 13:48:08 2019

@author: bradly
"""

# import Libraries
# Built-in Python libraries
import os # functions for interacting w operating system
import sys # access to functions/variables at the level of the interpreter

#import operator tools for list manipulations
from itertools import groupby
from operator import itemgetter

# 3rd-party libraries
import numpy as np # module for low-level scientific computing
import scipy as sp # library for working with NumPy arrays
import scipy.io as sio # read/write data to various formats
from scipy import signal # signal processing module
import matplotlib.pyplot as plt # makes matplotlib work like MATLAB. ’pyplot’ functions.
from scipy.stats import sem  
import easygui
import tables
import pickle #for data storage and retreival

# Define SEM function for smoothing figures (they are messy because so long)
def sliding_mean(data_array, window=5):  
    data_array = np.array(data_array)  
    new_list = []  
    for i in range(len(data_array)):  
        indices = range(max(i - window + 1, 0),  
                        min(i + window + 1, len(data_array)))  
        avg = 0  
        for j in indices:  
            avg += data_array[j]  
        avg /= float(len(indices))  
        new_list.append(avg)  
          
    return np.array(new_list)  

#Get name of directory where you want to save output files to
save_name = easygui.diropenbox(msg = 'Choose directory you want output files sent to (and/or where ".dir" files are)',default = '/home/bradly/drive2/data')

#Get data_saving name
msg   = "What condition are you analyzing first?"
subplot_check1 = easygui.buttonbox(msg,choices = ["Saline","LiCl","Other"])
    
# Ask the user for the hdf5 files that need to be plotted together (fist condition)
dirs_1 = []
dir_name = easygui.diropenbox(msg = 'Choose first condition directory with a hdf5 file')
dirs_1.append(dir_name)
    
# Ask the user for the hdf5 files that need to be plotted together (second condition)
#Get data_saving name
msg   = "What condition are you analyzing second?"
subplot_check2 = easygui.buttonbox(msg,choices = ["Saline","LiCl","Other"])
    
dirs_2 = []
dir_name = easygui.diropenbox(msg = 'Choose second condition directory with a hdf5 file')
dirs_2.append(dir_name)
    
#modify save names
if subplot_check1==subplot_check2:
    subplot_check2=subplot_check2+'_2'    
	
# Get the psth paramaters from the user
params = easygui.multenterbox(msg = 'Enter the parameters for making firing rate spreads', fields = ['Pre stimulus (ms)','Window size (ms)', 'Step size (ms)','Smoothing Spline (1-10; 5 is conservative)'],values = ['0','20000','10000','5'])
for i in range(len(params)):
	params[i] = int(params[i])

#Get name of directory where the data files and hdf5 file sits, and change to that directory for processing
dirs_1_spike_rates =[];dirs_2_spike_rates =[]
for dir_name in dirs_1:
	#Change to the directory
	os.chdir(dir_name)
	#Locate the hdf5 file
	file_list = os.listdir('./')
	hdf5_name = ''
	for files in file_list:
		if files[-2:] == 'h5':
			hdf5_name = files
			
	#Open the hdf5 file
	hf5 = tables.open_file(hdf5_name, 'r')
	
	# Make directory to store the FRHs plots. Delete and remake the directory if it exists
	try:
		os.system('rm -r '+'./FRH')
	except:
		pass
	os.mkdir('./FRH')
    
	# Get the list of spike trains by digital input channels
	trains_dig_in = hf5.list_nodes('/spike_trains')
	
	# Plot FRHs by unit and channels
	for dig_in in trains_dig_in:
		os.mkdir('./FRH/'+str.split(dig_in._v_pathname, '/')[-1])
		trial_avg_spike_array = np.mean(dig_in.spike_array[:], axis = 0)
		for unit in range(trial_avg_spike_array.shape[0]):
			time = []
			spike_rate = []
			for i in range(0, trial_avg_spike_array.shape[1] - params[1], params[2]):
				time.append(i - params[0])
				spike_rate.append(1000.0*np.sum(trial_avg_spike_array[unit, i:i+params[1]])/float(params[1]))
			dirs_1_spike_rates.append(spike_rate)
			fig = plt.figure()
			plt.title('Condition: %s, Window: %i ms, Step: %i ms' % (subplot_check1, params[1], params[2]) + '\n' + 'Electrode: %i, Single Unit: %i, RSU: %i, FS: %i' % (hf5.root.unit_descriptor[unit]['electrode_number'], hf5.root.unit_descriptor[unit]['single_unit'], hf5.root.unit_descriptor[unit]['regular_spiking'], hf5.root.unit_descriptor[unit]['fast_spiking']))
			plt.xlabel('Time from injection (ms)')
			plt.ylabel('Firing rate (Hz)')
			plt.plot(time, spike_rate)
			plt.tight_layout()
			fig.savefig('./FRH/'+str.split(dig_in._v_pathname, '/')[-1]+'/Unit%i.png' % (unit))
			plt.close('all')
		
		#Plot FRH for all units together
		fig = plt.figure(figsize=(10, 8))
		plt.title('Condition: %s, Window: %i ms, Step: %i ms' % (subplot_check1, params[1], params[2]) + '\n' + 'All Units: %i' % (np.size(dirs_1_spike_rates,axis=0)))
		plt.xlabel('Time from injection (ms)')
		plt.ylabel('Firing rate (Hz)')
		plt.plot(time, np.mean(dirs_1_spike_rates,axis=0))
		plt.tight_layout()
		fig.savefig('./FRH/'+'/All_Units_%i.png' % (np.size(dirs_1_spike_rates,axis=0)))
		
		#Plot FRH smoothed
		smooth_FRH = sliding_mean(np.mean(dirs_1_spike_rates,axis=0),window=params[3])
		sem_FRH = sem(dirs_1_spike_rates,axis=0)
		fig = plt.figure( figsize=(10, 8))
		plt.title('Condition: %s, Window: %i ms, Step: %i ms' % (subplot_check1, params[1], params[2]) + '\n' + 'All Units: %i, Smoothing Spline: %ith order' % (np.size(dirs_1_spike_rates,axis=0), params[3]))
		plt.xlabel('Time from injection (ms)')
		plt.ylabel('Firing rate (Hz)')
		plt.plot(time, np.mean(dirs_1_spike_rates,axis=0),color='skyblue', linewidth=1,label='Mean Firing Rate')
		plt.plot(time, smooth_FRH,color='blue', linewidth=1,label='Smoothed Firing Rate')
		plt.fill_between(time, np.mean(dirs_1_spike_rates,axis=0) - sem_FRH,  
		                 np.mean(dirs_1_spike_rates,axis=0) + sem_FRH, color="grey",alpha=0.3,label='SEM') 
		plt.legend() 
		fig.savefig('./FRH/'+'/All_Units_%i_Smoothed.png' % (np.size(dirs_1_spike_rates,axis=0)))
		plt.close('all')

    #Close the hdf5 file
	hf5.close()	

for dir_name in dirs_2:
	#Change to the directory
	os.chdir(dir_name)
	#Locate the hdf5 file
	file_list = os.listdir('./')
	hdf5_name = ''
	for files in file_list:
		if files[-2:] == 'h5':
			hdf5_name = files

	#Open the hdf5 file
	hf5 = tables.open_file(hdf5_name, 'r')
	
	# Make directory to store the FRHs plots. Delete and remake the directory if it exists
	try:
		os.system('rm -r '+'./FRH')
	except:
		pass
	os.mkdir('./FRH')
    
	# Get the list of spike trains by digital input channels
	trains_dig_in = hf5.list_nodes('/spike_trains')
	
	# Plot FRHs by unit and channels
	for dig_in in trains_dig_in:
		os.mkdir('./FRH/'+str.split(dig_in._v_pathname, '/')[-1])
		trial_avg_spike_array = np.mean(dig_in.spike_array[:], axis = 0)
		for unit in range(trial_avg_spike_array.shape[0]):
			time = []
			spike_rate = []
			for i in range(0, trial_avg_spike_array.shape[1] - params[1], params[2]):
				time.append(i - params[0])
				spike_rate.append(1000.0*np.sum(trial_avg_spike_array[unit, i:i+params[1]])/float(params[1]))
			dirs_2_spike_rates.append(spike_rate)
			fig = plt.figure()
			plt.title('Condition: %s, Window: %i ms, Step: %i ms' % (subplot_check2, params[1], params[2]) + '\n' + 'Electrode: %i, Single Unit: %i, RSU: %i, FS: %i' % (hf5.root.unit_descriptor[unit]['electrode_number'], hf5.root.unit_descriptor[unit]['single_unit'], hf5.root.unit_descriptor[unit]['regular_spiking'], hf5.root.unit_descriptor[unit]['fast_spiking']))
			plt.xlabel('Time from injection (ms)')
			plt.ylabel('Firing rate (Hz)')
			plt.plot(time, spike_rate)
			plt.tight_layout()
			fig.savefig('./FRH/'+str.split(dig_in._v_pathname, '/')[-1]+'/Unit%i.png' % (unit))
			plt.close('all')
		
		#Plot FRH for all units together
		fig = plt.figure(figsize=(10, 8))
		plt.title('Condition: %s, Window: %i ms, Step: %i ms' % (subplot_check2, params[1], params[2]) + '\n' + 'All Units: %i' % (np.size(dirs_2_spike_rates,axis=0)))
		plt.xlabel('Time from injection (ms)')
		plt.ylabel('Firing rate (Hz)')
		plt.plot(time, np.mean(dirs_2_spike_rates,axis=0))
		plt.tight_layout()
		fig.savefig('./FRH/'+'/All_Units_%i.png' % (np.size(dirs_2_spike_rates,axis=0)))
		
		#Plot FRH smoothed
		smooth_FRH = sliding_mean(np.mean(dirs_2_spike_rates,axis=0),window=params[3])
		sem_FRH = sem(dirs_2_spike_rates,axis=0)
		fig = plt.figure( figsize=(10, 8))
		plt.title('Condition: %s, Window: %i ms, Step: %i ms' % (subplot_check2, params[1], params[2]) + '\n' + 'All Units: %i, Smoothing Spline: %ith order' % (np.size(dirs_2_spike_rates,axis=0), params[3]))
		plt.xlabel('Time from injection (ms)')
		plt.ylabel('Firing rate (Hz)')
		plt.plot(time, np.mean(dirs_2_spike_rates,axis=0),color='skyblue', linewidth=1,label='Mean Firing Rate')
		plt.plot(time, smooth_FRH,color='blue', linewidth=1,label='Smoothed Firing Rate')
		plt.fill_between(time, np.mean(dirs_2_spike_rates,axis=0) - sem_FRH,  
		                 np.mean(dirs_2_spike_rates,axis=0) + sem_FRH, color="grey",alpha=0.3,label='SEM') 
		plt.legend() 
		fig.savefig('./FRH/'+'/All_Units_%i_Smoothed.png' % (np.size(dirs_2_spike_rates,axis=0)))
		plt.close('all')

    #Close the hdf5 file
	hf5.close()

#Plot smoothed FRH from both conditions
fig = plt.figure( figsize=(10, 8))
plt.title('Window: %i ms, Step: %i ms' % (params[1], params[2]) + '\n' + 'Condition 1 Units: %i, Condition 2 Units: %i, Smoothing Spline: %ith order' % (np.size(dirs_1_spike_rates,axis=0),np.size(dirs_2_spike_rates,axis=0), params[3]))
plt.xlabel('Time from injection (ms)')
plt.ylabel('Firing rate (Hz)')
plt.plot(time, sliding_mean(np.mean(dirs_1_spike_rates,axis=0),window=params[3]),color='blue', linewidth=1,label='%s:Smoothed Firing Rate' %subplot_check1)
plt.plot(time, sliding_mean(np.mean(dirs_2_spike_rates,axis=0),window=params[3]),color='green', linewidth=1,label='%s:Smoothed Firing Rate'%subplot_check2)		
plt.legend() 
saving_name =  os.path.join(save_name, '%s Combined_sessions_Smoothed.png' %(hdf5_name[:4]))
fig.savefig(saving_name)
plt.close('all')


fig = plt.figure( figsize=(10, 8))
plt.plot(time, (sliding_mean(np.mean(dirs_2_spike_rates,axis=0),window=params[3])-sliding_mean(np.mean(dirs_1_spike_rates,axis=0),window=params[3]))/sliding_mean(np.mean(dirs_1_spike_rates,axis=0),window=params[3]))
plt.axhline(0, linestyle='--', color='grey', linewidth=2)
plt.xlabel('Time from injection (ms)')
plt.ylabel(r'$\Delta$'+ ' Firing rate (Hz)')