#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 16:11:10 2019

@author: bradly
"""

# import Libraries
# Built-in Python libraries
import os # functions for interacting w operating system

# 3rd-party libraries
import numpy as np # module for low-level scientific computing
import matplotlib.pyplot as plt # makes matplotlib work like MATLAB. ’pyplot’ functions.
from scipy.stats import sem  
import easygui
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
save_name = easygui.diropenbox(msg = 'Choose directory you want output files sent to (and/or where ".dir" files are)',default = '/home/bradly/drive2/data/Affective_State_Protocol/LiCl/Combined_Passive_Data/LiCl_Saline/freq_bands/LiCl')

#establish which dump files to choose from
dir_name_1 = easygui.diropenbox(msg = 'Choose directory with first ".dump" file.')
dir_name_2 = easygui.diropenbox(msg = 'Choose directory with second ".dump" file.')

# Get the psth paramaters from the user
params = easygui.multenterbox(msg = 'Enter the parameters for making firing rate spreads', fields = ['Pre stimulus (ms)','Window size (ms)', 'Step size (ms)','Smoothing Spline (1-10; 5 is conservative)'],values = ['0','10000','5000','5'])
for i in range(len(params)):
	params[i] = int(params[i])

#set conditoni variables
cond1 = dir_name_1.split("/")[-1]
cond2 = dir_name_2.split("/")[-1]

#load tuples
#Change to the directory
os.chdir(dir_name_1)
#Locate the dump file
file_list = os.listdir('./')
for files in file_list:
	if files[-4:] == 'dump':
		tup_name = files
dir_1_tup = pickle.load(open(tup_name, 'rb'))

#Change to the directory
os.chdir(dir_name_2)
#Locate the dump file
file_list = os.listdir('./')
for files in file_list:
	if files[-4:] == 'dump':
		tup_name = files
dir_2_tup = pickle.load(open(tup_name, 'rb'))

#combine dictionaries
new_dict = {}
new_dict['a']=dir_1_tup; new_dict['b']=dir_2_tup

#hold variables for naming figure
held_units = []; num_animals = []; change_scored_combined = []; 

#Plot using for loop through dictionaries
fig = plt.figure( figsize=(10, 8))

for key in new_dict:
	conditions = []
	for cond in range(0,2):
		
		#grab condtion label
		conditions.append(new_dict[key][list(new_dict[key].keys())[cond]]['condition'])
		
		#extract FR arrays based on conditionin cohort
		if cond == 0:
			held_FRs_cond1_2 = new_dict[key][list(new_dict[key].keys())[cond]]['FRs_binned']
		else:			
			held_FRs_cond2_2 = new_dict[key][list(new_dict[key].keys())[cond]]['FRs_binned']
			
			
		#extract common variables
		time = new_dict[key][list(new_dict[key].keys())[cond]]['time']
		p_vector = new_dict[key][list(new_dict[key].keys())[cond]]['p_vals']
		
		#Mean of all animals in this cohort
		all_anmimals_1 = sliding_mean(np.mean(held_FRs_cond1_2,axis=0),window=params[3])
		all_anmimals_2 = sliding_mean(np.mean(held_FRs_cond2_2,axis=0),window=params[3])
		change_scored = (all_anmimals_2-all_anmimals_1)/all_anmimals_1
		sem_change_scored = (sem(held_FRs_cond2_2,axis=0)-sem(held_FRs_cond1_2,axis=0))/sem(held_FRs_cond1_2,axis=0)
		
		#Locate significant bins			
		sig_bins = list(np.where(np.array(p_vector) <= 0.05))	
		
	#hold change score arrays for analyses with animal and cell count
	num_animals.append(np.size(new_dict[key][list(new_dict[key].keys())[cond]]['held_cells']))
	held_units.append(np.size(held_FRs_cond1_2,axis=0))
	change_scored_combined.append(change_scored)
	
	#create label
	for i in range(np.size(conditions)):
		conditions[i] = conditions[i].replace("_2","")
		
	cond_label = '/'.join([conditions[0],conditions[1]])
	
	#Plot
	if key == 'a':
		plt.plot(time,change_scored, linestyle='-', color='darkblue', linewidth=4, label = cond_label)
		plt.fill_between(time, change_scored - sem_change_scored,  
                 change_scored + sem_change_scored, color="grey",alpha=0.3)
		plt.plot(time,change_scored,'rD',markevery=sig_bins)
	else:
		plt.plot(time,change_scored, linestyle='-', color='darkgreen', alpha=0.9,linewidth=4, label = cond_label)
		plt.fill_between(time, change_scored - sem_change_scored,  
                 change_scored + sem_change_scored, color="grey",alpha=0.3,label='SEM')
		plt.plot(time,change_scored,'rD',markevery=sig_bins,label='p < 0.05')	

plt.title('Held Unit Firing Rate Histogram' +'\n' + 'Cond1 Animals: %i, Cond2 Animals: %i' % (num_animals[0],num_animals[1]) +'\n' + 'Window: %i ms, Step: %i ms' %(params[1], params[2]) + '\n' + 'Units: %i, Units: %i, Smoothing Spline: %ith order' % (held_units[0],held_units[1], params[3]))
plt.xlabel('Time from injection (ms)')
plt.ylabel(r'$\Delta$'+ ' Firing rate (Hz)')
plt.legend()

#Save figure
fig.savefig(save_name+'/Compared_Held_Stats_Smoothed_%ims_window.png' % (params[1]))
plt.close('all')
