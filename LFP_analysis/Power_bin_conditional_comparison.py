#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 08:46:57 2020

@author: bradly
"""

import os
import pickle
import easygui
import numpy as np
import matplotlib.pyplot as plt 
from scipy.signal import savgol_filter
# =============================================================================
# #Establish files to work on
# =============================================================================
# Ask the user for the hdf5 files that need to be plotted together (fist condition)
pickle_dirs = []
while True:
	dir_name = easygui.diropenbox(msg = 'Choose Directories with pickle files, hit cancel to stop choosing')
	try:
		if len(dir_name) > 0:	
			pickle_dirs.append(dir_name)
	except:
		break   

#Establish conditional names
group_conditions = easygui.multenterbox(msg = 'Input condition names:', fields = ['Cohort 1:','Cohort 2:'], values = ['Control','Experimental'])

#specify frequency bands
iter_freqs = [
        ('Theta', 4, 7),
        ('Mu', 8, 12),
        ('Beta', 13, 25),
        ('Gamma', 30, 45)]

#Flip through files and store pickle names
corr_pickles = []; pickles = []
for dir_name in pickle_dirs:	
	#Change to the directoryfrom itertools import islice
	os.chdir(dir_name)
	#Locate the pickle files
	file_list = os.listdir('./')
	pickle_name = ''
	for files in file_list:
		if files[-6:] == 'corr.p':
			pickle_name_corr = files
		if files[-6:] == 'tats.p':
			pickle_name = files
	corr_pickles.append(pickle_name_corr)
	pickles.append(pickle_name)

#Load pickles for plotting
file_name = pickle_dirs[0] +'/'+ corr_pickles[0]
set1_corr = pickle.load( open(file_name, "rb" ) )
	
file_name = pickle_dirs[1] +'/'+ corr_pickles[1]
set2_corr = pickle.load( open(file_name, "rb" ) )	
	
#Plot together
#Initiate figure
fig,axes = plt.subplots(nrows=2, ncols=2,sharex=True, sharey=True,
	        figsize=(12, 12), squeeze=False)
fig.text(0.075, 0.5,'Percentage of Animals', va='center', 
        rotation='vertical',fontsize=15, fontweight='bold')
fig.text(0.5, 0.05, 'Time Post-Injection (min)', ha='center',fontsize=15, fontweight='bold')
axes_list = [item for sublist in axes for item in sublist]
for band in range(len(iter_freqs)):
	#indicate axes
	ax = axes_list.pop(0)
		
	#Extract data from lists
	sig_set1 = np.where(np.array([item[band] for item in set1_corr]) <= 0.05)
	sig_set2 = np.where(np.array([item[band] for item in set2_corr]) <= 0.05)
	
	#Get occurrences
	sig_occ1 = np.unique(list(sig_set1[1]), return_counts=True)
	sig_occ2 = np.unique(list(sig_set2[1]), return_counts=True)
	
	#Plot
	ax.plot(savgol_filter(100*sig_occ1[1]/len(set1_corr),9,1),\
		 color='midnightblue',linewidth=2,label=group_conditions[0], marker='1',markersize=6)
	ax.plot(savgol_filter(100*sig_occ2[1]/len(set2_corr),9,3),\
		 color='salmon',linewidth=2,label=group_conditions[1], marker='2',markersize=6)
	ax.scatter(sig_occ1[0],100*sig_occ1[1]/len(set1_corr),\
		 color='midnightblue', marker='1',alpha=0.5)
	ax.scatter(sig_occ2[0],100*sig_occ2[1]/len(set2_corr),\
		 color='salmon', marker='2',alpha=0.5)
	
	#Set binning parameters (assumes shapes are the same across conditions)
	step_size = 60000
	n_bins = 1200000//step_size
	bins = np.arange(0, 1200000, step_size)
	barWidth = 0.25
	
	#Formatting
	ax.set_xticks([])
	ax.set_xticks([r + barWidth/2 for r in range(20)])
	ax.set_xticklabels([str(x+1) for x in range(20)])
	ax.set_title('%s' %(iter_freqs[band][0]))
	
	#Set global legend 
	if band == len(iter_freqs)-1:
		handles, _ = ax.get_legend_handles_labels()
		ax.legend(handles, group_conditions,\
			bbox_to_anchor=(1.0, 1.0),ncol=1,fontsize=15)
		
#Save	
fig.subplots_adjust(hspace=0.2,wspace = 0.05)
plt.suptitle('State-dependent Power Modulation\n Animals: %i - %s /  %i - %s' \
			 %(len(set1_corr), group_conditions[0],len(set2_corr),group_conditions[1]),size=18,fontweight="bold")  
fig.savefig(pickle_dirs[0]+'/'+'%s_%s_Power_stats_comparison.png' %(len(set1_corr),len(set2_corr)))   
plt.close(fig)		

	

	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	