#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 15:08:17 2019

@author: bradly
"""

# =============================================================================
# Import stuff
# =============================================================================

# import Libraries
# Built-in Python libraries
import os # functions for interacting w operating system
from os.path import basename #For saving purposes
from datetime import datetime

# 3rd-party libraries
import numpy as np # module for low-level scientific computing
import matplotlib.pyplot as plt # makes matplotlib work like MATLAB. ’pyplot’ functions.
import easygui
import sys
import tables
from tqdm import trange
import pandas as pd
import scipy as sp # library for working with NumPy arrays
from scipy import signal # signal processing module
from scipy.ndimage.filters import gaussian_filter
import matplotlib.cm as cm # colormap module
import re
import scipy.stats
from scipy.signal import hilbert 
from scipy.signal import butter
from scipy.signal import filtfilt

# =============================================================================
# #Define Functions to be used
# =============================================================================
#define bandpass filter parameters to parse out frequencies
def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut /nyq
    high = highcut/nyq
    b, a = butter(order, [low, high], btype='bandpass')
    y = filtfilt(b, a, data)
    return y

#Get name of directory where you want to save output files to
save_name = easygui.diropenbox(msg = 'Choose directory you want output files sent to (and/or where ".dir" files are)',default = '/home/bradly/drive2/data/Affective_State_Protocol/LiCl/Combined_Passive_Data/LiCl_Saline')

#Ask user to check LFP traces to ensure channels are not shorted/bad in order to remove said channel from further processing
folder_check = 	easygui.multchoicebox(msg = 'Choose the folders containing .hf5 files to compare.', choices = tuple([i for i in next(os.walk(save_name))[1]]))
if folder_check:
	
	#Create empty dataframe to store the combined sets
	grouped_df=pd.DataFrame()
	
	#Flip through directories to store data
	for i in range(len(folder_check)):
		dir_name = save_name+'/%s' %(folder_check[i])
		os.chdir(dir_name)
	
		#Look for the hdf5 file in the directory
		file_list = os.listdir('./')
		hdf5_name = ''
		for files in file_list:
			if files[-2:] == 'h5':
				hdf5_name = files
		
		#establish naming variables
		cwd = os.getcwd()
		basename(dir_name)
		
		#pull in dframes
		dframe = pd.read_hdf(hdf5_name,'/%s' %(basename(dir_name)),'r+')
		
		#add identifier
		dframe.insert(0,'Condition',basename(dir_name))
		
		#store into merged set
		grouped_df = pd.concat([grouped_df,dframe],sort=True)
		
	#reset dataframe index
	grouped_df.reset_index(drop=True, inplace=True)
	
if folder_check is None:
	print('You have not chosen the correct number of folers to compare. Think about what you are doing and try again')
	sys.exit()

# =============================================================================
# #Estalbish parameters
# =============================================================================
taste_params = easygui.multenterbox(msg = 'Input taste identities:', fields = ['Taste 1 (dig_in_1)', 'Taste 2 (dig_in_2)','Taste 3 (dig_in_3)','Taste 4 (dig_in_4)'],values = ['NaCl','Sucrose','Citric Acid','QHCl'])
analysis_params = easygui.multenterbox(msg = 'Input analysis paramters:', fields = ['Pre-stimulus signal duration (ms; from set-up)','Post-stimulus signal duration (ms; from set-up)','Pre-Taste array start time (ms)', 'Taste array end time (ms)', 'Sampling Rate (samples per second)', 'Signal Window (ms)', 'Window Overlap (ms; default 90%)'], values = ['2000','5000','0','2500','1000','900','850'])
    
#create timing variables
pre_stim = int(analysis_params[0])
post_stim = int(analysis_params[1])
lower = int(analysis_params[2])
upper = int(analysis_params[3])
Fs = int(analysis_params[4])
signal_window = int(analysis_params[5])
window_overlap = int(analysis_params[6])
base_time = 2000   
  
#establish meshing and plotting paramters
plotting_params = easygui.multenterbox(msg = 'Input plotting paramters:', fields = ['Minimum frequency (Hz):','Maximum frequency (Hz):', 'Pre-stim plot time (ms):', 'Post-stim plot time (ms):'], values = ['3','40', '1000',int(upper)])

#specify frequency bands
iter_freqs = [
        ('Theta', 4, 7),
        ('Mu', 7, 12),
        ('Beta', 13, 25),
        ('Gamma', 30, 45)]

#Set t vector for plotting
t = np.array(list(range(0,int(analysis_params[0])+int(analysis_params[1]))))-pre_stim

#Flip through tastes and bands to produce figures to assess GEP
for taste in range(len(grouped_df['Taste'].unique())):
    #Query data
	query1 = grouped_df.query('Taste == @taste and Condition == @folder_check[0]')
	query2 = grouped_df.query('Taste == @taste and Condition == @folder_check[1]')

	fig = plt.figure(figsize=(11,8))
	fig,axes = plt.subplots(nrows=len(iter_freqs), ncols=2,sharex=True, sharey=False,figsize=(12, 8), squeeze=True)
	fig.text(0.5, 0.05, 'Milliseconds', ha='center',fontsize=15)
	axes_list = [item for sublist in axes for item in sublist]
	
	#Flip through bands and create figures	
	for ax,band in zip(axes.flatten(),trange(len(iter_freqs), desc = 'bands')):
		band_filt_sig1 = butter_bandpass_filter(data = query1.iloc[:,4:], 
	                                    lowcut = iter_freqs[band][1], 
	                                    highcut =  iter_freqs[band][2], 
	                                    fs = 1000)
		analytic_signal1 = hilbert(band_filt_sig1)
		instantaneous_phase1 = np.angle(analytic_signal1)
		x_power1 = np.abs(analytic_signal1)**2
		
		band_filt_sig2 = butter_bandpass_filter(data = query2.iloc[:,4:], 
	                                    lowcut = iter_freqs[band][1], 
	                                    highcut =  iter_freqs[band][2], 
	                                    fs = 1000)
		analytic_signal2 = hilbert(band_filt_sig2)
		instantaneous_phase2 = np.angle(analytic_signal2)
		x_power2 = np.abs(analytic_signal2)**2
		
		#Plot raw versus filtered LFP
		ax = axes_list.pop(0)
		ax.plot(t,np.mean(query1.iloc[:,4:].T,axis=1),'k-',alpha=0.3,lw=1,label='_nolegend_'); 
		ax.plot(t,np.mean(analytic_signal1.T,axis=1),'r',alpha=0.8,lw=2);
		
		ax.plot(t,np.mean(query2.iloc[:,4:].T,axis=1),'k-',alpha=0.3,lw=1,label='_nolegend_'); 
		ax.plot(t,np.mean(analytic_signal2.T,axis=1),'r',alpha=0.4,lw=2);
		handles, labels = ax.get_legend_handles_labels()
		
		#Formatting
		ax.set_xlim([-int(plotting_params[2]),int(plotting_params[3])])
		ax.set_ylim([np.min(np.mean(query1.iloc[:,4:].T,axis=1)),100])
		ax.vlines(x=0, ymin=np.min(np.mean(query1.iloc[:,4:].T,axis=1)),ymax=100, linewidth=3, color='k',linestyle=':',label='_nolegend_')
		ax.text(0.83,0.9,'%s (%i - %iHz)' %(iter_freqs[band][0],iter_freqs[band][1],iter_freqs[band][2]), ha='center', va='center', transform=ax.transAxes)
		
		#Plot Power over time
		ax = axes_list.pop(0)
		ax.plot(t,np.mean(x_power1.T,axis=1),'b',alpha=0.8,lw=2); 
		ax.plot(t,np.mean(x_power2.T,axis=1),'b',alpha=0.4,lw=2); 
		
		#Formatting
		ax.set_xlim([-int(plotting_params[2]),int(plotting_params[3])])
		ax.vlines(x=0, ymin=np.min(np.mean(x_power1.T,axis=1)),ymax=np.max([np.mean(x_power1.T,axis=1),np.mean(x_power2.T,axis=1)]), linewidth=3, color='k',linestyle=':')
		
	plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=-0.08)
	fig.legend( handles, labels=folder_check*2, loc='best', bbox_to_anchor=(0.9, 0.98), ncol=2 )
	plt.suptitle('Hilbert Transform and Instantaneous Power \nTaste: %s' %(taste_params[taste]),size=16)
	fig.savefig(save_name+'/%s_N_%i_Animals' %(taste_params[taste],len(grouped_df.Animal.unique())) + '_ComparedHilbert.png')   
	plt.show()
	plt.close(fig)

#Plot grouped tastants
fig = plt.figure(figsize=(11,8))
fig,axes = plt.subplots(nrows=len(iter_freqs), ncols=2,sharex=True, sharey=False,figsize=(12, 8), squeeze=True)
fig.text(0.5, 0.05, 'Milliseconds', ha='center',fontsize=15)
axes_list = [item for sublist in axes for item in sublist]

#Flip through bands and create figures	
for ax,band in zip(axes.flatten(),trange(len(iter_freqs), desc = 'bands')):
	#Query data
	query1 = grouped_df.query('Condition == @folder_check[0]')
	query2 = grouped_df.query('Condition == @folder_check[1]')
	
	band_filt_sig1 = butter_bandpass_filter(data = query1.iloc[:,4:], 
                                    lowcut = iter_freqs[band][1], 
                                    highcut =  iter_freqs[band][2], 
                                    fs = 1000)
	analytic_signal1 = hilbert(band_filt_sig1)
	instantaneous_phase1 = np.angle(analytic_signal1)
	x_power1 = np.abs(analytic_signal1)**2
	
	band_filt_sig2 = butter_bandpass_filter(data = query2.iloc[:,4:], 
                                    lowcut = iter_freqs[band][1], 
                                    highcut =  iter_freqs[band][2], 
                                    fs = 1000)
	analytic_signal2 = hilbert(band_filt_sig2)
	instantaneous_phase2 = np.angle(analytic_signal2)
	x_power2 = np.abs(analytic_signal2)**2
	
	#Plot raw versus filtered LFP
	ax = axes_list.pop(0)
	ax.plot(t,np.mean(query1.iloc[:,4:].T,axis=1),'k-',alpha=0.3,lw=1,label='_nolegend_'); 
	ax.plot(t,np.mean(analytic_signal1.T,axis=1),'r',alpha=0.8,lw=2);
	
	ax.plot(t,np.mean(query2.iloc[:,4:].T,axis=1),'k-',alpha=0.3,lw=1,label='_nolegend_'); 
	ax.plot(t,np.mean(analytic_signal2.T,axis=1),'r',alpha=0.4,lw=2);
	handles, labels = ax.get_legend_handles_labels()
	
	#Formatting
	ax.set_xlim([-int(plotting_params[2]),int(plotting_params[3])])
	ax.set_ylim([np.min(np.mean(query1.iloc[:,4:].T,axis=1)),100])
	ax.vlines(x=0, ymin=np.min(np.mean(query1.iloc[:,4:].T,axis=1)),ymax=100, linewidth=3, color='k',linestyle=':',label='_nolegend_')
	ax.text(0.83,0.9,'%s (%i - %iHz)' %(iter_freqs[band][0],iter_freqs[band][1],iter_freqs[band][2]), ha='center', va='center', transform=ax.transAxes)
	
	#Plot Power over time
	ax = axes_list.pop(0)
	ax.plot(t,np.mean(x_power1.T,axis=1),'b',alpha=0.8,lw=2); 
	ax.plot(t,np.mean(x_power2.T,axis=1),'b',alpha=0.4,lw=2); 
	
	#Formatting
	ax.set_xlim([-int(plotting_params[2]),int(plotting_params[3])])
	ax.vlines(x=0, ymin=np.min(np.mean(x_power1.T,axis=1)),ymax=np.max([np.mean(x_power1.T,axis=1),np.mean(x_power2.T,axis=1)]), linewidth=3, color='k',linestyle=':')
	
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=-0.08)
fig.legend( handles, labels=folder_check*2, loc='best', bbox_to_anchor=(0.9, 0.98), ncol=2 )
plt.suptitle('Hilbert Transform and Instantaneous Power \nAll Tastes',size=16)
fig.savefig(save_name+'/AllTastes_N_%i_Animals' %(len(grouped_df.Animal.unique())) + '_ComparedHilbert.png')   
plt.show()
plt.close(fig)	






























