#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 11:27:18 2019

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


#define bandpass filter parameters to parse out frequencies
def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut /nyq
    high = highcut/nyq
    b, a = butter(order, [low, high], btype='bandpass')
    y = filtfilt(b, a, data)
    return y

#Get name of directory where the data files and hdf5 file sits, and change to that directory for processing
dir_name = easygui.diropenbox()
os.chdir(dir_name)

#Look for the hdf5 file in the directory
file_list = os.listdir('./')
hdf5_name = ''
for files in file_list:
        if files[-2:] == 'h5':
                hdf5_name = files

#Open the hdf5 file and create list of child paths
hf5 = tables.open_file(hdf5_name, 'r+')

# Get the list of spike trains and LFPs by digital input channels
trains_dig_in = hf5.list_nodes('/spike_trains')
all_spikes = np.squeeze(np.asarray([spikes.spike_array[:] for spikes in trains_dig_in])[:,0,:,:],axis=0)
dig_in_LFP_nodes = hf5.list_nodes('/Parsed_LFP')
LFP_data = np.array(dig_in_LFP_nodes[0][:]) 

#Get/Set parameters
analysis_params = easygui.multenterbox(
		msg = 'Input analysis paramters:', 
		fields = ['Pre-stimulus signal duration (ms; from set-up)',
                    'Post-stimulus signal duration (ms; from set-up)'], 
					values = ['0','1200000'])
    
#create timing variables
pre_stim = int(analysis_params[0])
post_stim = int(analysis_params[1])
time = np.linspace(pre_stim,post_stim,post_stim)

#specify frequency bands
iter_freqs = [('Delta',1,3),
        ('Theta', 4, 7),
        ('Mu', 8, 12),
        ('Beta', 13, 25),
        ('Gamma', 30, 45)]

#Extract LFPs
filt_sig = []; sig_power = [];
for band in trange(len(iter_freqs), desc = 'bands'):
	band_filt_sig = butter_bandpass_filter(data = np.squeeze(LFP_data,axis=1), 
                                    lowcut = iter_freqs[band][1], 
                                    highcut =  iter_freqs[band][2], 
                                    fs = 1000)
	analytic_signal = hilbert(band_filt_sig)
	instantaneous_phase1 = np.angle(analytic_signal)
	x_power = np.abs(analytic_signal)**2
	filt_sig.append(analytic_signal)
	sig_power.append(x_power)

#Plotting	
xmin = 0; xmax = 1200000		

#Sort cells based on spike count
sorted_spikes = all_spikes[np.argsort(all_spikes.sum(axis=1))]

#Identify spike locations
sorted_spikes_ID = np.where(sorted_spikes)

#Initiate figure
fig, axs = plt.subplots(2, sharex=True, sharey=False,\
						figsize=(12, 8), gridspec_kw={'hspace': 0})

#Plot LFP Power
axs[0].plot(time[xmin:xmax],np.array(np.mean(sig_power[1],axis=0))[xmin:xmax],'midnightblue'); 

#Formatting
axs[0].set_ylabel('Power ($mV^2/Hz$)',fontweight='bold',size=14)
axs[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

#Plot raster for all cells sorted
axs[1].scatter(sorted_spikes_ID[1],sorted_spikes_ID[0], s=0.5,color='black');plt.xlim(xmin,xmax)

#Formatting
axs[1].set_ylabel('Cell',fontweight='bold',size=14)
axs[1].set_xlim(xmin,xmax)

axs[1].set_xlabel('Time Post-Injection (ms)',fontweight='bold',size=18)


# =============================================================================
# #Smaller window
# =============================================================================

#Window setting
xmin = 900000; xmax = 930000	

#Initiate figure
fig, axs = plt.subplots(2, sharex=True, sharey=False,\
						figsize=(12, 8), gridspec_kw={'hspace': 0})

#Plot LFP Power
axs[0].plot(time[xmin:xmax],np.array(np.mean(sig_power[1],axis=0))[xmin:xmax],'midnightblue'); 

#Formatting
axs[0].set_ylabel('Power ($mV^2/Hz$)',fontweight='bold',size=14)
axs[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

#Plot raster for all cells sorted
axs[1].scatter(sorted_spikes_ID[1],sorted_spikes_ID[0], s=0.5,color='black');plt.xlim(xmin,xmax)

#Formatting
axs[1].set_ylabel('Cell',fontweight='bold',size=14)
axs[1].set_xlim(xmin,xmax)

axs[1].set_xlabel('Time Post-Injection (ms)',fontweight='bold',size=18)






df = pd.DataFrame(data=raster_arr, index=raster_arr_df.index, columns=raster_arr_df.columns)






















r=[]
for i in range(0,sorted_spikes.shape[0]):
    #r[k] = np.corrcoef(sorted_spikes[i,:], brad)[0,1]
    r.append(np.corrcoef(sorted_spikes[i,:], np.array(np.mean(sig_power[1],axis=0)))[0,1])


r=[]
for i in range(0,sorted_spikes.shape[0]):
    #r[k] = np.corrcoef(sorted_spikes[i,:], brad)[0,1]
    r.append(np.corrcoef(sorted_spikes[i,:], np.array(np.mean(sig_power[1],axis=0)))[0,1])


# =============================================================================
# 
# =============================================================================
# fig, axes = plt.subplots(nrows=2, ncols=1,sharex=True, sharey=False,
# 	 figsize=(12, 8))
# 
# axes[0, 0].plot(time[xmin:],np.array(np.mean(sig_power[1],axis=0))[xmin:],'midnightblue')
# axes[1, 0].scatter(sorted_spikes_ID[1],sorted_spikes_ID[0],s=.5);
# 
# =============================================================================
# 		
# fig= plt.figure(figsize=(12, 8))
# fig.add_subplot(211)
# plt.plot(time[xmin:],np.array(np.mean(sig_power[1],axis=0))[xmin:],'midnightblue')
# plt.plot(time[xmin:],np.array(np.mean(sig_power[1],axis=0))[xmin:],'darkred')
# plt.xlim(xmin,xmax)
# plt.ylabel('Power')
# 
# #axes_list = [item for sublist in axes for item in sublist]
# for unit in range(all_spikes.shape[0]):
# 	#ax = axes_list.pop(0)
# 	x = np.where(all_spikes[unit,:] > 0.0)[0]
# 	fig.add_subplot(212)
# 	plt.vlines(x, unit, unit + 1, colors = 'black',alpha=0.3)
# 	plt.xlim(xmin,xmax)
# 	plt.ylabel('Cell #')
# 
# plt.xlabel('Time Post-Injection (ms)')
# 
# 
# #Plotting	
# xmin = 900000; xmax = 930000		
# 		
# fig= plt.figure(figsize=(12, 8))
# fig.add_subplot(211)
# plt.plot(time[xmin:],np.array(np.mean(sig_power[1],axis=0))[xmin:],'midnightblue')
# #plt.plot(time[xmin:],np.array(filt_sig[1][0])[xmin:],'darkred')
# plt.xlim(xmin,xmax)
# plt.ylabel('Power')
# 
# #axes_list = [item for sublist in axes for item in sublist]
# for unit in range(all_spikes.shape[0]):
# 	#ax = axes_list.pop(0)
# 	x = np.where(all_spikes[unit,:] > 0.0)[0]
# 	fig.add_subplot(212)
# 	plt.vlines(x, unit, unit + 1, colors = 'black',alpha=0.3)
# 	plt.xlim(xmin,xmax)
# 	plt.ylabel('Cell #')
# 
# plt.xlabel('Time Post-Injection (ms)')
# 
# 
# 
# 
# 
# 
# 
# 
# 
# fig=plt.figure(figsize=(12,8));plt.scatter(sorted_spikes_ID[1],sorted_spikes_ID[0],s=.5); plt.xlim(10000,100500)
# 
# 
# =============================================================================


spikes_transformed = np.where(all_spikes)
fig=plt.figure(figsize=(12,8));plt.scatter(spikes_transformed[1],spikes_transformed[0],s=.5); plt.xlim(0,100000)



brad2 = np.where(brad)
fig=plt.figure(figsize=(12,8));plt.scatter(brad2[1],brad2[0],s=.5); plt.xlim(0,100000)


