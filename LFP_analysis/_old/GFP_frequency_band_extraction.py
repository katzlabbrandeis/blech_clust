#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 08:19:17 2018

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
from scipy.signal import hilbert #Hilbert transform to determine the amplitude envelope and instantaneous frequency of an amplitude-modulated signal
from scipy.signal import butter
from scipy.signal import filtfilt
from scipy.signal import argrelextrema #for peak amp coverage
import matplotlib.pyplot as plt # makes matplotlib work like MATLAB. ’pyplot’ functions.
import easygui
import tables
import pickle #for data storage and retreival

#Get name of directory where you want to save output files to
save_name = easygui.diropenbox(msg = 'Choose directory you want output files sent to (and/or where ".dir" files are)',default = '/home/bradly/drive2/data/Affective_State_Protocol/LiCl/Combined_Passive_Data/LiCl_Saline/freq_bands/LiCl')

#Ask user if they have ran set-up before
msg   = "Have you performed directory set-up before (ie. do you have '.dir' files in output folder) ?"
dir_check = easygui.buttonbox(msg,choices = ["Yes","No"])

if dir_check == "No":
    #Get data_saving name
    msg   = "What condition are you analyzing first?"
    subplot_check1 = easygui.buttonbox(msg,choices = ["Saline","LiCl","Other"])
    
    # Ask the user for the hdf5 files that need to be plotted together (fist condition)
    dirs_1 = []
    while True:
    	dir_name = easygui.diropenbox(msg = 'Choose first condition directory with a hdf5 file, hit cancel to stop choosing')
    	try:
    		if len(dir_name) > 0:	
    			dirs_1.append(dir_name)
    	except:
    		break   
    
    # Ask the user for the hdf5 files that need to be plotted together (second condition)
    #Get data_saving name
    msg   = "What condition are you analyzing second?"
    subplot_check2 = easygui.buttonbox(msg,choices = ["Saline","LiCl","Other"])
    
    dirs_2 = []
    while True:
    	dir_name = easygui.diropenbox(msg = 'Choose second condition directory with a hdf5 file, hit cancel to stop choosing')
    	try:
    		if len(dir_name) > 0:	
    			dirs_2.append(dir_name)
    	except:
    		break
    
    #Dump the directory names into chosen output location for each condition
    #condition 1
    completeName_1 = os.path.join(save_name, 'dirs_cond1.dir') 
    f_1 = open(completeName_1, 'w')
    for item in dirs_1:
        f_1.write("%s\n" % item)
    f_1.close()
    
    #condition 2
    completeName_2 = os.path.join(save_name, 'dirs_cond2.dir') 
    f_2 = open(completeName_2, 'w')
    for item in dirs_2:
        f_2.write("%s\n" % item)
    f_2.close()

if dir_check == "Yes":
    
    #Get data_saving name
    msg   = "What condition are you analyzing first?"
    subplot_check1 = easygui.buttonbox(msg,choices = ["Saline","LiCl","Other"])
    
    #establish directories to flip through
    #condition 1
    dirs_1_path = os.path.join(save_name, 'dirs_cond1.dir')
    dirs_1_file = open(dirs_1_path,'r')
    dirs_1 = dirs_1_file.read().splitlines()
    dirs_1_file.close()
    
    #Get data_saving name
    msg   = "What condition are you analyzing second?"
    subplot_check2 = easygui.buttonbox(msg,choices = ["Saline","LiCl","Other"])
    
    #condition 2
    dirs_2_path = os.path.join(save_name, 'dirs_cond2.dir')
    dirs_2_file = open(dirs_2_path,'r')
    dirs_2 = dirs_2_file.read().splitlines()
    dirs_2_file.close()

#modify save names
if subplot_check1==subplot_check2:
    subplot_check2=subplot_check2+'_2'    

#Define functions used in loop
#define bandpass filter parameters to parse out frequencies
def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut /nyq
    high = highcut/nyq
    b, a = butter(order, [low, high], btype='bandpass')
    y = filtfilt(b, a, data)
    return y

#define joining parameters
def inner_join(a, b):
    L = a + b
    L.sort(key=itemgetter(0)) # sort by the first column
    for _, group in groupby(L, itemgetter(0)):
        row_a, row_b = next(group), next(group, None)
        if row_b is not None: # join
            yield row_a + row_b[1:] # cut 1st column from 2nd row
            
#Get name of directory where the data files and hdf5 file sits, and change to that directory for processing
#dir_name = easygui.diropenbox()
#os.chdir(dir_name)

combined_tuples1 = list(); combined_tuples2 = list()
            
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
    
    #Establish child dig_in number/name for LFP data
	dig_name = hf5.root.Parsed_LFP._f_iter_nodes()
	this_dig_in = next(dig_name)
	lfp_signal = np.squeeze(this_dig_in[:],axis=1) # The signal itself. This is a dataframe of electrodes x shape times

    #Close the hdf5 file
	hf5.close()
    
    #processing variables
	sfreq=1000 # Sampling frequency
	tmin = 0  #In seconds
	tmax = 1200 #In seconds
	samples = int(sfreq*np.size(lfp_signal,axis=1)/1000)
	times = np.arange(samples) / sfreq
    
    #frequency bands
	iter_freqs = [
        ('Theta', 4, 7),
        ('Alpha', 8, 12),
        ('Beta', 13, 25),
        ('Gamma', 30, 45)]
    
    #detrend the signal to remove linear trend
	#lfp_signal_detrend = signal.detrend(lfp_signal)    
	lfp_signal_detrend = lfp_signal
    
    #create list to store outputs into
	frequency_map = list(); envelope_map = list();  extrema_map = list()
    
    #run through frequency bands and compile list using hilbert transform
	for band, fmin, fmax in iter_freqs:
        
        #apply filter to compute specified bands 
		band_filt_sig = butter_bandpass_filter(np.mean(lfp_signal_detrend,axis=0), fmin, fmax, 1000)
         
        # bandpass filter and compute Hilbert
		analytic_signal = hilbert(band_filt_sig) 
        
        #complex vector created, so must perform list creation to extract only real parts of number
		analytic_signal = np.asarray([num.real for num in analytic_signal])
		amplitude_envelope = np.abs(analytic_signal)
		extrema = argrelextrema(amplitude_envelope, np.greater)
		instantaneous_phase = np.unwrap(np.angle(analytic_signal)) #first (angle) in radians, then unwrap radian phase
		instantaneous_frequency = (np.diff(instantaneous_phase) /(2.0*np.pi) * sfreq)
    
        #store and move on
		frequency_map.append(((band, fmin, fmax), analytic_signal))    
		envelope_map.append(((band, fmin, fmax), amplitude_envelope)) 
		extrema_map.append(((band, fmin, fmax), np.squeeze(np.asarray(extrema),axis=0))) 
    
    #plot the results (by frequency band and a 2.0s time window for visualization)
	fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True, sharey=True)
	colors = plt.get_cmap('winter_r')(np.linspace(0, 1, 4))
	for ((freq_name, fmin, fmax), average), color, ax in zip(
            frequency_map, colors, axes.ravel()[::-1]):
    
		gfp = average
		ax.plot(times, gfp, label=freq_name, color=color, linewidth=2.5)
		ax.axhline(0, linestyle='--', color='grey', linewidth=2)
		ax.set_ylabel('GFP')
		ax.annotate('%s (%d-%dHz)' % (freq_name, fmin, fmax),
                    xy=(0.95, 0.8),
                    horizontalalignment='right',
                    xycoords='axes fraction')
		ax.set_xlim(0, 2)
		ax.set_ylim(-60,60)
    
	axes.ravel()[-1].set_xlabel('Time [s]')
	plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.00)
	plt.suptitle('Global Field Potential by Frequency Band: 2sec',size=20)
	file_name = hdf5_name[:4]+ '_' + subplot_check1 + '_smaller_passive_bands.png'
	fig_save = os.path.join(save_name, file_name)
	fig.savefig(fig_save)
	plt.close()
    
    #plot the results (full 20-minutes)
	fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True, sharey=True)
	colors = plt.get_cmap('winter_r')(np.linspace(0, 1, 4))
	for ((freq_name, fmin, fmax), average), color, ax in zip(
            frequency_map, colors, axes.ravel()[::-1]):
        
		gfp = average
		ax.plot(times, gfp, label=freq_name, color=color, linewidth=2.5)
		ax.axhline(0, linestyle='--', color='grey', linewidth=2)
		ax.set_ylabel('GFP')
		ax.annotate('%s (%d-%dHz)' % (freq_name, fmin, fmax),
                    xy=(0.95, 0.8),
                    horizontalalignment='right',
                    xycoords='axes fraction')
		ax.set_xlim(0, 1200)
        #ax.set_ylim(-50,40)
    
	axes.ravel()[-1].set_xlabel('Time [s]')
	plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.00)
	plt.suptitle('Global Field Potential by Frequency Band: Extended',size=20)
	file_name = hdf5_name[:4]+ '_' + subplot_check1 +'_full_passive_bands.png'
	fig_save = os.path.join(save_name, file_name)
	fig.savefig(fig_save)
	plt.close()
    
	fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True, sharey=True)
	colors = plt.get_cmap('winter_r')(np.linspace(0, 1, 4))
	for ((freq_name, fmin, fmax), average), color, ax in zip(
            envelope_map, colors, axes.ravel()[::-1]):
    
		gfp = average
		ax.plot(times, gfp, label=freq_name, color=color, linewidth=2.5)
		ax.axhline(0, linestyle='--', color='grey', linewidth=2)
		ax.set_ylabel('GFP')
		ax.annotate('%s (%d-%dHz)' % (freq_name, fmin, fmax),
                    xy=(0.95, 0.8),
                    horizontalalignment='right',
                    xycoords='axes fraction')
		ax.set_xlim(0, 1)
		ax.set_ylim(-1,40)
    
	axes.ravel()[-1].set_xlabel('Time [s]')
       
    #combine all items of the 3 lists based on key ids
	result = list(inner_join(frequency_map, envelope_map));result2= list(inner_join(result, extrema_map))
    
    # take second element for sorting list based on frequency
	def takeSecond(elem):
		return elem[:][0][1]
    
    # sort list with key based on the frequency
	result2.sort(key=takeSecond)
    
	fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True, sharey=True)
	colors = plt.get_cmap('winter_r')(np.linspace(0, 1, 4))
	for ((freq_name, fmin, fmax), average1,average2,average3), color, ax in zip(
            result2, colors, axes.ravel()[::-1]):
        
		hilb_signal = average1; env_signal = average2; maxpeak_signal = average3
        
		ax.plot(times[maxpeak_signal], np.abs(hilb_signal[maxpeak_signal]), label=freq_name, color=color, linewidth=2.5)
		ax.axhline(0, linestyle='--', color='grey', linewidth=2)
		ax.set_ylabel('GFP')
		ax.annotate('%s (%d-%dHz)' % (freq_name, fmin, fmax),
                    xy=(0.95, 0.8),
                    horizontalalignment='right',
                    xycoords='axes fraction')
		ax.set_xlim(0, 1200)
        #ax.set_ylim(-1,40)
    
	axes.ravel()[-1].set_xlabel('Time [s]')
    
    
	fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True, sharey=True)
	colors = plt.get_cmap('winter_r')(np.linspace(0, 1, 4))
	for ((freq_name, fmin, fmax), average1,average2,average3), color, ax in zip(
            result2, colors, axes.ravel()[::-1]):
    
		hilb_signal = average1; env_signal = average2; maxpeak_signal = average3
         
        #apply a Savitzky-Golay Filter to smooth/eliminate noise from dataset
		power_smoothed = signal.savgol_filter(np.abs(hilb_signal[maxpeak_signal]), 351, 1)
         
		ax.plot(times[maxpeak_signal], power_smoothed, label=freq_name, color=color, linewidth=2.5)
		ax.axhline(np.mean(power_smoothed), linestyle='--', color='grey', linewidth=2)
		ax.set_ylabel('GFP')
		ax.annotate('%s (%d-%dHz)' % (freq_name, fmin, fmax),
                    xy=(0.95, 0.8),
                    horizontalalignment='right',
                    xycoords='axes fraction')
		ax.set_xlim(0, 1200)
		#ax.set_ylim(-3,40)
    
	axes.ravel()[-1].set_xlabel('Time [s]')
	plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.00)
	plt.suptitle('Global Field Potential by Frequency Band: Savitzky-Golay Enveloped',size=20)
	file_name = hdf5_name[:4]+ '_' + subplot_check1 + '_full_passive_eveloped.png'
	fig_save = os.path.join(save_name, file_name)
	fig.savefig(fig_save)
	plt.close()
    
    #store animal's output into group list
	combined_tuples1.append((hdf5_name[:4], result2)) 
    
#Save output to .dump file for easy access plotting later
tuple_save = subplot_check1 + '_tuple.dump'
output_name =   os.path.join(save_name, tuple_save)
pickle.dump(combined_tuples1, open(output_name, 'wb'))  

#flip through next condition
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
    
    #Establish child dig_in number/name for LFP data
	dig_name = hf5.root.Parsed_LFP._f_iter_nodes()
	this_dig_in = next(dig_name)
	lfp_signal = np.squeeze(this_dig_in[:],axis=1) # The signal itself. This is a dataframe of electrodes x shape times

    #Close the hdf5 file
	hf5.close()
    
    #processing variables
	sfreq=1000 # Sampling frequency
	tmin = 0  #In seconds
	tmax = 1200 #In seconds
	samples = int(sfreq*np.size(lfp_signal,axis=1)/1000)
	times = np.arange(samples) / sfreq
    
    #frequency bands
	iter_freqs = [
        ('Theta', 4, 7),
        ('Alpha', 8, 12),
        ('Beta', 13, 25),
        ('Gamma', 30, 45)]
    
    #detrend the signal to remove linear trend
	lfp_signal_detrend = signal.detrend(lfp_signal)    
    
    #create list to store outputs into
	frequency_map = list(); envelope_map = list();  extrema_map = list()
    
    #run through frequency bands and compile list using hilbert transform
	for band, fmin, fmax in iter_freqs:
        
        #apply filter to compute specified bands 
		band_filt_sig = butter_bandpass_filter(np.mean(lfp_signal_detrend,axis=0), fmin, fmax, 1000)
         
        # bandpass filter and compute Hilbert
		analytic_signal = hilbert(band_filt_sig) 
        
        #complex vector created, so must perform list creation to extract only real parts of number
		analytic_signal = np.asarray([num.real for num in analytic_signal])
		amplitude_envelope = np.abs(analytic_signal)
		extrema = argrelextrema(amplitude_envelope, np.greater)
		instantaneous_phase = np.unwrap(np.angle(analytic_signal)) #first (angle) in radians, then unwrap radian phase
		instantaneous_frequency = (np.diff(instantaneous_phase) /(2.0*np.pi) * sfreq)
    
        #store and move on
		frequency_map.append(((band, fmin, fmax), analytic_signal))    
		envelope_map.append(((band, fmin, fmax), amplitude_envelope)) 
		extrema_map.append(((band, fmin, fmax), np.squeeze(np.asarray(extrema),axis=0))) 
    
    #plot the results (by frequency band and a 2.0s time window for visualization)
	fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True, sharey=True)
	colors = plt.get_cmap('winter_r')(np.linspace(0, 1, 4))
	for ((freq_name, fmin, fmax), average), color, ax in zip(
            frequency_map, colors, axes.ravel()[::-1]):
    
		gfp = average
		ax.plot(times, gfp, label=freq_name, color=color, linewidth=2.5)
		ax.axhline(0, linestyle='--', color='grey', linewidth=2)
		ax.set_ylabel('GFP')
		ax.annotate('%s (%d-%dHz)' % (freq_name, fmin, fmax),
                    xy=(0.95, 0.8),
                    horizontalalignment='right',
                    xycoords='axes fraction')
		ax.set_xlim(0, 2)
		ax.set_ylim(-60,60)
    
	axes.ravel()[-1].set_xlabel('Time [s]')
	plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.00)
	plt.suptitle('Global Field Potential by Frequency Band: 2sec',size=20)
	file_name = hdf5_name[:4]+ '_' + subplot_check2 + '_smaller_passive_bands.png'
	fig_save = os.path.join(save_name, file_name)
	fig.savefig(fig_save)
	plt.close()
    
    #plot the results (full 20-minutes)
	fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True, sharey=True)
	colors = plt.get_cmap('winter_r')(np.linspace(0, 1, 4))
	for ((freq_name, fmin, fmax), average), color, ax in zip(
            frequency_map, colors, axes.ravel()[::-1]):
        
		gfp = average
		ax.plot(times, gfp, label=freq_name, color=color, linewidth=2.5)
		ax.axhline(0, linestyle='--', color='grey', linewidth=2)
		ax.set_ylabel('GFP')
		ax.annotate('%s (%d-%dHz)' % (freq_name, fmin, fmax),
                    xy=(0.95, 0.8),
                    horizontalalignment='right',
                    xycoords='axes fraction')
		ax.set_xlim(0, 1200)
        #ax.set_ylim(-50,40)
    
	axes.ravel()[-1].set_xlabel('Time [s]')
	plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.00)
	plt.suptitle('Global Field Potential by Frequency Band: Extended',size=20)
	file_name = hdf5_name[:4]+ '_' + subplot_check2 +'_full_passive_bands.png'
	fig_save = os.path.join(save_name, file_name)
	fig.savefig(fig_save)
	plt.close()
    
	fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True, sharey=True)
	colors = plt.get_cmap('winter_r')(np.linspace(0, 1, 4))
	for ((freq_name, fmin, fmax), average), color, ax in zip(
            envelope_map, colors, axes.ravel()[::-1]):
    
		gfp = average
		ax.plot(times, gfp, label=freq_name, color=color, linewidth=2.5)
		ax.axhline(0, linestyle='--', color='grey', linewidth=2)
		ax.set_ylabel('GFP')
		ax.annotate('%s (%d-%dHz)' % (freq_name, fmin, fmax),
                    xy=(0.95, 0.8),
                    horizontalalignment='right',
                    xycoords='axes fraction')
		ax.set_xlim(0, 1)
		ax.set_ylim(-1,40)
    
	axes.ravel()[-1].set_xlabel('Time [s]')
       
    #combine all items of the 3 lists based on key ids
	result = list(inner_join(frequency_map, envelope_map));result2= list(inner_join(result, extrema_map))
    
    # take second element for sorting list based on frequency
	def takeSecond(elem):
		return elem[:][0][1]
    
    # sort list with key based on the frequency
	result2.sort(key=takeSecond)
    
	fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True, sharey=True)
	colors = plt.get_cmap('winter_r')(np.linspace(0, 1, 4))
	for ((freq_name, fmin, fmax), average1,average2,average3), color, ax in zip(
            result2, colors, axes.ravel()[::-1]):
        
		hilb_signal = average1; env_signal = average2; maxpeak_signal = average3
        
		ax.plot(times[maxpeak_signal], np.abs(hilb_signal[maxpeak_signal]), label=freq_name, color=color, linewidth=2.5)
		ax.axhline(0, linestyle='--', color='grey', linewidth=2)
		ax.set_ylabel('GFP')
		ax.annotate('%s (%d-%dHz)' % (freq_name, fmin, fmax),
                    xy=(0.95, 0.8),
                    horizontalalignment='right',
                    xycoords='axes fraction')
		ax.set_xlim(0, 1200)
        #ax.set_ylim(-1,40)
    
	axes.ravel()[-1].set_xlabel('Time [s]')
    
    
	fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True, sharey=True)
	colors = plt.get_cmap('winter_r')(np.linspace(0, 1, 4))
	for ((freq_name, fmin, fmax), average1,average2,average3), color, ax in zip(
            result2, colors, axes.ravel()[::-1]):
    
		hilb_signal = average1; env_signal = average2; maxpeak_signal = average3
         
        #apply a Savitzky-Golay Filter to smooth/eliminate noise from dataset
		power_smoothed = signal.savgol_filter(np.abs(hilb_signal[maxpeak_signal]), 351, 1)
         
		ax.plot(times[maxpeak_signal], power_smoothed, label=freq_name, color=color, linewidth=2.5)
		ax.axhline(np.mean(power_smoothed), linestyle='--', color='grey', linewidth=2)
		ax.set_ylabel('GFP')
		ax.annotate('%s (%d-%dHz)' % (freq_name, fmin, fmax),
                    xy=(0.95, 0.8),
                    horizontalalignment='right',
                    xycoords='axes fraction')
		ax.set_xlim(0, 1200)
		#ax.set_ylim(-3,40)
    
	axes.ravel()[-1].set_xlabel('Time [s]')
	plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.00)
	plt.suptitle('Global Field Potential by Frequency Band: Savitzky-Golay Enveloped',size=20)
	file_name = hdf5_name[:4]+ '_' + subplot_check2 + '_full_passive_eveloped.png'
	fig_save = os.path.join(save_name, file_name)
	fig.savefig(fig_save)
	plt.close()
    
    #store animal's output into group list
	combined_tuples2.append((hdf5_name[:4], result2)) 
    
#Save output to .dump file for easy access plotting later
tuple_save = subplot_check2 + '_tuple.dump'
output_name =   os.path.join(save_name, tuple_save)
pickle.dump(combined_tuples2, open(output_name, 'wb'))  