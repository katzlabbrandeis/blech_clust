#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 08:59:35 2018

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

#establish which dump files to choose from
dir_name_1 = easygui.diropenbox(msg = 'Choose first condition directory with a ".dump" file.')
dir_name_2 = easygui.diropenbox(msg = 'Choose second condition directory with a ".dump" file.')

#set conditoni variables
cond1 = dir_name_1.split("/")[-1]
cond2 = dir_name_2.split("/")[-1]

#load tuples
#Change to the directory
os.chdir(dir_name_1)
#Locate the hdf5 file
file_list = os.listdir('./')
for files in file_list:
	if files[-4:] == 'dump':
		tup_name = files
dir_1_tup = pickle.load(open(tup_name, 'rb'))

#Change to the directory
os.chdir(dir_name_2)
#Locate the hdf5 file
file_list = os.listdir('./')
for files in file_list:
	if files[-4:] == 'dump':
		tup_name = files
dir_2_tup = pickle.load(open(tup_name, 'rb'))

#establish variables
sfreq=1000 # Sampling frequency
passive_time = 1200000 #milliseconds
samples = passive_time
times = np.arange(samples)
#times = np.arange(1200*1000) #put into milliseconds

#define joining parameters
def inner_join(a, b):
    L = a + b
    L.sort(key=itemgetter(0)) # sort by the first column
    for _, group in groupby(L, itemgetter(0)):
        row_a, row_b = next(group), next(group, None)
        if row_b is not None: # join
            yield row_a + row_b[1:] # cut 1st column from 2nd row

#combine all items of the 3 lists based on key ids
result = list(inner_join(dir_1_tup, dir_2_tup));

## take second element for sorting list based on frequency
#def takeSecond(elem):
#	return elem[:][0][1]
#    
##sort list with key based on the frequency
#result2.sort(key=takeSecond)

##plot the results (by frequency band and a 2.0s time window for visualization)
#fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True, sharey=True)
#colors = plt.get_cmap('winter_r')(np.linspace(0, 1, 4))
#for ((freq_name, fmin, fmax), average1,average2,average3), color, ax in zip(dir_1_tup, colors, axes.ravel()[::-1]):
#    
#    #assign vector variables
#    hilb_signal = average1; env_signal = average2; maxpeak_signal = average3
#    
#    #apply a Savitzky-Golay Filter to smooth/eliminate noise from dataset
#    power_smoothed = signal.savgol_filter(np.abs(hilb_signal[maxpeak_signal]), 351, 1)
#         
#    ax.plot(times[maxpeak_signal], power_smoothed, label=freq_name, color=color, linewidth=2.5)
#    ax.axhline(0, linestyle='--', color='grey', linewidth=2)
#    ax.set_ylabel('GFP')
#    ax.annotate('%s (%d-%dHz)' % (freq_name, fmin, fmax),
#                xy=(0.95, 0.8),
#                horizontalalignment='right',
#                xycoords='axes fraction')
#    #ax.set_xlim(0, 2)
#    #ax.set_ylim(-50,40)
#
#axes.ravel()[-1].set_xlabel('Time [s]')
#plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.00)
#plt.suptitle('Global Field Potential by Frequency Band: 2sec',size=20)
#file_name = hdf5_name[:4]+ '_' + subplot_check + '_smaller_passive_bands.png'
#fig_save = os.path.join(save_name, file_name)
#fig.savefig(fig_save)
#plt.close()

fig_labels = [cond1,cond2]

for animal in range(len(result)):
    print('Working on...'+result[int(animal)][0])
    trimmed_tuple = result[animal]
    fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True, sharey=True)
    for condition in range(1,len(trimmed_tuple)):
        condition_tuple = trimmed_tuple[condition]
        #fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True, sharey=True)
        colors = plt.get_cmap('winter_r')(np.linspace(0, 1, 4))
        if condition==1:
            colors = colors
        else:
            colors = colors*.5
            
        for ((freq_name, fmin, fmax), average1,average2,average3), color, ax in zip(condition_tuple, colors, axes.ravel()[::-1]):
            
            #assign vector variables
            hilb_signal = average1; env_signal = average2; maxpeak_signal = average3
    
            #apply a Savitzky-Golay Filter to smooth/eliminate noise from dataset
            power_smoothed = signal.savgol_filter(np.abs(hilb_signal[maxpeak_signal]), 351, 1)
         
            ax.plot(times[maxpeak_signal], power_smoothed, label=freq_name, color=color, linewidth=1.5)
            #ax.axhline(np.mean(power_smoothed), linestyle='--', color='grey', linewidth=2)
            ax.set_ylabel('GFP')
            ax.annotate('%s (%d-%dHz)' % (freq_name, fmin, fmax),
                xy=(0.95, 0.8),
                horizontalalignment='right',
                xycoords='axes fraction')
            ax.set_ylim(0,200)
            ax.set_xlim(0, 1200*1000)
            
            if condition==2:
                ax.legend(labels=fig_labels,ncol=2,loc=2)
        ticks = ax.get_xticks()/1000
        ax.set_xticklabels(ticks)
        axes.ravel()[-1].set_xlabel('Time [s]')
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.00)
        plt.suptitle('Global Field Potential by Frequency Band: Savitzky-Golay Enveloped \n%s' %(result[int(animal)][0]),size=20)
    
    #save file
    file_name = result[int(animal)][0] + '_Savitzky-Golay_envelope_' + cond1 +'_' + cond2 + '.png'
    fig_save = os.path.join(save_name, file_name)
    fig.savefig(fig_save)
    
    #plt.legend(labels=fig_labels)
        
#        
#        
#        print('Working on...'+result[int(animal)][0])
#        for data_Set_2 in range(len(result[animal][data_set][0])):
#            print(data_Set)
#








