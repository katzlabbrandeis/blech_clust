#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:45:21 2019

@author: bradly
"""
# =============================================================================
# #Import Stuff
# =============================================================================

# Import stuff!
import numpy as np
import tables
import easygui
import sys
import os
import pandas as pd
from scipy import stats
from tqdm import trange

#Import specific functions in order to filter the data file
from scipy.signal import hilbert
from scipy.signal import butter
from scipy.signal import filtfilt

#Parallelizing
from joblib import Parallel, delayed

#Plotting imports
import matplotlib.pyplot as plt # makes matplotlib work like MATLAB. ’pyplot’ functions.

# =============================================================================
# #Define Functions
# =============================================================================

#define bandpass filter parameters to parse out frequencies
def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut /nyq
    high = highcut/nyq
    b, a = butter(order, [low, high], btype='bandpass')
    y = filtfilt(b, a, data)
    return y

def butter_bandpass_filter_parallel(data, iter_freqs, channel, band):
	band_filt_sig = butter_bandpass_filter(data = data[channel,:],
                                    lowcut = iter_freqs[band][1],
                                    highcut =  iter_freqs[band][2],
                                    fs = 1000)
	analytic_signal = hilbert(band_filt_sig)
	x_power = np.abs(analytic_signal)**2

	return band_filt_sig, analytic_signal, x_power

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

#Import necessary data
try:
	spike_times = np.array(hf5.list_nodes('/Whole_session_spikes')[0]) #by unit
	delivery_times = pd.read_hdf(hdf5_name,'/Whole_session_spikes/delivery_times','r+') #ordered by dig_in
	LFP_times = np.array(hf5.list_nodes('/Whole_session_raw_LFP')[0]) #by unit (time = ms)

except:
	print('Bruh, you need to set up your data file!')
	sys.exit()

# =============================================================================
# #Set parameters
# =============================================================================
Boxes = [[] for x in range(0,delivery_times.shape[0]+1)]
Boxes = ["Taste:" for x in Boxes]
Boxes[0] = 'Sampling Rate'

Def_labels = [30000,'NaCl','Sucrose','Citric Acid','QHCl']
Defaults = [[] for x in range(0,delivery_times.shape[0]+1)]
for x in range(len(Defaults)):
	Defaults[x] = Def_labels[x]

freqparam = easygui.multenterbox(
    'Specify data collection sampling rate and tastes delivered',
    'Setting Analysis Parameters',
    Boxes, Defaults)

#specify frequency bands
iter_freqs = [('Delta',1,3),
        ('Theta', 4, 7),
        ('Mu', 8, 12),
        ('Beta', 13, 25),
        ('Gamma', 30, 45)]

if "/Whole_session_raw_LFP/WS_LFP_filtered" in hf5:
	print('You already have done the filtering')
else:
	#Extract filtered LFPs
	all_output = []
	for band in trange(len(iter_freqs), desc = 'bands'):
		output = Parallel(n_jobs=6)(delayed(butter_bandpass_filter_parallel)\
				 (LFP_times, iter_freqs, channel, band) for channel in range(len(LFP_times)))

		all_output.append(output)

	#Store data	(channel X LFP(ms))
	hf5.create_array('/Whole_session_raw_LFP','WS_LFP_filtered',np.asarray(all_output))
	hf5.flush()

#Dynamically determine ITI and cut data accordingly
ITI = np.array(np.diff(np.sort(np.array(delivery_times),axis=None))//int(freqparam[0]))

if len(np.unique(ITI)) == 1:
	ITI = ITI[0]
else:
	ITI = ITI

ITIparam = easygui.multenterbox(
    'Specify how many seconds you want to skip extract POST-taste delivery',
    'Setting data slicing Parameters',
    ['Post Delivery time (s):','Slicing time (s):'], [8,10])

data_skip = int(ITIparam[0])*1000
slice_ext = int(ITIparam[1])*1000


#Extract taset delivery times
taste_delivery = np.sort(np.array(delivery_times),axis=None)

LFP_data_sliced = []; spike_data_sliced = []

Hilbert_all = np.asarray(all_output)[:,:,1,:]
for i in range(len(taste_delivery)):

	#Set slicing variables
	taste_start = int((taste_delivery[i]/int(freqparam[0]))*1000)
	slice_start = taste_start+data_skip

	spike_slice_start = taste_delivery[i]+data_skip*(int(freqparam[0])/1000)

	#Process LFP data
	LFP_data_sliced.append(Hilbert_all[:,:,slice_start:slice_start+slice_ext])

	#Process spike data
	spike_data_sliced.append(spike_times[:,np.array((spike_times>= spike_slice_start) & \
				   (spike_times <= spike_slice_start+slice_ext*(int(freqparam[0])/1000)))[1,:]])


#Stack LFP data cell-wise (dimensions: bandX trials X channel X duration (ms))
LFP_data_sliced = np.swapaxes(np.array(LFP_data_sliced),0,1)

plt.imshow(LFP_data_sliced[0,:,0,:].real,interpolation='nearest',aspect='auto')




fig = plt.figure(figsize=(20,10))
for i in range(len(spike_data_sliced)):
	plt.scatter((spike_data_sliced[i][1]-(taste_delivery[0]+\
				 data_skip*(int(freqparam[0])/1000)))/int(freqparam[0]), \
					spike_data_sliced[i][0], s=0.5,color='black')

plt.xlim(0,spike_data_sliced[-1][1][-1]/int(freqparam[0]))
plt.yticks(size=18)
plt.ylabel('Cell',fontweight='bold',size=20)
plt.xlabel('Time from %i (sec) Post-First Taste Delivery'\
		    %(data_skip/1000),fontweight='bold',size=20)
plt.title('Animal: %s \nFull Taste Session \nDate: %s' \
		  %(hdf5_name[:4],hdf5_name.split('_')[2].split('_')[0]), \
		  fontweight='bold',size=24)
plt.tight_layout()


hf5.close()
