#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 10:23:58 2019

@author: abuzarmahmood/bradly

Created on Wed Feb 13 19:36:13 2019

"""

# ____       _               
#/ ___|  ___| |_ _   _ _ __  
#\___ \ / _ \ __| | | | '_ \ 
# ___) |  __/ |_| |_| | |_) |
#|____/ \___|\__|\__,_| .__/ 
#                     |_|    

# =============================================================================
# Import stuff
# =============================================================================

# import Libraries
# Built-in Python libraries
import os # functions for interacting w operating system
import sys
import glob

# 3rd-party libraries
import numpy as np # module for low-level scientific computing
#Hilbert transform to determine the amplitude envelope and 
#instantaneous frequency of an amplitude-modulated signal
from scipy.signal import hilbert 
from scipy.signal import butter
from scipy.signal import filtfilt
import matplotlib.pyplot as plt # makes matplotlib work like MATLAB. ’pyplot’ functions.
import easygui
import tables
from tqdm import trange
import pandas as pd
import collections

# =============================================================================
# Define functions
# =============================================================================

#define bandpass filter parameters to parse out frequencies
def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut /nyq
    high = highcut/nyq
    b, a = butter(order, [low, high], btype='bandpass')
    y = filtfilt(b, a, data)
    return y

def getFlaggedLFPs(hf5_name):

    # Load flagged channels from HDF5 if present
    try:
        flag_frame = pd.read_hdf(hdf5_name,'/Parsed_LFP/flagged_channels')
        flagged_channel_bool = 1
    except:
        print('No flagged channels dataset present. Defaulting to not using flags.')
        flagged_channel_bool = 0

    #Open the hdf5 file
    hf5 = tables.open_file(hdf5_name, 'r+')

    # Pull LFPS and spikes
    # Make sure not taking anything other than a dig_in
    lfps_dig_in = [node for node in hf5.list_nodes('/Parsed_LFP') \
            if 'dig_in' in str(node)]

    # If flagged channels dataset present
    if flagged_channel_bool > 0:
        good_channel_list = [list(flag_frame.\
                query('Dig_In == {} and Error_Flag == 0'.format(dig_in))['Channel']) \
                for dig_in in range(len(lfps_dig_in))]
    else:
        good_channel_list = [list(np.arange(lfps_dig_in[dig_in][:].shape[0])) \
                for dig_in in range(len(lfps_dig_in))]

    # Load LFPs and remove flagged channels if present
    lfp_list = [dig_in[:][good_channel_list[dig_in_num],:] \
        for dig_in_num,dig_in in enumerate(lfps_dig_in)]

    hf5.close()

    return lfp_list

# =============================================================================
# Define common variables
# =============================================================================
#specify frequency bands
iter_freqs = [
        (0, 4, 7),
        (1, 7, 12),
        (2, 13, 25),
        (3, 30, 45)]

#Covert to dframe for storing
freq_dframe = pd.DataFrame.from_dict(iter_freqs)
freq_dframe.columns = ['band_num','low_freq','high_freq']

colors = plt.get_cmap('winter_r')(np.linspace(0, 1, len(iter_freqs)))

# =============================================================================
# Import/Open HDF5 File
# =============================================================================

# If directory provided with script, use that otherwise ask
try:
    #dir_name = os.path.dirname(sys.argv[1])
    dir_name = sys.argv[1]
except:
    dir_name = easygui.diropenbox(msg = 'Select directory with HDF5 file')

hdf5_name = glob.glob(dir_name + '/*.h5')[0]

lfp_list = getFlaggedLFPs(hdf5_name)

hf5 = tables.open_file(hdf5_name, 'r+')
trains_dig_in = hf5.list_nodes('/spike_trains')
spike_array = np.asarray([spikes.spike_array[:] for spikes in trains_dig_in])

# ____                              _             
#|  _ \ _ __ ___   ___ ___  ___ ___(_)_ __   __ _ 
#| |_) | '__/ _ \ / __/ _ \/ __/ __| | '_ \ / _` |
#|  __/| | | (_) | (_|  __/\__ \__ \ | | | | (_| |
#|_|   |_|  \___/ \___\___||___/___/_|_| |_|\__, |
#                                           |___/ 

# =============================================================================
# Calculate phases
# =============================================================================

# Create lists of phase array
# Save data as namedTuple so band and taste are annotated
filtered_tuple = collections.namedtuple('BandpassFilteredData',['Band','Taste','Data'])
filtered_signal_list = [ filtered_tuple (band, taste,
                                butter_bandpass_filter(
                                            data = dig_in,
                                            lowcut = iter_freqs[band][1], 
                                            highcut =  iter_freqs[band][2], 
                                            fs = 1000)) \
                                            for taste,dig_in in enumerate(lfp_list)\
                    for band in trange(len(iter_freqs), desc = 'bands') \
                    if len(dig_in) > 0]

# =============================================================================
# Use mean LFP (across channels) to calculate phase (since all channels have same phase)
# =============================================================================
# Process filtered signals to extract hilbert transform and phase 
# Find channel closest in phase to the mean phase of all channels and use
# that for the phase

phase_list = \
    [ filtered_tuple(x.Band, x.Taste, np.angle(hilbert(x.Data))) \
                    for x in filtered_signal_list ]

mean_phase_list = \
    [ filtered_tuple(x.Band, x.Taste, np.mean(x.Data,axis=0)) \
                    for x in phase_list]

error_list = [np.sum(np.abs(np.subtract(phase.Data, mean_phase.Data)),axis=(1,2)) \
        for (phase,mean_phase) in zip(phase_list,mean_phase_list)]

chosen_channel = np.argmin(np.sum(np.asarray(error_list),axis=0))

final_phase_list = [ filtered_tuple(x.Band, x.Taste, x.Data[chosen_channel,:,:])\
        for x in phase_list]

# Calculate wavenumber for every trial
wavelength_num_list = [ filtered_tuple(x.Band, x.Taste,\
        np.floor_divide(np.unwrap(x.Data), 2*np.pi)) for x in final_phase_list]

# Validation plots --please leave commented--
#plt.subplot(311)
#plt.plot(np.real(filtered_signal_list[0].Data[0,0,:2000]))
#plt.subplot(312)
#plt.plot(final_phase_list[0].Data[0,:2000])
#plt.subplot(313)
#plt.plot(wavelength_num_list[0].Data[0,:2000])
#plt.show()

# =============================================================================
# Calculate phase locking: for every spike, find phase for every band
# =============================================================================
# Find spiketimes
# Find what phase each spike occured
spike_times = spike_array.nonzero()
spikes_frame = pd.DataFrame(data = {'taste':spike_times[0],
                                    'trial':spike_times[1],
                                    'unit':spike_times[2],
                                    'time':spike_times[3]})

# Create array index identifiers
# Used to convert array to pandas dataframe
def make_array_identifiers(array):
    nd_idx_objs = []
    for dim in range(array.ndim):
        this_shape = np.ones(len(array.shape))
        this_shape[dim] = array.shape[dim]
        nd_idx_objs.append(
                np.broadcast_to(
                    np.reshape(
                        np.arange(array.shape[dim]),
                                this_shape.astype('int')), 
                    array.shape).flatten())
    return nd_idx_objs

# Run through all groups of mean phase, convert to pandas dataframe
# and concatenate into single dataframe
phase_frame = pd.concat(
        [pd.DataFrame( data = { 'band' : phase.Band,
                                'taste' : phase.Taste,
                                'trial' : idx[0].flatten(),
                                'time' : idx[1].flatten(),
                                'phase' : phase.Data.flatten(),
                                'wavelength_num': wave_num.Data.flatten()}) \

                for phase, wave_num, idx in \
    zip(final_phase_list,
        wavelength_num_list,
        map(lambda dat: make_array_identifiers(dat.Data),final_phase_list))  
        ]
        )
        

# Merge : Gives dataframe with length of (bands x numner of spikes)
final_phase_frame = pd.merge(spikes_frame,phase_frame,how='inner')

#  ___        _               _   
# / _ \ _   _| |_ _ __  _   _| |_ 
#| | | | | | | __| '_ \| | | | __|
#| |_| | |_| | |_| |_) | |_| | |_ 
# \___/ \__,_|\__| .__/ \__,_|\__|
#                |_|              

#Flush and close file
hf5.flush()
hf5.close()

# Delete Spike_Phase_Dframe node so it can be re-written
with tables.open_file(hdf5_name,'r+') as hf5:
    if '/Spike_Phase_Dframe' in hf5:
        print('Removing last instance of Spike_Phase_Dframe')
        hf5.remove_node('/Spike_Phase_Dframe',recursive=True)

#Save dframes into node within HdF5 file
final_phase_frame.to_hdf(hdf5_name,'Spike_Phase_Dframe/dframe', mode = 'a')
freq_dframe.to_hdf(hdf5_name,'Spike_Phase_Dframe/freq_keys', mode = 'a')

