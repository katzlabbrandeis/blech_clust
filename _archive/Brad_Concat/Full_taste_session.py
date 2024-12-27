#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:25:00 2019

@author: bradly
"""

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
# makes matplotlib work like MATLAB. ’pyplot’ functions.
import matplotlib.pyplot as plt

##########################
# Define functions to be used
##########################

def get_filtered_electrode(data, low_pass, high_pass, sampling_rate):
    el = 0.195*(data)
    m, n = butter(
            2,
            [2.0*int(low_pass)/sampling_rate, 2.0*int(high_pass)/sampling_rate],
            btype = 'bandpass')
    filt_el = filtfilt(m, n, el)
    return filt_el

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

##########################
# Begin Processing
##########################

#Ask user if they have ran set-up before
msg   = "Have you performed directory set-up before '\
        '(ie. do you have '.dir' files' in output folder) ?"
dir_check = easygui.buttonbox(msg,choices = ["Yes","No"])

if dir_check == "No":
    # Ask for the directory where to store the directory file
        save_name = easygui.diropenbox(msg = \
                'Choose where you want to save directory output file to...')

        # Ask the user for the hdf5 files that need to be plotted together (fist condition)
        dirs = []
        while True:
                dir_name = easygui.diropenbox(msg = \
                        'Choose directory with a hdf5 file, hit cancel to stop choosing')
                try:
                        if len(dir_name) > 0:
                                dirs.append(dir_name)
                except:
                        break

    #Dump the directory names into chosen output location for each condition
    #condition 1
        completeName = os.path.join(save_name, 'Taste_dirs.dir')
        f = open(completeName, 'w')
        for item in dirs:
                f.write("%s\n" % item)
        f.close()

if dir_check == "Yes":
        # Ask for the directory where to store the directory file
        dir_folder = easygui.diropenbox(msg = 'Choose where directory output file is...')

    #establish directories to flip through
    #condition 1
        dirs_path = os.path.join(dir_folder, 'Taste_dirs.dir')
        dirs_file = open(dirs_path,'r')
        dirs = dirs_file.read().splitlines()
        dirs_file.close()

#Establish whether parameters can be shared amongst all files in dir.
msg   = "Can all processing parameters be shared across files in dir?"
shared_parms = easygui.buttonbox(msg,choices = ["Yes","No"])

#Flip through directories to perform processing and initiate file counter
file_count = 0
for dir_name in dirs:

        #Change to the directory
        os.chdir(dir_name)
        #Locate the hdf5 file
        file_list = os.listdir('./')
        hdf5_name = ''
        for files in file_list:
                if files[-2:] == 'h5':
                        hdf5_name = files

        # Open the hdf5 file
        hf5 = tables.open_file(hdf5_name, 'r+')

        #Status update
        print('Working on %s...' %(hdf5_name))

        # Grab the names of the arrays containing digital inputs,
        # and pull the data into a numpy array
        dig_in_nodes = hf5.list_nodes('/digital_in')
        dig_in = []
        dig_in_pathname = []
        for node in dig_in_nodes:
                dig_in_pathname.append(node._v_pathname)
                exec("dig_in.append(hf5.root.digital_in.%s[:])" \
                        % dig_in_pathname[-1].split('/')[-1])
        dig_in = np.array(dig_in)

        # Get the stimulus delivery times - take the end of the
        # stimulus pulse as the time of delivery
        dig_on = []
        for i in range(len(dig_in)):
                dig_on.append(np.where(dig_in[i,:] == 1)[0])
        start_points = []
        end_points = []
        for on_times in dig_on:
                start = []
                end = []
                try:
                        start.append(on_times[0]) # Get the start of the first trial
                except:
                        # Continue without appending anything if this port wasn't on at all
                        pass
                for j in range(len(on_times) - 1):
                        if np.abs(on_times[j] - on_times[j+1]) > 30:
                                end.append(on_times[j])
                                start.append(on_times[j+1])
                try:
                        # append the last trial which will be missed by this method
                        end.append(on_times[-1])
                except:
                        pass # Continue without appending anything if this port wasn't on at all
                start_points.append(np.array(start))
                end_points.append(np.array(end))

        if shared_parms == "Yes":
                if file_count == 0:
                        # Show the user the number of trials on each digital input channel,
                        # and ask them to confirm
                        check = easygui.ynbox(\
                                msg = 'Digital input channels: ' + \
                                str(dig_in_pathname) + '\n' + 'No. of trials: ' + \
                                str([len(ends) for ends in end_points]), \
                                title = 'Check and confirm the number of trials'\
                                    'detected on digital input channels')
                        # Go ahead only if the user approves by saying yes
                        if check:
                                pass
                        else:
                                print("Well, if you don't agree, blech_clust can't do much!")
                                sys.exit()

                        # Ask the user which digital input channels should be
                        # used for getting spike train data, and convert the
                        # channel numbers into integers for pulling stuff out of change_points
                        dig_in_channels = easygui.multchoicebox(
                                msg = 'Which digital input channels should be used to '\
                                        'produce spike train data trial-wise?',
                                        choices = ([path for path in dig_in_pathname]))
                        dig_in_channel_nums = []
                        for i in range(len(dig_in_pathname)):
                                if dig_in_pathname[i] in dig_in_channels:
                                        dig_in_channel_nums.append(i)

        if shared_parms == "No":
                #Ask for each file
                # Show the user the number of trials on each digital input channel,
                # and ask them to confirm
                check = easygui.ynbox(
                        msg = 'Digital input channels: ' + \
                                str(dig_in_pathname) + '\n' + \
                                'No. of trials: ' + \
                                str([len(ends) for ends in end_points]), \
                                title = 'Check and confirm the number of trials'\
                                        ' detected on digital input channels')
                # Go ahead only if the user approves by saying yes
                if check:
                        pass
                else:
                        print("Well, if you don't agree, blech_clust can't do much!")
                        sys.exit()

                # Ask the user which digital input channels should be
                # used for getting spike train data, and convert the
                # channel numbers into integers for pulling stuff out of change_points
                dig_in_channels = easygui.multchoicebox(\
                        msg = 'Which digital input channels should be used '\
                        'to produce spike train data trial-wise?',
                        choices = ([path for path in dig_in_pathname]))
                dig_in_channel_nums = []
                for i in range(len(dig_in_pathname)):
                        if dig_in_pathname[i] in dig_in_channels:
                                dig_in_channel_nums.append(i)

        #Extract only channels that you want to analyze
        dig_in_data = [end_points[i] for i in dig_in_channel_nums]

        #=======================================================================
        ##Pull 20sec prior/after first/last taste delivery
        #first_delivery = min(list(map(min, dig_in_data)))
        #last_delivery = max(list(map(max, dig_in_data)))
        #
        #exp_dur = 1000*(round((last_delivery-first_delivery)/30000)+40)
        #exp_dur = round((last_delivery-first_delivery)+40)
        #=======================================================================

        # Get list of units under the sorted_units group.
        # Find the latest/largest spike time amongst the units, i
        # and get an experiment end time (to account for cases where
        # the headstage fell off mid-experiment)

        units = hf5.list_nodes('/sorted_units')
        expt_end_time = 0
        for unit in units:
                if unit.times[-1] > expt_end_time:
                        expt_end_time = unit.times[-1]

        #Store spikes by cell number
        spikes = np.zeros((len(units), int(expt_end_time)+1), dtype='uint8')
        for k in range(len(units)):
                spikes[k, units[k].times[:]] = 1

        #Find where spikes occur
        sorted_spikes_ID = np.where(spikes)

        #Check if whole session spike data is already within file and remove node if so.
        #Create new spike group within H5 file.
        try:
            hf5.remove_node('/Whole_session_spikes', recursive = True)
        except:
            pass
        hf5.create_group('/', 'Whole_session_spikes')

        #Store as (unitXdurations [conversion(min)=array/sampling rate(30000)])
        hf5.create_array('/Whole_session_spikes', 'all_spikes', sorted_spikes_ID)

        #Store data as dataframe into hf5
        df_deliver = pd.DataFrame(dig_in_data)
        df_deliver.to_hdf(hdf5_name, '/Whole_session_spikes' + '/delivery_times')

        hf5.flush()

        # =============================================================================
        # #Plotting
        # =============================================================================
        #Sorts taste delivery time into one large vector
        taste_delivery = np.sort([j for i in dig_in_data for j in i],axis=None)

        #Sort cells based on spike count
        sorted_spikes = spikes[np.argsort(spikes.sum(axis=1))]
        sorted_spikes_ID = np.where(sorted_spikes)

        fig = plt.figure(figsize=(40,20))
        plt.scatter(sorted_spikes_ID[1], sorted_spikes_ID[0], s=0.5,color='black')

        #Add markers for taste delivery
        for xc in taste_delivery:
            plt.axvline(x=xc, color='r', linestyle='-',alpha =0.5)

        #Formatting
        labels=np.arange(0, sorted_spikes_ID[1][-1], step=1800000*4)/(30000)
        plt.xticks(np.arange(0, sorted_spikes_ID[1][-1], step=1800000*4),labels,size=18)
        plt.yticks(size=18)
        plt.ylabel('Cell',fontweight='bold',size=20)
        plt.xlabel('Time from Recording Start (min)',fontweight='bold',size=20)
        plt.title('Animal: %s \nFull Taste Session \nDate: %s' \
                          %(hdf5_name[:4],hdf5_name.split('_')[2].split('_')[0]), \
                          fontweight='bold',size=24)
        plt.tight_layout()

        #Save output
        fig.savefig(dir_name+'/%s_%s_Full_Session_raster.png' \
                %(hdf5_name[:4],hdf5_name.split('_')[2].split('_')[0]))
        plt.close(fig)

        #Set parameters for LFPs
        if shared_parms == "Yes":
                if file_count == 0:
                        #Specify filtering parameters (linear-phase finite impulse response filter)
                        #to define filter specificity across electrodes
                        Boxes = ['low','high','Sampling Rate']
                        freqparam = list(map(int,easygui.multenterbox(
                            'Specify LFP bandpass filtering paramters and sampling rate',
                            'Low-Frequency Cut-off (Hz)',
                            Boxes, [1,300,30000])))

        if shared_parms == "No":
                #Do for every file
                #Specify filtering parameters (linear-phase finite impulse response filter)
                #to define filter specificity across electrodes
                Boxes = ['low','high','Sampling Rate']
                freqparam = list(map(int,easygui.multenterbox(
                    'Specify LFP bandpass filtering paramters and sampling rate',
                    'Low-Frequency Cut-off (Hz)',
                    Boxes, [1,300,30000])))

        #Use parameters to extract and plot FRs for all cells over session
        #Find whole second, Cut file to nearest one, get FR (Hz: spikes/sec)
        whole_session = int((spikes.shape[1]//freqparam[2]))

        WS_FRs = np.mean(spikes[:,:whole_session*freqparam[2]].\
                                   reshape((spikes.shape[0],whole_session,freqparam[2])), \
                                   axis=-1)*freqparam[2] #Multiply by Sampling Rate --> Hz

        #Sort cells based on Firing Rates
        sorted_FRs  = WS_FRs[np.argsort(WS_FRs.sum(axis=1))]

        #Sort cells based on Zscored Firing Rates
        z_FRs = stats.zscore(WS_FRs,axis=1)
        sorted_ZFRs  = z_FRs[np.argsort(z_FRs.sum(axis=1))]

        #Plot FRs
        fig = plt.figure(figsize=(20,10))
        plt.imshow(np.flipud(sorted_FRs),interpolation='nearest',aspect='auto')
        plt.colorbar().set_label(label='Firing Rate (Hz)',size=20)

        #Formatting
        plt.yticks(size=18); plt.xticks(size=18)
        plt.ylabel('Cell',fontweight='bold',size=20)
        plt.xlabel('Time from Recording Start (sec)',fontweight='bold',size=20)
        plt.title('Animal: %s \nFull Taste Session \nDate: %s' \
                          %(hdf5_name[:4],hdf5_name.split('_')[2].split('_')[0]), \
                          fontweight='bold',size=24)
        plt.tight_layout()

        #Save output
        fig.savefig(\
                dir_name+\
                '/%s_%s_Full_Session_FRs.png' \
                %(hdf5_name[:4],hdf5_name.split('_')[2].split('_')[0]))
        plt.close(fig)

        #Plot Zscored FRs
        fig = plt.figure(figsize=(20,10))
        plt.imshow(np.flipud(sorted_ZFRs),interpolation='nearest',aspect='auto')
        plt.colorbar().set_label(label='Firing Rate (Z-scored)',size=20)

        #Formatting
        plt.yticks(size=18); plt.xticks(size=18)
        plt.ylabel('Cell',fontweight='bold',size=20)
        plt.xlabel('Time from Recording Start (sec)',fontweight='bold',size=20)
        plt.title('Animal: %s \nFull Taste Session \nDate: %s' \
                          %(hdf5_name[:4],hdf5_name.split('_')[2].split('_')[0]), \
                          fontweight='bold',size=24)
        plt.tight_layout()

        #Save output
        fig.savefig(\
                dir_name+\
                '/%s_%s_Full_Session_ZFRs.png' \
                %(hdf5_name[:4],hdf5_name.split('_')[2].split('_')[0]))
        plt.close(fig)

        # =============================================================================
        # #Extract LFPs for full session
        # =============================================================================

        # ==============================
        # Select channels to read
        # ==============================

        #Create vector of electode numbers that have neurons
        #on them (from unit_descriptor table)
        #Some electrodes may record from more than one neuron
        #(shown as repeated number in unit_descriptor);
        #Remove these duplicates within array
        electrodegroup = np.unique(hf5.root.unit_descriptor[:]['electrode_number'])

        ## List all appropriate dat files
        # Look for the hdf5 file in the directory
        Raw_Electrodefiles = np.sort(np.array([f for f in os.listdir(dir_name) \
                                        if f.endswith('.dat') & ~f.find('amp')]))
        Raw_Electrodefiles = Raw_Electrodefiles[electrodegroup]

        # ==============================
        # Extract Raw Data
        # ==============================

        #Check if LFP data is already within file and remove node if so.
        #Create new raw LFP group within H5 file.
        try:
            hf5.remove_node('/Whole_session_raw_LFP', recursive = True)
        except:
            pass
        hf5.create_group('/', 'Whole_session_raw_LFP')

        #Loop through each neuron-recording electrode (from .dat files),
        #complie, filter data, and create array in new LFP node
        compiled_LFPS= []
        for i in range(len(Raw_Electrodefiles)):
                compiled_LFPS.append(np.fromfile(Raw_Electrodefiles[i], \
                                                dtype = np.dtype('int16')))

        #Read and filter data
        resampled_d = np.mean(np.array(compiled_LFPS).\
                                    reshape((len(compiled_LFPS),-1,30)),axis=-1)
        filt_el = get_filtered_electrode(data = resampled_d,
                                            low_pass = freqparam[0],
                                            high_pass = freqparam[1],
                                            sampling_rate = freqparam[2])

        #Store data     (channel X LFP(ms))
        hf5.create_array('/Whole_session_raw_LFP','WS_LFP',filt_el)
        hf5.flush()


        # =============================================================================
        # #Set parameters
        # =============================================================================
        if shared_parms == "Yes":
                if file_count == 0:
                        delivery_times = np.array(dig_in_data)
                        Boxes = [[] for x in range(0,delivery_times.shape[0]+1)]
                        Boxes = ["Taste:" for x in Boxes]
                        Boxes[0] = 'Sampling Rate'

                        Def_labels = [30000,'NaCl','Sucrose','Citric Acid','QHCl']
                        Defaults = [[] for x in range(0,delivery_times.shape[0]+1)]
                        for x in range(len(Defaults)):
                                Defaults[x] = Def_labels[x]

                        #specify frequency bands
                        iter_freqs = [('Delta',1,3),
                                ('Theta', 4, 7),
                                ('Mu', 8, 12),
                                ('Beta', 13, 25),
                                ('Gamma', 30, 45)]

        #Create array out of filtered electrode info
        LFP_times = np.array(filt_el)

        if "/Whole_session_raw_LFP/WS_LFP_filtered" in hf5:
                print('You already have done the filtering')
        else:
                #Extract filtered LFPs
                for band in trange(len(iter_freqs), desc = 'bands'):
                        output = Parallel(n_jobs=6)(delayed(butter_bandpass_filter_parallel)\
                                (LFP_times, iter_freqs, channel, band) \
                                for channel in range(len(LFP_times)))

                        hf5.create_array('/Whole_session_raw_LFP',
                                'WS_LFP_filtered_band_%i' %(band),np.asarray(output))
                        hf5.flush()

        hf5.close()

        #Update file counter
        file_count += 1
