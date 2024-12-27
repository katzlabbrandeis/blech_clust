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
import easygui
import sys
import os
from scipy import stats
import h5py as h5

#Plotting imports
# makes matplotlib work like MATLAB. ’pyplot’ functions.
import matplotlib.pyplot as plt
import seaborn as sns

#Ask user if they have ran set-up before
msg   = "Have you performed directory set-up before '\
        '(ie. do you have '.dir' files' in output folder) ?"
dir_check = easygui.buttonbox(msg,choices = ["Yes","No"])

if dir_check == "No":
        # Ask for the directory where to store the directory file
        save_name = easygui.diropenbox(\
                msg = 'Choose where you want to save directory output file to...')

        # Ask the user for the hdf5 files that need to be plotted together (fist condition)
        dirs = []
        while True:
                dir_name = easygui.diropenbox(\
                    msg = 'Choose directory with a hdf5 file, hit cancel to stop choosing')
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

        #establish directories to flip through condition 1
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

        #Look for the hdf5 file in the directory
        file_list = os.listdir('./')
        hdf5_name = ''
        for files in file_list:
                if files[-2:] == 'h5':
                        hdf5_name = files

        #Status update
        print('Working on %s...' %(hdf5_name))

        #Open the hdf5 file and create list of child paths
        #hf5 = tables.open_file(hdf5_name, 'r+')
        f = h5.File(hdf5_name, 'r')

        #Import necessary data
        try:
                #by unit
                spike_times = np.array(f['Whole_session_spikes/all_spikes'])
                #ordered by dig_in
                delivery_times = \
                        np.array(f['/Whole_session_spikes/delivery_times/block0_values'])

        except:
                print('Bruh, you need to set up your data file!')
                sys.exit()

        # Make directory to store all analyses plots. Delete and
        # remake the directory if it exists
        os.chdir(dir_name)
        try:
                os.system('rm -r '+'./Intertrial_analyses')
        except:
                pass
        os.mkdir('./Intertrial_analyses')

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

        if shared_parms == "Yes":
                if file_count == 0:
                        freqparam = easygui.multenterbox(
                            'Specify data collection sampling rate and tastes delivered',
                            'Setting Analysis Parameters',
                            Boxes, Defaults)

        if shared_parms == "No":
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

        #Dynamically determine ITI and cut data accordingly
        ITI = np.array(np.diff(np.sort(delivery_times,axis=None))//int(freqparam[0]))

        if len(np.unique(ITI)) == 1:
                ITI = ITI[0]
        else:
                ITI = ITI

        if shared_parms == "Yes":
                if file_count == 0:
                        ITIparam = easygui.multenterbox(
                            'Specify how many seconds you want to skip extract POST-taste delivery',
                            'Setting data slicing Parameters',
                            ['Post Delivery time (s):','Slicing time (s):'], [8,10])

        if shared_parms == "No":
                ITIparam = easygui.multenterbox(
                            'Specify how many seconds you want to skip extract POST-taste delivery',
                            'Setting data slicing Parameters',
                            ['Post Delivery time (s):','Slicing time (s):'], [8,10])

        data_skip = int(ITIparam[0])*1000
        slice_ext = int(ITIparam[1])*1000


        #Extract taset delivery times
        taste_delivery = np.sort(delivery_times,axis=None)

        #Initiate empty list for spike data
        spike_data_sliced = []; LFP_data_sliced = []

        #Flip through each band and extract spiking
        counter = 0
        for band in np.array(f['Whole_session_raw_LFP'])[1:]:
                band_slice = []
                #for band in all_arrays[1:]:
                for i in range(len(taste_delivery)):

                        #Set slicing variables
                        taste_start = int((taste_delivery[i]/int(freqparam[0]))*1000)
                        slice_start = taste_start+data_skip

                        spike_slice_start = taste_delivery[i]+data_skip*(int(freqparam[0])/1000)

                        if counter == 0:
                                #Process spike data
                            spike_data_sliced.append(\
                                spike_times[:,np.array((spike_times>= spike_slice_start) & \
                                (spike_times <= spike_slice_start+\
                                        slice_ext*(int(freqparam[0])/1000)))[1,:]])

                        band_slice.append(f['Whole_session_raw_LFP']\
                                [band][:,:,slice_start:slice_start+slice_ext])

                # Stack LFP data cell-wise
                # (dimensions: band X trials X channels X filtered signal tuple X duration (ms))
                LFP_data_sliced.append(band_slice)
                counter += 1

        #Create folder for saving all plots
        save_name = dir_name+'/Intertrial_analyses'

        #Plot data
        for band in range(len(LFP_data_sliced)):
                # Make directory to store all analyses plots. Delete and
                # remake the directory if it exists
                os.chdir(save_name)
                try:
                        os.system('rm -r '+'./%s' %(iter_freqs[band][0]))
                except:
                        pass
                os.mkdir('./%s' %(iter_freqs[band][0]))

                for channel in range(np.array(LFP_data_sliced[band]).shape[1]):
                        #Initiate figure for mean power
                        fig, axs = plt.subplots(2, sharex=True, sharey=False,\
                            figsize=(20,10), gridspec_kw={'hspace': 0})

                        #plot heatmaps
                        axs[0].imshow(np.array(LFP_data_sliced[band])\
                                [:,channel,2,:].real,interpolation='nearest',aspect='auto')

                        #plot mean power across trials
                        axs[1].plot(np.mean(np.array(LFP_data_sliced[band])\
                                [:,channel,2,:].real,axis=0),color='midnightblue')
                        axs[1].plot(np.mean(np.array(LFP_data_sliced[band])\
                                [:,channel,2,:].real,axis=0)+\
                         np.std(np.array(LFP_data_sliced[band])[:,channel,2,:].real,axis=0),\
                         color = 'black',linewidth = 1,linestyle='dashed',alpha=0.8)
                        axs[1].plot(np.mean(np.array(LFP_data_sliced[band])\
                                [:,channel,2,:].real,axis=0)-\
                         np.std(np.array(LFP_data_sliced[band])[:,channel,2,:].real,axis=0),\
                         color = 'black',linewidth = 1,linestyle='dashed',alpha=0.8)

                        #Formatting
                        axs[0].set_ylabel('Trial',fontweight='bold',size=20)

                        axs[0].set_title('Animal: %s \nFull Taste Session \nDate: %s' \
                          %(hdf5_name[:4],hdf5_name.split('_')[2].split('_')[0]), \
                          fontweight='bold',size=24)

                        axs[1].set_ylabel('Power',fontweight='bold',size=20)
                        axs[1].set_xlabel('Time from %i (sec) Post-First Taste Delivery'\
                            %(data_skip/1000),fontweight='bold',size=20)
                        axs[1].set_xticks([])
                        axs[1].set_xticks(np.arange(0, slice_ext, step=1000))
                        axs[1].set_xticklabels( [str(num) \
                                for num in np.linspace(0,slice_ext//1000,1+slice_ext//1000)])
                        axs[0].set_xlim(0,slice_ext)

                        plt.tight_layout()

                        #Save plots
                        fig.savefig(save_name+'/%s'  \
                            %(iter_freqs[band][0])+'/Channel_%i_intertrial_power.png' %(channel))
                        plt.close(fig)

                        #Initiate figure for zsored power
                        fig, axs = plt.subplots(2, sharex=True, sharey=False,\
                                figsize=(20,10), gridspec_kw={'hspace': 0})

                        #plot heatmaps
                        axs[0].imshow(np.array(LFP_data_sliced[band])\
                                [:,channel,2,:].real,interpolation='nearest',aspect='auto')

                        #plot zscored power across trials
                        axs[1].plot(np.mean(stats.zscore(np.array(LFP_data_sliced[band])\
                                [:,channel,2,:].real,axis=1),axis=0),color='midnightblue')

                        #Formatting
                        axs[0].set_ylabel('Trial',fontweight='bold',size=20)

                        axs[0].set_title('Animal: %s \nFull Taste Session \nDate: %s' \
                          %(hdf5_name[:4],hdf5_name.split('_')[2].split('_')[0]), \
                          fontweight='bold',size=24)

                        axs[1].set_ylabel('Z-Scored Power',fontweight='bold',size=20)
                        axs[1].set_xlabel('Time from %i (sec) Post-First Taste Delivery'\
                            %(data_skip/1000),fontweight='bold',size=20)
                        axs[1].set_xticks([])
                        axs[1].set_xticks(np.arange(0, slice_ext, step=1000))
                        axs[1].set_xticklabels( [str(num) \
                                for num in np.linspace(0,slice_ext//1000,1+slice_ext//1000)])
                        axs[0].set_xlim(0,slice_ext)

                        plt.tight_layout()

                        #Save plots
                        fig.savefig(save_name+'/%s'  \
                            %(iter_freqs[band][0])+'/Channel_%i_intertrial_Zpower.png' %(channel))
                        plt.close(fig)

                        #Plot the distributions of power atop eachother
                        bins =  np.percentile(\
                                    np.array(\
                                        LFP_data_sliced[band])\
                                            [:,channel,2,:].real,np.linspace(0,100,100))

                        power_hist = []
                        for trial in range(\
                                        len(\
                                            np.array(\
                                                LFP_data_sliced[band])[:,channel,2,:].real)):

                                power_hist.append(\
                                        np.histogram(\
                                            np.array(LFP_data_sliced[band])\
                                                [trial,channel,2,:].real,bins=bins)[0])

                        #initiate figure
                        fig = plt.figure(figsize=(20,10))
                        plt.imshow(np.array(power_hist).T,origin='lower',\
                          cmap='jet', interpolation = 'gaussian', aspect = 'auto')

                        #Formatting
                        plt.ylabel('Percentile',fontweight='bold',size=20)
                        plt.xlabel('Trials',fontweight='bold',size=20)
                        plt.colorbar().set_label(label='Density',size=20)
                        plt.title('Animal: %s \nPower over Full Taste Session \nDate: %s' \
                          %(hdf5_name[:4],hdf5_name.split('_')[2].split('_')[0]), \
                          fontweight='bold',size=24)

                        plt.tight_layout()

                        #Save plots
                        fig.savefig(save_name+'/%s'  \
                                %(iter_freqs[band][0])+'/Channel_%i_Power.png' %(channel))
                        plt.close(fig)

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

        #Save plots
        fig.savefig(save_name+'/%s_%s_Intertrial_raster.png' \
                %(hdf5_name[:4],hdf5_name.split('_')[2].split('_')[0]))
        plt.close(fig)

        #Close file
        f.close()

        #Update file counter
        file_count += 1
