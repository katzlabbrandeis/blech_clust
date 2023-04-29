#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 14:41:43 2017

@author: bradly
"""

#Import necessary tools
import numpy as np
import easygui
import tables
import os
import scipy.io as sio

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

#Ask if file needs to be split, if yes, split it
split_response = easygui.indexbox(msg='Do you need to split these trials?', title='Split trials', choices=('Yes', 'No'), image=None, default_choice='Yes', cancel_choice='No')

# Ask if this analysis is looking at more than 1 trial and/or taste
msg   = "Do you want to create an LFP matlab file for more than ONE trial (ie. Do you have several tastes) ?"
trial_check = easygui.buttonbox(msg,choices = ["Yes","No"])

if trial_check == "Yes":
    total_trials = hf5.root.Parsed_LFP.dig_in_1_LFPs[:].shape[1]
if trial_check == "No":
    total_trials = 1

dig_in_channels = hf5.list_nodes('/digital_in')
dig_in_LFP_nodes = hf5.list_nodes('/Parsed_LFP')

if split_response == 0:
    trial_split = easygui.multenterbox(msg = 'Put in the number of trials to parse from each of the LFP arrays (only integers)', fields = [node._v_name for node in dig_in_LFP_nodes], values = ['15' for node in dig_in_LFP_nodes])
    #Convert all values to integers
    trial_split = list(map(int,trial_split))
    total_sessions = int(total_trials/int(trial_split[0]))
else:    
    #if int(trial_split[0]) == 0:
    total_sessions = 1
    trial_split = list(map(int,[total_trials for node in dig_in_LFP_nodes]))

#Build dictionary, extract arrays, and store in dictionary
LFP_data = {}

if trial_check == "Yes":
    for sessions in range(total_sessions):
    
        LFP_data['Session_%i' % sessions] = [np.array(dig_in_channels[node][-8:]) for node in range(len(dig_in_LFP_nodes))]
    
        #LFP_data['Session_%i' % sessions] = np.zeros(len(trial_split))
        for node in range(len(dig_in_LFP_nodes)):
            exec("LFP_array = hf5.root.Parsed_LFP.dig_in_%i_LFPs[:] " % node)
            if sessions == 0:
                LFP_data['Session_%i' % sessions][node] = LFP_array[:,0:trial_split[node],:]
            else:
                LFP_data['Session_%i' % sessions][node] = LFP_array[:,trial_split[node]:int((total_trials/int(trial_split[0]))*(trial_split[node])),:]
    
        #Save arrays into .mat format for processing in MATLAB
        sio.savemat(hdf5_name[:-12] + '_all_tastes.mat', {'all_tastes':LFP_data})
        
if trial_check == "No":
    # Ask if this is for drug or saline
    msg   = "Are you using dig_in_4 (Saline) or dig_in_5 (Drug)?"
    LFP_dig_in = easygui.buttonbox(msg,choices = ["4","5"])
    
    #Put LFP data into dictionary
    LFP_data['Session_%i' % 0] = [np.array(dig_in_LFP_nodes[0])]
    exec("LFP_array = hf5.root.Parsed_LFP.dig_in_%i_LFPs[:] " % int(LFP_dig_in[0]))
    LFP_data['Session_%i' % 0][0] = LFP_array[:,0:trial_split[0],:]
    
    #Save arrays into .mat format for processing in MATLAB
    sio.savemat(hdf5_name[:-12] + '_affective_session.mat', {'affective_states':LFP_data})

#Indicate process complete
print('*.mat files saved')

#Close file
hf5.flush()
hf5.close() 