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
dig_in_channels = hf5.list_nodes('/digital_in')
spike_train_dig = hf5.list_nodes('/spike_trains')

#Build dictionary, extract arrays, and store in dictionary
spiketrain_data = {}; attemptingnow = {}

for dig_in in range(len(spike_train_dig)):
    exec("spike_array = hf5.root.spike_trains.dig_in_%i.spike_array[:] " % dig_in)
    spiketrain_data['dig_in_%i' % dig_in] = [np.array(spike_array)]

    #Save arrays into .mat format for processing in MATLAB
    sio.savemat(hdf5_name[:-12] + '_spike_trains.mat', {'spike_trains':spiketrain_data})
        

#Indicate process complete
print('*.mat files saved')

#Close file
hf5.flush()
hf5.close() 