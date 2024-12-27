#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 13:09:53 2020

@author: bradly
"""

# import Libraries
# Built-in Python libraries
import os # functions for interacting w operating system

# 3rd-party libraries
import numpy as np # module for low-level scientific computing
import easygui
import tables
import matplotlib.pyplot as plt
from scipy.signal import argrelmax
from datetime import date
import pandas as pd

#Establish files to import into dframe
# Ask for the directory where to store the directory file
dir_folder = easygui.diropenbox(msg = 'Choose where directory output file is...')

dirs_path = os.path.join(dir_folder, 'all_dirs.dir')
dirs_file = open(dirs_path,'r')
dirs = dirs_file.read().splitlines()
dirs_file.close()

#Create list of animal variable for ease of use
animals = [str(x.split('/')[-1][0:4]) for x in sorted(dirs)]
animals = list(set(animals))

#Flip through directories to perform processing and initiate file counter
file_count = 0; all_data = []; file_names = []
for dir_name in dirs:

	#Change to the directory
	os.chdir(dir_name)

	#Look for the hdf5 file in the directory
	file_list = os.listdir('./')
	hdf5_name = ''
	for files in file_list:
	        if files[-2:] == 'h5':
	                hdf5_name = files

	#store name of files for worksheet labeling
	file_names.append(hdf5_name[0:-3])

	# Get the amplifier ports used
	ports = list(set(f[4] for f in file_list if f[:3] == 'amp'))
	# Sort the ports in alphabetical order
	ports.sort()

	#Open hdf5
	hf5 = tables.open_file(hdf5_name, 'r+')

	animal_data = []
	for port in ports:
		for channel in range(32):

			#extract the data
			data = np.fromfile('amp-' + port + '-%03d'%channel + '.dat', dtype = np.dtype('int16'))
#			data = 0.195*(data-32768) #converts data to microvolts

			#downsample data (assumes 30k sampling rate)
			animal_data.append(data.reshape(-1, 30).mean(axis=1))

		hf5.flush()

	hf5.close()

	#append to larger list
	all_data.append(animal_data)

#Create list of max session binning (assumes all channels have same length)
bin_factor = 1000	#binning in ms
session_floor = [np.floor(all_data[x][0].shape[0]/bin_factor).astype(int) for x in range(len(all_data))]

#create workbook for data storage using XlsxWriter as the engine
save_path = dir_folder + '/parameterization_' + date.today().strftime("%d_%m_%Y") + '.xlsx'
writer = pd.ExcelWriter(save_path, engine='xlsxwriter')

#bin the data and then calculate root mean square
rms_all = []; stnr_all = []
for file in range(len(all_data)):

	binned = [all_data[file][x][:session_floor[file]*bin_factor].reshape(-1, bin_factor) for x in range(len(all_data[file]))]

	rms_channel = []; stnr_channel =[]
	for channel in range(len(binned)):
		rms = np.array([np.sqrt(np.mean(binned[channel][x]**2)) for x in range(len(binned[channel]))])
		stnr =  np.array([np.mean(binned[channel][x])/np.std(binned[channel][x]) for x in range(len(binned[channel]))])

		#append by channel
		rms_channel.append(rms)
		stnr_channel.append(stnr)

	#append to larger list
	rms_all.append(rms_channel)
	stnr_all.append(stnr_channel)

	#set up data for file writing
	id_rms = np.hstack((np.reshape(np.arange(32).T, (-1, 1)),np.array(rms_channel)))
	id_stnr = np.hstack((np.reshape(np.arange(32).T, (-1, 1)),np.array(stnr_channel)))
	merged_dat = np.vstack((id_rms,id_stnr))

	#convert to dataframe for storage
	df = pd.DataFrame(merged_dat)
	df.rename(columns={0: 'channel'},inplace = True)
	df.insert(loc=1,column='metric',value=np.hstack((['RMS'] * 32,['STN'] * 32)))

	#write data to excel file for ease of later use
	name = file_names[file].split('_')[0].strip() + '_' + \
			file_names[file].split('_')[2].strip() + '_' +\
			file_names[file].split('_')[3].strip()
	df.to_excel(writer, sheet_name=name)


	#PLOTTING
	#Heatmaps - RMS
	fig = plt.figure(figsize=(12,8))
	plt.imshow(np.array(rms_all[file]),aspect='auto',interpolation ='nearest')
	plt.title('Root Mean Square' + '\n%s' % (file_names[file][0:42]))
	plt.xlabel('Time from recording start (ms)')
	plt.ylabel('Channel #')
	fig.savefig(dir_folder + '/%s_RMS.png' %(name))
	plt.close('all')

	#Heatmaps - STN
	fig = plt.figure(figsize=(12,8))
	plt.imshow(np.array(stnr_all[file]),aspect='auto',interpolation ='nearest')
	plt.title('Signal to Noise Ratio' + '\n%s' % (file_names[file][0:42]))
	plt.xlabel('Time from recording start (ms)')
	plt.ylabel('Channel #')
	fig.savefig(dir_folder + '/%s_STN.png' %(name))
	plt.close('all')

	#RMS Distributions
	rms_mode = []
	fig = plt.figure(figsize=(12,8))
	for i in range(len(rms_all[file])):
		n, bins = np.histogram(rms_all[file][i],np.linspace(0,1500,500))
		x = np.linspace(0,1500,500)

		#find index of min between modes
		ind_max = argrelmax(n)
		x_max = x[ind_max]
		y_max = n[ind_max]

		#find first/second y values in y_max
		index_first_max = np.argmax(y_max)
		maximum_y = y_max[index_first_max]

		plt.hist(rms_all[file][i],np.linspace(0,1500,500),\
					histtype='step',density=False, label = i, alpha=0.3)

		#label the rms modes and append to list
		plt.scatter(x_max[index_first_max],y_max[index_first_max],color='black')
		rms_mode.append(np.array([x_max[index_first_max],y_max[index_first_max]]))

	plt.legend()
	plt.ylim(0,80)
	plt.title('RMS Distibution' + '\n%s' % (file_names[file][0:42]))
	plt.xlabel('Voltage (mV)')
	plt.ylabel('Frequency')
	fig.savefig(dir_folder + '/%s_RMSdist.png' %(name))
	plt.close('all')

	#Plot the mode of eaach channel
	fig = plt.figure(figsize=(12,8))
	plt.bar(np.arange(32),[x[0] for x in rms_mode])
	plt.xlabel('Channel #')
	plt.ylabel('RMS Mode')
	plt.title('RMS Mode by Channel' + '\n%s' % (file_names[file][0:42]))
	fig.savefig(dir_folder + '/%s_RMSmode.png' %(name))
	plt.close('all')

	#STN Distributions
	stn_mode = []
	fig = plt.figure(figsize=(12,8))
	for i in range(len(stnr_all[file])):
		n, bins = np.histogram(stnr_all[file][i],np.linspace(-2,2,500))
		x = np.linspace(-2,2,500)

		#find index of min between modes
		ind_max = argrelmax(n)
		x_max = x[ind_max]
		y_max = n[ind_max]

		#find first/second y values in y_max
		index_first_max = np.argmax(y_max)
		maximum_y = y_max[index_first_max]

		plt.hist(stnr_all[file][i],np.linspace(-2,2,500),\
					histtype='step',density=False, label = i, alpha=0.3)

		#label the rms modes and append to list
		plt.scatter(x_max[index_first_max],y_max[index_first_max],color='black')
		stn_mode.append(np.array([x_max[index_first_max],y_max[index_first_max]]))

	plt.legend()
	plt.ylim(0,85)
	plt.title('STN Distibution' + '\n%s' % (file_names[file][0:42]))
	plt.xlabel('Voltage (mV)')
	plt.ylabel('Frequency')
	fig.savefig(dir_folder + '/%s_STNdist.png' %(name))
	plt.close('all')

	#Plot the mode of eaach channel
	fig = plt.figure(figsize=(12,8))
	plt.bar(np.arange(32),[x[0] for x in stn_mode])
	plt.xlabel('Channel #')
	plt.ylabel('STN Mode')
	plt.title('STN Mode by Channel' + '\n%s' % (file_names[file][0:42]))
	fig.savefig(dir_folder + '/%s_STNmode.png' %(name))
	plt.close('all')

#Close and save excel sheet
writer.save()
