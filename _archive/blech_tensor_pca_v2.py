#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 13:56:18 2020

@author: bradly
"""
# import Libraries
# Built-in Python libraries
import os # functions for interacting w operating system

# 3rd-party libraries
import numpy as np # module for low-level scientific computing
import easygui
import pandas as pd
import tables

#import stat packages
from scipy.stats import pearsonr
from scipy.stats import rankdata
from scipy import stats

#Import plotting tools
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt

#Import tensor package
import tensortools as tt

#Define  function to calculate firing rates
def _calc_firing_rates(step_size, window_size, dt , spike_array):
        """
        step_size
        window_size :: params :: In milliseconds. For moving window firing rate
                                calculation
        sampling_rate :: params :: In ms, To calculate total number of bins
        spike_array :: params :: 3D array with time as last dimension
        """

        if np.sum([step_size % dt , window_size % dt]) > 1e-14:
            raise Exception('Step size and window size must be integer multiples'\
                    ' of the inter-sample interval')

        fin_step_size, fin_window_size = \
            int(step_size/dt), int(window_size/dt)
        total_time = spike_array.shape[-1]

        bin_inds = (0,fin_window_size)
        total_bins = int((total_time - fin_window_size + 1) / fin_step_size) + 1
        bin_list = [(bin_inds[0]+step,bin_inds[1]+step) \
                for step in np.arange(total_bins)*fin_step_size ]

        firing_rate = np.empty((spike_array.shape[0],spike_array.shape[1],total_bins))
        for bin_inds in bin_list:
            firing_rate[:,:,bin_inds[0]//fin_step_size] = \
                    np.sum(spike_array[:,:,bin_inds[0]:bin_inds[1]], axis=-1)

        return firing_rate
# =============================================================================
# Import/Open HDF5 File and variables
# =============================================================================
dir_folder = easygui.diropenbox(msg = 'Choose where directory output file is...')
dirs_path = os.path.join(dir_folder, 'Taste_dirs.dir')
dirs_file = open(dirs_path,'r')
dirs = dirs_file.read().splitlines()
dirs_file.close()

#Get name of directory where the data files and xlsx file sits, and change to that directory for processing
tensor_dir_name = easygui.diropenbox('Please select the folder that you want to store outputs in.',\
									 default = '/home/bradly/drive2/data/Affective_State_Protocol/LiCl/all_taste_dir/for_paper/Tensor_PCA')

#Ask user about held units anlysis and perform analyses as seen fit
msg   = "Do you want to include a held cell analyses?"
held_check = easygui.buttonbox(msg,choices = ["Yes","No"])

if held_check == "Yes":
	#Ask if held cells were detected using CAR or Raw?
	msg   = "Were held cells detected using Raw data or Common Average Referenced?"
	type_check = easygui.buttonbox(msg,choices = ["Raw","C.A.R."])

	held_dir_name = easygui.diropenbox(msg = 'Choose directory with all "_held_units.txt" files in it.',default = '/home/bradly/drive2/spike_locking')

	#Create empty arrays for storing FRH later
	#held_FRs_cond1 = []; held_FRs_cond2 = [];
	held_FRs_cond1_2 = []; held_FRs_cond2_2 = []

	#Flip through all files and create a held_units list
	for file in held_dir_name:
		#Change to the directory
		os.chdir(held_dir_name)
		#Locate the hdf5 file
		file_list = os.listdir('./')
		held_file_name = ''

		day1 =[]; day2=[]; #Create arrays for each day
		all_day1=[]; all_day2=[] #Create arrays for combined days
		for files in sorted(file_list):
			if files[-3:] == 'txt':

				if type_check == "C.A.R.":
					with open(files,'r') as splitfile:
						for columns in [line.split() for line in splitfile]:
							day1.append(columns[0]);	day2.append(columns[1])
						all_day1.append(day1); all_day2.append(day2)
						day1 = []; day2 = []    #Clear day array

				if type_check == "Raw":

					#Check to see if animal had held cells, if not store blank
					#Read in table, skipping unnecessary headers
					try:
						frame = pd.read_table(files, sep='\s+',skiprows=2)

						#Determine order of days to store in correct output files
						date_1=frame.columns[4][(frame.columns[4]).rfind('_')-6:-7]
						date_2=frame.columns[5][(frame.columns[5]).rfind('_')-6:-7]

						#Store units into array based on condition order
						if date_1<date_2:
							day_1_units = list(frame[str(frame.columns[4])])
							day_2_units = list(frame[str(frame.columns[5])])

							for x,y in zip(day_1_units,day_2_units):
							    day1.append(int(''.join(c for c in x if c.isdigit())))
							    day2.append(int(''.join(c for c in y if c.isdigit())))

							#append larger sets
							all_day1.append(day1); all_day2.append(day2)
							day1 = []; day2 = []    #Clear day array

						if date_1>date_2:
							day_1_units = list(frame[str(frame.columns[5])])
							day_2_units = list(frame[str(frame.columns[4])])

							for x,y in zip(day_1_units,day_2_units):
							    day1.append(int(''.join(c for c in x if c.isdigit())))
							    day2.append(int(''.join(c for c in y if c.isdigit())))

							#append larger sets
							all_day1.append(day1); all_day2.append(day2)
							day1 = []; day2 = []    #Clear day array

					except:
						#append larger sets
						all_day1.append([]); all_day2.append([])
						print('%s has no held units, sorry buddy!' %(files[:4]))

	if type_check == "C.A.R.":
		#Remove 'Day' from all lists
		for sublist in all_day1:
		    del sublist[0]

		for sublist in all_day2:
		    del sublist[0]

	#Account for out of order units
	sorted_day1 = [];sorted_day2=[]
	for animal in range(len(all_day1)):
		sorted_day1.append(sorted(all_day1[animal], key=int))
		sorted_day2.append(sorted(all_day2[animal], key=int))

#Create list of animal variable for ease of use
animals = []

#Flip through each directory and perform tensor pca (store in larger list)
all_factors = []; all_data = [];
file_count=0;held_count1=0; held_count2=0;
for dir_name in dirs:
	# Change to the directory
	os.chdir(dir_name)
	# Locate the hdf5 file
	file_list = os.listdir('./')
	hdf5_name = ''
	for files in file_list:
		if files[-2:] == 'h5':
			hdf5_name = files

	# Open the hdf5 file
	hf5 = tables.open_file(hdf5_name, 'r')

	#Animal
	hdf5_animal = hdf5_name[:4]
	animals.append(hdf5_animal)

	# Get the spike trains for each taste available
	trains_dig_in = hf5.list_nodes('/spike_trains')
	train_in = []
	train_in_pathname = []
	for node in trains_dig_in:
		train_in_pathname.append(node._v_pathname)
		#extract only held units
		if file_count % 2 == 0:
			if len(sorted_day1[held_count1]) > 0:
				exec("train_in.append(hf5.root.spike_trains.%s.spike_array[:,sorted_day1[held_count1],:])" \
				  % train_in_pathname[-1].split('/')[-1])
			else:
				train_in = []
		if file_count % 2 != 0:
			if len(sorted_day2[held_count2]) > 0:
				exec("train_in.append(hf5.root.spike_trains.%s.spike_array[:,sorted_day2[held_count2],:])" \
				  % train_in_pathname[-1].split('/')[-1])
			else:
				train_in = []

	if file_count % 2 == 0:
		held_count1+=1

	if file_count % 2 != 0:
		held_count2+=1

	#append together
	all_data.append(train_in)
	hf5.close()

	#update counter
	file_count+=1

#remove files that have no held units
splice = [x for x in zip(all_data,animals) if x[0] != []]
all_data = [x[0] for x in splice]
animals = [x[1] for x in splice]
animals = sorted(list(set(animals)))

# =============================================================================
# =============================================================================
# # #EXPOSE DIFFERENCES ACROSS TRIALS
# =============================================================================
# =============================================================================
#stack conditions, calculate firing rate, and zscore
counter = 0; merged = []; all_factors=[]
for file in range(len(all_data)//2):
	trial_arrays_1 = all_data[counter]
	trial_arrays_2 = all_data[counter+1]

	#slice to even out trials
	min_trials = min(np.min(np.unique([x.shape[0] for x in trial_arrays_1])),\
				  np.min(np.unique([x.shape[0] for x in trial_arrays_2])))

	trial_array_1_min = np.array(trial_arrays_1)[:,0:min_trials,:,:]
	trial_array_2_min = np.array(trial_arrays_2)[:,0:min_trials,:,:]

	#stack conditions along trial dimension
	cond_stacked = np.concatenate((trial_array_1_min,trial_array_2_min),axis=1)

	#Flip through each taste and fit tensor
	neuron_FRs = []; all_FRzscore = []
	for neuron in range(cond_stacked.shape[2]):

		#Calculate Firing rate
		fr_array = _calc_firing_rates(25,250,1,cond_stacked[:,:,neuron,:])

		fr_array += np.random.random(fr_array.shape)*1e-6
		zscore_fr_array = np.array([stats.zscore(x,axis=None) \
								for x in fr_array.swapaxes(0,1)])
		all_FRzscore.append(zscore_fr_array)

	all_FRzscore = np.array(all_FRzscore)
	taste_factors = []
	for taste in range(all_FRzscore.shape[2]):

		#Fit CP tensor decomposition on Zscored array
		U = tt.cp_als(all_FRzscore[:,:,taste,:], rank=4, verbose=True)
		taste_factors.append(list(U.factors))

		#fig, _, _ = tt.plot_factors(U.factors)
	counter+=2
	all_factors.append(taste_factors)

#Plotting a different way
#Start counter
counter = 0
for animal in range(len(all_factors)):
	trial_arrays_1 = [x[1][:x[1].shape[0]//2] for x in all_factors[animal]]
	trial_arrays_2 = [x[1][x[1].shape[0]//2:] for x in all_factors[animal]]

	cell_arrays_1 = [x[0] for x in all_factors[animal]]
	cell_arrays_2 = [x[0] for x in all_factors[animal]]

	time_arrays_1 = [x[2] for x in all_factors[animal]]
	time_arrays_2 = [x[2] for x in all_factors[animal]]

	#Extract means, std, and sem across tastes
	t_means_1 = np.mean(np.array(trial_arrays_1),axis=0)
	t_means_2 = np.mean(np.array(trial_arrays_2),axis=0)

	t_std_1 = np.std(np.array(trial_arrays_1),axis=0)
	t_std_2 = np.std(np.array(trial_arrays_2),axis=0)

	t_sem_1 = stats.sem(np.array(trial_arrays_1),axis=0)
	t_sem_2 = stats.sem(np.array(trial_arrays_2),axis=0)

	#plot by rank
	#Initiate figure
	fig,axes = plt.subplots(nrows=2, ncols=2,sharex=True, sharey=True,
		        figsize=(12, 8), squeeze=False)
	fig.text(0.075, 0.5,'Factor Value', va='center',
	        rotation='vertical',fontsize=15, fontweight='bold')
	fig.text(0.5, 0.05,'Trial Number', ha='center',\
		  fontsize=15, fontweight='bold')
	axes_list = [item for sublist in axes for item in sublist]

	for rank in range(t_means_1.shape[1]):
		ax = axes_list.pop(0)

		ax.plot(t_means_1[:,rank],color='salmon')
		ax.fill_between(np.arange(0,t_means_1[:,rank].shape[0]),\
						 t_means_1[:,rank]-t_sem_1[:,rank],\
						 t_means_1[:,rank]+t_sem_1[:,rank],\
						 color='salmon',alpha=0.3)

		ax.plot(t_means_2[:,rank],color='midnightblue')
		ax.fill_between(np.arange(0,t_means_2[:,0].shape[0]),\
						 t_means_2[:,rank]-t_sem_2[:,rank],\
						 t_means_2[:,rank]+t_sem_2[:,rank],\
						 color='midnightblue',alpha=0.2)

		ax.set_title('Rank: %s' %(rank))

	plt.suptitle('%s\nTensor PCA: Trials' %(sorted(animals)[animal]),fontsize=20,fontweight='bold')
	fig.savefig(tensor_dir_name + '/%s_TensorTrial.png' %(sorted(animals)[animal]))
	plt.close('all')

# =============================================================================
# =============================================================================
# # #EXPOSE DIFFERENCES ACROSS TIME
# =============================================================================
# =============================================================================
#stack conditions, calculate firing rate, and zscore
counter = 0; merged = []; all_factors=[]
for file in range(len(all_data)//2):
	trial_arrays_1 = all_data[counter]
	trial_arrays_2 = all_data[counter+1]

	#slice to even out trials
	min_trials = min(np.min(np.unique([x.shape[0] for x in trial_arrays_1])),\
				  np.min(np.unique([x.shape[0] for x in trial_arrays_2])))

	trial_array_1_min = np.array(trial_arrays_1)[:,0:min_trials,:,:]
	trial_array_2_min = np.array(trial_arrays_2)[:,0:min_trials,:,:]

	#stack conditions along trial dimension
	cond_stacked = np.concatenate((trial_array_1_min,trial_array_2_min),axis=-1)

	#Flip through each taste and fit tensor
	neuron_FRs = []; all_FRzscore = []
	for neuron in range(cond_stacked.shape[2]):

		#Calculate Firing rate
		fr_array = _calc_firing_rates(25,250,1,cond_stacked[:,:,neuron,:])

		fr_array += np.random.random(fr_array.shape)*1e-6
		zscore_fr_array = np.array([stats.zscore(x,axis=None) \
								for x in fr_array.swapaxes(0,1)])
		all_FRzscore.append(zscore_fr_array)

	all_FRzscore = np.array(all_FRzscore)
	taste_factors = []
	for taste in range(all_FRzscore.shape[2]):

		#Fit CP tensor decomposition on Zscored array
		U = tt.cp_als(all_FRzscore[:,:,taste,:], rank=4, verbose=True)
		taste_factors.append(list(U.factors))

		#fig, _, _ = tt.plot_factors(U.factors)
	counter+=2
	all_factors.append(taste_factors)

counter = 0
for animal in range(len(all_factors)):
	trial_arrays_1 = [x[1] for x in all_factors[animal]]
	trial_arrays_2 = [x[1] for x in all_factors[animal]]

	cell_arrays_1 = [x[0] for x in all_factors[animal]]
	cell_arrays_2 = [x[0] for x in all_factors[animal]]

	time_arrays_1 = [x[2][:((x[2].shape[0]-1)//2)] for x in all_factors[animal]]
	time_arrays_2 = [x[2][((x[2].shape[0]-1)//2):] for x in all_factors[animal]]

	t1 = [x[2][:((x[2].shape[0]-1)//2)] for x in all_factors[animal]]
	t2 = [x[2][((x[2].shape[0]-1)//2):] for x in all_factors[animal]]

	#Extract means, std, and sem across tastes
	t_means_1 = np.mean(np.array(time_arrays_1),axis=0)
	t_means_2 = np.mean(np.array(time_arrays_2),axis=0)

	t_std_1 = np.std(np.array(time_arrays_1),axis=0)
	t_std_2 = np.std(np.array(time_arrays_2),axis=0)

	t_sem_1 = stats.sem(np.array(time_arrays_1),axis=0)
	t_sem_2 = stats.sem(np.array(time_arrays_2),axis=0)

	#Set xticks
	xtick = np.arange(time_arrays_1[0].shape[0])*25

	#Initiate figure
	fig,axes = plt.subplots(nrows=2, ncols=2,sharex=True, sharey=True,
		        figsize=(12, 8), squeeze=False)
	fig.text(0.075, 0.5,'Factor Value', va='center',
	        rotation='vertical',fontsize=15, fontweight='bold')
	fig.text(0.5, 0.05,'Time from taste delivery (ms)', ha='center',\
		  fontsize=15, fontweight='bold')
	axes_list = [item for sublist in axes for item in sublist]

	for rank in range(t_means_1.shape[1]):
		ax = axes_list.pop(0)

		ax.plot(t_means_1[:,rank],color='salmon')
		ax.fill_between(np.arange(0,t_means_1[:,rank].shape[0]),\
						 t_means_1[:,rank]-t_sem_1[:,rank],\
						 t_means_1[:,rank]+t_sem_1[:,rank],\
						 color='salmon',alpha=0.3)

		ax.plot(t_means_2[:,rank],color='midnightblue')
		ax.fill_between(np.arange(0,t_means_2[:,0].shape[0]),\
						 t_means_2[:,rank]-t_sem_2[:,rank],\
						 t_means_2[:,rank]+t_sem_2[:,rank],\
						 color='midnightblue',alpha=0.2)

		ax.set_xticks(np.arange(0, time_arrays_1[0].shape[0], step=20))
		ax.set_xticklabels(np.arange(-2000, 5000, step=500))

		ax.set_title('Rank: %s' %(rank))
		ax.set_xlim(60,180)
		ax.axvline(80,linestyle = '--',color='black')
	plt.suptitle('%s\nTensor PCA: Time' %(sorted(animals)[animal]),fontsize=20,fontweight='bold')
	fig.savefig(tensor_dir_name + '/%s_TensorTime.png' %(sorted(animals)[animal]))
	plt.close('all')

	#update counter
	counter+=2
