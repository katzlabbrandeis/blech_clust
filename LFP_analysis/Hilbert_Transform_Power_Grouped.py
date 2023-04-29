#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 11:17:42 2019

@author: bradly
"""

# =============================================================================
# Import stuff
# =============================================================================

# import Libraries
# Built-in Python libraries
import os # functions for interacting w operating system
from os.path import basename #For saving purposes
from datetime import datetime

# 3rd-party libraries
import numpy as np # module for low-level scientific computing
import matplotlib.pyplot as plt # makes matplotlib work like MATLAB. ’pyplot’ functions.
import easygui
import tables
from tqdm import trange
import pandas as pd
import scipy as sp # library for working with NumPy arrays
from scipy import signal # signal processing module
from scipy.ndimage.filters import gaussian_filter
import matplotlib.cm as cm # colormap module
import re
import scipy.stats
from scipy.signal import hilbert 
from scipy.signal import butter
from scipy.signal import filtfilt

# =============================================================================
# #Define Functions to be used
# =============================================================================

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

#define bandpass filter parameters to parse out frequencies
def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut /nyq
    high = highcut/nyq
    b, a = butter(order, [low, high], btype='bandpass')
    y = filtfilt(b, a, data)
    return y

# =============================================================================
# #Establish files to work on
# =============================================================================

#Get name of directory where you want to save output files to
save_name = easygui.diropenbox(msg = 'Choose directory you want output files sent to (and/or where ".dir" files are)',default = '/home/bradly/drive2/data/Affective_State_Protocol/LiCl/Combined_Passive_Data/LiCl_Saline')

#Ask user if they have ran set-up before
msg   = "Have you performed directory set-up before (ie. do you have '.dir' files in output folder) ?"
dir_check = easygui.buttonbox(msg,choices = ["Yes","No"])

if dir_check == "No":
    # Ask the user for the hdf5 files that need to be plotted together (fist condition)
    dirs_1 = []
    while True:
    	dir_name = easygui.diropenbox(msg = 'Choose first condition directory with a hdf5 file, hit cancel to stop choosing')
    	try:
    		if len(dir_name) > 0:	
    			dirs_1.append(dir_name)
    	except:
    		break   

    
    #Dump the directory names into chosen output location for each condition
    #condition 1
    completeName_1 = os.path.join(save_name, 'dirs_cond1.dir') 
    f_1 = open(completeName_1, 'w')
    for item in dirs_1:
        f_1.write("%s\n" % item)
    f_1.close()


if dir_check == "Yes":
    #establish directories to flip through
    #condition 1
    dirs_1_path = os.path.join(save_name, 'dirs_cond1.dir')
    dirs_1_file = open(dirs_1_path,'r')
    dirs_1 = dirs_1_file.read().splitlines()
    dirs_1_file.close()

# =============================================================================
# #Estalbish parameters
# =============================================================================
taste_params = easygui.multenterbox(msg = 'Input taste identities:', fields = ['Taste 1 (dig_in_1)', 'Taste 2 (dig_in_2)','Taste 3 (dig_in_3)','Taste 4 (dig_in_4)'],values = ['NaCl','Sucrose','Citric Acid','QHCl'])
analysis_params = easygui.multenterbox(msg = 'Input analysis paramters:', fields = ['Pre-stimulus signal duration (ms; from set-up)','Post-stimulus signal duration (ms; from set-up)','Pre-Taste array start time (ms)', 'Taste array end time (ms)', 'Sampling Rate (samples per second)', 'Signal Window (ms)', 'Window Overlap (ms; default 90%)'], values = ['2000','5000','0','2500','1000','900','850'])
    
#create timing variables
pre_stim = int(analysis_params[0])
post_stim = int(analysis_params[1])
lower = int(analysis_params[2])
upper = int(analysis_params[3])
Fs = int(analysis_params[4])
signal_window = int(analysis_params[5])
window_overlap = int(analysis_params[6])
base_time = 2000   
  
#establish meshing and plotting paramters
plotting_params = easygui.multenterbox(msg = 'Input plotting paramters:', fields = ['Minimum frequency (Hz):','Maximum frequency (Hz):', 'Pre-stim plot time (ms):', 'Post-stim plot time (ms):'], values = ['3','40', '1000',int(upper)])

#specify frequency bands
iter_freqs = [
        ('Theta', 4, 7),
        ('Mu', 7, 12),
        ('Beta', 13, 25),
        ('Gamma', 30, 45)]

#Create blank Dataframe for storing all animals
grouped_df=pd.DataFrame()

#Create blank list to store animal names
all_animals = [[] for x in range(len(dirs_1))]

#Start counter
file_count =0

#Flip through files and extract LFP data and store into large arrays
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
	hf5 = tables.open_file(hdf5_name, 'r+')

	#Ask if file needs to be split, if yes, split it
	split_response = easygui.indexbox(msg='Do you need to split these trials? \nFile Name: %s' %(hdf5_name), title='Split trials', choices=('Yes', 'No'), image=None, default_choice='Yes', cancel_choice='No')
	total_trials = hf5.root.Parsed_LFP.dig_in_1_LFPs[:].shape[1]
	dig_in_channels = hf5.list_nodes('/digital_in')
	dig_in_LFP_nodes = hf5.list_nodes('/Parsed_LFP')
	
	if split_response == 0:
	    trial_split = easygui.multenterbox(msg = 'Put in the number of trials to parse from each of the LFP arrays (only integers)', fields = [node._v_name for node in dig_in_LFP_nodes], values = ['19' for node in dig_in_LFP_nodes])
	    #Convert all values to integers
	    trial_split = list(map(int,trial_split))
	    total_sessions = int(total_trials/int(trial_split[0]))
	    #Create dictionary of all parsed LFP arrays
	    LFP_data = [np.array(dig_in_LFP_nodes[node][:,0:trial_split[node],:]) for node in range(len(dig_in_LFP_nodes))]
	    
	else:    
	    total_sessions = 1
	    trial_split = list(map(int,[total_trials for node in dig_in_LFP_nodes]))
	    #Create dictionary of all parsed LFP arrays
	    LFP_data = [np.array(dig_in_LFP_nodes[node][:]) for node in range(len(dig_in_LFP_nodes))]
	    	
	# =============================================================================
	# #Channel Check Processing
	# =============================================================================
	channel_data = np.mean(LFP_data[0],axis=1).T #Assumes same number of tirlas per taste
	
	#Ask user to check LFP traces to ensure channels are not shorted/bad in order to remove said channel from further processing
	channel_check = 	easygui.multchoicebox(msg = 'Choose the channel numbers that you want to REMOVE from further analyses. Click clear all and ok if all channels are good', choices = tuple([i for i in range(np.size(channel_data,axis=1))]))
	if channel_check:
		for i in range(len(channel_check)):
			channel_check[i] = int(channel_check[i])
	
	#set channel_check to an empty list if no channels were chosen
	if channel_check is None:
		channel_check = []
	channel_check.sort()
	
	cleaned_LFP = []
	for taste in range(len(LFP_data)):		
		cleaned_LFP.append(np.delete(LFP_data[taste][:], channel_check, axis=0))
	
	#Create blank dataframe    
	df = pd.DataFrame(columns=range(np.array(cleaned_LFP[0][0][1].shape)[0]+1)) #Add one columns for descriptors
	df.rename(columns={0:'Taste'}) #Set Descriptors
	
	#Append dataframe      
	for taste in range(len(cleaned_LFP)):
		#Establish lengths for stacking
	    m,n,r = cleaned_LFP[taste].shape
		
		#Stack data
	    out_arr = np.column_stack((np.repeat(np.arange(m),n),cleaned_LFP[taste].reshape(m*n,-1)))
	    
		#Create data frame and add descriptor columns
	    outdf=pd.DataFrame(out_arr)
	    outdf.insert(0,'Taste',taste)
	    df = pd.concat([df,outdf],sort=False)
	    
	#Reset column order and add column name for Channel
	cols = list(df.columns)
	cols = [cols[-1]] + cols[:-1]
	df = df[cols]	
	df.rename(columns={0:'Channel'},inplace=True)
	df.insert(0,'Animal',hdf5_name[0:4])
	
	#Stack dataframes onto eachother
	grouped_df = pd.concat([grouped_df,df],sort=True)
	
	#store animal name
	all_animals[file_count] = hdf5_name[0:4]
	file_count+=1
	#Close the hdf5 file
	hf5.close()

#reset dataframe index
grouped_df.reset_index(drop=True, inplace=True)

#Further dataframe clean up
#Ask user to indicate which tastes to remove (if any) based on animal files
taste_check = easygui.multchoicebox(msg = 'Choose the animals numbers that you want to REMOVE tastes from further analyses. Click clear all and ok if all tastes are good', choices = tuple([all_animals[i] for i in range(np.size(all_animals,axis=0))]))
if taste_check:
	for i in range(len(taste_check)):
		taste_remove = easygui.multchoicebox(msg = 'Choose the taste numbers that you want to REMOVE from further analyses. Click clear all and ok if all tastes are good. \nAnimal: %s' %(taste_check[i]), choices = tuple([taste_params[x] for x in range(np.size(taste_params,axis=0))]))
		for j in range(len(taste_remove)):
			indexNames = grouped_df[(grouped_df['Animal'] == taste_check[i]) & (grouped_df['Taste'] == taste_params.index(taste_remove[j]))].index
			grouped_df.drop(indexNames , inplace=True)

#Store into HDF5 file
cwd = os.getcwd()
grouped_df.to_hdf(save_name+'/%s_LFP_N%i_%s.h5' %(basename(save_name), len(grouped_df.Animal.unique()),datetime.today().strftime('%Y%m%d')), key=basename(save_name))

# =============================================================================
# #Bandfilter signal and produce figures
# =============================================================================
# Make directory to store the LFP trace plots. Delete and remake the directory if it exists
try:
	os.system('rm -r '+save_name+'/LFP_signals')
except:
	pass
os.mkdir(save_name+'/LFP_signals')	
	
#Set t vector for plotting
t = np.array(list(range(0,np.size(channel_data,axis=0))))-pre_stim
	
#Flip through tastes and bands to produce figures to assess GEP
for taste in range(len(grouped_df['Taste'].unique())):
    #Query data
	query = grouped_df.query('Taste== @taste')

	fig = plt.figure(figsize=(11,8))
	fig,axes = plt.subplots(nrows=len(iter_freqs), ncols=2,sharex=True, sharey=False,figsize=(12, 8), squeeze=True)
	fig.text(0.5, 0.05, 'Milliseconds', ha='center',fontsize=15)
	axes_list = [item for sublist in axes for item in sublist]
	
	#Flip through bands and create figures	
	for ax,band in zip(axes.flatten(),trange(len(iter_freqs), desc = 'bands')):
		band_filt_sig = butter_bandpass_filter(data = query.iloc[:,3:], 
	                                    lowcut = iter_freqs[band][1], 
	                                    highcut =  iter_freqs[band][2], 
	                                    fs = 1000)
		analytic_signal = hilbert(band_filt_sig)
		instantaneous_phase = np.angle(analytic_signal)
		x_power = np.abs(analytic_signal)**2
		
		#Plot raw versus filtered LFP
		ax = axes_list.pop(0)
		ax.plot(t,np.mean(query.iloc[:,3:].T,axis=1),'k-',alpha=0.3,lw=1); 
		ax.plot(t,np.mean(analytic_signal.T,axis=1),'r',lw=1);
		ax.set_xlim([-int(plotting_params[2]),int(plotting_params[3])])
		ax.set_ylim([np.min(np.mean(query.iloc[:,3:].T,axis=1)),100])
		ax.vlines(x=0, ymin=np.min(np.mean(query.iloc[:,3:].T,axis=1)),ymax=100, linewidth=3, color='k',linestyle=':')
		ax.text(0.83,0.9,'%s (%i - %iHz)' %(iter_freqs[band][0],iter_freqs[band][1],iter_freqs[band][2]), ha='center', va='center', transform=ax.transAxes)
		
		#Plot Power over time
		ax = axes_list.pop(0)
		ax.plot(t,np.mean(x_power.T,axis=1)); 
		ax.set_xlim([-int(plotting_params[2]),int(plotting_params[3])])
		ax.vlines(x=0, ymin=np.min(np.mean(x_power.T,axis=1)),ymax=np.max(np.mean(x_power.T,axis=1)), linewidth=3, color='k',linestyle=':')
		
	plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=-0.08)
	plt.suptitle('Hilbert Transform and Instantaneous Power \nTaste: %s' %(taste_params[taste]),size=16)
	fig.savefig(save_name+'/LFP_signals/' + '%s_N_%i_Animals' %(taste_params[taste],len(grouped_df.Animal.unique())) + '_HilbertTransform.png')   
	plt.show()
	plt.close(fig)

#Plot grouped tastants
fig = plt.figure(figsize=(11,8))
fig,axes = plt.subplots(nrows=len(iter_freqs), ncols=2,sharex=True, sharey=False,figsize=(12, 8), squeeze=True)
fig.text(0.5, 0.05, 'Milliseconds', ha='center',fontsize=15)
axes_list = [item for sublist in axes for item in sublist]

#Flip through bands and create figures	
for ax,band in zip(axes.flatten(),trange(len(iter_freqs), desc = 'bands')):
	band_filt_sig = butter_bandpass_filter(data = grouped_df.iloc[:,3:], 
                                    lowcut = iter_freqs[band][1], 
                                    highcut =  iter_freqs[band][2], 
                                    fs = 1000)
	analytic_signal = hilbert(band_filt_sig)
	instantaneous_phase = np.angle(analytic_signal)
	x_power = np.abs(analytic_signal)**2
	
	#Plot raw versus filtered LFP
	ax = axes_list.pop(0)
	ax.plot(t,np.mean(grouped_df.iloc[:,3:].T,axis=1),'k-',alpha=0.3,lw=1); 
	ax.plot(t,np.mean(analytic_signal.T,axis=1),'r',lw=1);
	ax.set_xlim([-int(plotting_params[2]),int(plotting_params[3])])
	ax.set_ylim([-50,50])
	ax.vlines(x=0, ymin=np.min(np.mean(grouped_df.iloc[:,3:].T,axis=1)),ymax=100, linewidth=3, color='k',linestyle=':')
	ax.text(0.83,0.9,'%s (%i - %iHz)' %(iter_freqs[band][0],iter_freqs[band][1],iter_freqs[band][2]), ha='center', va='center', transform=ax.transAxes)
	
	#Plot Power over time
	ax = axes_list.pop(0)
	ax.plot(t,np.mean(x_power.T,axis=1)); 
	ax.set_xlim([-int(plotting_params[2]),int(plotting_params[3])])
	ax.vlines(x=0, ymin=np.min(np.mean(x_power.T,axis=1)),ymax=np.max(np.mean(x_power.T,axis=1)), linewidth=3, color='k',linestyle=':')
	
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=-0.08)
plt.suptitle('Hilbert Transform and Instantaneous Power \nAll Tastes',size=16)
fig.savefig(save_name+'/LFP_signals/' + 'AllTastes_N_%i_Animals' %(len(grouped_df.Animal.unique())) + '_HilbertTransform.png')   
plt.show()
plt.close(fig)	






















	
	
	
	
	
	