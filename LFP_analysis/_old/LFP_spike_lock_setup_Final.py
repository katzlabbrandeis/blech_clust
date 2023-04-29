#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 10:23:58 2019

@author: abuzarmahmood/bradly

Created on Wed Feb 13 19:36:13 2019

"""
# =============================================================================
# Import stuff
# =============================================================================

# import Libraries
# Built-in Python libraries
import os # functions for interacting w operating system

# 3rd-party libraries
import numpy as np # module for low-level scientific computing
from scipy.signal import hilbert #Hilbert transform to determine the amplitude envelope and instantaneous frequency of an amplitude-modulated signal
from scipy.signal import butter
from scipy.signal import filtfilt
import pickle #for data storage and retreival
import matplotlib.pyplot as plt # makes matplotlib work like MATLAB. ’pyplot’ functions.
import easygui
import tables
from tqdm import trange
import pandas as pd

#plotting functionalities
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

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

# =============================================================================
# Define common variables
# =============================================================================
#frequency bands
iter_freqs = [
        ('Theta', 4, 7),
        ('Alpha', 8, 12),
        ('Beta', 13, 25),
        ('Gamma', 30, 45)]

colors = plt.get_cmap('winter_r')(np.linspace(0, 1, len(iter_freqs)))

# =============================================================================
# Identity set-up
# =============================================================================

msg   = "Is this a passive or taste session (ie. do you have more than one trial)?"
session_type = easygui.buttonbox(msg,choices = ["Passive","Tastes"])

# =============================================================================
# Create/Import directories
# =============================================================================

#Get name of directory where you want to save output files to
save_name = easygui.diropenbox(msg = 'Choose directory you want output files sent to (and/or where ".dir" files are)',default = '/home/bradly/drive2/spike_locking')

#Ask user if they have ran set-up before
msg   = "Have you performed directory set-up before (ie. do you have '.dir' files in output folder) ?"
dir_check = easygui.buttonbox(msg,choices = ["Yes","No"])

if dir_check == "No":
    #Get data_saving name
    msg   = "What condition are you analyzing first?"
    subplot_check1 = easygui.buttonbox(msg,choices = ["Saline","LiCl","Other"])
    
    # Ask the user for the hdf5 files that need to be plotted together (fist condition)
    dirs_1 = []
    while True:
    	dir_name = easygui.diropenbox(msg = 'Choose first condition directory with a hdf5 file, hit cancel to stop choosing')
    	try:
    		if len(dir_name) > 0:	
    			dirs_1.append(dir_name)
    	except:
    		break   
    
    # Ask the user for the hdf5 files that need to be plotted together (second condition)
    #Get data_saving name
    msg   = "What condition are you analyzing second?"
    subplot_check2 = easygui.buttonbox(msg,choices = ["Saline","LiCl","Other"])
    
    dirs_2 = []
    while True:
    	dir_name = easygui.diropenbox(msg = 'Choose second condition directory with a hdf5 file, hit cancel to stop choosing')
    	try:
    		if len(dir_name) > 0:	
    			dirs_2.append(dir_name)
    	except:
    		break
    
    #Dump the directory names into chosen output location for each condition
    #condition 1
    completeName_1 = os.path.join(save_name, 'dirs_cond1.dir') 
    f_1 = open(completeName_1, 'w')
    for item in dirs_1:
        f_1.write("%s\n" % item)
    f_1.close()
    
    #condition 2
    completeName_2 = os.path.join(save_name, 'dirs_cond2.dir') 
    f_2 = open(completeName_2, 'w')
    for item in dirs_2:
        f_2.write("%s\n" % item)
    f_2.close()

if dir_check == "Yes":
    
    #Get data_saving name
    msg   = "What condition are you analyzing first?"
    subplot_check1 = easygui.buttonbox(msg,choices = ["Saline","LiCl","Other"])
    
    #establish directories to flip through
    #condition 1
    dirs_1_path = os.path.join(save_name, 'dirs_cond1.dir')
    dirs_1_file = open(dirs_1_path,'r')
    dirs_1 = dirs_1_file.read().splitlines()
    dirs_1_file.close()
    
    #Get data_saving name
    msg   = "What condition are you analyzing second?"
    subplot_check2 = easygui.buttonbox(msg,choices = ["Saline","LiCl","Other"])
    
    #condition 2
    dirs_2_path = os.path.join(save_name, 'dirs_cond2.dir')
    dirs_2_file = open(dirs_2_path,'r')
    dirs_2 = dirs_2_file.read().splitlines()
    dirs_2_file.close()

#modify save names
if subplot_check1==subplot_check2:
    subplot_check2=subplot_check2+'_2'    

# =============================================================================
# Processing
# =============================================================================
all_conds = [subplot_check1]*len(dirs_1)+[subplot_check2]*len(dirs_2) #merge conditional lists
all_dirs = dirs_1+dirs_2

#flip through merged directories
dir_count=0; all_phase_frame = pd.DataFrame()
for dir_name in all_dirs:
	#Change to the directory
	os.chdir(dir_name)
	#Locate the hdf5 file
	file_list = os.listdir('./')
	hdf5_name = ''
	for files in file_list:
		if files[-2:] == 'h5':
			hdf5_name = files

	#Open the hdf5 file
	hf5 = tables.open_file(hdf5_name, 'r')
	
	# Make directory where .hdf5 is to store the phase plots. Delete and remake the directory if it exists
	try:
		os.system('rm -r '+'./Phase_lock_all_units')
	except:
		pass
	os.mkdir('./Phase_lock_all_units')

	#Open the hdf5 file
	hf5 = tables.open_file(hdf5_name, 'r')
	
	# Pull LFPS and spikes
	lfps_dig_in = hf5.list_nodes('/Parsed_LFP')
	trains_dig_in = hf5.list_nodes('/spike_trains')
	lfp_array = np.asarray([lfp[:] for lfp in lfps_dig_in])
	spike_array = np.asarray([spikes.spike_array[:] for spikes in trains_dig_in])
	
	#create subdirectories for storing plots
	for dig_in in trains_dig_in:
		os.mkdir('./Phase_lock_all_units/'+str.split(dig_in._v_pathname, '/')[-1])
 	
	hf5.close()
# =============================================================================
# Calculate phases
# =============================================================================

	# Create processed lfp arrays
	analytic_signal_array = np.zeros((len(iter_freqs),) + lfp_array.shape, dtype = np.complex128)
	phase_array = np.zeros((len(iter_freqs),) + lfp_array.shape)
	
	for band in trange(len(iter_freqs), desc = 'bands'):
	    for taste in range(lfp_array.shape[0]):
	        for channel in range(lfp_array.shape[1]):
	            for trial in range(lfp_array.shape[2]):
	                band_filt_sig = butter_bandpass_filter(data = lfp_array[taste,channel,trial,:], 
	                                                       lowcut = iter_freqs[band][1], 
	                                                       highcut =  iter_freqs[band][2], 
	                                                       fs = 1000)
	                analytic_signal = hilbert(band_filt_sig)
	                instantaneous_phase = np.angle(analytic_signal)
	                
	                analytic_signal_array[band,taste,channel,trial,:] = analytic_signal
	                phase_array[band,taste,channel,trial,:] = instantaneous_phase
	     	
	# =============================================================================
	# Use mean LFP (across channels) to calculate phase (since all channels have same phase)
	# =============================================================================
	mean_analytic_signal = np.mean(analytic_signal_array,axis=2)
	mean_phase_array = np.angle(mean_analytic_signal)
	
	# =============================================================================
	# Plot phase from raw signal with mean signal across lfp channels and perform inlay view
	# =============================================================================
	
	fig,axes = plt.subplots(nrows=2, ncols=2,sharex=True, sharey=True,figsize=(20,12), squeeze=False)
	fig.text(0.08, 0.5, 'Period', va='center', rotation='vertical',fontsize=18)
	fig.text(0.5, 0.05, 'Time (ms)', ha='center',fontsize=18)
	axes_list = [item for sublist in axes for item in sublist]
	for ax,band,color in zip(axes.flatten(), range(len(iter_freqs)),colors):
		#formatting main plots
		ax = axes_list.pop(0); ax.tick_params(labelsize=18)
		ax.set_title('Band: %s (%i-%iHz)' %(iter_freqs[band][0],iter_freqs[band][1],iter_freqs[band][2]),size=18)
		ax.set_xlim(0,phase_array[band,0,0:-1,0,:1000].shape[1]); ax.set_ylim(-1*np.pi,np.pi);
		ax.set_yticks(np.linspace(-1*np.pi,np.pi,5));labels=[r"-$\pi$",r"-$\pi/2$","$0$",r"$\pi/2$",r"$\pi$"];
		ax.set_yticklabels(labels);  
		
		#normal data plot set-up
		ax.plot(phase_array[band,0,0:-1,0,:1000].T,color=color,alpha=0.3)
		ax.plot(phase_array[band,0,-1:,0,:1000].T,color=color,alpha=0.3,label='raw phase')
		ax.plot(mean_phase_array[band,0,0,:1000],linestyle='--',color='black',alpha=0.8,label='mean phase'); 
		ax.legend(loc=2,prop={'size': 14}) 
# =============================================================================
# 		#zoomed data plot set-up
# 		axins = zoomed_inset_axes(ax, 6, loc=2) #zoom factor
# 		axins.plot(phase_array[band,0,0:-1,0,:].T,color=color,alpha=0.3)
# 		axins.plot(phase_array[band,0,-1:,0,:].T,color=color,alpha=0.3,label='raw phase')
# 		axins.plot(mean_phase_array[band,0,0,:],linestyle='--',color='black',alpha=0.8,label='mean phase');  
# 		x1, x2, y1, y2 = 2000, 2500, -1*np.pi/2, 0 # specify the limits
# 		axins.set_xlim(x1, x2);  axins.set_ylim(y1, y2) # apply the x and y-limits
# 		plt.yticks(visible=False); plt.xticks(visible=False)	#hide inlay ticks
# 		mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.1",linewidth=3)
# 		
# =============================================================================
	fig.suptitle('%s' %(hdf5_name[:4]) +'\n'+ 'Phase Allignment for %i LFP channels'%(phase_array[0,0,:,0,:].shape[0]),size=20)
	fig.subplots_adjust(hspace=0.3,wspace = 0.05)
	fig.savefig(dir_name+'/%s_LFP_channel_phase_traces.png' % (hdf5_name[:4]))
	plt.close(fig)
	
	# =============================================================================
	# Calculate phase locking: for every spike, find phase for every band
	# =============================================================================
	# Find spiketimes and convert to dataframe
	# Convert phases to dataframe and merge both to find at what phase spikes occured
	
	spike_times = spike_array.nonzero()
	spikes_frame = pd.DataFrame(data = {'taste':spike_times[0],
	                                    'trial':spike_times[1],
	                                    'unit':spike_times[2],
	                                    'time':spike_times[3]})
	# Create array index identifiers
	nd_idx_objs = []
	for dim in range(mean_phase_array.ndim):
	    this_shape = np.ones(len(mean_phase_array.shape))
	    this_shape[dim] = mean_phase_array.shape[dim]
	    nd_idx_objs.append(np.broadcast_to( np.reshape(np.arange(mean_phase_array.shape[dim]),this_shape.astype('int')), mean_phase_array.shape).flatten())
	
	
	phase_frame = pd.DataFrame(data = {'band' : nd_idx_objs[0].flatten(),
	                                   'taste' : nd_idx_objs[1].flatten(),
	                                   'trial' : nd_idx_objs[2].flatten(),
	                                   'time' : nd_idx_objs[3].flatten(),
	                                   'phase' : mean_phase_array.flatten()})
	
	# Merge : Gives dataframe with length of (bands x numner of spikes)
	final_phase_frame = pd.merge(spikes_frame,phase_frame,how='inner')
	
	# Add columns for animal and condition
	final_phase_frame['animal'], final_phase_frame['condition'], final_phase_frame['session_type'], final_phase_frame['filename']= [hdf5_name[:4],all_conds[dir_count],session_type,hdf5_name]
	all_phase_frame= all_phase_frame.append(final_phase_frame)
	dir_count+=1
	
#put into dump file
tuple_save = '%s_Phaselock_%i_%s_%i_%s.dump' %(session_type,len(dirs_1), subplot_check1,len(dirs_2), subplot_check2)
output_name =   os.path.join(save_name, tuple_save)
pickle.dump(all_phase_frame, open(output_name, 'wb'))  
 			
