#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 11:26:55 2019

@author: bradly
"""
# =============================================================================
# Import stuff
# =============================================================================

# import Libraries
# Built-in Python libraries
import os # functions for interacting w operating system

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

# =============================================================================
# #Establish User inputs for Variables
# =============================================================================

#Ask if file needs to be split, if yes, split it
split_response = easygui.indexbox(
        msg='Do you need to split these trials?', 
        title='Split trials', choices=('Yes', 'No'), 
        image=None, default_choice='Yes', cancel_choice='No')

# Ask if this analysis is looking at more than 1 trial and/or taste
msg   = "Do you want to perform LFP analyses for more than ONE trial" \
                    "(ie. Do you have several tastes) ?"
trial_check = easygui.buttonbox(msg,choices = ["Yes","No"])

if trial_check == "Yes":
    total_trials = hf5.root.Parsed_LFP.dig_in_1_LFPs[:].shape[1]
    # Ask about subplotting
    msg   = "Do you want saved outputs for each tastant?"
    subplot_check = easygui.buttonbox(msg,choices = ["Yes","No"])

if trial_check == "No":
    total_trials = 1

dig_in_channels = hf5.list_nodes('/digital_in')
dig_in_LFP_nodes = hf5.list_nodes('/Parsed_LFP')

if split_response == 0:
    trial_split = easygui.multenterbox(
            msg = "Put in the number of trials to parse from each of "\
                    "the LFP arrays (only integers)", 
                    fields = [node._v_name for node in dig_in_LFP_nodes], 
                    values = ['15' for node in dig_in_LFP_nodes])

    #Convert all values to integers
    trial_split = list(map(int,trial_split))
    total_sessions = int(total_trials/int(trial_split[0]))

    #Create dictionary of all parsed LFP arrays
    LFP_data = [np.array(dig_in_LFP_nodes[node][:,0:trial_split[node],:]) \
            for node in range(len(dig_in_LFP_nodes))]
    
else:    
    total_sessions = 1
    trial_split = list(map(int,[total_trials for node in dig_in_LFP_nodes]))
    #Create dictionary of all parsed LFP arrays
    LFP_data = [np.array(dig_in_LFP_nodes[node][:]) \
            for node in range(len(dig_in_LFP_nodes))]
    
#Establish timing parameters
if trial_check == "No":
    analysis_params = easygui.multenterbox(
            msg = 'Input analysis paramters:', 
            fields = ['Taste array start time (ms)', 
                    'Taste array end time (ms)', 
                    'Sampling Rate (samples per second)', 
                    'Signal Window (ms)', 
                    'Window Overlap (ms; default 90%)'], 
            values = ['0','1200000','1000','1000','900'])
        
    #create timing variables
    pre_stim = 0
    lower = int(analysis_params[0])
    upper = int(analysis_params[1])
    Fs = int(analysis_params[2])
    signal_window = int(analysis_params[3])
    window_overlap = int(analysis_params[4])
    
    #establish meshing and plotting paramters
    plotting_params = easygui.multenterbox(
            msg = 'Input plotting paramters:', 
            fields = ['Minimum frequency (Hz):','Maximum frequency (Hz):'], 
            values = ['3','40'])
    
    base_time = lower 
else:    
    analysis_params = easygui.multenterbox(
            msg = 'Input analysis paramters:', 
            fields = ['Pre-stimulus signal duration (ms; from set-up)',
                    'Post-stimulus signal duration (ms; from set-up)',
                    'Pre-Taste array start time (ms)', 
                    'Taste array end time (ms)', 
                    'Sampling Rate (samples per second)', 
                    'Signal Window (ms)', 
                    'Window Overlap (ms; default 90%)'], 
            values = ['2000','5000','0','2500','1000','900','850'])
    
    #create timing variables
    pre_stim = int(analysis_params[0])
    post_stim = int(analysis_params[1])
    lower = int(analysis_params[2])
    upper = int(analysis_params[3])
    Fs = int(analysis_params[4])
    signal_window = int(analysis_params[5])
    window_overlap = int(analysis_params[6])
    
    #establish meshing and plotting paramters
    plotting_params = easygui.multenterbox(
            msg = 'Input plotting paramters:', 
            fields = ['Minimum frequency (Hz):',
                    'Maximum frequency (Hz):', 
                    'Pre-stim plot time (ms):', 
                    'Post-stim plot time (ms):'], 
            values = ['3','40', '1000',int(upper)])

    # Ask if this analysis is an average of normalization 
    # (not supported for passive sitting yet)
    msg   = "Do you want to normalize to baseline (time before stimulus) "\
            "or have a mean (post-stimulus)?"
    analysis_type = easygui.buttonbox(msg,choices = ["Normalize","Mean"])
    taste_params = easygui.multenterbox(
            msg = 'Input taste identities:', 
            fields = ['Taste 1 (dig_in_1)', 
                    'Taste 2 (dig_in_2)',
                    'Taste 3 (dig_in_3)',
                    'Taste 4 (dig_in_4)'],
            values = ['NaCl','Sucrose','Citric Acid','QHCl'])
    
    #Adjust array parameters
    if analysis_type == "Normalize":
        analysis_params_2 = easygui.multenterbox(
                msg = 'Input normalizing paramter:', 
                fields = ['Baseline (pre-stimulus signal duration (ms))'], 
                values = ['2000'])
        base_time = int(analysis_params_2[0])
        
    else:
        base_time = 2000    

#Detail selections to user
print('analysis will use signals: ' + str(base_time) + 'ms' + \
        ' before stim onset & ' + str(upper) + 'ms after stim onset')

#specify frequency bands
iter_freqs = [
        ('Theta', 4, 7),
        ('Mu', 8, 12),
        ('Beta', 13, 25),
        ('Gamma', 30, 45)]

# =============================================================================
# #Channel Check
# =============================================================================
# Make directory to store the LFP trace plots. Delete and remake the directory if it exists
try:
        os.system('rm -r '+'./LFP_channel_check')
except:
        pass
os.mkdir('./LFP_channel_check')

#Check to make sure LFPs are "normal" and allow user to remove any that are not
for taste in range(len(LFP_data)):
        #Set data
        channel_data = np.mean(LFP_data[taste],axis=1).T
        t=np.array(list(range(0,np.size(channel_data,axis=0))))
        
        #Create figure
        fig,axes = plt.subplots(nrows=np.size(channel_data,axis=1), 
                ncols=1,sharex=True, sharey=False,figsize=(12, 8), squeeze=False)
        fig.text(0.5, 0.05, 'Milliseconds', ha='center',fontsize=15)
        axes_list = [item for sublist in axes for item in sublist]
        
        for ax, chan in zip(axes.flatten(),range(np.size(channel_data,axis=1))):
        
                ax = axes_list.pop(0)
                ax.set_yticks([])
                ax.plot(t, channel_data[:,chan])
                h = ax.set_ylabel('Channel %s' %(chan))
                h.set_rotation(0)
                ax.vlines(x=pre_stim, ymin=np.min(channel_data[:,chan]),
                        ymax=np.max(channel_data[:,chan]), linewidth=4, color='r')
                
        fig.subplots_adjust(hspace=0,wspace = -0.15)
        fig.suptitle('%s - Channel Check: %s' %(taste_params[taste], 
            hdf5_name[0:4])+'\n' + 'Raw LFP Traces; Date: %s' %(re.findall(r'_(\d{6})', 
                hdf5_name)[0]),size=16,fontweight='bold')
        fig.savefig('./LFP_channel_check/' + hdf5_name[0:4] + \
                '_ %s_%s' %(re.findall(r'_(\d{6})', hdf5_name)[0],
                    taste_params[taste]) + '_channelcheck.png')   
        plt.show()
        plt.close(fig)

# =============================================================================
# #Channel Check Processing
# =============================================================================

#Ask user to check LFP traces to ensure channels are not shorted/bad in order to remove said channel from further processing
channel_check = easygui.multchoicebox(msg = 'Choose the channel numbers that you want to REMOVE from further analyses. Click clear all and ok if all channels are good', choices = tuple([i for i in range(np.size(channel_data,axis=1))]))
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

#Plot all channels and tastes together
#Create figure
fig,axes = plt.subplots(nrows=len(cleaned_LFP), ncols=1,sharex=True, sharey=False,figsize=(12, 8), squeeze=False)
fig.text(0.5, 0.05, 'Milliseconds', ha='center',fontsize=15)
axes_list = [item for sublist in axes for item in sublist]

for ax,taste in zip(axes.flatten(),range(len(cleaned_LFP))):
        
        #Set data
        channel_data = np.mean(cleaned_LFP[taste],axis=1).T
        t=np.array(list(range(0,np.size(channel_data,axis=0))))
        ax = axes_list.pop(0)
        
        #Create color map
        colors = plt.get_cmap('winter_r')(np.linspace(0, 1, np.size(channel_data,axis=1)))
                                
        for chan in range(np.size(channel_data,axis=1)):
                ax.plot(t, channel_data[:,chan],color=colors[chan],alpha=0.35)
                        
                h = ax.set_ylabel('%s' %(taste_params[taste]),fontsize=15)
                h.set_rotation(0)
                ax.yaxis.set_label_position("right")
                ax.vlines(x=pre_stim, ymin=np.min(channel_data[:,chan]),ymax=np.max(channel_data[:,chan]), linewidth=4, color='r')
                
fig.subplots_adjust(hspace=0,wspace = -0.15)
fig.suptitle('All Taste Channel Check: %s' %(hdf5_name[0:4])+'\n' + 'Raw LFP Traces; Date: %s' %(re.findall(r'_(\d{6})', hdf5_name)[0]),size=16,fontweight='bold')
fig.savefig('./LFP_channel_check/' + hdf5_name[0:4] + '_ %s' %(re.findall(r'_(\d{6})', hdf5_name)[0]) + '_alltaste_channelcheck.png')   
plt.show()
plt.close(fig)

# =============================================================================
# #Create Dataframe
# =============================================================================
        
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


# =============================================================================
# #BEGIN PROCESSING
# =============================================================================
#Set Variables for processing based on user input criteria and data
array_start = pre_stim-base_time 
max_pow = ((pre_stim+post_stim)-window_overlap)/(signal_window-window_overlap)

#create mean arary to hold all taste LFP arrays
mean_array = []; mean_LFP_array= []; norm_array = []

#Set up for figure build
fig = plt.figure(figsize=(11,8))
fig,axes = plt.subplots(nrows=len(cleaned_LFP), ncols=1,sharex=True, sharey=False,figsize=(12, 8), squeeze=False)
fig.text(0.5, 0.05, 'Milliseconds', ha='center',fontsize=15)
axes_list = [item for sublist in axes for item in sublist]

for taste in trange(len(df['Taste'].unique()), desc = 'Taste'):
    #Query data
        query = df.query('Taste== @taste')
        
        #create processing variables 
        mean_dem = np.size(query,axis=0) #ChannelsXTrials
        mean_pwelch = [] #empy array for pwelch values
        big_norm_P = np.empty([mean_dem,np.int((signal_window/2)+1),np.int(max_pow)])
        big_mean_P = np.empty([mean_dem,np.int((signal_window/2)+1),np.int(max_pow)])
        
        for lfp in range(len(query)):
                
                 #Spectrogram calculations
                 f, t_spec, x_spec = sp.signal.spectrogram(sp.signal.detrend(query.iloc[lfp,2:]), fs=Fs, window='hanning', nperseg=signal_window, noverlap=signal_window-(signal_window-window_overlap), mode='psd')
                 x_mesh, y_mesh = np.meshgrid(t_spec, f[f<int(plotting_params[1])])      
                 
                 if analysis_type == "Normalize":
                         #establish time and power array values to normalize against        
                         condition = t_spec<=base_time/1000                          #create conditional array based on baseline-ing params
                         preP = np.mean(x_spec[:,condition],axis =1)                 #takes mean of power values of pre-taste times
                         meanPP = np.mean(x_spec[:,condition],axis=1)
                    
                         #normalize
                         normPP = (x_spec/np.broadcast_to(meanPP,(np.size(x_spec,axis=1),len(meanPP))).T)

                     #store in array
                         big_norm_P[lfp,:,:]=normPP

                 else:
                        #store in array
                         big_mean_P[lfp,:,:]= x_spec
                         
                         
                 # Use the signal processing module (scipy.signal) to estimate the power spectral density (PSD).
                 # PSD shows how the strength of a signal is distributed across a range of frequencies.
                 #Pwelch analysis
                 f_spec, psd_spec =sp.signal.welch(query.iloc[lfp,2:], Fs, window='hanning',nperseg=1000)
                 mean_pwelch.append(psd_spec)
        
        #Plot tastes together
        plt.subplot(2,2,taste+1)
        plt.title('Taste: %s' %(taste_params[taste]), size=14)
		
        if analysis_type == "Normalize":
			      z_mesh = (np.log10(gaussian_filter(np.mean(big_norm_P,axis=0),sigma=1)))[f<int(plotting_params[1])]
        if analysis_type == "Mean":
			      z_mesh = (np.log10(gaussian_filter(np.mean(big_mean_P,axis=0),sigma=1)))[f<int(plotting_params[1])]
		
		#Plot
        im= plt.pcolormesh(x_mesh,y_mesh, z_mesh,cmap=plt.cm.winter); 
        plt.xlabel('Time (s)', size = 16)
        plt.ylabel('Frequency (Hz)', size = 16)
        plt.vlines(x=pre_stim/1000, ymin=0,ymax=np.max(y_mesh), linewidth=4, color='r',linestyle=':')

cbar_ax = fig.add_axes([.91,.3,.03,.4])
fig.colorbar(im, cax=cbar_ax,label="PSD (dB/Hz)")
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.35)
plt.suptitle('Time-Frequency Profile of Local Field Potential: %s \nHanning: Signal Window = %i ms, Window Overlap= %i ms' %(hdf5_name[0:4],signal_window,window_overlap), size=16)
fig.savefig(hdf5_name[0:4] + '_%s' %(re.findall(r'_(\d{6})', hdf5_name)[0]) +'_%s' %(analysis_type) +'_all_tastes_spectrograms_hanning_w_' + f'{np.int((window_overlap/signal_window)*100)}%_overlap.png')                  
plt.close('all')  


# =============================================================================
# #Bandfilter signal and produce figures
# =============================================================================
# Make directory to store the LFP trace plots. Delete and remake the directory if it exists
try:
        os.system('rm -r '+'./LFP_signals')
except:
        pass
os.mkdir('./LFP_signals')

#Set t vector for plotting
t = np.array(list(range(0,np.size(channel_data,axis=0))))-pre_stim

#Flip through tastes and bands to produce figures to assess GEP
for taste in range(len(df['Taste'].unique())):
    #Query data
        query = df.query('Taste== @taste')

        fig = plt.figure(figsize=(11,8))
        fig,axes = plt.subplots(nrows=4, ncols=2,sharex=True, sharey=False,figsize=(12, 8), squeeze=True)
        fig.text(0.5, 0.05, 'Milliseconds', ha='center',fontsize=15)
        axes_list = [item for sublist in axes for item in sublist]
        
        #Flip through bands and create figures  
        for ax,band in zip(axes.flatten(),trange(len(iter_freqs), desc = 'bands')):
                band_filt_sig = butter_bandpass_filter(data = query.iloc[:,2:], 
                                            lowcut = iter_freqs[band][1], 
                                            highcut =  iter_freqs[band][2], 
                                            fs = 1000,order=2)
                analytic_signal = hilbert(band_filt_sig)
                instantaneous_phase = np.angle(analytic_signal)
                x_power = np.abs(analytic_signal)**2
                
                #Plot raw versus filtered LFP
                ax = axes_list.pop(0)
                ax.plot(t,np.mean(query.iloc[:,2:].T,axis=1),'k-',alpha=0.3,lw=1); 
                ax.plot(t,np.mean(analytic_signal.T,axis=1),'r',lw=1);
                ax.set_xlim([-int(plotting_params[2]),int(plotting_params[3])])
                ax.set_ylim([np.min(np.mean(query.iloc[:,2:].T,axis=1)),100])
                ax.vlines(x=0, ymin=np.min(np.mean(query.iloc[:,2:].T,axis=1)),ymax=100, linewidth=3, color='k',linestyle=':')
                ax.text(0.83,0.9,'%s (%i - %iHz)' %(iter_freqs[band][0],iter_freqs[band][1],iter_freqs[band][2]), ha='center', va='center', transform=ax.transAxes)
                
                #Plot Power over time
                ax = axes_list.pop(0)
                ax.plot(t,np.mean(x_power.T,axis=1)); 
                ax.set_xlim([-int(plotting_params[2]),int(plotting_params[3])])
                ax.vlines(x=0, ymin=np.min(np.mean(x_power.T,axis=1)),ymax=np.max(np.mean(x_power.T,axis=1)), linewidth=3, color='k',linestyle=':')
                
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=-0.08)
        plt.suptitle('Hilbert Transform and Instantaneous Power: %s \nTaste: %s; Date: %s' %(hdf5_name[0:4], taste_params[taste],(re.findall(r'_(\d{6})', hdf5_name)[0])),size=16)
        fig.savefig('./LFP_signals/' + hdf5_name[0:4] + '_%s_ %s' %(taste_params[taste],re.findall(r'_(\d{6})', hdf5_name)[0]) + '_HilbertTransform.png')   
        plt.show()
        plt.close('all')  












        
        if analysis_type=="Normalize":
# =============================================================================
#                collapsed = np.mean(big_norm_P,axis=0)
#                z_mesh = collapsed[f<int(plotting_params[1])]
#                t_index = np.arange(0,int(analysis_params[1])/Fs,1/Fs) # time array
#             
# =============================================================================
                 #append spectrogram matrix into all taste matrix
                 norm_array.append(big_norm_P)
            
        else:
# =============================================================================
#                collapsed = np.mean(big_mean_P,axis=0)
#                z_mesh = collapsed[f<int(plotting_params[1])]
#                t_index = np.arange(0,(pre_stim+upper)/Fs,1/Fs)
# =============================================================================
                
                #append spectrogram matrix into all taste matrix
                mean_array.append(big_mean_P)
  
          
  
        #Smooth image with gaussian wrap
        z_mesh = gaussian_filter(z_mesh,sigma=1.5,mode='wrap')
        z_mesh -= np.min(z_mesh)    #normalize color
        z_mesh /= np.max(z_mesh)    #normalize color
        
        if analysis_type=="Normalize":
                if subplot_check == "Yes":
                        #Plotting
                        fig = plt.figure(figsize=(11,7))
                        plt.subplot(1,1,1)
                        plt.title('Taste: %s \nHanning: Signal Window = %i ms, Window Overlap= %i ms' %(taste_params[taste],signal_window,window_overlap), size=16)
                        plt.suptitle('Time-Frequency Profile of Local Field Potential', size = 20) 
                
                        #establish plotting variables
                        start_plot = pre_stim-int(plotting_params[2])
                        end_plot = pre_stim+int(plotting_params[3])
                
                        #Plot
                        plt.pcolormesh(x_mesh, y_mesh, z_mesh, cmap=cm.gnuplot2)
                        plt.colorbar(label="Normalized Power")
                        plt.ylim((int(plotting_params[0]),int(plotting_params[1])-.9))
                        locs, labels = plt.xticks()
                
                        #structuring
                        locs_condition = locs <=base_time/Fs
                        plt.xticks(locs,np.linspace(-base_time/Fs,int(plotting_params[3])/Fs,10))
                        plt.xlim((int(plotting_params[2])/Fs,(int(analysis_params[0])/Fs+int(plotting_params[3])/Fs)-0.45))
                        plt.xlabel('time (s)')
                        plt.ylabel('Frequency (Hz)')
                        fig.savefig(hdf5_name[0:4] +'_norm_'+ taste_params[dat_array]  + '_spectrograms_hanning_w_' + f'{np.int((window_overlap/signal_window)*100)}%_overlap.png') 
          
        if analysis_type=="Mean":
                if subplot_check == "Yes":   
                        #Plotting
                        fig = plt.figure(figsize=(11,7))
                        plt.subplot(1,1,1)
                        plt.title('Taste: %s \nHanning: Signal Window = %i ms, Window Overlap= %i ms' %(taste_params[taste],signal_window,window_overlap), size=16)
                        plt.suptitle('Time-Frequency Profile of Local Field Potential', size = 20) 
                
                        #establish plotting variables
                        start_plot = pre_stim-int(plotting_params[2])
                        end_plot = pre_stim+int(plotting_params[3])
                
                        #Plot
                        plt.pcolormesh(t_spec, f, gaussian_filter(np.mean(big_mean_P,axis=0),sigma=1.5));
                        #plt.pcolormesh(x_mesh, y_mesh, z_mesh, cmap=cm.coolwarm)
                        plt.colorbar(label="Mean Power")
                        plt.ylim((int(plotting_params[0]),int(plotting_params[1])))
                        locs, labels = plt.xticks()
                        #plt.xticks(locs,np.linspace(-start_plot/Fs,end_plot/Fs,10))
                        #plt.xlim((int(plotting_params[2])/Fs,(int(analysis_params[0])/Fs+int(plotting_params[3])/Fs)-0.25))
                        plt.xlabel('time from stimulus (s)')
                        plt.ylabel('Frequency (Hz)')
                        fig.savefig(hdf5_name[0:4] +'_mean_'+ taste_params[dat_array]  + '_spectrograms_hanning_w_' + f'{np.int((window_overlap/signal_window)*100)}%_overlap.png')                   
  

        
        figure=plt.figure(figsize=(11,7))        
        plt.pcolormesh(t_spec, f, np.mean(big_mean_P,axis=0)); plt.ylim([0, 30])

        plt.pcolormesh(t_spec, f, gaussian_filter(np.mean(big_mean_P,axis=0),sigma=1.5)); plt.ylim([0, 30])             











  
