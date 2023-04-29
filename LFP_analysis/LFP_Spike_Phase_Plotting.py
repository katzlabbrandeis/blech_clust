#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 15:25:18 2019

@author: bradly
"""

#import Libraries
# Built-in Python libraries
import os # functions for interacting w operating system
#import tools for multiprocessing
import pandas as pd
import multiprocessing
from joblib import Parallel, delayed, parallel_backend #for parallel processing
# 3rd-party libraries
import numpy as np # module for low-level scientific computing
import matplotlib
import matplotlib.pyplot as plt 
from scipy import stats
import re
import math
import easygui
from pylab import text
import seaborn.apionly as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage.filters import gaussian_filter1d
import glob 
import tqdm

# =============================================================================
# =============================================================================
# # #Define functions used in code
# =============================================================================
# =============================================================================

# =============================================================================
# #Parellel processing function
# =============================================================================
def applyParallel(
        dfGrouped, 
        func, 
        parallel_kws={}, 
        backend='multiprocessing', 
        backend_kws={}):

    ''' Parallel version of pandas apply '''

    if not isinstance(dfGrouped, pd.core.groupby.GroupBy):
        raise TypeError(f'dfGrouped must be pandas.core.groupby.GroupBy,',
                'not {type(dfGrouped)}')

    # Set default parallel args
    default_parallel_kws =\
    dict(n_jobs=multiprocessing.cpu_count(), max_nbytes=None, verbose=11)

    for key,item in default_parallel_kws.items():
        parallel_kws.setdefault(key, item)
    print("Apply parallel with {} verbosity".format(parallel_kws["verbose"]))

    # Compute
    with parallel_backend(backend, **backend_kws): 
        # backend decides how job lib will run your jobs, 
        # e.g. threads/processes/dask/etc
        retLst = Parallel(**parallel_kws)(delayed(func)(group) \
                for name, group in dfGrouped)

    return pd.concat(retLst) # return concatonated result


# =============================================================================
# #Create dataframe to store phasic density matrices using KDE 
# =============================================================================
def spike_phase_density(data_frame,frequency_list,time_vector,bin_vals):    

    #Create empty dataframe for storing output
    density_df_all = pd.DataFrame()

    for band_num in range(len(frequency_list)):
        for taste_num in    range(len(data_frame.taste.unique())):
            density_all=[[None] *(bin_vals) for i in range(len(time_vector)-1)]
            for time_num in range(len(time_vector)-1):
            
                #Set query to perfrom statistical tests on
                query = data_frame.query('band == @band_num  \
                             and taste == @taste_num \
                             and time >= @time_vector[@time_num] \
                             and time < @time_vector[@time_num+1]')
                
                #Set query index
                queryidx = query.index
                
                 #Fills bin slot with nan for empty queries
                if len(queryidx) <= 1:
                    density_all[time_num] = np.full(bin_vals,'nan',dtype=np.float64)
                      
                #Ensures that idx is a possible boolean
                if len(queryidx) >= 2:

                    #Get density statistics
                    density = stats.gaussian_kde(np.array(query.phase)) 
                    
                    #Use histogram function to access densities based on phasic bin
                    n1,x1=np.histogram(np.array(query.phase), \
                            bins=np.linspace(-np.pi,np.pi,bin_vals), density=True);
        
                    density_all[time_num] = density(x1)
                    
            #store in data frame by unit,band,taste_num
            density_df = pd.DataFrame(density_all)
            density_df["band"] = band_num; density_df["taste"] = \
                    taste_num; density_df["unit"] = data_frame.unit.unique()[0]
            
            #concatenate dataframes
            density_df_all = pd.concat([density_df_all, density_df])
            
                    
    return density_df_all

# =============================================================================
# #Define column indexing function for plotting
# =============================================================================
def column_index(df, query_cols):
    cols = df.columns.values
    sidx = np.argsort(cols)
    return sidx[np.searchsorted(cols,query_cols,sorter=sidx)]

# =============================================================================
# =============================================================================
# # Begin working with Data
# =============================================================================
# =============================================================================

# =============================================================================
# #Load Data
# =============================================================================
# Get name of directory where the data files and hdf5 file sits, 
# and change to that directory for processing

# If directory provided with script, use that otherwise ask
try:
    #dir_name = os.path.dirname(sys.argv[1])
    dir_name = sys.argv[1]
except:
    dir_name = easygui.diropenbox(msg = 'Select directory with HDF5 file')

os.chdir(dir_name)
hdf5_name = glob.glob(dir_name + '/*.h5')[0]

#pull in dframes
dframe = pd.read_hdf(hdf5_name,'Spike_Phase_Dframe/dframe','r+')
freq_dframe = pd.read_hdf(hdf5_name,'Spike_Phase_Dframe/freq_keys','r+')
dframe_stat = pd.read_hdf(hdf5_name,'Spike_Phase_Dframe/stats_dframe','r+')

# =============================================================================
# #Establish variables for processing
# =============================================================================
#Create time vector (CHANGE THIS BASED ON BIN SIZING NEEDS)     
if np.size(dframe.taste.unique())>0:
    # Set paramaters from the user
    params = easygui.multenterbox(msg = 'Enter the parameters for plotting', 
            fields = ['Pre stimulus spike train (ms)',
                    'Post-stimulus spike train (ms)', 
                    'Bin size (ms)','Pre stimulus bin (ms)',
                    'Post-stimulus bin (ms)'],
            values = ['2000','5000','50','2000','2000'])
    for i in range(len(params)):
        params[i] = int(params[i])  
    
    t= np.linspace(0,params[0]+params[1],((params[0]+params[1])//100)+1)
    bins=params[2]
    
    identities = easygui.multenterbox(\
            msg = 'Put in the taste identities of the digital inputs', 
            fields = [tastant for tastant in range(len(dframe.taste.unique()))], 
            values=['NaCl','Sucrose','Citric Acid','QHCl'])
else:
    # Set paramaters from the user
    params = easygui.multenterbox(msg = 'Enter the parameters for plotting', 
            fields = ['Pre stimulus (ms)','Post-stimulus (ms)', 'Bin size (ms)'],
            values = ['0','1200000','50'])
    for i in range(len(params)):
        params[i] = int(params[i])  
    #Change this dependending on the session type       
    t= np.linspace(0,params[0]+params[1],((params[0]+params[1])//100)+1)
    bins=params[2]
    
#Exctract frequency names
freq_bands = np.array(freq_dframe.iloc[:][0]).astype(str).\
        reshape(np.array(freq_dframe.iloc[:][0]).size,1)
freq_vals = freq_dframe.to_dict('index')

#define color palette
colors = plt.get_cmap('winter_r')(np.linspace(0, 1, 4))

# =============================================================================
# #Parellel processing for statistics
# =============================================================================
#tempfunction to escape issue with using lambda
def tmpfunc(x, freq_bands=freq_bands, t=t,bins=bins):
    return spike_phase_density(x,freq_bands,t,bins)

#Perform statistics multiprocessing using taste as the group
dfnew = applyParallel(dframe.groupby(dframe.unit), tmpfunc)

#Save dframe into node within HdF5 file
dfnew.to_hdf(hdf5_name,'Spike_Phase_Dframe/kde_dframe')

# =============================================================================
# #Plotting
# =============================================================================
# Make directory to store all phaselocking plots. Delete and 
# remake the directory if it exists
try:
        os.system('rm -r '+'./Phase_lock_analyses')
except:
        pass
os.mkdir('./Phase_lock_analyses')

# Make directory to store histogram plots. Delete and 
# remake the directory if it exists
try:
        os.system('rm -r '+'./Phase_lock_analyses/Phase_histograms')
except:
        pass
os.mkdir('./Phase_lock_analyses/Phase_histograms')

# =============================================================================
# Phase Raste plot 
# =============================================================================

# For single band
plot_dir = dir_name + '/Phase_lock_analyses/Phase_Rasters'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

bin_num = np.min([dframe.trial.max(),dframe.wavelength_num.max()])
dframe = (dframe.
        assign(wavelength_id = lambda x : (x.wavelength_num*bin_num)+x.trial))

stim_time = int(easygui.enterbox('Enter time of stimulus delivery (sec)')) 

def make_rasters(single_band_frame, band):
    stim_wavelength = len(np.unique(dframe.trial)) * \
            np.mean(freq_dframe.loc[band,1:]) * stim_time
    g = sns.FacetGrid(single_band_frame , col = 'unit', row = 'taste',
            sharey=True)
    #g.map(plt.scatter, 'phase', 'plot_index', s = 48, alpha = 0.3)
    g.map(plt.scatter, 'phase', 'wavelength_id', s = 48, alpha = 0.3)
    g.map(plt.hlines, y = stim_wavelength, xmin = -np.pi, xmax = np.pi, color  = 'r' )
    plt.savefig(plot_dir + \
            '/{}_raster.png'.format(freq_dframe[0][band]))
    h = sns.FacetGrid(single_band_frame, col = 'unit', row = 'taste', sharey=False)
    h.map(plt.hist,'phase', bins = 20)
    plt.savefig(plot_dir +\
        '/{}_histogram.png'.format(freq_dframe[0][band]))

#for band in dframe.band_num.unique():
Parallel(n_jobs = len(dframe.band.unique()))(delayed(make_rasters)\
        (dframe.query('band == @band'),band) \
        for band in tqdm.tqdm(dframe.band.unique()))

# =============================================================================
# Creates Histograms for spikes by phase
# =============================================================================
for taste, color in zip(dframe.taste.unique(),colors):
    for band in sorted(dframe.band.unique()):
        #Set up axes for plotting all tastes together
        fig,axes = plt.subplots(nrows=math.ceil(len(dframe.unit.unique())/4), 
                ncols=4,sharex=True, sharey=False,figsize=(12, 8), squeeze=False)
        fig.text(0.07, 0.5,'Number of Spikes', 
                va='center', rotation='vertical',fontsize=14)
        fig.text(0.5, 0.05, 'Phase', ha='center',fontsize=14)
        axes_list = [item for sublist in axes for item in sublist]
        
        for ax, unit in zip(axes.flatten(),np.sort(dframe.unit.unique())):
            query_check = dframe.query('band == @band and unit == @unit and '+\
                    'taste == @taste and time>=@params[3] and ' +\
                    'time<=@params[3]+@params[4]')
            df_var = query_check.phase
            
            ax = axes_list.pop(0)
            im =ax.hist(np.array(df_var), bins=25, color=color, alpha=0.7)
            ax.set_title(unit,size=12,y=0.55,x=0.9)
            ax.set_xticks(np.linspace(-np.pi,np.pi,5))
            ax.set_xticklabels([r"-$\pi$",r"-$\pi/2$","$0$",r"$\pi/2$",r"$\pi$"])
    
        fig.subplots_adjust(hspace=0.25,wspace = 0.05)
        fig.suptitle('Taste: %s' %(identities[taste])+'\n' +\
                'Freq. Band: %s (%i - %iHz)' \
                %(freq_vals[band][0],freq_vals[band][1],freq_vals[band][2])+'\n'+\
                'Time: %i - %ims post-delivery' %(params[0]-params[3],params[4]),\
                        size=16,fontweight='bold')                        
        fig.savefig('./Phase_lock_analyses/Phase_histograms/' + '%s_%s_hist.png' \
                        %(identities[taste],freq_vals[band][0]))   
        plt.close(fig)

# =============================================================================
# =============================================================================
# # Sig locking over time (density)
# =============================================================================
# =============================================================================

# Make directory to store histogram plots. Delete and 
# remake the directory if it exists
try:
        os.system('rm -r '+'./Phase_lock_analyses/KDEs')
except:
        pass
os.mkdir('./Phase_lock_analyses/KDEs')
            
#Creates Heatmaps for density estimationg (KDE) of spikes within phase over time
for unit in dfnew.unit.unique():
    for band in dfnew.band.unique():
        #Set up axes for plotting all tastes together
        fig,axes = plt.subplots(nrows=2, ncols=2,sharex=True, sharey=True,
                figsize=(12, 8), squeeze=False)
        fig.text(0.12, 0.5,'Phase', va='center', rotation='vertical',fontsize=15)
        fig.text(0.5, 0.05, 'Seconds', ha='center',fontsize=15)
        axes_list = [item for sublist in axes for item in sublist]
    
        for ax, taste, color in zip(axes.flatten(),dfnew.taste.unique(),colors):
            query_check = dfnew.query('band == @band and unit == @unit and '+\
                    'taste == @taste')
            plt_query = query_check[query_check.columns.difference(\
                    ['band', 'unit','taste'])] 
            #Excludes labeling columns for accurate imshow
            
            ax = axes_list.pop(0)
            im =ax.imshow(plt_query.T)
            ax.set_title(identities[taste],size=15,y=1)
            ax.set_yticks(np.linspace(0,bins-1,5))
            ax.set_yticklabels([r"-$\pi$",r"-$\pi/2$","$0$",r"$\pi/2$",r"$\pi$"])
            ax.set_xticks(np.linspace(0,len(t)-1,((len(t)-1)//10)+1))
            ax.set_xticklabels(np.arange(0,(len(t)-1)//10,1))       
            ax.axvline(x=np.where(t==params[0]), linewidth=4, color='r')
            
        fig.subplots_adjust(hspace=0.25,wspace = -0.15)
        fig.suptitle('Unit %i'
                %(sorted(dframe_stat['unit'].unique())[unit])+'\n' +\
                        'Freq. Band: %s (%i - %iHz)' %(freq_vals[band][0],\
                        freq_vals[band][1],freq_vals[band][2]),size=16,\
                        fontweight='bold')
        fig.savefig('./Phase_lock_analyses/KDEs/'+'Unit_%i_%s_KDE.png' \
                %(sorted(dframe_stat['unit'].unique())[unit],freq_vals[band][0]))   
        plt.close(fig)

# =============================================================================
# =============================================================================
# # Sig locking over time (unit dependent)
# =============================================================================
# =============================================================================

# Make directory to store histogram plots. 
# Delete and remake the directory if it exists
try:
        os.system('rm -r '+'./Phase_lock_analyses/ZPMs')
except:
        pass
os.mkdir('./Phase_lock_analyses/ZPMs')

# Creates Heatmaps of Rayleigh pvals based on distribution 
# of spikes within band over time
#Create categorical values for pvals

dframe_stat['pval_cat'] = ''
dframe_stat.loc[(dframe_stat['Raytest_p'] > 0) & (dframe_stat['Raytest_p'] <= 0.05),
        'pval_cat'] = int(0)
dframe_stat.loc[(dframe_stat['Raytest_p'] > 0.05) & (dframe_stat['Raytest_p'] <= 0.1),
        'pval_cat'] = int(1)
dframe_stat.loc[(dframe_stat['Raytest_p'] > 0.1),'pval_cat'] = int(2)
dframe_stat = dframe_stat.replace('',np.nan)

#Set colorbar values
cmap = LinearSegmentedColormap.from_list('Custom', colors[0:3,:], len(colors[0:3,:]))

#Plot by unit
for unit in dframe_stat.unit.unique():
    
    fig,axes = plt.subplots(nrows=2, ncols=2,sharex=True, sharey=True,
            figsize=(12, 8), squeeze=False)
    fig.text(0.07, 0.5,'Bands', va='center', rotation='vertical',fontsize=14)
    fig.text(0.5, 0.05, 'Time', ha='center',fontsize=14)
    axes_list = [item for sublist in axes for item in sublist]
    
    #Subplot by Tastant
    for ax, taste in zip(axes.flatten(),dframe_stat.taste.unique()):
        query = dframe_stat.query('unit == @unit and taste == @taste')
        
        ax = axes_list.pop(0)
        piv = pd.pivot_table(query, values="pval_cat",index=["band"], 
                columns=["time_bin"], fill_value=2)
        im = sns.heatmap(piv,yticklabels=[x[0] for x in \
                list(freq_vals.values())],annot_kws = {"color": "white"},
                cmap=cmap, ax=ax)

        # Manually specify colorbar labelling after it's been generated
        colorbar = ax.collections[0].colorbar
        colorbar.set_ticks([0.25, 1, 1.75])
        colorbar.set_ticklabels(['p < 0.05', 'p < 0.1', 'p > 0.1'])
        
        ax.axvline(x=column_index(piv,params[0]), linewidth=4, color='r')
        ax.set_title(identities[taste],size=15,y=1)
        
    fig.suptitle('Unit %i' %(sorted(dframe_stat['unit'].unique())[unit])+ '\n'+\
            'Ztest pval Matrices',size=16)
    fig.savefig('./Phase_lock_analyses/ZPMs/'+'Unit_%i_ZPMs.png' \
            %(sorted(dframe_stat['unit'].unique())[unit]))   
    plt.close(fig)

# =============================================================================
# =============================================================================
# # Sig locking over time (unit independent)
# =============================================================================
# =============================================================================
    
#Plot counts of sig phaselocking units over time
#define color palette
colors_new = plt.get_cmap('RdBu')(np.linspace(0, 1, 4))
fig,axes = plt.subplots(nrows=2, ncols=2,sharex=True, sharey=True,
        figsize=(12, 8), squeeze=False)
fig.text(0.075, 0.5,'Percentage of Units', va='center', 
        rotation='vertical',fontsize=15)
fig.text(0.5, 0.05, 'Time', ha='center',fontsize=15)
axes_list = [item for sublist in axes for item in sublist]

for band in sorted(dframe_stat.band.unique()):
    ax = axes_list.pop(0)
    for taste in sorted(dframe_stat.taste.unique()):
        query = dframe_stat.query('taste == @taste and band ==@band and pval_cat == 0')
        
        #Applied a first order gaussian filter to smooth lines
        p1 = ax.plot(sorted(dframe_stat.time_bin.unique()),
                gaussian_filter1d(np.array([query['time_bin'].value_counts()[x] \
                    if x in query['time_bin'].unique() else 0 for x in \
                    sorted(dframe_stat.time_bin.unique())])/\
                    len(dframe_stat.unit.unique())*100,sigma=1),
                color = colors_new[taste],linewidth = 2)
        ax.set_title([x[0] for x in list(freq_vals.values())][band],size=15,y=1)

fig.legend(identities,loc = (0.3, 0), ncol=4)
fig.subplots_adjust(hspace=0.25,wspace = 0.1)
fig.suptitle('Animal: %s; Date: %s' %(hdf5_name[:4],re.findall(r'_(\d{6})', 
    hdf5_name)[0])+ '\n' + 'Taste effect on phase-locking' + '\n' +\
            'Units = %i' %(len(dframe_stat.unit.unique())),size=16)
fig.savefig('./Phase_lock_analyses/ZPMs/'+'%s_SPL_Distribution.png' %(hdf5_name[:4]))   
plt.close(fig)

# =============================================================================
# =============================================================================
# # #Grouped Bar plots
# =============================================================================
# =============================================================================

#Create grouped bar plots detailing taste evoked spike-phase locking stats
zpal_params = easygui.multenterbox(msg = 'Enter the parameters for grouped' +\
        'bars', fields = ['Pre-stimulus time (ms)','Post-stimulus time (ms)'],
        values = ['2000','2000'])
for i in range(len(zpal_params)):
    zpal_params[i] = int(zpal_params[i])    

# Make directory to store histogram plots. Delete and remake the directory if it exists
try:
        os.system('rm -r '+'./Phase_lock_analyses/Frequency_Plots')
except:
        pass
os.mkdir('./Phase_lock_analyses/Frequency_Plots')

#Create smaller dataframes to work with
pre_dataframe = dframe_stat.query('time_bin <= @zpal_params[0]')
post_dataframe = dframe_stat.query('time_bin > @zpal_params[0] and time_bin<='+\
        '@params[0]+@zpal_params[1]')

for unit in dframe_stat.unit.unique():
        
    fig,axes = plt.subplots(nrows=2, ncols=2,sharex=True, 
            sharey=True,figsize=(13, 8), squeeze=False)
    fig.text(0.07, 0.5,'Number of significant bins', va='center', 
            rotation='vertical',fontsize=14)
    fig.text(0.5, 0.05, 'Freq. Band', ha='center',fontsize=14)
    axes_list = [item for sublist in axes for item in sublist]
    ind = np.arange(len(dframe_stat['band'].unique())) #x locations for groups
    width = 0.35 #set widths of bars
    
    #Subplot by Tastant
    for ax, taste in zip(axes.flatten(),dframe_stat.taste.unique()):
        pre_query = pre_dataframe.query('unit == @unit and taste == '+\
                '@taste and pval_cat == 0')
        post_query = post_dataframe.query('unit == @unit and taste == '+\
               ' @taste and pval_cat == 0')
        
        #get frequency count before after taste delivery
        pre_freq = [pre_query['band'].value_counts()[x] if x in \
                pre_query['band'].unique() else 0 for x in \
                sorted(dframe_stat.band.unique())]
        post_freq = [post_query['band'].value_counts()[x] if x in \
                post_query['band'].unique() else 0 for x in \
                sorted(dframe_stat.band.unique())]
        
        ax = axes_list.pop(0)
        p1 = ax.bar(ind, pre_freq, width, color = colors[0])
        p2 = ax.bar(ind+width, post_freq, width, color = colors[1])

        ax.set_xticks(ind + width / 2)
        ax.set_xticklabels(([x[0] for x in list(freq_vals.values())]))
        ax.set_title(identities[taste],size=15,y=1)
        
    ax.legend(('Pre','Post'), bbox_to_anchor=(1.05, 0), loc='lower left', 
            fontsize = 13, borderaxespad=0.)
    fig.subplots_adjust(hspace=0.2,wspace = 0.1)
    fig.suptitle('Unit %i' %(sorted(dframe_stat['unit'].unique())[unit])+ '\n' + \
            'Taste effect on phase-lock frequency' + '\n' + \
            'Pre: %ims & Post: %ims' %(zpal_params[0],zpal_params[1]),size=16)
    fig.savefig('./Phase_lock_analyses/Frequency_Plots/'+\
    'Unit_%i_SPFreqency_plots.png' %(sorted(dframe_stat['unit'].unique())[unit]))   
    plt.close(fig)

