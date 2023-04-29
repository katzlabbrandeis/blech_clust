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
import matplotlib.pyplot as plt # makes matplotlib work like MATLAB. ’pyplot’ functions.
from scipy.stats import sem
from scipy import stats
from scipy.stats import chi2_contingency
from pycircstat import descriptive, swap2zeroaxis 
import math
import sys
import easygui
import tables
from matplotlib.ticker import PercentFormatter
from pylab import text
import pylab as P
from astropy.stats import rayleightest
from astropy.stats import circmean
from astropy.stats import vonmisesmle
from astropy import units as u
from pycircstat import kuiper
import seaborn.apionly as sns

# =============================================================================
# #Define functions used in code
# =============================================================================

#Parellel processing function
def applyParallel(dfGrouped, func, parallel_kws={}, backend='multiprocessing', backend_kws={}):
    ''' Parallel version of pandas apply '''

    if not isinstance(dfGrouped, pd.core.groupby.GroupBy):
        raise TypeError(f'dfGrouped must be pandas.core.groupby.GroupBy, not {type(dfGrouped)}')

    # Set default parallel args
    default_parallel_kws = dict(n_jobs=multiprocessing.cpu_count(), max_nbytes=None, verbose=11)
    for key,item in default_parallel_kws.items():
        parallel_kws.setdefault(key, item)
    print("Apply parallel with {} verbosity".format(parallel_kws["verbose"]))

    # Compute
    with parallel_backend(backend, **backend_kws): # backend decides how job lib will run your jobs, e.g. threads/processes/dask/etc
        retLst = Parallel(**parallel_kws)(delayed(func)(group) for name, group in dfGrouped)

    return pd.concat(retLst) # return concatonated result

# =============================================================================
# #Create dataframe to store phasic density matrices using KDE 
# =============================================================================
def spike_phase_density(data_frame,frequency_list,time_vector,bin_vals):	

	#Create empty dataframe for storing output
	density_df_all = pd.DataFrame()
	
	for band_num in range(len(frequency_list)):
		for taste_num in	range(len(data_frame.taste.unique())):
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
					n1,x1=np.histogram(np.array(query.phase), bins=np.linspace(-np.pi,np.pi,bin_vals), density=True);
		
					density_all[time_num] = density(x1)
					
			#store in data frame by unit,band,taste_num
			density_df = pd.DataFrame(density_all)
			density_df["band"] = band_num; density_df["taste"] = taste_num; density_df["unit"] = data_frame.unit.unique()[0]
			
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
# load the data
# =============================================================================

#Get name of directory where the data files and hdf5 file sits, and change to that directory for processing
dir_name = easygui.diropenbox()
os.chdir(dir_name)

#Look for the hdf5 file in the directory
file_list = os.listdir('./')
hdf5_name = ''
for files in file_list:
	if files[-2:] == 'h5':
		hdf5_name = files

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
	params = easygui.multenterbox(msg = 'Enter the parameters for plotting', fields = ['Pre stimulus spike train (ms)','Post-stimulus spike train (ms)', 'Bin size (ms)','Pre stimulus bin (ms)','Post-stimulus bin (ms)'],values = ['2000','5000','50','2000','2000'])
	for i in range(len(params)):
		params[i] = int(params[i])	
	
	t= np.linspace(0,params[0]+params[1],((params[0]+params[1])//100)+1)
	bins=params[2]
	
	identities = easygui.multenterbox(msg = 'Put in the taste identities of the digital inputs', fields = [tastant for tastant in range(len(dframe.taste.unique()))], values=['NaCl','Sucrose','Citric Acid','QHCl'])
else:
	# Set paramaters from the user
	params = easygui.multenterbox(msg = 'Enter the parameters for plotting', fields = ['Pre stimulus (ms)','Post-stimulus (ms)', 'Bin size (ms)'],values = ['0','1200000','50'])
	for i in range(len(params)):
		params[i] = int(params[i])	
	#Change this dependending on the session type		
	t= np.linspace(0,params[0]+params[1],((params[0]+params[1])//100)+1)
	bins=params[2]
	
#Exctract frequency names
freq_bands = np.array(freq_dframe.iloc[:][0]).astype(str).reshape(np.array(freq_dframe.iloc[:][0]).size,1)
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

plt_query = dfnew.query('band == 3 and unit == 0 and taste == 3')
plt_query2 = plt_query[plt_query.columns.difference(['band', 'unit','taste'])]
plt.imshow(plt_query2.T)

plt.imshow(plt_query.loc[:, plt_query.columns != 'band'].T)
plt.imshow(plt_query.T)


# =============================================================================
# #Allow user to choose how they want their data represented
# =============================================================================
msg = "Enter your personal information"
title = "Credit Card Application"
fieldNames = ["Name","Street Address","City","State","ZipCode"]
fieldValues = []  # we start with blanks for the values
fieldValues = easygui.multchoicebox(msg,title, fieldNames)

#Creates Histograms for spikes by phase
for taste, color in zip(dframe.taste.unique(),colors):
	for band in dframe.band.unique():
		#Set up axes for plotting all tastes together
		fig,axes = plt.subplots(nrows=math.ceil(len(dframe.unit.unique())/4), ncols=4,sharex=True, sharey=True,figsize=(12, 8), squeeze=False)
		fig.text(0.07, 0.5,'Number of Spikes', va='center', rotation='vertical',fontsize=14)
		fig.text(0.5, 0.05, 'Taste', ha='center',fontsize=14)
		axes_list = [item for sublist in axes for item in sublist]
		
		for ax, unit in zip(axes.flatten(),np.sort(dframe.unit.unique())):
			query_check = dframe.query('band == @band and unit == @unit and taste == @taste and time>=@params[3] and time<=@params[3]+@params[4]')
			df_var = query_check.phase
			
			ax = axes_list.pop(0)
			im =ax.hist(np.array(df_var), bins=25, color=color, alpha=0.7)
			ax.set_title(unit,size=12,y=0.55,x=0.9)
			ax.set_xticks(np.linspace(-np.pi,np.pi,5))
			ax.set_xticklabels([r"-$\pi$",r"-$\pi/2$","$0$",r"$\pi/2$",r"$\pi$"])
	
		fig.subplots_adjust(hspace=0.25,wspace = 0.05)
		fig.suptitle('Taste: %s' %(identities[taste])+'\n' + 'Freq. Band: %s (%i - %iHz)' %(freq_vals[band][0],freq_vals[band][1],freq_vals[band][2])+'\n' + 'Time: %i - %ims' %(params[0]-params[3],params[4]),size=16,fontweight='bold')						

#Creates Heatmaps for density estimationg (KDE) of spikes within phase over time
for unit in dfnew.unit.unique():
	for band in dfnew.band.unique():
		#Set up axes for plotting all tastes together
		fig,axes = plt.subplots(nrows=2, ncols=2,sharex=True, sharey=True,figsize=(12, 8), squeeze=False)
		fig.text(0.12, 0.5,'Phase', va='center', rotation='vertical',fontsize=15)
		fig.text(0.5, 0.05, 'Seconds', ha='center',fontsize=15)
		axes_list = [item for sublist in axes for item in sublist]
	
		for ax, taste, color in zip(axes.flatten(),dfnew.taste.unique(),colors):
			query_check = dfnew.query('band == @band and unit == @unit and taste == @taste')
			plt_query = query_check[query_check.columns.difference(['band', 'unit','taste'])] #Excludes labeling columns for accurate imshow
			
			ax = axes_list.pop(0)
			im =ax.imshow(plt_query.T)
			ax.set_title(identities[taste],size=15,y=1)
			ax.set_yticks(np.linspace(0,bins-1,5))
			ax.set_yticklabels([r"-$\pi$",r"-$\pi/2$","$0$",r"$\pi/2$",r"$\pi$"])
			ax.set_xticks(np.linspace(0,len(t)-1,((len(t)-1)//10)+1))
			ax.set_xticklabels(np.arange(0,(len(t)-1)//10,1))		
			ax.axvline(x=np.where(t==params[0]), linewidth=4, color='r')
			
		fig.subplots_adjust(hspace=0.25,wspace = -0.15)
		fig.suptitle('Unit %i' %(dframe.unit.unique()[unit])+'\n' + 'Freq. Band: %s (%i - %iHz)' %(freq_vals[band][0],freq_vals[band][1],freq_vals[band][2]),size=16,fontweight='bold')
		fig.savefig('Unit_%i_%s_KDE.png' %(dframe.unit.unique()[unit],freq_vals[band][0]))   
		plt.close(fig)

# =============================================================================
# plt_query = density_df_all.query('band == 3 and unit == 0 and taste == 3')
# plt_query2 = plt_query[plt_query.columns.difference(['band', 'unit','taste'])]
# plt.imshow(plt_query2.T)
# 
# =============================================================================

	


for unit in dframe_stat.unit.unique():
	plt.figure()
	for taste in dframe_stat.taste.unique():
		query = dframe_stat.query('unit == @unit and band == 0 and taste == @taste')
		#test = query.groupby('time_bin')
		plt.plot(query.time_bin,query.Raytest_p)


for unit in dframe_stat.unit.unique():
	#plot by unit
	fig,axes = plt.subplots(nrows=2, ncols=2,sharex=True, sharey=True,figsize=(12, 8), squeeze=False)
	fig.text(0.07, 0.5,'Bands', va='center', rotation='vertical',fontsize=14)
	fig.text(0.5, 0.05, 'Time', ha='center',fontsize=14)
	axes_list = [item for sublist in axes for item in sublist]
	
	for ax, taste in zip(axes.flatten(),dframe_stat.taste.unique()):
		query = dframe_stat.query('unit == @unit and taste == @taste')
		
		ax = axes_list.pop(0)
		piv = pd.pivot_table(query, values="Raytest_p",index=["band"], columns=["time_bin"], fill_value=0)
		im = sns.heatmap(piv,vmin=0, vmax=0.1,cmap="YlGnBu", ax=ax)
		ax.axvline(x=column_index(piv,params[0]), linewidth=4, color='r')
		ax.set_title(taste,size=15,y=1)
		
	fig.suptitle('Unit %i' %(dframe_stat.unit.unique()[unit])+ '\n' + 'Ztest pval Matrices',size=16)
	fig.savefig('Unit_%i_ZPMs.png' %(dframe_stat.unit.unique()[unit]))   
	plt.close(fig)





