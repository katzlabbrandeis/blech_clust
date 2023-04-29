#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 18:01:25 2019

@author: bradly
"""

# ____       _               
#/ ___|  ___| |_ _   _ _ __  
#\___ \ / _ \ __| | | | '_ \ 
# ___) |  __/ |_| |_| | |_) |
#|____/ \___|\__|\__,_| .__/ 
#                     |_|    

#import Libraries
# Built-in Python libraries
import os # functions for interacting w operating system
import sys
import glob

#import tools for multiprocessing
import pandas as pd
import multiprocessing
from joblib import Parallel, delayed, parallel_backend #for parallel processing

# 3rd-party libraries
import numpy as np # module for low-level scientific computing
import easygui
from astropy.stats import rayleightest
from astropy.stats import vonmisesmle

# =============================================================================
# #Define functions used in code
# =============================================================================

#Parellel processing function
def applyParallel(dfGrouped, func, parallel_kws={}, 
                        backend='multiprocessing', backend_kws={}):
    ''' Parallel version of pandas apply '''

    if not isinstance(dfGrouped, pd.core.groupby.GroupBy):
        raise TypeError(f'dfGrouped must be pandas.core.groupby.GroupBy, '\
                'not {type(dfGrouped)}')

    # Set default parallel args
    default_parallel_kws = \
                dict(n_jobs=multiprocessing.cpu_count(), max_nbytes=None, verbose=1)

    for key,item in default_parallel_kws.items():
        parallel_kws.setdefault(key, item)
    print("Apply parallel with {} verbosity".format(parallel_kws["verbose"]))

    # Compute
    with parallel_backend(backend, **backend_kws): 
        # backend decides how job lib will run your jobs, e.g. threads/processes/dask/etc
        retLst = Parallel(**parallel_kws)(delayed(func)(name,group) \
                        for name, group in dfGrouped)
    
    return retLst
    #return pd.concat(retLst) # return concatonated result

# Lambda function which calculates stats and returns list with indices
# and output stats
def run_stats(name, group):
    return list(name) + [rayleightest(np.array(group.phase)),
                    vonmisesmle(np.array(group.phase))[0],
                    vonmisesmle(np.array(group.phase))[1]
                    ]
    

 #Statistical running funtion
def spike_phase_stats(data_frame, time_vector):
    """
    Create dataframe to store statistics with column for Rayleigh p values, mean circular 
    angle of data, magnitude of non-uniformity, and distribution tests (comparitive)
    """
    
    # Pull out unique values for each group
    frame_dict = {'time_bin' : np.asarray(time_vector[:-1]).astype(int),
             'taste' : data_frame.taste.unique(),
             'unit' : data_frame.unit.unique(),
             'band' : data_frame.band.unique()}

    # Make list of dataframes with sequences of unique values to be merged later
    frame_list = [pd.DataFrame(data = val,columns=[key]) for key,val in frame_dict.items()]
    for frame in frame_list:
        frame['tmp'] = 1

    # Merge all dataframes in list
    # This allows pandas to handle generating all possible combinations
    stats_frame = frame_list[0]
    for frame_num in range(1,len(frame_list)):
        stats_frame = stats_frame.merge(frame_list[frame_num],how='outer')
    stats_frame = stats_frame.drop(['tmp'],axis=1)
    

    # Bin spiketimes in time-bins
    data_frame['time_bin'] = \
                pd.cut(x=data_frame.time,bins=t,labels=np.asarray(t[:-1]).astype(int))
    
    # Group by all variables to iterate over groups
    group_frame = data_frame.groupby(['band','taste','unit','time_bin'])
    
    
    # Parallel loop over all groups and run stats (using "applyParallel" 
    # function defined above).
    # Returns list of lists containing output of lambda function
    stats_list = applyParallel(
            group_frame, 
            run_stats, 
            parallel_kws={}, backend='multiprocessing', backend_kws={})
    
    # Compile lists of list into dataframe to be output
    temp_stats_frame = pd.DataFrame(stats_list,columns = \
            ['band','taste','unit','time_bin',\
            'Raytest_p','circ_mean','circ_kappa'])\
            .apply(pd.to_numeric, errors = 'coerce')
    
    # Merge with original stats frame
    ### This prevents skipping of groups since iterating over groups only returns ###
    ### non-empty groups ###
    stats_frame = stats_frame.merge(temp_stats_frame,how='outer')
    
    return stats_frame

# ___                            _     ____        _        
#|_ _|_ __ ___  _ __   ___  _ __| |_  |  _ \  __ _| |_ __ _ 
# | || '_ ` _ \| '_ \ / _ \| '__| __| | | | |/ _` | __/ _` |
# | || | | | | | |_) | (_) | |  | |_  | |_| | (_| | || (_| |
#|___|_| |_| |_| .__/ \___/|_|   \__| |____/ \__,_|\__\__,_|
#              |_|                                          

# =============================================================================
# Import/Open HDF5 File and variables
# =============================================================================

# If directory provided with script, use that otherwise ask
try:
    dir_name = sys.argv[1]
except:
    dir_name = easygui.diropenbox(msg = 'Select directory with HDF5 file')

hdf5_name = glob.glob(dir_name + '/*.h5')[0]

#pull in dframes
dframe = pd.read_hdf(hdf5_name,'Spike_Phase_Dframe/dframe','r+')
freq_dframe = pd.read_hdf(hdf5_name,'Spike_Phase_Dframe/freq_keys','r+')

#Exctract frequency names
freq_bands = np.array(freq_dframe.iloc[:][0]).astype(str).reshape(np.array(freq_dframe.iloc[:][0]).size,1)

#Create time vector (CHANGE THIS BASED ON BIN SIZING NEEDS)        
if np.size(dframe.taste.unique())>0:
    #Change this dependending on the session type        
    #Chose this bin size (250ms) based on smalled Frequency (4Hz)
	t= np.linspace(0,7000,29) 
	bins=29
else:
    #Change this dependending on the session type        
    t= np.linspace(0,1200000,50)
    bins=50

# ____                              _             
#|  _ \ _ __ ___   ___ ___  ___ ___(_)_ __   __ _ 
#| |_) | '__/ _ \ / __/ _ \/ __/ __| | '_ \ / _` |
#|  __/| | | (_) | (_|  __/\__ \__ \ | | | | (_| |
#|_|   |_|  \___/ \___\___||___/___/_|_| |_|\__, |
#                                           |___/ 

# =============================================================================
# #Parellel processing for statistics
# =============================================================================

#Perform statistics multiprocessing using taste as the group
dfnew = spike_phase_stats(dframe, t)

#convert strings to numerical data
dfnew = dfnew.apply(pd.to_numeric, errors = 'coerce')

#  ___        _               _   
# / _ \ _   _| |_ _ __  _   _| |_ 
#| | | | | | | __| '_ \| | | | __|
#| |_| | |_| | |_| |_) | |_| | |_ 
# \___/ \__,_|\__| .__/ \__,_|\__|
#                |_|              

# =============================================================================
# #Store back into HDF5
# =============================================================================

#Save dframe into node within HdF5 file
dfnew.to_hdf(hdf5_name,'Spike_Phase_Dframe/stats_dframe')
