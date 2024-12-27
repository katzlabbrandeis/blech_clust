"""
Code to test run ephys_data module
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import zscore

os.chdir('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
from visualize import firing_overview, imshow

dat = \
    ephys_data('/media/bigdata/Abuzar_Data/gc_only/AM34/AM34_4Tastes_201215_115133')
dat.firing_rate_params = dat.default_firing_params

dat.get_unit_descriptors()
dat.get_spikes()
dat.get_firing_rates()
#dat.get_lfps(re_extract= True)
dat.get_lfps()

dat.get_region_units()
dat.get_lfp_electrodes()

#dat.get_stft(recalculate = True)
dat.get_stft()

aggregate_amplitude = dat.get_mean_stft_amplitude()

def normalize_timeseries(array, time_vec, stim_time):
    mean_baseline = np.mean(array[...,time_vec<stim_time],axis=-1)[...,np.newaxis]
    array = array/mean_baseline
    return array

time_vec = dat.time_vec
stim_time = 2
fig,ax = plt.subplots(1,len(aggregate_amplitude))
for num,(region,this_ax) in enumerate(zip(aggregate_amplitude, ax.flatten())):
    this_ax.imshow(zscore(
                    normalize_timeseries(region, time_vec, stim_time),axis=-1),
            aspect='auto', origin = 'lower')
    this_ax.set_title(dat.region_names[num])
    this_ax.set_yticks(np.arange(len(dat.freq_vec)))
    this_ax.set_yticklabels(dat.freq_vec)
plt.show()
