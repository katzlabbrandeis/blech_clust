"""
Code to calculate STFT for emg data
"""

# Import stuff
import numpy as np
from scipy.signal import butter, filtfilt, periodogram
import easygui
import os
import sys
sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
import visualize as vz
from ephys_data import ephys_data
from tqdm import tqdm
import jax
import pylab as plt
from scipy.stats import zscore

## Get name of directory with the data files
#if len(sys.argv) > 1:
#    dir_name = os.path.abspath(sys.argv[1])
#    if dir_name[-1] != '/':
#        dir_name += '/'
#else:
#    dir_name = easygui.diropenbox(msg = 'Please select data directory')
#dir_name = '/media/fastdata/NB3_EMG_4tastes_211210_103154/'
dir_name = '/media/fastdata/KM45/KM45_5tastes_210620_113227'
os.chdir(dir_name)

# Load the data
emg_data = np.load('emg_data.npy')
#emg_filt = np.load('emg_filt.npy')
emg_filt = np.load('env.npy')

# Calc stft
stft_params = {
                'Fs' : 1000, 
                'signal_window' : 1000,
                'window_overlap' : 995,
                'max_freq' : 20,
                'time_range_tuple' : (1,6)
                }

stft_iters = list(np.ndindex(emg_filt.shape[:3])) 
stft_list = [ephys_data.calc_stft(emg_filt[this_iter], **stft_params)\
        for this_iter in tqdm(stft_iters)]

freq_vec = stft_list[0][0]
time_vec = stft_list[0][1]
fin_stft_list = [x[-1] for x in stft_list]
del stft_list
# (taste, channel, trial, frequencies, time)
stft_array = ephys_data.convert_to_array(fin_stft_list, stft_iters)
del fin_stft_list
amplitude_array = jax.numpy.square(jax.numpy.abs(stft_array))
del stft_array

# Aggregate across trials
agg_amplitude = np.median(amplitude_array,axis=2)

for taste in agg_amplitude:
    #vz.firing_overview(zscore(taste,axis=-1))
    #vz.firing_overview(np.log10(taste[:,4:11]))
    vz.firing_overview(np.log10(taste))
plt.show()
