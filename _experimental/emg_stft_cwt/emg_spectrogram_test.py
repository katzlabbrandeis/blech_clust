"""
Comparison of information in filtered emg vs envelope
"""

# Import stuff
import numpy as np
import scipy
from scipy.signal import butter, filtfilt, periodogram, gaussian, convolve
from scipy.signal import savgol_filter as savgol
import easygui
import os
import pylab as plt
from tqdm import tqdm
import sys
sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
import visualize as vz
import tables
from glob import glob
from scipy.stats import zscore

# Define function to parse out only wanted frequencies in STFT
def calc_stft(trial, max_freq,time_range_tuple,\
            Fs,signal_window,window_overlap):
    """
    trial : 1D array
    max_freq : where to lob off the transform
    time_range_tuple : (start,end) in seconds, time_lims of spectrogram
                            from start of trial snippet`
    """
    f,t,this_stft = scipy.signal.stft(
                scipy.signal.detrend(trial),
                fs=Fs,
                window='hann',
                nperseg=signal_window,
                noverlap=signal_window-(signal_window-window_overlap))
    this_stft =  this_stft[np.where(f<max_freq)[0]]
    this_stft = this_stft[:,np.where((t>=time_range_tuple[0])*\
                                            (t<time_range_tuple[1]))[0]]
    fin_freq = f[f<max_freq]
    fin_t = t[np.where((t>=time_range_tuple[0])*(t<time_range_tuple[1]))]
    return  fin_freq, fin_t, this_stft

def rolling_zscore(x,window):
    """
    Perform a zscore using a rolling window

    Inputs:
        x : 1D array
        window : int, size of window to use

    Outputs:
        zscore : 1D array, same size as x
    """
    zscore = np.zeros(x.shape)
    for i in range(len(x)):
        zscore[i] = (x[i] - np.mean(x[i-window:i])) / np.std(x[i-window:i])
    # Drop nan's
    zscore[np.isnan(zscore)] = 0
    return zscore

def calc_stft_mode_freq(dat, **stft_params, BSA_output = True):
    """
    Calculate the mode frequency of the STFT of a signal

    Inputs:
        dat : 1D array, signal to be analyzed
        stft_params : dict, parameters for STFT calculation
        BSA_output : bool, whether to output in a similar format to the BSA

    Outputs:
        freq_vec : 1D array, frequency vector for STFT
        t_vec : 1D array, time vector for STFT
        weight_mean : 1D array, time averaged mode frequency
    """
    freq_vec, t_vec, stft  = calc_stft(dat, **stft_params)
    mag = np.abs(stft)
    time_norm_mag = mag / mag.sum(axis=0)
    weight_mean = (freq_vec[:,None] * time_norm_mag).sum(axis=0)
    if BSA_output:
        # This is only needed if stft_step size > 1
        # Interpolate output to same length as input
        new_t_vec = np.arange(stft_params['time_range_tuple'][0],
                              stft_params['time_range_tuple'][1],
                              1/stft_params['Fs'])
        new_weight_mean = np.interp(new_t_vec, t_vec, weight_mean)
        new_freq = np.arange(0,10,0.5)
        new_weight_mean = np.zeros((len(new_freq), len(t_vec)))
        closest_freq = np.argmin(np.abs(weight_mean[:,None] - new_freq), axis=1)
        new_weight_mean[closest_freq, np.arange(len(t_vec))] = 1
        return new_freq, new_t_vec, new_weight_mean
    else:
        return freq_vec, t_vec, weight_mean


# Ask for the directory where the data (emg_data.npy) sits
dir_name = '/media/bigdata/firing_space_plot/NM_gape_analysis/raw_data/NM51_2500ms_161030_130155_new/emg_output/emg'
#dir_name = easygui.diropenbox()
os.chdir(dir_name)

#emg_data = np.load('emg_data.npy')
emg_filt = np.load('emg_filt.npy')
emg_env = np.load('emg_env.npy')
sig_trials = np.load('sig_trials.npy')

stft_params = dict(
        max_freq = 20,
        time_range_tuple = (0,7),
        Fs = 1000,
        signal_window = 400,
        window_overlap = 399
        )

dat = emg_env[0,0]
x = np.arange(len(dat)) / 1000
freq_vec, t_vec, stft  = calc_stft(dat, **stft_params)
mag = np.abs(stft)
# max_mag = np.zeros(mag.shape)
# max_mag[np.argmax(mag, axis=0), np.arange(mag.shape[1])] = 1
# power = np.abs(stft)**2
# log_spectrum = 20*np.log10(np.abs(stft))
# time_norm_mag = mag / mag.sum(axis=0)
# weight_mean = (freq_vec[:,None] * time_norm_mag).sum(axis=0)
freq_vec, t_vec, weight_mean = calc_stft_mode_freq(dat, **stft_params)
fig,ax = plt.subplots(3,1, sharex=True)
ax[0].plot(x, dat)
ax[1].pcolormesh(t_vec, freq_vec, mag)
ax[2].pcolormesh(t_vec, freq_vec, max_mag)
ax[0].set_xlim(*stft_params['time_range_tuple'])
ax[2].plot(t_vec, weight_mean, color = 'r')
plt.show()

############################################################
inds = list(np.ndindex(emg_env.shape[:2]))
freq_vec, t_vec, test_stft  = calc_stft(emg_env[inds[0]], **stft_params)
stft_array = np.zeros((*emg_env.shape[:2], *test_stft.shape), dtype = np.complex)
for i in tqdm(inds):
    stft_array[i] = calc_stft(emg_env[i], **stft_params)[-1]

mag = np.abs(stft_array)
max_inds = np.argmax(mag, axis=2)
max_mag = np.zeros(mag.shape)
for i in inds:
    max_mag[i[0], i[1], max_inds[i], np.arange(len(max_inds[i]))] = 1

time_norm_mag = mag / mag.sum(axis=2)[:,:,None,:]
weight_mean = (freq_vec[:,None] * time_norm_mag).sum(axis=2)

#plt.pcolormesh(t_vec, freq_vec, max_mag[0,0])
##plt.imshow(mag[0,0], interpolation = 'nearest', aspect = 'auto')
#plt.show()
#
#mean_max_mag = max_mag.mean(axis=1)
#vz.firing_overview(mean_max_mag, cmap = 'viridis');plt.show()

############################################################
## Compare max STFT output with BSA
############################################################
#basename = dir_name.split('/')[-3]
#dir_list_path = '/media/bigdata/firing_space_plot/NM_gape_analysis/fin_NM_emg_dat.txt'
#dir_list = open(dir_list_path,'r').readlines()
#wanted_dir_path = [x for x in dir_list if basename in x][0].strip()
wanted_dir_path = "/".join(dir_name.split('/')[:-2])
wanted_h5_path = glob(os.path.join(wanted_dir_path,'*.h5'))[0]

h5 = tables.open_file(wanted_h5_path,'r')
bsa_p_nodes = [x for x in h5.get_node('/emg_BSA_results/emg')._f_iter_nodes() \
        if 'taste' in x.name]
bsa_out = np.stack([x[:] for x in bsa_p_nodes]).swapaxes(2,3)
#bsa_out = h5.get_node('/emg_BSA_results','taste0_p')[:]
bsa_freq = h5.get_node('/emg_BSA_results','omega')[:]
bsa_time = np.arange(bsa_out.shape[-1])/1000

h5.close()
##############################
# Convert BSA out and max_mag to timeseries rather than images
##############################
plot_dir = os.path.join(wanted_dir_path, 'bsa_stft_comparison')
if not os.path.isdir(plot_dir):
    os.mkdir(plot_dir)

bsa_inds = np.argmax(bsa_out, axis=2)
bsa_line = bsa_freq[bsa_inds]

stft_line = freq_vec[max_inds]

# Calculate mean abs distance between BSA and STFT (max and weighted mean)
bsa_stft_dist = np.abs(bsa_line - stft_line).mean(axis=-1)
bsa_stft_dist_w = np.abs(bsa_line - weight_mean).mean(axis=-1)

# Calculate same for 2000-4500 ms
bsa_stft_dist_2 = np.abs(bsa_line - stft_line)[...,2000:4500].mean(axis=-1)
bsa_stft_dist_w_2 = np.abs(bsa_line - weight_mean)[...,2000:4500].mean(axis=-1)

# Plot histogram of distances
fig,ax = plt.subplots(2,1, sharex=True)
ax[0].hist(bsa_stft_dist.flatten(), bins = 30, alpha = 0.5, label = 'Max')
ax[0].hist(bsa_stft_dist_w.flatten(), bins = 30, alpha = 0.5, label = 'Weighted')
# Also plot vline for mean values, and mark with text
ax[0].axvline(bsa_stft_dist.mean(), color = 'k', linestyle = '--', alpha = 0.5)
ax[0].axvline(bsa_stft_dist_w.mean(), color = 'k', linestyle = '--', alpha = 0.5)
ax[0].text(bsa_stft_dist.mean(), 0, 'Mean: {:.2f}'.format(bsa_stft_dist.mean()), rotation = 90, fontsize = 8)
ax[0].text(bsa_stft_dist_w.mean(), 0, 'Mean: {:.2f}'.format(bsa_stft_dist_w.mean()), rotation = 90, fontsize = 8)
ax[0].legend()
ax[1].hist(bsa_stft_dist_2.flatten(), bins = 30, alpha = 0.5, label = 'Max')
ax[1].hist(bsa_stft_dist_w_2.flatten(), bins = 30, alpha = 0.5, label = 'Weighted')
# Also plot vline for mean values, and mark with text
ax[1].axvline(bsa_stft_dist_2.mean(), color = 'k', linestyle = '--', alpha = 0.5)
ax[1].axvline(bsa_stft_dist_w_2.mean(), color = 'k', linestyle = '--', alpha = 0.5)
ax[1].text(bsa_stft_dist_2.mean(), 0, 'Mean: {:.2f}'.format(bsa_stft_dist_2.mean()), rotation = 90, fontsize = 8)
ax[1].text(bsa_stft_dist_w_2.mean(), 0, 'Mean: {:.2f}'.format(bsa_stft_dist_w_2.mean()), rotation = 90, fontsize = 8)
ax[1].set_xlabel('Distance (mean absolute)')
ax[0].set_ylabel('Count')
ax[1].set_ylabel('Count')
fig.suptitle('BSA vs STFT distance')
ax[0].set_title('All time')
ax[1].set_title('2000-4500 ms')
fig.savefig(os.path.join(plot_dir, 'bsa_stft_dist.png'), dpi = 300)
plt.close(fig)
#plt.show()

# Plot BSA and STFT lines
this_plot_dir = os.path.join(plot_dir, 'bsa_stft_raw_dat')
if not os.path.isdir(this_plot_dir):
    os.mkdir(this_plot_dir)
t_vec_s = t_vec.copy()
t_vec_l = np.arange(bsa_line.shape[-1])/1000

#ind = (3,1)
for ind in tqdm(np.ndindex(bsa_out.shape[:2])):
    fig,ax = plt.subplots(6,1, sharex=True, figsize = (7,10))
    ax[0].pcolormesh(bsa_time, bsa_freq, bsa_out[ind])
    ax[1].pcolormesh(t_vec, freq_vec, max_mag[ind])
    ax[2].pcolormesh(t_vec, freq_vec, time_norm_mag[ind])
    ax[2].plot(t_vec, weight_mean[ind], color = 'y', linewidth = 3,
               label = 'Weighted Mean')
    ax[2].legend()
    ax[0].set_ylim([0,13])
    ax[1].set_ylim([0,13])
    ax[2].set_ylim([0,13])
    ax[4].set_ylim([0,13])
    ax[5].set_ylim([0,13])
    ax[3].plot(t_vec_l, emg_env[ind])
    ax[4].plot(t_vec_l, bsa_line[ind], label = "BSA")
    ax[4].plot(t_vec_s, stft_line[ind], label = "STFT")
    ax[5].plot(t_vec_l, bsa_line[ind], label = "BSA")
    ax[5].plot(t_vec_l, weight_mean[ind], label = "weighted mean")
    ax[4].legend(loc = 'lower left')
    ax[5].legend(loc = 'lower left')
    ax[0].set_title('BSA')
    ax[1].set_title('STFT Max Power')
    ax[2].set_title('STFT and Weighted Mean')
    ax[3].set_title('EMG Envelope')
    ax[4].set_title('Overlay Comparison')
    ax[0].set_ylabel('Freq (Hz)')
    ax[1].set_ylabel('Freq (Hz)')
    ax[3].set_ylabel('Amplitude')
    ax[4].set_ylabel('Freq (Hz)')
    ax[-1].set_xlabel('Time (s)')
    fig.suptitle(f'Taste {ind[0]} Trial {ind[1]}')
    plt.tight_layout()
    #plt.show()
    fig.savefig(os.path.join(this_plot_dir, f'stft_bsa_comparison_{ind}.png'),
                bbox_inches = 'tight')
    plt.close(fig)


############################################################
############################################################
# Test all methods using dummy data
# Generate dummy data using random walk through frequency space

from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

t_max = 100
fs = 1000
dt = 1/fs
f_max = 10
n_anchors = 20

x_samples = np.arange(0, t_max*1.1, t_max/n_anchors)
freq_samples = np.random.random(x_samples.shape) * 10

x = np.arange(0, t_max, dt)
dx = np.full_like(x, dt)       # Change in x

interpolation = interp1d(x_samples, freq_samples, kind='quadratic')
freq = interpolation(x)

x_plot = (freq * dx).cumsum()    # Cumsum freq * change in x
y = np.sin(x_plot)

stft_params = dict(
        max_freq = 20,
        time_range_tuple = (0,t_max),
        Fs = fs,
        signal_window = 1000,
        window_overlap = 950
        )

freq_vec, t_vec, weight_mean = calc_stft_mode_freq(y, **stft_params)

plt.plot(x, y, label="sin(freq(x) * x)")
plt.plot(x, freq, label="freq(x)")
plt.plot(t_vec, weight_mean, label="Weighted Mean")
plt.legend()
plt.show()

freq, t, stft = calc_stft(y, **stft_params)
plt.pcolormesh(t, freq, np.abs(stft))
plt.show()
