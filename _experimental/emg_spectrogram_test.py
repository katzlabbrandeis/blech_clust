"""
Comparison of information in filtered emg vs envelope
"""

# Import stuff
import numpy as np
import scipy
from scipy.signal import butter, filtfilt, periodogram
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
                window='hanning',
                nperseg=signal_window,
                noverlap=signal_window-(signal_window-window_overlap))
    this_stft =  this_stft[np.where(f<max_freq)[0]]
    this_stft = this_stft[:,np.where((t>=time_range_tuple[0])*\
                                            (t<time_range_tuple[1]))[0]]
    fin_freq = f[f<max_freq]
    fin_t = t[np.where((t>=time_range_tuple[0])*(t<time_range_tuple[1]))]
    return  fin_freq, fin_t, this_stft


# Ask for the directory where the data (emg_data.npy) sits
dir_name = '/media/bigdata/firing_space_plot/NM_gape_analysis/raw_data/NM51_2500ms_161030_130155/emg_0'
#dir_name = easygui.diropenbox()
os.chdir(dir_name)

#emg_data = np.load('emg_data.npy')
emg_filt = np.load('emg_filt.npy')
emg_env = np.load('env.npy')
sig_trials = np.load('sig_trials.npy')

stft_params = dict(
        max_freq = 20,
        time_range_tuple = (0,7),
        Fs = 1000,
        signal_window = 300,
        window_overlap = 299
        )

dat = emg_env[0,0]
x = np.arange(len(dat)) / 1000
freq_vec, t_vec, stft  = calc_stft(dat, **stft_params)
mag = np.abs(stft)
max_mag = np.zeros(mag.shape)
max_mag[np.argmax(mag, axis=0), np.arange(mag.shape[1])] = 1
power = np.abs(stft)**2
log_spectrum = 20*np.log10(np.abs(stft))

fig,ax = plt.subplots(3,1, sharex=True)
ax[0].plot(x, dat)
ax[1].pcolormesh(t_vec, freq_vec, mag)
ax[2].pcolormesh(t_vec, freq_vec, max_mag) 
ax[0].set_xlim(*stft_params['time_range_tuple'])
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

plt.pcolormesh(t_vec, freq_vec, max_mag[0,0])
#plt.imshow(mag[0,0], interpolation = 'nearest', aspect = 'auto')
plt.show()

mean_max_mag = max_mag.mean(axis=1)
vz.firing_overview(mean_max_mag, cmap = 'viridis');plt.show()

############################################################
## Compare max STFT output with BSA
############################################################
basename = dir_name.split('/')[-2]
dir_list_path = '/media/bigdata/firing_space_plot/NM_gape_analysis/fin_NM_emg_dat.txt'
dir_list = open(dir_list_path,'r').readlines()
wanted_dir_path = [x for x in dir_list if basename in x][0].strip()
wanted_h5_path = glob(os.path.join(wanted_dir_path,'*.h5'))[0]

h5 = tables.open_file(wanted_h5_path,'r')
bsa_p_nodes = [x for x in h5.get_node('/emg_BSA_results')._f_iter_nodes() \
        if 'taste' in x.name]
bsa_out = np.stack([x[:] for x in bsa_p_nodes]).swapaxes(2,3)
#bsa_out = h5.get_node('/emg_BSA_results','taste0_p')[:]
bsa_freq = h5.get_node('/emg_BSA_results','omega')[:]
bsa_time = np.arange(bsa_out.shape[-1])/1000

h5.close()
##############################
# Convert BSA out and max_mag to timeseries rather than images
##############################
bsa_inds = np.argmax(bsa_out, axis=2)
bsa_line = bsa_freq[bsa_inds]

stft_line = freq_vec[max_inds]

plot_dir = '/media/bigdata/firing_space_plot/NM_gape_analysis/plots'

ind = (2,0)
fig,ax = plt.subplots(5,1, sharex=True, figsize = (7,7))
ax[0].pcolormesh(bsa_time, bsa_freq, bsa_out[ind]) 
ax[1].pcolormesh(t_vec, freq_vec, max_mag[ind])
ax[0].set_ylim([0,13])
ax[1].set_ylim([0,13])
ax[2].plot(t_vec, emg_env[ind])
ax[3].plot(t_vec, bsa_line[ind], label = "BSA")
ax[3].plot(t_vec, stft_line[ind], label = "STFT")
ax[4].plot(t_vec, bsa_line[ind], label = "BSA")
ax[4].plot(t_vec, savgol(stft_line[ind], 301, 2), label = "Filtered STFT")
ax[3].legend(loc = 'lower left')
ax[4].legend(loc = 'lower left')
ax[0].set_title('BSA')
ax[1].set_title('STFT Max Power')
ax[2].set_title('EMG Envelope')
ax[3].set_title('Overlay Comparison')
ax[0].set_ylabel('Freq (Hz)')
ax[1].set_ylabel('Freq (Hz)')
ax[2].set_ylabel('Amplitude')
ax[3].set_ylabel('Freq (Hz)')
ax[3].set_xlabel('Time (s)')
plt.tight_layout()
plt.show()
#fig.savefig(os.path.join(plot_dir, f'stft_bsa_comparison_{ind}.png'))
#plt.close(fig)

