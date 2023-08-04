"""
Extract periods of noise from electrode data
Order of operations
1. Bandpass filter data from 750-1750 Hz
2. Calculate mean of absolute value of signal across all electrodes
3. Smooth signal with 1-500 Hz bandpass filter
"""

import numpy as np
import tables
from glob import glob
import pylab as plt
from scipy.signal import spectrogram
from tqdm import tqdm

def calc_threshold(filt_el, threshold_mult=5):
	m = np.mean(filt_el)
	th = threshold_mult*np.median(np.abs(filt_el)/0.6745)
	return m, th

# Use bandpass signal from 750-1750 Hz as marker for noisy periods
from scipy.signal import butter, sosfiltfilt
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	sos = butter(order, [low, high], analog=False, btype='bandpass', output='sos')
	y = sosfiltfilt(sos, data)
	return y


data_dir = '/media/storage/gc_only/AS18/AS18_4Tastes_200228_151511_old_car/'
hf5_path = glob(data_dir + '*.h5')[0] 	
hf5 = tables.open_file(hf5_path, 'r')
car_electrodes = hf5.list_nodes('/raw')

noise_band = [750, 1750]
smooth_band = [1,500]
fs = 30000

# Noise bandpass filter
noise_filt = []
for electrode in car_electrodes:
	print(electrode)
	data = electrode[:].flatten()
	data = butter_bandpass_filter(data, noise_band[0], noise_band[1], fs, order=5)
	noise_filt.append(data)

# Calculate mean of absolute value of signal across all electrodes
noise_filt = [np.abs(x) for x in noise_filt] 
noise_filt = np.mean(noise_filt, axis=0)

# Smooth signal with 1-500 Hz bandpass filter
smooth_filt = butter_bandpass_filter(noise_filt, smooth_band[0], smooth_band[1], fs, order=5)

noise_median = np.median(smooth_filt)
noise_MAD = np.median(np.abs(smooth_filt - noise_median))

# Plot fraction of time that is noise against MAD threshold
MAD_thresholds = np.arange(0, 100, 1)
frac_noise = []
for MAD_threshold in MAD_thresholds:
	print(MAD_threshold)
	noise_periods = np.where(smooth_filt > noise_median + MAD_threshold*noise_MAD)[0]
	frac_noise.append(len(noise_periods)/len(smooth_filt))

plt.plot(MAD_thresholds, frac_noise, '-x')
plt.xlabel('MAD threshold')
plt.ylabel('Fraction of time that is noise')
plt.show()

MAD_threshold = 5

# Find periods of noise
noise_periods = np.where(smooth_filt > noise_median + MAD_threshold*noise_MAD)[0]
not_noise_periods = np.where(smooth_filt < noise_median + MAD_threshold*noise_MAD)[0]

# Fraction of time that is noise
print(len(noise_periods)/len(smooth_filt))

inds = [0,1000000]
# Get smooth filt greater than threshold
smooth_filt_plot = smooth_filt[inds[0]:inds[1]]
noise_inds = np.where(smooth_filt_plot > noise_median + MAD_threshold*noise_MAD)[0]
plt.plot(smooth_filt_plot, 'k', linewidth=2, zorder = 10)
plt.plot(noise_inds, smooth_filt_plot[noise_inds], 'r.', markersize=5, zorder = 10)
for this_dat in car_electrodes:
	plt.plot(this_dat[inds[0]:inds[1]], alpha=0.1)
plt.axhline(noise_median + MAD_threshold*noise_MAD, color='r', linestyle='--')
plt.show()

##############################
# Calculate threshold before and after taking out noise

pre_thresholds = [calc_threshold(x[::100]) for x in tqdm(car_electrodes)]
post_thresholds = [calc_threshold(x[not_noise_periods][::100]) for x in tqdm(car_electrodes)]
pre_thresholds = np.array([x[1] for x in pre_thresholds])
post_thresholds = np.array([x[1] for x in post_thresholds])

print(np.vstack((pre_thresholds, post_thresholds)).T)
