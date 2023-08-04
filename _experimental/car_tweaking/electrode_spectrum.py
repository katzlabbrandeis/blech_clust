import numpy as np
import tables
from glob import glob
import pylab as plt
from scipy.signal import spectrogram

data_dir = '/media/storage/gc_only/AS18/AS18_4Tastes_200228_151511_old_car/'
hf5_path = glob(data_dir + '*.h5')[0] 	
hf5 = tables.open_file(hf5_path, 'r')
raw_electrodes_list = hf5.list_nodes('/raw')
dat_len = raw_electrodes_list[0].shape[0]
dat_sd = [np.std(node[::1000]) for node in raw_electrodes_list]
dat_mean_sd = np.mean(dat_sd)

def return_dat_cut(start, end):
	return [node[start:end] for node in raw_electrodes_list]

window_len = dat_len//10000


time_lims = [
		[4410966, 4422080],
		[4165312, 4177026],
		[4574335, 4585449],
		[5881288, 5892402],
		]
t_diff = [x[1] - x[0] for x in time_lims]
min_t_diff = min(t_diff)
time_lims = [[x[0], x[0] + min_t_diff] for x in time_lims]
dat_cut = np.stack([np.stack(return_dat_cut(*x)) for x in time_lims])

#fig,ax = plt.subplots(len(dat_cut),1)
#for i in range(len(dat_cut)):
#	ax[i].plot(dat_cut[i].T)
#plt.show()

car_dat = [x - np.mean(x, axis=0) for x in dat_cut]


# Generate spectrograms for each electrode
def iter_spectrogram(x, fs=30000, nperseg=500, noverlap=100, nfft=1000):
	f, t, Sxx = spectrogram(x, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
	return Sxx

fs = 30000
nperseg = 100
noverlap = 10
nfft = 100
spec_params = {'fs':fs, 'nperseg':nperseg, 'noverlap':noverlap, 'nfft':nfft}

f, t, _ = spectrogram(dat_cut[0], **spec_params)
t = np.linspace(0, min_t_diff, len(t))
Sxx_raw = [iter_spectrogram(x,**spec_params) for x in dat_cut]
Sxx_car = [iter_spectrogram(x,**spec_params) for x in car_dat]

# Use bandpass signal from 750-1750 Hz as marker for noisy periods
from scipy.signal import butter, sosfiltfilt
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	sos = butter(order, [low, high], analog=False, btype='bandpass', output='sos')
	y = sosfiltfilt(sos, data)
	return y

car_dat_filt = [butter_bandpass_filter(x, 750, 1750, fs) for x in car_dat]
mean_abs_filt = [np.mean(np.abs(x),axis=0) for x in car_dat_filt]

# Now low-pass filter the mean_abs_filt to get a smoothed version
mean_abs_filt_smooth = [butter_bandpass_filter(x, 1, 500, fs) for x in mean_abs_filt]

# Plot mean spectrum
fig,ax = plt.subplots(3,len(Sxx_raw), sharex='col', sharey='row')
for i in range(len(Sxx_raw)):
	ax[0,i].plot(dat_cut[i].T, alpha=0.1)
	ax[0,i].plot(mean_abs_filt_smooth[i], color = 'k', alpha = 0.7)
	ax[1,i].pcolormesh(t, f, Sxx_raw[i].mean(axis=0), cmap='viridis', shading='gouraud')
	ax[2,i].pcolormesh(t, f, Sxx_car[i].mean(axis=0), cmap='viridis', shading='gouraud')
plt.show()



## Apply bandstop filter to remove 500-1600 Hz from car_dat
#from scipy.signal import butter, sosfiltfilt
#def butter_bandstop_filter(data, lowcut, highcut, fs, order=5):
#	nyq = 0.5 * fs
#	low = lowcut / nyq
#	high = highcut / nyq
#	sos = butter(order, [low, high], analog=False, btype='bandstop', output='sos')
#	y = sosfiltfilt(sos, data)
#	return y
#
#car_dat_filt = [butter_bandstop_filter(x, 500, 1600, fs) for x in car_dat]
#Sxx_car_filt = [iter_spectrogram(x) for x in car_dat_filt]
#
#
## Plot mean spectrum
#fig,ax = plt.subplots(5,len(Sxx_raw), sharex='col', sharey='row')
#for i in range(len(Sxx_raw)):
#	ax[0,i].plot(dat_cut[i].T, alpha=0.1)
#	ax[1,i].pcolormesh(t, f, Sxx_raw[i].mean(axis=0), cmap='viridis', shading='gouraud')
#	ax[2,i].pcolormesh(t, f, Sxx_car[i].mean(axis=0), cmap='viridis', shading='gouraud')
#	ax[3,i].plot(car_dat_filt[i].T, alpha=0.1)
#	ax[4,i].pcolormesh(t, f, Sxx_car_filt[i].mean(axis=0), cmap='viridis', shading='gouraud')
#plt.show()
#
#
