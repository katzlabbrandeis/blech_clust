"""
Compare emg_data, emg_env, emg_filt
"""

import numpy as np
import os
import pylab as plt
from scipy.stats import zscore
from scipy.signal import correlate


dirs = [
    '/media/fastdata/KM45/KM45_5tastes_210620_113227_old',
    '/media/fastdata/KM45/KM45_5tastes_210620_113227_new/emg_output'
]

emg_data_list = []
for this_dir in dirs:
    this_dat = np.load(
        os.path.join(this_dir, 'emg_data.npy')
    )
    emg_data_list.append(this_dat)

emg_shapes = [x.shape for x in emg_data_list]

reshape_emg_data = [np.swapaxes(x, 0, 1).reshape(
    (-1, 25, 7000)) for x in emg_data_list]

fig, ax = plt.subplots(2, 1)
for this_dat, this_ax in zip(reshape_emg_data, ax.flatten()):
    this_ax.imshow(this_dat[:, :, 0], interpolation='nearest', aspect='auto')
plt.show()

# plot_dat = np.array(list(zip(*[x[0] for x in reshape_emg_data])))
plot_dat = np.array([x[0, 3] for x in emg_data_list]).swapaxes(0, 1)
fig, ax = plt.subplots(25, 1, sharey=True)
for this_dat, this_ax in zip(plot_dat, ax.flatten()):
    zscore_dat = zscore(this_dat, axis=-1)
    this_ax.plot(zscore_dat.T)
plt.show()

# Cross corelation of traces
xcorr = [correlate(x[0], x[1]) for x in plot_dat]
# fig,ax = plt.subplots(25,1, sharey=True)
# for this_dat, this_ax in zip(xcorr, ax.flatten()):
#    zscore_dat = zscore(this_dat, axis=-1)
#    this_ax.plot(zscore_dat)
# plt.show()

max_peak_inds = np.array([np.argmax(x) for x in xcorr])
offset = max_peak_inds - len(xcorr[0])/2
print(offset)
