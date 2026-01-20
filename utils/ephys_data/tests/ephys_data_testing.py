"""
Code to test run ephys_data module
"""

# from visualize import firing_overview, imshow
# from ephys_data import ephys_data
from blech_clust.utils.ephys_data import ephys_data
from blech_clust.utils.ephys_data.visualize import imshow, firing_overview
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import zscore

# os.chdir('/media/bigdata/firing_space_plot/ephys_data')
from importlib import reload
reload(ephys_data)

data_dir = '/media/storage/abu_resorted/bla_gc/AM35_4Tastes_201228_124547' 
dat = ephys_data.ephys_data(data_dir)

dat.firing_rate_params = dat.default_firing_params

dat.get_unit_descriptors()
dat.get_spikes()
dat.get_firing_rates()

firing_overview(dat.all_normalized_firing)
plt.show()

# shape: tastes, units, trials, time-bins
# dat.firing_array
# shape: tastes, units, trials, time
# np.array(dat.spikes)

# # Interpolate firing-rates to 1ms bins to match spike-trains
# spikes_len = dat.spikes[0].shape[-1]
# firing_len = dat.firing_array.shape[-1]
# n_repeats = np.ceil(spikes_len / firing_len).astype(int) 
# interp_firing_array = dat.firing_array.repeat(n_repeats, axis=-1)[..., :spikes_len]
#
# firing_overview(np.concatenate(interp_firing_array).swapaxes(0,1))
# plt.show()

bin_size = dat.firing_rate_params['step_size']
spike_array = np.array(dat.spikes)
binned_spikes = np.add.reduceat( 
    spike_array, np.arange(0, spike_array.shape[-1], bin_size), axis=-1)
# Chopped version to match firing rates
binned_spikes = binned_spikes[..., :dat.firing_array.shape[-1]]

firing_overview(np.concatenate(binned_spikes).swapaxes(0,1))
firing_overview(np.concatenate(dat.firing_array.swapaxes(1,2)).swapaxes(0,1))
plt.show()

# Match dimensions
# Shapes: (units, tastes*trials, time-bins)
# matched_binned_spikes = np.concatenate(binned_spikes).swapaxes(0,1)
# matched_firing_array = np.concatenate(dat.firing_array.swapaxes(1,2)).swapaxes(0,1)
matched_binned_spikes = np.concatenate(binned_spikes.swapaxes(1,2))
matched_firing_array = np.concatenate(dat.firing_array)

time_vec = dat.time_vector.copy()
assert len(time_vec) == matched_firing_array.shape[-1], "Time vector length does not match data length"

wanted_time_lims = [-500, 2000]
time_inds = np.where(
    (time_vec >= wanted_time_lims[0]) &
    (time_vec <= wanted_time_lims[1])
    )[0]

matched_binned_spikes = matched_binned_spikes[..., time_inds]
matched_firing_array = matched_firing_array[..., time_inds]

# Calculate bits per spike for rolling window rates
bps_list = []
for nrn_ind in range(matched_firing_array.shape[0]):
    bps = dat.compute_bits_per_spike(
            matched_firing_array[nrn_ind],
            matched_binned_spikes[nrn_ind],
            )
    bps_list.append(bps)

# Sort by bits per spike
sorted_inds = np.argsort(bps_list)[::-1]
sorted_firing_array = matched_firing_array[sorted_inds]

fig, ax=  firing_overview(sorted_firing_array)
for this_ax, this_bps in zip(ax.flatten(), np.array(bps_list)[sorted_inds]): 
    this_ax.set_title(f'{this_bps:.3f}')
# plt.tight_layout()
plt.show()

# dat.get_lfps(re_extract= True)
dat.get_lfps()

dat.get_region_units()
dat.get_lfp_electrodes()

# dat.get_stft(recalculate = True)
dat.get_stft()

aggregate_amplitude = dat.get_mean_stft_amplitude()


def normalize_timeseries(array, time_vec, stim_time):
    mean_baseline = np.mean(
        array[..., time_vec < stim_time], axis=-1)[..., np.newaxis]
    array = array/mean_baseline
    return array


time_vec = dat.time_vec
stim_time = 2
fig, ax = plt.subplots(1, len(aggregate_amplitude))
for num, (region, this_ax) in enumerate(zip(aggregate_amplitude, ax.flatten())):
    this_ax.imshow(zscore(
        normalize_timeseries(region, time_vec, stim_time), axis=-1),
        aspect='auto', origin='lower')
    this_ax.set_title(dat.region_names[num])
    this_ax.set_yticks(np.arange(len(dat.freq_vec)))
    this_ax.set_yticklabels(dat.freq_vec)
plt.show()
