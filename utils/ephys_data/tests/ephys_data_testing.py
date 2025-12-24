"""
Code to test run ephys_data module
"""

# from visualize import firing_overview, imshow
# from ephys_data import ephys_data
from blech_clust.utils.ephys_data import ephys_data
from blech_clust.utils.ephys_data.visualize import firing_overview, imshow
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import zscore

# os.chdir('/media/bigdata/firing_space_plot/ephys_data')
# data_dir = '/media/bigdata/Abuzar_Data/bla_gc/AM11/AM11_4Tastes_191030_114043_copy'
data_dir = '/media/storage/NM_resorted_data/laser_2500ms/NM43_2500ms_160515_104159'

from importlib import reload

reload(ephys_data)
dat = ephys_data.ephys_data(data_dir)
dat.firing_rate_params = dat.default_firing_params
dat.profile_units(recalculate=True)

firing_overview(dat.all_normalized_firing)
plt.show()

dat.separate_laser_firing()
stacked_off_firing = np.concatenate(dat.off_firing).swapaxes(0,1)

# Convert laser_tuple to str
unit_profile = dat.unit_profile.copy()
unit_profile['laser_tuple'] = unit_profile['laser_tuple'].apply(lambda x: str(x))
off_profile = dat.unit_profile.loc[dat.unit_profile.laser_tuple=='(0, 0)']

# Plot all units for off_firing and their p-values
pval_cols = [col for col in off_profile.columns if 'pval' in col]

for unit_id, unit_firing in enumerate(stacked_off_firing):
    pvals = off_profile.loc[off_profile.neuron_num==unit_id, pval_cols]
    fig, ax = plt.subplots(1, 1)
    ax.imshow(unit_firing, aspect='auto', origin='lower')
    ax.set_title(f'Unit {unit_id}\n{pvals.T}') 
    ax.set_xlabel('Time Bins')
    ax.set_ylabel('Trials')
    plt.tight_layout()
plt.show()

firing_overview(stacked_off_firing)
plt.show()

dat.get_unit_descriptors()
dat.get_spikes()
dat.get_firing_rates()
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
