# Description: Test script for lfp artifact removal


############################################################
# Imports
############################################################

from scipy.stats import median_absolute_deviation as MAD
import pylab as plt
import numpy as np
from ephys_data import ephys_data
import os
import sys
ephys_data_dir = '/media/bigdata/firing_space_plot/ephys_data'
sys.path.append(ephys_data_dir)

############################################################
# Load Data
############################################################

dir_list_path = '/media/bigdata/projects/pytau/pytau/data/fin_inter_list_3_14_22.txt'
with open(dir_list_path, 'r') as f:
    dir_list = f.read().splitlines()
dir_name = dir_list[0]

dat = ephys_data(dir_name)
# Region lfps shape : (n_tastes, n_channels, n_trials, n_timepoints)
region_lfps, region_names = dat.return_region_lfps()

# Find channel per region closest to mean
# mean_region_lfp shape : (n_regions, n_tastes, n_trials, n_timepoints)
mean_region_lfp = np.stack([np.mean(x, axis=1) for x in region_lfps])

# Find channel per region closest to mean
wanted_channel_inds = []
for this_mean, this_region in zip(mean_region_lfp, region_lfps):
    diff_lfp = np.abs(this_region - this_mean[:, np.newaxis, :, :])
    mean_diff_lfp = diff_lfp.mean(axis=(0, 2, 3))
    min_diff_lfp = np.argmin(mean_diff_lfp)
    wanted_channel_inds.append(min_diff_lfp)

wanted_lfp_electrodes = np.array([x[:, y]
                                  for x, y in zip(region_lfps, wanted_channel_inds)])

# Plot mean ERP and per trial data for each region
flat_lfp = wanted_lfp_electrodes.reshape(
    wanted_lfp_electrodes.shape[0],
    -1,
    wanted_lfp_electrodes.shape[-1])
mean_lfp = np.mean(flat_lfp, axis=1)

# Plot mean ERP and per trial data for each region
fig, ax = plt.subplots(2, len(region_names), sharex=True, sharey='row')
for region_ind, (this_region, this_name) in enumerate(zip(flat_lfp, region_names)):
    ax[0, region_ind].imshow(this_region,
                             interpolation='nearest', aspect='auto')
    ax[0, region_ind].set_title(this_name)
    ax[1, region_ind].plot(mean_lfp[region_ind], color='k', linewidth=3)
    ax[1, region_ind].set_title(this_name)
plt.show()

############################################################
# Artifact Removal
############################################################
# Can do this in two ways:
# 1) Use something like:
# 1.1) MAD, or
# 1.2) deviation from t-distribution
# 2) Use ICA to decompose and manually remove components

# 1.1) MAD
lfp_median = np.median(flat_lfp, axis=1)
lfp_MAD = MAD(flat_lfp, axis=1)
MAD_threshold = 3

# Plot per_trial data with MAD threshold
fig, ax = plt.subplots(2, len(region_names), sharex=True, sharey='row')
for region_ind, (this_region, this_name) in enumerate(zip(flat_lfp, region_names)):
    ax[0, region_ind].imshow(this_region,
                             interpolation='nearest', aspect='auto')
    ax[0, region_ind].set_title(this_name)
    thresh_high = lfp_median[region_ind] + MAD_threshold*lfp_MAD[region_ind]
    thresh_low = lfp_median[region_ind] - MAD_threshold*lfp_MAD[region_ind]
    mask_bool = np.logical_or(this_region > thresh_high[np.newaxis, :],
                              this_region < thresh_low[np.newaxis, :])
    masked_region = np.ma.masked_where(mask_bool, this_region)
    cmap = plt.cm.viridis
    cmap.set_bad(color='red')
    ax[1, region_ind].imshow(masked_region,
                             interpolation='nearest', aspect='auto', cmap=cmap)
    ax[1, region_ind].set_title(this_name)
plt.show()

# Use total deviation per trial scaled by MAD to remove trial
mean_trial_deviation = np.mean(
    np.abs(flat_lfp - lfp_median[:, np.newaxis, :])/lfp_MAD[:, None], axis=2)
deviation_median = np.median(mean_trial_deviation, axis=1)
deviation_MAD = MAD(mean_trial_deviation, axis=1)
deviation_threshold = 3
fin_deviation_threshold = deviation_median + deviation_threshold*deviation_MAD

# Plot trial data and plot total deviation next to it
summed_mad_thresh = np.sum(MAD_threshold*lfp_MAD, axis=1)
fig, ax = plt.subplots(2, len(region_names), sharey=True, sharex='col')
for region_ind, (this_region, this_name) in enumerate(zip(flat_lfp, region_names)):
    ax[region_ind, 0].imshow(this_region,
                             interpolation='nearest', aspect='auto')
    ax[region_ind, 0].set_title(this_name)
    ax[region_ind, 1].plot(mean_trial_deviation[region_ind],
                           np.arange(this_region.shape[0]))
    ax[region_ind, 1].axvline(fin_deviation_threshold[region_ind], color='k')
    ax[region_ind, 1].set_title(this_name)
plt.show()

# Remove trials with high deviation
good_trials_bool = mean_trial_deviation < fin_deviation_threshold[:, np.newaxis]
# Take only trials good for both regions
good_trials_bool = np.all(good_trials_bool, axis=0)
# good_lfp_data = flat_lfp[:,good_trials_bool]
good_lfp_data = flat_lfp.copy()
good_lfp_data[:, ~good_trials_bool] = np.nan

# good_lfp_trials_bool = dat.lfp_processing.return_good_lfp_trial_inds(dat.all_lfp_array)
# good_lfp_data = dat.lfp_processing.return_good_lfp_trial_inds(dat.all_lfp_array)

# Plot full and good lfp trial data
fig, ax = plt.subplots(2, len(region_names), sharex=True, sharey=True)
for region_ind, (this_region, this_name) in enumerate(zip(flat_lfp, region_names)):
    ax[region_ind, 0].imshow(this_region,
                             interpolation='nearest', aspect='auto')
    ax[region_ind, 0].set_title(this_name)
    ax[region_ind, 1].imshow(good_lfp_data[region_ind],
                             interpolation='nearest', aspect='auto')
    ax[region_ind, 1].set_title(this_name)
    ax[region_ind, 0].set_ylabel('Trial Number')
plt.show()


def return_good_lfp_trial_inds(data, MAD_threshold=3, summed_MAD_threshold=3):
    """Return boolean array of good trials (for both regions) based on MAD threshold
    data : shape (n_regions, n_trials, n_timepoints)
    MAD_threshold : number of MADs to use as threshold for individual timepoints
    summed_MAD_threshold : number of MADs to use as threshold for summed MADs
    """
    lfp_median = np.median(data, axis=1)
    lfp_MAD = MAD(data, axis=1)
    # Use total deviation per trial scaled by MAD to remove trial
    mean_trial_deviation = np.mean(
        np.abs(data - lfp_median[:, np.newaxis, :])/lfp_MAD[:, None], axis=2)
    deviation_median = np.median(mean_trial_deviation, axis=1)
    deviation_MAD = MAD(mean_trial_deviation, axis=1)
    deviation_threshold = 3
    fin_deviation_threshold = deviation_median + deviation_threshold*deviation_MAD
    # Remove trials with high deviation
    good_trials_bool = mean_trial_deviation < fin_deviation_threshold[:, np.newaxis]
    # Take only trials good for both regions
    good_trials_bool = np.all(good_trials_bool, axis=0)
    return good_trials_bool


def return_good_lfp_trials(data, MAD_threshold=3, summed_MAD_threshold=3):
    """Return good trials (for both regions) based on MAD threshold
    data : shape (n_regions, n_trials, n_timepoints)
    MAD_threshold : number of MADs to use as threshold for individual timepoints
    summed_MAD_threshold : number of MADs to use as threshold for summed MADs
    """
    good_trials_bool = return_good_lfp_trial_inds(
        data, MAD_threshold, summed_MAD_threshold)
    good_lfp_data = data.copy()
    return good_lfp_data[:, good_trials_bool]
