# Use the results in Li et al. 2016 to get gapes on taste trials

import os
import sys
from glob import glob

import numpy as np
import tables
import pylab as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import white_tophat
from sklearn.neighbors import NeighborhoodComponentsAnalysis
import pandas as pd
from scipy.stats import zscore
from scipy.signal import welch

# Have to be in blech_clust/emg/gape_QDA_classifier dir
from detect_peaks import detect_peaks
from QDA_classifier import QDA
sys.path.append('../..')
from utils.blech_utils import imp_metadata
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
import itertools
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, AgglomerativeClustering


#from xgboost import XGBClassifier
##import shap
#from umap import UMAP


# TODO: Add function to check for and chop up segments with double peaks

def extract_movements(this_trial_dat, size = 250):
    filtered_dat = white_tophat(this_trial_dat, size=size)
    segments_raw = np.where(filtered_dat)[0]
    segments = np.zeros_like(filtered_dat)
    segments[segments_raw] = 1
    segment_starts = np.where(np.diff(segments) == 1)[0]
    segment_ends = np.where(np.diff(segments) == -1)[0]
    # If first start is after first end, drop first end
    # and last start
    if segment_starts[0] > segment_ends[0]:
        segment_starts = segment_starts[:-1]
        segment_ends = segment_ends[1:]
    segment_dat = [this_trial_dat[x:y]
                   for x, y in zip(segment_starts, segment_ends)]
    return segment_starts, segment_ends, segment_dat


def normalize_segments(segment_dat):
    """
    Perform min-max normalization on each segment
    And make length of each segment equal to the longest segment
    """
    max_len = max([len(x) for x in segment_dat])
    interp_segment_dat = [np.interp(
        np.linspace(0, 1, max_len),
        np.linspace(0, 1, len(x)),
        x)
        for x in segment_dat]
    interp_segment_dat = np.vstack(interp_segment_dat)
    # Normalize
    interp_segment_dat = interp_segment_dat - \
        np.min(interp_segment_dat, axis=-1)[:, None]
    interp_segment_dat = interp_segment_dat / \
        np.max(interp_segment_dat, axis=-1)[:, None]
    return interp_segment_dat


def extract_features(segment_dat, segment_starts, segment_ends):
    """
    # Features to extract
    # 1. Duration of movement
    # 2. Amplitude
    # 3. Left and Right intervals
    # 4. PCA of time-adjusted waveform
    """
    peak_inds = [np.argmax(x) for x in segment_dat]
    peak_times = [x+y for x, y in zip(segment_starts, peak_inds)]
    # Drop first and last segments because we can't get intervals for them
    segment_dat = segment_dat[1:-1]
    segment_starts = segment_starts[1:-1]
    segment_ends = segment_ends[1:-1]

    durations = [len(x) for x in segment_dat]
    amplitudes_rel = [np.max(x) - np.min(x) for x in segment_dat]
    amplitude_abs = [np.max(x) for x in segment_dat]
    left_intervals = [peak_times[i] - peak_times[i-1]
                      for i in range(1, len(peak_times))][:-1]
    right_intervals = [peak_times[i+1] - peak_times[i]
                       for i in range(len(peak_times)-1)][1:]
    interp_segment_dat = normalize_segments(segment_dat)
    pca_segment_dat = PCA(n_components=3).fit_transform(interp_segment_dat)

    welch_out = [welch(x, fs=1000, axis=-1) for x in segment_dat]
    max_freq = [x[0][np.argmax(x[1], axis=-1)] for x in welch_out]

    feature_list = [
        durations,
        amplitudes_rel,
        amplitude_abs,
        left_intervals,
        right_intervals,
        pca_segment_dat,
        max_freq,
    ]
    feature_list = [np.atleast_2d(x) for x in feature_list]
    feature_list = [x if len(x) == len(pca_segment_dat) else x.T
                    for x in feature_list]
    feature_array = np.concatenate(feature_list, axis=-1)

    feature_names = [
        'duration',
        'amplitude_rel',
        'amplitude_abs',
        'left_interval',
        'right_interval',
        'pca_1',
        'pca_2',
        'pca_3',
        'max_freq',
    ]
    return feature_array, feature_names, segment_dat, segment_starts, segment_ends

def find_segment(gape_locs, segment_starts, segment_ends):
    segment_bounds = list(zip(segment_starts, segment_ends))
    all_segment_inds = []
    for this_gape in gape_locs:
        this_segment_inds = []
        for i, bounds in enumerate(segment_bounds):
            if bounds[0] < this_gape < bounds[1]:
                this_segment_inds.append(i)
        if len(this_segment_inds) ==0:
            this_segment_inds.append(np.nan)
        all_segment_inds.append(this_segment_inds)
    return np.array(all_segment_inds).flatten()


############################################################
# Load Data
############################################################


# Ask for the directory where the hdf5 file sits, and change to that directory
# Get name of directory with the data files
#data_dir = '/media/fastdata/KM45/KM45_5tastes_210620_113227_new'
data_dir = '/home/abuzarmahmood/Desktop/blech_clust/pipeline_testing/test_data_handling/test_data/KM45_5tastes_210620_113227_new'
metadata_handler = imp_metadata([[], data_dir])
data_dir = metadata_handler.dir_name
os.chdir(data_dir)

# Open the hdf5 file
hf5 = tables.open_file(metadata_handler.hdf5_name, 'r+')

# Extract taste dig-ins from experimental info file
info_dict = metadata_handler.info_dict
params_dict = metadata_handler.params_dict
pre_stim, post_stim = params_dict['spike_array_durations']
taste_names = info_dict['taste_params']['tastes']

############################################################
# Load and Process Data
############################################################
emg_output_dir = os.path.join(data_dir, 'emg_output')
# Get dirs for each emg CAR
dir_list = glob(os.path.join(emg_output_dir, 'emg*'))
dir_list = [x for x in dir_list if os.path.isdir(x)]

# Load the unique laser duration/lag combos and the trials that correspond
# to them from the ancillary analysis node
# Shape : (laser conditions x trials per laser condition)
trials = hf5.root.ancillary_analysis.trials[:]
laser_cond_num = len(trials)
unique_lasers = hf5.root.ancillary_analysis.laser_combination_d_l[:]

# Pull out a specific channel
num = 0
dir_name = dir_list[num]

emg_basename = os.path.basename(dir_name)
print(f'Processing {emg_basename}')

if 'emg_env.npy' not in os.listdir(dir_name):
    raise Exception(f'emg_env.py not found for {dir_name}')
    exit()

os.chdir(dir_name)

# Paths for plotting
plot_dir = f'emg_output/gape_classifier_plots/overview/{emg_basename}'
fin_plot_dir = os.path.join(data_dir, plot_dir)
if not os.path.exists(fin_plot_dir):
    os.makedirs(fin_plot_dir)

# Load the required emg data (the envelope and sig_trials)
env = np.load('emg_env.npy')
num_tastes, num_trials, time_len = env.shape
env = np.vstack(env)
sig_trials = np.load('sig_trials.npy').flatten()

# Now arrange these arrays by (laser condition X taste X trials X time)
# Shape : (laser conditions x tastes x trials x time)
env_final = np.reshape(
    env,
    (
        laser_cond_num,
        num_tastes,
        int(num_trials/laser_cond_num),
        time_len
    ),
)

# Make an array to store gapes (with 1s)
gapes_Li = np.zeros(env_final.shape)

# Shape : (laser conditions x tastes x trials)
sig_trials_final = np.reshape(
    sig_trials,
    (
        laser_cond_num,
        num_tastes,
        int(num_trials/laser_cond_num),
    ),
)

# Also make an array to store the time of first gape on every trial
first_gape = np.empty(sig_trials_final.shape, dtype=int)

# Run through the trials and get burst times,
# intervals and durations. Also check if these bursts are gapes -
# if they are, put 1s in the gape array
inds = list(np.ndindex(sig_trials_final.shape[:3]))
segment_dat_list = []
for this_ind in inds:
    # Get peak indices
    this_trial_dat = env_final[this_ind]
    this_laser_prestim_dat = env_final[this_ind[0], :, :, :pre_stim]

    peak_ind = detect_peaks(
        this_trial_dat,
        mpd=85,
        mph=np.mean(this_laser_prestim_dat) +
        np.std(this_laser_prestim_dat)
    )

    segment_starts, segment_ends, segment_dat = extract_movements(
        this_trial_dat, size=200)

    #plt.plot(this_trial_dat, linewidth=2)
    #for this_start, this_end, this_dat in zip(segment_starts, segment_ends, segment_dat):
    #    plt.plot(np.arange(this_start, this_end), this_dat,
    #             linewidth = 5, alpha = 0.7)
    #plt.show()

    (feature_array,
     feature_names,
     segment_dat,
     segment_starts,
     segment_ends) = extract_features(
        segment_dat, segment_starts, segment_ends)

    segment_bounds = list(zip(segment_starts, segment_ends))
    merged_dat = [feature_array, segment_dat, segment_bounds]
    segment_dat_list.append(merged_dat)

    # Get the indices, in the smoothed signal,
    # that are below the mean of the smoothed signal
    below_mean_ind = np.where(this_trial_dat <=
                              np.mean(this_laser_prestim_dat))[0]

    # Throw out peaks if they happen in the pre-stim period
    accept_peaks = np.where(peak_ind > pre_stim)[0]
    peak_ind = peak_ind[accept_peaks]

    # Run through the accepted peaks, and append their breadths to durations.
    # There might be peaks too close to the end of the trial -
    # skip those. Append the surviving peaks to final_peak_ind
    durations = []
    final_peak_ind = []
    for peak in peak_ind:
        try:
            left_end = np.where(below_mean_ind < peak)[0][-1]
            right_end = np.where(below_mean_ind > peak)[0][0]
        except:
            continue
        dur = below_mean_ind[right_end]-below_mean_ind[left_end]
        if dur > 20.0 and dur <= 200.0:
            durations.append(dur)
            final_peak_ind.append(peak)
    durations = np.array(durations)
    peak_ind = np.array(final_peak_ind)

    # In case there aren't any peaks or just one peak
    # (very unlikely), skip this trial and mark it 0 on sig_trials
    if len(peak_ind) <= 1:
        sig_trials_final[this_ind] = 0
    else:
        # Get inter-burst-intervals for the accepted peaks,
        # convert to Hz (from ms)
        intervals = []
        for peak in range(len(peak_ind)):
            # For the first peak,
            # the interval is counted from the second peak
            if peak == 0:
                intervals.append(
                    1000.0/(peak_ind[peak+1] - peak_ind[peak]))
            # For the last peak, the interval is
            # counted from the second to last peak
            elif peak == len(peak_ind) - 1:
                intervals.append(
                    1000.0/(peak_ind[peak] - peak_ind[peak-1]))
            # For every other peak, take the largest interval
            else:
                intervals.append(
                    1000.0/(
                        np.amax([(peak_ind[peak] - peak_ind[peak-1]),
                                 (peak_ind[peak+1] - peak_ind[peak])])
                    )
                )
        intervals = np.array(intervals)

        # Now run through the intervals and durations of the accepted
        # movements, and see if they are gapes.
        # If yes, mark them appropriately in gapes_Li
        # Do not use the first movement/peak in the trial -
        # that is usually not a gape
        for peak in range(len(durations) - 1):
            gape = QDA(intervals[peak+1], durations[peak+1])
            if gape and peak_ind[peak+1] - pre_stim <= post_stim:
                gapes_Li[this_ind[0], this_ind[1], this_ind[2], peak_ind[peak+1]] = 1.0

        # If there are no gapes on a trial, mark these as 0
        # on sig_trials_final and 0 on first_gape.
        # Else put the time of the first gape in first_gape
        if np.sum(gapes_Li[this_ind]) == 0.0:
            sig_trials_final[this_ind] = 0
            first_gape[this_ind] = 0
        else:
            first_gape[this_ind] = np.where(
                gapes_Li[this_ind] > 0.0)[0][0]


############################################################
## Cluster waveforms
############################################################
# For each cluster, return:
# 1) Features
# 2) Mean waveform
# 3) Fraction of classifier gapes

# Convert segment_dat and gapes_Li to pandas dataframe for easuer handling
gape_frame = pd.DataFrame(data = inds,
                          columns = ['channel', 'taste', 'trial'])
# Standardize features
gape_frame['features'] = [x[0] for x in segment_dat_list]
gape_frame['segment_raw'] = [x[1] for x in segment_dat_list]
gape_frame['segment_bounds'] = [x[2] for x in segment_dat_list]
gape_frame = gape_frame.explode(['features','segment_raw','segment_bounds'])

# Standardize features
raw_features = np.stack(gape_frame['features'].values)
scaled_features = StandardScaler().fit_transform(raw_features)
gape_frame['features'] = [x for x in scaled_features]

# Add index for each segment
gape_frame['segment_num'] = gape_frame.groupby(['channel', 'taste', 'trial']).cumcount()
gape_frame = gape_frame.reset_index(drop=True)

# Add classifier boolean
for row_ind, this_row in gape_frame.iterrows():
    this_ind = (this_row['channel'], this_row['taste'], this_row['trial'])
    bounds = this_row['segment_bounds']
    if gapes_Li[this_ind][bounds[0]:bounds[1]].any():
        gape_frame.loc[row_ind, 'classifier'] = 1
    else:
        gape_frame.loc[row_ind, 'classifier'] = 0
# Convert to int
gape_frame['classifier'] = gape_frame['classifier'].astype(int)

############################################################
# Plot all segmented data for visual inspection
this_plot_dir = os.path.join(plot_dir, 'segmented_data')
if not os.path.exists(this_plot_dir):
    os.makedirs(this_plot_dir)

taste_groups = list(gape_frame.groupby(['taste']))
for taste_num, this_taste in taste_groups:
    trial_count = this_taste.trial.nunique()
    fig,ax = plt.subplots(trial_count,1, sharex=True, sharey=True,
                          figsize=(10,trial_count*2))
    for num, this_row in this_taste.iterrows():
        ax[this_row.trial].plot(np.arange(*this_row.segment_bounds), this_row.segment_raw,
                                linewidth = 3)
    for this_trial in range(env_final[:,taste_num].shape[1]):
        ax[this_trial].plot(env_final[:,taste_num][:,this_trial].flatten(),
                            color = 'k', linewidth = 0.5)
    fig.suptitle(str(this_taste.taste.unique()[0]))
    fig.savefig(os.path.join(this_plot_dir, 'taste_' + str(this_taste.taste.unique()[0]) + '.png'),
                dpi = 300, bbox_inches = 'tight')
    plt.close(fig)

############################################################

############################################################
# Compare clusters
def median_zscore(x, axis=0):
    """
    Subtract median and divide by MAD
    """
    return (x - np.median(x, axis=axis)) / np.median(np.abs(x - np.median(x, axis=axis)), axis=axis)

n_components = 20
#gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0)
#gmm.fit(X)
#labels = gmm.predict(X)
#kmeans = KMeans(n_clusters=n_components, random_state=0).fit(X)
#labels = kmeans.labels_
## Project features onto 3D
#pca = PCA(n_components=3)
#X_pca = pca.fit_transform(scaled_features)

# Use agglomerative clustering
clustering = AgglomerativeClustering(n_clusters=n_components).fit(scaled_features)
#clustering = AgglomerativeClustering(n_clusters=n_components).fit(X_pca)
labels = clustering.labels_

# Classifier gapes by labels
gape_bool = gape_frame['classifier'].values
class_gape_per_label = [np.mean(gape_bool[labels == x]) for x in range(n_components)]
class_gape_per_label = np.round(class_gape_per_label, 3)

# Plot
# Sorted features by label, concatenated with cluster_labels
sorted_labels = np.sort(labels)
sorted_features = np.stack(gape_frame['features'].values)[np.argsort(labels)]
#plot_dat = np.concatenate((sorted_features, sorted_labels[:,None]), axis=1)
plot_dat = sorted_features
plot_dat = median_zscore(plot_dat,axis=0)

# Plot n representative waveforms per cluster
# and overlay the mean waveform
plot_n = 50
clust_waveforms = []
mean_waveforms = []
for this_clust in range(n_components):
    clust_inds = np.where(labels == this_clust)[0]
    clust_dat = gape_frame['segment_raw'].values[clust_inds]
    # Get n random waveforms
    this_plot_n = np.min([plot_n, len(clust_dat)])
    rand_inds = np.random.choice(np.arange(len(clust_dat)), this_plot_n, replace=False)
    clust_waveforms.append(clust_dat[rand_inds])
    ## Get mean waveform
    #mean_waveforms.append(np.mean(clust_dat, axis=0))

# Plot
fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(1, 10)
ax = [fig.add_subplot(gs[0, :1]),
      fig.add_subplot(gs[0, 2:7]),
      fig.add_subplot(gs[0, 8:])]
im = ax[1].imshow(plot_dat,
          aspect='auto', cmap='viridis', interpolation='none',
          vmin=-5, vmax=5, origin='lower')
plt.colorbar(im, ax=ax[1])
ax[1].set_title('Features')
ax[1].set_xticks(np.arange(len(feature_names)))
ax[1].set_xticklabels(feature_names, rotation=90)
ax[2].barh(np.arange(n_components), class_gape_per_label)
ax[2].set_xlim(0,1)
ax[2].set_title('Mean Gape probability')
ax[2].set_xlabel('Probability')
ax[2].set_ylabel('Cluster')
# Add a subplot to plot the cluster labels and share the y-axis
# with the image
im = ax[0].imshow(sorted_labels[::-1, None], aspect='auto', cmap='tab20', interpolation='none',)
# Plot number of each cluster at center of cluster
for this_clust in range(n_components):
    ax[0].text(0, len(sorted_labels) - np.mean(np.where(sorted_labels == this_clust)), this_clust,
               ha='center', va='center', color='k', fontsize=12)
ax[0].set_title('Cluster labels')
fig.suptitle('Gape cluster breakdown')
fig.savefig(os.path.join(this_plot_dir, 'gape_cluster_breakdown.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

fig,ax = plt.subplots(n_components,1,sharex=True, sharey=True,
                      figsize=(3,n_components))
for this_clust in range(n_components):
    for this_wave in clust_waveforms[this_clust]:
        ax[this_clust].plot(this_wave, color='grey', alpha=0.3)
    ax[this_clust].set_title(f'Cluster {this_clust}, mean_prob = {class_gape_per_label[this_clust]}')
ax[-1].set_xlabel('Time (ms)')
plt.suptitle('Random waveforms from each cluster')
fig.savefig(os.path.join(this_plot_dir, 'gape_cluster_waveforms.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

############################################################
############################################################
# If given a waveform or set of waveforms, find the closest cluster

# Cluster with highest probability of being a gape
gape_clust = np.argmax(class_gape_per_label)

# Extract waveforms from gape cluster
gape_waveforms = clust_waveforms[gape_clust]
