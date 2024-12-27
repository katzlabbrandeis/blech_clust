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

from detect_peaks import detect_peaks
from QDA_classifier import QDA
sys.path.append('../..')
from utils.blech_utils import imp_metadata
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
import itertools
from sklearn.mixture import GaussianMixture

from umap import UMAP

from xgboost import XGBClassifier
import shap


# TODO: Add function to check for and chop up segments with double peaks

def extract_movements(this_trial_dat, size):
    filtered_dat = white_tophat(this_trial_dat, size=200)
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

    feature_list = [
        durations,
        amplitudes_rel,
        amplitude_abs,
        left_intervals,
        right_intervals,
        pca_segment_dat
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

psth_durs = params_dict['psth_params']['durations']
psth_durs[0] *= -1
psth_inds = [int(x + pre_stim) for x in psth_durs]

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

    #sizes = [50, 100, 150, 200]
    #fig, ax = plt.subplots(len(sizes)+1, 1, sharex=True, sharey=True)
    #ax[0].plot(this_trial_dat)
    #for i, size in enumerate(sizes):
    #   ax[i+1].plot(white_tophat(this_trial_dat, size))
    #plt.show()

    segment_starts, segment_ends, segment_dat = extract_movements(
        this_trial_dat, size=200)

    ## Plot the segments
    #fig, ax = plt.subplots(2,1, sharex=True)
    #for this_start, this_end, this_dat in zip(segment_starts, segment_ends, segment_dat):
    #    ax[0].fill_between(np.arange(this_start, this_end), 0, this_dat, alpha = 0.7)
    #    ax[0].plot(np.arange(this_start, this_end), this_dat)
    #ax[1].plot(this_trial_dat)
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

    #scaled_feature_array = StandardScaler().fit_transform(feature_array)

    #plt.matshow(scaled_feature_array)
    #plt.show()

    #plt.plot(this_trial_dat)
    #for start, end, dat in zip(segment_starts, segment_ends, segment_dat):
    #    plt.plot(np.arange(start, end), dat, linewidth=3)
    #plt.show()

    #plt.plot(this_trial_dat)
    #plt.plot(peak_ind, this_trial_dat[peak_ind], 'ro')
    #plt.axhline(np.mean(this_laser_prestim_dat) +
    #              np.std(this_laser_prestim_dat),
    #              color='red', linestyle='--')
    #plt.show()

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
## Compare early vs late gapes
############################################################

# Plot overview of gape position per taste
plot_gapes = np.squeeze(gapes_Li)
fig, ax = plt.subplots(len(plot_gapes), 1, sharex=True, sharey=True)
for this_ind, this_ax in zip(range(len(plot_gapes)), ax):
    this_dat = plot_gapes[this_ind]
    spike_inds = np.where(this_dat)
    this_ax.scatter(spike_inds[1], spike_inds[0], s=1)
    this_ax.set_title(taste_names[this_ind])
plt.show()

quin_dat = plot_gapes[0]
quin_inds = np.where(quin_dat)

gape_dividers = [2600, 4000]
plt.hist(quin_inds[1], bins=50)
plt.xlim(2000,7000)
for x in gape_dividers:
    plt.axvline(x, color='red', linestyle='--')
plt.show()

# For each gape indicated by gapes_Li, find corresponding segment
# and see if the different kinds of gapes are separable
trial_inds_inds = [i for i, this_ind in enumerate(inds) if this_ind[:2] == (0,0)]
trial_inds = [inds[i] for i in trial_inds_inds]

wanted_gapes_Li = [gapes_Li[this_ind] for this_ind in trial_inds]
wanted_gapes_loc = [np.where(this_gape)[0] for this_gape in wanted_gapes_Li]

all_locs = []
gape_starts = []
non_gape_starts = []
all_gape_features = []
all_non_gape_features = []
for trial_num in range(len(trial_inds)):
    this_starts, this_ends = list(zip(*segment_dat_list[trial_num][-1]))
    this_starts = np.array(this_starts)
    this_ends = np.array(this_ends)
    gape_segment_inds = find_segment(wanted_gapes_loc[trial_num], this_starts, this_ends)
    #print(gape_segment_inds)
    not_nan_inds = np.where(~np.isnan(gape_segment_inds))[0]
    gape_segment_inds = gape_segment_inds[not_nan_inds]
    gape_segment_inds = np.vectorize(int)(gape_segment_inds)
    temp_wanted_gapes = wanted_gapes_loc[trial_num][not_nan_inds]
    gape_segment_features = segment_dat_list[trial_num][0][gape_segment_inds]
    non_gape_inds = [i for i in range(len(this_starts)) if i not in gape_segment_inds]
    non_gape_features = segment_dat_list[trial_num][0][non_gape_inds]
    gape_starts.append(this_starts[gape_segment_inds])
    non_gape_starts.append(this_starts[non_gape_inds])
    all_non_gape_features.append(non_gape_features)
    all_locs.append(temp_wanted_gapes)
    all_gape_features.append(gape_segment_features)
all_locs = np.concatenate(all_locs)
all_gape_features = np.concatenate(all_gape_features)
all_non_gape_features = np.concatenate(all_non_gape_features)
gape_starts = np.concatenate(gape_starts)
non_gape_starts = np.concatenate(non_gape_starts)

print(all_locs.shape)
print(all_gape_features.shape)

X_raw = all_gape_features.copy()
X = StandardScaler().fit_transform(X_raw)
y = np.digitize(all_locs, gape_dividers)

# Create scatter plots of the NCA data for each pair of groups
groups = np.unique(y)
group_names = ['Early', 'Middle', 'Late']
group_pairs = list(itertools.combinations(groups, 2))

fig, ax = plt.subplots(1, len(group_pairs),
                       figsize = (10,3))#, sharex=True, sharey=True)
for i, group_pair in enumerate(group_pairs):
    this_inds = np.where(np.isin(y, group_pair))[0]
    this_X = X[this_inds]
    this_y = y[this_inds]
    this_labels = [group_names[i] for i in group_pair]
    nca = NeighborhoodComponentsAnalysis(n_components=2, random_state=42)
    X_nca = nca.fit_transform(this_X, this_y)
    # Calculate classification accuracy
    clf = SVC(kernel='rbf', C=1)
    scores = cross_val_score(clf, this_X, this_y, cv=5)
    #print('Accuracy for {} vs {}: {:.2f} +/- {:.2f}'.format(
    #    this_labels[0], this_labels[1], np.mean(scores), np.std(scores)))
    scatter = ax[i].scatter(X_nca[:, 0], X_nca[:, 1],
                            c=this_y, cmap='rainbow', alpha = 0.7)
    ax[i].legend(handles=scatter.legend_elements()[0],
               labels=this_labels)
    ax[i].set_title('Accuracy: {:.2f} +/- {:.2f}'.format(
        np.mean(scores), np.std(scores)))
plt.show()

############################################################
# Run XGBoost with SHAP analysis to see which features are most important
# for differentiating early vs late gapes
############################################################
clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
clf.fit(X, y)
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X, plot_type="bar",
                  feature_names=feature_names)
# Cross-validation accuracy
scores = cross_val_score(clf, X, y, cv=5)
print('Accuracy: {:.2f} +/- {:.2f}'.format(
    np.mean(scores), np.std(scores)))

# Cross-validate using svm with rbf kernel
clf = SVC(kernel='rbf', C=1)
scores = cross_val_score(clf, X, y, cv=5)
print('Accuracy: {:.2f} +/- {:.2f}'.format(
    np.mean(scores), np.std(scores)))

############################################################
## Compare gapes with non-gapes
############################################################
X_raw = np.concatenate((all_gape_features, all_non_gape_features))
X = StandardScaler().fit_transform(X_raw)
y = np.concatenate((np.ones(all_gape_features.shape[0]),
                    np.zeros(all_non_gape_features.shape[0])))

# Plot X as matrix
plt.imshow(X, aspect='auto', interpolation='none')
plt.show()

# Run Classifier
clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
clf.fit(X, y)
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X, plot_type="bar",
                  feature_names=feature_names)
# Cross-validation accuracy
scores = cross_val_score(clf, X, y, cv=5)
print('Accuracy: {:.2f} +/- {:.2f}'.format(
    np.mean(scores), np.std(scores)))

# Cross-validate using svm with rbf kernel
clf = SVC(kernel='rbf', C=1)
scores = cross_val_score(clf, X, y, cv=5)
print('Accuracy: {:.2f} +/- {:.2f}'.format(
    np.mean(scores), np.std(scores)))

# Create umap plot
reducer = UMAP()
embedding = reducer.fit_transform(X)
plt.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap='rainbow')
plt.show()

# Create NCA plot
nca = NeighborhoodComponentsAnalysis(n_components=3, random_state=42)
X_nca = nca.fit_transform(X, y)

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_nca[:, 0], X_nca[:, 1], X_nca[:, 2], c=y, cmap='rainbow')
plt.show()

plt.scatter(X_nca[:, 0], X_nca[:, 1], c=y,
            cmap='rainbow', alpha = 0.5)
plt.show()

############################################################
## Include time as a feature
############################################################
X_raw = np.concatenate((all_gape_features, all_non_gape_features))
all_times = np.concatenate((gape_starts, non_gape_starts))
X_raw = np.concatenate((X_raw, all_times[:, np.newaxis]), axis=1)
feature_names = np.concatenate((feature_names, ['Time']))
X = StandardScaler().fit_transform(X_raw)
y = np.concatenate((np.ones(all_gape_features.shape[0]),
                    np.zeros(all_non_gape_features.shape[0])))

# Plot X as matrix
plt.imshow(X, aspect='auto', interpolation='none')
plt.show()

# Run Classifier
clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
clf.fit(X, y)
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X, plot_type="bar",
                  feature_names=feature_names)
# Cross-validation accuracy
scores = cross_val_score(clf, X, y, cv=5)
print('Accuracy: {:.2f} +/- {:.2f}'.format(
    np.mean(scores), np.std(scores)))

# Cross-validate using svm with rbf kernel
clf = SVC(kernel='rbf', C=1)
scores = cross_val_score(clf, X, y, cv=5)
print('Accuracy: {:.2f} +/- {:.2f}'.format(
    np.mean(scores), np.std(scores)))

# Create umap plot
reducer = UMAP()
embedding = reducer.fit_transform(X)
plt.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap='rainbow')
plt.show()

# Create NCA plot
nca = NeighborhoodComponentsAnalysis(n_components=3, random_state=42)
X_nca = nca.fit_transform(X, y)

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_nca[:, 0], X_nca[:, 1], X_nca[:, 2], c=y, cmap='rainbow')
plt.show()

#plt.scatter(X_nca[:, 0], X_nca[:, 1], c=y,
#            cmap='rainbow', alpha = 0.5)
#plt.show()

############################################################
## Cluster X and check if gapes are clustered together
############################################################
# Uses GMM to cluster X
gmm = GaussianMixture(n_components=8, random_state=42)
gmm.fit(X)
y_pred = gmm.predict(X)
y_pred_prob = gmm.predict_proba(X)

# Plot mean gape probability for each cluster
cluster_ind = np.argmax(y_pred_prob, axis=1)
cluster_pred = [np.mean(y_pred_prob[cluster_ind == i, 0]) for i in range(8)]
cluster_counts = [np.sum(cluster_ind == i) for i in range(8)]

# Sort X by cluster
cluster_ind_sorted = np.argsort(cluster_ind)
X_sorted = X[cluster_ind_sorted, :]
y_sorted = y[cluster_ind_sorted]
clust_num_sorted = cluster_ind[cluster_ind_sorted]

# Plot X  and y
fig, ax = plt.subplots(1,3, figsize=(5, 10), sharey=True)
ax[0].imshow(X_sorted, aspect='auto', interpolation='none')
ax[1].plot(y_sorted, np.arange(len(X)), '-x')
ax[2].plot(clust_num_sorted, np.arange(len(X)), '-x')
plt.show()

############################################################
## Use supervised clustering to cluster waveforms
## more predictive of quinine during the palatability epoch
############################################################
