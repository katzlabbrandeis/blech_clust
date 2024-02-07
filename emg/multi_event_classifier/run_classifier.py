import os
import sys
from glob import glob

import numpy as np
import tables
import pylab as plt
import pandas as pd
from tqdm import tqdm
from xgboost import XGBClassifier
import joblib
import json
from matplotlib import patches 
from scipy.ndimage import white_tophat

code_dir = os.path.expanduser('~/Desktop/blech_clust/emg/multi_event_classifier')
sys.path.append(code_dir)
from utils.gape_clust_funcs import (extract_movements,
                                            normalize_segments,
                                            extract_features,
                                            threshold_movement_lengths,
                                            JL_process,
                                            gen_gape_frame,
                                            )

def get_emg_envs(data_dir, channel_pattern='emgAD'):
    """
    Get emg envs from data_dir

    Inputs:
        data_dir: str
        channel_pattern: str, exact string to match in data_dir

    Outputs:
        env: np.array, tastes x trials x time
    """
    env_files = glob(os.path.join(data_dir,'**','*env.npy'), recursive=True)
    dir_basenames = [os.path.basename(os.path.dirname(x)) for x in env_files]

    # Selet only specific channel
    env_files = [this_env for this_env, this_dir in zip(env_files, dir_basenames) \
            if channel_pattern == this_dir] 
    if len(env_files) == 0:
        raise ValueError('No files found in {}'.format(data_dir))
    elif len(env_files) > 1:
        raise ValueError(f'Multiple files found in {data_dir} ::: {env_files}')
    else:
        env_file = env_files[0]

    # Load env and table files
    env = np.load(env_file)
    return env

##############################
plot_dir = os.path.join(code_dir, 'plots')
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

############################################################
# Load Classifier
############################################################
artifacts_dir = os.path.join(code_dir, 'data','artifacts')
scaler_path = os.path.join(artifacts_dir, 'feature_scaler.pkl')
clf_path = os.path.join(artifacts_dir, 'gape_classifier.pkl')
scaler_obj = joblib.load(open(scaler_path, 'rb'))
clf = joblib.load(open(clf_path, 'rb')) 

# Load y_map
y_map_path = os.path.join(artifacts_dir, 'y_map.json')
with open(y_map_path, 'r') as f:
    y_map = json.load(f)

############################################################
# Load data 
############################################################
data_base_dir = '/media/fastdata/NB27'
data_dir_list = os.listdir(data_base_dir)
data_dir_list = [os.path.join(data_base_dir, x) for x in data_dir_list]
data_dir_list = [x for x in data_dir_list if os.path.isdir(x)]

# Extract Envelopes
envs = np.stack([get_emg_envs(data_dir) for data_dir in tqdm(data_dir_list)])

filtered_envs = np.zeros(envs.shape)
inds = np.array(list(np.ndindex(envs.shape[:3])))
for this_ind in tqdm(inds):
    filtered_envs[tuple(this_ind)] = white_tophat(envs[tuple(this_ind)], size = 250)

############################################################
# Exctract features 
############################################################
pre_stim = 2000
post_stim = 5000
gapes_Li = np.zeros(envs.shape)

############################## 
# Preprocessing
############################## 
filtered_envs = np.zeros(envs.shape)
inds = np.array(list(np.ndindex(envs.shape[:3])))
for this_ind in tqdm(inds):
    filtered_envs[tuple(this_ind)] = white_tophat(envs[tuple(this_ind)], size = 250)

##############################

gapes_Li = np.zeros(envs.shape)

segment_dat_list = []
inds = list(np.ndindex(envs.shape[:3]))
for this_ind in inds:
    # this_trial_dat = envs[this_ind]
    this_trial_dat = filtered_envs[this_ind]

    ### Jenn Li Process ###
    # Get peak indices
    # this_day_prestim_dat = envs[this_ind[0], :, :, :pre_stim]
    this_day_prestim_dat = filtered_envs[this_ind[0], :, :, :pre_stim]
    gape_peak_inds = JL_process(
                        this_trial_dat, 
                        this_day_prestim_dat,
                        pre_stim,
                        post_stim,
                        this_ind,)
    if gape_peak_inds is not None:
        gapes_Li[this_ind][gape_peak_inds] = 1

    ### AM Process ###
    segment_starts, segment_ends, segment_dat = \
            extract_movements(this_trial_dat) 

    # Threshold movement lengths
    segment_starts, segment_ends, segment_dat = threshold_movement_lengths(
        segment_starts, segment_ends, segment_dat, 
        min_len = 50, max_len= 500)

    # plt.plot(this_trial_dat)
    # for i in range(len(segment_starts)):
    #     plt.plot(np.arange(segment_starts[i], segment_ends[i]),
    #              segment_dat[i], linewidth = 5, alpha = 0.5)
    # plt.show()

    (feature_array,
     feature_names,
     segment_dat,
     segment_starts,
     segment_ends) = extract_features(
        segment_dat, segment_starts, segment_ends)

    segment_bounds = list(zip(segment_starts, segment_ends))
    merged_dat = [feature_array, segment_dat, segment_bounds] 
    segment_dat_list.append(merged_dat)

gape_frame, scaled_features, scaler_obj = \
        gen_gape_frame(segment_dat_list, gapes_Li, inds)

# Bounds for gape_frame are in 0-7000 time
# Adjust to make -2000 -> 5000
# Adjust segment_bounds by removing pre_stim
all_segment_bounds = gape_frame.segment_bounds.values
adjusted_segment_bounds = [np.array(x)-pre_stim for x in all_segment_bounds]
gape_frame['segment_bounds'] = adjusted_segment_bounds
gape_frame.rename(columns={'channel': 'day_ind'}, inplace=True)

# Calculate segment centers
gape_frame['segment_center'] = [np.mean(x) for x in gape_frame.segment_bounds]

############################################################
# Perform prediction 
############################################################
X = np.stack(gape_frame['features'].values)
pred_y = clf.predict(X)
proba_y = clf.predict_proba(X)

inverse_y_map = {v: k for k, v in y_map.items()}
pred_y_label = [inverse_y_map[x] for x in pred_y] 

# Add to gape_frame
gape_frame['pred_y'] = pred_y
gape_frame['pred_y_label'] = pred_y_label
gape_frame['proba_y'] = proba_y[np.arange(len(pred_y)), pred_y] 

############################################################
# Generate plots 
############################################################

##############################
# Measure MAD amplitude at baseline
# So we can use it as threshold

# Get baseline data
baseline_dat = filtered_envs[:, :, :, :pre_stim]
baseline_med = np.median(baseline_dat, axis=None)
baseline_mad = np.median(np.abs(baseline_dat - baseline_med), axis=None)
filtered_MAD_threshold = baseline_med + 3*baseline_mad

baseline_dat = envs[:, :, :, :pre_stim]
baseline_med = np.median(baseline_dat, axis=None)
baseline_mad = np.median(np.abs(baseline_dat - baseline_med), axis=None)
MAD_threshold = baseline_med + 3*baseline_mad

# Plot envs above and below threshold
for i, this_taste in tqdm(enumerate(taste_inds)):
    fig, ax = plt.subplots(max_trials, 2, figsize=(7, 10),
                           sharey = 'row', sharex = True)
    for j, this_trial in enumerate(trial_inds[i]):
        filtered_env_dat = filtered_envs[wanted_day_ind, this_taste, this_trial]
        env_dat = envs[wanted_day_ind, this_taste, this_trial]
        greater_than = env_dat > MAD_threshold
        less_than = env_dat < MAD_threshold
        filtered_greater_than = filtered_env_dat > filtered_MAD_threshold
        filtered_less_than = filtered_env_dat < filtered_MAD_threshold
        ax[j,0].plot(envs_x[greater_than], env_dat[greater_than], 'r.',
                     linewidth = 1, markersize = 2)
        ax[j,0].plot(envs_x[less_than], env_dat[less_than], 'b.',
                     linewidth = 1, markersize = 2)
        ax[j,0].axvline(0, c = 'r', linestyle = '--', linewidth = 2)
        ax[j,1].plot(envs_x[filtered_greater_than], 
                     filtered_env_dat[filtered_greater_than], 'r.',
                     linewidth = 1, markersize = 2)
        ax[j,1].plot(envs_x[filtered_less_than], 
                     filtered_env_dat[filtered_less_than], 'b.',
                     linewidth = 1, markersize = 2)
        ax[j,1].axvline(0, c = 'r', linestyle = '--', linewidth = 2)
        ax[0,0].set_title('Raw')
        ax[0,1].set_title('Filtered')
        ax[j,0].set_ylabel(this_trial)
plt.show()

##############################

amp_list = [x[2] for x in gape_frame.features.values]
plt.hist(amp_list, bins=100, log=True)
plt.show()

wanted_day_ind = 0
wanted_frame = gape_frame[gape_frame.day_ind == wanted_day_ind]

cmap = plt.get_cmap('tab10')
# Plot all tastes on same plot
taste_inds = gape_frame.taste.unique()
trial_inds = gape_frame.groupby('taste').trial.unique()
max_trials = max([len(x) for x in trial_inds])
max_time = envs.shape[-1]

envs_x = np.arange(-pre_stim, post_stim)

for i, this_taste in tqdm(enumerate(taste_inds)):
    fig, ax = plt.subplots(max_trials, 1, figsize=(7, 10),
                           sharex=True, sharey=True)
    min_y = np.min([np.min(x) for x in envs[wanted_day_ind, this_taste]])
    max_y = np.max([np.max(x) for x in envs[wanted_day_ind, this_taste]])
    for j, this_trial in enumerate(trial_inds[i]):
        this_trial_dat = wanted_frame[(wanted_frame.taste == this_taste) \
                & (wanted_frame.trial == this_trial)]
        for k, this_row in this_trial_dat.iterrows():
            segment_bounds = this_row.segment_bounds
            x = np.arange(segment_bounds[0], segment_bounds[1])
            y = this_row.segment_raw
            ax[j].plot(envs_x, envs[wanted_day_ind, this_taste, this_trial], c = 'k')
            # ax[j, i].plot(x, y, c = 'k')
            # Also plot rectangle in background to indicate inferred event
            event_type = this_row.pred_y
            alpha = this_row.proba_y
            event_color = cmap(event_type)
            # Create rectangle
            rect = patches.Rectangle(
                (segment_bounds[0], min_y),
                segment_bounds[1] - segment_bounds[0],
                max_y - min_y,
                linewidth=0.5,
                edgecolor= 'k',
                facecolor=event_color,
                alpha=0.5,
                zorder = -1)
            ax[j].add_patch(rect)
            ax[j].set_ylabel(this_trial)
        ax[j].axvline(0, c = 'r', linestyle = '--', linewidth = 2)
    plt.suptitle(f'Taste {this_taste}')
    ax[0].set_title('Predicted Events')
    ax[-1].set_xlabel('Time (ms)')
    # Set legend at bottom
    handles = [patches.Patch(color=cmap(key), label=val) for key, val in inverse_y_map.items()]
    fig.legend(handles=handles, loc='lower center', ncol=len(inverse_y_map))
    fig.savefig(os.path.join(plot_dir, f'taste_{this_taste}_events.png'))
    plt.close(fig)
# plt.show()

fig, ax = plt.subplots(1, len(taste_inds), figsize=(20, 5),
                       sharex=True, sharey=True)
for i, this_taste in tqdm(enumerate(taste_inds)):
    for j, this_trial in enumerate(trial_inds[i]):
        this_trial_dat = wanted_frame[(wanted_frame.taste == this_taste) \
                & (wanted_frame.trial == this_trial)]
        for k, this_row in this_trial_dat.iterrows():
            segment_bounds = this_row.segment_bounds
            event_type = this_row.pred_y
            alpha = this_row.proba_y
            event_color = cmap(event_type)
            # Create rectangle
            rect = patches.Rectangle(
                (segment_bounds[0], this_trial - 0.5),
                segment_bounds[1] - segment_bounds[0],
                1,
                linewidth=1,
                edgecolor=event_color,
                facecolor=event_color,
                alpha=alpha)
            ax[i].add_patch(rect)
    ax[i].set_xlim(-pre_stim, post_stim)
    ax[i].axvline(0, color='k', linestyle='--')
    ax[i].set_ylim(-0.5, max_trials-0.5)
    ax[i].set_title(f'Taste: {this_taste}')
    ax[i].set_xlabel('Time (ms)')
    # Invert y axis so that trial 0 is at the top
    ax[i].invert_yaxis()
ax[0].set_ylabel('Trial #')
# Create horizontal legend below all plots
handles = [patches.Patch(color=cmap(key), label=val) for key, val in inverse_y_map.items()]
fig.legend(handles=handles, loc='lower center', ncol=len(inverse_y_map))
fig.suptitle('Inferrred Mouth Events')
plt.tight_layout()
plt.subplots_adjust(top=0.9, bottom=0.2)
# plt.show()
fig.savefig(os.path.join(plot_dir, 'inferred_mouth_events.png'),
            bbox_inches='tight')
plt.close(fig)
