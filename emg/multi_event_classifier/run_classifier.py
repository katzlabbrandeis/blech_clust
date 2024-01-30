import os
import sys
from glob import glob

import numpy as np
import tables
import pylab as plt
import pandas as pd
from tqdm import tqdm
from xgboost import XGBClassifier
import pickle

code_dir = os.path.expanduser('~/Desktop/blech_clust/emg/multi_event_classifier')
sys.path.append(code_dir)
from utils.gape_clust_funcs import (extract_movements,
                                            normalize_segments,
                                            extract_features,
                                            threshold_movement_lengths,
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

############################################################
# Load Classifier
############################################################
artifacts_dir = os.path.join(code_dir, 'data','artifacts')
clf_path = os.path.join(artifacts_dir, 'gape_classifier.pkl')
clf = pickle.load(open(clf_path, 'rb')) 

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

############################################################
# Exctract features 
############################################################
pre_stim = 2000
post_stim = 5000
gapes_Li = np.zeros(envs.shape)

segment_dat_list = []
inds = list(np.ndindex(envs.shape[:3]))
for this_ind in inds:
    this_trial_dat = envs[this_ind]

    ### Jenn Li Process ###
    # Get peak indices
    this_day_prestim_dat = envs[this_ind[0], :, :, :pre_stim]
    gape_peak_inds = JL_process(
                        this_trial_dat, 
                        this_day_prestim_dat,
                        pre_stim,
                        post_stim,
                        this_ind,)
    if gape_peak_inds is not None:
        gapes_Li[this_ind][gape_peak_inds] = 1

    ### AM Process ###
    segment_starts, segment_ends, segment_dat = extract_movements(
        this_trial_dat, size=200)

    # Threshold movement lengths
    segment_starts, segment_ends, segment_dat = threshold_movement_lengths(
        segment_starts, segment_ends, segment_dat, 
        min_len = 50, max_len= 500)

    (feature_array,
     feature_names,
     segment_dat,
     segment_starts,
     segment_ends) = extract_features(
        segment_dat, segment_starts, segment_ends)

    segment_bounds = list(zip(segment_starts, segment_ends))
    merged_dat = [feature_array, segment_dat, segment_bounds] 
    segment_dat_list.append(merged_dat)

gape_frame, scaled_features = gen_gape_frame(segment_dat_list, gapes_Li, inds)
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
X = np.stack(scored_gape_frame['features'].values)
pred_y = clf.predict(X)

inverse_y_map = {v: k for k, v in y_map.items()}
pred_y_label = [inverse_y_map[x] for x in pred_y] 
