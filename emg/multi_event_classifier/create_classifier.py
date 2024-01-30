"""
Create classifier from annotated data
"""
import os
import sys
from glob import glob

import numpy as np
import tables
import pylab as plt
import pandas as pd
import glob
from tqdm import tqdm
from matplotlib.patches import Patch
import seaborn as sns
from xgboost import XGBClassifier
import pickle
import json

# Have to be in blech_clust/emg/gape_QDA_classifier dir
code_dir = os.path.expanduser('~/Desktop/blech_clust/emg/multi_event_classifier')
sys.path.append(code_dir)
from utils.extract_scored_data import return_taste_orders, process_scored_data
from utils.gape_clust_funcs import (extract_movements,
                                            normalize_segments,
                                            extract_features,
                                            JL_process,
                                            gen_gape_frame,
                                            threshold_movement_lengths,
                                            )

import itertools
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA

artifacts_dir = os.path.join(code_dir, 'data','artifacts')
if not os.path.exists(artifacts_dir):
    os.makedirs(artifacts_dir)


scored_gape_frame_path = os.path.join(artifacts_dir, 'scored_gape_frame.pkl')
if os.path.exists(scored_gape_frame_path):
    scored_gape_frame = pd.read_pickle(scored_gape_frame_path)

else:
    ############################################################
    # Run pipeline
    data_dir = '/media/fastdata/NB27'

    # For each day of experiment, load env and table files
    data_subdirs = sorted(glob(os.path.join(data_dir,'*')))
    # Make sure that path is a directory
    data_subdirs = [subdir for subdir in data_subdirs if os.path.isdir(subdir)]
    # Make sure that subdirs are in order
    subdir_basenames = [os.path.basename(subdir).lower() for subdir in data_subdirs]

    env_files = [glob(os.path.join(subdir,'**','*env.npy'), recursive=True) \
            for subdir in data_subdirs]

    # Selet only specific channel
    channel_pattern = 'emgAD'
    env_files = [[x for x in y if channel_pattern in x] for y in env_files]
    env_files = [x[0] for x in env_files]
    env_files = sorted(env_files)

    # Load env and table files
    # days x tastes x trials x time
    envs = np.stack([np.load(env_file) for env_file in env_files])

    ############################################################
    # Get dig-in info
    ############################################################
    # Find HDF5 files
    h5_files = glob(os.path.join(data_dir,'**','*.h5'), recursive=True)
    h5_files = sorted(h5_files)
    h5_basenames = [os.path.basename(x) for x in h5_files]
    # Make sure order of h5 files is same as order of envs
    order_bool = [x.lower() in y.lower() for x,y in zip(subdir_basenames, h5_basenames)]

    if not all(order_bool):
        raise Exception('Bubble bubble, toil and trouble')

    all_taste_orders = return_taste_orders(h5_files)
    fin_scored_table = process_scored_data(data_subdirs, all_taste_orders)

    ############################################################
    # Extract mouth movements 
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

    # Create segment bounds for fin_scored_table
    fin_scored_table['segment_bounds'] = \
            list(zip(fin_scored_table['rel_time_start'], fin_scored_table['rel_time_stop']))

    # Make sure that each segment in gape_frame is in fin_score_table
    score_match_cols = ['day_ind','taste','taste_trial']
    gape_match_cols = ['day_ind','taste,','trial']

    score_bounds_list = []
    for ind, row in tqdm(gape_frame.iterrows()):
        day_ind = row.day_ind
        taste = row.taste
        taste_trial = row.trial
        segment_center = row.segment_center
        wanted_score_table = fin_scored_table.loc[
            (fin_scored_table.day_ind == day_ind) &
            (fin_scored_table.taste == taste) &
            (fin_scored_table.taste_trial == taste_trial)]
        if len(wanted_score_table):
            # Check if segment center is in any of the scored segments
            for _, score_row in wanted_score_table.iterrows():
                if (score_row.segment_bounds[0] <= segment_center) & (segment_center <= score_row.segment_bounds[1]):
                    gape_frame.loc[ind, 'scored'] = True
                    gape_frame.loc[ind, 'event_type'] = score_row.event  
                    score_bounds_list.append(score_row.segment_bounds)
                    break
                else:
                    gape_frame.loc[ind, 'scored'] = False

    scored_gape_frame = gape_frame.loc[gape_frame.scored == True]
    scored_gape_frame['score_bounds'] = score_bounds_list

    scored_gape_frame.to_pickle(scored_gape_frame_path)

############################################################
# Classifier comparison on gapes 
############################################################
############################################################
# Multiclass 

wanted_event_types = ['gape','tongue protrusion',]


# Train new classifier on data
X = np.stack(scored_gape_frame['features'].values)
y = scored_gape_frame['event_type']
y_bool = [x in wanted_event_types for x in y] 
X = X[y_bool]
y = y[y_bool]

# Convert y to categorical
y_map = dict(zip(wanted_event_types, range(len(wanted_event_types))))
y = y.map(y_map) 

# Write out y_map
y_map_path = os.path.join(artifacts_dir, 'y_map.json')
with open(y_map_path, 'w') as f:
    json.dump(y_map, f)

clf = XGBClassifier()
clf.fit(X, y)

# Save classifier in artifacts
clf_path = os.path.join(artifacts_dir, 'gape_classifier.pkl')
# clf.save_model(clf_path)
pickle.dump(clf, open(clf_path, 'wb'))
