import os
import sys
from glob import glob

import numpy as np
import tables
import pylab as plt
import pandas as pd

# Have to be in blech_clust/emg/gape_QDA_classifier dir
# os.chdir(os.path.expanduser('~/Desktop/blech_clust/emg/gape_QDA_classifier/_experimental/mouth_movement_clustering'))
sys.path.append(os.path.expanduser('~/Desktop/'))
# sys.path.append(os.path.expanduser('~/Desktop/blech_clust/emg/gape_QDA_classifier'))
from blech_clust.utils.blech_utils import imp_metadata
from utils.gape_clust_funcs import (extract_movements,
                                            normalize_segments,
                                            extract_features,
                                            find_segment,
                                            calc_peak_interval,
                                            JL_process,
                                            gen_gape_frame,
                                            )

import itertools
from sklearn.cluster import KMeans, AgglomerativeClustering

############################################################
############################################################
# Extract dig-ins

def return_taste_orders(h5_files):
    """
    Returns the order of tastes for each session

    Inputs:
        h5_files: list of paths to hdf5 files
    
    Outputs:
        all_taste_orders: array of shape (days, tastes)
    """

    dig_in_list = []
    for i, h5_file in enumerate(h5_files):
        session_dig_in_list = []
        h5 = tables.open_file(h5_file, 'r')
        for this_dig in h5.root.digital_in._f_iter_nodes():
            session_dig_in_list.append(this_dig[:])
        h5.close()
        dig_in_list.append(session_dig_in_list)

    all_starts = []
    for this_session in dig_in_list:
        session_starts = []
        for this_dig in this_session:
            starts = np.where(np.diff(this_dig) == 1)[0]
            session_starts.append(starts)
        all_starts.append(session_starts)

    all_starts = np.stack(all_starts)
    all_starts = all_starts / 30000
    all_starts = np.round(all_starts).astype(int)

    # Find order of deliveries for each session
    all_taste_orders = []
    for this_session in all_starts:
        bin_array = np.zeros((this_session.max(), len(this_session)))
        for i, this_dig in enumerate(this_session):
            bin_array[this_dig-1,i] = 1
        taste_order = np.where(bin_array)[1]
        all_taste_orders.append(taste_order)
    all_taste_orders = np.array(all_taste_orders)

    return all_taste_orders


############################################################
# Process scoring 
############################################################

def process_scored_data(data_subdirs, all_taste_orders):
    """
    Processes scored data from each day of experiment

    Inputs:
        data_subdirs: list of paths to scored data
        all_taste_orders: array of shape (days, tastes)

    Outputs:
        score_tables: list of pandas dataframes of scored data
    """

    table_files = []
    for subdir in data_subdirs:
        path_list = glob(os.path.join(subdir,'*scoring_table.npy'))
        if len(path_list) == 0:
            raise ValueError('No scoring table found in {}'.format(subdir))
        else:
            table_files.append(path_list[0])

    # Process score tables
    score_tables = [np.load(table_file, allow_pickle=True) for table_file in table_files]
    # Only first and last columns are relevant
    score_tables = [x[:,[0,-2,-1]] for x in score_tables]

    # Convert to pandas dataframes
    score_tables = [pd.DataFrame(x, columns=['event','mark_type','time']) for x in score_tables]
    updated_tables = []
    for i, table in enumerate(score_tables):
        table['day'] = subdir_basenames[i] 
        table.sort_values(by=['day','time'], inplace=True)
        table.reset_index(inplace=True, drop=True)

        # Mark trials in fin_table
        table['trial'] = None
        trial_count = -1
        for j in range(len(table)):
            if table.loc[j]['event'] == 'trial start':
                trial_count += 1
            table.loc[j]['trial'] = trial_count
        updated_tables.append(table)

    # Note, day 3 only has 117 trials
    fin_table = pd.concat(updated_tables)

    # Keep only trials with more than one event (that is more than trial start)
    #fin_table = fin_table.groupby('trial').filter(lambda x: len(x) > 1)
    fin_table['day_ind'] = [int(str(x)[-1])-1 for x in fin_table.day]
    fin_table.reset_index(inplace=True, drop=True)

    # Mark taste for each trial
    taste_list = []
    for i in range(len(fin_table)):
        day_ind = fin_table.loc[i]['day_ind']
        trial_ind = fin_table.loc[i]['trial']
        this_taste = all_taste_orders[day_ind, trial_ind]
        taste_list.append(this_taste)

    fin_table['taste'] = taste_list

    # Also find what delivery of the given taste this is
    group_list=  [x[1] for x in list(fin_table.groupby(['day','taste']))]
    updated_tables = []
    for this_table in group_list:
        unique_trials = this_table.trial.unique()
        trial_map = {x:i for i,x in enumerate(unique_trials)}
        this_table['taste_trial'] = [trial_map[x] for x in this_table.trial]
        updated_tables.append(this_table)

    fin_table = pd.concat(updated_tables)

    # Calculate relative time from start of each trial
    group_list = list(fin_table.groupby(['day','trial']))
    group_list = [x[1] for x in group_list]
    updated_tables = []
    for i, table in enumerate(group_list):
        table['rel_time'] = table['time'] - table['time'].iloc[0]
        updated_tables.append(table)
    fin_table = pd.concat(updated_tables)
    fin_table['rel_time'] *= 1000
    # Convert rel_time to int
    fin_table['rel_time'] = fin_table['rel_time'].astype(int)

    # Only point type mark is trial start
    # Now that we have rel_time, drop all trial starts and make starts and end columns
    # That is, convert long to wide
    fin_table = fin_table.loc[fin_table.event != 'trial start']
    fin_table.reset_index(inplace=True, drop=True)

    index_cols = ['event','trial','taste','day']
    # Group by day and sort by time
    group_list = list(fin_table.groupby(['day']))
    group_list = [x[1] for x in group_list]
    updated_tables = []
    for i, table in enumerate(group_list):
        table.sort_values(by=['time'], inplace=True)
        table.reset_index(inplace=True, drop=True)
        updated_tables.append(table)
    fin_table = pd.concat(updated_tables)

    start_table = fin_table.loc[fin_table.mark_type == 'START']
    stop_table = fin_table.loc[fin_table.mark_type == 'STOP']
    start_table['event_num'] = np.arange(len(start_table))
    stop_table['event_num'] = np.arange(len(stop_table))

    fin_table = start_table.merge(stop_table, how = 'outer', 
                      on = ['event','event_num','trial','taste','day','taste_trial', 'day_ind'], 
                      suffixes = ('_start','_stop'))
    fin_table.drop(['mark_type_start','mark_type_stop'], axis=1, inplace=True)

    return fin_table


############################################################
# Test plots
############################################################
if __name__ == '__main__':
    data_dir = '/home/abuzarmahmood/Desktop/blech_clust/emg/multi_event_classifier/data/scored_data/NB27'

    # For each day of experiment, load env and table files
    data_subdirs = sorted(glob(os.path.join(data_dir,'*')))
    # Make sure that path is a directory
    data_subdirs = [subdir for subdir in data_subdirs if os.path.isdir(subdir)]
    # Make sure that subdirs are in order
    subdir_basenames = [os.path.basename(subdir).lower() for subdir in data_subdirs]

    env_files = [glob(os.path.join(subdir,'*env.npy'))[0] for subdir in data_subdirs]
    # Load env and table files
    # days x tastes x trials x time
    envs = np.stack([np.load(env_file) for env_file in env_files])

    ############################################################
    # Get dig-in info
    ############################################################
    # Extract dig-in from datasets
    raw_data_dir = '/media/fastdata/NB_data/NB27'
    # Find HDF5 files
    h5_files = glob(os.path.join(raw_data_dir,'**','*','*.h5'))
    h5_files = sorted(h5_files)
    h5_basenames = [os.path.basename(x) for x in h5_files]
    # Make sure order of h5 files is same as order of envs
    order_bool = [x in y for x,y in zip(subdir_basenames, h5_basenames)]

    if not all(order_bool):
        raise Exception('Bubble bubble, toil and trouble')

    all_taste_orders = return_taste_orders(h5_files)
    fin_table = process_scored_data(data_subdirs, all_taste_orders)

    plot_group = list(fin_table.groupby(['day_ind','taste','taste_trial']))
    plot_inds = [x[0] for x in plot_group]
    plot_dat = [x[1] for x in plot_group]

    t = np.arange(-2000, 5000)

    event_types = fin_table.event.unique()
    cmap = plt.get_cmap('tab10')
    event_colors = {event_types[i]:cmap(i) for i in range(len(event_types))}

    # Generate custom legend
    from matplotlib.patches import Patch

    legend_elements = [Patch(facecolor=event_colors[event], edgecolor='k',
                             label=event) for event in event_types]

    plot_n = 15
    fig,ax = plt.subplots(plot_n, 1, sharex=True,
                          figsize = (10, plot_n*2))
    for i in range(plot_n):
        this_scores = plot_dat[i]
        this_inds = plot_inds[i]
        this_env = envs[this_inds]
        ax[i].plot(t, this_env)
        for _, this_event in this_scores.iterrows():
            event_type = this_event.event
            start_time = this_event.rel_time_start
            stop_time = this_event.rel_time_stop
            this_event_c = event_colors[event_type]
            ax[i].axvspan(start_time, stop_time, 
                          color=this_event_c, alpha=0.5, label=event_type)
    ax[0].legend(handles=legend_elements, loc='upper right',
                 bbox_to_anchor=(1.5, 1.1))
    ax[0].set_xlim([0, 5000])
    fig.subplots_adjust(right=0.75)
    plt.show()
