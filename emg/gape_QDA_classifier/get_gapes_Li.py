# Use the results in Li et al. 2016 to get gapes on taste trials

from utils.blech_utils import imp_metadata, pipeline_graph_check
import os
import sys
from glob import glob

import numpy as np
import tables
import pylab as plt
import pandas as pd
import seaborn as sns

from detect_peaks import detect_peaks
from QDA_classifier import QDA
sys.path.append('../..')


def QDA_process_single_trial(this_trial, pre_stim, pre_stim_mean):

    peak_ind = detect_peaks(
        this_trial,
        mpd=85,
        mph=np.mean(
            this_trial[:pre_stim]) +
        np.std(this_trial[:pre_stim]
               )
    )

    # Get the indices, in the smoothed signal,
    # that are below the mean of the smoothed signal
    below_mean_ind = np.where(this_trial <= pre_stim_mean)[0]

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
        sig_trial = 0
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
        gapes_Li = np.zeros_like(this_trial)
        for peak in range(len(durations) - 1):
            gape = QDA(intervals[peak+1], durations[peak+1])
            if gape and peak_ind[peak+1] - pre_stim <= post_stim:
                gapes_Li[peak_ind[peak+1]] = 1.0

        # If there are no gapes on a trial, mark these as 0
        # on sig_trials_final and 0 on first_gape.
        # Else put the time of the first gape in first_gape
        if np.sum(gapes_Li) == 0.0:
            sig_trial = 0
        else:
            sig_trial = 1

    return gapes_Li, sig_trial


############################################################
# Load Data
############################################################

# Ask for the directory where the hdf5 file sits, and change to that directory
# Get name of directory with the data files
metadata_handler = imp_metadata(sys.argv)
data_dir = metadata_handler.dir_name

# Perform pipeline graph check
script_path = os.path.realpath(__file__)
this_pipeline_check = pipeline_graph_check(data_dir)
this_pipeline_check.check_previous(script_path)
this_pipeline_check.write_to_log(script_path, 'attempted')

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
psth_ind_vec = np.arange(psth_inds[0], psth_inds[1])
psth_x = np.arange(psth_durs[0], psth_durs[1])

############################################################
# Load and Process Data
############################################################
emg_output_dir = os.path.join(data_dir, 'emg_output')

if 'flat_emg_env_data.npy' not in os.listdir(emg_output_dir):
    raise Exception(f'flat_emg_env_data.py not found in {emg_output_dir}')
    exit()

os.chdir(data_dir)

# Paths for plotting
plot_dir = f'emg_output/gape_classifier_plots/'
fin_plot_dir = os.path.join(data_dir, plot_dir)
if not os.path.exists(fin_plot_dir):
    os.makedirs(fin_plot_dir)

# Load the required emg data (the envelope and sig_trials)
# env = np.load('emg_env.npy')
env = np.load('emg_output/flat_emg_env_data.npy')

merge_frame = pd.read_csv('emg_output/emg_env_merge_df.csv')
merge_frame['laser'] = merge_frame['laser_duration_ms'].astype(str) + \
    '_' + merge_frame['laser_lag_ms'].astype(str)

merge_frame.drop(columns=['laser_duration_ms', 'laser_lag_ms'], inplace=True)
merge_frame.drop(columns=['dig_in_num_laser',
                 'dig_in_name_laser'], inplace=True)

env_pre_stim_mean = np.mean(env[:, :pre_stim], axis=None)
outs = [QDA_process_single_trial(x, pre_stim, env_pre_stim_mean) for x in env]
gapes_Li, sig_trials_final = zip(*outs)
gapes_Li = np.array(gapes_Li)
sig_trials_final = np.array(sig_trials_final)

first_gapes = [np.where(x)[0][0] if sum(x) > 0 else np.nan for x in gapes_Li]

############################################################
# Plots
############################################################
# 1) Raw traces with gapes
# 2) Scatters of gape times per condition
# 3) Gape "Rate" timeseries per condition

##############################
# 1) Raw traces with gapes
##############################

car_group_frames_list = [x[1] for x in list(merge_frame.groupby('car'))]

for this_car_frame in car_group_frames_list:
    car_name = this_car_frame.car.unique()[0]
    this_env_dat = env[this_car_frame.index.values]

    n_dig_ins = len(this_car_frame['dig_in_name_taste'].unique())
    max_trials = this_car_frame['taste_rel_trial_num'].max() + 1
    max_value = this_env_dat.max(axis=None)

    fig, ax = plt.subplots(max_trials, n_dig_ins,
                           figsize=(max_trials, n_dig_ins*4),
                           sharex=True, sharey=True)
    for i, (this_meta, this_dat, this_ax) in \
            enumerate(zip(this_car_frame.iterrows(), this_env_dat, ax.flatten(order='F'))):
        this_ax.plot(psth_x, this_dat[psth_ind_vec])
        gape_inds = np.where(gapes_Li[i][psth_ind_vec])[
            0] - pre_stim + psth_ind_vec[0]
        this_ax.scatter(gape_inds, max_value * 1.1 * np.ones(gape_inds.shape),
                        c='r', s=10)
        this_ax.set_ylabel(
            str(this_meta[1].taste_rel_trial_num) + ':' + this_meta[1].laser)
        # this_ax.set_title(this_meta[1]['dig_in_name_taste'])
    for i, this_name in enumerate(merge_frame['dig_in_name_taste'].unique()):
        ax[0, i].set_title(this_name)
        ax[-1, i].set_xlabel('Time (ms)')
    fig.suptitle(f'Trace with gapes for {car_name}')
    plt.tight_layout()
    fig.savefig(os.path.join(fin_plot_dir, f'{car_name}_traces_with_gapes.png'),
                bbox_inches='tight')
    plt.close(fig)

##############################
# 2) Scatters of gape times per condition
##############################

# Add gapes_Li to the merge_frame
gapes_frame_list = []
for i, this_row in merge_frame.iterrows():
    this_gapes = gapes_Li[i]
    this_gapes_frame = pd.DataFrame({'gapes': this_gapes[psth_ind_vec]})
    this_gapes_frame['time'] = psth_x
    this_gapes_frame['taste_rel_trial_num'] = this_row['taste_rel_trial_num']
    this_gapes_frame['laser'] = this_row['laser']
    this_gapes_frame['dig_in_name_taste'] = this_row['dig_in_name_taste']
    this_gapes_frame['car'] = this_row['car']
    gapes_frame_list.append(this_gapes_frame)
gapes_frame = pd.concat(gapes_frame_list)
gapes_frame = gapes_frame.loc[gapes_frame['gapes'] > 0]

car_group_frames_list = [x[1] for x in list(gapes_frame.groupby('car'))]
for this_car_frame in car_group_frames_list:
    g = sns.FacetGrid(gapes_frame, col='dig_in_name_taste', row='laser',
                      hue='gapes', size=4, aspect=1)
    g.map(plt.scatter, 'time', 'taste_rel_trial_num')
    for this_ax in g.axes.flatten():
        this_ax.set_xlabel('Time post-stim (ms)')
        this_ax.set_ylabel('Trial number')
        old_title = this_ax.get_title()
        new_title = '\n'.join(old_title.split('|'))
        this_ax.set_title(new_title)
        this_ax.set_xlim(*psth_durs)
    g.fig.suptitle(f'Gape times for {this_car_frame.car.unique()[0]}')
    plt.tight_layout()
    g.fig.savefig(os.path.join(fin_plot_dir, f'{this_car_frame.car.unique()[0]}_gape_times.png'),
                  bbox_inches='tight')
    plt.close(g.fig)

##############################
# 3) Gape "Rate" timeseries per condition
##############################
kern_len = 300
kern = np.ones(kern_len)/kern_len
gape_rate_list = []
for this_gapes in gapes_Li:
    conv_rate = np.convolve(
        this_gapes, kern, mode='same')
    gape_rate_list.append(conv_rate)
gape_rate = np.array(gape_rate_list)*1000

gapes_frame_list = []
for i, this_row in merge_frame.iterrows():
    this_gapes = gape_rate[i]
    this_gapes_frame = pd.DataFrame({'gape_rate': this_gapes[psth_ind_vec]})
    this_gapes_frame['time'] = psth_x
    this_gapes_frame['taste_rel_trial_num'] = this_row['taste_rel_trial_num']
    this_gapes_frame['laser'] = this_row['laser']
    this_gapes_frame['dig_in_name_taste'] = this_row['dig_in_name_taste']
    this_gapes_frame['car'] = this_row['car']
    gapes_frame_list.append(this_gapes_frame)
gapes_frame = pd.concat(gapes_frame_list)

mean_gapes_frame = gapes_frame.groupby(
    ['car', 'laser', 'dig_in_name_taste', 'time']).mean().reset_index()

g = sns.relplot(
    x='time', y='gape_rate',
    hue='dig_in_name_taste', col='laser',
    row='car', data=mean_gapes_frame, kind='line',
    aspect=1, height=4)
g.fig.suptitle('Gape rate timeseries')
# Move the legend to the right
sns.move_legend(g, "upper right", bbox_to_anchor=(1.2, 1))
g.fig.suptitle('Gape rate timeseries' + '\n' + f'Kernel length: {kern_len}')
plt.tight_layout()
g.fig.savefig(os.path.join(fin_plot_dir, 'gape_rate_timeseries.png'),
              bbox_inches='tight')
plt.close(g.fig)

############################################################

# Save these results to the hdf5 file
hf5_save_path = '/emg_gape_classifier'

if hf5_save_path not in hf5:
    hf5.create_group(
        os.path.dirname(hf5_save_path),
        os.path.basename(hf5_save_path),
        createparents=True)
try:
    hf5.remove_node(f'{hf5_save_path}/gapes_Li')
    # hf5.remove_node(f'{hf5_save_path}/gape_trials_Li')
    hf5.remove_node(f'{hf5_save_path}/first_gape_Li')
except:
    pass
hf5.create_array(hf5_save_path, 'gapes_Li', gapes_Li)
# hf5.create_array(hf5_save_path, 'gape_trials_Li', sig_trials_final)
hf5.create_array(hf5_save_path, 'first_gape_Li', first_gapes)
hf5.flush()

hf5.close()

# Write successful execution to log
this_pipeline_check.write_to_log(script_path, 'completed')
