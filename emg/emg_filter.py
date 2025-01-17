"""
This module processes EMG (electromyography) data by subtracting signals, filtering them, and saving the results. It handles data loading, filtering, differencing, and identifying significant trials based on activity changes.

- Imports necessary libraries and modules, including numpy, scipy, pandas, and custom utilities.
- Loads EMG data from an HDF5 file and retrieves metadata and parameters.
- Configures Butterworth highpass and lowpass filters for signal processing.
- Reads electrode layout and groups EMG channels by CAR (Common Average Reference) groups.
- Differentiates EMG signals within CAR groups, handling cases with one or two channels.
- Applies bandpass and lowpass filters to the differentiated signals.
- Identifies significant trials by comparing pre- and post-stimulus activity levels.
- Saves filtered signals, envelopes, and significant trial indicators back to the HDF5 file.
- Logs the execution status of the processing pipeline.
"""
# Subtracts the two emg signals and filters and saves the results.

# Import stuff
import numpy as np
from scipy.signal import butter, filtfilt
import os
import sys
import shutil
import glob
import pandas as pd
import tables
import ast

sys.path.append('..')
from utils.blech_utils import imp_metadata, pipeline_graph_check

# Get name of directory with the data files
metadata_handler = imp_metadata(sys.argv)
dir_name = metadata_handler.dir_name

# Perform pipeline graph check
script_path = os.path.realpath(__file__)
this_pipeline_check = pipeline_graph_check(dir_name)
this_pipeline_check.check_previous(script_path)
this_pipeline_check.write_to_log(script_path, 'attempted')

os.chdir(dir_name)
print(f'Processing : {dir_name}')

############################################################
# Load the data
# shape : channels x tastes x trials x time
# emg_data = np.load('emg_output/emg_data.npy')
with tables.open_file(metadata_handler.hdf5_name, 'r') as hf5:
    emg_digins = hf5.list_nodes('/emg_data')
    emg_digins = [x for x in emg_digins if 'dig_in' in x._v_name]
    emg_digin_names = [x._v_name for x in emg_digins]
    emg_data = [x.emg_array[:] for x in emg_digins]
    map_array = hf5.get_node('/emg_data/ind_electrode_map').read()
ind_electrode_map = ast.literal_eval(str(map_array)[2:-1])
# key = electrode_ind, value = index in array
inverse_map = {int(v.split('emg')[1]): k for k, v in ind_electrode_map.items()}

info_dict = metadata_handler.info_dict
params_dict = metadata_handler.params_dict

# Pull pre_stim duration from params file
durations = params_dict['spike_array_durations']
pre_stim = int(durations[0])
print(f'Using pre-stim duration : {pre_stim}' + '\n')

# Get coefficients for Butterworth filters
m, n = butter(2, 2.0*300.0/1000.0, 'highpass')
c, d = butter(2, 2.0*15.0/1000.0, 'lowpass')

## todo: This can be pulled from info file
## check how many EMG channels used in this experiment
layout_path = glob.glob(os.path.join(dir_name,"*layout.csv"))[0]
electrode_layout_frame = pd.read_csv(layout_path) 

# Change CAR_group col to lower
electrode_layout_frame['CAR_group'] = electrode_layout_frame['CAR_group'].str.lower()

# Allow for multiple emg CAR groups
wanted_rows = pd.DataFrame(
        [x for num,x in electrode_layout_frame.iterrows() \
                if 'emg' in x.CAR_group])
wanted_rows = wanted_rows.sort_values('electrode_ind')
wanted_rows.reset_index(inplace=True, drop=True)

# TODO: Ask about differencing pairs
# Difference by CAR emg labels
# If only 1 channel per emg CAR, do not difference if asked
emg_car_groups = [x[1] for x in wanted_rows.groupby('CAR_group')]
emg_car_names = [x.CAR_group.unique()[0] for x in emg_car_groups]
# emg_car_inds = [x.index.values for x in emg_car_groups]
emg_car_inds = [[inverse_map[x] for x in y.electrode_ind.values] for y in emg_car_groups]

print('EMG CAR Groups with more than 1 channel will be differenced')
print('EMG CAR groups as follows:')
for x in emg_car_groups:
    print(x)
    print()

# TODO: This question can go into an EMG params file
# Bandpass filter the emg signals, and store them in a numpy array. 
# Low pass filter the bandpassed signals, and store them in another array
# Take difference between pairs of channels
# emg_data = List of arrays (per dig-in) of shape : channels x trials x time
# emg_data_grouped = list of lists
#       outer list : emg CAR groups 
#       inner list : dig-ins 
#       element_array : channels x trials x time
emg_data_grouped = [[dat[x] for dat in emg_data] for x in emg_car_inds]
# Make sure all element arrays are 3D with shape: channels x trials x time
for x in emg_data_grouped:
    for y in x:
        if len(y.shape) < 3:
            y = np.expand_dims(y, axis=0)

emg_diff_data = []
for this_car in emg_data_grouped:
    this_car_diff = []
    for this_dig in this_car:
        if len(this_dig) > 1:
            this_car_diff.append(np.squeeze(np.diff(this_dig,axis=0)))
        elif len(this_dig) > 2:
            raise Exception("More than 2 per EMG CAR currently not supported")
        else:
            this_car_diff.append(np.squeeze(this_dig))
    emg_diff_data.append(this_car_diff)

# Iterate over trials and apply frequency filter
emg_filt_list = []
emg_env_list = []
for car_group in emg_diff_data:
    this_car_filt = []
    this_car_env = []
    for dig_in in car_group:
        emg_filt = np.zeros(dig_in.shape)
        emg_env = np.zeros(dig_in.shape)
        temp_filt = filtfilt(m, n, dig_in)
        temp_env = filtfilt(c, d, np.abs(temp_filt))
        this_car_filt.append(temp_filt)
        this_car_env.append(temp_env)
    emg_filt_list.append(this_car_filt)
    emg_env_list.append(this_car_env)

# Iterate and check for signficant changes in activity
n_cars = len(emg_diff_data)
n_dig = len(emg_diff_data[0])
trial_lens = [[len(x) for x in y] for y in emg_diff_data]

ind_frame = pd.DataFrame(
        dict(
            car_group = [x for x in range(n_cars) for y in range(n_dig)],
            dig_in = [y for x in range(n_cars) for y in range(n_dig)],
            trial_len = [np.arange(trial_lens[x][y]) for x in range(n_cars) for y in range(n_dig)]
            )
        )
# Explode the trial_len column
ind_frame = ind_frame.explode('trial_len')

sig_trials_list = []
for i, this_row in ind_frame.iterrows(): 
    this_ind = this_row.values
    this_dat = emg_filt_list[this_ind[0]][this_ind[1]][this_ind[2]]
    ## Get mean and std of baseline emg activity, 
    ## and use it to select trials that have significant post stimulus activity
    # sig_trials (assumed shape) : tastes x trials
    pre_m = np.mean(np.abs(this_dat[:pre_stim]))
    pre_s = np.std(np.abs(this_dat[:pre_stim]))

    post_m = np.mean(np.abs(this_dat[pre_stim:]))
    post_max = np.max(np.abs(this_dat[pre_stim:]))

    # If any of the channels passes the criteria, select that trial as significant
    # 1) mean post-stim activity > mean pre-stim activity
    # 2) max post-stim activity > mean pre-stim activity + 4*pre-stim STD
    mean_bool = post_m > pre_m
    std_bool = post_max > (pre_m + 4.0*pre_s)

    # Logical AND
    sig_trials = mean_bool * std_bool
    sig_trials_list.append(sig_trials)

ind_frame['sig_trials'] = sig_trials_list

# Save the highpass filtered signal, 
# the envelope and the indicator of significant trials as a np array
# Iterate over channels and save them in different directories 
ind_frame.to_hdf(metadata_handler.hdf5_name, '/emg_data/emg_sig_trials')

with tables.open_file(metadata_handler.hdf5_name, 'r+') as hf5:
    for digin_ind, digin_name in enumerate(emg_digin_names):
        digin_path = f'/emg_data/{digin_name}'
        if f'{digin_path}/processed_emg' in hf5:
            hf5.remove_node(f'{digin_path}/processed_emg', recursive = True)
        hf5.create_group(f'{digin_path}', 'processed_emg')
        for car_ind , this_car_name in enumerate(emg_car_names): 
            hf5.create_array(
                    f'{digin_path}/processed_emg', 
                    f'{this_car_name}_emg_filt', 
                    emg_filt_list[car_ind][digin_ind]
                    )
            hf5.create_array(
                    f'{digin_path}/processed_emg',
                    f'{this_car_name}_emg_env',
                    emg_env_list[car_ind][digin_ind]
                    )

# Write successful execution to log
this_pipeline_check.write_to_log(script_path, 'completed')
