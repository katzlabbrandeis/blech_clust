"""
This module performs post-processing cleanup of files created by `emg_local_BSA_execute.py`, saving output files to an HDF5 file under the node `emg_BSA_results`. It processes EMG data, removes unnecessary nodes, and calculates specific frequency arrays.

- Imports necessary libraries and utility functions.
- Retrieves the directory name and metadata using `imp_metadata`.
- Performs a pipeline graph check using `pipeline_graph_check`.
- Opens the HDF5 file and removes the `/raw_emg` node if it exists to reduce file size.
- Extracts experimental information and taste names from metadata.
- Loads trial data from a CSV file and frequency analysis results from NPY files.
- Processes and saves the first non-NaN omega data to disk.
- Calculates `gape_array` and `ltp_array` based on specific frequency ranges and saves them to disk.
- Logs the successful execution of the script.
"""

# Import stuff
import numpy as np
import easygui
import os
import tables
import glob
import json
import sys
import pandas as pd

test_bool = False

if test_bool:
    data_dir = '/home/abuzarmahmood/projects/blech_clust/pipeline_testing/test_data_handling/test_data/KM45_5tastes_210620_113227_new/'
    script_path = '/home/abuzarmahmood/projects/blech_clust/emg/emg_freq_setup.py'
    blech_clust_dir = os.path.dirname(os.path.dirname(script_path))
    sys.path.append(blech_clust_dir)
    print(f'blech_clust_dir: {blech_clust_dir}')
    # from utils.blech_process_utils import path_handler  # noqa: E402
    from utils.blech_utils import imp_metadata, pipeline_graph_check  # noqa: E402

    # Get name of directory with the data files
    metadata_handler = imp_metadata([[], data_dir])
    dir_name = metadata_handler.dir_name

else:
    script_path = os.path.realpath(__file__)

    blech_clust_dir = os.path.dirname(os.path.dirname(script_path))
    sys.path.append(blech_clust_dir)
    print(f'blech_clust_dir: {blech_clust_dir}')
    # from utils.blech_process_utils import path_handler  # noqa: E402
    from utils.blech_utils import imp_metadata, pipeline_graph_check  # noqa: E402

    # Get name of directory with the data files
    metadata_handler = imp_metadata(sys.argv)
    dir_name = metadata_handler.dir_name

    # Perform pipeline graph check
    this_pipeline_check = pipeline_graph_check(dir_name)
    this_pipeline_check.check_previous(script_path)
    this_pipeline_check.write_to_log(script_path, 'attempted')

os.chdir(dir_name)
print(f'Processing : {dir_name}')

# Open the hdf5 file
hf5 = tables.open_file(metadata_handler.hdf5_name, 'r+')

# Delete the raw_emg node, if it exists in the hdf5 file,
# to cut down on file size
try:
    hf5.remove_node('/raw_emg', recursive=1)
except:
    print("Raw EMG recordings have already been removed, so moving on ..")


# Extract info experimental info file
info_dict = metadata_handler.info_dict
taste_names = info_dict['taste_params']['tastes']

# Use trial count from emg_data to account for chopping down of trials
emg_trials_frame = pd.read_csv('emg_output/emg_env_df.csv', index_col=0)

# Load frequency analysis output
results_path = os.path.join(dir_name, 'emg_output', 'emg_BSA_results')
# p_files = sorted(glob.glob(os.path.join(results_path, '*_p.npy')))
# omega_files = sorted(glob.glob(os.path.join(results_path, '*_omega.npy')))
trial_inds = emg_trials_frame.index.values
p_files = [os.path.join(
    results_path, f'trial{x:03}_p.npy') for x in trial_inds]
omega_files = [os.path.join(
    results_path, f'trial{x:03}_omega.npy') for x in trial_inds]

p_data = [np.load(x) for x in p_files]
p_data = np.stack(p_data, axis=0)
omega_data = [np.load(x) for x in omega_files]
# Get first non nan omega
omega = [x for x in omega_data if not any(np.isnan(x))][0]

# Write out p_flat and omega to disk
# np.save(os.path.join('emg_output', 'p_flat.npy'), p_flat)
np.save(os.path.join('emg_output', 'omega.npy'), omega)

# Write out concatenated p_data to disk
np.save(os.path.join('emg_output', 'p_data.npy'), p_data)

############################################################
# ## Create gape and ltp arrays
############################################################
# Segment by frequencies

# gape_array = np.logical_and(
#         p_flat >= 3,
#         p_flat <= 5
#         )
# ltp_array = p_flat >= 5.5

# Find the frequency with the maximum EMG power at each time point on each trial
# Gapes are anything upto 4.6 Hz
# LTPs are from 5.95 Hz to 8.65 Hz
# Alternatively, gapes from 3.65-5.95 Hz (6-11). LTPs from 5.95 to 8.65 Hz (11-17)
gape_array = np.sum(p_data[:, :, 6:11], axis=2) /\
    np.sum(p_data, axis=2)
ltp_array = np.sum(p_data[:, :, 11:], axis=2) /\
    np.sum(p_data, axis=2)

# Write out
np.save('emg_output/gape_array.npy', gape_array)
np.save('emg_output/ltp_array.npy', ltp_array)

############################################################
# Also export to HDF5 as arrays 
# These will NOT be used for downstream processing
# but are exported for backwards compatibility

# Matching commit: 431ceb
# ==============================
# Add group to hdf5 file for emg BSA results
if '/emg_BSA_results' in hf5:
    hf5.remove_node('/','emg_BSA_results', recursive = True)
hf5.create_group('/', 'emg_BSA_results')

# Add omega to the hdf5 file
if '/emg_BSA_results/omega' not in hf5:
    atom = tables.Atom.from_dtype(omega.dtype)
    om = hf5.create_carray('/emg_BSA_results', 'omega', atom, omega.shape)
    om[:] = omega 
    hf5.flush()

# for num, this_dir in enumerate(channel_dirs):
#     os.chdir(this_dir)
#     this_basename = channels_discovered[num]
#     print(f'Processing data for : {this_basename}')
#
#     # Load sig_trials.npy to get number of tastes
#     sig_trials = np.load('sig_trials.npy')
#     tastes = sig_trials.shape[0]
#
#     print(f'Trials taken from emg_data.npy ::: {dict(zip(taste_names, trials))}')
#
#     # Change to emg_BSA_results
#     os.chdir('emg_BSA_results')
#
#     # Omega doesn't vary by trial, 
#     # so just pick it up from the 1st taste and trial, 
#     first_omega = 'taste00_trial00_omega.npy'
#     if os.path.exists(first_omega):
#         omega = np.load(first_omega)
#
#         # Load one of the p arrays to find out the time length of the emg data
#         p = np.load('taste00_trial00_p.npy')
#         time_length = p.shape[0]
#
#         # Go through the tastes and trials
#         # todo: Output to HDF5 needs to be named by channel
#         for i in range(tastes):
#             # Make an array for posterior probabilities for each taste
#             #p = np.zeros((trials[i], time_length, 20))
#             # Make array with highest numbers of trials, so uneven trial numbers
#             # can be accomadated
#             p = np.zeros((np.max(trials), time_length, 20))
#             for j in range(trials[i]):
#                 p[j, :, :] = np.load(f'taste{i:02}_trial{j:02}_p.npy')
#             # Save p to hdf5 file
#             atom = tables.Atom.from_dtype(p.dtype)
#             prob = hf5.create_carray(
#                     os.path.join(base_dir, this_basename), 
#                     'taste%i_p' % i, 
#                     atom, 
#                     p.shape)
#             prob[:, :, :] = p
#         hf5.flush()
############################################################

max_n_trials = np.max(emg_trials_frame['trial_inds']) + 1

emg_trials_frame['taste_num'] = emg_trials_frame['dig_in'].rank(method='dense') - 1
emg_trials_frame['taste_num'] = emg_trials_frame['taste_num'].astype(int)

car_grouped_df = emg_trials_frame.groupby('car')

base_dir = '/emg_BSA_results'
for _, this_df in car_grouped_df:
    this_basename = this_df['car'].unique()[0]

    if os.path.join(base_dir, this_basename) in hf5:
        hf5.remove_node(base_dir, this_basename, recursive = True)
    hf5.create_group(base_dir, this_basename)

    dig_in_grouped_df = this_df.groupby('dig_in')
    for dig_in_ind, dig_df in dig_in_grouped_df:

        this_p_array = np.zeros((max_n_trials, *p_data.shape[1:]))
        # fill with NaNs
        this_p_array.fill(np.nan)

        for row_ind, this_row in dig_df.iterrows(): 
            this_trial_ind = this_row['trial_inds']
            this_p_array[this_trial_ind] = p_data[row_ind] 

        # Save p to hdf5 file
        atom = tables.Atom.from_dtype(this_p_array.dtype)
        taste_num = this_row['taste_num']
        hf5.create_array(
            os.path.join(base_dir, this_basename), 
            'taste%i_p' % taste_num,
            this_p_array,
            atom=atom)
        hf5.flush()

hf5.close()
############################################################
# Write successful execution to log
this_pipeline_check.write_to_log(script_path, 'completed')
