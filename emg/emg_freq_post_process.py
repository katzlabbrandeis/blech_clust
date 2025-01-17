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

# Open the hdf5 file
hf5 = tables.open_file(metadata_handler.hdf5_name, 'r+')

# Delete the raw_emg node, if it exists in the hdf5 file, 
# to cut down on file size
try:
    hf5.remove_node('/raw_emg', recursive = 1)
except:
    print("Raw EMG recordings have already been removed, so moving on ..")

hf5.close()

# Extract info experimental info file
info_dict = metadata_handler.info_dict
taste_names = info_dict['taste_params']['tastes']

# Use trial count from emg_data to account for chopping down of trials
emg_trials_frame = pd.read_csv('emg_output/emg_env_df.csv', index_col = 0)

# Load frequency analysis output
results_path = os.path.join(dir_name, 'emg_output', 'emg_BSA_results')
#p_files = sorted(glob.glob(os.path.join(results_path, '*_p.npy')))
#omega_files = sorted(glob.glob(os.path.join(results_path, '*_omega.npy')))
trial_inds = emg_trials_frame.index.values
p_files = [os.path.join(results_path, f'trial{x:03}_p.npy') for x in trial_inds]
omega_files = [os.path.join(results_path, f'trial{x:03}_omega.npy') for x in trial_inds]

p_data = [np.load(x) for x in p_files]
p_data = np.stack(p_data, axis = 0) 
omega_data = [np.load(x) for x in omega_files]
# Get first non nan omega
omega = [x for x in omega_data if not any(np.isnan(x))][0]

# Write out p_flat and omega to disk
# np.save(os.path.join('emg_output', 'p_flat.npy'), p_flat)
np.save(os.path.join('emg_output', 'omega.npy'), omega)

############################################################
# ## Create gape and ltp arrays
############################################################
# Segment by frequencies

# gape_array = np.logical_and(
#         p_flat >= 3,
#         p_flat <= 5
#         )
# ltp_array = p_flat >= 5.5

## Find the frequency with the maximum EMG power at each time point on each trial
## Gapes are anything upto 4.6 Hz
## LTPs are from 5.95 Hz to 8.65 Hz
#Alternatively, gapes from 3.65-5.95 Hz (6-11). LTPs from 5.95 to 8.65 Hz (11-17) 
gape_array = np.sum(p_data[:, :, 6:11], axis = 2)/\
		np.sum(p_data, axis = 2)
ltp_array = np.sum(p_data[:, :, 11:], axis = 2)/\
		np.sum(p_data, axis = 2)

# Write out
np.save('emg_output/gape_array.npy', gape_array)
np.save('emg_output/ltp_array.npy', ltp_array)

# Write successful execution to log
this_pipeline_check.write_to_log(script_path, 'completed')
