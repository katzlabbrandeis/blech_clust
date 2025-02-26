"""
blech_common_avg_reference.py - Common Average Reference (CAR) processing for neural recordings

This script performs common average referencing on raw electrode data to reduce noise and artifacts.
It processes electrode recordings by:

1. Data Organization:
   - Groups electrodes based on anatomical regions and recording ports
   - Excludes EMG channels and electrodes marked as 'none' from CAR processing
   - Handles multiple CAR groups independently

2. Reference Calculation:
   - Calculates common average reference for each electrode group
   - Averages voltage values across all electrodes in a group
   - Processes groups sequentially to optimize memory usage

3. Signal Processing:
   - Subtracts appropriate common average reference from each electrode
   - Updates electrode data in-place in the HDF5 file
   - Maintains data integrity through careful memory management

Usage:
    python blech_common_avg_reference.py <dir_name>

Arguments:
    dir_name : Directory containing the HDF5 file with raw electrode data

Dependencies:
    - numpy, tables, tqdm
    - Custom utility modules from blech_clust package

Notes:
    - Requires properly formatted electrode layout information in the experiment info file
    - CAR groups are defined by the 'CAR_group' column in the electrode layout
    - EMG channels and electrodes marked as 'none' are automatically excluded

Author: Abuzar Mahmood
"""

# Import stuff!
import os
from utils.blech_utils import imp_metadata, pipeline_graph_check
import json
import glob
from tqdm import tqdm
import sys
import easygui
import tables
import numpy as np


def identify_dead_channels(raw_electrodes, abs_threshold=0.01, rel_threshold=0.05):
    """
    Identify dead channels based on absolute and relative variance thresholds.

    Channels are considered dead if either:
    1. Their variance is below the absolute threshold
    2. Their variance is below rel_threshold * median variance of all channels

    Args:
        raw_electrodes (list): List of electrode data arrays.
        abs_threshold (float): Absolute variance threshold to identify dead channels.
        rel_threshold (float): Relative variance threshold as a fraction of median variance.

    Returns:
        list: Indices of dead channels.
    """
    # Calculate variance for all electrodes
    variances = np.array([np.var(electrode[:])
                         for electrode in raw_electrodes])

    # Calculate median variance across all channels
    median_variance = np.median(variances)

    # Identify dead channels using both absolute and relative thresholds
    dead_channels = []
    for i, variance in enumerate(variances):
        if variance < abs_threshold or variance < (rel_threshold * median_variance):
            dead_channels.append(i)

    print(
        f"Identified {len(dead_channels)} dead channels out of {len(raw_electrodes)}")
    if len(dead_channels) > 0:
        print(f"Median variance: {median_variance:.6f}")
        print(f"Relative threshold: {rel_threshold * median_variance:.6f}")

    return dead_channels


def get_electrode_by_name(raw_electrodes, name):
    """
    Get the electrode data from the list of raw electrodes
    by the name of the electrode
    """
    str_name = f"electrode{name:02}"
    wanted_electrode_ind = [
        x for x in raw_electrodes if str_name in x._v_pathname][0]
    return wanted_electrode_ind

############################################################


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

# Read CAR groups from info file
# Every region is a separate group, multiple ports under single region is a separate group,
# emg is a separate group
info_dict = metadata_handler.info_dict
electrode_layout_frame = metadata_handler.layout
# Pull out the raw electrode nodes of the HDF5 file
raw_electrodes = hf5.list_nodes('/raw')

# Identify dead channels
dead_channels = identify_dead_channels(raw_electrodes)

# Remove emg and none channels from the electrode layout frame
emg_bool = ~electrode_layout_frame.CAR_group.str.contains('emg')
none_bool = ~electrode_layout_frame.CAR_group.str.contains('none')
fin_bool = np.logical_and(emg_bool, none_bool)
electrode_layout_frame = electrode_layout_frame[fin_bool]

# Since electrodes are already in monotonic numbers (A : 0-31, B: 32-63)
# we can directly pull them
grouped_layout = list(electrode_layout_frame.groupby('CAR_group'))
all_car_group_names = [x[0] for x in grouped_layout]
# Note: electrodes in HDF5 are also saved according to inds
# specified in the layout file
all_car_group_vals = [x[1].electrode_ind.values for x in grouped_layout]

# Exclude dead channels from CAR groups
CAR_electrodes = [np.setdiff1d(group, dead_channels)
                  for group in all_car_group_vals]
num_groups = len(CAR_electrodes)
print(f" Number of groups : {num_groups}")
for region, vals in zip(all_car_group_names, all_car_group_vals):
    print(f" {region} :: {vals}")

# Pull out the raw electrode nodes of the HDF5 file
raw_electrodes = hf5.list_nodes('/raw')

# First get the common average references by averaging across
# the electrodes picked for each group
print(
    "Calculating common average reference for {:d} groups".format(num_groups))
common_average_reference = np.zeros(
    (num_groups, raw_electrodes[0][:].shape[0]))
print('Calculating mean values')
for group_num, group_name in tqdm(enumerate(all_car_group_names)):
    print(f"Processing group {group_name}")
    # First add up the voltage values from each electrode to the same array
    # then divide by number of electrodes to get the average
    # This is more memory efficient than loading all the electrode data into
    # a single array and then averaging
    for electrode_name in tqdm(CAR_electrodes[group_num]):
        common_average_reference[group_num, :] += \
            get_electrode_by_name(raw_electrodes, electrode_name)[:]
    common_average_reference[group_num,
                             :] /= float(len(CAR_electrodes[group_num]))

print("Common average reference for {:d} groups calculated".format(num_groups))

# Now run through the raw electrode data and
# subtract the common average reference from each of them
print('Performing background subtraction')
for group_num, group_name in tqdm(enumerate(all_car_group_names)):
    print(f"Processing group {group_name}")
    for electrode_num in tqdm(all_car_group_vals[group_num]):
        # Subtract the common average reference for that group from the
        # voltage data of the electrode
        wanted_electrode = get_electrode_by_name(raw_electrodes, electrode_num)
        referenced_data = wanted_electrode[:] - \
            common_average_reference[group_num]
        # Overwrite the electrode data with the referenced data
        wanted_electrode[:] = referenced_data
        hf5.flush()
        del referenced_data

hf5.close()
print("Modified electrode arrays written to HDF5 file after "
      "subtracting the common average reference")

# Write successful execution to log
this_pipeline_check.write_to_log(script_path, 'completed')
