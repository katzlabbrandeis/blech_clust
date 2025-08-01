"""
This module processes electrophysiological data files, organizes them into an HDF5 format, and reads data from various file types. It handles both single-file and multi-file data formats, creating necessary groups in the HDF5 file for raw data, EMG data, and digital inputs/outputs.

- Imports necessary Python and custom modules for file handling, data processing, and path management.
- Determines the type of data files present (either one file per signal type or one file per channel).
- Creates or opens an HDF5 file, setting up groups for raw data, EMG data, and digital inputs/outputs.
- Identifies and sorts lists of amplifier and digital input files based on the file type.
- Reads metadata from an info file to determine sampling rates and other parameters.
- Reads and processes digital input data and electrode channels, handling both single-file and multi-file formats.
- Utilizes utility functions from imported modules to read and append data to the HDF5 file.
"""
# Necessary python modules
import os
import tables
import sys
import numpy as np
import multiprocessing
import json
import glob
import pandas as pd
import shutil

# Get blech_clust path
script_path = os.path.realpath(__file__)
blech_clust_dir = os.path.dirname(os.path.basename(script_path))
sys.path.append(blech_clust_dir)
# Necessary blech_clust modules
from utils.blech_process_utils import path_handler  # noqa: E402
from utils.blech_utils import entry_checker, imp_metadata  # noqa: E402
from utils import qa_utils as qa  # noqa: E402
from utils import read_file  # noqa: E402

############################################################

metadata_handler = imp_metadata(sys.argv)
dir_name = metadata_handler.dir_name
print(f'Processing : {dir_name}')
os.chdir(dir_name)

info_dict = metadata_handler.info_dict
file_list = metadata_handler.file_list


# Get the type of data files (.rhd or .dat)
if 'auxiliary.dat' in file_list:
    file_type = ['one file per signal type']
else:
    file_type = ['one file per channel']

# Create hdf5 file, and make groups for raw data, raw emgs,
# digital outputs and digital inputs, and close

# Grab directory name to create the name of the hdf5 file
# If HDF5 present, use that, otherwise, create new one
h5_search = glob.glob('*.h5')
if len(h5_search):
    hdf5_name = h5_search[0]
    print(f'HDF5 file found...Using file {hdf5_name}')
    hf5 = tables.open_file(hdf5_name, 'r+')
else:
    hdf5_name = str(os.path.dirname(dir_name)).split('/')[-1]+'.h5'
    print(f'No HDF5 found...Creating file {hdf5_name}')
    hf5 = tables.open_file(hdf5_name, 'w', title=hdf5_name[-1])

group_list = ['raw', 'raw_emg', 'digital_in', 'digital_out']
for this_group in group_list:
    if '/'+this_group in hf5:
        hf5.remove_node('/', this_group, recursive=True)
    hf5.create_group('/', this_group)
hf5.close()
print(f'Created nodes in HF5 : {group_list}')

# Get lists of amplifier and digital input files
if file_type == ['one file per signal type']:
    electrodes_list = ['amplifier.dat']
    dig_in_list = ['digitalin.dat']
elif file_type == ['one file per channel']:
    electrodes_list = [name for name in file_list if name.startswith('amp-')]
    dig_in_list = [name for name in file_list if name.startswith('board-DI')]

electrodes_list = sorted(electrodes_list)
dig_in_list = sorted(dig_in_list)

# Use info file for port list calculation
info_file_path = os.path.join(dir_name, 'info.rhd')
if os.path.exists(info_file_path):
    info_file = np.fromfile(info_file_path, dtype=np.dtype('float32'))
    sampling_rate = int(info_file[2])
else:
    print("info.rhd file not found. Please enter the sampling rate manually:")
    sampling_rate = int(input("Sampling rate (Hz): "))

# Read the time.dat file for use in separating out
# the one file per signal type data
num_recorded_samples = len(np.fromfile(
    dir_name + '/' + 'time.dat', dtype=np.dtype('float32')))
total_recording_time = num_recorded_samples/sampling_rate  # In seconds

check_str = f'Amplifier files: {electrodes_list} \nSampling rate: {sampling_rate} Hz'\
    f'\nDigital input files: {dig_in_list} \n ---------- \n \n'
print(check_str)

ports = info_dict['ports']

if file_type == ['one file per channel']:
    print("\tOne file per CHANNEL Detected")

    # Read dig-in data
    # Pull out the digital input channels used,
    # and convert them to integers
    dig_in = [x.split('-')[-1].split('.')[0] for x in dig_in_list]
    dig_in = sorted([int(x) for x in dig_in])

elif file_type == ['one file per signal type']:

    print("\tOne file per SIGNAL Detected")
    dig_in = np.arange(info_dict['dig_ins']['count'])

check_str = f'ports used: {ports} \n sampling rate: {sampling_rate} Hz'\
    f'\n digital inputs on intan board: {dig_in}'

print(check_str)

all_car_group_vals = []
for region_name, region_elecs in info_dict['electrode_layout'].items():
    if not region_name == 'emg':
        for group in region_elecs:
            if len(group) > 0:
                all_car_group_vals.append(group)
all_electrodes = [electrode for region in all_car_group_vals
                  for electrode in region]

emg_info = info_dict['emg']
emg_port = emg_info['port']
emg_channels = sorted(emg_info['electrodes'])


layout_path = glob.glob(os.path.join(dir_name, "*layout.csv"))[0]
electrode_layout_frame = pd.read_csv(layout_path)


# Read data files, and append to electrode arrays
if file_type == ['one file per channel']:
    read_file.read_digins(hdf5_name, dig_in, dig_in_list)
    read_file.read_electrode_channels(hdf5_name, electrode_layout_frame)
    if len(emg_channels) > 0:
        read_file.read_emg_channels(hdf5_name, electrode_layout_frame)
elif file_type == ['one file per signal type']:
    read_file.read_digins_single_file(hdf5_name, dig_in, dig_in_list)
    # This next line takes care of both electrodes and emgs
    read_file.read_electrode_emg_channels_single_file(
        hdf5_name, electrode_layout_frame, electrodes_list, num_recorded_samples, emg_channels)
