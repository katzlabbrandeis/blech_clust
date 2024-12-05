"""
blech_clust.py - Main script for processing and clustering neural recording data

This script handles the initial processing of neural recording data from Intan files,
creating an HDF5 file structure to store the processed data. It performs several key functions:

1. Data Import and Organization:
   - Reads raw data files (.rhd or .dat format)
   - Creates HDF5 file structure for organized data storage
   - Handles different file formats (one file per channel, one file per signal type, traditional)

2. Directory Setup:
   - Creates necessary directories for storing:
     * Spike waveforms
     * Spike times
     * Clustering results
     * Analysis plots
     * Memory monitoring data

3. Quality Assurance:
   - Performs channel correlation analysis
   - Generates QA plots and reports
   - Validates digital input signals

4. Processing Pipeline:
   - Sets up parallel processing scripts for spike sorting
   - Handles parameter file creation and management
   - Integrates with the broader blech_clust pipeline

Usage:
    python blech_clust.py <dir_name> [--force_run]

Arguments:
    dir_name    : Directory containing the raw data files
    --force_run : Optional flag to bypass user confirmations

Dependencies:
    - numpy, tables, pandas, matplotlib
    - Custom utility modules from blech_clust package
"""

# Necessary python modules
import argparse

parser = argparse.ArgumentParser(description='Load data and create hdf5 file')
parser.add_argument('dir_name', type=str, help='Directory name with data files')
parser.add_argument('--force_run' , action='store_true', help='Force run the script without asking user')
args = parser.parse_args()
force_run = args.force_run

import os
import tables
import sys
import numpy as np
import multiprocessing
import json
import glob
import pandas as pd
import shutil
import pylab as plt

# Necessary blech_clust modules
from utils import read_file
from utils.qa_utils import channel_corr
from utils.blech_utils import entry_checker, imp_metadata, pipeline_graph_check
from utils.blech_process_utils import path_handler
from utils.importrhdutilities import read_header


class HDF5Handler:
    """Handles HDF5 file operations for blech_clust"""
    
    def __init__(self, dir_name, force_run=False):
        """Initialize HDF5 handler
        
        Args:
            dir_name: Directory containing the data
            force_run: Whether to force operations without asking user
        """
        self.dir_name = dir_name
        self.force_run = force_run
        self.group_list = ['raw', 'raw_emg', 'digital_in', 'digital_out']
        self.setup_hdf5()
        
    def setup_hdf5(self):
        """Setup or load HDF5 file"""
        h5_search = glob.glob('*.h5')
        if len(h5_search):
            self.hdf5_name = h5_search[0]
            print(f'HDF5 file found...Using file {self.hdf5_name}')
            self.hf5 = tables.open_file(self.hdf5_name, 'r+')
        else:
            self.hdf5_name = str(os.path.dirname(self.dir_name)).split('/')[-1]+'.h5'
            print(f'No HDF5 found...Creating file {self.hdf5_name}')
            self.hf5 = tables.open_file(self.hdf5_name, 'w', title=self.hdf5_name[-1])
            
    def initialize_groups(self):
        """Initialize HDF5 groups"""
        found_list = []
        for this_group in self.group_list:
            if '/'+this_group in self.hf5:
                found_list.append(this_group)

        if len(found_list) > 0 and not self.force_run:
            reload_msg = f'Data already present: {found_list}' + '\n' +\
                        'Reload data? (yes/y/n/no) ::: '
            reload_data_str, continue_bool = entry_checker(
                    msg= reload_msg,
                    check_func=lambda x: x in ['y', 'yes', 'n', 'no'],
                    fail_response='Please enter (yes/y/n/no)')
        else:
            continue_bool = True
            reload_data_str = 'y'

        if continue_bool:
            if reload_data_str in ['y', 'yes']:
                for this_group in self.group_list:
                    if '/'+this_group in self.hf5:
                        self.hf5.remove_node('/', this_group, recursive=True)
                    self.hf5.create_group('/', this_group)
                print('Created nodes in HF5')
        
        self.hf5.close()
        return continue_bool, reload_data_str

    def get_digital_inputs(self, sampling_rate):
        """Get digital input data from HDF5 file
        
        Args:
            sampling_rate: Sampling rate of the data
            
        Returns:
            numpy array of digital input data
        """
        with tables.open_file(self.hdf5_name, 'r') as hf5:
            dig_in_list = [self._process_digital_input(x[:], sampling_rate) 
                          for x in hf5.root.digital_in]
        return np.stack(dig_in_list)
    
    @staticmethod
    def _process_digital_input(data, sampling_rate):
        """Process a single digital input channel
        
        Args:
            data: Raw digital input data
            sampling_rate: Sampling rate
            
        Returns:
            Processed digital input data
        """
        len_dig_in = len(data)
        truncated = data[:(len_dig_in//sampling_rate)*sampling_rate]
        return np.reshape(truncated, (-1, sampling_rate)).sum(axis=-1)


def generate_processing_scripts(dir_name, blech_clust_dir, electrode_layout_frame, 
                              all_electrodes, all_params_dict):
    """Generate bash scripts for running single and parallel processing
    
    Args:
        dir_name: Directory containing the data
        blech_clust_dir: Directory containing blech_clust code
        electrode_layout_frame: DataFrame with electrode layout info
        all_electrodes: List of all electrode indices
        all_params_dict: Dictionary of processing parameters
    """
    script_save_path = os.path.join(dir_name, 'temp')
    if not os.path.exists(script_save_path):
        os.mkdir(script_save_path)

    # Generate single electrode processing script
    with open(os.path.join(script_save_path, 'blech_process_single.sh'), 'w') as f:
        f.write('#!/bin/bash \n')
        f.write(f'BLECH_DIR={blech_clust_dir} \n')
        f.write(f'DATA_DIR={dir_name} \n')
        f.write('ELECTRODE_NUM=$1 \n')
        f.write('python $BLECH_DIR/blech_process.py $DATA_DIR $ELECTRODE_NUM \n')
        f.write('python $BLECH_DIR/utils/cluster_stability.py $DATA_DIR $ELECTRODE_NUM \n')

    # Generate parallel processing script
    num_cpu = multiprocessing.cpu_count()
    
    electrode_bool = electrode_layout_frame.loc[
        electrode_layout_frame.electrode_ind.isin(all_electrodes)]
    not_none_bool = electrode_bool.loc[~electrode_bool.CAR_group.isin(
        ["none", "None", 'na'])]
    not_emg_bool = not_none_bool.loc[
        ~not_none_bool.CAR_group.str.contains('emg')
    ]
    bash_electrode_list = not_emg_bool.electrode_ind.values
    job_count = np.min(
            (
                len(bash_electrode_list), 
                int(num_cpu-2), 
                all_params_dict["max_parallel_cpu"]
                )
            )

    with open(os.path.join(script_save_path, 'blech_process_parallel.sh'), 'w') as f:
        f.write('#!/bin/bash \n')
        f.write(f'DIR={dir_name} \n')
        print(f"parallel -k -j {job_count} --noswap --load 100% --progress " +
              "--memfree 4G --ungroup --retry-failed " +
              f"--joblog $DIR/results.log " +
              "bash $DIR/temp/blech_process_single.sh " +\
              f"::: {' '.join([str(x) for x in bash_electrode_list])}",
              file=f)


# Get blech_clust path
script_path = os.path.realpath(__file__)
blech_clust_dir = os.path.dirname(script_path)

# Check that template file is present
params_template_path = os.path.join(
    blech_clust_dir,
    'params/sorting_params_template.json')
if not os.path.exists(params_template_path):
    print('=== Sorting Params Template file not found. ===')
    print('==> Please copy [[ blech_clust/params/_templates/sorting_params_template.json ]] to [[ blech_clust/params/sorting_params_template.json ]] and update as needed.')
    exit()
############################################################


metadata_handler = imp_metadata([[], args.dir_name])
dir_name = metadata_handler.dir_name

# Perform pipeline graph check
this_pipeline_check = pipeline_graph_check(dir_name)
# If info_dict present but execution log is not
# just create the execution log with blech_exp_info marked
if 'info_dict' in dir(metadata_handler) and not os.path.exists(metadata_handler.dir_name + '/execution_log.json'):
    blech_exp_info_path = os.path.join(blech_clust_dir, 'blech_exp_info.py')
    this_pipeline_check.write_to_log(blech_exp_info_path, 'attempted')
    this_pipeline_check.write_to_log(blech_exp_info_path, 'completed')
    print('Execution log created for blech_exp_info')
this_pipeline_check.check_previous(script_path)
this_pipeline_check.write_to_log(script_path, 'attempted')

print(f'Processing : {dir_name}')
os.chdir(dir_name)

info_dict = metadata_handler.info_dict
file_list = metadata_handler.file_list


# Get the type of data files (.rhd or .dat)
if 'auxiliary.dat' in file_list:
    file_type = ['one file per signal type']
elif sum(['rhd' in x for x in file_list]) > 1: # multiple .rhd files
    file_type = ['traditional']
else:
    file_type = ['one file per channel']


# Create HDF5 handler and initialize groups
hdf5_handler = HDF5Handler(dir_name, force_run)
continue_bool, reload_data_str = hdf5_handler.initialize_groups()
hdf5_name = hdf5_handler.hdf5_name

# Create directories to store waveforms, spike times, clustering results, and plots
dir_list = ['spike_waveforms', 'spike_times', 'clustering_results', 
            'Plots', 'memory_monitor_clustering']

dir_exists = [x for x in dir_list if os.path.exists(x)]
if dir_exists and not force_run:
    recreate_msg = f'Following dirs are present:\n{dir_exists}\nOverwrite dirs? (yes/y/n/no) ::: '
    recreate_str, continue_bool = entry_checker(
        msg=recreate_msg,
        check_func=lambda x: x in ['y', 'yes', 'n', 'no'],
        fail_response='Please enter (yes/y/n/no)')
else:
    continue_bool = True
    recreate_str = 'y'

if not continue_bool:
    quit()
    
if recreate_str in ['y', 'yes']:
    [shutil.rmtree(x) for x in dir_list if os.path.exists(x)]
    [os.makedirs(x) for x in dir_list]

print('Created dirs in data folder')

# Get lists of amplifier and digital input files
file_lists = {
    'one file per signal type': {
        'electrodes': ['amplifier.dat'],
        'dig_in': ['digitalin.dat']
    },
    'one file per channel': {
        'electrodes': sorted([name for name in file_list if name.startswith('amp-')]),
        'dig_in': sorted([name for name in file_list if name.startswith('board-DI')])
    },
    'traditional': {
        'rhd': sorted([name for name in file_list if name.endswith('.rhd')])
    }
}

if file_type[0] != 'traditional':
    electrodes_list = file_lists[file_type[0]]['electrodes']
    dig_in_file_list = file_lists[file_type[0]]['dig_in']

    # Use info file for port list calculation
    info_file = np.fromfile(dir_name + '/info.rhd', dtype=np.dtype('float32'))
    sampling_rate = int(info_file[2])

    # Read the time.dat file for use in separating out 
    # the one file per signal type data
    num_recorded_samples = len(np.fromfile(
        dir_name + '/' + 'time.dat', dtype=np.dtype('float32')))
    total_recording_time = num_recorded_samples/sampling_rate  # In seconds

    check_str = f'Amplifier files: {electrodes_list} \nSampling rate: {sampling_rate} Hz'\
            f'\nDigital input files: {dig_in_file_list} \n ---------- \n \n'
    print(check_str)
    ports = info_dict['ports']

if file_type[0] == 'traditional':
    rhd_file_list = file_lists[file_type[0]]['rhd']
    with open(rhd_file_list[0], 'rb') as f:
        header = read_header(f)
    # temp_file, data_present = importrhdutilities.load_file(file_list[0])
    amp_channel_ports = [x['port_prefix'] for x in header['amplifier_channels']]
    amp_channel_names = [x['native_channel_name'] for x in header['amplifier_channels']]
    dig_in_channels = [x['native_channel_name'] for x in header['board_dig_in_channels']]
    sampling_rate = int(header['sample_rate'])
    ports = np.unique(amp_channel_ports)

    check_str = f"""
    == Amplifier channels: \n{amp_channel_names}\n
    == Digital input channels: \n{dig_in_channels}\n
    == Sampling rate: {sampling_rate} Hz\n
    == Ports: {ports}\n
    """
    print(check_str)



if file_type == ['one file per channel']:
    print("\tOne file per CHANNEL Detected")
    # Read dig-in data
    # Pull out the digital input channels used,
    # and convert them to integers
    dig_in_int = [x.split('-')[-1].split('.')[0] for x in dig_in_file_list]
    dig_in_int = sorted([(x) for x in dig_in_int])
elif file_type == ['one file per signal type']:
    print("\tOne file per SIGNAL Detected")
    dig_in_int = np.arange(info_dict['dig_ins']['count'])
elif file_type == ['traditional']:
    print('Tranditional INTAN file format detected')
    dig_in_int = sorted([x.split('-')[-1].split('.')[0] for x in dig_in_channels])

check_str = f'ports used: {ports} \n sampling rate: {sampling_rate} Hz'\
            f'\n digital inputs on intan board: {dig_in_int}'

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
if reload_data_str in ['y', 'yes']:
    if file_type == ['one file per channel']:
        read_file.read_digins(hdf5_name, dig_in_int, dig_in_file_list)
        read_file.read_electrode_channels(hdf5_name, electrode_layout_frame)
        if len(emg_channels) > 0:
            read_file.read_emg_channels(hdf5_name, electrode_layout_frame)
    elif file_type == ['one file per signal type']:
        read_file.read_digins_single_file(hdf5_name, dig_in_int, dig_in_file_list)
        # This next line takes care of both electrodes and emgs
        read_file.read_electrode_emg_channels_single_file(
            hdf5_name, electrode_layout_frame, electrodes_list, num_recorded_samples, emg_channels)
    elif file_type == ['traditional']:
        read_file.read_traditional_intan(
                hdf5_name, 
                rhd_file_list, 
                electrode_layout_frame,
                dig_in_int,
                )
else:
    print('Data already present...Not reloading data')

# Write out template params file to directory if not present
params_template = json.load(open(params_template_path, 'r'))
# Info on taste digins and laser should be in exp_info file
all_params_dict = params_template.copy()
all_params_dict['sampling_rate'] = sampling_rate

params_out_path = hdf5_name.split('.')[0] + '.params'
if not os.path.exists(params_out_path):
    print('No params file found...Creating new params file')
    with open(params_out_path, 'w') as params_file:
        json.dump(all_params_dict, params_file, indent=4)
else:
    print("Params file already present...not writing a new one")

##############################
# Test correlation between channels for quality check
print()
print('Calculating correlation matrix for quality check')
# qa_down_rate = all_params_dict["qa_params"]["downsample_rate"]
n_corr_samples = all_params_dict["qa_params"]["n_corr_samples"]
qa_threshold = all_params_dict["qa_params"]["bridged_channel_threshold"]
down_dat_stack, chan_names = channel_corr.get_all_channels(
        hdf5_name, 
        n_corr_samples = n_corr_samples)
corr_mat = channel_corr.intra_corr(down_dat_stack)
qa_out_path = os.path.join(dir_name, 'QA_output')
if not os.path.exists(qa_out_path):
    os.mkdir(qa_out_path)
else:
    # Delete dir and remake
    shutil.rmtree(qa_out_path)
    os.mkdir(qa_out_path)
channel_corr.gen_corr_output(corr_mat, 
                   qa_out_path, 
                   qa_threshold,)
##############################

##############################
# Also output a plot with digin and laser info

# Get digin and laser info
print('Getting trial markers from digital inputs')
dig_in_array = hdf5_handler.get_digital_inputs(sampling_rate)
# Downsample to 10 seconds
# dig_in_array = dig_in_array[:, :(dig_in_array.shape[1]//sampling_rate)*sampling_rate]
# dig_in_array = np.reshape(dig_in_array, (len(dig_in_array), -1, sampling_rate)).sum(axis=2)
dig_in_markers = np.where(dig_in_array > 0)
del dig_in_array

# Check if laser is present
laser_dig_in = info_dict['laser_params']['dig_in']

dig_in_map = {}
for num, name in zip(info_dict['taste_params']['dig_ins'], info_dict['taste_params']['tastes']):
    dig_in_map[num] = name
for num in laser_dig_in:
    dig_in_map[num] = 'laser'

# Sort dig_in_map
dig_in_map = {num:dig_in_map[num] for num in sorted(list(dig_in_map.keys()))}
dig_in_str = [f'{num}: {dig_in_map[num]}' for num in dig_in_map.keys()]

plt.scatter(dig_in_markers[1], dig_in_markers[0], s=50, marker='|', c='k')
# If there is a laser_dig_in, mark laser trials with axvline
if len(laser_dig_in) > 0:
    laser_markers = np.where(dig_in_markers[0] == laser_dig_in)[0]
    for marker in laser_markers:
        plt.axvline(dig_in_markers[1][marker], c='yellow', lw=2, alpha = 0.5,
                    zorder = -1)
plt.yticks(np.array(list(dig_in_map.keys())), dig_in_str)
plt.title('Digital Inputs')
plt.xlabel('Time (s)')
plt.ylabel('Digital Input Channel')
plt.savefig(os.path.join(qa_out_path, 'digital_inputs.png'))
plt.close()

##############################


# Generate the processing scripts
generate_processing_scripts(dir_name, blech_clust_dir, electrode_layout_frame,
                          all_electrodes, all_params_dict)

print('blech_clust.py complete \n')
print('*** Please check params file to make sure all is good ***\n')

# Write success to log
this_pipeline_check.write_to_log(script_path, 'completed')
