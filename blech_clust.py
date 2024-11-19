import os
import glob
import json
import shutil
import numpy as np
import multiprocessing
import pandas as pd
import tables
import matplotlib.pyplot as plt

# Local imports
from utils import read_file
from utils.blech_utils import imp_metadata, pipeline_graph_check, entry_checker
from utils.importrhdutilities import read_header
from utils.qa_utils import channel_corr

class DataProcessor:
    """Handles data processing operations for blech_clust"""
    
    def __init__(self, dir_name, force_run=False):
        self.dir_name = dir_name
        self.force_run = force_run
        self.sampling_rate = None
        self.hdf5_name = None
        self.file_type = None
        
    def setup_hdf5(self):
        """Create or open HDF5 file and setup basic structure"""
        h5_search = glob.glob('*.h5')
        if len(h5_search):
            self.hdf5_name = h5_search[0]
            print(f'HDF5 file found...Using file {self.hdf5_name}')
        else:
            self.hdf5_name = str(os.path.dirname(self.dir_name)).split('/')[-1]+'.h5'
            print(f'No HDF5 found...Creating file {self.hdf5_name}')
            
        return self.setup_hdf5_groups()
    
    def setup_hdf5_groups(self):
        """Setup HDF5 file groups"""
        hf5 = tables.open_file(self.hdf5_name, 'r+' if os.path.exists(self.hdf5_name) else 'w')
        group_list = ['raw', 'raw_emg', 'digital_in', 'digital_out']
        found_list = [g for g in group_list if '/'+g in hf5]
        
        if found_list and not self.force_run:
            print(f'Data already present: {found_list}')
            reload_data_str, continue_bool = entry_checker(
                msg='Reload data? (yes/y/n/no) ::: ',
                check_func=lambda x: x in ['y', 'yes', 'n', 'no'],
                fail_response='Please enter (yes/y/n/no)')
        else:
            continue_bool = True
            reload_data_str = 'y'
            
        if continue_bool and reload_data_str in ['y', 'yes']:
            for group in group_list:
                if '/'+group in hf5:
                    hf5.remove_node('/', group, recursive=True)
                hf5.create_group('/', group)
            print('Created nodes in HF5')
            
        hf5.close()
        return continue_bool, reload_data_str

class DirectoryManager:
    """Manages directory creation and verification"""
    
    @staticmethod
    def setup_directories(force_run=False):
        """Create necessary directories for analysis outputs"""
        dir_list = [
            'spike_waveforms',
            'spike_times', 
            'clustering_results',
            'Plots',
            'memory_monitor_clustering'
        ]
        
        dir_exists = [x for x in dir_list if os.path.exists(x)]
        
        if dir_exists and not force_run:
            recreate_msg = (f'Following dirs are present:\n{dir_exists}\n'
                          'Overwrite dirs? (yes/y/n/no) ::: ')
            recreate_str, continue_bool = entry_checker(
                msg=recreate_msg,
                check_func=lambda x: x in ['y', 'yes', 'n', 'no'],
                fail_response='Please enter (yes/y/n/no)')
        else:
            continue_bool = True
            recreate_str = 'y'
            
        if continue_bool and recreate_str in ['y', 'yes']:
            for x in dir_list:
                if os.path.exists(x):
                    shutil.rmtree(x)
                os.makedirs(x)
            print('Created dirs in data folder')
            
        return continue_bool

class Config:
    """Handles configuration and argument parsing"""
    
    @staticmethod
    def parse_args():
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('dir_name', help='Directory containing the data files')
        parser.add_argument('--force-run', action='store_true', 
                          help='Force run without user prompts')
        return parser.parse_args()

    @staticmethod
    def check_params_template(blech_clust_dir):
        """Check and return path to params template"""
        template_name = 'blech_params.json'
        return os.path.join(blech_clust_dir, template_name)

def main():
    # Parse arguments and setup
    args = Config.parse_args()
    force_run = args.force_run
    
    script_path = os.path.realpath(__file__)
    blech_clust_dir = os.path.dirname(script_path)
    
    # Check template and setup metadata
    params_template_path = Config.check_params_template(blech_clust_dir)
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
    found_list = []
    for this_group in group_list:
        if '/'+this_group in hf5:
            found_list.append(this_group)

    if len(found_list) > 0 and not force_run:
        print(f'Data already present: {found_list}')
        reload_data_str, continue_bool = entry_checker(
                msg='Reload data? (yes/y/n/no) ::: ',
                check_func=lambda x: x in ['y', 'yes', 'n', 'no'],
                fail_response='Please enter (yes/y/n/no)')
    else:
        continue_bool = True
        reload_data_str = 'y'

    if continue_bool:
        if reload_data_str in ['y', 'yes']:
            for this_group in group_list:
                if '/'+this_group in hf5:
                    hf5.remove_node('/', this_group, recursive=True)
                hf5.create_group('/', this_group)
            print('Created nodes in HF5')
    hf5.close()

    # Create directories to store waveforms, spike times, clustering results, and plots
    # And a directory for dumping files talking about memory usage in blech_process.py
    # Check if dirs are already present, if they are, ask to delete and continue
    # or abort
    dir_list = ['spike_waveforms',
                'spike_times',
                'clustering_results',
                'Plots',
                'memory_monitor_clustering']
    dir_exists = [x for x in dir_list if os.path.exists(x)]
    recreate_msg = f'Following dirs are present :' + '\n' + f'{dir_exists}' + \
        '\n' + 'Overwrite dirs? (yes/y/n/no) ::: '

    # If dirs exist, check with user
    if len(dir_exists) > 0 and not force_run:
        recreate_str, continue_bool = entry_checker(
            msg=recreate_msg,
            check_func=lambda x: x in ['y', 'yes', 'n', 'no'],
            fail_response='Please enter (yes/y/n/no)')
    # Otherwise, create all of them
    else:
        continue_bool = True
        recreate_str = 'y'

    # Break if user said n/no or gave exit signal
    if continue_bool:
        if recreate_str in ['y', 'yes']:
            for x in dir_list:
                if os.path.exists(x):
                    shutil.rmtree(x)
                os.makedirs(x)
    else:
        return

    print('Created dirs in data folder')

    # Get lists of amplifier and digital input files
    if file_type == ['one file per signal type']:
        electrodes_list = ['amplifier.dat']
        dig_in_file_list = ['digitalin.dat']
    elif file_type == ['one file per channel']:
        electrodes_list = [name for name in file_list if name.startswith('amp-')]
        dig_in_file_list = [name for name in file_list if name.startswith('board-DI')]
    elif file_type == ['traditional']:
        rhd_file_list = sorted([name for name in file_list if name.endswith('.rhd')])
    

if not file_type == ['traditional']:
    electrodes_list = sorted(electrodes_list)
    dig_in_file_list = sorted(dig_in_file_list)

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
else:
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
dig_in_list = []
with tables.open_file(hdf5_name, 'r') as hf5:
    for i, this_dig_in in enumerate(hf5.root.digital_in):
        this_dig_in = this_dig_in[:]
        len_dig_in = len(this_dig_in)
        this_dig_in = this_dig_in[:(len_dig_in//sampling_rate)*sampling_rate]
        this_dig_in = np.reshape(this_dig_in, (-1, sampling_rate)).sum(axis=-1)
        dig_in_list.append(this_dig_in)
        # dig_in_array = np.stack([x[:] for x in hf5.root.digital_in])
dig_in_array = np.stack(dig_in_list)
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

# Write single runner file to data directory
script_save_path = os.path.join(dir_name, 'temp')
if not os.path.exists(script_save_path):
    os.mkdir(script_save_path)

with open(os.path.join(script_save_path, 'blech_process_single.sh'), 'w') as f:
    f.write('#!/bin/bash \n')
    f.write(f'BLECH_DIR={blech_clust_dir} \n')
    f.write(f'DATA_DIR={dir_name} \n')
    f.write('ELECTRODE_NUM=$1 \n')
    f.write('python $BLECH_DIR/blech_process.py $DATA_DIR $ELECTRODE_NUM \n')
    f.write('python $BLECH_DIR/utils/cluster_stability.py $DATA_DIR $ELECTRODE_NUM \n')

# Dump shell file(s) for running GNU parallel job on the user's 
# blech_clust folder on the desktop
# First get number of CPUs - parallel be asked to run num_cpu-1 threads in parallel
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
f = open(os.path.join(script_save_path, 'blech_process_parallel.sh'), 'w')
f.write('#!/bin/bash \n')
f.write(f'DIR={dir_name} \n')
print(f"parallel -k -j {job_count} --noswap --load 100% --progress " +
      "--memfree 4G --ungroup --retry-failed " +
      f"--joblog $DIR/results.log " +
      "bash $DIR/temp/blech_process_single.sh " +\
      f"::: {' '.join([str(x) for x in bash_electrode_list])}",
      file=f)
f.close()

print('blech_clust.py complete \n')
print('*** Please check params file to make sure all is good ***\n')

# Write success to log
this_pipeline_check.write_to_log(script_path, 'completed')
