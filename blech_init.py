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

import argparse  # noqa
parser = argparse.ArgumentParser(description='Load data and create hdf5 file')
parser.add_argument('dir_name', type=str,
                    help='Directory name with data files')
parser.add_argument('--force_run', action='store_true',
                    help='Force run the script without asking user')
args = parser.parse_args()
force_run = args.force_run

# Necessary blech_clust modules
from blech_clust.utils.importrhdutilities import read_header  # noqa
from blech_clust.utils.blech_process_utils import path_handler  # noqa
from blech_clust.utils.blech_utils import entry_checker, imp_metadata, pipeline_graph_check  # noqa
from blech_clust.utils.qa_utils import channel_corr  # noqa
from blech_clust.utils import read_file  # noqa
from blech_clust.utils.blech_channel_profile import plot_channels  # noqa
# Necessary python modules
from ast import literal_eval  # noqa
import pylab as plt  # noqa
import shutil  # noqa
import pandas as pd  # noqa
import glob  # noqa
import json  # noqa
import multiprocessing  # noqa
import numpy as np  # noqa
import sys  # noqa
import tables  # noqa
import os  # noqa


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
            self.hdf5_name = str(os.path.dirname(
                self.dir_name)).split('/')[-1]+'.h5'
            print(f'No HDF5 found...Creating file {self.hdf5_name}')
            self.hf5 = tables.open_file(
                self.hdf5_name, 'w', title=self.hdf5_name[-1])

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
                msg=reload_msg,
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
        f.write(
            'python $BLECH_DIR/utils/cluster_stability.py $DATA_DIR $ELECTRODE_NUM \n')

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
              "bash $DIR/temp/blech_process_single.sh " +
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

file_type = info_dict['file_type']

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
    },
    'one file per channel': {
        'electrodes': sorted([name for name in file_list if name.startswith('amp-')]),
    },
    'traditional': {
        'rhd': sorted([name for name in file_list if name.endswith('.rhd')])
    }
}

# Valid file types
VALID_FILE_TYPES = ['one file per signal type',
                    'one file per channel', 'traditional']
if file_type not in VALID_FILE_TYPES:
    raise ValueError(
        f"Invalid file_type: {file_type}. Must be one of: {VALID_FILE_TYPES}")

# Get digin and laser info
print('Getting trial markers from digital inputs')
# dig_in_array = hdf5_handler.get_digital_inputs(sampling_rate)
this_dig_handler = read_file.DigInHandler(dir_name, file_type)
this_dig_handler.load_dig_in_frame()

print('DigIn data loaded')
print(this_dig_handler.dig_in_frame.drop(columns='pulse_times'))

if file_type != 'traditional':
    electrodes_list = file_lists[file_type]['electrodes']

    if file_type == 'one file per channel':
        print("\tOne file per CHANNEL Detected")
    elif file_type == 'one file per signal type':
        print("\tOne file per SIGNAL Detected")

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
    ports = info_dict['ports']

    check_str = f'Amplifier files: {electrodes_list} \nSampling rate: {sampling_rate} Hz'\
        + '\n Ports : {ports} \n---------- \n \n'
    print(check_str)

if file_type == 'traditional':
    print('Tranditional INTAN file format detected')
    rhd_file_list = file_lists[file_type]['rhd']
    with open(rhd_file_list[0], 'rb') as f:
        header = read_header(f)
    # temp_file, data_present = importrhdutilities.load_file(file_list[0])
    amp_channel_ports = [x['port_prefix']
                         for x in header['amplifier_channels']]
    amp_channel_names = [x['native_channel_name']
                         for x in header['amplifier_channels']]
    sampling_rate = int(header['sample_rate'])
    ports = np.unique(amp_channel_ports)

    check_str = f"""
    == Amplifier channels: \n{amp_channel_names}\n
    == Sampling rate: {sampling_rate} Hz\n
    == Ports: {ports}\n
    """
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
    if file_type == 'one file per channel':
        # read_file.read_digins(hdf5_name, dig_in_int, dig_in_file_list)
        read_file.read_electrode_channels(hdf5_name, electrode_layout_frame)
        if len(emg_channels) > 0:
            read_file.read_emg_channels(hdf5_name, electrode_layout_frame)
    elif file_type == 'one file per signal type':
        # read_file.read_digins_single_file(hdf5_name, dig_in_int, dig_in_file_list)
        # This next line takes care of both electrodes and emgs
        read_file.read_electrode_emg_channels_single_file(
            hdf5_name, electrode_layout_frame, electrodes_list, num_recorded_samples, emg_channels)
    elif file_type == 'traditional':
        read_file.read_traditional_intan(
            hdf5_name,
            rhd_file_list,
            electrode_layout_frame,
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
    n_corr_samples=n_corr_samples)
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
# Also write out the correlation matrix to qa_out_path
np.save(os.path.join(qa_out_path, 'channel_corr_mat.npy'), corr_mat)

# Check average intra-CAR similarity and write a warning if it's below threshold
try:
    avg_threshold = all_params_dict.get('qa_params', {}).get('avg_intra_car_similarity_threshold', None)
    if avg_threshold is not None:
        # Exclude emg and none CAR groups from analysis
        emg_bool = ~electrode_layout_frame.CAR_group.str.contains('emg', case=False)
        none_bool = ~electrode_layout_frame.CAR_group.str.contains('none', case=False)
        fin_bool = np.logical_and(emg_bool, none_bool)
        elec_frame = electrode_layout_frame[fin_bool]

        group_means = []
        for group_name in elec_frame.CAR_group.unique():
            # Get electrode numbers for this group
            try:
                electrode_nums = elec_frame.loc[elec_frame.CAR_group == group_name, 'electrode_num'].astype(int).values
            except Exception:
                # If electrode_num column is missing, fall back to electrode_ind
                electrode_nums = elec_frame.loc[elec_frame.CAR_group == group_name, 'electrode_ind'].astype(int).values

            # Find indices in chan_names corresponding to these electrode numbers
            # chan_names is returned from get_all_channels (list of channel numbers)
            try:
                chan_inds = [i for i, c in enumerate(chan_names) if int(c) in electrode_nums]
            except Exception:
                chan_inds = []

            if len(chan_inds) > 1:
                submat = corr_mat[np.ix_(chan_inds, chan_inds)].astype(float)
                # Exclude diagonal
                if submat.size > 1:
                    triu = np.triu_indices(submat.shape[0], k=1)
                    values = submat[triu]
                    # Drop NaNs and compute mean
                    values = values[~np.isnan(values)]
                    if values.size > 0:
                        group_means.append(np.mean(values))
        if len(group_means) > 0:
            overall_avg = float(np.mean(group_means))
            warnings_path = os.path.join(qa_out_path, 'warnings.txt')
            if overall_avg < float(avg_threshold):
                with open(warnings_path, 'a') as wf:
                    wf.write('\n')
                    wf.write('=== Average intra-CAR similarity warning ===\n')
                    wf.write(f'Average intra-CAR similarity across groups: {overall_avg:.4f}\n')
                    wf.write(f'Threshold: {avg_threshold:.4f}\n')
                    wf.write('Per-group mean similarities:\n')
                    for gname, gmean in zip(elec_frame.CAR_group.unique(), group_means):
                        wf.write(f'  {gname}: {gmean:.4f}\n')
                    wf.write('=== End Average intra-CAR similarity warning ===\n')
except Exception:
    # Do not fail the pipeline QA step if this check errors
    pass

# Generate channel profile plots for non-traditional file types
if file_type in ['one file per channel', 'one file per signal type']:
    print('\nGenerating channel profile plots')
    plot_channels(dir_name, qa_out_path, file_type)
##############################

##############################
# Also output a plot with digin and laser info

# Downsample to 10 seconds
dig_in_pulses = this_dig_handler.dig_in_frame.pulse_times.values
dig_in_pulses = [literal_eval(x) for x in dig_in_pulses]
# Take starts of pulses
dig_in_pulses = [[x[0] for x in this_dig] for this_dig in dig_in_pulses]
dig_in_markers = [np.array(x) / sampling_rate for x in dig_in_pulses]

# Check if laser is present
laser_dig_in = info_dict['laser_params']['dig_in_nums']

dig_in_map = {}
for num, name in zip(info_dict['taste_params']['dig_in_nums'], info_dict['taste_params']['tastes']):
    dig_in_map[num] = name
for num in laser_dig_in:
    dig_in_map[num] = 'laser'

# Sort dig_in_map
dig_in_map = {num: dig_in_map[num] for num in sorted(list(dig_in_map.keys()))}
dig_in_str = [f'{num}: {dig_in_map[num]}' for num in dig_in_map.keys()]

for i, vals in enumerate(dig_in_markers):
    plt.scatter(vals,
                np.ones_like(vals)*i,
                s=50, marker='|', c='k')
# If there is a laser_dig_in, mark laser trials with axvline
if len(laser_dig_in) > 0:
    # laser_markers = np.where(dig_in_markers[0] == laser_dig_in)[0]
    laser_markers = dig_in_markers[laser_dig_in[0]]
    for marker in laser_markers:
        plt.axvline(marker, c='yellow', lw=2, alpha=0.5,
                    zorder=-1)
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
