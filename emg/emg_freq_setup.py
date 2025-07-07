"""
This module sets up EMG data for running the envelope of EMG recordings through a local Bayesian Spectrum Analysis (BSA). It requires an installation of R and the R library BaSAR. The script is a preparatory step for `emg_local_BSA_execute.py`.

- Imports necessary libraries and modules, including custom utilities for metadata and path handling.
- Retrieves the directory name containing data files using metadata.
- Performs a pipeline graph check to ensure the script's execution order.
- Sets up parameters for processing, including durations and plotting parameters.
- Retrieves paths for data directories and checks for the existence of necessary parameter files.
- Loads EMG envelope and filtered data from an HDF5 file and organizes it into lists.
- Converts the EMG data into a pandas DataFrame for further processing.
- Creates output directories and saves the EMG envelope data to a CSV file and a NumPy file.
- Deletes previous results and logs, and sets up shell scripts for running parallel jobs using GNU parallel.
- Merges the EMG data with trial information from a CSV file and saves the merged data.
- Generates plots of the EMG envelope and filtered data, saving them to the output directory.
- Logs the successful execution of the script.
"""
# Sets up emg data for running the envelope of emg recordings (env.npy) through
# a local Bayesian Spectrum Analysis (BSA).
# Needs an installation of R (installing Rstudio on Ubuntu is enough) -
# in addition, the R library BaSAR needs to be installed from the CRAN
# archives (https://cran.r-project.org/src/contrib/Archive/BaSAR/)
# This is the starting step for emg_local_BSA_execute.py

# Import stuff
import numpy as np
import os
import multiprocessing
import sys
import shutil
from glob import glob
import json
import tables
import pandas as pd
import pylab as plt

test_bool = False

if test_bool:
    # data_dir = '/home/abuzarmahmood/projects/blech_clust/pipeline_testing/test_data_handling/test_data/KM45_5tastes_210620_113227_new/'
    data_dir = '/home/abuzarmahmood/Desktop/blech_clust/pipeline_testing/test_data_handling/test_data/eb24_behandephys_11_12_24_241112_114659_copy'
    # script_path = '/home/abuzarmahmood/projects/blech_clust/emg/emg_freq_setup.py'
    script_path = '/home/abuzarmahmood/Desktop/blech_clust/emg/emg_freq_setup.py'
    blech_clust_dir = os.path.dirname(os.path.dirname(script_path))
    sys.path.append(blech_clust_dir)
    print(f'blech_clust_dir: {blech_clust_dir}')
    # from utils.blech_process_utils import path_handler  # noqa: E402
    from utils.blech_utils import imp_metadata, pipeline_graph_check  # noqa: E402

    # Get name of directory with the data files
    metadata_handler = imp_metadata([[], data_dir])
    data_dir = metadata_handler.dir_name

else:
    script_path = os.path.realpath(__file__)

    blech_clust_dir = os.path.dirname(os.path.dirname(script_path))
    sys.path.append(blech_clust_dir)
    print(f'blech_clust_dir: {blech_clust_dir}')
    # from utils.blech_process_utils import path_handler  # noqa: E402
    from utils.blech_utils import imp_metadata, pipeline_graph_check  # noqa: E402

    # Get name of directory with the data files
    metadata_handler = imp_metadata(sys.argv)
    data_dir = metadata_handler.dir_name

    # Perform pipeline graph check
    this_pipeline_check = pipeline_graph_check(data_dir)
    this_pipeline_check.check_previous(script_path)
    this_pipeline_check.write_to_log(script_path, 'attempted')

# Get paths
# this_path_handler = path_handler()
# blech_clust_dir = this_path_handler.blech_clust_dir
blech_emg_dir = os.path.join(blech_clust_dir, 'emg')
print(f'blech_emg_dir: {blech_emg_dir}')

os.chdir(data_dir)
print(f'Processing : {data_dir}')

##############################
# Setup params
##############################
params_dict = metadata_handler.params_dict
durations = params_dict['spike_array_durations']
pre_stim = int(durations[0])
plot_params = params_dict['psth_params']['durations']

fin_inds = [pre_stim - plot_params[0], pre_stim + plot_params[1]]
time_vec = np.arange(-plot_params[0], plot_params[1])


print(f'blech_clust_dir: {blech_clust_dir}')
print()
emg_params_path = os.path.join(blech_clust_dir, 'params', 'emg_params.json')

if not os.path.exists(emg_params_path):
    print('=== EMG params file not found. ===')
    print(
        '==> Please copy [[ blech_clust/params/_templates/emg_params.json ]] to [[ blech_clust/params/emg_params.json ]] and update as needed.')
    exit()

emg_params_dict = json.load(open(emg_params_path, 'r'))
use_BSA_bool = emg_params_dict['use_BSA']

hf5 = tables.open_file(metadata_handler.hdf5_name, 'r')

# Get all emg_env data
# Structure /emg_data/dig_in_<xx>/processed_emg/<xx>_emg_env
dig_in_nodes = hf5.list_nodes('/emg_data')
dig_in_nodes = [x for x in dig_in_nodes if 'dig_in' in x._v_name]
emg_env_node_names = []
emg_env_data = []
emg_filt_data = []
for node in dig_in_nodes:
    node_list = hf5.list_nodes(
        os.path.join(node._v_pathname, 'processed_emg'),
        classname='Array')
    env_node_list = [x for x in node_list if 'emg_env' in x._v_name]
    env_dat_list = [x.read() for x in env_node_list]
    emg_env_data.extend(env_dat_list)
    emg_env_node_names.extend([x._v_pathname for x in env_node_list])

    filt_node_list = [x for x in node_list if 'emg_filt' in x._v_name]
    filt_dat_list = [x.read() for x in filt_node_list]
    emg_filt_data.extend(filt_dat_list)

hf5.close()

# Save everything as pandas dataframe
dig_in_list = [x.split('/')[2] for x in emg_env_node_names]
car_list = [x.split('_')[-3].split('/')[1] for x in emg_env_node_names]
trial_lens = [x.shape[0] for x in emg_env_data]

fin_dig_list = [[x]*y for x, y in zip(dig_in_list, trial_lens)]
fin_car_list = [[x]*y for x, y in zip(car_list, trial_lens)]
fin_dig_list = [item for sublist in fin_dig_list for item in sublist]
fin_car_list = [item for sublist in fin_car_list for item in sublist]
trial_inds = [list(range(x)) for x in trial_lens]
trial_inds = [item for sublist in trial_inds for item in sublist]
flat_emg_env_data = np.stack(
    [item for sublist in emg_env_data for item in sublist]
)
flat_emg_filt_data = np.stack(
    [item for sublist in emg_filt_data for item in sublist]
)

emg_env_df = pd.DataFrame(
    dict(
        dig_in=[int(x.split('_')[-1]) for x in fin_dig_list],
        car=fin_car_list,
        trial_inds=trial_inds,
    )
)

emg_env_df['dig_in_ind'] = emg_env_df.dig_in.rank(
    method='dense').astype(int) - 1


emg_output_dir = os.path.join(data_dir, 'emg_output')
plot_dir = os.path.join(emg_output_dir, 'plots')
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

print(f'emg_output_dir: {emg_output_dir}')
os.chdir(emg_output_dir)

# Write the emg_filt data to a file
np.save('flat_emg_filt_data.npy', flat_emg_filt_data)

# Write the emg_env data to a file
emg_env_df.to_csv('emg_env_df.csv')
np.save('flat_emg_env_data.npy', flat_emg_env_data)

############################################################
# Also export to numpy
# These will NOT be used for downstream processing
# but are exported for backwards compatibility

# Matching commit: 431ceb
# ==============================
# # NOTE: Currently DIFFERENT sig_trials for each channel
# # Save the highpass filtered signal,
# # the envelope and the indicator of significant trials as a np array
# # Iterate over channels and save them in different directories
# for num,this_name in enumerate(emg_car_names):
#     #dir_path = f'emg_output/emg_channel{num}'
#     dir_path = f'emg_output/{this_name}'
#     if os.path.exists(dir_path):
#         shutil.rmtree(dir_path)
#     os.makedirs(dir_path)
#     # emg_filt (output shape): tastes x trials x time
#     np.save(os.path.join(dir_path, f'emg_filt.npy'), emg_filt_list[num])
#     # env (output shape): tastes x trials x time
#     np.save(os.path.join(dir_path, f'emg_env.npy'), emg_env_list[num])
#     # sig_trials (output shape): tastes x trials
#     np.save(os.path.join(dir_path, 'sig_trials.npy'), sig_trials_list[num])

max_n_trials = emg_env_df.trial_inds.max() + 1
n_dig_ins = emg_env_df.dig_in.nunique()

emg_env_array = np.zeros((n_dig_ins, max_n_trials, flat_emg_env_data.shape[-1]),
                         dtype=np.float32)
emg_filt_array = np.zeros((n_dig_ins, max_n_trials, flat_emg_filt_data.shape[-1]),
                          dtype=np.float32)

# Fill both with nans
emg_env_array.fill(np.nan)
emg_filt_array.fill(np.nan)

# Fill the arrays with the data
for i, this_row in emg_env_df.iterrows():
    dig_in_ind = this_row['dig_in_ind']
    trial_ind = this_row['trial_inds']
    emg_env_array[dig_in_ind, trial_ind, :] = flat_emg_env_data[i]
    emg_filt_array[dig_in_ind, trial_ind, :] = flat_emg_filt_data[i]

# Save the arrays to numpy files
np.save(os.path.join(emg_output_dir, 'emg_env.npy'), emg_env_array)
np.save(os.path.join(emg_output_dir, 'emg_filt.npy'), emg_filt_array)

############################################################

print('Deleting emg_BSA_results')
if os.path.exists('emg_BSA_results'):
    shutil.rmtree('emg_BSA_results')
os.makedirs('emg_BSA_results')

# Also delete log
print('Deleting results.log')
if os.path.exists('results.log'):
    os.remove('results.log')

# Dump shell file(s) for running GNU parallel job on the
# user's blech_clust folder on the desktop
# First get number of CPUs - parallel be asked to run num_cpu-1
# threads in parallel
num_cpu = multiprocessing.cpu_count()
# Then produce the file generating the parallel command
f = open(os.path.join(blech_emg_dir, 'blech_emg_jetstream_parallel.sh'), 'w')
format_args = (
    int(num_cpu)-1,
    data_dir,
    len(emg_env_df)-1)
print(
    "parallel -k -j {:d} --noswap --load 100% --progress --ungroup --joblog {:s}/results.log bash blech_emg_jetstream_parallel1.sh ::: {{0..{:d}}}".format(
        *format_args),
    file=f)
f.close()

# Then produce the file that runs blech_process.py
if use_BSA_bool:
    file_name = 'emg_local_BSA_execute.py'
    print(' === Using BSA for frequency estimation ===')
else:
    file_name = 'emg_local_STFT_execute.py'
    print(' === Using STFT for frequency estimation ===')
f = open(os.path.join(blech_emg_dir, 'blech_emg_jetstream_parallel1.sh'), 'w')
print("export OMP_NUM_THREADS=1", file=f)
print(f"python {file_name} $1", file=f)
f.close()

# Finally dump a file with the data directory's location (blech.dir)
# If there is more than one emg group, this will iterate over them
f = open(os.path.join(blech_emg_dir, 'BSA_run.dir'), 'w')
print(data_dir, file=f)
f.close()

############################################################
# Merge the emg_env_df with trial_info_df
############################################################
# Also get trial_info_frame
trial_info_frame = pd.read_csv(os.path.join(data_dir, 'trial_info_frame.csv'))

merge_frame = pd.merge(emg_env_df, trial_info_frame,
                       left_on=['dig_in', 'trial_inds'],
                       right_on=['dig_in_num_taste', 'taste_rel_trial_num'],
                       how='left')
merge_frame.drop(
    columns=['dig_in', 'trial_inds',
             'start_taste', 'end_taste',
             'start_laser', 'end_laser',
             'laser_duration', 'laser_lag',
             'start_laser_ms', 'end_laser_ms',
             'start_taste_ms', 'end_taste_ms',
             ], inplace=True)

# Write out merge frame
merge_frame.to_csv(os.path.join(data_dir, 'emg_output/emg_env_merge_df.csv'))


############################################################
# Plots
############################################################
# Plot env using flat_emg_env and emg_env_merge_df

car_group = list(merge_frame.groupby('car'))

max_trials = merge_frame.taste_rel_trial_num.max() + 1

for car_name, car_data in car_group:
    n_digs = car_data.dig_in_num_taste.nunique()
    fig, ax = plt.subplots(max_trials, n_digs,
                           sharex=True, sharey=True,
                           figsize=(n_digs*4, max_trials)
                           )
    for i, (dig_name, dig_data) in enumerate(car_data.groupby('dig_in_name_taste')):
        ax[0, i].set_title(dig_name)
        dat_inds = dig_data.index.values
        dig_filt = flat_emg_env_data[dat_inds][:, fin_inds[0]:fin_inds[1]]
        for j, trial in enumerate(dig_filt):
            ax[j, i].plot(time_vec, trial)
            ax[j, i].axvline(0, color='r', linestyle='--')
    fig.suptitle(f'{car_name} EMG Filt')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{car_name}_emg_env.png'),
                bbox_inches='tight')
    plt.close()

for car_name, car_data in car_group:
    n_digs = car_data.dig_in_num_taste.nunique()
    fig, ax = plt.subplots(max_trials, n_digs,
                           sharex=True, sharey=True,
                           figsize=(n_digs*4, max_trials)
                           )
    for i, (dig_name, dig_data) in enumerate(car_data.groupby('dig_in_name_taste')):
        ax[0, i].set_title(dig_name)
        dat_inds = dig_data.index.values
        dig_filt = flat_emg_filt_data[dat_inds][:, fin_inds[0]:fin_inds[1]]
        for j, trial in enumerate(dig_filt):
            ax[j, i].plot(time_vec, trial)
            ax[j, i].axvline(0, color='r', linestyle='--')
    fig.suptitle(f'{car_name} EMG Filt')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{car_name}_emg_filt.png'),
                bbox_inches='tight')
    plt.close()

# Write successful execution to log
this_pipeline_check.write_to_log(script_path, 'completed')
