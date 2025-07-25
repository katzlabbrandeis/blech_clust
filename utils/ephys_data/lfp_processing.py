"""
lfp_processing.py - LFP extraction and processing utilities

This module provides functions for extracting and processing Local Field Potential (LFP)
data from electrophysiology recordings. Adapted from blech_clust LFP analysis tools.

Key Functions:
    extract_lfps: Main function for LFP extraction and processing
    extract_emgs: Similar processing for EMG signals
    get_filtered_electrode: Apply bandpass filtering to electrode signals
    return_good_lfp_trials: Quality control for LFP trials
    return_good_lfp_trial_inds: Get indices of good quality trials

Features:
    - Automatic trial segmentation based on digital inputs
    - Configurable filtering parameters
    - Data quality visualization
    - EMG-specific processing
    - Trial quality assessment using MAD thresholds

Dependencies:
    - numpy, scipy, tables
    - matplotlib for visualization

EXAMPLE WORKFLOWS:

This module provides functions for extracting and processing LFP data from electrophysiology recordings.
Here are some common usage patterns:

Workflow 1: Basic LFP Extraction
-----------------------------------------------------
from utils.ephys_data import lfp_processing

# Set parameters for LFP extraction
params = {
    'freq_bounds': [1, 300],          # Frequency range in Hz
    'sampling_rate': 30000,           # Original sampling rate
    'taste_signal_choice': 'Start',   # Trial alignment
    'fin_sampling_rate': 1000,        # Final sampling rate
    'dig_in_list': [0, 1, 2, 3],      # Digital inputs to process
    'trial_durations': [2000, 5000]   # Pre/post trial durations in ms
}

# Extract LFPs from raw data files
lfp_processing.extract_lfps(
    dir_name='/path/to/data',
    **params
)

# After extraction, the LFP data is stored in the HDF5 file in the /Parsed_LFP group
# You can then load and analyze it using the ephys_data class

Workflow 2: EMG Extraction
-----------------------------------------------------
from utils.ephys_data import lfp_processing

# Set parameters for EMG extraction
params = {
    'emg_electrode_nums': [0, 1],     # EMG electrode numbers
    'freq_bounds': [10, 500],         # Frequency range in Hz (higher for EMG)
    'sampling_rate': 30000,           # Original sampling rate
    'taste_signal_choice': 'Start',   # Trial alignment
    'fin_sampling_rate': 1000,        # Final sampling rate
    'dig_in_list': [0, 1, 2, 3],      # Digital inputs to process
    'trial_durations': [2000, 5000]   # Pre/post trial durations in ms
}

# Extract EMGs from raw data files
lfp_processing.extract_emgs(
    dir_name='/path/to/data',
    **params
)

Workflow 3: Quality Control for LFP Trials
-----------------------------------------------------
import numpy as np
import tables
import matplotlib.pyplot as plt
from utils.ephys_data import lfp_processing

# Load LFP data from HDF5 file
with tables.open_file('/path/to/data/session.h5', 'r') as hf5:
    # Get LFP data for a specific taste
    lfp_data = hf5.root.Parsed_LFP.dig_in_0_LFPs[:]  # Shape: (channels, trials, time)

# Identify good quality trials
good_trials_bool = lfp_processing.return_good_lfp_trial_inds(
    data=lfp_data,
    MAD_threshold=3  # Number of MADs to use as threshold
)

# Get only the good trials
good_lfp_data = lfp_data[:, good_trials_bool, :]

# Visualize the results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Plot all trials for one channel
channel = 0
ax1.set_title(f'All Trials - Channel {channel}')
for trial in range(lfp_data.shape[1]):
    color = 'blue' if good_trials_bool[trial] else 'red'
    ax1.plot(lfp_data[channel, trial], color=color, alpha=0.5)
ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('Amplitude')

# Plot mean of good vs. bad trials
ax2.set_title('Mean LFP')
ax2.plot(np.mean(lfp_data[:, good_trials_bool, :], axis=1).T, 'b-', label='Good Trials')
ax2.plot(np.mean(lfp_data[:, ~good_trials_bool, :], axis=1).T, 'r-', label='Bad Trials')
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Amplitude')
ax2.legend()

plt.tight_layout()
plt.show()

Usage:
    >>> from utils.ephys_data import lfp_processing
    >>>
    >>> # Extract LFPs with default parameters
    >>> lfp_processing.extract_lfps(
    ...     dir_name='/path/to/data',
    ...     freq_bounds=[1, 300],          # Frequency range in Hz
    ...     sampling_rate=30000,           # Original sampling rate
    ...     taste_signal_choice='Start',   # Trial alignment
    ...     fin_sampling_rate=1000,        # Final sampling rate
    ...     dig_in_list=[0,1,2,3],        # Digital inputs to process
    ...     trial_durations=[2000,5000]    # Pre/post trial durations
    ... )
    >>>
    >>> # Extract EMGs similarly
    >>> lfp_processing.extract_emgs(
    ...     dir_name='/path/to/data',
    ...     emg_electrode_nums=[0,1],      # EMG electrode numbers
    ...     freq_bounds=[1, 300],
    ...     sampling_rate=30000,
    ...     taste_signal_choice='Start',
    ...     fin_sampling_rate=1000,
    ...     dig_in_list=[0,1,2,3],
    ...     trial_durations=[2000,5000]
    ... )
    >>>
    >>> # Filter individual electrode data
    >>> filtered_data = lfp_processing.get_filtered_electrode(
    ...     data=raw_data,
    ...     low_pass=1,
    ...     high_pass=300,
    ...     sampling_rate=1000
    ... )
    >>>
    >>> # Get good quality trials
    >>> good_trials = lfp_processing.return_good_lfp_trials(
    ...     data=lfp_data,
    ...     MAD_threshold=3
    ... )

Installation:
    Required packages can be installed via pip:
    $ pip install numpy scipy tables matplotlib
"""
# ==============================
# Setup
# ==============================

# Import necessary tools
import numpy as np
import tables
import os
import glob
import matplotlib.pyplot as plt
import re
from tqdm import tqdm, trange
import shutil
# Import specific functions in order to filter the data file
from scipy.signal import butter
from scipy.signal import filtfilt
try:
    from scipy.stats import median_abs_deviation as MAD
    try_again = False
except:
    print('Could not import median_abs_deviation, using deprecated version')
    try_again = True

if try_again:
    try:
        from scipy.stats import median_absolute_deviation as MAD
    except:
        raise ImportError('Could not import median_absolute_deviation')

# ==============================
# Define Functions
# ==============================


def get_filtered_electrode(data, low_pass, high_pass, sampling_rate):
    el = 0.195*(data)
    m, n = butter(
        2,
        [2.0*int(low_pass)/sampling_rate, 2.0*int(high_pass)/sampling_rate],
        btype='bandpass')
    filt_el = filtfilt(m, n, el)
    return filt_el

# ==============================
# Collect user input needed for later processing
# ==============================

# call by: extract_lfps(dir_name, **lfp_param_dict)


def extract_lfps(dir_name,
                 freq_bounds,
                 sampling_rate,
                 taste_signal_choice,
                 fin_sampling_rate,
                 dig_in_list,
                 trial_durations,
                 trial_info_frame,
                 ):

    if taste_signal_choice == 'Start':
        # diff_val = 1
        dig_col = 'start_taste_ms'
    elif taste_signal_choice == 'End':
        # diff_val = -1
        dig_col = 'end_taste_ms'

    # ==============================
    # Open HDF5 File
    # ==============================

    # Look for the hdf5 file in the directory
    hdf5_path = glob.glob(os.path.join(dir_name, '**.h5'))[0]

    # Open the hdf5 file
    hf5 = tables.open_file(hdf5_path, 'r+')

    # ==============================
    # Select channels to read
    # ==============================

    # Create vector of electode numbers that have neurons on them
    # (from unit_descriptor table).
    # Some electrodes may record from more than one neuron
    # (shown as repeated number in unit_descriptor);
    # Remove these duplicates within array

    electrodegroup = np.unique(hf5.root.unit_descriptor[:]['electrode_number'])

    # List all appropriate dat files
    Raw_Electrodefiles = np.sort(
        glob.glob(os.path.join(dir_name, '*amp*dat*')))
    Raw_Electrodefiles = Raw_Electrodefiles[electrodegroup]

    # ==============================
    # Extract Raw Data
    # ==============================

    # Check if LFP data is already within file and remove node if so.
    # Create new raw LFP group within H5 file.
    if '/raw_LFP' in hf5:
        hf5.remove_node('/raw_LFP', recursive=True)
    hf5.create_group('/', 'raw_LFP')

    # Loop through each neuron-recording electrode (from .dat files),
    # filter data, and create array in new LFP node

    # How many folds to downsample data by
    new_intersample_interval = sampling_rate/fin_sampling_rate

    # Pull out signal for each electrode, down_sample,
    # bandpass filter and store in HDF5
    print('Extracting raw LFPs')
    for i in trange(len(Raw_Electrodefiles)):
        data = np.fromfile(Raw_Electrodefiles[i], dtype=np.dtype('int16'))
        remaining_inds = int(data.shape[0] % new_intersample_interval)
        if remaining_inds:
            data = data[:-remaining_inds]
        data_down = np.mean(
            data.reshape((-1, int(new_intersample_interval))), axis=-1)
        filt_el_down = get_filtered_electrode(data=data_down,
                                              low_pass=freq_bounds[0],
                                              high_pass=freq_bounds[1],
                                              sampling_rate=fin_sampling_rate)

        # Zero padding to 3 digits because code get screwy with sorting electrodes
        # if that isn't done
        hf5.create_array('/raw_LFP', 'electrode{:0>3}'.
                         format(electrodegroup[i]), filt_el_down)
        hf5.flush()
        del data, data_down, filt_el_down

    # ==============================
    # Write-Out Extracted LFP
    # ==============================

    # Grab the names of the arrays containing LFP recordings
    lfp_nodes = hf5.list_nodes('/raw_LFP')

    # Make the Parsed_LFP node in the hdf5 file if it doesn't exist, else move on
    if '/Parsed_LFP' in hf5:
        hf5.remove_node('/Parsed_LFP', recursive=True)
    hf5.create_group('/', 'Parsed_LFP')

    # Create array marking which channel were chosen for further analysis
    # Made in root folder for backward compatibility of code
    # Code further below simply enumerates arrays in Parsed_LFP
    if "/Parsed_LFP_channels" in hf5:
        hf5.remove_node('/Parsed_LFP_channels')
    hf5.create_array('/', 'Parsed_LFP_channels', electrodegroup)
    hf5.flush()

    # Make markers to slice trials for every dig_on
    trial_info_frame['dig_in_ind'] = trial_info_frame['dig_in_num_taste'].rank(
        method='dense') - 1
    change_points_fin = [
        np.vectorize(int)(x[dig_col].values) for _, x in trial_info_frame.groupby('dig_in_ind')
    ]
    all_trial_markers = [[(x-trial_durations[0], x+trial_durations[1])
                          for x in this_dig_in_markers]
                         for this_dig_in_markers in change_points_fin]

    # Cut off dig-inds by lowest number of trials
    trial_counts = [x.shape[0] for x in change_points_fin]
    # min_trial_counts = np.min(trial_counts)
    max_trial_counts = np.max(trial_counts)
    # all_trial_markers = [np.array(x)[:min_trial_counts, :]
    #                      for x in all_trial_markers]

    # Extract trials for every channel for every dig_in
    print('Parsing LFPs')
    all_channel_trials = []
    for channel in tqdm(lfp_nodes):
        this_channel_trials = [
            np.asarray([channel[marker_tuple[0]:marker_tuple[1]]
                        for marker_tuple in this_dig_in])
            for this_dig_in in all_trial_markers
        ]
        all_channel_trials.append(this_channel_trials)

    # Resort data to have 4 arrays (one for every dig_in)
    # with dims (channels , trials, time)
    for dig_in in dig_in_list:
        this_taste_LFP = np.asarray([
            channel[dig_in] for channel in all_channel_trials])

        # Put the LFP data for this taste in hdf5 file under /Parsed_LFP
        hf5.create_array('/Parsed_LFP', 'dig_in_%i_LFPs'
                         % (dig_in), this_taste_LFP)
        hf5.flush()

    # Delete data
    hf5.remove_node('/raw_LFP', recursive=True)
    hf5.flush()

    # ================================================
    # Make plots to visually check quality of channels
    # ================================================

    # Code copied from LFP_Spectrogram_Stone.py
    # Might need cleanup

    dig_in_channels = hf5.list_nodes('/digital_in')
    dig_in_LFP_nodes = hf5.list_nodes('/Parsed_LFP')

    # Create dictionary of all parsed LFP arrays
    LFP_data = [np.array(dig_in_LFP_nodes[node][:])
                for node in range(len(dig_in_LFP_nodes))]

    # =============================================================================
    # #Channel Check
    # =============================================================================
    # Make directory to store the LFP trace plots.
    # Delete and remake the directory if it exists
    channel_check_dir = os.path.join(dir_name, 'LFP_channel_check')
    if os.path.exists(channel_check_dir):
        shutil.rmtree(channel_check_dir)
    # try:
    #        os.system('rm -r '+'./LFP_channel_check')
    # except:
    #        pass
    os.mkdir(channel_check_dir)
    hdf5_name = os.path.basename(hdf5_path)

    # Check to make sure LFPs are "normal" and allow user to remove any that are not
    ########################################
    # Channel check plots are now made automatically (Abu 2/3/19)
    ########################################
    # if subplot_check is "Yes":
    for taste in range(len(LFP_data)):

        # Set data
        channel_data = np.mean(LFP_data[taste], axis=1).T
        t = np.array(list(range(0, np.size(channel_data, axis=0))))

        mean_val = np.mean(channel_data.flatten())
        std_val = np.std(channel_data.flatten())
        # Create figure
        fig, axes = plt.subplots(nrows=np.size(channel_data, axis=1),
                                 ncols=1, sharex=True, sharey=True, figsize=(12, 8), squeeze=False)
        fig.text(0.5, 0.05, 'Milliseconds', ha='center', fontsize=15)
        axes_list = [item for sublist in axes for item in sublist]

        for ax, chan in zip(axes.flatten(), range(np.size(channel_data, axis=1))):

            ax = axes_list.pop(0)
            ax.set_yticks([])
            ax.plot(np.squeeze(t), np.squeeze(channel_data[:, chan]))
            ax.set_ylim([mean_val - 3*std_val, mean_val + 3*std_val])
            h = ax.set_ylabel('Channel %s' % (chan))
            h.set_rotation(0)
            ax.vlines(x=trial_durations[0], ymin=np.min(channel_data[:, chan]),
                      ymax=np.max(channel_data[:, chan]), linewidth=4, color='r')

        fig.subplots_adjust(hspace=0, wspace=-0.15)
        fig.suptitle('Dig in {} - '.format(taste) +
                     '%s - Channel Check: %s' % (taste,
                                                 hdf5_name[0:4])+'\n' + 'Raw LFP Traces; Date: %s'
                     % (re.findall(r'_(\d{6})',
                                   hdf5_name)[0]), size=16, fontweight='bold')
        fig.savefig(
            os.path.join(channel_check_dir,
                         hdf5_name[0:4] +
                         '_dig_in{}'.format(taste) +
                         '_ %s_%s' % (re.findall(r'_(\d{6})', hdf5_name)[0],
                                      taste) + '_channelcheck.png'))

    # ==============================
    # Close Out
    # ==============================
    print("If you want to compress the file to release disk space, " +
          "run 'blech_hdf5_repack.py' upon completion.")
    hf5.flush()
    hf5.close()


def extract_emgs(dir_name,
                 emg_electrode_nums,
                 freq_bounds,
                 sampling_rate,
                 taste_signal_choice,
                 fin_sampling_rate,
                 dig_in_list,
                 trial_durations):
    """Extract EMG data from raw recordings

    Args:
        dir_name: Directory containing data files
        emg_electrode_nums: List of electrode numbers for EMG channels
        freq_bounds: [low, high] frequency bounds for filtering
        sampling_rate: Original sampling rate
        taste_signal_choice: 'Start' or 'End' for trial alignment
        fin_sampling_rate: Final sampling rate after downsampling
        dig_in_list: List of digital input channels to process
        trial_durations: [pre_trial, post_trial] durations in ms
    """

    if taste_signal_choice == 'Start':
        diff_val = 1
    elif taste_signal_choice == 'End':
        diff_val = -1

    # ==============================
    # Open HDF5 File
    # ==============================

    # Look for the hdf5 file in the directory
    hdf5_path = glob.glob(os.path.join(dir_name, '**.h5'))[0]

    # Open the hdf5 file
    hf5 = tables.open_file(hdf5_path, 'r+')

    # ==============================
    # Select channels to read
    # ==============================

    # Create vector of electode numbers that have neurons on them
    # (from unit_descriptor table).
    # Some electrodes may record from more than one neuron
    # (shown as repeated number in unit_descriptor);
    # Remove these duplicates within array

    # List all appropriate dat files
    Raw_Electrodefiles = np.sort(
        glob.glob(os.path.join(dir_name, '*amp*dat*')))
    Raw_Electrodefiles = [Raw_Electrodefiles[x] for x in emg_electrode_nums]

    # ==============================
    # Extract Raw Data
    # ==============================

    # Check if LFP data is already within file and remove node if so.
    # Create new raw LFP group within H5 file.
    if '/raw_emg' in hf5:
        hf5.remove_node('/raw_emg', recursive=True)
    hf5.create_group('/', 'raw_emg')

    # Loop through each neuron-recording electrode (from .dat files),
    # filter data, and create array in new LFP node

    # How many folds to downsample data by
    new_intersample_interval = sampling_rate/fin_sampling_rate

    # Pull out signal for each electrode, down_sample,
    # bandpass filter and store in HDF5
    print('Extracting raw EMGs')
    for i in trange(len(Raw_Electrodefiles)):
        data = np.fromfile(Raw_Electrodefiles[i], dtype=np.dtype('int16'))
        data_down = np.mean(
            data.reshape((-1, int(new_intersample_interval))), axis=-1)
        filt_el_down = get_filtered_electrode(data=data_down,
                                              low_pass=freq_bounds[0],
                                              high_pass=freq_bounds[1],
                                              sampling_rate=fin_sampling_rate)

        # Zero padding to 3 digits because code get screwy with sorting electrodes
        # if that isn't done
        hf5.create_array('/raw_emg', 'electrode{:0>3}'.
                         format(emg_electrode_nums[i]), filt_el_down)
        hf5.flush()
        del data, data_down, filt_el_down

    # Grab the names of the arrays containing digital inputs,
    # and pull the data into a numpy array
    dig_in_nodes = hf5.list_nodes('/digital_in')
    dig_in = []
    dig_in_pathname = []
    for node in dig_in_nodes:
        dig_in_pathname.append(node._v_pathname)
        exec("dig_in.append(hf5.root.digital_in.%s[:])"
             % dig_in_pathname[-1].split('/')[-1])
    dig_in = np.array(dig_in)

    # The tail end of the pulse generates a negative value when passed through diff
    # This method removes the need for a "for" loop

    diff_points = list(np.where(np.diff(dig_in) == diff_val))
    diff_points[1] = diff_points[1]//new_intersample_interval
    change_points = [diff_points[1][diff_points[0] == this_dig_in]
                     for this_dig_in in range(len(dig_in))]

    # ==============================
    # Write-Out Extracted LFP
    # ==============================

    # Grab the names of the arrays containing LFP recordings
    lfp_nodes = hf5.list_nodes('/raw_emg')

    # Make the Parsed_LFP node in the hdf5 file if it doesn't exist, else move on
    if '/Parsed_emg' in hf5:
        hf5.remove_node('/Parsed_emg', recursive=True)
    hf5.create_group('/', 'Parsed_emg')

    # Create array marking which channel were chosen for further analysis
    # Made in root folder for backward compatibility of code
    # Code further below simply enumerates arrays in Parsed_LFP
    if "/Parsed_emg_channels" in hf5:
        hf5.remove_node('/Parsed_emg_channels')
    hf5.create_array('/', 'Parsed_emg_channels', emg_electrode_nums)
    hf5.flush()

    # Remove dig_ins which are not relevant
    change_points_fin = [change_points[x] for x in range(len(change_points))
                         if x in dig_in_list]

    # Make markers to slice trials for every dig_on
    all_trial_markers = [[(x-trial_durations[0], x+trial_durations[1])
                          for x in this_dig_in_markers]
                         for this_dig_in_markers in change_points_fin]

    # Extract trials for every channel for every dig_in
    print('Parsing EMGs')
    all_channel_trials = []
    for channel in tqdm(lfp_nodes):
        this_channel_trials = [
            np.asarray([channel[marker_tuple[0]:marker_tuple[1]]
                        for marker_tuple in this_dig_in])
            for this_dig_in in all_trial_markers
        ]
        all_channel_trials.append(this_channel_trials)

    # Resort data to have 4 arrays (one for every dig_in)
    # with dims (channels , trials, time)
    for dig_in in dig_in_list:
        this_taste_LFP = np.asarray([
            channel[dig_in] for channel in all_channel_trials])

        # Put the LFP data for this taste in hdf5 file under /Parsed_LFP
        hf5.create_array('/Parsed_emg', 'dig_in_%i_emg'
                         % (dig_in), this_taste_LFP)
        hf5.flush()

    # Delete data
    hf5.remove_node('/raw_emg', recursive=True)
    hf5.flush()

    # ================================================
    # Make plots to visually check quality of channels
    # ================================================

    # Code copied from LFP_Spectrogram_Stone.py
    # Might need cleanup

    dig_in_channels = hf5.list_nodes('/digital_in')
    dig_in_LFP_nodes = hf5.list_nodes('/Parsed_emg')

    # Create dictionary of all parsed LFP arrays
    LFP_data = [np.array(dig_in_LFP_nodes[node][:])
                for node in range(len(dig_in_LFP_nodes))]

    # =============================================================================
    # #Channel Check
    # =============================================================================
    # Make directory to store the LFP trace plots.
    # Delete and remake the directory if it exists
    channel_check_dir = os.path.join(dir_name, 'emg_channel_check')
    if os.path.exists(channel_check_dir):
        shutil.rmtree(channel_check_dir)
    # try:
    #        os.system('rm -r '+'./LFP_channel_check')
    # except:
    #        pass
    os.mkdir(channel_check_dir)
    hdf5_name = os.path.basename(hdf5_path)

    # Check to make sure LFPs are "normal" and allow user to remove any that are not
    ########################################
    # Channel check plots are now made automatically (Abu 2/3/19)
    ########################################
    # if subplot_check is "Yes":
    for taste in range(len(LFP_data)):

        # Set data
        channel_data = np.mean(LFP_data[taste], axis=1).T
        t = np.array(list(range(0, np.size(channel_data, axis=0))))

        mean_val = np.mean(channel_data.flatten())
        std_val = np.std(channel_data.flatten())
        # Create figure
        fig, axes = plt.subplots(nrows=np.size(channel_data, axis=1),
                                 ncols=1, sharex=True, sharey=True, figsize=(12, 8), squeeze=False)
        fig.text(0.5, 0.05, 'Milliseconds', ha='center', fontsize=15)
        axes_list = [item for sublist in axes for item in sublist]

        for ax, chan in zip(axes.flatten(), range(np.size(channel_data, axis=1))):

            ax = axes_list.pop(0)
            ax.set_yticks([])
            ax.plot(np.squeeze(t), np.squeeze(channel_data[:, chan]))
            ax.set_ylim([mean_val - 3*std_val, mean_val + 3*std_val])
            h = ax.set_ylabel('Channel %s' % (chan))
            h.set_rotation(0)
            ax.vlines(x=trial_durations[0], ymin=np.min(channel_data[:, chan]),
                      ymax=np.max(channel_data[:, chan]), linewidth=4, color='r')

        fig.subplots_adjust(hspace=0, wspace=-0.15)
        fig.suptitle('Dig in {} - '.format(taste) +
                     '%s - Channel Check: %s' % (taste,
                                                 hdf5_name[0:4])+'\n' + 'Raw LFP Traces; Date: %s'
                     % (re.findall(r'_(\d{6})',
                                   hdf5_name)[0]), size=16, fontweight='bold')
        fig.savefig(
            os.path.join(channel_check_dir,
                         hdf5_name[0:4] +
                         '_dig_in{}'.format(taste) +
                         '_ %s_%s' % (re.findall(r'_(\d{6})', hdf5_name)[0],
                                      taste) + '_channelcheck.png'))

    # ==============================
    # Close Out
    # ==============================
    print("If you want to compress the file to release disk space, " +
          "run 'blech_hdf5_repack.py' upon completion.")
    hf5.flush()
    hf5.close()


def return_good_lfp_trial_inds(data, MAD_threshold=3):
    """
    Return boolean array of good trials (for all channels) based on MAD threshold
    Remove trials based on deviation from median LFP per trial

    Inputs:
        data : shape (n_channels, n_trials, n_timepoints)
        MAD_threshold : number of MADs to use as threshold for individual timepoints
        deviation_threshold : number of MADs to use as threshold for trial deviation
                             (defaults to MAD_threshold if None)

    Outputs:
        good_trials_bool : boolean array of good trials
    """
    lfp_median = np.median(data, axis=1)
    lfp_MAD = MAD(data, axis=1)
    # Use total deviation per trial scaled by MAD to remove trial
    mean_trial_deviation = np.mean(
        np.abs(data - lfp_median[:, np.newaxis, :])/lfp_MAD[:, None], axis=2)
    deviation_median = np.median(mean_trial_deviation, axis=1)
    deviation_MAD = MAD(mean_trial_deviation, axis=1)
    fin_deviation_threshold = deviation_median + MAD_threshold*deviation_MAD
    # Remove trials with high deviation
    good_trials_bool = mean_trial_deviation < fin_deviation_threshold[:, np.newaxis]
    # Take only trials good for both regions
    good_trials_bool = np.all(good_trials_bool, axis=0)
    return good_trials_bool


def return_good_lfp_trials(data, MAD_threshold=3):
    """Return good trials (for all channels) based on MAD threshold
        data : shape (n_channels, n_trials, n_timepoints)
        MAD_threshold : number of MADs to use as threshold for individual timepoints
    """
    good_trials_bool = return_good_lfp_trial_inds(
        data, MAD_threshold)
    return data[:, good_trials_bool]
