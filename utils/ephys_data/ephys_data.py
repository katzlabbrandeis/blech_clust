"""
This module provides a class for streamlined electrophysiology data analysis, focusing on handling and analyzing data from multiple files. It includes features for automatic data loading, spike train and LFP data processing, firing rate calculation, digital input parsing, trial segmentation, region-based analysis, laser condition handling, and data quality checks.

EXAMPLE WORKFLOWS:

This class provides a comprehensive interface for analyzing electrophysiology data.
Below are examples of common analysis workflows:

Workflow 1: Basic Data Loading and Processing
-----------------------------------------------------
from blech_clust.utils.ephys_data.ephys_data import ephys_data

# Initialize with data directory
data = ephys_data(data_dir='/path/to/data')

# Load and process data
data.get_unit_descriptors()  # Get unit information
data.get_spikes()            # Extract spike data
data.get_firing_rates()      # Calculate firing rates
data.get_lfps()              # Extract LFP data

# Access processed data
spikes = data.spikes         # Access spike data
firing = data.firing_array   # Access firing rate data
lfps = data.lfp_array        # Access LFP data

Workflow 2: Region-Based Analysis
-----------------------------------------------------
from blech_clust.utils.ephys_data.ephys_data import ephys_data
import matplotlib.pyplot as plt

# Initialize and load data
data = ephys_data(data_dir='/path/to/data')
data.extract_and_process()   # Extract and process all data

# Get region information
data.get_region_units()      # Get units by brain region

# Analyze specific brain regions
for region in data.region_names:
    # Get spikes for this region
    region_spikes = data.return_region_spikes(region)

    # Get firing rates for this region
    region_firing = data.get_region_firing(region)

    # Get LFPs for this region
    region_lfps, _ = data.return_region_lfps()

    # Example: Plot mean firing rate for this region
    if region_firing is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(np.mean(region_firing, axis=(0, 1)))
        plt.title(f'Mean Firing Rate - {region}')
        plt.xlabel('Time (bins)')
        plt.ylabel('Firing Rate (Hz)')
        plt.show()

Workflow 3: Laser Condition Analysis
-----------------------------------------------------
from blech_clust.utils.ephys_data.ephys_data import ephys_data
import matplotlib.pyplot as plt
import numpy as np

# Initialize and load data
data = ephys_data(data_dir='/path/to/data')
data.extract_and_process()   # Extract and process all data

# Check if laser trials exist
data.check_laser()

if data.laser_exists:
    # Separate data by laser condition
    data.separate_laser_data()

    # Compare firing rates between laser conditions
    on_firing = data.all_on_firing   # Laser on trials
    off_firing = data.all_off_firing # Laser off trials

    # Example: Plot mean firing rate comparison
    plt.figure(figsize=(12, 6))

    # Calculate mean across trials and neurons
    mean_on = np.mean(on_firing, axis=(0, 1))
    mean_off = np.mean(off_firing, axis=(0, 1))

    plt.plot(mean_on, 'r-', label='Laser On')
    plt.plot(mean_off, 'b-', label='Laser Off')
    plt.title('Mean Firing Rate Comparison')
    plt.xlabel('Time (bins)')
    plt.ylabel('Firing Rate (Hz)')
    plt.legend()
    plt.show()

Workflow 4: Palatability Analysis
-----------------------------------------------------
from blech_clust.utils.ephys_data.ephys_data import ephys_data
import matplotlib.pyplot as plt
import numpy as np

# Initialize and load data
data = ephys_data(data_dir='/path/to/data')
data.extract_and_process()   # Extract and process all data

# Calculate palatability correlation
data.calc_palatability()

# Plot palatability correlation over time
plt.figure(figsize=(10, 6))
plt.imshow(data.pal_array, aspect='auto', cmap='viridis')
plt.colorbar(label='|Palatability Correlation|')
plt.xlabel('Time (bins)')
plt.ylabel('Neuron')
plt.title('Palatability Correlation Over Time')
plt.show()

# Find neurons with strong palatability coding
strong_pal_neurons = np.where(np.max(data.pal_array, axis=1) > 0.7)[0]
print(f"Neurons with strong palatability coding: {strong_pal_neurons}")

Workflow 5: Time-Frequency Analysis
-----------------------------------------------------
from blech_clust.utils.ephys_data.ephys_data import ephys_data
import matplotlib.pyplot as plt

# Initialize and load data
data = ephys_data(data_dir='/path/to/data')
data.get_lfps()  # Extract LFP data

# Set STFT parameters
data.stft_params = {
    'Fs': 1000,
    'signal_window': 500,
    'window_overlap': 499,
    'max_freq': 100,
    'time_range_tuple': (0, 5)
}

# Calculate STFT
data.get_stft(recalculate=True, dat_type=['amplitude', 'phase'])

# Plot STFT amplitude for a specific channel and trial
taste = 0
channel = 0
trial = 0

plt.figure(figsize=(12, 8))
plt.pcolormesh(data.time_vec, data.freq_vec,
              data.amplitude_array[taste, channel, trial],
              shading='gouraud', cmap='viridis')
plt.colorbar(label='Power')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title(f'STFT Amplitude - Taste {taste}, Channel {channel}, Trial {trial}')
plt.show()

- `ephys_data`: Main class for data handling and analysis.
  - `__init__`: Initializes the class with optional data directory.
  - `calc_stft`: Computes the Short-Time Fourier Transform (STFT) of a trial.
  - `parallelize`: Utilizes parallel processing to apply a function over an iterator.
  - `_calc_conv_rates`: Calculates firing rates using a convolution method.
  - `_calc_baks_rate`: Calculates firing rates using Bayesian Adaptive Kernel Smoother (BAKS).
  - `get_hdf5_path`: Finds the HDF5 file in the specified directory.
  - `convert_to_array`: Converts a list to a numpy array.
  - `remove_node`: Removes a node from an HDF5 file.
  - `extract_and_process`: Extracts and processes unit descriptors, spikes, firing rates, and LFPs.
  - `separate_laser_data`: Separates data into laser on and off conditions.
  - `get_unit_descriptors`: Extracts unit descriptors from an HDF5 file.
  - `check_laser`: Checks for the presence of laser trials.
  - `get_spikes`: Extracts spike arrays from HDF5 files.
  - `separate_laser_spikes`: Separates spike arrays into laser on and off conditions.
  - `extract_lfps`: Extracts LFPs from raw data files and saves them to HDF5.
  - `get_lfp_channels`: Extracts parsed LFP channels.
  - `get_lfps`: Initiates LFP extraction or retrieves LFP arrays from HDF5.
  - `separate_laser_lfp`: Separates LFP arrays into laser on and off conditions.
  - `firing_rate_method_selector`: Selects the method for firing rate calculation.
  - `get_firing_rates`: Converts spikes to firing rates.
  - `calc_palatability`: Calculates single neuron palatability from firing rates.
  - `separate_laser_firing`: Separates firing rates into laser on and off conditions.
  - `get_info_dict`: Loads information from a JSON file.
  - `get_region_electrodes`: Extracts electrodes for each region from a JSON file.
  - `get_region_units`: Extracts indices of units by region of electrodes.
  - `return_region_spikes`: Returns spikes for a specified brain region.
  - `get_region_firing`: Returns firing rates for a specified brain region.
  - `get_lfp_electrodes`: Extracts indices of LFP electrodes by region.
  - `get_stft`: Retrieves or calculates STFT and saves it to HDF5.
  - `return_region_lfps`: Returns LFPs for each region and region names.
  - `return_representative_lfp_channels`: Returns one electrode per region closest to the mean.
  - `get_mean_stft_amplitude`: Calculates the mean STFT amplitude for each region.
  - `get_trial_info_frame`: Loads trial information from a CSV file.
  - `sequester_trial_inds`: Sequesters trials into categories based on tastes and laser conditions.
  - `get_sequestered_spikes`: Sequesters spikes into categories based on tastes and laser conditions.
  - `get_sequestered_firing`: Sequesters firing rates into categories based on tastes and laser conditions.
  - `get_sequestered_data`: Sequesters both spikes and firing rates into categories based on tastes and laser conditions.
  - `get_stable_units`: Loads drift check results and marks units as stable or unstable.
  - `profile_units`: Generates a DataFrame with unit characteristics (responsiveness, discriminability, palatability, dynamicity, stability).
"""
import os
import warnings
import numpy as np
import tables
import copy
import multiprocessing as mp
from scipy.special import gamma
from scipy.stats import zscore, spearmanr, ttest_rel
import scipy
import scipy.signal
import glob
import json
import easygui
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm
from itertools import product
from .BAKS import BAKS
from . import lfp_processing
from pprint import pprint as pp
import numpy as np
import pandas as pd
import pingouin as pg

#  ______       _                      _____        _
# |  ____|     | |                    |  __ \      | |
# | |__   _ __ | |__  _   _ ___ ______| |  | | __ _| |_ __ _
# |  __| | '_ \| '_ \| | | / __|______| |  | |/ _` | __/ _` |
# | |____| |_) | | | | |_| \__ \      | |__| | (_| | || (_| |
# |______| .__/|_| |_|\__, |___/      |_____/ \__,_|\__\__,_|
#        | |           __/ |
#        |_|          |___/

"""
ephys_data.py - Class for streamlined electrophysiology data analysis

This module provides a class for analyzing electrophysiology data from multiple files.
The class provides containers and functions for data analysis with automatic loading
capabilities.

Key Features:
    - Automatic data loading from specified files
    - Spike train and LFP data processing
    - Firing rate calculation with multiple methods
    - Digital input parsing and trial segmentation
    - Region-based analysis for multi-region recordings
    - Laser condition handling for optogenetic experiments
    - Data quality checks and visualization

Classes:
    ephys_data: Main class for data handling and analysis

Dependencies:
    - numpy, scipy, tables, pandas
    - BAKS (Bayesian Adaptive Kernel Smoother)
    - Custom LFP processing utilities

Usage:
    >>> from utils.ephys_data.ephys_data import ephys_data
    >>> # Initialize with data directory
    >>> data = ephys_data(data_dir='/path/to/data')
    >>>
    >>> # Load and process data
    >>> data.get_unit_descriptors()  # Get unit information
    >>> data.get_spikes()           # Extract spike data
    >>> data.get_firing_rates()     # Calculate firing rates
    >>> data.get_lfps()            # Extract LFP data
    >>>
    >>> # Access processed data
    >>> spikes = data.spikes       # Access spike data
    >>> firing = data.firing_array # Access firing rate data
    >>> lfps = data.lfp_array     # Access LFP data
    >>>
    >>> # Region-based analysis
    >>> data.get_region_units()    # Get units by brain region
    >>> region_spikes = data.return_region_spikes('region_name')
    >>>
    >>> # Handle laser conditions (if present)
    >>> data.check_laser()         # Check for laser trials
    >>> data.separate_laser_data() # Split data by laser condition

Installation:
    Required packages can be installed via pip:
    $ pip install numpy scipy tables pandas tqdm matplotlib
"""


class ephys_data():

    ######################
    # Define static methods
    #####################

    @staticmethod
    def calc_stft(
        trial,
        max_freq,
        time_range_tuple,
        Fs,
        signal_window,
        window_overlap
    ):
        """Compute Short-Time Fourier Transform (STFT) of a trial

        Args:
            trial: 1D array of signal data
            max_freq: Maximum frequency to include in Hz (frequencies above are truncated)
            time_range_tuple: (start, end) time limits in seconds for spectrogram
            Fs: Sampling rate in Hz
            signal_window: Window size for spectrogram in samples
            window_overlap: Overlap between windows in samples

        Returns:
            tuple: (fin_freq, fin_t, this_stft)
                - fin_freq: Frequency vector (Hz) after truncation
                - fin_t: Time vector (seconds) after truncation
                - this_stft: Complex STFT array, shape (n_freqs, n_times)
        """
        f, t, this_stft = scipy.signal.stft(
            scipy.signal.detrend(trial),
            fs=Fs,
            window='hann',
            nperseg=signal_window,
            noverlap=signal_window-(signal_window-window_overlap))
        this_stft = this_stft[np.where(f < max_freq)[0]]
        this_stft = this_stft[:, np.where((t >= time_range_tuple[0]) *
                                          (t < time_range_tuple[1]))[0]]
        fin_freq = f[f < max_freq]
        fin_t = t[np.where((t >= time_range_tuple[0])
                           * (t < time_range_tuple[1]))]
        return fin_freq, fin_t, this_stft

    # Calculate absolute and phase
    @staticmethod
    def parallelize(func, iterator):
        """Execute function in parallel across iterator using joblib

        Args:
            func: Function to apply to each element
            iterator: Iterable to process in parallel

        Returns:
            List of results from applying func to each element
        """
        return Parallel(n_jobs=mp.cpu_count()-2)(delayed(func)(this_iter) for this_iter in tqdm(iterator))

    @staticmethod
    def _calc_conv_rates(step_size, window_size, dt, spike_array):
        """Calculate firing rates using convolution with moving window

        Args:
            step_size: Step size in milliseconds for moving window
            window_size: Window size in milliseconds for firing rate calculation
            dt: Inter-sample interval in milliseconds
            spike_array: N-D array with time as last dimension, binary spike data

        Returns:
            tuple: (firing_rate, time_vector)
                - firing_rate: Calculated firing rates, shape (*spike_array.shape[:-1], n_bins)
                - time_vector: Time vector in ms relative to stimulus delivery

        Raises:
            Exception: If step_size or window_size are not integer multiples of dt
        """

        if np.sum([step_size % dt, window_size % dt]) > 1e-14:
            raise Exception('Step size and window size must be integer multiples'
                            ' of the inter-sample interval')

        fin_step_size, fin_window_size = \
            int(step_size/dt), int(window_size/dt)
        total_time = spike_array.shape[-1]

        bin_inds = (0, fin_window_size)
        total_bins = int(
            (total_time - fin_window_size + 1) / fin_step_size) + 1
        bin_list = [(bin_inds[0]+step, bin_inds[1]+step)
                    for step in np.arange(total_bins)*fin_step_size]

        firing_rate = np.empty((spike_array.shape[0],
                                spike_array.shape[1], total_bins))

        for bin_inds in bin_list:
            firing_rate[..., bin_inds[0]//fin_step_size] = \
                np.sum(spike_array[..., bin_inds[0]:bin_inds[1]], axis=-1)

        # Calculate time vector relative to stimulus delivery
        # Center of each bin in milliseconds
        time_vector = np.array([x[0] for x in bin_list]) * dt + \
            (fin_window_size//2) * dt

        return firing_rate, time_vector

    @staticmethod
    def _calc_baks_rate(resolution, dt, spike_array):
        """Calculate firing rates using Bayesian Adaptive Kernel Smoother (BAKS)

        Args:
            resolution: Resolution of output firing rate in seconds
            dt: Resolution of input spike trains in seconds
            spike_array: N-D array with time as last dimension, binary spike data

        Returns:
            tuple: (firing_rate_array, time_vector)
                - firing_rate_array: Calculated firing rates, shape (*spike_array.shape[:-1], n_time_points)
                - time_vector: Time vector in seconds relative to stimulus delivery
        """
        t = np.arange(0, spike_array.shape[-1]*dt, resolution)
        array_inds = list(np.ndindex((spike_array.shape[:-1])))
        spike_times = [np.where(spike_array[this_inds])[0]*dt
                       for this_inds in array_inds]

        firing_rates = [BAKS(this_spike_times, t)
                        for this_spike_times in tqdm(spike_times)]

        # Put back into array
        firing_rate_array = np.zeros((*spike_array.shape[:-1], len(t)))
        for this_inds, this_firing in zip(array_inds, firing_rates):
            firing_rate_array[this_inds] = this_firing

        # Time vector is already calculated as t (in seconds)
        time_vector = t

        return firing_rate_array, time_vector

    @staticmethod
    def get_hdf5_path(data_dir):
        """Find HDF5 file in the specified directory

        Args:
            data_dir: Directory path to search for HDF5 file

        Returns:
            str: Path to HDF5 file

        Raises:
            Exception: If no HDF5 file found or if multiple files found (prompts user selection)
        """
        hdf5_path = glob.glob(
            os.path.join(data_dir, '**.h5'))
        if not len(hdf5_path) > 0:
            raise Exception('No HDF5 file detected' +
                            f'Looking in {data_dir}')
        elif len(hdf5_path) > 1:
            selection_list = ['{}) {} \n'.format(num, os.path.basename(file))
                              for num, file in enumerate(hdf5_path)]
            selection_string = \
                'Multiple HDF5 files detected, please select a number:\n{}'.\
                format("".join(selection_list))
            file_selection = input(selection_string)
            return hdf5_path[int(file_selection)]
        else:
            return hdf5_path[0]

    # Convert list to array
    @staticmethod
    def convert_to_array(iterator, iter_inds):
        """Convert list of arrays to a single multi-dimensional array

        Args:
            iterator: List of arrays to combine
            iter_inds: List of index tuples indicating where each array should be placed

        Returns:
            temp_array: Combined multi-dimensional numpy array
        """
        temp_array =\
            np.empty(
                tuple((*(np.max(np.array(iter_inds), axis=0) + 1),
                       *iterator[0].shape)),
                dtype=np.dtype(iterator[0].flatten()[0]))
        for iter_num, this_iter in enumerate(tqdm(iter_inds)):
            temp_array[this_iter] = iterator[iter_num]
        return temp_array

    @staticmethod
    def remove_node(path_to_node, hf5):
        """Remove a node from HDF5 file if it exists

        Args:
            path_to_node: Full path to the node in HDF5 file
            hf5: Open HDF5 file handle
        """
        if path_to_node in hf5:
            hf5.remove_node(
                os.path.dirname(path_to_node), os.path.basename(path_to_node))

    ####################
    # Initialize instance
    ###################

    def __init__(self,
                 data_dir=None):
        """Initialize ephys_data object for electrophysiology data analysis

        Args:
            data_dir: Directory path containing HDF5 file and data
                     If None, opens dialog for user to select directory

        Attributes Created:
            data_dir: Path to data directory
            hdf5_path: Full path to HDF5 file
            hdf5_name: Basename of HDF5 file
            firing_rate_params: Dictionary of firing rate calculation parameters
            lfp_params: Dictionary of LFP extraction parameters
            default_firing_params: Default firing rate parameters
            default_lfp_params: Default LFP parameters
            stft_params: STFT calculation parameters
        """
        self.lfp_processing = lfp_processing

        if data_dir is None:
            self.data_dir = easygui.diropenbox(
                'Please select directory with HDF5 file')
        else:
            self.data_dir = data_dir
            self.hdf5_path = self.get_hdf5_path(data_dir)
            self.hdf5_name = os.path.basename(self.hdf5_path)

            # self.spikes = None

        # Create environemnt variable to allow program to know
        # if file is currently accessed
        # Created for multiprocessing of fits
        # os.environ[self.hdf5_name]="0"

        self.firing_rate_params = {
            'type':   None,
            'step_size':   None,
            'window_size':   None,
            'dt':   None,
            'baks_resolution': None,
            'baks_dt': None
        }

        self.lfp_params = {
            'freq_bounds': None,
            'sampling_rate': None,
            'taste_signal_choice': None,
            'fin_sampling_rate': None,
            'trial_durations': None
        }

        self.default_firing_params = {
            'type':   'conv',
            'step_size':   25,
            'window_size':   250,
            'dt':   1,
            'baks_resolution': 25e-3,
            'baks_dt':   1e-3
        }

        self.default_lfp_params = {
            'freq_bounds': [1, 300],
            'sampling_rate': 30000,
            'taste_signal_choice': 'Start',
            'fin_sampling_rate': 1000,
            'trial_durations': [2000, 5000]
        }

        # Resolution has to be increased for phase of higher frequencies
        # Can be passed as kwargs to "calc_stft"
        self.stft_params = {
            'Fs': 1000,
            'signal_window': 500,
            'window_overlap': 499,
            'max_freq': 20,
            'time_range_tuple': (0, 5)
        }

    # class access:
    #    def __init__(self, key_name):
    #        os.environ[key_name] = '0'

    #    def check(self):
    #        access_bool =

    def extract_and_process(self):
        """Extract and process all data types (units, spikes, firing rates, LFPs)

        Convenience method that calls all extraction methods in sequence.

        Side Effects:
            Sets attributes:
                - unit_descriptors: Array of unit descriptor records from HDF5
                - spikes: List of spike arrays, one per taste/dig_in
                - firing_array: 4D array of firing rates (n_tastes, n_neurons, n_trials, n_timepoints)
                - lfp_array: 4D array of LFP data (n_tastes, n_channels, n_trials, n_timepoints)
        """
        self.get_unit_descriptors()
        self.get_spikes()
        self.get_firing_rates()
        self.get_lfps()

    def separate_laser_data(self):
        """Separate all data types into laser on and off conditions

        Convenience method that separates spikes, firing rates, and LFPs by laser condition.

        Side Effects:
            Sets attributes:
                - on_spikes: Spike arrays for laser on trials
                - off_spikes: Spike arrays for laser off trials
                - on_firing: Firing rate arrays for laser on trials
                - off_firing: Firing rate arrays for laser off trials
                - on_lfp: LFP arrays for laser on trials
                - off_lfp: LFP arrays for laser off trials
        """
        self.separate_laser_spikes()
        self.separate_laser_firing()
        self.separate_laser_lfp()

    def get_unit_descriptors(self):
        """Extract unit descriptors from HDF5 file

        Reads the unit_descriptor table from the HDF5 file containing information
        about each sorted unit (neuron).

        Side Effects:
            Sets self.unit_descriptors: Structured array with fields including:
                - electrode_number: Electrode where unit was recorded
                - single_unit: Boolean indicating if unit is single or multi-unit
                - regular_spiking: Boolean indicating regular spiking classification
                - fast_spiking: Boolean indicating fast spiking classification
        """
        with tables.open_file(self.hdf5_path, 'r+') as hf5_file:
            self.unit_descriptors = hf5_file.root.unit_descriptor[:]
            # Round SNR to 1 decimal place
            for descriptor in self.unit_descriptors:
                descriptor['snr'] = round(descriptor['snr'], 1)

    def check_laser(self):
        """Check if laser trials exist in the data

        Examines trial_info_frame to determine if any trials have non-zero laser durations.

        Side Effects:
            Sets attributes:
                - laser_exists: bool, True if laser trials exist, False otherwise
                - laser_durations: list of np.arrays, laser durations for each taste in ms
        """
        if 'trial_info_frame' not in dir(self):
            print('Trial info frame not found...Loading')
            self.get_trial_info_frame()

        laser_durations = [x['laser_duration_ms'].values for _,
                           x in self.trial_info_frame.groupby('dig_in_num_taste')]
        if any([np.any(x > 0) for x in laser_durations]):
            self.laser_exists = True
        else:
            self.laser_exists = False
        self.laser_durations = laser_durations

        # with tables.open_file(self.hdf5_path, 'r+') as hf5:
        #     dig_in_list = \
        #         [x for x in hf5.list_nodes('/spike_trains')
        #          if 'dig_in' in x.__str__()]
        #
        #     # Mark whether laser exists or not
        #     self.laser_durations_exists = sum([dig_in.__contains__('laser_durations')
        #                                        for dig_in in dig_in_list]) > 0
        #
        #     # If it does, pull out laser durations
        #     if self.laser_durations_exists:
        #         self.laser_durations = np.array([dig_in.laser_durations[:]
        #                                          for dig_in in dig_in_list])
        #
        #         non_zero_laser_durations = np.any(
        #             np.sum(self.laser_durations, axis=0) > 0)
        #
        #     # If laser_durations exists, only non_zero durations
        #     # will indicate laser
        #     # If it doesn't exist, then mark laser as absent
        #     if self.laser_durations_exists:
        #         if non_zero_laser_durations:
        #             self.laser_exists = True
        #         else:
        #             self.laser_exists = False
        #     else:
        #         self.laser_exists = False

    def get_spikes(self):
        """Extract spike arrays from HDF5 file

        Side Effects:
            Sets attributes:
                - spikes: List of spike arrays, one per taste/dig_in
                - dig_in_name_list: List of digital input names
                - dig_in_num_list: List of digital input numbers

        Raises:
            Exception: If no spike trains found in HDF5 file
        """
        print('Loading spikes')
        with tables.open_file(self.hdf5_path, 'r+') as hf5:
            if '/spike_trains' in hf5:
                dig_in_list = \
                    [x for x in hf5.list_nodes('/spike_trains')
                     if 'dig_in' in x.__str__()]
                # Sort dig_in_list by the digital input number to ensure consistent ordering
                dig_in_list = sorted(
                    dig_in_list, key=lambda x: int(x._v_name.split('_')[-1]))
                self.dig_in_name_list = [x._v_name for x in dig_in_list]
                self.dig_in_num_list = [int(x.split('_')[-1])
                                        for x in self.dig_in_name_list]
            else:
                raise Exception('No spike trains found in HF5')

            print('Spike trains loaded from following dig-ins')
            print(
                "\n".join([f'{i}. {x} (dig_in_{self.dig_in_num_list[i]})' for i, x in enumerate(self.dig_in_name_list)]))
            # list of length n_tastes, each element is a 3D array
            # array dimensions are (n_trials, n_neurons, n_timepoints)
            self.spikes = [dig_in.spike_array[:] for dig_in in dig_in_list]

    def separate_laser_spikes(self):
        """Separate spike arrays into laser on and off conditions

        Side Effects:
            Sets attributes:
                - on_spikes: Spike arrays for laser on trials
                - off_spikes: Spike arrays for laser off trials

        Raises:
            Exception: If no laser trials exist in the experiment
        """
        if 'laser_exists' not in dir(self):
            self.check_laser()
        if 'spikes' not in dir(self):
            self.get_spikes()
        if self.laser_exists:
            self.on_spikes = np.array([taste[laser > 0] for taste, laser in
                                       zip(self.spikes, self.laser_durations)])
            self.off_spikes = np.array([taste[laser == 0] for taste, laser in
                                        zip(self.spikes, self.laser_durations)])
        else:
            raise Exception('No laser trials in this experiment')

    def extract_lfps(self):
        """Wrapper function to extract LFPs from raw data files and save to HDF5

        Loads experiment information and calls lfp_processing.extract_lfps to
        process raw electrode data into LFP arrays.

        Side Effects:
            - Loads info_dict and trial_info_frame if not present
            - Calls lfp_processing.extract_lfps which creates /Parsed_LFP node in HDF5
            - Creates dig_in_X_LFPs arrays in HDF5 for each digital input
            - Generates LFP channel check plots for quality control
        """
        if 'info_dict' not in dir(self):
            print('Info dict not found...Loading')
            self.get_info_dict()
        if 'trial_info_frame' not in dir(self):
            print('Trial info frame not found...Loading')
            self.get_trial_info_frame()
        taste_dig_ins = self.info_dict['taste_params']['dig_in_nums']
        # Add final argument to argument list
        if None in self.lfp_params.values():
            print('No LFP params found...using default LFP params')
            self.lfp_params = self.default_lfp_params
        self.lfp_params.update({'dig_in_list': taste_dig_ins})
        lfp_processing.extract_lfps(
            self.data_dir,
            **self.lfp_params,
            trial_info_frame=self.trial_info_frame
        )

    def get_lfp_channels(self):
        """Extract Parsed_LFP_channels from HDF5 file

        This is done separately from get_lfps to avoid the overhead of
        reading the large LFP arrays when only channel information is needed.

        Side Effects:
            Sets self.parsed_lfp_channels: 1D array of electrode channel indices used for LFP recording
        """
        with tables.open_file(self.hdf5_path, 'r+') as hf5:
            if '/Parsed_LFP_channels' not in hf5:
                extract_bool = True
            else:
                extract_bool = False

        if extract_bool:
            self.extract_lfps()

        with tables.open_file(self.hdf5_path, 'r+') as hf5:
            self.parsed_lfp_channels = \
                hf5.root.Parsed_LFP_channels[:]

    def check_file_type(self):
        """Check if file type is compatible with LFP processing

        Verifies that the data directory contains raw .dat files needed for LFP extraction.
        Prints warning if incompatible (traditional format).

        Returns:
            bool: True if file type is compatible (not traditional), False if traditional format
        """
        if 'info_dict' not in dir(self):
            print('Info dict not found...Loading')
            self.get_info_dict()

        if self.info_dict['file_type'] == 'traditional':
            # raise Exception('This method is not yet compatible with traditional data files. ')
            print('LFP Processing is not yet compatible with traditional data files. ')
            return False
        else:
            return True

    def get_lfps(self, re_extract=False):
        """Extract or load LFP arrays from HDF5 file

        Wrapper function to either initiate LFP extraction from raw data
        or load existing LFP arrays from HDF5 file.

        Args:
            re_extract: If True, force re-extraction even if LFPs exist in HDF5

        Side Effects:
            Sets attributes:
                - lfp_array: LFP data, shape (n_tastes, n_channels, n_trials, n_timepoints)
                - all_lfp_array: Reshaped LFP data with tastes concatenated

        Note:
            Not compatible with traditional data files
        """

        if not self.check_file_type():
            return

        with tables.open_file(self.hdf5_path, 'r+') as hf5:

            if ('/Parsed_LFP' not in hf5) or (re_extract == True):
                extract_bool = True
            else:
                extract_bool = False

        if extract_bool:
            self.extract_lfps()

        with tables.open_file(self.hdf5_path, 'r+') as hf5:
            lfp_nodes = [node for node in hf5.list_nodes('/Parsed_LFP')
                         if 'dig_in' in node.__str__()]
            # Account for parsed LFPs being different
            self.lfp_array = np.asarray([node[:] for node in lfp_nodes])
            self.all_lfp_array = \
                self.lfp_array.\
                swapaxes(1, 2).\
                reshape(-1, self.lfp_array.shape[1],
                        self.lfp_array.shape[-1]).\
                swapaxes(0, 1)

    def separate_laser_lfp(self):
        """Separate LFP arrays into laser on and off conditions

        Side Effects:
            Sets attributes:
                - on_lfp: LFP arrays for laser on trials
                - off_lfp: LFP arrays for laser off trials
                - all_on_lfp: Reshaped on_lfp with tastes concatenated
                - all_off_lfp: Reshaped off_lfp with tastes concatenated

        Raises:
            Exception: If no laser trials exist in the experiment
        """
        if 'laser_exists' not in dir(self):
            self.check_laser()
        if 'lfp_array' not in dir(self):
            self.get_lfps()
        if self.laser_exists:
            self.on_lfp = np.array([taste.swapaxes(0, 1)[laser > 0]
                                    for taste, laser in
                                    zip(self.lfp_array, self.laser_durations)])
            self.off_lfp = np.array([taste.swapaxes(0, 1)[laser == 0]
                                     for taste, laser in
                                     zip(self.lfp_array, self.laser_durations)])
            self.all_on_lfp =\
                np.reshape(self.on_lfp, (-1, *self.on_lfp.shape[-2:]))
            self.all_off_lfp =\
                np.reshape(self.off_lfp, (-1, *self.off_lfp.shape[-2:]))
        else:
            raise Exception('No laser trials in this experiment')

    def firing_rate_method_selector(self):
        """Select and configure firing rate calculation method

        Validates firing rate parameters and returns appropriate calculation function
        based on the specified method type ('conv' or 'baks').

        Returns:
            function: Configured firing rate calculation function that takes spike_array
                     as input and returns (firing_rate, time_vector) tuple

        Raises:
            Exception: If firing_rate_params['type'] is not 'conv' or 'baks'
            Exception: If required parameters for selected method are missing

        Raises:
            Exception: If required parameters are missing or invalid
        """
        params = self.firing_rate_params

        type_list = ['conv', 'baks']
        type_exists_bool = 'type' in params.keys()
        if not type_exists_bool:
            raise Exception('Firing rate calculation type not specified.'
                            '\nPlease use: \n {}'.format('\n'.join(type_list)))
        if params['type'] not in type_list:
            raise Exception('Firing rate calculation type not recognized.'
                            '\nPlease use: \n {}'.format('\n'.join(type_list)))

        def check_firing_rate_params(params, param_name_list):
            """Validate that all required firing rate parameters are present and not None

            Args:
                params: Dictionary of firing rate parameters
                param_name_list: List of required parameter names

            Raises:
                Exception: If any required parameters are missing or None
            """
            param_exists_bool = [True if x in params.keys() else False
                                 for x in param_name_list]
            if not all(param_exists_bool):
                raise Exception('All required firing rate parameters'
                                ' have not been specified \n{}'.format(
                                    '\n'.join(map(str,
                                                  list(zip(param_exists_bool, param_name_list))))))
            param_present_bool = [params[x] is not None
                                  for x in param_name_list]
            if not all(param_present_bool):
                raise Exception('All required firing rate parameters'
                                ' have not been specified \n{}'.format(
                                    '\n'.join(map(str,
                                                  list(zip(param_present_bool, param_name_list))))))

        if params['type'] == 'conv':
            param_name_list = ['step_size', 'window_size', 'dt']

            # This checks if anything is missing
            # And raises exception if anything missing
            check_firing_rate_params(params, param_name_list)

            # If all good, define the function to be used
            def calc_firing_func(data):
                """Calculate firing rates using convolution method"""
                firing_rate, time_vector = \
                    self._calc_conv_rates(
                        step_size=self.firing_rate_params['step_size'],
                        window_size=self.firing_rate_params['window_size'],
                        dt=self.firing_rate_params['dt'],
                        spike_array=data)
                return firing_rate, time_vector

        if params['type'] == 'baks':
            param_name_list = ['baks_resolution', 'baks_dt']
            check_firing_rate_params(params, param_name_list)

            def calc_firing_func(data):
                """Calculate firing rates using BAKS method"""
                firing_rate, time_vector = \
                    self._calc_baks_rate(
                        resolution=self.firing_rate_params['baks_resolution'],
                        dt=self.firing_rate_params['baks_dt'],
                        spike_array=data)
                return firing_rate, time_vector

        return calc_firing_func

    def get_firing_rates(self):
        """Convert spikes to firing rates using specified method

        Uses firing_rate_params to determine calculation method (conv or baks)
        and computes firing rates for all spike data.

        Side Effects:
            Sets attributes:
                - firing_list: List of firing rate arrays, one per taste
                  Each element shape: (n_trials, n_neurons, n_timepoints)
                - time_vector: Time points in ms relative to stimulus delivery
                - firing_array: 4D array (n_tastes, n_neurons, n_trials, n_timepoints)
                  Only created if all tastes have equal dimensions
                - normalized_firing: Normalized firing rates (same shape as firing_array)
                - all_firing_array: Reshaped firing with tastes concatenated
                - all_normalized_firing: Reshaped normalized firing with tastes concatenated
        """

        if 'spikes' not in dir(self):
            # raise Exception('Run method "get_spikes" first')
            print('No spikes found, getting spikes ...')
            self.get_spikes()
        if None in self.firing_rate_params.values():
            # raise Exception('Specify "firing_rate_params" first')
            print('No firing rate params found...using default firing params')
            pp(self.default_firing_params)
            print('If you want specific firing params, set them manually')
            self.firing_rate_params = self.default_firing_params
        if 'sorting_params_dict' not in dir(self):
            print('No sorting params found, getting info dict ...')
            self.get_sorting_params_dict()
        spike_train_lims = self.sorting_params_dict['spike_array_durations']

        calc_firing_func = self.firing_rate_method_selector()
        results = [calc_firing_func(spikes) for spikes in self.spikes]
        self.firing_list = [result[0] for result in results]
        # Store the time vector from the first result (they should all be the same)
        # Adjust time relative to stimulus delivery
        raw_time_vector = results[0][1] - spike_train_lims[0]
        self.time_vector = np.array(raw_time_vector)

        if np.sum([self.firing_list[0].shape == x.shape
                   for x in self.firing_list]) == len(self.firing_list):
            print('All tastes have equal dimensions,'
                  'concatenating and normalizing')

            # Reshape for backward compatiblity
            self.firing_array = np.asarray(self.firing_list).swapaxes(1, 2)
            # Concatenate firing across all tastes
            self.all_firing_array = \
                self.firing_array.\
                swapaxes(1, 2).\
                reshape(-1, self.firing_array.shape[1],
                        self.firing_array.shape[-1]).\
                swapaxes(0, 1)

            # Calculate normalized firing
            min_vals = [np.min(self.firing_array[:, nrn, :, :], axis=None)
                        for nrn in range(self.firing_array.shape[1])]
            max_vals = [np.max(self.firing_array[:, nrn, :, :], axis=None)
                        for nrn in range(self.firing_array.shape[1])]
            self.normalized_firing = np.asarray(
                [(self.firing_array[:, nrn, :, :] - min_vals[nrn]) /
                 (max_vals[nrn] - min_vals[nrn])
                 for nrn in range(self.firing_array.shape[1])]).\
                swapaxes(0, 1)

            # Concatenate normalized firing across all tastes
            self.all_normalized_firing = \
                self.normalized_firing.\
                swapaxes(1, 2).\
                reshape(-1, self.normalized_firing.shape[1],
                        self.normalized_firing.shape[-1]).\
                swapaxes(0, 1)

            print('Generated attributes:')
            print(
                f'  firing_list: {len(self.firing_list)} arrays of shape {self.firing_list[0].shape}')
            print(f'  time_vector: shape {self.time_vector.shape}')
            print(f'  firing_array: shape {self.firing_array.shape}')
            print(f'  all_firing_array: shape {self.all_firing_array.shape}')
            print(f'  normalized_firing: shape {self.normalized_firing.shape}')
            print(
                f'  all_normalized_firing: shape {self.all_normalized_firing.shape}')

        else:
            print('Uneven numbers of trials...not stacking into firing rates array')
            print('Generated attributes:')
            print(
                f'  firing_list: {len(self.firing_list)} arrays with shapes {[x.shape for x in self.firing_list]}')
            print(f'  time_vector: shape {self.time_vector.shape}')

    def calc_palatability(self):
        """
        Calculate single neuron (absolute) palatability from firing rates

        Requires:
            - info_dict
                - palatability ranks
                - taste names
            - firing rates

        Generates:
            - pal_df : pandas dataframe
                - shape: tastes x 3 cols (dig_ins, taste_names, pal_ranks)
            - pal_array : np.array
                - shape : neurons x time_bins
        """

        if 'info_dict' not in dir(self):
            print('Info dict not found...Loading')
            self.get_info_dict()
        if 'firing_list' not in dir(self):
            print('Firing list not found...Loading')
            self.get_firing_rates()

        # Get taste information from info_dict
        self.taste_names = self.info_dict['taste_params']['tastes']
        self.palatability_ranks = self.info_dict['taste_params']['pal_rankings']

        # Get digital input information from info_dict to ensure correct mapping
        taste_dig_ins = self.info_dict['taste_params']['dig_in_nums']

        # Create a mapping from dig_in numbers to indices in the taste arrays
        dig_in_to_index = {dig_in: i for i, dig_in in enumerate(taste_dig_ins)}

        # Reorder taste_names and palatability_ranks to match the order in dig_in_name_list
        ordered_taste_names = []
        ordered_pal_ranks = []

        for dig_in_num in self.dig_in_num_list:
            if dig_in_num in dig_in_to_index:
                idx = dig_in_to_index[dig_in_num]
                if idx < len(self.taste_names):
                    ordered_taste_names.append(self.taste_names[idx])
                    ordered_pal_ranks.append(self.palatability_ranks[idx])
                else:
                    warnings.warn(
                        f"Index {idx} out of range for taste_names and palatability_ranks")
                    ordered_taste_names.append(f"Unknown-{dig_in_num}")
                    ordered_pal_ranks.append(0)  # Default palatability rank
            else:
                warnings.warn(
                    f"Digital input {dig_in_num} not found in taste_params.dig_ins")
                ordered_taste_names.append(f"Unknown-{dig_in_num}")
                ordered_pal_ranks.append(0)  # Default palatability rank

        print('Calculating palatability with following order:')
        self.pal_df = pd.DataFrame(
            dict(
                dig_ins=self.dig_in_name_list,
                dig_in_nums=self.dig_in_num_list,
                taste_names=ordered_taste_names,
                pal_ranks=ordered_pal_ranks,
            )
        )
        print(self.pal_df)
        trial_counts = [x.shape[0] for x in self.firing_list]
        # Use the ordered palatability ranks from the pal_df DataFrame
        pal_vec = np.concatenate(
            [np.repeat(x, y) for x, y in zip(self.pal_df['pal_ranks'], trial_counts)])
        cat_firing = np.concatenate(self.firing_list, axis=0).T
        # Add very small noise to avoid issues with zero or same firing rates
        # when calculating Spearman correlation
        cat_firing += np.random.normal(0, 1e-6, cat_firing.shape)
        inds = list(np.ndindex(cat_firing.shape[:2]))
        pal_rho_array = np.zeros(cat_firing.shape[:2])
        pal_p_array = np.zeros(cat_firing.shape[:2])
        for this_ind in tqdm(inds):
            rho, p_val = spearmanr(cat_firing[tuple(this_ind)], pal_vec)
            pal_rho_array[tuple(this_ind)] = rho
            pal_p_array[tuple(this_ind)] = p_val
        self.pal_rho_array = np.abs(pal_rho_array).T
        self.pal_p_array = pal_p_array.T

    def separate_laser_firing(self):
        """Separate firing rate arrays into laser on and off conditions

        Side Effects:
            Sets attributes:
                - on_firing: Firing rates for laser on trials
                - off_firing: Firing rates for laser off trials
                - all_on_firing: Reshaped on_firing with tastes concatenated
                - all_off_firing: Reshaped off_firing with tastes concatenated

        Raises:
            Exception: If no laser trials exist in the experiment
        """
        if 'laser_exists' not in dir(self):
            self.check_laser()
        if 'firing_array' not in dir(self):
            self.get_firing_rates()
        if self.laser_exists:
            self.on_firing = np.array([taste[laser > 0] for taste, laser in
                                       zip(self.firing_list, self.laser_durations)])
            self.off_firing = np.array([taste[laser == 0] for taste, laser in
                                        zip(self.firing_list, self.laser_durations)])
            self.all_on_firing =\
                np.reshape(self.on_firing, (-1, *self.on_firing.shape[-2:]))
            self.all_off_firing =\
                np.reshape(self.off_firing, (-1, *self.off_firing.shape[-2:]))
        else:
            raise Exception('No laser trials in this experiment')

    def get_info_dict(self):
        """Load experiment information from .info JSON file

        Loads the .info file containing experimental parameters and metadata.

        Side Effects:
            Sets self.info_dict: Dictionary containing:
                - file_type: Type of data file ('one file' or 'traditional')
                - taste_params: Dictionary with taste delivery parameters
                - electrode_layout: Dictionary mapping brain regions to electrode lists
                - palatability_ranks: List of palatability rankings for each taste

        Raises:
            Exception: If no .info file is found in data directory
        """
        json_path = glob.glob(os.path.join(self.data_dir, "**.info"))[0]
        if os.path.exists(json_path):
            self.info_dict = json_dict = json.load(open(json_path, 'r'))
        else:
            raise Exception('No info file found')

    def get_sorting_params_dict(self):
        """Extract sorting parameters from .params file

        Side Effects:
            Sets self.sorting_params_dict: Dictionary containing:
                - spike_array_durations: [pre_stim, post_stim] durations in ms
                - sampling_rate: Sampling rate in Hz
                - voltage_cutoff: Voltage threshold for spike detection
                - max_breach_rate: Maximum breach rate for artifact rejection
                - max_secs_above_cutoff: Maximum seconds above cutoff for artifact rejection

        Raises:
            Exception: If no .params file is found in data directory
        """
        json_path = glob.glob(os.path.join(self.data_dir, "**.params"))[0]
        if os.path.exists(json_path):
            self.sorting_params_dict = json.load(open(json_path, 'r'))
        else:
            raise Exception('No info file found')

    def get_region_electrodes(self):
        """Extract electrode assignments for each brain region from .info file

        Side Effects:
            Sets attributes:
                - region_electrode_dict: Dictionary mapping region names to electrode lists
                - region_names: List of region names (excludes 'emg' and 'none')

        Raises:
            Exception: If .info file is not found
        """
        # json_name = self.hdf5_path.split('.')[0] + '.info'
        # json_path = os.path.join(self.data_dir, json_name)
        json_path = glob.glob(os.path.join(self.data_dir, "**.info"))[0]
        if os.path.exists(json_path):
            json_dict = json.load(open(json_path, 'r'))
            self.region_electrode_dict = json_dict["electrode_layout"]
            # Drop 'emg' or 'none' regions
            self.region_electrode_dict = {k: v for k, v in
                                          self.region_electrode_dict.items()
                                          if 'emg' not in k and 'none' not in k}
            self.region_names = [x for x in self.region_electrode_dict.keys()
                                 if 'emg' not in x]
        else:
            raise Exception("Cannot find json file. Make sure it's present")

    def get_region_units(self):
        """Extract unit indices organized by brain region and CAR group

        Side Effects:
            Sets attributes:
                - car_names: List of CAR (Common Average Reference) group names
                - car_electrodes: List of electrode lists for each CAR group
                - car_units: List of unit indices for each CAR group
                - region_units: List of unit indices for each brain region
        """
        if "region_electrode_dict" not in dir(self):
            self.get_region_electrodes()
        if "unit_descriptors" not in dir(self):
            self.get_unit_descriptors()

        unit_electrodes = [x['electrode_number']
                           for x in self.unit_descriptors]
        region_electrode_vals = [val for key, val in
                                 self.region_electrode_dict.items() if key != 'emg']

        car_name = []
        car_electrodes = []
        for key, val in self.region_electrode_dict.items():
            if key != 'emg':
                for num, this_car in enumerate(val):
                    car_electrodes.append(this_car)
                    car_name.append(key+str(num))

        self.car_names = car_name
        self.car_electrodes = car_electrodes

        car_ind_vec = np.zeros(len(unit_electrodes))
        for num, val in enumerate(self.car_electrodes):
            for elec_num, elec in enumerate(unit_electrodes):
                if elec in val:
                    # This tells you which car group each neuron is in
                    car_ind_vec[elec_num] = num

        self.car_units = [np.where(car_ind_vec == x)[0]
                          for x in np.unique(car_ind_vec)]

        region_ind_vec = np.zeros(len(unit_electrodes))
        for elec_num, elec in enumerate(unit_electrodes):
            for region_num, region in enumerate(region_electrode_vals):
                for car in region:
                    if elec in car:
                        region_ind_vec[elec_num] = region_num

        self.region_units = [np.where(region_ind_vec == x)[0]
                             for x in np.unique(region_ind_vec)]

    def return_region_spikes(self, region_name='all'):
        """Return spike data for a specific brain region

        Args:
            region_name: Name of brain region to extract, or 'all' for all spikes

        Returns:
            np.array: Spike data for specified region, shape (n_tastes, n_trials, n_neurons, n_timepoints)
                     Returns None if no units found in region

        Raises:
            Exception: If region_name not found or multiple matches found
        """
        if 'region_names' not in dir(self):
            self.get_region_units()
        if self.spikes is None:
            self.get_spikes()

        if not region_name == 'all':
            region_ind = [num for num, x in enumerate(self.region_names)
                          if x == region_name]
            if not len(region_ind) == 1:
                raise Exception('Region name not found, or too many matches found, '
                                'acceptable options are' +
                                '\n' + f"===> {self.region_names, 'all'}")
            else:
                if region_ind[0] < len(self.region_units):
                    this_region_units = self.region_units[region_ind[0]]
                    region_spikes = [x[:, this_region_units]
                                     for x in self.spikes]
                    return np.array(region_spikes)
                else:
                    print(f'No units found in this region: {region_name}')
                    return None
        else:
            return self.spikes

    def get_region_firing(self, region_name='all'):
        """Return firing rate data for a specific brain region

        Args:
            region_name: Name of brain region to extract, or 'all' for all firing rates

        Returns:
            np.array or list: Firing rate data for specified region
                             Returns array if trials are even, list if uneven
                             Shape: (n_tastes, n_neurons, n_trials, n_timepoints) for arrays

        Raises:
            Exception: If region_name not found or multiple matches found
        """
        if 'region_units' not in dir(self):
            self.get_region_units()
        if 'firing_array' not in dir(self):
            self.get_firing_rates()
        # If firing_array is still not generated, that means firing cannot be stack (uneven trials)
        if 'firing_array' not in dir(self):
            firing_obj = self.firing_list
            uneven_trials = True
        else:
            firing_obj = self.firing_array
            uneven_trials = False

        if not region_name == 'all':
            region_ind = [num for num, x in enumerate(self.region_names)
                          if x == region_name]
            if not len(region_ind) == 1:
                raise Exception('Region name not found, or too many matches found, '
                                'acceptable options are' +
                                '\n' + f"===> {self.region_names, 'all'}")
            else:
                this_region_units = self.region_units[region_ind[0]]
                region_firing = [x[this_region_units]
                                 for x in firing_obj]
                if uneven_trials:
                    # If firing cannot be stacked, return list of arrays
                    return region_firing
                else:
                    return np.array(region_firing)
        else:
            if uneven_trials:
                # If firing cannot be stacked, return list of arrays
                return firing_obj
            else:
                return np.array(self.firing_array)

    def get_lfp_electrodes(self):
        """Extract LFP electrode indices organized by brain region

        Maps LFP channels to brain regions based on electrode layout information
        from the .info file.

        Side Effects:
            Sets self.lfp_region_electrodes: List of arrays, each containing electrode
                                            indices for a specific brain region
        """
        if not self.check_file_type():
            return

        if 'parsed_lfp_channels' not in dir(self):
            self.get_lfp_channels()
        if 'region_electrode_dict' not in dir(self):
            self.get_region_electrodes()

        region_electrode_vals = [val for key, val in
                                 self.region_electrode_dict.items() if key != 'emg']
        region_ind_vec = np.zeros(len(self.parsed_lfp_channels))
        for elec_num, elec in enumerate(self.parsed_lfp_channels):
            for region_num, region in enumerate(region_electrode_vals):
                for car in region:
                    if elec in car:
                        region_ind_vec[elec_num] = region_num

        self.lfp_region_electrodes = [np.where(region_ind_vec == x)[0]
                                      for x in np.unique(region_ind_vec)]

    def get_stft(
            self,
            recalculate=False,
            dat_type=['amplitude'],
            write_out=True):
        """Retrieve or calculate Short-Time Fourier Transform of LFP data

        If STFT exists in HDF5, retrieves it. Otherwise calculates STFT
        for all LFP data and optionally saves to HDF5.

        Args:
            recalculate: If True, force recalculation even if STFT exists
            dat_type: List of data types to load/calculate: 'raw', 'amplitude', 'phase'
            write_out: If True, write calculated STFT to HDF5 file

        Side Effects:
            Sets attributes:
                - freq_vec: Frequency vector for STFT
                - time_vec: Time vector for STFT
                - stft_array: Raw complex STFT (if 'raw' in dat_type)
                - amplitude_array: STFT amplitude (if 'amplitude' in dat_type)
                - phase_array: STFT phase (if 'phase' in dat_type)
        """

        if not self.check_file_type():
            return

        # Check if STFT in HDF5
        # If present, only load what user has asked for
        if not recalculate:
            self.calc_stft_bool = 0
            with tables.open_file(self.hdf5_path, 'r+') as hf5:
                if ('/stft/stft_array' in hf5) and (not recalculate):
                    self.freq_vec = hf5.root.stft.freq_vec[:]
                    self.time_vec = hf5.root.stft.time_vec[:]
                    if 'raw' in dat_type:
                        self.stft_array = hf5.root.stft.stft_array[:]
                    if 'amplitude' in dat_type:
                        self.amplitude_array = hf5.root.stft.amplitude_array[:]
                    if 'phase' in dat_type:
                        self.phase_array = hf5.root.stft.phase_array[:]

                    # If everything there, then don't calculate
                    # Unless forced to
                    self.calc_stft_bool = 0

                else:
                    self.calc_stft_bool = 1

                if ('/stft' not in hf5):
                    hf5.create_group('/', 'stft')
                    hf5.flush()
        else:
            self.calc_stft_bool = 1

        if self.calc_stft_bool:
            print('Calculating STFT')

            # Get LFPs to calculate STFT
            if "lfp_array" not in dir(self):
                self.get_lfps()

            # Generate list of individual trials to be fed into STFT function
            stft_iters = list(
                product(
                    *list(map(np.arange, self.lfp_array.shape[:3]))
                )
            )

            # Calculate STFT over lfp array
            try:
                stft_list = Parallel(n_jobs=mp.cpu_count()-2)(delayed(self.calc_stft)(self.lfp_array[this_iter],
                                                                                      **self.stft_params)
                                                              for this_iter in tqdm(stft_iters))
            except:
                warnings.warn("Couldn't process STFT in parallel."
                              "Running serial loop")
            # stft_list = [self.calc_stft(self.lfp_array[this_iter],
            #                            **self.stft_params)\
            #        for this_iter in tqdm(stft_iters)]

            self.freq_vec = stft_list[0][0]
            self.time_vec = stft_list[0][1]
            fin_stft_list = [x[-1] for x in stft_list]
            del stft_list
            amplitude_list = self.parallelize(np.abs, fin_stft_list)
            phase_list = self.parallelize(np.angle, fin_stft_list)

            # (taste, channel, trial, frequencies, time)
            self.stft_array = self.convert_to_array(fin_stft_list, stft_iters)
            del fin_stft_list
            self.amplitude_array = self.convert_to_array(
                amplitude_list, stft_iters)**2
            del amplitude_list
            self.phase_array = self.convert_to_array(phase_list, stft_iters)
            del phase_list

            # After recalculating, only keep what was asked for
            object_names = ['freq_vec', 'time_vec']
            object_list = [self.freq_vec, self.time_vec]

            if 'raw' in dat_type:
                object_names.append('stft_array')
                object_list.append(self.stft_array)
            else:
                del self.stft_array

            if 'amplitude' in dat_type:
                object_names.append('amplitude_array')
                object_list.append(self.amplitude_array)
            else:
                del self.amplitude_array

            if 'phase' in dat_type:
                object_names.append('phase_array')
                object_list.append(self.phase_array)
            else:
                del self.phase_array

            if write_out:
                dir_path = '/stft'

                with tables.open_file(self.hdf5_path, 'r+') as hf5:
                    for name, obj in zip(object_names, object_list):
                        self.remove_node(os.path.join(dir_path, name), hf5)
                        hf5.create_array(dir_path, name, obj)

    def return_region_lfps(self):
        """Return list containing LFPs for each region and region names

        Extracts LFP data organized by brain region based on electrode assignments.

        Returns:
            tuple: (region_lfp, region_names)
                - region_lfp: List of arrays, each with shape (n_tastes, n_channels_in_region, n_trials, n_timepoints)
                - region_names: List of region name strings
        """
        if not self.check_file_type():
            return

        if 'lfp_array' not in dir(self):
            self.get_lfps()
        if 'lfp_region_electrodes' not in dir(self):
            self.get_lfp_electrodes()
        region_lfp = [self.lfp_array[:, x, :, :]
                      for x in self.lfp_region_electrodes]
        return region_lfp, self.region_names

    def return_representative_lfp_channels(self):
        """Return one electrode per region that is closest to the mean

        Identifies a single representative LFP channel for each brain region by
        finding the channel with minimum deviation from the region's mean LFP.

        Returns:
            tuple: (wanted_channel_inds, region_names, region_lfps)
                - wanted_channel_inds: List of channel indices, one per region
                - region_names: List of region name strings (sorted alphabetically)
                - region_lfps: List of LFP arrays for each region
        """
        if not self.check_file_type():
            return
        # Region lfps shape : (n_tastes, n_channels, n_trials, n_timepoints)
        region_lfps, region_names = self.return_region_lfps()
        # Drop 'none' if in region_names
        keep_inds = [i for i, x in enumerate(region_names) if x != 'none']
        region_lfps = [region_lfps[i] for i in keep_inds]
        region_names = [region_names[i] for i in keep_inds]

        # Sort by region_names to make sure order is always same
        sort_inds = np.argsort(region_names)
        region_lfps = [region_lfps[x] for x in sort_inds]
        region_names = [region_names[x] for x in sort_inds]

        # Find channel per region closest to mean
        # mean_region_lfp shape : (n_regions, n_tastes, n_trials, n_timepoints)
        mean_region_lfp = np.stack([np.mean(x, axis=1) for x in region_lfps])

        # Find channel per region closest to mean
        wanted_channel_inds = []
        for this_mean, this_region in zip(mean_region_lfp, region_lfps):
            diff_lfp = np.abs(this_region - this_mean[:, np.newaxis, :, :])
            mean_diff_lfp = diff_lfp.mean(axis=(0, 2, 3))
            min_diff_lfp = np.argmin(mean_diff_lfp)
            wanted_channel_inds.append(min_diff_lfp)

        wanted_lfp_electrodes = np.array([x[:, y]
                                          for x, y in zip(region_lfps, wanted_channel_inds)])
        return wanted_channel_inds, wanted_lfp_electrodes, region_names

    def get_mean_stft_amplitude(self):
        """Calculate mean STFT amplitude for each brain region

        Computes the median STFT amplitude across tastes, trials, and channels
        for each brain region.

        Returns:
            np.array: Mean STFT amplitude per region, shape (n_regions, n_freqs, n_times)
        """
        if not self.check_file_type():
            return
        if 'amplitude_array' not in dir(self):
            self.get_stft()
        if 'lfp_region_electrodes' not in dir(self):
            self.get_lfp_electrodes()

        aggregate_amplitude = []
        for region in self.lfp_region_electrodes:
            aggregate_amplitude.append(
                np.median(self.amplitude_array[:, region], axis=(0, 1, 2)))
        return np.array(aggregate_amplitude)

    def get_trial_info_frame(self):
        """Load trial information from CSV file

        Loads trial_info_frame.csv containing trial metadata including
        taste identities, laser conditions, and timing information.

        Side Effects:
            Sets self.trial_info_frame: pandas DataFrame with columns including:
                - dig_in_num_taste: Digital input number for taste delivery
                - laser_duration_ms: Duration of laser stimulation in ms
                - start_taste_ms: Trial start time in ms
                - end_taste_ms: Trial end time in ms
        """
        self.trial_info_frame = pd.read_csv(
            os.path.join(self.data_dir, 'trial_info_frame.csv'))

    def sequester_trial_inds(self):
        """Organize trial indices by taste and laser condition

        Groups trials by taste identity and laser parameters, creating
        a DataFrame with trial indices for each unique combination.

        Side Effects:
            Sets self.trial_inds_frame: pandas DataFrame with columns:
                - dig_in_num_taste: Digital input number
                - laser_duration_ms: Laser duration in ms
                - trial_inds: List of trial indices for this condition

        Side Effects:
            Sets self.trial_inds_frame: DataFrame with columns:
                - dig_in_num_taste: Digital input number
                - laser_duration_ms: Laser duration
                - laser_lag_ms: Laser lag
                - trial_inds: List of trial indices for this group
        """

        wanted_cols = [
            'dig_in_num_taste',
            'dig_in_name_taste',
            'taste',
            'laser_duration_ms',
            'laser_lag_ms',
            'taste_rel_trial_num',
        ]
        group_cols = ['dig_in_num_taste', 'laser_duration_ms', 'laser_lag_ms']
        if 'trial_info_frame' not in dir(self):
            self.get_trial_info_frame()
        wanted_frame = self.trial_info_frame[wanted_cols]
        grouped_frame = wanted_frame.groupby(group_cols)
        group_list = list(grouped_frame)
        group_names = [x[0] for x in group_list]
        group_name_frame = pd.DataFrame(group_names, columns=group_cols)
        grouped_frame_list = [x[1] for x in group_list]
        # Aggregate 'taste_rel_trial_num' into a list for each group
        trial_inds = [x['taste_rel_trial_num'].tolist()
                      for x in grouped_frame_list]
        group_name_frame['trial_inds'] = trial_inds
        self.trial_inds_frame = group_name_frame

    def get_sequestered_spikes(self):
        """Organize spike data by taste and laser condition

        Extracts spike data for each unique combination of taste and laser
        parameters, creating both array and DataFrame representations.

        Side Effects:
            Sets attributes:
                - sequestered_spikes: List of spike arrays for each group
                - sequestered_spikes_frame: DataFrame with spike indices and metadata
                - trial_inds_frame['spikes']: Adds spikes column to trial_inds_frame
        """
        if 'trial_inds_frame' not in dir(self):
            self.sequester_trial_inds()
        if 'spikes' not in dir(self):
            self.get_spikes()
        # Get trial inds for each group
        trial_inds_frame = self.trial_inds_frame.copy()
        self.sequestered_spikes = []
        sequestered_spikes_frame_list = []
        for i, this_row in trial_inds_frame.iterrows():
            taste_ind = np.where(
                np.array(self.dig_in_num_list) == int(
                    this_row['dig_in_num_taste'])
            )[0][0]
            trial_inds = this_row['trial_inds']
            this_seq_spikes = self.spikes[taste_ind][this_row['trial_inds']]
            self.sequestered_spikes.append(this_seq_spikes)
            spike_inds = np.where(this_seq_spikes)
            this_seq_spikes = pd.DataFrame(
                dict(
                    trial_num=spike_inds[0],
                    neuron_num=spike_inds[1],
                    time_num=spike_inds[2],
                )
            )
            this_seq_spikes['taste_num'] = taste_ind
            this_seq_spikes['laser_tuple'] = str(
                (this_row['laser_lag_ms'], this_row['laser_duration_ms']))
            sequestered_spikes_frame_list.append(this_seq_spikes)
        self.trial_inds_frame['spikes'] = self.sequestered_spikes
        self.sequestered_spikes_frame = pd.concat(
            sequestered_spikes_frame_list)
        print('Added sequestered spikes to trial_inds_frame')

    def get_sequestered_firing(self):
        """Organize firing rate data by taste and laser condition

        Extracts firing rate data for each unique combination of taste and
        laser parameters, creating both array and DataFrame representations.

        Side Effects:
            Sets attributes:
                - sequestered_firing: List of firing rate arrays for each group
                - sequestered_firing_frame: DataFrame with firing rates and metadata
                - trial_inds_frame['firing']: Adds firing column to trial_inds_frame
        """
        if 'trial_inds_frame' not in dir(self):
            self.sequester_trial_inds()
        if 'firing_array' not in dir(self):
            self.get_firing_rates()
        group_cols = ['dig_in_num_taste', 'laser_duration_ms', 'laser_lag_ms']
        # Get trial inds for each group
        trial_inds_frame = self.trial_inds_frame.copy()
        self.sequestered_firing = []
        sequestered_firing_frame_list = []
        for i, this_row in trial_inds_frame.iterrows():
            taste_ind = np.where(
                np.array(self.dig_in_num_list) == int(
                    this_row['dig_in_num_taste'])
            )[0][0]
            trial_inds = this_row['trial_inds']
            laser_tuple = (this_row['laser_lag_ms'],
                           this_row['laser_duration_ms'])
            this_seq_firing = self.firing_list[taste_ind][trial_inds]
            self.sequestered_firing.append(this_seq_firing)
            inds = np.array(list(np.ndindex(this_seq_firing.shape)))
            this_seq_firing = pd.DataFrame(
                dict(
                    trial_num=inds[:, 0],
                    neuron_num=inds[:, 1],
                    time_num=inds[:, 2],
                    firing=this_seq_firing.flatten(),
                )
            )
            this_seq_firing['taste_num'] = taste_ind
            this_seq_firing['laser_tuple'] = str(laser_tuple)
            sequestered_firing_frame_list.append(this_seq_firing)
        self.trial_inds_frame['firing'] = self.sequestered_firing
        self.sequestered_firing_frame = pd.concat(
            sequestered_firing_frame_list)
        print('Added sequestered firing to trial_inds_frame')

    def get_sequestered_data(self):
        """
        Sequester spikes and firing into different categories:
            - Tastes
            - Laser conditions
        """
        self.get_sequestered_spikes()
        self.get_sequestered_firing()

    def get_stable_units(self, p_val_threshold=0.05):
        """
        Load drift check results from a CSV file and mark units as stable or unstable
        based on a p-value threshold.

        Parameters:
        -----------
        p_val_threshold : float, default=0.05
            Threshold for p-value to determine stability.
            Units with p-values >= p_val_threshold are considered stable.

        Returns:
        --------
        None
            Results are stored as class attributes:
            - drift_results: DataFrame containing the loaded CSV data
            - stable_units: Boolean array indicating which units are stable
            - unstable_units: Boolean array indicating which units are unstable

        Example:
        --------
        >>> data = ephys_data(data_dir='/path/to/data')
        >>> # Access stable units
        >>> stable_unit_indices = np.where(data.stable_units)[0]
        """

        csv_path = os.path.join(
            self.data_dir, 'QA_output', 'post_drift_p_vals.csv')

        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Drift check results file not found: {csv_path}")

        # Load the CSV file
        self.drift_results = pd.read_csv(csv_path, index_col=0)

        # Rename "trial_bin" column to "p_val"
        self.drift_results.rename(columns={'trial_bin': 'p_val'}, inplace=True)

        # Mark stable
        self.drift_results['stable'] = self.drift_results['p_val'] >= p_val_threshold

        # Get the indices of stable and unstable units
        self.stable_units = self.drift_results[self.drift_results['stable']]['unit'].values
        self.unstable_units = self.drift_results[~self.drift_results['stable']]['unit'].values

        print(
            f"Loaded drift check results for {len(self.drift_results)} units")
        print(
            f"Found {len(self.stable_units)} stable units and {len(self.unstable_units)} unstable units")
        print(f"Using p-value threshold of {p_val_threshold}")

    def calculate_responsiveness(self, stim_time, params_dict):
        """
        Calculate unit responsiveness based on pre and post-stimulus firing rates.

        Args:
            stim_time: Time of stimulus presentation.
            seq_spikes_frame: DataFrame containing spike data.
            params_dict: Dictionary of parameters for responsiveness calculation.

        Returns:
            dict: A dictionary of responsiveness p-values for each unit.
        """

        if 'sequestered_firing_frame' not in dir(self) or \
                'sequestered_spikes_frame' not in dir(self):
            self.get_sequestered_data()

        print("="*40)
        print("Calculating unit responsiveness")
        print("="*40)

        # Get data in responsive window
        responsive_window = params_dict.get(
            'responsiveness_pre_post_durations', [1000, 2000])
        responsive_inds = ((stim_time-responsive_window[0], stim_time),
                           (stim_time, stim_time+responsive_window[1]))
        min_ind, max_ind = min(responsive_inds[0]), max(responsive_inds[1])

        seq_spikes_frame = self.sequestered_spikes_frame.copy()
        seq_spikes_frame = seq_spikes_frame.loc[
            (seq_spikes_frame.time_num >= min_ind) &
            (seq_spikes_frame.time_num < max_ind)
        ]
        seq_spikes_frame['spikes'] = 1

        # Mark pre/post stimulus
        seq_spikes_frame['post_stim'] = seq_spikes_frame['time_num'] >= stim_time

        # Get mean spikes per condition
        seq_spike_counts = seq_spikes_frame.groupby(
            ['trial_num', 'neuron_num', 'taste_num', 'laser_tuple', 'post_stim']
        ).count().reset_index()
        seq_spike_counts.drop(
            columns=['time_num'], inplace=True, errors='ignore')

        # Adjust for differing window sizes
        pre_window, post_window = responsive_window
        seq_spike_counts['window_len'] = seq_spike_counts['post_stim'].apply(
            lambda x: post_window if x else pre_window)
        seq_spike_counts['rate'] = seq_spike_counts['spikes'] / \
            seq_spike_counts['window_len']

        # Correct for missing zero-spike entries
        index_cols = ['trial_num', 'neuron_num', 'taste_num', 'laser_tuple']
        firing_frame_group_inds = list(
            self.sequestered_firing_frame.groupby(index_cols).groups.keys())

        firing_frame_group_inds = pd.DataFrame(
            firing_frame_group_inds, columns=index_cols)

        # Stack 2 for post/pre
        firing_frame_group_inds = pd.concat(
            [firing_frame_group_inds.assign(post_stim=False),
             firing_frame_group_inds.assign(post_stim=True)],
            ignore_index=True,
        )

        # Make sure post-stim in both is boolean
        firing_frame_group_inds['post_stim'] = firing_frame_group_inds['post_stim'].astype(
            bool)
        seq_spike_counts['post_stim'] = seq_spike_counts['post_stim'].astype(
            bool)

        seq_spike_counts = pd.merge(
            firing_frame_group_inds,
            seq_spike_counts,
            on=index_cols + ['post_stim'],
            how='left',
        )
        seq_spike_counts.fillna(0, inplace=True)

        # Calculate responsiveness p-values
        resp_pvals = {}
        group_cols = ['neuron_num', 'taste_num', 'laser_tuple']
        for (nrn, taste, laser), group in tqdm(seq_spike_counts.groupby(group_cols)):
            pval = ttest_rel(
                group.loc[group.post_stim, 'rate'].values,
                group.loc[~group.post_stim, 'rate'].values,
            )[1]
            resp_pvals[(nrn, laser)] = min(
                resp_pvals.get((nrn, laser), 1.0), pval)

        return resp_pvals

    def calculate_discriminability_dynamicity(self, stim_time, params_dict):
        """
        Calculate unit discriminability based on ANOVA analysis.

        Args:
            seq_spikes_anova: DataFrame containing spike data for ANOVA.
            params_dict: Dictionary of parameters for discriminability calculation.

        Returns:
            dict: A dictionary of discriminability p-values for each unit.
        """

        print("="*40)
        print("Calculating unit discriminability and dynamicity")
        print("="*40)

        if 'sequestered_spikes_frame' not in dir(self):
            self.get_sequestered_data()

        discrim_params = params_dict.get(
            'discrim_analysis_params', {'bin_width': 500, 'bin_num': 4})
        anova_bin_width = discrim_params['bin_width']
        anova_bin_num = discrim_params['bin_num']
        bin_lims = np.vectorize(int)(np.linspace(
            stim_time,
            stim_time + (anova_bin_num*anova_bin_width),
            anova_bin_num+1))

        seq_spikes_anova = self.sequestered_spikes_frame.copy()
        min_lim, max_lim = min(bin_lims), max(bin_lims)
        seq_spikes_anova = seq_spikes_anova.loc[
            (seq_spikes_anova.time_num >= min_lim) &
            (seq_spikes_anova.time_num < max_lim)
        ]
        seq_spikes_anova['bin_num'] = pd.cut(
            seq_spikes_anova.time_num,
            bin_lims,
            labels=np.arange(anova_bin_num),
            include_lowest=True,
        )
        seq_spikes_anova['spikes'] = 1
        # Since all bins are the same size, no need to normalize spike counts
        seq_spike_anova_counts = seq_spikes_anova.groupby(
            ['trial_num', 'neuron_num', 'taste_num', 'laser_tuple', 'bin_num']
        ).count().reset_index()
        seq_spike_anova_counts.drop(
            columns=['time_num'], inplace=True, errors='ignore')
        seq_spike_anova_counts.fillna(0, inplace=True)

        # Correct for missing zero-spike entries
        index_cols = ['trial_num', 'neuron_num', 'taste_num', 'laser_tuple']
        firing_frame_group_inds = list(
            self.sequestered_firing_frame.groupby(index_cols).groups.keys())

        firing_frame_group_inds = pd.DataFrame(
            firing_frame_group_inds, columns=index_cols)

        # Stack for all bins
        firing_frame_group_inds = pd.concat(
            [firing_frame_group_inds.assign(bin_num=bin_num)
             for bin_num in range(anova_bin_num)],
            ignore_index=True,
        )

        seq_spike_anova_counts = pd.merge(
            firing_frame_group_inds,
            seq_spike_anova_counts,
            on=index_cols + ['bin_num'],
            how='left',
        )
        seq_spike_anova_counts.fillna(0, inplace=True)

        # Calculate discriminability p-values
        discrim_pvals = {}
        dynamicity_pvals = {}
        group_cols = ['neuron_num', 'laser_tuple']
        for (nrn, laser), group in tqdm(seq_spike_anova_counts.groupby(group_cols)):
            anova_out = pg.anova(
                data=group,
                dv='spikes',
                between=['taste_num', 'bin_num'],
            )
            anova_out = anova_out.loc[anova_out.Source != 'Residual']
            taste_pval = anova_out.loc[
                anova_out.Source == 'taste_num', 'p-unc'].values
            bin_pval = anova_out.loc[
                anova_out.Source == 'bin_num', 'p-unc'].values
            taste_pval = taste_pval[0] if len(taste_pval) > 0 else 1.0
            bin_pval = bin_pval[0] if len(bin_pval) > 0 else 1.0
            discrim_pvals[(nrn, laser)] = taste_pval
            dynamicity_pvals[(nrn, laser)] = bin_pval

        return discrim_pvals, dynamicity_pvals

    def calculate_palatability(self, pal_ranks, firing_t_vec, stim_time):
        """
        Calculate unit palatability based on correlation with palatability rankings.

        Args:
            seq_firing_frame: DataFrame containing firing data.
            pal_ranks: List of palatability rankings for each taste.

        Returns:
            dict: A dictionary of palatability p-values for each unit.
        """

        if 'sequestered_firing_frame' not in dir(self):
            self.get_sequestered_data()

        print("="*40)
        print("Calculating unit palatability")
        print("="*40)

        seq_firing_frame = self.sequestered_firing_frame.copy()

        pal_pvals = {}
        seq_firing_frame['time_val'] = [firing_t_vec[x]
                                        for x in seq_firing_frame.time_num]
        seq_firing_frame['pal_rank'] = [pal_ranks[i]
                                        for i in seq_firing_frame.taste_num]
        group_cols = ['neuron_num', 'time_val', 'laser_tuple']
        for (nrn, time_val, laser), group in tqdm(seq_firing_frame.groupby(group_cols)):
            rho, pval = spearmanr(group.firing, group.pal_rank)
            pal_pvals[(nrn, laser, time_val)] = pval

        # Aggregate across time
        pal_pval_df = pd.DataFrame([
            {'neuron_num': key[0],
             'laser_tuple': key[1],
             'time_val': key[2],
             'p_val': val}
            for key, val in pal_pvals.items()
        ])

        if 'sorting_params_dict' not in dir(self):
            self.get_sorting_params_dict()

        pal_window = self.sorting_params_dict.get('palatability_window')
        pal_window = np.array(pal_window) + stim_time

        wanted_pal_pval_df = pal_pval_df.loc[
            (pal_pval_df.time_val >= pal_window[0]) &
            (pal_pval_df.time_val < pal_window[1])
        ]

        # Get median p-val across time window
        wanted_pal_pval_df = wanted_pal_pval_df.groupby(
            ['neuron_num', 'laser_tuple']
        ).median().reset_index()
        wanted_pal_pval_df.drop(columns=['time_val'], inplace=True)

        # Convert to dictionary
        wanted_pal_pval_df.set_index(
            ['neuron_num', 'laser_tuple'], inplace=True)
        pal_pvals = wanted_pal_pval_df['p_val'].to_dict()

        return pal_pvals, pal_pval_df

    def profile_units(self, save_to_file=True, alpha=0.05, recalculate=False):
        """
        Generate a DataFrame containing unit characteristics including:
        - Unit responsiveness (pre vs post-stimulus firing rate comparison)
        - Unit taste discrimination (ANOVA for taste differences)
        - Unit palatability (correlation with palatability rankings)
        - Unit drift/stability (from drift check results)
        """

        print("="*40)
        print("Unit Profiling")
        print("="*40)
        csv_path = os.path.join(self.data_dir, 'unit_profile.csv')
        if os.path.exists(csv_path) and not recalculate:
            print(f"Loading existing unit profile from {csv_path}")
            print("***To recalculate, set recalculate=True***")
            profile_df = pd.read_csv(csv_path)
            print('Storing unit profile in self.unit_profile')
            self.unit_profile = profile_df
            return

        # Ensure we have the necessary data loaded
        if not hasattr(self, 'sequestered_spikes_frame') or \
           not hasattr(self, 'sequestered_firing_frame'):
            self.get_sequestered_data()

        # Load info and params
        self.get_info_dict()
        self.get_sorting_params_dict()

        params_dict = self.sorting_params_dict
        info_dict = self.info_dict

        stim_time = params_dict['spike_array_durations'][0]

        # Get firing time vector
        firing_t_vec = np.arange(self.sequestered_firing_frame.time_num.max()+1) * \
            self.firing_rate_params['step_size']
        firing_t_vec += self.firing_rate_params['window_size']
        firing_t_vec -= stim_time

        # Calculate responsiveness
        resp_pvals = self.calculate_responsiveness(
            stim_time, params_dict)

        # Calculate discriminability
        discrim_pvals, dynamic_pvals = self.calculate_discriminability_dynamicity(
            stim_time, params_dict)

        # Calculate palatability
        pal_ranks = info_dict['taste_params'].get('pal_rankings', None)
        pal_pvals = self.calculate_palatability(
            pal_ranks, firing_t_vec, stim_time)[0]

        # Check stability
        if 'drift_results' not in dir(self):
            self.get_stable_units(p_val_threshold=alpha)

        # ==================== BUILD RESULTS DATAFRAME ====================
        results = []
        for nrn in self.sequestered_firing_frame.neuron_num.unique():
            for laser in self.sequestered_firing_frame.laser_tuple.unique():
                key = (nrn, laser)
                resp_p = resp_pvals.get(key, 1.0)
                discrim_p = discrim_pvals.get(key, 1.0)
                pal_p = pal_pvals.get(key, np.nan)
                stable = nrn in self.stable_units
                stable_pval = self.drift_results.loc[
                    self.drift_results['unit'] == nrn, 'p_val'].values[0]
                dynamic_p = dynamic_pvals.get(key, 1.0)
                dynamic = dynamic_p < alpha

                results.append({
                    'neuron_num': nrn,
                    'laser_tuple': laser,
                    'responsive': resp_p < alpha,
                    'responsive_pval': resp_p,
                    'discriminative': discrim_p < alpha,
                    'discriminative_pval': discrim_p,
                    'palatable': pal_p < alpha if not np.isnan(pal_p) else False,
                    'palatable_pval': pal_p,
                    'dynamic': dynamic,
                    'dynamic_pval': dynamic_p,
                    'stable': stable,
                    'stable_pval': stable_pval,
                })

        profile_df = pd.DataFrame(results)

        # Store as attribute
        print('Storing unit profile in self.unit_profile')
        self.unit_profile = profile_df

        # Save to file if requested
        if save_to_file:
            profile_df.to_csv(csv_path, index=False)
            print(f"Unit profile saved to {csv_path}")
