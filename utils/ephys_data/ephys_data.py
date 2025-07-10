"""
This module provides a class for streamlined electrophysiology data analysis, focusing on handling and analyzing data from multiple files. It includes features for automatic data loading, spike train and LFP data processing, firing rate calculation, digital input parsing, trial segmentation, region-based analysis, laser condition handling, and data quality checks.

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
"""
import os
import warnings
import numpy as np
import tables
import copy
import multiprocessing as mp
from scipy.special import gamma
from scipy.stats import zscore, spearmanr
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
    class SpikeProcessing:
        def get_spikes(self):
            """Extract spike arrays from specified HD5 files"""
            pass

        def separate_laser_spikes(self):
            """Separate spike arrays into laser on and off conditions"""
            pass

        def get_sequestered_spikes(self):
            """Sequester spikes into different categories"""
            pass

    class LFPProcessing:
        def extract_lfps(self):
            """Extract LFPs from raw data files and save to HDF5"""
            pass

        def get_lfp_channels(self):
            """Extract parsed LFP channels"""
            pass

        def get_lfps(self, re_extract=False):
            """Initiate LFP extraction or retrieve LFP arrays from HDF5"""
            pass

        def separate_laser_lfp(self):
            """Separate LFP arrays into laser on and off conditions"""
            pass

    class LaserConditionHandling:
        def separate_laser_data(self):
            """Separate data into laser on and off conditions"""
            pass

        def check_laser(self):
            """Check for the presence of laser trials"""
            pass

        def separate_laser_firing(self):
            """Separate firing rates into laser on and off conditions"""
            pass

    class RegionBasedAnalysis:
        def get_region_electrodes(self):
            # Existing code for get_region_electrodes
            pass

        def get_region_units(self):
            # Existing code for get_region_units
            pass

        def return_region_spikes(self, region_name='all'):
            # Existing code for return_region_spikes
            pass

        def get_region_firing(self, region_name='all'):
            # Existing code for get_region_firing
            pass

        def get_lfp_electrodes(self):
            # Existing code for get_lfp_electrodes
            pass

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
        """
        trial : 1D array
        max_freq : where to lob off the transform
        time_range_tuple : (start,end) in seconds, time_lims of spectrogram
                                from start of trial snippet`
        Fs : sampling rate
        signal_window : window size for spectrogram
        window_overlap : overlap between windows
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
        return Parallel(n_jobs=mp.cpu_count()-2)(delayed(func)(this_iter) for this_iter in tqdm(iterator))

    @staticmethod
    def _calc_conv_rates(step_size, window_size, dt, spike_array):
        """
        step_size
        window_size :: params :: In milliseconds. For moving window firing rate
                                calculation
        sampling_rate :: params :: In ms, To calculate total number of bins
        spike_array :: params :: N-D array with time as last dimension
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

        return firing_rate

    @staticmethod
    def _calc_baks_rate(resolution, dt, spike_array):
        """
        resolution : resolution of output firing rate (sec)
        dt : resolution of input spike trains (sec)
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
        return firing_rate_array

    @staticmethod
    def get_hdf5_path(data_dir):
        """
        Look for the hdf5 file in the directory
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
        if path_to_node in hf5:
            hf5.remove_node(
                os.path.dirname(path_to_node), os.path.basename(path_to_node))

    ####################
    # Initialize instance
    ###################

    def __init__(self,
                 data_dir=None):
        """
        data_dirs : where to look for hdf5 file
            : get_data() loads data from this directory
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

    class SpikeProcessing:
        def get_spikes(self):
            """Extract spike arrays from specified HD5 files"""
            pass

        def separate_laser_spikes(self):
            """Separate spike arrays into laser on and off conditions"""
            pass

        def get_sequestered_spikes(self):
            """Sequester spikes into different categories"""
            pass

    class LFPProcessing:
        def extract_lfps(self):
            """Extract LFPs from raw data files and save to HDF5"""
            pass

        def get_lfp_channels(self):
            """Extract parsed LFP channels"""
            pass

        def get_lfps(self, re_extract=False):
            """Initiate LFP extraction or retrieve LFP arrays from HDF5"""
            pass

        def separate_laser_lfp(self):
            """Separate LFP arrays into laser on and off conditions"""
            pass

    class LaserConditionHandling:
        def separate_laser_data(self):
            """Separate data into laser on and off conditions"""
            pass

        def check_laser(self):
            """Check for the presence of laser trials"""
            pass

        def separate_laser_firing(self):
            """Separate firing rates into laser on and off conditions"""
            pass

    class RegionBasedAnalysis:
        def get_region_electrodes(self):
            # Existing code for get_region_electrodes

        def get_region_units(self):
            # Existing code for get_region_units

        def return_region_spikes(self, region_name='all'):
            # Existing code for return_region_spikes

        def get_region_firing(self, region_name='all'):
            # Existing code for get_region_firing

        def get_lfp_electrodes(self):
            # Existing code for get_lfp_electrodes

    def get_stft(
            self,
            recalculate=False,
            dat_type=['amplitude'],
            write_out=True):
        """
        If STFT present in HDF5 then retrieve it
        If not, then calculate it and save it into HDF5 file

        Inputs:
            recalculate: bool, if True then recalculate STFT
            dat_type: list of strings, options are 'raw', 'amplitude', 'phase'
            write_out: bool, if True then write out STFT to HDF5 file
        """

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
        """
        Return list containing LFPs for each region and region names
        """
        if 'lfp_array' not in dir(self):
            self.get_lfps()
        if 'lfp_region_electrodes' not in dir(self):
            self.get_lfp_electrodes()
        region_lfp = [self.lfp_array[:, x, :, :]
                      for x in self.lfp_region_electrodes]
        return region_lfp, self.region_names

    def return_representative_lfp_channels(self):
        """
        Return one electrode per region that is closest to the mean
        """
        # Region lfps shape : (n_tastes, n_channels, n_trials, n_timepoints)
        region_lfps, region_names = self.return_region_lfps()

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
        self.trial_info_frame = pd.read_csv(
            os.path.join(self.data_dir, 'trial_info_frame.csv'))

    def sequester_trial_inds(self):
        """
        Sequester trials into different categories:
            - Tastes
            - Laser conditions
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
        """
        Sequester spikes into different categories:
            - Tastes
            - Laser conditions
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
        """
        Sequester spikes into different categories:
            - Tastes
            - Laser conditions
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
        >>> data.get_stable_units('/path/to/drift_results.csv')
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
