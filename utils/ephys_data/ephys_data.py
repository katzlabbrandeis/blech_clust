import os
import warnings
import numpy as np
import tables
import copy
import multiprocessing as mp
from scipy.special import gamma
from scipy.stats import zscore
import scipy, scipy.signal
import glob
import json
import easygui
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm
from itertools import product
from .BAKS import BAKS
from . import lfp_processing

#  ______       _                      _____        _         
# |  ____|     | |                    |  __ \      | |        
# | |__   _ __ | |__  _   _ ___ ______| |  | | __ _| |_ __ _  
# |  __| | '_ \| '_ \| | | / __|______| |  | |/ _` | __/ _` | 
# | |____| |_) | | | | |_| \__ \      | |__| | (_| | || (_| | 
# |______| .__/|_| |_|\__, |___/      |_____/ \__,_|\__\__,_| 
#        | |           __/ |                                  
#        |_|          |___/ 

"""
Class to streamline data analysis from multiple files
Class has a container for data from different files and functions for analysis
Functions in class can autoload data from specified files according to specified 
paramters
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
        """
        trial : 1D array
        max_freq : where to lob off the transform
        time_range_tuple : (start,end) in seconds, time_lims of spectrogram
                                from start of trial snippet`
        Fs : sampling rate
        signal_window : window size for spectrogram
        window_overlap : overlap between windows
        """
        f,t,this_stft = scipy.signal.stft(
                    scipy.signal.detrend(trial),
                    fs=Fs,
                    window='hann',
                    nperseg=signal_window,
                    noverlap=signal_window-(signal_window-window_overlap))
        this_stft =  this_stft[np.where(f<max_freq)[0]]
        this_stft = this_stft[:,np.where((t>=time_range_tuple[0])*\
                                                (t<time_range_tuple[1]))[0]]
        fin_freq = f[f<max_freq]
        fin_t = t[np.where((t>=time_range_tuple[0])*(t<time_range_tuple[1]))]
        return  fin_freq, fin_t, this_stft
                            
    # Calculate absolute and phase
    @staticmethod
    def parallelize(func, iterator):
        return Parallel(n_jobs = mp.cpu_count()-2)\
                (delayed(func)(this_iter) for this_iter in tqdm(iterator))


    @staticmethod
    def _calc_conv_rates(step_size, window_size, dt , spike_array):
        """
        step_size 
        window_size :: params :: In milliseconds. For moving window firing rate
                                calculation
        sampling_rate :: params :: In ms, To calculate total number of bins 
        spike_array :: params :: N-D array with time as last dimension
        """

        if np.sum([step_size % dt , window_size % dt]) > 1e-14:
            raise Exception('Step size and window size must be integer multiples'\
                    ' of the inter-sample interval')

        fin_step_size, fin_window_size = \
            int(step_size/dt), int(window_size/dt)
        total_time = spike_array.shape[-1]

        bin_inds = (0,fin_window_size)
        total_bins = int((total_time - fin_window_size + 1) / fin_step_size) + 1
        bin_list = [(bin_inds[0]+step,bin_inds[1]+step) \
                for step in np.arange(total_bins)*fin_step_size ]

        firing_rate = np.empty((spike_array.shape[0],
                                spike_array.shape[1],total_bins))

        for bin_inds in bin_list:
            firing_rate[...,bin_inds[0]//fin_step_size] = \
                    np.sum(spike_array[...,bin_inds[0]:bin_inds[1]], axis=-1)

        return firing_rate

    @staticmethod
    def _calc_baks_rate(resolution, dt, spike_array):
        """
        resolution : resolution of output firing rate (sec)
        dt : resolution of input spike trains (sec)
        """
        t = np.arange(0,spike_array.shape[-1]*dt, resolution)
        array_inds = list(np.ndindex((spike_array.shape[:-1])))
        spike_times = [np.where(spike_array[this_inds])[0]*dt \
                for this_inds in array_inds]

        firing_rates = [BAKS(this_spike_times,t) \
                for this_spike_times in tqdm(spike_times)]

        # Put back into array
        firing_rate_array = np.zeros((*spike_array.shape[:-1],len(t)))
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
            raise Exception('No HDF5 file detected')
        elif len(hdf5_path) > 1:
            selection_list = ['{}) {} \n'.format(num,os.path.basename(file)) \
                    for num,file in enumerate(hdf5_path)]
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
        temp_array  =\
                np.empty(
                    tuple((*(np.max(np.array(iter_inds),axis=0) + 1),
                            *iterator[0].shape)),
                        dtype=np.dtype(iterator[0].flatten()[0]))
        for iter_num, this_iter in enumerate(tqdm(iter_inds)):
            temp_array[this_iter] = iterator[iter_num]
        return temp_array

    @staticmethod
    def remove_node(path_to_node, hf5):
        if path_to_node in hf5:
            hf5.remove_node(
                    os.path.dirname(path_to_node),os.path.basename(path_to_node))

    ####################
    # Initialize instance
    ###################

    def __init__(self, 
            data_dir = None):
        
        """
        data_dirs : where to look for hdf5 file
            : get_data() loads data from this directory
        """
        self.lfp_processing = lfp_processing

        if data_dir is None:
            self.data_dir = easygui.diropenbox(
                    'Please select directory with HDF5 file')
        else:
            self.data_dir =     data_dir
            self.hdf5_path =    self.get_hdf5_path(data_dir) 
            self.hdf5_name =    os.path.basename(self.hdf5_path) 

            self.spikes = None

        # Create environemnt variable to allow program to know
        # if file is currently accessed
        # Created for multiprocessing of fits
        #os.environ[self.hdf5_name]="0"
        
        self.firing_rate_params = {
            'type'          :   None,
            'step_size'     :   None,
            'window_size'   :   None,
            'dt'            :   None,
            'baks_resolution' : None,
            'baks_dt'       : None
                }

        self.default_firing_params = {
            'type'          :   'conv',
            'step_size'     :   25,
            'window_size'   :   250,
            'dt'            :   1,
            'baks_resolution' : 25e-3,
            'baks_dt'       :   1e-3
                }

        self.default_lfp_params = {
            'freq_bounds' : [1,300],
            'sampling_rate' : 30000,
            'taste_signal_choice': 'Start',
            'fin_sampling_rate' : 1000,
            'trial_durations' : [2000,5000]
                }


        # Resolution has to be increased for phase of higher frequencies
        # Can be passed as kwargs to "calc_stft"
        self.stft_params = {
                'Fs' : 1000, 
                'signal_window' : 500,
                'window_overlap' : 499,
                'max_freq' : 20,
                'time_range_tuple' : (0,5)
                }
         
    #class access:
    #    def __init__(self, key_name):
    #        os.environ[key_name] = '0'

    #    def check(self):
    #        access_bool = 

    def extract_and_process(self):
        self.get_unit_descriptors()
        self.get_spikes()
        self.get_firing_rates()
        self.get_lfps()

    def separate_laser_data(self):
        self.separate_laser_spikes()
        self.separate_laser_firing()
        self.separate_laser_lfp()


    def get_unit_descriptors(self):
        """
        Extract unit descriptors from HDF5 file
        """
        with tables.open_file(self.hdf5_path, 'r+') as hf5_file:
            self.unit_descriptors = hf5_file.root.unit_descriptor[:]

    def check_laser(self):
        with tables.open_file(self.hdf5_path, 'r+') as hf5: 
            dig_in_list = \
                [x for x in hf5.list_nodes('/spike_trains') \
                if 'dig_in' in x.__str__()]

            # Mark whether laser exists or not
            self.laser_durations_exists = sum([dig_in.__contains__('laser_durations') \
                for dig_in in dig_in_list]) > 0
        
            # If it does, pull out laser durations
            if self.laser_durations_exists:
                self.laser_durations = [dig_in.laser_durations[:] \
                        for dig_in in dig_in_list]

                non_zero_laser_durations = np.sum(self.laser_durations) > 0

            # If laser_durations exists, only non_zero durations 
            # will indicate laser
            # If it doesn't exist, then mark laser as absent
            if self.laser_durations_exists:
                if non_zero_laser_durations:
                    self.laser_exists = True
                else:
                    self.laser_exists = False
            else:
                self.laser_exists = False

    def get_spikes(self):
        """
        Extract spike arrays from specified HD5 files
        """
        with tables.open_file(self.hdf5_path, 'r+') as hf5: 
            if '/spike_trains' in hf5:
                dig_in_list = \
                    [x for x in hf5.list_nodes('/spike_trains') \
                    if 'dig_in' in x.__str__()]
            else:
                raise Exception('No spike trains found in HF5')
            
            self.spikes = [dig_in.spike_array[:] for dig_in in dig_in_list]

    def separate_laser_spikes(self):
        """
        Separate spike arrays into laser on and off conditions
        """
        if 'laser_exists' not in dir(self):
            self.check_laser()
        if 'spikes' not in dir(self):
            self.get_spikes()
        if self.laser_exists:
            self.on_spikes = np.array([taste[laser>0] for taste,laser in \
                    zip(self.spikes,self.laser_durations)])
            self.off_spikes = np.array([taste[laser==0] for taste,laser in \
                    zip(self.spikes,self.laser_durations)])
        else:
            raise Exception('No laser trials in this experiment')

    def extract_lfps(self):
        """
        Wrapper function to extract LFPs from raw data files and save to HDF5
        Loads relevant information for .info file
        """
        json_path = glob.glob(os.path.join(self.data_dir, "**.info"))[0] 
        if os.path.exists(json_path):
            json_dict = json.load(open(json_path,'r'))
            taste_dig_ins = json_dict['taste_params']['dig_ins'] 
        else:
            raise Exception("Cannot find json file. Make sure it's present")
        # Add final argument to argument list
        self.default_lfp_params.update({'dig_in_list': taste_dig_ins})
        lfp_processing.extract_lfps(self.data_dir, **self.default_lfp_params)

    def get_lfp_channels(self):
        """
        Extract Parsed_LFP_channels
        This is done separately from "get_lfps" to avoid
        the overhead of reading the large lfp arrays
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

    def get_lfps(self, re_extract = False):
        """
        Wrapper function to either
        - initiate LFP extraction, or
        - pull LFP arrays from HDF5 file
        """
        with tables.open_file(self.hdf5_path, 'r+') as hf5: 

            if ('/Parsed_LFP' not in hf5) or (re_extract == True):
                extract_bool = True
            else:
                extract_bool = False

        if extract_bool:
            self.extract_lfps()

        with tables.open_file(self.hdf5_path, 'r+') as hf5: 
            lfp_nodes = [node for node in hf5.list_nodes('/Parsed_LFP')\
                    if 'dig_in' in node.__str__()]
            # Account for parsed LFPs being different 
            self.lfp_array = np.asarray([node[:] for node in lfp_nodes])
            self.all_lfp_array = \
                    self.lfp_array.\
                        swapaxes(1,2).\
                        reshape(-1, self.lfp_array.shape[1],\
                                self.lfp_array.shape[-1]).\
                        swapaxes(0,1)


    def separate_laser_lfp(self):
        """
        Separate spike arrays into laser on and off conditions
        """
        if 'laser_exists' not in dir(self):
            self.check_laser()
        if 'lfp_array' not in dir(self):
            self.get_lfps()
        if self.laser_exists:
            self.on_lfp = np.array([taste.swapaxes(0,1)[laser>0] \
                    for taste,laser in \
                    zip(self.lfp_array,self.laser_durations)])
            self.off_lfp = np.array([taste.swapaxes(0,1)[laser==0] \
                    for taste,laser in \
                    zip(self.lfp_array,self.laser_durations)])
            self.all_on_lfp =\
                np.reshape(self.on_lfp,(-1,*self.on_lfp.shape[-2:]))
            self.all_off_lfp =\
                np.reshape(self.off_lfp,(-1,*self.off_lfp.shape[-2:]))
        else:
            raise Exception('No laser trials in this experiment')

    def firing_rate_method_selector(self):
        params = self.firing_rate_params

        type_exists_bool = 'type' in params.keys()
        if not type_exists_bool:
            raise Exception('Firing rate calculation type not specified.'\
                    '\nPlease use: \n {}'.format('\n'.join(type_list)))
        type_list = ['conv','baks']
        if params['type'] not in type_list:
            raise Exception('Firing rate calculation type not recognized.'\
                    '\nPlease use: \n {}'.format('\n'.join(type_list)))

        def check_firing_rate_params(params, param_name_list):
            param_exists_bool = [True if x in params.keys() else False \
                    for x in param_name_list]
            if not all(param_exists_bool):
                raise Exception('All required firing rate parameters'\
                        ' have not been specified \n{}'.format(\
                        '\n'.join(map(str,\
                        list(zip(param_exists_bool,param_name_list))))))
            param_present_bool = [params[x] is not None \
                    for x in param_name_list]
            if not all(param_present_bool):
                raise Exception('All required firing rate parameters'\
                        ' have not been specified \n{}'.format(\
                        '\n'.join(map(str,\
                        list(zip(param_present_bool,param_name_list))))))

        if params['type'] == 'conv':
            param_name_list = ['step_size','window_size','dt']
            
            # This checks if anything is missing
            # And raises exception if anything missing
            check_firing_rate_params(params,param_name_list)

            # If all good, define the function to be used
            def calc_firing_func(data):
                firing_rate = \
                        self._calc_conv_rates(\
                        step_size = self.firing_rate_params['step_size'],
                        window_size = self.firing_rate_params['window_size'],
                        dt = self.firing_rate_params['dt'],
                        spike_array = data)
                return firing_rate

        if params['type'] == 'baks':
            param_name_list = ['baks_resolution','baks_dt']
            check_firing_rate_params(params,param_name_list)
            def calc_firing_func(data):
                firing_rate = \
                        self._calc_baks_rate(\
                        resolution = self.firing_rate_params['baks_resolution'],
                        dt = self.firing_rate_params['baks_dt'],
                        spike_array = data)
                return firing_rate

        return calc_firing_func

    def get_firing_rates(self):
        """
        Converts spikes to firing rates
        """
        
        if self.spikes is None:
            raise Exception('Run method "get_spikes" first')
        if None in self.firing_rate_params.values():
            raise Exception('Specify "firing_rate_params" first')

        calc_firing_func = self.firing_rate_method_selector()
        self.firing_list = [calc_firing_func(spikes) for spikes in self.spikes]
        #self.firing_list = [self._calc_conv_rates(
        #    step_size = self.firing_rate_params['step_size'],
        #    window_size = self.firing_rate_params['window_size'],
        #    dt = self.firing_rate_params['dt'],
        #    spike_array = spikes)
        #                    for spikes in self.spikes]
        
        if np.sum([self.firing_list[0].shape == x.shape \
                for x in self.firing_list]) == len(self.firing_list):
            print('All tastes have equal dimensions,' \
                    'concatenating and normalizing')
            
            # Reshape for backward compatiblity
            self.firing_array = np.asarray(self.firing_list).swapaxes(1,2)
            # Concatenate firing across all tastes
            self.all_firing_array = \
                    self.firing_array.\
                        swapaxes(1,2).\
                        reshape(-1, self.firing_array.shape[1],\
                                self.firing_array.shape[-1]).\
                        swapaxes(0,1)
            
            # Calculate normalized firing
            min_vals = [np.min(self.firing_array[:,nrn,:,:],axis=None) \
                    for nrn in range(self.firing_array.shape[1])] 
            max_vals = [np.max(self.firing_array[:,nrn,:,:],axis=None) \
                    for nrn in range(self.firing_array.shape[1])] 
            self.normalized_firing = np.asarray(\
                    [(self.firing_array[:,nrn,:,:] - min_vals[nrn])/\
                        (max_vals[nrn] - min_vals[nrn]) \
                        for nrn in range(self.firing_array.shape[1])]).\
                        swapaxes(0,1)

            # Concatenate normalized firing across all tastes
            self.all_normalized_firing = \
                    self.normalized_firing.\
                        swapaxes(1,2).\
                        reshape(-1, self.normalized_firing.shape[1],\
                                self.normalized_firing.shape[-1]).\
                        swapaxes(0,1)

        else:
            raise Exception('Cannot currently handle different'\
                    'numbers of trials')

    def separate_laser_firing(self):
        """
        Separate spike arrays into laser on and off conditions
        """
        if 'laser_exists' not in dir(self):
            self.check_laser()
        if 'firing_array' not in dir(self):
            self.get_firing_rates()
        if self.laser_exists:
            self.on_firing = np.array([taste[laser>0] for taste,laser in \
                    zip(self.firing_list,self.laser_durations)])
            self.off_firing = np.array([taste[laser==0] for taste,laser in \
                    zip(self.firing_list,self.laser_durations)])
            self.all_on_firing =\
                np.reshape(self.on_firing,(-1,*self.on_firing.shape[-2:]))
            self.all_off_firing =\
                np.reshape(self.off_firing,(-1,*self.off_firing.shape[-2:]))
        else:
            raise Exception('No laser trials in this experiment')

    def get_info_dict(self):
        json_path = glob.glob(os.path.join(self.data_dir, "**.info"))[0] 
        if os.path.exists(json_path):
            self.info_dict = json_dict = json.load(open(json_path,'r'))
        else:
            raise Exception('No info file found')

    def get_region_electrodes(self):
        """
        If the appropriate json file is present in the data_dir,
        extract the electrodes for each region
        """
        #json_name = self.hdf5_path.split('.')[0] + '.info'
        #json_path = os.path.join(self.data_dir, json_name)
        json_path = glob.glob(os.path.join(self.data_dir, "**.info"))[0] 
        if os.path.exists(json_path):
            json_dict = json.load(open(json_path,'r'))
            self.region_electrode_dict = json_dict["electrode_layout"]
            self.region_names = [x for x in self.region_electrode_dict.keys() \
                    if 'emg' not in x]
        else:
            raise Exception("Cannot find json file. Make sure it's present")

    def get_region_units(self):
        """
        Extracts indices of units by region of electrodes
        `"""
        if "region_electrode_dict" not in dir(self): 
            self.get_region_electrodes()
        if "unit_descriptors" not in dir(self):
            self.get_unit_descriptors()

        unit_electrodes = [x['electrode_number'] for x in self.unit_descriptors]
        region_electrode_vals = [val for key,val in \
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
            for elec_num,elec in enumerate(unit_electrodes):
                if elec in val:
                    # This tells you which car group each neuron is in
                    car_ind_vec[elec_num] = num 

        self.car_units = [np.where(car_ind_vec == x)[0] \
                for x in np.unique(car_ind_vec)]

        region_ind_vec = np.zeros(len(unit_electrodes))
        for elec_num,elec in enumerate(unit_electrodes):
            for region_num, region in enumerate(region_electrode_vals):
                for car in region:
                    if elec in car:
                        region_ind_vec[elec_num] = region_num

        self.region_units = [np.where(region_ind_vec == x)[0] \
                for x in np.unique(region_ind_vec)]

    def return_region_spikes(self, region_name = 'all'):
        if 'region_names' not in dir(self):
            self.get_region_units()
        if self.spikes is None:
            self.get_spikes()

        if not region_name == 'all':
            region_ind = [num for num,x in enumerate(self.region_names) \
                    if x == region_name]
            if not len(region_ind) == 1 :
                raise Exception('Region name not found, or too many matches found, '\
                'acceptable options are' + \
                '\n' + f"===> {self.region_names, 'all'}")
            else:
                this_region_units = self.region_units[region_ind[0]]
                region_spikes = [x[:,this_region_units] for x in self.spikes] 
                return np.array(region_spikes)
        else:
            return np.array(self.spikes)

    def get_region_firing(self, region_name = 'all'):
        if 'region_units' not in dir(self):
            self.get_region_units()
        if 'firing_array' not in dir(self):
            self.get_firing_rates()

        if not region_name == 'all':
            region_ind = [num for num,x in enumerate(self.region_names) \
                    if x == region_name]
            if not len(region_ind) == 1 :
                raise Exception('Region name not found, or too many matches found, '\
                'acceptable options are' + \
                '\n' + f"===> {self.region_names, 'all'}")
            else:
                this_region_units = self.region_units[region_ind[0]]
                region_firing = [x[this_region_units] for x in self.firing_array] 
                return np.array(region_firing)
        else:
            return np.array(self.firing_array)


    def get_lfp_electrodes(self):
        """
        Extracts indices of lfp_electrodes according to region
        """
        if 'parsed_lfp_channels' not in dir(self):
            self.get_lfp_channels()
        if 'region_electrode_dict' not in dir(self):
            self.get_region_electrodes()

        region_electrode_vals = [val for key,val in \
                self.region_electrode_dict.items() if key != 'emg']
        region_ind_vec = np.zeros(len(self.parsed_lfp_channels))
        for elec_num,elec in enumerate(self.parsed_lfp_channels):
            for region_num, region in enumerate(region_electrode_vals):
                for car in region:
                    if elec in car:
                        region_ind_vec[elec_num] = region_num

        self.lfp_region_electrodes = [np.where(region_ind_vec == x)[0] \
                for x in np.unique(region_ind_vec)]
        
    def get_stft(
            self, 
            recalculate = False, 
            dat_type = ['amplitude'],
            write_out = True):
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
            with tables.open_file(self.hdf5_path,'r+') as hf5:
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
                    hf5.create_group('/','stft')
                    hf5.flush()
        else:
            self.calc_stft_bool = 1

        if self.calc_stft_bool:
            print('Calculating STFT' )

            # Get LFPs to calculate STFT
            if "lfp_array" not in dir(self):
                self.get_lfps()

            # Generate list of individual trials to be fed into STFT function
            stft_iters = list(
                    product(
                        *list(map(np.arange,self.lfp_array.shape[:3]))
                        )
                    )

            # Calculate STFT over lfp array
            try:
                stft_list = Parallel(n_jobs = mp.cpu_count()-2)\
                        (delayed(self.calc_stft)(self.lfp_array[this_iter],
                                            **self.stft_params)\
                        for this_iter in tqdm(stft_iters))
            except:
                warnings.warn("Couldn't process STFT in parallel."\
                        "Running serial loop")
            #stft_list = [self.calc_stft(self.lfp_array[this_iter],
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
            self.amplitude_array = self.convert_to_array(amplitude_list, stft_iters)**2
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

                with tables.open_file(self.hdf5_path,'r+') as hf5:
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
        region_lfp = [self.lfp_array[:,x,:,:] for x in self.lfp_region_electrodes]
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
            diff_lfp = np.abs(this_region - this_mean[:,np.newaxis,:,:])
            mean_diff_lfp = diff_lfp.mean(axis=(0,2,3))
            min_diff_lfp = np.argmin(mean_diff_lfp)
            wanted_channel_inds.append(min_diff_lfp)

        wanted_lfp_electrodes = np.array([x[:,y] \
                for x,y in zip(region_lfps, wanted_channel_inds)])
        return wanted_channel_inds, wanted_lfp_electrodes, region_names

    def get_mean_stft_amplitude(self):
        if 'amplitude_array' not in dir(self):
            self.get_stft()
        if 'lfp_region_electrodes' not in dir(self):
            self.get_lfp_electrodes()
        
        aggregate_amplitude = []
        for region in self.lfp_region_electrodes:
           aggregate_amplitude.append(\
                   np.median(self.amplitude_array[:,region],axis=(0,1,2)))
        return np.array(aggregate_amplitude)

