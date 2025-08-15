"""
This module provides functionality for handling digital inputs and reading data from Intan format files, saving the data into HDF5 format. It includes a class for managing digital inputs and several functions for reading and processing data.

- `DigInHandler` class: Manages digital input files across different formats.
  - `__init__`: Initializes the handler with a data directory and file type.
  - `get_dig_in_files`: Retrieves digital input files and their metadata based on the specified file type.
  - `get_trial_data`: Extracts trial data (start and end times) from digital input files.
  - `write_out_frame`: Saves the digital input data frame to a CSV file.
  - `load_dig_in_frame`: Loads the digital input data frame from a CSV file.

- `read_traditional_intan`: Reads traditional Intan format data and saves it to an HDF5 file, organizing amplifier and EMG data.

- `read_emg_channels`: Reads EMG data from amplifier channels and saves it to an HDF5 file.

- `read_electrode_channels`: Reads electrode data from amplifier channels, excluding EMG channels, and saves it to an HDF5 file.

- `read_electrode_emg_channels_single_file`: Reads both electrode and EMG data from a single file and saves it to an HDF5 file, organizing data by channel index.
"""
import tables
import os
import numpy as np
from tqdm import tqdm
import pandas as pd

from utils.importrhdutilities import load_file, read_header

class IntanDataHandler:
    def __init__(self, hdf5_name, electrode_layout_frame):
        self.hdf5_name = hdf5_name
        self.electrode_layout_frame = electrode_layout_frame

    def _read_data_from_file(self, filename):
        return np.fromfile(filename, dtype=np.dtype('int16'))

    def _create_earray(self, group, array_name, atom):
        hf5 = tables.open_file(self.hdf5_name, 'r+')
        if os.path.join(group, array_name) not in hf5:
            return hf5.create_earray(group, array_name, atom, (0,))
        else:
            return hf5.get_node(group, array_name)

    def read_traditional_intan(self, file_list):
        atom = tables.IntAtom()
        hf5 = tables.open_file(self.hdf5_name, 'r+')

        pbar = tqdm(total=len(file_list))
        for this_file in file_list:
            pbar.set_description(os.path.basename(this_file))
            this_file_data, data_present = load_file(this_file)
            for i, this_amp in enumerate(this_file_data['amplifier_data']):
                if 'emg' in self.electrode_layout_frame.loc[i].CAR_group.lower():
                    array_name = f'emg{i:02}'
                    hf5_el_array = self._create_earray(
                        '/raw_emg', array_name, atom)
                    hf5_el_array.append(this_amp)
                elif self.electrode_layout_frame.loc[i].CAR_group.lower() not in ['none', 'na']:
                    array_name = f'electrode{i:02}'
                    hf5_el_array = self._create_earray(
                        '/raw', array_name, atom)
                    hf5_el_array.append(this_amp)
                hf5.flush()
            pbar.update(1)
        pbar.close()
        hf5.close()

    def read_emg_channels(self):
        atom = tables.IntAtom()
        hf5 = tables.open_file(self.hdf5_name, 'r+')
        for num, row in tqdm(self.electrode_layout_frame.iterrows()):
            if 'emg' in row.CAR_group.lower():
                print(f'Reading : {row.filename, row.CAR_group}')
                data = self._read_data_from_file(row.filename)
                array_name = f'emg{row.electrode_ind:02}'
                hf5_el_array = self._create_earray(
                    '/raw_emg', array_name, atom)
                hf5_el_array.append(data)
                hf5.flush()
        hf5.close()

    def read_electrode_channels(self):
        atom = tables.IntAtom()
        hf5 = tables.open_file(self.hdf5_name, 'r+')
        for num, row in tqdm(self.electrode_layout_frame.iterrows()):
            emg_bool = 'emg' not in row.CAR_group.lower()
            none_bool = row.CAR_group.lower() not in ['none', 'na']
            if emg_bool and none_bool:
                print(f'Reading : {row.filename, row.CAR_group}')
                data = self._read_data_from_file(row.filename)
                array_name = f'electrode{row.electrode_ind:02}'
                hf5_el_array = self._create_earray('/raw', array_name, atom)
                hf5_el_array.append(data)
                hf5.flush()
        hf5.close()

    def read_electrode_emg_channels_single_file(self, electrodes_list, num_recorded_samples, emg_channels):
        hf5 = tables.open_file(self.hdf5_name, 'r+')
        atom = tables.IntAtom()
        amplifier_data = self._read_data_from_file(electrodes_list[0])
        num_electrodes = int(len(amplifier_data) / num_recorded_samples)
        amp_reshape = np.reshape(amplifier_data, (int(
            len(amplifier_data) / num_electrodes), num_electrodes)).T
        for num, row in tqdm(self.electrode_layout_frame.iterrows()):
            emg_bool = 'emg' not in row.CAR_group.lower()
            none_bool = row.CAR_group.lower() not in ['none', 'na']
            if emg_bool and none_bool:
                print(f'Reading : {row.filename, row.CAR_group}')
                el = self._create_earray(
                    '/raw', f'electrode{row.electrode_ind:02}', atom)
                exec(
                    f"hf5.root.raw.electrode{row.electrode_ind:02}.append(amp_reshape[num,:])")
                hf5.flush()
            elif not emg_bool and none_bool:
                el = self._create_earray(
                    '/raw_emg', f'emg{row.electrode_ind:02}', atom)
                exec(
                    f"hf5.root.raw_emg.emg{row.electrode_ind:02}.append(amp_reshape[num,:])")
        hf5.close()


class DigInHandler:
    """
    Class to unify handling of digital inputs across different file formats

    Methods:
            get_dig_in_files: Get digital input
            get_trial_data: Get trial data (start or end times)

    """

    def __init__(self, data_dir, file_type):
        """
        Initializes DigInHandler object

        Input:
                data_dir: str
                        Directory containing digital input files
                file_format: str
                        Format of digital input files
        """
        self.data_dir = data_dir
        self.file_type = file_type

    def get_dig_in_files(self):
        """
        Get digital input files
        Keep track of:
                1- Actual filename, this would be :
                        - .dat for one-file-per-channel
                        - digitalin.dat for one-file-per-signal-type
                        - all .rhd files for traditional
                2- dig-in name
                        - str before .dat for one-file-per-channel
                        - ??? for one-file-per-signal-type
                        - 'board_dig_in_channels' for traditional
                3- The dig-in number (NOT the index)

        Output:
                dig_in_files: list
                        List of digital input files
        """

        file_list = os.listdir(self.data_dir)

        if self.file_type == 'one file per signal type':
            dig_in_file_list = ['digitalin.dat']
            dig_in_name = 'digitalin'
            dig_in_num = []
        elif self.file_type == 'one file per channel':
            dig_in_file_list = [
                name for name in file_list if name.startswith('board-DI')]
            dig_in_name = [x.split('.')[0] for x in dig_in_file_list]
            dig_in_num = [int(x.split('-')[-1]) for x in dig_in_name]

            # Sort by dig-in number
            sort_inds = np.argsort(dig_in_num)
            dig_in_file_list = [dig_in_file_list[x] for x in sort_inds]
            dig_in_name = [dig_in_name[x] for x in sort_inds]
            dig_in_num = [dig_in_num[x] for x in sort_inds]

        else:
            rhd_file_list = [x for x in file_list if 'rhd' in x]
            with open(os.path.join(self.data_dir, rhd_file_list[0]), 'rb') as f:
                header = read_header(f)
            dig_in_file_list = sorted([x for x in rhd_file_list if 'rhd' in x])
            dig_in_name = [x['native_channel_name'].lower()
                           for x in header['board_dig_in_channels']]
            dig_in_num = [int(x.split('-')[-1]) for x in dig_in_name]

            # Sort by dig-in number
            sort_inds = np.argsort(dig_in_num)
            dig_in_name = [dig_in_name[x] for x in sort_inds]
            dig_in_num = [dig_in_num[x] for x in sort_inds]

        self.file_list = file_list
        self.dig_in_file_list = dig_in_file_list
        self.dig_in_name = dig_in_name
        self.dig_in_num = dig_in_num
        dig_in_file_str = '\n'.join(dig_in_file_list)
        dig_in_num_str = ', '.join([str(x) for x in dig_in_num])
        print(f'Digital input files found: \n{dig_in_file_str}')
        print(f'Digital input numbers found: \n{dig_in_num_str}')

    def get_trial_data(self):
        """
        Get trial data (start or end times)
        NOTE: Keep track of dig-ins using their indices, but also save their filenames

        Input:
                trial_type: str
                        Type of trial data to get (start or end)

        Output:
                trial_data: list
                        List of trial data
        """
        if self.file_type == 'one file per channel':
            pulse_times = {}
            for this_dig_num, filename in zip(self.dig_in_num, self.dig_in_file_list):
                dig_inputs = np.array(np.fromfile(
                    os.path.join(self.data_dir, filename),
                    dtype=np.dtype('uint16')))
                dig_inputs = dig_inputs.astype('int')

                d_diff = np.ediff1d(dig_inputs)
                start_ind = np.where(d_diff == 1)[0]
                end_ind = np.where(d_diff == -1)[0]
                pulse_times[this_dig_num] = list(zip(start_ind, end_ind))

        elif self.file_type == 'one file per signal type':
            all_dig_ins = np.fromfile(
                os.path.join(self.data_dir, self.dig_in_file_list[0]),
                dtype=np.dtype('uint16'))[:]
            all_dig_ins = all_dig_ins.astype('int')
            pulse_times = {}
            for i, dig_inputs in enumerate(all_dig_ins):
                d_diff = np.diff(dig_inputs)
                start_ind = np.where(d_diff == 1)[0]
                end_ind = np.where(d_diff == -1)[0]
                pulse_times[i] = list(zip(start_ind, end_ind))

        elif self.file_type == 'traditional':

            pulse_times = {}
            counter = 0
            for this_file in tqdm(self.dig_in_file_list):
                this_file_data, data_present = load_file(
                    os.path.join(self.data_dir, this_file))
                dig_inputs = this_file_data['board_dig_in_data']
                dig_inputs = dig_inputs.astype('int')
                # Shape: (num_dig_ins, num_samples)
                d_diff = np.diff(dig_inputs, axis=-1)
                start_ind = np.where(d_diff == 1)
                end_ind = np.where(d_diff == -1)

                # Need to keep track of how many files have been read
                # to account for pulse index (as pulses detected are simply from the
                # start of single files)
                start_ind = (start_ind[0], start_ind[1] + counter)
                end_ind = (end_ind[0], end_ind[1] + counter)

                for i, ind in enumerate(start_ind[0]):
                    this_dig_num = self.dig_in_num[ind]
                    if this_dig_num not in pulse_times.keys():
                        pulse_times[this_dig_num] = []
                    # In some cases for the traditional format, start and end
                    # might be in different files.
                    # Not dealing with this right now.
                    # Since we only use start times, we can just append None for end times
                    this_start = start_ind[1][i]
                    # this_end = end_ind[1][i]
                    this_end = None
                    pulse_times[this_dig_num].append((this_start, this_end))

                counter += dig_inputs.shape[1]

        dig_in_trials = [len(pulse_times[x]) for x in pulse_times.keys()]
        dig_in_frame = pd.DataFrame(
            dict(
                dig_in_nums=pulse_times.keys(),
                trial_counts=dig_in_trials,
                pulse_times=pulse_times.values()
            )
        )
        num_name_map = dict(zip(self.dig_in_num, self.dig_in_name))
        dig_in_frame['dig_in_names'] = dig_in_frame['dig_in_nums'].map(
            num_name_map)

        dig_in_frame.sort_values(by='dig_in_nums', inplace=True)
        dig_in_frame.reset_index(inplace=True, drop=True)
        # dig_in_frame['filenames'] = self.dig_in_file_list

        bad_dig_ins = dig_in_frame[dig_in_frame['trial_counts']
                                   == 0]['dig_in_nums'].values
        fin_dig_in_list = dig_in_frame[dig_in_frame['trial_counts']
                                       > 0]['dig_in_nums'].values
        fin_dig_in_trials = dig_in_frame[dig_in_frame['trial_counts']
                                         > 0].trial_counts.values

        if len(bad_dig_ins) > 0:
            bad_dig_ins = [str(element) for element in bad_dig_ins]
            bad_dig_in_str = '\n'.join(bad_dig_ins)
            print(f"== No deliveries detected for following dig-ins ==" + '\n')
            print('\n'+f"== {bad_dig_in_str} ==" + '\n')
            print('== They will be REMOVED from the list of dig-ins ==')
            print('== Remaining dig-ins ==' + '\n')

        dig_in_print_str = "Dig-ins : \n" + \
            str(dig_in_frame[['dig_in_nums', 'dig_in_names', 'trial_counts']])
        print(dig_in_print_str)

        dig_in_frame.reset_index(inplace=True, drop=True)
        self.dig_in_frame = dig_in_frame

    def write_out_frame(self):
        # Write out the dig-in frame
        self.dig_in_frame.to_csv(os.path.join(
            self.data_dir, 'dig_in_frame.csv'))
        print('Dig-in frame written out to dig_in_frame.csv')

    def load_dig_in_frame(self):
        # Load the dig-in frame
        self.dig_in_frame = pd.read_csv(os.path.join(self.data_dir, 'dig_in_frame.csv'),
                                        index_col=0)
        print('Dig-in frame loaded from dig_in_frame.csv')


class IntanDataHandler:
    def __init__(self, hdf5_name, electrode_layout_frame):
        self.hdf5_name = hdf5_name
        self.electrode_layout_frame = electrode_layout_frame

    def _read_data_from_file(self, filename):
        return np.fromfile(filename, dtype=np.dtype('int16'))

    def _create_earray(self, group, array_name, atom):
        hf5 = tables.open_file(self.hdf5_name, 'r+')
        if os.path.join(group, array_name) not in hf5:
            return hf5.create_earray(group, array_name, atom, (0,))
        else:
            return hf5.get_node(group, array_name)

    def read_traditional_intan(self, file_list):
        atom = tables.IntAtom()
        hf5 = tables.open_file(self.hdf5_name, 'r+')

        pbar = tqdm(total=len(file_list))
        for this_file in file_list:
            pbar.set_description(os.path.basename(this_file))
            this_file_data, data_present = load_file(this_file)
            for i, this_amp in enumerate(this_file_data['amplifier_data']):
                if 'emg' in self.electrode_layout_frame.loc[i].CAR_group.lower():
                    array_name = f'emg{i:02}'
                    hf5_el_array = self._create_earray(
                        '/raw_emg', array_name, atom)
                    hf5_el_array.append(this_amp)
                elif self.electrode_layout_frame.loc[i].CAR_group.lower() not in ['none', 'na']:
                    array_name = f'electrode{i:02}'
                    hf5_el_array = self._create_earray(
                        '/raw', array_name, atom)
                    hf5_el_array.append(this_amp)
                hf5.flush()
            pbar.update(1)
        pbar.close()
        hf5.close()

    def read_emg_channels(self):
        atom = tables.IntAtom()
        hf5 = tables.open_file(self.hdf5_name, 'r+')
        for num, row in tqdm(self.electrode_layout_frame.iterrows()):
            if 'emg' in row.CAR_group.lower():
                print(f'Reading : {row.filename, row.CAR_group}')
                data = self._read_data_from_file(row.filename)
                array_name = f'emg{row.electrode_ind:02}'
                hf5_el_array = self._create_earray(
                    '/raw_emg', array_name, atom)
                hf5_el_array.append(data)
                hf5.flush()
        hf5.close()

    def read_electrode_channels(self):
        atom = tables.IntAtom()
        hf5 = tables.open_file(self.hdf5_name, 'r+')
        for num, row in tqdm(self.electrode_layout_frame.iterrows()):
            emg_bool = 'emg' not in row.CAR_group.lower()
            none_bool = row.CAR_group.lower() not in ['none', 'na']
            if emg_bool and none_bool:
                print(f'Reading : {row.filename, row.CAR_group}')
                data = self._read_data_from_file(row.filename)
                array_name = f'electrode{row.electrode_ind:02}'
                hf5_el_array = self._create_earray('/raw', array_name, atom)
                hf5_el_array.append(data)
                hf5.flush()
        hf5.close()

    def read_electrode_emg_channels_single_file(self, electrodes_list, num_recorded_samples, emg_channels):
        hf5 = tables.open_file(self.hdf5_name, 'r+')
        atom = tables.IntAtom()
        amplifier_data = self._read_data_from_file(electrodes_list[0])
        num_electrodes = int(len(amplifier_data) / num_recorded_samples)
        amp_reshape = np.reshape(amplifier_data, (int(
            len(amplifier_data) / num_electrodes), num_electrodes)).T
        for num, row in tqdm(self.electrode_layout_frame.iterrows()):
            emg_bool = 'emg' not in row.CAR_group.lower()
            none_bool = row.CAR_group.lower() not in ['none', 'na']
            if emg_bool and none_bool:
                print(f'Reading : {row.filename, row.CAR_group}')
                el = self._create_earray(
                    '/raw', f'electrode{row.electrode_ind:02}', atom)
                exec(
                    f"hf5.root.raw.electrode{row.electrode_ind:02}.append(amp_reshape[num,:])")
                hf5.flush()
            elif not emg_bool and none_bool:
                el = self._create_earray(
                    '/raw_emg', f'emg{row.electrode_ind:02}', atom)
                exec(
                    f"hf5.root.raw_emg.emg{row.electrode_ind:02}.append(amp_reshape[num,:])")
        hf5.close()
