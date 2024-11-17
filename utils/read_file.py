# Import stuff!
import tables
import os
import numpy as np
import tqdm

# Code for loading traditional intan format from 
# https://github.com/Intan-Technologies/load-rhd-notebook-python

# import sys
# sys.path.append('/home/abuzarmahmood/projects/load-rhd-notebook-python')
from utils.importrhdutilities import load_file, read_header

# dir_name = '/home/abuzarmahmood/projects/blech_clust/pipeline_testing/test_data_handling/test_data/eb24_behandephys_11_12_24_241112_114659'
# # dir_name = '/home/abuzarmahmood/projects/blech_clust/pipeline_testing/test_data_handling/test_data/trad_intan_test'
# file_list = sorted([os.path.join(dir_name, x) for x in os.listdir(dir_name) if '.rhd' in x])

def read_traditional_intan(
		hdf5_name, 
		file_list, 
		electrode_layout_frame,
		dig_in_int
		):
	"""
	Reads traditional intan format data and saves to hdf5

	Input:
		hdf5_name: str
			Name of hdf5 file to save data to
		file_list: list
			List of file names to read
		electrode_layout_frame: pandas.DataFrame
			Dataframe containing details of electrode layout
		dig_in_int: list
			List of digital input numbers to read (as some may be empty)
			Note: This is the digital input number, not the index in the array

	Writes:
		hdf5 file with raw and raw_emg data
		- raw: amplifier data
		- raw_emg: EMG data
		- digital_in: digital input data
	"""
	atom = tables.IntAtom()
	# Read EMG data from amplifier channels
	# hdf5_path = os.path.join(dir_name, hdf5_name)
	hf5 = tables.open_file(hdf5_name, 'r+')

	pbar = tqdm.tqdm(total = len(file_list))
	for this_file in file_list:
		# Update progress bar with file name
		pbar.set_description(os.path.basename(this_file))
		# this_file_data = read_data(this_file)
		this_file_data, data_present = load_file(this_file)
		# Get channel details
		# For each anplifier channel, read data and save to hdf5
		for i, this_amp in enumerate(this_file_data['amplifier_data']):
			# If the amplifier channel is an EMG channel, save to raw_emg
			# Otherwise, save to raw
			if 'emg' in electrode_layout_frame.loc[i].CAR_group.lower():
				array_name = f'emg{i:02}'
				if os.path.join('/raw_emg', array_name) not in hf5: 
					hf5_el_array = hf5.create_earray('/raw_emg', array_name, atom, (0,))
				else:
					hf5_el_array = hf5.get_node('/raw_emg', array_name)
				hf5_el_array.append(this_amp)
			else:
				array_name = f'electrode{i:02}'
				if os.path.join('/raw', array_name) not in hf5: 
					hf5_el_array = hf5.create_earray('/raw', array_name, atom, (0,))
				else:
					hf5_el_array = hf5.get_node('/raw', array_name)
				hf5_el_array.append(this_amp)
			hf5.flush()
		# Do the same for digital inputs
		dig_in_channels = [x['native_channel_name'] for x in this_file_data['board_dig_in_channels']]
		# for i, this_dig_in in enumerate(this_file_data['board_dig_in_data']):
		for i, this_dig in enumerate(dig_in_channels):
			this_dig_in_int = int(this_dig.split('-')[-1].split('.')[0]) 
			if this_dig_in_int not in dig_in_int:
				continue
			array_name = f'dig_in_{this_dig_in_int}'
			array_data = this_file_data['board_dig_in_data'][i]
			if os.path.join('/digital_in', array_name) not in hf5:
				hf5_dig_array = hf5.create_earray('/digital_in', array_name, atom, (0,))
			else:
				hf5_dig_array = hf5.get_node('/digital_in', array_name)
			hf5_dig_array.append(array_data)
			hf5.flush()
		pbar.update(1)
	pbar.close()
	hf5.close()


def read_digins(hdf5_name, dig_in_int, dig_in_file_list): 
	atom = tables.IntAtom()
	hf5 = tables.open_file(hdf5_name, 'r+')
	# Read digital inputs, and append to the respective hdf5 arrays
	print('Reading dig-ins')
	for i, (dig_int, dig_in_filename) in \
			enumerate(zip(dig_in_int, dig_in_file_list)):
		dig_in_name = f'dig_in_{dig_int}'
		print(f'Reading {dig_in_name}')
		inputs = np.fromfile(dig_in_filename,
					   dtype = np.dtype('uint16'))
		hf5_dig_array = hf5.create_earray('/digital_in', dig_in_name, atom, (0,))
		hf5_dig_array.append(inputs)
		hf5.flush()
	hf5.close()

		
def read_digins_single_file(hdf5_name, dig_in, dig_in_file_list): 
	num_dig_ins = len(dig_in)
	hf5 = tables.open_file(hdf5_name, 'r+')
	# Read digital inputs, and append to the respective hdf5 arrays
	print('Reading dig-ins')
	atom = tables.IntAtom()
	for i in dig_in:
		dig_inputs = hf5.create_earray('/digital_in', 'dig_in_%i' % i, atom, (0,))
	d_inputs = np.fromfile(dig_in_file_list[0], dtype=np.dtype('uint16'))
	d_inputs_str = d_inputs.astype('str')
	d_in_str_int = d_inputs_str.astype('int64')
	d_diff = np.diff(d_in_str_int)
	dig_inputs = np.zeros((num_dig_ins,len(d_inputs)))
	for n_i in range(num_dig_ins):
		start_ind = np.where(d_diff == n_i + 1)[0]
		end_ind = np.where(d_diff == -1*(n_i + 1))[0]
		for s_i in range(len(start_ind)):
			dig_inputs[n_i,start_ind[s_i]:end_ind[s_i]] = 1
	for i in tqdm.tqdm(range(num_dig_ins)):		
		exec("hf5.root.digital_in.dig_in_"+str(i)+".append(dig_inputs[i,:])")
	hf5.flush()
	hf5.close()

# TODO: Remove exec statements throughout file
def read_emg_channels(hdf5_name, electrode_layout_frame):
	atom = tables.IntAtom()
	# Read EMG data from amplifier channels
	hf5 = tables.open_file(hdf5_name, 'r+')
	for num,row in tqdm.tqdm(electrode_layout_frame.iterrows()):
		# Loading should use file name 
		# but writing should use channel ind so that channels from 
		# multiple boards are written into a monotonic sequence
		if 'emg' in row.CAR_group.lower():
			print(f'Reading : {row.filename, row.CAR_group}')
			port = row.port
			channel_ind = row.electrode_ind
			data = np.fromfile(row.filename, dtype = np.dtype('int16'))
			# Label raw_emg with electrode_ind so it's more easily identifiable
			array_name = f'emg{channel_ind:02}'
			hf5_el_array = hf5.create_earray('/raw_emg', array_name, atom, (0,))
			hf5_el_array.append(data)
			hf5.flush()
	hf5.close()

def read_electrode_channels(hdf5_name, electrode_layout_frame):
	"""
	# Loading should use file name 
	# but writing should use channel ind so that channels from 
	# multiple boards are written into a monotonic sequence
	# Note: That channels inds may not be contiguous if there are
	# EMG channels in the middle
	"""
	atom = tables.IntAtom()
	# Read EMG data from amplifier channels
	hf5 = tables.open_file(hdf5_name, 'r+')
	for num,row in tqdm.tqdm(electrode_layout_frame.iterrows()):
		emg_bool = 'emg' not in row.CAR_group.lower()
		none_bool = row.CAR_group.lower() not in ['none','na']
		if emg_bool and none_bool:
			print(f'Reading : {row.filename, row.CAR_group}')
			port = row.port
			channel_ind = row.electrode_ind
			data = np.fromfile(row.filename, dtype = np.dtype('int16'))
			# Label raw_emg with electrode_ind so it's more easily identifiable
			array_name = f'electrode{channel_ind:02}'
			hf5_el_array = hf5.create_earray('/raw', array_name, atom, (0,))
			hf5_el_array.append(data)
			hf5.flush()
	hf5.close()
	
def read_electrode_emg_channels_single_file(
        hdf5_name, 
        electrode_layout_frame, 
        electrodes_list, 
        num_recorded_samples, 
        emg_channels):
    # Read EMG data from amplifier channels
	hf5 = tables.open_file(hdf5_name, 'r+')
	atom = tables.IntAtom()
	amplifier_data = np.fromfile(electrodes_list[0], dtype = np.dtype('int16'))
	num_electrodes = int(len(amplifier_data)/num_recorded_samples)
	amp_reshape = np.reshape(amplifier_data,(int(len(amplifier_data)/num_electrodes),num_electrodes)).T
	for num,row in tqdm.tqdm(electrode_layout_frame.iterrows()):
        # Loading should use file name 
        # but writing should use channel ind so that channels from 
        # multiple boards are written into a monotonic sequence
		emg_bool = 'emg' not in row.CAR_group.lower()
		none_bool = row.CAR_group.lower() not in ['none','na']
		if emg_bool and none_bool:
			print(f'Reading : {row.filename, row.CAR_group}')
			port = row.port
			channel_ind = row.electrode_ind
            #el = hf5.create_earray('/raw_emg', f'emg{emg_counter:02}', atom, (0,))
            # Label raw_emg with electrode_ind so it's more easily identifiable
			el = hf5.create_earray('/raw', f'electrode{channel_ind:02}', atom, (0,))
			exec(f"hf5.root.raw.electrode{channel_ind:02}.append(amp_reshape[num,:])")
			hf5.flush()
		elif not(emg_bool) and none_bool:
			port = row.port
			channel_ind = row.electrode_ind
			el = hf5.create_earray('/raw_emg', f'emg{channel_ind:02}', atom, (0,))
			exec(f"hf5.root.raw_emg.emg{channel_ind:02}.append(amp_reshape[num,:])")
	hf5.close()
