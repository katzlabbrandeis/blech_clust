# Import stuff!
import tables
import os
import numpy as np
import tqdm

#todo: Separate functions for electrode, EMG, and dig-in channels
def read_digins(hdf5_dir_name, data_dir, dig_in, dig_in_list): 
	os.chdir(data_dir)
	hf5 = tables.open_file(hdf5_dir_name, 'r+')
	# Read digital inputs, and append to the respective hdf5 arrays
	print('Reading dig-ins')
	atom = tables.IntAtom()
	for i in dig_in:
		#To handle the case of multiple files being concatenated
		try:
			dig_inputs = hf5.create_earray('/digital_in', 'dig_in_%i' % i, atom, (0,))
		except:
			#Dig-in data of some form already exists, so concatenate instead
			exist_dig_inputs = hf5.root.digital_in['dig_in_' + str(i)][:]
			hf5.remove_node('/digital_in','dig_in_%i' % i)
			dig_inputs = hf5.create_earray('/digital_in', 'dig_in_%i' % i, atom, (0,))
		dig_name = [d_n for d_n in dig_in_list if int(d_n.split('.')[-2][-2:]) == i]
		inputs = np.fromfile(dig_name[0],dtype = np.dtype('uint16'))
		try:
			full_inputs = np.concatenate((exist_dig_inputs,inputs))
			print("\tExisting inputs length = " + str(len(exist_dig_inputs)))
			print("\tNew inputs length = " + str(len(inputs)))
			print("\tHDF5 appended length = " + str(len(full_inputs)))
		except:
			full_inputs = inputs
		exec("hf5.root.digital_in.dig_in_"+str(i)+".append(full_inputs[:])")
	hf5.flush()
	hf5.close()
		
#todo: Separate functions for electrode, EMG, and dig-in channels
def read_digins_single_file(hdf5_dir_name, data_dir, dig_in, dig_in_list): 
	os.chdir(data_dir)
	num_dig_ins = len(dig_in)
	hf5 = tables.open_file(hdf5_dir_name, 'r+')
	# Read digital inputs, and append to the respective hdf5 arrays
	print('Reading dig-ins')
	atom = tables.IntAtom()
	#First prepare data
	d_inputs = np.fromfile(dig_in_list[0], dtype=np.dtype('uint16'))
	d_inputs_str = d_inputs.astype('str')
	d_in_str_int = d_inputs_str.astype('int64')
	d_diff = np.diff(d_in_str_int)
	inputs_array = np.zeros((num_dig_ins,len(d_inputs)))
	for n_i in range(num_dig_ins):
		start_ind = np.where(d_diff == n_i + 1)[0]
		end_ind = np.where(d_diff == -1*(n_i + 1))[0]
		for s_i in range(len(start_ind)):
			inputs_array[n_i,start_ind[s_i]:end_ind[s_i]] = 1
	for i in dig_in:
		#To handle the case of multiple files being concatenated
		try:
			dig_inputs = hf5.create_earray('/digital_in', 'dig_in_%i' % i, atom, (0,))
		except:
			#Dig-in data of some form already exists, so concatenate instead
			exist_dig_inputs = hf5.root.digital_in['dig_in_' + str(i)][:]
			hf5.remove_node('/digital_in','dig_in_%i' % i)
			dig_inputs = hf5.create_earray('/digital_in', 'dig_in_%i' % i, atom, (0,))
		inputs = inputs_array[i,:]
		try:
			full_inputs = np.concatenate((exist_dig_inputs,inputs))
			print("\tExisting inputs length = " + str(len(exist_dig_inputs)))
			print("\tNew inputs length = " + str(len(inputs)))
			print("\tHDF5 appended length = " + str(len(full_inputs)))
		except:
			full_inputs = inputs
		exec("hf5.root.digital_in.dig_in_"+str(i)+".append(full_inputs)")
	hf5.flush()
	hf5.close()

# TODO: Remove exec statements throughout file
def read_emg_channels(hdf5_dir_name, data_dir, electrode_layout_frame):
	os.chdir(data_dir)
	# Read EMG data from amplifier channels
	hf5 = tables.open_file(hdf5_dir_name, 'r+')
	atom = tables.IntAtom()
	#emg_counter = 0
	for num,row in tqdm.tqdm(electrode_layout_frame.iterrows()):
		# Loading should use file name 
		# but writing should use channel ind so that channels from 
		# multiple boards are written into a monotonic sequence
		if 'emg' in row.CAR_group.lower():
			print(f'Reading : {row.filename, row.CAR_group}')
			port = row.port
			channel_ind = row.electrode_ind
			data = np.fromfile(row.filename, dtype = np.dtype('int16'))
			#el = hf5.create_earray('/raw_emg', f'emg{emg_counter:02}', atom, (0,))
			# Label raw_emg with electrode_ind so it's more easily identifiable
			try:
				el = hf5.create_earray('/raw_emg', f'emg{channel_ind:02}', atom, (0,))
			except:
				#Data of some form already exists, so concatenate instead
				exist_emg_inputs = hf5.root.raw_emg['emg' + str(f"{channel_ind:02d}")][:]
				hf5.remove_node('/raw_emg',f'emg{channel_ind:02}')
				el = hf5.create_earray('/raw_emg', f'emg{channel_ind:02}', atom, (0,))
			try:
				full_inputs = np.concatenate((exist_emg_inputs,data[:]))
				print("\tExisting inputs length = " + str(len(exist_emg_inputs)))
				print("\tNew inputs length = " + str(len(data[:])))
				print("\tHDF5 appended length = " + str(len(full_inputs)))
			except:
				full_inputs = data[:]
			exec(f"hf5.root.raw_emg.emg{channel_ind:02}.append(full_inputs[:])")
			#emg_counter += 1
	hf5.flush()
	hf5.close()

def read_electrode_channels(hdf5_dir_name, data_dir, electrode_layout_frame):
	os.chdir(data_dir)
	# Read EMG data from amplifier channels
	hf5 = tables.open_file(hdf5_dir_name, 'r+')
	atom = tables.IntAtom()
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
			data = np.fromfile(row.filename, dtype = np.dtype('int16'))
			#el = hf5.create_earray('/raw_emg', f'emg{emg_counter:02}', atom, (0,))
			# Label raw_emg with electrode_ind so it's more easily identifiable
			try:
				el = hf5.create_earray('/raw', f'electrode{channel_ind:02}', atom, (0,))
			except:
				#Data of some form already exists, so concatenate instead
				exist_inputs = hf5.root.raw['electrode' + str(f"{channel_ind:02d}")][:]
				hf5.remove_node('/raw',f'electrode{channel_ind:02}')
				el = hf5.create_earray('/raw', f'electrode{channel_ind:02}', atom, (0,))
			try:
				full_inputs = np.concatenate((exist_inputs,data[:]))
				print("\tExisting inputs length = " + str(len(exist_inputs)))
				print("\tNew inputs length = " + str(len(data[:])))
				print("\tHDF5 appended length = " + str(len(full_inputs)))
			except:
				full_inputs = data[:]
			exec(f"hf5.root.raw.electrode{channel_ind:02}.append(full_inputs[:])")
	hf5.flush()
	hf5.close()
	
def read_electrode_emg_channels_single_file(hdf5_dir_name, data_dir, electrode_layout_frame, electrodes_list, num_recorded_samples, emg_channels):
	os.chdir(data_dir)
	# Read EMG data from amplifier channels
	hf5 = tables.open_file(hdf5_dir_name, 'r+')
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
			try:
				el = hf5.create_earray('/raw', f'electrode{channel_ind:02}', atom, (0,))
			except:
				#Data of some form already exists, so concatenate instead
				exist_inputs = hf5.root.raw['electrode' + str(f"{channel_ind:02d}")][:]
				hf5.remove_node('/raw',f'electrode{channel_ind:02}')
				el = hf5.create_earray('/raw', f'electrode{channel_ind:02}', atom, (0,))
			inputs = amp_reshape[num,:]
			try:
				full_inputs = np.concatenate((exist_inputs,inputs))
				print("\tExisting inputs length = " + str(len(exist_inputs)))
				print("\tNew inputs length = " + str(len(inputs)))
				print("\tHDF5 appended length = " + str(len(full_inputs)))
			except:
				full_inputs = inputs
			exec(f"hf5.root.raw.electrode{channel_ind:02}.append(full_inputs[:])")
			hf5.flush()
		elif not(emg_bool) and none_bool:
			port = row.port
			channel_ind = row.electrode_ind
			try:
				el = hf5.create_earray('/raw_emg', f'emg{channel_ind:02}', atom, (0,))
			except:
				#Data of some form already exists, so concatenate instead
				exist_inputs = hf5.root.raw_emg['emg' + str(f"{channel_ind:02d}")][:]
				hf5.remove_node('/raw_emg',f'emg{channel_ind:02}')
				el = hf5.create_earray('/raw_emg', f'emg{channel_ind:02}', atom, (0,))
			inputs = amp_reshape[num,:]
			try:
				full_inputs = np.concatenate((exist_inputs,amp_reshape[num,:]))
				print("\tExisting inputs length = " + str(len(exist_inputs)))
				print("\tNew inputs length = " + str(len(inputs)))
				print("\tHDF5 appended length = " + str(len(full_inputs)))
			except:
				full_inputs = amp_reshape[num,:]
			exec(f"hf5.root.raw_emg.emg{channel_ind:02}.append(full_inputs[:])")
	hf5.close()
