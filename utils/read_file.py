# Import stuff!
import tables
import os
import numpy as np
import tqdm

#todo: Separate functions for electrode, EMG, and dig-in channels
def read_digins(hdf5_name, dig_in, dig_in_list): 
    hf5 = tables.open_file(hdf5_name, 'r+')
    # Read digital inputs, and append to the respective hdf5 arrays
    print('Reading dig-ins')
    atom = tables.IntAtom()
    for i in dig_in:
        dig_inputs = hf5.create_earray(\
                '/digital_in', 'dig_in_%i' % i, atom, (0,))
        dig_name = [d_n for d_n in dig_in_list if int(d_n.split('.')[-2][-2:]) == i]
        inputs = np.fromfile(dig_name[0], 
                                dtype = np.dtype('uint16'))
        exec("hf5.root.digital_in.dig_in_"+str(i)+".append(inputs[:])")
    hf5.flush()
    hf5.close()
		
#todo: Separate functions for electrode, EMG, and dig-in channels
def read_digins_single_file(hdf5_name, dig_in, dig_in_list): 
	num_dig_ins = len(dig_in)
	hf5 = tables.open_file(hdf5_name, 'r+')
	# Read digital inputs, and append to the respective hdf5 arrays
	print('Reading dig-ins')
	atom = tables.IntAtom()
	for i in dig_in:
		dig_inputs = hf5.create_earray('/digital_in', 'dig_in_%i' % i, atom, (0,))
	d_inputs = np.fromfile(dig_in_list[0], dtype=np.dtype('uint16'))
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
    # Read EMG data from amplifier channels
    hf5 = tables.open_file(hdf5_name, 'r+')
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
            el = hf5.create_earray('/raw_emg', f'emg{channel_ind:02}', atom, (0,))
            exec(f"hf5.root.raw_emg.emg{channel_ind:02}."\
                    "append(data[:])")
            #emg_counter += 1
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
    # Read EMG data from amplifier channels
    hf5 = tables.open_file(hdf5_name, 'r+')
    atom = tables.IntAtom()
    for num,row in tqdm.tqdm(electrode_layout_frame.iterrows()):
        emg_bool = 'emg' not in row.CAR_group.lower()
        none_bool = row.CAR_group.lower() not in ['none','na']
        if emg_bool and none_bool:
            print(f'Reading : {row.filename, row.CAR_group}')
            port = row.port
            channel_ind = row.electrode_ind
            data = np.fromfile(row.filename, dtype = np.dtype('int16'))
            # Label raw_emg with electrode_ind so it's more easily identifiable
            el = hf5.create_earray('/raw', f'electrode{channel_ind:02}', atom, (0,))
            exec(f"hf5.root.raw.electrode{channel_ind:02}."\
                    "append(data[:])")
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
