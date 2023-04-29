# Import stuff!
import numpy as np
import tables
import sys
import os
import re
import glob
import pandas as pd
from tqdm import tqdm
from utils.clustering import get_filtered_electrode
from utils.blech_process_utils import return_cutoff_values
from utils.blech_utils import (
        entry_checker,
        imp_metadata,
        )

def get_dig_in_data(hf5):
    dig_in_nodes = hf5.list_nodes('/digital_in')
    dig_in_data = []
    dig_in_pathname = []
    for node in dig_in_nodes:
        dig_in_pathname.append(node._v_pathname)
        dig_in_data.append(node[:])
    dig_in_basename = [os.path.basename(x) for x in dig_in_pathname]
    dig_in_data = np.array(dig_in_data)
    return dig_in_pathname, dig_in_basename, dig_in_data

def create_spike_trains_for_digin(
        taste_starts_cutoff,
        dig_in_ind,
        this_dig_in,
        durations,
        sampling_rate_ms,
        units,
        hf5,
        ):
        spike_train = []
        for this_start in this_dig_in: 
            spikes = np.zeros((len(units), durations[0] + durations[1]))
            for k in range(len(units)):
                # Get the spike times around the end of taste delivery
                trial_bounds = [
                        this_start + durations[1]*sampling_rate_ms,
                        this_start - durations[0]*sampling_rate_ms
                        ]
                spike_inds = np.logical_and(
                                units[k].times[:] <= trial_bounds[0],
                                units[k].times[:] >= trial_bounds[1] 
                            )
                spike_times = units[k].times[spike_inds]
                spike_times = spike_times - this_start 
                spike_times = (spike_times/sampling_rate_ms).astype(int) + durations[0]
                # Drop any spikes that are too close to the ends of the trial
                spike_times = spike_times[\
                        np.where((spike_times >= 0)*(spike_times < durations[0] + \
                        durations[1]))[0]]
                spikes[k, spike_times] = 1
                            
            # Append the spikes array to spike_train 
            spike_train.append(spikes)

        # And add spike_train to the hdf5 file
        hf5.create_group('/spike_trains', dig_in_basename[i])
        spike_array = hf5.create_array(
                f'/spike_trains/{dig_in_basename[i]}', 
                'spike_array', np.array(spike_train))
        hf5.flush()

def create_laser_params_for_digin(
        i,
        this_dig_in,
        start_points_cutoff,
        end_points_cutoff,
        sampling_rate,
        sampling_rate_ms,
        laser_digin_inds,
        dig_in_basename,
        hf5,
        ):

    # Even if laser is not present, create arrays for laser parameters
    laser_duration = np.zeros(len(this_dig_in))
    laser_start = np.zeros(len(this_dig_in))

    if len(laser_digin_inds):
        selected_laser_digin = laser_digin_inds[0]
        print(f'Processing laser from {dig_in_basename[selected_laser_digin]}')

        # Else run through the lasers and check if the lasers 
        # went off within 5 secs of the stimulus delivery time
        time_diff = \
                this_dig_in[:,np.newaxis] - \
                start_points_cutoff[selected_laser_digin][:,np.newaxis].T
        time_diff = np.abs(time_diff)
        laser_trial_bool = time_diff <= 5*sampling_rate
        which_taste_trial = np.sum(laser_trial_bool, axis = 1) > 0
        which_laser_trial = np.sum(laser_trial_bool, axis = 0) > 0

        all_laser_durations = \
                end_points_cutoff[selected_laser_digin] - \
                start_points_cutoff[selected_laser_digin]
        wanted_laser_durations = all_laser_durations[which_laser_trial]
        wanted_laser_starts = \
                start_points_cutoff[selected_laser_digin][which_laser_trial] - \
                this_dig_in[which_taste_trial]
        # If the lasers did go off around stimulus delivery, 
        # get the duration and start time in ms 
        # (from end of taste delivery) of the laser trial 
        # (as a multiple of 10 - so 53 gets rounded off to 50)
        vector_int = np.vectorize(np.int32)
        wanted_laser_durations = \
                10*vector_int(wanted_laser_durations/(sampling_rate_ms*10))
        wanted_laser_starts = \
                10*vector_int(wanted_laser_starts/(sampling_rate_ms*10))

        laser_duration[which_taste_trial] = wanted_laser_durations
        laser_start[which_taste_trial] = wanted_laser_starts

    else:
        print('No lasers specified')

    if '/spike_trains' not in hf5:
        hf5.create_group('/', 'spike_trains')

    dig_in_path = f'/spike_trains/{dig_in_basename[i]}'
    if dig_in_path not in hf5:
        hf5.create_group('/spike_trains', dig_in_basename[i])

    # Write the conditional stimulus duration array to the hdf5 file
    if f'{dig_in_path}/laser_durations' in hf5:
        hf5.remove_node(dig_in_path, 'laser_durations')
    if f'{dig_in_path}/laser_onset_lag' in hf5:
        hf5.remove_node(dig_in_path, 'laser_onset_lag')
    laser_durations = hf5.create_array(
            dig_in_path,
            'laser_durations', laser_duration)
    laser_onset_lag = hf5.create_array(
            dig_in_path,
            'laser_onset_lag', laser_start)
    hf5.flush() 

# _                    _   ____        _        
#| |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
#| |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
#| |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
#|_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
#                                               

if __name__ == '__main__':

    # Ask for the directory where the hdf5 file sits, and change to that directory
    # Get name of directory with the data files
    metadata_handler = imp_metadata(sys.argv)
    os.chdir(metadata_handler.dir_name)

    # Open the hdf5 file
    hf5 = tables.open_file(metadata_handler.hdf5_name, 'r+')

    # Grab the names of the arrays containing digital inputs, 
    # and pull the data into a numpy array
    dig_in_pathname, dig_in_basename, dig_in_data = get_dig_in_data(hf5)
    dig_in_diff = np.diff(dig_in_data,axis=-1)
    # Calculate start and end points of pulses
    start_points = [np.where(x==1)[0] for x in dig_in_diff]
    end_points = [np.where(x==-1)[0] for x in dig_in_diff]

    # Extract taste dig-ins from experimental info file
    info_dict = metadata_handler.info_dict
    params_dict = metadata_handler.params_dict
    sampling_rate = params_dict['sampling_rate']
    sampling_rate_ms = sampling_rate/1000

    # Pull out taste dig-ins
    taste_digin_inds = info_dict['taste_params']['dig_ins']
    taste_digin_channels = [dig_in_basename[x] for x in taste_digin_inds]
    taste_str = "\n".join(taste_digin_channels)

    # Extract laser dig-in from params file
    laser_digin_inds = [info_dict['laser_params']['dig_in']][0]

    # Pull laser digin from hdf5 file
    if len(laser_digin_inds) == 0:
        laser_digin_channels = []
        laser_str = 'None'
    else:
        laser_digin_channels = [dig_in_basename[x] for x in laser_digin_inds]
        laser_str = "\n".join(laser_digin_channels)

    print(f'Taste dig_ins ::: \n{taste_str}\n')
    print(f'Laser dig_in ::: \n{laser_str}\n')


    # Get list of units under the sorted_units group. 
    # Find the latest/largest spike time amongst the units, 
    # and get an experiment end time 
    # (to account for cases where the headstage fell off mid-experiment)

    # TODO: Move this out of here...maybe make it a util
    #============================================================#
    # NOTE: Calculate headstage falling off same way for all not "none" channels 
    # Pull out raw_electrode and raw_emg data
    if '/raw' in hf5:
        raw_electrodes = [x for x in hf5.get_node('/','raw')]
    else:
        raw_electrodes = []
    if '/raw_emg' in hf5:
        raw_emg_electrodes = [x for x in hf5.get_node('/','raw_emg')]
    else:
        raw_emg = []

    all_electrodes = [raw_electrodes, raw_emg_electrodes] 
    all_electrodes = [x for y in all_electrodes for x in y]
    # If raw channel data is present, use that to calcualte cutoff
    # This would explicitly be the case if only EMG was recorded
    if len(all_electrodes) > 0:
        all_electrode_names = [x._v_pathname for x in all_electrodes]
        electrode_names = list(zip(*[x.split('/')[1:] for x in all_electrode_names]))

        print('Calculating cutoff times')
        cutoff_data = []
        for this_el in tqdm(all_electrodes): 
            raw_el = this_el[:]
            # High bandpass filter the raw electrode recordings
            filt_el = get_filtered_electrode(
                raw_el,
                freq=[params_dict['bandpass_lower_cutoff'],
                      params_dict['bandpass_upper_cutoff']],
                sampling_rate=params_dict['sampling_rate'])

            # Cut data to have integer number of seconds
            sampling_rate = params_dict['sampling_rate']
            filt_el = filt_el[:int(sampling_rate)*int(len(file_el)/sampling_rate)]

            # Delete raw electrode recording from memory
            del raw_el

            # Get parameters for recording cutoff
            this_out = return_cutoff_values(
                            filt_el,
                            params_dict['sampling_rate'],
                            params_dict['voltage_cutoff'],
                            params_dict['max_breach_rate'],
                            params_dict['max_secs_above_cutoff'],
                            params_dict['max_mean_breach_rate_persec']
                            ) 
            # First output of recording cutoff is processed filtered electrode 
            cutoff_data.append(this_out[1:])


        elec_cutoff_frame = pd.DataFrame(
                data = cutoff_data,
                columns = [
                    'breach_rate', 
                    'breaches_per_sec', 
                    'secs_above_cutoff', 
                    'mean_breach_rate_persec',
                    'recording_cutoff'
                    ],
                )
        elec_cutoff_frame['electrode_type'] = all_electrode_names[0]
        elec_cutoff_frame['electrode_name'] = all_electrode_names[1]

        # Write out to HDF5
        hf5.close()
        elec_cutoff_frame.to_hdf(
                metadata_handler.hdf5_name,
                '/cutoff_frame'
                )
        hf5 = tables.open_file(metadata_handler.hdf5_name, 'r+')

        expt_end_time = elec_cutoff_frame['recording_cutoff'].min()*sampling_rate
    else:
        # Else use spiketimes
        units = hf5.get_node('/','sorted_units')
        expt_end_time = np.max([x.times[-1] for x in units]) 
    #============================================================#

    # ____                              _             
    #|  _ \ _ __ ___   ___ ___  ___ ___(_)_ __   __ _ 
    #| |_) | '__/ _ \ / __/ _ \/ __/ __| | '_ \ / _` |
    #|  __/| | | (_) | (_|  __/\__ \__ \ | | | | (_| |
    #|_|   |_|  \___/ \___\___||___/___/_|_| |_|\__, |
    #                                           |___/ 

    #TODO: Creating spike-trians + laser arrays can CERTAINLY be made cleaner
    # Go through the taste_digin_inds and make an array of spike trains 
    # of dimensions (# trials x # units x trial duration (ms)) - 
    # use START of digital input pulse as the time of taste delivery
    # Refer to https://github.com/narendramukherjee/blech_clust/pull/14

    # Check start points prior to loop and print results
    dig_in_trials = np.array([len(x) for x in start_points])
    start_points_cutoff = [x[x<expt_end_time] for x in start_points]
    end_points_cutoff = [x[x<expt_end_time] for x in end_points]
    trials_before_cutoff = np.array([len(x) for x in start_points_cutoff])
    cutoff_frame = pd.DataFrame(
            data = dict(
                dig_ins = dig_in_basename,
                trials_before_cutoff = trials_before_cutoff,
                trials_after_cutoff = dig_in_trials - trials_before_cutoff
                )
            )
    print(cutoff_frame)

    taste_starts_cutoff = [start_points_cutoff[i] for i in taste_digin_inds]

    # Load durations from params file
    durations = params_dict['spike_array_durations']
    print(f'Using durations ::: {durations}')

    # Only make spike-trains if sorted units present
    if '/sorted_units' in hf5:
        print('Sorted units found ==> Making spike trains')
        units = hf5.list_nodes('/sorted_units')

        # Delete the spike_trains node in the hdf5 file if it exists, 
        # and then create it
        if '/spike_trains' in hf5:
            hf5.remove_node('/spike_trains', recursive = True)
        hf5.create_group('/', 'spike_trains')

        # Pull out spike trains
        for i, this_dig_in in enumerate(taste_starts_cutoff): 
            print(f'Creating spike-trains for {dig_in_basename[i]}')
            create_spike_trains_for_digin(
                    taste_starts_cutoff,
                    i,
                    this_dig_in,
                    durations,
                    sampling_rate_ms,
                    units,
                    hf5,
                    )
    else:
        print('No sorted units found...NOT MAKING SPIKE TRAINS')

    # Separate out laser loop
    for i, this_dig_in in enumerate(taste_starts_cutoff): 
        print(f'Creating laser info for {dig_in_basename[i]}')
        create_laser_params_for_digin(
                i,
                this_dig_in,
                start_points_cutoff,
                end_points_cutoff,
                sampling_rate,
                sampling_rate_ms,
                laser_digin_inds,
                dig_in_basename,
                hf5,
                )

    if '/raw_emg' in hf5:
        if len(list(hf5.get_node('/','raw_emg'))) > 0:
        
            print('EMG Data found ==> Making EMG Trial Arrays')

            # Grab the names of the arrays containing emg recordings
            emg_nodes = hf5.list_nodes('/raw_emg')
            emg_pathname = []
            for node in emg_nodes:
                emg_pathname.append(node._v_pathname)

            # Create a numpy array to store emg data by trials
            # Shape : Channels x Tastes x Trials x Time
            # Use max number of trials to define array, this allows people with uneven
            # numbers of trials to continue working
            trial_counts = [len(x) for x in taste_starts_cutoff]
            if len(np.unique(trial_counts)) > 1:
                print(f'!! Uneven numbers of trials !! {trial_counts}')
                print(f'Using {np.max(trial_counts)} as trial count')
                print('== EMG ARRAY WILL HAVE EMPTY TRIALS ==')

            # Shape : channels x dig_ins x max_trials x duration 
            emg_data = np.ndarray((
                len(emg_pathname), 
                len(taste_starts_cutoff), 
                np.max(trial_counts), 
                durations[0]+durations[1]))

            # And pull out emg data into this array
            for i in range(len(emg_pathname)):
                data = hf5.get_node(emg_pathname[i])[:]
                for j, this_taste_digin in enumerate(taste_starts_cutoff):
                    for k, this_start in enumerate(this_taste_digin):
                        trial_bounds = [
                                int(this_start - durations[0]*sampling_rate_ms),
                                int(this_start + durations[1]*sampling_rate_ms)
                                ]
                        raw_emg_data = data[trial_bounds[0]:trial_bounds[1]]
                        raw_emg_data = 0.195*raw_emg_data
                        # Downsample the raw data by averaging the 30 samples 
                        # per millisecond, and assign to emg_data
                        emg_data[i, j, k, :] = \
                                np.mean(
                                    raw_emg_data.reshape((-1, int(sampling_rate_ms))), 
                                    axis = 1)

            # Write out booleans for non-zero trials
            nonzero_trial = np.abs(emg_data.mean(axis=(0,3))) > 0

            # Save output in emg dir
            if not os.path.exists('emg_output'):
                os.makedirs('emg_output')

            # Save the emg_data
            np.save('emg_output/emg_data.npy', emg_data)
            np.save('emg_output/nonzero_trials.npy', nonzero_trial)

            # Also write out README to explain CAR groups and order of emg_data for user
            with open('emg_output/emg_data_readme.txt','w') as f:
                f.write(f'Channels used : {emg_pathname}')
                f.write('\n')
                f.write('Numbers indicate "electrode_ind" in electrode_layout_frame')

        else:
            print('No EMG Data Found...NOT MAKING EMG ARRAYS')

    hf5.close()

