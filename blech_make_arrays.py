"""
This module processes neural and EMG data from an HDF5 file, extracting and organizing
spike trains and EMG trials based on digital input events. It also handles metadata and
logs the processing steps.

Functions:
- `create_spike_trains_for_digin`: Generates spike trains for specified digital input
  events and stores them in the HDF5 file.
- `create_emg_trials_for_digin`: Extracts EMG trial data for specified digital input
  events and stores it in the HDF5 file.

The main script:
  - Loads metadata and performs a pipeline graph check.
  - Creates trial information frame using blech_trial_info module.
  - Determines experiment end time based on EMG or spike data.
  - Creates spike trains if sorted units are present and saves them to the HDF5 file.
  - Creates EMG trial arrays if EMG data is available and saves them to the HDF5 file.
  - Logs the successful completion of the processing steps.
"""
# Import stuff!
import numpy as np
import tables
import sys
import os
import pandas as pd
from tqdm import tqdm
from blech_clust.utils.clustering import get_filtered_electrode
from blech_clust.utils.blech_process_utils import return_cutoff_values
from blech_clust.utils.blech_utils import imp_metadata, pipeline_graph_check
from blech_clust.utils.read_file import DigInHandler


def create_spike_trains_for_digin(
        this_starts,
        this_dig_name,
        durations,
        sampling_rate_ms,
        units,
        hf5,
):
    spike_train = []
    for this_start in this_starts:
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
            spike_times = (
                spike_times/sampling_rate_ms).astype(int) + durations[0]
            # Drop any spikes that are too close to the ends of the trial
            spike_times = spike_times[
                np.where((spike_times >= 0)*(spike_times < durations[0] +
                                             durations[1]))[0]]
            spikes[k, spike_times] = 1

        # Append the spikes array to spike_train
        spike_train.append(spikes)

    # And add spike_train to the hdf5 file
    hf5.create_group('/spike_trains', this_dig_name)
    spike_array = hf5.create_array(
        f'/spike_trains/{this_dig_name}',
        'spike_array', np.array(spike_train))
    hf5.flush()


def create_emg_trials_for_digin(
        this_starts,
        dig_in_basename,
        durations,
        sampling_rate_ms,
        emg_nodes,
        hf5,
):
    emg_data = [[this_emg[this_start - durations[0]*sampling_rate_ms:
                          this_start + durations[1]*sampling_rate_ms]
                 for this_start in this_starts]
                for this_emg in emg_nodes]
    emg_data = np.stack(emg_data)*0.195

    emg_data = np.mean(
        emg_data.reshape((*emg_data.shape[:2], -1, int(sampling_rate_ms))),
        axis=-1)

    # Write out ind:name map for each node
    ind_name_map = {i: node._v_name for i, node in enumerate(emg_nodes)}
    str_dict = str(ind_name_map)
    if '/emg_data/ind_electrode_map' in hf5:
        hf5.remove_node('/emg_data', 'ind_electrode_map')
    hf5.create_array('/emg_data', 'ind_electrode_map', np.array(str_dict))

    # And add emg_data to the hdf5 file
    hf5.create_group('/emg_data', dig_in_basename)
    # Shape = (n_channels, n_trials, n_samples)
    hf5.create_array(
        f'/emg_data/{dig_in_basename}',
        'emg_array', np.array(emg_data))
    hf5.flush()

############################################################
# Run Main
############################################################


if __name__ == '__main__':

    # Ask for the directory where the hdf5 file sits, and change to that directory
    # Get name of directory with the data files

    test_bool = False

    if test_bool:
        data_dir = '/media/storage/NM_resorted_data/NM43/NM43_500ms_160510_125413'
        metadata_handler = imp_metadata([[], data_dir])
    else:
        metadata_handler = imp_metadata(sys.argv)

        # Perform pipeline graph check
        script_path = os.path.realpath(__file__)
        this_pipeline_check = pipeline_graph_check(metadata_handler.dir_name)
        this_pipeline_check.check_previous(script_path)
        this_pipeline_check.write_to_log(script_path, 'attempted')

    os.chdir(metadata_handler.dir_name)
    print(f'Processing: {metadata_handler.dir_name}')

    # Open the hdf5 file
    hf5 = tables.open_file(metadata_handler.hdf5_name, 'r+')

    # Extract taste dig-ins from experimental info file
    info_dict = metadata_handler.info_dict
    params_dict = metadata_handler.params_dict
    sampling_rate = params_dict['sampling_rate']
    sampling_rate_ms = sampling_rate/1000

    ##############################
    # Load trial info frame
    print('Loading trial info frame...')

    # Try to load from HDF5 first
    try:
        trial_info_frame = pd.read_hdf(
            metadata_handler.hdf5_name, 'trial_info_frame')
        print('Trial info frame loaded from HDF5')
    except (KeyError, FileNotFoundError):
        # Fall back to CSV if HDF5 doesn't have it
        csv_path = os.path.join(
            metadata_handler.dir_name, 'trial_info_frame.csv')
        if os.path.exists(csv_path):
            trial_info_frame = pd.read_csv(csv_path)
            print('Trial info frame loaded from CSV')
        else:
            raise FileNotFoundError(
                "trial_info_frame not found in HDF5 or CSV. "
                "Please run blech_exp_info.py first to generate it."
            )

    print(trial_info_frame.head())

    # Pull out taste dig-ins from trial_info_frame
    taste_digin_names = trial_info_frame['dig_in_name_taste'].unique().tolist()
    taste_str = "\n".join([str(x) for x in taste_digin_names])

    # Extract laser dig-in info from trial_info_frame
    has_laser = trial_info_frame['laser'].any()
    if has_laser:
        laser_digin_names = trial_info_frame[trial_info_frame['laser']
                                            ]['dig_in_name_laser'].unique().tolist()
        laser_str = "\n".join([str(x) for x in laser_digin_names])
    else:
        laser_digin_names = []
        laser_str = 'None'

    print(f'Taste dig_ins ::: \n{taste_str}\n')
    print(f'Laser dig_in ::: \n{laser_str}\n')

    # Get list of units under the sorted_units group.
    # Find the latest/largest spike time amongst the units,
    # and get an experiment end time
    # (to account for cases where the headstage fell off mid-experiment)

    # TODO: Move this out of here...maybe make it a util
    # ============================================================#
    # NOTE: Calculate headstage falling off same way for all not "none" channels
    # Pull out raw_electrode and raw_emg data

    # If sorting hasn't been done, use only emg channels
    # to calculate cutoff...don't need to go through all channels

    raw_emg_electrodes = [x for x in hf5.get_node('/', 'raw_emg')]

    if len(raw_emg_electrodes) > 0:
        emg_electrode_names = [x._v_pathname for x in raw_emg_electrodes]
        electrode_names = list(zip(*[x.split('/')[1:]
                               for x in emg_electrode_names]))

        print('Calculating cutoff times using following EMG electrodes...')
        print(emg_electrode_names)
        print('===============================================')
        cutoff_data = []
        for this_el in tqdm(raw_emg_electrodes):
            raw_el = this_el[:]
            # High bandpass filter the raw electrode recordings
            filt_el = get_filtered_electrode(
                raw_el,
                freq=[params_dict['bandpass_lower_cutoff'],
                      params_dict['bandpass_upper_cutoff']],
                sampling_rate=params_dict['sampling_rate'])

            # Cut data to have integer number of seconds
            sampling_rate = params_dict['sampling_rate']
            filt_el = filt_el[:int(sampling_rate) *
                              int(len(filt_el)/sampling_rate)]

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
            cutoff_data.append(this_out)

        elec_cutoff_frame = pd.DataFrame(
            data=cutoff_data,
            columns=[
                'breach_rate',
                'breaches_per_sec',
                'secs_above_cutoff',
                'mean_breach_rate_persec',
                'recording_cutoff'
            ],
        )
        elec_cutoff_frame['electrode_type'] = electrode_names[0]
        elec_cutoff_frame['electrode_name'] = electrode_names[1]

        # Write out to HDF5
        hf5.close()
        elec_cutoff_frame.to_hdf(
            metadata_handler.hdf5_name,
            '/cutoff_frame'
        )
        hf5 = tables.open_file(metadata_handler.hdf5_name, 'r+')

        expt_end_time = elec_cutoff_frame['recording_cutoff'].min(
        )*sampling_rate
    else:
        # Else use spiketimes
        units = hf5.get_node('/', 'sorted_units')
        expt_end_time = np.max([x.times[-1] for x in units])

    # Check if any trials were cutoff
    cutoff_bool = np.logical_and(
        trial_info_frame.start_taste > expt_end_time,
        trial_info_frame.end_taste > expt_end_time
    )
    cutoff_frame = trial_info_frame.loc[cutoff_bool, :]
    cutoff_frame = cutoff_frame[[
        'dig_in_name_taste', 'start_taste', 'end_taste']]

    if len(cutoff_frame) > 0:
        print('=== Cutoff frame ===')
        print(cutoff_frame)
    else:
        print('=== No trials were cutoff ===')

    # ============================================================#

    ############################################################
    # Processing
    ############################################################

    taste_starts_cutoff = trial_info_frame.loc[~cutoff_bool].\
        groupby('dig_in_name_taste').start_taste.apply(np.array).tolist()

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
            hf5.remove_node('/spike_trains', recursive=True)
        hf5.create_group('/', 'spike_trains')

        # Pull out spike trains
        for name, this_starts in zip(taste_digin_names, taste_starts_cutoff):
            print(f'Creating spike-trains for {name}')
            create_spike_trains_for_digin(
                this_starts,
                name,
                durations,
                sampling_rate_ms,
                units,
                hf5,
            )
        ###############
        # Write out laser_duration and lag to hdf5 file
        if True in trial_info_frame['laser'] and '/spike_trains' in hf5:
            trial_info_group = \
                [x[1] for x in trial_info_frame.groupby('dig_in_name_taste')]
            for this_group in trial_info_group:
                this_group = this_group.sort_values('taste_rel_trial_num')
                laser_durations = this_group['laser_duration_ms'].values
                laser_lags = this_group['laser_lag_ms'].values
                this_dig_in_name = this_group['dig_in_name_taste'].values[0]
                dig_in_path = f'/spike_trains/{this_dig_in_name}'
                if f'{dig_in_path}/laser_durations' in hf5:
                    hf5.remove_node(dig_in_path, 'laser_durations')
                if f'{dig_in_path}/laser_onset_lag' in hf5:
                    hf5.remove_node(dig_in_path, 'laser_onset_lag')
                hf5.create_array(
                    dig_in_path,
                    'laser_durations', laser_durations)
                hf5.create_array(
                    dig_in_path,
                    'laser_onset_lag', laser_lags)
                hf5.flush()

    else:
        print('No sorted units found...NOT MAKING SPIKE TRAINS')

    # Test for EMG Data and then use it
    if len(raw_emg_electrodes) > 0:
        print('EMG Data found ==> Making EMG Trial Arrays')

        # Grab the names of the arrays containing emg recordings
        emg_nodes = hf5.list_nodes('/raw_emg')
        emg_pathname = []
        for node in emg_nodes:
            emg_pathname.append(node._v_pathname)

        # Delete /emg_data in hf5 file if it exists, and then create it
        if '/emg_data' in hf5:
            hf5.remove_node('/emg_data', recursive=True)
        hf5.create_group('/', 'emg_data')

        # Pull out emg trials
        for name, this_starts in zip(taste_digin_names, taste_starts_cutoff):
            print(f'Creating emg-trials for {name}')
            create_emg_trials_for_digin(
                this_starts,
                name,
                durations,
                sampling_rate_ms,
                emg_nodes,
                hf5,
            )

        # Save output in emg dir
        if not os.path.exists('emg_output'):
            os.makedirs('emg_output')

        # Also write out README to explain CAR groups and order of emg_data for user
        with open('emg_output/emg_data_readme.txt', 'w') as f:
            f.write(f'Channels used : {emg_pathname}')
            f.write('\n')
            f.write('Numbers indicate "electrode_ind" in electrode_layout_frame')
    else:
        print('No EMG Data Found...NOT MAKING EMG ARRAYS')

    hf5.close()

    # Write successful execution to log
    this_pipeline_check.write_to_log(script_path, 'completed')
