"""
This module processes neural and EMG data from an HDF5 file, extracting and organizing spike trains and EMG trials based on digital input events. It also handles metadata and logs the processing steps.

- `create_spike_trains_for_digin(this_starts, this_dig_name, durations, sampling_rate_ms, units, hf5)`: Generates spike trains for specified digital input events and stores them in the HDF5 file.
- `create_emg_trials_for_digin(this_starts, dig_in_basename, durations, sampling_rate_ms, emg_nodes, hf5)`: Extracts EMG trial data for specified digital input events and stores it in the HDF5 file.
- The main script:
  - Loads metadata and performs a pipeline graph check.
  - Extracts digital input data and organizes it into a trial information frame.
  - Calculates laser timing corrections and saves trial information to the HDF5 file and a CSV.
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
from ast import literal_eval
import matplotlib.pyplot as plt

# def get_dig_in_data(hf5):
#     dig_in_nodes = hf5.list_nodes('/digital_in')
#     dig_in_data = []
#     dig_in_pathname = []
#     for node in dig_in_nodes:
#         dig_in_pathname.append(node._v_pathname)
#         dig_in_data.append(node[:])
#     dig_in_basename = [os.path.basename(x) for x in dig_in_pathname]
#     dig_in_data = np.array(dig_in_data)
#     return dig_in_pathname, dig_in_basename, dig_in_data


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

    # Grab the names of the arrays containing digital inputs,
    # and pull the data into a numpy array
    # dig_in_pathname, dig_in_basename, dig_in_data = get_dig_in_data(hf5)
    # dig_in_diff = np.diff(dig_in_data,axis=-1)
    # # Calculate start and end points of pulses
    # start_points = [np.where(x==1)[0] for x in dig_in_diff]
    # end_points = [np.where(x==-1)[0] for x in dig_in_diff]

    # Extract taste dig-ins from experimental info file
    info_dict = metadata_handler.info_dict
    params_dict = metadata_handler.params_dict
    sampling_rate = params_dict['sampling_rate']
    sampling_rate_ms = sampling_rate/1000

    this_dig_handler = DigInHandler(
        metadata_handler.dir_name,
        info_dict['file_type']
    )
    this_dig_handler.load_dig_in_frame()
    print('DigIn data loaded')
    print(this_dig_handler.dig_in_frame.drop(columns='pulse_times'))

    # Pull out taste dig-ins
    taste_digin_nums = info_dict['taste_params']['dig_in_nums']
    # taste_digin_channels = [dig_in_basename[x] for x in taste_digin_inds]
    # taste_str = "\n".join(taste_digin_channels)
    taste_str = "\n".join([str(x) for x in taste_digin_nums])

    # Extract laser dig-in from params file
    laser_digin_nums = [info_dict['laser_params']['dig_in_nums']][0]

    # Pull laser digin from hdf5 file
    if len(laser_digin_nums) == 0:
        laser_digin_channels = []
        laser_str = 'None'
    else:
        # laser_digin_channels = [dig_in_basename[x] for x in laser_digin_inds]
        laser_str = "\n".join([str(x) for x in laser_digin_nums])

    print(f'Taste dig_ins ::: \n{taste_str}\n')
    print(f'Laser dig_in ::: \n{laser_str}\n')

    ##############################
    # Create trial info frame with following information
    # 1. Trial # (from 1 to n)
    # 2. Trial # (per taste)
    # 3. Taste dig-in
    # 4. Taste name (from info file)
    # 5. Laser dig-in
    # 6. Laser duration and lag
    # 7. Start and end times of taste delivery
    # 8. Start and end times of laser delivery
    # 9. Start and end times of taste delivery (in ms)
    # 10. Start and end times of laser delivery (in ms)
    # 11. Laser duration and lag (in ms)

    taste_info_list = []
    for ind, num in enumerate(taste_digin_nums):
        this_dig = this_dig_handler.dig_in_frame.loc[
            this_dig_handler.dig_in_frame['dig_in_nums'] == num]
        pulse_times = this_dig['pulse_times'].values[0]
        pulse_times = literal_eval(pulse_times)
        dig_in_name = this_dig['dig_in_names'].values[0]
        this_frame = pd.DataFrame(
            dict(
                dig_in_num=num,
                dig_in_name=dig_in_name,
                taste=this_dig['taste'].values[0],
                start=[x[0] for x in pulse_times],
                end=[x[1] for x in pulse_times],
            )
        )
        taste_info_list.append(this_frame)
    taste_info_frame = pd.concat(taste_info_list)
    taste_info_frame.sort_values(by=['start'], inplace=True)
    taste_info_frame.reset_index(drop=True, inplace=True)
    taste_info_frame['abs_trial_num'] = taste_info_frame.index

    # Calculate taste duration
    taste_info_frame['taste_duration'] = (
        taste_info_frame['end'] - taste_info_frame['start']
    )
    taste_info_frame['taste_duration_ms'] = (
        taste_info_frame['taste_duration'] / sampling_rate
    ) * 1000

    # Add taste_rel_trial_num
    taste_grouped = taste_info_frame.groupby('dig_in_num')
    fin_group = []
    for name, group in taste_grouped:
        group['taste_rel_trial_num'] = np.arange(group.shape[0])
        fin_group.append(group)
    taste_info_frame = pd.concat(fin_group)
    taste_info_frame.sort_values(by=['start'], inplace=True)

    laser_info_list = []
    # for ind, num in enumerate(laser_digin_inds):
    for ind, num in enumerate(laser_digin_nums):
        this_dig = this_dig_handler.dig_in_frame.loc[
            this_dig_handler.dig_in_frame['dig_in_nums'] == num]
        # pulse_times = this_dig_handler.dig_in_frame['pulse_times'][num]
        pulse_times = this_dig['pulse_times'].values[0]
        pulse_times = literal_eval(pulse_times)
        # dig_in_name = this_dig_handler.dig_in_frame['dig_in_nums'][num]
        dig_in_name = this_dig['dig_in_names'].values[0]
        this_frame = pd.DataFrame(
            dict(
                dig_in_num=num,
                dig_in_name=dig_in_name,
                laser=True,
                start=[x[0] for x in pulse_times],
                end=[x[1] for x in pulse_times],
            )
        )
        laser_info_list.append(this_frame)

    if len(laser_info_list) > 0:
        laser_info_frame = pd.concat(laser_info_list)
        laser_starts = laser_info_frame['start'].values
        match_trials_ind = []

        if len(info_dict['laser_params']['onset_duration']) == 1:

            # Match laser starts to taste starts within tolerance
            match_tol = (2*sampling_rate)/10  # 200 ms
            print(
                f'Aligning laser to taste using exact match with tolerance of {match_tol/sampling_rate} sec')
            for this_start in laser_starts:
                match_ind = np.where(
                    np.abs(taste_info_frame['start'] - this_start) < match_tol
                )[0]
                if not len(match_ind) == 1:
                    error_str = f'Exact match not found between taste and laser signals given tolerance of {(match_tol)/sampling_rate} sec'
                    raise ValueError(error_str)
                match_trials_ind.append(match_ind[0])

        else:
            print('Aligning laser to taste using closest trial match')
            for this_start in laser_starts:
                match_ind = np.argmin(
                    np.abs(taste_info_frame['start'] - this_start)
                )
                match_trials_ind.append(match_ind)

        match_trials = taste_info_frame.iloc[match_trials_ind]['abs_trial_num'].values
        laser_info_frame['abs_trial_num'] = match_trials

    else:

        # Dummy (place-holder) data
        laser_info_frame = pd.DataFrame(
            dict(
                dig_in_num=np.nan,
                dig_in_name=np.nan,
                laser=False,
                start=np.nan,
                end=np.nan,
                abs_trial_num=taste_info_frame['abs_trial_num'].values,
            ),
        )

    # Merge taste and laser info frames
    trial_info_frame = taste_info_frame.merge(
        laser_info_frame,
        on='abs_trial_num',
        how='left',
            suffixes=('_taste', '_laser')
    )

    # Calculate laser lag and duration
    trial_info_frame['laser_duration'] = (
        trial_info_frame['end_laser'] - trial_info_frame['start_laser']
    )
    trial_info_frame['laser_lag'] = (
        trial_info_frame['start_laser'] - trial_info_frame['start_taste']
    )

    # Convert to sec
    sec_cols = ['start_taste', 'end_taste', 'start_laser', 'end_laser',
                'laser_duration', 'laser_lag']
    for col in sec_cols:
        new_col_name = col + '_ms'
        trial_info_frame[new_col_name] = (
            trial_info_frame[col] / sampling_rate)*1000

    ###############
    # Correct laser timing using info_dict

    # laser_onset = info_dict['laser_params']['onset']
    # laser_duration = info_dict['laser_params']['duration']
    laser_onset_duration_params = np.array(
        info_dict['laser_params']['onset_duration'])

    trial_info_frame['laser_duration_ms'].fillna(0, inplace=True)
    trial_info_frame['laser_lag_ms'].fillna(0, inplace=True)

    trial_info_frame['laser_duration_ms'] = \
        trial_info_frame['laser_duration_ms'].astype(int)
    trial_info_frame['laser_lag_ms'] = \
        trial_info_frame['laser_lag_ms'].astype(int)

    # Save originals to make figure
    orig_duration = trial_info_frame['laser_duration_ms'].copy()
    orig_lag = trial_info_frame['laser_lag_ms'].copy()

    # Match by closest vector match
    if len(laser_onset_duration_params) > 0:
        nonzero_inds = trial_info_frame['laser_duration_ms'] > 0
        onset_duration_vectors = np.array(
            [trial_info_frame['laser_lag_ms'].values[nonzero_inds],
             trial_info_frame['laser_duration_ms'].values[nonzero_inds]]
        ).T
        match_ind = [
            np.argmin(np.linalg.norm(x - laser_onset_duration_params, axis=1))
            for x in onset_duration_vectors
        ]
        trial_info_frame.loc[nonzero_inds, 'laser_lag_ms'] = [
            laser_onset_duration_params[x][0] for x in match_ind
        ]
        trial_info_frame.loc[nonzero_inds,
                             'laser_duration_ms'] = [
            laser_onset_duration_params[x][1] for x in match_ind
        ]

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(orig_lag, orig_duration, label='Original', alpha=0.5)
    ax.scatter(trial_info_frame['laser_lag_ms'],
               trial_info_frame['laser_duration_ms'], label='Corrected')
    ax.set_xlabel('Laser Lag (ms)')
    ax.set_ylabel('Laser Duration (ms)')
    ax.legend()
    # plt.show()
    fig.savefig(os.path.join(metadata_handler.dir_name, 'QA_output', 'laser_timing_correction.png'),
                bbox_inches='tight')
    plt.close()

    ##############################
    # Create barplot of taste_duration_ms across dig-ins
    fig, ax = plt.subplots(figsize=(8, 6))
    taste_duration_data = trial_info_frame.groupby(
        'dig_in_name_taste')['taste_duration_ms'].apply(list)
    positions = np.arange(len(taste_duration_data))

    for i, (dig_in_name, durations) in enumerate(taste_duration_data.items()):
        ax.bar(i, np.mean(durations), yerr=np.std(durations),
               capsize=5, alpha=0.7, label=dig_in_name)

    ax.set_xlabel('Dig-in')
    ax.set_ylabel('Taste Duration (ms)')
    ax.set_title('Taste Duration Across Dig-ins')
    ax.set_xticks(positions)
    ax.set_xticklabels(taste_duration_data.index, rotation=45, ha='right')
    fig.tight_layout()
    fig.savefig(os.path.join(metadata_handler.dir_name, 'QA_output', 'taste_duration_barplot.png'),
                bbox_inches='tight')
    plt.close()

    ##############################
    # Save trial info frame to hdf5 file and csv

    trial_info_frame.to_hdf(metadata_handler.hdf5_name,
                            'trial_info_frame', mode='a')
    csv_path = os.path.join(metadata_handler.dir_name, 'trial_info_frame.csv')
    trial_info_frame.to_csv(csv_path, index=False)

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
        groupby('dig_in_num_taste').start_taste.apply(np.array).tolist()

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
        for num, this_starts in zip(taste_digin_nums, taste_starts_cutoff):
            # dig_in_basename = this_dig_handler.dig_in_frame.loc[i, 'dig_in_nums']
            dig_in_basename = f'dig_in_{num}'
            # print(f'Creating spike-trains for {dig_in_basename[i]}')
            print(f'Creating spike-trains for dig-in {num}')
            create_spike_trains_for_digin(
                this_starts,
                dig_in_basename,
                durations,
                sampling_rate_ms,
                units,
                hf5,
            )
        ###############
        # Write out laser_duration and lag to hdf5 file
        if True in trial_info_frame['laser'] and '/spike_trains' in hf5:
            trial_info_group = \
                [x[1] for x in trial_info_frame.groupby('dig_in_num_taste')]
            for this_group in trial_info_group:
                this_group = this_group.sort_values('taste_rel_trial_num')
                laser_durations = this_group['laser_duration_ms'].values
                laser_lags = this_group['laser_lag_ms'].values
                # this_dig_in_name = this_group['dig_in_name_taste'].values[0]
                # dig_in_path = f'/spike_trains/{this_dig_in_name}'
                this_dig_in_num = this_group['dig_in_num_taste'].values[0]
                dig_in_path = f'/spike_trains/dig_in_{this_dig_in_num}'
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
        # for i, this_starts in zip(taste_digin_inds, taste_starts_cutoff):
        for num, this_starts in zip(taste_digin_nums, taste_starts_cutoff):
            # dig_in_basename = this_dig_handler.dig_in_frame.loc[i, 'dig_in_nums']
            dig_in_basename = f'dig_in_{num}'
            print(f'Creating emg-trials for dig-in {num}')
            create_emg_trials_for_digin(
                this_starts,
                # num,
                dig_in_basename,
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
