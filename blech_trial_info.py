"""
This module handles the creation and management of trial information frames
from digital input data.

Functions:
- create_trial_info_frame: Creates a comprehensive trial information dataframe
  from taste and laser digital inputs
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from ast import literal_eval


def create_trial_info_frame(
        dig_handler,
        taste_digin_nums,
        laser_digin_nums,
        info_dict,
        sampling_rate,
        output_dir=None
):
    """
    Create a trial information frame from digital input data.
    
    Parameters
    ----------
    dig_handler : DigInHandler
        Handler object containing digital input data
    taste_digin_nums : list
        List of digital input numbers for taste stimuli
    laser_digin_nums : list
        List of digital input numbers for laser stimuli
    info_dict : dict
        Dictionary containing experimental information
    sampling_rate : float
        Sampling rate in Hz
    output_dir : str, optional
        Directory to save correction plots
        
    Returns
    -------
    pd.DataFrame
        Trial information frame with taste and laser timing data
    """
    
    # Create taste info frame
    taste_info_list = []
    for ind, num in enumerate(taste_digin_nums):
        this_dig = dig_handler.dig_in_frame.loc[
            dig_handler.dig_in_frame['dig_in_nums'] == num]
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

    # Add taste_rel_trial_num
    taste_grouped = taste_info_frame.groupby('dig_in_num')
    fin_group = []
    for name, group in taste_grouped:
        group['taste_rel_trial_num'] = np.arange(group.shape[0])
        fin_group.append(group)
    taste_info_frame = pd.concat(fin_group)
    taste_info_frame.sort_values(by=['start'], inplace=True)

    # Create laser info frame
    laser_info_list = []
    for ind, num in enumerate(laser_digin_nums):
        this_dig = dig_handler.dig_in_frame.loc[
            dig_handler.dig_in_frame['dig_in_nums'] == num]
        pulse_times = this_dig['pulse_times'].values[0]
        pulse_times = literal_eval(pulse_times)
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

    # Correct laser timing using info_dict
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
        trial_info_frame.loc[nonzero_inds, 'laser_duration_ms'] = [
            laser_onset_duration_params[x][1] for x in match_ind
        ]

    # Create correction plot if output directory provided
    if output_dir is not None:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(orig_lag, orig_duration, label='Original', alpha=0.5)
        ax.scatter(trial_info_frame['laser_lag_ms'],
                   trial_info_frame['laser_duration_ms'], label='Corrected')
        ax.set_xlabel('Laser Lag (ms)')
        ax.set_ylabel('Laser Duration (ms)')
        ax.legend()
        fig.savefig(os.path.join(output_dir, 'laser_timing_correction.png'),
                    bbox_inches='tight')
        plt.close()

    return trial_info_frame
