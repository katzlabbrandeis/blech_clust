"""
This module handles the creation and management of trial information frames
from digital input data.

Functions:
- create_trial_info_frame: Creates a comprehensive trial information dataframe
  from taste and laser digital inputs
- _create_taste_info_frame: Extract taste trial information from digital inputs
- _create_laser_info_frame: Extract laser trial information from digital inputs
- _match_laser_to_taste_trials: Match laser pulses to corresponding taste trials
- _merge_trial_info: Merge taste and laser information into single dataframe
- _correct_laser_timing: Apply timing corrections based on experimental parameters
- _plot_laser_correction: Visualize laser timing corrections
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from ast import literal_eval


def _create_taste_info_frame(dig_handler, taste_digin_names):
    """
    Create taste information frame from digital input data.

    Parameters
    ----------
    dig_handler : DigInHandler
        Handler object containing digital input data
    taste_digin_names : list
        List of digital input names for taste stimuli

    Returns
    -------
    pd.DataFrame
        Taste information frame with trial timing and metadata
    """
    taste_info_list = []
    for ind, name in enumerate(taste_digin_names):
        this_dig = dig_handler.dig_in_frame.loc[
            dig_handler.dig_in_frame['dig_in_names'] == name]
        pulse_times = this_dig['pulse_times'].values[0]
        pulse_times = literal_eval(pulse_times)
        dig_in_num = this_dig['dig_in_nums'].values[0]
        this_frame = pd.DataFrame(
            dict(
                dig_in_num=dig_in_num,
                dig_in_name=name,
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
    taste_grouped = taste_info_frame.groupby('dig_in_name')
    fin_group = []
    for name, group in taste_grouped:
        group['taste_rel_trial_num'] = np.arange(group.shape[0])
        fin_group.append(group)
    taste_info_frame = pd.concat(fin_group)
    taste_info_frame.sort_values(by=['start'], inplace=True)

    return taste_info_frame


def _create_laser_info_frame(dig_handler, laser_digin_names):
    """
    Create laser information frame from digital input data.

    Parameters
    ----------
    dig_handler : DigInHandler
        Handler object containing digital input data
    laser_digin_names : list
        List of digital input names for laser stimuli

    Returns
    -------
    pd.DataFrame or None
        Laser information frame with pulse timing, or None if no laser inputs
    """
    if len(laser_digin_names) == 0:
        return None

    laser_info_list = []
    for ind, name in enumerate(laser_digin_names):
        this_dig = dig_handler.dig_in_frame.loc[
            dig_handler.dig_in_frame['dig_in_names'] == name]
        pulse_times = this_dig['pulse_times'].values[0]
        pulse_times = literal_eval(pulse_times)
        dig_in_num = this_dig['dig_in_nums'].values[0]
        this_frame = pd.DataFrame(
            dict(
                dig_in_num=dig_in_num,
                dig_in_name=name,
                laser=True,
                start=[x[0] for x in pulse_times],
                end=[x[1] for x in pulse_times],
            )
        )
        laser_info_list.append(this_frame)

    laser_info_frame = pd.concat(laser_info_list)
    return laser_info_frame


def _match_laser_to_taste_trials(laser_info_frame, taste_info_frame,
                                 info_dict, sampling_rate):
    """
    Match laser pulses to corresponding taste trials.

    Parameters
    ----------
    laser_info_frame : pd.DataFrame
        Laser information frame
    taste_info_frame : pd.DataFrame
        Taste information frame
    info_dict : dict
        Dictionary containing experimental information
    sampling_rate : float
        Sampling rate in Hz

    Returns
    -------
    pd.DataFrame
        Laser info frame with added abs_trial_num column

    Raises
    ------
    ValueError
        If any laser pulses don't match within tolerance
    """
    laser_starts = laser_info_frame['start'].values
    laser_ends = laser_info_frame['end'].values
    taste_starts = taste_info_frame['start'].values
    match_trials_ind = []
    mismatches = []

    if len(info_dict['laser_params']['onset_duration']) == 1:
        # Match laser starts to taste starts within tolerance
        match_tol = (2*sampling_rate)/10  # 200 ms
        print(
            f'Aligning laser to taste using exact match with tolerance of {match_tol/sampling_rate} sec')

        for idx, this_start in enumerate(laser_starts):
            match_ind = np.where(
                np.abs(taste_info_frame['start'] - this_start) < match_tol
            )[0]

            if not len(match_ind) == 1:
                # Find previous and next taste trials
                taste_diffs = taste_starts - this_start
                prev_taste_idx = np.where(taste_diffs < 0)[0]
                next_taste_idx = np.where(taste_diffs > 0)[0]

                prev_taste_time = taste_starts[prev_taste_idx[-1]] / \
                    sampling_rate if len(prev_taste_idx) > 0 else None
                next_taste_time = taste_starts[next_taste_idx[0]] / \
                    sampling_rate if len(next_taste_idx) > 0 else None

                laser_time = this_start / sampling_rate
                laser_duration = (laser_ends[idx] - this_start) / sampling_rate

                print(
                    f"\nWARNING: Laser pulse {idx} does not match within tolerance:")
                print(f"  Laser pulse time: {laser_time:.3f} sec")
                print(f"  Laser pulse duration: {laser_duration:.3f} sec")
                print(
                    f"  Previous taste trial time: {prev_taste_time:.3f} sec" if prev_taste_time is not None else "  Previous taste trial time: None (no previous trial)")
                print(
                    f"  Next taste trial time: {next_taste_time:.3f} sec" if next_taste_time is not None else "  Next taste trial time: None (no next trial)")
                print(f"  Tolerance: {match_tol/sampling_rate:.3f} sec")

                mismatches.append({
                    'laser_idx': idx,
                    'laser_time': laser_time,
                    'laser_duration': laser_duration,
                    'prev_taste_time': prev_taste_time,
                    'next_taste_time': next_taste_time
                })

                # Use closest match for now
                match_ind = np.array(
                    [np.argmin(np.abs(taste_starts - this_start))])

            match_trials_ind.append(match_ind[0])

        if mismatches:
            error_msg = f"\n{'='*60}\nERROR: {len(mismatches)} laser pulse(s) did not match within tolerance of {match_tol/sampling_rate:.3f} sec\n{'='*60}"
            print(error_msg)
            print('=== You might want to make a new laser_params onset_duration entry for this session, or check the timing of your laser pulses. ===')
            raise ValueError(error_msg)

    else:
        print('Aligning laser to taste using closest trial match')
        for this_start in laser_starts:
            match_ind = np.argmin(
                np.abs(taste_info_frame['start'] - this_start)
            )
            match_trials_ind.append(match_ind)

    match_trials = taste_info_frame.iloc[match_trials_ind]['abs_trial_num'].values
    laser_info_frame = laser_info_frame.copy()
    laser_info_frame['abs_trial_num'] = match_trials
    return laser_info_frame


def _merge_trial_info(taste_info_frame, laser_info_frame, sampling_rate):
    """
    Merge taste and laser information into a single trial info frame.

    Parameters
    ----------
    taste_info_frame : pd.DataFrame
        Taste information frame
    laser_info_frame : pd.DataFrame or None
        Laser information frame, or None if no laser data
    sampling_rate : float
        Sampling rate in Hz

    Returns
    -------
    pd.DataFrame
        Merged trial information frame with timing calculations
    """
    if laser_info_frame is None:
        # Create dummy laser data
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

    # Convert to ms
    sec_cols = ['start_taste', 'end_taste', 'start_laser', 'end_laser',
                'laser_duration', 'laser_lag']
    for col in sec_cols:
        new_col_name = col + '_ms'
        trial_info_frame[new_col_name] = (
            trial_info_frame[col] / sampling_rate)*1000

    return trial_info_frame


def _correct_laser_timing(trial_info_frame, info_dict):
    """
    Correct laser timing using experimental parameters.

    Parameters
    ----------
    trial_info_frame : pd.DataFrame
        Trial information frame with laser timing
    info_dict : dict
        Dictionary containing laser parameters

    Returns
    -------
    tuple
        (corrected_trial_info_frame, orig_duration, orig_lag)
    """
    laser_onset_duration_params = np.array(
        info_dict['laser_params']['onset_duration'])

    trial_info_frame['laser_duration_ms'].fillna(0, inplace=True)
    trial_info_frame['laser_lag_ms'].fillna(0, inplace=True)

    trial_info_frame['laser_duration_ms'] = \
        trial_info_frame['laser_duration_ms'].astype(int)
    trial_info_frame['laser_lag_ms'] = \
        trial_info_frame['laser_lag_ms'].astype(int)

    # Save originals for plotting
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

    return trial_info_frame, orig_duration, orig_lag


def _plot_laser_correction(trial_info_frame, orig_lag, orig_duration, output_dir):
    """
    Create visualization of laser timing corrections.

    Parameters
    ----------
    trial_info_frame : pd.DataFrame
        Trial information frame with corrected timing
    orig_lag : pd.Series
        Original laser lag values
    orig_duration : pd.Series
        Original laser duration values
    output_dir : str
        Directory to save the plot
    """
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


def create_trial_info_frame(
        dig_handler,
        taste_digin_names,
        laser_digin_names,
        info_dict,
        sampling_rate,
        output_dir=None
):
    """
    Create a trial information frame from digital input data.

    This function orchestrates the creation of a comprehensive trial information
    dataframe by extracting taste and laser timing data, matching them together,
    and applying timing corrections.

    Parameters
    ----------
    dig_handler : DigInHandler
        Handler object containing digital input data
    taste_digin_names : list
        List of digital input names for taste stimuli
    laser_digin_names : list
        List of digital input names for laser stimuli
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
    # Extract taste trial information
    taste_info_frame = _create_taste_info_frame(dig_handler, taste_digin_names)

    # Extract laser trial information
    laser_info_frame = _create_laser_info_frame(dig_handler, laser_digin_names)

    # Match laser pulses to taste trials
    if laser_info_frame is not None:
        laser_info_frame = _match_laser_to_taste_trials(
            laser_info_frame, taste_info_frame, info_dict, sampling_rate
        )

    # Merge taste and laser information
    trial_info_frame = _merge_trial_info(
        taste_info_frame, laser_info_frame, sampling_rate
    )

    # Correct laser timing based on experimental parameters
    trial_info_frame, orig_duration, orig_lag = _correct_laser_timing(
        trial_info_frame, info_dict
    )

    # Create correction plot if output directory provided
    if output_dir is not None:
        _plot_laser_correction(
            trial_info_frame, orig_lag, orig_duration, output_dir
        )

    return trial_info_frame
