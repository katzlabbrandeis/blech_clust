# os.chdir(os.path.expanduser('~/Desktop/blech_clust/emg/gape_QDA_classifier'))
import sys
import os
sys.path.append(os.path.expanduser('~/Desktop/blech_clust/emg/gape_QDA_classifier'))
import numpy as np
from scipy.signal import welch
from scipy.ndimage import white_tophat
from sklearn.decomposition import PCA
from detect_peaks import detect_peaks
import pandas as pd
from sklearn.preprocessing import StandardScaler
from QDA_classifier import QDA


def extract_movements(this_trial_dat, size = 250):
    filtered_dat = white_tophat(this_trial_dat, size=size)
    segments_raw = np.where(filtered_dat)[0]
    segments = np.zeros_like(filtered_dat)
    segments[segments_raw] = 1
    segment_starts = np.where(np.diff(segments) == 1)[0]
    segment_ends = np.where(np.diff(segments) == -1)[0]
    # If first start is after first end, drop first end
    # and last start
    if segment_starts[0] > segment_ends[0]:
        segment_starts = segment_starts[:-1]
        segment_ends = segment_ends[1:]
    segment_dat = [this_trial_dat[x:y]
                   for x, y in zip(segment_starts, segment_ends)]
    return segment_starts, segment_ends, segment_dat

def threshold_movement_lengths(
        segment_starts,
        segment_ends,
        segment_dat,
        min_len = 50,
        max_len = 500):
    """
    Threshold movement lengths
    """
    keep_inds = [x for x, y in enumerate(segment_dat) if len(y) > min_len and len(y) < max_len]
    segment_starts = segment_starts[keep_inds]
    segment_ends = segment_ends[keep_inds]
    segment_dat = [segment_dat[x] for x in keep_inds]
    return segment_starts, segment_ends, segment_dat

def normalize_segments(segment_dat):
    """
    Perform min-max normalization on each segment
    And make length of each segment equal to the longest segment
    """
    max_len = max([len(x) for x in segment_dat])
    interp_segment_dat = [np.interp(
        np.linspace(0, 1, max_len),
        np.linspace(0, 1, len(x)),
        x)
        for x in segment_dat]
    interp_segment_dat = np.vstack(interp_segment_dat)
    # Normalize
    interp_segment_dat = interp_segment_dat - \
        np.min(interp_segment_dat, axis=-1)[:, None]
    interp_segment_dat = interp_segment_dat / \
        np.max(interp_segment_dat, axis=-1)[:, None]
    return interp_segment_dat


def extract_features(segment_dat, segment_starts, segment_ends):
    """
    # Features to extract
    # 1. Duration of movement
    # 2. Amplitude
    # 3. Left and Right intervals
    # 4. PCA of time-adjusted waveform
    """
    peak_inds = [np.argmax(x) for x in segment_dat]
    peak_times = [x+y for x, y in zip(segment_starts, peak_inds)]
    # Drop first and last segments because we can't get intervals for them
    segment_dat = segment_dat[1:-1]
    segment_starts = segment_starts[1:-1]
    segment_ends = segment_ends[1:-1]

    durations = [len(x) for x in segment_dat]
    amplitudes_rel = [np.max(x) - np.min(x) for x in segment_dat]
    amplitude_abs = [np.max(x) for x in segment_dat]
    left_intervals = [peak_times[i] - peak_times[i-1]
                      for i in range(1, len(peak_times))][:-1]
    right_intervals = [peak_times[i+1] - peak_times[i]
                       for i in range(len(peak_times)-1)][1:]

    interp_segment_dat = normalize_segments(segment_dat)
    pca_segment_dat = PCA(n_components=3).fit_transform(interp_segment_dat)

    welch_out = [welch(x, fs=1000, axis=-1) for x in segment_dat]
    max_freq = [x[0][np.argmax(x[1], axis=-1)] for x in welch_out]

    feature_list = [
        durations,
        amplitudes_rel,
        amplitude_abs,
        left_intervals,
        right_intervals,
        pca_segment_dat,
        max_freq,
    ]
    feature_list = [np.atleast_2d(x) for x in feature_list]
    feature_list = [x if len(x) == len(pca_segment_dat) else x.T
                    for x in feature_list]
    feature_array = np.concatenate(feature_list, axis=-1)

    feature_names = [
        'duration',
        'amplitude_rel',
        'amplitude_abs',
        'left_interval',
        'right_interval',
        'pca_1',
        'pca_2',
        'pca_3',
        'max_freq',
    ]
    return feature_array, feature_names, segment_dat, segment_starts, segment_ends

def find_segment(gape_locs, segment_starts, segment_ends):
    segment_bounds = list(zip(segment_starts, segment_ends))
    all_segment_inds = []
    for this_gape in gape_locs:
        this_segment_inds = []
        for i, bounds in enumerate(segment_bounds):
            if bounds[0] < this_gape < bounds[1]:
                this_segment_inds.append(i)
        if len(this_segment_inds) ==0:
            this_segment_inds.append(np.nan)
        all_segment_inds.append(this_segment_inds)
    return np.array(all_segment_inds).flatten()

def calc_peak_interval(peak_ind):
    """
    This function calculates the inter-burst-interval

    Inputs:
    peak_ind : (num_peaks,)

    Outputs:
    intervals : (num_peaks,)
    """

    # Get inter-burst-intervals for the accepted peaks,
    # convert to Hz (from ms)
    intervals = []
    for peak in range(len(peak_ind)):
        # For the first peak,
        # the interval is counted from the second peak
        if peak == 0:
            intervals.append(
                1000.0/(peak_ind[peak+1] - peak_ind[peak]))
        # For the last peak, the interval is
        # counted from the second to last peak
        elif peak == len(peak_ind) - 1:
            intervals.append(
                1000.0/(peak_ind[peak] - peak_ind[peak-1]))
        # For every other peak, take the largest interval
        else:
            intervals.append(
                1000.0/(
                    np.amax([(peak_ind[peak] - peak_ind[peak-1]),
                             (peak_ind[peak+1] - peak_ind[peak])])
                )
            )
    intervals = np.array(intervals)

    return intervals

def get_peak_edges(peak_ind, below_mean_ind):
    """
    Get edges of peaks which are "below_mean_ind"
    If one edge doesn't come back below baseline, drop that peak
    """
    left_end_list = []
    right_end_list = []
    keep_peaks = []
    for i, this_peak in enumerate(peak_ind):
        left_end_ind_list = np.where(below_mean_ind < this_peak)[0]
        right_end_ind_list = np.where(below_mean_ind > this_peak)[0]
        # Don't take peaks which are missing an end
        if len(left_end_ind_list) and len(right_end_ind_list):
            left_end = below_mean_ind[left_end_ind_list[-1]]
            right_end = below_mean_ind[right_end_ind_list[0]]
            left_end_list.append(left_end)
            right_end_list.append(right_end)
            keep_peaks.append(i)
    return (
            np.array(left_end_list), 
            np.array(right_end_list), 
            np.array(keep_peaks)
            )

def JL_process(
        this_trial_dat, 
        this_laser_prestim_dat, 
        pre_stim,
        post_stim,
        this_ind):
    """
    This function takes in a trial of data and returns the gapes
    according to Jenn Li's pipeline

    Inputs:
    trial_data : (time,)
    stim_t : 1

    Outputs:
    gapes : (time,)
    """

    peak_ind = detect_peaks(
        this_trial_dat,
        mpd=85,
        mph=np.mean(this_laser_prestim_dat) +
        np.std(this_laser_prestim_dat)
    )


    ## Drop first and last peaks
    #peak_ind = peak_ind[1:-1]

    # Get the indices, in the smoothed signal,
    # that are below the mean of the smoothed signal
    below_mean_ind = np.where(this_trial_dat <=
                              np.mean(this_laser_prestim_dat))[0]

    #plt.plot(this_trial_dat)
    #plt.plot(peak_ind, this_trial_dat[peak_ind], 'ro')
    #plt.axhline(np.mean(this_laser_prestim_dat), color='k')
    #plt.scatter(below_mean_ind, this_trial_dat[below_mean_ind], color='g')
    #plt.show()

    # Throw out peaks if they happen in the pre-stim period
    accept_peaks = np.where(peak_ind > pre_stim)[0]
    peak_ind = peak_ind[accept_peaks]
    if len(peak_ind) == 0:
        return None


    # Run through the accepted peaks, and append their breadths to durations.
    # There might be peaks too close to the end of the trial -
    # skip those. Append the surviving peaks to final_peak_ind
    # Also threshold by durations
    left_end_list, right_end_list, keep_peaks = \
            get_peak_edges(peak_ind, below_mean_ind)
    if len(peak_ind) == 0:
        return None
    peak_ind = peak_ind[keep_peaks]

        #left_end_list = [np.where(below_mean_ind < peak)[0][-1] \
        #        for peak in peak_ind]
        #right_end_list = [np.where(below_mean_ind > peak)[0][0] \
        #        for peak in peak_ind]

    #dur = below_mean_ind[np.array(right_end_list)]-below_mean_ind[np.array(left_end_list)]
    dur = right_end_list - left_end_list 
    dur_bool = np.logical_and(dur > 20.0, dur <= 200.0)
    durations = dur[dur_bool]
    peak_ind = peak_ind[dur_bool]
    if len(peak_ind) == 0:
        return None

    # In case there aren't any peaks or just one peak
    # (very unlikely), skip this trial 
    if len(peak_ind) > 1:
        intervals = calc_peak_interval(peak_ind)

        gape_bool = [QDA(intervals[peak], durations[peak]) for peak in range(len(durations))]
        # Drop first one
        gape_bool = gape_bool[1:]
        peak_ind = peak_ind[1:]
        # Make sure the peaks are within 2000-5000 ms of the stimulus
        peak_bool = np.logical_and(peak_ind > pre_stim, peak_ind <= post_stim)
        fin_bool = np.logical_and(gape_bool, peak_bool)

        gape_peak_ind = peak_ind[fin_bool]
    else:
        gape_peak_ind = None

    return gape_peak_ind

# Convert segment_dat and gapes_Li to pandas dataframe for easuer handling
def gen_gape_frame(segment_dat_list, gapes_Li, inds):
    """
    Generate a dataframe with the following columns:
    channel, taste, trial, segment_num, features, segment_raw, segment_bounds

    Inputs:
    segment_dat_list : list of lists

    Returns:
    gape_frame : pandas dataframe
    """

    gape_frame = pd.DataFrame(data = inds, 
                              columns = ['channel', 'taste', 'trial'])
    # Standardize features
    gape_frame['features'] = [x[0] for x in segment_dat_list]
    gape_frame['segment_raw'] = [x[1] for x in segment_dat_list]
    gape_frame['segment_bounds'] = [x[2] for x in segment_dat_list]
    gape_frame = gape_frame.explode(['features','segment_raw','segment_bounds'])

    # Standardize features
    raw_features = np.stack(gape_frame['features'].values)
    scaled_features = StandardScaler().fit_transform(raw_features)
    gape_frame['features'] = [x for x in scaled_features]

    # Add index for each segment
    gape_frame['segment_num'] = gape_frame.groupby(['channel', 'taste', 'trial']).cumcount()
    gape_frame = gape_frame.reset_index(drop=True)

    # Add classifier boolean
    for row_ind, this_row in gape_frame.iterrows():
        this_ind = (this_row['channel'], this_row['taste'], this_row['trial'])
        bounds = this_row['segment_bounds']
        if gapes_Li[this_ind][bounds[0]:bounds[1]].any():
            gape_frame.loc[row_ind, 'classifier'] = 1
        else:
            gape_frame.loc[row_ind, 'classifier'] = 0
    # Convert to int
    gape_frame['classifier'] = gape_frame['classifier'].astype(int)

    return gape_frame, scaled_features
