"""
This module performs a local BSA (Blind Source Analysis) on a single trial of EMG (Electromyography) data, running on a High-Performance Computing (HPC) environment. It calculates the Short-Time Fourier Transform (STFT) and processes the results for further analysis.

- `Logger` class: A custom logger that writes messages to both the terminal and a log file, with timestamps.
- `calc_stft(trial, max_freq, time_range_tuple, Fs, signal_window, window_overlap)`: Computes the Short-Time Fourier Transform (STFT) of a given trial signal, returning the frequency and time vectors, and the STFT matrix.
- `calc_stft_mode_freq(dat, BSA_output=True, **stft_params)`: Calculates the mode frequency of the STFT of a signal, optionally formatting the output to match BSA requirements. Returns frequency and time vectors, and the time-averaged mode frequency.
- The script sets up parameters by loading them from a JSON file and checks the pipeline status using a utility function.
- It processes a specified trial of EMG data, checking for non-zero data before running the BSA, and saves the results to files.
- The script logs the progress and results of the analysis, ensuring traceability and error checking.
"""
# Runs a local BSA analysis (see emg_local_BSA.py) on one trial of EMG data. Runs on the HPC

############################################################
# Imports
############################################################

# Import stuff
import numpy as np
import easygui
import os
import sys
import datetime
from scipy import signal
import json
script_path = os.path.realpath(__file__)
blech_clust_dir = os.path.dirname(os.path.dirname(script_path))
sys.path.append(blech_clust_dir)
from blech_clust.utils.blech_utils import pipeline_graph_check  # noqa: E402


class Logger(object):
    def __init__(self, log_file_path):
        self.terminal = sys.stdout
        self.log = open(log_file_path, "a")

    def append_time(self, message):
        now = str(datetime.datetime.now())
        ap_msg = f'[{now}] {message}'
        return ap_msg

    def write(self, message):
        ap_msg = self.append_time(message)
        self.terminal.write(ap_msg)
        self.log.write(ap_msg)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

############################################################
# Define functions
############################################################


def calc_stft(trial, max_freq, time_range_tuple,
              Fs, signal_window, window_overlap):
    """
    trial : 1D array
    max_freq : where to lob off the transform
    time_range_tuple : (start,end) in seconds, time_lims of spectrogram
                            from start of trial snippet`
    """
    f, t, this_stft = signal.stft(
        signal.detrend(trial),
        fs=Fs,
        # window='hann', # don't specify window to avoid version issues
        nperseg=signal_window,
        noverlap=signal_window-(signal_window-window_overlap))
    freq_bool = f <= max_freq
    this_stft = this_stft[np.where(freq_bool)[0]]
    this_stft = this_stft[:, np.where((t >= time_range_tuple[0]) *
                                      (t < time_range_tuple[1]))[0]]
    fin_freq = f[freq_bool]
    fin_t = t[np.where((t >= time_range_tuple[0])*(t < time_range_tuple[1]))]
    return fin_freq, fin_t, this_stft


def calc_stft_mode_freq(dat, BSA_output=True, **stft_params):
    """
    Calculate the mode frequency of the STFT of a signal

    Inputs:
        dat : 1D array, signal to be analyzed
        stft_params : dict, parameters for STFT calculation
        BSA_output : bool, whether to output in a similar format to the BSA

    Outputs:
        freq_vec : 1D array, frequency vector for STFT
        t_vec : 1D array, time vector for STFT
        weight_mean : 1D array, time averaged mode frequency
    """
    freq_vec, t_vec, stft = calc_stft(dat, **stft_params)
    mag = np.abs(stft)
    time_norm_mag = mag / mag.sum(axis=0)
    weight_mean = (freq_vec[:, None] * time_norm_mag).sum(axis=0)
    if BSA_output:
        # This is only needed if stft_step size > 1
        # Interpolate output to same length as input
        new_t_vec = np.arange(stft_params['time_range_tuple'][0],
                              stft_params['time_range_tuple'][1],
                              1/stft_params['Fs'])
        new_weight_mean = np.interp(new_t_vec, t_vec, weight_mean)
        new_freq = np.arange(0, 10, 0.5)
        new_weight_mean = np.zeros((len(new_freq), len(t_vec)))
        closest_freq = np.argmin(
            np.abs(weight_mean[:, None] - new_freq), axis=1)
        new_weight_mean[closest_freq, np.arange(len(t_vec))] = 1
        return new_freq, new_t_vec, new_weight_mean
    else:
        return freq_vec, t_vec, weight_mean


##############################
# Setup params
##############################
# script_path = os.path.realpath(__file__)
script_dir = os.path.dirname(script_path)
blech_clust_dir = os.path.dirname(script_dir)
emg_params_path = os.path.join(blech_clust_dir, 'params', 'emg_params.json')

if not os.path.exists(emg_params_path):
    print('=== Environment params file not found. ===')
    print(
        '==> Please copy [[ blech_clust/params/_templates/emg_params.json ]] to [[ blech_clust/params/env_params.json ]] and update as needed.')
    exit()

emg_params_dict = json.load(open(emg_params_path, 'r'))
stft_params = emg_params_dict['stft_params']

##############################
# Calculate STFT
##############################
# Read blech.dir, and cd to that directory.
with open('BSA_run.dir', 'r') as f:
    dir_name = [x.strip() for x in f.readlines()][0]

# Perform pipeline graph check
this_pipeline_check = pipeline_graph_check(dir_name)
this_pipeline_check.check_previous(script_path)
this_pipeline_check.write_to_log(script_path, 'attempted')

# If there is more than one dir in BSA_run.dir,
# loop over both, as both sets will have the same number of trials
# for dir_name in dir_list:
sys.stdout = Logger(os.path.join(dir_name, 'BSA_log.txt'))
os.chdir(os.path.join(dir_name, 'emg_output'))

# Read the data files
# emg_env = np.load('emg_env.npy')
emg_env = np.load('flat_emg_env_data.npy')
# sig_trials = np.load('sig_trials.npy')

task = int(sys.argv[1])

# print(f'Processing taste {taste}, trial {trial}')
print(f'Processing Trial {task}')

# Run BSA on trial 'trial' of taste 'taste' and assign the results to p and omega.
# input_data = emg_env[taste, trial, :]
input_data = emg_env[task]
# Check that trial is non-zero, if it isn't, don't try to run BSA
if not any(np.isnan(input_data)):

    omega, t_vec, p = calc_stft_mode_freq(
        input_data, **stft_params, BSA_output=True)
    p = p.T
    print(f'Trial {task:03} succesfully processed')
else:
    print(f'NANs in trial {task:03}, BSA will also output NANs')
    p = np.zeros((7000, 20))
    omega = np.zeros(20)
    p[:] = np.nan
    omega = np.nan

# Save p and omega by taste and trial number
np.save(os.path.join('emg_BSA_results', f'trial{task:03}_p.npy'), p)
np.save(os.path.join('emg_BSA_results', f'trial{task:03}_omega.npy'), omega)

# Write successful execution to log
this_pipeline_check.write_to_log(script_path, 'completed')
