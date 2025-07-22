"""
This module performs a local Bayesian Spectral Analysis (BSA) on a single trial of Electromyography (EMG) data using the High-Performance Computing (HPC) environment. It integrates Python with R to execute the BSA using the BaSAR package.

- `Logger` class: A custom logger that writes messages to both the terminal and a log file, appending timestamps to each message.
- Reads the directory name from 'BSA_run.dir' and changes the working directory to the specified path.
- Utilizes the `pipeline_graph_check` from `blech_utils` to verify the pipeline's previous state and log the attempt.
- Loads EMG data from a NumPy file and processes a specific trial based on a command-line argument.
- Interfaces with R using `rpy2` to perform BSA on the EMG data, checking for NaN values before processing.
- Saves the BSA results (`p` and `omega`) as NumPy files in the 'emg_BSA_results' directory.
- Logs the completion of the process using the pipeline check utility.
"""
# Runs a local BSA analysis (see emg_local_BSA.py) on one trial of EMG data. Runs on the HPC

# Import stuff
from rpy2.robjects import r
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
import rpy2.robjects as ro
import readline
import numpy as np
import easygui
import os
import sys
import datetime
script_path = os.path.realpath(__file__)
blech_clust_dir = os.path.dirname(os.path.dirname(script_path))
sys.path.append(blech_clust_dir)
from utils.blech_utils import pipeline_graph_check  # noqa: E402


class Logger(object):
    def __init__(self, log_file_path):
        self.terminal = sys.stdout
        log_dir = os.path.join(os.path.dirname(log_file_path), "logs")
        os.makedirs(log_dir, exist_ok=True)
        self.log = open(os.path.join(log_dir, os.path.basename(log_file_path)), "a")

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
# Read blech.dir, and cd to that directory.
with open('BSA_run.dir', 'r') as f:
    dir_name = [x.strip() for x in f.readlines()][0]

# Perform pipeline graph check
# script_path = os.path.realpath(__file__)
this_pipeline_check = pipeline_graph_check(dir_name)
this_pipeline_check.check_previous(script_path)
this_pipeline_check.write_to_log(script_path, 'attempted')

# If there is more than one dir in BSA_run.dir,
# loop over both, as both sets will have the same number of trials
# for dir_name in dir_list:
sys.stdout = Logger(os.path.join(dir_name, 'BSA_log.txt'))
os.chdir(os.path.join(dir_name, 'emg_output'))

# Read the data files
emg_env = np.load('flat_emg_env_data.npy')

task = int(sys.argv[1])

# print(f'Processing taste {taste}, trial {trial}')
print(f'Processing Trial {task}')

# Import R related stuff - use rpy2 for Python->R and pandas for R->Python
# Needed for the next line to work on Anaconda.
# Also needed to do conda install -c r rpy2 at the command line
rpy2.robjects.numpy2ri.activate()
# rpy.common got deprecated in newer versions of pandas. So we use rpy2 instead
# import pandas.rpy.common as com

# Fire up BaSAR on R
basar = importr('BaSAR')

# Make the time array and assign it to t on R
T = (np.arange(7000) + 1)/1000.0
t_r = ro.r.matrix(T, nrow=1, ncol=7000)
ro.r.assign('t_r', t_r)
ro.r('t = c(t_r)')

# Run BSA on trial 'trial' of taste 'taste' and assign the results to p and omega.
# input_data = emg_env[taste, trial, :]
input_data = emg_env[task]
# Check that trial is non-zero, if it isn't, don't try to run BSA
if not any(np.isnan(input_data)):

    Br = ro.r.matrix(input_data, nrow=1, ncol=7000)
    ro.r.assign('B', Br)
    ro.r('x = c(B[1,])')

    # x is the data,
    # we scan periods from 0.1s (10 Hz) to 1s (1 Hz) in 20 steps.
    # Window size is 300ms.
    # There are no background functions (=0)
    ro.r('r_local = BaSAR.local(x, 0.1, 1, 20, t, 0, 300)')
    p_r = r['r_local']
    # r_local is returned as a length 2 object,
    # with the first element being omega and the second being the
    # posterior probabilities. These need to be recast as floats
    p = np.array(p_r[1]).astype('float')
    omega = np.array(p_r[0]).astype('float')/(2.0*np.pi)
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
