"""
Post processing cleanup of the mess of files created by emg_local_BSA_execute.py. 
All the output files will be saved to p (named by tastes) and omega 
in the hdf5 file under the node emg_BSA_results
"""

# Import stuff
import numpy as np
import easygui
import os
import tables
import glob
import json
import sys

sys.path.append('..')
from utils.blech_utils import imp_metadata

# Get name of directory with the data files
metadata_handler = imp_metadata(sys.argv)
dir_name = metadata_handler.dir_name
os.chdir(dir_name)
print(f'Processing : {dir_name}')

# Open the hdf5 file
hf5 = tables.open_file(metadata_handler.hdf5_name, 'r+')

# Delete the raw_emg node, if it exists in the hdf5 file, 
# to cut down on file size
try:
    hf5.remove_node('/raw_emg', recursive = 1)
except:
    print("Raw EMG recordings have already been removed, so moving on ..")

# Extract info experimental info file
info_dict = metadata_handler.info_dict
taste_names = info_dict['taste_params']['tastes']
trials = [int(x) for x in info_dict['taste_params']['trial_count']]

if len(np.unique(trials)) > 1:
    print(f'Uneven numbers of trials detected : {trials}')
    print(f'Using max number of trials for array : {np.max(trials)}') 
    print(f'!! WARNING !! emg_BSA_results will have trials with zero data')

############################################################
## Following will be looped over emg channels
# In case there is more than one pair/location or differencing did not happen
############################################################
output_list = glob.glob(os.path.join(dir_name,'emg_output/*'))
output_list = [x for x in output_list if 'emg' in os.path.basename(x)]
channel_dirs = sorted([x for x in output_list if os.path.isdir(x)])
channels_discovered = [os.path.basename(x) for x in channel_dirs]
print(f'Creating plots for : {channels_discovered}\n')

# Add group to hdf5 file for emg BSA results
if '/emg_BSA_results' in hf5:
    hf5.remove_node('/','emg_BSA_results', recursive = True)
hf5.create_group('/', 'emg_BSA_results')

for num, this_dir in enumerate(channel_dirs):
    os.chdir(this_dir)
    this_basename = channels_discovered[num]
    print(f'Processing data for : {this_basename}')

    # Load sig_trials.npy to get number of tastes
    sig_trials = np.load('sig_trials.npy')
    tastes = sig_trials.shape[0]

    print(f'Trials taken from info file ::: {dict(zip(taste_names, trials))}')

    # Change to emg_BSA_results
    os.chdir('emg_BSA_results')

    # Omega doesn't vary by trial, 
    # so just pick it up from the 1st taste and trial, 
    first_omega = 'taste00_trial00_omega.npy'
    if os.path.exists(first_omega):
        omega = np.load(first_omega)

        # Add omega to the hdf5 file
        if '/emg_BSA_results/omega' not in hf5:
            atom = tables.Atom.from_dtype(omega.dtype)
            om = hf5.create_carray('/emg_BSA_results', 'omega', atom, omega.shape)
            om[:] = omega 
            hf5.flush()

        base_dir = '/emg_BSA_results'
        if os.path.join(base_dir, this_basename) in hf5:
            hf5.remove_node(base_dir, this_basename, recursive = True)
        hf5.create_group(base_dir, this_basename)


        # Load one of the p arrays to find out the time length of the emg data
        p = np.load('taste00_trial00_p.npy')
        time_length = p.shape[0]

        # Go through the tastes and trials
        # todo: Output to HDF5 needs to be named by channel
        for i in range(tastes):
            # Make an array for posterior probabilities for each taste
            #p = np.zeros((trials[i], time_length, 20))
            # Make array with highest numbers of trials, so uneven trial numbers
            # can be accomadated
            p = np.zeros((np.max(trials), time_length, 20))
            for j in range(trials[i]):
                p[j, :, :] = np.load(f'taste{i:02}_trial{j:02}_p.npy')
            # Save p to hdf5 file
            atom = tables.Atom.from_dtype(p.dtype)
            prob = hf5.create_carray(
                    os.path.join(base_dir, this_basename), 
                    'taste%i_p' % i, 
                    atom, 
                    p.shape)
            prob[:, :, :] = p
        hf5.flush()

        # TODO: Since BSA returns most dominant frequency, BSA output is 
        #       HIGHLY compressible. Change to utilizing timeseries rather than
        #       time-frequency representation

        # Since BSA is an expensive process, don't delete anything
        # In case things need to be reanalyzed

    else:
        print(f'No data found for channel {this_basename}')
        print('Computer will self-destruct in T minus 10 seconds')
    print('\n')
    print('================================')

# Close the hdf5 file
hf5.close()
