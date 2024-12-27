# Import stuff!
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tables
import easygui
import sys
import os
import pymc3 as pm
import theano.tensor as tt
import pickle
import identity_palatability_switch_functions as fn

# Read the blech_MCMC.dir file and change to the data directory
f = open('blech_MCMC.dir', 'r')
dir_name = []
for line in f.readlines():
	dir_name.append(line)
f.close()
os.chdir(dir_name[0][:-1])

# If running on jetstream (or personal computer) using GNU parallel, get sys.argv[1] - this is the trial number to be looked at
trial_num = int(sys.argv[1]) - 1

# Look for the hdf5 file in the directory
file_list = os.listdir('./')
hdf5_name = ''
for files in file_list:
	if files[-2:] == 'h5':
		hdf5_name = files

# Open the hdf5 file
hf5 = tables.open_file(hdf5_name, 'r')

# Read in the array of spikes
spikes_cat = hf5.root.MCMC_switch.categorical_spikes[:]

# Get the number of emissions in the data (number of unique neurons + 1) (+1 for the case when none of the neurons spike)
num_emissions = len(np.unique(spikes_cat))

# Get the laser condition and taste and trial number for this thread (indexed by trial_num above)
laser_condition = int(trial_num/(spikes_cat.shape[1]*spikes_cat.shape[2]))
taste_num = int((trial_num - laser_condition*spikes_cat.shape[1]*spikes_cat.shape[2])/spikes_cat.shape[2])
trial = int((trial_num - laser_condition*spikes_cat.shape[1]*spikes_cat.shape[2]) % spikes_cat.shape[2])

# Change to the correct laser/taste directory
os.chdir('MCMC_switch/Laser{:d}/Taste{:d}'.format(laser_condition, taste_num))

# Choose the switch function according to the laser condition being used
switch_functions = {'0': fn.laser_off_trials, '1': fn.laser_early_trials, '2': fn.laser_middle_trials, '3': fn.laser_late_trials}

# Get the model and trace after fitting the switching model with MCMC
model, tr = switch_functions[str(laser_condition)](spikes_cat[laser_condition, taste_num, trial, :], num_emissions)

# Set up things to plot the traceplot for this trial
fig, axarr = plt.subplots(4, 2)
axarr = pm.traceplot(tr, ax = axarr)
fig.savefig("Trial{:d}.png".format(trial + 1))
plt.close('all')

# Save the trace for this trial
with open('Trial{:d}_trace.pickle'.format(trial + 1), 'wb') as handle:
	pickle.dump(tr, handle, protocol = pickle.HIGHEST_PROTOCOL)

# Save the Gelam-Rubin convergence statistics for this trial
with open('Trial{:d}_Gelman_Rubin.pickle'.format(trial + 1), 'wb') as handle:
	pickle.dump(pm.gelman_rubin(tr), handle, protocol = pickle.HIGHEST_PROTOCOL)

hf5.close()
