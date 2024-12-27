# Import stuff!
import numpy as np
import tables
import easygui
import sys
import os
import multiprocessing

# Ask for the directory where the hdf5 file sits, and change to that directory
dir_name = easygui.diropenbox()
os.chdir(dir_name)

# Look for the hdf5 file in the directory
file_list = os.listdir('./')
hdf5_name = ''
for files in file_list:
	if files[-2:] == 'h5':
		hdf5_name = files

# Open the hdf5 file
hf5 = tables.open_file(hdf5_name, 'r+')

# Ask the user for the parameters to process spike trains
spike_params = easygui.multenterbox(msg = 'Fill in the parameters for processing your spike trains', fields = ['Pre-stimulus time used for making spike trains (ms)', 'Bin size for switchpoint detection (ms) - assumes 10', 'Post-stimulus time for switchpoint detection (ms) - assumes 2500'])
pre_stim = int(spike_params[0])
bin_size = int(spike_params[1])
post_stim = int(spike_params[2])

# Get the digital inputs/tastes available
trains_dig_in = hf5.list_nodes('/spike_trains')

# Ask the user about the type of units they want to do the calculations on (single or all units)
unit_type = easygui.multchoicebox(msg = 'Which type of units do you want to use?', choices = ('All units', 'Single units', 'Multi units', 'Custom choice'))
all_units = np.arange(trains_dig_in[0].spike_array.shape[1])
single_units = np.array([i for i in range(len(all_units)) if hf5.root.unit_descriptor[i]["single_unit"] == 1])
multi_units = np.array([i for i in range(len(all_units)) if hf5.root.unit_descriptor[i]["single_unit"] == 0])
chosen_units = []
if unit_type[0] == 'All units':
	chosen_units = all_units
elif unit_type[0] == 'Single units':
	chosen_units = single_units
elif unit_type[0] == 'Multi units':
	chosen_units = multi_units
else:
	chosen_units = easygui.multchoicebox(msg = 'Which units do you want to choose?', choices = ([i for i in all_units]))
	for i in range(len(chosen_units)):
		chosen_units[i] = int(chosen_units[i])
	chosen_units = np.array(chosen_units)

# Get the digital inputs/tastes available, then ask the user to rank them in order of palatability
trains_dig_in = hf5.list_nodes('/spike_trains')
palatability_rank = easygui.multenterbox(msg = 'Rank the digital inputs in order of palatability (1 for the lowest, only integers)', fields = [train._v_name for train in trains_dig_in])
for i in range(len(palatability_rank)):
	palatability_rank[i] = int(palatability_rank[i])

# Now ask the user to put in the identities of the digital inputs
identities = easygui.multenterbox(msg = 'Put in the identities of the digital inputs (only integers)', fields = [train._v_name for train in trains_dig_in])
for i in range(len(identities)):
	identities[i] = int(identities[i])
# Now make arrays to pull the data out
num_trials = trains_dig_in[0].spike_array.shape[0]
num_units = len(chosen_units)
time = trains_dig_in[0].spike_array.shape[2]
num_tastes = len(trains_dig_in)

# Pull out unique laser conditions if they exist
laser_conditions = 0
try:
	lasers = np.zeros((num_trials, 2))
	lasers[:, 0] = trains_dig_in[0].laser_durations[:]
	lasers[:, 1] = trains_dig_in[0].laser_onset_lag[:]
	unique_lasers = np.vstack({tuple(row) for row in lasers})
	unique_lasers = unique_lasers[unique_lasers[:, 0].argsort(), :]
	unique_lasers = unique_lasers[unique_lasers[:, 1].argsort(), :]
	num_laser_trials = len([i for i in range(lasers.shape[0]) if np.array_equal(lasers[i, :], unique_lasers[0, :])])
	laser_conditions += 1
except:
	pass

# Make the data arrays
if laser_conditions > 0:
	spikes = np.empty(shape = (unique_lasers.shape[0], num_laser_trials*num_tastes, int(post_stim/bin_size), num_units), dtype = int)
	palatability = np.empty(shape = (unique_lasers.shape[0], num_laser_trials*num_tastes), dtype = int)
	identity = np.empty(shape = (unique_lasers.shape[0], num_laser_trials*num_tastes), dtype = int)
else:
	spikes = np.empty(shape = (1, num_trials*num_tastes, int(post_stim/bin_size), num_units), dtype = int)
	palatability = np.empty(shape = (1, num_trials*num_tastes), dtype = int)
	identity = np.empty(shape = (1, num_trials*num_tastes), dtype = int)

# Fill in the data
for i in range(spikes.shape[0]):
	for j in range(num_tastes):
		if laser_conditions > 0:
			lasers = np.zeros((num_trials, 2))
			lasers[:, 0] = trains_dig_in[j].laser_durations[:]
			lasers[:, 1] = trains_dig_in[j].laser_onset_lag[:]
			# Get the trials for this laser condition and this taste
			these_trials = np.array([trial for trial in range(lasers.shape[0]) if np.array_equal(lasers[trial, :], unique_lasers[i, :])])
			# Get the spikes for these trials
			these_spikes = trains_dig_in[j].spike_array[these_trials, :, :]
			these_spikes = these_spikes[:, chosen_units, :]

			palatability[i, num_laser_trials*j : num_laser_trials*(j + 1)] = palatability_rank[j] * np.ones(num_laser_trials)
			identity[i, num_laser_trials*j : num_laser_trials*(j + 1)] = identities[j] * np.ones(num_laser_trials)

			# Bin the spiking data
			for k in range(pre_stim, pre_stim + post_stim, bin_size):
				spikes[i, num_laser_trials*j : num_laser_trials*(j + 1), int((k - pre_stim)/bin_size), :] = np.sum(these_spikes[:, :, k : k + bin_size], axis = -1)

		else:
			palatability[i, num_trials*j : num_trials*(j + 1)] = palatability_rank[j] * np.ones(num_trials)
			identity[i, num_trials*j : num_trials*(j + 1)] = identities[j] * np.ones(num_trials)
			for k in range(pre_stim, pre_stim + post_stim, bin_size):
				spikes[i, num_trials*j : num_trials*(j + 1), int((k - pre_stim)/bin_size), :] = np.sum(trains_dig_in[j].spike_array[:, chosen_units, k : k + bin_size], axis = -1)

# Also make an array of Bernoulli (0/1) spikes
spikes_bernoulli = np.zeros(spikes.shape)
spikes_bernoulli[spikes > 0.0] = 1.0

# Also make an array of categorical spikes
spikes_cat = np.empty(shape = spikes.shape[:-1], dtype = int)
for i in range(spikes.shape[0]):
	for j in range(spikes.shape[1]):
		for k in range(spikes.shape[2]):
			# Find active units
			active_units = np.where(spikes[i, j, k, :] > 0)[0] + 1
			if len(active_units) == 0:
				spikes_cat[i, j, k] = 0
			else:
				spikes_cat[i, j, k] = np.random.choice(active_units)

# Make a directory to store the results of the switching analysis
try:
	os.system("rm -r ./MCMC_switch")
except:
	pass
os.mkdir('MCMC_switch')

# Make directories for the laser conditions and tastes
for i in range(spikes_cat.shape[0]):
	os.chdir('MCMC_switch')
	os.mkdir('Laser{:d}'.format(i))
	os.chdir('Laser{:d}'.format(i))
	for j in range(num_tastes):
		os.mkdir('Taste{:d}'.format(j))
	os.chdir(dir_name)

# Reshape the categorical spikes to # laser conditions x # of tastes x # of trials x time
# Reshape the spikes and bernoulli spikes too
spikes_cat = spikes_cat.reshape((spikes_cat.shape[0], num_tastes, int(spikes_cat.shape[1]/num_tastes), spikes_cat.shape[2]))
spikes = spikes.reshape((spikes.shape[0], num_tastes, int(spikes.shape[1]/num_tastes), spikes.shape[2], spikes.shape[3]))
spikes_bernoulli = spikes_bernoulli.reshape((spikes_bernoulli.shape[0], num_tastes, int(spikes_bernoulli.shape[1]/num_tastes), spikes_bernoulli.shape[2], spikes_bernoulli.shape[3]))

# Reshape the identity and palatability arrays too
identity = identity.reshape((identity.shape[0], num_tastes, int(identity.shape[1]/num_tastes)))
palatability = palatability.reshape((palatability.shape[0], num_tastes, int(palatability.shape[1]/num_tastes)))

# Make a node to store the results of switching analysis in the hdf5 file
try:
	hf5.remove_node('/MCMC_switch', recursive = True)
except:
	pass
hf5.create_group('/', 'MCMC_switch')

# Save the reshaped categorical spikes to this node
hf5.create_array('/MCMC_switch', 'categorical_spikes', spikes_cat)
hf5.create_array('/MCMC_switch', 'spikes', spikes)
hf5.create_array('/MCMC_switch', 'bernoulli_spikes', spikes_bernoulli)
hf5.create_array('/MCMC_switch', 'identity', identity)
hf5.create_array('/MCMC_switch', 'palatability', palatability)
hf5.flush()

# Also save the unique laser conditions if they exist
try:
	hf5.create_array('/MCMC_switch', 'unique_lasers', unique_lasers)
	hf5.flush()
except:
	pass

# Also drop a .dir file in the user's blech_clust/additional_analyses folder on the Desktop (the process code will use this to know what directory to use data from)
# Grab Brandeis/Jetstream/personal computer unet username
username = easygui.multenterbox(msg = 'Enter your Brandeis/Jetstream/personal computer id', fields = ['username'])
os.chdir('/home/{:s}/Desktop/blech_clust/additional_analyses'.format(username[0]))
f = open('blech_MCMC.dir', 'w')
print(dir_name, file=f)
f.close()

# Also drop the GNU parallel shell file in the blech_clust/additional_analyses folder
f = open('identity_palatability_switch_parallel.sh', 'w')
# First get number of CPUs - parallel be asked to run num_cpu-1 threads in parallel - assign 2 cores per job (for sampling 2 chains)
num_cpu = multiprocessing.cpu_count()
print("parallel -k -j {:d} --noswap --load 100% --progress --memfree 4G --retry-failed --joblog {:s}/MCMC_switch/results.log python identity_palatability_switch_process.py ::: {{1..{:d}}}".format(int((num_cpu-1)/2), dir_name, num_trials*num_tastes), file = f)
f.close()

hf5.close()
