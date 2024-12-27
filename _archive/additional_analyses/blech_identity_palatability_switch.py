# Import stuff!
import numpy as np
import tables
import easygui
import sys
import os
import pymc3 as pm
import theano.tensor as tt
import multiprocessing
from scipy.stats import pearsonr
from collections import Counter

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
spike_params = easygui.multenterbox(msg = 'Fill in the parameters for processing your spike trains', fields = ['Pre-stimulus time used for making spike trains (ms)', 'Bin size for switchpoint detection (ms) - usually 10', 'Post-stimulus time for switchpoint detection (ms)'])
pre_stim = int(spike_params[0])
bin_size = int(spike_params[1])
post_stim = int(spike_params[2])

# Get the digital inputs/tastes available, then ask the user to rank them in order of palatability
trains_dig_in = hf5.list_nodes('/spike_trains')
palatability_rank = easygui.multenterbox(msg = 'Rank the digital inputs in order of palatability (1 for the lowest, only integers)', fields = [train._v_name for train in trains_dig_in])
for i in range(len(palatability_rank)):
	palatability_rank[i] = int(palatability_rank[i])

# Now ask the user to put in the identities of the digital inputs
identities = easygui.multenterbox(msg = 'Put in the identities of the digital inputs (only integers)', fields = [train._v_name for train in trains_dig_in])
for i in range(len(identities)):
	identities[i] = int(identities[i])

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

# Get the number of unique emissions in the array of categorical spikes
num_emissions = len(np.unique(spikes_cat))
# Also add 2 to the palatability ranks (so that they are distinct from identity labels)
palatability = palatability + 2

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

# Make lists to save 1.) The converged trial numbers, and 2.) The switchpoints on those trials in every laser condition
converged_trial_nums = []
switchpoints = []

# Also make lists to save the palatability rank and firing rates in every laser condition
pal = []
firing = []

# Another list to save the spiking data with the laser inactivations taken into account
inactivated_spikes = []

# Now run the MCMC inference for every laser condition

# Laser off trials
print("===========================================")
print("Running laser off trials")
with pm.Model() as model:
	# Dirichlet prior on the emission/spiking probabilities - 7 states (1 start, 2 identity, 4 palatability)
	p = pm.Dirichlet('p', np.ones(num_emissions), shape = (7, num_emissions))

	# Uniform switch times
	# Switch to identity firing
	t1 = pm.Uniform('t1', lower = 20, upper = 60, shape = num_trials)
	# Switch from identity to palatability firing
	t2 = pm.Uniform('t2', lower = t1 + 20, upper = 130, shape = num_trials)

	# Get the actual state numbers based on the switch times
	states1 = tt.switch(t1 >= np.repeat(np.arange(150)[:, None], num_trials, axis = 1), 0, identity[0, :])
	states = tt.switch(t2 >= np.repeat(np.arange(150)[:, None], num_trials, axis = 1), states1, palatability[0, :])

	# Define the log-likelihood function
	def logp(value):
		value = tt.cast(value, 'int32')
		return tt.sum(tt.log(p[states.T, value]))

	# Categorical observations
	obs = pm.DensityDist('obs', logp, observed = {'value': spikes_cat[0, :, :150]})

	# Inference button :D
	tr = pm.sample(1000000, init = None, step = pm.Metropolis(), njobs = 2, trace = [t1, t2], start = {'t1': np.ones(num_trials)*25.0, 't2': np.ones(num_trials)*120.0})

# Make a list to save the converged trial numbers and switchpoints for this laser condition
this_converged_trial_nums = []
this_switchpoints = []
# Lists for palatability ranks and firing rates in this laser condition
this_pal = []
this_firing = []
# Get the spiking data for this laser condition
inactivated_spikes.append(spikes[0, :, :150, :])
# Get the Gelman-Rubin convergence statistics
converged = pm.gelman_rubin(tr)
# Run through the trials in this condition
for i in range(num_trials):
	# Check if this trial converged
	if converged['t1'][i] < 1.1 and converged['t2'][i] < 1.1:
		# Save 1.) Trial number
		this_converged_trial_nums.append(i)
		# 2.) Switchpoints (averaged over the last 100k samples, skipping 100 samples at a time)
		start = int(np.mean(tr[-100000::100]['t1'][:, i]))
		end = int(np.mean(tr[-100000::100]['t2'][:, i]))
		#start = int(Counter(tr[-100000::100]['t1'][:, i].astype('int')).most_common()[0][0])
		#end = int(Counter(tr[-100000::100]['t2'][:, i].astype('int')).most_common()[0][0])
		this_switchpoints.append([start, end])
		# 3.) Palatability rank
		this_pal.append(palatability[0, i])
		# 4.) Firing rates
		this_firing.append([np.mean(inactivated_spikes[0][i, start:end, :], axis = 0), np.mean(inactivated_spikes[0][i, end:, :], axis = 0)])
# Append the lists for this laser condition to the overall lists
converged_trial_nums.append(np.array(this_converged_trial_nums))
switchpoints.append(np.array(this_switchpoints))
pal.append(np.array(this_pal))
firing.append(np.array(this_firing))

print("Laser off trials done")
print("==========================================")

# Laser early trials
print("===========================================")
print("Running laser early trials")
with pm.Model() as model:
	# Dirichlet prior on the emission/spiking probabilities - 7 states (1 start, 2 identity, 4 palatability)
	p = pm.Dirichlet('p', np.ones(num_emissions), shape = (7, num_emissions))

	# Uniform switch times
	# Switch to identity firing
	t1 = pm.Uniform('t1', lower = 10, upper = 60, shape = num_trials)
	# Switch from identity to palatability firing
	t2 = pm.Uniform('t2', lower = t1 + 20, upper = 130, shape = num_trials)

	# Get the actual state numbers based on the switch times
	states1 = tt.switch(t1 >= np.repeat(np.arange(150)[:, None], num_trials, axis = 1), 0, identity[1, :])
	states = tt.switch(t2 >= np.repeat(np.arange(150)[:, None], num_trials, axis = 1), states1, palatability[1, :])

	# Define the log-likelihood function
	def logp(value):
		value = tt.cast(value, 'int32')
		return tt.sum(tt.log(p[states.T, value]))

	# Categorical observations
	obs = pm.DensityDist('obs', logp, observed = {'value': spikes_cat[1, :, 50:200]})

	# Inference button :D
	tr = pm.sample(1000000, init = None, step = pm.Metropolis(), njobs = 2, trace = [t1, t2], start = {'t1': np.ones(num_trials)*25.0, 't2': np.ones(num_trials)*120.0})

# Make a list to save the converged trial numbers and switchpoints for this laser condition
this_converged_trial_nums = []
this_switchpoints = []
# Lists for palatability ranks and firing rates in this laser condition
this_pal = []
this_firing = []
# Get the spiking data for this laser condition
inactivated_spikes.append(spikes[1, :, 50:200, :])
# Get the Gelman-Rubin convergence statistics
converged = pm.gelman_rubin(tr)
# Run through the trials in this condition
for i in range(num_trials):
	# Check if this trial converged
	if converged['t1'][i] < 1.1 and converged['t2'][i] < 1.1:
		# Save 1.) Trial number
		this_converged_trial_nums.append(i)
		# 2.) Switchpoints (averaged over the last 100k samples, skipping 100 samples at a time)
		start = int(np.mean(tr[-100000::100]['t1'][:, i]))
		end = int(np.mean(tr[-100000::100]['t2'][:, i]))
#		start = int(Counter(tr[-100000::100]['t1'][:, i].astype('int')).most_common()[0][0])
#		end = int(Counter(tr[-100000::100]['t2'][:, i].astype('int')).most_common()[0][0])
		this_switchpoints.append([start, end])
		# 3.) Palatability rank
		this_pal.append(palatability[1, i])
		# 4.) Firing rates
		this_firing.append([np.mean(inactivated_spikes[1][i, start:end, :], axis = 0), np.mean(inactivated_spikes[1][i, end:, :], axis = 0)])
# Append the lists for this laser condition to the overall lists
converged_trial_nums.append(np.array(this_converged_trial_nums))
switchpoints.append(np.array(this_switchpoints))
pal.append(np.array(this_pal))
firing.append(np.array(this_firing))

print("Laser early trials done")
print("==========================================")

# Laser middle trials
print("===========================================")
print("Running laser middle trials")
with pm.Model() as model:
	# Dirichlet prior on the emission/spiking probabilities - 7 states (1 start, 2 identity, 4 palatability)
	p = pm.Dirichlet('p', np.ones(num_emissions), shape = (7, num_emissions))

	# Uniform switch times
	# Switch to identity firing
	t1 = pm.Uniform('t1', lower = 20, upper = 60, shape = num_trials)
	# Switch from identity to palatability firing
	t2 = pm.Uniform('t2', lower = t1 + 10, upper = 130, shape = num_trials)

	# Get the actual state numbers based on the switch times
	states1 = tt.switch(t1 >= np.repeat(np.arange(150)[:, None], num_trials, axis = 1), 0, identity[2, :])
	states = tt.switch(t2 >= np.repeat(np.arange(150)[:, None], num_trials, axis = 1), states1, palatability[2, :])

	# Define the log-likelihood function
	def logp(value):
		value = tt.cast(value, 'int32')
		return tt.sum(tt.log(p[states.T, value]))

	# Categorical observations
	obs = pm.DensityDist('obs', logp, observed = {'value': np.concatenate((spikes_cat[2, :, :70], spikes_cat[2, :, 120:200]), axis = 1)})

	# Inference button :D
	tr = pm.sample(1000000, init = None, step = pm.Metropolis(), njobs = 2, trace = [t1, t2], start = {'t1': np.ones(num_trials)*25.0, 't2': np.ones(num_trials)*90.0})

# Make a list to save the converged trial numbers and switchpoints for this laser condition
this_converged_trial_nums = []
this_switchpoints = []
# Lists for palatability ranks and firing rates in this laser condition
this_pal = []
this_firing = []
# Get the spiking data for this laser condition
inactivated_spikes.append(np.concatenate((spikes[2, :, :70, :], spikes[2, :, 120:200, :]), axis = 1))
# Get the Gelman-Rubin convergence statistics
converged = pm.gelman_rubin(tr)
# Run through the trials in this condition
for i in range(num_trials):
	# Check if this trial converged
	if converged['t1'][i] < 1.1 and converged['t2'][i] < 1.1:
		# Save 1.) Trial number
		this_converged_trial_nums.append(i)
		# 2.) Switchpoints (averaged over the last 100k samples, skipping 100 samples at a time)
		start = int(np.mean(tr[-100000::100]['t1'][:, i]))
		end = int(np.mean(tr[-100000::100]['t2'][:, i]))
#		start = int(Counter(tr[-100000::100]['t1'][:, i].astype('int')).most_common()[0][0])
#		end = int(Counter(tr[-100000::100]['t2'][:, i].astype('int')).most_common()[0][0])
		this_switchpoints.append([start, end])
		# 3.) Palatability rank
		this_pal.append(palatability[2, i])
		# 4.) Firing rates
		this_firing.append([np.mean(inactivated_spikes[2][i, start:end, :], axis = 0), np.mean(inactivated_spikes[2][i, end:, :], axis = 0)])
# Append the lists for this laser condition to the overall lists
converged_trial_nums.append(np.array(this_converged_trial_nums))
switchpoints.append(np.array(this_switchpoints))
pal.append(np.array(this_pal))
firing.append(np.array(this_firing))

print("Laser middle trials done")
print("==========================================")

# Laser late trials
print("===========================================")
print("Running laser late trials")
with pm.Model() as model:
	# Dirichlet prior on the emission/spiking probabilities - 7 states (1 start, 2 identity, 4 palatability)
	p = pm.Dirichlet('p', np.ones(num_emissions), shape = (7, num_emissions))

	# Uniform switch times
	# Switch to identity firing
	t1 = pm.Uniform('t1', lower = 20, upper = 60, shape = num_trials)
	# Switch from identity to palatability firing
	t2 = pm.Uniform('t2', lower = t1 + 20, upper = 130, shape = num_trials)

	# Get the actual state numbers based on the switch times
	states1 = tt.switch(t1 >= np.repeat(np.arange(150)[:, None], num_trials, axis = 1), 0, identity[3, :])
	states = tt.switch(t2 >= np.repeat(np.arange(150)[:, None], num_trials, axis = 1), states1, palatability[3, :])

	# Define the log-likelihood function
	def logp(value):
		value = tt.cast(value, 'int32')
		return tt.sum(tt.log(p[states.T, value]))

	# Categorical observations
	obs = pm.DensityDist('obs', logp, observed = {'value': np.concatenate((spikes_cat[3, :, :140], spikes_cat[3, :, 190:200]), axis = 1)})

	# Inference button :D
	tr = pm.sample(1000000, init = None, step = pm.Metropolis(), njobs = 2, trace = [t1, t2], start = {'t1': np.ones(num_trials)*25.0, 't2': np.ones(num_trials)*120.0})

# Make a list to save the converged trial numbers and switchpoints for this laser condition
this_converged_trial_nums = []
this_switchpoints = []
# Lists for palatability ranks and firing rates in this laser condition
this_pal = []
this_firing = []
# Get the spiking data for this laser condition
inactivated_spikes.append(np.concatenate((spikes[3, :, :140, :], spikes[3, :, 190:200, :]), axis = 1))
# Get the Gelman-Rubin convergence statistics
converged = pm.gelman_rubin(tr)
# Run through the trials in this condition
for i in range(num_trials):
	# Check if this trial converged
	if converged['t1'][i] < 1.1 and converged['t2'][i] < 1.1:
		# Save 1.) Trial number
		this_converged_trial_nums.append(i)
		# 2.) Switchpoints (averaged over the last 100k samples, skipping 100 samples at a time)
		start = int(np.mean(tr[-100000::100]['t1'][:, i]))
		end = int(np.mean(tr[-100000::100]['t2'][:, i]))
#		start = int(Counter(tr[-100000::100]['t1'][:, i].astype('int')).most_common()[0][0])
#		end = int(Counter(tr[-100000::100]['t2'][:, i].astype('int')).most_common()[0][0])
		this_switchpoints.append([start, end])
		# 3.) Palatability rank
		this_pal.append(palatability[3, i])
		# 4.) Firing rates
		this_firing.append([np.mean(inactivated_spikes[3][i, start:end, :], axis = 0), np.mean(inactivated_spikes[3][i, end:, :], axis = 0)])
# Append the lists for this laser condition to the overall lists
converged_trial_nums.append(np.array(this_converged_trial_nums))
switchpoints.append(np.array(this_switchpoints))
pal.append(np.array(this_pal))
firing.append(np.array(this_firing))

print("Laser late trials done")
print("==========================================")

# Save all these lists to the HDF5 file
# Inactivated spikes is a homogeneously sized list, so it can be saved to the HDF5 file on its own
hf5.create_array('/MCMC_switch', 'inactivated_spikes', inactivated_spikes)
# All the other need to be saved on a laser condition-by-condition basis
hf5.create_group('/MCMC_switch', 'converged_trial_nums')
hf5.create_group('/MCMC_switch', 'switchpoints')
hf5.create_group('/MCMC_switch', 'converged_trial_palatability')
hf5.create_group('/MCMC_switch', 'converged_trial_firing')
# Now run through the laser conditions and save the arrays for that condition to file
for laser in range(len(inactivated_spikes)):
	hf5.create_array('/MCMC_switch/converged_trial_nums', 'laser_condition_{:d}'.format(laser), converged_trial_nums[laser])
	hf5.create_array('/MCMC_switch/switchpoints', 'laser_condition_{:d}'.format(laser), switchpoints[laser])
	hf5.create_array('/MCMC_switch/converged_trial_palatability', 'laser_condition_{:d}'.format(laser), pal[laser])
	hf5.create_array('/MCMC_switch/converged_trial_firing', 'laser_condition_{:d}'.format(laser), firing[laser])
hf5.flush()

# ---------------------------Palatability correlation calculation---------------------------------------------------

# Make arrays to store r_pearson and p_pearson values
r_pearson = np.zeros((len(inactivated_spikes), 2, inactivated_spikes[0].shape[-1]))
p_pearson = np.ones((len(inactivated_spikes), 2, inactivated_spikes[0].shape[-1]))

# Run through the laser conditions
for laser in range(len(inactivated_spikes)):
	for unit in range(inactivated_spikes[0].shape[-1]):
		# Calculate palatability correlation in epoch 1
		r_pearson[laser, 0, unit], p_pearson[laser, 0, unit] = pearsonr(firing[laser][:, 0, unit], pal[laser])
		# If NaNs are produced (happens when firing rate is 0 for all palatability conditions), say that r = 0 and p = 1 (no correlation with palatability)
		if np.isnan(r_pearson[laser, 0, unit]):
			r_pearson[laser, 0, unit] = 0.0
			p_pearson[laser, 0, unit] = 1.0

		# Calculate palatability correlation in epoch 2
		r_pearson[laser, 1, unit], p_pearson[laser, 1, unit] = pearsonr(firing[laser][:, 1, unit], pal[laser])
		# If NaNs are produced (happens when firing rate is 0 for all palatability conditions), say that r = 0 and p = 1 (no correlation with palatability)
		if np.isnan(r_pearson[laser, 1, unit]):
			r_pearson[laser, 1, unit] = 0.0
			p_pearson[laser, 1, unit] = 1.0

# Save these correlation arrays to the hdf5 file
hf5.create_array('/MCMC_switch', 'r_pearson', r_pearson)
hf5.create_array('/MCMC_switch', 'p_pearson', p_pearson)
hf5.flush()
# ---------------------------End Palatability correlation calculation-----------------------------------------------

hf5.close()
