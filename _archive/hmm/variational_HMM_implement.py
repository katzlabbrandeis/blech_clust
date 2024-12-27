# Import stuff!
import numpy as np
import tables
import easygui
import sys
import os
import pylab as plt
import multiprocessing as mp
import pickle
# Import PyHMM
sys.path.append('/home/narendra/Desktop/PyHMM/PyHMM')
import DiscreteHMM as dhmm
import variationalHMM as vhmm
from hinton import hinton

# Read blech.dir
f = open('blech.dir', 'r')
dir_name = []
for line in f.readlines():
	dir_name.append(line)
f.close()

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def implement_categorical(data = None, restarts = None, num_states = None, num_emissions = None, n_cpu = None, max_iter = None, threshold = None):
	pool = mp.Pool(processes = n_cpu)
	results = [pool.apply_async(run_categorical, args = (data, num_states, num_emissions, restart, max_iter, threshold,)) for restart in range(restarts)]

	output = [p.get() for p in results]
	return output

def run_categorical(data = None, num_states = None, num_emissions = None, restart = None, max_iter = None, threshold = None):
	np.random.seed(restart)
	model_MAP = dhmm.CategoricalHMM(num_states = num_states, num_emissions = num_emissions, max_iter = max_iter, threshold = threshold)
	model_MAP.fit(data = data, p_transitions = np.random.random((num_states, num_states)), p_emissions = np.random.random((num_states, num_emissions)), p_start = np.random.random(num_states), \
transition_pseudocounts = np.random.random((num_states, num_states)), emission_pseudocounts = np.random.random((num_states, num_emissions)), start_pseudocounts = np.random.random(num_states), verbose = False)

	model_VI = vhmm.CategoricalHMM(num_states = num_states, num_emissions = num_emissions, max_iter = max_iter, threshold = threshold)
	model_VI.fit(data = data, transition_hyperprior=1, emission_hyperprior=1, start_hyperprior=1, initial_emission_counts=80*model_MAP.p_emissions, initial_transition_counts=80*model_MAP.p_transitions, \
initial_start_counts=8*model_MAP.p_start, verbose = False)

	return model_MAP, model_VI

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Pull out the NSLOTS - number of CPUs allotted
#n_cpu = int(os.getenv('NSLOTS'))
n_cpu = mp.cpu_count()
#n_cpu = int(sys.argv[1])

# Change to the data directory, get the names of all files in it, and find the .params and hdf5 (.h5) file
os.chdir(dir_name[0][:-1])
file_list = os.listdir('./')
hdf5_name = ''
params_file = ''
units_file = ''
for files in file_list:
	if files[-2:] == 'h5':
		hdf5_name = files
	if files[-10:] == 'hmm_params':
		params_file = files
	if files[-9:] == 'hmm_units':
		units_file = files

# Read the .hmm_params file
f = open(params_file, 'r')
params = []
for line in f.readlines():
	params.append(line)
f.close()

# Assign the params to variables
min_states = int(params[0])
max_states = int(params[1])
max_iterations = int(params[2])
threshold = float(params[3])
seeds = int(params[4])
taste = int(params[5])
pre_stim = int(params[6])
bin_size = int(params[7])
pre_stim_hmm = int(params[8])
post_stim_hmm = int(params[9])

# Read the chosen units
f = open(units_file, 'r')
chosen_units = []
for line in f.readlines():
	chosen_units.append(int(line))
chosen_units = np.array(chosen_units)

# Open up hdf5 file
hf5 = tables.open_file(hdf5_name, 'r+')

# Get the spike array from the required taste/input
exec('spikes = hf5.root.spike_trains.dig_in_%i.spike_array[:]' % taste)

# Slice out the required portion of the spike array, and bin it
spikes = spikes[:, chosen_units, pre_stim - pre_stim_hmm:pre_stim + post_stim_hmm]
binned_spikes = np.zeros((spikes.shape[0], int((pre_stim_hmm + post_stim_hmm)/bin_size)))
time = []
for i in range(spikes.shape[0]):
	time = []
	for k in range(0, spikes.shape[2], bin_size):
		time.append(k - pre_stim_hmm)
		n_firing_units = np.where(np.sum(spikes[i, :, k:k+bin_size], axis = 1) > 0)[0]
		if n_firing_units.size:
			n_firing_units = n_firing_units + 1
		else:
			n_firing_units = [0]
		binned_spikes[i, int(k/bin_size)] = np.random.choice(n_firing_units)

# Get the laser and non-laser trials for this taste
exec('dig_in = hf5.root.spike_trains.dig_in_%i' % taste)
on_trials = np.where(dig_in.laser_durations[:] > 0.0)[0]
off_trials = np.where(dig_in.laser_durations[:] == 0.0)[0]

# Delete the categorical_vb_hmm_results node under /spike_trains/dig_in_(taste)/ if it exists
try:
	hf5.remove_node('/spike_trains/dig_in_%i/categorical_vb_hmm_results' % taste, recursive = True)
except:
	pass

# Then create the categorical_vb_hmm_results group
hf5.create_group('/spike_trains/dig_in_%i' % taste, 'categorical_vb_hmm_results')
hf5.flush()

# Delete the Categorical folder within HMM_plots if it exists for this taste
try:
	os.system("rm -r ./variational_HMM_plots/dig_in_%i/Categorical" % taste)
except:
	pass

# Make a folder for plots of Multinomial HMM analysis
os.mkdir("variational_HMM_plots/dig_in_%i/Categorical" % taste)

# Running laser off trials first---------------------------------------------------------------------------------------------------------------------------------

# Implement a variational categorical HMM for no. of states defined by min_states and max_states
hmm_results = []
for n_states in range(min_states, max_states + 1):
	# Run the variational HMM (initialized with MAP parameters)
	result = implement_categorical(data = binned_spikes[off_trials, :], restarts = seeds, num_states = n_states, num_emissions = np.unique(binned_spikes).shape[0], n_cpu = n_cpu, max_iter = max_iterations, threshold = threshold)
	hmm_results.append(result)

# Clean up the results from the HMM analysis by just retaining the seed (for each number of states) that has the highest ELBO
cleaned_results = []
for result in hmm_results:
	# Pick only the seeds that converged
	converged_results = [seed for seed in result if seed[1].converged]
	# Skip to the next number of states if none of the seeds converged
	if len(converged_results) == 0:
		continue
	else:
		# Get the ELBO of all the seeds that converged
		ELBO = [seed[1].ELBO[-1] for seed in converged_results]
		# Append the seed with the highest ELBO to the cleaned_results
		cleaned_results.append(converged_results[np.argmax(ELBO)])

# Delete the laser_off node under /spike_trains/dig_in_(taste)/categorical_vb_hmm_results/ if it exists
try:
	exec("hf5.remove_node('/spike_trains/dig_in_%i/categorical_vb_hmm_results/laser_off' % taste, recursive = True)")
except:
	pass

# Then create the laser_off node under the categorical_vb_hmm_results group
exec("hf5.create_group('/spike_trains/dig_in_%i/categorical_vb_hmm_results' % taste, 'laser_off')")
hf5.flush()

# Delete the laser_off folder within variational_HMM_plots/(taste)/Categorical if it exists for this taste
try:
	os.system("rm -r ./variational_HMM_plots/dig_in_%i/Categorical/laser_off" % taste)
except:
	pass

# Make a folder for plots of Multinomial HMM analysis on laser off trials
os.mkdir("variational_HMM_plots/dig_in_%i/Categorical/laser_off" % taste)

# Go through the cleaned_results, and make plots for each state and each trial
for result in cleaned_results:
	# Make a plotting directory for this number of states
	os.mkdir("variational_HMM_plots/dig_in_%i/Categorical/laser_off/states_%i" % (taste, result[1].num_states))

	# Make a group under categorical_vb_hmm_results for this number of states
	hf5.create_group('/spike_trains/dig_in_%i/categorical_vb_hmm_results/laser_off' % taste, 'states_%i' % (result[1].num_states))

	# Write the start, transition and emission parameters and the posterior probabilities of the states only from the variational solution
	# First get the posterior probabilities of the states by doing an E-step
	alpha, beta, scaling, expected_latent_state, expected_latent_state_pair = result[1].E_step()
	start_counts = hf5.create_array('/spike_trains/dig_in_%i/categorical_vb_hmm_results/laser_off/states_%i' % (taste, result[1].num_states), 'start_counts', result[1].start_counts)
	transition_counts = hf5.create_array('/spike_trains/dig_in_%i/categorical_vb_hmm_results/laser_off/states_%i' % (taste, result[0].num_states), 'transition_counts', result[1].transition_counts)
	emission_counts = hf5.create_array('/spike_trains/dig_in_%i/categorical_vb_hmm_results/laser_off/states_%i' % (taste, result[0].num_states), 'emission_counts', result[1].emission_counts)
	posterior_proba_VB = hf5.create_array('/spike_trains/dig_in_%i/categorical_vb_hmm_results/laser_off/states_%i' % (taste, result[0].num_states), 'posterior_proba_VB', expected_latent_state)
	# Also write the ELBO to file
	ELBO = hf5.create_array('/spike_trains/dig_in_%i/categorical_vb_hmm_results/laser_off/states_%i' % (taste, result[0].num_states), 'ELBO', result[1].ELBO[-1])
	hf5.flush()
	# Also write the posterior probabilities of the states from the MAP solution to file
	alpha, beta, scaling, expected_latent_state, expected_latent_state_pair = result[0].E_step()
	posterior_proba_MAP = hf5.create_array('/spike_trains/dig_in_%i/categorical_vb_hmm_results/laser_off/states_%i' % (taste, result[0].num_states), 'posterior_proba_MAP', expected_latent_state)
	hf5.flush()

	# Go through laser off trials and plot the trial-wise posterior probabilities and raster plots
	# First make a dictionary of colors for the rasters
	raster_colors = {'regular_spiking': 'red', 'fast_spiking': 'blue', 'multi_unit': 'black'}
	for i in range(off_trials.shape[0]):
		# Plotting the variational solution first
		fig = plt.figure()
		for j in range(posterior_proba_VB.shape[0]):
			plt.plot(time, len(chosen_units)*posterior_proba_VB[j, i, :])
		for unit in range(len(chosen_units)):
			# Determine the type of unit we are looking at - the color of the raster will depend on that
			if hf5.root.unit_descriptor[chosen_units[unit]]['regular_spiking'] == 1:
				unit_type = 'regular_spiking'
			elif hf5.root.unit_descriptor[chosen_units[unit]]['fast_spiking'] == 1:
				unit_type = 'fast_spiking'
			else:
				unit_type = 'multi_unit'
			for j in range(spikes.shape[2]):
				if spikes[off_trials[i], unit, j] > 0:
					plt.vlines(j - pre_stim_hmm, unit, unit + 0.5, color = raster_colors[unit_type], linewidth = 0.5)
		plt.xlabel('Time post stimulus (ms)')
		plt.ylabel('Probability of HMM states')
		plt.title('VB_Trial %i, Dur: %ims, Lag:%ims' % (off_trials[i]+1, dig_in.laser_durations[off_trials[i]], dig_in.laser_onset_lag[off_trials[i]]) + '\n' + 'RSU: red, FS: blue, Multi: black')
		fig.savefig('variational_HMM_plots/dig_in_%i/Categorical/laser_off/states_%i/Trial_%i_VB.png' % (taste, result[1].num_states, off_trials[i] + 1))
		plt.close("all")

		# Now plotting the MAP solution
		fig = plt.figure()
		for j in range(posterior_proba_MAP.shape[0]):
			plt.plot(time, len(chosen_units)*posterior_proba_MAP[j, i, :])
		for unit in range(len(chosen_units)):
			# Determine the type of unit we are looking at - the color of the raster will depend on that
			if hf5.root.unit_descriptor[chosen_units[unit]]['regular_spiking'] == 1:
				unit_type = 'regular_spiking'
			elif hf5.root.unit_descriptor[chosen_units[unit]]['fast_spiking'] == 1:
				unit_type = 'fast_spiking'
			else:
				unit_type = 'multi_unit'
			for j in range(spikes.shape[2]):
				if spikes[off_trials[i], unit, j] > 0:
					plt.vlines(j - pre_stim_hmm, unit, unit + 0.5, color = raster_colors[unit_type], linewidth = 0.5)
		plt.xlabel('Time post stimulus (ms)')
		plt.ylabel('Probability of HMM states')
		plt.title('MAP_Trial %i, Dur: %ims, Lag:%ims' % (off_trials[i]+1, dig_in.laser_durations[off_trials[i]], dig_in.laser_onset_lag[off_trials[i]]) + '\n' + 'RSU: red, FS: blue, Multi: black')
		fig.savefig('variational_HMM_plots/dig_in_%i/Categorical/laser_off/states_%i/Trial_%i_MAP.png' % (taste, result[1].num_states, off_trials[i] + 1))
		plt.close("all")

	# Also pickle the model objects themselves to file in the plotting directory
	with open("variational_HMM_plots/dig_in_%i/Categorical/laser_off/states_%i/MAP_model.out" % (taste, result[0].num_states), "wb") as f:
		pickle.dump(result[0], f, pickle.HIGHEST_PROTOCOL)
	with open("variational_HMM_plots/dig_in_%i/Categorical/laser_off/states_%i/variational_model.out" % (taste, result[1].num_states), "wb") as f:
		pickle.dump(result[1], f, pickle.HIGHEST_PROTOCOL)

# Laser off trials done------------------------------------------------------------------------------------------------------------------------------------------


# Running laser on trials----------------------------------------------------------------------------------------------------------------------------------------

# Implement a variational categorical HMM for no. of states defined by min_states and max_states
hmm_results = []
for n_states in range(min_states, max_states + 1):
	# Run the variational HMM (initialized with MAP parameters)
	result = implement_categorical(data = binned_spikes[on_trials, :], restarts = seeds, num_states = n_states, num_emissions = np.unique(binned_spikes).shape[0], n_cpu = n_cpu, max_iter = max_iterations, threshold = threshold)
	hmm_results.append(result)

# Clean up the results from the HMM analysis by just retaining the seed (for each number of states) that has the highest ELBO
cleaned_results = []
for result in hmm_results:
	# Pick only the seeds that converged
	converged_results = [seed for seed in result if seed[1].converged]
	# Skip to the next number of states if none of the seeds converged
	if len(converged_results) == 0:
		continue
	else:
		# Get the ELBO of all the seeds that converged
		ELBO = [seed[1].ELBO[-1] for seed in converged_results]
		# Append the seed with the highest ELBO to the cleaned_results
		cleaned_results.append(converged_results[np.argmax(ELBO)])

# Delete the laser_on node under /spike_trains/dig_in_(taste)/categorical_vb_hmm_results/ if it exists
try:
	exec("hf5.remove_node('/spike_trains/dig_in_%i/categorical_vb_hmm_results/laser_on' % taste, recursive = True)")
except:
	pass

# Then create the laser_off node under the categorical_vb_hmm_results group
exec("hf5.create_group('/spike_trains/dig_in_%i/categorical_vb_hmm_results' % taste, 'laser_on')")
hf5.flush()

# Delete the laser_off folder within variational_HMM_plots/(taste)/Categorical if it exists for this taste
try:
	os.system("rm -r ./variational_HMM_plots/dig_in_%i/Categorical/laser_on" % taste)
except:
	pass

# Make a folder for plots of Multinomial HMM analysis on laser off trials
os.mkdir("variational_HMM_plots/dig_in_%i/Categorical/laser_on" % taste)

# Go through the cleaned_results, and make plots for each state and each trial
for result in cleaned_results:
	# Make a plotting directory for this number of states
	os.mkdir("variational_HMM_plots/dig_in_%i/Categorical/laser_on/states_%i" % (taste, result[1].num_states))

	# Make a group under categorical_vb_hmm_results for this number of states
	hf5.create_group('/spike_trains/dig_in_%i/categorical_vb_hmm_results/laser_on' % taste, 'states_%i' % (result[1].num_states))

	# Write the start, transition and emission parameters and the posterior probabilities of the states only from the variational solution
	# First get the posterior probabilities of the states by doing an E-step
	alpha, beta, scaling, expected_latent_state, expected_latent_state_pair = result[1].E_step()
	start_counts = hf5.create_array('/spike_trains/dig_in_%i/categorical_vb_hmm_results/laser_on/states_%i' % (taste, result[1].num_states), 'start_counts', result[1].start_counts)
	transition_counts = hf5.create_array('/spike_trains/dig_in_%i/categorical_vb_hmm_results/laser_on/states_%i' % (taste, result[0].num_states), 'transition_counts', result[1].transition_counts)
	emission_counts = hf5.create_array('/spike_trains/dig_in_%i/categorical_vb_hmm_results/laser_on/states_%i' % (taste, result[0].num_states), 'emission_counts', result[1].emission_counts)
	posterior_proba_VB = hf5.create_array('/spike_trains/dig_in_%i/categorical_vb_hmm_results/laser_on/states_%i' % (taste, result[0].num_states), 'posterior_proba_VB', expected_latent_state)
	# Also write the ELBO to file
	ELBO = hf5.create_array('/spike_trains/dig_in_%i/categorical_vb_hmm_results/laser_on/states_%i' % (taste, result[0].num_states), 'ELBO', result[1].ELBO[-1])
	hf5.flush()
	# Also write the posterior probabilities of the states from the MAP solution to file
	alpha, beta, scaling, expected_latent_state, expected_latent_state_pair = result[0].E_step()
	posterior_proba_MAP = hf5.create_array('/spike_trains/dig_in_%i/categorical_vb_hmm_results/laser_on/states_%i' % (taste, result[0].num_states), 'posterior_proba_MAP', expected_latent_state)
	hf5.flush()

	# Go through laser off trials and plot the trial-wise posterior probabilities and raster plots
	# First make a dictionary of colors for the rasters
	raster_colors = {'regular_spiking': 'red', 'fast_spiking': 'blue', 'multi_unit': 'black'}
	for i in range(on_trials.shape[0]):
		# Plotting the variational solution first
		fig = plt.figure()
		for j in range(posterior_proba_VB.shape[0]):
			plt.plot(time, len(chosen_units)*posterior_proba_VB[j, i, :])
		for unit in range(len(chosen_units)):
			# Determine the type of unit we are looking at - the color of the raster will depend on that
			if hf5.root.unit_descriptor[chosen_units[unit]]['regular_spiking'] == 1:
				unit_type = 'regular_spiking'
			elif hf5.root.unit_descriptor[chosen_units[unit]]['fast_spiking'] == 1:
				unit_type = 'fast_spiking'
			else:
				unit_type = 'multi_unit'
			for j in range(spikes.shape[2]):
				if spikes[on_trials[i], unit, j] > 0:
					plt.vlines(j - pre_stim_hmm, unit, unit + 0.5, color = raster_colors[unit_type], linewidth = 0.5)
		plt.xlabel('Time post stimulus (ms)')
		plt.ylabel('Probability of HMM states')
		plt.title('VB_Trial %i, Dur: %ims, Lag:%ims' % (on_trials[i]+1, dig_in.laser_durations[on_trials[i]], dig_in.laser_onset_lag[on_trials[i]]) + '\n' + 'RSU: red, FS: blue, Multi: black')
		fig.savefig('variational_HMM_plots/dig_in_%i/Categorical/laser_on/states_%i/Trial_%i_VB.png' % (taste, result[1].num_states, on_trials[i] + 1))
		plt.close("all")

		# Now plotting the MAP solution
		fig = plt.figure()
		for j in range(posterior_proba_MAP.shape[0]):
			plt.plot(time, len(chosen_units)*posterior_proba_MAP[j, i, :])
		for unit in range(len(chosen_units)):
			# Determine the type of unit we are looking at - the color of the raster will depend on that
			if hf5.root.unit_descriptor[chosen_units[unit]]['regular_spiking'] == 1:
				unit_type = 'regular_spiking'
			elif hf5.root.unit_descriptor[chosen_units[unit]]['fast_spiking'] == 1:
				unit_type = 'fast_spiking'
			else:
				unit_type = 'multi_unit'
			for j in range(spikes.shape[2]):
				if spikes[on_trials[i], unit, j] > 0:
					plt.vlines(j - pre_stim_hmm, unit, unit + 0.5, color = raster_colors[unit_type], linewidth = 0.5)
		plt.xlabel('Time post stimulus (ms)')
		plt.ylabel('Probability of HMM states')
		plt.title('MAP_Trial %i, Dur: %ims, Lag:%ims' % (on_trials[i]+1, dig_in.laser_durations[on_trials[i]], dig_in.laser_onset_lag[on_trials[i]]) + '\n' + 'RSU: red, FS: blue, Multi: black')
		fig.savefig('variational_HMM_plots/dig_in_%i/Categorical/laser_on/states_%i/Trial_%i_MAP.png' % (taste, result[1].num_states, on_trials[i] + 1))
		plt.close("all")

	# Also pickle the model objects themselves to file in the plotting directory
	with open("variational_HMM_plots/dig_in_%i/Categorical/laser_on/states_%i/MAP_model.out" % (taste, result[0].num_states), "wb") as f:
		pickle.dump(result[0], f, pickle.HIGHEST_PROTOCOL)
	with open("variational_HMM_plots/dig_in_%i/Categorical/laser_on/states_%i/variational_model.out" % (taste, result[1].num_states), "wb") as f:
		pickle.dump(result[1], f, pickle.HIGHEST_PROTOCOL)

# Laser on trials done-------------------------------------------------------------------------------------------------------------------------------------------

# Close the HDF5 file
hf5.close()
