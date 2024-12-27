# Goes through files that were run through the variational HMM, picks the best solution (highest ELBO) for each taste, and lines up trials by the state transition time (like Sadacca et al)
# Does that separately for the laser and non-laser trials

import numpy as np
import tables
import easygui
import sys
import os
import pylab as plt
from scipy.stats import rankdata
from scipy.stats import spearmanr
from scipy.stats import pearsonr

# Ask the user for the hdf5 files that need to be plotted together
dirs = []
while True:
	dir_name = easygui.diropenbox(msg = 'Choose a directory with a hdf5 file, hit cancel to stop choosing')
	try:
		if len(dir_name) > 0:
			dirs.append(dir_name)
	except:
		break

# Ask the user for the pre and post stimulus times and the bin size used in the HMM analysis
params = easygui.multenterbox(msg = 'Fill in the parameters used for running the variational HMM analysis', fields = ['Pre stimulus time for making spike trains (ms)', 'Pre stimulus time for HMM (ms)', 'Post stimulus time for HMM (ms)', 'Bin size (ms)'])
pre_stim = int(params[0])
pre_stim_hmm = int(params[1])
post_stim_hmm = int(params[2])
bin_size = int(params[3])

# Make an array of time points with the provided parameters
time = np.arange(pre_stim_hmm, post_stim_hmm, bin_size)

# Ask the user for the time window to use while looking for the palatability state
palatability_time_window = easygui.multenterbox(msg = 'Enter the time limits to search for the palatability state in the latent state sequence', fields = ['Lower limit (ms), usually 800', 'Upper limit (ms), usually 2000'])
for i in range(len(palatability_time_window)):
	palatability_time_window[i] = int(palatability_time_window[i])
# Use the palatability window to find the time indices that fall in that window
palatability_times = np.where((time >= palatability_time_window[0])*(time <= palatability_time_window[1]))[0]

# Ask the user to enter the number of tastes in the files and their palatabilities
num_tastes = easygui.multenterbox(msg = 'Enter the number of tastes used in the experiments', fields = ['Number of tastes'])
num_tastes = int(num_tastes[0])
palatability = easygui.multenterbox(msg = 'Rank the tastes in order of palatability (1 for the lowest, only integers)', fields = ['taste {:d}'.format(i) for i in range(num_tastes)])
for i in range(len(palatability)):
	palatability[i] = int(palatability[i])

# Get the palatability calculation paramaters from the user
params = easygui.multenterbox(msg = 'Enter the parameters for palatability calculation', fields = ['Window size (ms)', 'Step size (ms)'])
for i in range(len(params)):
	params[i] = int(params[i])

# Now run through the directories
# Empty lists for appending palatability correlation data for both laser conditions
r_spearman_laser_off = []
p_spearman_laser_off = []
r_spearman_laser_on = []
p_spearman_laser_on = []
for dir_name in dirs:
	os.chdir(dir_name)
	# Locate the hdf5 file
	file_list = os.listdir('./')
	hdf5_name = ''
	for files in file_list:
		if files[-2:] == 'h5':
			hdf5_name = files

	# Open the hdf5 file
	hf5 = tables.open_file(hdf5_name, 'r')

	# Get all the digital inputs that the file has under /spike_trains
	dig_in = hf5.list_nodes("/spike_trains")

	# Make lists to store the spiking data for both laser conditions for all tastes
	laser_off_aligned = []
	laser_off_unaligned = []
	laser_on_aligned = []
	laser_on_unaligned = []

	# Find the single units in the file, only single units will be used in the correlation analysis
	all_units = hf5.list_nodes('/sorted_units')
	all_units = np.array([int(str(unit).split('/')[-1][4:7]) for unit in all_units])
	single_units = np.array([i for i in range(len(all_units)) if hf5.root.unit_descriptor[i]["single_unit"] == 1])

	# Indicator variable to show that more than 1 trial of each taste is recorded after HMM alignment
	trials_zero = 0

	# Run through the digital inputs
	for taste in dig_in:
		# Pull out the ELBO from every variational HMM solution for this taste
		# laser off trials first
		laser_off_solutions = hf5.list_nodes(taste.categorical_vb_hmm_results.laser_off)
		ELBO_laser_off = []
		for solution in laser_off_solutions:
			ELBO_laser_off.append(solution.ELBO.read())
		# Then do the laser on solutions
		laser_on_solutions = hf5.list_nodes(taste.categorical_vb_hmm_results.laser_on)
		ELBO_laser_on = []
		for solution in laser_on_solutions:
			ELBO_laser_on.append(solution.ELBO.read())

		# Pick the solution with the highest ELBO in each case
		best_solution_laser_off = laser_off_solutions[np.argmax(ELBO_laser_off)]
		best_solution_laser_on = laser_on_solutions[np.argmax(ELBO_laser_on)]

		# Get the posterior probabilities of the states from these solutions
		posterior_proba_laser_off = best_solution_laser_off.posterior_proba_VB[:]
		posterior_proba_laser_on = best_solution_laser_on.posterior_proba_VB[:]

		# First get the most dominant state across time on every trial
		laser_off_states = np.argmax(posterior_proba_laser_off, axis = 0)
		laser_on_states = np.argmax(posterior_proba_laser_on, axis = 0)

		# Then subset the time to the palatability window and find the state that dominates the most
		laser_off_states = np.unique(laser_off_states[:, palatability_times], return_counts = True)
		laser_off_pal_state = laser_off_states[0][np.argmax(laser_off_states[1])]
		laser_on_states = np.unique(laser_on_states[:, palatability_times], return_counts = True)
		laser_on_pal_state = laser_on_states[0][np.argmax(laser_on_states[1])]

		# Get the laser on and off trial numbers for this taste
		laser_off_trials = np.where(taste.laser_durations[:] == 0)[0]
		laser_on_trials = np.where(taste.laser_durations[:] > 0)[0]

		# Lists for storing spiking data for this taste
		this_taste_laser_off_aligned = []
		this_taste_laser_off_unaligned = []
		this_taste_laser_on_aligned = []
		this_taste_laser_on_unaligned = []

		# Run through the laser off trials
		for trial in range(len(laser_off_trials)):
			# Find the time that the dominant state first comes on
			state_onset = np.where(posterior_proba_laser_on[laser_on_pal_state, trial, :] > 0.5)[0]
			later_onset = np.where(np.ediff1d(state_onset) > 1)[0]
			# If the dominant state does go over 0.8 in probability during the trial, pick its onset
			if len(state_onset) > 0:
				if state_onset[0]*bin_size > 100:
					state_onset = state_onset[0]
				elif len(later_onset) > 0:
					state_onset = later_onset[0]
				else:
					continue
				print(state_onset)
				# Append spiking data for a total of 2.5s - 1s before to 1.5s after the state onset
				this_taste_laser_off_aligned.append(taste.spike_array[laser_off_trials[trial], single_units, pre_stim + state_onset*bin_size + pre_stim_hmm - 1000:pre_stim + state_onset*bin_size + pre_stim_hmm + 2500])
				# Append an equal 2.5s worth of unaligned spiking data
				this_taste_laser_off_unaligned.append(taste.spike_array[laser_off_trials[trial], single_units, pre_stim - 1000:pre_stim + 2500])

		# Do the same with the laser on trials
		for trial in range(len(laser_on_trials)):
			# Find the time that the dominant state first comes on
			state_onset = np.where(posterior_proba_laser_on[laser_on_pal_state, trial, :] > 0.5)[0]
			later_onset = np.where(np.ediff1d(state_onset) > 1)[0]
			#state_onset = np.where(np.ediff1d(state_times) > 1)[0]
			# If the dominant state does go over 0.8 in probability during the trial, pick its onset
			if len(state_onset) > 0:
				if state_onset[0]*bin_size > 100:
					state_onset = state_onset[0]
				elif len(later_onset) > 0:
					state_onset = later_onset[0]
				else:
					continue
				print(state_onset)
				# Append spiking data for a total of 2.5s - 1s before to 1.5s after the state onset
				this_taste_laser_on_aligned.append(taste.spike_array[laser_on_trials[trial], single_units, pre_stim + state_onset*bin_size + pre_stim_hmm - 1000:pre_stim + state_onset*bin_size + pre_stim_hmm + 2500])
				# Append an equal 2.5s worth of unaligned spiking data
				this_taste_laser_on_unaligned.append(taste.spike_array[laser_on_trials[trial], single_units, pre_stim - 1000:pre_stim + 2500])

		# Append the the taste specific spiking lists to the overall lists for this dataset
		if len(this_taste_laser_off_aligned) == 0 or len(this_taste_laser_on_aligned) == 0:
			trials_zero = 1
		#	break

		laser_off_aligned.append(this_taste_laser_off_aligned)
		laser_off_unaligned.append(this_taste_laser_off_unaligned)
		laser_on_aligned.append(this_taste_laser_on_aligned)
		laser_on_unaligned.append(this_taste_laser_on_unaligned)

	# In case the HMM solution resulted in state onsets under 100ms on all trials of a taste, skip this entire file
	if trials_zero == 1:
		hf5.close()
		continue

	# Consolidate firing data from all the tastes so that they can be used for palatability correlation calculation
	palatability_laser_off = np.concatenate([np.repeat(palatability[i], len(laser_off_aligned[i])) for i in range(num_tastes)])
	laser_off_aligned = np.vstack([np.array(laser_off_aligned[i]) for i in range(num_tastes)])
	laser_off_unaligned = np.vstack([np.array(laser_off_unaligned[i]) for i in range(num_tastes)])
	palatability_laser_on = np.concatenate([np.repeat(palatability[i], len(laser_on_aligned[i])) for i in range(num_tastes)])
	laser_on_aligned = np.vstack([np.array(laser_on_aligned[i]) for i in range(num_tastes)])
	laser_on_unaligned = np.vstack([np.array(laser_on_unaligned[i]) for i in range(num_tastes)])

	# Now bin the consolidated data in preparation for correlation analysis
	response_laser_off_aligned = np.array([np.mean(laser_off_aligned[:, :, time:time+params[0]], axis = -1) for time in range(0, 3500 - params[0] + params[1], params[1])])
	response_laser_off_unaligned = np.array([np.mean(laser_off_unaligned[:, :, time:time+params[0]], axis = -1) for time in range(0, 3500 - params[0] + params[1], params[1])])
	response_laser_on_aligned = np.array([np.mean(laser_on_aligned[:, :, time:time+params[0]], axis = -1) for time in range(0, 3500 - params[0] + params[1], params[1])])
	response_laser_on_unaligned = np.array([np.mean(laser_on_unaligned[:, :, time:time+params[0]], axis = -1) for time in range(0, 3500 - params[0] + params[1], params[1])])

	# Run through the neurons and correlate their binned responses with the taste palatability
	for unit in range(single_units.shape[0]):
		# First set of sublists is for aligned data, second set for unaligned data
		this_unit_r_laser_off = [[], []]
		this_unit_p_laser_off = [[], []]
		this_unit_r_laser_on = [[], []]
		this_unit_p_laser_on = [[], []]
		# Run through the time points and carry out a Spearman correlation for each time bin
		for time in range(response_laser_off_aligned.shape[0]):
			ranks = rankdata(response_laser_off_aligned[time, :, unit])
			r, p = spearmanr(ranks, palatability_laser_off)
			this_unit_r_laser_off[0].append(r)
			this_unit_p_laser_off[0].append(p)
			ranks = rankdata(response_laser_off_unaligned[time, :, unit])
			r, p = spearmanr(ranks, palatability_laser_off)
			this_unit_r_laser_off[1].append(r)
			this_unit_p_laser_off[1].append(p)
			ranks = rankdata(response_laser_on_aligned[time, :, unit])
			r, p = spearmanr(ranks, palatability_laser_on)
			this_unit_r_laser_on[0].append(r)
			this_unit_p_laser_on[0].append(p)
			ranks = rankdata(response_laser_on_unaligned[time, :, unit])
			r, p = spearmanr(ranks, palatability_laser_on)
			this_unit_r_laser_on[1].append(r)
			this_unit_p_laser_on[1].append(p)

		# Append the unit specific lists to the main lists of r and p
		r_spearman_laser_off.append(this_unit_r_laser_off)
		p_spearman_laser_off.append(this_unit_p_laser_off)
		r_spearman_laser_on.append(this_unit_r_laser_on)
		p_spearman_laser_on.append(this_unit_p_laser_on)

	# Close the hdf5 file
	hf5.close()

r_spearman_laser_off = np.array(r_spearman_laser_off)
p_spearman_laser_off = np.array(p_spearman_laser_off)
r_spearman_laser_on = np.array(r_spearman_laser_on)
p_spearman_laser_on = np.array(p_spearman_laser_on)
