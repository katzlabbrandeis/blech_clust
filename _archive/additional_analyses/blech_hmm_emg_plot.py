import matplotlib
matplotlib.use('Agg')

# Import stuff!
import numpy as np
import tables
import easygui
import sys
import os
import matplotlib.pyplot as plt

# Ask for the directory where the hdf5 file sits, and change to that directory
dir_name = easygui.diropenbox()
os.chdir(dir_name)

# Look for the hdf5 file in the directory
file_list = os.listdir('./')
hdf5_name = ''
units_file = ''
for files in file_list:
	if files[-2:] == 'h5':
		hdf5_name = files
	if files[-9:] == 'hmm_units':
		units_file = files

# Read the chosen units
f = open(units_file, 'r')
chosen_units = []
for line in f.readlines():
	chosen_units.append(int(line))
chosen_units = np.array(chosen_units)

# Open the hdf5 file
hf5 = tables.open_file(hdf5_name, 'r+')

# Get the digital inputs/tastes
trains_dig_in = hf5.list_nodes('/spike_trains')

# Get the pre stimulus time from the hdf5 file
pre_stim = int(hf5.root.ancillary_analysis.pre_stim.read())

# Delete the directory for storing HMM-EMG plots if it exists, and make a new one
try:
	os.system('rm -r ./HMM_EMG_plots')
except:
	pass
os.mkdir('./HMM_EMG_plots')

# Pull out the EMG BSA results
gapes = hf5.root.ancillary_analysis.gapes[:]
ltps = hf5.root.ancillary_analysis.ltps[:]

# Pull out the gapes according to Li et al. 2016
gapes_Li = hf5.root.ancillary_analysis.gapes_Li[:]

# Pull out the significant trials on EMG
sig_trials = hf5.root.ancillary_analysis.sig_trials[:]

# Pull outthe laser conditions and the trials organized by laser condition
lasers = hf5.root.ancillary_analysis.laser_combination_d_l[:]
trials = hf5.root.ancillary_analysis.trials[:]

# Run through the digital inputs
for dig_in in trains_dig_in:

	# Get the taste number
	taste_num = int(str.split(dig_in._v_pathname, '/')[-1][-1])

	# Get the spike array from the required taste/input
	spikes = dig_in.spike_array[:]
	# Take only the units that were chosen while fitting the HMM
	spikes = spikes[:, chosen_units, :]

	# Make a directory for this digital input
	os.mkdir('./HMM_EMG_plots/dig_in_{:d}'.format(taste_num))

	# First check if this digital input has multinomial_hmm_results
	if hf5.__contains__('/spike_trains/dig_in_{:d}/multinomial_hmm_results'.format(taste_num)):
		# If it does, then make a folder for multinomial hmm plots
		os.mkdir('./HMM_EMG_plots/dig_in_{:d}/multinomial'.format(taste_num))

		# List the nodes under multinomial_hmm_results
		hmm_nodes = hf5.list_nodes('/spike_trains/dig_in_{:d}/multinomial_hmm_results'.format(taste_num))

		# Run through the hmm_nodes, make folders for each of them, and plot the posterior probabilities
		for node in hmm_nodes:
			# Check if the current node is the laser node
			if str.split(node._v_pathname, '/')[-1] == 'laser':
				# Get the nodes with the laser results
				laser_nodes = hf5.list_nodes('/spike_trains/dig_in_{:d}/multinomial_hmm_results/laser'.format(taste_num))

				# Run through the laser_nodes, make folders for each of them, and plot the posterior probabilities
				os.mkdir('./HMM_EMG_plots/dig_in_{:d}/multinomial/laser'.format(taste_num))
				for laser_node in laser_nodes:
					# Make a folder for this node
					os.mkdir('./HMM_EMG_plots/dig_in_{:d}/multinomial/laser/{:s}'.format(taste_num, str.split(laser_node._v_pathname, '/')[-1]))
					# Change to this directory
					os.chdir('./HMM_EMG_plots/dig_in_{:d}/multinomial/laser/{:s}'.format(taste_num, str.split(laser_node._v_pathname, '/')[-1]))

					# Get the HMM time
					time = laser_node.time[:]
					# And the posterior probability to plot
					posterior_proba = laser_node.posterior_proba[:]

					# Make an array of posterior probabilities arranged by laser conditions
					final_proba = np.zeros((lasers.shape[0], int(posterior_proba.shape[0]/lasers.shape[0]), posterior_proba.shape[1], posterior_proba.shape[2]))

					# Get the limits of plotting
					start = 100*(int(time[0]/100))
					end = 100*(int(time[-1]/100) + 1)

					# Slice out the required length of the array of spikes
					spikes_current = spikes[:, :, pre_stim - np.abs(start) : pre_stim + np.abs(end)]

					# Make directories for the plots
					os.mkdir('./gapes')
					os.mkdir('./ltps')
					# Make folders by laser conditions too
					for condition in lasers:
						os.mkdir('./gapes/Dur%i,Lag%i' % (int(condition[0]), int(condition[1])))
						os.mkdir('./ltps/Dur%i,Lag%i' % (int(condition[0]), int(condition[1])))
					# Run through the trials
					for i in range(posterior_proba.shape[0]):
						# Locate this trial number in the lasers X trial X.. array called trials
						laser_condition = int(np.where(trials == posterior_proba.shape[0]*taste_num + i)[0][0])
						this_taste_trials = np.where((trials[laser_condition] >= posterior_proba.shape[0]*taste_num) * (trials[laser_condition] <= posterior_proba.shape[0]*(taste_num + 1)))
						this_trial = int(np.where(trials[laser_condition, this_taste_trials][0] == posterior_proba.shape[0]*taste_num + i)[0])

						# Fill up the final_proba array
						final_proba[laser_condition, this_trial, :, :] = posterior_proba[i, :, :]

						# Plot the gapes, gapes_Li and posterior_proba
						fig = plt.figure()
						for j in range(posterior_proba.shape[2]):
							plt.plot(time, len(chosen_units)*posterior_proba[i, :, j])
						if sig_trials[laser_condition, taste_num, this_trial] > 0.0:
							plt.plot(np.arange(end), len(chosen_units)*gapes[laser_condition, taste_num, this_trial, :end])
							plt.plot(np.arange(end), len(chosen_units)*gapes_Li[laser_condition, taste_num, this_trial, pre_stim : pre_stim + end], linewidth = 2.0, color = 'black')
						# First make a dictionary of colors for the rasters
						raster_colors = {'regular_spiking': 'red', 'fast_spiking': 'blue', 'multi_unit': 'black'}
						for unit in range(len(chosen_units)):
							# Determine the type of unit we are looking at - the color of the raster will depend on that
							if hf5.root.unit_descriptor[chosen_units[unit]]['regular_spiking'] == 1:
								unit_type = 'regular_spiking'
							elif hf5.root.unit_descriptor[chosen_units[unit]]['fast_spiking'] == 1:
								unit_type = 'fast_spiking'
							else:
								unit_type = 'multi_unit'
							for j in range(spikes_current.shape[2]):
								if spikes_current[i, unit, j] > 0:
									plt.vlines(j - np.abs(start), unit, unit + 0.5, color = raster_colors[unit_type], linewidth = 0.5)

						plt.xlabel('Time post stimulus (ms)')
						plt.ylabel('Probability of HMM states' + '\n' + '% Power < 4.6Hz, Gapes from Li et al')
						plt.title('Trial %i, Dur: %ims, Lag:%ims' % (i+1, dig_in.laser_durations[i], dig_in.laser_onset_lag[i]))
						fig.savefig('./gapes/Dur%i,Lag%i/Trial_%i.png' % (int(lasers[laser_condition, 0]), int(lasers[laser_condition, 1]), i+1))
						plt.close("all")

						# Plot the ltps, and posterior_proba
						fig = plt.figure()
						for j in range(posterior_proba.shape[2]):
							plt.plot(time, posterior_proba[i, :, j])
						if sig_trials[laser_condition, taste_num, this_trial] > 0.0:
							plt.plot(np.arange(end), len(chosen_units)*ltps[laser_condition, taste_num, this_trial, :end])

						# First make a dictionary of colors for the rasters
						raster_colors = {'regular_spiking': 'red', 'fast_spiking': 'blue', 'multi_unit': 'black'}
						for unit in range(len(chosen_units)):
							# Determine the type of unit we are looking at - the color of the raster will depend on that
							if hf5.root.unit_descriptor[chosen_units[unit]]['regular_spiking'] == 1:
								unit_type = 'regular_spiking'
							elif hf5.root.unit_descriptor[chosen_units[unit]]['fast_spiking'] == 1:
								unit_type = 'fast_spiking'
							else:
								unit_type = 'multi_unit'
							for j in range(spikes_current.shape[2]):
								if spikes_current[i, unit, j] > 0:
									plt.vlines(j - np.abs(start), unit, unit + 0.5, color = raster_colors[unit_type], linewidth = 0.5)
						plt.xlabel('Time post stimulus (ms)')
						plt.ylabel('Probability of HMM states' + '\n' + '% Power in 5.95-8.6Hz')
						plt.title('Trial %i, Dur: %ims, Lag:%ims' % (i+1, dig_in.laser_durations[i], dig_in.laser_onset_lag[i]))
						fig.savefig('./ltps/Dur%i,Lag%i/Trial_%i.png' % (int(lasers[laser_condition, 0]), int(lasers[laser_condition, 1]), i+1))
						plt.close("all")


					# Plot the trial-averaged HMM posterior probabilities
					mean_proba = np.mean(final_proba, axis = 1)
					for i in range(mean_proba.shape[0]):
						fig = plt.figure()
						for j in range(mean_proba.shape[2]):
							plt.plot(time, mean_proba[i, :, j])
						plt.xlabel('Time post stimulus (ms)')
						plt.ylabel('Trial averaged probabilities' + '\n' + 'of HMM states')
						plt.title('Dur: %ims, Lag:%ims' % (int(lasers[i, 0]), int(lasers[i, 1])))
						fig.savefig('./gapes/Dur%i,Lag%i/Average_HMM.png' % (int(lasers[i, 0]), int(lasers[i, 1])))
						fig.savefig('./ltps/Dur%i,Lag%i/Average_HMM.png' % (int(lasers[i, 0]), int(lasers[i, 1])))
						plt.close('all')

					# Plot the trial-averaged gape and ltp frequencies
					mean_gapes = np.mean(gapes[:, :, :, :], axis = 2)
					mean_ltps = np.mean(ltps[:, :, :, :], axis = 2)
					for i in range(mean_gapes.shape[0]):
						fig = plt.figure()
						plt.plot(np.arange(end), mean_gapes[i, taste_num, :end])
						plt.xlabel('Time post stimulus (ms)')
						plt.ylabel('Trial-averaged fraction of' + '\n' + 'power in the 4-6Hz range')
						plt.title('Dur: %ims, Lag:%ims' % (int(lasers[i, 0]), int(lasers[i, 1])))
						fig.savefig('./gapes/Dur%i,Lag%i/Average_gapes.png' % (int(lasers[i, 0]), int(lasers[i, 1])))
						plt.close('all')

						fig = plt.figure()
						plt.plot(np.arange(end), mean_ltps[i, taste_num, :end])
						plt.xlabel('Time post stimulus (ms)')
						plt.ylabel('Trial-averaged fraction of' + '\n' + 'power in the 6-10Hz range')
						plt.title('Dur: %ims, Lag:%ims' % (int(lasers[i, 0]), int(lasers[i, 1])))
						fig.savefig('./ltps/Dur%i,Lag%i/Average_ltps.png' % (int(lasers[i, 0]), int(lasers[i, 1])))
						plt.close('all')

					# Save the final_proba array to the hdf5 file
					hf5.create_array(laser_node, 'final_proba', final_proba)
					hf5.flush()

					# Go back to the data directory
					os.chdir(dir_name)

			else:
				# Make a folder for this node
				os.mkdir('./HMM_EMG_plots/dig_in_{:d}/multinomial/{:s}'.format(taste_num, str.split(node._v_pathname, '/')[-1]))
				# Change to this directory
				os.chdir('./HMM_EMG_plots/dig_in_{:d}/multinomial/{:s}'.format(taste_num, str.split(node._v_pathname, '/')[-1]))
				# Get the HMM time
				time = node.time[:]
				# And the posterior probability to plot
				posterior_proba = node.posterior_proba[:]

				# Make an array of posterior probabilities arranged by laser conditions
				final_proba = np.zeros((lasers.shape[0], int(posterior_proba.shape[0]/lasers.shape[0]), posterior_proba.shape[1], posterior_proba.shape[2]))

				# Get the limits of plotting
				start = 100*(int(time[0]/100))
				end = 100*(int(time[-1]/100) + 1)

				# Make directories for the plots
				os.mkdir('./gapes')
				os.mkdir('./ltps')
				# Run through the trials
				for i in range(posterior_proba.shape[0]):
					# Locate this trial number in the lasers X trial X.. array called trials
					laser_condition = int(np.where(trials == posterior_proba.shape[0]*taste_num + i)[0][0])
					this_taste_trials = np.where((trials[laser_condition] >= posterior_proba.shape[0]*taste_num) * (trials[laser_condition] <= posterior_proba.shape[0]*(taste_num + 1)))
					this_trial = int(np.where(trials[laser_condition, this_taste_trials][0] == posterior_proba.shape[0]*taste_num + i)[0])

					# Fill up the final_proba array
					final_proba[laser_condition, this_trial, :, :] = posterior_proba[i, :, :]

					# Plot the gapes, gapes_Li and posterior_proba
					fig = plt.figure()
					for j in range(posterior_proba.shape[2]):
						plt.plot(time, posterior_proba[i, :, j])
					if sig_trials[laser_condition, taste_num, this_trial] > 0.0:
						plt.plot(np.arange(end), gapes[laser_condition, taste_num, this_trial, :end])
						plt.plot(np.arange(end), gapes_Li[laser_condition, taste_num, this_trial, pre_stim : pre_stim + end], linewidth = 2.0, color = 'black')
					plt.xlabel('Time post stimulus (ms)')
					plt.ylabel('Probability of HMM states' + '\n' + '% Power < 4.6Hz, Gapes from Li et al')
					plt.title('Trial %i' % (i+1))
					fig.savefig('./gapes/Trial_%i.png' % (i+1))
					plt.close("all")

					# Plot the ltps, and posterior_proba
					fig = plt.figure()
					for j in range(posterior_proba.shape[2]):
						plt.plot(time, posterior_proba[i, :, j])
					if sig_trials[laser_condition, taste_num, this_trial] > 0.0:
						plt.plot(np.arange(end), ltps[laser_condition, taste_num, this_trial, :end])
					plt.xlabel('Time post stimulus (ms)')
					plt.ylabel('Probability of HMM states' + '\n' + '% Power in 5.95-8.6Hz')
					plt.title('Trial %i' % (i+1))
					fig.savefig('./ltps/Trial_%i.png' % (i+1))
					plt.close("all")

				# Plot the trial-averaged HMM posterior probabilities
				mean_proba = np.mean(final_proba, axis = (0, 1))
				fig = plt.figure()
				for j in range(mean_proba.shape[1]):
					plt.plot(time, mean_proba[:, j])
				plt.xlabel('Time post stimulus (ms)')
				plt.ylabel('Trial averaged probabilities' + '\n' + 'of HMM states')
				plt.title('Trial-averaged HMM')
				fig.savefig('./gapes/Average_HMM.png')
				fig.savefig('./ltps/Average_HMM.png')
				plt.close('all')

				# Plot the trial-averaged gape and ltp frequencies
				mean_gapes = np.mean(gapes[:, :, :, :], axis = (0, 2))
				mean_ltps = np.mean(ltps[:, :, :, :], axis = (0, 2))
				fig = plt.figure()
				plt.plot(np.arange(end), mean_gapes[taste_num, :end])
				plt.xlabel('Time post stimulus (ms)')
				plt.ylabel('Trial-averaged fraction of' + '\n' + 'power in the 4-6Hz range')
				plt.title('Trial-averaged power in gaping range')
				fig.savefig('./gapes/Average_gapes.png')
				plt.close('all')

				fig = plt.figure()
				plt.plot(np.arange(end), mean_ltps[taste_num, :end])
				plt.xlabel('Time post stimulus (ms)')
				plt.ylabel('Trial-averaged fraction of' + '\n' + 'power in the 6-10Hz range')
				plt.title('Trial-averaged power in LTP range')
				fig.savefig('./ltps/Average_ltps.png')
				plt.close('all')

				# Save the final_proba array to the hdf5 file
				hf5.create_array(node, 'final_proba', final_proba)
				hf5.flush()

				# Go back to the data directory
				os.chdir(dir_name)

	# Now check if this digital input has generic_poisson_hmm_results
	if hf5.__contains__('/spike_trains/dig_in_{:d}/generic_poisson_hmm_results'.format(taste_num)):
		# If it does, then make a folder for multinomial hmm plots
		os.mkdir('./HMM_EMG_plots/dig_in_{:d}/generic_poisson'.format(taste_num))

		# List the nodes under multinomial_hmm_results
		hmm_nodes = hf5.list_nodes('/spike_trains/dig_in_{:d}/generic_poisson_hmm_results'.format(taste_num))

		# Run through the hmm_nodes, make folders for each of them, and plot the posterior probabilities
		for node in hmm_nodes:
			# Check if the current node is the laser node
			if str.split(node._v_pathname, '/')[-1] == 'laser':
				# Get the nodes with the laser results
				laser_nodes = hf5.list_nodes('/spike_trains/dig_in_{:d}/generic_poisson_hmm_results/laser'.format(taste_num))

				# Run through the laser_nodes, make folders for each of them, and plot the posterior probabilities
				os.mkdir('./HMM_EMG_plots/dig_in_{:d}/generic_poisson/laser'.format(taste_num))
				for laser_node in laser_nodes:
					# Make a folder for this node
					os.mkdir('./HMM_EMG_plots/dig_in_{:d}/generic_poisson/laser/{:s}'.format(taste_num, str.split(laser_node._v_pathname, '/')[-1]))
					# Change to this directory
					os.chdir('./HMM_EMG_plots/dig_in_{:d}/generic_poisson/laser/{:s}'.format(taste_num, str.split(laser_node._v_pathname, '/')[-1]))

					# Get the HMM time
					time = laser_node.time[:]
					# And the posterior probability to plot
					posterior_proba = laser_node.posterior_proba[:]

					# Make an array of posterior probabilities arranged by laser conditions
					final_proba = np.zeros((lasers.shape[0], int(posterior_proba.shape[0]/lasers.shape[0]), posterior_proba.shape[1], posterior_proba.shape[2]))

					# Get the limits of plotting
					start = 100*(int(time[0]/100))
					end = 100*(int(time[-1]/100) + 1)

					# Slice out the required length of the array of spikes
					spikes_current = spikes[:, :, pre_stim - np.abs(start) : pre_stim + np.abs(end)]

					# Make directories for the plots
					os.mkdir('./gapes')
					os.mkdir('./ltps')
					# Make folders by laser conditions too
					for condition in lasers:
						os.mkdir('./gapes/Dur%i,Lag%i' % (int(condition[0]), int(condition[1])))
						os.mkdir('./ltps/Dur%i,Lag%i' % (int(condition[0]), int(condition[1])))
					# Run through the trials
					for i in range(posterior_proba.shape[0]):
						# Locate this trial number in the lasers X trial X.. array called trials
						laser_condition = int(np.where(trials == posterior_proba.shape[0]*taste_num + i)[0][0])
						this_taste_trials = np.where((trials[laser_condition] >= posterior_proba.shape[0]*taste_num) * (trials[laser_condition] <= posterior_proba.shape[0]*(taste_num + 1)))
						this_trial = int(np.where(trials[laser_condition, this_taste_trials][0] == posterior_proba.shape[0]*taste_num + i)[0])

						# Fill up the final_proba array
						final_proba[laser_condition, this_trial, :, :] = posterior_proba[i, :, :]

						# Plot the gapes, gapes_Li and posterior_proba
						fig = plt.figure()
						for j in range(posterior_proba.shape[2]):
							plt.plot(time, posterior_proba[i, :, j])
						if sig_trials[laser_condition, taste_num, this_trial] > 0.0:
							plt.plot(np.arange(end), len(chosen_units)*gapes[laser_condition, taste_num, this_trial, :end])
							plt.plot(np.arange(end), len(chosen_units)*gapes_Li[laser_condition, taste_num, this_trial, pre_stim : pre_stim + end], linewidth = 2.0, color = 'black')
						# First make a dictionary of colors for the rasters
						raster_colors = {'regular_spiking': 'red', 'fast_spiking': 'blue', 'multi_unit': 'black'}
						for unit in range(len(chosen_units)):
							# Determine the type of unit we are looking at - the color of the raster will depend on that
							if hf5.root.unit_descriptor[chosen_units[unit]]['regular_spiking'] == 1:
								unit_type = 'regular_spiking'
							elif hf5.root.unit_descriptor[chosen_units[unit]]['fast_spiking'] == 1:
								unit_type = 'fast_spiking'
							else:
								unit_type = 'multi_unit'
							for j in range(spikes_current.shape[2]):
								if spikes_current[i, unit, j] > 0:
									plt.vlines(j - np.abs(start), unit, unit + 0.5, color = raster_colors[unit_type], linewidth = 0.5)

						plt.xlabel('Time post stimulus (ms)')
						plt.ylabel('Probability of HMM states' + '\n' + '% Power < 4.6Hz, Gapes from Li et al')
						plt.title('Trial %i, Dur: %ims, Lag:%ims' % (i+1, dig_in.laser_durations[i], dig_in.laser_onset_lag[i]))
						fig.savefig('./gapes/Dur%i,Lag%i/Trial_%i.png' % (int(lasers[laser_condition, 0]), int(lasers[laser_condition, 1]), i+1))
						plt.close("all")

						# Plot the ltps, and posterior_proba
						fig = plt.figure()
						for j in range(posterior_proba.shape[2]):
							plt.plot(time, posterior_proba[i, :, j])
						if sig_trials[laser_condition, taste_num, this_trial] > 0.0:
							plt.plot(np.arange(end), len(chosen_units)*ltps[laser_condition, taste_num, this_trial, :end])
						# First make a dictionary of colors for the rasters
						raster_colors = {'regular_spiking': 'red', 'fast_spiking': 'blue', 'multi_unit': 'black'}
						for unit in range(len(chosen_units)):
							# Determine the type of unit we are looking at - the color of the raster will depend on that
							if hf5.root.unit_descriptor[chosen_units[unit]]['regular_spiking'] == 1:
								unit_type = 'regular_spiking'
							elif hf5.root.unit_descriptor[chosen_units[unit]]['fast_spiking'] == 1:
								unit_type = 'fast_spiking'
							else:
								unit_type = 'multi_unit'
							for j in range(spikes_current.shape[2]):
								if spikes_current[i, unit, j] > 0:
									plt.vlines(j - np.abs(start), unit, unit + 0.5, color = raster_colors[unit_type], linewidth = 0.5)

						plt.xlabel('Time post stimulus (ms)')
						plt.ylabel('Probability of HMM states' + '\n' + '% Power in 5.95-8.6Hz')
						plt.title('Trial %i, Dur: %ims, Lag:%ims' % (i+1, dig_in.laser_durations[i], dig_in.laser_onset_lag[i]))
						fig.savefig('./ltps/Dur%i,Lag%i/Trial_%i.png' % (int(lasers[laser_condition, 0]), int(lasers[laser_condition, 1]), i+1))
						plt.close("all")

					# Plot the trial-averaged HMM posterior probabilities
					mean_proba = np.mean(final_proba, axis = 1)
					for i in range(mean_proba.shape[0]):
						fig = plt.figure()
						for j in range(mean_proba.shape[2]):
							plt.plot(time, mean_proba[i, :, j])
						plt.xlabel('Time post stimulus (ms)')
						plt.ylabel('Trial averaged probabilities' + '\n' + 'of HMM states')
						plt.title('Dur: %ims, Lag:%ims' % (int(lasers[i, 0]), int(lasers[i, 1])))
						fig.savefig('./gapes/Dur%i,Lag%i/Average_HMM.png' % (int(lasers[i, 0]), int(lasers[i, 1])))
						fig.savefig('./ltps/Dur%i,Lag%i/Average_HMM.png' % (int(lasers[i, 0]), int(lasers[i, 1])))
						plt.close('all')

					# Plot the trial-averaged gape and ltp frequencies
					mean_gapes = np.mean(gapes[:, :, :, :], axis = 2)
					mean_ltps = np.mean(ltps[:, :, :, :], axis = 2)
					for i in range(mean_gapes.shape[0]):
						fig = plt.figure()
						plt.plot(np.arange(end), mean_gapes[i, taste_num, :end])
						plt.xlabel('Time post stimulus (ms)')
						plt.ylabel('Trial-averaged fraction of' + '\n' + 'power in the 4-6Hz range')
						plt.title('Dur: %ims, Lag:%ims' % (int(lasers[i, 0]), int(lasers[i, 1])))
						fig.savefig('./gapes/Dur%i,Lag%i/Average_gapes.png' % (int(lasers[i, 0]), int(lasers[i, 1])))
						plt.close('all')

						fig = plt.figure()
						plt.plot(np.arange(end), mean_ltps[i, taste_num, :end])
						plt.xlabel('Time post stimulus (ms)')
						plt.ylabel('Trial-averaged fraction of' + '\n' + 'power in the 6-10Hz range')
						plt.title('Dur: %ims, Lag:%ims' % (int(lasers[i, 0]), int(lasers[i, 1])))
						fig.savefig('./ltps/Dur%i,Lag%i/Average_ltps.png' % (int(lasers[i, 0]), int(lasers[i, 1])))
						plt.close('all')

					# Save the final_proba array to the hdf5 file
					hf5.create_array(laser_node, 'final_proba', final_proba)
					hf5.flush()

					# Go back to the data directory
					os.chdir(dir_name)

			else:
				# Make a folder for this node
				os.mkdir('./HMM_EMG_plots/dig_in_{:d}/generic_poisson/{:s}'.format(taste_num, str.split(node._v_pathname, '/')[-1]))
				# Change to this directory
				os.chdir('./HMM_EMG_plots/dig_in_{:d}/generic_poisson/{:s}'.format(taste_num, str.split(node._v_pathname, '/')[-1]))
				# Get the HMM time
				time = node.time[:]
				# And the posterior probability to plot
				posterior_proba = node.posterior_proba[:]

				# Make an array of posterior probabilities arranged by laser conditions
				final_proba = np.zeros((lasers.shape[0], int(posterior_proba.shape[0]/lasers.shape[0]), posterior_proba.shape[1], posterior_proba.shape[2]))

				# Get the limits of plotting
				start = 100*(int(time[0]/100))
				end = 100*(int(time[-1]/100) + 1)

				# Make directories for the plots
				os.mkdir('./gapes')
				os.mkdir('./ltps')
				# Run through the trials
				for i in range(posterior_proba.shape[0]):
					# Locate this trial number in the lasers X trial X.. array called trials
					laser_condition = int(np.where(trials == posterior_proba.shape[0]*taste_num + i)[0][0])
					this_taste_trials = np.where((trials[laser_condition] >= posterior_proba.shape[0]*taste_num) * (trials[laser_condition] <= posterior_proba.shape[0]*(taste_num + 1)))
					this_trial = int(np.where(trials[laser_condition, this_taste_trials][0] == posterior_proba.shape[0]*taste_num + i)[0])

					# Fill up the final_proba array
					final_proba[laser_condition, this_trial, :, :] = posterior_proba[i, :, :]

					# Plot the gapes, gapes_Li and posterior_proba
					fig = plt.figure()
					for j in range(posterior_proba.shape[2]):
						plt.plot(time, posterior_proba[i, :, j])
					if sig_trials[laser_condition, taste_num, this_trial] > 0.0:
						plt.plot(np.arange(end), gapes[laser_condition, taste_num, this_trial, :end])
						plt.plot(np.arange(end), gapes_Li[laser_condition, taste_num, this_trial, pre_stim : pre_stim + end], linewidth = 2.0, color = 'black')
					plt.xlabel('Time post stimulus (ms)')
					plt.ylabel('Probability of HMM states' + '\n' + '% Power < 4.6Hz, Gapes from Li et al')
					plt.title('Trial %i' % (i+1))
					fig.savefig('./gapes/Trial_%i.png' % (i+1))
					plt.close("all")

					# Plot the ltps, and posterior_proba
					fig = plt.figure()
					for j in range(posterior_proba.shape[2]):
						plt.plot(time, posterior_proba[i, :, j])
					if sig_trials[laser_condition, taste_num, this_trial] > 0.0:
						plt.plot(np.arange(end), ltps[laser_condition, taste_num, this_trial, :end])
					plt.xlabel('Time post stimulus (ms)')
					plt.ylabel('Probability of HMM states' + '\n' + '% Power in 5.95-8.6Hz')
					plt.title('Trial %i' % (i+1))
					fig.savefig('./ltps/Trial_%i.png' % (i+1))
					plt.close("all")

				# Plot the trial-averaged HMM posterior probabilities
				mean_proba = np.mean(final_proba, axis = (0, 1))
				fig = plt.figure()
				for j in range(mean_proba.shape[1]):
					plt.plot(time, mean_proba[:, j])
				plt.xlabel('Time post stimulus (ms)')
				plt.ylabel('Trial averaged probabilities' + '\n' + 'of HMM states')
				plt.title('Trial-averaged HMM')
				fig.savefig('./gapes/Average_HMM.png')
				fig.savefig('./ltps/Average_HMM.png')
				plt.close('all')

				# Plot the trial-averaged gape and ltp frequencies
				mean_gapes = np.mean(gapes[:, :, :, :], axis = (0, 2))
				mean_ltps = np.mean(ltps[:, :, :, :], axis = (0, 2))
				fig = plt.figure()
				plt.plot(np.arange(end), mean_gapes[taste_num, :end])
				plt.xlabel('Time post stimulus (ms)')
				plt.ylabel('Trial-averaged fraction of' + '\n' + 'power in the 4-6Hz range')
				plt.title('Trial-averaged power in gaping range')
				fig.savefig('./gapes/Average_gapes.png')
				plt.close('all')

				fig = plt.figure()
				plt.plot(np.arange(end), mean_ltps[taste_num, :end])
				plt.xlabel('Time post stimulus (ms)')
				plt.ylabel('Trial-averaged fraction of' + '\n' + 'power in the 6-10Hz range')
				plt.title('Trial-averaged power in LTP range')
				fig.savefig('./ltps/Average_ltps.png')
				plt.close('all')

				# Save the final_proba array to the hdf5 file
				hf5.create_array(node, 'final_proba', final_proba)
				hf5.flush()

				# Go back to the data directory
				os.chdir(dir_name)

	# Now check if this digital input has feedforward_poisson_hmm_results
	if hf5.__contains__('/spike_trains/dig_in_{:d}/feedforward_poisson_hmm_results'.format(taste_num)):
		# If it does, then make a folder for multinomial hmm plots
		os.mkdir('./HMM_EMG_plots/dig_in_{:d}/feedforward_poisson'.format(taste_num))

		# List the nodes under multinomial_hmm_results
		hmm_nodes = hf5.list_nodes('/spike_trains/dig_in_{:d}/feedforward_poisson_hmm_results'.format(taste_num))

		# Run through the hmm_nodes, make folders for each of them, and plot the posterior probabilities
		for node in hmm_nodes:
			# Check if the current node is the laser node
			if str.split(node._v_pathname, '/')[-1] == 'laser':
				# Get the nodes with the laser results
				laser_nodes = hf5.list_nodes('/spike_trains/dig_in_{:d}/feedforward_poisson_hmm_results/laser'.format(taste_num))

				# Run through the laser_nodes, make folders for each of them, and plot the posterior probabilities
				os.mkdir('./HMM_EMG_plots/dig_in_{:d}/feedforward_poisson/laser'.format(taste_num))
				for laser_node in laser_nodes:
					# Make a folder for this node
					os.mkdir('./HMM_EMG_plots/dig_in_{:d}/feedforward_poisson/laser/{:s}'.format(taste_num, str.split(laser_node._v_pathname, '/')[-1]))
					# Change to this directory
					os.chdir('./HMM_EMG_plots/dig_in_{:d}/feedforward_poisson/laser/{:s}'.format(taste_num, str.split(laser_node._v_pathname, '/')[-1]))

					# Get the HMM time
					time = laser_node.time[:]
					# And the posterior probability to plot
					posterior_proba = laser_node.posterior_proba[:]

					# Make an array of posterior probabilities arranged by laser conditions
					final_proba = np.zeros((lasers.shape[0], int(posterior_proba.shape[0]/lasers.shape[0]), posterior_proba.shape[1], posterior_proba.shape[2]))

					# Get the limits of plotting
					start = 100*(int(time[0]/100))
					end = 100*(int(time[-1]/100) + 1)

					# Make directories for the plots
					os.mkdir('./gapes')
					os.mkdir('./ltps')
					# Make folders by laser conditions too
					for condition in lasers:
						os.mkdir('./gapes/Dur%i,Lag%i' % (int(condition[0]), int(condition[1])))
						os.mkdir('./ltps/Dur%i,Lag%i' % (int(condition[0]), int(condition[1])))
					# Run through the trials
					for i in range(posterior_proba.shape[0]):
						# Locate this trial number in the lasers X trial X.. array called trials
						laser_condition = int(np.where(trials == posterior_proba.shape[0]*taste_num + i)[0][0])
						this_taste_trials = np.where((trials[laser_condition] >= posterior_proba.shape[0]*taste_num) * (trials[laser_condition] <= posterior_proba.shape[0]*(taste_num + 1)))
						this_trial = int(np.where(trials[laser_condition, this_taste_trials][0] == posterior_proba.shape[0]*taste_num + i)[0])

						# Fill up the final_proba array
						final_proba[laser_condition, this_trial, :, :] = posterior_proba[i, :, :]

						# Plot the gapes, gapes_Li and posterior_proba
						fig = plt.figure()
						for j in range(posterior_proba.shape[2]):
							plt.plot(time, posterior_proba[i, :, j])
						if sig_trials[laser_condition, taste_num, this_trial] > 0.0:
							plt.plot(np.arange(end), gapes[laser_condition, taste_num, this_trial, :end])
							plt.plot(np.arange(end), gapes_Li[laser_condition, taste_num, this_trial, pre_stim : pre_stim + end], linewidth = 2.0, color = 'black')

						plt.xlabel('Time post stimulus (ms)')
						plt.ylabel('Probability of HMM states' + '\n' + '% Power < 4.6Hz, Gapes from Li et al')
						plt.title('Trial %i, Dur: %ims, Lag:%ims' % (i+1, dig_in.laser_durations[i], dig_in.laser_onset_lag[i]))
						fig.savefig('./gapes/Dur%i,Lag%i/Trial_%i.png' % (int(lasers[laser_condition, 0]), int(lasers[laser_condition, 1]), i+1))
						plt.close("all")

						# Plot the ltps, and posterior_proba
						fig = plt.figure()
						for j in range(posterior_proba.shape[2]):
							plt.plot(time, posterior_proba[i, :, j])
						if sig_trials[laser_condition, taste_num, this_trial] > 0.0:
							plt.plot(np.arange(end), ltps[laser_condition, taste_num, this_trial, :end])

						plt.xlabel('Time post stimulus (ms)')
						plt.ylabel('Probability of HMM states' + '\n' + '% Power in 5.95-8.6Hz')
						plt.title('Trial %i, Dur: %ims, Lag:%ims' % (i+1, dig_in.laser_durations[i], dig_in.laser_onset_lag[i]))
						fig.savefig('./ltps/Dur%i,Lag%i/Trial_%i.png' % (int(lasers[laser_condition, 0]), int(lasers[laser_condition, 1]), i+1))
						plt.close("all")

					# Plot the trial-averaged HMM posterior probabilities
					mean_proba = np.mean(final_proba, axis = 1)
					for i in range(mean_proba.shape[0]):
						fig = plt.figure()
						for j in range(mean_proba.shape[2]):
							plt.plot(time, mean_proba[i, :, j])
						plt.xlabel('Time post stimulus (ms)')
						plt.ylabel('Trial averaged probabilities' + '\n' + 'of HMM states')
						plt.title('Dur: %ims, Lag:%ims' % (int(lasers[i, 0]), int(lasers[i, 1])))
						fig.savefig('./gapes/Dur%i,Lag%i/Average_HMM.png' % (int(lasers[i, 0]), int(lasers[i, 1])))
						fig.savefig('./ltps/Dur%i,Lag%i/Average_HMM.png' % (int(lasers[i, 0]), int(lasers[i, 1])))
						plt.close('all')

					# Plot the trial-averaged gape and ltp frequencies
					mean_gapes = np.mean(gapes[:, :, :, :], axis = 2)
					mean_ltps = np.mean(ltps[:, :, :, :], axis = 2)
					for i in range(mean_gapes.shape[0]):
						fig = plt.figure()
						plt.plot(np.arange(end), mean_gapes[i, taste_num, :end])
						plt.xlabel('Time post stimulus (ms)')
						plt.ylabel('Trial-averaged fraction of' + '\n' + 'power in the 4-6Hz range')
						plt.title('Dur: %ims, Lag:%ims' % (int(lasers[i, 0]), int(lasers[i, 1])))
						fig.savefig('./gapes/Dur%i,Lag%i/Average_gapes.png' % (int(lasers[i, 0]), int(lasers[i, 1])))
						plt.close('all')

						fig = plt.figure()
						plt.plot(np.arange(end), mean_ltps[i, taste_num, :end])
						plt.xlabel('Time post stimulus (ms)')
						plt.ylabel('Trial-averaged fraction of' + '\n' + 'power in the 6-10Hz range')
						plt.title('Dur: %ims, Lag:%ims' % (int(lasers[i, 0]), int(lasers[i, 1])))
						fig.savefig('./ltps/Dur%i,Lag%i/Average_ltps.png' % (int(lasers[i, 0]), int(lasers[i, 1])))
						plt.close('all')

					# Save the final_proba array to the hdf5 file
					hf5.create_array(laser_node, 'final_proba', final_proba)
					hf5.flush()

					# Go back to the data directory
					os.chdir(dir_name)

			else:
				# Make a folder for this node
				os.mkdir('./HMM_EMG_plots/dig_in_{:d}/feedforward_poisson/{:s}'.format(taste_num, str.split(node._v_pathname, '/')[-1]))
				# Change to this directory
				os.chdir('./HMM_EMG_plots/dig_in_{:d}/feedforward_poisson/{:s}'.format(taste_num, str.split(node._v_pathname, '/')[-1]))
				# Get the HMM time
				time = node.time[:]
				# And the posterior probability to plot
				posterior_proba = node.posterior_proba[:]

				# Make an array of posterior probabilities arranged by laser conditions
				final_proba = np.zeros((lasers.shape[0], int(posterior_proba.shape[0]/lasers.shape[0]), posterior_proba.shape[1], posterior_proba.shape[2]))

				# Get the limits of plotting
				start = 100*(int(time[0]/100))
				end = 100*(int(time[-1]/100) + 1)

				# Make directories for the plots
				os.mkdir('./gapes')
				os.mkdir('./ltps')
				# Run through the trials
				for i in range(posterior_proba.shape[0]):
					# Locate this trial number in the lasers X trial X.. array called trials
					laser_condition = int(np.where(trials == posterior_proba.shape[0]*taste_num + i)[0][0])
					this_taste_trials = np.where((trials[laser_condition] >= posterior_proba.shape[0]*taste_num) * (trials[laser_condition] <= posterior_proba.shape[0]*(taste_num + 1)))
					this_trial = int(np.where(trials[laser_condition, this_taste_trials][0] == posterior_proba.shape[0]*taste_num + i)[0])

					# Fill up the final_proba array
					final_proba[laser_condition, this_trial, :, :] = posterior_proba[i, :, :]

					# Plot the gapes, gapes_Li and posterior_proba
					fig = plt.figure()
					for j in range(posterior_proba.shape[2]):
						plt.plot(time, posterior_proba[i, :, j])
					if sig_trials[laser_condition, taste_num, this_trial] > 0.0:
						plt.plot(np.arange(end), gapes[laser_condition, taste_num, this_trial, :end])
						plt.plot(np.arange(end), gapes_Li[laser_condition, taste_num, this_trial, pre_stim : pre_stim + end], linewidth = 2.0, color = 'black')
					plt.xlabel('Time post stimulus (ms)')
					plt.ylabel('Probability of HMM states' + '\n' + '% Power < 4.6Hz, Gapes from Li et al')
					plt.title('Trial %i' % (i+1))
					fig.savefig('./gapes/Trial_%i.png' % (i+1))
					plt.close("all")

					# Plot the ltps, and posterior_proba
					fig = plt.figure()
					for j in range(posterior_proba.shape[2]):
						plt.plot(time, posterior_proba[i, :, j])
					if sig_trials[laser_condition, taste_num, this_trial] > 0.0:
						plt.plot(np.arange(end), ltps[laser_condition, taste_num, this_trial, :end])
					plt.xlabel('Time post stimulus (ms)')
					plt.ylabel('Probability of HMM states' + '\n' + '% Power in 5.95-8.6Hz')
					plt.title('Trial %i' % (i+1))
					fig.savefig('./ltps/Trial_%i.png' % (i+1))
					plt.close("all")

				# Plot the trial-averaged HMM posterior probabilities
				mean_proba = np.mean(final_proba, axis = (0, 1))
				fig = plt.figure()
				for j in range(mean_proba.shape[1]):
					plt.plot(time, mean_proba[:, j])
				plt.xlabel('Time post stimulus (ms)')
				plt.ylabel('Trial averaged probabilities' + '\n' + 'of HMM states')
				plt.title('Trial-averaged HMM')
				fig.savefig('./gapes/Average_HMM.png')
				fig.savefig('./ltps/Average_HMM.png')
				plt.close('all')

				# Plot the trial-averaged gape and ltp frequencies
				mean_gapes = np.mean(gapes[:, :, :, :], axis = (0, 2))
				mean_ltps = np.mean(ltps[:, :, :, :], axis = (0, 2))
				fig = plt.figure()
				plt.plot(np.arange(end), mean_gapes[taste_num, :end])
				plt.xlabel('Time post stimulus (ms)')
				plt.ylabel('Trial-averaged fraction of' + '\n' + 'power in the 4-6Hz range')
				plt.title('Trial-averaged power in gaping range')
				fig.savefig('./gapes/Average_gapes.png')
				plt.close('all')

				fig = plt.figure()
				plt.plot(np.arange(end), mean_ltps[taste_num, :end])
				plt.xlabel('Time post stimulus (ms)')
				plt.ylabel('Trial-averaged fraction of' + '\n' + 'power in the 6-10Hz range')
				plt.title('Trial-averaged power in LTP range')
				fig.savefig('./ltps/Average_ltps.png')
				plt.close('all')

				# Save the final_proba array to the hdf5 file
				hf5.create_array(node, 'final_proba', final_proba)
				hf5.flush()

				# Go back to the data directory
				os.chdir(dir_name)

# Finally plot out the trial-averaged gaping and ltp power by laser conditions
# Ask the user for the post-stimulus time to plot
post_stim = easygui.multenterbox(msg = 'Fill in the post-stimulus period you want to plot for the average gapes and ltps', fields = ['Post-stimulus time (ms)'])
post_stim = int(post_stim[0])

# Now run through the laser conditions
for i in range(lasers.shape[0]):
	# And plot all the tastes on the same graph
	fig_gape = plt.figure()
	ax_gape = fig_gape.add_subplot(1, 1, 1)
	fig_ltp = plt.figure()
	ax_ltp = fig_ltp.add_subplot(1, 1, 1)
	for j in range(len(trains_dig_in)):
		ax_gape.plot(np.arange(post_stim), np.mean(gapes[i, j, :, :post_stim], axis = 0), label = 'Taste %i' % (j + 1))
		ax_ltp.plot(np.arange(post_stim), np.mean(ltps[i, j, :, :post_stim], axis = 0), label = 'Taste %i' % (j + 1))
		ax_gape.legend()
		ax_ltp.legend()
		ax_gape.set_xlabel('Time post stimlus (ms)')
		ax_ltp.set_xlabel('Time post stimlus (ms)')
		ax_gape.set_ylabel('Trial averaged power in 4.15-5.95 Hz')
		ax_ltp.set_ylabel('Trial averaged power in 5.95-8.65 Hz')
		ax_gape.set_title('Dur: %ims, Lag: %ims' % (int(lasers[i, 0]), int(lasers[i, 1])))
		ax_ltp.set_title('Dur: %ims, Lag: %ims' % (int(lasers[i, 0]), int(lasers[i, 1])))
		fig_gape.savefig('./HMM_EMG_plots/average_gape_Dur%i_Lag%i.png' % (int(lasers[i, 0]), int(lasers[i, 1])), bbox_inches = 'tight')
		fig_ltp.savefig('./HMM_EMG_plots/average_ltp_Dur%i_Lag%i.png' % (int(lasers[i, 0]), int(lasers[i, 1])), bbox_inches = 'tight')
		plt.close('all')

hf5.close()
