# Import stuff!
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tables
import easygui
import sys
import os
#import seaborn as sns
#sns.set(style="white", context="talk", font_scale=1.8)
#sns.set_color_codes(palette = 'colorblind')

# Ask the user for the hdf5 files that need to be plotted together
dirs = []
while True:
	dir_name = easygui.diropenbox(msg = 'Choose a directory with a hdf5 file, hit cancel to stop choosing')
	try:
		if len(dir_name) > 0:
			dirs.append(dir_name)
	except:
		break

# Ask the user to enter the number of laser conditions and tastes in the experiment
num_tastes = easygui.multenterbox(msg = 'Enter the number of tastes in the datasets', fields = ['Number of tastes (integer)'])
num_lasers = easygui.multenterbox(msg = 'Enter the number of laser conditions in the datasets', fields = ['Number of laser conditions (integer)'])
num_tastes = int(num_tastes[0])
num_lasers = int(num_lasers[0])

# Now run through the directories, and pull out the data
r_pearson = []
p_pearson = []
gapes = []
ltps = []
switchpoints = []
converged_trials = []
unique_lasers = []
num_trials = []
gapes_Li = []
p_pal_before_laser = []
posterior_prob_switchpoints = []
potential_switchpoint_array = []
pre_stim = []

for dir_name in dirs:
	# Change to the directory
	os.chdir(dir_name)
	# Locate the hdf5 file
	file_list = os.listdir('./')
	hdf5_name = ''
	for files in file_list:
		if files[-2:] == 'h5':
			hdf5_name = files

	# Open the hdf5 file
	hf5 = tables.open_file(hdf5_name, 'r')

	# Pull the data from the /EM_switch node
	unique_lasers.append(hf5.root.EM_switch.unique_lasers[:])
	r_pearson.append(hf5.root.EM_switch.r_pearson[:])
	p_pearson.append(hf5.root.EM_switch.p_pearson[:])
	# Read the gapes and LTPs
	gapes.append(hf5.root.ancillary_analysis.gapes[:, :, :, :])
	ltps.append(hf5.root.ancillary_analysis.ltps[:, :, :, :])
	gapes_Li.append(hf5.root.ancillary_analysis.gapes_Li[:, :, :, :])
	num_trials.append(np.array(hf5.root.EM_switch.inactivated_spikes[:]).shape[1] / num_tastes)
	# Read the pre-stimulus time
	pre_stim.append(int(hf5.root.ancillary_analysis.pre_stim.read()))
	# Make lists to pull the switchpoints, converged_trials, potential switchpoints and their posterior probabilities for this dataset
	this_switchpoints = []
	this_converged_trials = []
	this_potential_switchpoints = []
	this_posterior_prob = []
	# Now run through the laser conditions to get the switchpoints, converged trials, potential switchpoints and their posterior probabilities
	for laser in range(num_lasers):
		exec("this_switchpoints.append(hf5.root.EM_switch.switchpoints.laser_condition_{:d}[:])".format(laser))
		exec("this_converged_trials.append(hf5.root.EM_switch.converged_trial_nums.laser_condition_{:d}[:])".format(laser))
		exec("this_potential_switchpoints.append(hf5.root.EM_switch.potential_switchpoints.laser_condition_{:d}[:])".format(laser))
		exec("this_posterior_prob.append(hf5.root.EM_switch.posterior_prob_switchpoints.laser_condition_{:d}[:])".format(laser))
	# Now append these lists to the big switchpoints and converged_trials lists
	switchpoints.append(this_switchpoints)
	converged_trials.append(this_converged_trials)
	potential_switchpoint_array.append(this_potential_switchpoints)
	posterior_prob_switchpoints.append(this_posterior_prob)
	# Also get the posterior probability of the palatability switchpoint happening before the laser inactivation started
	p_pal_before_laser.append(hf5.root.EM_switch.palatability_before_laser_probability[:])

	# Close the hdf5 file
	hf5.close()

# Assuming that the pre-stimulus time is the same in all files, set pre_stim to the pre-stimulus time from the first file
pre_stim = pre_stim[0]

# Check if the number of laser activation/inactivation windows is same across files, raise an error and quit if it isn't
if all(unique_lasers[i].shape == unique_lasers[0].shape for i in range(len(unique_lasers))):
	pass
else:
	print("Number of inactivation/activation windows doesn't seem to be the same across days. Please check and try again")
	sys.exit()

# Now first set the ordering of laser trials straight across data files
laser_order = []
for i in range(len(unique_lasers)):
	# The first file defines the order
	if i == 0:
		laser_order.append(np.arange(unique_lasers[i].shape[0]))
	# And everyone else follows
	else:
		this_order = []
		for j in range(unique_lasers[i].shape[0]):
			for k in range(unique_lasers[i].shape[0]):
				if np.array_equal(unique_lasers[0][j, :], unique_lasers[i][k, :]):
					this_order.append(k)
		laser_order.append(np.array(this_order))

# Now join up all the data into big numpy arrays, maintaining the laser order defined in laser_order
# If there's only one data file, set the final arrays to the only array read in
if len(laser_order) == 1:
	r_pearson = r_pearson[0]
	p_pearson = p_pearson[0]
else:
	r_pearson = np.concatenate(tuple(r_pearson[i][laser_order[i], :, :] for i in range(len(r_pearson))), axis = 2)
	p_pearson = np.concatenate(tuple(p_pearson[i][laser_order[i], :, :] for i in range(len(p_pearson))), axis = 2)

# Ask the user for the directory to save plots etc in
dir_name = easygui.diropenbox(msg = 'Choose the output directory for identity-palatability switch analysis')
os.chdir(dir_name)

# Ask the user for the names of the tastes in the dataset
tastes = easygui.multenterbox(msg = 'Enter the names of the tastes used in the experiments', fields = ['Taste{:d}'.format(i+1) for i in range(num_tastes)])

# Plot a histogram of palatability correlation in the 2 epochs, for each laser condition
for laser in range(num_lasers):
	fig = plt.figure()
	plt.hist(r_pearson[laser, 0, :]**2, bins = 20, alpha = 0.4, label = "Epoch 1" + "\n" + "mean $r^2$ = {:01.3f} $\pm$ {:01.3f}".format(np.mean(r_pearson[laser, 0, :]**2), np.std(r_pearson[laser, 0, :]**2)/np.sqrt(r_pearson.shape[-1])))
	plt.hist(r_pearson[laser, 1, :]**2, bins = 20, alpha = 0.4, label = "Epoch 2" + "\n" + "mean $r^2$ = {:01.3f} $\pm$ {:01.3f}".format(np.mean(r_pearson[laser, 1, :]**2), np.std(r_pearson[laser, 0, :]**2)/np.sqrt(r_pearson.shape[-1])))
	plt.title("Dur: {:d}ms, Lag: {:d}ms".format(int(unique_lasers[0][laser, 0]), int(unique_lasers[0][laser, 1])))
	plt.xlabel("Palatability $r^2$")
	plt.ylabel("Number of neurons (Total = {:d})".format(r_pearson.shape[-1]))
	plt.legend()
	plt.tight_layout()
	fig.savefig('Pearson_correlation_palatability_Dur{:d}_Lag{:d}ms.png'.format(int(unique_lasers[0][laser, 0]), int(unique_lasers[0][laser, 1])), bbox_inches = 'tight')
	plt.close('all')

#----------------------------Splitting EMG data by switchpoints--------------------------------------------------
# Now set things up to plot the EMG data by the switchpoints found in the neural data

# First ask the user how many splits they want to do for their plots
num_splits = easygui.multenterbox(msg = 'Enter the number of switchpoint splits you want to plot', fields = ['Number of splits (integer)'])
num_splits = int(num_splits[0])

# Now get the switchpoint time for every split that the user wants to make
plot_switch = easygui.multenterbox(msg = 'Enter the switchpoint times you want to split the data by', fields = ['Time {:d} (in ms)'.format(i+1) for i in range(num_splits)])
# Convert the switchpoints for splitting the plots to the same scale as the neural analysis (10ms bins)
for i in range(len(plot_switch)):
	plot_switch[i] = int(float(plot_switch[i])/10)

#Ask the user for the time post stimulus to be plotted
post_stim = easygui.multenterbox(msg = 'What is the post stimulus time you want to plot?', fields = ['Post stimulus time (ms)'])
post_stim = int(post_stim[0])

# Run through the switchpoint splits
for split in plot_switch:
	# And run through the laser conditions
	for laser in range(num_lasers):
		# Make lists to pull out the gaping EMG data from the trials that have switchpoints before and after this switchpoint split
		gapes_before = [[] for i in range(num_tastes)]
		gapes_after = [[] for i in range(num_tastes)]

		# Also make list of lists to store the actual switchpoint times as well
		switchpoint1 = [[] for i in range(num_tastes)]
		switchpoint2 = [[] for i in range(num_tastes)]

		# Make list of lists to store the posterior probability that the palatability switchpoint came before the laser came on
		p_pal_before_laser_before = [[] for i in range(num_tastes)]
		p_pal_before_laser_after = [[] for i in range(num_tastes)]

		# Now run through the datasets
		for dataset in range(len(converged_trials)):
			# Run through the converged trials in this dataset for this laser condition
			for trial in range(converged_trials[dataset][laser].shape[0]):
				# Check if the palatability (2nd) switchpoint on this trial is before the switchpoint split
				if switchpoints[dataset][laser][trial, 1] < split:
					# Append the data for this trial to the proper taste in gapes_before
					gapes_before[int(converged_trials[dataset][laser][trial]/num_trials[dataset])].append(gapes[dataset][laser, int(converged_trials[dataset][laser][trial]/num_trials[dataset]), int(converged_trials[dataset][laser][trial] % num_trials[dataset]), :])
					p_pal_before_laser_before[int(converged_trials[dataset][laser][trial]/num_trials[dataset])].append(p_pal_before_laser[dataset][laser][trial])
				# If the switchpoint on this trial is after the switchpoint split, append the data to gapes_after
				else:
					gapes_after[int(converged_trials[dataset][laser][trial]/num_trials[dataset])].append(gapes[dataset][laser, int(converged_trials[dataset][laser][trial]/num_trials[dataset]), int(converged_trials[dataset][laser][trial] % num_trials[dataset]), :])
					p_pal_before_laser_after[int(converged_trials[dataset][laser][trial]/num_trials[dataset])].append(p_pal_before_laser[dataset][laser][trial])

				# Append the actual switchpoint times to the respective lists
				# Correct the switchpoint if it happened after the laser - add the laser duration to the switchpoint in this case
				# First check switchpoint 1 (identity switchpoint)
				if switchpoints[dataset][laser][trial, 0]*10 > unique_lasers[dataset][laser, 1]:
					switchpoint1[int(converged_trials[dataset][laser][trial]/num_trials[dataset])].append(switchpoints[dataset][laser][trial, 0]*10 + unique_lasers[dataset][laser, 0])
				else:
					switchpoint1[int(converged_trials[dataset][laser][trial]/num_trials[dataset])].append(switchpoints[dataset][laser][trial, 0]*10)
				# Now check if switchpoint 2 happened after the laser onset
				if switchpoints[dataset][laser][trial, 1]*10 > unique_lasers[dataset][laser, 1]:
					switchpoint2[int(converged_trials[dataset][laser][trial]/num_trials[dataset])].append(switchpoints[dataset][laser][trial, 1]*10 + unique_lasers[dataset][laser, 0])
				else:
					switchpoint2[int(converged_trials[dataset][laser][trial]/num_trials[dataset])].append(switchpoints[dataset][laser][trial, 1]*10)

		# Convert the gapes_before and gapes_after lists to numpy arrays
		gapes_before = [np.array(gapes_before[i]) for i in range(num_tastes)]
		gapes_after = [np.array(gapes_after[i]) for i in range(num_tastes)]
		# Convert the switchpoint lists to arrays as well
		switchpoint1 = [np.array(switchpoint1[i]) for i in range(num_tastes)]
		switchpoint2 = [np.array(switchpoint2[i]) for i in range(num_tastes)]
		# Conver the probability of palatability switchpoint before laser lists as well
		p_pal_before_laser_before = [np.array(p_pal_before_laser_before[i]) for i in range(num_tastes)]
		p_pal_before_laser_after = [np.array(p_pal_before_laser_after[i]) for i in range(num_tastes)]

		# Save these lists in the plot directory
		np.save("Gapes_before_{:d}_Dur{:d}_Lag{:d}.npy".format(split*10, int(unique_lasers[0][laser, 0]), int(unique_lasers[0][laser, 1])), gapes_before)
		np.save("Gapes_after_{:d}_Dur{:d}_Lag{:d}.npy".format(split*10, int(unique_lasers[0][laser, 0]), int(unique_lasers[0][laser, 1])), gapes_after)
		np.save("Switchpoint1_Dur{:d}_Lag{:d}.npy".format(int(unique_lasers[0][laser, 0]), int(unique_lasers[0][laser, 1])), switchpoint1)
		np.save("Switchpoint2_Dur{:d}_Lag{:d}.npy".format(int(unique_lasers[0][laser, 0]), int(unique_lasers[0][laser, 1])), switchpoint2)
		np.save("Prob_prelaser_switch_before_{:d}_trials_Dur{:d}_Lag{:d}.npy".format(split*10, int(unique_lasers[0][laser, 0]), int(unique_lasers[0][laser, 1])), p_pal_before_laser_before)
		np.save("Prob_prelaser_switch_after_{:d}_trials_Dur{:d}_Lag{:d}.npy".format(split*10, int(unique_lasers[0][laser, 0]), int(unique_lasers[0][laser, 1])), p_pal_before_laser_after)

		# Now make the EMG plots for this laser condition and switchpoint split
		# First plot all the tastes together (without error bars)
		# Gaping on trials with switchpoint before split
		fig = plt.figure()
		for i in range(num_tastes):
			plt.plot(np.mean(gapes_before[i][:, pre_stim:pre_stim + post_stim], axis = 0), label = tastes[i])
		plt.legend()
		plt.xlabel("Time post stimulus (ms)")
		plt.ylabel("Mean fraction of power in 4-6Hz")
		plt.title("Palatability switchpoint < {:d}ms".format(split*10) + "\n" + "Dur: {:d}ms, Lag: {:d}ms, Trials: {}".format(int(unique_lasers[0][laser, 0]), int(unique_lasers[0][laser, 1]), [gapes_before[i].shape[0] for i in range(4)]))
		plt.tight_layout()
		fig.savefig("Before_{:d}_Dur{:d}_Lag{:d}.png".format(split*10, int(unique_lasers[0][laser, 0]), int(unique_lasers[0][laser, 1])), bbox_inches = "tight")
		# Gaping on trials with switchpoint after split
		fig = plt.figure()
		for i in range(num_tastes):
			plt.plot(np.mean(gapes_after[i][:, pre_stim:pre_stim + post_stim], axis = 0), label = tastes[i])
		plt.legend()
		plt.xlabel("Time post stimulus (ms)")
		plt.ylabel("Mean fraction of power in 4-6Hz")
		plt.title("Palatability switchpoint > {:d}ms".format(split*10) + "\n" + "Dur: {:d}ms, Lag: {:d}ms, Trials: {}".format(int(unique_lasers[0][laser, 0]), int(unique_lasers[0][laser, 1]), [gapes_after[i].shape[0] for i in range(4)]))
		plt.tight_layout()
		fig.savefig("After_{:d}_Dur{:d}_Lag{:d}.png".format(split*10, int(unique_lasers[0][laser, 0]), int(unique_lasers[0][laser, 1])), bbox_inches = "tight")
		plt.close("all")

		# Now plot only dil and concentrated quinine, with errorbars for comparisons
		# Gaping on trials with switchpoint before split
		fig = plt.figure()
		plt.plot(np.mean(gapes_before[2][:, pre_stim:pre_stim + post_stim], axis = 0), label = tastes[2])
		std_error = np.std(gapes_before[2][:, pre_stim:pre_stim + post_stim], axis = 0)/np.sqrt(gapes_before[2].shape[0])
		plt.fill_between(np.arange(post_stim), np.mean(gapes_before[2][:, pre_stim:pre_stim + post_stim], axis = 0) - std_error, np.mean(gapes_before[2][:, pre_stim:pre_stim + post_stim], axis = 0) + std_error, alpha = 0.3)
		plt.plot(np.mean(gapes_before[3][:, pre_stim:pre_stim + post_stim], axis = 0), label = tastes[3])
		std_error = np.std(gapes_before[3][:, pre_stim:pre_stim + post_stim], axis = 0)/np.sqrt(gapes_before[3].shape[0])
		plt.fill_between(np.arange(post_stim), np.mean(gapes_before[3][:, pre_stim:pre_stim + post_stim], axis = 0) - std_error, np.mean(gapes_before[3][:, pre_stim:pre_stim + post_stim], axis = 0) + std_error, alpha = 0.3)
		plt.legend()
		plt.xlabel("Time post stimulus (ms)")
		plt.ylabel("Mean fraction of power in 4-6Hz")
		plt.title("Palatability switchpoint < {:d}ms".format(split*10) + "\n" + "Dur: {:d}ms, Lag: {:d}ms, Trials: {}".format(int(unique_lasers[0][laser, 0]), int(unique_lasers[0][laser, 1]), [gapes_before[i+2].shape[0] for i in range(2)]))
		plt.tight_layout()
		fig.savefig("Before_{:d}_Dur{:d}_Lag{:d}_Quinine.png".format(split*10, int(unique_lasers[0][laser, 0]), int(unique_lasers[0][laser, 1])), bbox_inches = "tight")
		# Gaping on trials with switchpoint after split
		fig = plt.figure()
		plt.plot(np.mean(gapes_after[2][:, pre_stim:pre_stim + post_stim], axis = 0), label = tastes[2])
		std_error = np.std(gapes_after[2][:, pre_stim:pre_stim + post_stim], axis = 0)/np.sqrt(gapes_after[2].shape[0])
		plt.fill_between(np.arange(post_stim), np.mean(gapes_after[2][:, pre_stim:pre_stim + post_stim], axis = 0) - std_error, np.mean(gapes_after[2][:, pre_stim:pre_stim + post_stim], axis = 0) + std_error, alpha = 0.3)
		plt.plot(np.mean(gapes_after[3][:, pre_stim:pre_stim + post_stim], axis = 0), label = tastes[3])
		std_error = np.std(gapes_after[3][:, pre_stim:pre_stim + post_stim], axis = 0)/np.sqrt(gapes_after[3].shape[0])
		plt.fill_between(np.arange(post_stim), np.mean(gapes_after[3][:, pre_stim:pre_stim + post_stim], axis = 0) - std_error, np.mean(gapes_after[3][:, pre_stim:pre_stim + post_stim], axis = 0) + std_error, alpha = 0.3)
		plt.legend()
		plt.xlabel("Time post stimulus (ms)")
		plt.ylabel("Mean fraction of power in 4-6Hz")
		plt.title("Palatability switchpoint > {:d}ms".format(split*10) + "\n" + "Dur: {:d}ms, Lag: {:d}ms, Trials: {}".format(int(unique_lasers[0][laser, 0]), int(unique_lasers[0][laser, 1]), [gapes_after[i+2].shape[0] for i in range(2)]))
		plt.tight_layout()
		fig.savefig("After_{:d}_Dur{:d}_Lag{:d}_Quinine.png".format(split*10, int(unique_lasers[0][laser, 0]), int(unique_lasers[0][laser, 1])), bbox_inches = "tight")
		plt.close("all")

#----------------------------Splitting EMG data by switchpoints done--------------------------------------------------

#----------------------------Splitting EMG data by posterior probability of switchpoints------------------------------
# Run through the switchpoint splits
for split in plot_switch:
	# And run through the laser conditions
	for laser in range(num_lasers):
		# Make lists to pull out the gaping EMG data based on posterior probability of switchpoints
		gapes_before_posterior = [[] for i in range(num_tastes)]
		gapes_after_posterior = [[] for i in range(num_tastes)]

		# Now run through the datasets
		for dataset in range(len(converged_trials)):
			# Run through the converged trials in this dataset for this laser condition
			for trial in range(converged_trials[dataset][laser].shape[0]):
				# Find the total posterior probability of the switchpoint being before the split
				switchpoints_before_split = np.where(potential_switchpoint_array[dataset][laser][:, 1] < split)[0]
				prob_before_split = np.sum(posterior_prob_switchpoints[dataset][laser][trial, switchpoints_before_split])

				# Append to gapes before if the posterior probability is > 0.5
				if prob_before_split > 0.5:
					# Append the data for this trial to the proper taste in gapes_before
					gapes_before_posterior[int(converged_trials[dataset][laser][trial]/num_trials[dataset])].append(gapes[dataset][laser, int(converged_trials[dataset][laser][trial]/num_trials[dataset]), int(converged_trials[dataset][laser][trial] % num_trials[dataset]), :])
				# Else append to gapes after
				else:
					gapes_after_posterior[int(converged_trials[dataset][laser][trial]/num_trials[dataset])].append(gapes[dataset][laser, int(converged_trials[dataset][laser][trial]/num_trials[dataset]), int(converged_trials[dataset][laser][trial] % num_trials[dataset]), :])

		# Convert the gapes_before_posterior and gapes_after_posterior lists to numpy arrays
		gapes_before_posterior = [np.array(gapes_before_posterior[i]) for i in range(num_tastes)]
		gapes_after_posterior = [np.array(gapes_after_posterior[i]) for i in range(num_tastes)]

		# Save these lists in the plot directory
		np.save("Gapes_before_{:d}_Dur{:d}_Lag{:d}_posterior.npy".format(split*10, int(unique_lasers[0][laser, 0]), int(unique_lasers[0][laser, 1])), gapes_before_posterior)
		np.save("Gapes_after_{:d}_Dur{:d}_Lag{:d}_posterior.npy".format(split*10, int(unique_lasers[0][laser, 0]), int(unique_lasers[0][laser, 1])), gapes_after_posterior)

		# Now make the EMG plots for this laser condition and switchpoint split
		# First plot all the tastes together (without error bars)
		# Gaping on trials with switchpoint before split
		fig = plt.figure()
		for i in range(num_tastes):
			plt.plot(np.mean(gapes_before_posterior[i][:, pre_stim:pre_stim + post_stim], axis = 0), label = tastes[i])
		plt.legend()
		plt.xlabel("Time post stimulus (ms)")
		plt.ylabel("Mean fraction of power in 4-6Hz")
		plt.title("Palatability switchpoint < {:d}ms".format(split*10) + "\n" + "Dur: {:d}ms, Lag: {:d}ms, Trials: {}".format(int(unique_lasers[0][laser, 0]), int(unique_lasers[0][laser, 1]), [gapes_before_posterior[i].shape[0] for i in range(4)]))
		plt.tight_layout()
		fig.savefig("Before_{:d}_Dur{:d}_Lag{:d}_posterior.png".format(split*10, int(unique_lasers[0][laser, 0]), int(unique_lasers[0][laser, 1])), bbox_inches = "tight")
		# Gaping on trials with switchpoint after split
		fig = plt.figure()
		for i in range(num_tastes):
			plt.plot(np.mean(gapes_after_posterior[i][:, pre_stim:pre_stim + post_stim], axis = 0), label = tastes[i])
		plt.legend()
		plt.xlabel("Time post stimulus (ms)")
		plt.ylabel("Mean fraction of power in 4-6Hz")
		plt.title("Palatability switchpoint > {:d}ms".format(split*10) + "\n" + "Dur: {:d}ms, Lag: {:d}ms, Trials: {}".format(int(unique_lasers[0][laser, 0]), int(unique_lasers[0][laser, 1]), [gapes_after_posterior[i].shape[0] for i in range(4)]))
		plt.tight_layout()
		fig.savefig("After_{:d}_Dur{:d}_Lag{:d}_posterior.png".format(split*10, int(unique_lasers[0][laser, 0]), int(unique_lasers[0][laser, 1])), bbox_inches = "tight")
		plt.close("all")

		# Now plot only dil and concentrated quinine, with errorbars for comparisons
		# Gaping on trials with switchpoint before split
		fig = plt.figure()
		plt.plot(np.mean(gapes_before_posterior[2][:, pre_stim:pre_stim + post_stim], axis = 0), label = tastes[2])
		std_error = np.std(gapes_before_posterior[2][:, pre_stim:pre_stim + post_stim], axis = 0)/np.sqrt(gapes_before_posterior[2].shape[0])
		plt.fill_between(np.arange(post_stim), np.mean(gapes_before_posterior[2][:, pre_stim:pre_stim + post_stim], axis = 0) - std_error, np.mean(gapes_before_posterior[2][:, pre_stim:pre_stim + post_stim], axis = 0) + std_error, alpha = 0.3)
		plt.plot(np.mean(gapes_before_posterior[3][:, pre_stim:pre_stim + post_stim], axis = 0), label = tastes[3])
		std_error = np.std(gapes_before_posterior[3][:, pre_stim:pre_stim + post_stim], axis = 0)/np.sqrt(gapes_before_posterior[3].shape[0])
		plt.fill_between(np.arange(post_stim), np.mean(gapes_before_posterior[3][:, pre_stim:pre_stim + post_stim], axis = 0) - std_error, np.mean(gapes_before[3][:, pre_stim:pre_stim + post_stim], axis = 0) + std_error, alpha = 0.3)
		plt.legend()
		plt.xlabel("Time post stimulus (ms)")
		plt.ylabel("Mean fraction of power in 4-6Hz")
		plt.title("Palatability switchpoint < {:d}ms".format(split*10) + "\n" + "Dur: {:d}ms, Lag: {:d}ms, Trials: {}".format(int(unique_lasers[0][laser, 0]), int(unique_lasers[0][laser, 1]), [gapes_before_posterior[i+2].shape[0] for i in range(2)]))
		plt.tight_layout()
		fig.savefig("Before_{:d}_Dur{:d}_Lag{:d}_Quinine_posterior.png".format(split*10, int(unique_lasers[0][laser, 0]), int(unique_lasers[0][laser, 1])), bbox_inches = "tight")
		# Gaping on trials with switchpoint after split
		fig = plt.figure()
		plt.plot(np.mean(gapes_after_posterior[2][:, pre_stim:pre_stim + post_stim], axis = 0), label = tastes[2])
		std_error = np.std(gapes_after_posterior[2][:, pre_stim:pre_stim + post_stim], axis = 0)/np.sqrt(gapes_after_posterior[2].shape[0])
		plt.fill_between(np.arange(post_stim), np.mean(gapes_after_posterior[2][:, pre_stim:pre_stim + post_stim], axis = 0) - std_error, np.mean(gapes_after_posterior[2][:, pre_stim:pre_stim + post_stim], axis = 0) + std_error, alpha = 0.3)
		plt.plot(np.mean(gapes_after_posterior[3][:, pre_stim:pre_stim + post_stim], axis = 0), label = tastes[3])
		std_error = np.std(gapes_after_posterior[3][:, pre_stim:pre_stim + post_stim], axis = 0)/np.sqrt(gapes_after_posterior[3].shape[0])
		plt.fill_between(np.arange(post_stim), np.mean(gapes_after_posterior[3][:, pre_stim:pre_stim + post_stim], axis = 0) - std_error, np.mean(gapes_after_posterior[3][:, pre_stim:pre_stim + post_stim], axis = 0) + std_error, alpha = 0.3)
		plt.legend()
		plt.xlabel("Time post stimulus (ms)")
		plt.ylabel("Mean fraction of power in 4-6Hz")
		plt.title("Palatability switchpoint > {:d}ms".format(split*10) + "\n" + "Dur: {:d}ms, Lag: {:d}ms, Trials: {}".format(int(unique_lasers[0][laser, 0]), int(unique_lasers[0][laser, 1]), [gapes_after_posterior[i+2].shape[0] for i in range(2)]))
		plt.tight_layout()
		fig.savefig("After_{:d}_Dur{:d}_Lag{:d}_Quinine_posterior.png".format(split*10, int(unique_lasers[0][laser, 0]), int(unique_lasers[0][laser, 1])), bbox_inches = "tight")
		plt.close("all")


#----------------------------Splitting EMG data by posterior probability of switchpoints done-------------------------

'''
Not lining up by switchpoints anymore

#----------------------------Plotting EMG data lined up by switchpoints-----------------------------------------------
# Plot the EMG data, averaged across trials, lined up by the switchpoints
# First ask the user of the pre and post-switchpoint time they want to use in the plots
pre_switch = easygui.multenterbox(msg = 'Enter the time pre-switchpoint that you want to plot (Choose a small enough number, will give errors if the window reaches the end of trial)', fields = ['Time pre switchpoint (ms)'])
pre_switch = int(pre_switch[0])
post_switch = easygui.multenterbox(msg = 'Enter the time post-switchpoint that you want to plot', fields = ['Time post switchpoint (ms)'])
post_switch = int(post_switch[0])

# Make a list of lists to store the EMG data (gapes and ltps) for every laser condition - make two such lists, one for each switchpoint
gapes_plot1 = [[[] for j in range(num_tastes)] for i in range(num_lasers)]
gapes_plot2 = [[[] for j in range(num_tastes)] for i in range(num_lasers)]
ltps_plot1 = [[[] for j in range(num_tastes)] for i in range(num_lasers)]
ltps_plot2 = [[[] for j in range(num_tastes)] for i in range(num_lasers)]

# Now run through the datasets
for dataset in range(len(converged_trials)):
	# And run through the laser conditions in each dataset
	for laser in range(num_lasers):
		# Run through the converged trials in this dataset for this laser condition
		for trial in range(converged_trials[dataset][laser].shape[0]):
			# Check if switchpoint1 happens after laser inactivation starts
			if switchpoints[dataset][laser][trial, 0]*10 > unique_lasers[0][laser, 1]:
				# Append the EMG data for this trial to the respective lists - correct the switchpoint by adding the laser duration
				gapes_plot1[laser][int(converged_trials[dataset][laser][trial]/num_trials[dataset])].append(gapes[dataset][laser, int(converged_trials[dataset][laser][trial]/num_trials[dataset]), int(converged_trials[dataset][laser][trial] % num_trials[dataset]), switchpoints[dataset][laser][trial, 0]*10 + int(unique_lasers[0][laser, 0]) : switchpoints[dataset][laser][trial, 0]*10 + int(unique_lasers[0][laser, 0]) + post_switch])
				ltps_plot1[laser][int(converged_trials[dataset][laser][trial]/num_trials[dataset])].append(ltps[dataset][laser, int(converged_trials[dataset][laser][trial]/num_trials[dataset]), int(converged_trials[dataset][laser][trial] % num_trials[dataset]), switchpoints[dataset][laser][trial, 0]*10 + int(unique_lasers[0][laser, 0]) : switchpoints[dataset][laser][trial, 0]*10 + int(unique_lasers[0][laser, 0]) + post_switch])
			# Do not correct the switchpoint if it happened before the laser inactivation starts
			else:
				gapes_plot1[laser][int(converged_trials[dataset][laser][trial]/num_trials[dataset])].append(gapes[dataset][laser, int(converged_trials[dataset][laser][trial]/num_trials[dataset]), int(converged_trials[dataset][laser][trial] % num_trials[dataset]), switchpoints[dataset][laser][trial, 0]*10 : switchpoints[dataset][laser][trial, 0]*10 + post_switch])
				ltps_plot1[laser][int(converged_trials[dataset][laser][trial]/num_trials[dataset])].append(ltps[dataset][laser, int(converged_trials[dataset][laser][trial]/num_trials[dataset]), int(converged_trials[dataset][laser][trial] % num_trials[dataset]), switchpoints[dataset][laser][trial, 0]*10 : switchpoints[dataset][laser][trial, 0]*10 + post_switch])

			# Now check if switchpoint2 happens after laser inactivation starts
			if switchpoints[dataset][laser][trial, 1]*10 > unique_lasers[0][laser, 1]:
				# Append the EMG data for this trial to the respective lists - correct the switchpoint by adding the laser duration
				gapes_plot2[laser][int(converged_trials[dataset][laser][trial]/num_trials[dataset])].append(gapes[dataset][laser, int(converged_trials[dataset][laser][trial]/num_trials[dataset]), int(converged_trials[dataset][laser][trial] % num_trials[dataset]), switchpoints[dataset][laser][trial, 1]*10 + int(unique_lasers[0][laser, 0]) - pre_switch : switchpoints[dataset][laser][trial, 1]*10 + int(unique_lasers[0][laser, 0]) + post_switch])
				ltps_plot2[laser][int(converged_trials[dataset][laser][trial]/num_trials[dataset])].append(ltps[dataset][laser, int(converged_trials[dataset][laser][trial]/num_trials[dataset]), int(converged_trials[dataset][laser][trial] % num_trials[dataset]), switchpoints[dataset][laser][trial, 1]*10 + int(unique_lasers[0][laser, 0]) - pre_switch : switchpoints[dataset][laser][trial, 1]*10 + int(unique_lasers[0][laser, 0]) + post_switch])
			# Do not correct the switchpoint if it happened before the laser inactivation starts
			else:
				gapes_plot2[laser][int(converged_trials[dataset][laser][trial]/num_trials[dataset])].append(gapes[dataset][laser, int(converged_trials[dataset][laser][trial]/num_trials[dataset]), int(converged_trials[dataset][laser][trial] % num_trials[dataset]), switchpoints[dataset][laser][trial, 1]*10 - pre_switch : switchpoints[dataset][laser][trial, 1]*10 + post_switch])
				ltps_plot2[laser][int(converged_trials[dataset][laser][trial]/num_trials[dataset])].append(ltps[dataset][laser, int(converged_trials[dataset][laser][trial]/num_trials[dataset]), int(converged_trials[dataset][laser][trial] % num_trials[dataset]), switchpoints[dataset][laser][trial, 1]*10 - pre_switch : switchpoints[dataset][laser][trial, 1]*10 + post_switch])

# Convert these lists into numpy arrays to help in averaging across trials while plotting
for i in range(num_lasers):
	gapes_plot1[i] = [np.array(gapes_plot1[i][j]) for j in range(num_tastes)]
	gapes_plot2[i] = [np.array(gapes_plot2[i][j]) for j in range(num_tastes)]
	ltps_plot1[i] = [np.array(ltps_plot1[i][j]) for j in range(num_tastes)]
	ltps_plot2[i] = [np.array(ltps_plot2[i][j]) for j in range(num_tastes)]

# Now plot the results by laser conditions
for laser in range(num_lasers):
	# Make 4 separate figures - 2 each for gapes and ltps
	fig_gapes1, ax_gapes1 = plt.subplots()
	fig_gapes2, ax_gapes2 = plt.subplots()
	fig_ltps1, ax_ltps1 = plt.subplots()
	fig_ltps2, ax_ltps2 = plt.subplots()

	# Now run through the tastes
	for taste in range(num_tastes):
		# And plot to the respective set of axes
		ax_gapes1.plot(np.arange(post_switch), np.mean(gapes_plot1[laser][taste], axis = 0), label = tastes[taste])
		ax_gapes1.set_xlabel("Time post switchpoint1 (ms)")
		ax_gapes1.set_ylabel("Average posterior probability of 3.5-6Hz EMG activity")
		ax_gapes2.plot(np.arange(pre_switch + post_switch) - pre_switch, np.mean(gapes_plot2[laser][taste], axis = 0), label = tastes[taste])
		ax_gapes2.set_xlabel("Time post switchpoint2 (ms)")
		ax_gapes2.set_ylabel("Average posterior probability of 3.5-6Hz EMG activity")
		ax_ltps1.plot(np.arange(post_switch), np.mean(ltps_plot1[laser][taste], axis = 0), label = tastes[taste])
		ax_ltps1.set_xlabel("Time post switchpoint1 (ms)")
		ax_ltps1.set_ylabel("Average posterior probability of 6-10Hz EMG activity")
		ax_ltps2.plot(np.arange(pre_switch + post_switch) - pre_switch, np.mean(ltps_plot2[laser][taste], axis = 0), label = tastes[taste])
		ax_ltps2.set_xlabel("Time post switchpoint2 (ms)")
		ax_ltps2.set_ylabel("Average posterior probability of 6-10Hz EMG activity")

	# Add legends to all the figures
	ax_gapes1.legend()
	ax_gapes2.legend()
	ax_ltps1.legend()
	ax_ltps2.legend()

	# Save the figures and close the plots
	fig_gapes1.savefig("Gapes_Switchpoint1_Dur{:d}_Lag{:d}.png".format(int(unique_lasers[0][laser, 0]), int(unique_lasers[0][laser, 1])), bbox_inches = "tight")
	fig_gapes2.savefig("Gapes_Switchpoint2_Dur{:d}_Lag{:d}.png".format(int(unique_lasers[0][laser, 0]), int(unique_lasers[0][laser, 1])), bbox_inches = "tight")
	fig_ltps1.savefig("LTP_Switchpoint1_Dur{:d}_Lag{:d}.png".format(int(unique_lasers[0][laser, 0]), int(unique_lasers[0][laser, 1])), bbox_inches = "tight")
	fig_ltps2.savefig("LTP_Switchpoint2_Dur{:d}_Lag{:d}.png".format(int(unique_lasers[0][laser, 0]), int(unique_lasers[0][laser, 1])), bbox_inches = "tight")
	plt.close("all")

#----------------------------Plotting EMG data lined up by switchpoints done------------------------------------------

gapes_less_700 = [[[] for j in range(num_tastes)] for i in range(num_lasers)]
gapes_more_700 = [[[] for j in range(num_tastes)] for i in range(num_lasers)]
gapes_Li_less_700 = [[] for i in range(4)]
gapes_Li_more_700 = [[] for i in range(4)]
gape_lag = [[] for i in range(4)]
switch = [[] for i in range(4)]
	# Get the EMG data for the middle inactivation condition that switched into palatability firing before or after 700ms
	gapes = hf5.root.ancillary_analysis.gapes[:]
	switchpoints = hf5.root.MCMC_switch.switchpoints.laser_condition_2[:]
	converged_trials = hf5.root.MCMC_switch.converged_trial_nums.laser_condition_2[:]
	trains_dig_in = hf5.list_nodes('/spike_trains')
	num_tastes = len(trains_dig_in)
	num_trials = int(np.array(hf5.root.MCMC_switch.inactivated_spikes[:]).shape[1] / num_tastes)
	gapes_Li = hf5.root.ancillary_analysis.gapes_Li[:]
	for i in range(converged_trials.shape[0]):
		if switchpoints[i, 1] < 90.0:
			switch[int(converged_trials[i]/num_trials)].append([switchpoints[i, 0], switchpoints[i, 1]])
			gapes_less_700[int(converged_trials[i]/num_trials)].append(gapes[2, int(converged_trials[i]/num_trials), int(converged_trials[i] % num_trials), :])
			first_gape = np.where(gapes_Li[0, int(converged_trials[i]/num_trials), int(converged_trials[i] % num_trials), 2000:] > 0.0)[0]
			if len(first_gape) > 0:
				gapes_Li_less_700[int(converged_trials[i]/num_trials)].append(first_gape[0])
				gape_lag[int(converged_trials[i]/num_trials)].append([switchpoints[i, 1], first_gape[0]])
			else:
				gapes_Li_less_700[int(converged_trials[i]/num_trials)].append(-1.0)
		# elif switchpoints[i, 1] > 120.0:
		#	gapes_more_700[int(converged_trials[i]/num_trials)].append(gapes[2, int(converged_trials[i]/num_trials), int(converged_trials[i] % num_trials), :])
		else:
			switch[int(converged_trials[i]/num_trials)].append([switchpoints[i, 0], switchpoints[i, 1]])
			gapes_more_700[int(converged_trials[i]/num_trials)].append(gapes[2, int(converged_trials[i]/num_trials), int(converged_trials[i] % num_trials), :])
			first_gape = np.where(gapes_Li[0, int(converged_trials[i]/num_trials), int(converged_trials[i] % num_trials), 2000:] > 0.0)[0]
			if len(first_gape) > 0:
				gapes_Li_more_700[int(converged_trials[i]/num_trials)].append(first_gape[0])
				gape_lag[int(converged_trials[i]/num_trials)].append([switchpoints[i, 1], first_gape[0]])
			else:
				gapes_Li_more_700[int(converged_trials[i]/num_trials)].append(-1.0)

'''
