# Import stuff!
import numpy as np
import tables
import easygui
import sys
import os
import pandas as pd
import pymc3 as pm
import theano.tensor as tt
import matplotlib.pyplot as plt

# Ask the user for the hdf5 files that need to be plotted together
dirs = []
while True:
	dir_name = easygui.diropenbox(msg = 'Choose a directory with a hdf5 file, hit cancel to stop choosing')
	try:
		if len(dir_name) > 0:
			dirs.append(dir_name)
	except:
		break

# Now run through the directories, and pull out the data
unique_lasers = []
unscaled_neural_response = []
palatability = []
r_spearman = []
pre_stim = []
params = []
laser = []
num_units = []

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

	# Pull the data from the /ancillary_analysis node
	unique_lasers.append(hf5.root.ancillary_analysis.laser_combination_d_l[:])
	unscaled_neural_response.append(hf5.root.ancillary_analysis.unscaled_neural_response[:])
	palatability.append(hf5.root.ancillary_analysis.palatability[:])
	laser.append(hf5.root.ancillary_analysis.laser[:])
	r_spearman.append(hf5.root.ancillary_analysis.r_spearman[:])
	# Reading single values from the hdf5 file seems hard, needs the read() method to be called
	pre_stim.append(hf5.root.ancillary_analysis.pre_stim.read())
	params.append(hf5.root.ancillary_analysis.params[:])
	# Also maintain a counter of the number of units in the analysis
	num_units.append(hf5.root.ancillary_analysis.palatability.shape[1])

	# Close the hdf5 file
	hf5.close()

# Check if the number of laser activation/inactivation windows is same across files, raise an error and quit if it isn't
if all(unique_lasers[i].shape == unique_lasers[0].shape for i in range(len(unique_lasers))):
	pass
else:
	print("Number of inactivation/activation windows doesn't seem to be the same across days. Please check and try again")
	sys.exit()

# Now first set the ordering of laser trials straight across data files
# We don't really need to do this as the laser conditions have already been sorted into order when the ancillary analysis node was made
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

# Ask the user for the time window they want to run the regression analysis on
# First get an array of x (aka raw time) values
x = np.arange(0, unscaled_neural_response[0].shape[0]*params[0][1], params[0][1]) - pre_stim[0]
time_limits = easygui.multenterbox(msg = 'Enter the time limits for performing palatability regression analysis', fields = ['Start time (ms)', 'End time (ms)'])
for i in range(len(time_limits)):
	time_limits[i] = int(time_limits[i])
analyze_indices = np.where((x>=time_limits[0])*(x<=time_limits[1]))[0]

# Target: A pandas dataframe with time, neuron, palatability, and laser condition as columns.
# To make this dataframe, we first need to expand out those variables in 1D arrays, and finally concatenate the arrays from all the data files
time = []
palatability_array = []
response_array = []
laser_array = []
neuron = []
neuron_count = 0
# Run through the data appended from each file
for i in range(len(unique_lasers)):
	# Make an array of time of the same dimensional order as the palatability array
	this_time = np.tile(np.arange(len(analyze_indices)).reshape(len(analyze_indices), 1, 1), (1, palatability[i].shape[1], palatability[i].shape[2]))
	# Make a similar array of neuron numbers - start from the current count of neurons/units
	this_neurons = np.tile(np.arange(neuron_count, neuron_count + num_units[i], 1).reshape(1, num_units[i], 1), (len(analyze_indices), 1, palatability[i].shape[2]))
	# Add the neuron count in this file to the counter
	neuron_count += num_units[i]

	# Take a mean of the spearman correlations across laser conditions and time
	# This assumes that the sign of the palatability representation/correlation by a neuron doesn't change across laser conditions and time
	# If the correlation is negative, multiply the firing of that neuron by -1 so that eventually all regression coefficients turn out to be positive and can be averaged across
	mean_corr = np.mean(r_spearman[i][:, analyze_indices, :], axis = (0, 1))
	# Then convert this array of mean correlations into an array of 1s and -1s depending upon the sign of the correlation
	sign_corr = np.tile(np.sign(mean_corr).reshape(1, num_units[i], 1), (len(analyze_indices), 1, palatability[i].shape[2]))

	# Run through the unique laser conditions and mark each trial by the condition number
	laser_condition = np.zeros(this_time.flatten().shape)
	for condition in range(unique_lasers[i].shape[0]):
		laser_condition[(laser[i][analyze_indices, :, :, 0].flatten() == unique_lasers[i][laser_order[i][condition]][0])*(laser[i][analyze_indices, :, :, 1].flatten() == unique_lasers[i][laser_order[i][condition]][1])] = condition

	# Standardize the firing/response data by the mean firing for every neuron in every time bin - multiply by the sign of the average spearman correlation here
	this_response_mean = np.tile(np.mean(unscaled_neural_response[i][analyze_indices, :, :]*sign_corr, axis = -1).reshape((len(analyze_indices), unscaled_neural_response[i].shape[1], 1)), (1, 1, unscaled_neural_response[i].shape[2]))
	this_response_std = np.tile(np.std(unscaled_neural_response[i][analyze_indices, :, :]*sign_corr, axis = -1).reshape((len(analyze_indices), unscaled_neural_response[i].shape[1], 1)), (1, 1, unscaled_neural_response[i].shape[2]))
	this_response = (unscaled_neural_response[i][analyze_indices, :, :]*sign_corr - this_response_mean)/this_response_std
	# If the firing on all trials in a time bin is zero, its std would be 0 and that would give nans in this_response. Change nans to 0 if they occur
	this_response[np.isnan(this_response)] = 0.0

	# Now append these data to the respective arrays
	time.append(this_time.flatten())
	neuron.append(this_neurons.flatten())
	palatability_array.append(palatability[i][analyze_indices, :, :].flatten())
	response_array.append(this_response.flatten())
	laser_array.append(laser_condition)

# Now make a pandas dataframe with the data. (Subtract 1 from the palatability array to index palatabilities from 0 - not doing this since there are no main effects in the model anymore)
firing_data = pd.DataFrame({"Time": np.concatenate(time), "Neuron": np.concatenate(neuron).astype("int"), "Palatability": np.concatenate(palatability_array), "Laser": np.concatenate(laser_array), "Firing": np.concatenate(response_array)})

# Ask the user for the directory to save the results in
# Ask the user for the directory to save plots etc in
dir_name = easygui.diropenbox(msg = 'Choose the output directory for palatability regression analysis')
os.chdir(dir_name)
# Save the dataframe in this directory
firing_data.to_csv("firing_data.csv")

# Start setting up the model
# Since we already standardized the firing data by the mean firing of every neuron at every point of time, we do not need to include main effects to capture the variability of
# a neuron's firing across time. We just use palatability slopes, one for each time point
with pm.Model() as model:
	# Palatability slopes, one for each time point (one set for each laser condition)
	coeff_pal = pm.Normal("coeff_pal", mu = 0, sd = 1, shape = (len(analyze_indices), unique_lasers[0].shape[0]))
	# Observation standard deviation
	sd = pm.HalfCauchy("sd", 1)
	# Regression equation for the mean observation
	regression = coeff_pal[tt.cast(firing_data["Time"], 'int32'), tt.cast(firing_data["Laser"], 'int32')]*firing_data["Palatability"]
	# Actual observations
	obs = pm.Normal("obs", mu = regression, sd = sd, observed = firing_data["Firing"])

	# Metropolis sampling works best!
	tr = pm.sample(tune = 10000, draws = 50000, cores = 4, start = pm.find_MAP(), step = pm.Metropolis())

# Print the Gelman-Rubin rhat convergence statistics to a file
f = open("palatability_regression_convergence.txt", "w")
print(str(pm.gelman_rubin(tr)), file = f)
f.close()

# Save the trace to the output folder as a numpy array, for later reference
# Save every 10th sample from the trace, to avoid any autocorrelation issues
np.save("palatability_regression_trace.npy", tr[::10]["coeff_pal"])

# Convert the trace to a dataframe, and save that too
# Save every 10th sample from the trace, to avoid any autocorrelation issues
tr_df = pm.trace_to_dataframe(tr[::10])
tr_df.to_csv("palatability_regression_trace.csv")

# Plot the results of the palatability regression analysis
# First just plot the mean regression coefficients for every laser condition, across time
fig = plt.figure()
mean_coeff = np.mean(tr[::10]["coeff_pal"], axis = 0)
hpd_coeff = pm.hpd(tr[::10]["coeff_pal"], alpha = 0.05)
for condition in range(unique_lasers[0].shape[0]):
	plt.plot(x[analyze_indices], mean_coeff[:, condition], linewidth = 3.0, label = "Dur:{}ms, Lag:{}ms".format(unique_lasers[0][condition][0], unique_lasers[0][condition][1]))
plt.legend()
plt.xlabel("Time post taste delivery (ms)")
plt.ylabel("Mean posterior regression coefficient")
fig.savefig("palatability_regression_coefficients_mean.png", bbox_inches = "tight")
plt.close("all")
# Now plot the mean and SD of the regression coefficients for every laser condition, across time
fig = plt.figure()
for condition in range(unique_lasers[0].shape[0]):
	plt.plot(x[analyze_indices], np.mean(tr[::10]["coeff_pal"], axis = 0)[:, condition], linewidth = 3.0, label = "Dur:{}ms, Lag:{}ms".format(unique_lasers[0][condition][0], unique_lasers[0][condition][1]))
	plt.fill_between(x[analyze_indices], hpd_coeff[:, condition, 0], hpd_coeff[:, condition, 1], alpha = 0.5)
plt.legend()
plt.xlabel("Time post taste delivery (ms)")
plt.ylabel("Mean posterior regression coefficient")
fig.savefig("palatability_regression_coefficients_hpd.png", bbox_inches = "tight")
plt.close("all")
