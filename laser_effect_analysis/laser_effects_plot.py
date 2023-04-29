# Import stuff!
import numpy as np
import tables
import easygui
import sys
import os
import matplotlib
matplotlib.use('Agg')
import pylab as plt
import seaborn as sns
sns.set(style="white", context="talk", font_scale=1.8)
sns.set_color_codes(palette = 'colorblind')
#import matplotlib.pyplot as plt
#import seaborn as sns
#sns.set(context="poster")
import pandas as pd

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
mean_firing_rates = []
lasers = []
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

	mean_firing_rates.append(hf5.root.laser_effects_bayesian.mean_firing_rates[:])
	lasers.append(hf5.root.laser_effects_bayesian.laser_combination_d_l[:])

	# Close the hdf5 file
	hf5.close()

# Now join up all the data into big numpy arrays
# If there's only one data file, set the final arrays to the only array read in
if len(lasers) == 1:
	lasers = lasers[0]
	mean_firing_rates = mean_firing_rates[0]
else:
	lasers = lasers[0]
	mean_firing_rates = np.concatenate(tuple(mean_firing_rates[i] for i in range(len(mean_firing_rates))), axis = 0)

# Sort the laser conditions in ascending order, just to be sure
lasers = lasers[lasers[:, 0].argsort(), :]
lasers = lasers[lasers[:, 1].argsort(), :]

# Ask the user for the directory to save plots etc in
dir_name = easygui.diropenbox(msg = 'Choose the output directory for laser effect on firing rate analysis')
os.chdir(dir_name)

# Dump the arrays into this folder
np.save('mean_firing_rates.npy', mean_firing_rates)
np.save('lasers.npy', lasers)

# Start making the plots - make the plots by laser conditions
for condition in range(mean_firing_rates.shape[1]):
	fig = plt.figure()
	for taste in range(mean_firing_rates.shape[2]):
		# Calculate the difference in the mean firing rate between the control (laser off) and inactivated for the same taste (0 is control)
		diff = mean_firing_rates[:, condition, taste, :, 0] - mean_firing_rates[:, condition, taste, :, 1]

		# Get the proportion of the diff distribution above zero (as you'd expect laser on to inhibit firing for Arch)
		proportion_below_zero = np.sum(diff[:, :] >= 0.0, axis = 1)/diff.shape[1]

		# Now plot the distribution of these proportions across the units in the population
		sns.kdeplot(proportion_below_zero, cumulative = True, label = 'Taste {}'.format(taste))
		#plt.hist(proportion_below_zero, histtype = 'step', normed = 1, cumulative = True, label = 'Taste {}'.format(taste), linewidth = 2.0)		
		plt.xlabel('P{(laser_off firing - laser_on firing) >= 0}')
		plt.ylabel('Proportion of units (Total = {})'.format(mean_firing_rates.shape[0]))
		#plt.tick_params(axis='both', which='major', labelsize=20)
		plt.xlim([0.0, 1.0])
		plt.title('Proportion of units against P{(laser_off firing - laser_on firing) >= 0}') 
	plt.legend(loc = 'upper left', fontsize = 15)
	#fig.set_size_inches(18.5, 10.5)
	plt.tight_layout()
	fig.savefig('duration:{},lag:{}.png'.format(lasers[condition + 1, 0], lasers[condition + 1, 1]), bbox_inches = 'tight')
	plt.close('all')

# Now make plots by tastes
for taste in range(mean_firing_rates.shape[2]):
	fig = plt.figure()
	for condition in range(mean_firing_rates.shape[1]):
		# Calculate the difference in the mean firing rate between this condition+taste and control (laser off) for the same taste (0 is control)
		diff = mean_firing_rates[:, condition, taste, :, 0] - mean_firing_rates[:, condition, taste, :, 1]

		# Get the proportion of the diff distribution above zero (as you'd expect laser on to inhibit firing for Arch)
		proportion_below_zero = np.sum(diff[:, :] >= 0.0, axis = 1)/diff.shape[1]

		# Now plot the distribution of these proportions across the units in the population
		sns.kdeplot(proportion_below_zero, cumulative = True, label = 'Dur:{}ms,Lag:{}ms'.format(lasers[condition + 1, 0], lasers[condition + 1, 1]))
		#plt.hist(proportion_below_zero, histtype = 'step', normed = 1, cumulative = True, label = 'duration:{},lag:{}'.format(lasers[condition + 1, 0], lasers[condition + 1, 1]), linewidth = 2.0)		
		plt.xlabel('P{(laser_off firing - laser_on firing) >= 0}')
		plt.ylabel('Number of units (Total = {})'.format(mean_firing_rates.shape[0]))
		#plt.tick_params(axis='both', which='major', labelsize=20)
		plt.xlim([0.0, 1.0])
		plt.title('Proportion of units against P{(laser_off firing - laser_on firing) >= 0}') 
	plt.legend(loc = 'upper left', fontsize = 15)
	plt.tight_layout()
	#fig.set_size_inches(18.5, 10.5)
	fig.savefig('Taste{}.png'.format(taste), bbox_inches = 'tight')
	plt.close('all')

# Ask the user for the significance criteria to use
p_values = easygui.multenterbox(msg = 'Enter the significance criteria for laser effect calculation', fields = ['Significance level (Bayesian p value)'])
p_values = float(p_values[0])

# Now make barcharts by tastes and grouped by laser conditions
# First pull out the data according to the significance criterion
data = np.zeros((mean_firing_rates.shape[1], mean_firing_rates.shape[2]))
for condition in range(mean_firing_rates.shape[1]):
	for taste in range(mean_firing_rates.shape[2]):
		# Calculate the difference in the mean firing rate between this condition+taste and control (laser off) for the same taste (0 is control)
		diff = mean_firing_rates[:, condition, taste, :, 0] - mean_firing_rates[:, condition, taste, :, 1]

		# Get the proportion of the diff distribution below zero (as you'd expect laser on to inhibit firing for Arch)
		proportion_below_zero = np.sum(diff[:, :] <= 0.0, axis = 1)/diff.shape[1]

		# Now get the number of neurons that have this proportion <= p_values
		data[condition, taste] = len(np.where(proportion_below_zero <= p_values)[0])
# Make a pandas dataframe with this data
taste_names = ['Taste {:d}'.format(i + 1) for i in range(mean_firing_rates.shape[2])]
laser_condition_names = ['Dur: {}ms, Lag: {}ms'.format(lasers[i+1, 0], lasers[i+1, 1]) for i in range(mean_firing_rates.shape[1])]
plot_data = pd.DataFrame(data = data, columns = taste_names)
plot_data['lasers'] = laser_condition_names
plot_data = pd.melt(plot_data.reset_index(), id_vars = ['lasers'], value_vars = taste_names, value_name = 'Tastes')

# Make the barplot
fig = plt.figure()
sns.barplot(x = 'lasers', y = 'Tastes', hue = 'variable', data = plot_data)
plt.xlabel('Laser conditions')
plt.ylabel('Number of neurons with significant' + '\n' + 'suppression of firing')
plt.title('Total number of neurons = {:d}'.format(mean_firing_rates.shape[0]))
sns.despine(bottom=True)
plt.legend(loc = 'upper left', fontsize = 15)
plt.tight_layout(h_pad=3)
#fig.set_size_inches(18.5, 10.5)
fig.savefig('Significant_neurons.png', bbox_inches = 'tight')
plt.close('all')

# Make histograms, across tastes, of the change in firing produced by the laser as a percentage of the firing in the control/laser off condition. Plot such a histogram for each laser condition on top of each other
# First get the difference in mean firing rate produced by the laser (averaged across tastes) as percent of the mean control firing
mean_diff = np.mean(mean_firing_rates[:, :, :, :, 0] - mean_firing_rates[:, :, :, :, 1], axis = (2, 3))/np.mean(mean_firing_rates[:, :, :, :, 0], axis = (2, 3))
fig = plt.figure()
# Plot the laser conditions on the same histogram, on top of each other
for condition in range(mean_firing_rates.shape[1]):
	plt.hist(mean_diff[:, condition], alpha = 0.5, bins = 20, label = 'Dur:{}ms,Lag:{}ms'.format(lasers[condition + 1, 0], lasers[condition + 1, 1]))
plt.xlabel(r"$\frac{Firing_{LaserOFF} - Firing_{LaserON}}{Firing_{LaserOFF}}$")
plt.ylabel("Number of neurons")
plt.legend(loc = 'upper left', fontsize = 15)
plt.tight_layout(h_pad=3)
fig.savefig('Percent_change_firing_rate.png', bbox_inches = 'tight')
plt.close('all')



 



		
