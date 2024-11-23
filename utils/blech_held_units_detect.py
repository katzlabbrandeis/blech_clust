# Import stuff!
import numpy as np
import tables
import easygui
import sys
import os
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", context="talk", font_scale=2)

#@jit(nogil = True)
def calculate_J3(wf_day1, wf_day2):
	# Send these off to calculate J1 and J2
	J1 = calculate_J1(wf_day1, wf_day2)
	J2 = calculate_J2(wf_day1, wf_day2)
	# Get J3 as the ratio of J2 and J1
	J3 = J2/J1
	return J3

#@jit(nogil = True)
def calculate_J2(wf_day1, wf_day2):
	# Get the mean PCA waveforms on days 1 and 2
	day1_mean = np.mean(wf_day1, axis = 0)
	day2_mean = np.mean(wf_day2, axis = 0)
	
	# Get the overall inter-day mean
	overall_mean = np.mean(np.concatenate((wf_day1, wf_day2), axis = 0), axis = 0)

	# Get the distances of the daily means from the inter-day mean
	dist1 = cdist(day1_mean.reshape((-1, 3)), overall_mean.reshape((-1, 3)))
	dist2 = cdist(day2_mean.reshape((-1, 3)), overall_mean.reshape((-1, 3)))

	# Multiply the distances by the number of points on both days and sum to get J2
	J2 = wf_day1.shape[0]*np.sum(dist1) + wf_day2.shape[0]*np.sum(dist2)
	return J2 

#@jit(nogil = True)
def calculate_J1(wf_day1, wf_day2):
	# Get the mean PCA waveforms on days 1 and 2
	day1_mean = np.mean(wf_day1, axis = 0)
	day2_mean = np.mean(wf_day2, axis = 0)

	# Get the Euclidean distances of each day from its daily mean
	day1_dists = cdist(wf_day1, day1_mean.reshape((-1, 3)), metric = 'euclidean')
	day2_dists = cdist(wf_day2, day2_mean.reshape((-1, 3)), metric = 'euclidean')

	# Sum up the distances to get J1
	J1 = np.sum(day1_dists) + np.sum(day2_dists)
	return J1
	

# Ask for the directory where the first hdf5 file sits, and change to that directory
dir_name = easygui.diropenbox(msg = 'Where is the hdf5 file from the first day?', title = 'First day of data')
os.chdir(dir_name)

# Look for the hdf5 file in the directory
file_list = os.listdir('./')
hdf5_name = ''
for files in file_list:
	if files[-2:] == 'h5':
		hdf5_name = files

# Open the hdf5 file
hf51 = tables.open_file(hdf5_name, 'r')

# Now do the same for the second day of data
dir_name = easygui.diropenbox(msg = 'Where is the hdf5 file from the second day?', title = 'Second day of data')
os.chdir(dir_name)
file_list = os.listdir('./')
hdf5_name = ''
for files in file_list:
	if files[-2:] == 'h5':
		hdf5_name = files
hf52 = tables.open_file(hdf5_name, 'r')

# Ask the user for the output directory to save the held units and plots in
dir_name = easygui.diropenbox(msg = 'Where do you want to save the held units and plots?', title = 'Output directory')
os.chdir(dir_name)

# Ask the user for the percentile criterion to use to determine held units
percent_criterion = easygui.multenterbox(msg = 'What percentile of the intra unit J3 distribution do you want to use to pull out held units?', fields = ['Percentile criterion (1-100) - lower is more conservative'])
percent_criterion = float(percent_criterion[0])

# Make a file to save the numbers of the units that are deemed to have been held across days
f = open('held_units.txt', 'w')
print('Day1', '\t', 'Day2', file=f)

# Calculate the intra-unit J3 numbers by taking every unit, and calculating the J3 between the first 3rd and last 3rd of its spikes
intra_J3 = []
# Run through the units on day 1
for unit1 in range(len(hf51.root.unit_descriptor[:])):
	# Only go ahead if this is a single unit
	if hf51.root.unit_descriptor[unit1]['single_unit'] == 1:
		exec("wf_day1 = hf51.root.sorted_units.unit%03d.waveforms[:]" % (unit1))
		pca = PCA(n_components = 3)
		pca.fit(wf_day1)
		pca_wf_day1 = pca.transform(wf_day1)
		intra_J3.append(calculate_J3(pca_wf_day1[:int(wf_day1.shape[0]*(1.0/3.0)), :], pca_wf_day1[int(wf_day1.shape[0]*(2.0/3.0)):, :]))
# Run through the units on day 2
for unit2 in range(len(hf52.root.unit_descriptor[:])):
	# Only go ahead if this is a single unit
	if hf52.root.unit_descriptor[unit2]['single_unit'] == 1:
		exec("wf_day2 = hf52.root.sorted_units.unit%03d.waveforms[:]" % (unit2))
		pca = PCA(n_components = 3)
		pca.fit(wf_day2)
		pca_wf_day2 = pca.transform(wf_day2)
		intra_J3.append(calculate_J3(pca_wf_day2[:int(wf_day2.shape[0]*(1.0/3.0)), :], pca_wf_day2[int(wf_day2.shape[0]*(2.0/3.0)):, :]))

# Now calculate the inter unit J3 numbers for units of the same type on the same electrode - mark them as held if they're less than the 95th percentile of intra_J3
# Run through the units on day 1
inter_J3 = []
for unit1 in range(len(hf51.root.unit_descriptor[:])):
	# Only go ahead if this is a single unit
	if hf51.root.unit_descriptor[unit1]['single_unit'] == 1:
		# Run through the units on day 2 and check if it was present (same electrode and unit type)
		for unit2 in range(len(hf52.root.unit_descriptor[:])):
			print(unit1, unit2, len(hf51.root.unit_descriptor[:]), len(hf52.root.unit_descriptor[:]))
			if hf52.root.unit_descriptor[unit2] == hf51.root.unit_descriptor[unit1]:
			    # Load up the waveforms for unit1 and unit2
				exec("wf_day1 = hf51.root.sorted_units.unit%03d.waveforms[:]" % (unit1))
				exec("wf_day2 = hf52.root.sorted_units.unit%03d.waveforms[:]" % (unit2))

				#energy1 = np.sqrt(np.sum(wf_day1**2, axis = 1))/wf_day1.shape[1]
				#energy2 = np.sqrt(np.sum(wf_day2**2, axis = 1))/wf_day2.shape[1]

				#pca_wf_day1 = np.divide(wf_day1.T, energy1).T
				#pca_wf_day2 = np.divide(wf_day2.T, energy2).T

				# Run the PCA - pick the first 3 principal components
				pca = PCA(n_components = 3)
				pca.fit(np.concatenate((wf_day1, wf_day2), axis = 0))
				pca_wf_day1 = pca.transform(wf_day1)
				pca_wf_day2 = pca.transform(wf_day2)
				# Get inter-day J3
				inter_J3.append(calculate_J3(pca_wf_day1, pca_wf_day2))

				# Only say that this unit is held if inter_J3 <= 95th percentile of intra_J3
				#print inter_J3, np.percentile(intra_J3, 95.0)
				#wait = raw_input()
				if inter_J3[-1] <= np.percentile(intra_J3, percent_criterion):
					print(unit1, '\t', unit2, file=f)
					# Also plot both these units on the same graph
					fig = plt.figure(figsize=(18, 6))
					plt.subplot(121)
					t = np.arange(wf_day1.shape[1]/10)
					#plt.plot(t - 15, wf_day1[:, ::10].T, linewidth = 0.01, color = 'red')
					plt.plot(t - 15, np.mean(wf_day1[:, ::10], axis = 0), linewidth = 5.0, color = 'black')
					plt.plot(t - 15, np.mean(wf_day1[:, ::10], axis = 0) - np.std(wf_day1[:, ::10], axis = 0), linewidth = 2.0, color = 'black', alpha = 0.5)
					plt.plot(t - 15, np.mean(wf_day1[:, ::10], axis = 0) + np.std(wf_day1[:, ::10], axis = 0), linewidth = 2.0, color = 'black', alpha = 0.5)
					plt.xlabel('Time (samples (30 per ms))', fontsize = 35)
					plt.ylabel('Voltage (microvolts)', fontsize = 35)
					plt.ylim([np.min(np.concatenate((wf_day1, wf_day2), axis = 0)) - 20, np.max(np.concatenate((wf_day1, wf_day2), axis = 0)) + 20])
					plt.title('Unit %i, total waveforms = %i, Electrode: %i, J3: %f' % (unit1, wf_day1.shape[0], hf51.root.unit_descriptor[unit1]['electrode_number'], inter_J3[-1]) + '\n' + 'Single Unit: %i, RSU: %i, FS: %i' % (hf51.root.unit_descriptor[unit1]['single_unit'], hf51.root.unit_descriptor[unit1]['regular_spiking'], hf51.root.unit_descriptor[unit1]['fast_spiking']), fontsize = 20)
					plt.tick_params(axis='both', which='major', labelsize=32)

					plt.subplot(122)
					t = np.arange(wf_day2.shape[1]/10)
					#plt.plot(t - 15, wf_day2[:, ::10].T, linewidth = 0.01, color = 'red')
					plt.plot(t - 15, np.mean(wf_day2[:, ::10], axis = 0), linewidth = 5.0, color = 'black')
					plt.plot(t - 15, np.mean(wf_day2[:, ::10], axis = 0) - np.std(wf_day2[:, ::10], axis = 0), linewidth = 2.0, color = 'black', alpha = 0.5)
					plt.plot(t - 15, np.mean(wf_day2[:, ::10], axis = 0) + np.std(wf_day2[:, ::10], axis = 0), linewidth = 2.0, color = 'black', alpha = 0.5)
					plt.xlabel('Time (samples (30 per ms))', fontsize = 35)
					plt.ylabel('Voltage (microvolts)', fontsize = 35)
					plt.ylim([np.min(np.concatenate((wf_day1, wf_day2), axis = 0)) - 20, np.max(np.concatenate((wf_day1, wf_day2), axis = 0)) + 20])
					plt.title('Unit %i, total waveforms = %i, Electrode: %i' % (unit2, wf_day2.shape[0], hf52.root.unit_descriptor[unit2]['electrode_number']) + '\n' + 'Single Unit: %i, RSU: %i, FS: %i' % (hf52.root.unit_descriptor[unit2]['single_unit'], hf52.root.unit_descriptor[unit2]['regular_spiking'], hf52.root.unit_descriptor[unit2]['fast_spiking']), fontsize = 20)
					plt.tick_params(axis='both', which='major', labelsize=32)
					fig.savefig('Unit%i_and_Unit%i.png' % (unit1, unit2), bbox_inches = 'tight')
					plt.close('all')

# Plot the intra and inter J3 in a different file
fig = plt.figure()
plt.hist(inter_J3, bins = 20, alpha = 0.3, label = 'Across-session J3')
plt.hist(intra_J3, bins = 20, alpha = 0.3, label = 'Within-session J3')
# Draw a vertical line at the percentile criterion used to choose held units
plt.axvline(np.percentile(intra_J3, percent_criterion), linewidth = 5.0, color = 'black', linestyle = 'dashed')
plt.xlabel('J3', fontsize = 35)
plt.ylabel('Number of single unit pairs', fontsize = 35)
plt.tick_params(axis='both', which='major', labelsize=32)
fig.savefig('J3_distributions.png', bbox_inches = 'tight')
plt.close('all')

# Close the hdf5 files and the file with the held units
hf51.close()
hf52.close()
f.close()


'''
intra_J3 = []					
for unit1 in range(len(hf51.root.unit_descriptor[:])):
	if hf51.root.unit_descriptor[unit1]['single_unit'] == 1:
		exec("wf_day1 = hf51.root.sorted_units.unit%03d.waveforms[:]" % (unit1 + 1))
		
		
		#for run in range(num_random_runs):
		# Do it for day 1
		x = np.arange(wf_day1.shape[0])
		#np.random.shuffle(x)
		#print wf_day1[x[:wf_day1.shape[0]/2], :].shape, len(x)
		intra_J3.append(calculate_J3(wf_day1[x[:int(wf_day1.shape[0]*0.5)], :], wf_day1[x[int(wf_day1.shape[0]*0.5):], :]))

intra_J3 = []					
for unit1 in range(len(hf52.root.unit_descriptor[:])):
	J3 = []
	if hf52.root.unit_descriptor[unit1]['single_unit'] == 1:
		exec("wf_day1 = hf52.root.sorted_units.unit%03d.waveforms[:]" % (unit1 + 1))
		
		
		for run in range(num_random_runs):
		# Do it for day 1
			x = np.arange(wf_day1.shape[0])
			np.random.shuffle(x)
		#print wf_day1[x[:wf_day1.shape[0]/2], :].shape, len(x)
			J3.append(calculate_J3(wf_day1[x[:int(wf_day1.shape[0]*0.33)], :], wf_day1[x[int(wf_day1.shape[0]*0.67):], :]))	

		x = np.arange(wf_day1.shape[0])
		J3.append(calculate_J3(wf_day1[x[:int(wf_day1.shape[0]*0.33)], :], wf_day1[x[int(wf_day1.shape[0]*0.67):], :]))

		intra_J3.append(J3)			
'''
'''
				# Divide the waveforms for unit1 and unit2 in equal splits - do this randomly num_random_runs times
				intra_J3 = []
				for run in range(num_random_runs):
					# Do it for day 1
					x = np.arange(wf_day1.shape[0])
					np.random.shuffle(x)
					print wf_day1[x[:wf_day1.shape[0]/2], :].shape
					intra_J3.append(calculate_J3(pca_wf_day1[x[:wf_day1.shape[0]/2], :], pca_wf_day1[x[wf_day1.shape[0]/2:], :]))
					# and for day 2
					x = np.arange(wf_day2.shape[0])
					np.random.shuffle(x)
					intra_J3.append(calculate_J3(pca_wf_day2[x[:wf_day2.shape[0]/2], :], pca_wf_day2[x[wf_day2.shape[0]/2:], :]))

In [74]: plt.plot(np.arange(45), np.mean(data, axis = 0), linewidth = 5.0, color = 'black')
Out[74]: [<matplotlib.lines.Line2D at 0x7ff7312f8050>]

In [75]: plt.plot(np.arange(45), np.mean(data, axis = 0) + np.std(data, axis = 0), linewidth = 2.0, color = 'black', alpha = 0.3)
Out[75]: [<matplotlib.lines.Line2D at 0x7ff7312f8b50>]

In [76]: plt.plot(np.arange(45), np.mean(data, axis = 0) - np.std(data, axis = 0), linewidth = 2.0, color = 'black', alpha = 0.3)
Out[76]: [<matplotlib.lines.Line2D at 0x7ff731304190>]

In [77]: plt.show()

'''


