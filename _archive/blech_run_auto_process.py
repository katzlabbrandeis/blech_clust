"""
Script to automatically post-process blech data.

Autoclustering will be done using Bayesian Gaussian Mixture Models (BGMMs)
from scikit-learn.
"""
############################################################
# Imports and Settings
############################################################
import os
import tables
import numpy as np
import easygui
import ast
import re
import pylab as plt
import matplotlib.image as mpimg
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture as BGM
from sklearn.cluster import KMeans
import argparse
import pandas as pd
import matplotlib
from scipy.stats import chisquare
from utils.blech_process_utils import gen_isi_hist
from glob import glob

matplotlib.rcParams['font.size'] = 6

# Import 3rd party code
from utils import blech_waveforms_datashader
from utils.blech_utils import entry_checker, imp_metadata
import utils.blech_post_process_utils as post_utils

# Set seed to allow inter-run reliability
# Also allows reusing the same sorting sheets across runs
np.random.seed(0)

############################################################
# Input from user and setup data
############################################################
# Get directory where the hdf5 file sits, and change to that directory
# Get name of directory with the data files
# Create argument parser
parser = argparse.ArgumentParser(
        description = 'Spike extraction and sorting script')
parser.add_argument('--dir-name',  '-d',
                    help = 'Directory containing data files')
parser.add_argument('--show-plot', '-p',
        help = 'Show waveforms while iterating (True/False)', default = 'True')
parser.add_argument('--sort-file', '-f', help = 'CSV with sorted units',
                    default = None)
args = parser.parse_args()

##############################
# Instantiate sort_file_handler
this_sort_file_handler = post_utils.sort_file_handler(args.sort_file)

if args.dir_name is not None:
    metadata_handler = imp_metadata([[],args.dir_name])
else:
    metadata_handler = imp_metadata([])

params_dict = metadata_handler.params_dict
sampling_rate = params_dict['sampling_rate']

dir_name = metadata_handler.dir_name
os.chdir(dir_name)
file_list = metadata_handler.file_list
hdf5_name = metadata_handler.hdf5_name

# Since this needs classifier output to run, check if it exists
clf_list = glob('./spike_waveforms/electrode*/clf_prob.npy')
if len(clf_list) == 0:
	print()
	print('======================================')
	print('Classifier output not found, please run blech_run_process.sh with classifier.')
	print('======================================')
	print()
	exit()

# Clean up the memory monitor files, pass if clean up has been done already
post_utils.clean_memory_monitor_data()
# Delete the raw node, if it exists in the hdf5 file, to cut down on file size
repacked_bool = post_utils.delete_raw_recordings(hdf5_name)

# Open the hdf5 file
if repacked_bool:
    hdf5_name = hdf5_name[:-3] + '_repacked.h5'
hf5 = tables.open_file(hdf5_name, 'r+')

##############################
# Instantiate unit_descriptor_handler
this_descriptor_handler = post_utils.unit_descriptor_handler(hf5, dir_name)

# Make the sorted_units group in the hdf5 file if it doesn't already exist
if not '/sorted_units' in hf5:
    hf5.create_group('/', 'sorted_units')

autosort_output_dir = os.path.join(dir_name, 'autosort_output')
if not os.path.isdir(autosort_output_dir):
	os.mkdir(autosort_output_dir)

############################################################
# Main Processing Loop
############################################################
# Run an infinite loop as long as the user wants to
# pick clusters from the electrodes

electrode_list = os.listdir('./spike_waveforms/')
electrode_num_list = [int(re.findall(r'\d+', this_electrode)[0]) \
		for this_electrode in electrode_list]
electrode_num_list.sort()

for electrode_num in electrode_num_list:

	# electrode_output_dir = os.path.join(autosort_output_dir, f'electrode{electrode_num:02}')

	############################################################
	# Get unit details and load data
	############################################################

	print()
	print('======================================')
	print()

	# Iterate over electrodes and pull out spikes
	# Get classifier probabilities for each spike and use only
	# "good" spikes

	# Print out selections
	print(f'=== Processing Electrode {electrode_num:02} ===')


	# Load data from the chosen electrode and solution
	# We can simply load data from 2-clusters because we're not using
	# gmm-predictions
	(
		spike_waveforms,
		spike_times,
		pca_slices,
		energy,
		amplitudes,
		predictions,
		) = post_utils.load_data_from_disk(electrode_num, 2)

	clf_data_paths = [
			f'./spike_waveforms/electrode{electrode_num:02}/clf_prob.npy',
			f'./spike_waveforms/electrode{electrode_num:02}/clf_pred.npy',
			]
	clf_prob, clf_pred = [np.load(this_path) for this_path in clf_data_paths]

	# Toss all waveforms and times that are not predicted
	spike_waveforms = spike_waveforms[clf_pred]
	spike_times = spike_times[clf_pred]
	pca_slices = pca_slices[clf_pred]
	energy = energy[clf_pred]
	amplitudes = amplitudes[clf_pred]
	clf_prob = clf_prob[clf_pred]
	clf_pred = clf_pred[clf_pred]

	# Take everything
	data = post_utils.prepare_data(
			np.arange(len(spike_waveforms)),
			pca_slices,
			energy,
			amplitudes,
			)

	############################################################
	# Perform clustering
	############################################################

	# Since the dirichlet process WILL find more clusters for more data,
	# (whereas we know that more spikes doens't mean more neurons),
	# we need to keep the "COUNT" constant
	# One way to do this is to use KMeans centroids as data points

	max_k = 1000
	n_max_train = 10000
	if len(data) > max_k:
		kmean_obj = KMeans(n_clusters = max_k)
		if len(data) < n_max_train:
			kmean_obj.fit(data)
		else:
			inds = np.random.choice(len(data), n_max_train, replace = False)
			kmean_obj.fit(data[inds])
		train_data = kmean_obj.cluster_centers_
		kmeans_used = True
	else:
		train_data = data
		kmeans_used = False

	max_components = 10
	g = BGM(
			n_components = max_components,
			covariance_type = 'full',
			max_iter = 1000,
			n_init = 10,
			weight_concentration_prior_type = 'dirichlet_process',
			# This can be systematically adjusted to match
			# actual data
			weight_concentration_prior = 0.1,
			verbose = 10,
			)

	#n_max_train = 20000
	#if len(data) < n_max_train:
	#	g.fit(data)
	#else:
	#	inds = np.random.choice(len(data), n_max_train, replace = False)
	#	g.fit(data[inds])

	# Train on kmeans centroids but predict on the actual data
	g.fit(train_data)
	split_predictions = g.predict(data)

	##############################
	# If any cluster has less than threshold weight, drop it
	cluster_weights = g.weights_

	# Plot cluster weights as bars along with threshold
	fig, ax = plt.subplots(figsize = (5, 5))
	ax.bar(np.arange(len(cluster_weights)), cluster_weights)
	#ax.axhline(weight_threshold, color = 'r', label = 'Threshold')
	#ax.legend()
	ax.set_xlabel('Cluster Number')
	ax.set_ylabel('Cluster Weight')
	ax.set_title(f'Electrode {electrode_num:02}')
	ax.set_ylim([0, 1])
	plt.savefig(os.path.join(autosort_output_dir, f'electrode{electrode_num:02}_cluster_weights.png'),
			 bbox_inches = 'tight')
	plt.close()

	##############################

	# Once selections have been made, save data
	# Waveforms of originally chosen cluster
	subcluster_inds = [np.where(split_predictions == this_split)[0] \
			for this_split in np.unique(split_predictions)]
	subcluster_waveforms = [spike_waveforms[this_inds] \
			for this_inds in subcluster_inds]
	subcluster_prob = [clf_prob[this_inds] \
			for this_inds in subcluster_inds]
	subcluster_times = [spike_times[this_inds] \
			for this_inds in subcluster_inds]
	mean_waveforms = [np.mean(this_waveform, axis = 0) for this_waveform in subcluster_waveforms]
	std_waveforms = [np.std(this_waveform, axis = 0) for this_waveform in subcluster_waveforms]

	# Check that the probability distributions are not uniform
	# That indicates that the cluster is likely generated by noise

	prob_dists = [np.histogram(this_prob, bins = 10, density = True)[0]
			   for this_prob in subcluster_prob]
	chi_out = [chisquare(this_dist) for this_dist in prob_dists]
	count_threshold = 2000
	alpha = 0.05

	chi_bool = [this_chi[1] < alpha for this_chi in chi_out]
	count_bool = [len(this_waveform) > count_threshold for this_waveform in subcluster_waveforms]
	fin_bool = np.logical_and(chi_bool, count_bool)

	# Plot all subclusters and their prob distribution
	# Indicate chi-square p-value in titles

	# Also add:
	# 1. ISI distribution
	# 2. Histogram of spikes over time
	# 3. Mean +/- std of waveform
	n_max_plot = 5000
	fig, ax = plt.subplots(5, len(subcluster_waveforms),
						figsize = (5*len(subcluster_prob), 20),
						sharex = False, sharey = False)
	for this_ax, this_waveforms in zip(ax[0], subcluster_waveforms):
		waveform_count = len(this_waveforms)
		if waveform_count > n_max_plot:
			this_waveforms = this_waveforms[np.random.choice(
				len(this_waveforms), n_max_plot, replace = False)]
		this_ax.plot(this_waveforms.T, color = 'k', alpha = 0.01)
		this_ax.set_title('Waveform Count: {}'.format(waveform_count))
	for this_ax, this_dist, this_chi in zip(ax[2], subcluster_prob, chi_out):
		this_ax.hist(this_dist, bins = 10, alpha = 0.5, density = True)
		this_ax.hist(this_dist, bins = 10, alpha = 0.5, density = True,
			   histtype = 'step', color = 'k', linewidth = 3)
		this_ax.set_title('Chi-Square p-value: {:.3f}'.format(this_chi.pvalue))
		this_ax.set_xlabel('Classifier Probability')
		this_ax.set_xlim([0, 1])
	# If chi-square p-value is less than alpha, create green border around subplots
	for i in range(len(subcluster_prob)):
		if fin_bool[i]:
			for this_ax in ax[:,i]:
				for this_spine in this_ax.spines.values():
					this_spine.set_edgecolor('green')
					this_spine.set_linewidth(5)
	ax[0,0].set_ylabel('Waveform Amplitude')
	ax[2,0].set_ylabel('Count')
	for this_ax, this_mean, this_std in zip(ax[1], mean_waveforms, std_waveforms):
		this_ax.plot(this_mean, color = 'k')
		this_ax.fill_between(np.arange(len(this_mean)),
					   y1 = this_mean - this_std,
					   y2 = this_mean + this_std,
					   color = 'k', alpha = 0.5)
		this_ax.set_title('Mean +/- Std')
		this_ax.set_xlabel('Time (samples)')
		this_ax.set_ylabel('Amplitude')
	for this_ax, this_times in zip(ax[3], subcluster_times):
		this_ax.hist(this_times, bins = 30, alpha = 0.5, density = True)
		this_ax.set_title('Spike counts over time')
	for this_ax, this_times in zip(ax[4], subcluster_times):
		fig, this_ax = gen_isi_hist(
				this_times,
				np.arange(len(this_times)),
				sampling_rate,
				ax = this_ax,
				)
		this_ax.hist(np.diff(this_times), bins = 30, alpha = 0.5, density = True)
		#this_ax.set_title('ISI distribution')
	# For first 2 rows, equalize y limits
	lims_list = [this_ax.get_ylim() for this_ax in ax[:2,:].flatten()]
	min_lim = np.min(lims_list)
	max_lim = np.max(lims_list)
	for this_ax in ax[:2,:].flatten():
		this_ax.set_ylim([min_lim, max_lim])
	plt.tight_layout()
	fig.savefig(os.path.join(autosort_output_dir, f'{electrode_num:02}_subclusters.png'))
	plt.close()
	#plt.show()

	############################################################
	# Subsetting this set of waveforms to include only the chosen split

	# for this_sub in range(len(subcluster_waveforms)):
	# 	if fin_bool[this_sub]:
	# 		continue_bool, unit_name = this_descriptor_handler.save_unit(
	# 				subcluster_waveforms[this_sub],
	# 				subcluster_times[this_sub],
	# 				electrode_num,
	# 				this_sort_file_handler,
	# 				split_or_merge = None,
	# 				override_ask = True,
	# 				)

	# hf5.flush()

	print('==== {} Complete ===\n'.format(unit_name))
	print('==== Iteration Ended ===\n')

############################################################
# Final write of unit_descriptor and cleanup
############################################################
# Sort unit_descriptor by unit_number
# This will be needed if sort_table is used, as using sort_table
# will add merge/split marked units first
print()
print('==== Sorting Units and writing Unit Descriptor ====\n')
this_descriptor_handler.write_unit_descriptor_from_sorted_units()
this_descriptor_handler.resort_units()
hf5.flush()

current_unit_table = this_descriptor_handler.table_to_frame()
print()
print('==== Unit Table ====\n')
print(current_unit_table)


print()
print('== Post-processing exiting ==')
# Close the hdf5 file
hf5.close()
