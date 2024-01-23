############################################################
# Imports and Settings
############################################################
import os
import tables
import numpy as np
import pylab as plt
from sklearn.mixture import GaussianMixture
import argparse
import pandas as pd
import matplotlib
from glob import glob
import re
from scipy.stats import chisquare

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

# Extract parameters for automatic processing
params_dict = metadata_handler.params_dict
auto_params = params_dict['clustering_params']['auto_params']
auto_cluster = auto_params['auto_cluster']
if auto_cluster:
    max_autosort_clusters = auto_params['max_autosort_clusters']
auto_post_process = auto_params['auto_post_process']
count_threshold = auto_params['cluster_count_threshold']
chi_square_alpha = auto_params['chi_square_alpha'] 

dir_name = metadata_handler.dir_name
os.chdir(dir_name)
file_list = metadata_handler.file_list
hdf5_name = metadata_handler.hdf5_name

# Delete the raw node, if it exists in the hdf5 file, to cut down on file size
repacked_bool = post_utils.delete_raw_recordings(hdf5_name)

# Open the hdf5 file
if repacked_bool:
    hdf5_name = hdf5_name[:-3] + '_repacked.h5'
hf5 = tables.open_file(hdf5_name, 'r+')

##############################
# Instantiate unit_descriptor_handler
this_descriptor_handler = post_utils.unit_descriptor_handler(hf5, dir_name)

# Clean up the memory monitor files, pass if clean up has been done already
post_utils.clean_memory_monitor_data()  


# Make the sorted_units group in the hdf5 file if it doesn't already exist
if not '/sorted_units' in hf5:
    hf5.create_group('/', 'sorted_units')

############################################################
# Main Processing Loop 
############################################################
# Run an infinite loop as long as the user wants to 
# pick clusters from the electrodes   

# This section will run if not auto_post_process
while not auto_post_process:

    ############################################################
    # Get unit details and load data
    ############################################################

    print()
    print('======================================')
    print()

    # If sort_file given, iterate through that, otherwise ask user
    continue_bool, electrode_num, num_clusters, clusters = \
            post_utils.get_electrode_details(this_sort_file_handler)

    # For all other continue_bools, if false, end iteration
    # That will return them to this one
    # At that point, if continue_bool is False, exit
    if not continue_bool: break

    # Print out selections
    print('||| Electrode {}, Solution {}, Cluster {} |||'.\
            format(electrode_num, num_clusters, clusters))


    # Load data from the chosen electrode and solution
    (
        spike_waveforms,
        spike_times,
        pca_slices,
        energy,
        amplitudes,
        predictions,
    ) = post_utils.load_data_from_disk(electrode_num, num_clusters)

    # Re-show images of neurons so dumb people like Abu can make sure they
    # picked the right ones
    #if ast.literal_eval(args.show_plot):
    if args.show_plot == 'True':
        post_utils.gen_select_cluster_plot(electrode_num, num_clusters, clusters)

    ############################################################
    # Get unit details and load data
    ############################################################

    this_split_merge_signal = post_utils.split_merge_signal(
            clusters, 
            this_sort_file_handler,)
    split_or_merge = np.logical_or(this_split_merge_signal.split,
                                   this_split_merge_signal.merge)

    # If the user asked to split/re-cluster, 
    # ask them for the clustering parameters and perform clustering
    if this_split_merge_signal.split: 
        ##############################
        ## Split sequence
        ##############################
        # Get clustering parameters from user
        continue_bool, n_clusters, n_iter, thresh, n_restarts = \
                post_utils.get_clustering_params()
        if not continue_bool: continue

        # Make data array to be put through the GMM - 5 components: 
        # 3 PCs, scaled energy, amplitude
        # Clusters is a list, and for len(clusters) == 1,
        # the code below will always work
        this_cluster_inds = np.where(predictions == int(clusters[0]))[0]
        this_cluster_waveforms = spike_waveforms[this_cluster_inds]
        this_cluster_times = spike_times[this_cluster_inds]

        data = post_utils.prepare_data(
                            this_cluster_inds,
                            pca_slices,
                            energy,
                            amplitudes,
                            )

        # Cluster the data
        g = GaussianMixture(
                n_components = n_clusters, 
                covariance_type = 'full', 
                tol = thresh, 
                max_iter = n_iter, 
                n_init = n_restarts)
        g.fit(data)
    
        # Show the cluster plots if the solution converged
        if g.converged_:
            split_predictions = g.predict(data)
            post_utils.generate_cluster_plots(
                            split_predictions, 
                            spike_waveforms, 
                            spike_times, 
                            n_clusters, 
                            this_cluster_inds)
        else:
            split_predictions = []
            print("Solution did not converge "\
                    "- try again with higher number of iterations "\
                    "or lower convergence criterion")
            continue


        # Ask the user for the split clusters they want to choose
        continue_bool, chosen_split = \
                post_utils.get_split_cluster_choice(n_clusters)
        if not continue_bool: continue

        # Once selections have been made, save data
        # Waveforms of originally chosen cluster
        subcluster_inds = [np.where(split_predictions == this_split)[0] \
                            for this_split in chosen_split]
        subcluster_waveforms = [this_cluster_waveforms[this_inds] \
                for this_inds in subcluster_inds]
        fin_inds = np.concatenate(subcluster_inds)


        ############################################################ 
        # Subsetting this set of waveforms to include only the chosen split
        unit_waveforms = this_cluster_waveforms[fin_inds]

        # Do the same thing for the spike times
        unit_times = this_cluster_times[fin_inds] 
        ############################################################ 


        # Plot selected clusters again after merging splits
        post_utils.generate_datashader_plot(
                unit_waveforms, 
                unit_times,
                title = 'Merged Splits',
                )
        # Generate plot showing merged units in different colors
        post_utils.plot_merged_units(
                subcluster_waveforms,
                chosen_split,
                max_n_per_cluster = 1000,
                sd_bound = 1,
                )
        plt.show()

    ##################################################

    # If only 1 cluster was chosen (and it wasn't split), 
    # add that as a new unit in /sorted_units. 
    # Ask if the isolated unit is an almost-SURE single unit
    elif not split_or_merge:
        ##############################
        ## Single cluster selected 
        ##############################
        fin_inds = np.where(predictions == int(clusters[0]))[0]

        unit_waveforms = spike_waveforms[fin_inds, :]
        unit_times = spike_times[fin_inds]


    elif this_split_merge_signal.merge: 
        ##############################
        ## Merge Sequence 
        ##############################
        # If the chosen units are going to be merged, merge them
        cluster_inds = [np.where(predictions == int(this_cluster))[0] \
                for this_cluster in clusters]

        cluster_waveforms = [spike_waveforms[cluster, :] \
                for cluster in cluster_inds]

        fin_inds = np.concatenate(cluster_inds)

        unit_waveforms = spike_waveforms[fin_inds, :] 
        unit_times = spike_times[fin_inds]

        # Generate plot for merged unit
        violations1, violations2,_,_ = post_utils.generate_datashader_plot(
                unit_waveforms, 
                unit_times,
                title = 'Merged Unit',
                )

        # Generate plot showing merged units in different colors
        post_utils.plot_merged_units(
                cluster_waveforms,
                clusters,
                max_n_per_cluster = 1000,
                sd_bound = 1,
                )

        plt.show()

        # Warn the user about the frequency of ISI violations 
        # in the merged unit
        continue_bool, proceed = \
                    post_utils.generate_violations_warning(
                            violations1,
                            violations2,
                            unit_times,
                            )
        if not continue_bool: continue

        # Create unit if the user agrees to proceed, 
        # else abort and go back to start of the loop 
        if not proceed:     
            continue


    ############################################################  
    # Finally, save the unit to the HDF5 file
    ############################################################  

    continue_bool, unit_name = this_descriptor_handler.save_unit(
            unit_waveforms,
            unit_times,
            electrode_num,
            this_sort_file_handler,
            split_or_merge,
            )

    if continue_bool and (this_sort_file_handler.sort_table is not None):
        this_sort_file_handler.mark_current_unit_saved()

    hf5.flush()


    print('==== {} Complete ===\n'.format(unit_name))
    print('==== Iteration Ended ===\n')

# Run auto-processing only if clustering was ALSO automatic
# As currently, this does not have functionality to determine
# correct number of clusters
if auto_post_process and auto_cluster:
    print('==== Auto Post-Processing ====\n')

    # Since this needs classifier output to run, check if it exists
    clf_list = glob('./spike_waveforms/electrode*/clf_prob.npy')
    if len(clf_list) == 0:
        print()
        print('======================================')
        print('Classifier output not found, please run blech_run_process.sh with classifier.')
        print('======================================')
        print()
        exit()

    electrode_list = os.listdir('./spike_waveforms/')
    electrode_num_list = [int(re.findall(r'\d+', this_electrode)[0]) \
            for this_electrode in electrode_list]
    electrode_num_list.sort()

    for electrode_num in electrode_num_list:
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

        # Load data from the chosen electrode 
        # We can pick any soluation, but need to know what
        # solutions are present

        (
            spike_waveforms,
            spike_times,
            pca_slices,
            energy,
            amplitudes,
            split_predictions,
        ) = post_utils.load_data_from_disk(electrode_num, max_autosort_clusters) 

        clf_data_paths = [
                f'./spike_waveforms/electrode{electrode_num:02}/clf_prob.npy',
                f'./spike_waveforms/electrode{electrode_num:02}/clf_pred.npy',
                ]
        clf_prob, clf_pred = [np.load(this_path) for this_path in clf_data_paths]

        # If auto-clustering was done, data has already been trimmed
        # Only clf_pred needs to be trimmed
        clf_prob = clf_prob[clf_pred]
        clf_pred = clf_pred[clf_pred]

        # Take everything
        data = post_utils.prepare_data(
                np.arange(len(spike_waveforms)),
                pca_slices,
                energy,
                amplitudes,
                )

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

        chi_bool = [this_chi[1] < chi_square_alpha for this_chi in chi_out]
        count_bool = [len(this_waveform) > count_threshold for this_waveform in subcluster_waveforms]
        fin_bool = np.logical_and(chi_bool, count_bool)

        ##############################
        # Generate plots for each subcluster
        ##############################

        autosort_output_dir = os.path.join(
            metadata_handler.dir_name,
            'autosort_outputs'
        )

        post_utils.gen_autosort_plot(
            subcluster_prob,
            subcluster_waveforms,
            chi_out,
            mean_waveforms,
            std_waveforms,
            subcluster_times,
            fin_bool,
            electrode_num,
            params_dict['sampling_rate'],
            autosort_output_dir,
            n_max_plot=5000,
        )

        ############################################################
        # Finally, save the unit to the HDF5 file
        ############################################################  

        ############################################################ 
        # Subsetting this set of waveforms to include only the chosen split

        for this_sub in range(len(subcluster_waveforms)):
            if fin_bool[this_sub]:
                continue_bool, unit_name = this_descriptor_handler.save_unit(
                        subcluster_waveforms[this_sub],
                        subcluster_times[this_sub],
                        electrode_num,
                        this_sort_file_handler,
                        split_or_merge = None,
                        override_ask = True,
                        )

        if continue_bool and (this_sort_file_handler.sort_table is not None):
            this_sort_file_handler.mark_current_unit_saved()

        hf5.flush()

    print('==== Auto Post-Processing Complete ====\n')
    print('==== Post-Processing Exiting ====\n')

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
