"""
blech_common_avg_reference.py - Common Average Reference (CAR) processing for neural recordings

This script performs common average referencing on raw electrode data to reduce noise and artifacts.
It processes electrode recordings by:

1. Data Organization:
   - Groups electrodes based on anatomical regions and recording ports
   - Excludes EMG channels and electrodes marked as 'none' from CAR processing
   - Handles multiple CAR groups independently

2. Reference Calculation:
   - Calculates common average reference for each electrode group
   - Averages voltage values across all electrodes in a group
   - Processes groups sequentially to optimize memory usage

3. Signal Processing:
   - Subtracts appropriate common average reference from each electrode
   - Updates electrode data in-place in the HDF5 file
   - Maintains data integrity through careful memory management

Usage:
    python blech_common_avg_reference.py <dir_name>

Arguments:
    dir_name : Directory containing the HDF5 file with raw electrode data

Dependencies:
    - numpy, tables, tqdm
    - Custom utility modules from blech_clust package

Notes:
    - Requires properly formatted electrode layout information in the experiment info file
    - CAR groups are defined by the 'CAR_group' column in the electrode layout
    - EMG channels and electrodes marked as 'none' are automatically excluded

Author: Abuzar Mahmood
"""

# Import stuff!
import tables
import numpy as np
import os
import easygui
import sys
from tqdm import tqdm
import glob
import json
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utils.blech_utils import imp_metadata, pipeline_graph_check
from utils.qa_utils import channel_corr


def get_electrode_by_name(raw_electrodes, name):
    """
    Get the electrode data from the list of raw electrodes
    by the name of the electrode
    """
    str_name = f"electrode{name:02}"
    wanted_electrode_ind = [
        x for x in raw_electrodes if str_name in x._v_pathname][0]
    return wanted_electrode_ind


# def extract_electrode_features(raw_electrodes, electrode_indices, n_samples=10000):
#     """
#     Extract features from electrode data for clustering using correlation matrix and PCA
#
#     Parameters:
#     -----------
#     raw_electrodes : list
#         List of electrode nodes from HDF5 file
#     electrode_indices : list
#         List of electrode indices to extract features from
#     n_samples : int
#         Number of samples to use for feature extraction
#
#     Returns:
#     --------
#     features : numpy.ndarray
#         Array of features for each electrode after PCA
#     """
#     print("Extracting features from electrodes using correlation matrix and PCA...")
#
#     # Initialize data array to store electrode signals
#     n_electrodes = len(electrode_indices)
#     electrode_data = np.zeros((n_electrodes, n_samples))
#     data_len = len(get_electrode_by_name(
#         raw_electrodes, electrode_indices[0])[:])
#     step_size = data_len // n_samples
#     sample_indices = np.arange(0, data_len, step_size)[:n_samples]
#
#     for i, electrode_idx in enumerate(tqdm(electrode_indices)):
#         # Get electrode data
#         full_data = get_electrode_by_name(
#             raw_electrodes, electrode_idx)[:]
#
#         # Subsample if needed
#         if len(full_data) > n_samples:
#             electrode_data[i, :] = full_data[sample_indices]
#         else:
#             # If data is shorter than n_samples, pad with zeros
#             electrode_data[i, :len(full_data)] = full_data[:n_samples]
#
#     # Calculate correlation matrix between electrodes
#     corr_matrix = channel_corr.intra_corr(electrode_data)
#     # Make nan values 0
#     corr_matrix[np.isnan(corr_matrix)] = 0
#     # Make sure the correlation matrix is symmetric
#     corr_matrix = corr_matrix + corr_matrix.T
#
#     # Apply PCA to the correlation matrix to retain 95% of variance
#     pca = PCA(n_components=0.95)
#     features = pca.fit_transform(corr_matrix)
#
#     print(
#         f"PCA reduced features from {corr_matrix.shape[1]} to {features.shape[1]} dimensions (95% variance retained)")
#
#     return features

def get_channel_corr_mat(data_dir):
    qa_out_path = os.path.join(data_dir, 'QA_output')
    return np.load(os.path.join(qa_out_path, 'channel_corr_mat.npy'))


def cluster_electrodes(features, n_components=10, n_iter=100, threshold=1e-3):
    """
    Cluster electrodes using Bayesian Gaussian Mixture model

    Parameters:
    -----------
    features : numpy.ndarray
        Array of features for each electrode (from PCA on correlation matrix)
    n_components : int
        Maximum number of components for the Bayesian Gaussian Mixture model
    n_iter : int
        Number of iterations for the model
    threshold : float
        Convergence threshold for the model

    Returns:
    --------
    predictions : numpy.ndarray
        Array of cluster assignments for each electrode
    model : BayesianGaussianMixture
        Fitted model
    reduced_features : numpy.ndarray
        Features reduced to 2D for visualization
    """
    print("Clustering electrodes...")

    # No need to standardize features as they're already from PCA
    # But we'll reduce to 2D for visualization if dimensions are higher
    if features.shape[1] > 2:
        pca_viz = PCA(n_components=2)
        reduced_features = pca_viz.fit_transform(features)
    else:
        reduced_features = features.copy()

    # Fit Bayesian Gaussian Mixture model on the full feature set
    model = BayesianGaussianMixture(
        n_components=n_components,
        covariance_type='full',
        max_iter=n_iter,
        tol=threshold,
        random_state=42,
        weight_concentration_prior_type='dirichlet_process',
        weight_concentration_prior=1e-2
    )
    model.fit(features)

    # Get cluster assignments
    predictions = model.predict(features)

    return predictions, model, reduced_features


def plot_clustered_corr_mat(
    corr_matrix, predictions, electrode_names, plot_path,
    cmap='jet'
):
    """
    Plot clustered correlation matrix for electrodes

    Parameters:
    -----------
    corr_matrix : numpy.ndarray
    predictions : numpy.ndarray
    electrode_indices : list
    cluster_indices : list
    plot_path : str
    """
    # Sort electrodes by cluster assignment
    sorted_indices = np.argsort(predictions)
    sorted_corr_matrix = corr_matrix[sorted_indices, :][:, sorted_indices]
    sorted_names = [electrode_names[i] for i in sorted_indices]

    # Plot clustered correlation matrix
    fig, ax = plt.subplots(1, 3, figsize=(20, 10),
                           # gridspec_kw={'width_ratios': [1, 1, 0.1]}
                           )
    ax[0].imshow(corr_matrix, cmap=cmap)
    ax[0].set_title('Original Correlation Matrix')
    ax[0].set_xticks(np.arange(len(electrode_names)))
    ax[0].set_yticks(np.arange(len(electrode_names)))
    ax[0].set_xticklabels(electrode_names, rotation=90)
    ax[0].set_yticklabels(electrode_names)
    ax[1].imshow(sorted_corr_matrix, cmap=cmap)
    ax[1].set_title('Clustered Correlation Matrix')
    ax[1].set_xticks(np.arange(len(sorted_names)))
    ax[1].set_yticks(np.arange(len(sorted_names)))
    ax[1].set_xticklabels(sorted_names, rotation=90)
    ax[1].set_yticklabels(sorted_names)
    # Sharey between ax[1] and ax[2]
    # Make a cluster matrix from sorted predictions
    sorted_predictions = predictions[sorted_indices]+1
    cluster_matrix = np.expand_dims(
        sorted_predictions, axis=1) == np.expand_dims(sorted_predictions, axis=0)
    # Mark unique clusters
    cluster_matrix = cluster_matrix * \
        np.expand_dims(sorted_predictions, axis=1)

    ax[2].imshow(cluster_matrix, cmap='tab10')
    ax[2].set_title('Cluster Assignments')
    ax[2].set_xticks(np.arange(len(sorted_names)))
    ax[2].set_yticks(np.arange(len(sorted_names)))
    ax[2].set_xticklabels(sorted_names, rotation=90)
    ax[2].set_yticklabels(sorted_names)
    # ax[2].imshow(np.expand_dims(predictions[sorted_indices], axis=1), cmap='tab10',
    #              aspect='auto', interpolation='nearest')
    # ax[2].sharey(ax[1])
    # # ax[2] height doesn't match ax[1] height, so adjust it
    # ax[2].set_ylim(ax[1].get_ylim())

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

############################################################
############################################################


testing_bool = True

if not testing_bool:
    # Get name of directory with the data files
    metadata_handler = imp_metadata(sys.argv)
    this_pipeline_check = pipeline_graph_check(dir_name)
    this_pipeline_check.check_previous(script_path)
    this_pipeline_check.write_to_log(script_path, 'attempted')
    # Perform pipeline graph check
    script_path = os.path.realpath(__file__)
else:
    data_dir = '/media/storage/for_transfer/bla_gc/AM35_4Tastes_201228_124547'
    metadata_handler = imp_metadata([[], data_dir])

dir_name = metadata_handler.dir_name

os.chdir(dir_name)
print(f'Processing : {dir_name}')

# Open the hdf5 file
hf5 = tables.open_file(metadata_handler.hdf5_name, 'r+')

# Read CAR groups from info file
# Every region is a separate group, multiple ports under single region is a separate group,
# emg is a separate group
info_dict = metadata_handler.info_dict
electrode_layout_frame = metadata_handler.layout
# Remove emg and none channels from the electrode layout frame
emg_bool = ~electrode_layout_frame.CAR_group.str.contains('emg')
none_bool = ~electrode_layout_frame.CAR_group.str.contains('none')
fin_bool = np.logical_and(emg_bool, none_bool)
electrode_layout_frame = electrode_layout_frame[fin_bool]
electrode_layout_frame['channel_name'] = \
    electrode_layout_frame['port'].astype(str) + '_' + \
    electrode_layout_frame['electrode_num'].astype(str)

# # Since electrodes are already in monotonic numbers (A : 0-31, B: 32-63)
# # we can directly pull them
# grouped_layout = list(electrode_layout_frame.groupby('CAR_group'))
# all_car_group_names = [x[0] for x in grouped_layout]
# # Note: electrodes in HDF5 are also saved according to inds
# # specified in the layout file
# all_car_group_vals = [x[1].electrode_ind.values for x in grouped_layout]

# CAR_electrodes = all_car_group_vals
num_groups = electrode_layout_frame.CAR_group.nunique()
print(f" Number of groups : {num_groups}")
# for region, vals in zip(all_car_group_names, all_car_group_vals):
#     print(f" {region} :: {vals}")
for group_num, group_name in enumerate(electrode_layout_frame.CAR_group.unique()):
    group_channel_names = electrode_layout_frame[electrode_layout_frame.CAR_group ==
                                                 group_name].channel_name.values
    print(f" {group_name} :: {len(group_channel_names)} channels :: \n{group_channel_names}")
    print()

# Pull out the raw electrode nodes of the HDF5 file
raw_electrodes = hf5.list_nodes('/raw')

# Check if auto_CAR parameters are enabled in the parameters
if hasattr(metadata_handler, 'params_dict') and metadata_handler.params_dict:
    auto_car_section = metadata_handler.params_dict.get('auto_CAR', {})
    auto_car_inference = auto_car_section.get('use_auto_CAR')
    max_clusters = auto_car_section.get('max_clusters')

# If auto_car_inference is enabled, perform clustering on each CAR group
if auto_car_inference:
    print("\nPerforming automatic CAR group inference...")

    # Create a directory for cluster plots if it doesn't exist
    # Extract features from electrodes
    # features = extract_electrode_features(
    #     raw_electrodes, electrode_indices)
    corr_mat = get_channel_corr_mat(dir_name)
    # Convert nan to 0
    corr_mat[np.isnan(corr_mat)] = 0
    # Make symmetric
    corr_mat = corr_mat + corr_mat.T

    # Perform PCA
    pca = PCA(n_components=5)
    features = pca.fit_transform(corr_mat)

    # Cluster electrodes
    predictions, model, reduced_features = cluster_electrodes(
        features,
        n_components=min(max_clusters, len(electrode_indices)),
        n_iter=100,
        threshold=1e-3
    )

    print(f"Found {len(np.unique(predictions))} clusters')

    electrode_layout_frame['predicted_clusters'] = predictions

    # Plot clusters
    plot_path = os.path.join(dir_name, 'QA_output', 'clustered_corr_mat.png')
    plot_clustered_corr_mat(
        corr_mat, predictions, electrode_layout_frame.channel_name.values, plot_path
    )

    # Process each CAR group
    for group_num, group_name in enumerate(electrode_layout_frame.CAR_group.unique()):
        print(f"\nProcessing group {group_name} for auto-CAR inference")

        this_car_frame = electrode_layout_frame[electrode_layout_frame.CAR_group == group_name]

        # Get electrode indices for this group
        # electrode_indices = CAR_electrodes[group_num]
        electrode_indices = this_car_frame.electrode_ind.values

        # Skip if there are too few electrodes
        if len(electrode_indices) < 2:
            print(
                f"Group {group_name} has fewer than 3 electrodes. Skipping clustering.")
            continue

        # Count number of electrodes in each cluster
        unique_clusters = np.unique(predictions)

        for cluster in unique_clusters:
            cluster_electrodes = [electrode_indices[i] for i in range(len(electrode_indices))
                                  if predictions[i] == cluster]
            print(
                f"  Cluster {cluster}: {len(cluster_electrodes)} electrodes - {cluster_electrodes}")

        # Save cluster assignments to a JSON file
        cluster_info = {
            'group_name': group_name,
            'clusters': {}
        }

        for cluster in unique_clusters:
            cluster_electrodes = [int(electrode_indices[i]) for i in range(len(electrode_indices))
                                  if predictions[i] == cluster]
            cluster_info['clusters'][f'cluster_{cluster}'] = cluster_electrodes

        cluster_file = os.path.join(plots_dir, f'{group_name}_clusters.json')
        with open(cluster_file, 'w') as f:
            json.dump(cluster_info, f, indent=4)

        print(f"Cluster information saved to: {cluster_file}")


# First get the common average references by averaging across
# the electrodes picked for each group
print(
    "Calculating common average reference for {:d} groups".format(num_groups))
common_average_reference = np.zeros(
    (num_groups, raw_electrodes[0][:].shape[0]))
print('Calculating mean values')
for group_num, group_name in tqdm(enumerate(all_car_group_names)):
    print(f"Processing group {group_name}")
    # First add up the voltage values from each electrode to the same array
    # then divide by number of electrodes to get the average
    # This is more memory efficient than loading all the electrode data into
    # a single array and then averaging
    for electrode_name in tqdm(CAR_electrodes[group_num]):
        common_average_reference[group_num, :] += \
            get_electrode_by_name(raw_electrodes, electrode_name)[:]
    common_average_reference[group_num,
                             :] /= float(len(CAR_electrodes[group_num]))

print("Common average reference for {:d} groups calculated".format(num_groups))

# Now run through the raw electrode data and
# subtract the common average reference from each of them
print('Performing background subtraction')
for group_num, group_name in tqdm(enumerate(all_car_group_names)):
    print(f"Processing group {group_name}")
    for electrode_num in tqdm(all_car_group_vals[group_num]):
        # Subtract the common average reference for that group from the
        # voltage data of the electrode
        wanted_electrode = get_electrode_by_name(raw_electrodes, electrode_num)
        referenced_data = wanted_electrode[:] - \
            common_average_reference[group_num]
        # Overwrite the electrode data with the referenced data
        wanted_electrode[:] = referenced_data
        hf5.flush()
        del referenced_data


hf5.close()
print("Modified electrode arrays written to HDF5 file after "
      "subtracting the common average reference")

# Write successful execution to log
this_pipeline_check.write_to_log(script_path, 'completed')
