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


def get_electrode_by_name(raw_electrodes, name):
    """
    Get the electrode data from the list of raw electrodes
    by the name of the electrode
    """
    str_name = f"electrode{name:02}"
    wanted_electrode_ind = [
        x for x in raw_electrodes if str_name in x._v_pathname][0]
    return wanted_electrode_ind


def extract_electrode_features(raw_electrodes, electrode_indices, n_samples=10000):
    """
    Extract features from electrode data for clustering using correlation matrix and PCA

    Parameters:
    -----------
    raw_electrodes : list
        List of electrode nodes from HDF5 file
    electrode_indices : list
        List of electrode indices to extract features from
    n_samples : int
        Number of samples to use for feature extraction

    Returns:
    --------
    features : numpy.ndarray
        Array of features for each electrode after PCA
    """
    print("Extracting features from electrodes using correlation matrix and PCA...")

    # Initialize data array to store electrode signals
    n_electrodes = len(electrode_indices)
    electrode_data = np.zeros((n_electrodes, n_samples))

    for i, electrode_idx in enumerate(tqdm(electrode_indices)):
        # Get electrode data
        full_data = get_electrode_by_name(
            raw_electrodes, electrode_idx)[:]

        # Subsample if needed
        if len(full_data) > n_samples:
            indices = np.random.choice(
                len(full_data), n_samples, replace=False)
            electrode_data[i, :] = full_data[indices]
        else:
            # If data is shorter than n_samples, pad with zeros
            electrode_data[i, :len(full_data)] = full_data[:n_samples]

    # Calculate correlation matrix between electrodes
    corr_matrix = np.corrcoef(electrode_data)
    
    # Apply PCA to the correlation matrix to retain 95% of variance
    pca = PCA(n_components=0.95)
    features = pca.fit_transform(corr_matrix)
    
    print(f"PCA reduced features from {corr_matrix.shape[1]} to {features.shape[1]} dimensions (95% variance retained)")
    
    return features


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


def plot_electrode_clusters(features, predictions, electrode_indices, output_dir):
    """
    Plot electrode clusters and correlation matrix

    Parameters:
    -----------
    features : numpy.ndarray
        Array of features for each electrode (reduced to 2D)
    predictions : numpy.ndarray
        Array of cluster assignments for each electrode
    electrode_indices : list
        List of electrode indices
    output_dir : str
        Directory to save the plot
    """
    print("Plotting electrode clusters and correlation matrix...")

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create figure for cluster plot
    plt.figure(figsize=(12, 10))

    # Plot clusters
    scatter = plt.scatter(features[:, 0], features[:, 1], c=predictions,
                          cmap='viridis', s=100, alpha=0.8)

    # Add electrode labels
    for i, electrode_idx in enumerate(electrode_indices):
        plt.annotate(str(electrode_idx), (features[i, 0], features[i, 1]),
                     fontsize=8, ha='center', va='center')

    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Cluster')

    # Set labels and title
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Electrode Clusters using Bayesian Gaussian Mixture Model')

    # Save figure
    plt.tight_layout()
    cluster_plot_path = os.path.join(output_dir, 'electrode_clusters.png')
    plt.savefig(cluster_plot_path)
    plt.close()

    # Plot correlation matrix with cluster assignments
    plt.figure(figsize=(14, 12))
    
    # Sort electrodes by cluster
    sorted_indices = np.argsort(predictions)
    sorted_electrodes = [electrode_indices[i] for i in sorted_indices]
    
    # Create a correlation matrix for visualization (we'll recalculate it here)
    # This is just for visualization purposes
    n_electrodes = len(electrode_indices)
    corr_matrix = np.zeros((n_electrodes, n_electrodes))
    
    # Fill the upper triangle with cluster information
    for i in range(n_electrodes):
        for j in range(i, n_electrodes):
            # 1 if same cluster, 0 if different
            corr_matrix[i, j] = 1 if predictions[i] == predictions[j] else 0
            corr_matrix[j, i] = corr_matrix[i, j]  # Make symmetric
    
    # Plot the correlation matrix
    plt.imshow(corr_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Cluster Similarity')
    
    # Add electrode labels
    plt.xticks(range(n_electrodes), electrode_indices, rotation=90, fontsize=8)
    plt.yticks(range(n_electrodes), electrode_indices, fontsize=8)
    
    plt.title('Electrode Correlation Matrix (Clustered)')
    plt.tight_layout()
    corr_plot_path = os.path.join(output_dir, 'electrode_correlation_matrix.png')
    plt.savefig(corr_plot_path)
    plt.close()

    # Save cluster assignments to a text file
    cluster_file_path = os.path.join(output_dir, 'electrode_clusters.txt')
    with open(cluster_file_path, 'w') as f:
        f.write("Electrode,Cluster\n")
        for i, electrode_idx in enumerate(electrode_indices):
            f.write(f"{electrode_idx},{predictions[i]}\n")

    print(f"Cluster plots saved to: {cluster_plot_path} and {corr_plot_path}")
    print(f"Cluster assignments saved to: {cluster_file_path}")
    
    return cluster_plot_path

############################################################
############################################################


# Get name of directory with the data files
metadata_handler = imp_metadata(sys.argv)
dir_name = metadata_handler.dir_name

# Perform pipeline graph check
script_path = os.path.realpath(__file__)
this_pipeline_check = pipeline_graph_check(dir_name)
this_pipeline_check.check_previous(script_path)
this_pipeline_check.write_to_log(script_path, 'attempted')


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

# Since electrodes are already in monotonic numbers (A : 0-31, B: 32-63)
# we can directly pull them
grouped_layout = list(electrode_layout_frame.groupby('CAR_group'))
all_car_group_names = [x[0] for x in grouped_layout]
# Note: electrodes in HDF5 are also saved according to inds
# specified in the layout file
all_car_group_vals = [x[1].electrode_ind.values for x in grouped_layout]

CAR_electrodes = all_car_group_vals
num_groups = len(CAR_electrodes)
print(f" Number of groups : {num_groups}")
for region, vals in zip(all_car_group_names, all_car_group_vals):
    print(f" {region} :: {vals}")

# Pull out the raw electrode nodes of the HDF5 file
raw_electrodes = hf5.list_nodes('/raw')

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

# Check if auto_car_inference is enabled in the parameters
auto_car_inference = False
if hasattr(metadata_handler, 'params_dict') and metadata_handler.params_dict:
    auto_car_inference = metadata_handler.params_dict.get(
        'auto_car_inference', False)

# If auto_car_inference is enabled, perform clustering on each CAR group
if auto_car_inference:
    print("\nPerforming automatic CAR group inference...")

    # Create a directory for cluster plots if it doesn't exist
    plots_dir = os.path.join(dir_name, 'car_cluster_plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Process each CAR group
    for group_num, group_name in enumerate(all_car_group_names):
        print(f"\nProcessing group {group_name} for auto-CAR inference")

        # Get electrode indices for this group
        electrode_indices = CAR_electrodes[group_num]

        # Skip if there are too few electrodes
        if len(electrode_indices) < 3:
            print(
                f"Group {group_name} has fewer than 3 electrodes. Skipping clustering.")
            continue

        # Extract features from electrodes
        features = extract_electrode_features(
            raw_electrodes, electrode_indices)

        # Cluster electrodes
        predictions, model, reduced_features = cluster_electrodes(
            features,
            n_components=min(10, len(electrode_indices)),
            n_iter=100,
            threshold=1e-3
        )

        # Plot clusters
        plot_path = plot_electrode_clusters(
            reduced_features,
            predictions,
            electrode_indices,
            plots_dir
        )

        # Count number of electrodes in each cluster
        unique_clusters = np.unique(predictions)
        print(f"Found {len(unique_clusters)} clusters in group {group_name}")

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

hf5.close()
print("Modified electrode arrays written to HDF5 file after "
      "subtracting the common average reference")

# Write successful execution to log
this_pipeline_check.write_to_log(script_path, 'completed')
