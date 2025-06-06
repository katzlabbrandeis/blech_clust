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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utils.blech_utils import imp_metadata, pipeline_graph_check
from utils.qa_utils import channel_corr
from utils.ephys_data.visualize import gen_square_subplots
import pandas as pd

try:
    from scipy.stats import median_abs_deviation as MAD
    try_again = False
except:
    print('Could not import median_abs_deviation, using deprecated version')
    try_again = True

if try_again:
    try:
        from scipy.stats import median_absolute_deviation as MAD
    except:
        raise ImportError('Could not import median_absolute_deviation')


def get_electrode_by_name(raw_electrodes, name):
    """
    Get the electrode data from the list of raw electrodes
    by the name of the electrode
    """
    str_name = f"electrode{name:02}"
    wanted_electrode_ind = [
        x for x in raw_electrodes if str_name in x._v_pathname][0]
    return wanted_electrode_ind


def get_channel_corr_mat(data_dir):
    qa_out_path = os.path.join(data_dir, 'QA_output')
    return np.load(os.path.join(qa_out_path, 'channel_corr_mat.npy'))


def calculate_bic(kmeans, X):
    """
    Calculate the Bayesian Information Criterion (BIC) for a K-Means model.

    Parameters:
    -----------
    kmeans : KMeans
        Fitted KMeans model
    X : numpy.ndarray
        Data used to fit the model

    Returns:
    --------
    bic : float
        BIC score (lower is better)
    """
    # Get model parameters
    n_samples, n_features = X.shape
    k = kmeans.n_clusters

    # Calculate log-likelihood
    # Compute distances from each point to its assigned cluster center
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    distances = np.zeros(n_samples)

    for i in range(n_samples):
        distances[i] = np.sum((X[i] - centers[labels[i]])**2)

    # Estimate variance (assuming spherical clusters)
    variance = np.sum(distances) / (n_samples - k)
    if variance <= 0:
        variance = 1e-10  # Avoid division by zero or negative variance

    # Log-likelihood
    log_likelihood = -0.5 * \
        (n_samples * np.log(2 * np.pi * variance) + n_samples)

    # Number of free parameters: k cluster centers (each with n_features dimensions) + 1 variance parameter
    n_params = k * n_features + 1

    # BIC = -2 * log-likelihood + n_params * log(n_samples)
    bic = -2 * log_likelihood + n_params * np.log(n_samples)

    return bic


def cluster_electrodes(features, max_clusters=10):
    """
    Cluster electrodes using K-Means and BIC

    Parameters:
    -----------
    features : numpy.ndarray
        Array of features for each electrode (from PCA on correlation matrix)
    max_clusters : int
        Maximum number of clusters to consider

    Returns:
    --------
    predictions : numpy.ndarray
        Array of cluster assignments for each electrode
    best_kmeans : KMeans
        Fitted KMeans model with optimal number of clusters
    scores : tuple
        Tuple containing (cluster_range, bic_scores)
    """
    print("Clustering electrodes with K-Means using BIC...")

    # Initialize variables to track the best model
    best_bic = np.inf  # For BIC, lower is better
    best_kmeans = None
    best_predictions = None
    bic_scores = []
    cluster_range = []

    # Handle the case where we have very few samples
    max_possible_clusters = min(max_clusters, len(features) - 1)

    # Try different numbers of clusters, starting from 1
    min_clusters = 1

    # Try different numbers of clusters
    for n_clusters in range(min_clusters, max_possible_clusters + 1):
        cluster_range.append(n_clusters)
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10  # Multiple initializations to find best solution
        )
        kmeans.fit(features)
        predictions = kmeans.labels_

        # Calculate BIC score
        bic = calculate_bic(kmeans, features)
        bic_scores.append(bic)
        print(f"  K={n_clusters}, BIC={bic:.4f}")

        if bic < best_bic:
            best_bic = bic
            best_kmeans = kmeans
            best_predictions = predictions

    print(
        f"Selected optimal number of clusters: {len(np.unique(best_predictions))}")
    return best_predictions, best_kmeans, (cluster_range, bic_scores)


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

    data_df = pd.DataFrame(
        dict(
            predictions=predictions,
            electrode_names=electrode_names
        )
    )

    data_df.sort_values(by=['predictions', 'electrode_names'], inplace=True)

    # Sort electrodes by cluster assignment
    # sorted_indices = np.argsort(predictions)
    sorted_indices = data_df.index.values
    sorted_corr_matrix = corr_matrix[sorted_indices, :][:, sorted_indices]
    # sorted_names = [electrode_names[i] for i in sorted_indices]
    sorted_names = data_df.electrode_names.values

    # Plot clustered correlation matrix
    fig, ax = plt.subplots(1, 3, figsize=(24, 8),
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
    # Plot axlines
    line_locs = np.where(np.abs(np.diff(sorted_predictions)))[0]
    for this_ax in [ax[1], ax[2]]:
        for i in line_locs:
            this_ax.axvline(x=i+0.5, color='black', linewidth=0.5)
            this_ax.axhline(y=i+0.5, color='black', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

############################################################
############################################################


testing_bool = False

if not testing_bool:
    # Get name of directory with the data files
    metadata_handler = imp_metadata(sys.argv)
    # Define script path first
    script_path = os.path.realpath(__file__)
    # Get directory name from metadata handler
    dir_name = metadata_handler.dir_name
    # Now create pipeline check with the correct dir_name
    this_pipeline_check = pipeline_graph_check(dir_name)
    this_pipeline_check.check_previous(script_path)
    this_pipeline_check.write_to_log(script_path, 'attempted')
else:
    # data_dir = '/media/storage/for_transfer/bla_gc/AM35_4Tastes_201228_124547'
    data_dir = '/media/storage/abu_resorted/gc_only/AM34_4Tastes_201216_105150/'
    # data_dir = '/home/abuzarmahmood/Desktop/blech_clust/pipeline_testing/test_data_handling/test_data/KM45_5tastes_210620_113227_new'
    metadata_handler = imp_metadata([[], data_dir])
    print(' ==== Running in test mode ====')

# dir_name already defined above for non-testing case
if testing_bool:
    dir_name = metadata_handler.dir_name

os.chdir(dir_name)
print(f'Processing : {dir_name}')

# Open the hdf5 file
hf5 = tables.open_file(metadata_handler.hdf5_name, 'r+')

# Read CAR groups from info file
# Every region is a separate group, multiple ports under single region is a separate group,
# emg is a separate group
info_dict = metadata_handler.info_dict
electrode_layout_frame = metadata_handler.layout.copy()
# Remove emg and none channels from the electrode layout frame
emg_bool = ~electrode_layout_frame.CAR_group.str.contains('emg')
none_bool = ~electrode_layout_frame.CAR_group.str.contains('none')
fin_bool = np.logical_and(emg_bool, none_bool)
electrode_layout_frame = electrode_layout_frame[fin_bool]
electrode_layout_frame['channel_name'] = electrode_layout_frame.apply(
    lambda row: f"{row['port']}_{row['electrode_num']:02}", axis=1
)
# electrode_layout_frame['port'].astype(str) + '_' + \
# electrode_layout_frame['electrode_num'].astype(str)

num_groups = electrode_layout_frame.CAR_group.nunique()
print(f" Number of groups : {num_groups}")
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
    auto_car_inference = auto_car_section.get('use_auto_CAR', False)
    max_clusters = auto_car_section.get(
        'max_clusters', 10)  # Default to 10 if not specified
    # Use BIC for clustering by default
    use_bic = auto_car_section.get('use_bic', True)
else:
    auto_car_inference = False
    max_clusters = 10

plots_dir = os.path.join(dir_name, 'QA_output')
os.makedirs(plots_dir, exist_ok=True)

# If auto_car_inference is enabled, perform clustering on each CAR group
if auto_car_inference:
    print("\nPerforming automatic CAR group inference...")

    # Create a directory for cluster plots if it doesn't exist

    # Get correlation matrix using the utility function
    corr_mat = get_channel_corr_mat(dir_name)
    # Convert nan to 0
    corr_mat[np.isnan(corr_mat)] = 0
    # Make symmetric
    # Average to ensure perfect symmetry
    corr_mat = (corr_mat + corr_mat.T) / 2

    # Index corr_mat by the electrode layout frame
    index_bool = emg_bool[none_bool].values
    corr_mat = corr_mat[index_bool, :][:, index_bool]

    # Create a dictionary to store cluster predictions for each CAR group
    all_predictions = np.zeros(len(electrode_layout_frame), dtype=int)

    # Process each CAR group separately
    for group_idx, group_name in enumerate(electrode_layout_frame.CAR_group.unique()):
        print(f"\nProcessing CAR group: {group_name}")

        # Get indices for this CAR group
        group_mask = electrode_layout_frame.CAR_group == group_name
        group_indices = np.where(group_mask)[0]

        if len(group_indices) <= 1:
            print(
                f"  Skipping group {group_name} - only {len(group_indices)} channels")
            continue

        # Extract correlation submatrix for this group
        group_corr_mat = corr_mat[group_indices, :][:, group_indices]

        # Perform PCA on this group's correlation matrix
        n_components = min(5, len(group_corr_mat) - 1)
        if n_components <= 0:
            print(
                f"  Skipping group {group_name} - insufficient channels for PCA")
            continue

        pca = PCA(n_components=n_components)
        group_features = pca.fit_transform(group_corr_mat)

        # Cluster electrodes within this group
        group_predictions, model, (cluster_range, scores) = cluster_electrodes(
            group_features,
            max_clusters=min(max_clusters, len(group_corr_mat) - 1)
        )

        print(
            f"  Found {len(np.unique(group_predictions))} clusters in group {group_name}")

        # Store predictions for this group
        for i, idx in enumerate(group_indices):
            all_predictions[idx] = group_predictions[i]

        # Plot K-Means BIC scores for this group
        plt.figure(figsize=(10, 6))
        plt.plot(cluster_range, scores, 'o-', color='blue')
        plt.title(f'K-Means Clustering BIC Scores - Group {group_name}')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('BIC Score (lower is better)')
        plt.grid(True)
        plt.savefig(os.path.join(
            plots_dir, f'kmeans_bic_scores_{group_name}.png'))
        plt.close()

    # Store all predictions in the electrode layout frame
    electrode_layout_frame['predicted_clusters'] = all_predictions

    # Save original CAR_groups as backup
    electrode_layout_frame['original_CAR_group'] = electrode_layout_frame['CAR_group']

    # Append cluster numbers to CAR group names
    electrode_layout_frame['CAR_group'] = electrode_layout_frame.apply(
        lambda row: f"{row['CAR_group']}-{row['predicted_clusters']:02}", axis=1
    )
    num_groups = electrode_layout_frame.CAR_group.nunique()

    pred_map = dict(zip(
        electrode_layout_frame.CAR_group.unique(),
        np.arange(num_groups)
    )
    )

    # Plot clusters for all electrodes
    plot_path = os.path.join(dir_name, 'QA_output', 'clustered_corr_mat.png')
    plot_clustered_corr_mat(
        corr_matrix=corr_mat,
        predictions=electrode_layout_frame.CAR_group.map(pred_map).values,
        electrode_names=electrode_layout_frame.channel_name.values,
        plot_path=plot_path
    )

    print(
        "Calculating common average reference for {:d} groups".format(num_groups))
    print('Updated CAR groups with channel counts')
    print(electrode_layout_frame.groupby('CAR_group').size())

    # Write out an updated version of the electrode layout frame
    # Care needs to be taken to preserve all the original information AND
    # not mess with the additional information added to the layout frame above
    layout_frame_path = glob.glob(os.path.join(
        dir_name, '*_electrode_layout.csv'))[0]
    out_electrode_layout_frame = metadata_handler.layout.copy()
    out_electrode_layout_frame.at[fin_bool,
                                  'predicted_clusters'] = all_predictions
    out_electrode_layout_frame.to_csv(layout_frame_path)
    print(f"Updated electrode layout frame written to {layout_frame_path}")


# First get the common average references by adjusting mean and standard deviation
# of all channels in a CAR group before taking the average
# This is important because all channels are not recorded at the same scale due to
# differences in impedance and other factors

# Also plot normalized channels by cluster


fig, ax = gen_square_subplots(len(electrode_layout_frame),
                              sharex=True, sharey=True, figsize=(15, 15))
n_plot_points = 10_000
rec_length = raw_electrodes[0][:].shape[0]
plot_inds = np.arange(0, rec_length, rec_length // n_plot_points)
plot_counter = 0
cmap = plt.get_cmap('tab10')

common_average_reference = np.zeros(
    (num_groups, rec_length), dtype=np.float32)

print('Calculating mean values')
for group_num, group_name in enumerate(electrode_layout_frame.CAR_group.unique()):
    print(f"\nProcessing group {group_name}")

    this_car_frame = electrode_layout_frame[electrode_layout_frame.CAR_group == group_name]
    print(f" {len(this_car_frame)} channels :: \n{this_car_frame.channel_name.values}")

    # Get electrode indices for this group
    electrode_indices = this_car_frame.electrode_ind.values

    # Load and normalize all electrode data for this group
    CAR_sum = np.zeros(raw_electrodes[0][:].shape[0])
    for electrode_name in tqdm(electrode_indices):
        channel_data = get_electrode_by_name(raw_electrodes, electrode_name)[:]
        # Normalize each channel by subtracting mean and dividing by std
        # channel_mean = np.mean(channel_data)
        # channel_std = np.std(channel_data)
        channel_mean = np.median(channel_data[::100])
        channel_std = MAD(channel_data[::100])
        normalized_channel = (channel_data - channel_mean) / channel_std
        CAR_sum += normalized_channel

        # Plot normalized channels
        ax.flatten()[plot_counter].plot(
            normalized_channel[plot_inds], color=cmap(group_num))
        ax.flatten()[plot_counter].set_title(
            f"{group_name} :: {electrode_name}")
        plot_counter += 1

    # Calculate the average of normalized channels
    if len(electrode_indices) > 0:
        common_average_reference[group_num,
                                 :] = CAR_sum / len(electrode_indices)

fig.suptitle('Normalized Channels by Cluster')
fig.savefig(os.path.join(plots_dir, 'normalized_channels.png'))
plt.close(fig)

print("Common average reference for {:d} groups calculated".format(num_groups))
print()

# Now run through the raw electrode data and
# subtract the common average reference from each of them
print('Performing background subtraction')
for group_num, group_name in enumerate(electrode_layout_frame.CAR_group.unique()):
    print(f"Processing group {group_name}")
    this_car_frame = electrode_layout_frame[electrode_layout_frame.CAR_group == group_name]
    electrode_indices = this_car_frame.electrode_ind.values
    if len(electrode_indices) > 1:
        for electrode_num in tqdm(electrode_indices):
            # Get the electrode data
            wanted_electrode = get_electrode_by_name(
                raw_electrodes, electrode_num)
            electrode_data = wanted_electrode[:]

            # Normalize the electrode data
            # electrode_mean = np.mean(electrode_data)
            # electrode_std = np.std(electrode_data)
            electrode_mean = np.median(electrode_data[::100])
            electrode_std = MAD(electrode_data[::100])
            normalized_data = (electrode_data - electrode_mean) / electrode_std

            # Subtract the common average reference for that group
            referenced_data = normalized_data - \
                common_average_reference[group_num]

            # Convert back to original scale
            final_data = (referenced_data * electrode_std) + electrode_mean

            # Overwrite the electrode data with the referenced data
            wanted_electrode[:] = final_data
            hf5.flush()
            del referenced_data, final_data, normalized_data, electrode_data


hf5.close()
print("Modified electrode arrays written to HDF5 file after "
      "subtracting the common average reference")

# Write successful execution to log
if not testing_bool:
    this_pipeline_check.write_to_log(script_path, 'completed')
