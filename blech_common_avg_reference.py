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
from utils.ephys_data.visualize import gen_square_subplots

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

    return predictions, model


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
    data_dir = '/home/abuzarmahmood/Desktop/blech_clust/pipeline_testing/test_data_handling/test_data/KM45_5tastes_210620_113227_new'
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
electrode_layout_frame['channel_name'] = \
    electrode_layout_frame['port'].astype(str) + '_' + \
    electrode_layout_frame['electrode_num'].astype(str)

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

    # Perform PCA - use min of 5 or the number of channels to avoid errors
    n_components = min(10, len(corr_mat) - 1)
    pca = PCA(n_components=n_components)
    features = pca.fit_transform(corr_mat)

    # Cluster electrodes
    predictions, model = cluster_electrodes(
        features,
        n_components=min(max_clusters, len(corr_mat) - 1),
        n_iter=100,
        threshold=1e-3
    )

    print(f"Found {len(np.unique(predictions))} clusters")

    electrode_layout_frame['predicted_clusters'] = predictions

    # Plot clusters
    plot_path = os.path.join(dir_name, 'QA_output', 'clustered_corr_mat.png')
    plot_clustered_corr_mat(
        corr_mat, predictions, electrode_layout_frame.channel_name.values, plot_path
    )

    # Save original CAR_groups as backup
    electrode_layout_frame['original_CAR_group'] = electrode_layout_frame['CAR_group']

    # Append cluster numbers to CAR group names
    electrode_layout_frame['CAR_group'] = electrode_layout_frame.apply(
        lambda row: f"{row['CAR_group']}-{row['predicted_clusters']:02}", axis=1
    )

    num_groups = electrode_layout_frame.CAR_group.nunique()
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
    out_electrode_layout_frame.at[fin_bool, 'predicted_clusters'] = predictions
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
