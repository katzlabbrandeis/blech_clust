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
from sklearn.mixture import BayesianGaussianMixture as BGMM
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from blech_clust.utils.blech_utils import imp_metadata, pipeline_graph_check
from blech_clust.utils.qa_utils import channel_corr
from blech_clust.utils.ephys_data.visualize import gen_square_subplots
import pandas as pd
import argparse
import xarray as xr
import seaborn as sns

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
    # Load netcdf file
    # return np.load(os.path.join(qa_out_path, 'channel_corr_mat.npy'))
    corr_mat_path = glob.glob(os.path.join(
        qa_out_path, 'channel_corr_mat.nc'))[0]
    corr_ds = xr.open_dataarray(corr_mat_path)
    return corr_ds


def calculate_group_averages(raw_electrodes, electrode_layout_frame, rec_length):
    """
    Calculate common average reference for each CAR group by normalizing channels
    and computing their average.

    Parameters:
    -----------
    raw_electrodes : list
        List of raw electrode data arrays
    electrode_layout_frame : pandas.DataFrame
        DataFrame containing electrode layout information with CAR_group column
    rec_length : int
        Length of the recording

    Returns:
    --------
    numpy.ndarray
        Common average reference array of shape (num_groups, rec_length)
    """

    num_groups = electrode_layout_frame.CAR_group.nunique()

    common_average_reference = np.zeros(
        (num_groups, rec_length), dtype=np.float32)

    print('Calculating mean values')
    for group_num, group_name in enumerate(electrode_layout_frame.CAR_group.unique()):
        print(f"\nProcessing group {group_name}")

        this_car_frame = electrode_layout_frame[electrode_layout_frame.CAR_group == group_name]
        print(
            f" {len(this_car_frame)} channels :: \n{this_car_frame.channel_name.values}")

        # Get electrode indices for this group
        electrode_indices = this_car_frame.electrode_ind.values

        # Load and normalize all electrode data for this group
        CAR_sum = np.zeros(raw_electrodes[0][:].shape[0])
        for electrode_name in tqdm(electrode_indices):
            channel_data = get_electrode_by_name(
                raw_electrodes, electrode_name)[:]
            # Normalize each channel by subtracting mean and dividing by std
            channel_mean = np.median(channel_data[::100])
            channel_std = MAD(channel_data[::100])
            normalized_channel = (channel_data - channel_mean) / channel_std
            CAR_sum += normalized_channel

        # Calculate the average of normalized channels
        if len(electrode_indices) > 0:
            common_average_reference[group_num,
                                     :] = CAR_sum / len(electrode_indices)

    return common_average_reference


def perform_background_subtraction(raw_electrodes, electrode_layout_frame,
                                   common_average_reference):
    """
    Subtract the common average reference from each electrode and update the data.

    Parameters:
    -----------
    raw_electrodes : list
        List of raw electrode data arrays
    electrode_layout_frame : pandas.DataFrame
        DataFrame containing electrode layout information with CAR_group column
    common_average_reference : numpy.ndarray
        Common average reference array of shape (num_groups, rec_length)
    """
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
                electrode_mean = np.median(electrode_data[::100])
                electrode_std = MAD(electrode_data[::100])
                normalized_data = (
                    electrode_data - electrode_mean) / electrode_std

                # Subtract the common average reference for that group
                referenced_data = normalized_data - \
                    common_average_reference[group_num]

                # Convert back to original scale
                final_data = (referenced_data * electrode_std) + electrode_mean

                # Overwrite the electrode data with the referenced data
                wanted_electrode[:] = final_data
                del referenced_data, final_data, normalized_data, electrode_data


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


def cluster_electrodes_kmeans(features, max_clusters=10):
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


def cluster_electrodes_bgmm(features, max_clusters=10):
    """
    Cluster electrodes using Bayesian Gaussian Mixture Model (BGMM)

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
    best_bgmm : BGMM
        Fitted BGMM model with optimal number of clusters
    scores : tuple
        Tuple containing (cluster_range, bic_scores)
    """
    print("Clustering electrodes with BGMM using BIC...")

    # Handle the case where we have very few samples
    max_possible_clusters = min(max_clusters, len(features) - 1)

    # Try different numbers of clusters
    bgmm = BGMM(
        n_components=max_possible_clusters,
        covariance_type='full',
        random_state=42,
        max_iter=500,
    )
    bgmm.fit(features)
    predictions = bgmm.predict(features)
    probs = bgmm.predict_proba(features)

    return predictions, bgmm, ([], [])  # BGMM does not require score tracking

# Wrapper function to choose clustering algorithm


def cluster_electrodes(features, max_clusters=10, cluster_algo='kmeans'):
    assert cluster_algo in [
        'kmeans', 'bgmm'], "cluster_algo must be 'kmeans' or 'bgmm'"
    if cluster_algo == 'kmeans':
        return cluster_electrodes_kmeans(features, max_clusters)
    elif cluster_algo == 'bgmm':
        return cluster_electrodes_bgmm(features, max_clusters)
    else:
        raise ValueError(f"Unknown clustering algorithm: {cluster_algo}")


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
    electrode_names : list, channel labels (e.g., "GC:0", "PC:16")
    plot_path : str
    cmap : str, colormap name
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

    # Adjust figure size and font based on number of channels
    n_chans = len(electrode_names)
    fig_width = max(24, n_chans * 0.4)
    fig_height = max(8, n_chans * 0.15)
    fontsize = max(4, 8 - n_chans // 20)

    # Plot clustered correlation matrix with shared colorbar
    fig, ax = plt.subplots(1, 3, figsize=(fig_width, fig_height),
                           gridspec_kw={'width_ratios': [
                               1, 1, 1], 'wspace': 0.3}
                           )

    # Create images for correlation matrices
    im0 = ax[0].imshow(corr_matrix, cmap=cmap, vmin=0, vmax=1)
    ax[0].set_title('Original Correlation Matrix')
    ax[0].set_xticks(np.arange(len(electrode_names)))
    ax[0].set_yticks(np.arange(len(electrode_names)))
    ax[0].set_xticklabels(electrode_names, rotation=90, fontsize=fontsize)
    ax[0].set_yticklabels(electrode_names, fontsize=fontsize)

    im1 = ax[1].imshow(sorted_corr_matrix, cmap=cmap, vmin=0, vmax=1)
    ax[1].set_title('Clustered Correlation Matrix')
    ax[1].set_xticks(np.arange(len(sorted_names)))
    ax[1].set_yticks(np.arange(len(sorted_names)))
    ax[1].set_xticklabels(sorted_names, rotation=90, fontsize=fontsize)
    ax[1].set_yticklabels(sorted_names, fontsize=fontsize)
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
    ax[2].set_xticklabels(sorted_names, rotation=90, fontsize=fontsize)
    ax[2].set_yticklabels(sorted_names, fontsize=fontsize)
    # Plot axlines
    line_locs = np.where(np.abs(np.diff(sorted_predictions)))[0]
    for this_ax in [ax[1], ax[2]]:
        for i in line_locs:
            this_ax.axvline(x=i+0.5, color='black', linewidth=0.5)
            this_ax.axhline(y=i+0.5, color='black', linewidth=0.5)

    # Add shared colorbar for the correlation matrices below all subplots
    fig.subplots_adjust(bottom=0.15)  # Make room for colorbar
    # [left, bottom, width, height]
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])
    fig.colorbar(im0, cax=cbar_ax, orientation='horizontal',
                 label='Correlation Coefficient')

    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close()


def calculate_post_car_correlations(hf5_path, electrode_layout_frame, n_corr_samples=10000):
    """
    Calculate correlation matrix after CAR has been applied.

    Parameters:
    -----------
    hf5_path : str
        Path to HDF5 file
    electrode_layout_frame : pandas.DataFrame
        DataFrame containing electrode layout information
    n_corr_samples : int
        Number of samples to use for correlation calculation

    Returns:
    --------
    corr_mat : numpy.ndarray
        Correlation matrix
    chan_names : numpy.ndarray
        Channel names/indices
    """
    all_chans, chan_names, _ = channel_corr.get_all_channels(
        hf5_path,
        electrode_layout_frame=electrode_layout_frame,
        n_corr_samples=n_corr_samples
    )
    corr_mat = channel_corr.intra_corr(all_chans)
    return corr_mat, chan_names


def plot_car_correlation_comparison(pre_corr_mat, post_corr_mat,
                                    electrode_layout_frame, plot_dir,
                                    valid_electrode_inds):
    """
    Generate comparison plots showing correlation changes before and after CAR.

    Creates:
    1. Side-by-side heatmaps of correlation matrices (before/after)
    2. Paired points plot (swarm) showing correlation values before and after

    Parameters:
    -----------
    pre_corr_mat : numpy.ndarray
        Correlation matrix before CAR
    post_corr_mat : numpy.ndarray
        Correlation matrix after CAR
    electrode_layout_frame : pandas.DataFrame
        DataFrame containing electrode layout information
    plot_dir : str
        Directory to save plots
    valid_electrode_inds : numpy.ndarray
        Indices of valid electrodes (non-EMG, non-none)
    """
    # Extract upper triangle values (excluding diagonal)
    n_chans = pre_corr_mat.shape[0]
    triu_inds = np.triu_indices(n_chans, k=1)

    pre_vals = pre_corr_mat[triu_inds]
    post_vals = post_corr_mat[triu_inds]

    # Remove NaN values
    valid_mask = ~(np.isnan(pre_vals) | np.isnan(post_vals))
    pre_vals = pre_vals[valid_mask]
    post_vals = post_vals[valid_mask]

    # 1. Heatmaps comparison
    fig_width = max(16, n_chans * 0.3)
    fig_height = max(6, n_chans * 0.12)
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height))

    # Get channel labels
    chan_labels = electrode_layout_frame.channel_name.values
    fontsize = max(4, 8 - n_chans // 20)

    # Before CAR
    im0 = axes[0].imshow(pre_corr_mat, cmap='jet', vmin=0, vmax=1)
    axes[0].set_title('Before CAR', fontsize=12)
    axes[0].set_xlabel('Channel')
    axes[0].set_ylabel('Channel')
    if len(chan_labels) <= 64:
        axes[0].set_xticks(np.arange(len(chan_labels)))
        axes[0].set_yticks(np.arange(len(chan_labels)))
        axes[0].set_xticklabels(chan_labels, rotation=90, fontsize=fontsize)
        axes[0].set_yticklabels(chan_labels, fontsize=fontsize)
    fig.colorbar(im0, ax=axes[0], shrink=0.8)

    # After CAR
    im1 = axes[1].imshow(post_corr_mat, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title('After CAR', fontsize=12)
    axes[1].set_xlabel('Channel')
    axes[1].set_ylabel('Channel')
    if len(chan_labels) <= 64:
        axes[1].set_xticks(np.arange(len(chan_labels)))
        axes[1].set_yticks(np.arange(len(chan_labels)))
        axes[1].set_xticklabels(chan_labels, rotation=90, fontsize=fontsize)
        axes[1].set_yticklabels(chan_labels, fontsize=fontsize)
    fig.colorbar(im1, ax=axes[1], shrink=0.8)

    # Difference (Before - After)
    diff_mat = pre_corr_mat - post_corr_mat
    max_diff = np.nanmax(np.abs(diff_mat))
    im2 = axes[2].imshow(diff_mat, cmap='RdBu_r',
                         vmin=-max_diff, vmax=max_diff)
    axes[2].set_title('Difference (Before - After)', fontsize=12)
    axes[2].set_xlabel('Channel')
    axes[2].set_ylabel('Channel')
    if len(chan_labels) <= 64:
        axes[2].set_xticks(np.arange(len(chan_labels)))
        axes[2].set_yticks(np.arange(len(chan_labels)))
        axes[2].set_xticklabels(chan_labels, rotation=90, fontsize=fontsize)
        axes[2].set_yticklabels(chan_labels, fontsize=fontsize)
    fig.colorbar(im2, ax=axes[2], shrink=0.8)

    fig.suptitle('Channel Correlation: Before vs After CAR', fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'car_correlation_heatmaps.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 2. Paired points plot (swarm)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'Before CAR': pre_vals,
        'After CAR': post_vals
    })

    # Melt for swarm plot
    plot_df_melted = plot_df.melt(
        var_name='Condition', value_name='Correlation')

    # Swarm plot with paired lines
    # Subsample if too many points for visibility
    n_pairs = len(pre_vals)
    max_plot_pairs = 500
    if n_pairs > max_plot_pairs:
        sample_inds = np.random.choice(n_pairs, max_plot_pairs, replace=False)
        pre_sample = pre_vals[sample_inds]
        post_sample = post_vals[sample_inds]
    else:
        pre_sample = pre_vals
        post_sample = post_vals

    # Paired points with connecting lines
    for i in range(len(pre_sample)):
        axes[0].plot([0, 1], [pre_sample[i], post_sample[i]],
                     color='gray', alpha=0.1, linewidth=0.5)

    # Add jittered scatter
    jitter = 0.05
    axes[0].scatter(np.zeros(len(pre_sample)) + np.random.uniform(-jitter, jitter, len(pre_sample)),
                    pre_sample, alpha=0.3, s=10, color='blue', label='Before CAR')
    axes[0].scatter(np.ones(len(post_sample)) + np.random.uniform(-jitter, jitter, len(post_sample)),
                    post_sample, alpha=0.3, s=10, color='red', label='After CAR')

    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['Before CAR', 'After CAR'])
    axes[0].set_ylabel('Correlation')
    axes[0].set_title('Paired Channel Correlations')
    axes[0].set_ylim(-0.1, 1.1)

    # Add summary statistics
    mean_pre = np.mean(pre_vals)
    mean_post = np.mean(post_vals)
    axes[0].axhline(mean_pre, color='blue', linestyle='--',
                    alpha=0.7, label=f'Mean Before: {mean_pre:.3f}')
    axes[0].axhline(mean_post, color='red', linestyle='--',
                    alpha=0.7, label=f'Mean After: {mean_post:.3f}')
    axes[0].legend(loc='upper right', fontsize=8)

    # Histogram of differences
    diff_vals = pre_vals - post_vals
    axes[1].hist(diff_vals, bins=50, edgecolor='black',
                 alpha=0.7, color='purple')
    axes[1].axvline(0, color='black', linestyle='--', linewidth=2)
    axes[1].axvline(np.mean(diff_vals), color='red', linestyle='-', linewidth=2,
                    label=f'Mean: {np.mean(diff_vals):.3f}')
    axes[1].set_xlabel('Correlation Change (Before - After)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Distribution of Correlation Changes')
    axes[1].legend()

    # Add text with statistics
    pct_decreased = np.sum(diff_vals > 0) / len(diff_vals) * 100
    stats_text = f'Pairs with decreased correlation: {pct_decreased:.1f}%\n'
    stats_text += f'Mean change: {np.mean(diff_vals):.3f}\n'
    stats_text += f'Median change: {np.median(diff_vals):.3f}'
    axes[1].text(0.95, 0.95, stats_text, transform=axes[1].transAxes,
                 verticalalignment='top', horizontalalignment='right',
                 fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle('CAR Effect on Channel Correlations', fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'car_correlation_paired_plot.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Print summary
    print("\n=== CAR Correlation Analysis ===")
    print(f"Mean correlation before CAR: {mean_pre:.4f}")
    print(f"Mean correlation after CAR: {mean_post:.4f}")
    print(f"Mean change: {np.mean(diff_vals):.4f}")
    print(f"Pairs with decreased correlation: {pct_decreased:.1f}%")
    print("================================\n")

    return diff_vals


############################################################
############################################################


testing_bool = False

if not testing_bool:
    parser = argparse.ArgumentParser(
        description='Load data and create hdf5 file')
    parser.add_argument('dir_name', type=str,
                        help='Directory name with data files')
    # Allow kmeans or bgmm clustering
    parser.add_argument('--cluster_algo', type=str, choices=['kmeans', 'bgmm'],
                        help='Clustering algorithm to use for auto CAR, BGMM tends to allow more clusters',
                        default='kmeans')
    parser.add_argument('--overwrite_dependencies', action='store_true',
                        help='Overwrite dependency check and continue even if previous script was not run')
    args = parser.parse_args()

    # Get name of directory with the data files
    metadata_handler = imp_metadata([[], args.dir_name])
    # Define script path first
    script_path = os.path.realpath(__file__)
    # Get directory name from metadata handler
    dir_name = metadata_handler.dir_name
    # Now create pipeline check with the correct dir_name
    overwrite_dependencies = args.overwrite_dependencies
    this_pipeline_check = pipeline_graph_check(dir_name, overwrite_dependencies)
    this_pipeline_check.check_previous(script_path)
    this_pipeline_check.write_to_log(script_path, 'attempted')

    cluster_algo = args.cluster_algo
else:
    from pprint import pprint as pp
    # data_dir = '/media/storage/for_transfer/bla_gc/AM35_4Tastes_201228_124547'
    # data_dir = '/media/storage/abu_resorted/gc_only/AM34_4Tastes_201216_105150/'
    # data_dir = '/media/storage/abu_resorted/bla_gc/AM11_4Tastes_191030_114043_copy'
    # data_dir = '/media/storage/abu_resorted/bla_gc/AM35_4Tastes_201230_115322'
    data_dir = '/media/bigdata/.blech_clust_test_data/KM45_5tastes_210620_113227_new'
    # data_dir = '/home/abuzarmahmood/Desktop/blech_clust/pipeline_testing/test_data_handling/test_data/KM45_5tastes_210620_113227_new'
    metadata_handler = imp_metadata([[], data_dir])
    cluster_algo = 'kmeans'
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
original_layout_frame = metadata_handler.layout.copy()

# Convert CAR_group column to string type to handle float values (e.g., NaN)
original_layout_frame['CAR_group'] = original_layout_frame['CAR_group'].astype(
    str)

# If'original_CAR_group' in layout, use it to overwrite CAR_group for processing
if 'original_CAR_group' in original_layout_frame.columns:
    print('='*60)
    print("original_CAR_group found in layout")
    print("Over-writing CAR_group with original_CAR_group for processing")
    print('='*60)
    original_layout_frame['CAR_group'] = original_layout_frame['original_CAR_group'].astype(
        str)

# Remove emg and none channels from the electrode layout frame
# Use fin_bool to track which channels to process (not emg or none)
fin_bool = ~(original_layout_frame.CAR_group.str.contains('emg') |
             original_layout_frame.CAR_group.str.contains('none'))
electrode_layout_frame = original_layout_frame[fin_bool].copy()
electrode_layout_frame['channel_name'] = electrode_layout_frame.apply(
    lambda row: f"{row['CAR_group']}:{row['electrode_num']}", axis=1
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

# Process correlation matrix
# Create a directory for cluster plots if it doesn't exist
# Get correlation matrix using the utility function
corr_xarray = get_channel_corr_mat(dir_name)
# Index corr_mat using electrode_ind from the filtered electrode_layout_frame
# The correlation matrix is ordered by electrode index (from channel_corr.get_all_channels)
# Use electrode_ind to directly select the channels we want to process
valid_electrode_inds = electrode_layout_frame.electrode_ind.values
# corr_mat = corr_mat[valid_electrode_inds, :][:, valid_electrode_inds]
corr_xarray = corr_xarray.sel(
    channel_1=valid_electrode_inds, channel_2=valid_electrode_inds)

corr_mat = corr_xarray.values
# Convert nan to 0
corr_mat[np.isnan(corr_mat)] = 1
# Make symmetric
# Average to ensure perfect symmetry
corr_mat = (corr_mat + corr_mat.T) / 2
# Make diagonal 1
np.fill_diagonal(corr_mat, 1)

# plt.matshow(corr_mat, cmap='jet');plt.title('Electrode Correlation Matrix');plt.show()

# If auto_car_inference is enabled, perform clustering on each CAR group
if auto_car_inference:
    print("\nPerforming automatic CAR group inference...")

    # Create a dictionary to store cluster predictions for each CAR group
    all_predictions = np.zeros(len(electrode_layout_frame), dtype=int)

    # Process each CAR group separately
    car_groups = electrode_layout_frame.CAR_group.unique()
    for group_idx, group_name in enumerate(car_groups):
        print(f"\nProcessing CAR group: {group_name}")

        group_mask = electrode_layout_frame.CAR_group == group_name
        group_indices = np.where(group_mask)[0]

        if len(group_indices) <= 1:
            print(
                f"  Skipping group {group_name} - only {len(group_indices)} channels")
            continue

        # Extract correlation submatrix for this group
        group_corr_mat = corr_mat[group_indices, :][:, group_indices]

        # plt.matshow(group_corr_mat, cmap='jet');plt.show()

        # Perform PCA on this group's correlation matrix
        n_components = min(3, len(group_corr_mat) - 1)
        if n_components <= 0:
            print(
                f"  Skipping group {group_name} - insufficient channels for PCA")
            continue

        pca = PCA(n_components=n_components)
        group_features = pca.fit_transform(group_corr_mat)

        # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        # ax[0].imshow(group_corr_mat, cmap='jet')
        # ax[0].set_title(f'Correlation Matrix - Group {group_name}')
        # ax[1].imshow(group_features, cmap='viridis', aspect='auto')
        # ax[1].set_title(f'PCA Features - Group {group_name}')
        # plt.tight_layout()
        # plt.show()

        # Cluster electrodes within this group
        group_predictions, model, (cluster_range, scores) = cluster_electrodes(
            group_features,
            max_clusters=min(max_clusters, len(group_corr_mat) - 1),
            cluster_algo=cluster_algo
        )

        print(
            f"  Found {len(np.unique(group_predictions))} clusters in group {group_name}")

        # Store predictions for this group
        for i, idx in enumerate(group_indices):
            all_predictions[idx] = group_predictions[i]

        if cluster_algo == 'kmeans':
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
    assert len(all_predictions) == len(electrode_layout_frame), \
        "Length of predictions does not match number of electrodes"
    electrode_layout_frame['predicted_clusters'] = all_predictions

    # Save original CAR_groups as backup only if not already present
    if 'original_CAR_group' not in electrode_layout_frame.columns:
        electrode_layout_frame['original_CAR_group'] = electrode_layout_frame['CAR_group']

    # Append cluster numbers to CAR group names
    # This will be the same as using original_CAR_group + cluster number as
    # CAR group names were over-written above
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
    # Preserve original_CAR_group if it exists
    if 'original_CAR_group' in electrode_layout_frame.columns:
        out_electrode_layout_frame.at[fin_bool,
                                      'original_CAR_group'] = electrode_layout_frame['original_CAR_group'].values
    # Make sure CAR_group and original_CAR_group are strings
    out_electrode_layout_frame['CAR_group'] = out_electrode_layout_frame['CAR_group'].astype(
        'string')
    if 'original_CAR_group' in out_electrode_layout_frame.columns:
        out_electrode_layout_frame['original_CAR_group'] = out_electrode_layout_frame['original_CAR_group'].astype(
            'string')

    # Fill NaN with 'none' for non-processed channels
    out_electrode_layout_frame['original_CAR_group'].fillna(
        'none', inplace=True)

    out_electrode_layout_frame.to_csv(layout_frame_path)
    print(f"Updated electrode layout frame written to {layout_frame_path}")

# Check average intra-CAR similarity and write a warning if below threshold
# This runs after clustering so warnings apply to the actual CAR groups being processed
try:
    avg_threshold = metadata_handler.params_dict['qa_params']['avg_intra_car_similarity_threshold']
    if avg_threshold is not None:
        car_groups = electrode_layout_frame.CAR_group.unique()
        group_means = {}
        for group_name in car_groups:
            group_inds = np.where(
                electrode_layout_frame.CAR_group == group_name)[0]
            if len(group_inds) > 1:
                submat = corr_mat[np.ix_(group_inds, group_inds)]
                triu_vals = submat[np.triu_indices(len(group_inds), k=1)]
                triu_vals = triu_vals[~np.isnan(triu_vals)]
                if len(triu_vals) > 0:
                    group_means[group_name] = np.mean(triu_vals)

        if group_means:
            overall_avg = np.mean(list(group_means.values()))
            if any(np.array(list(group_means.values())) < avg_threshold):
                warnings_path = os.path.join(plots_dir, 'warnings.txt')
                warning_lines = [
                    '\n=== Average intra-CAR similarity warning ===',
                    f'Average intra-CAR similarity across groups: {overall_avg:.4f}',
                    f'Threshold: {avg_threshold:.4f}',
                    'Per-group mean similarities:',
                ] + [f'  {name}: {mean:.4f}' for name, mean in group_means.items()] + [
                    '=== End Average intra-CAR similarity warning ===\n'
                ]
                warning_message = '\n'.join(warning_lines)
                print(warning_message)
                with open(warnings_path, 'a') as wf:
                    wf.write(warning_message)
except Exception as e:
    raise RuntimeError(
        f"Error checking average intra-CAR similarity: {e}")

# First get the common average references by adjusting mean and standard deviation
# of all channels in a CAR group before taking the average
# This is important because all channels are not recorded at the same scale due to
# differences in impedance and other factors

# Perform PCA on the correlation matrix so that it can be plotted later
pca = PCA(n_components=2)
pca_features = pca.fit_transform(corr_mat)
var_explained = sum(pca.explained_variance_ratio_)

# Get group assignments for plotting
group_assignments = electrode_layout_frame.CAR_group.map(
    {name: num for num, name in enumerate(
        electrode_layout_frame.CAR_group.unique()
    )}).values

# Plot the PCA features
fig, ax = plt.subplots(figsize=(5, 5))
for group_num in range(num_groups):
    group_inds = np.where(group_assignments == group_num)[0]
    ax.scatter(pca_features[group_inds, 0],
               pca_features[group_inds, 1],
               label=f'Group {group_num}', alpha=0.7)
ax.set_title('PCA of Electrode Correlation Matrix')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
# put legend outside
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
# Annotate points with electrode names
for i, electrode_name in enumerate(electrode_layout_frame.channel_name.values):
    plt.annotate(electrode_name, (pca_features[i, 0], pca_features[i, 1]),
                 textcoords="offset points", xytext=(0, 5), ha='center', fontsize=10)
fig.suptitle(
    f'PCA of Electrode Correlation Matrix (Variance Explained: {var_explained:.2%})')
plt.tight_layout()
fig.savefig(os.path.join(
    plots_dir, 'electrode_corr_pca.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
# plt.show()

# Also plot normalized channels by cluster
fig, ax = gen_square_subplots(len(electrode_layout_frame),
                              sharex=True, sharey=True, figsize=(15, 15))
n_plot_points = 10_000
rec_length = raw_electrodes[0][:].shape[0]
plot_inds = np.arange(0, rec_length, rec_length // n_plot_points)
plot_counter = 0
cmap = plt.get_cmap('tab10')

# Calculate group averages using the new function
common_average_reference = calculate_group_averages(
    raw_electrodes, electrode_layout_frame, rec_length)

# Now need to add plotting code back for normalized channels
print('Plotting normalized channels')
for group_num, group_name in enumerate(electrode_layout_frame.CAR_group.unique()):
    this_car_frame = electrode_layout_frame[electrode_layout_frame.CAR_group == group_name]
    electrode_indices = this_car_frame.electrode_ind.values

    for electrode_name in tqdm(electrode_indices):
        channel_data = get_electrode_by_name(raw_electrodes, electrode_name)[:]
        channel_mean = np.median(channel_data[::100])
        channel_std = MAD(channel_data[::100])
        normalized_channel = (channel_data - channel_mean) / channel_std

        # Plot normalized channels
        ax.flatten()[plot_counter].plot(
            normalized_channel[plot_inds], color=cmap(group_num))
        ax.flatten()[plot_counter].set_title(
            f"{group_name} :: {electrode_name}")
        plot_counter += 1

fig.suptitle('Normalized Channels by Cluster')
fig.savefig(os.path.join(plots_dir, 'normalized_channels.png'))
plt.close(fig)

print("Common average reference for {:d} groups calculated".format(num_groups))
print()

# Perform background subtraction using the new function
perform_background_subtraction(raw_electrodes, electrode_layout_frame,
                               common_average_reference)

hf5.flush()
hf5.close()
print("Modified electrode arrays written to HDF5 file after "
      "subtracting the common average reference")

# Calculate post-CAR correlations and generate comparison plots
print("\nCalculating post-CAR correlations for comparison...")
try:
    # Get n_corr_samples from params if available
    if hasattr(metadata_handler, 'params_dict') and metadata_handler.params_dict:
        n_corr_samples = metadata_handler.params_dict.get(
            'qa_params', {}).get('n_corr_samples', 10000)
    else:
        n_corr_samples = 10000

    # Calculate post-CAR correlation matrix
    post_corr_mat, post_chan_names = calculate_post_car_correlations(
        metadata_handler.hdf5_name,
        electrode_layout_frame,
        n_corr_samples=n_corr_samples
    )

    # Filter to valid electrodes (same as pre-CAR matrix)
    post_corr_xarray = xr.DataArray(
        post_corr_mat,
        coords={
            'channel_1': post_chan_names,
            'channel_2': post_chan_names
        },
        dims=['channel_1', 'channel_2']
    )
    post_corr_xarray = post_corr_xarray.sel(
        channel_1=valid_electrode_inds, channel_2=valid_electrode_inds)
    post_corr_mat_filtered = post_corr_xarray.values

    # Make symmetric and handle NaN
    post_corr_mat_filtered[np.isnan(post_corr_mat_filtered)] = 0
    post_corr_mat_filtered = (
        post_corr_mat_filtered + post_corr_mat_filtered.T) / 2
    np.fill_diagonal(post_corr_mat_filtered, 1)

    # Save post-CAR correlation matrix
    post_corr_xarray_out = xr.DataArray(
        post_corr_mat_filtered,
        coords={
            'channel_1': valid_electrode_inds,
            'channel_2': valid_electrode_inds
        },
        dims=['channel_1', 'channel_2']
    )
    post_corr_xarray_out.to_netcdf(os.path.join(
        plots_dir, 'channel_corr_mat_post_car.nc'))

    # Generate comparison plots
    plot_car_correlation_comparison(
        pre_corr_mat=corr_mat,
        post_corr_mat=post_corr_mat_filtered,
        electrode_layout_frame=electrode_layout_frame,
        plot_dir=plots_dir,
        valid_electrode_inds=valid_electrode_inds
    )
    print("CAR correlation comparison plots saved to QA_output/")

except Exception as e:
    print(f"Warning: Could not generate CAR correlation comparison: {e}")
    import traceback
    traceback.print_exc()

# Write successful execution to log
if not testing_bool:
    this_pipeline_check.write_to_log(script_path, 'completed')
