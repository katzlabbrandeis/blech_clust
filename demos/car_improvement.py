"""
Demo script to compare CAR (Common Average Reference) methods.

Calculates and saves correlations between channels before and after CAR for:
1. User-specified CAR groups without scaling (original blech_clust method)
2. User-specified CAR groups with scaling
3. Clustering-based CAR groups with scaling

Outputs correlation comparison plots and statistics to demonstrate the
improvement from scaling and clustering approaches.

Usage:
    python demos/car_improvement.py <data_dir>

Arguments:
    data_dir: Directory containing the HDF5 file with raw electrode data
              and electrode layout CSV file.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tables
import xarray as xr
from tqdm import tqdm
from scipy.stats import pearsonr, zscore
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from itertools import combinations

try:
    from scipy.stats import median_abs_deviation as MAD
except ImportError:
    from scipy.stats import median_absolute_deviation as MAD

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blech_clust.utils.blech_utils import imp_metadata
from blech_clust.utils.qa_utils import channel_corr


def get_electrode_by_name(raw_electrodes, name):
    """Get electrode data by electrode index."""
    str_name = f"electrode{name:02}"
    wanted_electrode = [x for x in raw_electrodes if str_name in x._v_pathname][0]
    return wanted_electrode


def calculate_correlation_matrix(data_array):
    """
    Calculate correlation matrix for channels.
    
    Parameters
    ----------
    data_array : np.ndarray
        Shape (n_channels, n_samples)
    
    Returns
    -------
    corr_mat : np.ndarray
        Shape (n_channels, n_channels)
    """
    n_chans = data_array.shape[0]
    X = zscore(data_array, axis=-1)
    inds = list(combinations(range(n_chans), 2))
    corr_mat = np.zeros((n_chans, n_chans))
    for i, j in inds:
        corr_mat[i, j] = pearsonr(X[i, :], X[j, :])[0]
        corr_mat[j, i] = corr_mat[i, j]
    np.fill_diagonal(corr_mat, 1)
    return corr_mat


def apply_car_no_scaling(raw_data, electrode_layout_frame):
    """
    Apply CAR without scaling (original blech_clust method).
    
    Simply averages channels in each CAR group and subtracts.
    
    Parameters
    ----------
    raw_data : np.ndarray
        Shape (n_channels, n_samples)
    electrode_layout_frame : pd.DataFrame
        Must have 'CAR_group' and 'electrode_ind' columns
    
    Returns
    -------
    referenced_data : np.ndarray
        Shape (n_channels, n_samples)
    """
    referenced_data = raw_data.copy()
    
    for group_name in electrode_layout_frame.CAR_group.unique():
        group_frame = electrode_layout_frame[electrode_layout_frame.CAR_group == group_name]
        group_indices = group_frame.index.values
        
        if len(group_indices) <= 1:
            continue
        
        # Simple average without scaling
        group_data = raw_data[group_indices, :]
        car = np.mean(group_data, axis=0)
        
        # Subtract CAR from each channel
        for idx in group_indices:
            referenced_data[idx, :] = raw_data[idx, :] - car
    
    return referenced_data


def apply_car_with_scaling(raw_data, electrode_layout_frame):
    """
    Apply CAR with channel scaling (normalize before averaging).
    
    Normalizes each channel by median and MAD before computing CAR,
    then converts back to original scale.
    
    Parameters
    ----------
    raw_data : np.ndarray
        Shape (n_channels, n_samples)
    electrode_layout_frame : pd.DataFrame
        Must have 'CAR_group' and 'electrode_ind' columns
    
    Returns
    -------
    referenced_data : np.ndarray
        Shape (n_channels, n_samples)
    """
    referenced_data = raw_data.copy()
    
    for group_name in electrode_layout_frame.CAR_group.unique():
        group_frame = electrode_layout_frame[electrode_layout_frame.CAR_group == group_name]
        group_indices = group_frame.index.values
        
        if len(group_indices) <= 1:
            continue
        
        # Normalize each channel and compute CAR
        normalized_sum = np.zeros(raw_data.shape[1])
        for idx in group_indices:
            channel_data = raw_data[idx, :]
            channel_median = np.median(channel_data[::100])
            channel_mad = MAD(channel_data[::100])
            if channel_mad > 0:
                normalized = (channel_data - channel_median) / channel_mad
            else:
                normalized = channel_data - channel_median
            normalized_sum += normalized
        
        car_normalized = normalized_sum / len(group_indices)
        
        # Subtract normalized CAR and convert back to original scale
        for idx in group_indices:
            channel_data = raw_data[idx, :]
            channel_median = np.median(channel_data[::100])
            channel_mad = MAD(channel_data[::100])
            if channel_mad > 0:
                normalized = (channel_data - channel_median) / channel_mad
                referenced_normalized = normalized - car_normalized
                referenced_data[idx, :] = (referenced_normalized * channel_mad) + channel_median
            else:
                referenced_data[idx, :] = channel_data - car_normalized
    
    return referenced_data


def cluster_electrodes(corr_mat, max_clusters=10):
    """
    Cluster electrodes based on correlation matrix using K-Means with BIC.
    
    Parameters
    ----------
    corr_mat : np.ndarray
        Correlation matrix (n_channels, n_channels)
    max_clusters : int
        Maximum number of clusters to try
    
    Returns
    -------
    predictions : np.ndarray
        Cluster assignments for each channel
    """
    n_chans = corr_mat.shape[0]
    max_possible = min(max_clusters, n_chans - 1)
    
    if max_possible < 2:
        return np.zeros(n_chans, dtype=int)
    
    # PCA on correlation matrix
    n_components = min(3, n_chans - 1)
    pca = PCA(n_components=n_components)
    features = pca.fit_transform(corr_mat)
    
    # Find optimal k using BIC
    best_bic = np.inf
    best_predictions = None
    
    for k in range(1, max_possible + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(features)
        
        # Calculate BIC
        n_samples, n_features = features.shape
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        
        distances = np.array([np.sum((features[i] - centers[labels[i]])**2) 
                              for i in range(n_samples)])
        variance = np.sum(distances) / max(n_samples - k, 1)
        variance = max(variance, 1e-10)
        
        log_likelihood = -0.5 * (n_samples * np.log(2 * np.pi * variance) + n_samples)
        n_params = k * n_features + 1
        bic = -2 * log_likelihood + n_params * np.log(n_samples)
        
        if bic < best_bic:
            best_bic = bic
            best_predictions = labels
    
    return best_predictions


def apply_car_clustered_scaling(raw_data, corr_mat, max_clusters=10):
    """
    Apply CAR with clustering and scaling.
    
    Clusters channels based on correlation, then applies scaled CAR
    within each cluster.
    
    Parameters
    ----------
    raw_data : np.ndarray
        Shape (n_channels, n_samples)
    corr_mat : np.ndarray
        Pre-CAR correlation matrix
    max_clusters : int
        Maximum clusters for K-Means
    
    Returns
    -------
    referenced_data : np.ndarray
        Shape (n_channels, n_samples)
    cluster_assignments : np.ndarray
        Cluster assignment for each channel
    """
    # Cluster based on correlation
    cluster_assignments = cluster_electrodes(corr_mat, max_clusters)
    
    referenced_data = raw_data.copy()
    
    for cluster_id in np.unique(cluster_assignments):
        cluster_indices = np.where(cluster_assignments == cluster_id)[0]
        
        if len(cluster_indices) <= 1:
            continue
        
        # Normalize each channel and compute CAR
        normalized_sum = np.zeros(raw_data.shape[1])
        for idx in cluster_indices:
            channel_data = raw_data[idx, :]
            channel_median = np.median(channel_data[::100])
            channel_mad = MAD(channel_data[::100])
            if channel_mad > 0:
                normalized = (channel_data - channel_median) / channel_mad
            else:
                normalized = channel_data - channel_median
            normalized_sum += normalized
        
        car_normalized = normalized_sum / len(cluster_indices)
        
        # Subtract normalized CAR and convert back
        for idx in cluster_indices:
            channel_data = raw_data[idx, :]
            channel_median = np.median(channel_data[::100])
            channel_mad = MAD(channel_data[::100])
            if channel_mad > 0:
                normalized = (channel_data - channel_median) / channel_mad
                referenced_normalized = normalized - car_normalized
                referenced_data[idx, :] = (referenced_normalized * channel_mad) + channel_median
            else:
                referenced_data[idx, :] = channel_data - car_normalized
    
    return referenced_data, cluster_assignments


def extract_upper_triangle(corr_mat):
    """Extract upper triangle values (excluding diagonal)."""
    n = corr_mat.shape[0]
    triu_inds = np.triu_indices(n, k=1)
    return corr_mat[triu_inds]


def plot_correlation_comparison(pre_corr, post_corrs, method_names, output_dir):
    """
    Generate comparison plots for different CAR methods.
    
    Parameters
    ----------
    pre_corr : np.ndarray
        Pre-CAR correlation matrix
    post_corrs : list of np.ndarray
        Post-CAR correlation matrices for each method
    method_names : list of str
        Names for each method
    output_dir : str
        Directory to save plots
    """
    n_methods = len(post_corrs)
    n_chans = pre_corr.shape[0]
    
    # Extract upper triangle values
    pre_vals = extract_upper_triangle(pre_corr)
    post_vals_list = [extract_upper_triangle(pc) for pc in post_corrs]
    
    # 1. Heatmaps comparison
    fig_width = max(20, n_chans * 0.15 * (n_methods + 2))
    fig_height = max(6, n_chans * 0.1)
    fig, axes = plt.subplots(1, n_methods + 1, figsize=(fig_width, fig_height))
    
    # Pre-CAR
    im = axes[0].imshow(pre_corr, cmap='jet', vmin=0, vmax=1)
    axes[0].set_title('Before CAR', fontsize=12)
    axes[0].set_xlabel('Channel')
    axes[0].set_ylabel('Channel')
    fig.colorbar(im, ax=axes[0], shrink=0.8)
    
    # Post-CAR for each method
    for i, (post_corr, name) in enumerate(zip(post_corrs, method_names)):
        im = axes[i + 1].imshow(post_corr, cmap='jet', vmin=0, vmax=1)
        axes[i + 1].set_title(f'After CAR:\n{name}', fontsize=10)
        axes[i + 1].set_xlabel('Channel')
        axes[i + 1].set_ylabel('Channel')
        fig.colorbar(im, ax=axes[i + 1], shrink=0.8)
    
    fig.suptitle('Channel Correlation Matrices: CAR Method Comparison', fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'car_comparison_heatmaps.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # 2. Paired points plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subsample for visibility
    n_pairs = len(pre_vals)
    max_plot = 500
    if n_pairs > max_plot:
        sample_inds = np.random.choice(n_pairs, max_plot, replace=False)
    else:
        sample_inds = np.arange(n_pairs)
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_methods))
    
    # Plot paired points
    x_positions = np.arange(n_methods + 1)
    jitter = 0.08
    
    for i in range(len(sample_inds)):
        idx = sample_inds[i]
        y_vals = [pre_vals[idx]] + [pv[idx] for pv in post_vals_list]
        axes[0].plot(x_positions, y_vals, color='gray', alpha=0.05, linewidth=0.5)
    
    # Scatter points
    axes[0].scatter(np.zeros(len(sample_inds)) + np.random.uniform(-jitter, jitter, len(sample_inds)),
                    pre_vals[sample_inds], alpha=0.3, s=10, color='black', label='Before CAR')
    
    for i, (post_vals, name) in enumerate(zip(post_vals_list, method_names)):
        axes[0].scatter(np.ones(len(sample_inds)) * (i + 1) + np.random.uniform(-jitter, jitter, len(sample_inds)),
                        post_vals[sample_inds], alpha=0.3, s=10, color=colors[i], label=name)
    
    axes[0].set_xticks(x_positions)
    axes[0].set_xticklabels(['Before'] + [f'Method {i+1}' for i in range(n_methods)], rotation=45, ha='right')
    axes[0].set_ylabel('Correlation')
    axes[0].set_title('Paired Channel Correlations')
    axes[0].set_ylim(-0.1, 1.1)
    axes[0].legend(loc='upper right', fontsize=8)
    
    # Add mean lines
    axes[0].axhline(np.mean(pre_vals), color='black', linestyle='--', alpha=0.7)
    for i, post_vals in enumerate(post_vals_list):
        axes[0].axhline(np.mean(post_vals), color=colors[i], linestyle='--', alpha=0.7)
    
    # Histogram of correlation changes
    for i, (post_vals, name) in enumerate(zip(post_vals_list, method_names)):
        diff_vals = pre_vals - post_vals
        axes[1].hist(diff_vals, bins=50, alpha=0.5, color=colors[i], 
                     label=f'{name}\nMean: {np.mean(diff_vals):.3f}', edgecolor='black', linewidth=0.5)
    
    axes[1].axvline(0, color='black', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Correlation Change (Before - After)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Distribution of Correlation Changes')
    axes[1].legend(fontsize=8)
    
    fig.suptitle('CAR Effect on Channel Correlations', fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'car_comparison_paired_plot.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # 3. Summary statistics bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    mean_pre = np.mean(pre_vals)
    mean_posts = [np.mean(pv) for pv in post_vals_list]
    pct_decreased = [np.sum(pre_vals > pv) / len(pre_vals) * 100 for pv in post_vals_list]
    
    x = np.arange(n_methods)
    width = 0.35
    
    bars1 = ax.bar(x - width/2, mean_posts, width, label='Mean Correlation After CAR', color=colors[:n_methods])
    ax.axhline(mean_pre, color='black', linestyle='--', linewidth=2, label=f'Mean Before CAR: {mean_pre:.3f}')
    
    ax.set_xlabel('CAR Method')
    ax.set_ylabel('Mean Correlation')
    ax.set_title('Mean Channel Correlation by CAR Method')
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=45, ha='right')
    ax.legend()
    
    # Add value labels
    for bar, val in zip(bars1, mean_posts):
        ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'car_comparison_summary.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return mean_pre, mean_posts, pct_decreased


def save_correlation_data(pre_corr, post_corrs, method_names, output_dir):
    """Save correlation matrices as NetCDF files."""
    # Save pre-CAR
    pre_xr = xr.DataArray(
        pre_corr,
        dims=['channel_1', 'channel_2'],
        name='correlation'
    )
    pre_xr.to_netcdf(os.path.join(output_dir, 'car_demo_pre_corr.nc'))
    
    # Save post-CAR for each method
    for post_corr, name in zip(post_corrs, method_names):
        safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').lower()
        post_xr = xr.DataArray(
            post_corr,
            dims=['channel_1', 'channel_2'],
            name='correlation'
        )
        post_xr.to_netcdf(os.path.join(output_dir, f'car_demo_post_corr_{safe_name}.nc'))


def main(data_dir, n_samples=10000, max_clusters=10):
    """
    Main function to run CAR comparison demo.
    
    Parameters
    ----------
    data_dir : str
        Directory containing HDF5 and electrode layout files
    n_samples : int
        Number of samples to use for correlation calculation
    max_clusters : int
        Maximum clusters for clustering method
    """
    print(f"Loading data from {data_dir}")
    
    # Load metadata
    metadata_handler = imp_metadata([[], data_dir])
    hf5_path = metadata_handler.hdf5_name
    electrode_layout = metadata_handler.layout.copy()
    
    # Filter out EMG and none channels
    valid_mask = ~(electrode_layout.CAR_group.str.contains('emg', case=False, na=False) |
                   electrode_layout.CAR_group.str.contains('none', case=False, na=False))
    electrode_layout_filtered = electrode_layout[valid_mask].copy()
    electrode_layout_filtered = electrode_layout_filtered.reset_index(drop=True)
    
    print(f"Processing {len(electrode_layout_filtered)} channels")
    print(f"CAR groups: {electrode_layout_filtered.CAR_group.unique()}")
    
    # Load raw data
    print("Loading raw electrode data...")
    hf5 = tables.open_file(hf5_path, 'r')
    raw_electrodes = hf5.list_nodes('/raw')
    
    # Get recording length and sample indices
    rec_length = raw_electrodes[0][:].shape[0]
    sample_inds = np.random.choice(rec_length, min(n_samples, rec_length), replace=False)
    sample_inds = np.sort(sample_inds)
    
    # Load data for valid electrodes
    n_chans = len(electrode_layout_filtered)
    raw_data = np.zeros((n_chans, len(sample_inds)))
    
    for i, row in tqdm(electrode_layout_filtered.iterrows(), total=n_chans, desc="Loading channels"):
        electrode_ind = row['electrode_ind']
        str_name = f"electrode{electrode_ind:02}"
        electrode_node = [x for x in raw_electrodes if str_name in x._v_pathname][0]
        raw_data[i, :] = electrode_node[:][sample_inds]
    
    hf5.close()
    
    # Calculate pre-CAR correlation
    print("Calculating pre-CAR correlations...")
    pre_corr = calculate_correlation_matrix(raw_data)
    
    # Apply CAR methods
    print("\nApplying CAR Method 1: No scaling (original)...")
    data_no_scaling = apply_car_no_scaling(raw_data, electrode_layout_filtered)
    
    print("Applying CAR Method 2: With scaling...")
    data_with_scaling = apply_car_with_scaling(raw_data, electrode_layout_filtered)
    
    print("Applying CAR Method 3: Clustering + scaling...")
    data_clustered, cluster_assignments = apply_car_clustered_scaling(
        raw_data, pre_corr, max_clusters=max_clusters
    )
    n_clusters = len(np.unique(cluster_assignments))
    print(f"  Found {n_clusters} clusters")
    
    # Calculate post-CAR correlations
    print("\nCalculating post-CAR correlations...")
    post_corr_no_scaling = calculate_correlation_matrix(data_no_scaling)
    post_corr_with_scaling = calculate_correlation_matrix(data_with_scaling)
    post_corr_clustered = calculate_correlation_matrix(data_clustered)
    
    # Create output directory
    output_dir = os.path.join(data_dir, 'QA_output', 'car_demo')
    os.makedirs(output_dir, exist_ok=True)
    
    # Method names
    method_names = [
        'No Scaling (Original)',
        'With Scaling',
        f'Clustering + Scaling ({n_clusters} clusters)'
    ]
    
    post_corrs = [post_corr_no_scaling, post_corr_with_scaling, post_corr_clustered]
    
    # Generate plots
    print("\nGenerating comparison plots...")
    mean_pre, mean_posts, pct_decreased = plot_correlation_comparison(
        pre_corr, post_corrs, method_names, output_dir
    )
    
    # Save correlation data
    print("Saving correlation data...")
    save_correlation_data(pre_corr, post_corrs, method_names, output_dir)
    
    # Print summary
    print("\n" + "=" * 60)
    print("CAR COMPARISON SUMMARY")
    print("=" * 60)
    print(f"Mean correlation before CAR: {mean_pre:.4f}")
    print()
    for name, mean_post, pct in zip(method_names, mean_posts, pct_decreased):
        print(f"{name}:")
        print(f"  Mean correlation after CAR: {mean_post:.4f}")
        print(f"  Correlation reduction: {mean_pre - mean_post:.4f}")
        print(f"  Pairs with decreased correlation: {pct:.1f}%")
        print()
    print("=" * 60)
    print(f"\nPlots saved to: {output_dir}")
    
    # Save summary to text file
    summary_path = os.path.join(output_dir, 'car_comparison_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("CAR COMPARISON SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Data directory: {data_dir}\n")
        f.write(f"Number of channels: {n_chans}\n")
        f.write(f"Samples used: {len(sample_inds)}\n\n")
        f.write(f"Mean correlation before CAR: {mean_pre:.4f}\n\n")
        for name, mean_post, pct in zip(method_names, mean_posts, pct_decreased):
            f.write(f"{name}:\n")
            f.write(f"  Mean correlation after CAR: {mean_post:.4f}\n")
            f.write(f"  Correlation reduction: {mean_pre - mean_post:.4f}\n")
            f.write(f"  Pairs with decreased correlation: {pct:.1f}%\n\n")
    
    print(f"Summary saved to: {summary_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare CAR methods by calculating channel correlations'
    )
    parser.add_argument('data_dir', type=str,
                        help='Directory containing HDF5 and electrode layout files')
    parser.add_argument('--n_samples', type=int, default=10000,
                        help='Number of samples for correlation calculation (default: 10000)')
    parser.add_argument('--max_clusters', type=int, default=10,
                        help='Maximum clusters for clustering method (default: 10)')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.data_dir):
        print(f"Error: {args.data_dir} is not a valid directory")
        sys.exit(1)
    
    main(args.data_dir, n_samples=args.n_samples, max_clusters=args.max_clusters)
