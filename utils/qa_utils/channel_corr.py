"""
This module provides utilities for quality assurance of channel data, focusing on correlation analysis between channels.

- `get_all_channels(hf5_path, electrode_layout_frame, n_corr_samples=10000)`: Extracts all channels from an HDF5 file, specifically from nodes 'raw' and 'raw_emg'. It returns the channel data, their names, and CAR group labels, using a specified number of samples for correlation calculation.
- `intra_corr(X)`: Computes the correlation matrix for all channels in the input array `X`, using Pearson correlation. It returns a matrix of correlation coefficients.
- `gen_corr_output(corr_mat, plot_dir, threshold=0.9, chan_labels=None)`: Generates and saves plots of the raw and thresholded correlation matrices. It also outputs a table of thresholded correlation values and logs warnings for channels with correlations above the specified threshold.
"""

import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr, zscore
from itertools import combinations
import matplotlib.pyplot as plt
import pandas as pd
import tables
import os


def get_all_channels(
    hf5_path,
    electrode_layout_frame=None,
    n_corr_samples=10000,
):
    """
    Get all channels in a file from nodes ['raw','raw_emg']

    Input:
            hf5_path: str, path to hdf5 file
            electrode_layout_frame: pd.DataFrame, electrode layout with CAR_group column (optional)
            n_corr_samples: int, number of samples to use for correlation calculation

    Output:
            all_chans: np.array (n_chans, n_samples)
            chan_names: np.array (n_chans,)
            chan_labels: list of str, labels combining CAR group and channel number (or None if no layout provided)
    """
    hf5 = tables.open_file(hf5_path, 'r')
    raw = hf5.list_nodes('/raw')
    raw_emg = hf5.list_nodes('/raw_emg')
    if len(raw) > 0:
        n_samples = raw[0].shape[0]
    elif len(raw_emg) > 0:
        n_samples = raw_emg[0].shape[0]
    else:
        raise ValueError('No data found in either /raw or /raw_emg')
    sample_inds = np.random.choice(n_samples, n_corr_samples, replace=False)
    all_chans = []
    chan_names = []
    for node in [raw, raw_emg]:
        for chan in tqdm(node):
            all_chans.append(chan[:][sample_inds])
            if 'emg' in chan._v_name:
                chan_names.append(int(chan._v_name.split('emg')[-1]))
            else:
                chan_names.append(int(chan._v_name.split('electrode')[-1]))
    hf5.close()
    # Sort everything by channel number
    sort_order = np.argsort(chan_names)
    chan_names = np.array(chan_names)[sort_order]
    all_chans = np.stack(all_chans)[sort_order]

    # Build channel labels with CAR group names if layout provided
    chan_labels = None
    if electrode_layout_frame is not None:
        chan_labels = []
        for chan_num in chan_names:
            row = electrode_layout_frame.loc[
                electrode_layout_frame['electrode_ind'] == chan_num]
            if len(row) > 0 and 'CAR_group' in row.columns:
                car_group = row['CAR_group'].values[0]
                chan_labels.append(f"{car_group}:{chan_num}")
            else:
                chan_labels.append(str(chan_num))

    return all_chans, np.array(chan_names), chan_labels


def intra_corr(X):
    """
    Correlations between all channels in X

    Input:
            X: np.array (n_chans, n_samples)

    Output:
            corr_mat: np.array (n_chans, n_chans)
    """
    X = zscore(X, axis=-1)
    inds = list(combinations(range(X.shape[0]), 2))
    corr_mat = np.zeros((X.shape[0], X.shape[0]))
    for i, j in tqdm(inds):
        corr_mat[i, j] = pearsonr(X[i, :], X[j, :])[0]
        corr_mat[j, i] = np.nan
    return corr_mat


def gen_corr_output(corr_mat, plot_dir, threshold=0.9, chan_labels=None):
    """
    Generate a plot of the raw, and thresholded correlation matrices

    Input:
            corr_mat: np.array (n_chans, n_chans)
            plot_dir: str, directory to save plots
            threshold: float, correlation threshold for highlighting
            chan_labels: list of str, labels for each channel (e.g., "GC:0", "PC:16")

    Output:
            fig: matplotlib figure
    """
    thresh_corr = corr_mat.copy()
    thresh_corr[thresh_corr < threshold] = np.nan

    save_path = os.path.join(plot_dir, 'raw_channel_corr_plot.png')

    # Adjust figure size based on number of channels for readability
    n_chans = corr_mat.shape[0]
    fig_width = max(12, n_chans * 0.2)
    fig_height = max(6, n_chans * 0.1)
    fig, ax = plt.subplots(1, 2, figsize=(fig_width, fig_height))

    im = ax[0].imshow(corr_mat, cmap='jet', vmin=0, vmax=1)
    ax[0].set_title('Raw Correlation Matrix')
    ax[0].set_xlabel('Channel')
    ax[0].set_ylabel('Channel')
    fig.colorbar(im, ax=ax[0])

    im = ax[1].imshow(thresh_corr, cmap='jet')
    ax[1].set_title('Thresholded Correlation Matrix')
    ax[1].set_xlabel('Channel')
    ax[1].set_ylabel('Channel')
    ax[1].imshow(thresh_corr,
                 interpolation='nearest')
    cbar = fig.colorbar(im, ax=ax[1])

    # Add channel labels with CAR group names if provided
    if chan_labels is not None:
        for a in ax:
            a.set_xticks(range(len(chan_labels)))
            a.set_yticks(range(len(chan_labels)))
            a.set_xticklabels(chan_labels, rotation=90,
                              fontsize=max(4, 8 - n_chans // 20))
            a.set_yticklabels(chan_labels, fontsize=max(4, 8 - n_chans // 20))

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    # Also output a table with only the thresholded values
    upper_thresh_corr = thresh_corr.copy()
    # Convert to pd.DataFrame
    inds = np.array(list(np.ndindex(upper_thresh_corr.shape)))
    upper_thresh_frame = pd.DataFrame(
        dict(
            chan1=inds[:, 0],
            chan2=inds[:, 1],
            corr=upper_thresh_corr.flatten()
        )
    )
    upper_thresh_frame = upper_thresh_frame.dropna()
    upper_thresh_frame.to_csv(
        os.path.join(plot_dir, 'raw_channel_corr_table.txt'),
        index=False, sep='\t')

    # If there are any channels with a correlation above the threshold,
    # output a list of those channels to warnings.txt
    warnings = upper_thresh_frame.loc[upper_thresh_frame['corr'] > threshold]
    if len(warnings) > 0:
        with open(os.path.join(plot_dir, 'warnings.txt'), 'a') as f:
            print('', file=f)
            f.write('=== Channel Correlation Warnings ===\n')
            f.write('The following channels have a correlation above the threshold of {}:\n'.format(
                threshold))
            f.write('Correlation Threshold: {}\n'.format(threshold))
            print('', file=f)
            f.write(warnings.to_string())
            print('', file=f)
            f.write('=== End Channel Correlation Warnings ===\n')
            print('', file=f)
