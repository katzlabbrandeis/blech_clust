"""
GLM-based firing rate estimation using nemos.

This module fits Generalized Linear Models (GLMs) to electrophysiological spike data
to infer firing rates. The GLM can incorporate:
- Spike history (autoregressive effects)
- Stimulus timing
- Coupling between simultaneously recorded neurons

Model comparison is provided via bits per spike metric against binned rates.

Usage:
    python infer_glm_rates.py <data_dir> [options]

Options:
    --bin_size: Bin size in ms for spike binning (default: 25)
    --history_window: History window in ms for autoregressive effects (default: 250)
    --n_basis_funcs: Number of basis functions for history filter (default: 8)
    --time_lims: Time limits for analysis [start, end] in ms (default: [1500, 4500])
    --include_coupling: Include coupling between neurons (default: False)
    --separate_tastes: Fit separate models for each taste (default: False)
    --separate_regions: Fit separate models for each region (default: False)
    --retrain: Force retraining even if model exists (default: False)
"""

import argparse
import os

test_mode = False
if test_mode:
    print('====================')
    print('Running in test mode')
    print('====================')
    data_dir = '/home/abuzarmahmood/projects/blech_clust/pipeline_testing/test_data_handling/test_data/KM45_5tastes_210620_113227_new'
    script_path = '/home/abuzarmahmood/projects/blech_clust/utils/infer_glm_rates.py'
    blech_clust_path = os.path.dirname(os.path.dirname(script_path))
    args = argparse.Namespace(
        data_dir=data_dir,
        bin_size=25,
        history_window=250,
        n_basis_funcs=8,
        time_lims=[1500, 4500],
        include_coupling=False,
        separate_tastes=True,
        separate_regions=True,
        retrain=False,
    )
else:
    parser = argparse.ArgumentParser(
        description='Infer firing rates using GLM (nemos)')
    parser.add_argument('data_dir', help='Path to data directory')
    parser.add_argument('--bin_size', type=int, default=25,
                        help='Bin size in ms for spike binning (default: %(default)s)')
    parser.add_argument('--history_window', type=int, default=250,
                        help='History window in ms for autoregressive effects (default: %(default)s)')
    parser.add_argument('--n_basis_funcs', type=int, default=8,
                        help='Number of basis functions for history filter (default: %(default)s)')
    parser.add_argument('--time_lims', type=int, nargs=2, default=[1500, 4500],
                        help='Time limits for analysis [start, end] in ms (default: %(default)s)')
    parser.add_argument('--include_coupling', action='store_true',
                        help='Include coupling between neurons (default: %(default)s)')
    parser.add_argument('--separate_tastes', action='store_true',
                        help='Fit separate models for each taste (default: %(default)s)')
    parser.add_argument('--separate_regions', action='store_true',
                        help='Fit separate models for each region (default: %(default)s)')
    parser.add_argument('--retrain', action='store_true',
                        help='Force retraining of model (default: %(default)s)')

    args = parser.parse_args()
    data_dir = args.data_dir
    script_path = os.path.abspath(__file__)
    blech_clust_path = os.path.dirname(os.path.dirname(script_path))

############################################################
############################################################

import tables
import matplotlib.pyplot as plt
import numpy as np
import sys
from pprint import pprint
import json
from itertools import product
import pandas as pd
import pickle

sys.path.append(blech_clust_path)

from blech_clust.utils.blech_utils import imp_metadata, pipeline_graph_check
from blech_clust.utils.ephys_data import visualize as vz
from blech_clust.utils.ephys_data import ephys_data

# Import nemos for GLM fitting
try:
    import nemos as nmo
except ImportError:
    raise ImportError(
        'nemos is required for GLM fitting. Install with: pip install nemos'
    )

############################################################
############################################################


def compute_bits_per_spike(model, X, y, baseline_rate=None):
    """
    Compute bits per spike metric for model comparison.
    
    Bits per spike measures how much better the model predicts spikes
    compared to a baseline (homogeneous Poisson) model.
    
    Args:
        model: Fitted nemos GLM model
        X: Feature matrix
        y: Spike counts
        baseline_rate: Baseline firing rate (if None, uses mean rate)
    
    Returns:
        bits_per_spike: Information gain in bits per spike
    """
    # Get model predictions
    pred_rate = model.predict(X)
    
    # Compute log-likelihood under model
    # For Poisson: LL = y * log(rate) - rate - log(y!)
    # We ignore the log(y!) term as it cancels in comparison
    eps = 1e-10
    ll_model = np.sum(y * np.log(pred_rate + eps) - pred_rate)
    
    # Compute log-likelihood under baseline (homogeneous Poisson)
    if baseline_rate is None:
        baseline_rate = np.mean(y, axis=0, keepdims=True)
    ll_baseline = np.sum(y * np.log(baseline_rate + eps) - baseline_rate)
    
    # Bits per spike = (LL_model - LL_baseline) / (n_spikes * log(2))
    n_spikes = np.sum(y)
    if n_spikes > 0:
        bits_per_spike = (ll_model - ll_baseline) / (n_spikes * np.log(2))
    else:
        bits_per_spike = 0.0
    
    return bits_per_spike


def bin_spikes(spike_data, bin_size):
    """
    Bin spike data.
    
    Args:
        spike_data: Array of shape (trials, neurons, time)
        bin_size: Bin size in samples
    
    Returns:
        binned: Array of shape (trials, neurons, n_bins)
    """
    n_trials, n_neurons, n_time = spike_data.shape
    n_bins = n_time // bin_size
    trimmed = spike_data[..., :n_bins * bin_size]
    binned = trimmed.reshape(n_trials, n_neurons, n_bins, bin_size).sum(axis=-1)
    return binned


def create_stimulus_feature(n_samples, stim_bin, n_basis=5, window_size=10):
    """
    Create stimulus feature using raised cosine basis.
    
    Args:
        n_samples: Number of time samples
        stim_bin: Bin index of stimulus onset
        n_basis: Number of basis functions
        window_size: Window size for basis
    
    Returns:
        stim_features: Stimulus feature matrix
    """
    # Create stimulus indicator
    stim = np.zeros(n_samples)
    if 0 <= stim_bin < n_samples:
        stim[stim_bin] = 1.0
    
    # Convolve with raised cosine basis
    basis = nmo.basis.RaisedCosineLogConv(n_basis, window_size=window_size)
    stim_features = basis.compute_features(stim.reshape(-1, 1))
    
    return stim_features


def fit_glm_single_neuron(
    binned_spikes,
    neuron_idx,
    history_window_bins,
    n_basis_funcs,
    stim_bin=None,
    other_neurons=None,
    include_coupling=False,
):
    """
    Fit GLM for a single neuron.
    
    Args:
        binned_spikes: Binned spike counts (trials, neurons, time)
        neuron_idx: Index of target neuron
        history_window_bins: History window in bins
        n_basis_funcs: Number of basis functions
        stim_bin: Stimulus onset bin (optional)
        other_neurons: Indices of other neurons for coupling (optional)
        include_coupling: Whether to include coupling
    
    Returns:
        model: Fitted GLM
        X: Feature matrix
        y: Target spike counts
        feature_info: Dictionary with feature information
    """
    n_trials, n_neurons, n_bins = binned_spikes.shape
    
    # Reshape to (samples, neurons) by concatenating trials
    spikes_flat = binned_spikes.transpose(0, 2, 1).reshape(-1, n_neurons)
    
    # Target neuron spikes
    y = spikes_flat[:, neuron_idx]
    
    # Build feature matrix
    feature_list = []
    feature_info = {'names': [], 'slices': []}
    current_idx = 0
    
    # 1. Spike history basis for target neuron
    history_basis = nmo.basis.RaisedCosineLogConv(
        n_basis_funcs, 
        window_size=history_window_bins
    )
    history_features = history_basis.compute_features(
        spikes_flat[:, neuron_idx:neuron_idx+1]
    )
    feature_list.append(history_features)
    feature_info['names'].append('history')
    feature_info['slices'].append(slice(current_idx, current_idx + n_basis_funcs))
    current_idx += n_basis_funcs
    
    # 2. Stimulus features (if provided)
    if stim_bin is not None:
        n_stim_basis = 5
        stim_window = min(20, history_window_bins)
        
        # Create stimulus indicator for each trial
        stim_indicator = np.zeros((n_trials * n_bins,))
        for trial in range(n_trials):
            trial_stim_idx = trial * n_bins + stim_bin
            if 0 <= trial_stim_idx < len(stim_indicator):
                stim_indicator[trial_stim_idx] = 1.0
        
        stim_basis = nmo.basis.RaisedCosineLogConv(n_stim_basis, window_size=stim_window)
        stim_features = stim_basis.compute_features(stim_indicator.reshape(-1, 1))
        feature_list.append(stim_features)
        feature_info['names'].append('stimulus')
        feature_info['slices'].append(slice(current_idx, current_idx + n_stim_basis))
        current_idx += n_stim_basis
    
    # 3. Coupling from other neurons (if requested)
    if include_coupling and other_neurons is not None and len(other_neurons) > 0:
        coupling_basis = nmo.basis.RaisedCosineLogConv(
            n_basis_funcs // 2,  # Fewer basis functions for coupling
            window_size=history_window_bins // 2
        )
        for other_idx in other_neurons:
            coupling_features = coupling_basis.compute_features(
                spikes_flat[:, other_idx:other_idx+1]
            )
            feature_list.append(coupling_features)
            feature_info['names'].append(f'coupling_{other_idx}')
            n_coupling_feats = coupling_features.shape[1]
            feature_info['slices'].append(slice(current_idx, current_idx + n_coupling_feats))
            current_idx += n_coupling_feats
    
    # Concatenate all features
    X = np.hstack(feature_list)
    
    # Handle NaN values from convolution edges
    valid_mask = ~np.any(np.isnan(X), axis=1)
    X_valid = X[valid_mask]
    y_valid = y[valid_mask]
    
    # Fit GLM
    model = nmo.glm.GLM(
        regularizer=nmo.regularizer.Ridge(regularizer_strength=0.01)
    )
    model.fit(X_valid, y_valid)
    
    return model, X_valid, y_valid, feature_info, valid_mask


def predict_firing_rates(model, X, valid_mask, original_shape):
    """
    Generate firing rate predictions and reshape to original trial structure.
    
    Args:
        model: Fitted GLM
        X: Feature matrix (valid samples only)
        valid_mask: Boolean mask for valid samples
        original_shape: Original shape (n_trials, n_bins)
    
    Returns:
        pred_rates: Predicted firing rates (n_trials, n_bins)
    """
    n_trials, n_bins = original_shape
    
    # Predict on valid samples
    pred_valid = model.predict(X)
    
    # Reconstruct full array with NaN for invalid samples
    pred_full = np.full(n_trials * n_bins, np.nan)
    pred_full[valid_mask] = pred_valid
    
    # Reshape to (trials, time)
    pred_rates = pred_full.reshape(n_trials, n_bins)
    
    return pred_rates


############################################################
############################################################

if not test_mode:
    metadata_handler = imp_metadata([[], args.data_dir])
    this_pipeline_check = pipeline_graph_check(args.data_dir)
    this_pipeline_check.check_previous(script_path)
    this_pipeline_check.write_to_log(script_path, 'attempted')

# Setup output directories
output_path = os.path.join(data_dir, 'glm_output')
artifacts_dir = os.path.join(output_path, 'artifacts')
plots_dir = os.path.join(output_path, 'plots')

for dir_path in [output_path, artifacts_dir, plots_dir]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

print(f'Processing data from {data_dir}')
print(f'Parameters:')
print(f'  bin_size: {args.bin_size} ms')
print(f'  history_window: {args.history_window} ms')
print(f'  n_basis_funcs: {args.n_basis_funcs}')
print(f'  time_lims: {args.time_lims}')
print(f'  include_coupling: {args.include_coupling}')
print(f'  separate_tastes: {args.separate_tastes}')
print(f'  separate_regions: {args.separate_regions}')

# Save parameters
params_dict = {
    'bin_size': args.bin_size,
    'history_window': args.history_window,
    'n_basis_funcs': args.n_basis_funcs,
    'time_lims': args.time_lims,
    'include_coupling': args.include_coupling,
    'separate_tastes': args.separate_tastes,
    'separate_regions': args.separate_regions,
}
with open(os.path.join(artifacts_dir, 'params.json'), 'w') as f:
    json.dump(params_dict, f, indent=4)

############################################################
# Load data
############################################################

basename = os.path.basename(data_dir)
data = ephys_data.ephys_data(data_dir)
data.get_spikes()
data.get_region_units()

# Build region mapping
region_dict = dict(zip(data.region_names, data.region_units))
region_vec = np.zeros(len(np.concatenate(data.region_units)), dtype=object)
for region_name, unit_list in region_dict.items():
    region_vec[unit_list] = region_name

n_tastes = len(data.spikes)
n_neurons = data.spikes[0].shape[1]

print(f'Loaded {n_tastes} tastes, {n_neurons} neurons')
print(f'Regions: {data.region_names}')

############################################################
# Process data
############################################################

bin_size = args.bin_size
history_window = args.history_window
history_window_bins = history_window // bin_size
n_basis_funcs = args.n_basis_funcs
time_lims = args.time_lims

# Stimulus time (typically at 2000ms)
stim_time = 2000
stim_bin = (stim_time - time_lims[0]) // bin_size

# Determine processing groups
group_by_list = []
if args.separate_tastes:
    group_by_list.append('taste')
if args.separate_regions:
    group_by_list.append('region')

# Results storage
results = {
    'pred_firing': [],
    'binned_spikes': [],
    'bits_per_spike': [],
    'models': [],
    'taste_idx': [],
    'region_name': [],
    'neuron_idx': [],
}

# Process each taste
for taste_idx, taste_spikes in enumerate(data.spikes):
    print(f'\n=== Processing taste {taste_idx} ===')
    
    # Cut to time limits
    taste_spikes = taste_spikes[..., time_lims[0]:time_lims[1]]
    
    # Bin spikes
    binned = bin_spikes(taste_spikes, bin_size)
    n_trials, n_neurons_taste, n_bins = binned.shape
    
    print(f'  Trials: {n_trials}, Neurons: {n_neurons_taste}, Bins: {n_bins}')
    
    # Determine which neurons to process
    if args.separate_regions:
        regions_to_process = [(name, units) for name, units in region_dict.items() 
                              if name.lower() != 'none']
    else:
        regions_to_process = [('all', list(range(n_neurons_taste)))]
    
    for region_name, region_units in regions_to_process:
        print(f'  Region: {region_name} ({len(region_units)} neurons)')
        
        for neuron_idx in region_units:
            # Determine coupling neurons
            if args.include_coupling:
                other_neurons = [i for i in region_units if i != neuron_idx]
            else:
                other_neurons = None
            
            # Model save path
            model_name = f'taste_{taste_idx}_region_{region_name}_neuron_{neuron_idx}'
            model_path = os.path.join(artifacts_dir, f'{model_name}.pkl')
            
            if os.path.exists(model_path) and not args.retrain:
                print(f'    Loading existing model for neuron {neuron_idx}')
                with open(model_path, 'rb') as f:
                    saved = pickle.load(f)
                model = saved['model']
                X_valid = saved['X']
                y_valid = saved['y']
                feature_info = saved['feature_info']
                valid_mask = saved['valid_mask']
            else:
                print(f'    Fitting GLM for neuron {neuron_idx}')
                model, X_valid, y_valid, feature_info, valid_mask = fit_glm_single_neuron(
                    binned,
                    neuron_idx,
                    history_window_bins,
                    n_basis_funcs,
                    stim_bin=stim_bin,
                    other_neurons=other_neurons,
                    include_coupling=args.include_coupling,
                )
                
                # Save model
                with open(model_path, 'wb') as f:
                    pickle.dump({
                        'model': model,
                        'X': X_valid,
                        'y': y_valid,
                        'feature_info': feature_info,
                        'valid_mask': valid_mask,
                    }, f)
            
            # Predict firing rates
            pred_rates = predict_firing_rates(
                model, X_valid, valid_mask, (n_trials, n_bins)
            )
            
            # Compute bits per spike
            bps = compute_bits_per_spike(model, X_valid, y_valid)
            
            # Store results
            results['pred_firing'].append(pred_rates)
            results['binned_spikes'].append(binned[:, neuron_idx, :])
            results['bits_per_spike'].append(bps)
            results['models'].append(model)
            results['taste_idx'].append(taste_idx)
            results['region_name'].append(region_name)
            results['neuron_idx'].append(neuron_idx)

############################################################
# Generate plots
############################################################

print('\n=== Generating plots ===')

# Bits per spike summary
bps_df = pd.DataFrame({
    'taste': results['taste_idx'],
    'region': results['region_name'],
    'neuron': results['neuron_idx'],
    'bits_per_spike': results['bits_per_spike'],
})

# Save summary
bps_df.to_csv(os.path.join(output_path, 'bits_per_spike_summary.csv'), index=False)

# Plot bits per spike distribution
fig, ax = plt.subplots(figsize=(10, 6))
for region in bps_df['region'].unique():
    region_bps = bps_df[bps_df['region'] == region]['bits_per_spike']
    ax.hist(region_bps, alpha=0.5, label=region, bins=20)
ax.set_xlabel('Bits per Spike')
ax.set_ylabel('Count')
ax.set_title('GLM Model Performance: Bits per Spike')
ax.legend()
fig.savefig(os.path.join(plots_dir, 'bits_per_spike_distribution.png'), dpi=150)
plt.close(fig)

# Plot mean bits per spike by taste and region
fig, ax = plt.subplots(figsize=(10, 6))
bps_summary = bps_df.groupby(['taste', 'region'])['bits_per_spike'].mean().unstack()
bps_summary.plot(kind='bar', ax=ax)
ax.set_xlabel('Taste')
ax.set_ylabel('Mean Bits per Spike')
ax.set_title('GLM Performance by Taste and Region')
ax.legend(title='Region')
fig.savefig(os.path.join(plots_dir, 'bits_per_spike_by_taste_region.png'), dpi=150)
plt.close(fig)

# Plot example neuron predictions
ind_plot_dir = os.path.join(plots_dir, 'individual_neurons')
if not os.path.exists(ind_plot_dir):
    os.makedirs(ind_plot_dir)

print('Plotting individual neuron predictions...')
for i in range(min(10, len(results['pred_firing']))):  # Plot first 10 neurons
    taste_idx = results['taste_idx'][i]
    region = results['region_name'][i]
    neuron_idx = results['neuron_idx'][i]
    pred = results['pred_firing'][i]
    binned = results['binned_spikes'][i]
    bps = results['bits_per_spike'][i]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    # Raster of binned spikes
    axes[0].imshow(binned, aspect='auto', cmap='Greys', interpolation='none')
    axes[0].set_ylabel('Trial')
    axes[0].set_title(f'Binned Spikes - Taste {taste_idx}, {region}, Neuron {neuron_idx}')
    
    # Predicted rates
    axes[1].imshow(pred, aspect='auto', cmap='viridis', interpolation='none')
    axes[1].set_ylabel('Trial')
    axes[1].set_title(f'GLM Predicted Rates (bits/spike: {bps:.3f})')
    
    # Mean comparison
    time_bins = np.arange(pred.shape[1]) * bin_size + time_lims[0]
    axes[2].plot(time_bins, np.nanmean(binned, axis=0), 'k-', label='Binned', alpha=0.7)
    axes[2].plot(time_bins, np.nanmean(pred, axis=0), 'r-', label='GLM Pred', alpha=0.7)
    axes[2].axvline(stim_time, color='b', linestyle='--', label='Stimulus')
    axes[2].set_xlabel('Time (ms)')
    axes[2].set_ylabel('Firing Rate')
    axes[2].legend()
    axes[2].set_title('Mean Firing Rate Comparison')
    
    plt.tight_layout()
    fig.savefig(os.path.join(ind_plot_dir, f'taste_{taste_idx}_{region}_neuron_{neuron_idx}.png'), dpi=150)
    plt.close(fig)

############################################################
# Write results to HDF5
############################################################

print('\n=== Writing results to HDF5 ===')

hdf5_path = data.hdf5_path
with tables.open_file(hdf5_path, 'r+') as hf5:
    # Remove existing GLM output if present
    if '/glm_output' in hf5:
        hf5.remove_node('/glm_output', recursive=True)
    
    # Create GLM output group
    hf5.create_group('/', 'glm_output', 'GLM-based firing rate estimates')
    glm_grp = hf5.get_node('/glm_output')
    
    # Store parameters
    hf5.create_array(glm_grp, 'bin_size', np.array([bin_size]))
    hf5.create_array(glm_grp, 'history_window', np.array([history_window]))
    hf5.create_array(glm_grp, 'time_lims', np.array(time_lims))
    
    # Create regions group
    hf5.create_group('/glm_output', 'regions', 'Region-specific GLM output')
    regions_grp = hf5.get_node('/glm_output/regions')
    
    # Organize results by taste and region
    for i in range(len(results['pred_firing'])):
        taste_idx = results['taste_idx'][i]
        region = results['region_name'][i]
        neuron_idx = results['neuron_idx'][i]
        
        group_name = f'taste_{taste_idx}_region_{region}_neuron_{neuron_idx}'
        
        neuron_grp = hf5.create_group(regions_grp, group_name, 
                                       f'Taste {taste_idx}, Region {region}, Neuron {neuron_idx}')
        
        hf5.create_array(neuron_grp, 'pred_firing', results['pred_firing'][i])
        hf5.create_array(neuron_grp, 'binned_spikes', results['binned_spikes'][i])
        hf5.create_array(neuron_grp, 'bits_per_spike', np.array([results['bits_per_spike'][i]]))

print(f'\nResults saved to {hdf5_path}')
print(f'Plots saved to {plots_dir}')
print(f'Summary saved to {os.path.join(output_path, "bits_per_spike_summary.csv")}')

# Print summary statistics
print('\n=== Summary ===')
print(f'Total neurons processed: {len(results["pred_firing"])}')
print(f'Mean bits per spike: {np.mean(results["bits_per_spike"]):.4f}')
print(f'Std bits per spike: {np.std(results["bits_per_spike"]):.4f}')

# Write successful execution to log
if not test_mode:
    this_pipeline_check.write_to_log(script_path, 'completed')
