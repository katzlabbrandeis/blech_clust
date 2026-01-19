"""
GLM fitting script using nemos.

This script runs in the nemos virtual environment and fits GLM models to
spike data extracted by _glm_extract_data.py.

This script is called by infer_glm_rates.py and should not be run directly.
"""

import sys
import os
import json
import numpy as np
import pickle
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Import nemos
try:
    import nemos as nmo
except ImportError:
    print("ERROR: nemos not found. Install with: pip install nemos")
    sys.exit(1)


############################################################
# GLM fitting functions
############################################################

def compute_bits_per_spike(model, X, y, baseline_rate=None):
    """
    Compute bits per spike metric for model comparison.
    
    Bits per spike measures how much better the model predicts spikes
    compared to a baseline (homogeneous Poisson) model.
    """
    # Get model predictions
    pred_rate = model.predict(X)
    
    # Compute log-likelihood under model
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
        valid_mask: Boolean mask for valid samples
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
        n_coupling_basis = max(2, n_basis_funcs // 2)
        coupling_window = max(2, history_window_bins // 2)
        coupling_basis = nmo.basis.RaisedCosineLogConv(
            n_coupling_basis,
            window_size=coupling_window
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
# Main
############################################################

def main():
    if len(sys.argv) != 2:
        print("Usage: _glm_fit_models.py <params_path>")
        sys.exit(1)
    
    params_path = sys.argv[1]
    
    # Load parameters
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    data_dir = params['data_dir']
    temp_dir = params['temp_dir']
    output_path = params['output_path']
    bin_size = params['bin_size']
    history_window = params['history_window']
    n_basis_funcs = params['n_basis_funcs']
    time_lims = params['time_lims']
    include_coupling = params['include_coupling']
    separate_tastes = params['separate_tastes']
    separate_regions = params['separate_regions']
    retrain = params['retrain']
    
    history_window_bins = history_window // bin_size
    
    # Stimulus time (typically at 2000ms)
    stim_time = 2000
    stim_bin = (stim_time - time_lims[0]) // bin_size
    
    # Setup directories
    artifacts_dir = os.path.join(output_path, 'artifacts')
    plots_dir = os.path.join(output_path, 'plots')
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load extracted data
    print("Loading extracted spike data...")
    extracted = np.load(os.path.join(temp_dir, 'extracted_data.npz'), allow_pickle=True)
    
    spike_data_keys = extracted['spike_data_keys']
    region_names = list(extracted['region_names'])
    region_vec = list(extracted['region_vec'])
    n_tastes = int(extracted['n_tastes'])
    n_neurons = int(extracted['n_neurons'])
    hdf5_path = str(extracted['hdf5_path'])
    
    # Load region units
    with open(os.path.join(temp_dir, 'region_units.json'), 'r') as f:
        region_dict = json.load(f)
    
    # Load spike arrays
    spike_data = {}
    for key in spike_data_keys:
        spike_data[key] = np.load(os.path.join(temp_dir, f'{key}.npy'))
    
    print(f"  Tastes: {n_tastes}")
    print(f"  Neurons: {n_neurons}")
    print(f"  Regions: {region_names}")
    
    # Save parameters
    with open(os.path.join(artifacts_dir, 'params.json'), 'w') as f:
        json.dump(params, f, indent=2)
    
    # Results storage
    results = {
        'pred_firing': [],
        'binned_spikes': [],
        'bits_per_spike': [],
        'taste_idx': [],
        'region_name': [],
        'neuron_idx': [],
    }
    
    # Process each taste
    for taste_idx in range(n_tastes):
        print(f"\n=== Processing taste {taste_idx} ===")
        
        binned = spike_data[f'taste_{taste_idx}']
        n_trials, n_neurons_taste, n_bins = binned.shape
        
        print(f"  Trials: {n_trials}, Neurons: {n_neurons_taste}, Bins: {n_bins}")
        
        # Determine which neurons to process
        if separate_regions:
            regions_to_process = [(name, units) for name, units in region_dict.items() 
                                  if name.lower() != 'none']
        else:
            regions_to_process = [('all', list(range(n_neurons_taste)))]
        
        for region_name, region_units in regions_to_process:
            print(f"  Region: {region_name} ({len(region_units)} neurons)")
            
            for neuron_idx in region_units:
                # Determine coupling neurons
                if include_coupling:
                    other_neurons = [i for i in region_units if i != neuron_idx]
                else:
                    other_neurons = None
                
                # Model save path
                model_name = f'taste_{taste_idx}_region_{region_name}_neuron_{neuron_idx}'
                model_path = os.path.join(artifacts_dir, f'{model_name}.pkl')
                
                if os.path.exists(model_path) and not retrain:
                    print(f"    Loading existing model for neuron {neuron_idx}")
                    with open(model_path, 'rb') as f:
                        saved = pickle.load(f)
                    model = saved['model']
                    X_valid = saved['X']
                    y_valid = saved['y']
                    feature_info = saved['feature_info']
                    valid_mask = saved['valid_mask']
                else:
                    print(f"    Fitting GLM for neuron {neuron_idx}")
                    model, X_valid, y_valid, feature_info, valid_mask = fit_glm_single_neuron(
                        binned,
                        neuron_idx,
                        history_window_bins,
                        n_basis_funcs,
                        stim_bin=stim_bin,
                        other_neurons=other_neurons,
                        include_coupling=include_coupling,
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
                results['taste_idx'].append(taste_idx)
                results['region_name'].append(region_name)
                results['neuron_idx'].append(neuron_idx)
    
    ############################################################
    # Generate plots
    ############################################################
    
    print("\n=== Generating plots ===")
    
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
    os.makedirs(ind_plot_dir, exist_ok=True)
    
    print("Plotting individual neuron predictions...")
    for i in range(min(10, len(results['pred_firing']))):
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
    
    print("\n=== Writing results to HDF5 ===")
    
    # Import tables here to write HDF5
    import tables
    
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
    
    print(f"\nResults saved to {hdf5_path}")
    print(f"Plots saved to {plots_dir}")
    print(f"Summary saved to {os.path.join(output_path, 'bits_per_spike_summary.csv')}")
    
    # Print summary statistics
    print("\n=== Summary ===")
    print(f"Total neurons processed: {len(results['pred_firing'])}")
    print(f"Mean bits per spike: {np.mean(results['bits_per_spike']):.4f}")
    print(f"Std bits per spike: {np.std(results['bits_per_spike']):.4f}")


if __name__ == '__main__':
    main()
