"""
Data extraction script for GLM fitting.

This script runs in the blech_clust conda environment and extracts spike data
using ephys_data. The extracted data is saved as numpy files for the GLM
fitting script to load.

This script is called by infer_glm_rates.py and should not be run directly.
"""

import sys
import os
import json
import numpy as np

def main():
    if len(sys.argv) != 2:
        print("Usage: _glm_extract_data.py <params_path>")
        sys.exit(1)
    
    params_path = sys.argv[1]
    
    # Load parameters
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    data_dir = params['data_dir']
    temp_dir = params['temp_dir']
    time_lims = params['time_lims']
    bin_size = params['bin_size']
    
    # Add blech_clust to path
    script_path = os.path.abspath(__file__)
    blech_clust_path = os.path.dirname(os.path.dirname(script_path))
    sys.path.insert(0, blech_clust_path)
    
    # Import blech_clust modules
    from blech_clust.utils.blech_utils import imp_metadata, pipeline_graph_check
    from blech_clust.utils.ephys_data import ephys_data
    
    print(f"Extracting data from: {data_dir}")
    
    # Pipeline check
    this_pipeline_check = pipeline_graph_check(data_dir)
    this_pipeline_check.check_previous(script_path)
    this_pipeline_check.write_to_log(script_path, 'attempted')
    
    # Load data
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
    
    print(f"  Tastes: {n_tastes}")
    print(f"  Neurons: {n_neurons}")
    print(f"  Regions: {data.region_names}")
    
    # Process and save spike data for each taste
    spike_data = {}
    for taste_idx, taste_spikes in enumerate(data.spikes):
        # Cut to time limits
        taste_spikes = taste_spikes[..., time_lims[0]:time_lims[1]]
        
        # Bin spikes
        n_trials, n_neurons_taste, n_time = taste_spikes.shape
        n_bins = n_time // bin_size
        trimmed = taste_spikes[..., :n_bins * bin_size]
        binned = trimmed.reshape(n_trials, n_neurons_taste, n_bins, bin_size).sum(axis=-1)
        
        spike_data[f'taste_{taste_idx}'] = binned
        print(f"  Taste {taste_idx}: {binned.shape}")
    
    # Save extracted data
    extracted_data = {
        'spike_data': spike_data,
        'region_names': list(data.region_names),
        'region_units': {name: list(units) for name, units in region_dict.items()},
        'region_vec': region_vec.tolist(),
        'n_tastes': n_tastes,
        'n_neurons': n_neurons,
        'hdf5_path': data.hdf5_path,
    }
    
    # Save as numpy file
    output_file = os.path.join(temp_dir, 'extracted_data.npz')
    np.savez(output_file, **{
        'spike_data_keys': list(spike_data.keys()),
        'region_names': np.array(data.region_names, dtype=object),
        'region_vec': np.array(region_vec, dtype=object),
        'n_tastes': n_tastes,
        'n_neurons': n_neurons,
        'hdf5_path': data.hdf5_path,
    })
    
    # Save spike arrays separately (npz doesn't handle nested dicts well)
    for key, arr in spike_data.items():
        np.save(os.path.join(temp_dir, f'{key}.npy'), arr)
    
    # Save region units as JSON
    with open(os.path.join(temp_dir, 'region_units.json'), 'w') as f:
        json.dump({name: list(map(int, units)) for name, units in region_dict.items()}, f)
    
    print(f"\nData saved to: {temp_dir}")
    
    # Write to pipeline log
    this_pipeline_check.write_to_log(script_path, 'completed')


if __name__ == '__main__':
    main()
