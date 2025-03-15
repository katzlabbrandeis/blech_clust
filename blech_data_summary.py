"""
blech_data_summary.py - Generate a summary of dataset characteristics and quality

This script analyzes data from a blech_clust experiment and generates a JSON summary file
containing both unit-level and session-level parameters, including:

1. Unit-Level Parameters:
   - Responsiveness p-values
   - Discriminability p-values
   - Palatability p-values
   - Dynamicity p-values

2. Session-Level Parameters:
   - Drift analysis results (pre-stimulus and post-stimulus)
   - Channel correlation violations
   - ELBO change point analysis results

The summary is useful for quickly assessing dataset quality and characteristics
without having to examine multiple individual analysis files.

Usage:
    python blech_data_summary.py <dir_name>

Arguments:
    dir_name : Directory containing the processed data files
"""

import os
import sys
import json
import glob
import numpy as np
import pandas as pd
import tables
import argparse
from tqdm import tqdm

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate summary of dataset characteristics')
parser.add_argument('dir_name', type=str, help='Directory containing data files')
args = parser.parse_args()

# Import necessary modules from blech_clust
script_path = os.path.realpath(__file__)
blech_clust_dir = os.path.dirname(script_path)
sys.path.append(blech_clust_dir)

from utils.blech_utils import imp_metadata, pipeline_graph_check
from utils.qa_utils.channel_corr import get_all_channels, intra_corr

def extract_unit_characteristics(dir_name):
    """
    Extract unit-level characteristics from aggregated_characteristics.csv
    
    Args:
        dir_name: Directory containing the data
        
    Returns:
        DataFrame with unit-level characteristics
    """
    characteristics_path = os.path.join(dir_name, 'aggregated_characteristics.csv')
    
    if not os.path.exists(characteristics_path):
        print(f"Warning: {characteristics_path} not found. Run blech_units_characteristics.py first.")
        return pd.DataFrame()
    
    # Load the characteristics data
    unit_data = pd.read_csv(characteristics_path)
    
    # Create a more structured summary by pivoting the data
    # Group by neuron_num and laser_tuple
    summary = []
    
    for (neuron, laser), group in unit_data.groupby(['neuron_num', 'laser_tuple']):
        neuron_summary = {
            'neuron_num': neuron,
            'laser_tuple': laser
        }
        
        # Extract values for each source type
        for source in group['Source'].unique():
            source_data = group[group['Source'] == source]
            if not source_data.empty:
                neuron_summary[f"{source}_sig"] = bool(source_data['sig'].iloc[0])
        
        summary.append(neuron_summary)
    
    return pd.DataFrame(summary)

def extract_drift_parameters(dir_name):
    """
    Extract drift analysis parameters from QA_output directory
    
    Args:
        dir_name: Directory containing the data
        
    Returns:
        Dictionary with drift analysis results
    """
    qa_dir = os.path.join(dir_name, 'QA_output')
    
    if not os.path.exists(qa_dir):
        print(f"Warning: {qa_dir} not found. Run drift_check.py first.")
        return {}
    
    baseline_drift_path = os.path.join(qa_dir, 'baseline_drift_p_vals.csv')
    post_drift_path = os.path.join(qa_dir, 'post_drift_p_vals.csv')
    
    drift_results = {
        'baseline_drift': None,
        'post_stimulus_drift': None
    }
    
    # Extract baseline drift results
    if os.path.exists(baseline_drift_path):
        baseline_drift = pd.read_csv(baseline_drift_path)
        # Count significant p-values
        alpha = 0.05
        sig_counts = {
            'trial_bin': (baseline_drift['trial_bin'] < alpha).sum(),
            'taste': (baseline_drift['taste'] < alpha).sum(),
            'interaction': (baseline_drift['trial_bin * taste'] < alpha).sum()
        }
        drift_results['baseline_drift'] = sig_counts
    
    # Extract post-stimulus drift results
    if os.path.exists(post_drift_path):
        post_drift = pd.read_csv(post_drift_path)
        # Count significant p-values
        alpha = 0.05
        sig_count = (post_drift['trial_bin'] < alpha).sum()
        drift_results['post_stimulus_drift'] = {'trial_bin': sig_count}
    
    return drift_results

def extract_channel_correlation_violations(dir_name):
    """
    Extract channel correlation violations from QA_output directory
    
    Args:
        dir_name: Directory containing the data
        
    Returns:
        Dictionary with channel correlation violations
    """
    qa_dir = os.path.join(dir_name, 'QA_output')
    
    if not os.path.exists(qa_dir):
        print(f"Warning: {qa_dir} not found. Run blech_clust.py first.")
        return {}
    
    # Look for correlation matrix files
    corr_files = glob.glob(os.path.join(qa_dir, '*corr*.csv'))
    
    if not corr_files:
        # Try to generate correlation data from HDF5 file
        metadata_handler = imp_metadata([[], dir_name])
        hdf5_path = metadata_handler.hdf5_name
        
        if os.path.exists(hdf5_path):
            try:
                # Get threshold from params file
                params_dict = metadata_handler.params_dict
                threshold = params_dict["qa_params"]["bridged_channel_threshold"]
                
                # Calculate correlation matrix
                down_dat_stack, chan_names = get_all_channels(
                    hdf5_path,
                    n_corr_samples=params_dict["qa_params"]["n_corr_samples"]
                )
                corr_mat = intra_corr(down_dat_stack)
                
                # Find violations
                violations = []
                for i in range(corr_mat.shape[0]):
                    for j in range(i+1, corr_mat.shape[1]):
                        if corr_mat[i, j] > threshold:
                            violations.append({
                                'channel1': chan_names[i],
                                'channel2': chan_names[j],
                                'correlation': float(corr_mat[i, j])
                            })
                
                return {
                    'threshold': threshold,
                    'violations': violations,
                    'violation_count': len(violations)
                }
            except Exception as e:
                print(f"Error calculating correlation matrix: {e}")
                return {}
        
        return {}
    
    # If correlation files exist, parse them
    violations = []
    threshold = 0.9  # Default threshold
    
    for corr_file in corr_files:
        try:
            corr_data = pd.read_csv(corr_file)
            # Extract violations based on file format
            # This is a simplified approach and may need adjustment based on actual file format
            if 'channel1' in corr_data.columns and 'channel2' in corr_data.columns:
                for _, row in corr_data.iterrows():
                    violations.append({
                        'channel1': row['channel1'],
                        'channel2': row['channel2'],
                        'correlation': float(row['correlation'])
                    })
        except Exception as e:
            print(f"Error parsing correlation file {corr_file}: {e}")
    
    return {
        'threshold': threshold,
        'violations': violations,
        'violation_count': len(violations)
    }

def extract_elbo_drift_results(dir_name):
    """
    Extract ELBO drift analysis results from QA_output directory
    
    Args:
        dir_name: Directory containing the data
        
    Returns:
        Dictionary with ELBO drift analysis results
    """
    qa_dir = os.path.join(dir_name, 'QA_output')
    
    if not os.path.exists(qa_dir):
        print(f"Warning: {qa_dir} not found. Run elbo_drift.py first.")
        return {}
    
    # Check for best_change.txt file
    best_change_path = os.path.join(qa_dir, 'best_change.txt')
    
    if not os.path.exists(best_change_path):
        return {}
    
    # Read best change value
    with open(best_change_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if not line.startswith('#'):
                best_change = int(line.strip())
                break
    
    # Look for artifact files
    artifact_dir = os.path.join(qa_dir, 'artifacts')
    if os.path.exists(artifact_dir):
        csv_files = glob.glob(os.path.join(artifact_dir, '*_taste_trial_change_elbo.csv'))
        
        if csv_files:
            try:
                elbo_data = pd.read_csv(csv_files[0])
                # Group by changes and get median ELBO
                median_elbo = elbo_data.groupby('changes')['elbo'].median().reset_index()
                # Sort by ELBO (ascending)
                sorted_elbo = median_elbo.sort_values('elbo', ascending=True)
                
                # Get top 3 best changes
                top_3_changes = sorted_elbo.head(3)['changes'].tolist()
                top_3_elbos = sorted_elbo.head(3)['elbo'].tolist()
                
                return {
                    'best_change': best_change,
                    'top_3_changes': top_3_changes,
                    'top_3_elbos': top_3_elbos,
                    'all_changes': sorted_elbo['changes'].tolist(),
                    'all_elbos': sorted_elbo['elbo'].tolist()
                }
            except Exception as e:
                print(f"Error parsing ELBO data: {e}")
    
    # If we couldn't get detailed data, just return the best change
    return {'best_change': best_change}

def generate_data_summary(dir_name):
    """
    Generate a comprehensive summary of dataset characteristics
    
    Args:
        dir_name: Directory containing the data
        
    Returns:
        Dictionary with dataset summary
    """
    # Initialize pipeline graph check
    script_path = os.path.realpath(__file__)
    this_pipeline_check = pipeline_graph_check(dir_name)
    this_pipeline_check.check_previous(script_path)
    this_pipeline_check.write_to_log(script_path, 'attempted')
    
    print(f"Generating data summary for {dir_name}")
    
    # Extract metadata
    metadata_handler = imp_metadata([[], dir_name])
    
    # Get basic experiment info
    try:
        info_dict = metadata_handler.info_dict
        basic_info = {
            'experiment_name': os.path.basename(dir_name.rstrip('/')),
            'tastes': info_dict['taste_params']['tastes'],
            'palatability_rankings': info_dict['taste_params']['pal_rankings'],
            'laser_present': len(info_dict['laser_params']['dig_in_nums']) > 0
        }
    except Exception as e:
        print(f"Error extracting basic info: {e}")
        basic_info = {
            'experiment_name': os.path.basename(dir_name.rstrip('/'))
        }
    
    # Extract unit characteristics
    print("Extracting unit characteristics...")
    unit_data = extract_unit_characteristics(dir_name)
    
    # Extract drift parameters
    print("Extracting drift parameters...")
    drift_data = extract_drift_parameters(dir_name)
    
    # Extract channel correlation violations
    print("Extracting channel correlation violations...")
    correlation_data = extract_channel_correlation_violations(dir_name)
    
    # Extract ELBO drift results
    print("Extracting ELBO drift results...")
    elbo_data = extract_elbo_drift_results(dir_name)
    
    # Count units by type
    unit_counts = {}
    if not unit_data.empty:
        # Count responsive units
        if 'responsiveness_sig' in unit_data.columns:
            unit_counts['responsive_units'] = unit_data['responsiveness_sig'].sum()
        
        # Count discriminatory units
        if 'discriminability_sig' in unit_data.columns:
            unit_counts['discriminatory_units'] = unit_data['discriminability_sig'].sum()
        
        # Count palatability units
        if 'palatability_sig' in unit_data.columns:
            unit_counts['palatability_units'] = unit_data['palatability_sig'].sum()
        
        # Count dynamic units
        if 'Dynamic_sig' in unit_data.columns:
            unit_counts['dynamic_units'] = unit_data['Dynamic_sig'].sum()
        
        # Total units
        unit_counts['total_units'] = len(unit_data)
    
    # Compile the summary
    data_summary = {
        'basic_info': basic_info,
        'unit_counts': unit_counts,
        'drift_analysis': drift_data,
        'channel_correlation': correlation_data,
        'elbo_analysis': elbo_data
    }
    
    # Add unit-level data if available
    if not unit_data.empty:
        data_summary['unit_data'] = unit_data.to_dict(orient='records')
    
    # Mark as completed
    this_pipeline_check.write_to_log(script_path, 'completed')
    
    return data_summary

if __name__ == "__main__":
    dir_name = args.dir_name
    
    # Generate the data summary
    data_summary = generate_data_summary(dir_name)
    
    # Save to JSON file
    output_path = os.path.join(dir_name, 'data_summary.json')
    with open(output_path, 'w') as f:
        json.dump(data_summary, f, indent=4)
    
    print(f"Data summary saved to {output_path}")
