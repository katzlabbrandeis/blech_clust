"""
blech_data_summary.py - Generate a summary of dataset characteristics and quality

This script analyzes data from a blech_clust experiment and generates a JSON summary file
containing both unit-level and session-level parameters, including:

1. Unit-Level Parameters:
   - Responsiveness p-values
   - Discriminability p-values
   - Palatability p-values
   - Dynamicity p-values
   - Unit counts per region

2. Session-Level Parameters:
   - Drift analysis results (pre-stimulus and post-stimulus)
   - Channel correlation violations
   - ELBO change point analysis results
   - Number of laser conditions
   - Total neuron count

The summary is useful for quickly assessing dataset quality and characteristics
without having to examine multiple individual analysis files.

Usage:
    python blech_data_summary.py <dir_name>

Arguments:
    dir_name : Directory containing the processed data files
"""

from utils.qa_utils.channel_corr import get_all_channels, intra_corr
from utils.blech_utils import imp_metadata, pipeline_graph_check
import os
import sys
import json
import glob
import numpy as np
import pandas as pd
import tables
import argparse
from tqdm import tqdm

test_bool = False
# Parse command line arguments
if test_bool:
    args = argparse.Namespace(
        dir_name='/home/abuzarmahmood/Desktop/blech_clust/pipeline_testing/test_data_handling/test_data/KM45_5tastes_210620_113227_new'
    )
    script_path = '/home/abuzarmahmood/Desktop/blech_clust/utils/blech_data_summary.py'
else:
    parser = argparse.ArgumentParser(
        description='Generate summary of dataset characteristics')
    parser.add_argument('dir_name', type=str,
                        help='Directory containing data files')
    args = parser.parse_args()

    script_path = os.path.realpath(__file__)

# Import necessary modules from blech_clust
blech_clust_dir = os.path.dirname(os.path.dirname(script_path))
sys.path.append(blech_clust_dir)


def extract_unit_characteristics(dir_name):
    """
    Extract unit-level characteristics from aggregated_characteristics.csv
    Get count and fraction of significant units for each source type

    Args:
        dir_name: Directory containing the data

    Returns:
        DataFrame with unit-level characteristics
    """
    characteristics_path = os.path.join(
        dir_name, 'aggregated_characteristics.csv')

    if not os.path.exists(characteristics_path):
        print(
            f"Warning: {characteristics_path} not found. Run blech_units_characteristics.py first.")
        return pd.DataFrame()

    # Load the characteristics data
    unit_data = pd.read_csv(characteristics_path)

    sig_count = unit_data.groupby(['Source', 'laser_tuple']).sum()[
        'sig'].reset_index()
    sig_frac = unit_data.groupby(['Source', 'laser_tuple']).mean()[
        'sig'].reset_index()
    sig_count.rename(columns={'sig': 'sig_count'}, inplace=True)
    sig_frac.rename(columns={'sig': 'sig_fraction'}, inplace=True)
    unit_data = pd.merge(sig_count, sig_frac, on=['Source', 'laser_tuple'])

    # In source, rename: 'bin_num' --> 'dynamic', 'taste_num' -> 'discriminability'
    unit_data['Source'] = unit_data['Source'].replace(
        {'bin_num': 'dynamic', 'taste_num': 'discriminability'})

    return unit_data


def extract_drift_parameters(dir_name, alpha=0.05):
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

    baseline_drift = pd.read_csv(baseline_drift_path, index_col=0)
    post_drift = pd.read_csv(post_drift_path, index_col=0)

    # Process baseline drift data
    baseline_drift_sig = baseline_drift.copy()
    baseline_drift_sig.drop(columns=['taste', 'Residual'], inplace=True)
    base_dat_cols = ['trial_bin', 'trial_bin * taste']
    baseline_drift_sig[base_dat_cols] = baseline_drift_sig[base_dat_cols].apply(
        lambda x: np.where(x < alpha, 1, 0))

    # Process post-drift data
    post_drift_sig = post_drift.copy()
    post_drift_sig.drop(columns=['Error'], inplace=True)
    dat_cols = ['trial_bin']
    post_drift_sig[dat_cols] = post_drift_sig[dat_cols].apply(
        lambda x: np.where(x < alpha, 1, 0))

    # Get counts and fractions of significant units
    baseline_drift_counts = baseline_drift_sig[base_dat_cols].sum()
    baseline_drift_frac = baseline_drift_sig[base_dat_cols].mean()
    post_drift_counts = post_drift_sig['trial_bin'].sum()
    post_drift_frac = post_drift_sig['trial_bin'].mean()

    # Combine results
    baseline_frame = pd.DataFrame({
        'comparison': baseline_drift_counts.index,
        'period': 'baseline',
        'sig_count': baseline_drift_counts.values,
        'sig_fraction': baseline_drift_frac.values
    })
    post_frame = pd.DataFrame({
        'comparison': 'trial_bin',
        'period': 'post',
        'sig_count': [post_drift_counts],
        'sig_fraction': [post_drift_frac]
    })
    drift_results = pd.concat([baseline_frame, post_frame], axis=0)

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

    metadata_handler = imp_metadata([[], dir_name])
    params_dict = metadata_handler.params_dict
    threshold = params_dict["qa_params"]["bridged_channel_threshold"]

    # Look for correlation matrix files
    channel_corr_mat = np.load(
        os.path.join(qa_dir, 'channel_corr_mat.npy'), allow_pickle=True)

    ut_inds = np.triu_indices(
        channel_corr_mat.shape[0], k=1)
    corr_vals = channel_corr_mat[ut_inds]

    # Check for violations
    violation_bool = corr_vals > threshold
    viol_count = np.sum(violation_bool)
    viol_frac = np.mean(violation_bool)

    viol_dict = {
        'threshold': threshold,
        'viol_count': viol_count,
        'viol_fraction': viol_frac
    }

    return viol_dict


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
        csv_files = glob.glob(os.path.join(
            artifact_dir, '*_taste_trial_change_elbo.csv'))

        if csv_files:
            try:
                elbo_data = pd.read_csv(csv_files[0])
                # Group by changes and get median ELBO
                median_elbo = elbo_data.groupby(
                    'changes')['elbo'].median().reset_index()
                # Sort by ELBO (ascending)
                sorted_elbo = median_elbo.sort_values('elbo', ascending=True)

                # Get top 3 best changes
                top_3_changes = sorted_elbo.head(3)['changes'].tolist()
                top_3_elbos = sorted_elbo.head(3)['elbo'].tolist()

                return {
                    'best_change': best_change,
                    'top_3_changes': np.vectorize(int)(top_3_changes),
                    'top_3_elbos': top_3_elbos,
                    'all_changes': np.vectorize(int)(sorted_elbo['changes'].tolist()),
                    'all_elbos': sorted_elbo['elbo'].tolist()
                }
            except Exception as e:
                print(f"Error parsing ELBO data: {e}")

    # If we couldn't get detailed data, just return the best change
    return {'best_change': best_change}


def extract_region_unit_counts(dir_name):
    """
    Extract neuron counts per region using ephys_data

    Args:
        dir_name: Directory containing the data

    Returns:
        Dictionary with neuron counts per region
    """
    try:
        # Initialize ephys_data with the directory
        from utils.ephys_data.ephys_data import ephys_data
        data = ephys_data(data_dir=dir_name)
        
        # Get region electrodes and units
        data.get_region_electrodes()
        data.get_region_units()
        
        # Create a dictionary with neuron counts per region
        region_unit_counts = {region: len(units) 
                             for region, units in zip(data.region_names, data.region_units)}
        
        return {
            'region_unit_counts': region_unit_counts,
            'total_units': sum(region_unit_counts.values())
        }
    except Exception as e:
        print(f"Error extracting region unit counts: {e}")
        return {}


def extract_laser_conditions(dir_name):
    """
    Extract laser conditions from trial_info_frame.csv

    Args:
        dir_name: Directory containing the data

    Returns:
        Dictionary with laser condition information
    """
    trial_info_path = os.path.join(dir_name, 'trial_info_frame.csv')
    
    if not os.path.exists(trial_info_path):
        print(f"Warning: {trial_info_path} not found.")
        return {}
    
    try:
        # Load trial info frame
        trial_info = pd.read_csv(trial_info_path)
        
        # Extract unique laser conditions
        laser_conditions = trial_info[['laser_duration_ms', 'laser_lag_ms']].drop_duplicates()
        
        # Count trials per laser condition
        condition_counts = trial_info.groupby(['laser_duration_ms', 'laser_lag_ms']).size().reset_index(name='trial_count')
        
        return {
            'n_laser_conditions': len(laser_conditions),
            'laser_conditions': laser_conditions.to_dict('records'),
            'condition_counts': condition_counts.to_dict('records')
        }
    except Exception as e:
        print(f"Error extracting laser conditions: {e}")
        return {}


def generate_data_summary(dir_name):
    """
    Generate a comprehensive summary of dataset characteristics

    Args:
        dir_name: Directory containing the data

    Returns:
        Dictionary with dataset summary
    """

    print(f"Generating data summary for {dir_name}")

    # Extract metadata
    metadata_handler = imp_metadata([[], dir_name])

    # Get basic experiment info
    try:
        info_dict = metadata_handler.info_dict
        basic_info = {
            'experiment_name': os.path.basename(dir_name.rstrip('/')),
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
    
    # Extract region unit counts
    print("Extracting region unit counts...")
    region_unit_data = extract_region_unit_counts(dir_name)
    
    # Extract laser conditions
    print("Extracting laser conditions...")
    laser_condition_data = extract_laser_conditions(dir_name)

    # Since unit_data and drift_data are DataFrames, we need to convert them to dicts
    unit_data_dict = unit_data.set_index(['Source', 'laser_tuple']).T.to_dict()
    # Convert keys to strings for JSON serialization
    unit_data_dict = {str(k): v for k, v in unit_data_dict.items()}
    drift_data_dict = drift_data.set_index(
        ['period', 'comparison']).T.to_dict()
    # Convert keys to strings for JSON serialization
    drift_data_dict = {str(k): v for k, v in drift_data_dict.items()}

    # Compile the summary
    data_summary = {
        'basic_info': basic_info,
        'unit_counts': unit_data_dict,
        'drift_analysis': drift_data_dict,
        'channel_correlation': correlation_data,
        'elbo_analysis': elbo_data,
        'region_units': region_unit_data,
        'laser_conditions': laser_condition_data
    }

    return data_summary


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


if __name__ == "__main__":
    dir_name = args.dir_name

    # Perform pipeline graph check
    this_pipeline_check = pipeline_graph_check(dir_name)
    this_pipeline_check.check_previous(script_path)
    this_pipeline_check.write_to_log(script_path, 'attempted')

    # Generate the data summary
    data_summary = generate_data_summary(dir_name)

    # Save to JSON file
    output_path = os.path.join(dir_name, 'data_summary.json')
    with open(output_path, 'w') as f:
        json.dump(data_summary, f, indent=4, cls=NpEncoder)

    print(f"Data summary saved to {output_path}")
    
    # Mark as completed
    this_pipeline_check.write_to_log(script_path, 'completed')
