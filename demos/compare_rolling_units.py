#!/usr/bin/env python3
"""
Compare spike sorting results between rolling window and non-rolling window methods.

This script compares:
1. Number of units detected
2. Waveform count per unit
3. Average classifier probability for each unit

Between two sets of recordings - one processed with rolling window thresholding
and one without.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import h5py

# Add utils to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from ephys_data.ephys_data import ephys_data


def load_unit_data(data_dir):
    """
    Load unit data from a processed recording directory.
    
    Returns:
        dict: Contains unit counts, waveform counts, and classifier probabilities
    """
    try:
        # Load ephys data
        dat = ephys_data(data_dir)
        
        # Get unit descriptors
        unit_descriptors = dat.get_unit_descriptors()
        
        # Initialize results
        results = {
            'n_units': 0,
            'waveform_counts': [],
            'classifier_probs': [],
            'unit_info': []
        }
        
        # Process each electrode
        hdf5_path = dat.get_hdf5_path(data_dir)
        with h5py.File(hdf5_path, 'r') as hf5:
            for electrode in range(len(unit_descriptors)):
                electrode_key = f'sorted_units/electrode_{electrode:02d}'
                
                if electrode_key not in hf5:
                    continue
                    
                electrode_group = hf5[electrode_key]
                
                # Get clusters for this electrode
                if 'clusters' in electrode_group:
                    clusters = list(electrode_group['clusters'].keys())
                    
                    for cluster in clusters:
                        cluster_group = electrode_group['clusters'][cluster]
                        
                        # Count waveforms
                        if 'waveforms' in cluster_group:
                            n_waveforms = cluster_group['waveforms'].shape[0]
                        else:
                            n_waveforms = 0
                            
                        # Get classifier probability if available
                        classifier_prob = np.nan
                        if 'classifier_prob' in cluster_group:
                            classifier_prob = cluster_group['classifier_prob'][()]
                        elif 'predictions' in cluster_group:
                            # If predictions available, calculate mean probability
                            predictions = cluster_group['predictions'][:]
                            if len(predictions) > 0:
                                classifier_prob = np.mean(predictions)
                        
                        # Store unit info
                        unit_info = {
                            'electrode': electrode,
                            'cluster': cluster,
                            'n_waveforms': n_waveforms,
                            'classifier_prob': classifier_prob
                        }
                        
                        results['unit_info'].append(unit_info)
                        results['waveform_counts'].append(n_waveforms)
                        if not np.isnan(classifier_prob):
                            results['classifier_probs'].append(classifier_prob)
        
        results['n_units'] = len(results['unit_info'])
        
        return results
        
    except Exception as e:
        print(f"Error loading data from {data_dir}: {e}")
        return None


def compare_datasets(rolling_dir, non_rolling_dir, output_dir=None):
    """
    Compare two datasets and generate comparison plots and statistics.
    """
    print("Loading rolling window dataset...")
    rolling_data = load_unit_data(rolling_dir)
    
    print("Loading non-rolling window dataset...")
    non_rolling_data = load_unit_data(non_rolling_dir)
    
    if rolling_data is None or non_rolling_data is None:
        print("Error: Could not load one or both datasets")
        return
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    print(f"\nNumber of Units:")
    print(f"  Rolling window:     {rolling_data['n_units']}")
    print(f"  Non-rolling window: {non_rolling_data['n_units']}")
    print(f"  Difference:         {rolling_data['n_units'] - non_rolling_data['n_units']}")
    
    # Waveform count statistics
    rolling_wf_counts = np.array(rolling_data['waveform_counts'])
    non_rolling_wf_counts = np.array(non_rolling_data['waveform_counts'])
    
    print(f"\nWaveform Counts per Unit:")
    print(f"  Rolling window:")
    print(f"    Mean: {np.mean(rolling_wf_counts):.1f}")
    print(f"    Median: {np.median(rolling_wf_counts):.1f}")
    print(f"    Std: {np.std(rolling_wf_counts):.1f}")
    
    print(f"  Non-rolling window:")
    print(f"    Mean: {np.mean(non_rolling_wf_counts):.1f}")
    print(f"    Median: {np.median(non_rolling_wf_counts):.1f}")
    print(f"    Std: {np.std(non_rolling_wf_counts):.1f}")
    
    # Classifier probability statistics
    if rolling_data['classifier_probs'] and non_rolling_data['classifier_probs']:
        rolling_probs = np.array(rolling_data['classifier_probs'])
        non_rolling_probs = np.array(non_rolling_data['classifier_probs'])
        
        print(f"\nClassifier Probabilities:")
        print(f"  Rolling window:")
        print(f"    Mean: {np.mean(rolling_probs):.3f}")
        print(f"    Median: {np.median(rolling_probs):.3f}")
        print(f"    Std: {np.std(rolling_probs):.3f}")
        
        print(f"  Non-rolling window:")
        print(f"    Mean: {np.mean(non_rolling_probs):.3f}")
        print(f"    Median: {np.median(non_rolling_probs):.3f}")
        print(f"    Std: {np.std(non_rolling_probs):.3f}")
    
    # Generate plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Rolling vs Non-Rolling Window Comparison', fontsize=16)
    
    # Plot 1: Unit count comparison
    ax1 = axes[0, 0]
    methods = ['Rolling', 'Non-Rolling']
    unit_counts = [rolling_data['n_units'], non_rolling_data['n_units']]
    bars = ax1.bar(methods, unit_counts, color=['skyblue', 'lightcoral'])
    ax1.set_ylabel('Number of Units')
    ax1.set_title('Total Units Detected')
    
    # Add value labels on bars
    for bar, count in zip(bars, unit_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom')
    
    # Plot 2: Waveform count distributions
    ax2 = axes[0, 1]
    ax2.hist(rolling_wf_counts, bins=30, alpha=0.7, label='Rolling', color='skyblue')
    ax2.hist(non_rolling_wf_counts, bins=30, alpha=0.7, label='Non-Rolling', color='lightcoral')
    ax2.set_xlabel('Waveforms per Unit')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Waveform Count Distribution')
    ax2.legend()
    
    # Plot 3: Waveform count box plot
    ax3 = axes[1, 0]
    data_to_plot = [rolling_wf_counts, non_rolling_wf_counts]
    box_plot = ax3.boxplot(data_to_plot, labels=['Rolling', 'Non-Rolling'], patch_artist=True)
    box_plot['boxes'][0].set_facecolor('skyblue')
    box_plot['boxes'][1].set_facecolor('lightcoral')
    ax3.set_ylabel('Waveforms per Unit')
    ax3.set_title('Waveform Count Comparison')
    
    # Plot 4: Classifier probability comparison
    ax4 = axes[1, 1]
    if rolling_data['classifier_probs'] and non_rolling_data['classifier_probs']:
        ax4.hist(rolling_probs, bins=20, alpha=0.7, label='Rolling', color='skyblue')
        ax4.hist(non_rolling_probs, bins=20, alpha=0.7, label='Non-Rolling', color='lightcoral')
        ax4.set_xlabel('Classifier Probability')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Classifier Probability Distribution')
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'Classifier probabilities\nnot available', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Classifier Probability Distribution')
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'rolling_comparison.png'), dpi=300, bbox_inches='tight')
        print(f"\nPlots saved to: {output_dir}/rolling_comparison.png")
    
    plt.show()
    
    # Save detailed comparison to CSV
    if output_dir:
        # Create detailed comparison dataframe
        comparison_data = []
        
        for unit_info in rolling_data['unit_info']:
            comparison_data.append({
                'method': 'rolling',
                'electrode': unit_info['electrode'],
                'cluster': unit_info['cluster'],
                'n_waveforms': unit_info['n_waveforms'],
                'classifier_prob': unit_info['classifier_prob']
            })
        
        for unit_info in non_rolling_data['unit_info']:
            comparison_data.append({
                'method': 'non_rolling',
                'electrode': unit_info['electrode'],
                'cluster': unit_info['cluster'],
                'n_waveforms': unit_info['n_waveforms'],
                'classifier_prob': unit_info['classifier_prob']
            })
        
        df = pd.DataFrame(comparison_data)
        csv_path = os.path.join(output_dir, 'unit_comparison.csv')
        df.to_csv(csv_path, index=False)
        print(f"Detailed comparison saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare spike sorting results between rolling and non-rolling window methods'
    )
    parser.add_argument('rolling_dir', 
                       help='Path to directory with rolling window processed data')
    parser.add_argument('non_rolling_dir',
                       help='Path to directory with non-rolling window processed data')
    parser.add_argument('--output_dir', '-o',
                       help='Output directory for plots and results (optional)')
    
    args = parser.parse_args()
    
    # Validate input directories
    if not os.path.exists(args.rolling_dir):
        print(f"Error: Rolling directory does not exist: {args.rolling_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.non_rolling_dir):
        print(f"Error: Non-rolling directory does not exist: {args.non_rolling_dir}")
        sys.exit(1)
    
    # Run comparison
    compare_datasets(args.rolling_dir, args.non_rolling_dir, args.output_dir)


if __name__ == '__main__':
    main()
