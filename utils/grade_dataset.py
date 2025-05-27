import argparse
import os
import sys
import json
import pandas as pd
import numpy as np

"""
Dataset Grading Utility

This script grades neural recording datasets based on multiple quality metrics:
1. Unit count - Total number of recorded units
2. Unit quality - Significant unit counts and fractions for each taste
3. Drift metrics - Both unit-based drift and ELBO-based drift analysis

The grades are calculated using thresholds defined in a grading_metrics.json file
and saved to the dataset's QA_output directory.
"""


test_bool = False
if test_bool:
    data_dir = '/home/abuzarmahmood/Desktop/blech_clust/pipeline_testing/test_data_handling/test_data/KM45_5tastes_210620_113227_new'
    args = argparse.Namespace(data_dir=data_dir)
    blech_clust_dir = '/home/abuzarmahmood/Desktop/blech_clust'
else:
    parser = argparse.ArgumentParser(
        description='Grade a dataset based on specified criteria.')
    parser.add_argument(
        'data_dir', type=str, help='Path to the directory containing the dataset files.')
    args = parser.parse_args()
    file_path = os.path.abspath(__file__)
    blech_clust_dir = os.path.dirname(os.path.dirname(file_path))

data_summary_path = os.path.join(data_dir, 'data_summary.json')
if os.path.exists(data_summary_path):
    with open(data_summary_path, 'r') as file:
        data_summary = json.load(file)
else:
    print(f"Data summary file not found @ : data_
    sys.exit(1)

grading_crit_path=os.path.join(
    blech_clust_dir, 'utils', 'grading_metrics.json')
if not os.path.exists(grading_crit_path):
    print(f"Grading criteria file not found @: {grading_crit_path}")
    sys.exit(1)
with open(grading_crit_path, 'r') as file:
    grading_criteria=json.load(file)

def extract_summary_values(data_summary):
    """
    Extract relevant values from the data summary for grading.
    
    Parameters:
    -----------
    data_summary : dict
        Dictionary containing the dataset summary information
        
    Returns:
    --------
    dict
        Dictionary containing:
        - total_units: Total number of units in the dataset
        - unit_qual_frame: DataFrame with unit quality metrics
        - drift_frame: DataFrame with drift analysis results
        - best_elbo: Best ELBO (Evidence Lower Bound) change value from drift analysis
    """
    total_units=data_summary['basic_info']['region_units'][0]['total_units']

    unit_qual_frame=pd.DataFrame(data_summary['unit_counts'])
    unit_qual_frame.columns=[eval(x)[0] for x in unit_qual_frame.columns]

    drift_frame=pd.DataFrame(data_summary['drift_analysis'])
    drift_frame.columns=[eval(x)[0] for x in drift_frame.columns]

    best_elbo=data_summary['elbo_analysis']['best_change']

    return {
        'total_units': total_units,
        'unit_qual_frame': unit_qual_frame,
        'drift_frame': drift_frame,
        'best_elbo': best_elbo
    }

summary_values=extract_summary_values(data_summary)

def get_count_score(value, thresholds, scores):
    """
    Get the score based on the value and the defined thresholds.
    
    The function finds the first threshold that the value is less than or equal to,
    and returns the corresponding score from the scores list.
    
    Parameters:
    -----------
    value : int or float
        The value to be scored
    thresholds : list
        List of threshold values in ascending order
    scores : list
        List of scores corresponding to thresholds
        
    Returns:
    --------
    float
        The score corresponding to the appropriate threshold
    
    Example:
    --------
    If thresholds = [10, 20, 30] and scores = [3, 2, 1, 0],
    a value of 15 would return a score of 2 (second threshold)
    """
    index=np.where(value <= np.array(thresholds))[0][0]
    return scores[index]

def grade_dataset(summary_values, grading_criteria):
    """
    Grade the dataset based on multiple criteria.
    
    The grading process evaluates:
    1. Total unit count - Higher counts receive better scores
    2. Unit quality - Based on significant unit counts and fractions
    3. Drift metrics - Both unit-based drift and ELBO-based drift
    
    Parameters:
    -----------
    summary_values : dict
        Dictionary containing extracted summary values:
        - total_units: Total number of units
        - unit_qual_frame: DataFrame with unit quality metrics
        - drift_frame: DataFrame with drift analysis results
        - best_elbo: Best ELBO change value
        
    grading_criteria : dict
        Dictionary containing thresholds and scores for each criterion
        
    Returns:
    --------
    pd.Series
        Series containing scores for:
        - unit_count: Score based on total unit count
        - taste-specific scores: One score per taste condition
        - drift_unit: Score based on unit drift (1 - significant fraction)
        - drift_elbo: Score based on ELBO change
    
    Notes:
    ------
    - Unit quality scores are calculated as: sig_count_score * sig_fraction
    - Drift unit score is calculated as: 1 - significant_fraction_post
    - Lower ELBO values indicate less drift and receive higher scores
    """
    unit_count = summary_values['total_units']
    best_elbo = summary_values['best_elbo']
    
    unit_count_score=get_count_score(unit_count,
                                       grading_criteria['unit_count']['thresholds'],
                                       grading_criteria['unit_count']['scores'])

    elbo_score=get_count_score(best_elbo,
                               grading_criteria['drift_ELBO']['thresholds'],
                               grading_criteria['drift_ELBO']['scores'])

    # Calculate unit scores based on the unit quality frame
    unit_qual_frame=summary_values['unit_qual_frame']
    sig_counts=unit_qual_frame.T['sig_count']
    sig_count_scores=[]
    for this_count in sig_counts:
        thresholds=grading_criteria['unit_count']['thresholds']
        scores=grading_criteria['unit_count']['scores']
        sig_count_scores.append(get_count_score(
            int(this_count), thresholds, scores))

    sig_fracs=unit_qual_frame.T['sig_fraction']

    # Final unit scores are the product of count scores and significant fractions
    unit_scores=np.array(sig_count_scores) * np.array(sig_fracs)

    # Calculate drift scores based on the drift frame
    # Use only post-stimulus units for drift analysis
    drift_analysis=summary_values['drift_frame']
    drift_unit_score=1-drift_analysis.loc['sig_fraction', 'post']

    # Compile everything into a DataFrame
    grading_df=pd.Series({
        'unit_count': unit_count_score,
        **dict(zip(unit_qual_frame.columns, unit_scores)),
        'drift_unit': drift_unit_score,
        'drift_elbo': elbo_score,
    })

    return grading_df

grades=grade_dataset(summary_values, grading_criteria)

# Write out grades to a JSON file in the 'QA_output' directory
output_dir=os.path.join(data_dir, 'QA_output')
os.makedirs(output_dir, exist_ok=True)
grades_path=os.path.join(output_dir, 'grading.json')
with open(grades_path, 'w') as file:
    json.dump(grades.to_dict(), file, indent=4)
