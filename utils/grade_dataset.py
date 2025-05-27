import argparse
import os
import sys
import json
import pandas as pd
import numpy as np


test_bool = True
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
    """
    index=np.where(value <= np.array(thresholds))[0][0]
    return scores[index]

def grade_dataset(summary_values, grading_criteria):

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
