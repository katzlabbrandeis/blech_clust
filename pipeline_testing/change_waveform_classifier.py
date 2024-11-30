"""
Script to change the use_classifier parameter in waveform_classifier_params.json
"""
import os
import json
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description='Change use_classifier parameter in waveform_classifier_params.json')
parser.add_argument('use_classifier', type=int, choices=[0, 1],
                    help='Set use_classifier to True (1) or False (0)')
args = parser.parse_args()

# Get paths
script_path = os.path.realpath(__file__)
blech_clust_dir = os.path.dirname(os.path.dirname(script_path))
params_dir = os.path.join(blech_clust_dir, 'params')
params_file = os.path.join(params_dir, 'waveform_classifier_params.json')

# Create params dir if it doesn't exist
if not os.path.exists(params_dir):
    os.makedirs(params_dir)

# Load template if params file doesn't exist
if not os.path.exists(params_file):
    template_file = os.path.join(params_dir, '_templates', 'waveform_classifier_params.json')
    with open(template_file, 'r') as f:
        params = json.load(f)
else:
    with open(params_file, 'r') as f:
        params = json.load(f)

# Update use_classifier
params['use_classifier'] = bool(args.use_classifier)

# Write updated params
with open(params_file, 'w') as f:
    json.dump(params, f, indent=3)

print("Updated waveform_classifier_params.json:")
os.system(f"cat {params_file}")

print(f"Updated use_classifier to {bool(args.use_classifier)}")
