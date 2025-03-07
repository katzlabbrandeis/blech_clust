# Import stuff!
import numpy as np
import sys
import os
import json

bsa_bool = int(sys.argv[1])

# Use post-process sheet template to write out a new sheet for this dataset
script_path = os.path.realpath(__file__)
blech_clust_dir = os.path.dirname(os.path.dirname(script_path))
emg_params_path = os.path.join(blech_clust_dir, 'params', 'emg_params.json')

with open(emg_params_path) as f:
    emg_params = json.load(f)

if bsa_bool == 0:
    emg_params['use_BSA'] = False
else:
    emg_params['use_BSA'] = True

print(f'Use-BSA set to {emg_params["use_BSA"]}')

# Write out the new sheet
with open(emg_params_path, 'w') as f:
    json.dump(emg_params, f, indent=4, sort_keys=True)
