# Import stuff!
import numpy as np
import sys
import os
import tables

# Use post-process sheet template to write out a new sheet for this dataset
script_path = os.path.realpath(__file__)
blech_clust_dir = os.path.dirname(os.path.dirname(script_path))
sys.path.append(blech_clust_dir)
from utils.blech_utils import imp_metadata  # noqa

metadata_handler = imp_metadata(sys.argv)
os.chdir(metadata_handler.dir_name)

hf5 = tables.open_file(metadata_handler.hdf5_name, 'r+')
# Get emg data
emg_nodes = hf5.list_nodes('/emg_data')
emg_nodes = [x for x in emg_nodes if 'dig' in x._v_name]
# Shape for each array = (n_channels, n_trials, n_samples)
emg_array = [x.emg_array.read() for x in emg_nodes]
emg_array_paths = [x._v_pathname for x in emg_nodes]

total_trials = 2
emg_array = [x[:, :total_trials] for x in emg_array]

# Remove old nodes
for this_path in emg_array_paths:
    hf5.remove_node(this_path, 'emg_array')

# Write new nodes
for i, emg in enumerate(emg_array):
    hf5.create_array(emg_array_paths[i], 'emg_array', emg)

hf5.close()
