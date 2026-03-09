# Import stuff!
import numpy as np
import sys
import os
import tables

print("=" * 60)
print("EMG Trial Cutting Script")
print("=" * 60)
print("This script reduces the number of trials in EMG data arrays")
print()

# Use post-process sheet template to write out a new sheet for this dataset
script_path = os.path.realpath(__file__)
blech_clust_dir = os.path.dirname(os.path.dirname(script_path))
sys.path.append(blech_clust_dir)
from utils.blech_utils import imp_metadata  # noqa

print("Loading metadata...")
metadata_handler = imp_metadata(sys.argv)
os.chdir(metadata_handler.dir_name)
print(f"Working directory: {metadata_handler.dir_name}")
print(f"HDF5 file: {metadata_handler.hdf5_name}")
print()

print("Opening HDF5 file...")
hf5 = tables.open_file(metadata_handler.hdf5_name, 'r+')

# Get emg data
print("Reading EMG data nodes...")
emg_nodes = hf5.list_nodes('/emg_data')
emg_nodes = [x for x in emg_nodes if 'board-DIN' in x._v_name]
print(f"Found {len(emg_nodes)} EMG nodes")

# Shape for each array = (n_channels, n_trials, n_samples)
print("Loading EMG arrays...")
emg_array = [x.emg_array.read() for x in emg_nodes]
emg_array_paths = [x._v_pathname for x in emg_nodes]

for i, arr in enumerate(emg_array):
    print(f"  Node {i+1}: shape = {arr.shape} (channels, trials, samples)")
print()

total_trials = 2
print(f"Cutting EMG data to first {total_trials} trials...")
emg_array = [x[:, :total_trials] for x in emg_array]

for i, arr in enumerate(emg_array):
    print(f"  Node {i+1}: new shape = {arr.shape}")
print()

# Remove old nodes
print("Removing old EMG array nodes from HDF5...")
for i, this_path in enumerate(emg_array_paths):
    print(f"  Removing: {this_path}/emg_array")
    hf5.remove_node(this_path, 'emg_array')
print()

# Write new nodes
print("Writing updated EMG arrays to HDF5...")
for i, emg in enumerate(emg_array):
    print(f"  Writing: {emg_array_paths[i]}/emg_array")
    hf5.create_array(emg_array_paths[i], 'emg_array', emg)
print()

print("Closing HDF5 file...")
hf5.close()

print("=" * 60)
print("EMG trial cutting completed successfully!")
print("=" * 60)
