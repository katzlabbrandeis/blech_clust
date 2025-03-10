"""
Remove temporary files from blech_clust
"""
import os

script_path = os.path.realpath(__file__)
blech_clust_dir = os.path.dirname(os.path.dirname(script_path))

temp_file_paths = [
    'blech.dir',
    'blech_clust_jetstream_parallel.sh',
    'blech_clust_jetstream_parallel1.sh',
    'emg/BSA_run.dir',
    'emg/blech_emg_jetstream_parallel.sh',
    'emg/blech_emg_jetstream_parallel1.sh',
]

for path in temp_file_paths:
    full_path = os.path.join(blech_clust_dir, path)
    if os.path.exists(full_path):
        os.remove(full_path)
