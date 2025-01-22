"""
Takes DIR and flags as input and prepares data directory
with info according to inputs
"""
import argparse
import os
import shutil

# Get name of directory with the data files
# Create argument parser
# TODO: Hardcode directory name in future so file can simply be run
parser = argparse.ArgumentParser(
    description='Prepares data directory with info '
    'with info files for testing')
parser.add_argument('dir_name',  help='Directory containing data files')
parser.add_argument('-emg', action='store_true')
parser.add_argument('-spike', action='store_true')
parser.add_argument('-emg_spike', action='store_true')
args = parser.parse_args()

print(f'Processing directory: {args.dir_name}')
if args.emg:
    print('Data mode: EMG')
elif args.spike:
    print('Data mode: Spike')
elif args.emg_spike:
    print('Data mode: EMG + Spike')

dir_name = args.dir_name

# Assert that only one mode is selected
# Else, tell user to select only one mode
assert sum([args.emg, args.spike, args.emg_spike]) == 1, \
    'Please select only one mode'

# Given flag and directory name, copy info files to directory
info_dir_list = ['emg_only_info', 'spike_only_info', 'emg_spike_info']
# Get info directory
if args.emg:
    info_dir = info_dir_list[0]
elif args.spike:
    info_dir = info_dir_list[1]
elif args.emg_spike:
    info_dir = info_dir_list[2]
info_dir_path = os.path.join(dir_name, info_dir)

# Print which info files are being copied
print(f'Copying info files from {info_dir_path}')
print('Files:')
for file in os.listdir(info_dir_path):
    print(file)
# Copy info files to directory
for file in os.listdir(info_dir_path):
    shutil.copy(os.path.join(info_dir_path, file), dir_name)
# shutil.copytree(info_dir_path, dir_name)
