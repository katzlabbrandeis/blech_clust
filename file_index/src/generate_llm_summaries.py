"""
Iterate through files in a directory and generate a summary of each file.
"""
import os
import sys
from pathlib import Path
from glob import glob

file_path = Path(__file__).resolve() 
blech_dir = file_path.parents[2]

# Base dir files
iter_dirs = [
        blech_dir,
        blech_dir / 'emg',
        blech_dir / 'emg' / 'utils',
        blech_dir / 'utils',
        blech_dir / 'utils' / 'ephys_data',
        blech_dir / 'utils' / 'qa_utils',
        ]

# Iterarte through directories and find all python files
file_list = []
for dir in iter_dirs:
    for file in dir.glob('*.py'):
        file_list.append(file)
sorted_files = sorted(file_list)
