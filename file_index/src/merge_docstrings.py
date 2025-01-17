"""
Merge docstrings from all main files into a json

- Filename
- Filepath
- Docstring
"""

import os
import sys
import json
import ast
from tqdm import tqdm
from pathlib import Path
from glob import glob

def docstring_from_file(file_path):
    """ Get the docstring from a file """
    with file_path.open("r") as f:
        module_ast = ast.parse(f.read())

    docstring = ast.get_docstring(module_ast)
    return docstring

def get_file_list():
    """ Get a list of all python files in the specified directories """
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

    # Iterate through directories and find all python files
    file_list = []
    for dir in iter_dirs:
        for file in dir.glob('*.py'):
            file_list.append(file)
    sorted_files = sorted(file_list)

    # Remove __init__.py files
    sorted_files = [file for file in sorted_files if '__init__' not in file.name]

    return sorted_files

def merge_docstrings():
    """ Merge docstrings from all main files into a json """
    sorted_files = get_file_list()
    docstrings = []
    for file in tqdm(sorted_files):
        docstring = docstring_from_file(file)
        docstrings.append({
            "filename": file.name,
            "filepath": str(file),
            "docstring": docstring
        })

    output_path = file_path.parents[1] / 'data' / 'merged_docstrings.json'
    with open(output_path, "w") as f:
        json.dump(docstrings, f, indent=4)

if __name__ == "__main__":
    merge_docstrings()
