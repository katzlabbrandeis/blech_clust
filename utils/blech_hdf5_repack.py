"""
This module automates the process of cleaning and compressing an HDF5 file using the `ptrepack` tool. It involves selecting a directory, identifying the HDF5 file, and replacing it with a compressed version.

- Prompts the user to select a directory using a GUI dialog and changes the working directory to the selected path.
- Searches for an HDF5 file (with a `.h5` extension) in the selected directory.
- Uses the `ptrepack` command-line tool to create a compressed and optimized copy of the HDF5 file named `tmp.h5`.
- Deletes the original HDF5 file.
- Renames the compressed file `tmp.h5` back to the original HDF5 file name.

Example Usage:
    # After processing LFP or spike data that creates large HDF5 files:

    # 1. Run this script directly:
    $ python blech_hdf5_repack.py
    # This will open a GUI to select the directory

    # 2. Or import and use in code:
    >>> import os
    >>> from utils.blech_hdf5_repack import *
    >>> os.chdir('/path/to/data')  # Instead of GUI selection
    >>> # Rest of script will run automatically
"""

import os
import tables
import easygui

# Get directory where the hdf5 file sits, and change to that directory
dir_name = easygui.diropenbox()
os.chdir(dir_name)

# Look for the hdf5 file in the directory
file_list = os.listdir('./')
hdf5_name = ''
for files in file_list:
    if files[-2:] == 'h5':
        hdf5_name = files

# Use ptrepack to save a clean and fresh copy of the hdf5 file as tmp.hf5
os.system("ptrepack --chunkshape=auto --propindexes --complevel=9 --complib=blosc " +
          hdf5_name + " " + "tmp.h5")

# Delete the old hdf5 file
os.remove(hdf5_name)

# And rename the new file with the same old name
os.rename("tmp.h5", hdf5_name)

print("HDF5 file successfully repacked and optimized")
