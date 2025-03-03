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


# Example workflow when running this file directly
if __name__ == "__main__":
    # This script is already set up to run directly
    # When executed, it will:
    # 1. Open a GUI dialog to select a directory
    # 2. Find the HDF5 file in that directory
    # 3. Compress and optimize the file using ptrepack

    # Example manual usage in Python:
    # import os
    # from utils.blech_hdf5_repack import *
    #
    # # Instead of using the GUI dialog, set directory directly
    # os.chdir('/path/to/data_directory')
    # # The rest of the script will run automatically

    # Command-line alternative (without using this script):
    # $ cd /path/to/data_directory
    # $ ptrepack --chunkshape=auto --propindexes --complevel=9 --complib=blosc original.h5 tmp.h5
    # $ mv tmp.h5 original.h5

    print("HDF5 file successfully repacked and optimized")

"""
EXAMPLE WORKFLOW:

This script is typically used after processing large datasets that create HDF5 files,
to reduce file size and optimize access speed.

Workflow 1: Direct execution (recommended for most users)
-----------------------------------------------------
1. Run the script directly:
   $ python utils/blech_hdf5_repack.py

2. Use the GUI dialog to select the directory containing your HDF5 file
   - The script automatically finds the HDF5 file in the directory
   - Compresses and optimizes it in place

Workflow 2: Programmatic usage in analysis scripts
-----------------------------------------------------
# After processing data that creates large HDF5 files
import os
from utils.blech_hdf5_repack import *

# Option 1: Use with GUI dialog
# Just import and call the script - it runs automatically

# Option 2: Specify directory programmatically
os.chdir('/path/to/data_directory')  # Set directory containing HDF5 file
# The rest of the script runs automatically when imported

Workflow 3: Batch processing multiple datasets
-----------------------------------------------------
import os
import glob

# Find all data directories
data_dirs = glob.glob('/path/to/experiment/*/data')

for data_dir in data_dirs:
    os.chdir(data_dir)
    # Look for the hdf5 file in the directory
    file_list = os.listdir('./')
    hdf5_name = ''
    for files in file_list:
        if files[-2:] == 'h5':
            hdf5_name = files
    
    if hdf5_name:
        print(f"Repacking {data_dir}/{hdf5_name}")
        # Use ptrepack to compress
        os.system("ptrepack --chunkshape=auto --propindexes --complevel=9 --complib=blosc " +
                  hdf5_name + " " + "tmp.h5")
        os.remove(hdf5_name)
        os.rename("tmp.h5", hdf5_name)
"""
