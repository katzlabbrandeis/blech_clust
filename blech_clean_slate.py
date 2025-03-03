"""
This script resets a data folder to an almost raw form by deleting most processing files while retaining specific file types such as info files and sorting tables.

- Imports necessary modules and a utility function `imp_metadata` to handle metadata.
- Retrieves directory name and file list from metadata.
- Defines a list of file patterns to keep, including `.dat`, `.info`, `.rhd`, `.csv`, `*_info`, `.txt`, and `.xml`.
- Identifies files to be removed by excluding the files matching the keep patterns.
- Attempts to remove each file or directory not matching the keep patterns using `shutil.rmtree` or `os.remove`.
"""
import os
import shutil
import glob
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description='Reset a data folder to near-raw form while preserving specific file types')
parser.add_argument('dir_path', help='Path to the data directory to clean')
args = parser.parse_args()

dir_name = args.dir_path
file_list = os.listdir(dir_name)

# Keep certain files and remove everything else
keep_pattern = ['*.dat', '*.info', '*.rhd',
                '*.csv', "*_info", "*.txt", "*.xml"]
keep_files = []
for pattern in keep_pattern:
    keep_files.extend(glob.glob(os.path.join(dir_name, pattern)))
keep_files_basenames = [os.path.basename(x) for x in keep_files]

remove_files = [x for x in file_list if x not in keep_files_basenames]
remove_paths = [os.path.join(dir_name, x) for x in remove_files]

for x in remove_paths:
    try:
        shutil.rmtree(x)
    except:
        os.remove(x)
    finally:
        pass
