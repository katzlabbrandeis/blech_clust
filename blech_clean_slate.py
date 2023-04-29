"""
Script to reset data folder to (almost) raw form
Deleting (almost) all processing files
Info file and sorting table (csv) as kept
"""
# Import stuff!
import os
import shutil
import easygui
import sys
import glob
from utils.blech_utils import imp_metadata

metadata_handler = imp_metadata(sys.argv)
dir_name = metadata_handler.dir_name
file_list = metadata_handler.file_list

# Keep certain files and remove everything else
keep_pattern = ['*.dat','*.info','*.rhd', '*.csv']
keep_files = []
for pattern in keep_pattern:
    keep_files.extend(glob.glob(os.path.join(dir_name, pattern)))
keep_files_basenames = [os.path.basename(x) for x in keep_files]

remove_files = [x for x in file_list if x not in keep_files_basenames] 
remove_paths = [os.path.join(dir_name,x) for x in remove_files]

for x in remove_paths:
    try:
        shutil.rmtree(x)
    except:
        os.remove(x)
    finally:
        pass

