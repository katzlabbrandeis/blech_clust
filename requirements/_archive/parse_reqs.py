import os
import numpy as np

INPUT_FILE_PATH='/home/abuzarmahmood/Desktop/blech_clust/requirements/import_statements.txt'
REQUIREMENTS_DIR = os.path.dirname(INPUT_FILE_PATH)

input_dat = [x.strip() for x in open(INPUT_FILE_PATH,'r').readlines()]

# Remove lines with 'directory'
input_dat = [x for x in input_dat if 'directory' not in x]

first_words = [x.split(' ')[0] for x in input_dat]

# If first word is not 'import' or 'from' toss entry
input_dat = [x for x,first_word in zip(input_dat,first_words) \
                    if first_word in ['import','from']]

first_words = [x.split(' ')[0] for x in input_dat]

# if first word is 'from', grab next word
from_statements = [x for x,first_word in zip(input_dat,first_words) \
                    if first_word in ['from']]
from_libs = [x.split(' ')[1] for x in from_statements]
# Drop if local module
from_libs = [x for x in from_libs if x[0]!='.']
# Grab only main module
fin_from_libs = [x.split('.')[0] for x in from_libs]
from_libs_set = np.unique(fin_from_libs)

# if first word is 'import'
import_statements = [x for x,first_word in zip(input_dat,first_words) \
                    if first_word in ['import']]
# If ',' in statement, mark as multi-import
multi_import = [x for x in import_statements if ',' in x] # None, so moving on
single_import = [x for x in import_statements if ',' not in x]
single_import_libs = [x.split(' ')[1] for x in single_import]

import_libs_set = np.unique(single_import_libs)

# Merge
fin_import_libs = np.sort(np.concatenate([from_libs_set, import_libs_set]))

# Apparently some sub-module imports got through, split and take first
fin_import_libs = np.sort(np.unique([x.split('.')[0] for x in fin_import_libs]))

# Write out
with open(os.path.join(REQUIREMENTS_DIR, 'base_requirements.txt'),'w') as out_file:
        out_file.writelines('\n'.join(fin_import_libs))
