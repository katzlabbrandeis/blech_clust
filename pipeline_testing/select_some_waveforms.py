"""
Select a specified number of waveforms from each electrode present
This is intended for the emg pipeline patch
Therefore, select about 3000 waveforms from 1 electrode
"""

import sys
import os
import glob
import numpy as np
import re
import pandas as pd

# Use post-process sheet template to write out a new sheet for this dataset
script_path = os.path.realpath(__file__)
blech_clust_dir = os.path.dirname(os.path.dirname(script_path))
sys.path.append(blech_clust_dir)
from utils.blech_utils import imp_metadata  # noqa

csv_path = os.path.join(
    blech_clust_dir,
    'example_meta_files',
    'GC_PC_taste_odor_spont_210919_175343.csv')
sorting_table = pd.read_csv(csv_path, keep_default_na=False)
sorting_table.drop(list(range(0, len(sorting_table))), inplace=True)

# Get name of directory with the data files
metadata_handler = imp_metadata(sys.argv)
dir_name = metadata_handler.dir_name

basename = os.path.basename(dir_name[:-1])

print(f'Processing : {dir_name}')

os.chdir(dir_name)

# Load clustering results
clustering_path = os.path.join(dir_name, 'clustering_results')
pred_files = sorted(glob.glob(os.path.join(
    clustering_path, '*', '*', 'predictions.npy')))
# Only take 1 cluster per electrode with count > threshold
# Because we don't want the script to confirm with user

# max_clust_pred = pred_files[-1]
for inum, this_pred_file in enumerate(pred_files):
    max_clust_pred = this_pred_file
    preds = np.load(max_clust_pred)
    splits = max_clust_pred.split('/')
    electrode_str = splits[-3]
    cluster_str = splits[-2]

    num_pattern = '(\d+)'
    electrode_num = int(re.findall(num_pattern, electrode_str)[0])
    cluster_num = int(re.findall(num_pattern, cluster_str)[0])

    # Find smallest combination of clusters which passes threshold
    wanted_count = 3000
    cluster_vals = np.unique(preds)
    cluster_counts = np.array([sum(preds == x) for x in np.unique(preds)])
    count_order = np.argsort(cluster_counts)

    # If single cluster can suffice, pick that, else pick smallest combination
    # that will hit wanted_count
    if sum(cluster_counts > wanted_count):
        wanted_inds = np.where(cluster_counts > wanted_count)[0]
        min_ind = np.argmin(cluster_counts[wanted_inds])
        final_ind = wanted_inds[min_ind]
        wanted_clusts = [cluster_vals[final_ind]]

        this_table = pd.DataFrame(
            data=dict(
                Unit=int(inum),
                Chan=electrode_num,
                Solution=cluster_num,
                Cluster='+'.join([str(x) for x in wanted_clusts]),
                single_unit='',
                Type='',
                Split='',
                Comments='',
            ),
            index=[inum],
        )
        sorting_table = sorting_table.append(this_table)

# Only pick 1 cluster per electrode
sorting_table = sorting_table.drop_duplicates(subset=['Chan'])

# Write out to data folder
sorting_table.to_csv(
    os.path.join(dir_name, basename+'_sorted_units.csv'),
    index=False
)

print(f'Wrote out {basename}_sorted_units.csv')
print(sorting_table)
