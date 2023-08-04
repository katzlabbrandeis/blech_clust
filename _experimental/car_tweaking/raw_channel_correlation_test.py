# Necessary python modules
import os
import tables
import sys
import numpy as np
import multiprocessing
import json
import glob
import pandas as pd
import shutil
from tqdm import tqdm
import tables
import pylab as plt

# Necessary blech_clust modules
from utils import read_file
from utils.blech_utils import entry_checker, imp_metadata
from utils.blech_process_utils import path_handler

dir_list_path = '/media/bigdata/projects/pytau/pytau/data/fin_inter_list_3_14_22.txt' 
with open(dir_list_path, 'r') as f:
    dir_list = f.read().splitlines()

from scipy.stats import pearsonr
from itertools import combinations

def intra_corr(X):
    inds = list(combinations(range(X.shape[0]), 2))
    corr_mat = np.zeros((X.shape[0], X.shape[0]))
    for i,j in inds:
        corr_mat[i,j] = pearsonr(X[i,:], X[j,:])[0]
        corr_mat[j,i] = corr_mat[i,j]
    return corr_mat

############################################################

for this_dir in tqdm(dir_list[6:]):
    metadata_handler = imp_metadata([[], this_dir])
    dir_name = metadata_handler.dir_name
    print(f'Processing : {dir_name}')
    os.chdir(dir_name)

    hdf5_name = metadata_handler.hdf5_name
    info_dict = metadata_handler.info_dict
    file_list = metadata_handler.file_list

    layout_path = glob.glob(os.path.join(dir_name, "*layout.csv"))[0]
    electrode_layout_frame = pd.read_csv(layout_path)

    # Read data files, and append to electrode arrays
    with tables.open_file(hdf5_name, 'r+') as hf5: 
        if '/raw' not in hf5:
            hf5.create_group('/', 'raw')
    read_file.read_electrode_channels(hdf5_name, electrode_layout_frame)

############################################################
# Calculate correlations
down_rate = 100

#corr_list = []
elec_region_list = [] 
for this_dir in tqdm(dir_list):
    metadata_handler = imp_metadata([[], this_dir])
    hdf5_name = metadata_handler.hdf5_name
    #with tables.open_file(hdf5_name, 'r+') as hf5: 
    #    raw_elecs = hf5.list_nodes('/raw')
    #    down_dat = [x[:][::down_rate] for x in raw_elecs]
    #corr_mat = intra_corr(np.vstack(down_dat))
    #corr_list.append(corr_mat)
    layout_path = glob.glob(os.path.join(this_dir, "*layout.csv"))[0]
    electrode_layout_frame = pd.read_csv(layout_path)
    elec_region_list.append(electrode_layout_frame)
    #elec_region_list.append(electrode_layout_frame['CAR_group'].values)

wanted_elec_list = [x[['CAR_group', 'electrode_num']] for x in elec_region_list]
wanted_elec_list = [x.sort_values(['CAR_group','electrode_num']) for x in wanted_elec_list]
sort_inds_list = [x.index.values for x in wanted_elec_list]

############################################################
plot_dir = '/home/abuzarmahmood/Desktop/elec_corr_test'
if not os.path.isdir(plot_dir):
    os.mkdir(plot_dir)

for i, this_dir in enumerate(dir_list):
    corr_mat = corr_list[i]
    elec_regions = wanted_elec_list[i]
    # Merge columns as str into a single index
    elec_regions = elec_regions['CAR_group'].astype(str) + '_' + elec_regions['electrode_num'].astype(str)
    fig, ax = plt.subplots(1,1)
    sorted_corr_mat = corr_mat[sort_inds_list[i],:]
    sorted_corr_mat = sorted_corr_mat[:,sort_inds_list[i]]
    im = ax.matshow(sorted_corr_mat)
    this_basename = os.path.basename(this_dir)
    ax.set_title(this_basename)
    ax.set_xticks(np.arange(len(elec_regions)))
    ax.set_yticks(np.arange(len(elec_regions)))
    ax.set_xticklabels(elec_regions)
    ax.set_yticklabels(elec_regions)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{i}_{this_basename}_corr.png'))
    plt.close(fig)

