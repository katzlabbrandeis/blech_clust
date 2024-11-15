"""
Given an electrode, generate a hierarchical clustering of the spike waveforms data.

Generate 2 subplots for each clustering solution:
	1. Dendrogram of the hierarchical clustering
	2. Above plot but with the data points colored by cluster
"""

############################################################
# Imports and Settings
############################################################
import os
import sys
import tables
import numpy as np
import re
import pylab as plt
import pandas as pd
import matplotlib
from glob import glob
# from sklearn.cluster import AgglomerativeClustering as AC
import seaborn as sns
import matplotlib

def load_electrode_data(data_dir, electrode_num): 
    """
    Load data from disk
    """

    loading_paths = [\
        f'spike_waveforms/electrode{electrode_num:02}/spike_waveforms.npy',
        f'spike_times/electrode{electrode_num:02}/spike_times.npy',
        f'spike_waveforms/electrode{electrode_num:02}/pca_waveforms.npy',
        f'spike_waveforms/electrode{electrode_num:02}/energy.npy',
        f'spike_waveforms/electrode{electrode_num:02}/spike_amplitudes.npy',
        ]
    loading_paths = [os.path.join(data_dir, x) for x in loading_paths]

    loaded_dat = [np.load(x) for x in loading_paths]

    return loaded_dat

def load_cluster_predictions(electrode_num, num_clusters):
	"""
	Load cluster predictions from disk
	"""
	return np.load(f'./clustering_results/electrode{electrode_num:02}/'\
				f'clusters{num_clusters}/predictions.npy')

def return_clustering_solutions(electrode_num):
	"""
	Find all clustering solutions for a given electrode
	"""
	# Find all clustering solutions for a given electrode
	clustering_solutions = glob(
			f'./clustering_results/electrode{electrode_num:02}/clusters*')
	clustering_solutions = [int(x.split('clusters')[-1]) \
			for x in clustering_solutions]
	clustering_solutions.sort()

	return clustering_solutions

############################################################
#data_dir = '/home/abuzarmahmood/Desktop/blech_clust/pipeline_testing/test_data_handling/test_data/KM45_5tastes_210620_113227_new/' 
data_dir = sys.argv[1]
# metadata_handler = imp_metadata([[], data_dir_name])
os.chdir(data_dir)

electrode_num = int(sys.argv[2])
# electrode_num = 0

# Find how many clusters were used for this electrode
clustering_solutions = return_clustering_solutions(electrode_num)

# Will only need to reload predictions for each clustering solution
(
	spike_waveforms,
	spike_times,
	pca_slices,
	energy,
	amplitudes,
) = load_electrode_data(data_dir, electrode_num) 

# Load predictions for all clustering solutions
predictions = [load_cluster_predictions(electrode_num, x) \
		for x in clustering_solutions]

# Perform hierarchical clustering on spike features
# clusterer = AC(n_clusters=None, affinity='euclidean', linkage='ward')
clust_dat = np.hstack((pca_slices, energy, amplitudes))
feature_names = [f'PC{x}' for x in range(pca_slices.shape[1])] + \
		['energy', 'amplitude']

# Thin out dat if too large
max_dat_size = 10000
if clust_dat.shape[0] > max_dat_size:
	inds = np.random.choice(clust_dat.shape[0], max_dat_size, replace=False)
	clust_dat = clust_dat[inds, :]
	predictions = [x[inds] for x in predictions]

cmap = plt.cm.get_cmap('tab10')
# Map predictions to consecutive integers
pred_labels = [np.unique(x, return_inverse=True)[1] for x in predictions]

# Convert to pandas dataframe
clust_dat = pd.DataFrame(clust_dat, columns=feature_names)

for ind, clust_num in enumerate(clustering_solutions):
	row_colors=[cmap(x) for x in pred_labels[ind]]
	# Dictionary of row_colors with pred_labels as keys
	row_colors_dict = dict(zip(predictions[ind], row_colors))
	# Sort by cluster
	row_colors_dict = {k: row_colors_dict[k] \
			for k in sorted(row_colors_dict.keys())}
	h = sns.clustermap(clust_dat, method='ward', metric='euclidean',
			row_cluster=True, col_cluster=False, cmap='viridis',
			yticklabels=False, figsize=(7, 7),
					   row_colors=row_colors,
					   xticklabels=feature_names, 
					cbar_pos = None,
					dendrogram_ratio=(0.3, 0),
					)
	# Create legend using row_colors_dict
	legend_elements = [matplotlib.patches.Patch(facecolor=row_colors_dict[x],
												label=f'{x}') \
						for x in row_colors_dict.keys()]
	# Create horizontal legend above heatmap
	h.ax_heatmap.legend(handles=legend_elements, loc='upper center',
						ncol=len(legend_elements), bbox_to_anchor=(0.5, 1.2),
						fontsize='small', frameon=False,
						title='Cluster #')
	fig = h.fig
	fig.suptitle(f'Electrode {electrode_num:02} - {clust_num} clusters')
	fig.tight_layout()
	fig.subplots_adjust(top=0.95)
	fig.savefig(f'./Plots/{electrode_num:02}/clusters{clust_num:02}/' +\
			f'clustermap.png', dpi=300,
			 bbox_inches='tight')
	plt.close()
