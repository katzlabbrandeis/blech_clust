"""
Utilities to support blech_clust processing
"""
import easygui
import os
import glob
import json
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

def entry_checker(msg, check_func, fail_response):
    check_bool = False
    continue_bool = True
    exit_str = '"x" to exit :: '
    while not check_bool:
        msg_input = input(msg.join([' ',exit_str]))
        if msg_input == 'x':
            continue_bool = False
            break
        check_bool = check_func(msg_input)
        if not check_bool:
            print(fail_response)
    return msg_input, continue_bool


class imp_metadata():
    def __init__(self, args):
        self.dir_name = self.get_dir_name(args)
        self.get_file_list()
        self.get_hdf5_name()
        self.load_params()
        self.load_info()
        self.load_layout()

    def get_dir_name(self, args):
        if len(args) > 1:
            dir_name = os.path.abspath(args[1])
            if dir_name[-1] != '/':
                dir_name += '/'
        else:
            dir_name = easygui.diropenbox(msg = 'Please select data directory')
        return dir_name

    def get_file_list(self,):
        self.file_list = os.listdir(self.dir_name)
        
    def get_hdf5_name(self,):
        file_list = glob.glob(os.path.join(self.dir_name,'**.h5'))
        if len(file_list) > 0:
            self.hdf5_name = file_list[0]
        else:
            print('No HDF5 file found')

    def get_params_path(self,):
        file_list = glob.glob(os.path.join(self.dir_name,'**.params'))
        if len(file_list) > 0:
            self.params_file_path = file_list[0]
        else:
            print('No PARAMS file found')

    def get_layout_path(self,):
        file_list = glob.glob(os.path.join(self.dir_name,'**layout.csv'))
        if len(file_list) > 0:
            self.layout_file_path = file_list[0]
        else:
            print('No LAYOUT file found')

    def load_params(self,):
        self.get_params_path()
        if 'params_file_path' in dir(self):
            with open(self.params_file_path, 'r') as params_file_connect:
                self.params_dict = json.load(params_file_connect)

    def get_info_path(self,):
        file_list = glob.glob(os.path.join(self.dir_name,'**.info'))
        if len(file_list) > 0:
            self.info_file_path = file_list[0]
        else:
            print('No INFO file found')

    def load_info(self,):
        self.get_info_path()
        if 'info_file_path' in dir(self):
            with open(self.info_file_path, 'r') as info_file_connect:
                self.info_dict = json.load(info_file_connect)

    def load_layout(self,):
        self.get_layout_path()
        if 'layout_file_path' in dir(self):
            self.layout = pd.read_csv(self.layout_file_path, index_col = 0)

##############################
# Functions for handling inconsistency in CAR groups

def return_corr_mat_car_groups(corr_mat, car_vals, corr_mat_inds):
    """
    Takes a corrlations matrix over all channels, and returns a list
    of correlation matrices for each CAR group

    Inputs:
        corr_mat - 2D numpy array of correlation values 
        car_vals - list of lists of indices [according to layout frame] 
                for each CAR group
        corr_mat_inds - list of indices for each channel in the 
                correlation matrix [according to layout frame]

    Outputs:
        corr_mat_list - list of correlation matrices for each CAR group

    """
    # Map elecrtode_inds to corr_mat inds
    # indexing inds_map will return where that channel is in the corr_mat
    inds_map = dict(zip(corr_mat_inds, range(len(corr_mat_inds))))
    corr_mat_list = []
    group_inds_list = []
    for this_car in car_vals:
        this_inds_map = [inds_map[x] for x in this_car]
        group_inds_list.append(this_inds_map)
        corr_mat_list.append(corr_mat[this_inds_map,:][:,this_inds_map])
    return corr_mat_list, group_inds_list

def make_corr_mat_symmetric(corr_mat):
    """
    Correlation matrix is only upper triangle, this makes it symmetric

    Inputs:
        corr_mat - 2D numpy array of correlation values

    Outputs:
        corr_mat - 2D numpy array of correlation values with symmetric
    """
    corr_mat[np.isnan(corr_mat)] = 0
    corr_mat = corr_mat + corr_mat.T
    # Make diagonal 1
    np.fill_diagonal(corr_mat, 1)
    return corr_mat

def return_threshold_pca(corr_mat, threshold = 0.9):
    """
    Returns number of components to keep for PCA based on threshold
    of variance explained

    Inputs:
        corr_mat - 2D numpy array of correlation values
        threshold - threshold for variance explained

    Outputs:
        n_components - number of components to keep
        pca_data - Transformed data
    """
    pca = PCA().fit(corr_mat)
    n_components = np.where(np.cumsum(pca.explained_variance_ratio_) > threshold)[0][0]
    pca_data = pca.transform(corr_mat)[:,:n_components]
    return n_components, pca_data

def kmeans_bic(kmeans_obs, data):
    """
    Calculates BIC for kmeans clustering

    Inputs:
        kmeans_obs - kmeans object
        data - data used for clustering

    Outputs:
        bic - BIC value
    """
    # Calculate BIC
    centers = [kmeans_obs.cluster_centers_]
    labels = kmeans_obs.labels_
    m = kmeans_obs.n_clusters
    n = np.bincount(labels)
    N, d = data.shape
    cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(data[np.where(labels == i)], [centers[0][i]], 
            'euclidean')**2) for i in range(m)])
    const_term = 0.5 * m * np.log(N) * (d+1)
    bic = np.sum([n[i] * np.log(n[i]) -
            n[i] * np.log(N) -
            ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
            ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term
    return bic

def calc_mat_clusters(corr_mat, max_clusters = 5, repeats = 10, criterion = 'bic'):
    """
    Uses BIC to determine number of clusters in correlation matrix
    using GMM clustering
    Fit on PCA transformed data to minimize effect of high dimensionality

    Inputs:
        corr_mat - 2D numpy array of correlation values
        max_clusters - maximum number of clusters to test

    Outputs:
        n_clusters - number of clusters that minimizes BIC
    """
    # Get PCA data
    pca_corr_mat = return_threshold_pca(corr_mat, threshold = 0.9)[1] 

    # Calculate BIC for each number of clusters
    n_clusters = np.arange(1,max_clusters+1)
    score = []
    gmm_list = []
    cluster_list = []
    for i, n in enumerate(n_clusters):
        for j in range(repeats):
            # NOTE: It seems like GMM is clustering unlike datapoints together
            #       This is not what we want, we want to cluster similar channels
            #       together, so might be better to use KMeans
            gmm = GaussianMixture(n_components = n, covariance_type = 'full')
            gmm.fit(pca_corr_mat)
            if criterion == 'bic':
                score.append(gmm.bic(pca_corr_mat))
            elif criterion == 'aic':
                score.append(gmm.aic(pca_corr_mat))
            gmm_list.append(gmm)
            cluster_list.append(n)
    # Find minimum BIC
    min_score_ind = np.argmin(score)
    # get clusters for min BIC gmm
    wanted_gmm = gmm_list[min_score_ind]
    cluster_labels = wanted_gmm.predict(pca_corr_mat)
    return cluster_list[min_score_ind], cluster_labels


def plot_clustered_mat(mat_list, trace_list, mat_names, mat_labels, dat_inds, plot_dir):
    """
    Generate clustered plots of each correlation matrix provided

    Inputs:
        mat_list - list of correlation matrices
        mat_names - list of names for each correlation matrix
        mat_labels - list of cluster labels for each correlation matrix
        plot_dir - directory to save plots to
    
    Outputs:
        None
    """
    fig, ax = plt.subplots(2, len(mat_list), figsize = (len(mat_list)*5, 5))
    for i, mat in enumerate(mat_list):
        this_mat = mat_list[i]
        this_traces = trace_list[i]
        this_name = mat_names[i]
        this_labels = mat_labels[i]
        this_cluster_counts = len(np.unique(this_labels))
        this_dat_inds = np.array(dat_inds[i])
        sort_inds = np.argsort(this_labels)
        sorted_mat = this_mat[sort_inds,:][:,sort_inds]
        sorted_labels = this_labels[sort_inds]
        sorted_dat_inds = this_dat_inds[sort_inds]
        cluster_breaks = np.where(np.diff(sorted_labels))[0]
        sorted_traces = this_traces[sort_inds,:]
        im = ax[0,i].imshow(sorted_mat, cmap = 'RdBu', vmin = 0, vmax = 1)
        plt.colorbar(im, ax = ax[0,i], label = 'Distance')
        # Find MAD of sorted_traces to use as bounds for plotting
        dat_mad = np.median(
                np.abs(sorted_traces - np.median(sorted_traces, axis = None)), 
                axis = None)
        dat_med = np.median(sorted_traces, axis = None)
        ax[1,i].imshow(sorted_traces, 
                       aspect='auto', interpolation='nearest', cmap = 'RdBu',
                       vmin = dat_med - 3*dat_mad, vmax = dat_med + 3*dat_mad)
        for this_break in cluster_breaks:
            ax[0,i].axhline(this_break+0.5, color = 'k', linewidth = 2)
            ax[0,i].axvline(this_break+0.5, color = 'k', linewidth = 2)
            ax[1,i].axhline(this_break+0.5, color = 'k', linewidth = 2)
        ax[0,i].set_title(f'{this_name}, {this_cluster_counts} clusters')
        ax[0,i].set_xticks([])
        ax[0,i].set_yticks([])
        ax[0,i].set_xticklabels([])
        ax[0,i].set_yticklabels([])
        ax[0,i].set_xlabel('Channel')
        ax[0,i].set_ylabel('Channel')
        ax[0,i].set_xticks(np.arange(0, len(sorted_labels), 1))
        ax[0,i].set_yticks(np.arange(0, len(sorted_labels), 1))
        ax[0,i].set_xticklabels(sorted_dat_inds, rotation = 90)
        ax[0,i].set_yticklabels(sorted_dat_inds)
        ax[1,i].set_yticks([])
        ax[1,i].set_yticklabels([])
        ax[1,i].set_yticks(np.arange(0, len(sorted_labels), 1))
        ax[1,i].set_yticklabels(sorted_dat_inds)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'clustered_CAR_correlations.png'))
    plt.close()
