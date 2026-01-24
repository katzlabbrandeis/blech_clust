"""
This module is designed for handling and processing electrophysiological data, specifically focusing on clustering and classification of neural spike waveforms. It includes classes and functions for managing paths, clustering, classification, electrode data handling, and spike processing.

- **path_handler**: Manages directory paths for the module.
- **cluster_handler**: Handles clustering of spike waveforms.
  - `check_classifier_data_exists`: Checks if classifier data exists.
  - `return_training_set`: Returns a training set for clustering.
  - `fit_manual_model`: Fits a Gaussian Mixture Model manually.
  - `fit_auto_model`: Fits a Bayesian Gaussian Mixture Model automatically.
  - `get_cluster_labels`: Retrieves cluster labels from a model.
  - `perform_prediction`: Performs clustering and saves the model.
  - `remove_outliers`: Removes outliers from clusters.
  - `save_cluster_labels`: Saves cluster labels to a file.
  - `calc_mahalanobis_distance_matrix`: Calculates and saves Mahalanobis distance matrix.
  - `create_output_dir`: Creates directories for clustering results.
  - `create_classifier_plots`: Generates plots for classifier results.
  - `create_output_plots`: Generates output plots for clusters.

- **classifier_handler**: Manages classification of waveforms.
  - `download_neurecommend_models`: Downloads necessary models if not present.
  - `return_waveform_classifier_params_path`: Returns the path to classifier parameters.
  - `get_waveform_classifier_params`: Loads classifier parameters.
  - `load_pipelines`: Loads feature and prediction pipelines.
  - `classify_waveforms`: Classifies waveforms and saves results.
  - `write_out_recommendations`: Writes out recommendations based on classification.
  - `gen_plots`: Generates plots for predicted spikes and noise.

- **electrode_handler**: Handles electrode data processing.
  - `filter_electrode`: Filters raw electrode data.
  - `cut_to_int_seconds`: Cuts data to integer seconds.
  - `calc_recording_cutoff`: Calculates recording cutoff based on parameters.
  - `make_cutoff_plot`: Generates a plot showing recording cutoff.
  - `cutoff_electrode`: Cuts off electrode data at the calculated cutoff.

- **spike_handler**: Processes spikes from electrode data.
  - `extract_waveforms`: Extracts waveforms from filtered electrode data.
  - `dejitter_spikes`: Dejitters spikes to correct timing.
  - `interpolate_waveforms`: Interpolates waveforms to target sampling rate for classifier compatibility.
  - `extract_features`: Extracts features from spike waveforms.
  - `return_feature`: Returns specific features from spike data.
  - `write_out_spike_data`: Saves spike data to files.

- **Utility Functions**:
  - `ifisdir_rmdir`: Removes a directory if it exists.
  - `return_cutoff_values`: Calculates cutoff values for electrode recording.
  - `gen_window_plots`: Generates plots for data windows.
  - `gen_datashader_plot`: Generates datashader plots for waveforms.
  - `gen_isi_hist`: Generates inter-spike interval histograms.
  - `remove_too_large_waveforms`: Removes waveforms that are too large.
  - `feature_timeseries_plot`: Plots feature timeseries for clusters.
  - `register_labels`, `calc_linkage`, `sort_label_array`, `perform_agg_clustering`, `plot_waveform_dendogram`, `trim_data`: Helper functions for agglomerative clustering.
"""

# Set environment variables to limit the number of threads used by various libraries
# Do it at the start of the script to ensure it applies to all imported libraries
from blech_clust.utils.blech_utils import ifisdir_rmdir
import os  # noqa
os.environ['OMP_NUM_THREADS'] = '1'  # noqa
os.environ['MKL_NUM_THREADS'] = '1'  # noqa
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # noqa

from matplotlib.patches import ConnectionPatch
from scipy.cluster.hierarchy import cut_tree, linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering, KMeans
import pandas as pd
import matplotlib
import shutil
from glob import glob
import tables
import numpy as np
import json
import pylab as plt
from scipy.stats import zscore
import subprocess
from utils import blech_waveforms_datashader
from scipy.spatial.distance import mahalanobis
from sklearn.mixture import BayesianGaussianMixture as BGM
from sklearn.mixture import GaussianMixture as gmm
from joblib import load
import blech_clust.utils.clustering as clust
from blech_units_plot import plot_unit_summary, gen_isi_hist
# import subprocess
# import sys

############################################################
# Define Functions
############################################################


class UnitData():
    def __init__(self, waveforms, times):
        self.waveforms = waveforms
        self.times = times


class path_handler():

    def __init__(self):
        self.home_dir = os.getenv('HOME')
        file_path = os.path.abspath(__file__)
        blech_clust_dir = ('/').join(file_path.split('/')[:-2])
        self.blech_clust_dir = blech_clust_dir


class cluster_handler():
    """
    Class to handle clustering steps
    """

    def __init__(self, params_dict, data_dir, electrode_num, cluster_num,
                 spike_features=None, slices_dejittered=None, times_dejittered=None,
                 threshold=None, feature_names=None, fit_type='manual'):
        """
        Initialize cluster handler with specific data rather than a spike_set object

        Args:
            params_dict: Dictionary of parameters
            data_dir: Directory containing data
            electrode_num: Electrode number
            cluster_num: Number of clusters
            spike_features: Features extracted from spikes
            slices_dejittered: Dejittered spike waveforms
            times_dejittered: Dejittered spike times
            threshold: Threshold used for spike detection
            feature_names: Names of features
            fit_type: Type of fitting ('manual' or 'auto')
        """
        assert fit_type in [
            'manual', 'auto'], 'fit_type must be manual or auto'

        self.params_dict = params_dict
        self.dat_thresh = 10e3
        self.data_dir = data_dir
        self.electrode_num = electrode_num
        self.cluster_num = cluster_num
        self.fit_type = fit_type

        # Store specific data instead of the whole spike_set
        self.spike_features = spike_features
        self.slices_dejittered = slices_dejittered
        self.times_dejittered = times_dejittered
        self.threshold = threshold
        self.feature_names = feature_names

        self.create_output_dir()

    def check_classifier_data_exists(self, data_dir):
        clf_list = glob(os.path.join(
            data_dir,
            'spike_waveforms/electrode*/clf_prob.npy'))

        if len(clf_list) == 0:
            raise Exception(
                'Classifier output not found, please run blech_run_process.sh with classifier.')

    def return_training_set(self, data):
        """
        Return training set for clustering
        """
        train_set = data[
            np.random.choice(np.arange(data.shape[0]),
                             int(np.min((data.shape[0], self.dat_thresh))))]
        return train_set

    def fit_manual_model(self, train_set, clusters):
        """
        Cluster waveforms
        """
        model = gmm(
            random_state=0,
            n_components=clusters,
            max_iter=self.params_dict['clustering_params']['num_iter'],
            n_init=self.params_dict['clustering_params']['num_restarts'],
            tol=self.params_dict['clustering_params']['thresh'],
        ).fit(train_set)
        return model

    def fit_auto_model(self, train_set, clusters):
        """
        Since the dirichlet process WILL find more clusters for more data,
        (whereas we know that more spikes doens't mean more neurons),
        we need to keep the "COUNT" constant
        One way to do this is to use KMeans centroids as data points
        """

        max_k = 1000

        if len(train_set) > max_k:
            kmean_obj = KMeans(n_clusters=max_k)
            kmean_obj.fit(train_set)
            bgm_train_data = kmean_obj.cluster_centers_
        else:
            bgm_train_data = train_set

        g = BGM(
            random_state=0,
            n_components=clusters,
            max_iter=self.params_dict['clustering_params']['num_iter'],
            n_init=self.params_dict['clustering_params']['num_restarts'],
            tol=self.params_dict['clustering_params']['thresh'],
            covariance_type='full',
            weight_concentration_prior_type='dirichlet_process',
            # This can be systematically adjusted to match
            # actual data
            weight_concentration_prior=0.1,
            verbose=0,
        )

        # Train on kmeans centroids but predict on the actual data
        # This is not a problem as both sets belong to the same space/dimensions
        g.fit(bgm_train_data)
        return g

    def get_cluster_labels(self, data):
        """
        Get cluster labels using the fitted model
        """
        return self.model.predict(data)

    def fit_model(self, train_set):
        """
        Fit appropriate model based on fit_type
        """
        if self.fit_type == 'manual':
            return self.fit_manual_model(train_set, self.cluster_num)
        elif self.fit_type == 'auto':
            return self.fit_auto_model(train_set, self.cluster_num)

    def perform_clustering(self):
        """
        Complete clustering pipeline:
        1. Get training set
        2. Fit model
        3. Get cluster labels

        Returns:
            numpy.ndarray: Cluster labels
        """
        train_set = self.return_training_set(self.spike_features)
        self.model = self.fit_model(train_set)
        self.labels = self.get_cluster_labels(self.spike_features)
        return self.labels

    def ensure_continuous_labels(self):
        """
        Ensure cluster labels are continuous numbers
        """

        # Make sure cluster labels are continuous numbers
        # as auto-model can return non-continuous numbers
        # (e.g. 0, 1, 3, 4, 5, 6, 7, 8, 9, 10)
        # This is not a problem for manual model
        # Rename all but label=-1
        if self.fit_type == 'auto':
            current_clusters = list(np.unique(self.labels))
            if -1 in current_clusters:
                current_clusters.remove(-1)
            new_labels = np.arange(len(current_clusters))
            cluster_map = dict(zip(current_clusters, new_labels))
            # Add -1:-1 mapping
            cluster_map[-1] = -1
            print('Renaming Cluters')
            print(f'Cluster map: {cluster_map}')
            self.labels = np.array([cluster_map[x] for x in self.labels])
            self.cluster_map = cluster_map

    def remove_outliers(self, params_dict):
        """
        Clear large waveforms
        """
        # Sometimes large amplitude noise waveforms cluster with the
        # spike waveforms because the amplitude has been factored out of
        # the scaled slices.
        # Run through the clusters and find the waveforms that are more than
        # wf_amplitude_sd_cutoff larger than the cluster mean.
        # Set predictions = -1 at these points so that they aren't
        # picked up by blech_post_process
        wf_amplitude_sd_cutoff = params_dict['wf_amplitude_sd_cutoff']
        for cluster in np.unique(self.labels):
            cluster_points = np.where(self.labels[:] == cluster)[0]
            this_cluster = remove_too_large_waveforms(
                cluster_points,
                self.spike_features[:, [i for i, x in enumerate(
                    self.feature_names) if 'amplitude' in x][0]],
                self.labels,
                wf_amplitude_sd_cutoff)
            self.labels[cluster_points] = this_cluster

    def save_cluster_labels(self):
        np.save(
            os.path.join(
                self.clust_results_dir, 'predictions.npy'),
            self.labels)

    def calc_mahalanobis_distance_matrix(self):
        """
        Calculates matrix of mahalanobis distances between all pairs of clusters
        Saves matrix to file
        """
        # assert model in dir(self), 'Model not found'
        cluster_labels = np.unique(self.labels)
        mahal_matrix = np.zeros((len(cluster_labels), len(cluster_labels)))
        full_data = self.spike_features
        for i, clust_i in enumerate(cluster_labels):
            for j, clust_j in enumerate(cluster_labels):
                # Use sample covariances so we can use labels
                this_cluster_data = \
                    full_data[np.where(self.labels == clust_i)[0]]
                # If cluster is smaller than 3, it gives
                # singular covariance matrix
                if len(this_cluster_data) > 2:
                    this_cluster_mean = np.mean(this_cluster_data, axis=0)
                    this_cluster_cov = np.cov(this_cluster_data, rowvar=False)
                    # this_cluster_mean = self.model.means_[i]
                    # this_cluster_cov = self.model.covariances_[i]
                    inv_cov = np.linalg.pinv(this_cluster_cov)
                    other_cluster = np.where(self.labels == clust_j)[0]
                    other_cluster_data = full_data[other_cluster]
                    mahal_list = [
                        mahalanobis(x, this_cluster_mean, inv_cov)
                        for x in other_cluster_data
                    ]
                    mahal_matrix[i, j] = np.mean(mahal_list)
                else:
                    mahal_matrix[i, j] = np.nan
        # Needs to be saved for plotting
        self.mahal_matrix = mahal_matrix
        np.save(
            os.path.join(
                self.clust_results_dir, 'mahalanobis_distances.npy'),
            self.mahal_matrix)

    def create_output_dir(self):
        # Make folder for results of i+2 clusters, and store results there
        clust_results_dir = os.path.join(
            self.data_dir,
            'clustering_results',
            f'electrode{self.electrode_num:02}',
            f'clusters{self.cluster_num}'
        )
        clust_plot_dir = os.path.join(
            self.data_dir,
            'Plots',
            f'{self.electrode_num:02}',
            f'clusters{self.cluster_num:02}'
        )
        ifisdir_rmdir(clust_results_dir)
        ifisdir_rmdir(clust_plot_dir)
        os.makedirs(clust_results_dir)
        os.makedirs(clust_plot_dir)
        self.clust_results_dir = clust_results_dir
        self.clust_plot_dir = clust_plot_dir

    def create_classifier_plots(
            self,
            # classifier_handler
            classifier_pred,
            classifier_prob,
            clf_threshold,
            all_waveforms,
            all_times,
            labels=None,
    ):
        """
        For each cluster, plot:
            1. Pred Spikes
            2. Pred Noise
            3. Distribution and timeseries of probability
            4. Histogram of prediction probability
            5. Histogram of times for spikes and noise

        Input data can come from classifier_handler
        """

        # self.check_classifier_data_exists(self.data_dir)

        # classifier_pred = classifier_handler.clf_pred
        # classifier_prob = classifier_handler.clf_prob
        # clf_threshold = classifier_handler.clf_threshold
        # # Use directly stored data instead of accessing through spike_set
        # all_waveforms = self.slices_dejittered
        # all_times = self.times_dejittered

        if labels is None:
            labels = self.labels

        max_plot_count = 1000
        for cluster in np.unique(labels):
            cluster_bool = labels == cluster
            if sum(cluster_bool):

                fig = plt.figure(figsize=(5, 10))
                gs = fig.add_gridspec(5, 2,
                                      width_ratios=(4, 1),
                                      height_ratios=(1, 1, 1, 1, 1),
                                      left=0.2, right=0.9, bottom=0.1, top=0.9,
                                      wspace=0.05, hspace=0.05)
                spike_ax = fig.add_subplot(gs[0, 0])
                spike_hist_ax = fig.add_subplot(gs[1, 0])
                noise_ax = fig.add_subplot(gs[2, 0])
                noise_hist_ax = fig.add_subplot(gs[3, 0], sharex=spike_hist_ax)
                prob_ax = fig.add_subplot(gs[4, 0], sharex=spike_hist_ax)
                prob_hist_ax = fig.add_subplot(gs[4, 1], sharey=prob_ax)

                spike_ax.set_ylabel('Spike Waveforms')
                noise_ax.set_ylabel('Noise Waveforms')
                spike_hist_ax.set_ylabel('Spike Times')
                noise_hist_ax.set_ylabel('Noise Times')
                prob_ax.set_ylabel('Classifier Probs')

                spike_bool = np.logical_and(classifier_pred, cluster_bool)
                noise_bool = np.logical_and(
                    np.logical_not(classifier_pred), cluster_bool)

                if sum(spike_bool):
                    spike_inds = np.random.choice(
                        np.where(spike_bool)[0], max_plot_count)
                    spike_prob = classifier_prob[spike_inds]
                    spike_waves = all_waveforms[spike_inds]
                    spike_times = all_times[spike_inds]
                    spike_ax.plot(spike_waves.T, color='k', alpha=0.1)
                    # spike_ax.set_title(f'Count : {np.sum(spike_bool)}')
                    spike_ax.text(1, 0.5,
                                  f'Count : {np.sum(spike_bool)}' + '\n' +
                                  f'Mean prob : {spike_prob.mean():.3f}',
                                  rotation=270,
                                  verticalalignment='center',
                                  transform=spike_ax.transAxes)
                    spike_ax.axhline(self.threshold,
                                     color='red', linestyle='--')
                    spike_ax.axhline(-self.threshold,
                                     color='red', linestyle='--')
                    spike_hist_ax.hist(spike_times, bins=30)

                if sum(noise_bool):
                    noise_inds = np.random.choice(
                        np.where(noise_bool)[0], max_plot_count)
                    noise_prob = classifier_prob[noise_inds]
                    noise_waves = all_waveforms[noise_inds]
                    noise_times = all_times[noise_inds]
                    noise_ax.plot(noise_waves.T, color='k', alpha=0.1)
                    # noise_ax.set_title(f'Count : {np.sum(noise_bool)}')
                    noise_ax.text(1, 0.5,
                                  f'Count : {np.sum(noise_bool)}' + '\n' +
                                  f'Mean prob : {noise_prob.mean():.3f}',
                                  rotation=270,
                                  verticalalignment='center',
                                  transform=noise_ax.transAxes)
                    noise_ax.axhline(self.threshold,
                                     color='red', linestyle='--')
                    noise_ax.axhline(-self.threshold,
                                     color='red', linestyle='--')
                    noise_hist_ax.hist(noise_times, bins=30)

                colors = ['red', 'blue']
                for this_bool in np.unique(classifier_pred):
                    clf_inds = classifier_pred == this_bool
                    inds = np.logical_and(clf_inds, cluster_bool)
                    this_times = all_times[inds]
                    this_probs = classifier_prob[inds]
                    if sum(inds):
                        prob_ax.scatter(
                            this_times, this_probs,
                            color=colors[this_bool],
                            alpha=0.1
                        )
                        prob_ax.axhline(clf_threshold,
                                        linestyle='--', color='k')
                        prob_hist_ax.hist(this_probs,
                                          bins=np.linspace(0, 1, 30),
                                          color=colors[this_bool],
                                          orientation='horizontal')
                        prob_hist_ax.axhline(clf_threshold,
                                             linestyle='--', color='k')
                fig.suptitle(f'Cluster {cluster}')
                clf_fig_path = os.path.join(
                    self.clust_plot_dir, f'Cluster{cluster}_classifier.png')
                fig.savefig(clf_fig_path, bbox_inches='tight')
                plt.close(fig)

            # If cluster waveform plot exists, merge with that
            waveform_plot_path = os.path.join(
                self.clust_plot_dir, f'Cluster{cluster}_waveforms.png')
            if os.path.exists(waveform_plot_path):
                print(
                    f'Cluster{cluster} : Waveform plot exists, merging with classifier plot')
                wav_img = plt.imread(waveform_plot_path)
                clf_img = plt.imread(clf_fig_path)
                # Use gridspec to have wav img be 2x wide
                fig = plt.figure(figsize=(15, 10))
                gs = fig.add_gridspec(1, 2, width_ratios=(2, 1))
                ax0 = fig.add_subplot(gs[0, 0])
                ax1 = fig.add_subplot(gs[0, 1])
                ax0.imshow(wav_img)
                ax1.imshow(clf_img)
                ax0.axis('off')
                ax1.axis('off')
                fig.suptitle(f'Cluster {cluster}')
                fig.savefig(waveform_plot_path, bbox_inches='tight')
                plt.close(fig)
                # Delete the classifier plot
                os.remove(os.path.join(self.clust_plot_dir,
                          f'Cluster{cluster}_classifier.png'))
            else:
                print(
                    f'Cluster {cluster} : Waveform plot does not exist, classifier plot will be saved as is')
                # Just rename the plot
                os.rename(
                    os.path.join(self.clust_plot_dir,
                                 f'Cluster{cluster}_classifier.png'),
                    waveform_plot_path)
    # return fig, ax

    def create_output_plots(self,
                            params_dict):

        # Generate mahalanobis distance plot
        fig, ax = plt.subplots()
        im = ax.matshow(self.mahal_matrix)
        fig.colorbar(im, ax=ax)
        # Annotated the plot with values of the matrix
        for (i, j), z in np.ndenumerate(self.mahal_matrix):
            ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',
                    bbox=dict(boxstyle='round',
                              facecolor='white', edgecolor='0.3'),
                    fontweight='bold', color='red')
        cluster_labels = np.unique(self.labels)
        ax.set_xticks(np.arange(len(cluster_labels)))
        ax.set_yticks(np.arange(len(cluster_labels)))
        ax.set_xticklabels(cluster_labels)
        ax.set_yticklabels(cluster_labels)
        ax.set_title('Mahalanobis Distance')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Cluster')
        plt.tight_layout()
        fig.savefig(os.path.join(self.clust_plot_dir, 'mahalanobis_dist'),
                    bbox_inches='tight')
        plt.close(fig)

        # Use the directly stored data instead of accessing through spike_set
        slices_dejittered = self.slices_dejittered
        times_dejittered = self.times_dejittered
        standard_data = self.spike_features
        feature_names = self.feature_names
        threshold = self.threshold
        # Create file, and plot spike waveforms for the different clusters.
        # Plot 10 times downsampled dejittered/smoothed waveforms.
        # Additionally plot the ISI distribution of each cluster
        x = np.arange(len(slices_dejittered[0])) + 1
        for cluster in np.unique(self.labels):
            cluster_points = np.where(self.labels == cluster)[0]

            if len(cluster_points) > 0:
                # downsample = False, Prevents waveforms_datashader
                # from FURTHER downsampling the given waveforms for plotting
                # Because in the previous version they were upsampled for clustering

                # Create waveform datashader plot with envelope
                #############################
                fig, ax = plot_unit_summary(
                    unit_data=UnitData(
                        slices_dejittered[cluster_points],
                        times_dejittered[cluster_points]
                    ),
                    min_time=0,
                    max_time=times_dejittered.max(),
                    params_dict=params_dict,
                    return_only=True,
                    # Use electrode and cluster-specific directories
                    # to avoid over-writing by other parallel processes
                    output_dir=f'{self.clust_plot_dir}/{cluster:02}_temp',
                )
                # Since blech_waveforms_datashader.waveforms_datashader only removes
                # the inner temp_dir, remove the above outer dir manually
                ifisdir_rmdir(f'{self.clust_plot_dir}/{cluster:02}_temp')

                fig.suptitle(f'Cluster {cluster} waveforms')

                fig.savefig(os.path.join(
                    self.clust_plot_dir, f'Cluster{cluster}_waveforms'))
                plt.close("all")

                #############################
                fig, ax = feature_timeseries_plot(
                    standard_data,
                    times_dejittered,
                    feature_names,
                    cluster_points
                )
                fig.suptitle(f'Cluster {cluster} features')
                fig.savefig(os.path.join(
                    self.clust_plot_dir, f'Cluster{cluster}_features'))
                plt.close(fig)

            else:
                # Write out file that somehow there are no spikes
                #############################
                file_path = os.path.join(
                    self.clust_plot_dir, f'no_spikes_Cluster{cluster}')
                with open(file_path, 'w') as file_connect:
                    file_connect.write('')


class classifier_handler():
    """
    Class to handler classifier steps
    """

    def __init__(
            self,
            data_dir,
            electrode_num,
            params_dict,
    ):
        home_dir = os.environ.get("HOME")
        model_dir = f'{home_dir}/Desktop/neuRecommend/model'

        # Download neuRecommend if not found
        # self.download_neurecommend_models(home_dir, model_dir)

        pred_pipeline_path = f'{model_dir}/xgboost_full_pipeline.dump'
        feature_pipeline_path = f'{model_dir}/feature_engineering_pipeline.dump'

        # Load feature names
        feature_names_path = f'{model_dir}/feature_names.json'
        feature_names = json.load(open(feature_names_path, 'r'))

        clf_threshold_path = f'{model_dir}/proba_threshold.json'
        with open(clf_threshold_path, 'r') as this_file:
            out_dict = json.load(this_file)
        clf_threshold = out_dict['threshold']

        self.create_pipeline_path = f'{home_dir}/Desktop/neuRecommend/src/create_pipeline'
        self.pred_pipeline_path = pred_pipeline_path
        self.feature_pipeline_path = feature_pipeline_path
        self.clf_threshold = clf_threshold
        self.feature_names = feature_names

        self.data_dir = data_dir
        self.electrode_num = electrode_num
        self.params_dict = params_dict
        self.plot_dir = os.path.join(
            data_dir,
            f'Plots/{electrode_num:02}')
        self.get_waveform_classifier_params()

    def download_neurecommend_models(self, home_dir, model_dir):
        """
        If models are not present in the right place
        Attempt to download them
        """
        # If neuRecommend not present, clone it
        git_path = 'https://github.com/abuzarmahmood/neuRecommend.git'
        neurecommend_dir = f'{home_dir}/Desktop/neuRecommend'
        if not os.path.exists(neurecommend_dir):
            process = subprocess.Popen(
                f'git clone {git_path} {neurecommend_dir}', shell=True)
            # Forces process to complete before proceeding
            stdout, stderr = process.communicate()
            # Install requirements for neuRecommend
            process = subprocess.Popen(
                f'pip install -r {neurecommend_dir}/requirements.txt', shell=True)
            # Forces process to complete before proceeding
            stdout, stderr = process.communicate()

        # If model_dir doesn't exist, then download models
        if not os.path.exists(f'{model_dir}/.models_downloaded'):
            print('Model directory does not exist')
            print('Attempting to download model')

            process = subprocess.Popen(
                f'bash {neurecommend_dir}/src/utils/io/download_models.sh', shell=True)
            # Forces process to complete before proceeding
            stdout, stderr = process.communicate()

        # If model_dir still doesn't exist, then throw an error
        if not os.path.exists(model_dir):
            raise Exception("Couldn't download model, please refer to '\
                    'blech_clust/README.md#setup for instructions")

    @staticmethod
    def return_waveform_classifier_params_path(blech_clust_dir):
        """
        Inner function for get_waveform_classifier_params
        so that it can also be called externally
        """
        params_file_path = os.path.join(
            blech_clust_dir,
            'params',
            'waveform_classifier_params.json')
        # If params file doesn't exist, stop execution
        if not os.path.exists(params_file_path):
            print('=== Waveform Classifier Params file not found. ===')
            print(
                '==> Please copy [[ blech_clust/params/_templates/waveform_classifier_params.json ]] to [[ blech_clust/params/waveform_classifier_params.json ]] and update as needed.')
            exit()
        return params_file_path

    def get_waveform_classifier_params(self):
        this_path_handler = path_handler()
        self.blech_clust_dir = this_path_handler.blech_clust_dir
        params_file_path = self.return_waveform_classifier_params_path(
            self.blech_clust_dir)
        with open(params_file_path, 'r') as this_file:
            self.classifier_params = json.load(this_file)

    def load_pipelines(self):
        """
        Load feature and prediction pipelines
        """
        # from feature_engineering_pipeline import *
        self.feature_pipeline = load(self.feature_pipeline_path)
        self.pred_pipeline = load(self.pred_pipeline_path)

    def classify_waveforms(self, slices, spiketimes,):
        """
        Classify waveforms
        """
        # Get the probability of each slice being a spike
        clf_prob = self.pred_pipeline.predict_proba(slices)[:, 1]
        clf_pred = clf_prob >= self.clf_threshold
        pred_spike = slices[clf_pred == 1]
        pos_spike_times = spiketimes[clf_pred == 1]
        spike_prob = clf_prob[clf_pred == 1]

        # Pull out noise info
        pred_noise = slices[clf_pred == 0]
        noise_times = spiketimes[clf_pred == 0]
        noise_prob = clf_prob[clf_pred == 0]

        # Turn postitive and negative into dictionaries
        pos_spike_dict = {
            'waveforms': pred_spike,
            'spiketimes': pos_spike_times,
            'prob': spike_prob,
        }
        neg_spike_dict = {
            'waveforms': pred_noise,
            'spiketimes': noise_times,
            'prob': noise_prob,
        }

        self.clf_prob = clf_prob
        self.clf_pred = clf_pred
        self.pos_spike_dict = pos_spike_dict
        self.neg_spike_dict = neg_spike_dict

        # Write out both prob and pred
        out_dir = os.path.join(
            self.data_dir,
            'spike_waveforms',
            f'electrode{self.electrode_num:02}'
        )
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        np.save(os.path.join(out_dir, 'clf_prob.npy'), clf_prob)
        np.save(os.path.join(out_dir, 'clf_pred.npy'), clf_pred)

    def write_out_recommendations(self):
        """
        If number of predicted spikes > classifier_params['min_suggestion_count']
        Write out electrode number, count, mean prediction probability, and 5,95th percentiles
        """
        waveform_thresh = self.classifier_params['min_suggestion_count']
        out_file_path = self.data_dir + '/waveform_classifier_recommendations.csv'
        count = len(self.pos_spike_dict['waveforms'])
        if count > waveform_thresh:
            percentile_5 = np.percentile(self.pos_spike_dict['prob'], 5)
            percentile_95 = np.percentile(self.pos_spike_dict['prob'], 95)
            mean_prob = np.mean(self.pos_spike_dict['prob'])
            columns = ['electrode', 'count', 'mean_prob',
                       'percentile_5', 'percentile_95']
            data = [self.electrode_num, count,
                    mean_prob, percentile_5, percentile_95]
            new_df = pd.DataFrame(dict(zip(columns, data)),
                                  index=[self.electrode_num])
            round_cols = ['mean_prob', 'percentile_5', 'percentile_95']
            if not os.path.exists(out_file_path):
                new_df.sort_index(inplace=True)
                new_df[round_cols] = new_df[round_cols].round(3)
                new_df.to_csv(out_file_path)
            else:
                # Load pandas dataframe
                df = pd.read_csv(out_file_path, index_col=0)
                if self.electrode_num in df.index.values:
                    # If electrode number already in df, replace
                    df.loc[self.electrode_num, columns] = data
                else:
                    # Append new data to df
                    df = pd.concat([df, new_df])
                # Write out updated frame
                df.sort_index(inplace=True)
                df[round_cols] = df[round_cols].round(3)
                df.to_csv(out_file_path)

    def gen_plots(self):
        # Create plot for predicted spikes
        fig = plt.figure(figsize=(5, 10))
        gs = fig.add_gridspec(3, 2,
                              width_ratios=(4, 1), height_ratios=(1, 1, 1),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.05, hspace=0.05)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[1, 0])
        ax2 = fig.add_subplot(gs[2, 0], sharex=ax1)
        hist_ax = fig.add_subplot(gs[1, 1], sharey=ax1)
        waveform_data = self.pos_spike_dict['waveforms']
        x = np.arange(waveform_data.shape[1])
        # Adjust ylims to 8-SD of amplitudes to avoid domination
        # by outliers
        slice_mid = waveform_data.shape[1]//2
        amp_sd = np.std(waveform_data[:, slice_mid])
        amp_mean = np.mean(waveform_data[:, slice_mid])
        amp_lims = [amp_mean - 8*amp_sd, amp_mean + 8*amp_sd]
        ax0.plot(x, waveform_data
                 [::10].T, c='k', alpha=0.05)
        ax0.set_ylim(amp_lims)
        ax1.scatter(self.pos_spike_dict['spiketimes'],
                    self.pos_spike_dict['prob'], s=1)
        ax1.set_ylim([0, 1.1])
        hist_ax.hist(self.pos_spike_dict['prob'],
                     orientation='horizontal', bins=30)
        # ax1.hexbin(self.pos_spike_dict['spiketimes'],
        #            self.pos_spike_dict['prob'],
        #             cmap='Greys')
        ax1.set_ylabel('Spike probability')
        ax2.hist(self.pos_spike_dict['spiketimes'], bins=50)
        ax2.set_ylabel('Binned Counts')
        ax2.set_xlabel('Time')
        fig.suptitle('Predicted Spike Waveforms' + '\n' +
                     f'Count : {len(self.pos_spike_dict["waveforms"])}')
        fig.savefig(os.path.join(self.plot_dir,
                                 f'{self.electrode_num:02}_pred_spikes.png'),
                    bbox_inches='tight')
        plt.close(fig)

        # Create hierarchical clutstering plot for predicted spikes
        data = trim_data(self.pos_spike_dict['waveforms'], 20000)
        features = self.feature_pipeline.transform(data)
        cut_label_array, map_dict, clust_range = \
            perform_agg_clustering(
                features,
                max_clusters=4)
        plot_waveform_dendogram(
            data,
            cut_label_array,
            clust_range,
            map_dict,
            plot_n=1000,
            save_path=os.path.join(self.plot_dir,
                                   f'{self.electrode_num:02}_pred_spikes_dendogram.png'))

        # Cluster noise and plot waveforms + times on single plot
        # Pull out noise info
        noise_slices, noise_times, noise_probs = self.neg_spike_dict.values()

        noise_transformed = self.feature_pipeline.transform(noise_slices)
        zscore_noise_slices = zscore(noise_slices, axis=-1)

        # Cluster noise and plot waveforms + times on single plot
        dat_thresh = 10000
        noise_train_set_inds = np.random.choice(np.arange(noise_slices.shape[0]),
                                                int(np.min((noise_slices.shape[0], dat_thresh))))

        noise_transformed_train = noise_transformed[noise_train_set_inds]

        # Don't need multiple restarts, this is just for visualization, not actual clustering
        gmm_model = gmm(
            random_state=0,
            n_components=5,
            max_iter=self.params_dict['clustering_params']['num_iter'],
            n_init=1,
            tol=self.params_dict['clustering_params']['thresh']
        ).fit(noise_transformed_train)
        predictions = gmm_model.predict(noise_transformed)

        clust_num = len(np.unique(predictions))
        fig, ax = plt.subplots(clust_num, 2, figsize=(20, 10), sharex='col')
        ax[0, 0].set_title('Waveforms')
        ax[0, 1].set_title('Spike Times')
        plot_max = 1000  # Plot at most this many waveforms
        for num in range(clust_num):
            this_dat = zscore_noise_slices[predictions == num]
            inds = np.random.choice(
                np.arange(this_dat.shape[0]),
                int(np.min((
                    this_dat.shape[0],
                    plot_max
                )))
            )
            this_dat = this_dat[inds]
            ax[num, 0].plot(this_dat.T, color='k', alpha=0.01)
            ax[num, 0].set_ylabel(f'Clust {num}')
            this_times = noise_times[predictions == num]
            ax[num, 1].hist(this_times, bins=100)

        fig.suptitle('Predicted Noise Waveforms' + '\n' +
                     f'Count : {noise_slices.shape[0]}')
        fig.savefig(os.path.join(self.plot_dir,
                                 f'{self.electrode_num:02}_pred_noise.png'),
                    bbox_inches='tight')
        plt.close(fig)
        # plt.show()


class electrode_handler():
    """
    Class to handle electrode data
    """

    def __init__(self, hdf5_path, electrode_num, params_dict):
        self.params_dict = params_dict
        self.hdf5_path = hdf5_path
        self.electrode_num = electrode_num

        hf5 = tables.open_file(hdf5_path, 'r')
        el_path = f'/raw/electrode{electrode_num:02}'
        if el_path in hf5:
            self.raw_el = hf5.get_node(el_path)[:]
        else:
            raise Exception(f'{el_path} not in HDF5')
        hf5.close()

    def preprocess_electrode(self):
        """
        Complete preprocessing pipeline for electrode data:
        1. Filter the electrode data
        2. Cut to integer seconds
        3. Calculate recording cutoff
        4. Make cutoff plot
        5. Apply cutoff to electrode data
        """
        # Filter electrode
        self.filter_electrode()

        # Cut to integer seconds
        self.cut_to_int_seconds()

        # Calculate recording cutoff
        self.calc_recording_cutoff()

        # Make cutoff plot
        self.make_cutoff_plot()

        # Apply cutoff to electrode data
        self.cutoff_electrode()

        return self.filt_el

    def filter_electrode(self):
        # Raw units get multiplied by 0.195 to get MICROVOLTS
        self.filt_el = clust.get_filtered_electrode(
            self.raw_el,
            freq=[self.params_dict['bandpass_lower_cutoff'],
                  self.params_dict['bandpass_upper_cutoff']],
            sampling_rate=self.params_dict['sampling_rate'],)
        # Delete raw electrode recording from memory
        del self.raw_el

    def adjust_to_sampling_rate(data, sampling_rate):
        return

    def cut_to_int_seconds(self):
        """
        Cut data to have integer number of seconds

        data: numpy array
        sampling_rate: int
        """
        data = self.filt_el
        sampling_rate = self.params_dict['sampling_rate']
        self.filt_el = data[:int(sampling_rate)*int(len(data)/sampling_rate)]

    def calc_recording_cutoff(self):
        keywords = (
            'filt_el',
            'sampling_rate',
            'voltage_cutoff',
            'max_breach_rate',
            'max_secs_above_cutoff',
            'max_mean_breach_rate_persec'
        )
        values = (
            self.filt_el,
            self.params_dict['sampling_rate'],
            self.params_dict['voltage_cutoff'],
            self.params_dict['max_breach_rate'],
            self.params_dict['max_secs_above_cutoff'],
            self.params_dict['max_mean_breach_rate_persec'],
        )
        kwarg_dict = dict(zip(keywords, values))
        (
            breach_rate,
            breaches_per_sec,
            secs_above_cutoff,
            mean_breach_rate_persec,
            recording_cutoff
        ) = return_cutoff_values(**kwarg_dict)

        self.recording_cutoff = recording_cutoff

    def make_cutoff_plot(self):
        """
        Makes a plot showing where the recording was cut off at

        filt_el: numpy array
        recording_cutoff: int
        """
        fig = plt.figure()
        # filt_el is in microvolts
        second_data = np.reshape(
            self.filt_el,
            (-1, self.params_dict['sampling_rate']))
        mean_data = np.mean(second_data, axis=1)
        std_data = np.std(second_data, axis=1)
        plt.plot(mean_data, label='Mean')
        plt.fill_between(
            x=np.arange(len(mean_data)),
            y1=mean_data + std_data,
            y2=mean_data - std_data,
            label='STD',
        )
        plt.axvline(self.recording_cutoff,
                    color='k', linewidth=2, linestyle='--',
                    label='Recording cutoff')
        plt.axhline(self.params_dict['voltage_cutoff'],
                    color='r', linewidth=2.0, linestyle='--',
                    label='Voltage cutoff')
        plt.axhline(-self.params_dict['voltage_cutoff'],
                    color='r', linewidth=2.0, linestyle='--')
        plt.xlabel('Recording time (secs)')
        plt.ylabel('Average voltage recorded per sec (microvolts)')
        plt.title(f'Recording length : {len(second_data)}s' + '\n' +
                  f'Cutoff time : {self.recording_cutoff}s')
        plt.legend()
        plt.yscale('symlog')
        fig.savefig(
            f'./Plots/{self.electrode_num:02}/cutoff_time.png',
            bbox_inches='tight')
        plt.close("all")

    def cutoff_electrode(self):
        # TODO: Add warning if recording cutoff before the end
        # Warning should be printed out to file AND printed
        self.filt_el = self.filt_el[:self.recording_cutoff *
                                    self.params_dict['sampling_rate']]


class spike_handler():
    """
    Class to handle processing of spikes
    """

    def __init__(self, filt_el, params_dict, dir_name, electrode_num):
        self.filt_el = filt_el
        self.params_dict = params_dict
        self.dir_name = dir_name
        self.electrode_num = electrode_num

    def process_spikes(self):
        """
        Complete spike processing pipeline:
        1. Extract waveforms from filtered electrode
        2. Dejitter spikes and sort by time

        Returns:
            tuple: (slices_dejittered, times_dejittered, threshold, mean_val)
        """
        # Extract waveforms
        self.extract_waveforms()

        # Dejitter spikes
        self.dejitter_spikes()

        return (self.slices_dejittered, self.times_dejittered,
                self.threshold, self.mean_val, self.MAD_val)

    def extract_waveforms(self):
        """
        Extract waveforms from filtered electrode
        """
        slices, spike_times, polarity, mean_val, threshold, MAD_val = \
            clust.extract_waveforms_abu(
                self.filt_el,
                spike_snapshot=[self.params_dict['spike_snapshot_before'],
                                self.params_dict['spike_snapshot_after']],
                sampling_rate=self.params_dict['sampling_rate'],
                threshold_mult=self.params_dict['waveform_threshold'])

        self.slices = slices
        self.spike_times = spike_times
        self.polarity = polarity
        self.mean_val = mean_val
        self.threshold = threshold
        self.MAD_val = MAD_val

    def dejitter_spikes(self):
        """
        Dejitter spikes
        """
        slices_dejittered, times_dejittered = clust.dejitter_abu3(
            self.slices,
            self.spike_times,
            polarity=self.polarity,
            spike_snapshot=[self.params_dict['spike_snapshot_before'],
                            self.params_dict['spike_snapshot_after']],
            sampling_rate=self.params_dict['sampling_rate'])

        # Sort data by time
        spike_order = np.argsort(times_dejittered)
        times_dejittered = times_dejittered[spike_order]
        slices_dejittered = slices_dejittered[spike_order]
        polarity = self.polarity[spike_order]

        self.slices_dejittered = slices_dejittered
        self.times_dejittered = times_dejittered
        self.polarity = polarity
        del self.slices
        del self.spike_times

    def interpolate_waveforms(self, target_sampling_rate=30000.0):
        """
        Interpolate waveforms to match expected sampling rate for classifier.

        This allows the classifier to work with data recorded at different
        sampling rates as long as pre/post windows are the same.
        """
        interpolated_slices = clust.interpolate_waveforms_to_30khz(
            self.slices_dejittered,
            spike_snapshot=[self.params_dict['spike_snapshot_before'],
                            self.params_dict['spike_snapshot_after']],
            actual_sampling_rate=self.params_dict['sampling_rate'],
            target_sampling_rate=target_sampling_rate
        )
        return interpolated_slices

    def extract_features(self,
                         slices_dejittered,
                         feature_transformer,
                         feature_names,
                         fitted_transformer=True,
                         retain_features=True,
                         ):

        self.feature_names = feature_names
        if fitted_transformer:
            spike_features = feature_transformer.transform(
                slices_dejittered)
        else:
            spike_features = feature_transformer.fit_transform(
                slices_dejittered)

        if retain_features:
            # Retain features in the object
            self.spike_features = spike_features.copy()
        return spike_features

    def return_feature(self, wanted_feature):
        wanted_inds = [i for i, x in enumerate(self.feature_names)
                       if wanted_feature in x]
        wanted_data = self.spike_features[:, wanted_inds]
        if any([x == 0 for x in wanted_data.shape]):
            raise Exception(f'Feature {wanted_feature} seems to have 0 shape')
        return wanted_data

    def write_out_spike_data(self):
        """
        Save the pca_slices, energy and amplitudes to the
        spike_waveforms folder for this electrode
        Save slices/spike waveforms and their times to their respective folders
        """
        to_be_saved = ['slices_dejittered',
                       'pca_slices',
                       'energy',
                       'amplitude',
                       'times_dejittered']

        slices_dejittered = self.slices_dejittered
        times_dejittered = self.times_dejittered
        # pca_inds = [i for i, x in enumerate(self.feature_names) if 'pca' in x]
        # pca_slices = self.spike_features[:, pca_inds]
        # energy_inds = [i for i, x in enumerate(self.feature_names) if 'energy' in x]
        # energy = self.spike_features[:, energy_inds]
        # amp_inds = [i for i, x in enumerate(self.feature_names) if 'amplitude' in x]
        # amplitude = self.spike_features[:,amp_inds]
        pca_slices = self.return_feature('pca')
        energy = self.return_feature('energy')
        amplitude = self.return_feature('amplitude')

        electrode_num_str = f'electrode{self.electrode_num:02}'
        waveform_dir = f'{self.dir_name}/spike_waveforms/{electrode_num_str}'
        spiketime_dir = f'{self.dir_name}/spike_times/{electrode_num_str}'
        save_paths = [f'{waveform_dir}/spike_waveforms.npy',
                      f'{waveform_dir}/pca_waveforms.npy',
                      f'{waveform_dir}/energy.npy',
                      f'{waveform_dir}/spike_amplitudes.npy',
                      f'{spiketime_dir}/spike_times.npy',
                      ]

        for key, path in zip(to_be_saved, save_paths):
            if locals()[key].size > 0:
                np.save(path, locals()[key])
            else:
                raise Exception(f'Feature {key} seems to have 0 size')


# Import utility functions from the new module


def gen_window_plots(
    filt_el,
    window_len,
    window_count,
    sampling_rate,
    spike_times,
    mean_val,
    threshold,
):
    windows_in_data = len(filt_el) // (window_len * sampling_rate)
    window_markers = np.linspace(0,
                                 int(windows_in_data*(window_len * sampling_rate)),
                                 int(windows_in_data))
    window_markers = np.array([int(x) for x in window_markers])
    chosen_window_inds = np.vectorize(int)(np.sort(np.random.choice(
        np.arange(windows_in_data), window_count)))
    chosen_window_markers = [(window_markers[x-1], window_markers[x])
                             for x in chosen_window_inds]
    chosen_windows = [filt_el[start:end]
                      for (start, end) in chosen_window_markers]
    # For each window, extract detected spikes
    chosen_window_spikes = [np.array(spike_times)
                            [(spike_times > start)*(spike_times < end)] - start
                            for (start, end) in chosen_window_markers]

    fig, ax = plt.subplots(len(chosen_windows), 1,
                           sharex=True, sharey=True, figsize=(10, 10))
    for dat, spikes, this_ax in zip(chosen_windows, chosen_window_spikes, ax):
        this_ax.plot(dat, linewidth=0.5)
        this_ax.hlines(mean_val + threshold, 0, len(dat))
        this_ax.hlines(mean_val - threshold, 0, len(dat))
        if len(spikes) > 0:
            this_ax.scatter(spikes, np.repeat(
                mean_val, len(spikes)), s=5, c='red')
        this_ax.set_ylim((mean_val - 1.5*threshold,
                          mean_val + 1.5*threshold))
    return fig


def gen_datashader_plot(
        slices_dejittered,
        cluster_points,
        x,
        threshold,
        electrode_num,
        sampling_rate,
        cluster,
):
    # Create a figure with two subplots - one for datashader and one for envelope
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1])

    # Datashader plot in the top subplot
    ax1 = fig.add_subplot(gs[0])
    _, ax1 = blech_waveforms_datashader.waveforms_datashader(
        slices_dejittered[cluster_points, :],
        x,
        downsample=False,
        threshold=threshold,
        dir_name="Plots/temp_plots/" +
        "datashader_temp_el" + str(electrode_num),
        ax=ax1)

    ax1.set_xlabel('Sample ({:d} samples per ms)'.
                   format(int(sampling_rate/1000)))
    ax1.set_ylabel('Voltage (microvolts)')
    ax1.set_title('Cluster%i Waveforms' % cluster)

    # Envelope plot in the bottom subplot
    ax2 = fig.add_subplot(gs[1])
    _, ax2 = blech_waveforms_datashader.waveform_envelope_plot(
        slices_dejittered[cluster_points, :],
        x,
        threshold=threshold,
        ax=ax2)

    ax2.set_xlabel('Sample ({:d} samples per ms)'.
                   format(int(sampling_rate/1000)))
    ax2.set_title('Cluster%i Mean ± Std Dev' % cluster)

    plt.tight_layout()
    return fig, ax1


def remove_too_large_waveforms(
        cluster_points,
        amplitudes,
        predictions,
        wf_amplitude_sd_cutoff
):
    this_cluster = predictions[cluster_points]
    cluster_amplitudes = amplitudes[cluster_points]
    cluster_amplitude_mean = np.mean(cluster_amplitudes)
    cluster_amplitude_sd = np.std(cluster_amplitudes)
    reject_wf = np.where(cluster_amplitudes <= cluster_amplitude_mean
                         - wf_amplitude_sd_cutoff*cluster_amplitude_sd)[0]
    this_cluster[reject_wf] = -1
    return this_cluster


def feature_timeseries_plot(
        standard_data,
        times_dejittered,
        feature_names,
        cluster_points
):
    this_standard_data = standard_data[cluster_points]
    this_spiketimes = times_dejittered[cluster_points]
    fig, ax = plt.subplots(this_standard_data.shape[1] + 1, 1,
                           figsize=(7, 9), sharex=True)
    for this_label, this_dat, this_ax in \
            zip(feature_names, this_standard_data.T, ax[:-1]):
        this_ax.scatter(this_spiketimes, this_dat,
                        s=0.5, alpha=0.5)
        this_ax.set_ylabel(this_label)
    ax[-1].hist(this_spiketimes, bins=50)
    ax[-1].set_ylabel('Spiketime' + '\n' + 'Histogram')
    return fig, ax

##############################
# Agglomerative clustering helper functions
##############################


def register_labels(x, y):
    """
    Register labels from one level to the next
    Input:
        x: labels at level n
        y: labels at level n+1
    Output:
        map_dict: dictionary mapping labels from level n to level n+1
    """
    unique_x = np.unique(x)
    x_cluster_y = [np.unique(y[x == i]) for i in unique_x]
    return dict(zip(unique_x, x_cluster_y))


def calc_linkage(model):
    """
    Calculate linkage matrix from sklearn agglomerative clustering model
    Input:
        model: sklearn agglomerative clustering model
    Output:
        linkage_matrix: linkage matrix
    """
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    return linkage_matrix

# Resort mapping so that children are always in order


def sort_label_array(label_array):
    """
    Resort map_dict so that children are always in order
    We will need a remapping at every level

    Input:
        label_array: levels x samples

    Output:
        label_array: levels x samples
    """
    n_levels = label_array.shape[0]
    level_pairs = list(zip(np.arange(n_levels-1), np.arange(1, n_levels)))
    for x, y in level_pairs:
        parent_labels = np.unique(label_array[x])
        child_labels = np.unique(label_array[y])
        child_map = {}
        highest_so_far = 0
        for this_parent in parent_labels:
            this_child = label_array[y][label_array[x] == this_parent]
            this_child = np.unique(this_child)
            wanted_labels = np.arange(
                highest_so_far, highest_so_far + len(this_child))
            highest_so_far = np.max(wanted_labels) + 1
            for i in range(len(this_child)):
                child_map[this_child[i]] = wanted_labels[i]
        # Remap
        for i in range(len(label_array[y])):
            label_array[y][i] = child_map[label_array[y][i]]
    return label_array


def perform_agg_clustering(features, max_clusters=8):
    """
    Perform agglomerative clustering on features
    Input:
        features: samples x features
        max_clusters: maximum number of clusters to consider
    Output:
        cut_label_array: levels x samples
        map_dict: dictionary mapping labels from level n to level n+1
        clust_range: range of clusters considered
    """
    clust_range = np.arange(1, max_clusters+1)
    ward = AgglomerativeClustering(
        distance_threshold=0,
        n_clusters=None,
        linkage="ward").fit(features)
    linkage = calc_linkage(ward)
    clust_label_list = [
        cut_tree(
            linkage,
            n_clusters=this_num
        ).flatten()
        for this_num in clust_range
    ]

    cut_label_array = np.stack(clust_label_list)
    cut_label_array = sort_label_array(cut_label_array)

    map_dict = {}
    for i in range(len(clust_range)-1):
        map_dict[i] = register_labels(
            cut_label_array[i],
            cut_label_array[i+1]
        )
    return cut_label_array, map_dict, clust_range


def plot_waveform_dendogram(data,
                            cut_label_array,
                            clust_range,
                            map_dict,
                            plot_n=1000,
                            save_path=None):
    if save_path is None:
        raise ValueError('Please provide a save path')
    slice_mid = data.shape[1]//2
    # Adjust ylims to 8-SD of amplitudes to avoid domination
    # by outliers
    amp_sd = np.std(data[:, slice_mid])
    amp_mean = np.mean(data[:, slice_mid])
    amp_lims = [amp_mean - 8*amp_sd, amp_mean + 8*amp_sd]
    # Plot dendogram
    fig, ax = plt.subplots(len(clust_range), np.max(clust_range)+1,
                           figsize=(7, 7), sharex=True, sharey=True)
    for row in range(len(clust_range)):
        center = int(np.max(clust_range)//2)
        labels = cut_label_array.T[:, row]
        unique_labels = np.unique(labels)
        med_label = int(np.median(unique_labels))
        ax_inds = unique_labels - med_label + center
        parent_unique_labels = np.unique(cut_label_array.T[:, row-1])
        parent_med_label = int(np.median(parent_unique_labels))
        parent_ax_inds = parent_unique_labels - parent_med_label + center
        ax[row, 0].set_ylabel(f'{clust_range[row]} Clusters')
        for x in unique_labels:
            this_dat = data[labels == x]
            if len(this_dat) > plot_n:
                plot_dat = this_dat[np.random.choice(len(this_dat), plot_n)]
            else:
                plot_dat = this_dat
            ax[row, ax_inds[x]].plot(plot_dat.T,
                                     color='k', alpha=0.01)
            ax[row, ax_inds[x]].set_ylim(amp_lims)
        if row > 0:
            this_map = map_dict[row-1]
            for key, val in this_map.items():
                for child in val:
                    con = ConnectionPatch(
                        xyA=(slice_mid, 0), coordsA=ax[row-1, parent_ax_inds[key]].transData,
                        xyB=(slice_mid, 0), coordsB=ax[row, ax_inds[child]].transData,
                        arrowstyle="-|>"
                    )
                    fig.add_artist(con)
    # Remove box round each subplot
    for ax0 in ax.flatten():
        ax0.axis('off')
    fig.suptitle('Predicted Waveform Dendogram')
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def trim_data(data, n_max=20000):
    """
    Agglomerative clustering doesn't like large datasets
    If we have more than n_max samples, we will randomly sample n_max samples

    Input:
        data: samples x features
        n_max: maximum number of samples to use

    Output:
        data: samples x features
    """
    if len(data) > n_max:
        data = data[np.random.choice(len(data), n_max, replace=False)]
    return data


def return_cutoff_values(
    filt_el,
    sampling_rate,
    voltage_cutoff,
    max_breach_rate,
    max_secs_above_cutoff,
    max_mean_breach_rate_persec
):
    """
    Return the cutoff values for the electrode recording

    Inputs:
        filt_el: numpy array (in microvolts)
        sampling_rate: int
        voltage_cutoff: float
        max_breach_rate: float
        max_secs_above_cutoff: float
        max_mean_breach_rate_persec: float

    Outputs:
        breach_rate: float
        breaches_per_sec: numpy array
        secs_above_cutoff: int
        mean_breach_rate_persec: float
        recording_cutoff: int
    """

    breach_rate = float(len(np.where(filt_el > voltage_cutoff)[0])
                        * int(sampling_rate))/len(filt_el)
    test_el = np.reshape(filt_el, (-1, sampling_rate))
    breaches_per_sec = (test_el > voltage_cutoff).sum(axis=-1)
    secs_above_cutoff = (breaches_per_sec > 0).sum()
    if secs_above_cutoff == 0:
        mean_breach_rate_persec = 0
    else:
        mean_breach_rate_persec = np.mean(breaches_per_sec[
            breaches_per_sec > 0])

    # And if they all exceed the cutoffs,
    # assume that the headstage fell off mid-experiment
    recording_cutoff = int(len(filt_el)/sampling_rate)
    if breach_rate >= max_breach_rate and \
            secs_above_cutoff >= max_secs_above_cutoff and \
            mean_breach_rate_persec >= max_mean_breach_rate_persec:
        # Find the first 1 second epoch where the number of cutoff breaches
        # is higher than the maximum allowed mean breach rate
        recording_cutoff = np.where(breaches_per_sec >
                                    max_mean_breach_rate_persec)[0][0]

    return (breach_rate, breaches_per_sec, secs_above_cutoff,
            mean_breach_rate_persec, recording_cutoff)
