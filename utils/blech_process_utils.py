
import utils.clustering as clust
# import subprocess
from joblib import load
from sklearn.mixture import GaussianMixture as gmm
from utils import blech_waveforms_datashader
import subprocess
from scipy.stats import zscore
import pylab as plt
import json
# import sys
import numpy as np
import tables
import os
import shutil
import matplotlib
import pandas as pd
matplotlib.use('Agg')

############################################################
# Define Functions
############################################################


def get_dir_names():
    home_dir = os.getenv('HOME')
    blech_clust_dir = os.path.join(home_dir, 'Desktop', 'blech_clust')
    blech_dir_path = os.path.join(blech_clust_dir, 'blech.dir')
    with open(blech_dir_path, 'r') as f:
        lines = f.readlines()
    dir_name = lines[0][:-1]
    return home_dir, blech_clust_dir, dir_name


class cluster_handler():
    """
    Class to handle clustering steps
    """

    def __init__(self, params_dict,
                 data_dir, electrode_num, cluster_num,
                 spike_set):
        self.params_dict = params_dict
        self.dat_thresh = 10e3
        self.data_dir = data_dir
        self.electrode_num = electrode_num
        self.cluster_num = cluster_num
        self.spike_set = spike_set
        self.create_output_dir()

    def return_training_set(self, data):
        """
        Return training set for clustering
        """
        train_set = data[
            np.random.choice(np.arange(data.shape[0]),
                             int(np.min((data.shape[0], self.dat_thresh))))]
        return train_set

    def fit_model(self, train_set, clusters):
        """
        Cluster waveforms
        """
        model = gmm(
            n_components=clusters,
            max_iter=self.params_dict['num_iter'],
            n_init=self.params_dict['num_restarts'],
            tol=self.params_dict['thresh']).fit(train_set)
        return model

    def get_cluster_labels(self, data, model):
        """
        Get cluster labels
        """
        return model.predict(data)

    def perform_prediction(self):
        full_data = self.spike_set.spike_features
        train_set = self.return_training_set(full_data)
        model = self.fit_model(train_set, self.cluster_num)
        labels = self.get_cluster_labels(full_data, model)
        self.labels = labels

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
                self.spike_set.amplitudes,
                self.labels,
                wf_amplitude_sd_cutoff)
            self.labels[cluster_points] = this_cluster

    def save_cluster_labels(self):
        np.save(
            os.path.join(
                self.clust_results_dir, 'predictions.npy'),
            self.labels)

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
            f'clusters{self.cluster_num}'
        )
        ifisdir_rmdir(clust_results_dir)
        ifisdir_rmdir(clust_plot_dir)
        os.makedirs(clust_results_dir)
        os.makedirs(clust_plot_dir)
        self.clust_results_dir = clust_results_dir
        self.clust_plot_dir = clust_plot_dir

    def create_classifier_plots(self, classifier_handler):
        """
        For each cluster, plot:
            1. Pred Spikes
            2. Pred Noise
            3. Distribution and timeseries of probability
            4. Histogram of prediction probability
            5. Histogram of times for spikes and noise

        Input data can come from classifier_handler
        """

        classifier_pred = classifier_handler.clf_pred
        classifier_prob = classifier_handler.clf_prob
        clf_threshold = classifier_handler.clf_threshold
        all_waveforms = self.spike_set.slices_dejittered
        all_times = self.spike_set.times_dejittered

        max_plot_count = 1000
        for cluster in np.unique(self.labels):
            cluster_bool = self.labels == cluster
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
                    spike_ax.axhline(self.spike_set.threshold,
                                     color='red', linestyle='--')
                    spike_ax.axhline(-self.spike_set.threshold,
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
                    noise_ax.axhline(self.spike_set.threshold,
                                     color='red', linestyle='--')
                    noise_ax.axhline(-self.spike_set.threshold,
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
                fig.savefig(os.path.join(
                    self.clust_plot_dir, f'Cluster{cluster}_classifier'))
                plt.close(fig)
    # return fig, ax

    def create_output_plots(self,
                            params_dict):

        slices_dejittered = self.spike_set.slices_dejittered
        times_dejittered = self.spike_set.times_dejittered
        standard_data = self.spike_set.spike_features
        feature_names = self.spike_set.feature_names
        threshold = self.spike_set.threshold
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

                # Create waveform datashader plot
                #############################
                fig, ax = gen_datashader_plot(
                    slices_dejittered,
                    cluster_points,
                    x,
                    threshold,
                    self.electrode_num,
                    params_dict['sampling_rate'],
                    cluster,
                )
                fig.savefig(os.path.join(
                    self.clust_plot_dir, f'Cluster{cluster}_waveforms'))
                plt.close("all")

                # Create ISI distribution plot
                #############################
                fig = gen_isi_hist(
                    times_dejittered,
                    cluster_points,
                )
                fig.savefig(os.path.join(
                    self.clust_plot_dir, f'Cluster{cluster}_ISIs'))
                plt.close("all")

                # Create features timeseries plot
                # And plot histogram of spiketimes
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
        self.download_neurecommend_models(home_dir, model_dir)

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

    def get_waveform_classifier_params(self):
        _, self.blech_clust_dir, _ = get_dir_names()
        params_file_path = os.path.join(
            self.blech_clust_dir,
            'params',
            'waveform_classifier_params.json')
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
                    df = df.append(new_df)
                # Write out updated frame
                df.sort_index(inplace=True)
                df[round_cols] = df[round_cols].round(3)
                df.to_csv(out_file_path)

    def gen_plots(self):
        fig = plt.figure(figsize=(5, 10))
        gs = fig.add_gridspec(3, 2,
                              width_ratios=(4, 1), height_ratios=(1, 1, 1),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.05, hspace=0.05)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[1, 0])
        ax2 = fig.add_subplot(gs[2, 0], sharex=ax1)
        hist_ax = fig.add_subplot(gs[1, 1], sharey=ax1)
        x = np.arange(self.pos_spike_dict['waveforms'].shape[1])
        ax0.plot(x, self.pos_spike_dict['waveforms']
                 [::10].T, c='k', alpha=0.05)
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
                                 f'{self.electrode_num}_pred_spikes.png'),
                    bbox_inches='tight')
        plt.close(fig)

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
            n_components=5,
            max_iter=self.params_dict['num_iter'],
            n_init=1,
            tol=self.params_dict['thresh']).fit(noise_transformed_train)
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
                                 f'{self.electrode_num}_pred_noise.png'),
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

    def filter_electrode(self):
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
        second_data = np.reshape(
            self.filt_el,
            (-1, self.params_dict['sampling_rate']))
        plt.plot(np.mean(second_data, axis=1))
        plt.axvline(self.recording_cutoff,
                    color='k', linewidth=4.0, linestyle='--')
        plt.xlabel('Recording time (secs)')
        plt.ylabel('Average voltage recorded per sec (microvolts)')
        plt.title(f'Recording length : {len(second_data)}s' + '\n' +
                  f'Cutoff time : {self.recording_cutoff}s')
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
    Class to handler processing of spikes
    """

    def __init__(self, filt_el, params_dict, dir_name, electrode_num):
        self.filt_el = filt_el
        self.params_dict = params_dict
        self.dir_name = dir_name
        self.electrode_num = electrode_num

    def extract_waveforms(self):
        """
        Extract waveforms from filtered electrode
        """
        slices, spike_times, polarity, mean_val, threshold = \
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

    def extract_amplitudes(self):
        """
        Extract amplitudes from dejittered spikes
        """
        zero_ind = self.params_dict['spike_snapshot_before'] *\
            self.params_dict['sampling_rate']/1000
        zero_ind = int(zero_ind)
        self.amplitudes = self.slices_dejittered[:, zero_ind]

    # def pca_slices(self):
    #    """
    #    PCA on dejittered spikes
    #    """
    #    # Scale the dejittered slices by the energy of the waveforms
    #    scaled_slices, energy = clust.scale_waveforms(self.slices_dejittered)

    #    # Run PCA on the scaled waveforms
    #    pca_slices, explained_variance_ratio = clust.implement_pca(
    #        scaled_slices)

    # def create_pca_plot(self):
    #    # Create file for saving plots, and plot explained variance ratios of the PCA
    #    fig= plt.figure()
    #    x= np.arange(len(explained_variance_ratio))
    #    plt.plot(x, explained_variance_ratio, 'x')
    #    plt.title('Variance ratios explained by PCs')
    #    plt.xlabel('PC #')
    #    plt.ylabel('Explained variance ratio')
    #    fig.savefig(f'./Plots/{self.electrode_num:02}/pca_variance.png',
    #                bbox_inches='tight')
    #    plt.close("all")

    def extract_features(self,
                         feature_transformer,
                         feature_names,
                         fitted_transformer = True):

        self.feature_names = feature_names
        if fitted_transformer:
            self.spike_features = feature_transformer.transform(
                self.slices_dejittered)
        else:
            self.spike_features = feature_transformer.fit_transform(
                self.slices_dejittered)

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
        pca_inds = [i for i, x in enumerate(self.feature_names) if 'pca' in x]
        pca_slices = self.spike_features[:, pca_inds]
        energy = self.spike_features[:, np.where(
            self.feature_names == 'energy')[0]]
        amplitude = self.spike_features[:,
                                        np.where(self.feature_names == 'amplitude')[0]]

        waveform_dir = f'{self.dir_name}/spike_waveforms/electrode{self.electrode_num:02}'
        spiketime_dir = f'{self.dir_name}/spike_times/electrode{self.electrode_num:02}'
        save_paths = [f'{waveform_dir}/spike_waveforms.npy',
                      f'{waveform_dir}/pca_waveforms.npy',
                      f'{waveform_dir}/energy.npy',
                      f'{waveform_dir}/spike_amplitudes.npy',
                      f'{spiketime_dir}/spike_times.npy',
                      ]

        for key, path in zip(to_be_saved, save_paths):
            np.save(path, locals()[key])


def ifisdir_rmdir(dir_name):
    if os.path.isdir(dir_name):
        shutil.rmtree(dir_name)


def return_cutoff_values(
    filt_el,
    sampling_rate,
    voltage_cutoff,
    max_breach_rate,
    max_secs_above_cutoff,
    max_mean_breach_rate_persec
):

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
    chosen_window_inds = np.vectorize(np.int)(np.sort(np.random.choice(
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
    fig, ax = blech_waveforms_datashader.waveforms_datashader(
        slices_dejittered[cluster_points, :],
        x,
        downsample=False,
        threshold=threshold,
        dir_name="Plots/" + "datashader_temp_el" + str(electrode_num))

    ax.set_xlabel('Sample ({:d} samples per ms)'.
                  format(int(sampling_rate/1000)))
    ax.set_ylabel('Voltage (microvolts)')
    ax.set_title('Cluster%i' % cluster)
    return fig, ax


def gen_isi_hist(
        times_dejittered,
        cluster_points,
):
    fig = plt.figure()
    cluster_times = times_dejittered[cluster_points]
    ISIs = np.ediff1d(np.sort(cluster_times))
    ISIs = ISIs/30.0
    max_ISI_val = 20
    bin_count = 100
    neg_pos_ISI = np.concatenate((-1*ISIs, ISIs), axis=-1)
    hist_obj = plt.hist(
        neg_pos_ISI,
        bins=np.linspace(-max_ISI_val, max_ISI_val, bin_count))
    plt.xlim([-max_ISI_val, max_ISI_val])
    # Scale y-lims by all but the last value
    upper_lim = np.max(hist_obj[0][:-1])
    if upper_lim:
        plt.ylim([0, upper_lim])
    plt.title("2ms ISI violations = %.1f percent (%i/%i)"
              % ((float(len(np.where(ISIs < 2.0)[0])) /
                  float(len(cluster_times)))*100.0,
                 len(np.where(ISIs < 2.0)[0]),
                 len(cluster_times)) + '\n' +
              "1ms ISI violations = %.1f percent (%i/%i)"
              % ((float(len(np.where(ISIs < 1.0)[0])) /
                  float(len(cluster_times)))*100.0,
                 len(np.where(ISIs < 1.0)[0]), len(cluster_times)))
    return fig


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
