"""
This module processes single electrode waveforms for spike detection and clustering. It includes data loading, preprocessing, spike extraction, feature extraction, and clustering. The module also supports classification and logging of the processing steps.

- **Argument Parsing**: Parses command-line arguments for the data directory and electrode number.
- **Data Loading**: Loads data and metadata for the specified electrode, and initializes a processing log.
- **Preprocessing**:
  - Filters electrode data.
  - Calculates voltage parameters and cuts the recording accordingly.
  - Extracts spike times and waveforms.
  - Generates plots of filtered data with threshold overlays.
- **Spike Processing**:
  - Dejitters spike waveforms and extracts their maximum amplitudes.
- **Classification**:
  - Loads and applies a classifier if specified, using neuRecommend features.
  - Optionally throws out noise waveforms based on classification results.
- **Feature Extraction**:
  - Extracts features using either neuRecommend or blech_spike_features pipelines.
- **Clustering**:
  - Performs manual or automatic clustering using Gaussian Mixture Models (GMM).
  - Removes outliers and calculates Mahalanobis distance matrices.
  - Saves cluster labels and generates output plots.
- **Logging**:
  - Updates the processing log with start and end times, and status of processing.
  - Writes successful execution to a pipeline log.
"""

############################################################
# Imports
############################################################
import os  # noqa
import argparse  # noqa

test_bool = False

if test_bool:
    args = argparse.Namespace(
        data_dir='/home/abuzarmahmood/projects/blech_clust/pipeline_testing/test_data_handling/test_data/KM45_5tastes_210620_113227_new',
        electrode_num=0
    )
    data_dir_name = args.data_dir
else:
    parser = argparse.ArgumentParser(
        description='Process single electrode waveforms')
    parser.add_argument('data_dir', type=str, help='Path to data directory')
    parser.add_argument('electrode_num', type=int,
                        help='Electrode number to process')
    args = parser.parse_args()
    data_dir_name = args.data_dir

    from utils.blech_utils import imp_metadata, pipeline_graph_check  # noqa
    # Perform pipeline graph check
    script_path = os.path.realpath(__file__)
    this_pipeline_check = pipeline_graph_check(data_dir_name)
    this_pipeline_check.check_previous(script_path)
    this_pipeline_check.write_to_log(script_path, 'attempted')


# Set environment variables to limit the number of threads used by various libraries
# Do it at the start of the script to ensure it applies to all imported libraries
os.environ['OMP_NUM_THREADS'] = '1'  # noqa
os.environ['MKL_NUM_THREADS'] = '1'  # noqa
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # noqa

import pathlib  # noqa
import datetime  # noqa
import warnings  # noqa
import numpy as np  # noqa
import sys  # noqa
import json  # noqa
import pylab as plt  # noqa
import utils.blech_process_utils as bpu  # noqa

# Confirm sys.argv[1] is a path that exists
if not os.path.exists(args.data_dir):
    raise ValueError(f'Provided path {args.data_dir} does not exist')


# Ignore specific warning
warnings.filterwarnings(action="ignore", category=UserWarning,
                        message="Trying to unpickle estimator")

# Set seed to allow inter-run reliability
# Also allows reusing the same sorting sheets across runs
np.random.seed(0)

############################################################
# Load Data
############################################################

path_handler = bpu.path_handler()
blech_clust_dir = path_handler.blech_clust_dir

metadata_handler = imp_metadata([[], data_dir_name])
os.chdir(metadata_handler.dir_name)

electrode_num = int(args.electrode_num)
print(f'Processing electrode {electrode_num}')

# Initialize or load processing log
log_path = pathlib.Path(metadata_handler.dir_name) / 'blech_process.log'
if log_path.exists():
    with open(log_path) as f:
        process_log = json.load(f)
else:
    process_log = {}

# Log processing start
process_log[str(electrode_num)] = {
    'start_time': datetime.datetime.now().isoformat(),
    'status': 'attempted'
}
with open(log_path, 'w') as f:
    json.dump(process_log, f, indent=2)

params_dict = metadata_handler.params_dict
auto_params = params_dict['clustering_params']['auto_params']
auto_cluster = auto_params['auto_cluster']

# Check if the directories for this electrode number exist -
# if they do, delete them (existence of the directories indicates a
# job restart on the cluster, so restart afresh)
dir_list = [f'./Plots/{electrode_num:02}',
            f'./spike_waveforms/electrode{electrode_num:02}',
            f'./spike_times/electrode{electrode_num:02}',
            f'./clustering_results/electrode{electrode_num:02}']
for this_dir in dir_list:
    bpu.ifisdir_rmdir(this_dir)
    os.makedirs(this_dir)

############################################################
# Preprocessing
############################################################
# Open up hdf5 file, and load this electrode number
electrode = bpu.electrode_handler(
    metadata_handler.hdf5_name,
    electrode_num,
    params_dict)

electrode.filter_electrode()

# Calculate the 3 voltage parameters
electrode.cut_to_int_seconds()
electrode.calc_recording_cutoff()

# Dump a plot showing where the recording was cut off at
electrode.make_cutoff_plot()

# Then cut the recording accordingly
electrode.cutoff_electrode()

#############################################################
# Process Spikes
#############################################################

# Extract spike times and waveforms from filtered data
spike_set = bpu.spike_handler(electrode.filt_el,
                              params_dict, data_dir_name, electrode_num)
spike_set.extract_waveforms()

############################################################
# Extract windows from filt_el and plot with threshold overlayed
window_len = 0.2  # sec
window_count = 10
fig = bpu.gen_window_plots(
    electrode.filt_el,
    window_len,
    window_count,
    params_dict['sampling_rate'],
    spike_set.spike_times,
    spike_set.mean_val,
    spike_set.threshold,
)
fig.savefig(f'./Plots/{electrode_num:02}/bandapass_trace_snippets.png',
            bbox_inches='tight', dpi=300)
plt.close(fig)
############################################################

# Delete filtered electrode from memory
del electrode

# Dejitter these spike waveforms, and get their maximum amplitudes
# Slices are returned sorted by amplitude polaity
spike_set.dejitter_spikes()

############################################################
# Load classifier if specificed
classifier_params_path = \
    bpu.classifier_handler.return_waveform_classifier_params_path(
        blech_clust_dir)
classifier_params = json.load(open(classifier_params_path, 'r'))


if classifier_params['use_neuRecommend']:
    # If full classification pipeline was not loaded, still load
    # feature transformation pipeline so it may be used later
    classifier_handler = bpu.classifier_handler(
        data_dir_name, electrode_num, params_dict)
    sys.path.append(classifier_handler.create_pipeline_path)
    from feature_engineering_pipeline import *
    classifier_handler.load_pipelines()

    # If override_classifier_threshold is set, use that
    if classifier_params['override_classifier_threshold'] is not False:
        clf_threshold = classifier_params['threshold_override']
        print(f' == Overriding classifier threshold with {clf_threshold} ==')
        classifier_handler.clf_threshold = clf_threshold

if classifier_params['use_classifier'] and \
        classifier_params['use_neuRecommend']:
    print(' == Using neuRecommend classifier ==')
    # Full classification pipeline also has feature transformation pipeline
    classifier_handler.classify_waveforms(
        spike_set.slices_dejittered,
        spike_set.times_dejittered,
    )
    classifier_handler.gen_plots()
    classifier_handler.write_out_recommendations()

############################################################

if classifier_params['use_neuRecommend']:
    # If full classification pipeline was not loaded, still use
    # feature transformation pipeline
    print('Using neuRecommend features')
    spike_set.extract_features(
        classifier_handler.feature_pipeline,
        classifier_handler.feature_names,
        fitted_transformer=True,
    )
else:
    print('Using blech_spike_features')
    import utils.blech_spike_features as bsf
    bsf_feature_pipeline = bsf.return_feature_pipeline(data_dir_name)
    # Set fitted_transformer to False so transformer is fit to new data
    spike_set.extract_features(
        bsf_feature_pipeline,
        bsf.feature_names,
        fitted_transformer=False,
    )


throw_out_noise_bool = classifier_params['throw_out_noise'] and \
    classifier_params['use_classifier'] and \
    classifier_params['use_neuRecommend']

if auto_cluster == False:
    print('=== Performing manual clustering ===')
    # Run GMM, from 2 to max_clusters
    max_clusters = params_dict['clustering_params']['max_clusters']
    for cluster_num in range(2, max_clusters+1):
        cluster_handler = bpu.cluster_handler(
            params_dict,
            data_dir_name,
            electrode_num,
            cluster_num,
            spike_set,
            fit_type='manual',
            waveform_pred=classifier_handler.clf_pred,
        )
        cluster_handler.perform_prediction(
            throw_out_noise=throw_out_noise_bool)

        # Backup original data for plotting (aligned with cluster labels)
        cluster_handler.spike_set.slices_original = spike_set.slices_dejittered.copy()
        cluster_handler.spike_set.times_original = spike_set.times_dejittered.copy()
        classifier_handler.clf_prob_original = classifier_handler.clf_prob.copy()
        classifier_handler.clf_pred_original = classifier_handler.clf_prob > classifier_handler.clf_threshold

        # Remove noise at this step if wanted
        # This way, downstream processing is only on spikes
        if classifier_params['throw_out_noise']:
            print('== Throwing out noise waveforms for clustering ==')

            # Remaining data is now only spikes
            slices_dejittered, times_dejittered, clf_prob = \
                classifier_handler.pos_spike_dict.values()

            # Store original labels before modifying for downstream processing
            cluster_handler.labels_original = cluster_handler.labels.copy()

            # Store the original data for plotting later
            cluster_handler.spike_set.slices_for_clustering = slices_dejittered.copy()
            cluster_handler.spike_set.times_for_clustering = times_dejittered.copy()
            classifier_handler.clf_prob_for_clustering = clf_prob.copy()

            # Update the data for clustering
            cluster_handler.spike_set.slices_dejittered = slices_dejittered
            cluster_handler.spike_set.times_dejittered = times_dejittered
            classifier_handler.clf_prob = clf_prob
            # Reset prediction to match probability threshold for remaining spikes
            classifier_handler.clf_pred = clf_prob > classifier_handler.clf_threshold

        cluster_handler.remove_outliers(params_dict)
        cluster_handler.calc_mahalanobis_distance_matrix()
        cluster_handler.save_cluster_labels()
        cluster_handler.create_output_plots(params_dict)
        if classifier_params['use_classifier'] and \
                classifier_params['use_neuRecommend']:
            cluster_handler.create_classifier_plots(classifier_handler)
else:
    print('=== Performing auto_clustering ===')
    max_clusters = auto_params['max_autosort_clusters']
    cluster_handler = bpu.cluster_handler(
        params_dict,
        data_dir_name,
        electrode_num,
        max_clusters,
        spike_set,
        fit_type='auto',
        waveform_pred=classifier_handler.clf_pred,
    )
    cluster_handler.perform_prediction(throw_out_noise=throw_out_noise_bool)

    # Backup original data for plotting (aligned with cluster labels)
    cluster_handler.spike_set.slices_original = spike_set.slices_dejittered.copy()
    cluster_handler.spike_set.times_original = spike_set.times_dejittered.copy()
    classifier_handler.clf_prob_original = classifier_handler.clf_prob.copy()
    classifier_handler.clf_pred_original = classifier_handler.clf_prob > classifier_handler.clf_threshold

    # Remove noise at this step if wanted
    # This way, downstream processing is only on spikes
    if classifier_params['throw_out_noise']:
        print('== Throwing out noise waveforms for clustering ==')

        # Remaining data is now only spikes
        slices_dejittered, times_dejittered, clf_prob = \
            classifier_handler.pos_spike_dict.values()

        # Store original labels before modifying for downstream processing
        cluster_handler.labels_original = cluster_handler.labels.copy()

        # Store the original data for plotting later
        cluster_handler.spike_set.slices_for_clustering = slices_dejittered.copy()
        cluster_handler.spike_set.times_for_clustering = times_dejittered.copy()
        classifier_handler.clf_prob_for_clustering = clf_prob.copy()

        # Update the data for clustering
        cluster_handler.spike_set.slices_dejittered = slices_dejittered
        cluster_handler.spike_set.times_dejittered = times_dejittered
        classifier_handler.clf_prob = clf_prob
        # Reset prediction to match probability threshold for remaining spikes
        classifier_handler.clf_pred = clf_prob > classifier_handler.clf_threshold

    cluster_handler.remove_outliers(
        params_dict, throw_out_noise=throw_out_noise_bool)
    cluster_handler.calc_mahalanobis_distance_matrix(
        throw_out_noise=throw_out_noise_bool)
    cluster_handler.save_cluster_labels(throw_out_noise=throw_out_noise_bool)
    cluster_handler.create_output_plots(params_dict)

    # Plotting internally will use all original data
    if classifier_params['use_classifier'] and \
            classifier_params['use_neuRecommend']:
        # For classifier plots, use all waveforms including noise
        # Temporarily restore original data for plotting
        original_slices = cluster_handler.spike_set.slices_dejittered
        original_times = cluster_handler.spike_set.times_dejittered
        original_clf_prob = classifier_handler.clf_prob
        original_clf_pred = classifier_handler.clf_pred

        # Restore original data for plotting
        cluster_handler.spike_set.slices_dejittered = cluster_handler.spike_set.slices_original
        cluster_handler.spike_set.times_dejittered = cluster_handler.spike_set.times_original
        classifier_handler.clf_prob = classifier_handler.clf_prob_original
        classifier_handler.clf_pred = classifier_handler.clf_pred_original

        # Create plots with all waveforms
        cluster_handler.create_classifier_plots(classifier_handler)

        # Restore filtered data for further processing
        if classifier_params['throw_out_noise']:
            cluster_handler.spike_set.slices_dejittered = original_slices
            cluster_handler.spike_set.times_dejittered = original_times
            classifier_handler.clf_prob = original_clf_prob
            classifier_handler.clf_pred = original_clf_pred

if throw_out_noise_bool:
    # Updating features matrix as it will be written out
    # We only want to save the spike features for actual spikes
    cluster_handler.spike_set.spike_features = \
        cluster_handler.spike_set.spike_features[
            classifier_handler.clf_prob_original > classifier_handler.clf_threshold]

    # Make sure we're using the filtered data for writing
    cluster_handler.spike_set.slices_dejittered = cluster_handler.spike_set.slices_dejittered
    cluster_handler.spike_set.times_dejittered = cluster_handler.spike_set.times_dejittered
# Write out data after throw_out_noise step
cluster_handler.spike_set.write_out_spike_data()

print(f'Electrode {electrode_num} complete.')

# Update processing log with completion
with open(log_path) as f:
    process_log = json.load(f)
process_log[str(electrode_num)
            ]['end_time'] = datetime.datetime.now().isoformat()
process_log[str(electrode_num)]['status'] = 'complete'
with open(log_path, 'w') as f:
    json.dump(process_log, f, indent=2)

# Write successful execution to log
this_pipeline_check.write_to_log(script_path, 'completed')
