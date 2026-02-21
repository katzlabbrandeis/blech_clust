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
import time
import argparse  # noqa
import os  # noqa
from blech_clust.utils.blech_utils import imp_metadata, pipeline_graph_check  # noqa

test_bool = False
if test_bool:
    args = argparse.Namespace(
        data_dir='/media/storage/abu_resorted/gc_only/AM34_4Tastes_201215_115133/',
        electrode_num=0
    )
else:
    parser = argparse.ArgumentParser(
        description='Process single electrode waveforms')
    parser.add_argument('data_dir', type=str, help='Path to data directory')
    parser.add_argument('electrode_num', type=int,
                        help='Electrode number to process')
    args = parser.parse_args()

    # Perform pipeline graph check
    script_path = os.path.realpath(__file__)
    this_pipeline_check = pipeline_graph_check(args.data_dir)
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
import blech_clust.utils.blech_process_utils as bpu  # noqa
from itertools import product  # noqa

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
data_dir_name = args.data_dir

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

# Ensure top-level structure exists
if 'complete' not in process_log:
    process_log['complete'] = {}
if 'attempted' not in process_log:
    process_log['attempted'] = {}

# Remove electrode from 'complete' if it exists (re-processing)
if str(electrode_num) in process_log['complete']:
    del process_log['complete'][str(electrode_num)]

# Log processing start under 'attempted'
process_log['attempted'][str(electrode_num)] = {
    'start_time': datetime.datetime.now().isoformat(),
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

# Run complete preprocessing pipeline
filtered_data = electrode.preprocess_electrode()

#############################################################
# Process Spikes
#############################################################

# Extract and process spikes from filtered data
spike_set = bpu.spike_handler(filtered_data,
                              params_dict, data_dir_name, electrode_num)
(
    slices_dejittered,
    times_dejittered,
    threshold,
    mean_val,
    MAD_val,
) = spike_set.process_spikes()

# Write MAD_val and threshold to electrode_layout_frame
# Reload layout to make sure we have the latest version
# Run in a backoff loop to handle any errors with writing / mutliple processes
write_success = False
backoff_time = 1
while not write_success and backoff_time < 20:
    try:
        metadata_handler.load_layout()
        ind = metadata_handler.layout.electrode_ind == electrode_num
        metadata_handler.layout.at[ind, 'mad_val'] = MAD_val
        metadata_handler.layout.at[ind, 'threshold'] = threshold
        metadata_handler.layout.to_csv(
            metadata_handler.layout_file_path,
        )
        write_success = True
    except:
        print(
            f'Failed to write MAD val to layout file, retrying in {backoff_time} seconds')
        time.sleep(backoff_time)
        backoff_time *= 2

############################################################
# Extract windows from filt_el and plot with threshold overlayed
window_len = 0.2  # sec
window_count = 10
fig = bpu.gen_window_plots(
    filtered_data,
    window_len,
    window_count,
    params_dict['sampling_rate'],
    times_dejittered,
    mean_val,
    threshold,
)
fig.savefig(f'./Plots/{electrode_num:02}/bandapass_trace_snippets.png',
            bbox_inches='tight', dpi=300)
plt.close(fig)

# ------------------------------------------------------------
# Rolling threshold plot (only if rolling threshold is enabled)
if params_dict.get('use_rolling_threshold', False):
    window_len = params_dict.get('rolling_threshold_window', 5.0)
    step_len = params_dict.get('rolling_threshold_step', 5.0)
    rt_times, rt_thresholds = bpu.compute_rolling_threshold(
        filtered_data,
        params_dict['sampling_rate'],
        window_len=window_len,
        step_len=step_len,
        threshold_mult=params_dict['waveform_threshold'],
    )
    rolling_fig, _ = bpu.plot_rolling_threshold(rt_times, rt_thresholds)
    rolling_fig.savefig(
        f'./Plots/{electrode_num:02}/rolling_threshold.png',
        bbox_inches='tight', dpi=300,
    )
    plt.close(rolling_fig)

    # Save rolling thresholds for QA grid plot
    rolling_thresh_dir = './QA_output/rolling_thresholds'
    os.makedirs(rolling_thresh_dir, exist_ok=True)
    np.savez(
        f'{rolling_thresh_dir}/electrode{electrode_num:02}_rolling_threshold.npz',
        times=rt_times,
        thresholds=rt_thresholds,
        electrode_num=electrode_num,
    )
# ------------------------------------------------------------

############################################################

# Delete filtered electrode from memory
del electrode

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
    if classifier_params['classifier_threshold_override']['override'] is not False:
        clf_threshold = classifier_params['classifier_threshold_override']['threshold']
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

    if classifier_params['throw_out_noise'] or auto_cluster:
        throw_out_noise_bool = True
        print('== Throwing out noise waveforms ==')
        # Make copy of the original data
        slices_og = spike_set.slices_dejittered.copy()
        times_og = spike_set.times_dejittered.copy()
        clf_prob_og = classifier_handler.clf_prob.copy()

        # Remaining data is now only spikes
        slices_dejittered, times_dejittered, clf_prob = \
            classifier_handler.pos_spike_dict.values()
        # Update internal attributes of spike_set for later feature_extraction
        spike_set.slices_dejittered = slices_dejittered
        spike_set.times_dejittered = times_dejittered
        # classifier_handler.clf_prob = clf_prob
        # classifier_handler.clf_pred = clf_prob > classifier_handler.clf_threshold
    else:
        throw_out_noise_bool = False
        slices_og = spike_set.slices_dejittered
        times_og = spike_set.times_dejittered
        clf_prob_og = classifier_handler.clf_prob
else:
    # Make sure throw_out_noise bool exists
    throw_out_noise_bool = False

############################################################

if classifier_params['use_neuRecommend']:
    # If full classification pipeline was not loaded, still use
    # feature transformation pipeline
    print('Using neuRecommend features')
    feature_pipeline = classifier_handler.feature_pipeline
    feature_names = classifier_handler.feature_names
    use_fitted_transformer = True
else:
    print('Using blech_spike_features')
    import utils.blech_spike_features as bsf
    bsf_feature_pipeline = bsf.return_feature_pipeline(data_dir_name)
    # Use the feature pipeline from blech_spike_features
    feature_pipeline = bsf_feature_pipeline
    feature_names = bsf.feature_names
    # Set fitted_transformer to False so transformer is fit to new data
    use_fitted_transformer = False

_ = spike_set.extract_features(
    slices_dejittered,
    feature_pipeline,
    feature_names,
    fitted_transformer=use_fitted_transformer,
    retain_features=True,
)

# If throw_out_noise is set, also get features for all waveforms
if throw_out_noise_bool:
    all_features = spike_set.extract_features(
        slices_og,
        feature_pipeline,
        feature_names,
        # If throw out noise is true, fitted transformer WILL be used
        # If not, this will not be run
        # Therefore, we don't need to worry about different transformations
        # for `all_features` and `spike_set.spike_features`
        fitted_transformer=use_fitted_transformer,
        # We don't want spike_set to be updated with all features
        retain_features=False,
    )
else:
    all_features = spike_set.spike_features

spike_set.write_out_spike_data()


if auto_cluster == False:
    print('=== Performing manual clustering ===')
    # Run GMM, from 2 to max_clusters
    max_clusters = params_dict['clustering_params']['max_clusters']
    iters = product(
        np.arange(2, max_clusters+1),
        ['manual']
    )
else:
    print('=== Performing auto_clustering ===')
    max_clusters = auto_params['max_autosort_clusters']
    iters = [
        (max_clusters, 'auto')
    ]

for cluster_num, fit_type in iters:
    # Pass specific data instead of the whole spike_set
    cluster_handler = bpu.cluster_handler(
        params_dict,
        data_dir_name,
        electrode_num,
        cluster_num,
        spike_features=spike_set.spike_features,
        slices_dejittered=slices_dejittered,
        times_dejittered=times_dejittered,
        threshold=spike_set.threshold,
        feature_names=spike_set.feature_names,
        fit_type=fit_type,
    )
    # Use the new simplified clustering method
    cluster_handler.perform_clustering()
    cluster_handler.ensure_continuous_labels()

    # At this point, cluster_handler has a trained GMM
    # If 'throw_out_noise', then get labels for all waveforms
    # otherwise, use the labels from the GMM
    if throw_out_noise_bool:
        print('=== GMM trained using only classified spikes ===')
        all_labels = cluster_handler.get_cluster_labels(
            all_features,
        )
        # Since GMM will return predictions using original labels,
        # if auto_clustering, will need to relabel
        if auto_cluster:
            all_labels = np.array(
                [cluster_handler.cluster_map[label] for label in all_labels]
            )
    else:
        all_labels = cluster_handler.labels
    cluster_handler.remove_outliers(params_dict)
    cluster_handler.calc_mahalanobis_distance_matrix()
    cluster_handler.save_cluster_labels()
    cluster_handler.create_output_plots(params_dict)
    # NOTE: Classifier plots will not have outliers removed
    if throw_out_noise_bool:
        print('=== Classifier plots will NOT have outliers removed ===')
    if classifier_params['use_classifier'] and \
            classifier_params['use_neuRecommend']:
        cluster_handler.create_classifier_plots(
            # classifier_handler
            classifier_pred=clf_prob_og > classifier_handler.clf_threshold,
            classifier_prob=clf_prob_og,
            clf_threshold=classifier_handler.clf_threshold,
            all_waveforms=slices_og,
            all_times=times_og,
            labels=all_labels,
        )

print(f'Electrode {electrode_num} complete.')

# Update processing log with completion
with open(log_path) as f:
    process_log = json.load(f)

# Move electrode from 'attempted' to 'complete'
electrode_entry = process_log['attempted'].pop(str(electrode_num), {})
electrode_entry['end_time'] = datetime.datetime.now().isoformat()
process_log['complete'][str(electrode_num)] = electrode_entry

with open(log_path, 'w') as f:
    json.dump(process_log, f, indent=2)

# Write successful execution to log
if not test_bool:
    this_pipeline_check.write_to_log(script_path, 'completed')
