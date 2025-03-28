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
import argparse  # noqa
parser = argparse.ArgumentParser(
    description='Process single electrode waveforms')
parser.add_argument('data_dir', type=str, help='Path to data directory')
parser.add_argument('electrode_num', type=int,
                    help='Electrode number to process')
args = parser.parse_args()

# Set environment variables to limit the number of threads used by various libraries
# Do it at the start of the script to ensure it applies to all imported libraries
import os  # noqa
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
from utils.blech_utils import imp_metadata, pipeline_graph_check  # noqa

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

# Perform pipeline graph check
script_path = os.path.realpath(__file__)
this_pipeline_check = pipeline_graph_check(data_dir_name)
this_pipeline_check.check_previous(script_path)
this_pipeline_check.write_to_log(script_path, 'attempted')

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

    if classifier_params['throw_out_noise'] or auto_cluster:
        print('== Throwing out noise waveforms ==')
        # Remaining data is now only spikes
        slices_dejittered, times_dejittered, clf_prob = \
            classifier_handler.pos_spike_dict.values()
        spike_set.slices_dejittered = slices_dejittered
        spike_set.times_dejittered = times_dejittered
        classifier_handler.clf_prob = clf_prob
        classifier_handler.clf_pred = clf_prob > classifier_handler.clf_threshold

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

spike_set.write_out_spike_data()


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
        )
        cluster_handler.perform_prediction()
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
    )
    cluster_handler.perform_prediction()
    cluster_handler.remove_outliers(params_dict)
    cluster_handler.calc_mahalanobis_distance_matrix()
    cluster_handler.save_cluster_labels()
    cluster_handler.create_output_plots(params_dict)
    if classifier_params['use_classifier'] and \
            classifier_params['use_neuRecommend']:
        cluster_handler.create_classifier_plots(classifier_handler)


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
