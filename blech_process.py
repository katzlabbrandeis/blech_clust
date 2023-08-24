"""
Changes to be made:
    1) Add flag for throwing out waveforms at blech_clust step
    2) Convert pred_spikes + pred_noice plots to datashader
    3) Number of noise clusters = 7
    4) Number of spikes clusters = 3
    5) Save predictions so they can be used in UMAP plot

Model monitoring:
    1) Monitoring input and output data distributions

Steps:
    1) Load Data
    2) Preprocess
        a) Bandpass filter
        b) Extract and dejitter spikes
        c) Extract amplitude (accounting for polarity)
        d) Extract energy and scale waveforms by energy
        e) Perform PCA
        f) Scale all features using StandardScaler
    3) Perform clustering
"""

############################################################
# Imports
############################################################

import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'

from utils.blech_utils import (
    imp_metadata,
)
import utils.blech_process_utils as bpu
from utils import memory_monitor as mm
import pylab as plt
import json
import sys
import numpy as np

# Set seed to allow inter-run reliability
# Also allows reusing the same sorting sheets across runs
np.random.seed(0)

from utils.blech_utils import imp_metadata


############################################################
# Load Data
############################################################

# Read blech.dir, and cd to that directory
path_handler = bpu.path_handler()
blech_clust_dir = path_handler.blech_clust_dir
data_dir_name = path_handler.data_dir


metadata_handler = imp_metadata([[], data_dir_name])
os.chdir(metadata_handler.dir_name)

electrode_num = int(sys.argv[1])
print(f'Processing electrode {electrode_num}')
params_dict = metadata_handler.params_dict

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
window_len= 0.2  # sec
window_count= 10
fig= bpu.gen_window_plots(
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
classifier_params = json.load(
    open(os.path.join(
        blech_clust_dir,
        'params/waveform_classifier_params.json'), 'r'))

if classifier_params['use_classifier'] and \
    classifier_params['use_neuRecommend']:
    classifier_handler = bpu.classifier_handler(
            data_dir_name, electrode_num, params_dict)
    sys.path.append(classifier_handler.create_pipeline_path)
    from feature_engineering_pipeline import *
    classifier_handler.load_pipelines()
    classifier_handler.classify_waveforms(
            spike_set.slices_dejittered,
            spike_set.times_dejittered,
            )
    classifier_handler.gen_plots()
    classifier_handler.write_out_recommendations()

    # throw_out_noise = True
    if classifier_params['throw_out_noise']:
        # Remaining data is now only spikes
        slices_dejittered, times_dejittered, clf_prob = \
            classifier_handler.pos_spike_dict.values()
        spike_set.slices_dejittered = slices_dejittered
        spike_set.times_dejittered = times_dejittered
        classifier_handler.clf_prob = clf_prob
        classifier_handler.clf_pred = clf_prob > classifier_handler.clf_threshold

############################################################

if classifier_params['use_neuRecommend']:
    spike_set.extract_features(
            classifier_handler.feature_pipeline,
            classifier_handler.feature_names,
            fitted_transformer=True,
            )
else:
    import utils.blech_spike_features as bsf
    spike_set.extract_features(
            bsf.feature_pipeline,
            bsf.feature_names,
            fitted_transformer=False,
            )

spike_set.write_out_spike_data()


# Set a threshold on how many datapoints are used to FIT the gmm
# Run GMM, from 2 to max_clusters
for cluster_num in range(2, params_dict['max_clusters']+1):
    cluster_handler = bpu.cluster_handler(
            params_dict, 
            data_dir_name, 
            electrode_num,
            cluster_num,
            spike_set)
    cluster_handler.perform_prediction()
    cluster_handler.remove_outliers(params_dict)
    cluster_handler.save_cluster_labels()
    cluster_handler.create_output_plots( 
                            params_dict)
    if classifier_params['use_classifier'] and \
        classifier_params['use_neuRecommend']:
        cluster_handler.create_classifier_plots(classifier_handler)


# Make file for dumping info about memory usage
f= open(f'./memory_monitor_clustering/{electrode_num:02}.txt', 'w')
print(mm.memory_usage_resource(), file=f)
f.close()
print(f'Electrode {electrode_num} complete.')
