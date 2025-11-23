import os
import tables
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from numpy.linalg import norm
from scipy.optimize import minimize
from pprint import pprint as pp
from itertools import product
import json
import pandas as pd
from glob import glob
from joblib import Parallel, delayed
from functools import partial
from numba import njit

base_dir = '/home/abuzarmahmood/projects/blech_clust/_experimental/template_matching/'
plot_dir = os.path.join(base_dir, 'plots')
artifacts_dir = os.path.join(base_dir, 'artifacts')

# data_dir = '/home/abuzarmahmood/.blech_clust_test_data/KM45_5tastes_210620_113227_new'
data_dir = '/home/abuzarmahmood/Desktop/test_data/AC5_D4_odors_tastes_251102_090233'
# Find h5 file
h5_path = glob(os.path.join(data_dir, '*.h5'))[0]
h5 = tables.open_file(h5_path, mode='r')

# h5
# - root
#   - raw
#     - electrodeXX
# Get all electrode names
electrode_names = h5.root.raw._v_children.keys()
electrode_nums = [int(name.replace('electrode', '')) for name in electrode_names]

# Only electrode 2 and 29 have spikes
# Check how that compares with the template matching

# np.savez(
#     os.path.join(artifacts_dir, 'optimized_filters.npz'),
#     optimized_filters=optimized_filters,
#     filter_pca_components=filter_pca_components
#     )

# Load optimized filters
npz = np.load(os.path.join(artifacts_dir, 'optimized_filters.npz'))
# Load pca components
# optimized_filters = npz['optimized_filters'][0]
optimized_filters = npz['filter_pca_components'][0]

plt.plot(optimized_filters.T)
plt.show()

assert np.isclose(norm(optimized_filters), 1), "Optimized filters must be unit norm" 
assert np.isclose(np.mean(optimized_filters), 0), "Optimized filters must be mean subtracted"

# perform scaled cross-correlation with all electrodes
import blech_clust.utils.blech_process_utils as bpu  # noqa
from blech_clust.utils.blech_utils import imp_metadata, pipeline_graph_check  # noqa

path_handler = bpu.path_handler()
blech_clust_dir = path_handler.blech_clust_dir
data_dir_name = data_dir

metadata_handler = imp_metadata([[], data_dir_name])
params_dict = metadata_handler.params_dict

os.chdir(metadata_handler.dir_name)

this_plot_dir = os.path.join(
    plot_dir,
    'template_conv_results'
    )
os.makedirs(this_plot_dir, exist_ok=True)

# all_spike_waveforms = {}
# all_filtered_data = []
# for electrode_num in electrode_nums:
#
#     print(f"Processing Electrode {electrode_num}")
#     electrode = bpu.electrode_handler(
#         # metadata_handler.hdf5_name,
#         h5_path,
#         0,
#         params_dict)
#     raw_el = electrode.raw_el.copy()
#     filtered_data = electrode.preprocess_electrode()
#     all_filtered_data.append(filtered_data)
#
# all_filtered_data = np.stack(all_filtered_data, axis=0)

@njit
def scaled_xcorr(snippet, template):
    assert len(snippet) == len(template), "Snippet and template must be of same length"
    snippet = snippet - np.mean(snippet)
    snippet = snippet / norm(snippet)
    return np.dot(snippet, template)

threshold = 0.8
all_spike_waveforms = {}
for electrode_num in electrode_nums:

    print(f"Processing Electrode {electrode_num}")
    electrode = bpu.electrode_handler(
        # metadata_handler.hdf5_name,
        h5_path,
        electrode_num,
        params_dict)
    raw_el = electrode.raw_el.copy()
    filtered_data = electrode.preprocess_electrode()

    # Scaled cross-correlation
    # def scaled_cross_correlation(signal, template):
    #     assert np.isclose(np.mean(template), 0), "Template must be mean subtracted"
    #     assert norm(template) == 1, "Template must be normalized to unit norm"
    #     template_length = len(template)
    #     bin_generator = (signal[i:i + template_length] for i in range(len(signal) - template_length + 1))
    #     scc_values = []
    #     pbar = tqdm(bin_generator, total=(len(signal) - template_length + 1))
    #     for segment in pbar:
    #         # Mean subtract
    #         segment = segment - np.mean(segment)
    #         segment_norm = norm(segment)
    #         # Don't have to normalize since both segment and template are normalized
    #         scc = np.dot(segment, template) 
    #         scc_values.append(scc)
    #         pbar.set_description(f"Processing {len(scc_values)/(len(signal) - template_length + 1)*100:.2f}%") 



    # test_xcorr = scaled_xcorr(
    #     filtered_data[1000:1000 + len(optimized_filters)],
    #     template
    # )

    # data_snippets = np.lib.stride_tricks.sliding_window_view(
    #     all_filtered_data,
    #     window_shape=len(optimized_filters),
    #     axis=1
    # )
    data_snippets = np.lib.stride_tricks.sliding_window_view(
        filtered_data,
        window_shape=len(optimized_filters),
    )

    outs = [scaled_xcorr(data, optimized_filters) for data in tqdm(data_snippets)]
    outs = np.array(outs)
        
    # # # Plot filtered data downsampled
    downsample_factor = 100
    fig, ax = plt.subplots(3, 1, figsize=(15, 10), sharex=True, sharey=False)
    ax[0].plot(raw_el[::downsample_factor], color='gray', alpha=0.5)
    ax[0].set_title('Raw Data (Downsampled)')
    ax[1].plot(filtered_data[::downsample_factor], color='blue', alpha=0.5)
    ax[1].set_title('Filtered Data (Downsampled)')
    ax[2].plot(outs[::downsample_factor]**2, color='red', alpha=0.5)
    ax[2].set_title('Scaled Cross-Correlation Output (squared to emphasize peaks)')
    plt.xlabel('Samples (Downsampled)')
    plt.tight_layout()
    fig.savefig(
        os.path.join(
            this_plot_dir,
            f'electrode_{electrode_num}_data_and_scaled_xcorr.png'
            )
        )
    plt.close()

    # Pull out waveforms where abs(outs) > threshold
    spike_indices = np.where(np.abs(outs) > threshold)[0]
    spike_waveforms = data_snippets[spike_indices]
    spike_xcorr_values = outs[spike_indices]

    # Plot histogram of abs(outs)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(np.abs(outs[::downsample_factor]), bins=np.arange(0, 1.01, 0.01), color='green', alpha=0.7)
    plt.xlabel('Absolute Scaled Cross-Correlation Value')
    plt.ylabel('Frequency')
    plt.yscale('log')
    plt.axvline(threshold, color='red', linestyle='dashed', linewidth=2, label='Threshold')
    ax.set_title(f'Histogram of Absolute Scaled Cross-Correlation Output\n' +\
            f'Electrode {electrode_num}\nTotal Spikes Detected: {len(spike_indices)}')
    plt.tight_layout()
    # plt.show()
    plt.savefig(
        os.path.join(
            this_plot_dir,
            f'electrode_{electrode_num}_scaled_xcorr_histogram.png'
        )
    )
    plt.close()


    # Plot some spike waveforms
    fig, ax = plt.subplots(figsize=(10, 5))
    for waveform in spike_waveforms[:1000]:
        ax.plot(waveform, color='orange', alpha=0.05)
    ax.set_title('Detected Spike Waveforms')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    # plt.show()
    plt.savefig(
        os.path.join(
            this_plot_dir,
            f'electrode_{electrode_num}_detected_spike_waveforms.png'
        )
    )
    plt.close()

    all_spike_waveforms[electrode_num] = {
        'spike_indices': spike_indices,
        'spike_waveforms': spike_waveforms,
        'xcorr_values': spike_xcorr_values
    }

    # scaled_xcorr_partial = partial(
    #     scaled_xcorr,
    #     template=template
    # )
    #
    # outs = [scaled_xcorr_partial(data) for data in tqdm(data_snippets)]
    #
    # outs = Parallel(n_jobs=4)(
    #     delayed(scaled_xcorr_partial)(
    #         this_snippet 
    #     ) for this_snippet in tqdm(data_snippets)
    # )

# Save all spike waveforms
np.savez(
    os.path.join(artifacts_dir, 'template_matching_spike_waveforms.npz'),
    all_spike_waveforms=all_spike_waveforms
    )

############################################################
# Compare:
# 1) Timing of detected spikes with ground truth spikes from blech_clust
# 2) Classifier probs vs scaled xcorr values

all_spike_waveforms_npz = np.load(
    os.path.join(artifacts_dir, 'template_matching_spike_waveforms.npz'),
    allow_pickle=True
    )

all_xcorr_waveforms = all_spike_waveforms_npz['all_spike_waveforms'].item()

import blech_clust.utils.blech_post_process_utils as post_utils  # noqa

all_clust_waveforms = {}
for electrode_num in electrode_nums:
    load_bool, (
        spike_waveforms,
        spike_times,
        pca_slices,
        energy,
        amplitudes,
        predictions,
    ) = post_utils.load_data_from_disk(data_dir, electrode_num, 10)

    clf_list_path = glob(
            os.path.join(
                data_dir,
                f'spike_waveforms/electrode{electrode_num:02d}/clf_prob.npy'
                )
            )
    clf_probs = np.load(clf_list_path[0])

    all_clust_waveforms[electrode_num] = {
        'spike_waveforms': spike_waveforms,
        'spike_times': spike_times,
        'clf_probs': clf_probs
    }

for this_electrode_num in electrode_nums:
    xcorr_data = all_xcorr_waveforms[this_electrode_num]
    clust_data = all_clust_waveforms[this_electrode_num]

    xcorr_spike_times = xcorr_data['spike_indices']
    clust_spike_times = clust_data['spike_times']

    # Find waveforms within one waveform length
    waveform_length = clust_data['spike_waveforms'].shape[1]
    xcorr_matched_indices = []
    clust_matched_indices = []
    for xcorr_ind, xcorr_time in enumerate(xcorr_spike_times):
        diffs = np.abs(clust_spike_times - xcorr_time)
        if np.any(diffs <= waveform_length):
            clust_ind = np.argmin(diffs)
            clust_matched_indices.append(clust_ind)
            xcorr_matched_indices.append(xcorr_ind)

    xcorr_waveforms = xcorr_data['spike_waveforms']
    clust_waveforms = clust_data['spike_waveforms']

    # Plot waveform comparison
    fig, ax = plt.subplots(1,2, figsize=(15,5))
    for waveform in xcorr_waveforms[:1000]:
        ax[0].plot(waveform, color='orange', alpha=0.05)
    ax[0].set_title('Template Matching Detected Waveforms'+\
            f'\nTotal Detected: {len(xcorr_waveforms)}')
    for waveform in clust_waveforms[:1000]:
        ax[1].plot(waveform, color='purple', alpha=0.05)
    ax[1].set_title('Blech Clust Detected Waveforms'+\
            f'\nTotal Detected: {len(clust_waveforms)}')
    plt.suptitle(f'Electrode {this_electrode_num} Waveform Comparison')
    # plt.show()
    plt.tight_layout()
    fig.savefig( 
          os.path.join(
               this_plot_dir,
               f'electrode_{this_electrode_num}_waveform_comparison.png'
          )
     )
    plt.close()

    # For the matched indices, plot classifier probs vs scaled xcorr values
    matched_clf_probs = clust_data['clf_probs'][clust_matched_indices]
    matched_xcorr_values = xcorr_data['xcorr_values'][xcorr_matched_indices]
    fig, ax = plt.subplots(figsize=(10,5))
    ax.scatter(
        np.abs(matched_xcorr_values),
        matched_clf_probs,
        color='teal',
        alpha=0.5
    )
    ax.set_xlabel('Scaled Cross-Correlation Value')
    ax.set_ylabel('Classifier Probability')
    ax.set_title(f'Classifier Probability vs Scaled Cross-Correlation Value\nElectrode {this_electrode_num}')
    # plt.show()
    fig.savefig(
            os.path.join(
                 this_plot_dir,
                 f'electrode_{this_electrode_num}_clf_prob_vs_xcorr_value.png'
            )
         )
    plt.close()

    # Plot overlapping waveforms for matched spikes
    fig, ax = plt.subplots(1,2, figsize=(15,5))
    matched_clust_waveforms = clust_waveforms[clust_matched_indices]
    matched_xcorr_waveforms = xcorr_waveforms[xcorr_matched_indices]
    ax[0].plot(matched_xcorr_waveforms.T, color='orange', alpha=0.05)
    ax[0].set_title('Template Matching Matched Waveforms')
    ax[1].plot(matched_clust_waveforms.T, color='purple', alpha=0.05)
    ax[1].set_title('Blech Clust Matched Waveforms')
    plt.suptitle(f'Electrode {this_electrode_num} Matched Waveform Comparison\n'+\
            f'Number of Matched Spikes: {len(clust_matched_indices)}'
                 )
    # plt.show()
    fig.savefig( 
          os.path.join(
               this_plot_dir,
               f'electrode_{this_electrode_num}_matched_waveform_comparison.png'
          )
     )
    plt.close()

# Make plot of detected spike counts for all electrodes
all_xcorr_waveform_counts = [
    len(all_xcorr_waveforms[el_num]['spike_waveforms'])
    for el_num in electrode_nums
    ]

fig, ax = plt.subplots(figsize=(10,5))
ax.bar(electrode_nums, all_xcorr_waveform_counts, color='cyan', alpha=0.7)
# Put anootations on top of bars
for i, count in enumerate(all_xcorr_waveform_counts):
    ax.text(electrode_nums[i], count + 1, f"#{i}: {count}", ha='center', va='bottom', rotation=90, fontsize=8) 
ax.set_xlabel('Electrode Number')
ax.set_ylabel('Number of Detected Spikes (Template Matching)')
ax.set_title('Detected Spike Counts per Electrode (Template Matching)')
plt.tight_layout()
# plt.show()
fig.savefig(
    os.path.join(
        this_plot_dir,
        f'all_electrodes_detected_spike_counts_template_matching.png'
    )
)
plt.close()

