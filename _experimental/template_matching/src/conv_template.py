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

data_dir = '/home/abuzarmahmood/.blech_clust_test_data/KM45_5tastes_210620_113227_new'
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
    threshold = 0.9
    spike_indices = np.where(np.abs(outs) > threshold)[0]
    spike_waveforms = data_snippets[spike_indices]

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
        'spike_waveforms': spike_waveforms
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
