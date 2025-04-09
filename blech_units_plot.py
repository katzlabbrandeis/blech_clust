"""
This module processes neural data stored in an HDF5 file, generating and saving plots of unit waveforms, inter-spike interval (ISI) histograms, and spike count histograms. It also logs the execution of the processing pipeline.

- Imports necessary libraries and utility functions for data handling, plotting, and logging.
- Retrieves metadata and directory information using `imp_metadata`.
- Performs a pipeline graph check with `pipeline_graph_check` to ensure the correct sequence of operations.
- Opens an HDF5 file containing sorted neural units and retrieves unit data.
- Calculates the minimum and maximum times for plotting purposes.
- Creates a directory for storing waveform plots, deleting any existing directory with the same name.
- Iterates over each unit to:
  - Plot waveforms using `blech_waveforms_datashader`.
  - Plot mean and standard deviation of waveforms.
  - Generate ISI histograms using `gen_isi_hist`.
  - Plot spike count histograms over time.
  - Save the generated plots in the specified directory.
- Separately saves datashader and average unit plots in a subdirectory.
- Closes the HDF5 file after processing.
- Logs the successful completion of the pipeline.
"""
# Import stuff!
import numpy as np
import tables
import easygui
import sys
import os
import matplotlib.pyplot as plt
import shutil
from tqdm import tqdm, trange

# Import 3rd part code
from utils import blech_waveforms_datashader
from utils.blech_utils import imp_metadata, pipeline_graph_check
from utils.blech_process_utils import gen_isi_hist

# Get name of directory with the data files
metadata_handler = imp_metadata(sys.argv)
dir_name = metadata_handler.dir_name

# Perform pipeline graph check
script_path = os.path.realpath(__file__)
this_pipeline_check = pipeline_graph_check(dir_name)
this_pipeline_check.check_previous(script_path)
this_pipeline_check.write_to_log(script_path, 'attempted')

os.chdir(dir_name)
print(f'Processing : {dir_name}')

params_dict = metadata_handler.params_dict

# Open the hdf5 file
hf5 = tables.open_file(metadata_handler.hdf5_name, 'r+')

# Get all the units from the hdf5 file
units = hf5.list_nodes('/sorted_units')

# Find min-max time for plotting
min_time = np.min([x.times[0] for x in units])
max_time = np.max([x.times[-1] for x in units])

# Delete and remake a directory for storing the plots of the unit waveforms (if it exists)
try:
    shutil.rmtree("unit_waveforms_plots", ignore_errors=True)
except:
    pass
os.mkdir("unit_waveforms_plots")

# Now plot the waveforms from the units in this directory one by one
for unit in trange(len(units)):
    waveforms = units[unit].waveforms[:]
    # Convert sample indices to time in minutes
    sampling_rate = params_dict['sampling_rate']
    x = np.arange(waveforms.shape[1]) / \
        (sampling_rate * 60)  # Convert to minutes
    times = units[unit].times[:]
    ISIs = np.diff(times)

    fig, ax = plt.subplots(2, 2, figsize=(8, 6), dpi=200)
    fig.suptitle('Unit %i, total waveforms = %i' % (unit, waveforms.shape[0])
                 + '\n' + 'Electrode: %i, Single Unit: %i, RSU: %i, FS: %i' %
                 (hf5.root.unit_descriptor[unit]['electrode_number'],
                  hf5.root.unit_descriptor[unit]['single_unit'],
                  hf5.root.unit_descriptor[unit]['regular_spiking'],
                  hf5.root.unit_descriptor[unit]['fast_spiking']))

    _, ax[0, 0] = blech_waveforms_datashader.\
        waveforms_datashader(waveforms, x, downsample=False,
                             ax=ax[0, 0])
    ax[0, 0].set_xlabel('Time (minutes)')
    ax[0, 0].set_ylabel('Voltage (microvolts)')

    # Also plot the mean and SD for every unit -
    # downsample the waveforms 10 times to remove effects of upsampling during de-jittering
    # fig = plt.figure()
    ax[0, 1].plot(x, np.mean(waveforms, axis=0), linewidth=4.0)
    ax[0, 1].fill_between(
        x,
        np.mean(waveforms, axis=0) - np.std(waveforms, axis=0),
        np.mean(waveforms, axis=0) + np.std(waveforms, axis=0),
        alpha=0.4)
    ax[0, 1].set_xlabel('Time (minutes)')
    # Set ylim same as ax[0,0]
    # ax[0,1].set_ylim(ax[0,0].get_ylim())

    # Also plot time raster and ISI histogram for every unit
    ISI_threshold_ms = 10  # ms
    bin_count = 25
    bins = np.linspace(min_time, max_time, bin_count)

    _, ax[1, 0] = gen_isi_hist(
        times,
        np.ones(len(times)) > 0,  # mark all as selected
        params_dict['sampling_rate'],
        ax=ax[1, 0],
    )

    ax[1, 1].hist(times, bins=bins)
    ax[1, 1].set_xlabel('Sample ind')
    ax[1, 1].set_ylabel('Spike count')
    ax[1, 1].set_title('Counts over time')

    plt.tight_layout()
    fig.savefig('./unit_waveforms_plots/Unit%i.png' %
                (unit), bbox_inches='tight')
    plt.close("all")


# Also save datashader and average unit plots separately
plot_dir = os.path.join('unit_waveforms_plots', 'waveforms_only')
os.mkdir(plot_dir)

for unit in trange(len(units)):
    waveforms = units[unit].waveforms[:]
    # Convert sample indices to time in minutes
    sampling_rate = params_dict['sampling_rate']
    x = np.arange(waveforms.shape[1]) / \
        (sampling_rate * 60)  # Convert to minutes
    times = units[unit].times[:]
    ISIs = np.diff(times)

    fig, ax = blech_waveforms_datashader.\
        waveforms_datashader(waveforms, x, downsample=False,
                             )
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Voltage (microvolts)')
    fig.savefig(os.path.join(plot_dir, 'Unit%i_datashader.png' %
                (unit)), bbox_inches='tight')
    plt.close("all")

    # Also plot the mean and SD for every unit -
    # downsample the waveforms 10 times to remove effects of upsampling during de-jittering
    # fig = plt.figure()
    fig, ax = plt.subplots()
    ax.plot(x, np.mean(waveforms, axis=0), linewidth=4.0)
    ax.fill_between(
        x,
        np.mean(waveforms, axis=0) - np.std(waveforms, axis=0),
        np.mean(waveforms, axis=0) + np.std(waveforms, axis=0),
        alpha=0.4)
    ax.set_xlabel('Time (minutes)')
    fig.savefig(os.path.join(plot_dir, 'Unit%i_mean_sd.png' %
                (unit)), bbox_inches='tight')

hf5.close()

# Write successful execution to log
this_pipeline_check.write_to_log(script_path, 'completed')
