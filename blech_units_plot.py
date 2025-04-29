"""
This module processes neural data stored in an HDF5 file, generating and saving plots of unit waveforms,
inter-spike interval (ISI) histograms, and spike count histograms. It also logs the execution of the processing pipeline.
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


def setup_environment(args):
    """
    Set up the environment for processing neural data.

    Args:
        args: Command line arguments

    Returns:
        tuple: metadata_handler, dir_name, params_dict, layout_frame, pipeline_check
    """
    # Get name of directory with the data files
    metadata_handler = imp_metadata(args)
    dir_name = metadata_handler.dir_name

    # Perform pipeline graph check
    script_path = os.path.realpath(__file__)
    this_pipeline_check = pipeline_graph_check(dir_name)
    this_pipeline_check.check_previous(script_path)
    this_pipeline_check.write_to_log(script_path, 'attempted')

    os.chdir(dir_name)
    print(f'Processing : {dir_name}')

    params_dict = metadata_handler.params_dict
    layout_frame = metadata_handler.layout

    return metadata_handler, dir_name, params_dict, layout_frame, this_pipeline_check, script_path


def prepare_output_directory(output_dir="unit_waveforms_plots"):
    """
    Create a clean output directory for storing plots.

    Args:
        output_dir: Directory name for storing plots
    """
    # Delete and remake a directory for storing the plots of the unit waveforms (if it exists)
    try:
        shutil.rmtree(output_dir, ignore_errors=True)
    except:
        pass
    os.mkdir(output_dir)


def load_units_data(hdf5_name):
    """
    Load units data from HDF5 file.

    Args:
        hdf5_name: Path to the HDF5 file

    Returns:
        tuple: hf5 file handle, units data, min_time, max_time
    """
    # Open the hdf5 file
    hf5 = tables.open_file(hdf5_name, 'r+')

    # Get all the units from the hdf5 file
    units = hf5.list_nodes('/sorted_units')

    # Find min-max time for plotting
    min_time = np.min([x.times[0] for x in units])
    max_time = np.max([x.times[-1] for x in units])

    return hf5, units, min_time, max_time


def plot_unit_summary(
        unit_data,
        min_time,
        max_time,
        unit_index=None,
        unit_descriptor=None,
        layout_frame=None,
        params_dict=None,
        output_dir="unit_waveforms_plots",
        return_only=False,
):
    """
    Generate and save a summary plot for a single neural unit.

    Args:
        unit_index: Index of the unit
        unit_data: Data for the unit
            - Contains: waveforms, times
        unit_descriptor: Descriptor information for the unit
            - Dictionary-like object with keys: electrode_number, single_unit, regular_spiking, fast_spiking, snr
        layout_frame: Layout information
        params_dict: Parameters dictionary
        min_time: Minimum time for plotting
        max_time: Maximum time for plotting
        output_dir: Directory to save plots
    """
    waveforms = unit_data.waveforms[:]
    x = np.arange(waveforms.shape[1]) + 1
    times = unit_data.times[:]
    ISIs = np.diff(times)

    # Get threshold from layout_frame
    if (unit_index is None) and (unit_descriptor is None):
        layout_ind = layout_frame['electrode_num'] == unit_descriptor['electrode_number']
        threshold = layout_frame['threshold'][layout_ind].values[0]
    else:
        threshold = None

    if (unit_index is not None) and (unit_descriptor is not None):
        title_text = f'Unit {unit_index}, total waveforms = {waveforms.shape[0]}' + \
            f'\nElectrode: {unit_descriptor["electrode_number"]}, ' + \
            f'Single Unit: {unit_descriptor["single_unit"]}, ' + \
            f'RSU: {unit_descriptor["regular_spiking"]}, ' + \
            f'FS: {unit_descriptor["fast_spiking"]}, ' + \
            f'SNR: {unit_descriptor["snr"]:.2f}'
    else:
        title_text = None

    fig, ax = plt.subplots(2, 2, figsize=(8, 6), dpi=200)

    if title_text is None:
        fig.suptitle(title_text, fontsize=12)

    # Plot datashader waveforms
    _, ax[0, 0] = blech_waveforms_datashader.waveforms_datashader(
        waveforms,
        x,
        downsample=False,
        threshold=threshold,
        ax=ax[0, 0]
    )
    ax[0, 0].set_xlabel('Sample (30 samples per ms)')
    ax[0, 0].set_ylabel('Voltage (microvolts)')

    # Plot mean and SD waveforms
    ax[0, 1].plot(x, np.mean(waveforms, axis=0), linewidth=4.0)
    ax[0, 1].fill_between(
        x,
        np.mean(waveforms, axis=0) - np.std(waveforms, axis=0),
        np.mean(waveforms, axis=0) + np.std(waveforms, axis=0),
        alpha=0.4)
    # Plot threshold
    if threshold is not None:
        ax[0, 1].axhline(threshold, color='red', linewidth=1,
                         linestyle='--', alpha=0.5)
        ax[0, 1].axhline(-threshold, color='red', linewidth=1,
                         linestyle='--', alpha=0.5)
    ax[0, 1].set_xlabel('Sample (30 samples per ms)')

    # Plot ISI histogram
    bin_count = 25
    bins = np.linspace(min_time, max_time, bin_count)

    _, ax[1, 0] = gen_isi_hist(
        times,
        np.ones(len(times)) > 0,  # mark all as selected
        params_dict['sampling_rate'],
        ax=ax[1, 0],
    )

    # Plot spike count histogram
    ax[1, 1].hist(times, bins=bins)
    ax[1, 1].set_xlabel('Sample ind')
    ax[1, 1].set_ylabel('Spike count')
    ax[1, 1].set_title('Counts over time')

    plt.tight_layout()

    if return_only:
        return fig, ax
    else:
        fig.savefig(f'./{output_dir}/Unit{unit_index}.png',
                    bbox_inches='tight')
        plt.close("all")


def process_all_units(
        units,
        hf5,
        layout_frame,
        params_dict,
        min_time,
        max_time,
        output_dir="unit_waveforms_plots",
):
    """
    Process and plot all units.

    Args:
        units: Neural units data
        hf5: HDF5 file handle
        layout_frame: Layout information
        params_dict: Parameters dictionary
        min_time: Minimum time for plotting
        max_time: Maximum time for plotting
        output_dir: Directory to save plots
    """
    # Now plot the waveforms from the units in this directory one by one
    for unit in trange(len(units)):
        unit_descriptor = hf5.root.unit_descriptor[unit]
        plot_unit_summary(
            unit_data=units[unit],
            min_time=min_time,
            max_time=max_time,
            unit_index=unit,
            unit_descriptor=unit_descriptor,
            layout_frame=layout_frame,
            params_dict=params_dict,
            output_dir=output_dir
        )


def save_individual_plots(units, output_subdir="waveforms_only"):
    """
    Save individual datashader and average plots for each unit.

    Args:
        units: Neural units data
        output_subdir: Subdirectory name for individual plots
    """
    plot_dir = os.path.join('unit_waveforms_plots', output_subdir)
    os.mkdir(plot_dir)

    for unit in trange(len(units)):
        waveforms = units[unit].waveforms[:]
        x = np.arange(waveforms.shape[1]) + 1
        times = units[unit].times[:]

        # Save datashader plot
        fig, ax = blech_waveforms_datashader.waveforms_datashader(
            waveforms, x, downsample=False)
        ax.set_xlabel('Sample (30 samples per ms)')
        ax.set_ylabel('Voltage (microvolts)')
        fig.savefig(os.path.join(plot_dir, f'Unit{unit}_datashader.png'),
                    bbox_inches='tight')
        plt.close("all")

        # Save mean and SD plot
        fig, ax = plt.subplots()
        ax.plot(x, np.mean(waveforms, axis=0), linewidth=4.0)
        ax.fill_between(
            x,
            np.mean(waveforms, axis=0) - np.std(waveforms, axis=0),
            np.mean(waveforms, axis=0) + np.std(waveforms, axis=0),
            alpha=0.4)
        ax.set_xlabel('Sample (30 samples per ms)')
        fig.savefig(os.path.join(plot_dir, f'Unit{unit}_mean_sd.png'),
                    bbox_inches='tight')
        plt.close("all")


def main():
    """Main function to process neural data and generate plots."""
    # Setup environment
    metadata_handler, dir_name, params_dict, layout_frame, this_pipeline_check, script_path = setup_environment(
        sys.argv)

    # Prepare output directory
    prepare_output_directory()

    # Load data
    hf5, units, min_time, max_time = load_units_data(
        metadata_handler.hdf5_name)

    # Process all units
    process_all_units(units, hf5, layout_frame,
                      params_dict, min_time, max_time)

    # Save individual plots
    save_individual_plots(units)

    # Cleanup
    hf5.close()

    # Write successful execution to log
    this_pipeline_check.write_to_log(script_path, 'completed')


if __name__ == "__main__":
    main()
