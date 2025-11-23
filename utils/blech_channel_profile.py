"""
This module generates plots for the entire timeseries of digital inputs (DIG_INs) and
amplifier (AMP) channels from data files in a specified directory. It handles two types
of file structures: one file per channel and one file per signal type.
"""

import glob
import os
import numpy as np
import pylab as plt
from tqdm import tqdm
import tables


def plot_channels(dir_path, qa_out_path, file_type, downsample=100, hdf5_name=None):
    """
    Generate plots for all channels and digital inputs
    
    Memory-efficient implementation that can load data from either:
    1. HDF5 file (if electrode data has been loaded) - preferred method
    2. Raw data files using memory mapping - fallback method
    
    This avoids loading entire datasets into RAM, which is critical for large recordings
    (e.g., 64 channels can otherwise consume 16GB+ of RAM).

    Args:
        dir_path: Directory containing the data files
        qa_out_path: Directory to save the plots
        file_type: Either 'one file per channel' or 'one file per signal type'
        downsample: Downsampling factor for plotting (default: 100)
        hdf5_name: Optional path to HDF5 file. If provided and contains data, will use HDF5 instead of raw files
    """
    if file_type not in ['one file per channel', 'one file per signal type']:
        raise ValueError(
            "file_type must be either 'one file per channel' or 'one file per signal type'")

    # Create plot dir
    plot_dir = os.path.join(qa_out_path, "channel_profile_plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    # Check if HDF5 file exists and has data
    use_hdf5 = False
    hf5 = None
    if hdf5_name is None:
        # Try to find HDF5 file in dir_path
        h5_search = glob.glob(os.path.join(dir_path, '*.h5'))
        if len(h5_search) > 0:
            hdf5_name = h5_search[0]
    
    if hdf5_name and os.path.exists(hdf5_name):
        try:
            hf5 = tables.open_file(hdf5_name, 'r')
            # Check if raw data exists in HDF5
            if '/raw' in hf5 and len(hf5.root.raw._v_children) > 0:
                use_hdf5 = True
                print(f"Using HDF5 file for data: {hdf5_name}")
            else:
                hf5.close()
                hf5 = None
        except Exception as e:
            print(f"Could not open HDF5 file: {e}")
            if hf5:
                hf5.close()
            hf5 = None

    # Get files to read
    if file_type == 'one file per channel':
        amp_files = glob.glob(os.path.join(dir_path, "amp*dat"))
        amp_files = sorted(amp_files)
        digin_files = sorted(glob.glob(os.path.join(dir_path, "board-DI*")))
    elif file_type == 'one file per signal type':
        amp_files = glob.glob(os.path.join(dir_path, "amp*dat"))
        digin_files = glob.glob(os.path.join(dir_path, "dig*dat"))
        # Use info file for port list calculation
        info_file_path = os.path.join(dir_path, 'info.rhd')
        if os.path.exists(info_file_path):
            info_file = np.fromfile(info_file_path, dtype=np.dtype('float32'))
            sampling_rate = int(info_file[2])
        else:
            print("info.rhd file not found. Please enter the sampling rate manually:")
            sampling_rate = int(input("Sampling rate (Hz): "))
        # Read the time.dat file for use in separating out the one file per signal type data
        num_recorded_samples = len(np.fromfile(
            os.path.join(dir_path, 'time.dat'), dtype=np.dtype('float32')))
        total_recording_time = num_recorded_samples/sampling_rate  # In seconds

    if not use_hdf5 and len(amp_files) < 1:
        raise Exception("Couldn't find amp*.dat files in dir" + "\n" +
                        f"{dir_path}")

    # Plot files
    print("Now plotting amplifier signals")
    row_lim = 8
    
    if use_hdf5:
        # Get electrode data from HDF5
        electrode_nodes = sorted([node._v_name for node in hf5.root.raw._f_iter_nodes()])
        num_electrodes = len(electrode_nodes)
        
        row_num = np.min((row_lim, num_electrodes))
        col_num = int(np.ceil(num_electrodes/row_num))
        
        # Create plot
        fig, ax = plt.subplots(row_num, col_num,
                               sharex=True, sharey=True, figsize=(15, 10))
        if num_electrodes == 1:
            ax = [ax]
        else:
            ax = ax.flatten()
        
        for node_name, this_ax in tqdm(zip(electrode_nodes, ax)):
            # Read data from HDF5 with downsampling
            data = hf5.root.raw._f_get_child(node_name)[::downsample]
            this_ax.plot(data)
            this_ax.set_ylabel(node_name)
        
        plt.suptitle('Amplifier Data (from HDF5)')
        fig.savefig(os.path.join(plot_dir, 'amplifier_data'))
        plt.close(fig)
        
    elif file_type == 'one file per channel':
        row_num = np.min((row_lim, len(amp_files)))
        col_num = int(np.ceil(len(amp_files)/row_num))
        # Create plot
        fig, ax = plt.subplots(row_num, col_num,
                               sharex=True, sharey=True, figsize=(15, 10))
        for this_file, this_ax in tqdm(zip(amp_files, ax.flatten())):
            # Use memory mapping to avoid loading entire file
            data_mmap = np.memmap(this_file, dtype=np.dtype('int16'), mode='r')
            # Only read downsampled indices to save memory
            downsampled_data = data_mmap[::downsample]
            this_ax.plot(downsampled_data)
            this_ax.set_ylabel("_".join(os.path.basename(this_file)
                                        .split('.')[0].split('-')[1:]))
            # Delete memmap to free memory
            del data_mmap, downsampled_data
        plt.suptitle('Amplifier Data')
        fig.savefig(os.path.join(plot_dir, 'amplifier_data'))
        plt.close(fig)
    elif file_type == 'one file per signal type':
        # Use memory mapping to avoid loading entire file into memory
        amplifier_data_mmap = np.memmap(amp_files[0], dtype=np.dtype('uint16'), mode='r')
        num_electrodes = int(len(amplifier_data_mmap)/num_recorded_samples)
        row_num = np.min((row_lim, num_electrodes))
        col_num = int(np.ceil(num_electrodes/row_num))
        # Create plot
        fig, ax = plt.subplots(row_num, col_num,
                               sharex=True, sharey=True, figsize=(15, 10))
        for e_i in tqdm(range(num_electrodes)):
            # Extract only the data for this electrode, downsampled
            # Data is interleaved: [e0_t0, e1_t0, ..., eN_t0, e0_t1, e1_t1, ...]
            electrode_indices = np.arange(e_i, len(amplifier_data_mmap), num_electrodes)
            # Downsample the indices
            downsampled_indices = electrode_indices[::downsample]
            data = amplifier_data_mmap[downsampled_indices]
            ax_i = plt.subplot(row_num, col_num, e_i+1)
            ax_i.plot(data)
            ax_i.set_ylabel('amp_' + str(e_i))
            del data, electrode_indices, downsampled_indices
        plt.suptitle('Amplifier Data')
        fig.savefig(os.path.join(plot_dir, 'amplifier_data'))
        plt.close(fig)
        del amplifier_data_mmap

    print("Now plotting digital input signals")
    
    # Digital inputs are not stored in HDF5, so always read from raw files
    if use_hdf5:
        print("Note: Digital inputs must be read from raw files (not stored in HDF5)")
    
    if file_type == 'one file per channel':
        fig, ax = plt.subplots(len(digin_files),
                               sharex=True, sharey=True, figsize=(8, 10))
        for this_file, this_ax in tqdm(zip(digin_files, ax.flatten())):
            # Use memory mapping to avoid loading entire file
            data_mmap = np.memmap(this_file, dtype=np.dtype('uint16'), mode='r')
            downsampled_data = data_mmap[::downsample]
            this_ax.plot(downsampled_data)
            this_ax.set_ylabel("_".join(os.path.basename(this_file)
                                        .split('.')[0].split('-')[1:]))
            del data_mmap, downsampled_data
        plt.suptitle('DIGIN Data')
        fig.savefig(os.path.join(plot_dir, 'digin_data'))
        plt.close(fig)
    elif file_type == 'one file per signal type':
        # Use memory mapping for digital inputs
        d_inputs_mmap = np.memmap(digin_files[0], dtype=np.dtype('uint16'), mode='r')
        # Process in chunks to reduce memory usage
        # For diff calculation, we need the full data but can work with views
        d_inputs_str = d_inputs_mmap.astype('str')
        d_in_str_int = d_inputs_str.astype('int64')
        d_diff = np.diff(d_in_str_int)
        # Clean up intermediate arrays
        del d_inputs_str, d_in_str_int
        
        dig_in = list(np.unique(np.abs(d_diff)) - 1)
        dig_in.remove(-1)
        num_dig_ins = len(dig_in)
        
        # Instead of creating full dig_inputs array, process each channel separately
        fig, ax = plt.subplots(num_dig_ins, 1,
                               sharex=True, sharey=True, figsize=(8, 10))
        for d_i, n_i in tqdm(enumerate(range(num_dig_ins))):
            # Create sparse representation for this channel only
            start_ind = np.where(d_diff == n_i + 1)[0]
            end_ind = np.where(d_diff == -1*(n_i + 1))[0]
            # Create downsampled output directly
            dig_input_downsampled = np.zeros(int(np.ceil(len(d_inputs_mmap) / downsample)))
            for s_i in range(len(start_ind)):
                start_ds = start_ind[s_i] // downsample
                end_ds = end_ind[s_i] // downsample
                dig_input_downsampled[start_ds:end_ds] = 1
            
            ax_i = plt.subplot(num_dig_ins, 1, d_i+1)
            ax_i.plot(dig_input_downsampled)
            ax_i.set_ylabel('Dig_in_' + str(dig_in[d_i]))
            del dig_input_downsampled
        
        plt.suptitle('DIGIN Data')
        fig.savefig(os.path.join(plot_dir, 'digin_data'))
        plt.close(fig)
        del d_inputs_mmap, d_diff
    
    # Close HDF5 file if it was opened
    if hf5 is not None:
        hf5.close()


if __name__ == '__main__':
    import argparse
    import sys

    # Create argument parser
    parser = argparse.ArgumentParser(description='Plots DIG_INs and AMP files')

    # Create argument parser
    parser.add_argument('dir_path', type=str,
                        help='The directory containing the data files')
    parser.add_argument('--file-type', type=str, required=True,
                        choices=['one file per channel',
                                 'one file per signal type'],
                        help='The type of file organization')
    parser.add_argument('--downsample', type=int, default=100,
                        help='Downsampling factor for plotting (default: 100)')
    parser.add_argument('--hdf5', type=str, default=None,
                        help='Path to HDF5 file (optional, will auto-detect if not provided)')
    args = parser.parse_args()

    dir_path = args.dir_path

    plot_channels(dir_path, dir_path, args.file_type, args.downsample, args.hdf5)
