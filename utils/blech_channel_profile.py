"""
This module generates plots for the entire timeseries of digital inputs (DIG_INs) and
amplifier (AMP) channels from data files in a specified directory. It handles three types
of file structures: one file per channel, one file per signal type, and traditional.
"""

import glob
import os
import numpy as np
import pylab as plt
from tqdm import tqdm
from blech_clust.utils.importrhdutilities import load_file


def plot_channels(dir_path, qa_out_path, file_type):
    """
    Generate plots for all channels and digital inputs

    Args:
        dir_path: Directory containing the data files
        qa_out_path: Directory to save the plots
        file_type: Either 'one file per channel', 'one file per signal type', or 'traditional'
    """
    if file_type not in ['one file per channel', 'one file per signal type', 'traditional']:
        raise ValueError(
            "file_type must be either 'one file per channel', 'one file per signal type', or 'traditional'")

    # Create plot dir
    plot_dir = os.path.join(qa_out_path, "channel_profile_plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

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
    elif file_type == 'traditional':
        rhd_files = sorted(glob.glob(os.path.join(dir_path, "*.rhd")))
        if len(rhd_files) < 1:
            raise Exception("Couldn't find *.rhd files in dir" + "\n" +
                            f"{dir_path}")
        # Load first file to get structure
        print(
            f"Loading first traditional file: {os.path.basename(rhd_files[0])}")
        result, data_present = load_file(rhd_files[0])
        if not data_present:
            raise Exception(
                "No data present in .rhd file. This may be a header-only file.")
        amp_data = result['amplifier_data']
        num_electrodes = amp_data.shape[0]
        amp_channel_names = [ch['native_channel_name']
                             for ch in result['amplifier_channels']]
        # Check if digital input data is present
        if 'board_dig_in_data' in result and result['board_dig_in_data'] is not None:
            dig_in_data = result['board_dig_in_data']
            num_dig_ins = dig_in_data.shape[0]
        else:
            dig_in_data = None
            num_dig_ins = 0

    if file_type != 'traditional' and len(amp_files) < 1:
        raise Exception("Couldn't find amp*.dat files in dir" + "\n" +
                        f"{dir_path}")

    # Plot files
    print("Now plotting amplifier signals")
    downsample = 100
    row_lim = 8
    if file_type == 'one file per channel':
        row_num = np.min((row_lim, len(amp_files)))
        col_num = int(np.ceil(len(amp_files)/row_num))
        # Create plot
        fig, ax = plt.subplots(row_num, col_num,
                               sharex=True, sharey=True, figsize=(15, 10))
        for this_file, this_ax in tqdm(zip(amp_files, ax.flatten())):
            data = np.fromfile(this_file, dtype=np.dtype('int16'))
            this_ax.plot(data[::downsample])
            this_ax.set_ylabel("_".join(os.path.basename(this_file)
                                        .split('.')[0].split('-')[1:]))
        plt.suptitle('Amplifier Data')
        fig.savefig(os.path.join(plot_dir, 'amplifier_data'))
        plt.close(fig)
    elif file_type == 'one file per signal type':
        amplifier_data = np.fromfile(amp_files[0], dtype=np.dtype('uint16'))
        num_electrodes = int(len(amplifier_data)/num_recorded_samples)
        amp_reshape = np.reshape(amplifier_data, (int(
            len(amplifier_data)/num_electrodes), num_electrodes)).T
        row_num = np.min((row_lim, num_electrodes))
        col_num = int(np.ceil(num_electrodes/row_num))
        # Create plot
        fig, ax = plt.subplots(row_num, col_num,
                               sharex=True, sharey=True, figsize=(15, 10))
        for e_i in tqdm(range(num_electrodes)):
            data = amp_reshape[e_i, :]
            ax_i = plt.subplot(row_num, col_num, e_i+1)
            ax_i.plot(data[::downsample])
            ax_i.set_ylabel('amp_' + str(e_i))
        plt.suptitle('Amplifier Data')
        fig.savefig(os.path.join(plot_dir, 'amplifier_data'))
        plt.close(fig)
    elif file_type == 'traditional':
        row_num = np.min((row_lim, num_electrodes))
        col_num = int(np.ceil(num_electrodes/row_num))
        # Create plot
        fig, ax = plt.subplots(row_num, col_num,
                               sharex=True, sharey=True, figsize=(15, 10))
        for e_i in tqdm(range(num_electrodes)):
            data = amp_data[e_i, :]
            ax_i = plt.subplot(row_num, col_num, e_i+1)
            ax_i.plot(data[::downsample])
            ax_i.set_ylabel(amp_channel_names[e_i])
        plt.suptitle('Amplifier Data')
        fig.savefig(os.path.join(plot_dir, 'amplifier_data'))
        plt.close(fig)

    print("Now plotting digital input signals")
    if file_type == 'one file per channel':
        fig, ax = plt.subplots(len(digin_files),
                               sharex=True, sharey=True, figsize=(8, 10))
        for this_file, this_ax in tqdm(zip(digin_files, ax.flatten())):
            data = np.fromfile(this_file, dtype=np.dtype('uint16'))
            this_ax.plot(data[::downsample])
            this_ax.set_ylabel("_".join(os.path.basename(this_file)
                                        .split('.')[0].split('-')[1:]))
        plt.suptitle('DIGIN Data')
        fig.savefig(os.path.join(plot_dir, 'digin_data'))
        plt.close(fig)
    elif file_type == 'traditional':
        if dig_in_data is not None and num_dig_ins > 0:
            fig, ax = plt.subplots(num_dig_ins, 1,
                                   sharex=True, sharey=True, figsize=(8, 10))
            # Handle case where there's only one digital input
            if num_dig_ins == 1:
                ax = [ax]
            for d_i in tqdm(range(num_dig_ins)):
                ax[d_i].plot(dig_in_data[d_i, ::downsample])
                ax[d_i].set_ylabel(f'Dig_in_{d_i}')
            plt.suptitle('DIGIN Data')
            fig.savefig(os.path.join(plot_dir, 'digin_data'))
            plt.close(fig)
        else:
            print("No digital input data found in traditional file")
    elif file_type == 'one file per signal type':
        d_inputs = np.fromfile(digin_files[0], dtype=np.dtype('uint16'))
        d_inputs_str = d_inputs.astype('str')
        d_in_str_int = d_inputs_str.astype('int64')
        d_diff = np.diff(d_in_str_int)
        dig_in = list(np.unique(np.abs(d_diff)) - 1)
        dig_in.remove(-1)
        num_dig_ins = len(dig_in)
        dig_inputs = np.zeros((num_dig_ins, len(d_inputs)))
        for n_i in range(num_dig_ins):
            start_ind = np.where(d_diff == n_i + 1)[0]
            end_ind = np.where(d_diff == -1*(n_i + 1))[0]
            for s_i in range(len(start_ind)):
                dig_inputs[n_i, start_ind[s_i]:end_ind[s_i]] = 1
        fig, ax = plt.subplots(num_dig_ins, 1,
                               sharex=True, sharey=True, figsize=(8, 10))
        for d_i in tqdm(range(num_dig_ins)):
            ax_i = plt.subplot(num_dig_ins, 1, d_i+1)
            ax_i.plot(dig_inputs[d_i, ::downsample])
            ax_i.set_ylabel('Dig_in_' + str(dig_in[d_i]))
        plt.suptitle('DIGIN Data')
        fig.savefig(os.path.join(plot_dir, 'digin_data'))
        plt.close(fig)


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
                                 'one file per signal type',
                                 'traditional'],
                        help='The type of file organization')
    args = parser.parse_args()

    dir_path = args.dir_path

    plot_channels(dir_path, args.file_type)
