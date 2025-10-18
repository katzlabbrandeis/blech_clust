"""
This module provides utilities for quality assurance of electrode data by plotting bulk power spectrum for each electrode.

- `get_electrode_data(hf5_path, electrode_num, max_samples=None)`: Extracts data from a specific electrode
- `calculate_power_spectrum(data, sampling_rate, nperseg=None)`: Calculates power spectral density using Welch's method
- `plot_electrode_spectra(hf5_path, output_dir, max_samples=100000, nperseg=None)`: Main function to generate power spectrum plots for all electrodes
- `downsample_data(data, target_samples)`: Downsamples data to target number of samples for efficiency
"""

import numpy as np
import matplotlib.pyplot as plt
import tables
import os
from scipy import signal
from tqdm import tqdm
import warnings


def get_electrode_data(hf5_path, electrode_num, max_samples=None):
    """
    Extract data from a specific electrode.

    Parameters:
    -----------
    hf5_path : str
        Path to HDF5 file
    electrode_num : int
        Electrode number to extract
    max_samples : int, optional
        Maximum number of samples to extract for efficiency

    Returns:
    --------
    data : np.array
        Electrode data
    sampling_rate : float
        Sampling rate in Hz
    """
    with tables.open_file(hf5_path, 'r') as hf5:
        # Try to find the electrode in raw data
        electrode_name = f'electrode{electrode_num:02d}'

        try:
            electrode_node = getattr(hf5.root.raw, electrode_name)
            data = electrode_node[:]

            # Get sampling rate from digital_in if available
            try:
                sampling_rate = hf5.root.digital_in.dig_in_sampling_rate[0]
            except:
                # Default sampling rate if not found
                sampling_rate = 30000.0
                warnings.warn(
                    f"Could not find sampling rate, using default {sampling_rate} Hz")

        except AttributeError:
            raise ValueError(
                f"Electrode {electrode_num} not found in {hf5_path}")

    # Downsample if data is too large
    if max_samples is not None and len(data) > max_samples:
        data = downsample_data(data, max_samples)

    return data, sampling_rate


def downsample_data(data, target_samples):
    """
    Downsample data to target number of samples using random sampling.

    Parameters:
    -----------
    data : np.array
        Input data
    target_samples : int
        Target number of samples

    Returns:
    --------
    downsampled_data : np.array
        Downsampled data
    """
    if len(data) <= target_samples:
        return data

    # Use random sampling to avoid bias
    indices = np.random.choice(len(data), target_samples, replace=False)
    indices = np.sort(indices)  # Sort to maintain temporal order somewhat
    return data[indices]


def calculate_power_spectrum(data, sampling_rate, nperseg=None):
    """
    Calculate power spectral density using Welch's method.

    Parameters:
    -----------
    data : np.array
        Input signal data
    sampling_rate : float
        Sampling rate in Hz
    nperseg : int, optional
        Length of each segment for Welch's method

    Returns:
    --------
    frequencies : np.array
        Frequency bins
    psd : np.array
        Power spectral density
    """
    if nperseg is None:
        # Use a reasonable default based on data length
        nperseg = min(len(data) // 8, 2048)

    frequencies, psd = signal.welch(
        data,
        fs=sampling_rate,
        nperseg=nperseg,
        noverlap=nperseg//2,
        detrend='linear'
    )

    return frequencies, psd


def plot_electrode_spectra(hf5_path, output_dir=None, max_samples=100000, nperseg=None):
    """
    Generate power spectrum plots for all electrodes in the dataset.

    Parameters:
    -----------
    hf5_path : str
        Path to HDF5 file
    output_dir : str, optional
        Output directory for plots. If None, uses same directory as hf5_path
    max_samples : int
        Maximum number of samples to use per electrode for efficiency
    nperseg : int, optional
        Length of each segment for Welch's method
    """
    if output_dir is None:
        output_dir = os.path.dirname(hf5_path)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get list of available electrodes
    with tables.open_file(hf5_path, 'r') as hf5:
        raw_nodes = hf5.list_nodes('/raw')
        electrode_nums = []
        for node in raw_nodes:
            if 'electrode' in node._v_name:
                electrode_nums.append(
                    int(node._v_name.replace('electrode', '')))
        electrode_nums = sorted(electrode_nums)

    if not electrode_nums:
        raise ValueError("No electrodes found in the dataset")

    print(f"Found {len(electrode_nums)} electrodes: {electrode_nums}")
    print(f"Processing with max {max_samples} samples per electrode...")

    # Calculate spectra for all electrodes
    all_frequencies = []
    all_psds = []
    valid_electrodes = []

    for electrode_num in tqdm(electrode_nums, desc="Processing electrodes"):
        try:
            data, sampling_rate = get_electrode_data(
                hf5_path, electrode_num, max_samples)
            frequencies, psd = calculate_power_spectrum(
                data, sampling_rate, nperseg)

            all_frequencies.append(frequencies)
            all_psds.append(psd)
            valid_electrodes.append(electrode_num)

        except Exception as e:
            print(f"Warning: Could not process electrode {electrode_num}: {e}")
            continue

    if not valid_electrodes:
        raise ValueError("No valid electrodes could be processed")

    # Create comprehensive plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        f'Electrode Power Spectrum Analysis\nFile: {os.path.basename(hf5_path)}', fontsize=14)

    # Plot 1: Individual spectra (log scale)
    ax1 = axes[0, 0]
    for i, (freq, psd, elec_num) in enumerate(zip(all_frequencies, all_psds, valid_electrodes)):
        ax1.loglog(freq[1:], psd[1:], alpha=0.7, linewidth=0.8,
                   label=f'Elec {elec_num}' if i < 10 else "")
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Power Spectral Density')
    ax1.set_title('Individual Electrode Spectra (Log-Log)')
    ax1.grid(True, alpha=0.3)
    if len(valid_electrodes) <= 10:
        ax1.legend(fontsize=8)

    # Plot 2: Average spectrum with std
    ax2 = axes[0, 1]
    # Interpolate all PSDs to common frequency grid
    common_freq = all_frequencies[0]
    interpolated_psds = []
    for freq, psd in zip(all_frequencies, all_psds):
        interp_psd = np.interp(common_freq, freq, psd)
        interpolated_psds.append(interp_psd)

    interpolated_psds = np.array(interpolated_psds)
    mean_psd = np.mean(interpolated_psds, axis=0)
    std_psd = np.std(interpolated_psds, axis=0)

    ax2.loglog(common_freq[1:], mean_psd[1:], 'b-', linewidth=2, label='Mean')
    ax2.fill_between(common_freq[1:],
                     (mean_psd - std_psd)[1:],
                     (mean_psd + std_psd)[1:],
                     alpha=0.3, color='blue', label='±1 STD')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power Spectral Density')
    ax2.set_title('Average Spectrum Across Electrodes')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Heatmap of power in frequency bands
    ax3 = axes[1, 0]
    # Define frequency bands
    bands = {
        'Delta (1-4 Hz)': (1, 4),
        'Theta (4-8 Hz)': (4, 8),
        'Alpha (8-13 Hz)': (8, 13),
        'Beta (13-30 Hz)': (13, 30),
        'Gamma (30-100 Hz)': (30, 100),
        'High Gamma (100-300 Hz)': (100, 300)
    }

    band_powers = np.zeros((len(valid_electrodes), len(bands)))
    band_names = list(bands.keys())

    for i, (freq, psd) in enumerate(zip(all_frequencies, all_psds)):
        for j, (band_name, (low, high)) in enumerate(bands.items()):
            band_mask = (freq >= low) & (freq <= high)
            if np.any(band_mask):
                band_powers[i, j] = np.mean(psd[band_mask])

    im = ax3.imshow(band_powers.T, aspect='auto',
                    cmap='viridis', interpolation='nearest')
    ax3.set_xlabel('Electrode Number')
    ax3.set_ylabel('Frequency Band')
    ax3.set_title('Power by Frequency Band')
    ax3.set_xticks(range(len(valid_electrodes)))
    ax3.set_xticklabels([f'{e}' for e in valid_electrodes], rotation=45)
    ax3.set_yticks(range(len(band_names)))
    ax3.set_yticklabels(band_names)
    plt.colorbar(im, ax=ax3, label='Power')

    # Plot 4: Quality metrics
    ax4 = axes[1, 1]
    # Calculate some quality metrics
    total_powers = [np.sum(psd) for psd in all_psds]
    peak_frequencies = [freq[np.argmax(psd[1:])]
                        for freq, psd in zip(all_frequencies, all_psds)]

    ax4_twin = ax4.twinx()

    line1 = ax4.plot(valid_electrodes, total_powers, 'bo-',
                     label='Total Power', markersize=4)
    ax4.set_xlabel('Electrode Number')
    ax4.set_ylabel('Total Power', color='b')
    ax4.tick_params(axis='y', labelcolor='b')

    line2 = ax4_twin.plot(valid_electrodes, peak_frequencies,
                          'ro-', label='Peak Frequency', markersize=4)
    ax4_twin.set_ylabel('Peak Frequency (Hz)', color='r')
    ax4_twin.tick_params(axis='y', labelcolor='r')

    ax4.set_title('Quality Metrics by Electrode')
    ax4.grid(True, alpha=0.3)

    # Combine legends
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()

    # Save the plot
    output_path = os.path.join(output_dir, 'electrode_power_spectra.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Save summary statistics
    summary_path = os.path.join(output_dir, 'power_spectrum_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Power Spectrum Analysis Summary\n")
        f.write(f"File: {hf5_path}\n")
        f.write(f"Number of electrodes processed: {len(valid_electrodes)}\n")
        f.write(f"Electrodes: {valid_electrodes}\n")
        f.write(f"Sampling rate: {sampling_rate} Hz\n")
        f.write(f"Samples per electrode: {max_samples}\n")
        f.write(
            f"Frequency range: {common_freq[1]:.2f} - {common_freq[-1]:.2f} Hz\n\n")

        f.write("Frequency Band Power Summary:\n")
        for i, band_name in enumerate(band_names):
            mean_power = np.mean(band_powers[:, i])
            std_power = np.std(band_powers[:, i])
            f.write(f"{band_name}: {mean_power:.2e} ± {std_power:.2e}\n")

        f.write(f"\nTotal Power Statistics:\n")
        f.write(f"Mean: {np.mean(total_powers):.2e}\n")
        f.write(f"Std: {np.std(total_powers):.2e}\n")
        f.write(
            f"Min: {np.min(total_powers):.2e} (Electrode {valid_electrodes[np.argmin(total_powers)]})\n")
        f.write(
            f"Max: {np.max(total_powers):.2e} (Electrode {valid_electrodes[np.argmax(total_powers)]})\n")

    print(f"Power spectrum analysis complete!")
    print(f"Plot saved to: {output_path}")
    print(f"Summary saved to: {summary_path}")

    return output_path, summary_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python electrode_power_spectrum.py <data_directory>")
        print("This will look for .h5 files in the specified directory")
        sys.exit(1)

    data_dir = sys.argv[1]

    # Find HDF5 files in the directory
    h5_files = []
    for file in os.listdir(data_dir):
        if file.endswith('.h5'):
            h5_files.append(os.path.join(data_dir, file))

    if not h5_files:
        print(f"No .h5 files found in {data_dir}")
        sys.exit(1)

    print(f"Found {len(h5_files)} HDF5 files")

    for h5_file in h5_files:
        print(f"\nProcessing: {h5_file}")
        try:
            plot_electrode_spectra(h5_file, output_dir=data_dir)
        except Exception as e:
            print(f"Error processing {h5_file}: {e}")
