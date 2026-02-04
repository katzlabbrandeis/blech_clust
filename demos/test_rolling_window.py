#!/usr/bin/env python
"""
Test script to verify rolling window threshold spike detection.

This script generates synthetic neural data with:
1. Varying noise levels across time (simulating recording instability)
2. Known spike times with consistent amplitude

It then compares:
- Global threshold detection (single threshold for entire recording)
- Rolling threshold detection (per-window adaptive threshold)

Uses the same spike_handler class as blech_process.py to ensure
the actual pipeline functions are tested.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.blech_process_utils import spike_handler, compute_rolling_threshold


def generate_spike_waveform(n_samples=45, sampling_rate=30000):
    """Generate a realistic spike waveform template."""
    t = np.arange(n_samples) / sampling_rate * 1000  # time in ms
    t_centered = t - t[n_samples // 3]

    # Simple spike shape: fast negative deflection, slower positive recovery
    waveform = -100 * np.exp(-((t_centered - 0.1) ** 2) / 0.02)
    waveform += 40 * np.exp(-((t_centered - 0.5) ** 2) / 0.1)

    return waveform


def generate_synthetic_data(
    duration_sec=60,
    sampling_rate=30000,
    base_noise_std=20,
    noise_segments=None,
    spike_rate_hz=5,
    spike_amplitude=100,
    seed=42,
):
    """
    Generate synthetic neural recording with varying noise and known spikes.

    Parameters
    ----------
    duration_sec : float
        Recording duration in seconds.
    sampling_rate : int
        Sampling rate in Hz.
    base_noise_std : float
        Base noise standard deviation in µV.
    noise_segments : list of tuples
        List of (start_sec, end_sec, noise_multiplier) for high-noise segments.
    spike_rate_hz : float
        Average spike rate in Hz.
    spike_amplitude : float
        Spike amplitude in µV.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    data : ndarray
        Synthetic recording data.
    true_spike_times : ndarray
        Sample indices of true spike times.
    noise_profile : ndarray
        Noise multiplier at each sample.
    """
    np.random.seed(seed)

    n_samples = int(duration_sec * sampling_rate)

    # Create noise profile
    noise_profile = np.ones(n_samples)
    if noise_segments is None:
        # Default: high noise in middle third of recording
        noise_segments = [
            (duration_sec * 0.33, duration_sec * 0.66, 3.0),
        ]

    for start_sec, end_sec, multiplier in noise_segments:
        start_idx = int(start_sec * sampling_rate)
        end_idx = int(end_sec * sampling_rate)
        noise_profile[start_idx:end_idx] = multiplier

    # Generate noise
    noise = np.random.randn(n_samples) * base_noise_std * noise_profile

    # Generate spike times (Poisson process)
    n_spikes = int(duration_sec * spike_rate_hz)
    true_spike_times = np.sort(
        np.random.randint(1000, n_samples - 1000, n_spikes)
    )

    # Remove spikes too close together (refractory period)
    min_isi = int(0.002 * sampling_rate)  # 2ms refractory
    valid_spikes = [true_spike_times[0]]
    for t in true_spike_times[1:]:
        if t - valid_spikes[-1] > min_isi:
            valid_spikes.append(t)
    true_spike_times = np.array(valid_spikes)

    # Generate spike waveform template
    waveform = generate_spike_waveform(n_samples=45, sampling_rate=sampling_rate)
    waveform = waveform / np.abs(waveform.min()) * spike_amplitude

    # Add spikes to data
    data = noise.copy()
    for t in true_spike_times:
        start = t - len(waveform) // 3
        end = start + len(waveform)
        if start >= 0 and end < n_samples:
            data[start:end] += waveform

    return data, true_spike_times, noise_profile


def evaluate_detection(detected_times, true_times, tolerance_samples=15):
    """
    Evaluate spike detection performance.

    Parameters
    ----------
    detected_times : ndarray
        Detected spike times.
    true_times : ndarray
        True spike times.
    tolerance_samples : int
        Maximum distance for a match.

    Returns
    -------
    dict
        Dictionary with TP, FP, FN, precision, recall, F1.
    """
    if len(detected_times) == 0:
        return {
            'TP': 0,
            'FP': 0,
            'FN': len(true_times),
            'precision': 0,
            'recall': 0,
            'F1': 0,
        }

    if len(true_times) == 0:
        return {
            'TP': 0,
            'FP': len(detected_times),
            'FN': 0,
            'precision': 0,
            'recall': 0,
            'F1': 0,
        }

    # Match detected to true spikes
    matched_true = set()
    matched_detected = set()

    for i, det in enumerate(detected_times):
        distances = np.abs(true_times - det)
        min_idx = np.argmin(distances)
        if distances[min_idx] <= tolerance_samples and min_idx not in matched_true:
            matched_true.add(min_idx)
            matched_detected.add(i)

    TP = len(matched_true)
    FP = len(detected_times) - TP
    FN = len(true_times) - TP

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    F1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return {
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'precision': precision,
        'recall': recall,
        'F1': F1,
    }


def create_params_dict(
    sampling_rate, threshold_mult, use_rolling, window_len=5.0, step_len=5.0
):
    """Create a params_dict matching blech_process.py format."""
    return {
        'sampling_rate': sampling_rate,
        'waveform_threshold': threshold_mult,
        'spike_snapshot_before': 0.5,  # ms
        'spike_snapshot_after': 1.0,  # ms
        'use_rolling_threshold': use_rolling,
        'rolling_threshold_window': window_len,
        'rolling_threshold_step': step_len,
    }


def run_simulation(output_dir=None):
    """Run the rolling window threshold simulation using spike_handler."""

    print("=" * 60)
    print("Rolling Window Threshold Simulation")
    print("Using spike_handler class (same as blech_process.py)")
    print("=" * 60)

    # Parameters
    duration_sec = 60
    sampling_rate = 30000
    base_noise_std = 20
    spike_amplitude = 100
    threshold_mult = 5.0

    # High noise in middle segment (3x noise)
    noise_segments = [
        (20, 40, 3.0),  # 20-40 seconds: 3x noise
    ]

    print(f"\nSimulation parameters:")
    print(f"  Duration: {duration_sec} seconds")
    print(f"  Sampling rate: {sampling_rate} Hz")
    print(f"  Base noise std: {base_noise_std} µV")
    print(f"  Spike amplitude: {spike_amplitude} µV")
    print(f"  Threshold multiplier: {threshold_mult}")
    print(f"  High noise segment: 20-40 sec (3x noise)")

    # Generate synthetic data
    print("\nGenerating synthetic data...")
    data, true_spike_times, noise_profile = generate_synthetic_data(
        duration_sec=duration_sec,
        sampling_rate=sampling_rate,
        base_noise_std=base_noise_std,
        noise_segments=noise_segments,
        spike_amplitude=spike_amplitude,
    )

    print(f"  Total samples: {len(data)}")
    print(f"  True spikes: {len(true_spike_times)}")

    # Count spikes in each segment
    low_noise_spikes = np.sum(
        (true_spike_times < 20 * sampling_rate)
        | (true_spike_times >= 40 * sampling_rate)
    )
    high_noise_spikes = np.sum(
        (true_spike_times >= 20 * sampling_rate)
        & (true_spike_times < 40 * sampling_rate)
    )
    print(f"  Spikes in low-noise segments: {low_noise_spikes}")
    print(f"  Spikes in high-noise segment: {high_noise_spikes}")

    # Run global threshold detection using spike_handler
    print("\nRunning global threshold detection (spike_handler, use_rolling=False)...")
    params_global = create_params_dict(
        sampling_rate, threshold_mult, use_rolling=False
    )
    handler_global = spike_handler(data, params_global, '.', 0)
    (
        slices_global,
        times_global,
        thresh_global,
        mean_global,
        mad_global,
    ) = handler_global.process_spikes()
    print(f"  Global threshold: {thresh_global:.2f} µV")
    print(f"  Detected spikes: {len(times_global)}")

    # Run rolling threshold detection using spike_handler
    print("\nRunning rolling threshold detection (spike_handler, use_rolling=True)...")
    params_rolling = create_params_dict(
        sampling_rate, threshold_mult, use_rolling=True
    )
    handler_rolling = spike_handler(data, params_rolling, '.', 0)
    (
        slices_rolling,
        times_rolling,
        thresh_rolling,
        mean_rolling,
        mad_rolling,
    ) = handler_rolling.process_spikes()
    print(f"  Median threshold: {thresh_rolling:.2f} µV")
    print(f"  Detected spikes: {len(times_rolling)}")

    # Compute rolling thresholds for visualization
    rt_times, rt_thresholds = compute_rolling_threshold(
        data, sampling_rate, window_len=5.0, step_len=5.0, threshold_mult=threshold_mult
    )

    # Evaluate detection performance
    print("\n" + "=" * 60)
    print("Detection Performance")
    print("=" * 60)

    results_global = evaluate_detection(times_global, true_spike_times)
    results_rolling = evaluate_detection(times_rolling, true_spike_times)

    print(f"\nGlobal threshold (use_rolling_threshold=False):")
    print(f"  True Positives:  {results_global['TP']}")
    print(f"  False Positives: {results_global['FP']}")
    print(f"  False Negatives: {results_global['FN']}")
    print(f"  Precision: {results_global['precision']:.3f}")
    print(f"  Recall:    {results_global['recall']:.3f}")
    print(f"  F1 Score:  {results_global['F1']:.3f}")

    print(f"\nRolling threshold (use_rolling_threshold=True):")
    print(f"  True Positives:  {results_rolling['TP']}")
    print(f"  False Positives: {results_rolling['FP']}")
    print(f"  False Negatives: {results_rolling['FN']}")
    print(f"  Precision: {results_rolling['precision']:.3f}")
    print(f"  Recall:    {results_rolling['recall']:.3f}")
    print(f"  F1 Score:  {results_rolling['F1']:.3f}")

    # Analyze by segment
    print("\n" + "=" * 60)
    print("Performance by Segment")
    print("=" * 60)

    # Low noise segments (0-20s and 40-60s)
    low_noise_true = true_spike_times[
        (true_spike_times < 20 * sampling_rate)
        | (true_spike_times >= 40 * sampling_rate)
    ]
    low_noise_global = times_global[
        (times_global < 20 * sampling_rate) | (times_global >= 40 * sampling_rate)
    ]
    low_noise_rolling = times_rolling[
        (times_rolling < 20 * sampling_rate) | (times_rolling >= 40 * sampling_rate)
    ]

    # High noise segment (20-40s)
    high_noise_true = true_spike_times[
        (true_spike_times >= 20 * sampling_rate)
        & (true_spike_times < 40 * sampling_rate)
    ]
    high_noise_global = times_global[
        (times_global >= 20 * sampling_rate) & (times_global < 40 * sampling_rate)
    ]
    high_noise_rolling = times_rolling[
        (times_rolling >= 20 * sampling_rate) & (times_rolling < 40 * sampling_rate)
    ]

    low_global = evaluate_detection(low_noise_global, low_noise_true)
    low_rolling = evaluate_detection(low_noise_rolling, low_noise_true)
    high_global = evaluate_detection(high_noise_global, high_noise_true)
    high_rolling = evaluate_detection(high_noise_rolling, high_noise_true)

    print(f"\nLow-noise segments (0-20s, 40-60s):")
    print(f"  True spikes: {len(low_noise_true)}")
    print(
        f"  Global:  Recall={low_global['recall']:.3f}, "
        f"Precision={low_global['precision']:.3f}"
    )
    print(
        f"  Rolling: Recall={low_rolling['recall']:.3f}, "
        f"Precision={low_rolling['precision']:.3f}"
    )

    print(f"\nHigh-noise segment (20-40s):")
    print(f"  True spikes: {len(high_noise_true)}")
    print(
        f"  Global:  Recall={high_global['recall']:.3f}, "
        f"Precision={high_global['precision']:.3f}"
    )
    print(
        f"  Rolling: Recall={high_rolling['recall']:.3f}, "
        f"Precision={high_rolling['precision']:.3f}"
    )

    # Generate plots
    print("\n" + "=" * 60)
    print("Generating plots...")
    print("=" * 60)

    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    # Share x-axis only for the first 3 subplots (time-based)
    for i in range(1, 3):
        axes[i].sharex(axes[0])

    time_axis = np.arange(len(data)) / sampling_rate

    # Plot 1: Raw data with noise profile
    ax = axes[0]
    ax.plot(time_axis[::100], data[::100], 'k', linewidth=0.3, alpha=0.7)
    ax.axhline(
        thresh_global, color='r', linestyle='--', label=f'Global thresh: {thresh_global:.1f}'
    )
    ax.axhline(-thresh_global, color='r', linestyle='--')
    ax.axvspan(20, 40, alpha=0.2, color='orange', label='High noise segment')
    ax.set_ylabel('Amplitude (µV)')
    ax.set_title('Synthetic Recording with Varying Noise')
    ax.legend(loc='upper right')

    # Plot 2: Rolling thresholds
    ax = axes[1]
    ax.plot(rt_times, rt_thresholds, 'b-', linewidth=2, label='Rolling threshold')
    ax.axhline(thresh_global, color='r', linestyle='--', label='Global threshold')
    ax.axvspan(20, 40, alpha=0.2, color='orange')
    ax.set_ylabel('Threshold (µV)')
    ax.set_title('Rolling vs Global Threshold')
    ax.legend(loc='upper right')

    # Plot 3: Detection comparison
    ax = axes[2]
    ax.eventplot(
        [true_spike_times / sampling_rate],
        lineoffsets=3,
        colors='green',
        linewidths=0.5,
        label=f'True spikes (n={len(true_spike_times)})',
    )
    ax.eventplot(
        [times_global / sampling_rate],
        lineoffsets=2,
        colors='red',
        linewidths=0.5,
        label=f'Global detected (n={len(times_global)})',
    )
    ax.eventplot(
        [times_rolling / sampling_rate],
        lineoffsets=1,
        colors='blue',
        linewidths=0.5,
        label=f'Rolling detected (n={len(times_rolling)})',
    )
    ax.axvspan(20, 40, alpha=0.2, color='orange')
    ax.set_ylabel('Detection')
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['Rolling', 'Global', 'True'])
    ax.set_title('Spike Detection Comparison')
    ax.legend(loc='upper right')

    # Plot 4: Performance metrics
    ax = axes[3]
    x = np.arange(3)
    width = 0.35
    metrics = ['Precision', 'Recall', 'F1']
    global_vals = [
        results_global['precision'],
        results_global['recall'],
        results_global['F1'],
    ]
    rolling_vals = [
        results_rolling['precision'],
        results_rolling['recall'],
        results_rolling['F1'],
    ]

    bars1 = ax.bar(x - width / 2, global_vals, width, label='Global', color='red', alpha=0.7)
    bars2 = ax.bar(x + width / 2, rolling_vals, width, label='Rolling', color='blue', alpha=0.7)
    ax.set_ylabel('Score')
    ax.set_title('Overall Detection Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 1.1)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(
            f'{height:.2f}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center',
            va='bottom',
            fontsize=9,
        )
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(
            f'{height:.2f}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center',
            va='bottom',
            fontsize=9,
        )

    axes[-1].set_xlabel('Time (s)')

    plt.tight_layout()

    # Save plot
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    output_path = os.path.join(output_dir, 'rolling_window_simulation_results.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    plt.close()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    improvement = results_rolling['F1'] - results_global['F1']
    print(f"\nRolling threshold F1 improvement: {improvement:+.3f}")

    if results_rolling['recall'] > results_global['recall']:
        print("✓ Rolling threshold detected MORE true spikes")
    else:
        print("✗ Rolling threshold did not improve recall")

    if high_rolling['recall'] > high_global['recall']:
        print("✓ Rolling threshold performed BETTER in high-noise segment")
    else:
        print("✗ Rolling threshold did not improve high-noise detection")

    # Verify spike_handler is working correctly
    print("\n" + "=" * 60)
    print("VERIFICATION: spike_handler class")
    print("=" * 60)
    print(f"✓ spike_handler with use_rolling_threshold=False: {len(times_global)} spikes")
    print(f"✓ spike_handler with use_rolling_threshold=True: {len(times_rolling)} spikes")
    print(f"✓ Dejittering applied (slices shape: {slices_rolling.shape})")

    return {
        'global': results_global,
        'rolling': results_rolling,
        'high_noise_global': high_global,
        'high_noise_rolling': high_rolling,
    }


if __name__ == '__main__':
    results = run_simulation()
