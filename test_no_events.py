#!/usr/bin/env python3
"""
Test script to verify the interactive viewer works without keyboard and scroll events.
"""

from utils.interactive_viewer import InteractivePlotter
from utils.raw_data_loader import RawDataLoader
import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
import os

# Add the project root to the path
sys.path.insert(0, '/workspaces/blech_clust')


def create_test_data():
    """Create simple test data."""
    duration = 5.0
    sampling_rate = 30000
    n_samples = int(duration * sampling_rate)

    # Simple sine wave with noise and spikes
    time = np.linspace(0, duration, n_samples)
    data = 20 * np.sin(2 * np.pi * 10 * time) + \
        np.random.normal(0, 5, n_samples)

    # Add clear spikes
    spike_indices = [30000, 60000, 90000, 120000]  # At 1s, 2s, 3s, 4s
    for idx in spike_indices:
        if idx < len(data):
            data[idx:idx+30] += 50  # 1ms spike

    filename = '/tmp/test_no_events.h5'
    with h5py.File(filename, 'w') as f:
        raw_group = f.create_group('raw')
        raw_group.create_dataset('channel_0', data=data)
        f.attrs['sampling_rate'] = sampling_rate
        f.attrs['duration'] = duration
        f.attrs['n_channels'] = 1

    return filename


def test_viewer_without_events():
    """Test that viewer functions without keyboard/scroll events."""
    print("Testing viewer without keyboard and scroll events...")

    test_file = create_test_data()

    try:
        data_loader = RawDataLoader(test_file)
        viewer = InteractivePlotter(
            data_loader=data_loader,
            initial_channel='channel_0',
            initial_group='raw',
            window_duration=2.0
        )

        # Test that basic functionality works
        print("✅ Viewer created successfully")

        # Test navigation via methods (not events)
        initial_time = viewer.current_time
        viewer.jump_to_time(1.0)
        assert viewer.current_time == 1.0, f"Expected time 1.0, got {viewer.current_time}"
        print("✅ Time navigation via methods works")

        # Test channel switching via methods
        initial_channel = viewer.current_channel
        print(f"✅ Current channel: {initial_channel}")

        # Test threshold setting
        viewer.set_threshold(30.0)
        assert viewer.threshold_value == 30.0, f"Expected threshold 30.0, got {viewer.threshold_value}"
        print("✅ Threshold setting works")

        # Test snippet extraction
        viewer.show_snippets = True
        viewer._update_display()
        print(
            f"✅ Snippet extraction works (found {len(viewer.current_snippets)} snippets)")

        # Test that our custom event handlers are not present
        # Note: matplotlib widgets may still have their own event handlers

        # Check that our specific methods are not in the viewer
        assert not hasattr(viewer, '_on_scroll') or not callable(
            getattr(viewer, '_on_scroll', None)), "Custom scroll handler should be removed"
        assert not hasattr(viewer, '_on_key_press') or not callable(
            getattr(viewer, '_on_key_press', None)), "Custom key handler should be removed"
        print("✅ Custom event handlers properly removed")

        # Test that double-click still works (this should remain)
        assert hasattr(viewer, '_on_click') and callable(
            viewer._on_click), "Double-click handler should be preserved"
        print("✅ Double-click functionality preserved")

        # Test GUI controls work
        viewer._on_window_change('3.0')
        assert viewer.window_duration == 3.0, f"Expected duration 3.0, got {viewer.window_duration}"
        print("✅ GUI controls work")

        # Test filter controls
        viewer._on_max_freq_change('2000')
        assert viewer.max_freq == 2000.0, f"Expected max freq 2000, got {viewer.max_freq}"
        print("✅ Filter controls work")

        # Test snippet controls
        viewer._on_snippet_before_change('0.8')
        assert viewer.snippet_before_ms == 0.8, f"Expected before 0.8, got {viewer.snippet_before_ms}"
        print("✅ Snippet controls work")

        print(f"\n🎉 All tests passed!")
        print(f"The viewer works correctly without keyboard and scroll events.")
        print(f"Navigation is now exclusively through GUI controls:")
        print(f"- Time slider for navigation")
        print(f"- Prev/Next buttons")
        print(f"- Jump button")
        print(f"- Double-click to jump to time")
        print(f"- All text boxes and controls")

        # Don't show the plot in automated testing
        # viewer.show()

    finally:
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"\nCleaned up test file: {test_file}")


if __name__ == "__main__":
    test_viewer_without_events()
