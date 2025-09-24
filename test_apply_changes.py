#!/usr/bin/env python3
"""
Test script for apply changes functionality.
Tests that parameter changes can be batched and applied together.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
import os

# Add the project root to the path
sys.path.insert(0, '/workspaces/blech_clust')

from utils.raw_data_loader import RawDataLoader
from utils.interactive_viewer import InteractivePlotter

def create_test_data():
    """Create test data for apply changes testing."""
    duration = 3.0
    sampling_rate = 30000
    n_samples = int(duration * sampling_rate)
    
    # Create data with noise and spikes
    np.random.seed(42)
    data = np.random.normal(0, 10, n_samples)  # 10 µV noise
    
    # Add some spikes
    spike_times = [0.5, 1.0, 1.5, 2.0, 2.5]
    for spike_time in spike_times:
        spike_idx = int(spike_time * sampling_rate)
        if spike_idx < len(data) - 60:
            # Add spike
            spike_samples = 60
            spike_waveform = 50 * np.exp(-np.linspace(0, 3, spike_samples))
            data[spike_idx:spike_idx+spike_samples] += spike_waveform
    
    filename = '/tmp/test_apply_changes.h5'
    with h5py.File(filename, 'w') as f:
        raw_group = f.create_group('raw')
        raw_group.create_dataset('channel_0', data=data)
        f.attrs['sampling_rate'] = sampling_rate
        f.attrs['duration'] = duration
        f.attrs['n_channels'] = 1
    
    return filename

def test_apply_changes():
    """Test the apply changes functionality."""
    print("Testing apply changes functionality...")
    
    test_file = create_test_data()
    
    try:
        data_loader = RawDataLoader(test_file)
        viewer = InteractivePlotter(
            data_loader=data_loader,
            initial_channel='channel_0',
            initial_group='raw',
            window_duration=3.0
        )
        
        # Test initial state
        print("\n=== Testing Initial State ===")
        assert viewer.auto_update == True, f"Auto update should be True initially, got {viewer.auto_update}"
        assert viewer.pending_changes == False, f"Should have no pending changes initially, got {viewer.pending_changes}"
        assert hasattr(viewer, 'btn_apply'), "Apply button should exist"
        assert hasattr(viewer, 'auto_update_check'), "Auto update checkbox should exist"
        print("✅ Initial state correct")
        
        # Test auto-update mode (default behavior)
        print("\n=== Testing Auto-Update Mode ===")
        initial_threshold = viewer.threshold_value
        viewer._on_threshold_change('25')
        
        # In auto-update mode, changes should be applied immediately
        assert viewer.threshold_value == 25.0, f"Threshold should be updated immediately, got {viewer.threshold_value}"
        assert viewer.pending_changes == False, f"Should have no pending changes in auto mode, got {viewer.pending_changes}"
        print("✅ Auto-update mode works correctly")
        
        # Test switching to manual mode
        print("\n=== Testing Manual Mode ===")
        viewer._on_auto_update_change('Auto Update')  # Toggle off
        assert viewer.auto_update == False, f"Auto update should be False after toggle, got {viewer.auto_update}"
        print("✅ Switched to manual mode")
        
        # Test parameter changes in manual mode
        print("\n=== Testing Parameter Changes in Manual Mode ===")
        
        # Change threshold - should not update display immediately
        viewer._on_threshold_change('40')
        assert viewer.threshold_value == 40.0, f"Threshold value should be stored, got {viewer.threshold_value}"
        assert viewer.pending_changes == True, f"Should have pending changes, got {viewer.pending_changes}"
        print("✅ Threshold change deferred")
        
        # Change filter frequencies - should not update display immediately
        viewer._on_max_freq_change('2000')
        assert viewer.max_freq == 2000.0, f"Max freq should be stored, got {viewer.max_freq}"
        assert viewer.pending_changes == True, f"Should still have pending changes, got {viewer.pending_changes}"
        print("✅ Max freq change deferred")
        
        viewer._on_min_freq_change('300')
        assert viewer.min_freq == 300.0, f"Min freq should be stored, got {viewer.min_freq}"
        assert viewer.pending_changes == True, f"Should still have pending changes, got {viewer.pending_changes}"
        print("✅ Min freq change deferred")
        
        # Change Y limits - should not update display immediately
        viewer._on_ylim_change('-100')  # This will set ymin
        viewer.ymin_box.set_val('-100')
        viewer.ymax_box.set_val('100')
        viewer._on_ylim_change('100')  # This will set both
        assert viewer.manual_ylims == (-100.0, 100.0), f"Y limits should be stored, got {viewer.manual_ylims}"
        assert viewer.pending_changes == True, f"Should still have pending changes, got {viewer.pending_changes}"
        print("✅ Y-limit changes deferred")
        
        # Change snippet parameters - should not update display immediately
        viewer.show_snippets = True  # Enable snippets first
        viewer._on_snippet_before_change('0.8')
        assert viewer.snippet_before_ms == 0.8, f"Snippet before should be stored, got {viewer.snippet_before_ms}"
        assert viewer.pending_changes == True, f"Should still have pending changes, got {viewer.pending_changes}"
        print("✅ Snippet parameter changes deferred")
        
        # Test apply changes button
        print("\n=== Testing Apply Changes Button ===")
        viewer._on_apply_changes(None)
        assert viewer.pending_changes == False, f"Should have no pending changes after apply, got {viewer.pending_changes}"
        print("✅ Apply changes button works")
        
        # Test switching back to auto-update with pending changes
        print("\n=== Testing Auto-Update with Pending Changes ===")
        viewer._on_threshold_change('60')  # Make a change in manual mode
        assert viewer.pending_changes == True, f"Should have pending changes, got {viewer.pending_changes}"
        
        viewer._on_auto_update_change('Auto Update')  # Toggle back to auto
        assert viewer.auto_update == True, f"Auto update should be True after toggle, got {viewer.auto_update}"
        assert viewer.pending_changes == False, f"Pending changes should be applied when switching to auto, got {viewer.pending_changes}"
        assert viewer.threshold_value == 60.0, f"Threshold should be applied, got {viewer.threshold_value}"
        print("✅ Switching to auto-update applies pending changes")
        
        # Test that time-related changes still work immediately in manual mode
        print("\n=== Testing Time Changes in Manual Mode ===")
        viewer._on_auto_update_change('Auto Update')  # Switch back to manual
        initial_time = viewer.current_time
        viewer._on_time_change(1.0)
        assert viewer.current_time == 1.0, f"Time should change immediately, got {viewer.current_time}"
        print("✅ Time changes work immediately in manual mode")
        
        # Test window duration changes (should be immediate)
        initial_duration = viewer.window_duration
        viewer._on_window_change('2.0')
        assert viewer.window_duration == 2.0, f"Window duration should change immediately, got {viewer.window_duration}"
        print("✅ Window duration changes work immediately in manual mode")
        
        print(f"\n🎉 All apply changes tests passed!")
        print(f"Features working correctly:")
        print(f"- Auto-update mode: immediate parameter application")
        print(f"- Manual mode: deferred parameter application")
        print(f"- Apply changes button: applies all pending changes")
        print(f"- Auto-update toggle: applies pending changes when switching to auto")
        print(f"- Time-related changes: always immediate")
        print(f"- Pending changes tracking: visual feedback system")
        
        # Don't show plot in automated testing
        # viewer.show()
        
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"\nCleaned up test file: {test_file}")

if __name__ == "__main__":
    test_apply_changes()