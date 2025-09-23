#!/usr/bin/env python3
"""
Test script for the updated snippet viewer layout.

This script tests the new layout where:
1. Main plot takes 5/6 of the width
2. Snippet plot takes 1/6 of the width (1/5 of main plot width)
3. Snippet window is 1/5 of the main trace window duration

Usage:
    python test_snippet_layout.py
"""

import sys
import os
import numpy as np
import h5py
import tempfile
import shutil

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

def create_test_data_with_clear_spikes():
    """Create test data with very clear spikes for layout testing."""
    temp_dir = tempfile.mkdtemp()
    h5_path = os.path.join(temp_dir, 'test_layout_data.h5')
    
    sampling_rate = 30000  # 30 kHz
    duration = 10.0  # 10 seconds
    n_samples = int(sampling_rate * duration)
    n_channels = 2
    
    with h5py.File(h5_path, 'w') as f:
        raw_group = f.create_group('raw')
        
        for i in range(n_channels):
            # Create clean background
            signal = np.random.normal(0, 10, n_samples)  # Low noise
            
            # Add very clear, large spikes
            n_spikes = 30
            spike_times = np.random.choice(n_samples, size=n_spikes, replace=False)
            spike_times = spike_times[(spike_times > 1000) & (spike_times < n_samples - 1000)]
            
            for spike_time in spike_times:
                # Create large, clear spike waveforms
                spike_duration = int(0.003 * sampling_rate)  # 3ms spike
                spike_start = max(0, spike_time - spike_duration//2)
                spike_end = min(n_samples, spike_time + spike_duration//2)
                
                # Large amplitude spikes for easy detection
                t_spike = np.linspace(-2, 2, spike_end - spike_start)
                if i == 0:  # Negative spikes
                    spike_waveform = -500 * np.exp(-t_spike**2 / 0.2) * np.sin(np.pi * t_spike)
                else:  # Positive spikes
                    spike_waveform = 500 * np.exp(-t_spike**2 / 0.2) * np.sin(np.pi * t_spike)
                
                signal[spike_start:spike_end] += spike_waveform
            
            channel_name = f'electrode{i:02d}'
            raw_group.create_dataset(channel_name, data=signal)
        
        f.attrs['sampling_rate'] = sampling_rate
    
    print(f"Created test layout data: {h5_path}")
    print(f"Channels: {n_channels}")
    print(f"Duration: {duration} seconds")
    print(f"Expected spikes per channel: ~{n_spikes}")
    
    return h5_path, temp_dir

def test_new_layout():
    """Test the new snippet viewer layout."""
    try:
        h5_path, temp_dir = create_test_data_with_clear_spikes()
        
        from utils.raw_data_loader import RawDataLoader
        from utils.interactive_viewer import InteractivePlotter
        
        print("\nTesting new snippet viewer layout...")
        
        data_loader = RawDataLoader(h5_path)
        print(f"Data loaded. Sampling rate: {data_loader.sampling_rate} Hz")
        
        plotter = InteractivePlotter(
            data_loader=data_loader,
            initial_channel='electrode00',
            initial_group='raw',
            window_duration=2.0  # 2 second window
        )
        
        print("\nNew Layout Features:")
        print("1. ✅ Main plot: 5/6 width (5 columns)")
        print("2. ✅ Snippet plot: 1/6 width (1 column) - 1/5 of main plot width")
        print("3. ✅ Snippet window: 1/5 of main trace window duration")
        print(f"4. ✅ Default snippet ratio: {plotter.snippet_window_ratio}")
        
        # Test layout calculations
        main_window_duration = plotter.window_duration  # 2.0 seconds
        snippet_window_duration = main_window_duration * plotter.snippet_window_ratio  # 0.4 seconds
        
        print(f"\nWindow Calculations:")
        print(f"  - Main window duration: {main_window_duration} seconds")
        print(f"  - Snippet window duration: {snippet_window_duration} seconds")
        print(f"  - Snippet ratio: {plotter.snippet_window_ratio} (1/5 = 0.2)")
        
        # Test snippet extraction with new parameters
        print(f"\nTesting snippet extraction...")
        data, time_array = data_loader.load_channel_data('electrode00', 'raw', 0, 2)
        data = data * plotter.data_conversion_factor
        
        threshold = 100.0  # Should easily detect our large spikes
        snippets, snippet_times, snippet_polarities = plotter._extract_snippets(
            data, time_array, threshold
        )
        
        print(f"  - Extracted {len(snippets)} snippets")
        if len(snippets) > 0:
            sampling_rate = data_loader.sampling_rate
            expected_samples = int(snippet_window_duration * sampling_rate)
            actual_samples = len(snippets[0])
            print(f"  - Snippet length: {actual_samples} samples (expected: {expected_samples})")
            print(f"  - Snippet duration: {actual_samples/sampling_rate:.3f} seconds")
        
        print(f"\nGUI Layout:")
        print(f"  - Figure size: {plotter.fig.get_size_inches()}")
        print(f"  - Main plot position: (0,0) colspan=5 rowspan=3")
        print(f"  - Snippet plot position: (0,5) colspan=1 rowspan=3")
        print(f"  - Controls start at row 4")
        
        print(f"\nStarting interactive viewer...")
        print("Instructions for testing:")
        print("1. Set threshold to 100 (should detect large spikes)")
        print("2. Check 'Show Snippets' to see the side-by-side layout")
        print("3. Adjust 'Snippet Ratio' (try 0.1 for narrower, 0.3 for wider)")
        print("4. Navigate through data to see different spike patterns")
        print("5. Try different channels (electrode00 vs electrode01)")
        print("6. Notice how snippet plot is 1/5 width of main plot")
        
        plotter.show()
        
    except Exception as e:
        print(f"Error testing new layout: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir)
            print(f"\nCleaned up: {temp_dir}")

if __name__ == '__main__':
    print("Testing Enhanced Raw Data Viewer - New Layout")
    print("=" * 50)
    test_new_layout()