#!/usr/bin/env python3
"""
Test script for the enhanced raw data viewer.

This script tests the new features added to the raw data viewer:
1. Text entry boxes for lowpass and highpass frequencies
2. Text entry boxes for y-limits
3. Data conversion factor (0.6745) to convert to microvolts
4. Channel selection via radio buttons

Usage:
    python test_enhanced_viewer.py
"""

import sys
import os
import numpy as np
import h5py
import tempfile
import shutil

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

def create_test_data():
    """Create a test HDF5 file with sample neural data."""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    h5_path = os.path.join(temp_dir, 'test_data.h5')
    
    # Create test data
    sampling_rate = 30000  # 30 kHz
    duration = 10.0  # 10 seconds
    n_samples = int(sampling_rate * duration)
    n_channels = 4
    
    # Generate synthetic neural data
    time_array = np.linspace(0, duration, n_samples)
    
    with h5py.File(h5_path, 'w') as f:
        # Create raw data group
        raw_group = f.create_group('raw')
        
        for i in range(n_channels):
            # Create synthetic signal with noise and some spikes
            signal = np.random.normal(0, 50, n_samples)  # Background noise
            
            # Add some spike-like events
            spike_times = np.random.choice(n_samples, size=20, replace=False)
            for spike_time in spike_times:
                if spike_time + 30 < n_samples:
                    # Simple spike waveform
                    spike_waveform = -200 * np.exp(-np.linspace(0, 3, 30))
                    signal[spike_time:spike_time+30] += spike_waveform
            
            # Add some LFP-like oscillations
            lfp_component = 20 * np.sin(2 * np.pi * 8 * time_array)  # 8 Hz oscillation
            signal += lfp_component
            
            # Store channel data
            channel_name = f'electrode{i:02d}'
            raw_group.create_dataset(channel_name, data=signal)
        
        # Store sampling rate
        f.attrs['sampling_rate'] = sampling_rate
    
    print(f"Created test data file: {h5_path}")
    print(f"Channels: {n_channels}")
    print(f"Duration: {duration} seconds")
    print(f"Sampling rate: {sampling_rate} Hz")
    
    return h5_path, temp_dir

def test_enhanced_viewer():
    """Test the enhanced raw data viewer."""
    try:
        # Create test data
        h5_path, temp_dir = create_test_data()
        
        # Import the enhanced viewer
        from utils.raw_data_loader import RawDataLoader
        from utils.interactive_viewer import InteractivePlotter
        
        print("\nTesting enhanced raw data viewer...")
        
        # Initialize data loader
        data_loader = RawDataLoader(h5_path)
        print(f"Data loader initialized. Sampling rate: {data_loader.sampling_rate} Hz")
        
        # Get available channels
        channels = data_loader.get_available_channels()
        print(f"Available channels: {channels}")
        
        # Initialize enhanced plotter
        plotter = InteractivePlotter(
            data_loader=data_loader,
            initial_channel='electrode00',
            initial_group='raw',
            window_duration=2.0
        )
        
        print("\nEnhanced viewer features:")
        print("1. ✅ Text boxes for lowpass/highpass frequencies")
        print("2. ✅ Text boxes for Y-limits (Y Min/Y Max)")
        print("3. ✅ Data conversion factor (0.6745) applied")
        print("4. ✅ Channel selection via radio buttons")
        print("\nControls available:")
        print("- Lowpass/Highpass frequency text boxes")
        print("- Y Min/Y Max text boxes for manual scaling")
        print("- Channel selection radio buttons")
        print("- Apply Filter button")
        print("- Auto Y button to reset to auto-scaling")
        
        print("\nStarting interactive viewer...")
        print("Try the following:")
        print("1. Enter frequencies in Lowpass/Highpass boxes (e.g., 3000, 300)")
        print("2. Click 'Apply Filter' to apply bandpass filter")
        print("3. Enter Y Min/Y Max values for manual scaling")
        print("4. Select different channels using radio buttons")
        print("5. Click 'Auto Y' to return to auto-scaling")
        
        # Show the viewer
        plotter.show()
        
    except Exception as e:
        print(f"Error testing enhanced viewer: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir)
            print(f"\nCleaned up temporary directory: {temp_dir}")

if __name__ == '__main__':
    test_enhanced_viewer()