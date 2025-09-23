#!/usr/bin/env python3
"""
Raw Data Viewer - Interactive viewer for neural recording data

This application provides an interactive interface for viewing raw neural data
from HDF5 files with features including:
- Single channel visualization with time navigation
- Real-time filtering (bandpass, highpass, lowpass)
- Threshold line display
- Configurable viewing parameters
- Memory-efficient data loading

Usage:
    python raw_data_viewer.py <data_directory> [options]
    python raw_data_viewer.py <hdf5_file> [options]

Examples:
    python raw_data_viewer.py /path/to/data/
    python raw_data_viewer.py /path/to/data.h5 --channel electrode00
    python raw_data_viewer.py /path/to/data/ --window 5.0 --filter spike
"""

import argparse
import os
import sys
import glob
import json
import warnings
from typing import Optional, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

# Add utils to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

try:
    from utils.raw_data_loader import RawDataLoader
    from utils.signal_filters import SignalFilter, FilterBank
    from utils.interactive_viewer import InteractivePlotter
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running from the blech_clust directory")
    sys.exit(1)


class RawDataViewerApp:
    """
    Main application class for the raw data viewer.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the raw data viewer application.
        
        Args:
            config: Configuration dictionary with viewer parameters
        """
        self.config = config
        self.data_loader = None
        self.plotter = None
        
        # Initialize data loader
        self._initialize_data_loader()
        
        # Initialize plotter
        self._initialize_plotter()
    
    def _initialize_data_loader(self):
        """Initialize the data loader."""
        hdf5_path = self.config['hdf5_path']
        sampling_rate = self.config.get('sampling_rate')
        
        print(f"Loading data from: {hdf5_path}")
        
        try:
            self.data_loader = RawDataLoader(hdf5_path, sampling_rate)
            print(f"Data loaded successfully. Sampling rate: {self.data_loader.sampling_rate} Hz")
            
            # Print available channels
            channel_info = self.data_loader.get_available_channels()
            for group, channels in channel_info.items():
                print(f"Group '{group}': {len(channels)} channels")
                if len(channels) <= 10:
                    print(f"  Channels: {list(channels.keys())}")
                else:
                    print(f"  Channels: {list(channels.keys())[:5]} ... {list(channels.keys())[-5:]}")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            sys.exit(1)
    
    def _initialize_plotter(self):
        """Initialize the interactive plotter."""
        # Determine initial channel and group
        initial_group = self.config.get('group', 'raw')
        initial_channel = self.config.get('channel')
        
        # Validate group
        available_channels = self.data_loader.get_available_channels()
        if initial_group not in available_channels:
            available_groups = list(available_channels.keys())
            print(f"Group '{initial_group}' not found. Available groups: {available_groups}")
            initial_group = available_groups[0]
            print(f"Using group: {initial_group}")
        
        # Validate channel
        group_channels = list(available_channels[initial_group].keys())
        if initial_channel and initial_channel not in group_channels:
            print(f"Channel '{initial_channel}' not found in group '{initial_group}'")
            initial_channel = None
        
        if not initial_channel:
            initial_channel = group_channels[0]
            print(f"Using channel: {initial_channel}")
        
        # Create plotter
        try:
            self.plotter = InteractivePlotter(
                data_loader=self.data_loader,
                initial_channel=initial_channel,
                initial_group=initial_group,
                window_duration=self.config.get('window_duration', 10.0),
                update_callback=self._on_plotter_update
            )
            
            # Set initial filter if specified
            filter_type = self.config.get('filter_type')
            if filter_type:
                self._set_initial_filter(filter_type)
            
            # Set initial threshold if specified
            threshold = self.config.get('threshold')
            if threshold is not None:
                self.plotter.set_threshold(threshold)
            
            print("Interactive plotter initialized successfully")
            
        except Exception as e:
            print(f"Error initializing plotter: {e}")
            sys.exit(1)
    
    def _set_initial_filter(self, filter_type: str):
        """Set initial filter based on type."""
        sampling_rate = self.data_loader.sampling_rate
        
        if filter_type.lower() == 'spike':
            signal_filter = FilterBank.create_spike_filter(sampling_rate)
        elif filter_type.lower() == 'lfp':
            signal_filter = FilterBank.create_lfp_filter(sampling_rate)
        elif filter_type.lower() == 'emg':
            signal_filter = FilterBank.create_emg_filter(sampling_rate)
        elif filter_type.lower() == 'none':
            signal_filter = SignalFilter(sampling_rate, filter_type='none')
        else:
            # Try to parse custom filter
            try:
                parts = filter_type.split('-')
                if len(parts) == 2:
                    low_freq, high_freq = map(float, parts)
                    signal_filter = SignalFilter(
                        sampling_rate=sampling_rate,
                        filter_type='bandpass',
                        low_freq=low_freq,
                        high_freq=high_freq
                    )
                else:
                    raise ValueError("Invalid filter format")
            except:
                print(f"Invalid filter type: {filter_type}")
                print("Valid options: spike, lfp, emg, none, or 'low-high' (e.g., '300-3000')")
                signal_filter = SignalFilter(sampling_rate, filter_type='none')
        
        self.plotter.set_filter(signal_filter)
        print(f"Filter set: {signal_filter}")
    
    def _on_plotter_update(self, plotter):
        """Callback for plotter updates."""
        # This could be used for logging, saving state, etc.
        pass
    
    def run(self):
        """Run the interactive viewer."""
        print("\n" + "="*60)
        print("Raw Data Viewer - Interactive Mode")
        print("="*60)
        print("Controls:")
        print("  Mouse scroll: Navigate time")
        print("  Double-click: Jump to time")
        print("  Left/Right arrows: Navigate time")
        print("  Up/Down arrows: Change channel")
        print("  'f' key: Cycle through filters")
        print("  'r' key: Reset view")
        print("  Use GUI controls for precise adjustments")
        print("="*60)
        
        try:
            self.plotter.show()
        except KeyboardInterrupt:
            print("\nViewer closed by user")
        except Exception as e:
            print(f"Error running viewer: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        if self.plotter:
            self.plotter.close()
        print("Viewer closed")


def find_hdf5_file(data_path: str) -> str:
    """
    Find HDF5 file in the given path.
    
    Args:
        data_path: Path to directory or HDF5 file
        
    Returns:
        Path to HDF5 file
    """
    if os.path.isfile(data_path) and data_path.endswith('.h5'):
        return data_path
    
    if os.path.isdir(data_path):
        # Look for HDF5 files in directory
        h5_files = glob.glob(os.path.join(data_path, '*.h5'))
        if len(h5_files) == 1:
            return h5_files[0]
        elif len(h5_files) > 1:
            print(f"Multiple HDF5 files found: {h5_files}")
            print("Please specify the exact file path")
            sys.exit(1)
        else:
            print(f"No HDF5 files found in {data_path}")
            sys.exit(1)
    
    raise FileNotFoundError(f"Invalid path: {data_path}")


def load_config_file(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config file: {e}")
        return {}


def create_default_config() -> Dict[str, Any]:
    """Create default configuration."""
    return {
        'window_duration': 10.0,
        'group': 'raw',
        'channel': None,
        'filter_type': None,
        'threshold': None,
        'sampling_rate': None
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Interactive viewer for raw neural data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/data/
  %(prog)s /path/to/data.h5 --channel electrode00
  %(prog)s /path/to/data/ --window 5.0 --filter spike
  %(prog)s /path/to/data/ --config viewer_config.json
        """
    )
    
    parser.add_argument('data_path', 
                       help='Path to data directory or HDF5 file')
    
    parser.add_argument('--channel', '-c',
                       help='Initial channel to display')
    
    parser.add_argument('--group', '-g', 
                       default='raw',
                       help='Data group to use (default: raw)')
    
    parser.add_argument('--window', '-w',
                       type=float, default=10.0,
                       help='Window duration in seconds (default: 10.0)')
    
    parser.add_argument('--filter', '-f',
                       help='Filter type: spike, lfp, emg, none, or "low-high" (e.g., "300-3000")')
    
    parser.add_argument('--threshold', '-t',
                       type=float,
                       help='Threshold value to display')
    
    parser.add_argument('--sampling-rate', '-s',
                       type=float,
                       help='Sampling rate in Hz (auto-detect if not specified)')
    
    parser.add_argument('--config',
                       help='Path to JSON configuration file')
    
    parser.add_argument('--save-config',
                       help='Save current configuration to file and exit')
    
    args = parser.parse_args()
    
    # Load base configuration
    if args.config:
        config = load_config_file(args.config)
    else:
        config = create_default_config()
    
    # Override with command line arguments
    config['hdf5_path'] = find_hdf5_file(args.data_path)
    
    if args.channel:
        config['channel'] = args.channel
    if args.group:
        config['group'] = args.group
    if args.window:
        config['window_duration'] = args.window
    if args.filter:
        config['filter_type'] = args.filter
    if args.threshold is not None:
        config['threshold'] = args.threshold
    if args.sampling_rate:
        config['sampling_rate'] = args.sampling_rate
    
    # Save configuration if requested
    if args.save_config:
        try:
            with open(args.save_config, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"Configuration saved to {args.save_config}")
            return
        except Exception as e:
            print(f"Error saving configuration: {e}")
            sys.exit(1)
    
    # Create and run the application
    try:
        app = RawDataViewerApp(config)
        app.run()
    except Exception as e:
        print(f"Error running application: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()