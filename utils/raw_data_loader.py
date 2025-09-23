"""
Raw Data Loader for Interactive Viewer

This module provides efficient loading of raw neural data from HDF5 files
with memory management capabilities for the interactive raw data viewer.

Classes:
    RawDataLoader: Main class for loading raw data with chunked reading
"""

import numpy as np
import tables
import os
from typing import Optional, Tuple, List, Union
import warnings


class RawDataLoader:
    """
    Efficient loader for raw neural data from HDF5 files.

    Supports loading single channels and specific time windows to prevent
    memory overflow when working with large datasets.
    """

    def __init__(self, hdf5_path: str, sampling_rate: Optional[float] = None):
        """
        Initialize the raw data loader.

        Args:
            hdf5_path: Path to the HDF5 file containing raw data
            sampling_rate: Sampling rate in Hz (will try to detect if not provided)
        """
        self.hdf5_path = hdf5_path
        self.sampling_rate = sampling_rate
        self._hdf5_file = None
        self._channel_info = {}
        self._data_groups = ['raw', 'raw_emg']

        # Validate file exists
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

        # Initialize connection and gather metadata
        self._initialize_metadata()

    def _initialize_metadata(self):
        """Initialize metadata about available channels and data structure."""
        with tables.open_file(self.hdf5_path, 'r') as hf5:
            # Detect available data groups
            available_groups = []
            for group_name in self._data_groups:
                if f'/{group_name}' in hf5:
                    available_groups.append(group_name)

            if not available_groups:
                raise ValueError(
                    f"No raw data groups found in {self.hdf5_path}")

            self._available_groups = available_groups

            # Gather channel information
            for group_name in available_groups:
                group = hf5.get_node(f'/{group_name}')
                channels = {}

                for node in group:
                    if hasattr(node, 'shape'):  # It's an array
                        channel_name = node._v_name
                        channels[channel_name] = {
                            'shape': node.shape,
                            'dtype': node.dtype,
                            'size_mb': node.size_in_memory / (1024 * 1024)
                        }

                self._channel_info[group_name] = channels

            # Try to detect sampling rate if not provided
            if self.sampling_rate is None:
                self._detect_sampling_rate(hf5)

    def _detect_sampling_rate(self, hf5):
        """Attempt to detect sampling rate from HDF5 metadata."""
        # Look for sampling rate in common locations
        possible_locations = [
            '/sampling_rate',
            '/info/sampling_rate',
            '/metadata/sampling_rate'
        ]

        for location in possible_locations:
            try:
                if location in hf5:
                    self.sampling_rate = float(hf5.get_node(location).read())
                    return
            except:
                continue

        # Default fallback
        warnings.warn("Could not detect sampling rate, using default 30000 Hz")
        self.sampling_rate = 30000.0

    def get_available_channels(self) -> dict:
        """
        Get information about available channels.

        Returns:
            Dictionary with group names as keys and channel info as values
        """
        return self._channel_info.copy()

    def get_channel_list(self, group: str = 'raw') -> List[str]:
        """
        Get list of available channels in a group.

        Args:
            group: Data group name ('raw' or 'raw_emg')

        Returns:
            List of channel names
        """
        if group not in self._channel_info:
            raise ValueError(
                f"Group '{group}' not found. Available: {list(self._channel_info.keys())}")

        return list(self._channel_info[group].keys())

    def get_channel_duration(self, channel: str, group: str = 'raw') -> float:
        """
        Get duration of a channel in seconds.

        Args:
            channel: Channel name
            group: Data group name

        Returns:
            Duration in seconds
        """
        if group not in self._channel_info:
            raise ValueError(f"Group '{group}' not found")

        if channel not in self._channel_info[group]:
            raise ValueError(
                f"Channel '{channel}' not found in group '{group}'")

        n_samples = self._channel_info[group][channel]['shape'][0]
        return n_samples / self.sampling_rate

    def load_channel_data(self,
                          channel: str,
                          group: str = 'raw',
                          start_time: Optional[float] = None,
                          end_time: Optional[float] = None,
                          start_sample: Optional[int] = None,
                          end_sample: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data from a specific channel and time range.

        Args:
            channel: Channel name (e.g., 'electrode00', 'emg01')
            group: Data group ('raw' or 'raw_emg')
            start_time: Start time in seconds (mutually exclusive with start_sample)
            end_time: End time in seconds (mutually exclusive with end_sample)
            start_sample: Start sample index
            end_sample: End sample index

        Returns:
            Tuple of (data_array, time_array)
        """
        # Validate inputs
        if group not in self._channel_info:
            raise ValueError(
                f"Group '{group}' not found. Available: {list(self._channel_info.keys())}")

        if channel not in self._channel_info[group]:
            available = list(self._channel_info[group].keys())
            raise ValueError(
                f"Channel '{channel}' not found in group '{group}'. Available: {available}")

        # Convert time to samples if needed
        if start_time is not None or end_time is not None:
            if start_sample is not None or end_sample is not None:
                raise ValueError("Cannot specify both time and sample ranges")

            total_samples = self._channel_info[group][channel]['shape'][0]

            if start_time is not None:
                start_sample = int(start_time * self.sampling_rate)
            else:
                start_sample = 0

            if end_time is not None:
                end_sample = int(end_time * self.sampling_rate)
            else:
                end_sample = total_samples

        # Default to full range if nothing specified
        if start_sample is None:
            start_sample = 0
        if end_sample is None:
            end_sample = self._channel_info[group][channel]['shape'][0]

        # Validate sample range
        total_samples = self._channel_info[group][channel]['shape'][0]
        start_sample = max(0, start_sample)
        end_sample = min(total_samples, end_sample)

        if start_sample >= end_sample:
            raise ValueError(
                f"Invalid sample range: {start_sample} >= {end_sample}")

        # Load data
        with tables.open_file(self.hdf5_path, 'r') as hf5:
            node = hf5.get_node(f'/{group}', channel)
            data = node[start_sample:end_sample]

        # Create time array
        time_array = np.arange(start_sample, end_sample) / self.sampling_rate

        return data, time_array

    def load_multiple_channels(self,
                               channels: List[str],
                               group: str = 'raw',
                               start_time: Optional[float] = None,
                               end_time: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data from multiple channels simultaneously.

        Args:
            channels: List of channel names
            group: Data group name
            start_time: Start time in seconds
            end_time: End time in seconds

        Returns:
            Tuple of (data_array, time_array) where data_array is shape (n_channels, n_samples)
        """
        if not channels:
            raise ValueError("No channels specified")

        # Load first channel to get time array and validate parameters
        first_data, time_array = self.load_channel_data(
            channels[0], group, start_time, end_time
        )

        # Initialize output array
        data_array = np.zeros(
            (len(channels), len(first_data)), dtype=first_data.dtype)
        data_array[0] = first_data

        # Load remaining channels
        for i, channel in enumerate(channels[1:], 1):
            data, _ = self.load_channel_data(
                channel, group, start_time, end_time
            )
            data_array[i] = data

        return data_array, time_array

    def get_data_info(self) -> dict:
        """
        Get comprehensive information about the loaded data.

        Returns:
            Dictionary with data information
        """
        info = {
            'hdf5_path': self.hdf5_path,
            'sampling_rate': self.sampling_rate,
            'available_groups': self._available_groups,
            'channels': {}
        }

        for group_name, channels in self._channel_info.items():
            info['channels'][group_name] = {}
            for channel_name, channel_info in channels.items():
                duration = channel_info['shape'][0] / self.sampling_rate
                info['channels'][group_name][channel_name] = {
                    'duration_seconds': duration,
                    'n_samples': channel_info['shape'][0],
                    'dtype': str(channel_info['dtype']),
                    'size_mb': channel_info['size_mb']
                }

        return info

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._hdf5_file is not None:
            self._hdf5_file.close()

    def __repr__(self):
        """String representation."""
        n_channels = sum(len(channels)
                         for channels in self._channel_info.values())
        return f"RawDataLoader('{self.hdf5_path}', {n_channels} channels, {self.sampling_rate} Hz)"


def load_sample_data(hdf5_path: str,
                     channel: str = None,
                     duration: float = 10.0,
                     group: str = 'raw') -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to load a sample of data for testing/preview.

    Args:
        hdf5_path: Path to HDF5 file
        channel: Channel name (if None, uses first available)
        duration: Duration in seconds to load
        group: Data group name

    Returns:
        Tuple of (data, time_array)
    """
    with RawDataLoader(hdf5_path) as loader:
        if channel is None:
            available_channels = loader.get_channel_list(group)
            if not available_channels:
                raise ValueError(f"No channels found in group '{group}'")
            channel = available_channels[0]

        return loader.load_channel_data(
            channel=channel,
            group=group,
            start_time=0,
            end_time=duration
        )
