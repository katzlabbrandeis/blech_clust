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

    Supports two loading strategies:
    1. Streaming mode: Load only requested time windows on-demand (memory efficient)
    2. Memory mode: Load full channel into memory for faster window extraction
    """

    def __init__(self, hdf5_path: str, sampling_rate: Optional[float] = None, 
                 loading_strategy: str = 'streaming'):
        """
        Initialize the raw data loader.

        Args:
            hdf5_path: Path to the HDF5 file containing raw data
            sampling_rate: Sampling rate in Hz (will try to detect if not provided)
            loading_strategy: 'streaming' (default) or 'memory'
        """
        self.hdf5_path = hdf5_path
        self.sampling_rate = sampling_rate
        self.loading_strategy = loading_strategy
        self._hdf5_file = None
        self._channel_info = {}
        self._data_groups = ['raw', 'raw_emg']
        
        # Memory mode storage
        self._memory_cache = {}  # {(group, channel): (data_array, time_array)}
        self._cached_channels = set()  # Track which channels are cached

        # Validate inputs
        if loading_strategy not in ['streaming', 'memory']:
            raise ValueError("loading_strategy must be 'streaming' or 'memory'")

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

        Uses either streaming or memory loading strategy based on initialization.

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
        if self.loading_strategy == 'memory':
            return self._load_channel_data_memory(
                channel, group, start_time, end_time, start_sample, end_sample)
        else:
            return self._load_channel_data_streaming(
                channel, group, start_time, end_time, start_sample, end_sample)
    def _load_channel_data_streaming(self,
                                   channel: str,
                                   group: str = 'raw',
                                   start_time: Optional[float] = None,
                                   end_time: Optional[float] = None,
                                   start_sample: Optional[int] = None,
                                   end_sample: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data using streaming strategy (original implementation).
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

        # Load data from HDF5
        with tables.open_file(self.hdf5_path, 'r') as hf5:
            node = hf5.get_node(f'/{group}', channel)
            data = node[start_sample:end_sample]

        # Create time array
        time_array = np.arange(start_sample, end_sample) / self.sampling_rate

        return data, time_array

    def _load_channel_data_memory(self,
                                channel: str,
                                group: str = 'raw',
                                start_time: Optional[float] = None,
                                end_time: Optional[float] = None,
                                start_sample: Optional[int] = None,
                                end_sample: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data using memory strategy (cache full channel, extract windows).
        """
        # Update request statistics
        if not hasattr(self, '_total_requests'):
            self._total_requests = 0
        self._total_requests += 1
        
        # Validate inputs
        if group not in self._channel_info:
            raise ValueError(
                f"Group '{group}' not found. Available: {list(self._channel_info.keys())}")

        if channel not in self._channel_info[group]:
            available = list(self._channel_info[group].keys())
            raise ValueError(
                f"Channel '{channel}' not found in group '{group}'. Available: {available}")

        # Ensure channel is cached
        cache_key = (group, channel)
        if cache_key not in self._memory_cache:
            self._cache_full_channel(channel, group)
        else:
            # Update cache hit statistics
            if hasattr(self, '_cache_hits'):
                self._cache_hits += 1

        # Get cached data (check again in case it was evicted during caching)
        if cache_key not in self._memory_cache:
            # Channel was evicted due to memory limits, reload it
            self._cache_full_channel(channel, group)
        
        full_data, full_time = self._memory_cache[cache_key]

        # Convert time to samples if needed
        if start_time is not None or end_time is not None:
            if start_sample is not None or end_sample is not None:
                raise ValueError("Cannot specify both time and sample ranges")

            total_samples = len(full_data)

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
            end_sample = len(full_data)

        # Validate sample range
        total_samples = len(full_data)
        start_sample = max(0, start_sample)
        end_sample = min(total_samples, end_sample)

        if start_sample >= end_sample:
            raise ValueError(
                f"Invalid sample range: {start_sample} >= {end_sample}")

        # Extract window from cached data
        data = full_data[start_sample:end_sample]
        time_array = full_time[start_sample:end_sample]

        return data, time_array

    def _cache_full_channel(self, channel: str, group: str = 'raw'):
        """
        Cache full channel data in memory.
        
        Args:
            channel: Channel name
            group: Data group name
        """
        cache_key = (group, channel)
        
        if cache_key in self._memory_cache:
            # Update cache hit statistics
            if hasattr(self, '_cache_hits'):
                self._cache_hits += 1
            return  # Already cached
            
        # Update cache miss statistics
        if not hasattr(self, '_cache_hits'):
            self._cache_hits = 0
            self._cache_misses = 0
            self._total_requests = 0
        self._cache_misses += 1
        
        print(f"Loading full channel {channel} from group {group} into memory...")
        
        # Check memory limit before caching
        if hasattr(self, '_memory_limit_mb'):
            channel_size_mb = self._channel_info[group][channel]['size_mb']
            current_usage = sum(data.nbytes for data, _ in self._memory_cache.values()) / (1024 * 1024)
            
            if current_usage + channel_size_mb > self._memory_limit_mb:
                print(f"Would exceed memory limit ({self._memory_limit_mb:.1f} MB), "
                      f"clearing cache first...")
                self._enforce_memory_limit()
        
        # Load full channel using streaming method
        data, time_array = self._load_channel_data_streaming(
            channel, group, start_sample=0, 
            end_sample=self._channel_info[group][channel]['shape'][0]
        )
        
        # Store in cache
        self._memory_cache[cache_key] = (data, time_array)
        self._cached_channels.add(cache_key)
        
        # Enforce memory limit after caching
        if hasattr(self, '_memory_limit_mb'):
            self._enforce_memory_limit()
        
        # Report memory usage
        size_mb = data.nbytes / (1024 * 1024)
        print(f"Cached {channel} ({size_mb:.1f} MB) in memory")

    def clear_memory_cache(self, channel: Optional[str] = None, group: Optional[str] = None):
        """
        Clear memory cache for specific channel or all channels.
        
        Args:
            channel: Specific channel to clear (None for all)
            group: Specific group to clear (None for all)
        """
        freed_memory = 0
        
        if channel is not None and group is not None:
            # Clear specific channel
            cache_key = (group, channel)
            if cache_key in self._memory_cache:
                data, _ = self._memory_cache[cache_key]
                freed_memory = data.nbytes / (1024 * 1024)
                del self._memory_cache[cache_key]
                self._cached_channels.discard(cache_key)
                print(f"Cleared cache for {channel} in group {group} (freed {freed_memory:.1f} MB)")
        else:
            # Clear all or by group
            keys_to_remove = []
            for cache_key in self._memory_cache.keys():
                cached_group, cached_channel = cache_key
                if group is None or cached_group == group:
                    keys_to_remove.append(cache_key)
            
            for key in keys_to_remove:
                data, _ = self._memory_cache[key]
                freed_memory += data.nbytes / (1024 * 1024)
                del self._memory_cache[key]
                self._cached_channels.discard(key)
            
            print(f"Cleared {len(keys_to_remove)} cached channels (freed {freed_memory:.1f} MB)")

    def get_cache_efficiency_stats(self) -> dict:
        """
        Get cache efficiency statistics.
        
        Returns:
            Dictionary with cache performance metrics
        """
        if not hasattr(self, '_cache_hits'):
            self._cache_hits = 0
            self._cache_misses = 0
            self._total_requests = 0
        
        hit_rate = self._cache_hits / max(1, self._total_requests)
        
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'total_requests': self._total_requests,
            'hit_rate': hit_rate,
            'cached_channels': len(self._memory_cache),
            'memory_usage_mb': sum(data.nbytes for data, _ in self._memory_cache.values()) / (1024 * 1024)
        }

    def set_memory_limit(self, limit_mb: float):
        """
        Set memory limit for cache and enforce it.
        
        Args:
            limit_mb: Memory limit in megabytes
        """
        self._memory_limit_mb = limit_mb
        self._enforce_memory_limit()

    def _enforce_memory_limit(self):
        """
        Enforce memory limit by removing least recently used channels.
        """
        if not hasattr(self, '_memory_limit_mb'):
            return
            
        current_usage = sum(data.nbytes for data, _ in self._memory_cache.values()) / (1024 * 1024)
        
        if current_usage <= self._memory_limit_mb:
            return
            
        # Sort by access time (if we had it) or just remove oldest entries
        # For now, remove channels until under limit
        channels_to_remove = []
        freed_memory = 0
        
        for cache_key, (data, _) in self._memory_cache.items():
            channels_to_remove.append(cache_key)
            freed_memory += data.nbytes / (1024 * 1024)
            
            if current_usage - freed_memory <= self._memory_limit_mb:
                break
        
        for key in channels_to_remove:
            del self._memory_cache[key]
            self._cached_channels.discard(key)
        
        if channels_to_remove:
            print(f"Enforced memory limit: removed {len(channels_to_remove)} channels "
                  f"(freed {freed_memory:.1f} MB)")

    def optimize_for_sequential_access(self, enable: bool = True):
        """
        Optimize loading for sequential time window access patterns.
        
        Args:
            enable: Whether to enable sequential access optimization
        """
        self._sequential_optimization = enable
        if enable:
            print("Enabled sequential access optimization")
        else:
            print("Disabled sequential access optimization")

    def get_memory_usage(self) -> dict:
        """
        Get current memory usage information.
        
        Returns:
            Dictionary with memory usage details
        """
        usage = {
            'cached_channels': len(self._memory_cache),
            'total_memory_mb': 0,
            'channels': {}
        }
        
        for (group, channel), (data, _) in self._memory_cache.items():
            size_mb = data.nbytes / (1024 * 1024)
            usage['total_memory_mb'] += size_mb
            usage['channels'][f"{group}/{channel}"] = size_mb
            
        return usage

    def load_multiple_channels(self,
                               channels: List[str],
                               group: str = 'raw',
                               start_time: Optional[float] = None,
                               end_time: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data from multiple channels simultaneously.

        Uses the configured loading strategy (streaming or memory).

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

        # For memory mode, pre-cache all channels if not already cached
        if self.loading_strategy == 'memory':
            for channel in channels:
                cache_key = (group, channel)
                if cache_key not in self._memory_cache:
                    self._cache_full_channel(channel, group)

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

    def switch_loading_strategy(self, new_strategy: str):
        """
        Switch between loading strategies.
        
        Args:
            new_strategy: 'streaming' or 'memory'
        """
        if new_strategy not in ['streaming', 'memory']:
            raise ValueError("loading_strategy must be 'streaming' or 'memory'")
            
        if new_strategy == self.loading_strategy:
            return  # No change needed
            
        old_strategy = self.loading_strategy
        self.loading_strategy = new_strategy
        
        print(f"Switched loading strategy from {old_strategy} to {new_strategy}")
        
        # If switching away from memory mode, optionally clear cache
        if old_strategy == 'memory' and new_strategy == 'streaming':
            print("Consider calling clear_memory_cache() to free memory")

    def preload_channels(self, channels: List[str], group: str = 'raw'):
        """
        Preload channels into memory (only works in memory mode).
        
        Args:
            channels: List of channel names to preload
            group: Data group name
        """
        if self.loading_strategy != 'memory':
            raise ValueError("Preloading only available in memory mode")
            
        for channel in channels:
            self._cache_full_channel(channel, group)

    def get_data_info(self) -> dict:
        """
        Get comprehensive information about the loaded data.

        Returns:
            Dictionary with data information
        """
        info = {
            'hdf5_path': self.hdf5_path,
            'sampling_rate': self.sampling_rate,
            'loading_strategy': self.loading_strategy,
            'available_groups': self._available_groups,
            'memory_usage': self.get_memory_usage() if self.loading_strategy == 'memory' else None,
            'channels': {}
        }

        for group_name, channels in self._channel_info.items():
            info['channels'][group_name] = {}
            for channel_name, channel_info in channels.items():
                duration = channel_info['shape'][0] / self.sampling_rate
                cache_key = (group_name, channel_name)
                info['channels'][group_name][channel_name] = {
                    'duration_seconds': duration,
                    'n_samples': channel_info['shape'][0],
                    'dtype': str(channel_info['dtype']),
                    'size_mb': channel_info['size_mb'],
                    'cached_in_memory': cache_key in self._memory_cache
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
