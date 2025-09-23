"""
Unit tests for RawDataLoader class.
"""

from raw_data_loader import RawDataLoader, load_sample_data
import unittest
import numpy as np
import tables
import tempfile
import os
import shutil
from unittest.mock import patch, MagicMock

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))


class TestRawDataLoader(unittest.TestCase):
    """Test cases for RawDataLoader class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.hdf5_path = os.path.join(self.temp_dir, 'test_data.h5')
        self.sampling_rate = 30000.0

        # Create test HDF5 file
        self._create_test_hdf5()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def _create_test_hdf5(self):
        """Create a test HDF5 file with sample data."""
        with tables.open_file(self.hdf5_path, 'w') as hf5:
            # Create groups
            raw_group = hf5.create_group('/', 'raw')
            emg_group = hf5.create_group('/', 'raw_emg')

            # Create sample data
            n_samples = int(self.sampling_rate * 10)  # 10 seconds of data

            # Raw electrode data
            for i in range(3):
                data = np.random.randint(-1000, 1000,
                                         n_samples, dtype=np.int16)
                # Add some structure to the data
                data += (np.sin(2 * np.pi * 10 * np.arange(n_samples) /
                         self.sampling_rate) * 100).astype(np.int16)
                hf5.create_array(raw_group, f'electrode{i:02d}', data)

            # EMG data
            for i in range(2):
                data = np.random.randint(-500, 500, n_samples, dtype=np.int16)
                hf5.create_array(emg_group, f'emg{i:02d}', data)

            # Add sampling rate
            hf5.create_array('/', 'sampling_rate', self.sampling_rate)

    def test_initialization(self):
        """Test RawDataLoader initialization."""
        loader = RawDataLoader(self.hdf5_path)

        self.assertEqual(loader.hdf5_path, self.hdf5_path)
        self.assertEqual(loader.sampling_rate, self.sampling_rate)
        self.assertIn('raw', loader._available_groups)
        self.assertIn('raw_emg', loader._available_groups)

    def test_initialization_with_sampling_rate(self):
        """Test initialization with explicit sampling rate."""
        custom_rate = 25000.0
        loader = RawDataLoader(self.hdf5_path, sampling_rate=custom_rate)

        self.assertEqual(loader.sampling_rate, custom_rate)

    def test_initialization_file_not_found(self):
        """Test initialization with non-existent file."""
        with self.assertRaises(FileNotFoundError):
            RawDataLoader('/nonexistent/file.h5')

    def test_get_available_channels(self):
        """Test getting available channels."""
        loader = RawDataLoader(self.hdf5_path)
        channels = loader.get_available_channels()

        self.assertIn('raw', channels)
        self.assertIn('raw_emg', channels)
        self.assertEqual(len(channels['raw']), 3)
        self.assertEqual(len(channels['raw_emg']), 2)

        # Check channel names
        raw_channels = list(channels['raw'].keys())
        self.assertIn('electrode00', raw_channels)
        self.assertIn('electrode01', raw_channels)
        self.assertIn('electrode02', raw_channels)

    def test_get_channel_list(self):
        """Test getting channel list for a group."""
        loader = RawDataLoader(self.hdf5_path)

        raw_channels = loader.get_channel_list('raw')
        self.assertEqual(len(raw_channels), 3)
        self.assertIn('electrode00', raw_channels)

        emg_channels = loader.get_channel_list('raw_emg')
        self.assertEqual(len(emg_channels), 2)
        self.assertIn('emg00', emg_channels)

    def test_get_channel_list_invalid_group(self):
        """Test getting channel list for invalid group."""
        loader = RawDataLoader(self.hdf5_path)

        with self.assertRaises(ValueError):
            loader.get_channel_list('invalid_group')

    def test_get_channel_duration(self):
        """Test getting channel duration."""
        loader = RawDataLoader(self.hdf5_path)

        duration = loader.get_channel_duration('electrode00', 'raw')
        self.assertAlmostEqual(duration, 10.0, places=1)

    def test_load_channel_data_full(self):
        """Test loading full channel data."""
        loader = RawDataLoader(self.hdf5_path)

        data, time_array = loader.load_channel_data('electrode00', 'raw')

        self.assertEqual(len(data), int(self.sampling_rate * 10))
        self.assertEqual(len(time_array), len(data))
        self.assertAlmostEqual(time_array[-1], 10.0, places=1)
        self.assertEqual(data.dtype, np.int16)

    def test_load_channel_data_time_range(self):
        """Test loading channel data with time range."""
        loader = RawDataLoader(self.hdf5_path)

        data, time_array = loader.load_channel_data(
            'electrode00', 'raw', start_time=2.0, end_time=5.0
        )

        expected_samples = int(self.sampling_rate * 3)  # 3 seconds
        self.assertEqual(len(data), expected_samples)
        self.assertEqual(len(time_array), len(data))
        self.assertAlmostEqual(time_array[0], 2.0, places=3)
        self.assertAlmostEqual(time_array[-1], 5.0, places=1)

    def test_load_channel_data_sample_range(self):
        """Test loading channel data with sample range."""
        loader = RawDataLoader(self.hdf5_path)

        start_sample = int(self.sampling_rate * 1)  # 1 second
        end_sample = int(self.sampling_rate * 3)    # 3 seconds

        data, time_array = loader.load_channel_data(
            'electrode00', 'raw', start_sample=start_sample, end_sample=end_sample
        )

        expected_samples = end_sample - start_sample
        self.assertEqual(len(data), expected_samples)
        self.assertEqual(len(time_array), len(data))
        self.assertAlmostEqual(time_array[0], 1.0, places=3)

    def test_load_channel_data_invalid_channel(self):
        """Test loading data from invalid channel."""
        loader = RawDataLoader(self.hdf5_path)

        with self.assertRaises(ValueError):
            loader.load_channel_data('invalid_channel', 'raw')

    def test_load_channel_data_invalid_group(self):
        """Test loading data from invalid group."""
        loader = RawDataLoader(self.hdf5_path)

        with self.assertRaises(ValueError):
            loader.load_channel_data('electrode00', 'invalid_group')

    def test_load_channel_data_conflicting_params(self):
        """Test loading data with conflicting time/sample parameters."""
        loader = RawDataLoader(self.hdf5_path)

        with self.assertRaises(ValueError):
            loader.load_channel_data(
                'electrode00', 'raw',
                start_time=1.0, start_sample=1000
            )

    def test_load_multiple_channels(self):
        """Test loading multiple channels simultaneously."""
        loader = RawDataLoader(self.hdf5_path)

        channels = ['electrode00', 'electrode01']
        data_array, time_array = loader.load_multiple_channels(
            channels, 'raw', start_time=1.0, end_time=3.0
        )

        expected_samples = int(self.sampling_rate * 2)  # 2 seconds
        self.assertEqual(data_array.shape, (2, expected_samples))
        self.assertEqual(len(time_array), expected_samples)

    def test_load_multiple_channels_empty_list(self):
        """Test loading multiple channels with empty list."""
        loader = RawDataLoader(self.hdf5_path)

        with self.assertRaises(ValueError):
            loader.load_multiple_channels([], 'raw')

    def test_get_data_info(self):
        """Test getting comprehensive data information."""
        loader = RawDataLoader(self.hdf5_path)

        info = loader.get_data_info()

        self.assertEqual(info['hdf5_path'], self.hdf5_path)
        self.assertEqual(info['sampling_rate'], self.sampling_rate)
        self.assertIn('raw', info['available_groups'])
        self.assertIn('raw_emg', info['available_groups'])

        # Check channel info
        self.assertIn('raw', info['channels'])
        self.assertIn('electrode00', info['channels']['raw'])
        self.assertAlmostEqual(
            info['channels']['raw']['electrode00']['duration_seconds'],
            10.0, places=1
        )

    def test_context_manager(self):
        """Test using RawDataLoader as context manager."""
        with RawDataLoader(self.hdf5_path) as loader:
            self.assertIsInstance(loader, RawDataLoader)
            data, _ = loader.load_channel_data(
                'electrode00', 'raw', start_time=0, end_time=1)
            self.assertIsInstance(data, np.ndarray)

    def test_repr(self):
        """Test string representation."""
        loader = RawDataLoader(self.hdf5_path)
        repr_str = repr(loader)

        self.assertIn('RawDataLoader', repr_str)
        self.assertIn(self.hdf5_path, repr_str)
        self.assertIn('5 channels', repr_str)  # 3 raw + 2 emg
        self.assertIn(str(self.sampling_rate), repr_str)


class TestLoadSampleData(unittest.TestCase):
    """Test cases for load_sample_data function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.hdf5_path = os.path.join(self.temp_dir, 'test_data.h5')
        self.sampling_rate = 30000.0

        # Create test HDF5 file
        self._create_test_hdf5()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def _create_test_hdf5(self):
        """Create a test HDF5 file with sample data."""
        with tables.open_file(self.hdf5_path, 'w') as hf5:
            raw_group = hf5.create_group('/', 'raw')
            n_samples = int(self.sampling_rate * 20)  # 20 seconds of data

            data = np.random.randint(-1000, 1000, n_samples, dtype=np.int16)
            hf5.create_array(raw_group, 'electrode00', data)
            hf5.create_array('/', 'sampling_rate', self.sampling_rate)

    def test_load_sample_data_default(self):
        """Test loading sample data with defaults."""
        data, time_array = load_sample_data(self.hdf5_path)

        expected_samples = int(self.sampling_rate * 10)  # 10 seconds default
        self.assertEqual(len(data), expected_samples)
        self.assertEqual(len(time_array), len(data))
        self.assertAlmostEqual(time_array[-1], 10.0, places=1)

    def test_load_sample_data_custom_duration(self):
        """Test loading sample data with custom duration."""
        data, time_array = load_sample_data(self.hdf5_path, duration=5.0)

        expected_samples = int(self.sampling_rate * 5)  # 5 seconds
        self.assertEqual(len(data), expected_samples)
        self.assertAlmostEqual(time_array[-1], 5.0, places=1)

    def test_load_sample_data_custom_channel(self):
        """Test loading sample data with custom channel."""
        data, time_array = load_sample_data(
            self.hdf5_path, channel='electrode00', duration=3.0
        )

        expected_samples = int(self.sampling_rate * 3)  # 3 seconds
        self.assertEqual(len(data), expected_samples)


if __name__ == '__main__':
    unittest.main()
