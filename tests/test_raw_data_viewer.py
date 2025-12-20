"""
Integration tests for the raw data viewer application.
"""

import raw_data_viewer
from interactive_viewer import InteractivePlotter
from signal_filters import SignalFilter, FilterBank
from raw_data_loader import RawDataLoader
import sys
import unittest
import numpy as np
import tables
import tempfile
import os
import shutil
import json
from unittest.mock import patch, MagicMock
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))


class TestRawDataViewerIntegration(unittest.TestCase):
    """Integration tests for the complete raw data viewer workflow."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.hdf5_path = os.path.join(self.temp_dir, 'test_data.h5')
        self.config_path = os.path.join(self.temp_dir, 'test_config.json')
        self.sampling_rate = 30000.0

        # Create test HDF5 file
        self._create_test_hdf5()

        # Create test configuration
        self._create_test_config()

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
            n_samples = int(self.sampling_rate * 30)  # 30 seconds of data

            # Raw electrode data with different characteristics
            for i in range(4):
                # Create signal with different frequency content for each channel
                t = np.arange(n_samples) / self.sampling_rate
                base_signal = np.random.normal(0, 100, n_samples)

                # Add frequency components
                if i == 0:  # Low frequency channel
                    base_signal += 200 * np.sin(2 * np.pi * 10 * t)
                elif i == 1:  # High frequency channel
                    base_signal += 150 * np.sin(2 * np.pi * 1000 * t)
                elif i == 2:  # Mixed frequency channel
                    base_signal += 100 * np.sin(2 * np.pi * 50 * t)
                    base_signal += 80 * np.sin(2 * np.pi * 500 * t)
                else:  # Noisy channel
                    base_signal += np.random.normal(0, 200, n_samples)

                data = base_signal.astype(np.int16)
                hf5.create_array(raw_group, f'electrode{i:02d}', data)

            # EMG data
            for i in range(2):
                t = np.arange(n_samples) / self.sampling_rate
                emg_signal = np.random.normal(0, 50, n_samples)
                # EMG-like frequency
                emg_signal += 100 * np.sin(2 * np.pi * 20 * t)
                data = emg_signal.astype(np.int16)
                hf5.create_array(emg_group, f'emg{i:02d}', data)

            # Add sampling rate
            hf5.create_array('/', 'sampling_rate', self.sampling_rate)

    def _create_test_config(self):
        """Create a test configuration file."""
        config = {
            'window_duration': 5.0,
            'group': 'raw',
            'channel': 'electrode00',
            'filter_type': 'spike',
            'threshold': 100.0,
            'sampling_rate': self.sampling_rate
        }

        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)

    def test_data_loader_integration(self):
        """Test data loader with real HDF5 file."""
        loader = RawDataLoader(self.hdf5_path)

        # Test basic functionality
        self.assertEqual(loader.sampling_rate, self.sampling_rate)
        self.assertIn('raw', loader._available_groups)
        self.assertIn('raw_emg', loader._available_groups)

        # Test loading data
        data, time_array = loader.load_channel_data(
            'electrode00', 'raw', start_time=0, end_time=5.0
        )

        expected_samples = int(self.sampling_rate * 5)
        self.assertEqual(len(data), expected_samples)
        self.assertEqual(len(time_array), expected_samples)

        # Test multiple channels
        channels = ['electrode00', 'electrode01']
        data_array, time_array = loader.load_multiple_channels(
            channels, 'raw', start_time=0, end_time=2.0
        )

        expected_samples = int(self.sampling_rate * 2)
        self.assertEqual(data_array.shape, (2, expected_samples))

    def test_filter_integration(self):
        """Test signal filtering with real data."""
        loader = RawDataLoader(self.hdf5_path)

        # Load test data
        data, _ = loader.load_channel_data(
            'electrode01', 'raw', start_time=0, end_time=5.0
        )

        # Test different filters
        filters_to_test = [
            FilterBank.create_spike_filter(self.sampling_rate),
            FilterBank.create_lfp_filter(self.sampling_rate),
            FilterBank.create_emg_filter(self.sampling_rate)
        ]

        for filt in filters_to_test:
            filtered_data = filt.filter_data(data)

            # Filtered data should have same length
            self.assertEqual(len(filtered_data), len(data))

            # Filtered data should be different from original
            self.assertFalse(np.array_equal(filtered_data, data))

            # Check that filter is stable (no NaN or inf values)
            self.assertFalse(np.any(np.isnan(filtered_data)))
            self.assertFalse(np.any(np.isinf(filtered_data)))

    @patch('matplotlib.pyplot.show')
    def test_interactive_plotter_creation(self, mock_show):
        """Test creating interactive plotter (without actually showing)."""
        loader = RawDataLoader(self.hdf5_path)

        # Create plotter
        plotter = InteractivePlotter(
            data_loader=loader,
            initial_channel='electrode00',
            initial_group='raw',
            window_duration=5.0
        )

        # Test basic properties
        self.assertEqual(plotter.current_channel, 'electrode00')
        self.assertEqual(plotter.current_group, 'raw')
        self.assertEqual(plotter.window_duration, 5.0)

        # Test getting current data
        data, time_array = plotter.get_current_data()
        self.assertIsInstance(data, np.ndarray)
        self.assertIsInstance(time_array, np.ndarray)

        # Test setting filter
        spike_filter = FilterBank.create_spike_filter(self.sampling_rate)
        plotter.set_filter(spike_filter)
        self.assertEqual(plotter.signal_filter.filter_type, 'bandpass')

        # Test setting threshold
        plotter.set_threshold(150.0)
        self.assertEqual(plotter.threshold_value, 150.0)
        self.assertTrue(plotter.show_threshold)

        # Test jumping to time
        plotter.jump_to_time(10.0)
        self.assertEqual(plotter.current_time, 10.0)

        # Test changing channel
        plotter.set_channel('electrode01')
        self.assertEqual(plotter.current_channel, 'electrode01')

        # Clean up
        plotter.close()

    def test_find_hdf5_file(self):
        """Test finding HDF5 files."""
        # Test with direct file path
        found_path = raw_data_viewer.find_hdf5_file(self.hdf5_path)
        self.assertEqual(found_path, self.hdf5_path)

        # Test with directory containing one HDF5 file
        found_path = raw_data_viewer.find_hdf5_file(self.temp_dir)
        self.assertEqual(found_path, self.hdf5_path)

        # Test with non-existent path
        with self.assertRaises(FileNotFoundError):
            raw_data_viewer.find_hdf5_file('/nonexistent/path')

        # Test with directory containing no HDF5 files
        empty_dir = os.path.join(self.temp_dir, 'empty')
        os.makedirs(empty_dir)

        with self.assertRaises(SystemExit):
            raw_data_viewer.find_hdf5_file(empty_dir)

    def test_load_config_file(self):
        """Test loading configuration file."""
        config = raw_data_viewer.load_config_file(self.config_path)

        self.assertEqual(config['window_duration'], 5.0)
        self.assertEqual(config['group'], 'raw')
        self.assertEqual(config['channel'], 'electrode00')
        self.assertEqual(config['filter_type'], 'spike')
        self.assertEqual(config['threshold'], 100.0)

    def test_create_default_config(self):
        """Test creating default configuration."""
        config = raw_data_viewer.create_default_config()

        self.assertIn('window_duration', config)
        self.assertIn('group', config)
        self.assertIn('channel', config)
        self.assertIn('filter_type', config)
        self.assertIn('threshold', config)
        self.assertIn('sampling_rate', config)

    @patch('raw_data_viewer.RawDataViewerApp')
    def test_main_with_config(self, mock_app_class):
        """Test main function with configuration file."""
        mock_app = MagicMock()
        mock_app_class.return_value = mock_app

        # Mock sys.argv
        test_args = [
            'raw_data_viewer.py',
            self.temp_dir,
            '--config', self.config_path,
            '--window', '3.0',
            '--threshold', '200'
        ]

        with patch('sys.argv', test_args):
            raw_data_viewer.main()

        # Check that app was created and run
        mock_app_class.assert_called_once()
        mock_app.run.assert_called_once()

        # Check configuration was passed correctly
        # First positional argument (config)
        call_args = mock_app_class.call_args[0][0]
        self.assertEqual(call_args['hdf5_path'], self.hdf5_path)
        # Overridden by command line
        self.assertEqual(call_args['window_duration'], 3.0)
        # Overridden by command line
        self.assertEqual(call_args['threshold'], 200)
        self.assertEqual(call_args['group'], 'raw')  # From config file

    @patch('raw_data_viewer.RawDataViewerApp')
    def test_main_without_config(self, mock_app_class):
        """Test main function without configuration file."""
        mock_app = MagicMock()
        mock_app_class.return_value = mock_app

        # Mock sys.argv
        test_args = [
            'raw_data_viewer.py',
            self.hdf5_path,
            '--channel', 'electrode02',
            '--filter', 'lfp'
        ]

        with patch('sys.argv', test_args):
            raw_data_viewer.main()

        # Check that app was created and run
        mock_app_class.assert_called_once()
        mock_app.run.assert_called_once()

        # Check configuration
        call_args = mock_app_class.call_args[0][0]
        self.assertEqual(call_args['hdf5_path'], self.hdf5_path)
        self.assertEqual(call_args['channel'], 'electrode02')
        self.assertEqual(call_args['filter_type'], 'lfp')

    def test_save_config(self):
        """Test saving configuration."""
        test_config_save_path = os.path.join(
            self.temp_dir, 'saved_config.json')

        test_args = [
            'raw_data_viewer.py',
            self.hdf5_path,
            '--channel', 'electrode01',
            '--window', '8.0',
            '--save-config', test_config_save_path
        ]

        with patch('sys.argv', test_args):
            raw_data_viewer.main()

        # Check that config was saved
        self.assertTrue(os.path.exists(test_config_save_path))

        # Load and verify saved config
        with open(test_config_save_path, 'r') as f:
            saved_config = json.load(f)

        self.assertEqual(saved_config['hdf5_path'], self.hdf5_path)
        self.assertEqual(saved_config['channel'], 'electrode01')
        self.assertEqual(saved_config['window_duration'], 8.0)


class TestRawDataViewerApp(unittest.TestCase):
    """Test cases for RawDataViewerApp class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.hdf5_path = os.path.join(self.temp_dir, 'test_data.h5')
        self.sampling_rate = 30000.0

        # Create minimal test HDF5 file
        self._create_minimal_hdf5()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def _create_minimal_hdf5(self):
        """Create a minimal test HDF5 file."""
        with tables.open_file(self.hdf5_path, 'w') as hf5:
            raw_group = hf5.create_group('/', 'raw')

            n_samples = int(self.sampling_rate * 5)  # 5 seconds
            data = np.random.randint(-1000, 1000, n_samples, dtype=np.int16)
            hf5.create_array(raw_group, 'electrode00', data)
            hf5.create_array('/', 'sampling_rate', self.sampling_rate)

    @patch('matplotlib.pyplot.show')
    def test_app_initialization(self, mock_show):
        """Test RawDataViewerApp initialization."""
        config = {
            'hdf5_path': self.hdf5_path,
            'window_duration': 3.0,
            'group': 'raw',
            'channel': 'electrode00'
        }

        app = raw_data_viewer.RawDataViewerApp(config)

        # Check that components were initialized
        self.assertIsNotNone(app.data_loader)
        self.assertIsNotNone(app.plotter)
        self.assertEqual(app.config, config)

        # Clean up
        app.cleanup()

    @patch('matplotlib.pyplot.show')
    def test_app_with_invalid_hdf5(self, mock_show):
        """Test RawDataViewerApp with invalid HDF5 file."""
        config = {
            'hdf5_path': '/nonexistent/file.h5',
            'window_duration': 3.0
        }

        with self.assertRaises(SystemExit):
            raw_data_viewer.RawDataViewerApp(config)

    @patch('matplotlib.pyplot.show')
    def test_app_filter_initialization(self, mock_show):
        """Test RawDataViewerApp filter initialization."""
        config = {
            'hdf5_path': self.hdf5_path,
            'filter_type': 'spike',
            'threshold': 150.0
        }

        app = raw_data_viewer.RawDataViewerApp(config)

        # Check filter was set
        self.assertEqual(app.plotter.signal_filter.filter_type, 'bandpass')
        self.assertEqual(app.plotter.signal_filter.low_freq, 300)
        self.assertEqual(app.plotter.signal_filter.high_freq, 3000)

        # Check threshold was set
        self.assertEqual(app.plotter.threshold_value, 150.0)
        self.assertTrue(app.plotter.show_threshold)

        # Clean up
        app.cleanup()

    @patch('matplotlib.pyplot.show')
    def test_enhanced_plotter_features(self, mock_show):
        """Test the new enhanced features of the interactive plotter."""
        loader = RawDataLoader(self.hdf5_path)
        plotter = InteractivePlotter(
            data_loader=loader,
            initial_channel='electrode00',
            initial_group='raw',
            window_duration=5.0
        )

        # Test enhanced properties
        self.assertEqual(plotter.data_conversion_factor, 0.6745)
        self.assertEqual(plotter.lowpass_freq, 3000.0)
        self.assertEqual(plotter.highpass_freq, 300.0)
        self.assertIsNone(plotter.manual_ylims)

        # Test frequency controls
        plotter._on_lowpass_change('2000')
        self.assertEqual(plotter.lowpass_freq, 2000.0)

        plotter._on_highpass_change('500')
        self.assertEqual(plotter.highpass_freq, 500.0)

        # Test manual y-limits
        plotter.ymin_box.set_val('-100')
        plotter.ymax_box.set_val('100')
        plotter._on_ylim_change('')  # Trigger the callback
        self.assertEqual(plotter.manual_ylims, (-100.0, 100.0))
        self.assertFalse(plotter.auto_scale)

        # Test auto-scale reset
        plotter._on_autoscale_click(None)
        self.assertTrue(plotter.auto_scale)
        self.assertIsNone(plotter.manual_ylims)

        # Test data conversion factor is applied in get_current_data
        data, _ = plotter.get_current_data()
        self.assertIsInstance(data, np.ndarray)

        # Verify conversion factor is applied by checking if data values are reasonable for microvolts
        # (original data is in int16 range, converted should be much smaller)
        # Should be in microvolt range
        self.assertTrue(np.abs(data).max() < 1000)

        # Clean up
        plotter.close()


if __name__ == '__main__':
    # Set up test environment
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    unittest.main()
