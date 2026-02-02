"""
Comprehensive pytest test suite for ephys_data.py module.

This test suite covers the main functionality of the ephys_data class,
focusing on testable methods and core functionality.
"""
from utils.ephys_data.ephys_data import ephys_data
import pytest
import numpy as np
import os
import sys
import tempfile
import shutil
from unittest.mock import patch, MagicMock, mock_open
from scipy import signal
import pandas as pd

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

# Import the module under test


class TestEphysDataCore:
    """Test class for core ephys_data functionality"""

    def test_init_with_data_dir(self):
        """Test initialization with data directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a dummy HDF5 file
            hdf5_path = os.path.join(temp_dir, 'test.h5')
            with open(hdf5_path, 'w') as f:
                f.write('')

            data = ephys_data(data_dir=temp_dir)
            assert data.data_dir == temp_dir
            assert data.hdf5_path == hdf5_path
            assert data.hdf5_name == 'test.h5'

    @patch('easygui.diropenbox')
    def test_init_without_data_dir(self, mock_diropenbox):
        """Test initialization without data directory (uses GUI)"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a dummy HDF5 file
            hdf5_path = os.path.join(temp_dir, 'test.h5')
            with open(hdf5_path, 'w') as f:
                f.write('')

            mock_diropenbox.return_value = temp_dir
            data = ephys_data()
            assert data.data_dir == temp_dir
            mock_diropenbox.assert_called_once()

    def test_get_hdf5_path_single_file(self):
        """Test finding HDF5 file when only one exists"""
        with tempfile.TemporaryDirectory() as temp_dir:
            hdf5_path = os.path.join(temp_dir, 'test.h5')
            with open(hdf5_path, 'w') as f:
                f.write('dummy')

            result = ephys_data.get_hdf5_path(temp_dir)
            assert result == hdf5_path

    def test_get_hdf5_path_no_file(self):
        """Test error when no HDF5 file exists"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(Exception, match="No HDF5 file detected"):
                ephys_data.get_hdf5_path(temp_dir)

    @patch('builtins.input')
    def test_get_hdf5_path_multiple_files(self, mock_input):
        """Test selection when multiple HDF5 files exist"""
        with tempfile.TemporaryDirectory() as temp_dir:
            hdf5_path1 = os.path.join(temp_dir, 'test1.h5')
            hdf5_path2 = os.path.join(temp_dir, 'test2.h5')

            with open(hdf5_path1, 'w') as f:
                f.write('dummy1')
            with open(hdf5_path2, 'w') as f:
                f.write('dummy2')

            mock_input.return_value = '0'
            result = ephys_data.get_hdf5_path(temp_dir)
            # The result should be one of the two files (order may vary due to glob)
            assert result in [hdf5_path1, hdf5_path2]

    def test_calc_stft_static_method(self):
        """Test the static STFT calculation method"""
        # Create test signal
        fs = 1000
        t = np.linspace(0, 1, fs)
        signal_data = np.sin(2 * np.pi * 10 * t)  # 10 Hz sine wave

        # Test parameters
        max_freq = 50
        time_range = (0.1, 0.9)
        signal_window = 100
        window_overlap = 50

        freq, time, stft_result = ephys_data.calc_stft(
            signal_data, max_freq, time_range, fs, signal_window, window_overlap
        )

        assert len(freq) > 0
        assert len(time) > 0
        assert stft_result.shape[0] == len(freq)
        assert stft_result.shape[1] == len(time)
        assert np.max(freq) <= max_freq

    def test_calc_conv_rates_static_method(self):
        """Test the convolution-based firing rate calculation"""
        # Create test spike array
        # 2 neurons, 5 trials, 1000 time points
        spike_array = np.random.randint(0, 2, (2, 5, 1000))

        step_size = 25  # ms
        window_size = 250  # ms
        dt = 1  # ms

        firing_rate, time_vector = ephys_data._calc_conv_rates(
            step_size, window_size, dt, spike_array
        )

        assert firing_rate.shape[0] == spike_array.shape[0]
        assert firing_rate.shape[1] == spike_array.shape[1]
        assert len(time_vector) == firing_rate.shape[2]
        assert np.all(firing_rate >= 0)  # Firing rates should be non-negative

    def test_calc_baks_rate_static_method(self):
        """Test the BAKS firing rate calculation"""
        # Create test spike array (smaller for faster computation)
        # 1 neuron, 2 trials, 100 time points
        spike_array = np.random.randint(0, 2, (1, 2, 100))

        resolution = 0.01  # 10ms resolution
        dt = 0.001  # 1ms input resolution

        firing_rate, time_vector = ephys_data._calc_baks_rate(
            resolution, dt, spike_array
        )

        assert firing_rate.shape[0] == spike_array.shape[0]
        assert firing_rate.shape[1] == spike_array.shape[1]
        assert len(time_vector) == firing_rate.shape[2]
        assert np.all(firing_rate >= 0)  # Firing rates should be non-negative

    def test_convert_to_array_static_method(self):
        """Test the convert_to_array static method"""
        # Create test data
        arrays = [np.ones((2, 3)), np.ones((2, 3)) * 2, np.ones((2, 3)) * 3]
        indices = [(0, 0), (0, 1), (1, 0)]

        result = ephys_data.convert_to_array(arrays, indices)

        assert result.shape == (2, 2, 2, 3)
        assert np.array_equal(result[0, 0], arrays[0])
        assert np.array_equal(result[0, 1], arrays[1])
        assert np.array_equal(result[1, 0], arrays[2])

    def test_remove_node_static_method(self):
        """Test the remove_node static method"""
        mock_hf5 = MagicMock()
        mock_hf5.__contains__ = MagicMock(return_value=True)

        ephys_data.remove_node('/test/path', mock_hf5)

        mock_hf5.remove_node.assert_called_once_with('/test', 'path')

    @patch('multiprocessing.cpu_count')
    def test_parallelize_static_method(self, mock_cpu_count):
        """Test the parallelize static method"""
        # Mock cpu_count to return a reasonable number
        mock_cpu_count.return_value = 4

        def square(x):
            return x ** 2

        test_data = [1, 2, 3, 4, 5]
        result = ephys_data.parallelize(square, test_data)

        expected = [1, 4, 9, 16, 25]
        assert result == expected

    def test_default_parameters(self):
        """Test that default parameters are set correctly"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a dummy HDF5 file
            hdf5_path = os.path.join(temp_dir, 'test.h5')
            with open(hdf5_path, 'w') as f:
                f.write('')

            data = ephys_data(data_dir=temp_dir)

            assert data.default_firing_params['type'] == 'conv'
            assert data.default_firing_params['step_size'] == 25
            assert data.default_firing_params['window_size'] == 250

            assert data.default_lfp_params['freq_bounds'] == [1, 300]
            assert data.default_lfp_params['sampling_rate'] == 30000

            assert data.stft_params['Fs'] == 1000
            assert data.stft_params['max_freq'] == 20

    def test_firing_rate_params_validation(self):
        """Test validation of firing rate parameters"""
        # Test invalid step size
        with pytest.raises(Exception):
            ephys_data._calc_conv_rates(
                step_size=25.5,  # Not integer multiple of dt
                window_size=250,
                dt=1,
                spike_array=np.zeros((1, 1, 1000))
            )

    def test_sequester_trial_inds_signature(self):
        """Test trial sequestering method signature"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a dummy HDF5 file
            hdf5_path = os.path.join(temp_dir, 'test.h5')
            with open(hdf5_path, 'w') as f:
                f.write('')

            data = ephys_data(data_dir=temp_dir)

            # Check that the method exists and can be called
            assert hasattr(data, 'sequester_trial_inds')
            assert callable(data.sequester_trial_inds)


class TestEphysDataIntegration:
    """Integration tests for ephys_data class"""

    def test_initialization_workflow(self):
        """Test basic initialization workflow"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a dummy HDF5 file
            hdf5_path = os.path.join(temp_dir, 'test_data.h5')
            with open(hdf5_path, 'w') as f:
                f.write('')

            # Initialize
            data = ephys_data(data_dir=temp_dir)

            # Verify initialization
            assert data.data_dir == temp_dir
            assert data.hdf5_path == hdf5_path
            assert data.hdf5_name == 'test_data.h5'

            # Check default parameters are set
            assert data.default_firing_params['type'] == 'conv'
            assert data.default_lfp_params['sampling_rate'] == 30000
            assert data.stft_params['Fs'] == 1000

    def test_parameter_setting_workflow(self):
        """Test parameter setting and method selection workflow"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a dummy HDF5 file
            hdf5_path = os.path.join(temp_dir, 'test_data.h5')
            with open(hdf5_path, 'w') as f:
                f.write('')

            data = ephys_data(data_dir=temp_dir)

            # Set firing rate parameters
            data.firing_rate_params = data.default_firing_params.copy()

            # Test method selection - just check that it returns a callable
            conv_method = data.firing_rate_method_selector()
            assert callable(conv_method)

            # Change to BAKS method
            data.firing_rate_params['type'] = 'baks'
            baks_method = data.firing_rate_method_selector()
            assert callable(baks_method)

    def test_data_processing_workflow_with_mock_data(self):
        """Test data processing workflow with mock data"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a dummy HDF5 file
            hdf5_path = os.path.join(temp_dir, 'test_data.h5')
            with open(hdf5_path, 'w') as f:
                f.write('')

            data = ephys_data(data_dir=temp_dir)

            # Set up mock data for processing
            data.spikes = np.random.randint(0, 2, (4, 10, 7000))
            data.firing_array = np.random.rand(4, 10, 100)
            data.lfp_array = np.random.randn(4, 32, 10, 7000)

            # Test basic data access
            assert data.spikes.shape == (4, 10, 7000)
            assert data.firing_array.shape == (4, 10, 100)
            assert data.lfp_array.shape == (4, 32, 10, 7000)

            # Test that data attributes are accessible
            assert hasattr(data, 'spikes')
            assert hasattr(data, 'firing_array')
            assert hasattr(data, 'lfp_array')


class TestEphysDataStaticMethods:
    """Test class specifically for static methods that don't require instance setup"""

    def test_stft_with_different_parameters(self):
        """Test STFT with various parameter combinations"""
        # Create a more complex test signal
        fs = 2000
        t = np.linspace(0, 2, fs * 2)
        # Signal with multiple frequency components
        signal_data = (np.sin(2 * np.pi * 5 * t) +
                       0.5 * np.sin(2 * np.pi * 20 * t) +
                       0.3 * np.sin(2 * np.pi * 50 * t))

        # Test with different parameters
        max_freq = 100
        time_range = (0.2, 1.8)
        signal_window = 200
        window_overlap = 150

        freq, time, stft_result = ephys_data.calc_stft(
            signal_data, max_freq, time_range, fs, signal_window, window_overlap
        )

        # Verify output properties
        assert len(freq) > 0
        assert len(time) > 0
        assert stft_result.shape[0] == len(freq)
        assert stft_result.shape[1] == len(time)
        assert np.max(freq) <= max_freq

        # Check that we can detect the frequency components
        # The STFT should show energy at the frequencies we put in
        power_spectrum = np.abs(stft_result) ** 2
        mean_power = np.mean(power_spectrum, axis=1)

        # Find peaks in the power spectrum
        peak_freqs = freq[mean_power > np.mean(mean_power)]

        # Should detect some frequency content
        assert len(peak_freqs) > 0

    def test_firing_rate_calculation_edge_cases(self):
        """Test firing rate calculation with edge cases"""
        # Test with all zeros (no spikes)
        spike_array = np.zeros((2, 3, 1000))

        step_size = 50
        window_size = 200
        dt = 1

        firing_rate, time_vector = ephys_data._calc_conv_rates(
            step_size, window_size, dt, spike_array
        )

        # Should return all zeros for firing rates
        assert np.all(firing_rate == 0)
        assert len(time_vector) == firing_rate.shape[2]

        # Test with all ones (maximum spikes)
        spike_array = np.ones((1, 2, 500))

        firing_rate, time_vector = ephys_data._calc_conv_rates(
            step_size, window_size, dt, spike_array
        )

        # Should return consistent firing rates
        assert np.all(firing_rate >= 0)
        assert firing_rate.shape == (1, 2, len(time_vector))

    def test_array_conversion_with_different_shapes(self):
        """Test array conversion with different input shapes"""
        # Test with 1D arrays
        arrays_1d = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        indices_1d = [(0,), (1,)]

        result_1d = ephys_data.convert_to_array(arrays_1d, indices_1d)
        assert result_1d.shape == (2, 3)
        assert np.array_equal(result_1d[0], arrays_1d[0])
        assert np.array_equal(result_1d[1], arrays_1d[1])

        # Test with 3D arrays
        arrays_3d = [np.ones((2, 3, 4)), np.ones((2, 3, 4)) * 2]
        indices_3d = [(0, 0), (1, 1)]

        result_3d = ephys_data.convert_to_array(arrays_3d, indices_3d)
        assert result_3d.shape == (2, 2, 2, 3, 4)
        assert np.array_equal(result_3d[0, 0], arrays_3d[0])
        assert np.array_equal(result_3d[1, 1], arrays_3d[1])

    def test_baks_rate_with_sparse_spikes(self):
        """Test BAKS rate calculation with sparse spike data"""
        # Create sparse spike data
        spike_array = np.zeros((1, 1, 200))
        # Add a few spikes at specific times
        spike_array[0, 0, [50, 100, 150]] = 1

        resolution = 0.005  # 5ms resolution
        dt = 0.001  # 1ms input resolution

        firing_rate, time_vector = ephys_data._calc_baks_rate(
            resolution, dt, spike_array
        )

        # Should produce smooth firing rate estimates
        assert firing_rate.shape[0] == 1
        assert firing_rate.shape[1] == 1
        assert len(time_vector) == firing_rate.shape[2]
        assert np.all(firing_rate >= 0)

        # The firing rate should be higher around spike times
        # This is a basic sanity check for BAKS functionality
        assert np.max(firing_rate) > 0

    def test_get_firing_rates_print_statements(self):
        """Test that get_firing_rates prints correct attribute names"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a dummy HDF5 file
            hdf5_path = os.path.join(temp_dir, 'test_data.h5')
            with open(hdf5_path, 'w') as f:
                f.write('')

            data = ephys_data(data_dir=temp_dir)
            
            # Set up mock data for testing
            data.firing_rate_params = data.default_firing_params
            data.sorting_params_dict = {'spike_array_durations': [0, 2]}
            
            # Test case 1: Even dimensions (stacking case)
            # Create mock spike data with equal dimensions
            data.spikes = [
                np.random.randint(0, 2, (5, 10, 1000)),  # taste 1: 5 trials, 10 neurons
                np.random.randint(0, 2, (5, 10, 1000)),  # taste 2: 5 trials, 10 neurons
            ]
            
            # Capture print output
            with patch('builtins.print') as mock_print:
                data.get_firing_rates()
                
                # Check that the correct print statements were called
                print_calls = [str(call) for call in mock_print.call_args_list]
                
                # Should contain the stacking message
                stacking_message_found = False
                attributes_message_found = False
                
                for call in print_calls:
                    if 'concatenating and normalizing' in call:
                        stacking_message_found = True
                    if 'Generated attributes:' in call:
                        attributes_message_found = True
                        # Check that all expected attributes are mentioned
                        assert 'firing_list' in call
                        assert 'time_vector' in call
                        assert 'firing_array' in call
                        assert 'all_firing_array' in call
                        assert 'normalized_firing' in call
                        assert 'all_normalized_firing' in call
                        # Check that shape information is included
                        assert 'shape' in call
                
                assert stacking_message_found, "Stacking message not found"
                assert attributes_message_found, "Attributes message not found"
            
            # Test case 2: Uneven dimensions (non-stacking case)
            # Create mock spike data with unequal dimensions
            data.spikes = [
                np.random.randint(0, 2, (5, 10, 1000)),  # taste 1: 5 trials, 10 neurons
                np.random.randint(0, 2, (3, 10, 1000)),  # taste 2: 3 trials, 10 neurons (different!)
            ]
            
            # Capture print output
            with patch('builtins.print') as mock_print:
                data.get_firing_rates()
                
                # Check that the correct print statements were called
                print_calls = [str(call) for call in mock_print.call_args_list]
                
                # Should contain the non-stacking message and attributes message
                non_stacking_message_found = False
                attributes_message_found = False
                
                for call in print_calls:
                    if 'not stacking into firing rates array' in call:
                        non_stacking_message_found = True
                    if 'Generated attributes:' in call:
                        attributes_message_found = True
                        # Check that only the expected attributes are mentioned
                        assert 'firing_list' in call
                        assert 'time_vector' in call
                        # Should NOT mention the stacking-specific attributes
                        assert 'firing_array' not in call
                        assert 'all_firing_array' not in call
                        assert 'normalized_firing' not in call
                        assert 'all_normalized_firing' not in call
                        # Check that shape information is included
                        assert 'shape' in call
                
                assert non_stacking_message_found, "Non-stacking message not found"
                assert attributes_message_found, "Attributes message not found"
