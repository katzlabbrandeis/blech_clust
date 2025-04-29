import os
import sys
import pytest
import numpy as np
from unittest.mock import patch, MagicMock, mock_open

# Add the parent directory to sys.path to import blech_clust modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from blech_clust import (
    process_file_info,
    process_data,
    perform_quality_assurance,
    plot_digital_inputs
)

class TestProcessFileInfo:
    """Tests for the process_file_info function"""
    
    @patch('numpy.fromfile')
    @patch('builtins.open', new_callable=mock_open)
    @patch('blech_clust.read_header')
    def test_process_one_file_per_channel(self, mock_read_header, mock_open_file, mock_fromfile):
        """Test processing file info for 'one file per channel' type"""
        # Setup
        dir_name = 'test_dir'
        file_type = 'one file per channel'
        file_lists = {
            'one file per channel': {
                'electrodes': ['amp-001.dat', 'amp-002.dat']
            }
        }
        info_dict = {
            'ports': ['A', 'B']
        }
        
        # Mock numpy.fromfile to return sample data
        mock_fromfile.side_effect = [
            np.array([1.0, 2.0, 30000.0]),  # info.rhd
            np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # time.dat
        ]
        
        # Call the function
        result = process_file_info(dir_name, file_type, file_lists, info_dict)
        
        # Unpack the result
        electrodes_list, sampling_rate, num_recorded_samples, rhd_file_list = result
        
        # Assertions
        assert electrodes_list == ['amp-001.dat', 'amp-002.dat']
        assert sampling_rate == 30000
        assert num_recorded_samples == 5
        assert rhd_file_list is None
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('blech_clust.read_header')
    def test_process_traditional(self, mock_read_header, mock_open_file):
        """Test processing file info for 'traditional' type"""
        # Setup
        dir_name = 'test_dir'
        file_type = 'traditional'
        file_lists = {
            'traditional': {
                'rhd': ['file1.rhd', 'file2.rhd']
            }
        }
        info_dict = {}
        
        # Mock read_header to return sample header
        mock_header = {
            'amplifier_channels': [
                {'port_prefix': 'A', 'native_channel_name': 'A-01'},
                {'port_prefix': 'A', 'native_channel_name': 'A-02'},
                {'port_prefix': 'B', 'native_channel_name': 'B-01'}
            ],
            'sample_rate': 30000
        }
        mock_read_header.return_value = mock_header
        
        # Call the function
        result = process_file_info(dir_name, file_type, file_lists, info_dict)
        
        # Unpack the result
        electrodes_list, sampling_rate, num_recorded_samples, rhd_file_list = result
        
        # Assertions
        assert electrodes_list is None
        assert sampling_rate == 30000
        assert num_recorded_samples is None
        assert rhd_file_list == ['file1.rhd', 'file2.rhd']


class TestProcessData:
    """Tests for the process_data function"""
    
    @patch('blech_clust.read_file.read_electrode_channels')
    @patch('blech_clust.read_file.read_emg_channels')
    def test_process_one_file_per_channel(self, mock_read_emg, mock_read_electrode):
        """Test processing data for 'one file per channel' type"""
        # Setup
        file_type = 'one file per channel'
        reload_data_str = 'y'
        hdf5_name = 'test.h5'
        electrode_layout_frame = MagicMock()
        electrodes_list = ['amp-001.dat', 'amp-002.dat']
        num_recorded_samples = 1000
        emg_channels = [101, 102]
        rhd_file_list = None
        
        # Call the function
        process_data(file_type, reload_data_str, hdf5_name, electrode_layout_frame,
                    electrodes_list, num_recorded_samples, emg_channels, rhd_file_list)
        
        # Assertions
        mock_read_electrode.assert_called_once_with(hdf5_name, electrode_layout_frame)
        mock_read_emg.assert_called_once_with(hdf5_name, electrode_layout_frame)
    
    @patch('blech_clust.read_file.read_electrode_emg_channels_single_file')
    def test_process_one_file_per_signal_type(self, mock_read_electrode_emg):
        """Test processing data for 'one file per signal type'"""
        # Setup
        file_type = 'one file per signal type'
        reload_data_str = 'y'
        hdf5_name = 'test.h5'
        electrode_layout_frame = MagicMock()
        electrodes_list = ['amplifier.dat']
        num_recorded_samples = 1000
        emg_channels = [101, 102]
        rhd_file_list = None
        
        # Call the function
        process_data(file_type, reload_data_str, hdf5_name, electrode_layout_frame,
                    electrodes_list, num_recorded_samples, emg_channels, rhd_file_list)
        
        # Assertions
        mock_read_electrode_emg.assert_called_once_with(
            hdf5_name, electrode_layout_frame, electrodes_list, 
            num_recorded_samples, emg_channels
        )
    
    @patch('blech_clust.read_file.read_traditional_intan')
    def test_process_traditional(self, mock_read_traditional):
        """Test processing data for 'traditional' type"""
        # Setup
        file_type = 'traditional'
        reload_data_str = 'y'
        hdf5_name = 'test.h5'
        electrode_layout_frame = MagicMock()
        electrodes_list = None
        num_recorded_samples = None
        emg_channels = [101, 102]
        rhd_file_list = ['file1.rhd', 'file2.rhd']
        
        # Call the function
        process_data(file_type, reload_data_str, hdf5_name, electrode_layout_frame,
                    electrodes_list, num_recorded_samples, emg_channels, rhd_file_list)
        
        # Assertions
        mock_read_traditional.assert_called_once_with(
            hdf5_name, rhd_file_list, electrode_layout_frame
        )
    
    def test_no_reload(self):
        """Test when reload_data_str is 'n'"""
        # Setup
        file_type = 'one file per channel'
        reload_data_str = 'n'
        hdf5_name = 'test.h5'
        electrode_layout_frame = MagicMock()
        electrodes_list = ['amp-001.dat', 'amp-002.dat']
        num_recorded_samples = 1000
        emg_channels = [101, 102]
        rhd_file_list = None
        
        # We're not patching any functions because none should be called
        
        # Call the function
        process_data(file_type, reload_data_str, hdf5_name, electrode_layout_frame,
                    electrodes_list, num_recorded_samples, emg_channels, rhd_file_list)
        
        # No assertions needed - we're just verifying no exceptions are raised


class TestPerformQualityAssurance:
    """Tests for the perform_quality_assurance function"""
    
    @patch('blech_clust.channel_corr.get_all_channels')
    @patch('blech_clust.channel_corr.intra_corr')
    @patch('blech_clust.channel_corr.gen_corr_output')
    @patch('numpy.save')
    @patch('os.path.exists')
    @patch('os.mkdir')
    @patch('shutil.rmtree')
    @patch('blech_clust.plot_channels')
    def test_perform_quality_assurance(self, mock_plot_channels, mock_rmtree, mock_mkdir, 
                                     mock_exists, mock_np_save, mock_gen_corr, 
                                     mock_intra_corr, mock_get_channels):
        """Test performing quality assurance"""
        # Setup
        hdf5_name = 'test.h5'
        all_params_dict = {
            'qa_params': {
                'n_corr_samples': 1000,
                'bridged_channel_threshold': 0.9
            }
        }
        dir_name = 'test_dir'
        file_type = 'one file per channel'
        
        # Mock return values
        mock_exists.return_value = False
        mock_get_channels.return_value = (np.random.rand(10, 1000), ['ch1', 'ch2'])
        mock_intra_corr.return_value = np.random.rand(10, 10)
        
        # Call the function
        result = perform_quality_assurance(hdf5_name, all_params_dict, dir_name, file_type)
        
        # Assertions
        assert result == 'test_dir/QA_output'
        mock_get_channels.assert_called_once()
        mock_intra_corr.assert_called_once()
        mock_gen_corr.assert_called_once()
        mock_np_save.assert_called_once()
        mock_mkdir.assert_called_once()
        mock_plot_channels.assert_called_once()
    
    @patch('blech_clust.channel_corr.get_all_channels')
    @patch('blech_clust.channel_corr.intra_corr')
    @patch('blech_clust.channel_corr.gen_corr_output')
    @patch('numpy.save')
    @patch('os.path.exists')
    @patch('os.mkdir')
    @patch('shutil.rmtree')
    def test_perform_quality_assurance_traditional(self, mock_rmtree, mock_mkdir, 
                                                 mock_exists, mock_np_save, 
                                                 mock_gen_corr, mock_intra_corr, 
                                                 mock_get_channels):
        """Test performing quality assurance with traditional file type"""
        # Setup
        hdf5_name = 'test.h5'
        all_params_dict = {
            'qa_params': {
                'n_corr_samples': 1000,
                'bridged_channel_threshold': 0.9
            }
        }
        dir_name = 'test_dir'
        file_type = 'traditional'
        
        # Mock return values
        mock_exists.return_value = True
        mock_get_channels.return_value = (np.random.rand(10, 1000), ['ch1', 'ch2'])
        mock_intra_corr.return_value = np.random.rand(10, 10)
        
        # Call the function
        result = perform_quality_assurance(hdf5_name, all_params_dict, dir_name, file_type)
        
        # Assertions
        assert result == 'test_dir/QA_output'
        mock_get_channels.assert_called_once()
        mock_intra_corr.assert_called_once()
        mock_gen_corr.assert_called_once()
        mock_np_save.assert_called_once()
        mock_rmtree.assert_called_once()
        mock_mkdir.assert_called_once()


class TestPlotDigitalInputs:
    """Tests for the plot_digital_inputs function"""
    
    @patch('matplotlib.pyplot.scatter')
    @patch('matplotlib.pyplot.axvline')
    @patch('matplotlib.pyplot.yticks')
    @patch('matplotlib.pyplot.title')
    @patch('matplotlib.pyplot.xlabel')
    @patch('matplotlib.pyplot.ylabel')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('ast.literal_eval')
    def test_plot_digital_inputs(self, mock_literal_eval, mock_close, mock_savefig, 
                               mock_ylabel, mock_xlabel, mock_title, mock_yticks, 
                               mock_axvline, mock_scatter):
        """Test plotting digital inputs"""
        # Setup
        this_dig_handler = MagicMock()
        this_dig_handler.dig_in_frame.pulse_times.values = [
            '[(100, 200), (300, 400)]',
            '[(150, 250), (350, 450)]'
        ]
        
        info_dict = {
            'taste_params': {
                'dig_in_nums': [0, 1],
                'tastes': ['taste1', 'taste2']
            },
            'laser_params': {
                'dig_in_nums': [2]
            }
        }
        
        sampling_rate = 1000
        qa_out_path = 'test_dir/QA_output'
        
        # Mock literal_eval to return lists of tuples
        mock_literal_eval.side_effect = [
            [(100, 200), (300, 400)],
            [(150, 250), (350, 450)]
        ]
        
        # Call the function
        plot_digital_inputs(this_dig_handler, info_dict, sampling_rate, qa_out_path)
        
        # Assertions
        assert mock_scatter.call_count == 2
        mock_yticks.assert_called_once()
        mock_title.assert_called_once()
        mock_xlabel.assert_called_once()
        mock_ylabel.assert_called_once()
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])
