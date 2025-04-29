import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Add the parent directory to sys.path to import blech_clust modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from blech_clust import main

class TestMain:
    """Tests for the main function"""
    
    @patch('blech_clust.parse_arguments')
    @patch('blech_clust.check_template_file')
    @patch('blech_clust.imp_metadata')
    @patch('blech_clust.setup_pipeline_check')
    @patch('os.chdir')
    @patch('blech_clust.HDF5Handler')
    @patch('blech_clust.create_directories')
    @patch('blech_clust.get_file_lists')
    @patch('blech_clust.read_file.DigInHandler')
    @patch('blech_clust.process_file_info')
    @patch('blech_clust.get_electrode_info')
    @patch('glob.glob')
    @patch('pandas.read_csv')
    @patch('blech_clust.process_data')
    @patch('blech_clust.create_params_file')
    @patch('blech_clust.perform_quality_assurance')
    @patch('blech_clust.plot_digital_inputs')
    @patch('blech_clust.generate_processing_scripts')
    def test_main_function_flow(self, mock_generate_scripts, mock_plot_digins, 
                               mock_qa, mock_create_params, mock_process_data, 
                               mock_read_csv, mock_glob, mock_get_electrode_info, 
                               mock_process_file_info, mock_dig_handler, 
                               mock_get_file_lists, mock_create_dirs, 
                               mock_hdf5_handler, mock_chdir, mock_pipeline_check, 
                               mock_metadata, mock_check_template, mock_parse_args):
        """Test the overall flow of the main function"""
        # Setup all the mocks
        args = MagicMock()
        args.dir_name = 'test_dir'
        args.force_run = False
        mock_parse_args.return_value = args
        
        mock_check_template.return_value = '/path/to/template.json'
        
        metadata = MagicMock()
        metadata.dir_name = 'test_dir'
        metadata.info_dict = {'file_type': 'one file per channel'}
        metadata.file_list = ['file1.dat', 'file2.dat']
        mock_metadata.return_value = metadata
        
        pipeline_check = MagicMock()
        mock_pipeline_check.return_value = pipeline_check
        
        hdf5_handler = MagicMock()
        hdf5_handler.initialize_groups.return_value = (True, 'y')
        hdf5_handler.hdf5_name = 'test.h5'
        mock_hdf5_handler.return_value = hdf5_handler
        
        mock_create_dirs.return_value = True
        
        mock_get_file_lists.return_value = {'one file per channel': {'electrodes': ['amp-001.dat']}}
        
        dig_handler = MagicMock()
        mock_dig_handler.return_value = dig_handler
        
        mock_process_file_info.return_value = (['amp-001.dat'], 30000, 1000, None)
        
        mock_get_electrode_info.return_value = ([1, 2, 3], [101, 102])
        
        mock_glob.return_value = ['test_dir/test_layout.csv']
        
        mock_read_csv.return_value = MagicMock()
        
        mock_create_params.return_value = {'max_parallel_cpu': 4}
        
        mock_qa.return_value = 'test_dir/QA_output'
        
        # Call the main function
        main()
        
        # Verify the expected flow of function calls
        mock_parse_args.assert_called_once()
        mock_check_template.assert_called_once()
        mock_metadata.assert_called_once()
        mock_pipeline_check.assert_called_once()
        mock_chdir.assert_called_once_with('test_dir')
        mock_hdf5_handler.assert_called_once()
        mock_create_dirs.assert_called_once()
        mock_get_file_lists.assert_called_once()
        mock_dig_handler.assert_called_once()
        mock_process_file_info.assert_called_once()
        mock_get_electrode_info.assert_called_once()
        mock_glob.assert_called_once()
        mock_read_csv.assert_called_once()
        mock_process_data.assert_called_once()
        mock_create_params.assert_called_once()
        mock_qa.assert_called_once()
        mock_plot_digins.assert_called_once()
        mock_generate_scripts.assert_called_once()
        pipeline_check.write_to_log.assert_called_with(mock_pipeline_check.return_value.check_previous.return_value, 'completed')

if __name__ == '__main__':
    pytest.main(['-xvs', __file__])
