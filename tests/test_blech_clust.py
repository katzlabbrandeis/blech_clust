from blech_clust import (
    parse_arguments,
    HDF5Handler,
    create_directories,
    check_template_file,
    get_file_lists,
    process_file_info,
    get_electrode_info,
    create_params_file,
    generate_processing_scripts
)
import os
import sys
import pytest
import numpy as np
import pandas as pd
import tables
import glob
import json
import shutil
from unittest.mock import patch, MagicMock, mock_open

# Add the parent directory to sys.path to import blech_clust modules
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

# Import functions and classes from blech_clust.py


class TestParseArguments:
    """Tests for the parse_arguments function"""

    def test_parse_arguments_with_dir_name(self):
        """Test parsing arguments with only directory name"""
        with patch('sys.argv', ['blech_clust.py', 'test_dir']):
            args = parse_arguments()
            assert args.dir_name == 'test_dir'
            assert args.force_run is False

    def test_parse_arguments_with_force_run(self):
        """Test parsing arguments with force_run flag"""
        with patch('sys.argv', ['blech_clust.py', 'test_dir', '--force_run']):
            args = parse_arguments()
            assert args.dir_name == 'test_dir'
            assert args.force_run is True


class TestHDF5Handler:
    """Tests for the HDF5Handler class"""

    @patch('glob.glob')
    @patch('tables.open_file')
    def test_init_with_existing_hdf5(self, mock_open_file, mock_glob):
        """Test initialization with existing HDF5 file"""
        mock_glob.return_value = ['existing.h5']
        mock_file = MagicMock()
        mock_open_file.return_value = mock_file

        handler = HDF5Handler('test_dir', force_run=False)

        assert handler.dir_name == 'test_dir'
        assert handler.force_run is False
        assert handler.hdf5_name == 'existing.h5'
        mock_open_file.assert_called_once_with('existing.h5', 'r+')

    @patch('glob.glob')
    @patch('tables.open_file')
    @patch('os.path.dirname')
    def test_init_without_existing_hdf5(self, mock_dirname, mock_open_file, mock_glob):
        """Test initialization without existing HDF5 file"""
        mock_glob.return_value = []
        mock_dirname.return_value = '/path/to/test_folder'
        mock_file = MagicMock()
        mock_open_file.return_value = mock_file

        handler = HDF5Handler('test_dir/', force_run=False)

        assert handler.dir_name == 'test_dir/'
        assert handler.hdf5_name == 'test_folder.h5'
        mock_open_file.assert_called_once_with(
            'test_folder.h5', 'w', title='5')

    @patch('utils.blech_utils.entry_checker')
    def test_initialize_groups_with_existing_groups(self, mock_entry_checker):
        """Test initializing groups when groups already exist"""
        # Setup
        mock_hf5 = MagicMock()
        mock_hf5.__contains__.return_value = True

        handler = HDF5Handler('test_dir', force_run=False)
        handler.hf5 = mock_hf5

        # Mock entry_checker to simulate user choosing to reload data
        mock_entry_checker.return_value = ('y', True)

        # Call the method
        continue_bool, reload_data_str = handler.initialize_groups()

        # Assertions
        assert continue_bool is True
        assert reload_data_str == 'y'
        assert mock_hf5.remove_node.call_count > 0
        assert mock_hf5.create_group.call_count > 0

    @patch('utils.blech_utils.entry_checker')
    def test_initialize_groups_with_force_run(self, mock_entry_checker):
        """Test initializing groups with force_run=True"""
        # Setup
        mock_hf5 = MagicMock()
        mock_hf5.__contains__.return_value = True

        handler = HDF5Handler('test_dir', force_run=True)
        handler.hf5 = mock_hf5

        # Call the method
        continue_bool, reload_data_str = handler.initialize_groups()

        # Assertions
        assert continue_bool is True
        assert reload_data_str == 'y'
        # No entry_checker call should happen with force_run=True
        mock_entry_checker.assert_not_called()


class TestCreateDirectories:
    """Tests for the create_directories function"""

    @patch('os.path.exists')
    @patch('shutil.rmtree')
    @patch('os.makedirs')
    @patch('utils.blech_utils.entry_checker')
    def test_create_directories_with_existing_dirs(self, mock_entry_checker,
                                                   mock_makedirs, mock_rmtree, mock_exists):
        """Test creating directories when they already exist"""
        # Setup
        mock_exists.return_value = True
        mock_entry_checker.return_value = ('y', True)
        dir_list = ['dir1', 'dir2']

        # Call the function
        result = create_directories(dir_list, force_run=False)

        # Assertions
        assert result is True
        mock_entry_checker.assert_called_once()
        assert mock_rmtree.call_count == 2
        assert mock_makedirs.call_count == 2

    @patch('os.path.exists')
    @patch('shutil.rmtree')
    @patch('os.makedirs')
    @patch('utils.blech_utils.entry_checker')
    def test_create_directories_with_force_run(self, mock_entry_checker,
                                               mock_makedirs, mock_rmtree, mock_exists):
        """Test creating directories with force_run=True"""
        # Setup
        mock_exists.return_value = True
        dir_list = ['dir1', 'dir2']

        # Call the function
        result = create_directories(dir_list, force_run=True)

        # Assertions
        assert result is True
        mock_entry_checker.assert_not_called()
        assert mock_rmtree.call_count == 2
        assert mock_makedirs.call_count == 2

    @patch('os.path.exists')
    @patch('shutil.rmtree')
    @patch('os.makedirs')
    @patch('utils.blech_utils.entry_checker')
    def test_create_directories_user_declines(self, mock_entry_checker,
                                              mock_makedirs, mock_rmtree, mock_exists):
        """Test when user declines to overwrite existing directories"""
        # Setup
        mock_exists.return_value = True
        mock_entry_checker.return_value = ('n', False)
        dir_list = ['dir1', 'dir2']

        # Call the function
        result = create_directories(dir_list, force_run=False)

        # Assertions
        assert result is False
        mock_entry_checker.assert_called_once()
        mock_rmtree.assert_not_called()
        mock_makedirs.assert_not_called()


class TestCheckTemplateFile:
    """Tests for the check_template_file function"""

    @patch('os.path.exists')
    def test_template_file_exists(self, mock_exists):
        """Test when template file exists"""
        mock_exists.return_value = True
        blech_clust_dir = '/path/to/blech_clust'

        result = check_template_file(blech_clust_dir)

        expected_path = '/path/to/blech_clust/params/sorting_params_template.json'
        assert result == expected_path

    @patch('os.path.exists')
    def test_template_file_missing(self, mock_exists):
        """Test when template file is missing"""
        mock_exists.return_value = False
        blech_clust_dir = '/path/to/blech_clust'

        with pytest.raises(SystemExit):
            check_template_file(blech_clust_dir)


class TestGetFileLists:
    """Tests for the get_file_lists function"""

    def test_one_file_per_signal_type(self):
        """Test getting file lists for 'one file per signal type'"""
        file_list = ['amplifier.dat', 'time.dat', 'info.rhd']
        file_type = 'one file per signal type'

        result = get_file_lists(file_list, file_type)

        assert 'electrodes' in result[file_type]
        assert result[file_type]['electrodes'] == ['amplifier.dat']

    def test_one_file_per_channel(self):
        """Test getting file lists for 'one file per channel'"""
        file_list = ['amp-001.dat', 'amp-002.dat', 'time.dat', 'info.rhd']
        file_type = 'one file per channel'

        result = get_file_lists(file_list, file_type)

        assert 'electrodes' in result[file_type]
        assert result[file_type]['electrodes'] == [
            'amp-001.dat', 'amp-002.dat']

    def test_traditional(self):
        """Test getting file lists for 'traditional'"""
        file_list = ['file1.rhd', 'file2.rhd', 'other.dat']
        file_type = 'traditional'

        result = get_file_lists(file_list, file_type)

        assert 'rhd' in result[file_type]
        assert result[file_type]['rhd'] == ['file1.rhd', 'file2.rhd']

    def test_invalid_file_type(self):
        """Test with invalid file type"""
        file_list = ['file1.dat']
        file_type = 'invalid_type'

        with pytest.raises(ValueError):
            get_file_lists(file_list, file_type)


class TestGetElectrodeInfo:
    """Tests for the get_electrode_info function"""

    def test_get_electrode_info(self):
        """Test extracting electrode information from info_dict"""
        info_dict = {
            'electrode_layout': {
                'region1': [[1, 2], [3, 4]],
                'region2': [[5, 6]],
                'emg': [[101, 102]]
            },
            'emg': {
                'port': 'C',
                'electrodes': [101, 102]
            }
        }

        all_electrodes, emg_channels = get_electrode_info(info_dict)

        assert sorted(all_electrodes) == [1, 2, 3, 4, 5, 6]
        assert emg_channels == [101, 102]


class TestCreateParamsFile:
    """Tests for the create_params_file function"""

    @patch('json.load')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    @patch('json.dump')
    def test_create_new_params_file(self, mock_json_dump, mock_exists, mock_open_file, mock_json_load):
        """Test creating a new params file when none exists"""
        # Setup
        mock_exists.return_value = False
        mock_json_load.return_value = {'key': 'value'}

        params_template_path = '/path/to/template.json'
        hdf5_name = 'test.h5'
        sampling_rate = 30000

        # Call the function
        result = create_params_file(
            params_template_path, hdf5_name, sampling_rate)

        # Assertions
        assert result['key'] == 'value'
        assert result['sampling_rate'] == 30000
        mock_json_dump.assert_called_once()

    @patch('json.load')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    @patch('json.dump')
    def test_params_file_already_exists(self, mock_json_dump, mock_exists, mock_open_file, mock_json_load):
        """Test when params file already exists"""
        # Setup
        mock_exists.return_value = True
        mock_json_load.return_value = {'key': 'value'}

        params_template_path = '/path/to/template.json'
        hdf5_name = 'test.h5'
        sampling_rate = 30000

        # Call the function
        result = create_params_file(
            params_template_path, hdf5_name, sampling_rate)

        # Assertions
        assert result['key'] == 'value'
        assert result['sampling_rate'] == 30000
        mock_json_dump.assert_not_called()


class TestGenerateProcessingScripts:
    """Tests for the generate_processing_scripts function"""

    @patch('os.path.exists')
    @patch('os.mkdir')
    @patch('builtins.open', new_callable=mock_open)
    def test_generate_processing_scripts(self, mock_open_file, mock_mkdir, mock_exists):
        """Test generating processing scripts"""
        # Setup
        mock_exists.return_value = False

        dir_name = '/path/to/data'
        blech_clust_dir = '/path/to/blech_clust'

        # Create mock electrode layout frame
        electrode_layout_frame = pd.DataFrame({
            'electrode_ind': [1, 2, 3, 4],
            'CAR_group': ['group1', 'group1', 'emg', 'none']
        })

        all_electrodes = [1, 2, 3, 4]
        all_params_dict = {'max_parallel_cpu': 4}

        # Call the function
        generate_processing_scripts(dir_name, blech_clust_dir, electrode_layout_frame,
                                    all_electrodes, all_params_dict)

        # Assertions
        mock_mkdir.assert_called_once()
        assert mock_open_file.call_count == 2  # Two files should be created


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])
