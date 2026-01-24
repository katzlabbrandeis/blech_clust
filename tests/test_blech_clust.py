"""
Tests for the blech_clust.py module.
"""
import blech_clust
import pytest
import os
import sys
from unittest.mock import patch, MagicMock, mock_open
import numpy as np

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


class TestBlechClust:
    """Test class for blech_clust.py"""

    @pytest.fixture
    def mock_dir_name(self):
        """Create a mock directory name"""
        return "/test/data/dir"

    def test_hdf5_handler_initialization(self, mock_dir_name):
        """Test HDF5Handler initialization"""
        with patch('blech_clust.glob.glob', return_value=[]):
            with patch('blech_clust.tables.open_file') as mock_open:
                mock_hf5 = MagicMock()
                mock_open.return_value = mock_hf5

                handler = blech_clust.HDF5Handler(
                    mock_dir_name, force_run=True)

                assert handler.dir_name == mock_dir_name
                assert handler.force_run == True
                assert 'raw' in handler.group_list
                assert 'digital_in' in handler.group_list

    def test_hdf5_handler_existing_file(self, mock_dir_name):
        """Test HDF5Handler with existing file"""
        with patch('blech_clust.glob.glob', return_value=['test.h5']):
            with patch('blech_clust.tables.open_file') as mock_open:
                mock_hf5 = MagicMock()
                mock_open.return_value = mock_hf5

                handler = blech_clust.HDF5Handler(mock_dir_name)

                assert handler.hdf5_name == 'test.h5'
                mock_open.assert_called_once_with('test.h5', 'r+')

    def test_module_structure(self):
        """Test that module has expected structure"""
        assert hasattr(blech_clust, 'HDF5Handler')
        assert hasattr(blech_clust, 'parser')
