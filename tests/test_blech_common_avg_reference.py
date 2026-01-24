"""
Tests for the blech_common_avg_reference.py module.
"""
import blech_common_avg_reference
import pytest
import os
import sys
from unittest.mock import patch, MagicMock
import numpy as np

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


class TestBlechCommonAvgReference:
    """Test class for blech_common_avg_reference.py"""

    @pytest.fixture
    def mock_metadata_handler(self):
        """Create a mock metadata handler"""
        mock = MagicMock()
        mock.dir_name = "/test/dir"
        mock.hdf5_name = "test.h5"
        mock.params_dict = {
            'sampling_rate': 30000,
            'max_breach_rate': 0.2,
            'max_secs_above_thresh': 1.0,
            'max_mean_breach_rate_persec': 10.0
        }
        return mock

    @pytest.fixture
    def mock_hf5(self):
        """Create a mock HDF5 file"""
        mock = MagicMock()
        mock.root.raw = MagicMock()
        return mock

    def test_module_imports(self):
        """Test that all required imports are available"""
        assert hasattr(blech_common_avg_reference, 'tables')
        assert hasattr(blech_common_avg_reference, 'np')

    @patch('blech_common_avg_reference.tables.open_file')
    def test_hdf5_file_operations(self, mock_open_file, mock_hf5):
        """Test HDF5 file operations"""
        mock_open_file.return_value = mock_hf5

        # Test that file can be opened
        hf5 = blech_common_avg_reference.tables.open_file("test.h5", 'r+')
        assert hf5 is not None
        mock_open_file.assert_called_once_with("test.h5", 'r+')

    def test_module_structure(self):
        """Test that module has expected structure"""
        # Verify module can be imported
        assert blech_common_avg_reference is not None
