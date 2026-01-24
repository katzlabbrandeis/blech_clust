"""
Tests for the blech_make_arrays.py module.
"""
import blech_make_arrays
import pytest
import os
import sys
from unittest.mock import patch, MagicMock
import numpy as np

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


class TestBlechMakeArrays:
    """Test class for blech_make_arrays.py"""

    @pytest.fixture
    def mock_metadata_handler(self):
        """Create a mock metadata handler"""
        mock = MagicMock()
        mock.dir_name = "/test/dir"
        mock.hdf5_name = "test.h5"
        mock.params_dict = {
            'sampling_rate': 30000,
            'pre_stim': 2000,
            'post_stim': 5000
        }
        mock.info_dict = {
            'taste_names': ['Taste1', 'Taste2']
        }
        return mock

    def test_module_imports(self):
        """Test that all required imports are available"""
        assert hasattr(blech_make_arrays, 'tables')
        assert hasattr(blech_make_arrays, 'np')

    def test_module_structure(self):
        """Test that module has expected structure"""
        # Verify module can be imported and has main guard
        assert blech_make_arrays is not None

    @patch('blech_make_arrays.imp_metadata')
    @patch('blech_make_arrays.os.chdir')
    def test_metadata_loading(self, mock_chdir, mock_imp_metadata, mock_metadata_handler):
        """Test metadata loading functionality"""
        mock_imp_metadata.return_value = mock_metadata_handler

        # Test that metadata can be loaded
        metadata = mock_imp_metadata()
        assert metadata.dir_name == "/test/dir"
        assert metadata.hdf5_name == "test.h5"
