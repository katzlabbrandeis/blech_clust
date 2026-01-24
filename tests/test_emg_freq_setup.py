"""
Tests for the emg/emg_freq_setup.py module.
"""
from emg import emg_freq_setup
import pytest
import os
import sys
from unittest.mock import patch, MagicMock
import numpy as np

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


class TestEmgFreqSetup:
    """Test class for emg_freq_setup.py"""

    @pytest.fixture
    def mock_metadata_handler(self):
        """Create a mock metadata handler"""
        mock = MagicMock()
        mock.dir_name = "/test/dir"
        mock.hdf5_name = "test.h5"
        mock.params_dict = {
            'sampling_rate': 30000
        }
        return mock

    def test_module_imports(self):
        """Test that all required imports are available"""
        assert hasattr(emg_freq_setup, 'tables')
        assert hasattr(emg_freq_setup, 'np')

    def test_module_structure(self):
        """Test that module has expected structure"""
        # Verify module can be imported
        assert emg_freq_setup is not None
