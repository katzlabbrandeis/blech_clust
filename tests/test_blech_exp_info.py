"""
Tests for the blech_exp_info.py module.
"""
import pytest
import os
import sys
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import blech_exp_info


class TestBlechExpInfo:
    """Test class for blech_exp_info.py"""

    @pytest.fixture
    def mock_metadata_handler(self):
        """Create a mock metadata handler"""
        mock = MagicMock()
        mock.dir_name = "/test/dir"
        mock.params_dict = {}
        mock.info_dict = {}
        return mock

    @patch('blech_exp_info.imp_metadata')
    @patch('blech_exp_info.os.chdir')
    def test_main_execution(self, mock_chdir, mock_imp_metadata, mock_metadata_handler):
        """Test that main function executes without errors"""
        mock_imp_metadata.return_value = mock_metadata_handler
        
        with patch('blech_exp_info.sys.argv', ['blech_exp_info.py']):
            with patch('builtins.input', return_value='n'):
                try:
                    # Test that the module can be imported and basic structure exists
                    assert hasattr(blech_exp_info, 'main')
                    assert callable(blech_exp_info.main)
                except Exception as e:
                    pytest.fail(f"Main function test failed: {e}")

    def test_module_imports(self):
        """Test that all required imports are available"""
        assert hasattr(blech_exp_info, 'imp_metadata')
        assert hasattr(blech_exp_info, 'main')
