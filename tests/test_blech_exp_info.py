"""
Tests for blech_exp_info module functionality.
"""
import json
import os
import tempfile
import unittest.mock as mock
from unittest.mock import patch, MagicMock

import pytest

from blech_clust.blech_exp_info import (
    parse_arguments,
    process_dig_ins_programmatic,
    process_dig_ins_manual,
    display_existing_info,
    main
)


class TestBlechExpInfo:
    """Test class for blech_exp_info functionality."""

    def test_parse_arguments_includes_taste_delivery_day(self):
        """Test that taste-delivery-day argument is included in parser."""
        # Mock sys.argv to simulate command line arguments
        with patch('sys.argv', ['blech_exp_info.py', 'test_dir', '--taste-delivery-day', '3']):
            args = parse_arguments()
            assert hasattr(args, 'taste_delivery_day')
            assert args.taste_delivery_day == 3

    def test_parse_arguments_taste_delivery_day_optional(self):
        """Test that taste-delivery-day argument is optional."""
        with patch('sys.argv', ['blech_exp_info.py', 'test_dir']):
            args = parse_arguments()
            assert hasattr(args, 'taste_delivery_day')
            assert args.taste_delivery_day is None

    @patch('blech_clust.blech_exp_info.test_bool', True)
    def test_test_namespace_includes_taste_delivery_day(self):
        """Test that test namespace includes taste_delivery_day."""
        args = parse_arguments()
        assert hasattr(args, 'taste_delivery_day')
        assert args.taste_delivery_day is None

    def test_process_dig_ins_programmatic_with_taste_delivery_day(self):
        """Test programmatic processing includes taste delivery day."""
        # Create mock args with taste delivery day
        mock_args = MagicMock()
        mock_args.taste_digins = "1,2,3,4"
        mock_args.tastes = "NaCl,Quinine,Sucrose,Citric"
        mock_args.concentrations = "0.1,0.001,0.3,0.01"
        mock_args.palatability = "1,2,4,3"
        mock_args.taste_delivery_day = 2

        # Create mock dig handler
        mock_dig_handler = MagicMock()
        mock_dig_handler.dig_in_frame = MagicMock()
        mock_dig_handler.dig_in_frame.trial_counts = [10, 15, 12, 8]
        mock_dig_handler.dig_in_frame.loc = MagicMock()
        mock_dig_handler.dig_in_frame.dig_in_nums = [1, 2, 3, 4]

        # Mock the get_dig_in_files and get_trial_data methods
        mock_dig_handler.get_dig_in_files = MagicMock()
        mock_dig_handler.get_trial_data = MagicMock()

        result = process_dig_ins_programmatic(mock_dig_handler, mock_args)

        # Check that taste_delivery_day is included in the result
        assert len(result) == 7  # Should have 7 elements now
        taste_dig_inds, tastes, concs, pal_ranks, taste_digin_nums, taste_digin_trials, taste_delivery_day = result
        assert taste_delivery_day == 2

    def test_process_dig_ins_programmatic_without_taste_delivery_day_raises_error(self):
        """Test that programmatic processing raises error when taste delivery day not provided."""
        # Create mock args without taste delivery day
        mock_args = MagicMock()
        mock_args.taste_digins = "1,2,3,4"
        mock_args.tastes = "NaCl,Quinine,Sucrose,Citric"
        mock_args.concentrations = "0.1,0.001,0.3,0.01"
        mock_args.palatability = "1,2,4,3"
        mock_args.taste_delivery_day = None

        # Create mock dig handler
        mock_dig_handler = MagicMock()
        mock_dig_handler.dig_in_frame = MagicMock()
        mock_dig_handler.dig_in_frame.trial_counts = [10, 15, 12, 8]
        mock_dig_handler.get_dig_in_files = MagicMock()
        mock_dig_handler.get_trial_data = MagicMock()

        with pytest.raises(ValueError, match="Taste delivery day not provided"):
            process_dig_ins_programmatic(mock_dig_handler, mock_args)

    def test_display_existing_info_shows_taste_delivery_day(self):
        """Test that display_existing_info shows taste delivery day."""
        existing_info = {
            'taste_params': {
                'dig_in_nums': [1, 2, 3, 4],
                'tastes': ['NaCl', 'Quinine', 'Sucrose', 'Citric'],
                'concs': [0.1, 0.001, 0.3, 0.01],
                'pal_rankings': [1, 2, 4, 3],
                'taste_delivery_day': 3
            }
        }

        with patch('builtins.print') as mock_print:
            display_existing_info(existing_info)

            # Check that taste delivery day was printed
            print_calls = [str(call) for call in mock_print.call_args_list]
            taste_delivery_day_found = any(
                'Taste delivery day: 3' in call for call in print_calls)
            assert taste_delivery_day_found

    def test_display_existing_info_handles_missing_taste_delivery_day(self):
        """Test that display_existing_info handles missing taste delivery day gracefully."""
        existing_info = {
            'taste_params': {
                'dig_in_nums': [1, 2, 3, 4],
                'tastes': ['NaCl', 'Quinine', 'Sucrus', 'Citric'],
                'concs': [0.1, 0.001, 0.3, 0.01],
                'pal_rankings': [1, 2, 4, 3]
                # No taste_delivery_day
            }
        }

        with patch('builtins.print') as mock_print:
            display_existing_info(existing_info)

            # Check that "Not set" was printed for taste delivery day
            print_calls = [str(call) for call in mock_print.call_args_list]
            taste_delivery_day_found = any(
                'Taste delivery day: Not set' in call for call in print_calls)
            assert taste_delivery_day_found

    @patch('blech_clust.blech_exp_info.setup_experiment_info')
    @patch('blech_clust.blech_exp_info.DigInHandler')
    @patch('blech_clust.blech_exp_info.process_dig_ins_programmatic')
    @patch('blech_clust.blech_exp_info.process_laser_params_programmatic')
    @patch('blech_clust.blech_exp_info.process_electrode_files')
    @patch('blech_clust.blech_exp_info.process_electrode_layout')
    @patch('blech_clust.blech_exp_info.process_notes')
    @patch('blech_clust.blech_exp_info.extract_recording_params')
    @patch('blech_clust.blech_exp_info.json.dump')
    @patch('os.path.exists')
    @patch('os.listdir')
    def test_main_includes_taste_delivery_day_in_final_dict(self, mock_listdir, mock_exists, mock_json_dump,
                                                            mock_extract_recording, mock_process_notes,
                                                            mock_process_layout, mock_process_electrodes,
                                                            mock_process_laser, mock_process_taste,
                                                            mock_dig_handler, mock_setup):
        """Test that main function includes taste delivery day in final dictionary."""

        # Setup mocks
        mock_setup.return_value = (
            '/test/path', 'test_dir', '/cache/path', {}, {}, {}, None)
        mock_listdir.return_value = ['auxiliary.dat']
        mock_exists.return_value = True

        # Mock dig handler
        mock_handler_instance = MagicMock()
        mock_dig_handler.return_value = mock_handler_instance
        mock_handler_instance.dig_in_frame = MagicMock()
        mock_handler_instance.dig_in_frame.dig_in_nums = [1, 2, 3, 4]
        mock_handler_instance.dig_in_frame.trial_counts = [10, 15, 12, 8]
        mock_handler_instance.write_out_frame = MagicMock()

        # Mock taste processing to return taste_delivery_day
        mock_process_taste.return_value = ([1, 2, 3, 4], ['NaCl', 'Quinine', 'Sucrose', 'Citric'],
                                           [0.1, 0.001, 0.3, 0.01], [
                                               1, 2, 4, 3],
                                           [1, 2, 3, 4], [10, 15, 12, 8], 2)

        # Mock other processes
        mock_process_laser.return_value = (None, [], [], '', [])
        mock_process_electrodes.return_value = ([], [], [])
        mock_process_layout.return_value = ({}, None, [], '')
        mock_process_notes.return_value = 'Test notes'
        mock_extract_recording.return_value = None

        # Mock args for programmatic mode
        with patch('blech_clust.blech_exp_info.args') as mock_args:
            mock_args.programmatic = True
            mock_args.template = None

            result = main()

            # Check that taste_delivery_day is in the final dictionary
            assert 'taste_params' in result
            assert 'taste_delivery_day' in result['taste_params']
            assert result['taste_params']['taste_delivery_day'] == 2

    def test_manual_processing_with_taste_delivery_day(self):
        """Test manual processing includes taste delivery day with caching."""
        # Create mock args
        mock_args = MagicMock()
        mock_args.auto_defaults = False

        # Create mock dig handler
        mock_dig_handler = MagicMock()
        mock_dig_handler.dig_in_num = [1, 2, 3, 4]
        mock_dig_handler.dig_in_frame = MagicMock()
        mock_dig_handler.dig_in_frame.trial_counts = [10, 15, 12, 8]
        mock_dig_handler.dig_in_frame.loc = MagicMock()
        mock_dig_handler.dig_in_frame.dig_in_nums = [1, 2, 3, 4]
        mock_dig_handler.get_dig_in_files = MagicMock()
        mock_dig_handler.get_trial_data = MagicMock()

        existing_info = {}
        cache = {}
        cache_file_path = '/test/cache.json'

        with patch('blech_clust.blech_exp_info.populate_field_with_defaults') as mock_populate:
            # Mock the populate_field_with_defaults to return taste delivery day
            mock_populate.return_value = '3'

            with patch('blech_clust.blech_exp_info.save_to_cache'):
                result = process_dig_ins_manual(
                    mock_dig_handler, mock_args, existing_info, cache, cache_file_path)

                # Check that taste_delivery_day is included in the result
                assert len(result) == 7  # Should have 7 elements now
                taste_dig_inds, tastes, concs, pal_ranks, taste_digin_nums, taste_digin_trials, taste_delivery_day = result
                assert taste_delivery_day == 3

                # Check that cache was updated with taste delivery day
                assert 'taste_params' in cache
                assert 'taste_delivery_day' in cache['taste_params']
                assert cache['taste_params']['taste_delivery_day'] == 3


if __name__ == '__main__':
    pytest.main([__file__])
