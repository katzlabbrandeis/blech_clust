"""
Tests for DigInHandler class in utils/read_file.py

Focuses on testing arbitrary string name support in dig-in filenames.

Test Coverage:
- Traditional numeric format (board-DI-00.dat)
- DIN numeric format (board-DIN-09.dat)
- Arbitrary string names (board-DI-Suc.dat)
- String names with embedded numbers (board-DI-Suc300mM.dat)
- Mixed formats (edge case)
- Single file scenarios
- Case-sensitive sorting
- Special characters in names
- Empty directories
- Non-board-DI files filtering
- Long descriptive names
- Consistency across multiple runs
"""
from utils.read_file import DigInHandler
import os
import sys
import pytest
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


class TestDigInHandlerFilenamesParsing:
    """Test class for DigInHandler filename parsing with arbitrary strings"""

    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory for test files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    def create_dummy_dat_files(self, data_dir, filenames):
        """Helper to create dummy .dat files"""
        for filename in filenames:
            filepath = os.path.join(data_dir, filename)
            with open(filepath, 'wb') as f:
                # Write some dummy data
                f.write(b'\x00' * 100)

    def test_numeric_format_board_di(self, temp_data_dir):
        """Test traditional numeric format: board-DI-00.dat"""
        filenames = ['board-DI-00.dat', 'board-DI-01.dat', 'board-DI-02.dat']
        self.create_dummy_dat_files(temp_data_dir, filenames)

        handler = DigInHandler(temp_data_dir, 'one file per channel')
        handler.get_dig_in_files()

        assert handler.dig_in_num == [0, 1, 2]
        assert handler.dig_in_name == [
            'board-DI-00', 'board-DI-01', 'board-DI-02']
        assert len(handler.dig_in_file_list) == 3

    def test_numeric_format_board_din(self, temp_data_dir):
        """Test DIN numeric format: board-DIN-09.dat"""
        filenames = ['board-DIN-09.dat',
                     'board-DIN-11.dat', 'board-DIN-13.dat']
        self.create_dummy_dat_files(temp_data_dir, filenames)

        handler = DigInHandler(temp_data_dir, 'one file per channel')
        handler.get_dig_in_files()

        # Should extract numbers and sort numerically
        assert handler.dig_in_num == [9, 11, 13]
        assert handler.dig_in_name == [
            'board-DIN-09', 'board-DIN-11', 'board-DIN-13']

    def test_arbitrary_string_names(self, temp_data_dir):
        """Test arbitrary string names: board-DI-Suc.dat"""
        filenames = ['board-DI-Suc.dat',
                     'board-DI-NaCl.dat', 'board-DI-CitricAcid.dat']
        self.create_dummy_dat_files(temp_data_dir, filenames)

        handler = DigInHandler(temp_data_dir, 'one file per channel')
        handler.get_dig_in_files()

        # Should sort alphabetically and assign sequential numbers
        assert handler.dig_in_num == [0, 1, 2]
        # Alphabetical order: CitricAcid, NaCl, Suc
        assert handler.dig_in_name == [
            'board-DI-CitricAcid', 'board-DI-NaCl', 'board-DI-Suc']

    def test_arbitrary_string_names_different_order(self, temp_data_dir):
        """Test that arbitrary strings are sorted alphabetically"""
        filenames = ['board-DI-Water.dat',
                     'board-DI-Sucrose.dat', 'board-DI-Quinine.dat']
        self.create_dummy_dat_files(temp_data_dir, filenames)

        handler = DigInHandler(temp_data_dir, 'one file per channel')
        handler.get_dig_in_files()

        # Alphabetical: Quinine, Sucrose, Water
        assert handler.dig_in_num == [0, 1, 2]
        assert handler.dig_in_name == [
            'board-DI-Quinine', 'board-DI-Sucrose', 'board-DI-Water']

    def test_string_names_with_embedded_numbers(self, temp_data_dir):
        """Test string names with embedded numbers: board-DI-Suc300mM.dat"""
        filenames = ['board-DI-Suc300mM.dat',
                     'board-DI-NaCl500mM.dat', 'board-DI-Water0mM.dat']
        self.create_dummy_dat_files(temp_data_dir, filenames)

        handler = DigInHandler(temp_data_dir, 'one file per channel')
        handler.get_dig_in_files()

        # Should extract last number found (300, 500, 0) and sort numerically
        assert handler.dig_in_num == [0, 300, 500]
        assert handler.dig_in_name == [
            'board-DI-Water0mM', 'board-DI-Suc300mM', 'board-DI-NaCl500mM']

    def test_mixed_formats_not_recommended(self, temp_data_dir):
        """Test mixed numeric and string formats (edge case)"""
        # Note: Mixing formats is not recommended but should still work
        filenames = ['board-DI-00.dat', 'board-DI-Suc.dat', 'board-DI-02.dat']
        self.create_dummy_dat_files(temp_data_dir, filenames)

        handler = DigInHandler(temp_data_dir, 'one file per channel')
        handler.get_dig_in_files()

        # board-DI-00 -> 0, board-DI-Suc -> None, board-DI-02 -> 2
        # Since there's a None, should fall back to alphabetical sorting
        assert handler.dig_in_num == [0, 1, 2]
        # Alphabetical: board-DI-00, board-DI-02, board-DI-Suc
        assert handler.dig_in_name == [
            'board-DI-00', 'board-DI-02', 'board-DI-Suc']

    def test_single_arbitrary_string_file(self, temp_data_dir):
        """Test single file with arbitrary string name"""
        filenames = ['board-DI-Sucrose.dat']
        self.create_dummy_dat_files(temp_data_dir, filenames)

        handler = DigInHandler(temp_data_dir, 'one file per channel')
        handler.get_dig_in_files()

        assert handler.dig_in_num == [0]
        assert handler.dig_in_name == ['board-DI-Sucrose']

    def test_case_sensitive_sorting(self, temp_data_dir):
        """Test that sorting is case-sensitive (Python default)"""
        filenames = ['board-DI-water.dat',
                     'board-DI-Sucrose.dat', 'board-DI-QUININE.dat']
        self.create_dummy_dat_files(temp_data_dir, filenames)

        handler = DigInHandler(temp_data_dir, 'one file per channel')
        handler.get_dig_in_files()

        # Python sorts uppercase before lowercase
        # Expected order: QUININE, Sucrose, water
        assert handler.dig_in_num == [0, 1, 2]
        assert handler.dig_in_name == [
            'board-DI-QUININE', 'board-DI-Sucrose', 'board-DI-water']

    def test_special_characters_in_names(self, temp_data_dir):
        """Test filenames with special characters"""
        filenames = ['board-DI-H2O.dat',
                     'board-DI-NaCl_500mM.dat', 'board-DI-Citric-Acid.dat']
        self.create_dummy_dat_files(temp_data_dir, filenames)

        handler = DigInHandler(temp_data_dir, 'one file per channel')
        handler.get_dig_in_files()

        # H2O has number 2, NaCl_500mM has 500, Citric-Acid has no number
        # Since there's a None (Citric-Acid), falls back to alphabetical
        assert handler.dig_in_num == [0, 1, 2]
        # Alphabetical order
        expected_names = sorted(
            ['board-DI-H2O', 'board-DI-NaCl_500mM', 'board-DI-Citric-Acid'])
        assert handler.dig_in_name == expected_names

    def test_empty_directory(self, temp_data_dir):
        """Test behavior with no dig-in files"""
        handler = DigInHandler(temp_data_dir, 'one file per channel')
        handler.get_dig_in_files()

        assert handler.dig_in_num == []
        assert handler.dig_in_name == []
        assert handler.dig_in_file_list == []

    def test_non_board_di_files_ignored(self, temp_data_dir):
        """Test that non-board-DI files are ignored"""
        filenames = [
            'board-DI-Suc.dat',
            'other-file.dat',
            'board-AI-00.dat',  # Analog input, not digital
            'board-DI-NaCl.dat'
        ]
        self.create_dummy_dat_files(temp_data_dir, filenames)

        handler = DigInHandler(temp_data_dir, 'one file per channel')
        handler.get_dig_in_files()

        # Should only find board-DI files
        assert len(handler.dig_in_file_list) == 2
        assert handler.dig_in_num == [0, 1]
        assert handler.dig_in_name == ['board-DI-NaCl', 'board-DI-Suc']

    def test_long_descriptive_names(self, temp_data_dir):
        """Test long descriptive filenames"""
        filenames = [
            'board-DI-SucroseOctaacetate.dat',
            'board-DI-MonosodiumGlutamate.dat',
            'board-DI-QuinineHydrochloride.dat'
        ]
        self.create_dummy_dat_files(temp_data_dir, filenames)

        handler = DigInHandler(temp_data_dir, 'one file per channel')
        handler.get_dig_in_files()

        assert handler.dig_in_num == [0, 1, 2]
        # Alphabetical order
        assert handler.dig_in_name == [
            'board-DI-MonosodiumGlutamate',
            'board-DI-QuinineHydrochloride',
            'board-DI-SucroseOctaacetate'
        ]

    def test_numeric_stability_across_runs(self, temp_data_dir):
        """Test that the same files produce consistent numbering"""
        filenames = ['board-DI-Suc.dat',
                     'board-DI-NaCl.dat', 'board-DI-Water.dat']
        self.create_dummy_dat_files(temp_data_dir, filenames)

        # Run twice
        handler1 = DigInHandler(temp_data_dir, 'one file per channel')
        handler1.get_dig_in_files()

        handler2 = DigInHandler(temp_data_dir, 'one file per channel')
        handler2.get_dig_in_files()

        # Should produce identical results
        assert handler1.dig_in_num == handler2.dig_in_num
        assert handler1.dig_in_name == handler2.dig_in_name
        assert handler1.dig_in_file_list == handler2.dig_in_file_list
