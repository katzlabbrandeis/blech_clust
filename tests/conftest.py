import pytest
import os
import pandas as pd
import numpy as np
from unittest.mock import MagicMock


@pytest.fixture
def mock_dir_name():
    """Fixture providing a mock directory name"""
    return '/path/to/test_data/'


@pytest.fixture
def mock_file_list():
    """Fixture providing a mock file list"""
    return [
        'amp-001.dat', 'amp-002.dat', 'amp-003.dat',
        'time.dat', 'info.rhd', 'digital_in.dat',
        'test_layout.csv'
    ]


@pytest.fixture
def mock_info_dict():
    """Fixture providing a mock info dictionary"""
    return {
        'file_type': 'one file per channel',
        'ports': ['A', 'B'],
        'taste_params': {
            'dig_in_nums': [0, 1, 2, 3],
            'tastes': ['taste1', 'taste2', 'taste3', 'taste4']
        },
        'laser_params': {
            'dig_in_nums': [4]
        },
        'electrode_layout': {
            'region1': [[1, 2, 3], [4, 5, 6]],
            'region2': [[7, 8, 9]],
            'emg': [[101, 102]]
        },
        'emg': {
            'port': 'C',
            'electrodes': [101, 102]
        }
    }


@pytest.fixture
def mock_electrode_layout_frame():
    """Fixture providing a mock electrode layout DataFrame"""
    return pd.DataFrame({
        'electrode_ind': [1, 2, 3, 4, 5, 101, 102],
        'CAR_group': ['group1', 'group1', 'group1', 'group2', 'none', 'emg', 'emg'],
        'port': ['A', 'A', 'A', 'B', 'B', 'C', 'C'],
        'channel': [1, 2, 3, 1, 2, 1, 2],
        'region': ['region1', 'region1', 'region1', 'region2', 'none', 'emg', 'emg']
    })


@pytest.fixture
def mock_dig_handler():
    """Fixture providing a mock DigInHandler"""
    mock_handler = MagicMock()
    mock_handler.dig_in_frame = pd.DataFrame({
        'channel': [0, 1, 2, 3, 4],
        'taste': ['taste1', 'taste2', 'taste3', 'taste4', 'laser'],
        'n_pulses': [10, 10, 10, 10, 5],
        'pulse_times': [
            '[(100, 200), (300, 400)]',
            '[(150, 250), (350, 450)]',
            '[(200, 300), (400, 500)]',
            '[(250, 350), (450, 550)]',
            '[(500, 600)]'
        ]
    })
    return mock_handler


@pytest.fixture
def mock_hdf5_file():
    """Fixture providing a mock HDF5 file"""
    mock_file = MagicMock()
    return mock_file
