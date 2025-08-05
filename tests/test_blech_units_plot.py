"""
Tests for the blech_units_plot.py module.
"""
import blech_units_plot
import os
import sys
import numpy as np
import pytest
import tables
import matplotlib.pyplot as plt
import pandas as pd
from unittest.mock import patch, MagicMock, mock_open

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


class TestBlechUnitsPlot:
    """Test class for blech_units_plot.py"""

    @pytest.fixture
    def mock_metadata_handler(self):
        """Create a mock metadata handler"""
        mock = MagicMock()
        mock.dir_name = "/test/dir"
        mock.params_dict = {
            "sampling_rate": 30000
        }
        mock.layout = pd.DataFrame({
            'electrode_ind': [1, 2, 3],
            'threshold': [100, 150, 200]
        })
        mock.hdf5_name = "test.h5"
        return mock

    @pytest.fixture
    def mock_pipeline_check(self):
        """Create a mock pipeline check"""
        mock = MagicMock()
        return mock

    @pytest.fixture
    def mock_hf5(self):
        """Create a mock HDF5 file"""
        mock = MagicMock()

        # Create mock units
        unit1 = MagicMock()
        unit1.waveforms = np.random.randn(100, 30)
        unit1.times = np.sort(np.random.randint(0, 10000, 100))

        unit2 = MagicMock()
        unit2.waveforms = np.random.randn(150, 30)
        unit2.times = np.sort(np.random.randint(0, 10000, 150))

        # Set up the list_nodes method to return the mock units
        mock.list_nodes.return_value = [unit1, unit2]

        # Set up the unit_descriptor
        mock.root.unit_descriptor = [
            {"electrode_number": 1, "single_unit": 1, "regular_spiking": 1,
             "fast_spiking": 0, "snr": 2.5},
            {"electrode_number": 2, "single_unit": 0, "regular_spiking": 0,
             "fast_spiking": 1, "snr": 1.8}
        ]

        return mock

    @patch('os.chdir')
    @patch('os.path.realpath')
    def test_setup_environment(self, mock_realpath, mock_chdir, mock_metadata_handler, mock_pipeline_check):
        """Test the setup_environment function"""
        mock_realpath.return_value = "/path/to/script.py"

        with patch('blech_units_plot.imp_metadata', return_value=mock_metadata_handler):
            with patch('blech_units_plot.pipeline_graph_check', return_value=mock_pipeline_check):
                result = blech_units_plot.setup_environment(["script.py"])

                # Check that the function returns the expected values
                assert result[0] == mock_metadata_handler
                assert result[1] == "/test/dir"
                assert result[2] == mock_metadata_handler.params_dict
                assert result[3].equals(mock_metadata_handler.layout)
                assert result[4] == mock_pipeline_check

                # Check that the pipeline check methods were called
                mock_pipeline_check.check_previous.assert_called_once_with(
                    "/path/to/script.py")
                mock_pipeline_check.write_to_log.assert_called_once_with(
                    "/path/to/script.py", 'attempted')

                # Check that os.chdir was called with the correct directory
                mock_chdir.assert_called_once_with("/test/dir")

    @patch('os.mkdir')
    @patch('shutil.rmtree')
    def test_prepare_output_directory(self, mock_rmtree, mock_mkdir):
        """Test the prepare_output_directory function"""
        test_dir = "test_output_dir"
        blech_units_plot.prepare_output_directory(test_dir)

        # Check that rmtree and mkdir were called with the correct directory
        mock_rmtree.assert_called_once_with(test_dir, ignore_errors=True)
        mock_mkdir.assert_called_once_with(test_dir)

    @patch('tables.open_file')
    def test_load_units_data(self, mock_open_file, mock_hf5):
        """Test the load_units_data function"""
        mock_open_file.return_value = mock_hf5

        # Create mock units with times
        unit1 = MagicMock()
        unit1.times = np.array([100, 200, 300])
        unit2 = MagicMock()
        unit2.times = np.array([50, 150, 350])

        # Override the list_nodes method to return our specific mock units
        mock_hf5.list_nodes.return_value = [unit1, unit2]

        result = blech_units_plot.load_units_data("test.h5")

        # Check that the function returns the expected values
        assert result[0] == mock_hf5
        assert result[1] == [unit1, unit2]
        assert result[2] == 50  # min_time
        assert result[3] == 350  # max_time

        # Check that open_file was called with the correct filename
        mock_open_file.assert_called_once_with("test.h5", 'r+')

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.close')
    @patch('blech_units_plot.blech_waveforms_datashader.waveforms_datashader')
    @patch('blech_units_plot.gen_isi_hist')
    def test_plot_unit_summary(self, mock_gen_isi_hist, mock_waveforms_datashader,
                               mock_close, mock_subplots, mock_hf5):
        """Test the plot_unit_summary function"""
        # Setup mock figure and axes
        mock_fig = MagicMock()
        mock_ax = [[MagicMock(), MagicMock()], [MagicMock(), MagicMock()]]
        mock_subplots.return_value = (mock_fig, mock_ax)

        # Setup mock waveforms_datashader
        mock_waveforms_datashader.return_value = (mock_fig, mock_ax[0][0])

        # Setup mock gen_isi_hist
        mock_gen_isi_hist.return_value = (mock_fig, mock_ax[1][0])

        # Create test data
        unit_data = MagicMock()
        unit_data.waveforms = np.random.randn(100, 30)
        unit_data.times = np.sort(np.random.randint(0, 10000, 100))

        unit_descriptor = {
            "electrode_number": 1,
            "single_unit": 1,
            "regular_spiking": 1,
            "fast_spiking": 0,
            "snr": 2.5
        }

        layout_frame = pd.DataFrame({
            'electrode_ind': [1, 2, 3],
            'threshold': [100, 150, 200]
        })

        params_dict = {"sampling_rate": 30000}

        # Call the function
        with patch('matplotlib.pyplot.savefig'):
            blech_units_plot.plot_unit_summary(
                unit_data=unit_data,
                min_time=0,
                max_time=10000,
                unit_index=0,
                unit_descriptor=unit_descriptor,
                layout_frame=layout_frame,
                params_dict=params_dict,
                output_dir="test_output_dir"
            )

        # Check that the plotting functions were called
        mock_waveforms_datashader.assert_called_once()
        mock_gen_isi_hist.assert_called_once()
        mock_close.assert_called_once_with("all")

    @patch('os.path.join')
    @patch('os.mkdir')
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.close')
    @patch('blech_units_plot.blech_waveforms_datashader.waveforms_datashader')
    def test_save_individual_plots(self, mock_waveforms_datashader, mock_close,
                                   mock_subplots, mock_mkdir, mock_path_join):
        """Test the save_individual_plots function"""
        # Setup mock figure and axes
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        # Setup mock waveforms_datashader
        mock_waveforms_datashader.return_value = (mock_fig, mock_ax)

        # Setup mock path_join
        mock_path_join.side_effect = lambda *args: '/'.join(args)

        # Create test units
        unit1 = MagicMock()
        unit1.waveforms = np.random.randn(100, 30)
        unit1.times = np.sort(np.random.randint(0, 10000, 100))

        unit2 = MagicMock()
        unit2.waveforms = np.random.randn(150, 30)
        unit2.times = np.sort(np.random.randint(0, 10000, 150))

        units = [unit1, unit2]

        # Call the function
        with patch('matplotlib.pyplot.savefig'):
            blech_units_plot.save_individual_plots(units, "test_subdir")

        # Check that mkdir was called
        mock_mkdir.assert_called_once_with('unit_waveforms_plots/test_subdir')

        # Check that close was called the expected number of times (2 units * 2 plots)
        assert mock_close.call_count == 4

    @patch('blech_units_plot.setup_environment')
    @patch('blech_units_plot.prepare_output_directory')
    @patch('blech_units_plot.load_units_data')
    @patch('blech_units_plot.process_all_units')
    @patch('blech_units_plot.save_individual_plots')
    @patch('sys.argv', ['script.py'])
    def test_main(self, mock_save_individual_plots, mock_process_all_units,
                  mock_load_units_data, mock_prepare_output_directory,
                  mock_setup_environment, mock_metadata_handler, mock_pipeline_check):
        """Test the main function"""
        # Setup mock return values
        mock_setup_environment.return_value = (
            mock_metadata_handler,
            "/test/dir",
            {"sampling_rate": 30000},
            pd.DataFrame(),
            mock_pipeline_check,
            "/path/to/script.py"
        )

        mock_hf5 = MagicMock()
        mock_units = [MagicMock(), MagicMock()]
        mock_load_units_data.return_value = (mock_hf5, mock_units, 0, 10000)

        # Call the main function
        blech_units_plot.main()

        # Check that all the functions were called
        mock_setup_environment.assert_called_once_with(['script.py'])
        mock_prepare_output_directory.assert_called_once()
        mock_load_units_data.assert_called_once_with(
            mock_metadata_handler.hdf5_name)
        mock_process_all_units.assert_called_once_with(
            mock_units, mock_hf5, pd.DataFrame(
            ), {"sampling_rate": 30000}, 0, 10000
        )
        mock_save_individual_plots.assert_called_once_with(mock_units)

        # Check that hf5.close was called
        mock_hf5.close.assert_called_once()

        # Check that write_to_log was called with 'completed'
        mock_pipeline_check.write_to_log.assert_called_with(
            "/path/to/script.py", 'completed')
