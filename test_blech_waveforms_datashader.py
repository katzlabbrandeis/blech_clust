"""
Tests for blech_waveforms_datashader.py module.

This module tests the waveforms_datashader and waveform_envelope_plot functions
with various input scenarios including edge cases.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Import the functions to test
from utils.blech_waveforms_datashader import waveforms_datashader, waveform_envelope_plot


class TestWaveformsDatashader:
    """Test class for waveforms_datashader function."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create sample waveform data
        self.n_waveforms = 100
        self.n_timepoints = 50
        self.waveforms = np.random.randn(self.n_waveforms, self.n_timepoints) * 100
        self.x_values = np.linspace(-1, 1, self.n_timepoints)
        
    def teardown_method(self):
        """Clean up after each test method."""
        # Clean up any remaining temp directories
        temp_dirs = ['datashader_temp', 'test_temp_dir']
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

    def test_basic_functionality(self):
        """Test basic functionality with default parameters."""
        fig, ax = waveforms_datashader(self.waveforms, self.x_values)
        
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        
        # Check that the plot has been created
        assert len(ax.images) > 0
        
        plt.close(fig)

    def test_with_threshold(self):
        """Test functionality with threshold parameter."""
        threshold = 50.0
        fig, ax = waveforms_datashader(
            self.waveforms, self.x_values, threshold=threshold
        )
        
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        
        # Check that horizontal lines were added for threshold
        horizontal_lines = [line for line in ax.lines 
                          if hasattr(line, 'get_ydata') and 
                          len(set(line.get_ydata())) == 1]
        assert len(horizontal_lines) >= 2  # Should have positive and negative threshold lines
        
        plt.close(fig)

    def test_without_downsampling(self):
        """Test functionality with downsampling disabled."""
        fig, ax = waveforms_datashader(
            self.waveforms, self.x_values, downsample=False
        )
        
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        
        plt.close(fig)

    def test_custom_directory_name(self):
        """Test functionality with custom directory name."""
        custom_dir = "test_temp_dir"
        fig, ax = waveforms_datashader(
            self.waveforms, self.x_values, dir_name=custom_dir
        )
        
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        
        # Directory should be cleaned up after function execution
        assert not os.path.exists(custom_dir)
        
        plt.close(fig)

    def test_with_existing_axes(self):
        """Test functionality when providing existing axes."""
        fig, existing_ax = plt.subplots(figsize=(10, 8))
        
        returned_fig, returned_ax = waveforms_datashader(
            self.waveforms, self.x_values, ax=existing_ax
        )
        
        assert returned_fig is fig
        assert returned_ax is existing_ax
        
        plt.close(fig)

    def test_empty_waveforms(self):
        """Test with empty waveforms array."""
        empty_waveforms = np.array([]).reshape(0, self.n_timepoints)
        
        with pytest.raises((ValueError, IndexError)):
            waveforms_datashader(empty_waveforms, self.x_values)

    def test_single_waveform(self):
        """Test with single waveform."""
        single_waveform = self.waveforms[:1, :]
        
        fig, ax = waveforms_datashader(single_waveform, self.x_values)
        
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        
        plt.close(fig)

    def test_mismatched_dimensions(self):
        """Test with mismatched waveform and x_values dimensions."""
        wrong_x_values = np.linspace(-1, 1, self.n_timepoints + 10)
        
        with pytest.raises((ValueError, IndexError)):
            waveforms_datashader(self.waveforms, wrong_x_values)

    @patch('utils.blech_waveforms_datashader.export_image')
    @patch('utils.blech_waveforms_datashader.imread')
    def test_file_operations(self, mock_imread, mock_export):
        """Test that file operations are called correctly."""
        # Mock the image reading
        mock_img = np.random.randint(0, 255, (1200, 1600, 3), dtype=np.uint8)
        mock_imread.return_value = mock_img
        
        fig, ax = waveforms_datashader(self.waveforms, self.x_values)
        
        # Check that export_image was called
        mock_export.assert_called_once()
        
        # Check that imread was called
        mock_imread.assert_called_once()
        
        plt.close(fig)


class TestWaveformEnvelopePlot:
    """Test class for waveform_envelope_plot function."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create sample waveform data
        self.n_waveforms = 100
        self.n_timepoints = 50
        self.waveforms = np.random.randn(self.n_waveforms, self.n_timepoints) * 100
        self.x_values = np.linspace(-1, 1, self.n_timepoints)

    def test_basic_functionality(self):
        """Test basic functionality with default parameters."""
        fig, ax = waveform_envelope_plot(self.waveforms, self.x_values)
        
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        
        # Check that lines were plotted (mean line)
        assert len(ax.lines) > 0
        
        # Check that fill_between was used (for envelope)
        assert len(ax.collections) > 0
        
        plt.close(fig)

    def test_with_threshold(self):
        """Test functionality with threshold parameter."""
        threshold = 50.0
        fig, ax = waveform_envelope_plot(
            self.waveforms, self.x_values, threshold=threshold
        )
        
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        
        # Check that horizontal lines were added for threshold
        horizontal_lines = [line for line in ax.lines 
                          if hasattr(line, 'get_ydata') and 
                          len(set(line.get_ydata())) == 1]
        assert len(horizontal_lines) >= 2  # Should have positive and negative threshold lines
        
        plt.close(fig)

    def test_with_existing_axes(self):
        """Test functionality when providing existing axes."""
        fig, existing_ax = plt.subplots(figsize=(10, 8))
        
        returned_fig, returned_ax = waveform_envelope_plot(
            self.waveforms, self.x_values, ax=existing_ax
        )
        
        assert returned_fig is fig
        assert returned_ax is existing_ax
        
        plt.close(fig)

    def test_single_waveform(self):
        """Test with single waveform."""
        single_waveform = self.waveforms[:1, :]
        
        fig, ax = waveform_envelope_plot(single_waveform, self.x_values)
        
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        
        # Should still have mean line and envelope
        assert len(ax.lines) > 0
        assert len(ax.collections) > 0
        
        plt.close(fig)

    def test_empty_waveforms(self):
        """Test with empty waveforms array."""
        empty_waveforms = np.array([]).reshape(0, self.n_timepoints)
        
        with pytest.raises((ValueError, IndexError)):
            waveform_envelope_plot(empty_waveforms, self.x_values)

    def test_mismatched_dimensions(self):
        """Test with mismatched waveform and x_values dimensions."""
        wrong_x_values = np.linspace(-1, 1, self.n_timepoints + 10)
        
        with pytest.raises((ValueError, IndexError)):
            waveform_envelope_plot(self.waveforms, wrong_x_values)

    def test_axis_labels_and_legend(self):
        """Test that axis labels and legend are set correctly."""
        fig, ax = waveform_envelope_plot(self.waveforms, self.x_values)
        
        # Check axis labels
        assert ax.get_xlabel() == 'Sample'
        assert ax.get_ylabel() == 'Voltage (microvolts)'
        
        # Check legend
        legend = ax.get_legend()
        assert legend is not None
        
        plt.close(fig)

    def test_statistical_calculations(self):
        """Test that mean and std calculations are correct."""
        # Create known waveforms for testing
        test_waveforms = np.array([
            [1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
            [3, 4, 5, 6, 7]
        ])
        test_x_values = np.arange(5)
        
        fig, ax = waveform_envelope_plot(test_waveforms, test_x_values)
        
        # Get the plotted mean line
        mean_line = ax.lines[0]
        plotted_mean = mean_line.get_ydata()
        
        # Calculate expected mean
        expected_mean = np.mean(test_waveforms, axis=0)
        
        # Check that plotted mean matches expected mean
        np.testing.assert_array_almost_equal(plotted_mean, expected_mean)
        
        plt.close(fig)


class TestIntegration:
    """Integration tests for both functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.n_waveforms = 50
        self.n_timepoints = 30
        self.waveforms = np.random.randn(self.n_waveforms, self.n_timepoints) * 100
        self.x_values = np.linspace(-1, 1, self.n_timepoints)

    def test_both_functions_same_data(self):
        """Test that both functions work with the same input data."""
        # Test datashader function
        fig1, ax1 = waveforms_datashader(self.waveforms, self.x_values)
        assert isinstance(fig1, plt.Figure)
        assert isinstance(ax1, plt.Axes)
        
        # Test envelope function
        fig2, ax2 = waveform_envelope_plot(self.waveforms, self.x_values)
        assert isinstance(fig2, plt.Figure)
        assert isinstance(ax2, plt.Axes)
        
        plt.close(fig1)
        plt.close(fig2)

    def test_both_functions_with_threshold(self):
        """Test that both functions handle threshold parameter correctly."""
        threshold = 75.0
        
        # Test datashader function with threshold
        fig1, ax1 = waveforms_datashader(
            self.waveforms, self.x_values, threshold=threshold
        )
        
        # Test envelope function with threshold
        fig2, ax2 = waveform_envelope_plot(
            self.waveforms, self.x_values, threshold=threshold
        )
        
        # Both should have threshold lines
        for ax in [ax1, ax2]:
            horizontal_lines = [line for line in ax.lines 
                              if hasattr(line, 'get_ydata') and 
                              len(set(line.get_ydata())) == 1]
            assert len(horizontal_lines) >= 2
        
        plt.close(fig1)
        plt.close(fig2)


if __name__ == "__main__":
    pytest.main([__file__])
