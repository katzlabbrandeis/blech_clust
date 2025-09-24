"""
Unit tests for SignalFilter classes.
"""

from signal_filters import SignalFilter, FilterBank, NotchFilter, apply_multiple_filters
import unittest
import numpy as np
from scipy import signal
import warnings

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))


class TestSignalFilter(unittest.TestCase):
    """Test cases for SignalFilter class."""

    def setUp(self):
        """Set up test fixtures."""
        self.sampling_rate = 30000.0
        self.n_samples = int(self.sampling_rate * 1)  # 1 second of data

        # Create test signal with multiple frequency components
        t = np.arange(self.n_samples) / self.sampling_rate
        self.test_signal = (
            np.sin(2 * np.pi * 10 * t) +      # 10 Hz
            np.sin(2 * np.pi * 100 * t) +     # 100 Hz
            np.sin(2 * np.pi * 1000 * t) +    # 1000 Hz
            np.random.normal(0, 0.1, len(t))  # noise
        )

    def test_initialization_bandpass(self):
        """Test SignalFilter initialization with bandpass filter."""
        filt = SignalFilter(
            sampling_rate=self.sampling_rate,
            filter_type='bandpass',
            low_freq=50,
            high_freq=500,
            filter_order=4
        )

        self.assertEqual(filt.sampling_rate, self.sampling_rate)
        self.assertEqual(filt.filter_type, 'bandpass')
        self.assertEqual(filt.low_freq, 50)
        self.assertEqual(filt.high_freq, 500)
        self.assertEqual(filt.filter_order, 4)

    def test_initialization_lowpass(self):
        """Test SignalFilter initialization with lowpass filter."""
        filt = SignalFilter(
            sampling_rate=self.sampling_rate,
            filter_type='lowpass',
            high_freq=1000
        )

        self.assertEqual(filt.filter_type, 'lowpass')
        self.assertEqual(filt.high_freq, 1000)
        self.assertIsNone(filt.low_freq)

    def test_initialization_highpass(self):
        """Test SignalFilter initialization with highpass filter."""
        filt = SignalFilter(
            sampling_rate=self.sampling_rate,
            filter_type='highpass',
            low_freq=100
        )

        self.assertEqual(filt.filter_type, 'highpass')
        self.assertEqual(filt.low_freq, 100)
        self.assertIsNone(filt.high_freq)

    def test_initialization_none(self):
        """Test SignalFilter initialization with no filtering."""
        filt = SignalFilter(
            sampling_rate=self.sampling_rate,
            filter_type='none'
        )

        self.assertEqual(filt.filter_type, 'none')
        np.testing.assert_array_equal(filt.b, [1.0])
        np.testing.assert_array_equal(filt.a, [1.0])

    def test_initialization_invalid_parameters(self):
        """Test SignalFilter initialization with invalid parameters."""
        # Invalid sampling rate
        with self.assertRaises(ValueError):
            SignalFilter(sampling_rate=0, filter_type='bandpass')

        # Invalid filter type
        with self.assertRaises(ValueError):
            SignalFilter(sampling_rate=self.sampling_rate,
                         filter_type='invalid')

        # Missing frequency for bandpass
        with self.assertRaises(ValueError):
            SignalFilter(
                sampling_rate=self.sampling_rate,
                filter_type='bandpass',
                low_freq=100
            )

        # Frequency above Nyquist
        with self.assertRaises(ValueError):
            SignalFilter(
                sampling_rate=self.sampling_rate,
                filter_type='lowpass',
                high_freq=self.sampling_rate  # Equal to Nyquist
            )

        # Invalid frequency order for bandpass
        with self.assertRaises(ValueError):
            SignalFilter(
                sampling_rate=self.sampling_rate,
                filter_type='bandpass',
                low_freq=500,
                high_freq=100  # low > high
            )

    def test_filter_data_none(self):
        """Test filtering with no filter."""
        filt = SignalFilter(
            sampling_rate=self.sampling_rate, filter_type='none')

        filtered_data = filt.filter_data(self.test_signal)

        np.testing.assert_array_equal(filtered_data, self.test_signal)

    def test_filter_data_lowpass(self):
        """Test lowpass filtering."""
        filt = SignalFilter(
            sampling_rate=self.sampling_rate,
            filter_type='lowpass',
            high_freq=200
        )

        filtered_data = filt.filter_data(self.test_signal)

        # Check that high frequencies are attenuated
        freqs, psd_orig = signal.welch(self.test_signal, self.sampling_rate)
        freqs, psd_filt = signal.welch(filtered_data, self.sampling_rate)

        # Find indices for frequencies above cutoff
        high_freq_idx = freqs > 500

        # High frequencies should be more attenuated in filtered signal
        if np.any(high_freq_idx):
            self.assertLess(
                np.mean(psd_filt[high_freq_idx]),
                np.mean(psd_orig[high_freq_idx])
            )

    def test_filter_data_highpass(self):
        """Test highpass filtering."""
        filt = SignalFilter(
            sampling_rate=self.sampling_rate,
            filter_type='highpass',
            low_freq=50
        )

        filtered_data = filt.filter_data(self.test_signal)

        # Check that low frequencies are attenuated
        freqs, psd_orig = signal.welch(self.test_signal, self.sampling_rate)
        freqs, psd_filt = signal.welch(filtered_data, self.sampling_rate)

        # Find indices for frequencies below cutoff
        low_freq_idx = freqs < 25

        # Low frequencies should be more attenuated in filtered signal
        if np.any(low_freq_idx):
            self.assertLess(
                np.mean(psd_filt[low_freq_idx]),
                np.mean(psd_orig[low_freq_idx])
            )

    def test_filter_data_bandpass(self):
        """Test bandpass filtering."""
        filt = SignalFilter(
            sampling_rate=self.sampling_rate,
            filter_type='bandpass',
            low_freq=50,
            high_freq=500
        )

        filtered_data = filt.filter_data(self.test_signal)

        # Check that frequencies outside passband are attenuated
        freqs, psd_orig = signal.welch(self.test_signal, self.sampling_rate)
        freqs, psd_filt = signal.welch(filtered_data, self.sampling_rate)

        # Frequencies below low cutoff should be attenuated
        low_freq_idx = freqs < 25
        if np.any(low_freq_idx):
            self.assertLess(
                np.mean(psd_filt[low_freq_idx]),
                np.mean(psd_orig[low_freq_idx])
            )

        # Frequencies above high cutoff should be attenuated
        high_freq_idx = freqs > 1000
        if np.any(high_freq_idx):
            self.assertLess(
                np.mean(psd_filt[high_freq_idx]),
                np.mean(psd_orig[high_freq_idx])
            )

    def test_filter_data_2d(self):
        """Test filtering 2D data (multiple channels)."""
        # Create 2D test data (3 channels)
        test_data_2d = np.tile(self.test_signal, (3, 1))
        test_data_2d[1] *= 2  # Scale second channel
        test_data_2d[2] *= 0.5  # Scale third channel

        filt = SignalFilter(
            sampling_rate=self.sampling_rate,
            filter_type='lowpass',
            high_freq=200
        )

        filtered_data = filt.filter_data(test_data_2d, axis=1)

        self.assertEqual(filtered_data.shape, test_data_2d.shape)

        # Each channel should be filtered independently
        for i in range(3):
            single_filtered = filt.filter_data(test_data_2d[i])
            np.testing.assert_array_almost_equal(
                filtered_data[i], single_filtered, decimal=10
            )

    def test_filter_streaming(self):
        """Test streaming filter functionality."""
        filt = SignalFilter(
            sampling_rate=self.sampling_rate,
            filter_type='lowpass',
            high_freq=200
        )

        # Split signal into chunks
        chunk_size = len(self.test_signal) // 4
        chunks = [
            self.test_signal[i:i+chunk_size]
            for i in range(0, len(self.test_signal), chunk_size)
        ]

        # Filter chunks sequentially
        filtered_chunks = []
        for i, chunk in enumerate(chunks):
            reset_state = (i == 0)
            filtered_chunk = filt.filter_streaming(
                chunk, reset_state=reset_state)
            filtered_chunks.append(filtered_chunk)

        # Concatenate filtered chunks
        streaming_result = np.concatenate(filtered_chunks)

        # Compare with batch filtering (should be similar but not identical due to edge effects)
        batch_result = filt.filter_data(self.test_signal)

        # Results should be reasonably close (allowing for edge effects)
        correlation = np.corrcoef(
            streaming_result[:len(batch_result)], batch_result)[0, 1]
        self.assertGreater(correlation, 0.9)

    def test_get_frequency_response(self):
        """Test getting frequency response."""
        filt = SignalFilter(
            sampling_rate=self.sampling_rate,
            filter_type='lowpass',
            high_freq=1000
        )

        freqs, magnitude = filt.get_frequency_response()

        self.assertEqual(len(freqs), len(magnitude))
        self.assertAlmostEqual(freqs[0], 0)
        self.assertAlmostEqual(freqs[-1], self.sampling_rate / 2, places=0)

        # Magnitude should be close to 1 at low frequencies
        low_freq_idx = freqs < 100
        self.assertGreater(np.mean(magnitude[low_freq_idx]), 0.9)

        # Magnitude should be lower at high frequencies
        high_freq_idx = freqs > 5000
        self.assertLess(np.mean(magnitude[high_freq_idx]), 0.5)

    def test_get_frequency_response_none(self):
        """Test frequency response for no filter."""
        filt = SignalFilter(
            sampling_rate=self.sampling_rate, filter_type='none')

        freqs, magnitude = filt.get_frequency_response()

        # All frequencies should have magnitude 1
        np.testing.assert_array_almost_equal(
            magnitude, np.ones_like(magnitude))

    def test_get_filter_info(self):
        """Test getting filter information."""
        filt = SignalFilter(
            sampling_rate=self.sampling_rate,
            filter_type='bandpass',
            low_freq=300,
            high_freq=3000,
            filter_order=4
        )

        info = filt.get_filter_info()

        self.assertEqual(info['filter_type'], 'bandpass')
        self.assertEqual(info['low_freq'], 300)
        self.assertEqual(info['high_freq'], 3000)
        self.assertEqual(info['filter_order'], 4)
        self.assertEqual(info['sampling_rate'], self.sampling_rate)
        self.assertIn('is_stable', info)
        self.assertTrue(info['is_stable'])

    def test_update_parameters(self):
        """Test updating filter parameters."""
        filt = SignalFilter(
            sampling_rate=self.sampling_rate,
            filter_type='lowpass',
            high_freq=1000
        )

        # Update to bandpass
        filt.update_parameters(
            filter_type='bandpass',
            low_freq=300,
            high_freq=3000
        )

        self.assertEqual(filt.filter_type, 'bandpass')
        self.assertEqual(filt.low_freq, 300)
        self.assertEqual(filt.high_freq, 3000)

    def test_update_parameters_invalid(self):
        """Test updating with invalid parameters."""
        filt = SignalFilter(
            sampling_rate=self.sampling_rate,
            filter_type='lowpass',
            high_freq=1000
        )

        with self.assertRaises(ValueError):
            filt.update_parameters(invalid_param=123)

    def test_repr(self):
        """Test string representation."""
        filt = SignalFilter(
            sampling_rate=self.sampling_rate,
            filter_type='bandpass',
            low_freq=300,
            high_freq=3000
        )

        repr_str = repr(filt)
        self.assertIn('SignalFilter', repr_str)
        self.assertIn('bandpass', repr_str)
        self.assertIn('300', repr_str)
        self.assertIn('3000', repr_str)


class TestFilterBank(unittest.TestCase):
    """Test cases for FilterBank class."""

    def setUp(self):
        """Set up test fixtures."""
        self.sampling_rate = 30000.0

    def test_create_spike_filter(self):
        """Test creating spike filter."""
        filt = FilterBank.create_spike_filter(self.sampling_rate)

        self.assertEqual(filt.filter_type, 'bandpass')
        self.assertEqual(filt.low_freq, 300)
        self.assertEqual(filt.high_freq, 3000)

    def test_create_lfp_filter(self):
        """Test creating LFP filter."""
        filt = FilterBank.create_lfp_filter(self.sampling_rate)

        self.assertEqual(filt.filter_type, 'bandpass')
        self.assertEqual(filt.low_freq, 1)
        self.assertEqual(filt.high_freq, 300)

    def test_create_emg_filter(self):
        """Test creating EMG filter."""
        filt = FilterBank.create_emg_filter(self.sampling_rate)

        self.assertEqual(filt.filter_type, 'bandpass')
        self.assertEqual(filt.low_freq, 10)
        self.assertEqual(filt.high_freq, 500)

    def test_create_notch_filter(self):
        """Test creating notch filter."""
        filt = FilterBank.create_notch_filter(self.sampling_rate)

        self.assertIsInstance(filt, NotchFilter)
        self.assertEqual(filt.notch_freq, 60.0)
        self.assertEqual(filt.quality, 30.0)


class TestNotchFilter(unittest.TestCase):
    """Test cases for NotchFilter class."""

    def setUp(self):
        """Set up test fixtures."""
        self.sampling_rate = 30000.0
        self.n_samples = int(self.sampling_rate * 1)  # 1 second

        # Create test signal with 60 Hz noise
        t = np.arange(self.n_samples) / self.sampling_rate
        self.test_signal = (
            np.sin(2 * np.pi * 10 * t) +      # 10 Hz signal
            np.sin(2 * np.pi * 60 * t) * 2 +  # 60 Hz noise (stronger)
            np.sin(2 * np.pi * 100 * t)       # 100 Hz signal
        )

    def test_initialization(self):
        """Test NotchFilter initialization."""
        filt = NotchFilter(self.sampling_rate, notch_freq=60.0, quality=30.0)

        self.assertEqual(filt.sampling_rate, self.sampling_rate)
        self.assertEqual(filt.notch_freq, 60.0)
        self.assertEqual(filt.quality, 30.0)

    def test_filter_data(self):
        """Test notch filtering."""
        filt = NotchFilter(self.sampling_rate, notch_freq=60.0, quality=30.0)

        filtered_data = filt.filter_data(self.test_signal)

        # Check that 60 Hz component is attenuated
        freqs, psd_orig = signal.welch(self.test_signal, self.sampling_rate)
        freqs, psd_filt = signal.welch(filtered_data, self.sampling_rate)

        # Find index closest to 60 Hz
        freq_60_idx = np.argmin(np.abs(freqs - 60))

        # 60 Hz should be attenuated
        self.assertLess(psd_filt[freq_60_idx], psd_orig[freq_60_idx])

        # Other frequencies should be relatively preserved
        freq_10_idx = np.argmin(np.abs(freqs - 10))
        freq_100_idx = np.argmin(np.abs(freqs - 100))

        # Allow some attenuation but not too much
        self.assertGreater(psd_filt[freq_10_idx], psd_orig[freq_10_idx] * 0.5)
        self.assertGreater(psd_filt[freq_100_idx],
                           psd_orig[freq_100_idx] * 0.5)

    def test_repr(self):
        """Test string representation."""
        filt = NotchFilter(self.sampling_rate, notch_freq=60.0, quality=30.0)

        repr_str = repr(filt)
        self.assertIn('NotchFilter', repr_str)
        self.assertIn('60', repr_str)
        self.assertIn('30', repr_str)


class TestApplyMultipleFilters(unittest.TestCase):
    """Test cases for apply_multiple_filters function."""

    def setUp(self):
        """Set up test fixtures."""
        self.sampling_rate = 30000.0
        self.n_samples = int(self.sampling_rate * 1)  # 1 second

        # Create test signal
        t = np.arange(self.n_samples) / self.sampling_rate
        self.test_signal = (
            np.sin(2 * np.pi * 10 * t) +
            np.sin(2 * np.pi * 60 * t) +
            np.sin(2 * np.pi * 1000 * t) +
            np.random.normal(0, 0.1, len(t))
        )

    def test_apply_multiple_filters(self):
        """Test applying multiple filters in sequence."""
        # Create filters
        bandpass_filter = SignalFilter(
            sampling_rate=self.sampling_rate,
            filter_type='bandpass',
            low_freq=50,
            high_freq=500
        )

        notch_filter = NotchFilter(self.sampling_rate, notch_freq=60.0)

        filters = [bandpass_filter, notch_filter]

        # Apply filters
        filtered_data = apply_multiple_filters(self.test_signal, filters)

        # Check that result is different from original
        self.assertFalse(np.array_equal(filtered_data, self.test_signal))

        # Check that applying filters sequentially gives same result
        temp_data = bandpass_filter.filter_data(self.test_signal)
        expected_data = notch_filter.filter_data(temp_data)

        np.testing.assert_array_almost_equal(filtered_data, expected_data)

    def test_apply_multiple_filters_empty(self):
        """Test applying empty filter list."""
        filtered_data = apply_multiple_filters(self.test_signal, [])

        np.testing.assert_array_equal(filtered_data, self.test_signal)


if __name__ == '__main__':
    unittest.main()
