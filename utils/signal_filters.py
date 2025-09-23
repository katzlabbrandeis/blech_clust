"""
Signal Filtering Utilities for Raw Data Viewer

This module provides signal processing utilities for filtering raw neural data
in the interactive viewer, with support for real-time filtering of data chunks.

Classes:
    SignalFilter: Main class for configurable signal filtering
"""

import numpy as np
from scipy import signal
from typing import Optional, Tuple, Union, Dict, Any
import warnings


class SignalFilter:
    """
    Configurable signal filter for neural data processing.

    Supports various filter types with efficient processing for streaming
    or chunked data loading scenarios.
    """

    def __init__(self,
                 sampling_rate: float,
                 filter_type: str = 'bandpass',
                 low_freq: Optional[float] = None,
                 high_freq: Optional[float] = None,
                 filter_order: int = 4,
                 filter_method: str = 'butterworth'):
        """
        Initialize the signal filter.

        Args:
            sampling_rate: Sampling rate in Hz
            filter_type: Type of filter ('bandpass', 'lowpass', 'highpass', 'none')
            low_freq: Low cutoff frequency in Hz (for highpass and bandpass)
            high_freq: High cutoff frequency in Hz (for lowpass and bandpass)
            filter_order: Filter order (default: 4)
            filter_method: Filter method ('butterworth', 'chebyshev1', 'chebyshev2', 'elliptic')
        """
        self.sampling_rate = sampling_rate
        self.filter_type = filter_type.lower()
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.filter_order = filter_order
        self.filter_method = filter_method.lower()

        # Validate parameters
        self._validate_parameters()

        # Design filter coefficients
        self._design_filter()

        # Initialize filter state for continuous filtering
        self._filter_state = None

    def _validate_parameters(self):
        """Validate filter parameters."""
        if self.sampling_rate <= 0:
            raise ValueError("Sampling rate must be positive")

        if self.filter_type not in ['bandpass', 'lowpass', 'highpass', 'none']:
            raise ValueError(f"Invalid filter type: {self.filter_type}")

        if self.filter_type == 'none':
            return

        nyquist = self.sampling_rate / 2

        if self.filter_type in ['highpass', 'bandpass']:
            if self.low_freq is None:
                raise ValueError(
                    f"{self.filter_type} filter requires low_freq")
            if self.low_freq <= 0 or self.low_freq >= nyquist:
                raise ValueError(
                    f"low_freq must be between 0 and {nyquist} Hz")

        if self.filter_type in ['lowpass', 'bandpass']:
            if self.high_freq is None:
                raise ValueError(
                    f"{self.filter_type} filter requires high_freq")
            if self.high_freq <= 0 or self.high_freq >= nyquist:
                raise ValueError(
                    f"high_freq must be between 0 and {nyquist} Hz")

        if self.filter_type == 'bandpass':
            if self.low_freq >= self.high_freq:
                raise ValueError(
                    "low_freq must be less than high_freq for bandpass filter")

        if self.filter_order <= 0:
            raise ValueError("Filter order must be positive")

        if self.filter_method not in ['butterworth', 'chebyshev1', 'chebyshev2', 'elliptic']:
            raise ValueError(f"Invalid filter method: {self.filter_method}")

    def _design_filter(self):
        """Design filter coefficients based on parameters."""
        if self.filter_type == 'none':
            self.b = np.array([1.0])
            self.a = np.array([1.0])
            return

        nyquist = self.sampling_rate / 2

        # Determine critical frequencies
        if self.filter_type == 'lowpass':
            critical_freq = self.high_freq / nyquist
        elif self.filter_type == 'highpass':
            critical_freq = self.low_freq / nyquist
        elif self.filter_type == 'bandpass':
            critical_freq = [self.low_freq / nyquist, self.high_freq / nyquist]

        # Design filter based on method
        if self.filter_method == 'butterworth':
            self.b, self.a = signal.butter(
                self.filter_order, critical_freq, btype=self.filter_type
            )
        elif self.filter_method == 'chebyshev1':
            # 1 dB ripple in passband
            self.b, self.a = signal.cheby1(
                self.filter_order, 1, critical_freq, btype=self.filter_type
            )
        elif self.filter_method == 'chebyshev2':
            # 40 dB attenuation in stopband
            self.b, self.a = signal.cheby2(
                self.filter_order, 40, critical_freq, btype=self.filter_type
            )
        elif self.filter_method == 'elliptic':
            # 1 dB ripple in passband, 40 dB attenuation in stopband
            self.b, self.a = signal.ellip(
                self.filter_order, 1, 40, critical_freq, btype=self.filter_type
            )

    def filter_data(self, data: np.ndarray, axis: int = -1) -> np.ndarray:
        """
        Apply filter to data using zero-phase filtering.

        Args:
            data: Input data array
            axis: Axis along which to apply filter

        Returns:
            Filtered data array
        """
        if self.filter_type == 'none':
            return data.copy()

        # Use filtfilt for zero-phase filtering
        try:
            filtered_data = signal.filtfilt(self.b, self.a, data, axis=axis)
        except ValueError as e:
            warnings.warn(f"Filtering failed: {e}. Returning original data.")
            return data.copy()

        return filtered_data

    def filter_streaming(self, data: np.ndarray, reset_state: bool = False) -> np.ndarray:
        """
        Apply filter to streaming data maintaining filter state.

        This method is useful for real-time or chunked data processing
        where filter state needs to be maintained between chunks.

        Args:
            data: Input data chunk (1D array)
            reset_state: Whether to reset filter state

        Returns:
            Filtered data chunk
        """
        if self.filter_type == 'none':
            return data.copy()

        if reset_state or self._filter_state is None:
            # Initialize filter state
            self._filter_state = signal.lfilter_zi(self.b, self.a)
            if data.ndim > 1:
                # Expand state for multiple channels
                self._filter_state = np.tile(
                    self._filter_state[:, np.newaxis],
                    (1, data.shape[0])
                )

        # Apply filter with state
        if data.ndim == 1:
            filtered_data, self._filter_state = signal.lfilter(
                self.b, self.a, data, zi=self._filter_state
            )
        else:
            # Handle multiple channels
            filtered_data = np.zeros_like(data)
            for i in range(data.shape[0]):
                filtered_data[i], self._filter_state[:, i] = signal.lfilter(
                    self.b, self.a, data[i], zi=self._filter_state[:, i]
                )

        return filtered_data

    def get_frequency_response(self, n_points: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get frequency response of the filter.

        Args:
            n_points: Number of points in frequency response

        Returns:
            Tuple of (frequencies, magnitude_response)
        """
        if self.filter_type == 'none':
            freqs = np.linspace(0, self.sampling_rate/2, n_points)
            magnitude = np.ones(n_points)
            return freqs, magnitude

        w, h = signal.freqz(self.b, self.a, worN=n_points)
        freqs = w * self.sampling_rate / (2 * np.pi)
        magnitude = np.abs(h)

        return freqs, magnitude

    def get_filter_info(self) -> Dict[str, Any]:
        """
        Get information about the current filter configuration.

        Returns:
            Dictionary with filter information
        """
        info = {
            'filter_type': self.filter_type,
            'filter_method': self.filter_method,
            'filter_order': self.filter_order,
            'sampling_rate': self.sampling_rate,
            'low_freq': self.low_freq,
            'high_freq': self.high_freq
        }

        if self.filter_type != 'none':
            # Add filter stability information
            info['is_stable'] = np.all(np.abs(np.roots(self.a)) < 1)

            # Add frequency response at key points
            freqs, magnitude = self.get_frequency_response(1024)
            info['dc_gain'] = magnitude[0]
            info['nyquist_gain'] = magnitude[-1]

        return info

    def update_parameters(self, **kwargs):
        """
        Update filter parameters and redesign filter.

        Args:
            **kwargs: Filter parameters to update
        """
        # Update parameters
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")

        # Validate and redesign filter
        self._validate_parameters()
        self._design_filter()

        # Reset filter state
        self._filter_state = None

    def __repr__(self):
        """String representation."""
        if self.filter_type == 'none':
            return "SignalFilter(type='none')"

        freq_str = ""
        if self.low_freq is not None:
            freq_str += f"low={self.low_freq}Hz"
        if self.high_freq is not None:
            if freq_str:
                freq_str += ", "
            freq_str += f"high={self.high_freq}Hz"

        return f"SignalFilter(type='{self.filter_type}', {freq_str}, order={self.filter_order})"


class FilterBank:
    """
    Collection of commonly used filters for neural data.
    """

    @staticmethod
    def create_spike_filter(sampling_rate: float) -> SignalFilter:
        """
        Create a typical spike detection filter (300-3000 Hz bandpass).

        Args:
            sampling_rate: Sampling rate in Hz

        Returns:
            Configured SignalFilter
        """
        return SignalFilter(
            sampling_rate=sampling_rate,
            filter_type='bandpass',
            low_freq=300,
            high_freq=3000,
            filter_order=4
        )

    @staticmethod
    def create_lfp_filter(sampling_rate: float) -> SignalFilter:
        """
        Create a typical LFP filter (1-300 Hz bandpass).

        Args:
            sampling_rate: Sampling rate in Hz

        Returns:
            Configured SignalFilter
        """
        return SignalFilter(
            sampling_rate=sampling_rate,
            filter_type='bandpass',
            low_freq=1,
            high_freq=300,
            filter_order=4
        )

    @staticmethod
    def create_emg_filter(sampling_rate: float) -> SignalFilter:
        """
        Create a typical EMG filter (10-500 Hz bandpass).

        Args:
            sampling_rate: Sampling rate in Hz

        Returns:
            Configured SignalFilter
        """
        return SignalFilter(
            sampling_rate=sampling_rate,
            filter_type='bandpass',
            low_freq=10,
            high_freq=500,
            filter_order=4
        )

    @staticmethod
    def create_notch_filter(sampling_rate: float, notch_freq: float = 60.0, quality: float = 30.0) -> 'NotchFilter':
        """
        Create a notch filter for line noise removal.

        Args:
            sampling_rate: Sampling rate in Hz
            notch_freq: Frequency to notch out (default: 60 Hz)
            quality: Quality factor (higher = narrower notch)

        Returns:
            Configured NotchFilter
        """
        return NotchFilter(sampling_rate, notch_freq, quality)


class NotchFilter:
    """
    Notch filter for removing line noise (e.g., 60 Hz).
    """

    def __init__(self, sampling_rate: float, notch_freq: float = 60.0, quality: float = 30.0):
        """
        Initialize notch filter.

        Args:
            sampling_rate: Sampling rate in Hz
            notch_freq: Frequency to notch out
            quality: Quality factor (higher = narrower notch)
        """
        self.sampling_rate = sampling_rate
        self.notch_freq = notch_freq
        self.quality = quality

        # Design notch filter
        self.b, self.a = signal.iirnotch(
            notch_freq / (sampling_rate / 2), quality
        )

    def filter_data(self, data: np.ndarray, axis: int = -1) -> np.ndarray:
        """
        Apply notch filter to data.

        Args:
            data: Input data array
            axis: Axis along which to apply filter

        Returns:
            Filtered data array
        """
        return signal.filtfilt(self.b, self.a, data, axis=axis)

    def __repr__(self):
        """String representation."""
        return f"NotchFilter(freq={self.notch_freq}Hz, Q={self.quality})"


def apply_multiple_filters(data: np.ndarray,
                           filters: list,
                           axis: int = -1) -> np.ndarray:
    """
    Apply multiple filters in sequence.

    Args:
        data: Input data array
        filters: List of filter objects with filter_data method
        axis: Axis along which to apply filters

    Returns:
        Filtered data array
    """
    filtered_data = data.copy()

    for filt in filters:
        filtered_data = filt.filter_data(filtered_data, axis=axis)

    return filtered_data
