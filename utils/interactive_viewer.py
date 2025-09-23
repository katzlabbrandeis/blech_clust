"""
Interactive Plotting Interface for Raw Data Viewer

This module provides an interactive plotting interface for visualizing raw neural data
with time navigation, filtering, and threshold display capabilities.

Classes:
    InteractivePlotter: Main interactive plotting widget
    ViewerControls: Control panel for viewer parameters
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox, CheckButtons
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from matplotlib.widgets import RadioButtons
from typing import Optional, Callable, Dict, Any, Tuple, List
import warnings

try:
    from .raw_data_loader import RawDataLoader
    from .signal_filters import SignalFilter
except ImportError:
    # Handle relative imports when running as script
    from raw_data_loader import RawDataLoader
    from signal_filters import SignalFilter


class InteractivePlotter:
    """
    Interactive plotting widget for raw neural data visualization.

    Features:
    - Real-time plotting of single or multiple channels
    - Time navigation (scrolling and jumping)
    - Configurable time window display
    - Threshold line overlay
    - Zoom and pan functionality
    - Filter application
    """

    def __init__(self,
                 data_loader: RawDataLoader,
                 initial_channel: str = None,
                 initial_group: str = 'raw',
                 window_duration: float = 10.0,
                 update_callback: Optional[Callable] = None):
        """
        Initialize the interactive plotter.

        Args:
            data_loader: RawDataLoader instance
            initial_channel: Initial channel to display
            initial_group: Initial data group
            window_duration: Initial window duration in seconds
            update_callback: Optional callback function for updates
        """
        self.data_loader = data_loader
        self.current_group = initial_group
        self.window_duration = window_duration
        self.update_callback = update_callback

        # Get available channels
        self.available_channels = self.data_loader.get_channel_list(
            initial_group)
        if not self.available_channels:
            raise ValueError(f"No channels found in group '{initial_group}'")

        self.current_channel = initial_channel or self.available_channels[0]

        # Initialize display parameters
        self.current_time = 0.0
        self.threshold_value = None
        self.show_threshold = False
        self.auto_scale = True
        self.y_scale_factor = 1.0
        self.manual_ylims = None
        self.lowpass_freq = 3000.0
        self.highpass_freq = 300.0
        self.data_conversion_factor = 0.6745  # Convert to microvolts

        # Snippet extraction parameters
        self.show_snippets = False
        self.snippet_before_ms = 0.5  # ms before threshold crossing
        self.snippet_after_ms = 1.0   # ms after threshold crossing
        self.max_snippets = 50        # maximum snippets to display
        self.current_snippets = []    # extracted snippet waveforms
        self.snippet_times = []       # times of snippet peaks
        self.snippet_polarities = []  # polarity of each snippet

        # Get total duration
        self.total_duration = self.data_loader.get_channel_duration(
            self.current_channel, self.current_group
        )

        # Initialize filter
        self.signal_filter = SignalFilter(
            sampling_rate=self.data_loader.sampling_rate,
            filter_type='none'
        )

        # Create the plot
        self._create_plot()
        self._create_controls()
        self._update_display()

    def _create_plot(self):
        """Create the main plot and figure."""
        # Create figure with subplots for plot and controls
        self.fig = plt.figure(figsize=(16, 14))

        # Main plot area - adjust for snippet subplot
        self.ax_main = plt.subplot2grid((8, 6), (0, 0), colspan=6, rowspan=3)
        self.ax_main.set_xlabel('Time (s)')
        self.ax_main.set_ylabel('Amplitude (µV)')
        self.ax_main.grid(True, alpha=0.3)

        # Snippet plot area
        self.ax_snippets = plt.subplot2grid(
            (8, 6), (3, 0), colspan=6, rowspan=2)
        self.ax_snippets.set_xlabel('Time (ms)')
        self.ax_snippets.set_ylabel('Amplitude (µV)')
        self.ax_snippets.set_title('Extracted Waveform Snippets')
        self.ax_snippets.grid(True, alpha=0.3)
        self.ax_snippets.set_visible(False)  # Initially hidden

        # Initialize empty line objects
        self.data_line, = self.ax_main.plot(
            [], [], 'b-', linewidth=0.8, label='Raw Data')
        self.threshold_line = self.ax_main.axhline(y=0, color='r', linestyle='--',
                                                   linewidth=2, alpha=0.7, visible=False,
                                                   label='Threshold')

        # Add legend
        self.ax_main.legend(loc='upper right')

        # Set up interactive features
        self.ax_main.set_xlim(0, self.window_duration)

        # Connect mouse events
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)

    def _create_controls(self):
        """Create control widgets."""
        # Time navigation slider - Row 5
        ax_time = plt.subplot2grid((8, 6), (5, 0), colspan=3)
        self.time_slider = Slider(
            ax_time, 'Time (s)', 0, max(
                0, self.total_duration - self.window_duration),
            valinit=self.current_time, valfmt='%.1f'
        )
        self.time_slider.on_changed(self._on_time_change)

        # Window duration control
        ax_window = plt.subplot2grid((8, 6), (5, 3))
        self.window_box = TextBox(
            ax_window, 'Window (s)', initial=str(self.window_duration))
        self.window_box.on_submit(self._on_window_change)

        # Threshold control
        ax_threshold = plt.subplot2grid((8, 6), (5, 4))
        self.threshold_box = TextBox(ax_threshold, 'Threshold', initial='')
        self.threshold_box.on_submit(self._on_threshold_change)

        # Channel dropdown (simplified as radio buttons for now)
        ax_channel_select = plt.subplot2grid((8, 6), (5, 5))
        # Show first few channels in radio buttons
        visible_channels = self.available_channels[:min(
            5, len(self.available_channels))]
        self.channel_radio = RadioButtons(ax_channel_select, visible_channels)
        self.channel_radio.on_clicked(self._on_channel_radio_change)

        # Filter frequency controls - Row 6
        ax_lowpass = plt.subplot2grid((8, 6), (6, 0))
        self.lowpass_box = TextBox(
            ax_lowpass, 'Lowpass (Hz)', initial=str(self.lowpass_freq))
        self.lowpass_box.on_submit(self._on_lowpass_change)

        ax_highpass = plt.subplot2grid((8, 6), (6, 1))
        self.highpass_box = TextBox(
            ax_highpass, 'Highpass (Hz)', initial=str(self.highpass_freq))
        self.highpass_box.on_submit(self._on_highpass_change)

        # Y-limits controls
        ax_ymin = plt.subplot2grid((8, 6), (6, 2))
        self.ymin_box = TextBox(ax_ymin, 'Y Min', initial='')
        self.ymin_box.on_submit(self._on_ylim_change)

        ax_ymax = plt.subplot2grid((8, 6), (6, 3))
        self.ymax_box = TextBox(ax_ymax, 'Y Max', initial='')
        self.ymax_box.on_submit(self._on_ylim_change)

        # Snippet controls - Row 7
        ax_snippet_before = plt.subplot2grid((8, 6), (7, 0))
        self.snippet_before_box = TextBox(
            ax_snippet_before, 'Before (ms)', initial=str(self.snippet_before_ms))
        self.snippet_before_box.on_submit(self._on_snippet_before_change)

        ax_snippet_after = plt.subplot2grid((8, 6), (7, 1))
        self.snippet_after_box = TextBox(
            ax_snippet_after, 'After (ms)', initial=str(self.snippet_after_ms))
        self.snippet_after_box.on_submit(self._on_snippet_after_change)

        ax_max_snippets = plt.subplot2grid((8, 6), (7, 2))
        self.max_snippets_box = TextBox(
            ax_max_snippets, 'Max Snippets', initial=str(self.max_snippets))
        self.max_snippets_box.on_submit(self._on_max_snippets_change)

        # Show snippets checkbox
        ax_show_snippets = plt.subplot2grid((8, 6), (7, 3))
        self.show_snippets_check = CheckButtons(
            ax_show_snippets, ['Show Snippets'], [self.show_snippets])
        self.show_snippets_check.on_clicked(self._on_show_snippets_change)

        # Add control buttons (will be created in separate method)
        self._create_buttons()

    def _create_buttons(self):
        """Create control buttons."""
        # Create button axes - positioned at bottom
        button_height = 0.03
        button_width = 0.06
        y_pos = 0.01

        # Navigation buttons
        ax_prev = plt.axes([0.02, y_pos, button_width, button_height])
        ax_next = plt.axes([0.09, y_pos, button_width, button_height])
        ax_jump = plt.axes([0.16, y_pos, button_width, button_height])

        self.btn_prev = Button(ax_prev, '← Prev')
        self.btn_next = Button(ax_next, 'Next →')
        self.btn_jump = Button(ax_jump, 'Jump to')

        self.btn_prev.on_clicked(self._on_prev_click)
        self.btn_next.on_clicked(self._on_next_click)
        self.btn_jump.on_clicked(self._on_jump_click)

        # Filter and display buttons
        ax_filter = plt.axes([0.25, y_pos, button_width, button_height])
        ax_reset = plt.axes([0.32, y_pos, button_width, button_height])
        ax_autoscale = plt.axes([0.39, y_pos, button_width, button_height])

        self.btn_filter = Button(ax_filter, 'Apply Filter')
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_autoscale = Button(ax_autoscale, 'Auto Y')

        self.btn_filter.on_clicked(self._on_filter_click)
        self.btn_reset.on_clicked(self._on_reset_click)
        self.btn_autoscale.on_clicked(self._on_autoscale_click)

        # Additional control buttons
        ax_all_channels = plt.axes(
            [0.46, y_pos, button_width * 1.2, button_height])
        self.btn_all_channels = Button(ax_all_channels, 'All Channels')
        self.btn_all_channels.on_clicked(self._on_all_channels_click)

    def _on_snippet_before_change(self, text):
        """Handle snippet before time change."""
        try:
            time_ms = float(text)
            if time_ms > 0:
                self.snippet_before_ms = time_ms
                if self.show_snippets:
                    self._update_display()
        except ValueError:
            pass

    def _on_snippet_after_change(self, text):
        """Handle snippet after time change."""
        try:
            time_ms = float(text)
            if time_ms > 0:
                self.snippet_after_ms = time_ms
                if self.show_snippets:
                    self._update_display()
        except ValueError:
            pass

    def _on_max_snippets_change(self, text):
        """Handle max snippets change."""
        try:
            max_count = int(text)
            if max_count > 0:
                self.max_snippets = max_count
                if self.show_snippets:
                    self._update_display()
        except ValueError:
            pass

    def _on_show_snippets_change(self, label):
        """Handle show snippets checkbox change."""
        self.show_snippets = not self.show_snippets
        self.ax_snippets.set_visible(self.show_snippets)
        if self.show_snippets:
            self._update_display()
        else:
            self.ax_snippets.clear()
            self.ax_snippets.set_xlabel('Time (ms)')
            self.ax_snippets.set_ylabel('Amplitude (µV)')
            self.ax_snippets.set_title('Extracted Waveform Snippets')
            self.ax_snippets.grid(True, alpha=0.3)
        self.fig.canvas.draw()

    def _extract_snippets(self, data, time_array, threshold):
        """
        Extract waveform snippets that cross the threshold.

        Based on extract_waveforms_abu from utils/clustering.py
        """
        if threshold is None or len(data) == 0:
            return [], [], []

        sampling_rate = self.data_loader.sampling_rate

        # Convert snippet times from ms to samples
        before_samples = int((self.snippet_before_ms / 1000.0) * sampling_rate)
        after_samples = int((self.snippet_after_ms / 1000.0) * sampling_rate)

        # Find mean for threshold detection
        mean_val = np.mean(data)

        # Find threshold crossings
        negative_crossings = np.where(data <= mean_val - threshold)[0]
        positive_crossings = np.where(data >= mean_val + threshold)[0]

        # Find breaks in threshold crossings (separate events)
        def find_crossing_events(crossings):
            if len(crossings) == 0:
                return []
            changes = np.concatenate(
                ([0], np.where(np.diff(crossings) > 1)[0] + 1))
            events = [(crossings[changes[i]], crossings[changes[i+1]-1] if i+1 < len(changes) else crossings[-1])
                      for i in range(len(changes)-1)]
            if len(changes) > 0:
                events.append((crossings[changes[-1]], crossings[-1]))
            return events

        neg_events = find_crossing_events(negative_crossings)
        pos_events = find_crossing_events(positive_crossings)

        # Find extrema for each event
        minima = [np.argmin(data[start:end+1]) +
                  start for start, end in neg_events]
        maxima = [np.argmax(data[start:end+1]) +
                  start for start, end in pos_events]

        # Combine and sort by time
        spike_indices = np.array(minima + maxima)
        polarities = np.array([-1] * len(minima) + [1] * len(maxima))

        if len(spike_indices) == 0:
            return [], [], []

        # Sort by time
        sort_order = np.argsort(spike_indices)
        spike_indices = spike_indices[sort_order]
        polarities = polarities[sort_order]

        # Extract snippets
        snippets = []
        snippet_times = []
        snippet_polarities = []

        for i, (spike_idx, polarity) in enumerate(zip(spike_indices, polarities)):
            # Check if we have enough data around the spike
            start_idx = spike_idx - before_samples
            end_idx = spike_idx + after_samples

            if start_idx >= 0 and end_idx < len(data):
                snippet = data[start_idx:end_idx]
                snippets.append(snippet)

                # Convert spike time to relative time in current window
                spike_time = time_array[spike_idx] if spike_idx < len(
                    time_array) else time_array[-1]
                snippet_times.append(spike_time)
                snippet_polarities.append(polarity)

                # Limit number of snippets
                if len(snippets) >= self.max_snippets:
                    break

        return snippets, snippet_times, snippet_polarities

    def _plot_snippets(self):
        """Plot extracted waveform snippets."""
        if not self.show_snippets or len(self.current_snippets) == 0:
            return

        self.ax_snippets.clear()

        # Create time axis for snippets (in ms)
        sampling_rate = self.data_loader.sampling_rate
        before_samples = int((self.snippet_before_ms / 1000.0) * sampling_rate)
        after_samples = int((self.snippet_after_ms / 1000.0) * sampling_rate)
        total_samples = before_samples + after_samples

        # Time axis in ms, centered at 0 (threshold crossing)
        snippet_time_ms = np.linspace(-self.snippet_before_ms,
                                      self.snippet_after_ms, total_samples)

        # Plot snippets with different colors for positive/negative
        neg_snippets = []
        pos_snippets = []

        for snippet, polarity in zip(self.current_snippets, self.snippet_polarities):
            if len(snippet) == total_samples:  # Ensure correct length
                if polarity == -1:
                    neg_snippets.append(snippet)
                    self.ax_snippets.plot(
                        snippet_time_ms, snippet, 'b-', alpha=0.3, linewidth=0.5)
                else:
                    pos_snippets.append(snippet)
                    self.ax_snippets.plot(
                        snippet_time_ms, snippet, 'r-', alpha=0.3, linewidth=0.5)

        # Plot average waveforms
        if neg_snippets:
            neg_avg = np.mean(neg_snippets, axis=0)
            self.ax_snippets.plot(snippet_time_ms, neg_avg, 'b-',
                                  linewidth=2, label=f'Negative avg (n={len(neg_snippets)})')

        if pos_snippets:
            pos_avg = np.mean(pos_snippets, axis=0)
            self.ax_snippets.plot(snippet_time_ms, pos_avg, 'r-',
                                  linewidth=2, label=f'Positive avg (n={len(pos_snippets)})')

        # Add threshold line at y=0 (relative to mean)
        self.ax_snippets.axhline(
            y=0, color='k', linestyle='--', alpha=0.5, label='Threshold crossing')
        self.ax_snippets.axvline(
            x=0, color='k', linestyle=':', alpha=0.5, label='Peak time')

        # Formatting
        self.ax_snippets.set_xlabel('Time (ms)')
        self.ax_snippets.set_ylabel('Amplitude (µV)')
        self.ax_snippets.set_title(
            f'Extracted Waveform Snippets (n={len(self.current_snippets)})')
        self.ax_snippets.grid(True, alpha=0.3)
        self.ax_snippets.legend(loc='upper right')

    def _update_display(self):
        """Update the main display with current data."""
        try:
            # Calculate time range
            start_time = self.current_time
            end_time = min(self.current_time +
                           self.window_duration, self.total_duration)

            # Load data
            data, time_array = self.data_loader.load_channel_data(
                channel=self.current_channel,
                group=self.current_group,
                start_time=start_time,
                end_time=end_time
            )

            # Apply conversion factor to convert to microvolts
            data = data * self.data_conversion_factor

            # Apply filtering if enabled
            if self.signal_filter.filter_type != 'none':
                data = self.signal_filter.filter_data(data)

            # Update plot data
            self.data_line.set_data(time_array, data)

            # Update plot limits
            self.ax_main.set_xlim(start_time, end_time)

            if self.manual_ylims is not None:
                # Use manual y-limits
                self.ax_main.set_ylim(
                    self.manual_ylims[0], self.manual_ylims[1])
            elif self.auto_scale:
                if len(data) > 0:
                    data_range = np.ptp(data)
                    data_center = np.mean(data)
                    margin = data_range * 0.1
                    self.ax_main.set_ylim(
                        data_center - data_range/2 - margin,
                        data_center + data_range/2 + margin
                    )

            # Update threshold line
            if self.show_threshold and self.threshold_value is not None:
                self.threshold_line.set_ydata(
                    [self.threshold_value, self.threshold_value])
                self.threshold_line.set_visible(True)
            else:
                self.threshold_line.set_visible(False)

            # Extract and plot snippets if enabled
            if self.show_snippets and self.show_threshold and self.threshold_value is not None:
                self.current_snippets, self.snippet_times, self.snippet_polarities = \
                    self._extract_snippets(
                        data, time_array, self.threshold_value)
                self._plot_snippets()

            # Update title
            filter_info = f" | Filter: {self.signal_filter.filter_type}" if self.signal_filter.filter_type != 'none' else ""
            self.ax_main.set_title(
                f"Channel: {self.current_channel} | "
                f"Time: {start_time:.1f}-{end_time:.1f}s | "
                f"Duration: {self.window_duration:.1f}s"
                f"{filter_info}"
            )

            # Redraw
            self.fig.canvas.draw()

            # Call update callback if provided
            if self.update_callback:
                self.update_callback(self)

        except Exception as e:
            warnings.warn(f"Error updating display: {e}")

    def _on_time_change(self, val):
        """Handle time slider change."""
        self.current_time = val
        self._update_display()

    def _on_window_change(self, text):
        """Handle window duration change."""
        try:
            new_duration = float(text)
            if new_duration > 0:
                self.window_duration = new_duration
                # Update slider range
                self.time_slider.valmax = max(
                    0, self.total_duration - self.window_duration)
                self.time_slider.ax.set_xlim(0, self.time_slider.valmax)
                self._update_display()
        except ValueError:
            pass

    def _on_threshold_change(self, text):
        """Handle threshold value change."""
        try:
            if text.strip():
                self.threshold_value = float(text)
                self.show_threshold = True
            else:
                self.show_threshold = False
            self._update_display()
        except ValueError:
            pass

    def _on_channel_radio_change(self, label):
        """Handle channel radio button change."""
        if label in self.available_channels:
            self.current_channel = label
            # Update total duration for new channel
            self.total_duration = self.data_loader.get_channel_duration(
                self.current_channel, self.current_group
            )
            self.time_slider.valmax = max(
                0, self.total_duration - self.window_duration)
            self._update_display()

    def _on_lowpass_change(self, text):
        """Handle lowpass frequency change."""
        try:
            freq = float(text)
            if freq > 0:
                self.lowpass_freq = freq
                self._update_filter_from_frequencies()
        except ValueError:
            pass

    def _on_highpass_change(self, text):
        """Handle highpass frequency change."""
        try:
            freq = float(text)
            if freq > 0:
                self.highpass_freq = freq
                self._update_filter_from_frequencies()
        except ValueError:
            pass

    def _on_ylim_change(self, text):
        """Handle y-limit changes."""
        try:
            ymin_text = self.ymin_box.text.strip()
            ymax_text = self.ymax_box.text.strip()

            if ymin_text and ymax_text:
                ymin = float(ymin_text)
                ymax = float(ymax_text)
                if ymin < ymax:
                    self.manual_ylims = (ymin, ymax)
                    self.auto_scale = False
                    self._update_display()
            elif not ymin_text and not ymax_text:
                self.manual_ylims = None
                self.auto_scale = True
                self._update_display()
        except ValueError:
            pass

    def _update_filter_from_frequencies(self):
        """Update filter based on current frequency settings."""
        if self.highpass_freq > 0 and self.lowpass_freq > self.highpass_freq:
            self.signal_filter.update_parameters(
                filter_type='bandpass',
                low_freq=self.highpass_freq,
                high_freq=self.lowpass_freq
            )
        elif self.highpass_freq > 0:
            self.signal_filter.update_parameters(
                filter_type='highpass',
                low_freq=self.highpass_freq
            )
        elif self.lowpass_freq > 0:
            self.signal_filter.update_parameters(
                filter_type='lowpass',
                high_freq=self.lowpass_freq
            )
        else:
            self.signal_filter.update_parameters(filter_type='none')

        self._update_display()

    def _on_scroll(self, event):
        """Handle mouse scroll for time navigation."""
        if event.inaxes == self.ax_main:
            # Scroll through time
            scroll_amount = self.window_duration * 0.1
            if event.button == 'up':
                new_time = max(0, self.current_time - scroll_amount)
            else:
                new_time = min(
                    self.total_duration - self.window_duration,
                    self.current_time + scroll_amount
                )

            self.current_time = new_time
            self.time_slider.set_val(self.current_time)

    def _on_click(self, event):
        """Handle mouse click events."""
        if event.inaxes == self.ax_main and event.dblclick:
            # Double-click to jump to time
            if event.xdata is not None:
                new_time = max(0, min(
                    self.total_duration - self.window_duration,
                    event.xdata - self.window_duration / 2
                ))
                self.current_time = new_time
                self.time_slider.set_val(self.current_time)

    def _on_key_press(self, event):
        """Handle keyboard shortcuts."""
        if event.key == 'left':
            self._navigate_time(-self.window_duration * 0.5)
        elif event.key == 'right':
            self._navigate_time(self.window_duration * 0.5)
        elif event.key == 'up':
            self._change_channel(-1)
        elif event.key == 'down':
            self._change_channel(1)
        elif event.key == 'r':
            self._on_reset_click(None)
        elif event.key == 'f':
            self._on_filter_click(None)

    def _navigate_time(self, delta):
        """Navigate time by delta seconds."""
        new_time = max(0, min(
            self.total_duration - self.window_duration,
            self.current_time + delta
        ))
        self.current_time = new_time
        self.time_slider.set_val(self.current_time)

    def _change_channel(self, direction):
        """Change channel by direction (+1 or -1)."""
        current_idx = self.available_channels.index(self.current_channel)
        new_idx = (current_idx + direction) % len(self.available_channels)
        self.current_channel = self.available_channels[new_idx]
        self.btn_channel.label.set_text(f'Ch: {self.current_channel}')

        # Update total duration for new channel
        self.total_duration = self.data_loader.get_channel_duration(
            self.current_channel, self.current_group
        )
        self.time_slider.valmax = max(
            0, self.total_duration - self.window_duration)

        self._update_display()

    def _on_prev_click(self, event):
        """Handle previous button click."""
        self._navigate_time(-self.window_duration)

    def _on_next_click(self, event):
        """Handle next button click."""
        self._navigate_time(self.window_duration)

    def _on_jump_click(self, event):
        """Handle jump button click."""
        # In a full implementation, this would open a dialog
        # For now, jump to middle of recording
        middle_time = self.total_duration / 2 - self.window_duration / 2
        self.current_time = max(0, middle_time)
        self.time_slider.set_val(self.current_time)

    def _on_reset_click(self, event):
        """Handle reset button click."""
        self.current_time = 0.0
        self.time_slider.set_val(0.0)
        self.auto_scale = True
        self._update_display()

    def _on_autoscale_click(self, event):
        """Handle autoscale button click."""
        self.auto_scale = True
        self.manual_ylims = None
        # Clear y-limit text boxes
        self.ymin_box.set_val('')
        self.ymax_box.set_val('')
        self._update_display()

    def _on_channel_click(self, event):
        """Handle channel button click."""
        # Cycle through channels
        self._change_channel(1)

    def _on_all_channels_click(self, event):
        """Handle all channels button click - cycle through available channels."""
        # This could open a dialog with all channels, for now just cycle
        self._change_channel(1)

    def _on_filter_click(self, event):
        """Handle filter button click - apply current frequency settings."""
        self._update_filter_from_frequencies()

    def set_filter(self, filter_obj: SignalFilter):
        """Set the signal filter."""
        self.signal_filter = filter_obj
        self._update_display()

    def set_channel(self, channel: str, group: str = None):
        """Set the current channel and optionally group."""
        if group is not None:
            self.current_group = group
            self.available_channels = self.data_loader.get_channel_list(group)

        if channel in self.available_channels:
            self.current_channel = channel
            self.btn_channel.label.set_text(f'Ch: {self.current_channel}')

            # Update total duration for new channel
            self.total_duration = self.data_loader.get_channel_duration(
                self.current_channel, self.current_group
            )
            self.time_slider.valmax = max(
                0, self.total_duration - self.window_duration)

            self._update_display()
        else:
            raise ValueError(
                f"Channel '{channel}' not found in group '{self.current_group}'")

    def jump_to_time(self, time_seconds: float):
        """Jump to a specific time."""
        new_time = max(0, min(
            self.total_duration - self.window_duration,
            time_seconds
        ))
        self.current_time = new_time
        self.time_slider.set_val(self.current_time)

    def set_threshold(self, threshold: Optional[float]):
        """Set threshold value."""
        self.threshold_value = threshold
        self.show_threshold = threshold is not None
        self._update_display()

    def get_current_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get currently displayed data."""
        start_time = self.current_time
        end_time = min(self.current_time + self.window_duration,
                       self.total_duration)

        data, time_array = self.data_loader.load_channel_data(
            channel=self.current_channel,
            group=self.current_group,
            start_time=start_time,
            end_time=end_time
        )

        # Apply conversion factor
        data = data * self.data_conversion_factor

        if self.signal_filter.filter_type != 'none':
            data = self.signal_filter.filter_data(data)

        return data, time_array

    def show(self):
        """Show the interactive plot."""
        plt.show()

    def close(self):
        """Close the plot."""
        plt.close(self.fig)


class ViewerControls:
    """
    Separate control panel for advanced viewer parameters.
    """

    def __init__(self, plotter: InteractivePlotter):
        """
        Initialize viewer controls.

        Args:
            plotter: InteractivePlotter instance to control
        """
        self.plotter = plotter
        self._create_control_window()

    def _create_control_window(self):
        """Create the control window."""
        self.control_fig, self.control_ax = plt.subplots(figsize=(6, 8))
        self.control_ax.set_xlim(0, 1)
        self.control_ax.set_ylim(0, 1)
        self.control_ax.axis('off')
        self.control_fig.suptitle('Viewer Controls')

        # Add control widgets here
        # This would include filter parameter controls, channel selection, etc.
        # Implementation would depend on specific requirements

    def show(self):
        """Show the control window."""
        plt.show()

    def close(self):
        """Close the control window."""
        plt.close(self.control_fig)
