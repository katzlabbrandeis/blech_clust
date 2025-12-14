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
    - Time navigation via GUI controls
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
                 update_callback: Optional[Callable] = None,
                 loading_strategy: str = None):
        """
        Initialize the interactive plotter.

        Args:
            data_loader: RawDataLoader instance
            initial_channel: Initial channel to display
            initial_group: Initial data group
            window_duration: Initial window duration in seconds
            update_callback: Optional callback function for updates
            loading_strategy: Override data loader's loading strategy ('streaming' or 'memory')
        """
        self.data_loader = data_loader
        self.current_group = initial_group
        self.window_duration = window_duration
        self.update_callback = update_callback

        # Override loading strategy if specified
        if loading_strategy is not None:
            self.data_loader.switch_loading_strategy(loading_strategy)

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
        self.max_freq = 3000.0  # Maximum frequency (lowpass cutoff)
        self.min_freq = 300.0   # Minimum frequency (highpass cutoff)
        self.data_conversion_factor = 0.6745  # Convert to microvolts

        # Snippet extraction parameters
        self.show_snippets = False
        self.snippet_before_ms = 0.5  # time before threshold crossing (ms)
        self.snippet_after_ms = 1.0   # time after threshold crossing (ms)
        self.max_snippets = 50        # maximum snippets to display
        self.current_snippets = []    # extracted snippet waveforms
        self.snippet_times = []       # times of snippet peaks
        self.snippet_polarities = []  # polarity of each snippet

        # Apply changes system
        self.auto_update = True       # Enable immediate updates by default
        self.pending_changes = False  # Track if changes need applying

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
        self._update_apply_button_state()  # Initialize button state
        self._update_display()  # Initial display

    def _create_plot(self):
        """Create the main plot and figure."""
        # Create figure with subplots for plot and controls
        self.fig = plt.figure(figsize=(16, 14))

        # Main plot area - takes up 5/6 of the width
        self.ax_main = plt.subplot2grid((8, 6), (0, 0), colspan=5, rowspan=3)
        self.ax_main.set_xlabel('Time (s)')
        self.ax_main.set_ylabel('Amplitude (µV)')
        self.ax_main.grid(True, alpha=0.3)

        # Snippet plot area - 1/6 width of total (1/5 of main plot width)
        self.ax_snippets = plt.subplot2grid(
            (8, 6), (0, 5), colspan=1, rowspan=3)
        self.ax_snippets.set_xlabel('Time (ms)')
        self.ax_snippets.set_ylabel('Amplitude (µV)')
        self.ax_snippets.set_title('Snippets')
        self.ax_snippets.grid(True, alpha=0.3)
        self.ax_snippets.set_visible(False)  # Initially hidden

        # Initialize empty line objects
        self.data_line, = self.ax_main.plot(
            [], [], 'b-', linewidth=0.8, label='Raw Data')
        self.threshold_line_pos = self.ax_main.axhline(y=0, color='r', linestyle='--',
                                                       linewidth=2, alpha=0.7, visible=False,
                                                       label='+Threshold')
        self.threshold_line_neg = self.ax_main.axhline(y=0, color='r', linestyle='--',
                                                       linewidth=2, alpha=0.7, visible=False,
                                                       label='-Threshold')

        # Add legend
        self.ax_main.legend(loc='upper right')

        # Set up interactive features
        self.ax_main.set_xlim(0, self.window_duration)

        # Connect mouse events
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

    def _create_controls(self):
        """Create control widgets."""
        # Time navigation slider - Row 4 (moved up since snippet plot is now on the side)
        ax_time = plt.subplot2grid((8, 6), (4, 0), colspan=3)
        self.time_slider = Slider(
            ax_time, 'Time (s)', 0, max(
                0, self.total_duration - self.window_duration),
            valinit=self.current_time, valfmt='%.1f'
        )
        self.time_slider.on_changed(self._on_time_change)

        # Window duration control
        ax_window = plt.subplot2grid((8, 6), (4, 3))
        self.window_box = TextBox(
            ax_window, 'Window (s)', initial=str(self.window_duration))
        self.window_box.on_submit(self._on_window_change)

        # Threshold control
        ax_threshold = plt.subplot2grid((8, 6), (4, 4))
        self.threshold_box = TextBox(ax_threshold, 'Threshold', initial='')
        self.threshold_box.on_submit(self._on_threshold_change)

        # Channel selection dropdown
        ax_channel_select = plt.subplot2grid((8, 6), (4, 5))
        self.channel_dropdown = self._create_channel_dropdown(
            ax_channel_select)

        # Filter frequency controls - Row 5
        ax_max_freq = plt.subplot2grid((8, 6), (5, 0))
        self.max_freq_box = TextBox(
            ax_max_freq, 'Max Freq (Hz)', initial=str(self.max_freq))
        self.max_freq_box.on_submit(self._on_max_freq_change)

        ax_min_freq = plt.subplot2grid((8, 6), (5, 1))
        self.min_freq_box = TextBox(
            ax_min_freq, 'Min Freq (Hz)', initial=str(self.min_freq))
        self.min_freq_box.on_submit(self._on_min_freq_change)

        # Y-limits controls
        ax_ymin = plt.subplot2grid((8, 6), (5, 2))
        self.ymin_box = TextBox(ax_ymin, 'Y Min', initial='')
        self.ymin_box.on_submit(self._on_ylim_change)

        ax_ymax = plt.subplot2grid((8, 6), (5, 3))
        self.ymax_box = TextBox(ax_ymax, 'Y Max', initial='')
        self.ymax_box.on_submit(self._on_ylim_change)

        # Loading strategy control
        ax_loading_strategy = plt.subplot2grid((8, 6), (5, 4))
        current_strategy = self.data_loader.loading_strategy
        self.loading_strategy_check = CheckButtons(
            ax_loading_strategy, ['Memory Mode'], [current_strategy == 'memory'])
        self.loading_strategy_check.on_clicked(
            self._on_loading_strategy_change)

        # Memory management controls
        ax_clear_cache = plt.subplot2grid((8, 6), (5, 5))
        self.btn_clear_cache = Button(ax_clear_cache, 'Clear Cache')
        self.btn_clear_cache.on_clicked(self._on_clear_cache_click)

        # Snippet controls - Row 6
        ax_snippet_before = plt.subplot2grid((8, 6), (6, 0))
        self.snippet_before_box = TextBox(
            ax_snippet_before, 'Before (ms)', initial=str(self.snippet_before_ms))
        self.snippet_before_box.on_submit(self._on_snippet_before_change)

        ax_snippet_after = plt.subplot2grid((8, 6), (6, 1))
        self.snippet_after_box = TextBox(
            ax_snippet_after, 'After (ms)', initial=str(self.snippet_after_ms))
        self.snippet_after_box.on_submit(self._on_snippet_after_change)

        ax_max_snippets = plt.subplot2grid((8, 6), (6, 2))
        self.max_snippets_box = TextBox(
            ax_max_snippets, 'Max Snippets', initial=str(self.max_snippets))
        self.max_snippets_box.on_submit(self._on_max_snippets_change)

        # Show snippets checkbox
        ax_show_snippets = plt.subplot2grid((8, 6), (6, 3))
        self.show_snippets_check = CheckButtons(
            ax_show_snippets, ['Show Snippets'], [self.show_snippets])
        self.show_snippets_check.on_clicked(self._on_show_snippets_change)

        # Add control buttons (will be created in separate method)
        self._create_buttons()

    def _create_channel_dropdown(self, ax):
        """Create a custom dropdown for channel selection."""
        from matplotlib.patches import Rectangle

        # Create a simple dropdown using matplotlib primitives
        # For now, use a button that shows current channel and cycles through options
        current_idx = self.available_channels.index(self.current_channel)
        channel_label = f'{self.current_channel}'

        # Create the main button
        channel_button = Button(ax, channel_label)
        channel_button.on_clicked(self._on_channel_dropdown_click)

        # Store dropdown state
        self.channel_dropdown_open = False
        self.channel_dropdown_buttons = []
        self.channel_dropdown_axes = []

        return channel_button

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

        # Apply changes and auto-update controls
        ax_apply = plt.axes([0.46, y_pos, button_width * 1.2, button_height])
        self.btn_apply = Button(ax_apply, 'Apply Changes')
        self.btn_apply.on_clicked(self._on_apply_changes)

        ax_auto_update = plt.axes(
            [0.60, y_pos, button_width * 1.2, button_height])
        self.auto_update_check = CheckButtons(
            ax_auto_update, ['Auto Update'], [self.auto_update])
        self.auto_update_check.on_clicked(self._on_auto_update_change)

        # Additional control buttons
        ax_all_channels = plt.axes(
            [0.76, y_pos, button_width * 1.2, button_height])
        self.btn_all_channels = Button(ax_all_channels, 'All Channels')
        self.btn_all_channels.on_clicked(self._on_all_channels_click)

        # Status display for loading mode
        ax_status = plt.axes([0.02, y_pos + button_height + 0.005, 0.3, 0.02])
        ax_status.set_xlim(0, 1)
        ax_status.set_ylim(0, 1)
        ax_status.axis('off')
        self.status_text = ax_status.text(0, 0.5, self._get_status_text(),
                                          fontsize=8, va='center')

    def _on_apply_changes(self, event):
        """Apply all pending parameter changes."""
        self._update_display()
        self.pending_changes = False
        self._update_apply_button_state()

    def _on_auto_update_change(self, label):
        """Handle auto-update checkbox change."""
        self.auto_update = not self.auto_update
        if self.auto_update and self.pending_changes:
            # If switching to auto-update and there are pending changes, apply them
            self._update_display()
            self.pending_changes = False
        self._update_apply_button_state()

    def _on_snippet_before_change(self, text):
        """Handle snippet before time change."""
        try:
            before_ms = float(text)
            if before_ms > 0:  # Must be positive
                self.snippet_before_ms = before_ms
                if self.show_snippets:
                    self._defer_update_if_needed()
        except ValueError:
            pass

    def _on_snippet_after_change(self, text):
        """Handle snippet after time change."""
        try:
            after_ms = float(text)
            if after_ms > 0:  # Must be positive
                self.snippet_after_ms = after_ms
                if self.show_snippets:
                    self._defer_update_if_needed()
        except ValueError:
            pass

    def _on_max_snippets_change(self, text):
        """Handle max snippets change."""
        try:
            max_count = int(text)
            if max_count > 0:
                self.max_snippets = max_count
                if self.show_snippets:
                    self._defer_update_if_needed()
        except ValueError:
            pass

    def _on_show_snippets_change(self, label):
        """Handle show snippets checkbox change."""
        self.show_snippets = not self.show_snippets
        self.ax_snippets.set_visible(self.show_snippets)
        if self.show_snippets:
            self._defer_update_if_needed()
        else:
            self.ax_snippets.clear()
            self.ax_snippets.set_xlabel('Time (ms)')
            self.ax_snippets.set_ylabel('Amplitude (µV)')
            self.ax_snippets.set_title('Snippets')
            self.ax_snippets.grid(True, alpha=0.3)
        self.fig.canvas.draw()

    def _defer_update_if_needed(self):
        """Call _update_display() only if auto_update is enabled."""
        if self.auto_update:
            self._update_display()
        else:
            self.pending_changes = True
            self._update_apply_button_state()

    def _update_apply_button_state(self):
        """Update the apply button appearance based on pending changes."""
        if hasattr(self, 'btn_apply'):
            if self.pending_changes:
                self.btn_apply.label.set_text('Apply Changes*')
                self.btn_apply.color = 'lightcoral'
            else:
                self.btn_apply.label.set_text('Apply Changes')
                self.btn_apply.color = 'lightgray'
            self.fig.canvas.draw_idle()

    def _extract_snippets(self, data, time_array, threshold):
        """
        Extract waveform snippets that cross the threshold.

        Based on extract_waveforms_abu from utils/clustering.py
        """
        if threshold is None or len(data) == 0:
            return [], [], []

        sampling_rate = self.data_loader.sampling_rate

        # Calculate snippet window based on fixed time windows
        before_samples = int((self.snippet_before_ms / 1000.0) * sampling_rate)
        after_samples = int((self.snippet_after_ms / 1000.0) * sampling_rate)

        # Find mean for threshold detection
        mean_val = np.mean(data)

        # Use absolute threshold value for both positive and negative crossings
        abs_threshold = abs(threshold)

        # Find threshold crossings for both positive and negative directions
        negative_crossings = np.where(data <= mean_val - abs_threshold)[0]
        positive_crossings = np.where(data >= mean_val + abs_threshold)[0]

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
        self.ax_snippets.set_xlabel('Time (ms)', fontsize='small')
        self.ax_snippets.set_ylabel('Amplitude (µV)', fontsize='small')
        self.ax_snippets.set_title(
            f'Snippets (n={len(self.current_snippets)})', fontsize='small')
        self.ax_snippets.grid(True, alpha=0.3)
        self.ax_snippets.legend(loc='upper right', fontsize='x-small')
        self.ax_snippets.tick_params(
            axis='both', which='major', labelsize='x-small')

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

            # Update threshold lines (both positive and negative)
            if self.show_threshold and self.threshold_value is not None:
                # Calculate mean for threshold positioning
                mean_val = np.mean(data) if len(data) > 0 else 0
                abs_threshold = abs(self.threshold_value)

                # Set positive threshold line
                pos_threshold = mean_val + abs_threshold
                self.threshold_line_pos.set_ydata(
                    [pos_threshold, pos_threshold])
                self.threshold_line_pos.set_visible(True)

                # Set negative threshold line
                neg_threshold = mean_val - abs_threshold
                self.threshold_line_neg.set_ydata(
                    [neg_threshold, neg_threshold])
                self.threshold_line_neg.set_visible(True)
            else:
                self.threshold_line_pos.set_visible(False)
                self.threshold_line_neg.set_visible(False)

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
            self._defer_update_if_needed()
        except ValueError:
            pass

    def _on_channel_dropdown_click(self, event):
        """Handle channel dropdown click - toggle dropdown menu."""
        if self.channel_dropdown_open:
            self._close_channel_dropdown()
        else:
            self._open_channel_dropdown()

    def _open_channel_dropdown(self):
        """Open the channel dropdown menu."""
        if self.channel_dropdown_open or not self.available_channels:
            return

        try:
            self.channel_dropdown_open = True

            # Get position of the dropdown button
            ax_pos = self.channel_dropdown.ax.get_position()

            # Limit number of visible options to prevent overflow
            max_visible = min(10, len(self.available_channels))

            # Create dropdown items
            for i, channel in enumerate(self.available_channels[:max_visible]):
                # Position each dropdown item below the main button
                item_y = ax_pos.y0 - (i + 1) * 0.03
                if item_y < 0:  # Don't go below the figure
                    break

                item_ax = self.fig.add_axes(
                    [ax_pos.x0, item_y, ax_pos.width, 0.025])

                # Create button for this channel
                item_button = Button(item_ax, channel)
                item_button.on_clicked(
                    lambda event, ch=channel: self._select_channel_from_dropdown(ch))

                # Highlight current selection
                if channel == self.current_channel:
                    item_button.color = 'lightblue'
                else:
                    item_button.color = 'white'

                self.channel_dropdown_axes.append(item_ax)
                self.channel_dropdown_buttons.append(item_button)

            # Add "More..." option if there are more channels
            if len(self.available_channels) > max_visible:
                item_y = ax_pos.y0 - (max_visible + 1) * 0.03
                if item_y >= 0:
                    item_ax = self.fig.add_axes(
                        [ax_pos.x0, item_y, ax_pos.width, 0.025])
                    more_button = Button(
                        item_ax, f'... ({len(self.available_channels) - max_visible} more)')
                    more_button.on_clicked(self._show_more_channels)
                    more_button.color = 'lightgray'
                    self.channel_dropdown_axes.append(item_ax)
                    self.channel_dropdown_buttons.append(more_button)

            # Redraw the figure
            self.fig.canvas.draw()

        except Exception as e:
            # If dropdown creation fails, close it and continue
            warnings.warn(f"Error opening channel dropdown: {e}")
            self._close_channel_dropdown()

    def _close_channel_dropdown(self):
        """Close the channel dropdown menu."""
        if not self.channel_dropdown_open:
            return

        try:
            self.channel_dropdown_open = False

            # Remove dropdown items safely
            for ax in self.channel_dropdown_axes:
                try:
                    ax.remove()
                except:
                    pass  # Ignore errors if axes already removed

            self.channel_dropdown_axes.clear()
            self.channel_dropdown_buttons.clear()

            # Redraw the figure
            self.fig.canvas.draw()

        except Exception as e:
            # Ensure state is reset even if cleanup fails
            self.channel_dropdown_open = False
            self.channel_dropdown_axes.clear()
            self.channel_dropdown_buttons.clear()
            warnings.warn(f"Error closing channel dropdown: {e}")

    def _select_channel_from_dropdown(self, channel):
        """Handle channel selection from dropdown."""
        self.current_channel = channel
        self.channel_dropdown.label.set_text(channel)
        self._close_channel_dropdown()

        # Update total duration for new channel
        self.total_duration = self.data_loader.get_channel_duration(
            self.current_channel, self.current_group
        )
        self.time_slider.valmax = max(
            0, self.total_duration - self.window_duration)
        self._update_display()

    def _show_more_channels(self, event):
        """Show more channels (cycling through pages)."""
        # For now, just close dropdown and cycle to next channel
        self._close_channel_dropdown()
        current_idx = self.available_channels.index(self.current_channel)
        next_idx = (current_idx + 10) % len(self.available_channels)
        self._select_channel_from_dropdown(self.available_channels[next_idx])

    def _on_max_freq_change(self, text):
        """Handle maximum frequency change."""
        try:
            freq = float(text)
            if freq > 0:
                self.max_freq = freq
                self._update_filter_from_frequencies()
        except ValueError:
            pass

    def _on_min_freq_change(self, text):
        """Handle minimum frequency change."""
        try:
            freq = float(text)
            if freq > 0:
                self.min_freq = freq
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
                    self._defer_update_if_needed()
            elif not ymin_text and not ymax_text:
                self.manual_ylims = None
                self.auto_scale = True
                self._defer_update_if_needed()
        except ValueError:
            pass

    def _update_filter_from_frequencies(self):
        """Update filter based on current frequency settings."""
        if self.min_freq > 0 and self.max_freq > self.min_freq:
            self.signal_filter.update_parameters(
                filter_type='bandpass',
                low_freq=self.min_freq,
                high_freq=self.max_freq
            )
        elif self.min_freq > 0:
            self.signal_filter.update_parameters(
                filter_type='highpass',
                low_freq=self.min_freq
            )
        elif self.max_freq > 0:
            self.signal_filter.update_parameters(
                filter_type='lowpass',
                high_freq=self.max_freq
            )
        else:
            self.signal_filter.update_parameters(filter_type='none')

        self._defer_update_if_needed()

    def _on_click(self, event):
        """Handle mouse click events."""
        # Check if click is outside dropdown to close it
        if self.channel_dropdown_open:
            # Check if click is not on dropdown or its items
            if (event.inaxes != self.channel_dropdown.ax and
                    event.inaxes not in self.channel_dropdown_axes):
                self._close_channel_dropdown()

        if event.inaxes == self.ax_main and event.dblclick:
            # Double-click to jump to time
            if event.xdata is not None:
                new_time = max(0, min(
                    self.total_duration - self.window_duration,
                    event.xdata - self.window_duration / 2
                ))
                self.current_time = new_time
                self.time_slider.set_val(self.current_time)

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

        # Update channel dropdown label
        self.channel_dropdown.label.set_text(self.current_channel)

        # Update total duration for new channel
        self.total_duration = self.data_loader.get_channel_duration(
            self.current_channel, self.current_group
        )
        self.time_slider.valmax = max(
            0, self.total_duration - self.window_duration)

        # If in memory mode, preload the new channel
        if self.data_loader.loading_strategy == 'memory':
            cache_key = (self.current_group, self.current_channel)
            if cache_key not in self.data_loader._memory_cache:
                self.data_loader.preload_channels(
                    [self.current_channel], self.current_group)
                self._update_status_display()

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
        self._defer_update_if_needed()

    def _on_autoscale_click(self, event):
        """Handle autoscale button click."""
        self.auto_scale = True
        self.manual_ylims = None
        # Clear y-limit text boxes
        self.ymin_box.set_val('')
        self.ymax_box.set_val('')
        self._defer_update_if_needed()

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

    def _on_loading_strategy_change(self, label):
        """Handle loading strategy checkbox change."""
        is_memory_mode = self.loading_strategy_check.get_status()[0]
        new_strategy = 'memory' if is_memory_mode else 'streaming'

        # Switch strategy
        self.data_loader.switch_loading_strategy(new_strategy)

        # If switching to memory mode, preload current channel
        if new_strategy == 'memory':
            self.data_loader.preload_channels(
                [self.current_channel], self.current_group)

        # Update status display
        self._update_status_display()

        # Update display to reflect any performance changes
        self._defer_update_if_needed()

    def _get_status_text(self) -> str:
        """Generate status text showing loading mode and memory usage."""
        strategy = self.data_loader.loading_strategy
        status_parts = [f"Mode: {strategy.title()}"]

        if strategy == 'memory':
            memory_info = self.data_loader.get_memory_usage()
            if memory_info['cached_channels'] > 0:
                status_parts.append(
                    f"Cached: {memory_info['cached_channels']} channels")
                status_parts.append(
                    f"Memory: {memory_info['total_memory_mb']:.1f} MB")
            else:
                status_parts.append("No channels cached")

        return " | ".join(status_parts)

    def _update_status_display(self):
        """Update the status display text."""
        if hasattr(self, 'status_text'):
            self.status_text.set_text(self._get_status_text())
            self.fig.canvas.draw_idle()

    def _on_clear_cache_click(self, event):
        """Handle clear cache button click."""
        if self.data_loader.loading_strategy == 'memory':
            self.data_loader.clear_memory_cache()
            self._update_status_display()
            print("Memory cache cleared")
        else:
            print("Cache clearing only available in memory mode")

    def set_memory_limit(self, limit_mb: float):
        """
        Set memory limit for the data loader.

        Args:
            limit_mb: Memory limit in megabytes
        """
        if hasattr(self.data_loader, 'set_memory_limit'):
            self.data_loader.set_memory_limit(limit_mb)
            self._update_status_display()
            print(f"Memory limit set to {limit_mb:.1f} MB")

    def get_performance_stats(self) -> dict:
        """
        Get performance statistics for the viewer.

        Returns:
            Dictionary with performance metrics
        """
        stats = {
            'loading_strategy': self.data_loader.loading_strategy,
            'current_channel': self.current_channel,
            'window_duration': self.window_duration,
            'total_duration': self.total_duration
        }

        if self.data_loader.loading_strategy == 'memory':
            if hasattr(self.data_loader, 'get_cache_efficiency_stats'):
                cache_stats = self.data_loader.get_cache_efficiency_stats()
                stats.update(cache_stats)

            memory_usage = self.data_loader.get_memory_usage()
            stats['memory_usage'] = memory_usage

        return stats

    def optimize_for_workflow(self, workflow_type: str = 'sequential'):
        """
        Optimize viewer settings for specific workflows.

        Args:
            workflow_type: 'sequential', 'random', or 'channel_switching'
        """
        if workflow_type == 'sequential':
            # Optimize for sequential time navigation
            if hasattr(self.data_loader, 'optimize_for_sequential_access'):
                self.data_loader.optimize_for_sequential_access(True)

            # Use memory mode for better performance
            if self.data_loader.loading_strategy != 'memory':
                self.data_loader.switch_loading_strategy('memory')
                self.loading_strategy_check.set_active(0)

        elif workflow_type == 'channel_switching':
            # Optimize for frequent channel switching
            if self.data_loader.loading_strategy != 'memory':
                self.data_loader.switch_loading_strategy('memory')
                self.loading_strategy_check.set_active(0)

            # Preload all channels
            all_channels = self.data_loader.get_channel_list(
                self.current_group)
            print(
                f"Preloading {len(all_channels)} channels for channel switching workflow...")
            self.data_loader.preload_channels(all_channels, self.current_group)

        elif workflow_type == 'random':
            # Use streaming mode for random access to save memory
            if self.data_loader.loading_strategy != 'streaming':
                self.data_loader.switch_loading_strategy('streaming')
                self.loading_strategy_check.set_active(0)

        self._update_status_display()
        print(f"Optimized for {workflow_type} workflow")

    def set_filter(self, filter_obj: SignalFilter):
        """Set the signal filter."""
        self.signal_filter = filter_obj
        self._defer_update_if_needed()

    def set_channel(self, channel: str, group: str = None):
        """Set the current channel and optionally group."""
        if group is not None:
            self.current_group = group
            self.available_channels = self.data_loader.get_channel_list(group)

        if channel in self.available_channels:
            self.current_channel = channel
            self.channel_dropdown.label.set_text(self.current_channel)

            # Update total duration for new channel
            self.total_duration = self.data_loader.get_channel_duration(
                self.current_channel, self.current_group
            )
            self.time_slider.valmax = max(
                0, self.total_duration - self.window_duration)

            # If in memory mode, preload the new channel
            if self.data_loader.loading_strategy == 'memory':
                cache_key = (self.current_group, self.current_channel)
                if cache_key not in self.data_loader._memory_cache:
                    self.data_loader.preload_channels(
                        [self.current_channel], self.current_group)
                    self._update_status_display()

            self._defer_update_if_needed()
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
        self._defer_update_if_needed()

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
