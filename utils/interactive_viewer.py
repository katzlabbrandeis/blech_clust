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
        self.available_channels = self.data_loader.get_channel_list(initial_group)
        if not self.available_channels:
            raise ValueError(f"No channels found in group '{initial_group}'")
        
        self.current_channel = initial_channel or self.available_channels[0]
        
        # Initialize display parameters
        self.current_time = 0.0
        self.threshold_value = None
        self.show_threshold = False
        self.auto_scale = True
        self.y_scale_factor = 1.0
        
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
        self.fig = plt.figure(figsize=(14, 10))
        
        # Main plot area
        self.ax_main = plt.subplot2grid((4, 4), (0, 0), colspan=4, rowspan=3)
        self.ax_main.set_xlabel('Time (s)')
        self.ax_main.set_ylabel('Amplitude (µV)')
        self.ax_main.grid(True, alpha=0.3)
        
        # Initialize empty line objects
        self.data_line, = self.ax_main.plot([], [], 'b-', linewidth=0.8, label='Raw Data')
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
        # Time navigation slider
        ax_time = plt.subplot2grid((4, 4), (3, 0), colspan=2)
        self.time_slider = Slider(
            ax_time, 'Time (s)', 0, max(0, self.total_duration - self.window_duration),
            valinit=self.current_time, valfmt='%.1f'
        )
        self.time_slider.on_changed(self._on_time_change)
        
        # Window duration control
        ax_window = plt.subplot2grid((4, 4), (3, 2))
        self.window_box = TextBox(ax_window, 'Window (s)', initial=str(self.window_duration))
        self.window_box.on_submit(self._on_window_change)
        
        # Threshold control
        ax_threshold = plt.subplot2grid((4, 4), (3, 3))
        self.threshold_box = TextBox(ax_threshold, 'Threshold', initial='')
        self.threshold_box.on_submit(self._on_threshold_change)
        
        # Add control buttons (will be created in separate method)
        self._create_buttons()
    
    def _create_buttons(self):
        """Create control buttons."""
        # Create button axes
        button_height = 0.04
        button_width = 0.08
        button_spacing = 0.02
        
        # Navigation buttons
        ax_prev = plt.axes([0.02, 0.02, button_width, button_height])
        ax_next = plt.axes([0.12, 0.02, button_width, button_height])
        ax_jump = plt.axes([0.22, 0.02, button_width, button_height])
        
        self.btn_prev = Button(ax_prev, '← Prev')
        self.btn_next = Button(ax_next, 'Next →')
        self.btn_jump = Button(ax_jump, 'Jump to')
        
        self.btn_prev.on_clicked(self._on_prev_click)
        self.btn_next.on_clicked(self._on_next_click)
        self.btn_jump.on_clicked(self._on_jump_click)
        
        # Filter and display buttons
        ax_filter = plt.axes([0.35, 0.02, button_width, button_height])
        ax_reset = plt.axes([0.45, 0.02, button_width, button_height])
        ax_autoscale = plt.axes([0.55, 0.02, button_width, button_height])
        
        self.btn_filter = Button(ax_filter, 'Filter')
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_autoscale = Button(ax_autoscale, 'Auto Scale')
        
        self.btn_filter.on_clicked(self._on_filter_click)
        self.btn_reset.on_clicked(self._on_reset_click)
        self.btn_autoscale.on_clicked(self._on_autoscale_click)
        
        # Channel selection (simplified - in full implementation would be dropdown)
        ax_channel = plt.axes([0.68, 0.02, button_width * 1.5, button_height])
        self.btn_channel = Button(ax_channel, f'Ch: {self.current_channel}')
        self.btn_channel.on_clicked(self._on_channel_click)
    
    def _update_display(self):
        """Update the main display with current data."""
        try:
            # Calculate time range
            start_time = self.current_time
            end_time = min(self.current_time + self.window_duration, self.total_duration)
            
            # Load data
            data, time_array = self.data_loader.load_channel_data(
                channel=self.current_channel,
                group=self.current_group,
                start_time=start_time,
                end_time=end_time
            )
            
            # Apply filtering if enabled
            if self.signal_filter.filter_type != 'none':
                data = self.signal_filter.filter_data(data)
            
            # Update plot data
            self.data_line.set_data(time_array, data)
            
            # Update plot limits
            self.ax_main.set_xlim(start_time, end_time)
            
            if self.auto_scale:
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
                self.threshold_line.set_ydata([self.threshold_value, self.threshold_value])
                self.threshold_line.set_visible(True)
            else:
                self.threshold_line.set_visible(False)
            
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
                self.time_slider.valmax = max(0, self.total_duration - self.window_duration)
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
        self.time_slider.valmax = max(0, self.total_duration - self.window_duration)
        
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
    
    def _on_filter_click(self, event):
        """Handle filter button click."""
        # Toggle between common filter types
        if self.signal_filter.filter_type == 'none':
            # Switch to spike filter
            self.signal_filter.update_parameters(
                filter_type='bandpass',
                low_freq=300,
                high_freq=3000
            )
        elif self.signal_filter.filter_type == 'bandpass' and self.signal_filter.low_freq == 300:
            # Switch to LFP filter
            self.signal_filter.update_parameters(
                filter_type='bandpass',
                low_freq=1,
                high_freq=300
            )
        else:
            # Switch to no filter
            self.signal_filter.update_parameters(filter_type='none')
        
        self._update_display()
    
    def _on_reset_click(self, event):
        """Handle reset button click."""
        self.current_time = 0.0
        self.time_slider.set_val(0.0)
        self.auto_scale = True
        self._update_display()
    
    def _on_autoscale_click(self, event):
        """Handle autoscale button click."""
        self.auto_scale = not self.auto_scale
        if self.auto_scale:
            self._update_display()
    
    def _on_channel_click(self, event):
        """Handle channel button click."""
        # Cycle through channels
        self._change_channel(1)
    
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
            self.time_slider.valmax = max(0, self.total_duration - self.window_duration)
            
            self._update_display()
        else:
            raise ValueError(f"Channel '{channel}' not found in group '{self.current_group}'")
    
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
        end_time = min(self.current_time + self.window_duration, self.total_duration)
        
        data, time_array = self.data_loader.load_channel_data(
            channel=self.current_channel,
            group=self.current_group,
            start_time=start_time,
            end_time=end_time
        )
        
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