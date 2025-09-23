# Raw Data Viewer Enhancements

This document describes the enhancements made to the raw data viewer as part of issue #1 and subsequent snippet extraction features.

## Overview

The raw data viewer has been enhanced with new GUI controls and functionality to provide better user control over data visualization, filtering, and waveform snippet extraction from threshold crossings.

## New Features

### 1. Filter Frequency Controls

**Text Entry Boxes for Lowpass and Highpass Frequencies**

- **Lowpass Frequency Box**: Allows users to set the upper cutoff frequency for filtering
- **Highpass Frequency Box**: Allows users to set the lower cutoff frequency for filtering
- **Apply Filter Button**: Applies the bandpass filter using the specified frequencies
- **Default Values**: Lowpass = 3000 Hz, Highpass = 300 Hz (suitable for spike filtering)

**Usage:**
1. Enter desired frequencies in the text boxes
2. Click "Apply Filter" to apply the bandpass filter
3. The filter is applied in real-time to the displayed data

### 2. Manual Y-Axis Control

**Text Entry Boxes for Y-Limits**

- **Y Min Box**: Set the minimum value for the Y-axis
- **Y Max Box**: Set the maximum value for the Y-axis
- **Auto Y Button**: Reset to automatic scaling

**Usage:**
1. Enter minimum and maximum Y values in the respective boxes
2. Press Enter to apply the manual scaling
3. Click "Auto Y" to return to automatic scaling
4. Leave both boxes empty and press Enter to return to auto-scaling

### 3. Data Conversion Factor

**Automatic Conversion to Microvolts**

- **Conversion Factor**: 0.6745 (applied automatically to all loaded data)
- **Purpose**: Converts raw ADC values to microvolts for proper amplitude display
- **Application**: Applied to all data before filtering and display

### 4. Enhanced Channel Selection

**Channel Selection via Radio Buttons**

- **Radio Button Panel**: Shows available channels for quick selection
- **Real-time Switching**: Immediately switches to selected channel
- **Channel Information**: Displays up to 5 channels in the radio button panel
- **Fallback**: "All Channels" button for cycling through all available channels

### 5. Waveform Snippet Extraction

**Automatic Spike Detection and Visualization**

- **Threshold-Based Detection**: Extracts waveforms that cross the user-defined threshold
- **Configurable Windows**: Adjustable time before (default: 0.5ms) and after (default: 1.0ms) threshold crossing
- **Polarity Separation**: Distinguishes between positive and negative threshold crossings
- **Average Waveforms**: Displays average waveforms for each polarity type
- **Real-time Extraction**: Snippets update automatically as you navigate through data

**Usage:**
1. Set a threshold value in the threshold box
2. Check "Show Snippets" to enable snippet extraction
3. Adjust "Before (ms)" and "After (ms)" for snippet window size
4. Set "Max Snippets" to limit the number displayed (default: 50)
5. Snippets are automatically extracted from the current viewing window

## Technical Implementation

### Code Changes

#### InteractivePlotter Class Enhancements

```python
# New attributes added (Issue #1)
self.manual_ylims = None
self.lowpass_freq = 3000.0
self.highpass_freq = 300.0
self.data_conversion_factor = 0.6745

# New snippet extraction attributes
self.show_snippets = False
self.snippet_before_ms = 0.5
self.snippet_after_ms = 1.0
self.max_snippets = 50
self.current_snippets = []
self.snippet_times = []
self.snippet_polarities = []
```

#### New GUI Controls

1. **Filter Controls** (Row 6 of control grid):
   - Lowpass frequency text box
   - Highpass frequency text box

2. **Y-Limit Controls** (Row 6 of control grid):
   - Y Min text box
   - Y Max text box

3. **Channel Selection** (Row 5 of control grid):
   - Radio buttons for channel selection

4. **Snippet Controls** (Row 7 of control grid):
   - Before (ms) text box for pre-threshold time
   - After (ms) text box for post-threshold time
   - Max Snippets text box for display limit
   - Show Snippets checkbox to enable/disable extraction

5. **Snippet Display**:
   - Dedicated subplot below main plot for snippet visualization
   - Automatic scaling and color coding by polarity

#### New Callback Methods

**Issue #1 Methods:**
- `_on_lowpass_change()`: Handle lowpass frequency changes
- `_on_highpass_change()`: Handle highpass frequency changes
- `_on_ylim_change()`: Handle Y-limit changes
- `_on_channel_radio_change()`: Handle channel selection
- `_update_filter_from_frequencies()`: Apply filter based on frequency settings

**Snippet Extraction Methods:**
- `_on_snippet_before_change()`: Handle snippet before time changes
- `_on_snippet_after_change()`: Handle snippet after time changes
- `_on_max_snippets_change()`: Handle max snippets limit changes
- `_on_show_snippets_change()`: Handle show snippets checkbox toggle
- `_extract_snippets()`: Extract waveform snippets from threshold crossings
- `_plot_snippets()`: Plot extracted snippets with polarity separation

### Data Processing Pipeline

1. **Data Loading**: Raw data loaded from HDF5 file
2. **Conversion**: Data multiplied by 0.6745 conversion factor
3. **Filtering**: Bandpass filter applied if frequencies are set
4. **Display**: Data displayed with manual or automatic Y-scaling
5. **Snippet Extraction**: If enabled, extract waveforms crossing threshold
6. **Snippet Analysis**: Separate positive/negative crossings, calculate averages
7. **Snippet Display**: Plot individual snippets and averages in dedicated subplot

## Usage Examples

### Setting Custom Filter Frequencies

```python
# In the GUI:
# 1. Enter "500" in Highpass box
# 2. Enter "2000" in Lowpass box
# 3. Click "Apply Filter"
# Result: 500-2000 Hz bandpass filter applied
```

### Manual Y-Axis Scaling

```python
# In the GUI:
# 1. Enter "-200" in Y Min box
# 2. Enter "200" in Y Max box
# 3. Press Enter
# Result: Y-axis fixed to -200 to +200 microvolts
```

### Channel Selection

```python
# In the GUI:
# 1. Click on desired channel in radio button panel
# Result: Immediately switches to selected channel
```

### Snippet Extraction

```python
# In the GUI:
# 1. Set threshold value (e.g., "50")
# 2. Check "Show Snippets" checkbox
# 3. Adjust "Before (ms)" to "0.8" and "After (ms)" to "1.2"
# 4. Set "Max Snippets" to "30"
# Result: Extracts up to 30 waveform snippets with 0.8ms before and 1.2ms after threshold crossings
```

## Testing

### Test Coverage

The enhancements include comprehensive tests:

1. **Unit Tests**: Test individual callback methods
2. **Integration Tests**: Test complete workflow
3. **GUI Tests**: Test widget creation and interaction

### Test Files

- `tests/test_raw_data_viewer.py`: Enhanced with new feature tests
- `test_enhanced_viewer.py`: Standalone test script for Issue #1 features
- `test_snippet_viewer.py`: Standalone test script for snippet extraction features

### Running Tests

```bash
# Run enhanced viewer test (Issue #1 features)
python3 test_enhanced_viewer.py

# Run snippet extraction test
python3 test_snippet_viewer.py

# Run unit tests (requires pytest)
python3 -m pytest tests/test_raw_data_viewer.py -v
```

## Backward Compatibility

All enhancements are backward compatible:

- Existing functionality remains unchanged
- Default values maintain original behavior
- New features are optional and don't affect existing workflows

## Configuration

### Default Settings

```python
DEFAULT_SETTINGS = {
    # Issue #1 settings
    'lowpass_freq': 3000.0,      # Hz
    'highpass_freq': 300.0,      # Hz
    'conversion_factor': 0.6745,  # ADC to microvolts
    'auto_scale': True,          # Y-axis auto-scaling
    'manual_ylims': None,        # Manual Y-limits (disabled)

    # Snippet extraction settings
    'show_snippets': False,      # Snippet extraction disabled by default
    'snippet_before_ms': 0.5,    # Time before threshold crossing (ms)
    'snippet_after_ms': 1.0,     # Time after threshold crossing (ms)
    'max_snippets': 50           # Maximum snippets to display
}
```

### Customization

Users can modify default values by editing the `InteractivePlotter` initialization:

```python
plotter = InteractivePlotter(
    data_loader=loader,
    initial_channel='electrode00',
    initial_group='raw',
    window_duration=5.0
)

# Customize after creation
plotter.lowpass_freq = 2500.0
plotter.highpass_freq = 250.0
plotter.data_conversion_factor = 0.5

# Customize snippet extraction
plotter.snippet_before_ms = 0.8
plotter.snippet_after_ms = 1.2
plotter.max_snippets = 30
```

## Future Enhancements

Potential future improvements:

1. **Dropdown Menu**: Full dropdown for channel selection (currently radio buttons)
2. **Filter Presets**: Quick buttons for common filter settings (spike, LFP, EMG)
3. **Save/Load Settings**: Persist user preferences
4. **Multiple Channel View**: Display multiple channels simultaneously
5. **Export Functionality**: Save filtered data or screenshots
6. **Snippet Analysis**: Advanced snippet clustering and classification
7. **Template Matching**: Compare snippets against template waveforms
8. **Snippet Statistics**: Real-time amplitude, width, and shape measurements
9. **Snippet Export**: Save extracted snippets to files for further analysis
10. **Multi-threshold Detection**: Support for multiple threshold levels simultaneously

## Troubleshooting

### Common Issues

1. **Filter Not Applied**: Ensure both frequency values are entered and "Apply Filter" is clicked
2. **Y-Limits Not Working**: Both Y Min and Y Max must be entered for manual scaling
3. **Channel Selection Issues**: Use "All Channels" button if radio buttons don't show desired channel
4. **No Snippets Extracted**: Ensure threshold is set and "Show Snippets" is checked
5. **Snippets Not Visible**: Check that threshold value is appropriate for data amplitude
6. **Too Many/Few Snippets**: Adjust "Max Snippets" value or threshold sensitivity

### Error Handling

The enhanced viewer includes robust error handling:

- Invalid frequency values are ignored
- Invalid Y-limit values are ignored
- Missing channels fall back to first available channel
- Filter errors don't crash the application
- Invalid snippet parameters are ignored
- Snippet extraction failures don't affect main display
- Boundary conditions handled for snippet windows

## Performance Considerations

- **Real-time Filtering**: Filters are applied on-demand, not pre-computed
- **Memory Usage**: Conversion factor applied during display, not stored
- **GUI Responsiveness**: Controls update immediately without blocking
- **Data Loading**: Unchanged from original implementation
- **Snippet Extraction**: Only processes current viewing window, not entire dataset
- **Snippet Caching**: Snippets cached per window to avoid re-extraction
- **Display Optimization**: Limited snippet count prevents performance degradation
- **Memory Management**: Snippets cleared when window changes

## Conclusion

These enhancements significantly improve the usability and functionality of the raw data viewer while maintaining full backward compatibility. Users now have precise control over filtering, scaling, channel selection, and waveform snippet extraction, making the tool more suitable for detailed neural data analysis and spike detection workflows.

The addition of snippet extraction brings the viewer closer to a complete spike sorting workflow, allowing researchers to:
- Visualize individual spike waveforms in real-time
- Assess data quality and threshold settings
- Identify different spike types and polarities
- Validate preprocessing parameters before full analysis

This makes the raw data viewer not just a visualization tool, but an integral part of the spike sorting pipeline for quality assessment and parameter optimization.
