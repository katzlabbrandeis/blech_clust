# Raw Data Viewer Enhancements

This document describes the enhancements made to the raw data viewer as part of issue #1.

## Overview

The raw data viewer has been enhanced with new GUI controls and functionality to provide better user control over data visualization and filtering.

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

## Technical Implementation

### Code Changes

#### InteractivePlotter Class Enhancements

```python
# New attributes added
self.manual_ylims = None
self.lowpass_freq = 3000.0
self.highpass_freq = 300.0
self.data_conversion_factor = 0.6745
```

#### New GUI Controls

1. **Filter Controls** (Row 5 of control grid):
   - Lowpass frequency text box
   - Highpass frequency text box

2. **Y-Limit Controls** (Row 5 of control grid):
   - Y Min text box
   - Y Max text box

3. **Channel Selection** (Row 4 of control grid):
   - Radio buttons for channel selection

#### New Callback Methods

- `_on_lowpass_change()`: Handle lowpass frequency changes
- `_on_highpass_change()`: Handle highpass frequency changes
- `_on_ylim_change()`: Handle Y-limit changes
- `_on_channel_radio_change()`: Handle channel selection
- `_update_filter_from_frequencies()`: Apply filter based on frequency settings

### Data Processing Pipeline

1. **Data Loading**: Raw data loaded from HDF5 file
2. **Conversion**: Data multiplied by 0.6745 conversion factor
3. **Filtering**: Bandpass filter applied if frequencies are set
4. **Display**: Data displayed with manual or automatic Y-scaling

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

## Testing

### Test Coverage

The enhancements include comprehensive tests:

1. **Unit Tests**: Test individual callback methods
2. **Integration Tests**: Test complete workflow
3. **GUI Tests**: Test widget creation and interaction

### Test Files

- `tests/test_raw_data_viewer.py`: Enhanced with new feature tests
- `test_enhanced_viewer.py`: Standalone test script

### Running Tests

```bash
# Run enhanced viewer test
python3 test_enhanced_viewer.py

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
    'lowpass_freq': 3000.0,      # Hz
    'highpass_freq': 300.0,      # Hz
    'conversion_factor': 0.6745,  # ADC to microvolts
    'auto_scale': True,          # Y-axis auto-scaling
    'manual_ylims': None         # Manual Y-limits (disabled)
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
```

## Future Enhancements

Potential future improvements:

1. **Dropdown Menu**: Full dropdown for channel selection (currently radio buttons)
2. **Filter Presets**: Quick buttons for common filter settings (spike, LFP, EMG)
3. **Save/Load Settings**: Persist user preferences
4. **Multiple Channel View**: Display multiple channels simultaneously
5. **Export Functionality**: Save filtered data or screenshots

## Troubleshooting

### Common Issues

1. **Filter Not Applied**: Ensure both frequency values are entered and "Apply Filter" is clicked
2. **Y-Limits Not Working**: Both Y Min and Y Max must be entered for manual scaling
3. **Channel Selection Issues**: Use "All Channels" button if radio buttons don't show desired channel

### Error Handling

The enhanced viewer includes robust error handling:

- Invalid frequency values are ignored
- Invalid Y-limit values are ignored
- Missing channels fall back to first available channel
- Filter errors don't crash the application

## Performance Considerations

- **Real-time Filtering**: Filters are applied on-demand, not pre-computed
- **Memory Usage**: Conversion factor applied during display, not stored
- **GUI Responsiveness**: Controls update immediately without blocking
- **Data Loading**: Unchanged from original implementation

## Conclusion

These enhancements significantly improve the usability and functionality of the raw data viewer while maintaining full backward compatibility. Users now have precise control over filtering, scaling, and channel selection, making the tool more suitable for detailed neural data analysis.