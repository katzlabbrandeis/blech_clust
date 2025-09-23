# Raw Data Viewer Documentation

The Raw Data Viewer is an interactive tool for visualizing and exploring raw neural recording data stored in HDF5 format. It provides memory-efficient data loading, real-time filtering, and intuitive navigation controls.

## Features

- **Memory-efficient data loading**: Load only the data you need to view
- **Interactive time navigation**: Scroll through time or jump to specific moments
- **Real-time filtering**: Apply bandpass, highpass, lowpass, and notch filters
- **Threshold visualization**: Display threshold lines for spike detection
- **Multi-channel support**: Switch between different recording channels
- **Configurable parameters**: Customize viewing window, filters, and display options

## Installation

The Raw Data Viewer is part of the blech_clust package. Ensure you have the required dependencies:

```bash
pip install numpy scipy matplotlib tables
```

## Quick Start

### Basic Usage

```bash
# View data from a directory (will find the HDF5 file automatically)
python raw_data_viewer.py /path/to/data/

# View specific HDF5 file
python raw_data_viewer.py /path/to/data.h5

# Start with specific channel and filter
python raw_data_viewer.py /path/to/data/ --channel electrode00 --filter spike
```

### Command Line Options

```bash
python raw_data_viewer.py <data_path> [options]

Arguments:
  data_path             Path to data directory or HDF5 file

Options:
  -c, --channel         Initial channel to display
  -g, --group           Data group to use (default: raw)
  -w, --window          Window duration in seconds (default: 10.0)
  -f, --filter          Filter type: spike, lfp, emg, none, or "low-high"
  -t, --threshold       Threshold value to display
  -s, --sampling-rate   Sampling rate in Hz (auto-detect if not specified)
  --config              Path to JSON configuration file
  --save-config         Save current configuration to file and exit
```

### Examples

```bash
# View with 5-second window and spike filter
python raw_data_viewer.py /data/ --window 5.0 --filter spike

# View EMG data with custom threshold
python raw_data_viewer.py /data/ --group raw_emg --channel emg00 --threshold 100

# Use custom bandpass filter (300-3000 Hz)
python raw_data_viewer.py /data/ --filter "300-3000"

# Save configuration for later use
python raw_data_viewer.py /data/ --window 8.0 --filter lfp --save-config my_config.json

# Load saved configuration
python raw_data_viewer.py /data/ --config my_config.json
```

## Interactive Controls

### Mouse Controls
- **Scroll wheel**: Navigate through time
- **Double-click**: Jump to clicked time point
- **Click and drag**: Pan the view (if implemented)

### Keyboard Shortcuts
- **Left/Right arrows**: Navigate time by half window duration
- **Up/Down arrows**: Switch between channels
- **'f' key**: Cycle through filter types (none → spike → LFP → none)
- **'r' key**: Reset view to beginning
- **'a' key**: Toggle auto-scaling

### GUI Controls
- **Time slider**: Navigate to specific time
- **Window duration box**: Change viewing window length
- **Threshold box**: Set threshold value
- **Navigation buttons**: Previous/Next/Jump controls
- **Filter button**: Cycle through filter types
- **Channel button**: Cycle through available channels

## Configuration Files

You can save and load viewer configurations using JSON files:

```json
{
  "window_duration": 10.0,
  "group": "raw",
  "channel": "electrode00",
  "filter_type": "spike",
  "threshold": 150.0,
  "sampling_rate": 30000.0
}
```

### Filter Types

- **`none`**: No filtering applied
- **`spike`**: Bandpass filter (300-3000 Hz) for spike detection
- **`lfp`**: Bandpass filter (1-300 Hz) for local field potentials
- **`emg`**: Bandpass filter (10-500 Hz) for EMG signals
- **`"low-high"`**: Custom bandpass filter (e.g., "100-1000")

## Data Requirements

The viewer expects HDF5 files with the following structure:

```
data.h5
├── raw/                    # Raw electrode data
│   ├── electrode00         # Individual electrode arrays
│   ├── electrode01
│   └── ...
├── raw_emg/               # EMG data (optional)
│   ├── emg00
│   └── ...
└── sampling_rate          # Sampling rate in Hz
```

### Data Format
- **Data type**: int16 (recommended) or float32
- **Shape**: 1D arrays with time samples
- **Sampling rate**: Stored as `/sampling_rate` in HDF5 file

## API Reference

### RawDataLoader

The core data loading class:

```python
from utils.raw_data_loader import RawDataLoader

# Initialize loader
loader = RawDataLoader('data.h5', sampling_rate=30000)

# Get available channels
channels = loader.get_available_channels()

# Load data from specific time range
data, time = loader.load_channel_data(
    channel='electrode00',
    group='raw',
    start_time=10.0,
    end_time=20.0
)

# Load multiple channels
data_array, time = loader.load_multiple_channels(
    channels=['electrode00', 'electrode01'],
    group='raw',
    start_time=0,
    end_time=5.0
)
```

### SignalFilter

Real-time filtering capabilities:

```python
from utils.signal_filters import SignalFilter, FilterBank

# Create custom filter
filt = SignalFilter(
    sampling_rate=30000,
    filter_type='bandpass',
    low_freq=300,
    high_freq=3000
)

# Use predefined filters
spike_filter = FilterBank.create_spike_filter(30000)
lfp_filter = FilterBank.create_lfp_filter(30000)

# Apply filter
filtered_data = filt.filter_data(raw_data)
```

### InteractivePlotter

Interactive visualization:

```python
from utils.interactive_viewer import InteractivePlotter

# Create plotter
plotter = InteractivePlotter(
    data_loader=loader,
    initial_channel='electrode00',
    window_duration=10.0
)

# Set filter
plotter.set_filter(spike_filter)

# Set threshold
plotter.set_threshold(100.0)

# Jump to specific time
plotter.jump_to_time(30.0)

# Show interactive plot
plotter.show()
```

## Performance Considerations

### Memory Usage
- The viewer loads only the requested time window into memory
- Typical memory usage: ~10-50 MB for 10-second windows
- Large datasets (hours of recording) can be viewed without memory issues

### Loading Speed
- Initial loading: ~1-2 seconds for typical datasets
- Time navigation: Near-instantaneous for local data
- Filter application: Real-time for most filter types

### Optimization Tips
- Use shorter time windows for faster navigation
- Store data on fast storage (SSD) for best performance
- Consider downsampling very high sampling rate data for visualization

## Troubleshooting

### Common Issues

**"No HDF5 files found"**
- Ensure your data directory contains `.h5` files
- Check file permissions
- Verify HDF5 file structure

**"Channel not found"**
- Check available channels with `loader.get_available_channels()`
- Verify group name ('raw' vs 'raw_emg')
- Ensure channel naming follows expected format

**"Sampling rate not detected"**
- Add sampling rate to HDF5 file as `/sampling_rate`
- Specify sampling rate manually with `--sampling-rate` option

**Slow performance**
- Reduce window duration
- Check available memory
- Ensure data is on fast storage

**Filter artifacts**
- Try different filter orders
- Check filter stability with `get_filter_info()`
- Use zero-phase filtering for critical applications

### Debug Mode

Enable verbose output for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Advanced Usage

### Custom Filters

Create custom filters for specific applications:

```python
# Custom notch filter for line noise
from utils.signal_filters import NotchFilter
notch_60hz = NotchFilter(sampling_rate=30000, notch_freq=60.0, quality=30)

# Multiple filters in sequence
from utils.signal_filters import apply_multiple_filters
filters = [bandpass_filter, notch_60hz]
filtered_data = apply_multiple_filters(raw_data, filters)
```

### Batch Processing

Process multiple files or channels:

```python
import glob
from utils.raw_data_loader import load_sample_data

# Process all HDF5 files in directory
for hdf5_file in glob.glob('*.h5'):
    data, time = load_sample_data(hdf5_file, duration=10.0)
    # Process data...
```

### Integration with Analysis Pipelines

The viewer components can be integrated into analysis workflows:

```python
# Extract data for analysis
with RawDataLoader('data.h5') as loader:
    # Get spike band data
    spike_filter = FilterBank.create_spike_filter(loader.sampling_rate)
    raw_data, _ = loader.load_channel_data('electrode00', 'raw')
    spike_data = spike_filter.filter_data(raw_data)
    
    # Detect spikes, run analysis, etc.
```

## Contributing

To contribute to the Raw Data Viewer:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements/pip_requirements_base.txt

# Run tests
python -m pytest tests/test_raw_data_viewer.py

# Run specific test
python -m pytest tests/test_raw_data_loader.py::TestRawDataLoader::test_load_channel_data
```

## License

The Raw Data Viewer is part of the blech_clust package and follows the same license terms.

## Support

For issues and questions:
- Check the troubleshooting section above
- Review existing GitHub issues
- Create a new issue with detailed information about your problem
- Include sample data and configuration files when possible