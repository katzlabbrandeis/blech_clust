# Ephys Data

The `ephys_data` module provides comprehensive tools for handling electrophysiology data.

## Overview

Located in `utils/ephys_data/`, this module handles:

- Data loading and management
- Spike train access
- Unit information
- Trial alignment
- Metadata handling

## ephys_data Class

Main class for accessing electrophysiology data.

### Initialization

```python
from utils.ephys_data import ephys_data

# Create data handler
data = ephys_data('/path/to/data')
```

### Key Attributes

#### Spike Data

```python
# Access spike trains
spike_trains = data.spikes  # All spike trains
unit_spikes = data.spikes[unit_num]  # Specific unit

# Get spike times
spike_times = data.spike_times
```

#### Unit Information

```python
# Get unit descriptors
units = data.units

# Get number of units
n_units = data.n_units

# Get unit types
unit_types = data.unit_types
```

#### Trial Data

```python
# Get trial information
trials = data.trials

# Get stimulus information
dig_in_trials = data.dig_in_trials

# Get trial times
trial_times = data.trial_times
```

#### Metadata

```python
# Access experimental metadata
metadata = data.metadata

# Get taste names
taste_names = data.taste_names

# Get sampling rate
sampling_rate = data.sampling_rate
```

### Key Methods

#### Data Loading

```python
# Load spike data
data.load_spikes()

# Load unit information
data.load_units()

# Load trial data
data.load_trials()
```

#### Data Filtering

```python
# Filter by unit type
good_units = data.get_units_by_type('single')

# Filter by quality metrics
high_quality = data.get_units_by_quality(threshold=0.8)
```

#### Trial Alignment

```python
# Get aligned spike trains
aligned_spikes = data.get_aligned_spikes(
    unit_num=0,
    dig_in=0,
    time_window=[-1000, 2000]
)
```

## Data Structure

### HDF5 File Organization

```
data.h5
├── spike_times/          # Spike times for each unit
├── spike_waveforms/      # Spike waveforms
├── unit_descriptor/      # Unit metadata
├── digital_in/           # Digital input data
│   ├── dig_in_0/
│   ├── dig_in_1/
│   └── ...
└── metadata/             # Experimental metadata
```

### Spike Train Format

Spike trains are stored as arrays of spike times in milliseconds:

```python
# Example spike train
spike_train = [125.5, 342.1, 567.8, 891.2, ...]
```

### Unit Descriptor Format

Unit information includes:

- Unit number
- Electrode number
- Single/multi-unit classification
- Quality metrics
- Waveform statistics

## Usage Examples

### Basic Data Access

```python
from utils.ephys_data import ephys_data

# Load data
data = ephys_data('/path/to/data')

# Get all spike trains
all_spikes = data.spikes

# Get specific unit
unit_0_spikes = data.spikes[0]

# Get unit information
units = data.units
print(f"Found {len(units)} units")
```

### Trial-Aligned Analysis

```python
# Get spikes aligned to stimulus onset
aligned_spikes = data.get_aligned_spikes(
    unit_num=0,
    dig_in=0,
    time_window=[-1000, 2000]  # -1s to +2s around stimulus
)

# Calculate firing rate
import numpy as np
bin_size = 50  # ms
bins = np.arange(-1000, 2000, bin_size)
firing_rate, _ = np.histogram(aligned_spikes, bins=bins)
firing_rate = firing_rate / (bin_size / 1000)  # Convert to Hz
```

### Quality Filtering

```python
# Get high-quality single units
good_units = [
    unit for unit in data.units
    if unit['single_unit'] and unit['quality'] > 0.8
]

print(f"Found {len(good_units)} high-quality single units")
```

## Advanced Features

### Custom Data Loading

```python
# Load specific data subsets
data.load_spikes(units=[0, 1, 2])  # Load only specific units
data.load_trials(dig_ins=[0, 1])   # Load only specific trials
```

### Data Export

```python
# Export to common formats
data.export_to_nwb('/path/to/output.nwb')  # Neurodata Without Borders
data.export_to_mat('/path/to/output.mat')  # MATLAB format
```

## See Also

- [Core Pipeline](core-pipeline.md)
- [Utilities](utilities.md)
- [Tutorials](../tutorials.md)
- [Module Documentation](https://github.com/katzlabbrandeis/blech_clust/blob/master/utils/ephys_data/README.md)
