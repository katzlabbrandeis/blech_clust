# Ephys Data Module Documentation

The `ephys_data` module provides a comprehensive class for analyzing electrophysiology data. Here's how to use its main functionalities:

## Basic Usage

```python
from utils.ephys_data.ephys_data import ephys_data

# Initialize with data directory
data = ephys_data(data_dir='/path/to/data')

# Load basic data
data.get_unit_descriptors()  # Get unit information
data.get_spikes()           # Extract spike data

# Calculating firing rates and LFPs needs additional parameters
# If not set, the module will use default values
# See `Default Parameters` section for details

# data.firing_rate_params = { ... }
# or
# data.firing_rate_params = data.default_firing_params

# data.stft_params = { ... }
# or
# data.lfp_params = data.default_lfp_params

data.get_firing_rates()     # Calculate firing rates
data.get_lfps()            # Extract LFP data

# Same for STFT calculation
data.stft_params = { ... }
data.get_stft()            # Calculate spectrograms
```

## Key Features

### Spike and Firing Rate Analysis

```python
# Access spike data
spikes = data.spikes       # List of spike arrays per taste
firing = data.firing_array # 4D array of firing rates
```

### LFP Analysis

### Stable Units Analysis

```python
# Identify stable units across sessions
stable_units = data.get_stable_units()

# Access stable unit data
stable_spikes = data.stable_spikes
stable_firing = data.stable_firing
```

```python
# Access LFP data
lfps = data.lfp_array     # Raw LFP data
stft = data.stft_array    # Complex Spectrograms
amplitude = data.amplitude_array # STFT Amplitude (power)
phase = data.phase_array  # STFT Phase
```

### Region-Based Analysis

### Drift Results Analysis

```python
# Analyze drift results over time
drift_results = data.get_drift_results()

# Access drift data
drift_metrics = data.drift_metrics
drift_plots = data.drift_plots
```

```python
# Get units by brain region
data.get_region_units()
region_spikes = data.return_region_spikes('region_name')
region_firing = data.get_region_firing('region_name')

# Get LFPs by region
region_lfps, region_names = data.return_region_lfps()
```

### Laser Condition Analysis

```python
# Check for laser trials
data.check_laser()

# Separate data by laser condition
data.separate_laser_data()  # Separates spikes, firing rates, and LFPs

# Access separated data
on_spikes = data.on_spikes
off_spikes = data.off_spikes
on_firing = data.on_firing
off_firing = data.off_firing
```

### Trial Information

```python
# Get trial information
data.get_trial_info_frame()

# Sequester trials by condition
data.sequester_trial_inds()
data.get_sequestered_spikes()
data.get_sequestered_firing()
```

### Palatability Analysis

```python
# Calculate palatability correlations
data.calc_palatability()

# Access results
pal_df = data.pal_df       # Palatability rankings
pal_array = data.pal_array # Palatability correlations
```

## Default Parameters

The module comes with sensible defaults for various analyses:

```python
# Default firing rate parameters
default_firing_params = {
    'type': 'conv',
    'step_size': 25,
    'window_size': 250,
    'dt': 1,
    'baks_resolution': 25e-3,
    'baks_dt': 1e-3
}

# Default LFP parameters
default_lfp_params = {
    'freq_bounds': [1,300],
    'sampling_rate': 30000,
    'taste_signal_choice': 'Start',
    'fin_sampling_rate': 1000,
    'trial_durations': [2000,5000]
}
```

## Data Structure

The module expects data organized in a specific way:
- HDF5 file containing spike trains and LFP data
- JSON info file with experimental parameters
- Trial information CSV file

See the main repository documentation for details on data organization.
