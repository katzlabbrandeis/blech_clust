# Ephys Data Module Documentation

The `ephys_data` module provides a comprehensive suite of tools for analyzing electrophysiology data, including spike trains, local field potentials (LFPs), and EMG signals.

## Modules

### ephys_data.py
Core class for loading and analyzing electrophysiology data.

### visualize.py
Functions for visualizing neural data, including raster plots and firing rate heatmaps.

### lfp_processing.py
Tools for extracting and processing LFP data from raw recordings.

### BAKS.py
Implementation of Bayesian Adaptive Kernel Smoother for firing rate estimation.

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

```python
# Access LFP data
lfps = data.lfp_array     # Raw LFP data
stft = data.stft_array    # Complex Spectrograms
amplitude = data.amplitude_array # STFT Amplitude (power)
phase = data.phase_array  # STFT Phase
```

### Region-Based Analysis

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

## Example Workflows

### Ephys Data Processing

```python
from utils.ephys_data.ephys_data import ephys_data
import numpy as np
import matplotlib.pyplot as plt

# Load data
data = ephys_data(data_dir='/path/to/data')

# Get basic information
data.get_unit_descriptors()
data.get_spikes()
data.get_firing_rates()
data.get_lfps()

# Analyze by brain region
data.get_region_units()
print(f"Available regions: {data.region_names}")

# Get spikes for a specific region
region_spikes = data.return_region_spikes('GC')  # Gustatory cortex example

# Analyze laser conditions (if applicable)
data.check_laser()
if data.laser_exists:
    data.separate_laser_data()
    # Compare firing rates between conditions
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Laser OFF')
    plt.imshow(np.mean(data.off_firing[0], axis=0), aspect='auto', cmap='viridis')
    plt.subplot(1, 2, 2)
    plt.title('Laser ON')
    plt.imshow(np.mean(data.on_firing[0], axis=0), aspect='auto', cmap='viridis')
    plt.tight_layout()
    plt.show()

# Calculate palatability correlation
data.calc_palatability()
plt.figure(figsize=(10, 6))
plt.imshow(data.pal_array, aspect='auto', cmap='viridis')
plt.colorbar(label='|Palatability Correlation|')
plt.xlabel('Time (bins)')
plt.ylabel('Neuron')
plt.title('Palatability Correlation Over Time')
plt.show()
```

### Visualization

```python
import numpy as np
import matplotlib.pyplot as plt
from utils.ephys_data.visualize import raster, firing_overview

# Create sample spike data (binary array where 1 indicates a spike)
spike_array = np.zeros((10, 100))  # 10 trials, 100 time points
spike_array[2, 20:25] = 1  # Add some spikes
spike_array[5, 40:45] = 1
spike_array[7, 60:65] = 1

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Generate raster plot
raster(ax, spike_array, marker='|', color='black')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Trial')
ax.set_title('Example Raster Plot')
plt.show()

# Create sample firing rate data for multiple neurons
# Shape: (neurons, trials, time points)
n_neurons = 4
n_trials = 10
n_timepoints = 100
data = np.random.rand(n_neurons, n_trials, n_timepoints)

# Add some structure to the data
for i in range(n_neurons):
    # Create a peak at different times for each neuron
    peak_time = 20 + i*15
    data[i, :, peak_time-5:peak_time+5] += 2

# Generate firing rate overview
fig, ax = firing_overview(
    data,
    t_vec=np.arange(n_timepoints) * 10,  # 10ms bins
    cmap='viridis',
    cmap_lims='shared',
    subplot_labels=np.arange(n_neurons),
    zscore_bool=True,
    figsize=(12, 10)
)
plt.tight_layout()
plt.show()
```

### LFP Processing

```python
from utils.ephys_data import lfp_processing
import numpy as np
import tables
import matplotlib.pyplot as plt

# Set parameters for LFP extraction
params = {
    'freq_bounds': [1, 300],          # Frequency range in Hz
    'sampling_rate': 30000,           # Original sampling rate
    'taste_signal_choice': 'Start',   # Trial alignment
    'fin_sampling_rate': 1000,        # Final sampling rate
    'dig_in_list': [0, 1, 2, 3],      # Digital inputs to process
    'trial_durations': [2000, 5000]   # Pre/post trial durations in ms
}

# Extract LFPs from raw data files
lfp_processing.extract_lfps(
    dir_name='/path/to/data',
    **params
)

# After extraction, load and analyze the LFP data
with tables.open_file('/path/to/data/session.h5', 'r') as hf5:
    # Get LFP data for a specific taste
    lfp_data = hf5.root.Parsed_LFP.dig_in_0_LFPs[:]  # Shape: (channels, trials, time)

# Identify good quality trials
good_trials_bool = lfp_processing.return_good_lfp_trial_inds(
    data=lfp_data,
    MAD_threshold=3  # Number of MADs to use as threshold
)

# Get only the good trials
good_lfp_data = lfp_data[:, good_trials_bool, :]

# Plot mean of good trials
plt.figure(figsize=(12, 6))
plt.plot(np.mean(good_lfp_data, axis=(0, 1)), 'b-')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.title('Mean LFP (Good Trials Only)')
plt.show()
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
