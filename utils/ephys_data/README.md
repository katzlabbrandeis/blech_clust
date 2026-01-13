# Ephys Data Module

The `ephys_data` module provides tools for analyzing electrophysiology data, including spike trains, local field potentials (LFPs), and EMG signals.

**Full documentation:** [docs/reference/ephys-data.md](../../docs/reference/ephys-data.md)

## Quick Start

```python
from utils.ephys_data.ephys_data import ephys_data

# Initialize with data directory
data = ephys_data(data_dir='/path/to/data')

# Load basic data
data.get_unit_descriptors()
data.get_spikes()
data.get_firing_rates()
data.get_lfps()
```

## Modules

- **ephys_data.py** - Core class for loading and analyzing electrophysiology data
- **visualize.py** - Raster plots and firing rate heatmaps
- **lfp_processing.py** - LFP extraction and processing
- **BAKS.py** - Bayesian Adaptive Kernel Smoother for firing rate estimation
