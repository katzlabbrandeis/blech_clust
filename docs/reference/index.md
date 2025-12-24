# API Reference

This section contains the API documentation for blech_clust modules.

## Overview

The blech_clust codebase is organized into several key modules:

- **[Core Pipeline](core-pipeline.md)** - Main spike sorting pipeline modules
- **[Utilities](utilities.md)** - Helper functions and utility classes
- **[Ephys Data](ephys-data.md)** - Electrophysiology data analysis tools
- **[QA Tools](qa-tools.md)** - Quality assurance and validation tools
- **[EMG Analysis](emg-analysis.md)** - EMG signal processing and analysis

## Module Organization

### Core Pipeline Modules

Located in the repository root, these modules form the main spike sorting pipeline:

- `blech_exp_info.py` - Experiment setup and metadata
- `blech_init.py` - Directory initialization and data preparation
- `blech_common_avg_reference.py` - Common average referencing
- `blech_process.py` - Spike extraction and clustering
- `blech_post_process.py` - Post-processing and unit selection
- `blech_units_plot.py` - Waveform visualization
- `blech_make_arrays.py` - Spike train array generation

### Utility Modules

Located in `utils/`, these provide supporting functionality:

- `blech_utils.py` - Core utility functions
- `clustering/` - Clustering algorithms
- `ephys_data/` - Electrophysiology data handling
- `qa_utils/` - Quality assurance tools

### EMG Modules

Located in `emg/`, these handle EMG signal analysis:

- `emg_filter.py` - EMG signal filtering
- `emg_freq_setup.py` - Frequency analysis setup
- `get_gapes_Li.py` - Gape detection using QDA

## Using the API

### Importing Modules

```python
# Import utility functions
from utils.blech_utils import Tee, path_handler, imp_metadata

# Import ephys data tools
from utils.ephys_data import ephys_data

# Import clustering utilities
from utils.clustering import clustering
```

### Example Usage

```python
# Load experimental data
from utils.ephys_data import ephys_data

# Create data handler
data = ephys_data('/path/to/data')

# Access spike trains
spike_trains = data.spikes

# Get unit information
units = data.units
```

## Documentation Format

Each module page includes:

- **Overview** - Module purpose and functionality
- **Key Functions/Classes** - Main components with descriptions
- **Usage Examples** - Code examples demonstrating usage
- **Parameters** - Detailed parameter descriptions
- **Returns** - Return value descriptions

## Contributing

To improve the API documentation:

1. Update docstrings in the source code following NumPy format
2. Submit a pull request with your changes
3. Documentation will be automatically rebuilt

See [CONTRIBUTING.md](https://github.com/katzlabbrandeis/blech_clust/blob/master/CONTRIBUTING.md) for guidelines.
