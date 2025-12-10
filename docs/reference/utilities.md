# Utilities

Utility modules provide supporting functionality for the spike sorting pipeline.

## blech_utils

Core utility functions used throughout the codebase.

### Key Classes and Functions

#### Tee

Redirect stdout to both console and file for logging.

```python
from blech_utils import Tee

# Redirect output to file and console
sys.stdout = Tee('/path/to/logfile.txt')
print("This goes to both console and file")
```

#### path_handler

Handle file paths and directory operations.

```python
from blech_utils import path_handler

# Get data directory
data_dir = path_handler.get_data_dir()

# Create output directory
path_handler.make_dir('/path/to/output')
```

#### imp_metadata

Import and manage experimental metadata.

```python
from blech_utils import imp_metadata

# Load metadata
metadata = imp_metadata('/path/to/data')

# Access metadata fields
tastes = metadata['taste_names']
dig_ins = metadata['dig_in_channels']
```

## Clustering Utilities

Located in `utils/clustering/`, these modules provide clustering algorithms and tools.

### Key Functions

- Spike clustering algorithms
- Feature extraction
- Cluster validation
- Merge/split operations

## Data Management

### ephys_data Module

Comprehensive data handling for electrophysiology recordings.

See [Ephys Data](ephys-data.md) for detailed documentation.

## Quality Assurance

### qa_utils Module

Tools for dataset quality assessment and validation.

See [QA Tools](qa-tools.md) for detailed documentation.

## Helper Scripts

### infer_rnn_rates.py

Infer firing rates from spike trains using RNN.

```bash
python utils/infer_rnn_rates.py <data_dir> [options]
```

Options:

- `--train_steps`: Number of training steps
- `--hidden_size`: RNN hidden layer size
- `--bin_size`: Spike binning size
- `--retrain`: Force model retraining

### blech_data_summary.py

Generate comprehensive dataset summary.

```bash
python utils/blech_data_summary.py
```

### grade_dataset.py

Grade dataset quality based on metrics.

```bash
python utils/grade_dataset.py
```

## Configuration Management

### Parameter Files

Utilities for loading and managing parameter files:

- JSON parameter loading
- Parameter validation
- Default value handling

### Example

```python
import json

# Load parameters
with open('params/sorting_params.json', 'r') as f:
    params = json.load(f)

# Access parameters
max_clusters = params['max_clusters']
min_cluster_size = params['min_cluster_size']
```

## File I/O

### HDF5 Operations

Functions for reading and writing HDF5 files:

```python
import tables

# Open HDF5 file
with tables.open_file('data.h5', 'r') as hf5:
    # Read spike times
    spike_times = hf5.root.spike_times[:]

    # Read unit information
    units = hf5.root.units[:]
```

### Binary Data

Functions for reading Intan binary data:

- Amplifier data (`.dat` files)
- Digital input data (DIN files)
- Auxiliary input data

## See Also

- [Core Pipeline](core-pipeline.md)
- [Ephys Data](ephys-data.md)
- [QA Tools](qa-tools.md)
