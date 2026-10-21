# Core Pipeline

The core pipeline modules handle the main spike sorting workflow from raw data to sorted units.

## Pipeline Modules

### blech_exp_info.py

Pre-clustering step to annotate channels and save experimental parameters.

**Key Functions:**

- Electrode layout configuration
- Digital input selection
- Taste/stimulus naming
- Palatability ranking
- Laser configuration

**Usage:**

```bash
python blech_exp_info.py /path/to/data
```

### blech_init.py

Initialize directories and prepare data for clustering.

**Key Functions:**

- Directory structure creation
- Data file organization
- Initial parameter setup

**Usage:**

```bash
python blech_init.py
```

### blech_common_avg_reference.py

Perform common average referencing on electrode data.

**Key Functions:**

- CAR group processing
- Signal referencing
- Artifact reduction

**CAR Groups:**

CAR (Common Average Reference) groups are used to organize electrodes that should be referenced together during preprocessing. When configuring CAR groups:

- Group electrodes from the same brain region together (e.g., all electrodes in the right gustatory cortex as `GC1`, left gustatory cortex as `GC2`)
- Two CAR group names are reserved and have special behavior:
  - **`none`**: Channels marked with CAR group `none` are excluded from CAR processing and will not be analyzed
  - **`emg`** (or any name containing "emg"): Channels with CAR groups containing "emg" are treated as EMG channels and processed separately for EMG analysis

**Usage:**

```bash
python blech_common_avg_reference.py
```

### blech_process.py

Core spike extraction and clustering module.

**Key Functions:**

- Spike detection
- Feature extraction
- Clustering algorithms
- UMAP dimensionality reduction

**Spike Detection:**

By default, spike detection uses a rolling (adaptive) threshold that computes the noise level independently for each time window. This allows detection to adapt to local noise variations across the recording.

- `use_rolling_threshold`: Enable/disable rolling threshold (default: true)
- `rolling_threshold_window`: Window size in seconds (default: 5.0)
- `rolling_threshold_step`: Step size in seconds (default: 5.0)
- `waveform_threshold`: Threshold multiplier in standard deviations (default: 5)

When rolling threshold is enabled, the threshold is computed as `waveform_threshold * MAD / 0.6745` for each window, where MAD is the Median Absolute Deviation. This is more robust to outliers than standard deviation.

Set `use_rolling_threshold: false` in sorting_params.json to use a single global threshold computed over the entire recording.

Example rolling threshold plots in the [migration guide](../getting-started/migration-guide/qa-improvements.md)

**Usage:**

```bash
# Usually called via blech_run_process.sh for parallel execution
python blech_process.py <electrode_number>
```

### blech_post_process.py

Add selected units to HDF5 file after manual curation.

**Key Functions:**

- Unit selection
- HDF5 file updates
- Metadata management

**Usage:**

```bash
python blech_post_process.py
```

### blech_units_plot.py

Plot waveforms of selected spikes for visualization.

**Key Functions:**

- Waveform plotting
- Unit visualization
- Quality metrics display

**Usage:**

```bash
python blech_units_plot.py
```

### blech_make_arrays.py

Generate spike-train arrays for analysis.

**Key Functions:**

- Spike train generation
- Trial alignment
- Array formatting

**Usage:**

```bash
python blech_make_arrays.py
```

## Pipeline Flow

```
Raw Data
    ↓
blech_exp_info.py (Setup)
    ↓
blech_init.py (Initialization)
    ↓
blech_common_avg_reference.py (Referencing)
    ↓
blech_run_process.sh (Parallel Processing)
    ↓
blech_post_process.py (Unit Selection)
    ↓
blech_units_plot.py (Visualization)
    ↓
blech_make_arrays.py (Array Generation)
    ↓
Sorted Units
```

## Configuration Files

The pipeline uses several JSON configuration files:

- `sorting_params.json` - Clustering parameters
- `spike_detection_params.json` - Detection thresholds
- `waveform_classifier_params.json` - Classifier settings

## See Also

- [Getting Started](../getting-started/installation.md)
- [Tutorials](../tutorials.md)
- [Utilities](utilities.md)
