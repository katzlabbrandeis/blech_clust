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
