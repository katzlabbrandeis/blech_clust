# Quick Start

## First Steps

Once installed, you can start processing your data:

1. **Prepare your data**: Ensure you have Intan RHD2132 recordings
2. **Set up experiment info**: Run `python blech_exp_info.py` to annotate channels
3. **Configure parameters**: Edit the parameter files in the `params/` directory
4. **Run the pipeline**: Use the convenience scripts or run individual modules

## Basic Workflow

### 1. Experiment Setup

```bash
# Activate the environment
conda activate blech_clust

# Navigate to your data directory
cd /path/to/your/data

# Run experiment info setup
python /path/to/blech_clust/blech_exp_info.py
```

This will guide you through annotating your channels and setting up experimental parameters.

### 2. Clustering Setup

```bash
# Run clustering setup
python /path/to/blech_clust/blech_clust.py
```

This creates the necessary directory structure and parameter files.

### 3. Common Average Referencing

```bash
# Perform common average referencing
python /path/to/blech_clust/blech_common_avg_reference.py
```

### 4. Spike Extraction and Clustering

```bash
# Run parallel processing
bash /path/to/blech_clust/blech_run_process.sh
```

This runs spike extraction and clustering in parallel across electrodes.

### 5. Post-Processing

```bash
# Add selected units to HDF5
python /path/to/blech_clust/blech_post_process.py

# Plot waveforms
python /path/to/blech_clust/blech_units_plot.py

# Generate spike arrays
python /path/to/blech_clust/blech_make_arrays.py
```

### 6. Quality Assessment

```bash
# Run QA checks
bash /path/to/blech_clust/blech_run_QA.sh

# Analyze unit characteristics
python /path/to/blech_clust/blech_units_characteristics.py

# Generate data summary
python /path/to/blech_clust/blech_data_summary.py

# Grade dataset quality
python /path/to/blech_clust/grade_dataset.py
```

## EMG Analysis

If you have EMG data, you can run the EMG analysis pipeline:

```bash
# Filter EMG signals
python /path/to/blech_clust/emg/emg_filter.py

# Choose your analysis approach:

# Option 1: BSA/STFT analysis
python /path/to/blech_clust/emg/emg_BSA_STFT.py

# Option 2: QDA-based gape detection
python /path/to/blech_clust/emg/emg_QDA.py
```

## Parameter Configuration

Key parameter files to configure:

- **clustering_params.json**: Clustering algorithm parameters
- **spike_detection_params.json**: Spike detection thresholds
- **emg_params.json**: EMG analysis parameters (if using EMG)

See the [Getting Started wiki](https://github.com/abuzarmahmood/blech_clust/wiki/Getting-Started#setting-up-params) for detailed parameter descriptions.

## Tips

- Always activate the conda environment before running scripts
- Check log files in the output directories for debugging
- Use the test dataset to verify your installation
- Refer to the [API Reference](../reference/index.md) for detailed function documentation

## Next Steps

- Explore the [Tutorials](../tutorials.md) for detailed walkthroughs
- Check the [API Reference](../reference/index.md) for function details
- Visit the [Blog](https://katzlabbrandeis.github.io/blech_clust/blogs/blogs_main.html) for tips and updates
