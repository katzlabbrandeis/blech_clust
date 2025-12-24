# Workflow Diagrams

This page provides additional visual representations and workflow details for the blech_clust pipeline.

## Main Spike-Sorting Pipeline

For the complete spike-sorting pipeline workflow (including the operations workflow diagram, detailed steps, and nomnoml schema), please refer to the [README](https://github.com/katzlabbrandeis/blech_clust#main-spike-sorting-pipeline).

## Quality Assessment Workflow

```
blech_unit_characteristics.py → blech_data_summary.py → grade_dataset.py
```

## Additional Workflow Details

### Spike Sorting Text Flow

```
[blech_exp_info] -> [blech_init]
[blech_init] -> [blech_common_average_reference]
[blech_common_average_reference] -> [bash blech_run_process.sh]
[bash blech_run_process.sh] -> [blech_post_process]
[blech_post_process] -> [blech_units_plot]
[blech_units_plot] -> [blech_make_arrays]
[blech_make_arrays] -> [bash blech_run_QA.sh]
[bash blech_run_QA.sh] -> [blech_unit_characteristics]
[blech_unit_characteristics] -> [blech_data_summary]
[blech_data_summary] -> [grade_dataset]
```

### EMG Analysis Workflows

**Shared Steps:**

```
[blech_init] -> [blech_make_arrays]
[blech_make_arrays] -> [emg_filter]
```

**BSA/STFT Branch:**

```
[emg_filter] -> [emg_freq_setup]
[emg_freq_setup] -> [bash blech_emg_jetstream_parallel.sh]
[bash blech_emg_jetstream_parallel.sh] -> [emg_freq_post_process]
[emg_freq_post_process] -> [emg_freq_plot]
```

**QDA Branch (Jenn Li):**

```
[emg_freq_setup] -> [get_gapes_Li]
```

## Using the Diagrams

### Viewing the Workflow

The operations workflow visual (available in the [README](https://github.com/katzlabbrandeis/blech_clust#main-spike-sorting-pipeline)) provides a high-level overview of the entire pipeline, showing how different components interact.

### Generating Custom Diagrams

1. Visit [nomnoml.com](https://www.nomnoml.com/)
2. Copy the nomnoml schema code from the [README](https://github.com/katzlabbrandeis/blech_clust#nomnoml-schema)
3. Paste it into the editor
4. Modify as needed for your specific use case
5. Export as PNG or SVG

### Understanding the Flow

- **Spike Sorting**: The main pipeline processes raw Intan data through clustering and quality assessment
- **EMG Analysis**: Two parallel branches for different analysis approaches
  - **BSA/STFT**: Frequency-based analysis using Bayesian methods
  - **QDA**: Gape detection using quadratic discriminant analysis

## See Also

- [Main Spike-Sorting Pipeline (README)](https://github.com/katzlabbrandeis/blech_clust#main-spike-sorting-pipeline)
- [Core Pipeline Documentation](reference/core-pipeline.md)
- [Tutorials](tutorials.md)
- [Getting Started](getting-started/installation.md)
