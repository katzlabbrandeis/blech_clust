# File Mapping

This page provides a complete reference for files that were relocated from the original blech_clust to the current fork.

## Root Level to utils/

Utility scripts moved from the repository root to `utils/`:

| Original Location | New Location |
|-------------------|--------------|
| `clustering.py` | `utils/clustering.py` |
| `read_file.py` | `utils/read_file.py` |
| `split_h5_files.py` | `utils/blech_split_h5_files.py` |
| `blech_waveforms_datashader.py` | `utils/blech_waveforms_datashader.py` |
| `blech_held_units_detect.py` | `utils/blech_held_units_detect.py` |
| `blech_hdf5_repack.py` | `utils/blech_hdf5_repack.py` |
| `blech_nex_convert.py` | `utils/blech_nex_convert.py` |
| `fix_laser_sampling_errors.py` | `utils/fix_laser_sampling_errors.py` |
| `memory_monitor.py` | `utils/ram_monitor.py` |

### Import Changes

If you have scripts that imported from the original locations, update your imports:

```python
# Original
from clustering import ...
from read_file import ...

# Current
from utils.clustering import ...
from utils.read_file import ...
```

---

## EMG Scripts to emg/

All EMG-related scripts have been consolidated in the `emg/` directory:

| Original Location | New Location |
|-------------------|--------------|
| `filter_emg.py` | `emg/emg_filter.py` |
| `emg_local_BSA.py` | `emg/emg_local_BSA_execute.py` |
| `emg_local_BSA_execute.py` | `emg/emg_local_BSA_execute.py` |
| `emg_local_BSA_post_process.py` | `emg/emg_freq_post_process.py` |
| `emg_BSA_segmentation.py` | `emg/emg_freq_setup.py` |
| `emg_BSA_segmentation_plot.py` | `emg/emg_freq_plot.py` |
| `get_gapes_Li.py` | `emg/gape_QDA_classifier/get_gapes_Li.py` |
| `detect_peaks.py` | `emg/gape_QDA_classifier/detect_peaks.py` |
| `QDA_nostd_no_first.mat` | `emg/gape_QDA_classifier/QDA_nostd_no_first.mat` |

### Merged Functionality

Some EMG scripts were merged or integrated:

| Original | Current | Notes |
|----------|---------|-------|
| `emg_make_arrays.py` | `blech_make_arrays.py` | EMG array creation integrated into main array generation |

---

## Renamed Scripts

Some scripts were renamed for clarity or consistency:

| Original Name | New Name | Notes |
|---------------|----------|-------|
| `blech_unit_visualize.py` | `blech_units_plot.py` | Enhanced with additional visualizations |
| `blech_units_similarity.py` | `utils/qa_utils/unit_similarity.py` | Moved to QA utilities |
| `memory_monitor.py` | `utils/ram_monitor.py` | Renamed for clarity |

---

## New utils/ Structure

The `utils/` directory contains several new subdirectories:

```
utils/
├── __init__.py
├── blech_channel_profile.py
├── blech_data_summary.py
├── blech_hdf5_repack.py
├── blech_held_units_detect.py
├── blech_nex_convert.py
├── blech_post_process_utils.py
├── blech_process_utils.py
├── blech_reload_amp_digs.py
├── blech_spike_features.py
├── blech_split_h5_files.py
├── blech_utils.py
├── blech_waveforms_datashader.py
├── cluster_stability.py
├── clustering.py
├── fix_laser_sampling_errors.py
├── grade_dataset.py
├── grading_metrics.json
├── importrhdutilities.py
├── infer_rnn_rates.py
├── makeRaisedCosBasis.py
├── ram_monitor.py
├── read_file.py
├── ephys_data/
│   ├── BAKS.py
│   ├── ephys_data.py
│   ├── lfp_processing.py
│   ├── visualize.py
│   ├── convenience_scripts/
│   └── tests/
├── qa_utils/
│   ├── channel_corr.py
│   ├── drift_check.py
│   ├── elbo_drift.py
│   └── unit_similarity.py
└── umap_plotting/
    ├── bash_umap_parallel.sh
    └── umap_plots.py
```

---

## New emg/ Structure

The `emg/` directory organization:

```
emg/
├── __init__.py
├── emg_filter.py
├── emg_freq_plot.py
├── emg_freq_post_process.py
├── emg_freq_setup.py
├── emg_local_BSA_execute.py
├── emg_local_STFT_execute.py
├── emg_run_pipeline.sh
├── _archive/
│   └── emg_process_comparison.py
├── gape_QDA_classifier/
│   ├── QDA_classifier.py
│   ├── QDA_nostd_no_first.mat
│   ├── detect_peaks.py
│   ├── get_gapes_Li.py
│   └── _experimental/
└── utils/
    └── emg_reload_raw_data.py
```

---

## Command Line Usage Changes

If you were running scripts from the command line, update your paths:

```bash
# Original
python filter_emg.py
python get_gapes_Li.py
python blech_held_units_detect.py

# Current
python emg/emg_filter.py
python emg/gape_QDA_classifier/get_gapes_Li.py
python utils/blech_held_units_detect.py
```

Or use the convenience scripts:

```bash
# EMG pipeline
bash emg/emg_run_pipeline.sh /path/to/data

# Main pipeline
bash blech_autosort.sh /path/to/data
```

---

## Finding Functionality

If you're looking for specific functionality from the original codebase:

| Looking for... | Now in... |
|----------------|-----------|
| Spike clustering | `utils/clustering.py` |
| File reading | `utils/read_file.py`, `utils/importrhdutilities.py` |
| Waveform visualization | `utils/blech_waveforms_datashader.py` |
| Held unit detection | `utils/blech_held_units_detect.py` |
| HDF5 operations | `utils/blech_hdf5_repack.py`, `utils/blech_split_h5_files.py` |
| NEX file conversion | `utils/blech_nex_convert.py` |
| Memory monitoring | `utils/ram_monitor.py` |
| EMG filtering | `emg/emg_filter.py` |
| Gape detection | `emg/gape_QDA_classifier/get_gapes_Li.py` |
| LFP processing | `utils/ephys_data/lfp_processing.py` |
| Unit similarity | `utils/qa_utils/unit_similarity.py` |
| Drift detection | `utils/qa_utils/drift_check.py` |
| PSTH overlay | `blech_units_characteristics.py`, `utils/ephys_data/` |
