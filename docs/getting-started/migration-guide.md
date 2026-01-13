# Migration Guide from Original blech_clust

This guide documents the changes between the [original blech_clust](https://github.com/narendramukherjee/blech_clust) and the current [katzlabbrandeis fork](https://github.com/katzlabbrandeis/blech_clust). It covers added functionality, removed components, and relocated features.

## Overview of Changes

The katzlabbrandeis fork represents a significant modernization of the original codebase with:

- **131 new files** added
- **80 files** removed (mostly legacy code and redundant scripts)
- **5 files** substantially modified

The changes focus on improved reproducibility, testing, documentation, and user experience while maintaining the core spike-sorting functionality.

---

## New Functionality

### Installation and Environment

| Feature | Description |
|---------|-------------|
| **Makefile-based installation** | Automated setup via `make all` handles conda environment, dependencies, and R packages |
| **Dev container support** | `.devcontainer/` configuration for consistent development environments |
| **Pre-commit hooks** | `.pre-commit-config.yaml` for code quality enforcement |
| **Structured requirements** | Separate requirements files for base, dev, test, docs, and optional dependencies |

### Testing Infrastructure

| Feature | Description |
|---------|-------------|
| **GitHub Actions CI/CD** | Automated testing via `pytest_workflow.yml`, `python_workflow_test.yml`, `installation_test.yml` |
| **pytest test suite** | `tests/` directory with unit tests for core functionality |
| **Pipeline testing** | `pipeline_testing/` directory with end-to-end pipeline validation tools |
| **EMG functionality testing** | Dedicated tests for EMG processing components |

### Metadata and Parameter Recording

| Feature | Description |
|---------|-------------|
| **blech_exp_info.py** | Pre-clustering annotation of channels and experimental parameters |
| **Parameter templates** | `params/_templates/` with JSON templates for sorting, EMG, waveform classifier, and RNN parameters |
| **Dependency graph** | `params/dependency_graph.json` tracks parameter dependencies |
| **Example metadata files** | `example_meta_files/` with sample `.csv`, `.info`, and electrode layout files |

The original blech_clust used text entry boxes with no record of parameters or metadata. The current version explicitly defines and records all processing parameters.

### Common Average Reference Improvements

| Feature | Description |
|---------|-------------|
| **Dead channel visualization** | Plots to identify dead or significantly different channels within CAR groups |
| **Channel clustering** | Automatic clustering of channels within CAR groups |
| **Enhanced blech_common_avg_reference.py** | Substantially expanded from original (674+ lines vs original) |

### Unit Quality and Classification

| Feature | Description |
|---------|-------------|
| **Waveform classifier** | Machine learning classifier to recommend units |
| **Feature visualization** | Plots showing spike features over recording duration |
| **blech_units_characteristics.py** | Analyze unit characteristics (new file) |
| **utils/blech_spike_features.py** | Dedicated spike feature extraction |
| **utils/cluster_stability.py** | Assess cluster stability over time |

### Post-Processing

| Feature | Description |
|---------|-------------|
| **Automatic post-processing** | Integrated into autosorting pipeline via `blech_autosort.sh` |
| **Table-based post-processing** | Alternative workflow using tabular data |
| **utils/blech_post_process_utils.py** | 1600+ lines of post-processing utilities |

### Quality Assurance

| Feature | Description |
|---------|-------------|
| **blech_run_QA.sh** | Dedicated QA script |
| **utils/qa_utils/** | QA utilities including drift detection, channel correlation, unit similarity |
| **Drift testing** | Population and single-unit level drift detection during recording |
| **utils/grade_dataset.py** | Dataset quality grading based on defined metrics |
| **utils/grading_metrics.json** | Configurable grading criteria |

### ephys_data Module

The `utils/ephys_data/` module provides utilities for loading, processing, and visualizing data:

| Component | Description |
|-----------|-------------|
| **ephys_data.py** | Main data loading and processing class (1800+ lines) |
| **lfp_processing.py** | LFP analysis utilities (760+ lines) |
| **visualize.py** | Visualization functions |
| **BAKS.py** | Bayesian Adaptive Kernel Smoother implementation |

### EMG Analysis Reorganization

EMG functionality has been reorganized into the `emg/` directory:

| Component | Description |
|-----------|-------------|
| **emg_filter.py** | EMG signal filtering |
| **emg_freq_setup.py** | Frequency analysis setup |
| **emg_freq_post_process.py** | Post-processing for frequency analysis |
| **emg_freq_plot.py** | Visualization |
| **emg_local_BSA_execute.py** | BSA execution |
| **emg_local_STFT_execute.py** | STFT execution |
| **gape_QDA_classifier/** | QDA-based gape detection (Li et al. methodology) |

### File Format Support

| Feature | Description |
|---------|-------------|
| **Traditional format support** | Process both one-file-per-channel and traditional Intan formats |
| **utils/read_file.py** | Unified file reading utilities |
| **utils/importrhdutilities.py** | RHD file import utilities (1300+ lines) |

### Shell Scripts for Automation

| Script | Description |
|--------|-------------|
| **blech_autosort.sh** | Main autosorting script |
| **blech_autosort_batch.sh** | Batch processing multiple datasets |
| **blech_clust_pre.sh** | Pre-clustering setup |
| **blech_clust_post.sh** | Post-clustering steps |
| **blech_run_process.sh** | Parallel spike extraction |
| **emg/emg_run_pipeline.sh** | EMG pipeline automation |

### Documentation

| Component | Description |
|-----------|-------------|
| **MkDocs site** | Full documentation site with `mkdocs.yml` configuration |
| **docs/** | Structured documentation with getting-started guides, tutorials, API reference |
| **CONTRIBUTING.md** | Contribution guidelines |
| **GitHub Pages deployment** | Automated docs deployment via `.github/workflows/docs.yml` |

---

## Removed Functionality

### Legacy Code

The following legacy/deprecated code has been removed:

| Removed | Notes |
|---------|-------|
| `python2_legacy_code/` | Python 2 backups no longer needed |
| `blech_clust.py` | Replaced by modular pipeline scripts |
| `blech_dat_file_join.py` | Functionality integrated elsewhere |

### LFP Analysis

The `LFP_analysis/` directory has been removed. LFP functionality is now available through:

- `utils/ephys_data/lfp_processing.py` - Core LFP processing
- `utils/ephys_data/convenience_scripts/region_spectrogram_plot.py` - Spectrogram visualization

### HMM Analysis

HMM-related scripts have been removed from the main repository:

| Removed |
|---------|
| `blech_hmm.py` |
| `blech_multinomial_hmm.py` |
| `blech_poisson_hmm.py` |
| `blech_setup_hmm.py` |
| `variational_HMM_implement.py` |
| `variational_HMM_line_up_palatability_plot.py` |
| `variational_HMM_setup.py` |

### Palatability/Identity Analysis

The `additional_analyses/` directory has been removed:

| Removed |
|---------|
| `blech_palatability_identity_plot.py` |
| `blech_palatability_identity_setup.py` |
| `identity_palatability_switch_*.py` (multiple files) |
| `blech_palatability_regression.py` |

### Laser Effect Analysis

The `laser_effect_analysis/` directory has been removed.

### Miscellaneous Removed Files

| Removed | Replacement |
|---------|-------------|
| `memory_monitor.py` | `utils/ram_monitor.py` |
| `read_file.py` | `utils/read_file.py` |
| `split_h5_files.py` | `utils/blech_split_h5_files.py` |
| `clustering.py` | `utils/clustering.py` |
| `blech_waveforms_datashader.py` | `utils/blech_waveforms_datashader.py` |
| `blech_held_units_detect.py` | `utils/blech_held_units_detect.py` |
| `blech_hdf5_repack.py` | `utils/blech_hdf5_repack.py` |
| `blech_nex_convert.py` | `utils/blech_nex_convert.py` |
| `fix_laser_sampling_errors.py` | `utils/fix_laser_sampling_errors.py` |
| `units_make_arrays.py` | Functionality in `blech_make_arrays.py` |
| `blech_make_psth.py` | Functionality in other modules |
| `blech_unit_visualize.py` | `blech_units_plot.py` (enhanced) |
| `blech_units_organize.py` | Functionality integrated into pipeline |
| `blech_units_similarity.py` | `utils/qa_utils/unit_similarity.py` |
| `overlay_psth.py` | Functionality in ephys_data module |
| `filter_emg.py` | `emg/emg_filter.py` |
| `get_gapes_Li.py` | `emg/gape_QDA_classifier/get_gapes_Li.py` |
| `emg_*.py` (root level) | Reorganized into `emg/` directory |

---

## Relocated Functionality

### Root Level to utils/

Many utility scripts moved from root to `utils/`:

```
clustering.py                  → utils/clustering.py
read_file.py                   → utils/read_file.py
split_h5_files.py              → utils/blech_split_h5_files.py
blech_waveforms_datashader.py  → utils/blech_waveforms_datashader.py
blech_held_units_detect.py     → utils/blech_held_units_detect.py
blech_hdf5_repack.py           → utils/blech_hdf5_repack.py
blech_nex_convert.py           → utils/blech_nex_convert.py
fix_laser_sampling_errors.py   → utils/fix_laser_sampling_errors.py
memory_monitor.py              → utils/ram_monitor.py
```

### EMG Scripts to emg/

All EMG-related scripts consolidated in `emg/`:

```
filter_emg.py                  → emg/emg_filter.py
emg_local_BSA.py               → emg/emg_local_BSA_execute.py
emg_local_BSA_execute.py       → emg/emg_local_BSA_execute.py
emg_local_BSA_post_process.py  → emg/emg_freq_post_process.py
emg_BSA_segmentation.py        → emg/emg_freq_setup.py
emg_BSA_segmentation_plot.py   → emg/emg_freq_plot.py
emg_make_arrays.py             → Integrated into blech_make_arrays.py
get_gapes_Li.py                → emg/gape_QDA_classifier/get_gapes_Li.py
detect_peaks.py                → emg/gape_QDA_classifier/detect_peaks.py
QDA_nostd_no_first.mat         → emg/gape_QDA_classifier/QDA_nostd_no_first.mat
```

---

## Workflow Changes

### Original Workflow

The original workflow was less structured, with scripts run manually in sequence.

### Current Workflow

The current version provides automated shell scripts:

1. **Pre-clustering**: `python blech_exp_info.py /path/to/data`
2. **Main pipeline**: `bash blech_autosort.sh /path/to/data`
   - Runs: `blech_init.py` → `blech_common_avg_reference.py` → `blech_run_process.sh` → `blech_post_process.py` → `blech_units_plot.py` → `blech_make_arrays.py`
3. **Quality assurance**: `bash blech_run_QA.sh /path/to/data`
4. **EMG analysis**: `bash emg/emg_run_pipeline.sh /path/to/data`

---

## Key Improvements Summary

| Area | Original | Current |
|------|----------|---------|
| **Installation** | Manual dependency management | `make all` automated setup |
| **Testing** | None | pytest suite + CI/CD |
| **Parameters** | Text entry, no record | JSON templates, full recording |
| **Documentation** | Minimal README | Full MkDocs site |
| **CAR** | Basic | Visualization + channel clustering |
| **Unit quality** | Manual inspection | Classifier + feature plots |
| **QA** | None | Drift detection, grading metrics |
| **File formats** | One format | Multiple format support |
| **Code organization** | Flat structure | Modular directories |

---

## Getting Help

- **Documentation**: [katzlabbrandeis.github.io/blech_clust](https://katzlabbrandeis.github.io/blech_clust/)
- **Issues**: [GitHub Issues](https://github.com/katzlabbrandeis/blech_clust/issues)
- **Original repo**: [narendramukherjee/blech_clust](https://github.com/narendramukherjee/blech_clust)
