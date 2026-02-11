# Migration Guide from Original blech_clust

This guide documents the changes between the [original blech_clust](https://github.com/narendramukherjee/blech_clust) and the current [katzlabbrandeis fork](https://github.com/katzlabbrandeis/blech_clust).

## Overview of Changes

The katzlabbrandeis fork represents a significant modernization of the original codebase:

| Metric | Count |
|--------|-------|
| Files added | 131 |
| Files removed | 80 |
| Files modified | 5 |

The changes focus on improved reproducibility, testing, documentation, and user experience while maintaining the core spike-sorting functionality.

## Migration Guide Sections

- **[Removed Features](removed-features.md)** - Components removed from the original and their alternatives
- **[File Mapping](file-mapping.md)** - Where relocated files now live
- **[Quality Assurance](qa-improvements.md)** - New QA tools and drift detection

## Quick Reference: New Features

### Installation and Environment

| Feature | Description |
|---------|-------------|
| Makefile-based installation | `make all` handles conda environment, dependencies, and R packages |
| Pip-installable package | Alternative installation via pip |
| Dev container support | `.devcontainer/` for consistent development environments |
| Pre-commit hooks | Code quality enforcement |
| Structured requirements | Separate files for base, dev, test, docs, and optional dependencies |

See [Installation](../installation.md) for setup instructions.

### Testing Infrastructure

| Feature | Description |
|---------|-------------|
| GitHub Actions CI/CD | Automated testing on push/PR |
| pytest test suite | Unit tests in `tests/` |
| Pipeline testing | End-to-end validation in `pipeline_testing/` |

### Metadata and Parameter Recording

The original blech_clust used text entry boxes with no record of parameters. The current version provides:

| Feature | Description |
|---------|-------------|
| `blech_exp_info.py` | Pre-clustering annotation of channels and experimental parameters |
| Parameter templates | JSON templates in `params/_templates/` |
| Dependency graph | `params/dependency_graph.json` tracks parameter dependencies |
| Example metadata | Sample files in `example_meta_files/` |

### Common Average Reference Improvements

| Feature | Description |
|---------|-------------|
| Dead channel visualization | Plots to identify dead/different channels within CAR groups |
| Channel clustering | Automatic clustering of channels within CAR groups |

### Auto-clustering

| Feature | Description |
|---------|-------------|
| Auto-clustering | Fully automated clustering of spikes integrated into the main pipeline. See [Auto Sorting Resources](../../resources/autosorting.md) for details. |

### Unit Quality and Classification

| Feature | Description |
|---------|-------------|
| Waveform classifier | ML classifier to recommend units |
| Feature visualization | Spike features over recording duration |
| `blech_units_characteristics.py` | Unit characteristic analysis |
| `utils/cluster_stability.py` | Cluster stability assessment |
| Hierarchical clustering plots | Visual assessment of cluster quality |

### RNN-Based Firing Rate Inference

The `utils/infer_rnn_rates.py` module uses recurrent neural networks to infer firing rates from spike trains:

```bash
python utils/infer_rnn_rates.py /path/to/data --train_steps 15000 --hidden_size 8
```

Configuration via `params/_templates/blechrnn_params.json`.

### Spike Detection Improvements

| Feature | Description |
|---------|-------------|
| Rolling threshold | Adaptive per-window threshold that adjusts to local noise levels |
| MAD-based detection | Uses Median Absolute Deviation for outlier-robust noise estimation |
| Configurable windows | Adjustable window size and step for threshold computation |

The rolling threshold computes spike detection thresholds independently for each time window, allowing detection to adapt to noise variations across the recording. This is controlled by `use_rolling_threshold`, `rolling_threshold_window`, and `rolling_threshold_step` parameters.

### Performance Improvements

| Improvement | Description |
|-------------|-------------|
| Collision calculation | Optimized from O(n²) to O(n) |
| Parallel electrode processing | Concurrent processing across electrodes |
| Memory-efficient CAR | Finite samples instead of downsampled recording |

### Shell Scripts for Automation

| Script | Description |
|--------|-------------|
| `blech_autosort.sh` | Main autosorting script |
| `blech_autosort_batch.sh` | Batch processing |
| `blech_clust_pre.sh` | Pre-clustering setup |
| `blech_clust_post.sh` | Post-clustering steps |
| `blech_run_process.sh` | Parallel spike extraction |
| `emg/emg_run_pipeline.sh` | EMG pipeline automation |

See [Quick Start](../quickstart.md) and [Tutorials](../../tutorials.md) for workflow details.

### Code Organization

The codebase has been reorganized from a flat structure to modular directories:

| Directory | Contents |
|-----------|----------|
| `utils/` | Utility functions and helper modules |
| `emg/` | EMG analysis pipeline |
| `params/` | Parameter templates and configuration |
| `tests/` | pytest test suite |
| `docs/` | MkDocs documentation |
| `pipeline_testing/` | End-to-end pipeline tests |

### ephys_data Module

The `utils/ephys_data/` module provides utilities for loading, processing, and visualizing data. See [Ephys Data Reference](../../reference/ephys-data.md) for details.

### EMG Analysis

EMG functionality has been reorganized into the `emg/` directory. See [EMG Analysis Reference](../../reference/emg-analysis.md) for details.

## Key Improvements Summary

| Area | Original | Current |
|------|----------|---------|
| Installation | Manual dependency management | `make all` automated setup |
| Testing | None | pytest suite + CI/CD |
| Parameters | Text entry, no record | JSON templates, full recording |
| Documentation | Minimal README | Full MkDocs site |
| CAR | Basic | Visualization + channel clustering |
| Spike detection | Global threshold | Rolling adaptive threshold |
| Unit quality | Manual inspection | Classifier + feature plots |
| QA | None | Drift detection, grading metrics |
| File formats | One format | Multiple format support |
| Code organization | Flat structure | Modular directories |

## Getting Help

- **Documentation**: [katzlabbrandeis.github.io/blech_clust](https://katzlabbrandeis.github.io/blech_clust/)
- **Issues**: [GitHub Issues](https://github.com/katzlabbrandeis/blech_clust/issues)
- **Original repo**: [narendramukherjee/blech_clust](https://github.com/narendramukherjee/blech_clust)
