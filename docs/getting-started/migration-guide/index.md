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

### Unit Quality and Classification

| Feature | Description |
|---------|-------------|
| Waveform classifier | ML classifier to recommend units |
| Feature visualization | Spike features over recording duration |
| `blech_units_characteristics.py` | Unit characteristic analysis |
| `utils/cluster_stability.py` | Cluster stability assessment |

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
| Unit quality | Manual inspection | Classifier + feature plots |
| QA | None | Drift detection, grading metrics |
| File formats | One format | Multiple format support |
| Code organization | Flat structure | Modular directories |

## Getting Help

- **Documentation**: [katzlabbrandeis.github.io/blech_clust](https://katzlabbrandeis.github.io/blech_clust/)
- **Issues**: [GitHub Issues](https://github.com/katzlabbrandeis/blech_clust/issues)
- **Original repo**: [narendramukherjee/blech_clust](https://github.com/narendramukherjee/blech_clust)
