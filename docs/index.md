# blech_clust

**Python and R based code for clustering and sorting electrophysiology data**

[![DOI](https://zenodo.org/badge/119422765.svg)](https://doi.org/10.5281/zenodo.15175272)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/katzlabbrandeis/blech_clust/master.svg)](https://results.pre-commit.ci/latest/github/katzlabbrandeis/blech_clust/master)

## Overview

**blech_clust** is a comprehensive Python and R based toolkit for clustering and sorting electrophysiology data recorded using the Intan RHD2132 chips. Originally written for cortical multi-electrode recordings in Don Katz's lab at Brandeis University, it's optimized for high-performance computing clusters but can be easily modified to work in any parallel environment.

Visit the [Katz lab website](https://sites.google.com/a/brandeis.edu/katzlab/) for more information.

## Key Features

- **Automated Spike Sorting**: Complete pipeline from raw data to sorted units
- **EMG Analysis**: Multiple approaches including BSA/STFT and QDA-based gape detection
- **Quality Assessment**: Built-in tools for dataset quality grading and validation
- **Parallel Processing**: Optimized for HPC environments
- **Comprehensive Documentation**: Detailed API reference and tutorials

## Quick Links

- [Getting Started](getting-started/installation.md) - Installation and setup instructions
- [API Reference](reference/index.md) - Complete API documentation
- [Tutorials](tutorials.md) - Step-by-step guides
- [GitHub Repository](https://github.com/katzlabbrandeis/blech_clust)

## Resources

- [blech_clust Secrets Within (PDF)](blech_clust_secrets_within.pdf) - Comprehensive guide to blech_clust internals and advanced usage

## Pipeline Overview

### Spike Sorting Pipeline

For the complete main spike-sorting pipeline (including the operations workflow diagram, detailed steps, and nomnoml schema), please refer to the [README](https://github.com/katzlabbrandeis/blech_clust#main-spike-sorting-pipeline).

### EMG Analysis Pipelines

For details on EMG analysis workflows, see the [README](https://github.com/katzlabbrandeis/blech_clust#emg-analysis) and [Workflow Documentation](workflow.md).

## Installation

The installation process is managed through a Makefile that handles all dependencies:

```bash
# Clone the repository
git clone https://github.com/katzlabbrandeis/blech_clust.git
cd blech_clust

# Install everything
make all

# Activate the environment
conda activate blech_clust
```

For more detailed installation instructions, see the [Getting Started](getting-started/installation.md) guide.

## Contributing

We welcome contributions! Please read our [CONTRIBUTING.md](https://github.com/katzlabbrandeis/blech_clust/blob/master/CONTRIBUTING.md) file for guidelines.

### Contributing to Documentation

Help us improve the documentation:

- **Report issues**: Found an error or unclear explanation? [Open an issue](https://github.com/katzlabbrandeis/blech_clust/issues)
- **Suggest improvements**: Have ideas for better organization or content? We'd love to hear them
- **Submit changes**: See [docs/README.md](https://github.com/katzlabbrandeis/blech_clust/blob/master/docs/README.md) for instructions on building and updating documentation

The documentation is built with [MkDocs](https://www.mkdocs.org/) and automatically deployed via GitHub Actions.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{blech_clust_katz,
  author       = {Mahmood, Abuzar and
                  Mukherjee, Narendra and
                  Stone, Bradly and
                  Raymond, Martin and
                  Germaine, Hannah and
                  Lin, Jian-You and
                  Mazzio, Christina and
                  Katz, Donald},
  title        = {katzlabbrandeis/blech\_clust: v1.1.0},
  month        = apr,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {1.1.0},
  doi          = {10.5281/zenodo.15175273},
  url          = {https://doi.org/10.5281/zenodo.15175273}
}
```

## Acknowledgments

This work used ACCESS-allocated resources at Brandeis University through allocation BIO230103 from the [Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support](https://access-ci.org/) (ACCESS) program, which is supported by U.S. National Science Foundation grants #2138259, #2138286, #2138307, #2137603, and #2138296.

The project titled "Computational Processing and Modeling of Neural Ensembles in Identifying the Nonlinear Dynamics of Taste Perception" was led by PI Abuzar Mahmood. The computational allocation was active from 2023-06-26 to 2024-06-25.
