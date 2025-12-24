[![DOI](https://zenodo.org/badge/119422765.svg)](https://doi.org/10.5281/zenodo.15175272)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/katzlabbrandeis/blech_clust/master.svg)](https://results.pre-commit.ci/latest/github/katzlabbrandeis/blech_clust/master)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://katzlabbrandeis.github.io/blech_clust/)

# blech_clust

Python and R based code for clustering and sorting electrophysiology data recorded using the Intan RHD2132 chips. Originally written for cortical multi-electrode recordings in Don Katz's lab at Brandeis University.

**📚 [Full Documentation](https://katzlabbrandeis.github.io/blech_clust/)** | **🚀 [Getting Started](https://katzlabbrandeis.github.io/blech_clust/getting-started.html)** | **📖 [Tutorials](https://katzlabbrandeis.github.io/blech_clust/tutorials.html)** | **🔧 [API Reference](https://katzlabbrandeis.github.io/blech_clust/reference/)**

## Features

- **Automated Spike Sorting**: Complete pipeline from raw Intan data to sorted units
- **EMG Analysis**: BSA/STFT frequency analysis and QDA-based gape detection
- **Quality Assessment**: Built-in drift detection, unit similarity analysis, and dataset grading
- **Parallel Processing**: Optimized for HPC environments
- **Comprehensive Documentation**: Detailed guides, tutorials, and API reference

## Quick Start

```bash
# Clone the repository
git clone https://github.com/katzlabbrandeis/blech_clust.git
cd blech_clust

# Install everything
make all

# Activate the environment
conda activate blech_clust

# Run the pipeline
python blech_exp_info.py /path/to/data
bash blech_autosort.sh /path/to/data
```

For detailed instructions, see the [Getting Started Guide](https://katzlabbrandeis.github.io/blech_clust/getting-started.html).

## Documentation

For comprehensive documentation, visit **[katzlabbrandeis.github.io/blech_clust](https://katzlabbrandeis.github.io/blech_clust/)**

- **[Getting Started](https://katzlabbrandeis.github.io/blech_clust/getting-started.html)** - Installation and setup
- **[Tutorials](https://katzlabbrandeis.github.io/blech_clust/tutorials.html)** - Step-by-step workflows
- **[API Reference](https://katzlabbrandeis.github.io/blech_clust/reference/)** - Complete module documentation
- **[Blog](https://katzlabbrandeis.github.io/blech_clust/blogs/blogs_main.html)** - Updates and insights

# For building documentation locally:
```
pip install -r requirements/requirements-docs.txt
mkdocs serve
```

## Pipeline Overview

### Main Spike-Sorting Pipeline

Refer to the documentation for the main spike-sorting flow.
```
blech_exp_info → blech_clust → blech_common_avg_reference →
blech_run_process → blech_post_process → blech_units_plot →
blech_make_arrays → blech_run_QA → blech_unit_characteristics
```

### EMG Analysis
- **BSA/STFT**: Bayesian Spectrum Analysis and Short-Time Fourier Transform
- **QDA**: Quadratic Discriminant Analysis for gape detection

See the [Core Pipeline Documentation](https://katzlabbrandeis.github.io/blech_clust/reference/core-pipeline.html) for details.

## Test Dataset

Test data available at: [Google Drive](https://drive.google.com/drive/folders/1ne5SNU3Vxf74tbbWvOYbYOE1mSBkJ3u3?usp=sharing)

## Contributing

We welcome contributions! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{blech_clust_katz,
  author       = {Mahmood, Abuzar and Mukherjee, Narendra and
                  Stone, Bradly and Raymond, Martin and
                  Germaine, Hannah and Lin, Jian-You and
                  Mazzio, Christina and Katz, Donald},
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

This work used ACCESS-allocated resources at Brandeis University through allocation BIO230103 from the [Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support](https://access-ci.org/) (ACCESS) program, supported by U.S. National Science Foundation grants #2138259, #2138286, #2138307, #2137603, and #2138296.

## License

See [LICENSE](LICENSE) for details.

---

**Visit the Katz Lab:** [sites.google.com/a/brandeis.edu/katzlab](https://sites.google.com/a/brandeis.edu/katzlab/)
