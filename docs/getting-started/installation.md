# Installation

## Platform Support

> **⚠️ Important**: blech_clust is primarily tested and supported on Linux systems. The make-based installation process works only on Linux.

**For Windows Users**: We recommend installing a tested version of Ubuntu via Windows Subsystem for Linux (WSL). This provides a Linux environment within Windows where blech_clust can be installed and run properly.

## Prerequisites

Before installing blech_clust, ensure you have:

- **Linux Operating System**: Ubuntu 20.04, 22.04, or 24.04 (tested versions)
- **Conda/Miniconda**: Required for environment management
- **Git**: For cloning repositories
- **System packages**: GNU parallel (optional, for parallel processing)

## Quick Start (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/katzlabbrandeis/blech_clust.git
   cd blech_clust
   ```

2. **Install everything:**
   ```bash
   make all
   ```
   This installs the base environment, EMG analysis tools, neuRecommend classifier, and all optional dependencies.

3. **Activate the environment:**
   ```bash
   conda activate blech_clust
   ```

## Custom Installation

For more control over what gets installed:

```bash
# Core spike sorting functionality only
make core

# Or install components individually:
make base      # Base environment and core dependencies (required)
make emg       # EMG analysis requirements (BSA/STFT, QDA)
make neurec    # neuRecommend waveform classifier
make prefect   # Prefect workflow management (for testing)
make dev       # Development dependencies
make optional  # Optional analysis tools
```

## Parameter Setup

After installation, set up parameter templates:

```bash
# Copy parameter templates (if none exist)
make params

# Edit the parameter files according to your experimental setup
```

### Parameter Files

The following parameter files control pipeline behavior:

- **sorting_params.json** - Spike sorting parameters:
    - `bandpass_lower_cutoff` / `bandpass_upper_cutoff`: Filter frequencies (default: 300-3000 Hz)
    - `waveform_threshold`: Spike detection threshold in standard deviations (default: 5)
    - `spike_snapshot_before` / `spike_snapshot_after`: Waveform extraction window in ms
    - `clustering_params`: GMM clustering settings
    - `qa_params`: Quality assurance thresholds
    - `psth_params`: PSTH calculation parameters

- **emg_params.json** - EMG analysis parameters:
    - `stft_params`: Short-time Fourier transform settings
    - `use_BSA`: Whether to use Bayesian Spectrum Analysis

- **waveform_classifier_params.json** - neuRecommend classifier settings

## Troubleshooting

- **Clean installation:** If you encounter issues, remove the environment and start fresh:
  ```bash
  make clean
  make all
  ```

- **Partial installation:** If a component fails, you can retry individual components:
  ```bash
  make base    # Retry base installation
  make emg     # Retry EMG components
  ```

- **Environment activation:** Always activate the environment before running scripts:
  ```bash
  conda activate blech_clust
  ```

## Tested Platforms

blech_clust is continuously tested on the following platform combinations:

| Linux Distribution | Python Versions | Status |
|-------------------|-----------------|---------|
| Ubuntu 20.04      | 3.8, 3.9, 3.10, 3.11 | ✅ Tested |
| Ubuntu 22.04      | 3.8, 3.9, 3.10, 3.11 | ✅ Tested |
| Ubuntu 24.04      | 3.8, 3.9, 3.10, 3.11 | ✅ Tested |

> **Note**: While other Linux distributions and Python versions may work, only the combinations above are actively tested in our continuous integration pipeline. For best results, we recommend using one of the tested configurations.

## Test Dataset

A test dataset is available to verify your installation:

[Test Dataset on Google Drive](https://drive.google.com/drive/folders/1ne5SNU3Vxf74tbbWvOYbYOE1mSBkJ3u3?usp=sharing)

## Next Steps

- Explore the [API Reference](../reference/index.md) to understand available functions
- Read the [Tutorials](../tutorials.md) for step-by-step guides
