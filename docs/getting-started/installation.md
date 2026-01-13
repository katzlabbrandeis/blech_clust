# Installation

## Prerequisites

Before installing blech_clust, ensure you have:

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
   This installs the base environment, EMG analysis tools, neuRecommend classifier, BlechRNN, and all optional dependencies.

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
make blechrnn  # BlechRNN for firing rate estimation
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

- **blechrnn_params.json** - BlechRNN firing rate estimation settings

See the [Getting Started wiki](https://github.com/abuzarmahmood/blech_clust/wiki/Getting-Started#setting-up-params) for detailed parameter configuration.

## GPU Support (Optional)

If you plan to use GPU acceleration with BlechRNN:

1. Install CUDA toolkit separately (see [PyTorch GPU installation guide](https://medium.com/@jeanpierre_lv/installing-pytorch-with-gpu-support-on-ubuntu-a-step-by-step-guide-38dcf3f8f266))
2. Reinstall PyTorch with GPU support:
   ```bash
   conda activate blech_clust
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

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

## Test Dataset

A test dataset is available to verify your installation:

[Test Dataset on Google Drive](https://drive.google.com/drive/folders/1ne5SNU3Vxf74tbbWvOYbYOE1mSBkJ3u3?usp=sharing)

## Next Steps

- Explore the [API Reference](../reference/index.md) to understand available functions
- Read the [Tutorials](../tutorials.md) for step-by-step guides
- Check out the [Blog](https://katzlabbrandeis.github.io/blech_clust/blogs/blogs_main.html) for insights and updates
