# GLM-based Firing Rate Estimation

This module fits Generalized Linear Models (GLMs) to electrophysiological spike data using the [nemos](https://nemos.readthedocs.io/) library.

## Why Separate Scripts?

The nemos library has dependency conflicts with blech_clust's conda environment (particularly around numpy/jax versions). To work around this, the GLM fitting runs in a **separate virtual environment** dedicated to nemos.

This module uses a two-environment architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    infer_glm_rates.py                       │
│                   (Main Orchestrator)                       │
│         Can run from any Python environment                 │
└─────────────────────┬───────────────────────────────────────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
┌─────────────────────┐   ┌─────────────────────┐
│ _glm_extract_data.py│   │  _glm_fit_models.py │
│                     │   │                     │
│ Runs in blech_clust │   │  Runs in nemos      │
│ conda environment   │   │  virtual environment│
│                     │   │                     │
│ - Loads spike data  │   │  - Fits GLM models  │
│ - Uses ephys_data   │   │  - Uses nemos       │
│ - Saves to temp/    │   │  - Reads from temp/ │
└─────────────────────┘   └─────────────────────┘
          │                       │
          └───────────┬───────────┘
                      ▼
              ┌───────────────┐
              │   _temp/      │
              │ (intermediate │
              │    files)     │
              └───────────────┘
```

## Files

| File | Purpose | Environment |
|------|---------|-------------|
| `infer_glm_rates.py` | Main entry point. Detects environments, orchestrates data flow | Any |
| `_glm_extract_data.py` | Extracts spike data using ephys_data | blech_clust conda |
| `_glm_fit_models.py` | Fits GLM models, generates plots, writes HDF5 | nemos venv |
| `_temp/` | Temporary directory for intermediate numpy files (gitignored) | N/A |

## Setup

1. **blech_clust environment**: Should already exist if you're using blech_clust.

2. **nemos environment**: Create a separate virtual environment:
   ```bash
   python -m venv ~/nemos_env
   source ~/nemos_env/bin/activate
   pip install nemos matplotlib pandas tables
   ```

## Usage

```bash
python utils/glm/infer_glm_rates.py <data_dir> [options]
```

The script will:
1. Auto-detect the blech_clust conda environment
2. Auto-detect or prompt for the nemos virtual environment
3. Extract data using blech_clust's ephys_data
4. Fit GLM models using nemos
5. Save results to HDF5 and generate plots

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--bin_size` | 25 | Bin size in ms |
| `--history_window` | 250 | History window in ms for autoregressive effects |
| `--n_basis_funcs` | 8 | Number of basis functions for history filter |
| `--time_lims` | [1500, 4500] | Time limits for analysis [start, end] in ms |
| `--include_coupling` | False | Include coupling between neurons |
| `--separate_tastes` | False | Fit separate models for each taste |
| `--separate_regions` | False | Fit separate models for each region |
| `--retrain` | False | Force retraining even if model exists |
| `--blech_clust_env` | blech_clust | Name of blech_clust conda environment |
| `--nemos_env` | (auto) | Path to nemos venv Python interpreter |

### Example

```bash
# Basic usage
python utils/glm/infer_glm_rates.py /path/to/data

# With options
python utils/glm/infer_glm_rates.py /path/to/data \
    --separate_tastes \
    --separate_regions \
    --include_coupling \
    --nemos_env ~/nemos_env/bin/python
```

## Output

- **HDF5**: Results saved under `/glm_output/regions/` in the data's HDF5 file
- **CSV**: `glm_output/bits_per_spike_summary.csv` with model performance metrics
- **Plots**: `glm_output/plots/` with distribution and individual neuron plots
- **Models**: `glm_output/artifacts/` with pickled GLM models

## Model Details

The GLM includes:
- **Spike history**: Raised cosine log basis functions capture autoregressive effects
- **Stimulus features** (optional): Convolved indicator for stimulus onset
- **Neural coupling** (optional): Cross-neuron dependencies within region

Performance is measured using **bits per spike**, which quantifies information gain over a baseline homogeneous Poisson model.
