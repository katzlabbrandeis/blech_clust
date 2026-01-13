# Removed Features

This page documents components removed from the original blech_clust and their alternatives in the current fork.

## Legacy Code

### Python 2 Backups

The `python2_legacy_code/` directory has been removed. This contained `.bak` files from the Python 2 to Python 3 migration:

- `blech_clust.py.bak`
- `blech_held_units_detect.py.bak`
- `blech_multinomial_hmm.py.bak`
- `blech_palatability_identity_plot.py.bak`
- `blech_palatability_identity_setup.py.bak`
- `blech_post_process.py.bak`
- `blech_process.py.bak`
- `blech_setup_hmm.py.bak`
- `blech_units_distance.py.bak`
- `detect_peaks.py.bak`
- `emg_BSA_segmentation_plot.py.bak`
- `emg_local_BSA.py.bak`
- `emg_local_BSA_post_process.py.bak`
- `emg_make_arrays.py.bak`
- `memory_monitor.py.bak`
- `units_make_arrays.py.bak`

**Rationale**: Python 2 reached end-of-life in 2020. The current codebase is Python 3 only.

### Original Entry Point

`blech_clust.py` has been removed and replaced by a modular pipeline:

| Original | Current |
|----------|---------|
| `blech_clust.py` | `blech_exp_info.py` → `blech_init.py` → `blech_common_avg_reference.py` → `blech_run_process.sh` → ... |

The modular approach allows:

- Running individual steps independently
- Better error recovery (restart from failed step)
- Clearer separation of concerns

---

## LFP Analysis

The entire `LFP_analysis/` directory has been removed:

| Removed File | Description |
|--------------|-------------|
| `Compare_frequency_envelopes.py` | Frequency envelope comparison |
| `GFP_frequency_band_extraction.py` | Global field power extraction |
| `LFP_Processing_Final.py` | Main LFP processing |
| `LFP_Spectrogram_stone.py` | Spectrogram generation |
| `LFP_create_m_file.py` | MATLAB file creation |
| `LFP_create_m_file_daniel.py` | MATLAB file creation (variant) |
| `LFP_spike_lock_setup_Final.py` | Spike-LFP locking setup |
| `Laser_LFP_Parse_Final.py` | Laser trial LFP parsing |
| `compare_passive_FRH_dumps.py` | Passive firing rate comparison |
| `passive_FRH_grouped.py` | Grouped passive firing rates |
| `passive_FRH_grouped_sumchange.py` | Summed change analysis |
| `passive_firing_rates.py` | Passive firing rate calculation |
| `passive_firing_rates_delta.py` | Delta firing rate analysis |
| `_old/` subdirectory | Archived versions |

### Alternative

LFP functionality is now available through the `ephys_data` module:

```python
from utils.ephys_data.ephys_data import ephys_data
from utils.ephys_data import lfp_processing

# Load data
data = ephys_data(data_dir='/path/to/data')

# Set LFP parameters
data.lfp_params = {
    'freq_bounds': [1, 300],
    'sampling_rate': 30000,
    'taste_signal_choice': 'Start',
    'fin_sampling_rate': 1000,
    'trial_durations': [2000, 5000]
}

# Extract and load LFPs
data.extract_lfps()
data.get_lfps()
lfp_data = data.lfp_array
```

See [Ephys Data Reference](../../reference/ephys-data.md) for complete documentation.

---

## HMM Analysis

Hidden Markov Model analysis scripts have been removed:

| Removed File | Description |
|--------------|-------------|
| `blech_hmm.py` | Basic HMM implementation |
| `blech_multinomial_hmm.py` | Multinomial HMM |
| `blech_poisson_hmm.py` | Poisson HMM |
| `blech_setup_hmm.py` | HMM setup and configuration |
| `variational_HMM_implement.py` | Variational HMM implementation |
| `variational_HMM_line_up_palatability_plot.py` | Palatability alignment plots |
| `variational_HMM_setup.py` | Variational HMM setup |

### Rationale

HMM analysis is a specialized downstream analysis that:

- Requires significant computational resources
- Has complex dependencies
- Is not part of the core spike-sorting pipeline

Users requiring HMM analysis should consider dedicated HMM packages or contact the lab for the standalone HMM analysis code.

---

## Palatability/Identity Analysis

The `additional_analyses/` directory has been removed:

| Removed File | Description |
|--------------|-------------|
| `blech_palatability_identity_plot.py` | Palatability/identity visualization |
| `blech_palatability_identity_setup.py` | Analysis setup |
| `blech_palatability_regression.py` | Regression analysis |
| `blech_hmm_emg_plot.py` | HMM-EMG combined plots |
| `blech_identity_palatability_switch.py` | Switch point analysis |
| `identity_palatability_switch_EM.py` | EM-based switch detection |
| `identity_palatability_switch_EM_correlation_plot.py` | Correlation visualization |
| `identity_palatability_switch_EM_implement.py` | EM implementation |
| `identity_palatability_switch_correlation_plot.py` | Correlation plots |
| `identity_palatability_switch_functions.py` | Helper functions |
| `identity_palatability_switch_process.py` | Processing pipeline |
| `identity_palatability_switch_setup.py` | Setup configuration |

### Alternative

Basic palatability analysis is available through the `ephys_data` module:

```python
from utils.ephys_data.ephys_data import ephys_data

data = ephys_data(data_dir='/path/to/data')
data.get_unit_descriptors()
data.get_spikes()
data.get_firing_rates()

# Calculate palatability correlations
data.calc_palatability()

# Access results
pal_df = data.pal_df       # Palatability rankings
pal_array = data.pal_array # Palatability correlations over time
```

For advanced identity/palatability switch analysis, contact the lab.

---

## Laser Effect Analysis

The `laser_effect_analysis/` directory has been removed:

| Removed File | Description |
|--------------|-------------|
| `compare_laser_effects.py` | Laser effect comparison |
| `laser_effects_plot.py` | Laser effect visualization |

### Alternative

Laser trial separation is available through the `ephys_data` module:

```python
from utils.ephys_data.ephys_data import ephys_data

data = ephys_data(data_dir='/path/to/data')
data.get_spikes()
data.get_firing_rates()

# Check for laser trials
data.check_laser()

if data.laser_exists:
    # Separate data by laser condition
    data.separate_laser_data()

    # Access separated data
    on_spikes = data.on_spikes
    off_spikes = data.off_spikes
    on_firing = data.on_firing
    off_firing = data.off_firing
```

---

## Miscellaneous Removed Files

### Examples Directory

The `examples/` directory has been removed:

| Removed | Description |
|---------|-------------|
| `Half-Gaussian PSTH example.ipynb` | Jupyter notebook example |

Example usage is now documented in the [Tutorials](../../tutorials.md) and API reference pages.

### Root-Level Scripts

Several root-level scripts have been removed or relocated:

| Removed | Status |
|---------|--------|
| `blech_dat_file_join.py` | Functionality integrated into data loading |
| `overlay_psth.py` | Functionality in ephys_data module |

See [File Mapping](file-mapping.md) for scripts that were relocated rather than removed.
