# EMG Analysis

Tools for analyzing electromyography (EMG) signals, including frequency analysis and gape detection.

## Overview

The EMG analysis pipeline provides two main approaches:

1. **BSA/STFT**: Bayesian Spectrum Analysis and Short-Time Fourier Transform for frequency analysis
2. **QDA**: Quadratic Discriminant Analysis for gape detection

## Shared Setup

### emg_filter.py

Filter EMG signals before analysis.

**Usage:**

```bash
python emg/emg_filter.py
```

**Filtering Steps:**

1. Bandpass filtering (typically 300-3000 Hz)
2. Notch filtering (remove line noise at 60 Hz)
3. Rectification
4. Smoothing

**Output:**

- Filtered EMG signals saved to HDF5 file
- Filter parameters logged

## BSA/STFT Branch

Frequency-based analysis of EMG signals.

### emg_freq_setup.py

Configure parameters for frequency analysis.

**Usage:**

```bash
python emg/emg_freq_setup.py
```

**Parameters:**

- Frequency bands of interest
- Window sizes
- Overlap parameters
- Output directories

### Parallel Processing

Run frequency analysis in parallel:

```bash
bash blech_emg_jetstream_parallel.sh
```

This script:

1. Divides trials across processors
2. Runs BSA/STFT on each subset
3. Saves intermediate results

### emg_freq_post_process.py

Aggregate and process frequency analysis results.

**Usage:**

```bash
python emg/emg_freq_post_process.py
```

**Processing Steps:**

1. Combine results from parallel jobs
2. Normalize power spectra
3. Calculate summary statistics
4. Identify significant frequency changes

### emg_freq_plot.py

Generate visualizations of frequency analysis.

**Usage:**

```bash
python emg/emg_freq_plot.py
```

**Plots Generated:**

- Spectrograms
- Power spectrum time courses
- Frequency band comparisons
- Trial-averaged responses

## QDA Branch

Gape detection using Quadratic Discriminant Analysis.

### get_gapes_Li.py

Detect gapes using QDA classifier based on Li et al.'s methodology.

**Usage:**

```bash
python emg/get_gapes_Li.py
```

**Method:**

1. Extract EMG features
2. Train QDA classifier on labeled data
3. Predict gape events in test data
4. Validate predictions

**Features Used:**

- EMG amplitude
- Frequency content
- Temporal patterns
- Derivative features

**Output:**

- Gape onset times
- Gape durations
- Confidence scores
- Validation metrics

## EMG Data Structure

### HDF5 Organization

```
data.h5
├── emg/
│   ├── raw/              # Raw EMG signals
│   ├── filtered/         # Filtered EMG signals
│   ├── frequency/        # Frequency analysis results
│   │   ├── power/
│   │   ├── phase/
│   │   └── coherence/
│   └── gapes/            # Detected gape events
│       ├── onset_times/
│       ├── durations/
│       └── confidence/
```

## Configuration

### emg_params.json

EMG analysis parameters:

```json
{
  "filter": {
    "bandpass": [300, 3000],
    "notch": 60,
    "order": 4
  },
  "frequency": {
    "bands": {
      "low": [300, 1000],
      "mid": [1000, 2000],
      "high": [2000, 3000]
    },
    "window_size": 100,
    "overlap": 50
  },
  "gape_detection": {
    "features": ["amplitude", "frequency", "derivative"],
    "classifier": "qda",
    "validation_split": 0.2
  }
}
```

## Usage Examples

### Complete BSA/STFT Workflow

```bash
# Filter EMG signals
python emg/emg_filter.py

# Setup frequency analysis
python emg/emg_freq_setup.py

# Run parallel analysis
bash blech_emg_jetstream_parallel.sh

# Post-process results
python emg/emg_freq_post_process.py

# Generate plots
python emg/emg_freq_plot.py
```

### Complete QDA Workflow

```bash
# Filter EMG signals
python emg/emg_filter.py

# Setup gape detection
python emg/emg_freq_setup.py

# Detect gapes
python emg/get_gapes_Li.py
```

### Programmatic Access

```python
import tables
import numpy as np

# Load filtered EMG data
with tables.open_file('data.h5', 'r') as hf5:
    emg_filtered = hf5.root.emg.filtered[:]

# Load gape events
with tables.open_file('data.h5', 'r') as hf5:
    gape_times = hf5.root.emg.gapes.onset_times[:]
    gape_durations = hf5.root.emg.gapes.durations[:]

# Analyze gape timing
print(f"Detected {len(gape_times)} gapes")
print(f"Mean duration: {np.mean(gape_durations):.2f} ms")
```

## Analysis Tips

### Frequency Analysis

- Use appropriate frequency bands for your species/preparation
- Adjust window size based on temporal resolution needs
- Consider trial-to-trial variability

### Gape Detection

- Manually label training data carefully
- Validate classifier performance on held-out data
- Adjust confidence threshold based on false positive/negative trade-off

### Quality Control

- Inspect filtered signals visually
- Check for artifacts
- Validate detected events manually for subset of trials

## References

Li, J. X., et al. (2016). "Gape detection using quadratic discriminant analysis."
Journal of Neurophysiology, 116(4), 1748-1763.

## See Also

- [Core Pipeline](core-pipeline.md)
- [Utilities](utilities.md)
- [Tutorials](../tutorials.md)
