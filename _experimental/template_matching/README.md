# Template Matching for Spike Detection

This module implements template-based spike detection as an alternative to traditional amplitude thresholding methods. The approach uses optimized waveform templates to identify spikes based on shape similarity rather than just amplitude.

## Overview

The template matching system consists of three main components:

1. **Template Generation** (`generate_templates.py`) - Creates optimized waveform templates from labeled spike data
2. **Template Application** (`conv_template.py`) - Applies templates to detect spikes in new data
3. **Interactive Exploration** (`unit_explorer.py`) - Provides tools to explore and visualize the template database

## Template Generation Process

### 1. Data Preparation
- Loads positive (spike) and negative (non-spike) waveform examples from HDF5 database
- Z-score normalizes waveforms to focus on shape differences rather than amplitude
- Samples balanced datasets for training

### 2. Basis Function Construction
The system uses raised cosine basis functions to parameterize templates:
- **Forward basis**: Covers the main spike region (samples 30-75)
- **Backward basis**: Covers the pre-spike region (samples 0-30)
- **Mirrored combination**: Creates full-length templates spanning the entire waveform

### 3. Template Optimization
Templates are optimized using a custom loss function that:
- **Maximizes class separation**: Positive examples should have high correlation, negative examples low correlation
- **Enforces orthogonality**: Multiple templates should be orthogonal to capture different spike shapes
- **Uses L-BFGS-B optimization**: Finds optimal basis function weights

The loss function is:
```
loss = class_0_penalty + class_1_penalty + orthogonality_penalty
```
Where:
- `class_0_penalty`: Mean correlation score for negative examples (should be ~0)
- `class_1_penalty`: 1 - mean correlation score for positive examples (should be ~0)
- `orthogonality_penalty`: L2 penalty on off-diagonal elements of template correlation matrix

### 4. Template Selection
- Grid search over number of basis functions, templates, and orthogonality weights
- Selects parameters that minimize classification loss
- Saves optimized templates and PCA components for deployment

## Template Application Process

### 1. Signal Preprocessing
- Loads raw electrode data and applies same filtering as training data
- Ensures consistent sampling rate and signal conditioning

### 2. Scaled Cross-Correlation
For each time point, computes scaled cross-correlation between signal segment and template:
```python
def scaled_xcorr(snippet, template):
    snippet = snippet - np.mean(snippet)  # Mean subtract
    snippet = snippet / norm(snippet)     # Normalize
    return np.dot(snippet, template)      # Dot product
```

### 3. Spike Detection
- Applies correlation threshold (typically 0.8-0.9) to identify potential spikes
- Extracts waveforms at detected time points
- Records correlation values for quality assessment

### 4. Validation and Comparison
- Compares detected spikes with traditional clustering results
- Analyzes correlation between template scores and classifier probabilities
- Generates comprehensive visualization reports

## Key Features

- **Shape-based detection**: Focuses on waveform morphology rather than just amplitude
- **Multiple templates**: Can detect different spike types with orthogonal templates
- **Quantitative scoring**: Provides correlation scores for spike quality assessment
- **Validation framework**: Built-in comparison with existing spike sorting results
- **Interactive exploration**: Tools to visualize and understand template performance

## Usage

1. **Generate templates** from labeled data:
   ```bash
   python src/generate_templates.py
   ```

2. **Apply templates** to detect spikes:
   ```bash
   python src/conv_template.py
   ```

3. **Explore results** interactively:
   ```bash
   python src/unit_explorer.py
   ```

## Output

The system generates:
- Optimized template filters saved as `.npz` files
- Detected spike waveforms and timing for each electrode
- Comprehensive visualization plots comparing methods
- Performance metrics and validation statistics

## Advantages

- **Robust to noise**: Template matching is less sensitive to amplitude variations
- **Shape specificity**: Can distinguish between different spike types
- **Quantitative assessment**: Provides correlation scores for spike quality
- **Validation**: Built-in comparison with existing methods
