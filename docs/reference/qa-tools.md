# QA Tools

Quality assurance tools for validating and grading electrophysiology datasets.

## Overview

The QA tools provide automated assessment of dataset quality based on multiple metrics:

- Unit quality metrics
- Waveform characteristics
- Firing rate statistics
- Inter-spike interval analysis
- Cluster separation

## blech_run_QA.sh

Main QA script that runs comprehensive quality checks.

### Usage

```bash
bash blech_run_QA.sh
```

### Checks Performed

1. **Waveform Quality**
   - Signal-to-noise ratio
   - Waveform consistency
   - Peak-to-trough ratio

2. **Firing Statistics**
   - Mean firing rate
   - Coefficient of variation
   - Burst analysis

3. **Cluster Quality**
   - Isolation distance
   - L-ratio
   - Silhouette score

4. **ISI Violations**
   - Refractory period violations
   - ISI distribution analysis

## blech_units_characteristics.py

Analyze and compute unit characteristics.

### Usage

```bash
python blech_units_characteristics.py
```

### Computed Metrics

#### Waveform Metrics

- **Peak amplitude**: Maximum waveform amplitude
- **Trough amplitude**: Minimum waveform amplitude
- **Peak-to-trough time**: Time between peak and trough
- **Half-width**: Waveform width at half-maximum

#### Firing Metrics

- **Mean firing rate**: Average spikes per second
- **CV**: Coefficient of variation of ISIs
- **Fano factor**: Variance-to-mean ratio
- **Burst index**: Proportion of spikes in bursts

#### Quality Metrics

- **SNR**: Signal-to-noise ratio
- **Isolation distance**: Cluster separation metric
- **L-ratio**: Cluster quality metric
- **ISI violations**: Refractory period violations

### Output

Creates a CSV file with unit characteristics:

```
unit_num,electrode,single_unit,snr,firing_rate,cv,isolation_distance,...
0,1,True,8.5,12.3,0.45,25.6,...
1,1,False,4.2,8.7,0.62,12.3,...
...
```

## blech_data_summary.py

Generate comprehensive dataset summary.

### Usage

```bash
python utils/blech_data_summary.py
```

### Summary Contents

1. **Dataset Overview**
   - Number of electrodes
   - Number of units
   - Recording duration
   - Number of trials

2. **Unit Statistics**
   - Single vs. multi-unit counts
   - Quality distribution
   - Firing rate distribution

3. **Trial Information**
   - Stimuli presented
   - Trial counts per stimulus
   - Trial timing

4. **Quality Metrics**
   - Overall dataset quality score
   - Per-electrode quality
   - Recommended units for analysis

### Output

Creates a summary report:

```
Dataset Summary
===============
Data Directory: /path/to/data
Recording Date: 2024-01-15

Units:
  Total: 45
  Single Units: 32
  Multi Units: 13
  Mean Quality: 0.78

Trials:
  Total: 120
  Stimuli: 4
  Trials per Stimulus: 30

Quality Score: 8.5/10
```

## grade_dataset.py

Grade dataset quality based on comprehensive metrics.

### Usage

```bash
python utils/grade_dataset.py
```

### Grading Criteria

#### A Grade (9-10)
- High unit count (>30 single units)
- Excellent isolation (>20 isolation distance)
- Low ISI violations (<1%)
- High SNR (>8)

#### B Grade (7-8)
- Good unit count (20-30 single units)
- Good isolation (15-20 isolation distance)
- Moderate ISI violations (1-2%)
- Good SNR (6-8)

#### C Grade (5-6)
- Moderate unit count (10-20 single units)
- Fair isolation (10-15 isolation distance)
- Some ISI violations (2-5%)
- Fair SNR (4-6)

#### D Grade (<5)
- Low unit count (<10 single units)
- Poor isolation (<10 isolation distance)
- High ISI violations (>5%)
- Low SNR (<4)

### Output

Creates a grading report:

```
Dataset Grade: B+
===============

Overall Score: 7.8/10

Component Scores:
  Unit Count: 8/10 (28 single units)
  Isolation Quality: 7/10 (avg 17.5)
  ISI Violations: 9/10 (0.8% violations)
  SNR: 7/10 (avg 6.8)

Recommendations:
  - Good quality dataset suitable for analysis
  - Consider excluding 3 low-quality units
  - Electrode 5 shows lower quality, review manually
```

## Quality Metrics Reference

### Signal-to-Noise Ratio (SNR)

```
SNR = peak_amplitude / noise_std
```

- **Good**: SNR > 8
- **Fair**: SNR 4-8
- **Poor**: SNR < 4

### Isolation Distance

Mahalanobis distance-based cluster separation metric.

- **Good**: > 20
- **Fair**: 10-20
- **Poor**: < 10

### L-ratio

Cluster quality metric based on Mahalanobis distance.

- **Good**: < 0.1
- **Fair**: 0.1-0.3
- **Poor**: > 0.3

### ISI Violations

Percentage of spikes violating refractory period (typically 2ms).

- **Good**: < 1%
- **Fair**: 1-2%
- **Poor**: > 2%

## Usage Examples

### Basic QA Workflow

```bash
# Run comprehensive QA
bash blech_run_QA.sh

# Analyze unit characteristics
python blech_units_characteristics.py

# Generate summary
python utils/blech_data_summary.py

# Grade dataset
python utils/grade_dataset.py
```

### Programmatic Access

```python
import pandas as pd

# Load unit characteristics
units = pd.read_csv('unit_characteristics.csv')

# Filter high-quality units
good_units = units[
    (units['single_unit'] == True) &
    (units['snr'] > 6) &
    (units['isolation_distance'] > 15) &
    (units['isi_violations'] < 0.02)
]

print(f"Found {len(good_units)} high-quality units")
```

## See Also

- [Core Pipeline](core-pipeline.md)
- [Utilities](utilities.md)
- [Tutorials](../tutorials.md)
