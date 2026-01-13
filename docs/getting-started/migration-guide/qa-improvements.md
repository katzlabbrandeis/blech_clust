# Quality Assurance Improvements

The current fork includes quality assurance tools that were not present in the original blech_clust. These tools help identify issues with recordings and sorted units.

## Overview

| Tool | Purpose |
|------|---------|
| `blech_run_QA.sh` | Run all QA checks |
| `utils/qa_utils/drift_check.py` | Detect drift during recording |
| `utils/qa_utils/channel_corr.py` | Channel correlation analysis |
| `utils/qa_utils/unit_similarity.py` | Unit similarity metrics |
| `utils/qa_utils/elbo_drift.py` | ELBO-based drift detection |
| `utils/grade_dataset.py` | Dataset quality grading |
| `utils/grading_metrics.json` | Configurable grading criteria |
| `utils/blech_data_summary.py` | At-a-glance dataset insights |

## Running QA Checks

After completing the spike sorting pipeline, run QA checks:

```bash
bash blech_run_QA.sh /path/to/data
```

This script runs the QA utilities and generates reports.

---

## Drift Detection

### Population-Level Drift

The `drift_check.py` module detects drift at the population level by analyzing firing rate stability across the recording:

```python
from utils.qa_utils import drift_check

# Run drift analysis
drift_results = drift_check.analyze_drift(data_dir='/path/to/data')
```

### Single-Unit Drift

Drift can also be assessed at the single-unit level to identify units that may have drifted during the recording:

- Firing rate changes over time
- Waveform amplitude changes
- ISI distribution changes

### ELBO-Based Drift

The `elbo_drift.py` module uses Evidence Lower Bound (ELBO) metrics from variational inference to detect population drift. This approach provides a probabilistic framework for identifying when neural activity patterns change significantly.

```python
from utils.qa_utils import elbo_drift

# Analyze ELBO drift
elbo_results = elbo_drift.analyze(data_dir='/path/to/data')
```

Key features:
- Uses `pymc` for Bayesian statistical modeling
- Performs drift detection on spike-time histograms (more accurate than PCA of firing rates)
- Supports flexible changepoints
- Exports results to CSV for further analysis

---

## Channel Correlation

The `channel_corr.py` module analyzes correlations between channels to identify:

- Dead channels
- Channels with excessive noise
- Cross-talk between channels
- Channels that should be excluded from common average reference

```python
from utils.qa_utils import channel_corr

# Analyze channel correlations
corr_results = channel_corr.analyze(data_dir='/path/to/data')
```

This enhancement bridges the gap between raw channel data and meaningful quality metrics.

---

## Unit Similarity

The `unit_similarity.py` module (relocated from `blech_units_similarity.py`) computes similarity metrics between sorted units:

- Waveform similarity
- Firing pattern correlation
- Potential duplicate detection

```python
from utils.qa_utils import unit_similarity

# Analyze unit similarity
similarity_results = unit_similarity.analyze(data_dir='/path/to/data')
```

---

## Data Summary

The `blech_data_summary.py` script provides at-a-glance dataset insights:

```bash
python utils/blech_data_summary.py /path/to/data
```

This generates a summary including:
- Number of units and their quality
- Recording duration and stability
- Trial counts per condition
- Basic firing rate statistics

---

## Dataset Grading

### Grading Script

The `grade_dataset.py` script assigns quality grades based on configurable metrics, automating the assessment process:

```bash
python utils/grade_dataset.py /path/to/data
```

### Grading Metrics

The `grading_metrics.json` file defines the criteria for grading:

```json
{
    "unit_count": {
        "weight": 0.3,
        "thresholds": {
            "A": 20,
            "B": 10,
            "C": 5,
            "D": 1
        }
    },
    "firing_rate_stability": {
        "weight": 0.2,
        "thresholds": {
            "A": 0.9,
            "B": 0.7,
            "C": 0.5,
            "D": 0.3
        }
    }
    // ... additional metrics
}
```

You can customize these thresholds for your experimental requirements.

---

## Cluster Stability

The `utils/cluster_stability.py` module assesses the stability of clusters over time:

```python
from utils import cluster_stability

# Analyze cluster stability
stability = cluster_stability.analyze(data_dir='/path/to/data')
```

This helps identify:

- Clusters that split or merge during recording
- Units with unstable waveforms
- Potential sorting errors

### Hierarchical Clustering Visualization

The pipeline includes hierarchical clustering plots for visual assessment of cluster quality. These plots help identify:

- Well-separated clusters
- Clusters that may need to be merged
- Outlier waveforms

---

## Common Average Reference QA

The enhanced `blech_common_avg_reference.py` includes QA features:

### Dead Channel Detection

Automatically identifies channels that are:

- Flat (no signal)
- Saturated
- Significantly different from other channels in the CAR group

### Channel Clustering

Clusters channels within CAR groups to identify:

- Channels that should be excluded from CAR
- Potential grouping issues

### Visualization

Generates plots showing:

- Channel distributions within CAR groups
- Outlier channels
- CAR effectiveness

---

## Unit Quality Visualization

### Feature Over Time Plots

`blech_units_plot.py` generates plots showing spike features over the recording duration:

- Amplitude stability
- Waveform shape consistency
- ISI distribution changes

### Waveform Classifier

The waveform classifier provides quality recommendations:

```python
# Classifier recommendations are stored in the HDF5 file
# Access via:
import tables
with tables.open_file('data.h5', 'r') as hf5:
    recommendations = hf5.root.sorted_units.classifier_recommendations[:]
```

---

## Integration with Pipeline

QA checks are integrated into the pipeline workflow:

1. **During CAR**: Dead channel detection and visualization
2. **After sorting**: Cluster stability and unit similarity
3. **Post-processing**: Drift detection and grading
4. **Final**: Dataset grade assignment

### Automated QA

Use `blech_autosort.sh` with the `--qa` flag to include QA in the automated pipeline:

```bash
bash blech_autosort.sh /path/to/data --qa
```

Or run QA separately:

```bash
bash blech_run_QA.sh /path/to/data
```

---

## Interpreting QA Results

### Drift Indicators

| Indicator | Interpretation |
|-----------|----------------|
| Low drift score | Recording is stable |
| High drift score | Significant drift detected |
| Unit-specific drift | Individual unit may have moved |

### Grade Interpretation

| Grade | Interpretation |
|-------|----------------|
| A | High quality, suitable for all analyses |
| B | Good quality, suitable for most analyses |
| C | Acceptable quality, use with caution |
| D | Poor quality, may need re-sorting or exclusion |

### Recommended Actions

| Issue | Recommended Action |
|-------|-------------------|
| High drift | Consider splitting recording into segments |
| Low unit count | Review sorting parameters |
| Channel issues | Exclude problematic channels from CAR |
| Similar units | Review for potential duplicates |
