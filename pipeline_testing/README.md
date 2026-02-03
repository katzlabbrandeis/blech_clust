# Pipeline Testing Framework

This directory contains tools for testing the blech_clust pipeline workflows using Prefect.

## Overview

The testing framework allows automated testing of different components of the blech_clust pipeline:
- Spike sorting workflows
- EMG processing workflows (using both BSA and STFT methods)
- Gape detection using QDA classifier
- Combined spike sorting + EMG processing workflows

## Test Data

The framework uses small test datasets that can be automatically downloaded:
- OFPC format data: `KM45_5tastes_210620_113227_new`
- Traditional format data: `eb24_behandephys_11_12_24_241112_114659_copy`

## Configuration

Test data locations and dataset information are configured in `test_config.json`:

```json
{
    "test_data_dir": "~/.blech_clust_test_data",
    "datasets": {
        "ofpc": {
            "name": "KM45_5tastes_210620_113227_new",
            "gdrive_id": "1EcpUIqp81h3J89-6dEEueeULqBlKW5a7"
        },
        "trad": {
            "name": "eb24_behandephys_11_12_24_241112_114659_copy",
            "gdrive_id": "1aU2DWHhbVB3rDujbF4KRX9QLA1LlfpU3"
        }
    }
}
```

Use `test_config_loader.py` to access configuration values in Python code:

```python
from blech_clust.pipeline_testing.test_config_loader import (
    get_data_dirs_dict,
    get_test_data_dir,
)
```

## Running Tests

You can run specific test workflows or all tests:

```bash
# Run all tests
python pipeline_testing/prefect_pipeline.py --all

# Run only spike sorting tests
python pipeline_testing/prefect_pipeline.py -s

# Run only EMG tests
python pipeline_testing/prefect_pipeline.py -e

# Run only BSA frequency analysis tests
python pipeline_testing/prefect_pipeline.py --bsa

# Run only STFT frequency analysis tests
python pipeline_testing/prefect_pipeline.py --stft

# Run only QDA gape detection tests
python pipeline_testing/prefect_pipeline.py --qda

# Run combined spike sorting + EMG tests
python pipeline_testing/prefect_pipeline.py --spike-emg

# Raise exceptions on test failures
python pipeline_testing/prefect_pipeline.py --all --raise-exception
```

## Utility Scripts

The directory includes several utility scripts:
- `reset_blech_clust.py`: Removes temporary files
- `cut_emg_trials.py`: Reduces EMG trial count for faster testing
- `change_auto_params.py`: Toggles auto-clustering parameters
- `change_waveform_classifier.py`: Toggles waveform classifier usage
- `change_emg_freq_method.py`: Switches between BSA and STFT methods
- `mark_exp_info_success.py`: Marks experiment info as successfully completed
- `select_some_waveforms.py`: Selects a subset of waveforms for testing
- `create_exp_info_commands.py`: Generates experiment info commands

## Future Improvements

Future versions may explore other workflow management tools such as:
- Apache Airflow
- Luigi
- Dagster
