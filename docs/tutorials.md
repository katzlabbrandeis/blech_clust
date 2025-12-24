# Tutorials

## Workflow Walkthrough

This tutorial walks you through the complete spike sorting pipeline from raw data to analyzed units.

### Step 1: Experiment Information Setup

Open a terminal and run:

```bash
cd /path/to/blech_clust  # Make the blech_clust repository your working directory
conda activate blech_clust  # Activate blech_clust environment
DIR=/path/to/raw/data/files  # Save the path of the target Intan data to be sorted
python blech_exp_info.py $DIR  # Generate metadata and electrode layout
```

#### Configuring CAR Groups

Once you've started running the script, it will ask you to "fill in car groups". Go to the Intan data folder, where you'll find a file named `[...]_electrode_layout.csv`.

1. Open this file in a spreadsheet editor
2. Fill in the `CAR_group` column
3. Give all electrodes implanted in the same bundle the same identifier
4. Use different identifiers for different bundles
   - Example: All electrodes from a bundle in right GC are called `GC1`, and all electrodes from a bundle in left GC are called `GC2`
5. Return to the terminal and type `y` then press `enter`

#### Selecting Digital Inputs

The script will search your data folder for DIN files and print something like:

```
(0, 'board-DIN-09.dat'),
(1, 'board-DIN-11.dat'),
(2, 'board-DIN-12.dat'),
(3, 'board-DIN-13.dat')
```

These are the files for the Intan digital inputs that correspond to stimulus presentations and/or laser activations.

**Prompt:** `Taste dig_ins used (IN ORDER, anything separated) :: "x" to exit ::`

- Select the DINs to include in later analysis steps
- Example: To include DINs 11 and 13 but not 09 or 12, type `1,3` and press `enter`
- Note: If you have a DIN for laser activations, do not include it here; it will be requested later

#### Naming Tastes

**Prompt:** `Tastes names used (IN ORDER, anything separated) :: "x" to exit ::`

- Provide taste names for each selected DIN
- Example: If DIN-11 was DI H2O and DIN-13 was 300mM sucrose, enter `Water,Sucrose`
- Leave off the molarity (provided in the next step)

#### Specifying Concentrations

**Prompt:** `Corresponding concs used (in M, IN ORDER, COMMA separated) :: "x" to exit ::`

- Provide numeric inputs for concentrations in Molarity
- Example: For DI H2O and 300mM sucrose, enter `0,0.3`

#### Palatability Rankings

**Prompt:** `Enter palatability rankings used (anything separated), higher number = more palatable :: "x" to exit ::`

- Provide numeric rankings (> 0 and <= number of stimuli)
- Can be non-integer and accept duplicates
- Valid examples: `4,3,2,1` or `0.4,0.3,0.2,0.1` or `3,2,2,1`
- Invalid examples: `2,1,1,0` (contains 0) or `5,4,3,2` (exceeds number of stimuli)
- Example for water/sucrose: `1,2`

#### Laser Configuration

**Prompt:** `Laser dig_in index, <BLANK> for none::: "x" to exit ::`

- If you have a laser DIN, enter its index (e.g., `0` for DIN-09)
- If no laser, just press `enter`

#### Experiment Notes

**Prompt:** `::: Please enter any notes about the experiment.`

- Enter any pertinent comments or press `enter` to finish

### Step 2: Parameter Configuration

Before running the clustering pipeline, set up parameter files:

1. Copy `blech_clust/params/_templates/sorting_params_template.json` to `blech_clust/params/sorting_params_template.json`
2. Update the parameters as needed for your experiment
3. Also copy and adapt:
   - `waveform_classifier_params.json`
   - `emg_params.json`

### Step 3: Run the Pipeline

#### Using Convenience Scripts

```bash
bash blech_clust_pre.sh $DIR   # Perform steps up to spike extraction and UMAP
python blech_post_process.py   # Add sorted units to HDF5 (CLI or .CSV as input)
bash blech_clust_post.sh       # Perform steps up to PSTH generation
```

#### Or Use the Automated Script

```bash
bash blech_autosort.sh <data_directory> [--force]
```

- `<data_directory>`: Path to the directory containing the raw data files
- `--force`: Optional flag to force re-processing even if previous results exist

The `blech_autosort.sh` script:

- Checks for required parameter files
- Verifies that specific settings are enabled
- Executes the pre-processing, clustering, and post-processing steps in sequence

### Step 4: Quality Assessment

After processing, assess the quality of your dataset:

```bash
python blech_units_characteristics.py  # Analyze unit characteristics
python utils/blech_data_summary.py    # Generate comprehensive dataset summary
python utils/grade_dataset.py         # Grade dataset quality based on metrics
```

## EMG Analysis Tutorial

### Shared Setup

1. Complete spike sorting through `blech_make_arrays.py`
2. Filter EMG signals:
   ```bash
   python emg_filter.py
   ```

### BSA/STFT Branch

For Bayesian Spectrum Analysis and Short-Time Fourier Transform:

```bash
python emg_freq_setup.py              # Configure parameters and generate parallel processing scripts
bash blech_emg_jetstream_parallel.sh  # Run the generated parallel processing script
python emg_freq_post_process.py       # Aggregate and process results
python emg_freq_plot.py               # Generate visualizations
```

**Note:** The `emg_freq_setup.py` script generates the `blech_emg_jetstream_parallel.sh` script, which uses GNU parallel to process EMG signals in parallel.

### QDA Branch

For Quadratic Discriminant Analysis (gape detection):

```bash
python emg_freq_setup.py                      # Setup parameters for gape detection
python emg/gape_QDA_classifier/get_gapes_Li.py  # Detect gapes using QDA classifier
```

## Testing Your Installation

### Local Testing with Prefect

1. Start the Prefect server in a separate terminal:
   ```bash
   prefect server start
   ```

2. In another terminal, run the tests:
   ```bash
   cd <path_to_blech_clust>
   make prefect  # Install/update Prefect
   ```

3. Run specific test suites:
   ```bash
   # Run all tests
   python pipeline_testing/prefect_pipeline.py --all

   # Run only spike sorting tests
   python pipeline_testing/prefect_pipeline.py -s

   # Run only EMG analysis tests
   python pipeline_testing/prefect_pipeline.py -e

   # Run spike sorting followed by EMG analysis
   python pipeline_testing/prefect_pipeline.py --spike-emg
   ```

Monitor test progress at [http://localhost:4200](http://localhost:4200)

## Advanced Topics

### RNN-based Firing Rate Inference

Use the `infer_rnn_rates.py` utility to infer firing rates from spike trains:

```bash
python utils/infer_rnn_rates.py <data_dir> [options]
```

Options:

- `--override_config`: Override config file and use provided arguments
- `--train_steps TRAIN_STEPS`: Number of training steps (default: 15000)
- `--hidden_size HIDDEN_SIZE`: Hidden size of RNN (default: 8)
- `--bin_size BIN_SIZE`: Bin size for binning spikes (default: 25)
- `--train_test_split TRAIN_TEST_SPLIT`: Fraction of data for training (default: 0.75)
- `--no_pca`: Do not use PCA for preprocessing
- `--retrain`: Force retraining of model
- `--time_lims TIME_LIMS TIME_LIMS`: Time limits for inferred firing rates (default: [1500, 4500])

## Additional Resources

- [Module Documentation](https://github.com/katzlabbrandeis/blech_clust/blob/master/utils/ephys_data/README.md): Detailed documentation for the ephys_data module
- [Wiki](https://github.com/abuzarmahmood/blech_clust/wiki): Additional guides and information
- [Blog](https://katzlabbrandeis.github.io/blech_clust/blogs/blogs_main.html): Insights and updates from the development team
