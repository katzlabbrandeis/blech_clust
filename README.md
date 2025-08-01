[![DOI](https://zenodo.org/badge/119422765.svg)](https://doi.org/10.5281/zenodo.15175272)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/katzlabbrandeis/blech_clust/master.svg)](https://results.pre-commit.ci/latest/github/katzlabbrandeis/blech_clust/master)

# blech_clust

Python and R based code for clustering and sorting electrophysiology data
recorded using the Intan RHD2132 chips.  Originally written for cortical
multi-electrode recordings in Don Katz's lab at Brandeis.  Optimized for the
High performance computing cluster at Brandeis
(https://kb.brandeis.edu/display/SCI/High+Performance+Computing+Cluster) but
can be easily modified to work in any parallel environment. Visit the Katz lab
website at https://sites.google.com/a/brandeis.edu/katzlab/

### Order of operations

**Spike Sorting Pipeline:**
1. `python blech_exp_info.py`
    - Pre-clustering step. Annotate recorded channels and save experimental parameters
    - Takes template for info and electrode layout as argument
2. `python blech_clust.py`
    - Setup directories and define clustering parameters
3. `python blech_common_avg_reference.py`
    - Perform common average referencing to remove large artifacts
4. `bash blech_run_process.sh`
    - Parallel spike extraction and clustering
5. `python blech_post_process.py`
    - Add selected units to HDF5 file for further processing
6. `python blech_units_plot.py`
    - Plot waveforms of selected spikes
7. `python blech_make_arrays.py`
    - Generate spike-train arrays
8. `bash blech_run_QA.sh`
    - Quality assurance: spike-time collisions and drift analysis
9. `python blech_unit_characteristics.py`
    - Analyze unit characteristics
10. `python utils/blech_data_summary.py`
    - Generate comprehensive dataset summary
11. `python utils/grade_dataset.py`
    - Grade dataset quality based on metrics

**EMG Analysis Pipelines:**

*Shared Steps:*
1. Complete spike sorting through `blech_make_arrays.py`
    - Required for temporal alignment with neural data
2. `python emg_filter.py`
    - Filter EMG signals using bandpass filter

*BSA/STFT Branch:* (Bayesian Spectrum Analysis/Short-Time Fourier Transform)
1. `python emg_freq_setup.py`
    - Configure parameters for frequency analysis
2. `bash blech_emg_jetstream_parallel.sh`
    - Parallel processing of EMG signals using BSA/STFT
3. `python emg_freq_post_process.py`
    - Aggregate and process frequency analysis results
4. `python emg_freq_plot.py`
    - Generate visualizations of EMG frequency components

*QDA (Jenn Li) Branch:* (Quadratic Discriminant Analysis)
1. `python emg_freq_setup.py`
    - Setup parameters for gape detection
2. `python get_gapes_Li.py`
    - Detect gapes using QDA classifier
    - Based on Li et al.'s methodology for EMG pattern recognition

### Module Documentation

- [Ephys Data Module](utils/ephys_data/README.md): Documentation for analyzing electrophysiology data

### Dataset Quality Assessment

The pipeline includes tools for assessing and grading dataset quality:

1. **Data Summary Generation** (`utils/blech_data_summary.py`):
   - Aggregates key metrics from analysis outputs
   - Summarizes unit counts, responsiveness, drift metrics, and more
   - Creates a comprehensive `data_summary.json` file

2. **Dataset Grading** (`utils/grade_dataset.py`):
   - Evaluates dataset quality based on configurable criteria
   - Grades datasets on unit counts, unit quality, and drift metrics
   - Uses thresholds defined in `utils/grading_metrics.json`
   - Outputs grades to `QA_output/grading.json`

These tools help standardize quality assessment across datasets and identify recordings that meet experimental quality standards.

### Blog

For more insights and updates, visit our [Blog Page](https://katzlabbrandeis.github.io/blech_clust/blogs/blogs_main.html).

### Contributing

We welcome contributions to the blech_clust project! If you're interested in contributing, please read our [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to get started. Whether it's reporting issues, submitting pull requests, or improving documentation, your help is appreciated.

### Acknowledgments

This work used ACCESS-allocated resources at Brandeis University through allocation BIO230103 from the [Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support](https://access-ci.org/) (ACCESS) program, which is supported by U.S. National Science Foundation grants #2138259, #2138286, #2138307, #2137603, and #2138296.

The project titled "Computational Processing and Modeling of Neural Ensembles in Identifying the Nonlinear Dynamics of Taste Perception" was led by PI Abuzar Mahmood. The computational allocation was active from 2023-06-26 to 2024-06-25.

### Setup

The installation process is managed through a Makefile that handles all dependencies and environment setup.

To install everything (recommended):
```bash
make all
```

Or install components individually:
```bash
make base      # Install base environment and dependencies
make emg       # Install EMG analysis requirements (optional)
make neurec    # Install neuRecommend classifier
make blechrnn  # Install BlechRNN for firing rate estimation (optional)
```

**Note:** If you plan to use GPU with BlechRNN, you'll need to install CUDA separately.
See: [Installing PyTorch with GPU Support](https://medium.com/@jeanpierre_lv/installing-pytorch-with-gpu-support-on-ubuntu-a-step-by-step-guide-38dcf3f8f266)

To remove the environment and start fresh:
```bash
make clean
```
- Parameter files will need to be setup according to [Setting up params](https://github.com/abuzarmahmood/blech_clust/wiki/Getting-Started#setting-up-params)

### Testing

#### Local Testing with Prefect
The project uses Prefect for orchestrating test pipelines locally. To run tests:

1. Start the Prefect server in a separate terminal:
```bash
prefect server start
```

2. In another terminal, run the tests:
```bash
cd <path_to_blech_clust>                # Move to blech_clust directory
make prefect                            # Install/update Prefect
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

You can monitor test progress and results in the Prefect UI at http://localhost:4200

#### Continuous Integration
The project uses GitHub Actions for automated testing on pull requests:

- Pre-commit checks run automatically on all PRs to enforce code style and quality
- The full test suite runs on self-hosted runners when PRs are labeled with 'install'
- Test results are reported in the PR checks interface
- The workflow configuration is in `.github/workflows/python_workflow_test.yml`


### Convenience scripts
- blech_clust_pre.sh : Runs steps 2-5
- blech_clust_post.sh : Runs steps 7-14

### blech_autosort.sh Script

The `blech_autosort.sh` script is designed to automate the process of clustering and post-processing electrophysiology data. It ensures that the necessary parameters are set correctly and executes the pre-processing, clustering, and post-processing steps in sequence.

**Usage:**
```bash
bash blech_autosort.sh <data_directory> [--force]
```

- `<data_directory>`: Path to the directory containing the raw data files.
- `--force`: Optional flag to force re-processing even if previous results exist.

**Functionality:**
- Checks for the existence of required parameter files and verifies that specific settings are enabled.
- Executes the `blech_clust_pre.sh` script to perform initial processing.
- Runs `blech_post_process.py` to add sorted units to the HDF5 file.
- Completes the workflow with `blech_clust_post.sh` for further processing.

Ensure that the parameter files `sorting_params_template.json` and `waveform_classifier_params.json` are correctly configured before running this script.

### Operations Workflow Visual
![nomnoml](https://github.com/user-attachments/assets/5a30d8f3-3653-4ce7-ae68-0623e3885210)

### Quality Assessment Workflow
```
blech_unit_characteristics.py → blech_data_summary.py → grade_dataset.py
```

### Workflow Walkthrough
*This section is being expanded, in progress.*

Open a terminal, and run:
```
cd /path/to/blech_clust #make the blech_clust repository your working directory
conda activate blech_clust #activate blech_clust
DIR=/path/to/raw/data/files  #save the path of the target Intan data to be sorted
python blech_exp_info.py $DIR  # Generate metadata and electrode layout
```
Once you've started running the script, it will ask you to "fill in car groups". Go to the intan data folder, where you'll find a file named ```[...]_electrode_layout.csv```. Open this file in a spreadsheet editor, and fill in the ```CAR_group``` column. You should give all of the electrodes implanted in the same bundle the same identifier, and use different identifiers for different bundles (e.g. all electrodes from a bundle in right GC are called ```GC1```, and all electrodes from a bundle in left GC are called ```GC2```). Once you've edited the .csv, return to the terminal and type ```y``` ```enter```.
The script will then search your data folder for DIN files, and will print something like this, though the specific files may vary:
```
(0, 'board-DIN-09.dat'),
(1, 'board-DIN-11.dat'),
(2, 'board-DIN-12.dat'),
(3, 'board-DIN-13.dat')
```
These are the files for the Intan digital inputs that blech_clust has detected in your data folder, which should correspond to the on/off times of each stimulus presentation, and/or laser activations.

You'll also be given this prompt: ```Taste dig_ins used (IN ORDER, anything separated)  :: "x" to exit ::```. You can select as many or as few as you'd like to be included in later steps in the analysis; they don't impact initial spike-sorting. In this case, if we wanted to include DINs 11 and 13 but not 09 or 12, we would type ```1,3``` ```enter``` in the terminal, using the indices corresponding to the desired DINs. If you have a DIN for laser activations, do not include that here; it will be requested at a later step.

Next, you'll see this dialog: ```Tastes names used (IN ORDER, anything separated)  :: "x" to exit ::```, asking to provide taste names for each of your selected DINs. Supposing that DIN-11 was associated with DI H2O, and DIN-13 was 300mM sucrose, we would enter ```Water,Sucrose``` ```enter```, leaving off the molarity, which will be provided in the next step.

That prompt (```Corresponding concs used (in M, IN ORDER, COMMA separated)  :: "x" to exit ::```) should immediately follow. This requires numeric inputs, so for our stimuli of DI H2O and 300mM sucrose, the appropriate input would be ```0,0.3``` ```enter```, giving the molarity of Water as 0, and converting mM to M in the sucrose concentration.

The next prompt (```Enter palatability rankings used (anything separated), higher number = more palatable  :: "x" to exit ::```) asks for palatability rankings for the stimuli. This requires a numeric input > 0 and <= the number of stimuli, does not need to be integer, and accepts duplicate values (e.g. ```4,3,2,1``` is fine, ```0.4,0.3,0.2,0.1``` is also fine, even ```3,2,2,1``` is fine, but ```2,1,1,0``` is not, nor is ```5,4,3,2```). In our water/sucrose example, ```1,2``` ```enter``` would be an appropriate entry.

The next prompt (```Laser dig_in index, <BLANK> for none::: "x" to exit ::```) asks for the index of the DIN corresponding to laser activations. If DIN-09 was the channel for the laser, for example, we would type ```0``` ```enter```, using the index corresponding with DIN-09. Alternatively, if we had no laser, we would just hit ```enter``` to proceed.

Our final prompt (```::: Please enter any notes about the experiment.```) just asks for notes. Enter any pertinent comments, or just hit ```enter``` to finish running ```blech_exp_info.py```

Once we've finished with ```blech_exp_info.py```, we'll want to continue on with either blech_clust.py or blech_clust_pre.sh. However, before we can run either thing, we'll need to set up a params file. First, copy blech_clust/params/_templates/sorting_params_template.json to blech_clust/params/sorting_params_template.json and update as needed.

While you're there, you should also copy and adapt the other two params templates (`waveform_classifier_params.json` and `emg_params.json`), or you will probably be haunted by errors.

```
bash blech_clust_pre.sh $DIR   # Perform steps up to spike extraction and UMAP
python blech_post_process.py   # Add sorted units to HDF5 (CLI or .CSV as input)
bash blech_clust_post.sh       # Perform steps up to PSTH generation
```

### Utilities
#### utils/infer_rnn_rates.py

This script is used to infer firing rates from spike trains using a Recurrent Neural Network (RNN). The RNN is trained on the spike trains and the firing rates are inferred from the trained model. The script uses the `BlechRNN` library for training the RNN.

```
usage: infer_rnn_rates.py [-h] [--override_config] [--train_steps TRAIN_STEPS]
                          [--hidden_size HIDDEN_SIZE] [--bin_size BIN_SIZE]
                          [--train_test_split TRAIN_TEST_SPLIT] [--no_pca]
                          [--retrain] [--time_lims TIME_LIMS TIME_LIMS]
                          data_dir

Infer firing rates using RNN

positional arguments:
  data_dir              Path to data directory

optional arguments:
  -h, --help            show this help message and exit
  --override_config     Override config file and use provided
                        arguments(default: False)
  --train_steps TRAIN_STEPS
                        Number of training steps (default: 15000)
  --hidden_size HIDDEN_SIZE
                        Hidden size of RNN (default: 8)
  --bin_size BIN_SIZE   Bin size for binning spikes (default: 25)
  --train_test_split TRAIN_TEST_SPLIT
                        Fraction of data to use for training (default: 0.75)
  --no_pca              Do not use PCA for preprocessing (default: False)
  --retrain             Force retraining of model. Will overwrite existing
                        model (default: False)
  --time_lims TIME_LIMS TIME_LIMS
                        Time limits inferred firing rates (default: [1500,
                        4500])
```

### Test Dataset
We are grateful to Brandeis University Google Filestream for hosting this dataset <br>
Data to test workflow available at:<br>
https://drive.google.com/drive/folders/1ne5SNU3Vxf74tbbWvOYbYOE1mSBkJ3u3?usp=sharing

### Dependency Graph (for use with https://www.nomnoml.com/)

- **Spike Sorting**
- - [blech_exp_info] -> [blech_clust]
- - [blech_clust] -> [blech_common_average_reference]
- - [blech_common_average_reference] -> [bash blech_run_process.sh]
- - [bash blech_run_process.sh] -> [blech_post_process]
- - [blech_post_process] -> [blech_units_plot]
- - [blech_units_plot] -> [blech_make_arrays]
- - [blech_make_arrays] -> [bash blech_run_QA.sh]
- - [bash blech_run_QA.sh] -> [blech_unit_characteristics]
- - [blech_unit_characteristics] -> [blech_data_summary]
- - [blech_data_summary] -> [grade_dataset]

- **EMG shared**
- - [blech_clust] -> [blech_make_arrays]
- - [blech_make_arrays] -> [emg_filter]

- **BSA/STFT**
- - [emg_filter] -> [emg_freq_setup]
- - [emg_freq_setup] -> [bash blech_emg_jetstream_parallel.sh]
- - [bash blech_emg_jetstream_parallel.sh] -> [emg_freq_post_process]
- - [emg_freq_post_process] -> [emg_freq_plot]

- **QDA (Jenn Li)**
- - [emg_freq_setup] -> [get_gapes_Li]

### Citation
If you use this code in your research, please cite the following paper:

```bibtex
@software{blech_clust_katz,
  author       = {Mahmood, Abuzar and
                  Mukherjee, Narendra and
                  Stone, Bradly and
                  Raymond, Martin and
                  Germaine, Hannah and
                  Lin, Jian-You and
                  Mazzio, Christina and
                  Katz, Donald},
  title        = {katzlabbrandeis/blech\_clust: v1.1.0},
  month        = apr,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {1.1.0},
  doi          = {10.5281/zenodo.15175273},
  url          = {https://doi.org/10.5281/zenodo.15175273},
  swhid        = {swh:1:dir:3970ccd774c6854819d510a288871435b6df3102
                   ;origin=https://doi.org/10.5281/zenodo.15175272;vi
                   sit=swh:1:snp:8c3e68b1e08e872f071a89982e99270c5846
                   0e87;anchor=swh:1:rel:c90c05b9e71fc340ac44e6c742b2
                   00fe3c655588;path=katzlabbrandeis-
                   blech\_clust-10c7fab
                  },
}
```
