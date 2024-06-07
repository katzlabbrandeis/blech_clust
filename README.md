# blech_clust

Python and R based code for clustering and sorting electrophysiology data
recorded using the Intan RHD2132 chips.  Originally written for cortical
multi-electrode recordings in Don Katz's lab at Brandeis.  Optimized for the
High performance computing cluster at Brandeis
(https://kb.brandeis.edu/display/SCI/High+Performance+Computing+Cluster) but
can be easily modified to work in any parallel environment. Visit the Katz lab
website at https://sites.google.com/a/brandeis.edu/katzlab/

### Order of operations  
1. `python blech_exp_info.py`  
    - Pre-clustering step. Annotate recorded channels and save experimental parameters  
    - Takes template for info and electrode layout as argument

2. `python blech_clust.py`
    - Setup directories and define clustering parameters  
3. `python blech_common_avg_reference.py`  
    - Perform common average referencing to remove large artifacts  
4. `bash blech_run_process.sh` 
    - Embarrasingly parallel spike extraction and clustering  

5. `python blech_post_process.py`  
    - Add selected units to HDF5 file for further processing  

6. `bash blech_run_QA.sh`  
    - Run quality asurance steps: 1) spike-time collisions across units, 2) drift within units
7. `python blech_units_plot.py`  
    - Plot waveforms of selected spikes  
8. `python blech_make_arrays.py`  
    - Generate spike-train arrays  
9. `python blech_make_psth.py`  
    - Plots PSTHs and rasters for all selected units  
10. `python blech_palatability_identity_setup.py`  
12. `python blech_overlay_psth.py`  
    - Plot overlayed PSTHs for units with respective waveforms  

### Setup
```
conda deactivate                                            # Make sure we're in the base environemnt
conda update -n base conda                                  # Update conda to have the new Libmamba solver

cd <path_to_blech_clust>/requirements                       # Move into blech_clust folder with requirements files
conda clean --all                                           # Removes unused packages and caches
conda create --name blech_clust python=3.8.13               # Create "blech_clust" environment with conda requirements
conda activate blech_clust                                  # Activate blech_clust environment
bash conda_requirements_base.sh                             # Install main packages using conda/mamba
bash install_gnu_parallel.sh                                # Install GNU Parallel
pip install -r pip_requirements_base.txt                    # Install pip requirements (not covered by conda)
bash patch_dependencies.sh                                  # Fix issues with dependencies

### Install neuRecommend (classifier)
cd ~/Desktop                                                # Relocate to download classifier library
git clone https://github.com/abuzarmahmood/neuRecommend.git # Download classifier library
pip install -r neuRecommend/requirements.txt
```
- Parameter files will need to be setup according to [Setting up params](https://github.com/abuzarmahmood/blech_clust/wiki/Getting-Started#setting-up-params)

### Convenience scripts
- blech_clust_pre.sh : Runs steps 2-5  
- blech_clust_post.sh : Runs steps 7-14   

### Operations Workflow Visual 
![nomnoml (1)](https://github.com/abuzarmahmood/blech_clust/assets/12436309/3a44e1a7-af29-4f48-8aa1-427b3e983a81)


### Workflow Walkthrough
Open a terminal, and run:
```
cd /path/to/blech_clust #make the blech_clust repository your working directory
conda activate blech_clust #activate blech_clust
DIR=/path/to/raw/data/files  #save the path of the target Intan data to be sorted
python blech_exp_info.py $DIR  # Generate metadata and electrode layout  
```
Once you've started running the script, it will ask you to "fill in car groups". Go to the intan data folder, where you'll find a file named ```[...]_electrode_layout.csv```. Open this file in a spreadsheet editor, and fill in the ```CAR_group``` column. You should give all of the electrodes implanted in the same bundle the same identifier, and use different identifiers for different bundles (e.g. all electrodes from a bundle in right GC are called ```GC1```, and all electrodes from a bundle in left GC are called ```GC2```). Once you've edited the .csv, return to the terminal and type y/enter.
Next, you'll be asked to provide indices for the intan digital inputs.

```
bash blech_clust_pre.sh $DIR   # Perform steps up to spike extraction and UMAP  
python blech_post_process.py   # Add sorted units to HDF5 (CLI or .CSV as input)  
bash blech_clust_post.sh       # Perform steps up to PSTH generation
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
- - [blech_post_process] -> [bash blech_run_QA.sh]
- - [bash blech_run_QA.sh] -> [blech_units_plot]
- - [blech_units_plot] -> [blech_make_arrays]
- - [blech_make_arrays] -> [blech_make_psth]
- - [blech_make_psth] -> [blech_palatability_identity_setup]
- - [blech_palatability_identity_setup] -> [blech_overlay_psth]

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
