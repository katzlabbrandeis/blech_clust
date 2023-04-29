# blech_clust

Python and R based code for clustering and sorting electrophysiology data
recorded using the Intan RHD2132 chips.  Originally written for cortical
multi-electrode recordings in Don Katz's lab at Brandeis.  Optimized for the
High performance computing cluster at Brandeis
(https://kb.brandeis.edu/display/SCI/High+Performance+Computing+Cluster) but
can be easily modified to work in any parallel environment. Visit the Katz lab
website at https://sites.google.com/a/brandeis.edu/katzlab/

### Setup
```
cd <path_to_blech_clust>/requirements               # Move into blech_clust folder with requirements files
conda clean --all                                   # Removes unused packages and caches
conda create --name blech_clust python=3.6.13       # Create "blech_clust" environment with conda requirements
conda activate blech_clust                          # Activate blech_clust environment
bash conda_requirements_base.sh                     # Install main packages using conda/mamba
bash install_gnu_parallel.sh                        # Install GNU Parallel
pip install -r pip_requirements_base.txt            # Install pip requirements (not covered by conda)
```

blech_clust relies on [neuRecommend](https://github.com/abuzarmahmood/neuRecommend) to automatically classify
neurons. Please clone neuRecommend to your desktop, download the models used by it, and install requirements 
in your blech_clust environment...
```
cd ~/Desktop
git clone https://github.com/abuzarmahmood/neuRecommend.git
cd neuRecommend
bash src/utils/io/download_models.sh
conda activate blech_clust  
pip install -r requirements.txt
```

### Convenience scripts
- blech_clust_pre.sh : Runs steps 2-5  
- blech_clust_post.sh : Runs steps 7-14   

### Order of operations  
1. python blech_exp_info.py  
    - Pre-clustering step. Annotate recorded channels and save experimental parameters  
    - Takes template for info and electrode layout as argument

2. python blech_clust.py  
    - Setup directories and define clustering parameters  
3. python blech_common_avg_reference.py  
    - Perform common average referencing to remove large artifacts  
4. bash blech_clust_jetstream_parallel.sh  
    - Embarrasingly parallel spike extraction and clustering  
5. bash bash_umap_spike_scatter.sh  
    - UMAP embedding of spikes for visualization of clusters, and generate spike-time rasters  

6. python blech_post_process.sh  
    - Add selected units to HDF5 file for further processing  

7. python blech_units_similarity.py  
    - Check for collisions of spiketimes to assess double-counting of waveforms in clustering  
8. python blech_units_plot.py  
    - Plot waveforms of selected spikes  
9. python blech_units_make_arrays.py  
    - Generate spike-train arrays  
10. python blech_make_psth.py  
    - Plots PSTHs and rasters for all selected units  
11. python blech_palatability_identity_setup.py  
12. python blech_palatability_identity_plot.py  
    - Perform ancillary analyses on spike trains by calculating:  
        - Repsonsive and discriminatory neurons (ANOVA and LDA)  
        - Palatability correlation coefficient  
14. python blech_overlay_psth.py  
    - Plot overlayed PSTHs for units with respective waveforms  

### Operations Workflow Visual 
![blech_clust_outline - Flowchart](https://user-images.githubusercontent.com/12436309/223515611-c4756424-2370-4bf2-8e3f-092e5b91b48e.png)

### Example workflow
```
DIR=/path/to/raw/data/files  
python blech_exp_info.py $DIR  # Generate metadata and electrode layout  
bash blech_clust_pre.sh $DIR   # Perform steps up to spike extraction and UMAP  
python blech_post_process.py   # Add sorted units to HDF5 (CLI or .CSV as input)  
bash blech_clust_post.sh       # Perform steps up to PSTH generation
```

### Test Dataset
We are grateful to Brandeis University Google Filestream for hosting this dataset <br>
Data to test workflow available at:<br>
https://drive.google.com/drive/folders/1ne5SNU3Vxf74tbbWvOYbYOE1mSBkJ3u3?usp=sharing

