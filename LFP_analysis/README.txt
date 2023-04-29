BLECH CLUST LFP ANALYSIS README v0.1

Python suite to:
    - Extract LFPs from raw intracranial recordings
    - Bandpass filter recordings into LFP bands
    - Calculate spike-LFP phase locking
    - Run stats and generate plots for spike-LFP phase locking

Order of operation:
===================

1. LFP_Processing_Final.py
    - Extracts LFP (1-300 Hz) from raw signal and saves it in HDF5 file
2. LFP_Finalize_Dataset.py
    - Flags which channels will be used for further analysis
    - Allows removal of dead/problematic channels
3. LFP_Spike_Phase_Calc.py
    - Calculate phases of all spikes from the recording and save in HDF5
4. LFP_Spike_Phase Stats.py
    - Test that spike-phase distribution are not uniform and are different
        between conditions
5. LFP_Spike_Phase_Plotting.py
    - Make plots of the following varieties to display results
        - x
        - y
        - z

** Refer to LFP_wishlist.txt for future additions
