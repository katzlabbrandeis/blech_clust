# Sets up emg data for running the envelope of emg recordings (env.npy) through a local Bayesian Spectrum Analysis (BSA). 
# Needs an installation of R (installing Rstudio on Ubuntu is enough) - in addition, the R library BaSAR needs to be installed from the CRAN archives (https://cran.r-project.org/src/contrib/Archive/BaSAR/)
# This is the starting step for emg_local_BSA_execute.py

# Import stuff
import numpy as np
import easygui
import os
import multiprocessing
import sys
import shutil

# Change to the directory that has the emg data files (env.npy and sig_trials.npy). Make a directory for storing the BSA results
# Get name of directory with the data files
if len(sys.argv) > 1:
    dir_name = os.path.abspath(sys.argv[1])
    if dir_name[-1] != '/':
        dir_name += '/'
else:
    dir_name = easygui.diropenbox(msg = 'Please select data directory')

if 'emg_channel' not in os.path.basename(dir_name[:-1]):
    raise Exception('Please point to <data_dir>/emg_output/emg_channel directory')
    exit()

os.chdir(dir_name)

if os.path.exists('emg_BSA_results'):
    shutil.rmtree('emg_BSA_results')
os.makedirs('emg_BSA_results')

# Load the data files
#env = np.load('./emg_0/env.npy')
#sig_trials = np.load('./emg_0/sig_trials.npy')
env = np.load('./env.npy')
sig_trials = np.load('./sig_trials.npy')

# Ask for the HPC queue to use
# No longer asking for this, just submit to all.q
#queue = easygui.multchoicebox(msg = 'Which HPC queue do you want to use for EMG analysis?', choices = ('neuro.q', 'dk.q'))

# Grab Brandeis unet username
home_dir = os.getenv('HOME')
#username = easygui.multenterbox(
#        msg = 'Enter your Brandeis/Jetstream/personal computer username', 
#        fields = ['username'])

# Dump a shell file for the BSA analysis in the user's blech_clust directory on the desktop
emg_code_dir = os.path.join(home_dir, 'Desktop/blech_clust/emg')
os.chdir(emg_code_dir)
f = open('blech_emg.sh', 'w')
print("module load PYTHON/ANACONDA-2.5.0", file=f)
print("module load R", file=f)
print(f"cd {emg_code_dir}", file=f)
print("python emg_local_BSA_execute.py", file=f)
f.close()

# Dump shell file(s) for running GNU parallel job on the user's blech_clust folder on the desktop
# First get number of CPUs - parallel be asked to run num_cpu-1 threads in parallel
num_cpu = multiprocessing.cpu_count()
# Then produce the file generating the parallel command
f = open('blech_emg_jetstream_parallel.sh', 'w')
print("parallel -k -j {:d} --noswap --load 100% --progress --joblog {:s}/results.log bash blech_emg_jetstream_parallel1.sh ::: {{1..{:d}}}".format(int(num_cpu)-1, dir_name, sig_trials.shape[0]*sig_trials.shape[1]), file = f)
f.close()

# Then produce the file that runs blech_process.py
f = open('blech_emg_jetstream_parallel1.sh', 'w')
print("export OMP_NUM_THREADS=1", file = f)
print("python emg_local_BSA_execute.py $1", file = f)
f.close()
# Finally dump a file with the data directory's location (blech.dir)
f = open('blech.dir', 'w')
print(dir_name, file = f)
f.close()

print("Now logout of the compute node and go back to the login node. Then say: qsub -t 1-"+str(sig_trials.shape[0]*sig_trials.shape[1])+" -q all.q -ckpt reloc blech_emg.sh")
