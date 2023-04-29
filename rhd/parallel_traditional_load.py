import sys
import os
sys.path.append('/home/abuzarmahmood/Desktop/blech_clust/rhd')
#os.chdir('/home/abuzarmahmood/Desktop/blech_clust/rhd')
from load_intan_test import *
import rhd

import numpy as np
import jax.numpy as jnp
from glob import glob
import tables
#from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm, trange
import time
import psutil
import multiprocessing as mp
import pylab as plt

from sklearn.cluster import KMeans


file_path = '/media/bigdata/Abuzar_Data/trad_intan_test2'
hdf5_path = os.path.join(file_path, 'test.h5')

file_list = sorted(glob(os.path.join(file_path,'*.rhd')))

# Get size of all files
# Or simpler yet, get size of first and last file and number of files
# All files except the last one will have the same size
first_size = get_file_info(file_list[0])
last_size = get_file_info(file_list[-1])

full_size = first_size*(len(file_list)-1) + last_size

test = rhd.read_data(file_list[-1], no_floats = True)
test_data = test['amplifier_data']

atom = tables.Int16Atom()
hf5 = tables.open_file(hdf5_path, 'w')
hf5.close()
hf5 = tables.open_file(hdf5_path, 'r+')

amp_array = hf5.create_array('/','amp_data', atom = atom, \
        shape = (test_data.shape[0], full_size))
hf5.close()


# Using pool to handle number of processes at any given timepoint
# This will prevent memory overflow errors
array_path = '/amp_data'
# For now, ignore the last file
block_len = first_size

def write_to_file(hdf5_path, array_path, data, block_len, ind):
    """
    Use -1 for final ind
    """
    lock.acquire()
    print(f"Writing {ind} now")
    with tables.open_file(hdf5_path, 'r+') as hf5:
        array = hf5.get_node(\
                os.path.dirname(array_path),os.path.basename(array_path))
        if ind == -1:
            array[:,-data.shape[1]:] = data
        else:
            array[:,ind*block_len : (ind+1)*block_len] = data
        hf5.flush()
    lock.release()

def load_and_write(num,
                    file_list,
                    hdf5_path,
                    array_path,
                    block_len):
    loaded_dict = rhd.read_data(file_list[num], no_floats = True)
    loaded_dat = loaded_dict['amplifier_data']
    referenced_dat = loaded_dat - np.mean(loaded_dat,axis=0)
    #write_to_file(hdf5_path, array_path, loaded_dat, block_len, num)
    write_to_file(hdf5_path, array_path, referenced_dat, block_len, num)

def load_and_write_par(num):
    load_and_write(num,
                        file_list,
                        hdf5_path,
                        array_path,
                        block_len)

def init(l):
    global lock
    lock = l


# Calculate maximum number of processes that should be allowed
# Check free memory, and size of single binary files
available_memory = psutil.virtual_memory()[1]
# Check size of files
bin_file_size = os.path.getsize(file_list[0])

max_processes_possible = \
        int(np.floor((available_memory//bin_file_size)/2))
max_processes = \
        np.min((len(file_list), mp.cpu_count(), max_processes_possible))

start_t = time.time()
iterable = list(range(len(file_list)-1))
iterable.append(-1)
l = mp.Lock()
pool = mp.Pool(max_processes, initializer=init, initargs=(l,))
pool.map(load_and_write_par, iterable)
pool.close()
pool.join()
end_t = time.time()
print(end_t - start_t)

# Plot some data
with tables.open_file(hdf5_path, 'r') as hf5:
    full_test = hf5.root.amp_data[:,:]
    test = hf5.root.amp_data[0,:]
full_test_down = full_test[:,::100]

from scipy.stats import zscore
zscore_full = zscore(full_test_down,axis=0)

plt.imshow(zscore_full,aspect='auto')
plt.show()

plt.plot(test[::100])
# Plot vlines at block_lens to check if there are breaks
min_val,max_val = np.min(test), np.max(test)
plt.vlines(block_len*np.arange(len(iterable))//100, min_val, max_val)
plt.show()

### Create CAR groups using kmeans
from sklearn.metrics import pairwise_distances as distmat
num_clusters = 4
max_points = 1000
selected_inds = np.random.choice(np.arange(zscore_full.shape[-1]),
                                max_points, replace = False)
dist_sample = zscore_full[:,selected_inds]
channel_dists = distmat(dist_sample)

kmeans = KMeans(n_clusters=num_clusters).fit(channel_dists)
cluster_labels = kmeans.labels_

sorted_inds = np.argsort(cluster_labels)
sorted_dists = channel_dists[sorted_inds,:]
sorted_dists = sorted_dists[:,sorted_inds]

fig,ax = plt.subplots(1,2)
ax[0].imshow(channel_dists)
ax[1].imshow(sorted_dists)
plt.show()
