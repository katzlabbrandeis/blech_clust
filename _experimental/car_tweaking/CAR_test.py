# Import stuff!
import tables
import numpy as np
import os
import easygui
import sys
from tqdm import tqdm, trange
import glob
import json
from utils.blech_utils import imp_metadata
import xgboost as xgb
from sklearn.linear_model import LinearRegression as LR
import pylab as plt
from scipy.stats import pearsonr, zscore
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA as IPCA
from sklearn.cluster import KMeans
import time

# import standard scaler
from sklearn.preprocessing import StandardScaler as Scaler

# Import sklearn train test split
from sklearn.model_selection import train_test_split

# Import sklearn regression metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def return_subset(X, batch_size):
    """
    Return a random subset of the data
    """
    idx = np.random.choice(X.shape[0], batch_size, replace=False)
    return X[idx]

def truncated_ipca(
        X, 
        n_components = 10, 
        batch_size = None, 
        n_iter = 1000,
        tolerance = 1e-3,
        ):
    """
    Truncated IPCA
    """
    if batch_size is None and n_iter is not None:
        batch_size = X.shape[0] // n_iter

    delta_list = []
    components_list = []
    ipca = IPCA(n_components=n_components, batch_size=batch_size)
    ipca.partial_fit(return_subset(X, batch_size))
    components_list.append(ipca.components_)
    for i in range(n_iter):
        print(f'Iteration : {i}')
        ipca.partial_fit(return_subset(X, batch_size))
        delta = np.mean(np.abs(components_list[-1] - ipca.components_))
        delta_list.append(delta)
        components_list.append(ipca.components_)
        if delta <= tolerance:
            break
    return ipca, delta_list

############################################################
############################################################
# Get name of directory with the data files
#dir_name = open('blech.dir','r').readlines()[0].strip()
dir_name = '/media/storage/gc_only/AS18/AS18_4Tastes_200228_151511' 
metadata_handler = imp_metadata([[],dir_name])
dir_name = metadata_handler.dir_name
os.chdir(dir_name)
print(f'Processing : {dir_name}')

# Open the hdf5 file
hf5 = tables.open_file(metadata_handler.hdf5_name, 'r+')

# Read CAR groups from info file
# Every region is a separate group, multiple ports under single region is a separate group,
# emg is a separate group
info_dict = metadata_handler.info_dict

# Since electrodes are already in monotonic numbers (A : 0-31, B: 32-63)
# we can directly pull them
all_car_group_vals = []
all_car_group_names = []
for region_name, region_elecs in info_dict['electrode_layout'].items():
    for group in region_elecs:
        if len(group) > 0:
            all_car_group_vals.append(group)
            all_car_group_names.append(region_name)

# Select car groups which are not emg
all_car_group_vals, all_car_group_names = list(zip(*[
        [x,y] for x,y in zip(all_car_group_vals, all_car_group_names) \
                if (('emg' not in y) and ('none'!=y))
        ]))
num_groups = len(all_car_group_vals)

CAR_electrodes = all_car_group_vals
print(f" Number of groups : {len(CAR_electrodes)}")
for region,vals in zip(all_car_group_names, all_car_group_vals):
    print(f" {region} :: {vals}")

# Pull out the raw electrode nodes of the HDF5 file
raw_electrodes = hf5.list_nodes('/raw')
# Sort electrodes (just in case) so we can index them directly
sort_order = np.argsort([x.__str__() for x in raw_electrodes])
raw_electrodes = [raw_electrodes[i] for i in sort_order]
raw_electrodes_map = {
        int(str.split(electrode._v_pathname, 'electrode')[-1]):num \
                for num, electrode in enumerate(raw_electrodes)}

wanted_electrode_inds = [raw_electrodes_map[x] for x in CAR_electrodes[0]]
wanted_electrodes = np.stack([raw_electrodes[x][:] for x in wanted_electrode_inds])
# Kill 2 channels
wanted_electrodes[:2,:] = 0
down_rate = 100
down_electrodes = [x[::down_rate] for x in wanted_electrodes]
down_stack = np.stack(down_electrodes, axis=0)

##############################
# Plot correlations between channels
from scipy.stats import pearsonr
from itertools import combinations

def intra_corr(X):
    inds = list(combinations(range(X.shape[0]), 2))
    corr_mat = np.zeros((X.shape[0], X.shape[0]))
    for i,j in inds:
        corr_mat[i,j] = pearsonr(X[i,:], X[j,:])[0]
        corr_mat[j,i] = corr_mat[i,j]
    return corr_mat

corr_mat = intra_corr(down_stack)
# nan to 0
corr_mat[np.isnan(corr_mat)] = 0
# Cluster the channels using KMeans
n_clusters = np.arange(3,7)

def return_sorted_corr_mat(corr_mat, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(corr_mat)
    # Sort cluster by mean correlation
    centers = kmeans.cluster_centers_
    mean_cluster_centers = np.mean(centers, axis=1)
    label_map = dict(zip(np.arange(0,n_clusters), np.argsort(mean_cluster_centers)))
    fin_labels = [label_map[x] for x in kmeans.labels_]
    # Sort by cluster
    sort_order = np.argsort(fin_labels)
    # Sort the correlation matrix and labels
    corr_mat_sorted = corr_mat[sort_order,:][:,sort_order]
    kmeans_sorted = kmeans.labels_[sort_order]
    plot_cluster_inds = np.where(np.diff(kmeans_sorted))[0] + 1
    return corr_mat_sorted, plot_cluster_inds

corr_mat_sorted, plot_cluster_inds = \
        list(zip(*[return_sorted_corr_mat(corr_mat, x) for x in n_clusters]))

# Plot original and sorted correlation matrices
fig, ax = plt.subplots(1,1+len(n_clusters), figsize=(20,5))
ax[0].imshow(corr_mat, cmap='viridis', vmin=0, vmax=1)
ax[0].set_title('Original')
for i, (x, y) in enumerate(zip(corr_mat_sorted, n_clusters)):
    im = ax[i+1].imshow(x, cmap='viridis', vmin=0, vmax=1)
    ax[i+1].set_title(f'Sorted, {y} clusters')
    for ind in plot_cluster_inds[i]:
        ax[i+1].axvline(ind, color='r')
        ax[i+1].axhline(ind, color='r')
cax = fig.add_axes([0.95, 0.15, 0.01, 0.7])
fig.colorbar(im, cax=cax)
plt.show()

plt.hist(corr_mat.flatten(), bins=100)
plt.xlim([0,1])
plt.show()

##############################
# Calculate noise at different steps of processing
calc_rmnoise = lambda x: np.sqrt(np.mean(x**2, axis = -1))
print_stats = lambda x: f'{np.mean(x):.2f} +/- {np.std(x):.2f}'

# Downsample the data to 1kHz
raw_rmnoise = calc_rmnoise(down_stack)
print('Raw noise +/- std : ' + print_stats(raw_rmnoise))

down_car = np.mean(down_stack, axis=0)
car_stack = down_stack - down_car[np.newaxis,:]
car_rmnoise = calc_rmnoise(car_stack)
print('CAR noise +/- std : ' + print_stats(car_rmnoise)) 

##############################
# Subtract principal components
pca = PCA(n_components=10)
pca.fit(down_stack.T)

#plt.plot(np.cumsum(pca.explained_variance_ratio_), 'o-')
#plt.show()

pca_car = pca.transform(down_stack.T).T
pca_inv = pca.inverse_transform(pca_car.T).T

## Plot n-steps of both down_stack and pca_inv
n_steps = 10000
#fig,ax = plt.subplots(len(pca_inv),1,sharex=True, sharey=True)
#for i in range(len(pca_inv)):
#    ax[i].plot(down_stack[i,:n_steps], label='raw')
#    ax[i].plot(pca_inv[i,:n_steps], label='pca_inv')
#ax[-1].legend()
#plt.show()

# Subtract pca_inv from down_stack
pca_sub_car = down_stack - pca_inv
pca_sub_car_rmnoise = calc_rmnoise(pca_sub_car)
print('PCA noise +/- std : ' + print_stats(pca_sub_car_rmnoise))

##############################
# Make sure that PCA generated from downsampled data is
# still useful for the full data
cut_raw_dat = wanted_electrodes[:,:(wanted_electrodes.shape[1]//5)]
# Zscore the data
cut_raw_dat = zscore(cut_raw_dat, axis=-1)
# Set nan values to 0
cut_raw_dat[np.isnan(cut_raw_dat)] = 0
cut_car_dat = zscore(cut_raw_dat - np.mean(cut_raw_dat, axis=0), axis=-1)

#cut_down_pca_dat = \
#        cut_raw_dat - pca.inverse_transform(pca.transform(cut_raw_dat.T)).T

# Keep track of time taken
#pca_time = []
ipca_time = []
trunc_ipca_time = []
#pca_time.append(time.time())
#cut_pca = PCA(n_components=10).fit(cut_raw_dat.T)
#pca_time.append(time.time())
ipca_time.append(time.time())
cut_ipca = IPCA(n_components=10).fit(cut_raw_dat.T)
ipca_time.append(time.time())
trunc_ipca_time.append(time.time())
cut_trunc_ipca, delta_list = truncated_ipca(
        X = cut_raw_dat.T,
        n_components=3,
        n_iter=1000,
        tolerance = 1e-2
        )
trunc_ipca_time.append(time.time())

#cut_pca_dat = cut_raw_dat - cut_pca.inverse_transform(
#        cut_pca.transform(cut_raw_dat.T)).T
cut_ipca_dat = cut_raw_dat - cut_ipca.inverse_transform(
        cut_ipca.transform(cut_raw_dat.T)).T
cut_ipca_dat = zscore(cut_ipca_dat, axis=-1)
cut_trunc_ipca_dat = cut_raw_dat - cut_trunc_ipca.inverse_transform(
        cut_trunc_ipca.transform(cut_raw_dat.T)).T
cut_trunc_ipca_dat = zscore(cut_trunc_ipca_dat, axis=-1)

from matplotlib import colors
vmin,vmax = np.min(cut_trunc_ipca.components_), np.max(cut_trunc_ipca.components_)
divnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0., vmax=vmax)
im = plt.matshow(cut_trunc_ipca.components_, cmap='bwr', norm=divnorm)
plt.colorbar(im)
plt.show()

#print(f"PCA time : {pca_time[1] - pca_time[0]:.2f}")
print(f"IPCA time : {ipca_time[1] - ipca_time[0]:.2f}")
print(f"Truncated IPCA time : {trunc_ipca_time[1] - trunc_ipca_time[0]:.2f}")

cut_raw_rmnoise = calc_rmnoise(cut_raw_dat)
cut_car_rmnoise = calc_rmnoise(cut_car_dat)
#cut_pca_rmnoise = calc_rmnoise(cut_pca_dat)
#cut_down_pca_rmnoise = calc_rmnoise(cut_down_pca_dat)
cut_ipca_rmnoise = calc_rmnoise(cut_ipca_dat)
cut_trunc_ipca_rmnoise = calc_rmnoise(cut_trunc_ipca_dat)

print('Cut raw noise +/- std : ' + print_stats(cut_raw_rmnoise))
print('Cut CAR noise +/- std : ' + print_stats(cut_car_rmnoise))
#print('Cut PCA noise +/- std : ' + print_stats(cut_pca_rmnoise))
#print('Cut down PCA noise +/- std : ' + print_stats(cut_down_pca_rmnoise))
print('Cut IPCA noise +/- std : ' + print_stats(cut_ipca_rmnoise))
print('Cut truncated IPCA noise +/- std : ' + print_stats(cut_trunc_ipca_rmnoise))

down_plot_rate = 100
fig, ax = plt.subplots(4,1,sharex=True, sharey=True)
ax[0].plot(cut_raw_dat[:4,::down_plot_rate].T, alpha=0.3)
ax[1].plot(cut_car_dat[:4,::down_plot_rate].T, alpha=0.3)
ax[2].plot(cut_ipca_dat[:4,::down_plot_rate].T, alpha=0.3)
ax[3].plot(cut_trunc_ipca_dat[:4,::down_plot_rate].T, alpha=0.3)
ax[0].set_title('Raw, RMS = ' + print_stats(cut_raw_rmnoise))
ax[1].set_title('CAR, RMS = ' + print_stats(cut_car_rmnoise))
ax[2].set_title('IPCA, RMS = ' + print_stats(cut_ipca_rmnoise) + \
        ', Time : ' + f"{ipca_time[1] - ipca_time[0]:.2f} s")
ax[3].set_title('Trunc IPCA, RMS = ' + print_stats(cut_trunc_ipca_rmnoise) + \
        ', Time : ' + f"{trunc_ipca_time[1] - trunc_ipca_time[0]:.2f} s")
plt.show()

##############################
# Scale car_stack
scaler = Scaler()
scaler.fit(pca_sub_car.T)
scaled_pca_sub_car = scaler.transform(car_stack.T).T

# Perform noise subtraction using XGBoost 
XGBsub_car = np.zeros(pca_sub_car.shape)
for i in trange(pca_sub_car.shape[0]):
    X = np.stack(
            [pca_sub_car[j,:] for j in range(car_stack.shape[0]) if j!=i],
            axis=0).T
    y = pca_sub_car[i,:]
    reg = xgb.XGBRegressor(objective ='reg:squarederror', 
            n_estimators = 50, max_depth = 5, seed = 123)
    reg.fit(X, y)
    y_pred = reg.predict(X)
    XGBsub_car[i,:] = y - y_pred

XGBsub_car_rmnoise = calc_rmnoise(XGBsub_car)
print('XGBsub noise +/- std : ' + print_stats(XGBsub_car_rmnoise))

# Plot overlay of pca_sub_car vs pca_sub_car vs XGBsub_car
fig,axis = plt.subplots(4,1,sharex=True, sharey=True)
axis[0].plot(down_stack.T[:n_steps,:4], alpha=0.3)
axis[0].set_title('Downsampled Data, mean noise +/- std : ' + print_stats(raw_rmnoise))
axis[1].plot(car_stack.T[:n_steps,:4], alpha=0.3)
axis[1].set_title('CAR, mean noise +/- std : ' + print_stats(car_rmnoise))
axis[2].plot(pca_sub_car.T[:n_steps,:4], alpha=0.3)
axis[2].set_title('PCA subtracted CAR, mean noise +/- std : ' + print_stats(pca_sub_car_rmnoise))
axis[3].plot(XGBsub_car.T[:n_steps,:4], alpha=0.3)
axis[3].set_title('XGB subtracted CAR, mean noise +/- std : ' + print_stats(XGBsub_car_rmnoise))
plt.show()

