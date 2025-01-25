"""
Using a raised cosine basis set to fit PSTHs
"""

from utils.ephys_data import visualize as vz
from utils.ephys_data import ephys_data
from utils import makeRaisedCosBasis as mrcb
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import sys
blech_dir = '/home/abuzarmahmood/Desktop/blech_clust'
sys.path.append(blech_dir)


data_dir = '/media/storage/for_transfer/bla_gc/AM11_4Tastes_191030_114043_copy'
dat = ephys_data.ephys_data(data_dir)
dat.get_spikes()
dat.get_firing_rates()

# vz.firing_overview(dat.all_normalized_firing)
# plt.show()
#
# vz.imshow(dat.all_normalized_firing[10])
# plt.show()

spike_array = np.stack(dat.spikes)
# wanted_dat = spike_array[1,:, 16, 2000:4000]
pre_stim = 500
post_stim = 1500
stim_t = 2000
wanted_dat = spike_array[0, :, 10, stim_t-pre_stim:stim_t+post_stim]

# vz.raster(None, wanted_dat, marker = '|' , color = 'k')
# plt.show()

n = post_stim
n_basis = 20
lin_ctrs, lin_dctrs = mrcb.gen_spread(n, n_basis, spread='linear')
log_ctrs, log_dctrs = mrcb.gen_spread(n, n_basis, spread='log')
sig_ctrs, sig_dctrs = mrcb.gen_spread(
    n, n_basis, spread='sigmoid', a=0.0025, b=750)
stim_linear_basis_funcs = mrcb.gen_raised_cosine_basis(n, lin_ctrs, lin_dctrs)
stim_log_basis_funcs = mrcb.gen_raised_cosine_basis(n, log_ctrs, log_dctrs)
stim_sigmoid_basis_funcs = mrcb.gen_raised_cosine_basis(n, sig_ctrs, sig_dctrs)

pre_stim_basis_params = mrcb.gen_spread(pre_stim, 5, spread='linear')
pre_stim_basis = mrcb.gen_raised_cosine_basis(pre_stim, *pre_stim_basis_params)

full_basis = np.zeros((25-1, pre_stim + post_stim))
linear_basis_funcs = full_basis.copy()
linear_basis_funcs[:5, 0:pre_stim] = pre_stim_basis
linear_basis_funcs[4:, pre_stim:] = stim_linear_basis_funcs

log_basis_funcs = full_basis.copy()
log_basis_funcs[:5, 0:pre_stim] = pre_stim_basis
log_basis_funcs[4:, pre_stim:] = stim_log_basis_funcs

sigmoid_basis_funcs = full_basis.copy()
sigmoid_basis_funcs[:5, 0:pre_stim] = pre_stim_basis
sigmoid_basis_funcs[4:, pre_stim:] = stim_sigmoid_basis_funcs


fig, ax = plt.subplots(4, 1, sharex=True)
img_kwargs = {'aspect': 'auto', 'origin': 'lower', 'interpolation': 'nearest'}
ax[0].imshow(log_basis_funcs, **img_kwargs)
ax[0].set_title('Logarithmically spaced basis functions')
ax[1].imshow(linear_basis_funcs, **img_kwargs)
ax[1].set_title('Linearly spaced basis functions')
ax[2].imshow(sigmoid_basis_funcs, **img_kwargs)
ax[2].set_title('Sigmoidally spaced basis functions')
ax[3] = vz.raster(ax[3], wanted_dat, marker='|', color='k')
ax[3].set_title('Raster plot')
plt.show()

# Perform Linear regression to fit the PSTH with the basis functions
mean_firing = wanted_dat.mean(axis=0)


def fit_basis(basis_funcs, wanted_dat):
    """
    Fit basis functions to data

    args:
        basis_funcs: n_basis x n matrix of basis functions
        wanted_dat: n x 1 vector of data to fit

    returns:
        fit: n x 1 vector of fitted data
    """
    lr = LinearRegression()
    lr.fit(basis_funcs.T, wanted_dat)
    fit = lr.predict(basis_funcs.T)
    return fit


# Linear basis
linear_fit = np.stack([fit_basis(linear_basis_funcs, this_trial)
                      for this_trial in wanted_dat])

# Log basis
log_fit = np.stack([fit_basis(log_basis_funcs, this_trial)
                   for this_trial in wanted_dat])

# Sigmoid basis
sigmoid_fit = np.stack([fit_basis(sigmoid_basis_funcs, this_trial)
                       for this_trial in wanted_dat])

vmin = linear_fit.min()
vmax = linear_fit.max()
fig, ax = plt.subplots(4, 3, sharex=True)
ax[0, 0].plot(linear_basis_funcs.T, color='k')
ax[0, 0].plot(linear_basis_funcs.sum(axis=0), color='r', linewidth=3)
ax[0, 0].set_title('Linearly spaced basis functions')
ax[0, 1].plot(log_basis_funcs.T, color='k')
ax[0, 1].plot(log_basis_funcs.sum(axis=0), color='r', linewidth=3)
ax[0, 1].set_title('Logarithmically spaced basis functions')
ax[1, 0].imshow(linear_fit, **img_kwargs, vmin=vmin, vmax=vmax)
ax[1, 0].set_title('Linear fit')
ax[1, 1].imshow(log_fit, **img_kwargs, vmin=vmin, vmax=vmax)
ax[1, 1].set_title('Log fit')
ax[2, 0] = vz.raster(ax[2, 0], wanted_dat, marker='|', color='k')
ax[2, 0].set_title('Raster plot')
ax[2, 1] = vz.raster(ax[2, 1], wanted_dat, marker='|', color='k')
ax[2, 1].set_title('Linear fit')
ax[0, 2].plot(sigmoid_basis_funcs.T, color='k')
ax[0, 2].plot(sigmoid_basis_funcs.sum(axis=0), color='r', linewidth=3)
ax[0, 2].set_title('Sigmoidally spaced basis functions')
sigmoid_fit = np.stack([fit_basis(sigmoid_basis_funcs, this_trial)
                       for this_trial in wanted_dat])
ax[1, 2].imshow(sigmoid_fit, **img_kwargs, vmin=vmin, vmax=vmax)
ax[1, 2].set_title('Sigmoid fit')
ax[2, 2] = vz.raster(ax[2, 2], wanted_dat, marker='|', color='k')
ax[2, 2].set_title('Sigmoid fit')
ax[3, 0].plot(wanted_dat.sum(axis=0), alpha=0.5)
ax[3, 0].plot(linear_fit.sum(axis=0), alpha=0.5)
ax[3, 0].set_title('Linear fit')
ax[3, 1].plot(wanted_dat.sum(axis=0), alpha=0.5)
ax[3, 1].plot(log_fit.sum(axis=0), alpha=0.5)
ax[3, 1].set_title('Log fit')
ax[3, 2].plot(wanted_dat.sum(axis=0), alpha=0.5)
ax[3, 2].plot(sigmoid_fit.sum(axis=0), alpha=0.5)
ax[3, 2].set_title('Sigmoid fit')
plt.show()
