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
import pandas as pd
from tqdm import tqdm, trange
import os
blech_dir = '/home/abuzarmahmood/Desktop/blech_clust'
sys.path.append(blech_dir)


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


# wanted_dat = spike_array[1,:, 16, 2000:4000]
pre_stim = 500
post_stim = 1500
stim_t = 2000

# vz.raster(None, wanted_dat, marker = '|' , color = 'k')
# plt.show()

n_post_basis = int(post_stim//100)
lin_ctrs, lin_dctrs = mrcb.gen_spread(post_stim, n_post_basis, spread='linear')
log_ctrs, log_dctrs = mrcb.gen_spread(post_stim, n_post_basis, spread='log')
sig_ctrs, sig_dctrs = mrcb.gen_spread(
    post_stim, n_post_basis, spread='sigmoid', a=0.0025, b=750)
stim_linear_basis_funcs = mrcb.gen_raised_cosine_basis(
    post_stim, lin_ctrs, lin_dctrs)
stim_log_basis_funcs = mrcb.gen_raised_cosine_basis(
    post_stim, log_ctrs, log_dctrs)
stim_sigmoid_basis_funcs = mrcb.gen_raised_cosine_basis(
    post_stim, sig_ctrs, sig_dctrs)

n_pre_basis = int(pre_stim//100)
pre_stim_basis_params = mrcb.gen_spread(pre_stim, n_pre_basis, spread='linear')
pre_stim_basis = mrcb.gen_raised_cosine_basis(pre_stim, *pre_stim_basis_params)

n_total_basis = n_pre_basis + n_post_basis
full_basis = np.zeros((n_total_basis-1, pre_stim + post_stim))
linear_basis_funcs = full_basis.copy()
linear_basis_funcs[:n_pre_basis, 0:pre_stim] = pre_stim_basis
linear_basis_funcs[n_pre_basis-1:, pre_stim:] = stim_linear_basis_funcs

log_basis_funcs = full_basis.copy()
log_basis_funcs[:n_pre_basis, 0:pre_stim] = pre_stim_basis
log_basis_funcs[n_pre_basis-1:, pre_stim:] = stim_log_basis_funcs

sigmoid_basis_funcs = full_basis.copy()
sigmoid_basis_funcs[:n_pre_basis, 0:pre_stim] = pre_stim_basis
sigmoid_basis_funcs[n_pre_basis-1:, pre_stim:] = stim_sigmoid_basis_funcs


fig, ax = plt.subplots(3, 1, sharex=True)
img_kwargs = {'aspect': 'auto', 'origin': 'lower', 'interpolation': 'nearest'}
ax[0].imshow(log_basis_funcs, **img_kwargs)
ax[0].set_title('Logarithmically spaced basis functions')
ax[1].imshow(linear_basis_funcs, **img_kwargs)
ax[1].set_title('Linearly spaced basis functions')
ax[2].imshow(sigmoid_basis_funcs, **img_kwargs)
plt.show()


############################################################
############################################################
# Perform Linear regression to fit the PSTH with the basis functions

data_dir_list_path = '/media/storage/for_transfer/bla_gc/data_dir_list.txt'
data_base_dir = os.path.dirname(data_dir_list_path)
plot_dir = os.path.join(data_base_dir, 'plots')
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

artfact_dir = os.path.join(data_base_dir, 'artifacts')
if not os.path.exists(artfact_dir):
    os.makedirs(artfact_dir)

with open(data_dir_list_path, 'r') as file:
    data_dir_list = file.readlines()

dat_frame_list = []

for i, data_dir in enumerate(tqdm(data_dir_list)):

    data_dir = data_dir.rstrip()
    dat = ephys_data.ephys_data(data_dir)
    dat.get_spikes()
    taste_data = np.array(dat.spikes)

    inds = np.array(
        list(np.ndindex((len(taste_data), taste_data[0].shape[1]))))
    taste_nrn_list = [taste_data[i, :, j] for i, j in inds]

    this_frame = pd.DataFrame(inds, columns=['taste', 'neuron'])
    this_frame['spikes'] = taste_nrn_list
    this_frame['session'] = i
    dat_frame_list.append(this_frame)

    # wanted_dat = spike_array[0, :, 10, stim_t-pre_stim:stim_t+post_stim]

dat_frame = pd.concat(dat_frame_list, ignore_index=True)

basis_psth_list = []

for ind, this_row in tqdm(dat_frame.iterrows()):
    # save_path = os.path.join(
    #         plot_dir,
    #         f'{this_row["session"]}_{this_row["taste"]}_{this_row["neuron"]}.png')

    if os.path.exists(save_path):
        continue

    wanted_dat = this_row['spikes']
    wanted_dat = wanted_dat[:, stim_t-pre_stim:stim_t+post_stim]

    # Linear basis
    linear_fit = np.stack([fit_basis(linear_basis_funcs, this_trial)
                          for this_trial in wanted_dat])

    # Log basis
    log_fit = np.stack([fit_basis(log_basis_funcs, this_trial)
                       for this_trial in wanted_dat])

    # Sigmoid basis
    sigmoid_fit = np.stack([fit_basis(sigmoid_basis_funcs, this_trial)
                           for this_trial in wanted_dat])

    basis_psth_list.append([linear_fit, log_fit, sigmoid_fit])

    # vmin = linear_fit.min()
    # vmax = linear_fit.max()
    # fig, ax = plt.subplots(4, 3, sharex=True, figsize=(15, 10))
    # ax[0, 0].plot(linear_basis_funcs.T, color='k')
    # ax[0, 0].plot(linear_basis_funcs.sum(axis=0), color='r', linewidth=3)
    # ax[0, 0].set_title('Linearly spaced basis functions')
    # ax[0, 1].plot(log_basis_funcs.T, color='k')
    # ax[0, 1].plot(log_basis_funcs.sum(axis=0), color='r', linewidth=3)
    # ax[0, 1].set_title('Logarithmically spaced basis functions')
    # ax[1, 0].imshow(linear_fit, **img_kwargs, vmin=vmin, vmax=vmax)
    # ax[1, 0].set_title('Linear fit')
    # ax[1, 1].imshow(log_fit, **img_kwargs, vmin=vmin, vmax=vmax)
    # ax[1, 1].set_title('Log fit')
    # ax[2, 0] = vz.raster(ax[2, 0], wanted_dat, marker='|', color='k')
    # ax[2, 0].set_title('Raster plot')
    # ax[2, 1] = vz.raster(ax[2, 1], wanted_dat, marker='|', color='k')
    # ax[2, 1].set_title('Linear fit')
    # ax[0, 2].plot(sigmoid_basis_funcs.T, color='k')
    # ax[0, 2].plot(sigmoid_basis_funcs.sum(axis=0), color='r', linewidth=3)
    # ax[0, 2].set_title('Sigmoidally spaced basis functions')
    # sigmoid_fit = np.stack([fit_basis(sigmoid_basis_funcs, this_trial)
    #                        for this_trial in wanted_dat])
    # ax[1, 2].imshow(sigmoid_fit, **img_kwargs, vmin=vmin, vmax=vmax)
    # ax[1, 2].set_title('Sigmoid fit')
    # ax[2, 2] = vz.raster(ax[2, 2], wanted_dat, marker='|', color='k')
    # ax[2, 2].set_title('Sigmoid fit')
    # ax[3, 0].plot(wanted_dat.sum(axis=0), alpha=0.5)
    # ax[3, 0].plot(linear_fit.sum(axis=0), alpha=0.5)
    # ax[3, 0].set_title('Linear fit')
    # ax[3, 1].plot(wanted_dat.sum(axis=0), alpha=0.5)
    # ax[3, 1].plot(log_fit.sum(axis=0), alpha=0.5)
    # ax[3, 1].set_title('Log fit')
    # ax[3, 2].plot(wanted_dat.sum(axis=0), alpha=0.5)
    # ax[3, 2].plot(sigmoid_fit.sum(axis=0), alpha=0.5)
    # ax[3, 2].set_title('Sigmoid fit')
    # for this_ax in ax[1:].flatten():
    #     this_ax.axvline(pre_stim, color='r', linestyle='--')
    # plt.savefig(save_path)
    # # plt.show()

dat_frame['linear_fit'] = [this_fit[0] for this_fit in basis_psth_list]
dat_frame['log_fit'] = [this_fit[1] for this_fit in basis_psth_list]
dat_frame['sigmoid_fit'] = [this_fit[2] for this_fit in basis_psth_list]


##############################
# For each session, and neuron, plot all 4 tastes

kern_len = 250
conv_kernel = np.ones(kern_len)/kern_len


def calc_conv_psth(this_array):
    """
    Calculate the convolution rate of the input array

    args:
        this_array: trials x time array

    returns:
        conv_psth: time
    """
    mean_spikes = this_array.mean(axis=0)
    x = np.arange(len(mean_spikes))
    conv_psth = np.convolve(mean_spikes, conv_kernel, mode='valid')
    conv_x = np.convolve(x, conv_kernel, mode='valid')
    return conv_psth, conv_x


conv_psth_list = [
    calc_conv_psth(this_row['spikes']) for ind, this_row in dat_frame.iterrows()]
conv_x = conv_psth_list[0][1]

dat_frame['conv_psth'] = [this_psth[0] for this_psth in conv_psth_list]

dat_frame.to_pickle(os.path.join(data_base_dir, 'basis_psth_data.pkl'))

this_plot_dir = os.path.join(data_base_dir, 'taste_plots')
if not os.path.exists(this_plot_dir):
    os.makedirs(this_plot_dir)

wanted_conv_x = conv_x[stim_t-pre_stim:stim_t+post_stim] - stim_t
basis_x = np.arange(pre_stim + post_stim) - pre_stim
for (session, neuron), this_frame in tqdm(dat_frame.groupby(['session', 'neuron'])):
    save_path = os.path.join(
        this_plot_dir,
        f'{session}_{neuron}.png')
    if os.path.exists(save_path):
        continue

    plot_dat = [
        this_frame.conv_psth.to_numpy(),
        this_frame.linear_fit.to_numpy(),
        this_frame.log_fit.to_numpy(),
        this_frame.sigmoid_fit.to_numpy(),
    ]
    plot_dat = [np.stack(this_dat) for this_dat in plot_dat]
    dat_names = ['Convolved PSTH', 'Linear fit', 'Log fit', 'Sigmoid fit']
    fig, ax = plt.subplots(5, 1, sharex=True, figsize=(10, 20))
    for i, this_dat in enumerate(plot_dat):
        this_dat_name = dat_names[i]
        if this_dat_name == 'Convolved PSTH':
            ax[i].plot(wanted_conv_x, this_dat[:, stim_t -
                       pre_stim:stim_t+post_stim].T)
        else:
            ax[i].plot(basis_x, this_dat.mean(axis=1).T)
        ax[i].set_title(this_dat_name)
    this_spikes = np.stack(this_frame.spikes.to_numpy())
    n_trials = this_spikes.shape[1]
    n_tastes = this_spikes.shape[0]
    taste_inds = np.arange(n_tastes * n_trials) // n_trials
    cat_spikes = np.concatenate(this_spikes, axis=0)
    cat_spikes = cat_spikes[:, stim_t-pre_stim:stim_t+post_stim]
    spike_inds = np.where(cat_spikes)
    ax[-1].scatter(spike_inds[1]-pre_stim, spike_inds[0], marker='|',
                   color=[f'C{i}' for i in taste_inds[spike_inds[0]]])
    for this_ax in ax:
        this_ax.axvline(0, color='r', linestyle='--')
    plt.savefig(save_path)
    plt.close(fig)
