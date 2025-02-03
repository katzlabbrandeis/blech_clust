"""
Using a raised cosine basis set to fit PSTHs
"""

from joblib import Parallel, delayed
from utils.ephys_data import visualize as vz
from utils.ephys_data import ephys_data
from utils import makeRaisedCosBasis as mrcb
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
from tqdm import tqdm, trange
from scipy.optimize import curve_fit
import os
blech_dir = '/home/abuzarmahmood/Desktop/blech_clust'
sys.path.append(blech_dir)


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

kern_len = 250
conv_kernel = np.ones(kern_len)/kern_len

basis_data_path = os.path.join(data_base_dir, 'basis_psth_data.pkl')
if not os.path.exists(basis_data_path):

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

        # if os.path.exists(save_path):
        #     continue

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

    conv_psth_list = [
        calc_conv_psth(this_row['spikes']) for ind, this_row in dat_frame.iterrows()]

    dat_frame['conv_psth'] = [this_psth[0] for this_psth in conv_psth_list]

    dat_frame.to_pickle(basis_data_path)
else:
    dat_frame = pd.read_pickle(basis_data_path)

conv_x = calc_conv_psth(dat_frame.iloc[0]['spikes'])[1]

##############################
# Get session = 9, unit = 10
this_frame = dat_frame[(dat_frame['session'] == 9) &
                       (dat_frame['neuron'] == 10)]

spikes = np.stack(this_frame.spikes.to_numpy())
cat_spikes = np.concatenate(spikes, axis=0)

vz.raster(None, cat_spikes, marker='|', color='k')
plt.show()


def calc_isi_array(cat_spikes):
    """
    Calculate the inter-spike interval array

    args:
        cat_spikes: concatenated spikes array

    returns:
        isi_array: inter-spike interval array
    """
    spike_inds = [np.where(this_trial)[0] for this_trial in cat_spikes]
    isi_list = [np.diff(this_trial) for this_trial in spike_inds]

    isi_array = np.zeros(cat_spikes.shape)
    for trial_ind, (trial_spikes, trial_isis) in enumerate(zip(spike_inds, isi_list)):
        for i, this_isi in enumerate(trial_isis):
            isi_array[trial_ind][trial_spikes[i]:trial_spikes[i+1]] = this_isi
    return isi_array


isi_array = calc_isi_array(cat_spikes)

# # Calculate inter-spike intervals
# spike_inds = [np.where(this_trial)[0] for this_trial in cat_spikes]
# isi_list = [np.diff(this_trial) for this_trial in spike_inds]
#
# isi_array = np.zeros(cat_spikes.shape)
# for trial_ind, (trial_spikes, trial_isis) in enumerate(zip(spike_inds, isi_list)):
#     for i, this_isi in enumerate(trial_isis):
#         isi_array[trial_ind][trial_spikes[i]:trial_spikes[i+1]] = this_isi
#
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].imshow(isi_array, aspect='auto', origin='lower', interpolation='nearest')
ax[1].plot(isi_array.mean(axis=0))
plt.show()

pre_mean_isi = isi_array.mean(axis=0)[stim_t-pre_stim:stim_t].mean()


def sigmoid(x, a, b):
    return 1 / (1 + np.exp(-a*(x-b)))


# Fit sigmoid to post
post_isi = isi_array.mean(axis=0)[stim_t:stim_t+post_stim]
post_isi = post_isi - post_isi.min()
post_isi = post_isi / post_isi.max()

post_x = np.arange(len(post_isi))
popt, pcov = curve_fit(sigmoid, post_x, post_isi, p0=[0.025, 500])

plt.plot(post_x, post_isi)
plt.plot(post_x, sigmoid(post_x, *popt))
plt.show()

# sum_isi = np.zeros(cat_spikes.shape[1])
# count_isi = np.zeros(cat_spikes.shape[1])
#
# for trial_inds, trial_isis in zip(spike_inds, isi_list):
#     mean_isi[trial_inds[:-1]] += trial_isis
#     count_isi[trial_inds[:-1]] += 1
#
# mean_isi = mean_isi / count_isi
#
# plt.plot(mean_isi, 'x')
# plt.show()
#
# # Median filter
# kern_len = 10
# med_isi = np.array([np.nanmedian(mean_isi[max(0, i-kern_len):min(len(mean_isi), i+kern_len)]) \
#         for i in range(len(mean_isi))])
#
#
# plt.plot(mean_isi, 'x')
# plt.plot(med_isi, 'x')
# plt.show()

##############################
# Example regression

y = np.exp((basis_x + 500) / 1000)
y = y / y.max()
y_sigmoid_fit = fit_basis(sigmoid_basis_funcs, y)

plt.plot(y, label='True')
plt.plot(y_sigmoid_fit, label='Fit')
plt.legend()
plt.show()

##############################
# Convolve with half-gaussian kernel with std = proportional to mean ISI
isi_mean = isi_array.mean(axis=0)

plt.plot(isi_mean)
plt.show()


def half_gaussian(x, mu, sigma):
    return np.exp(-(x-mu)**2/(2*sigma**2)) * (x < mu)


def gen_half_gaussian_kern_array(isi_mean, isi_sigma_ratio=1):
    """
    Generate a kernel array for half-gaussian convolution

    args:
        isi_mean: mean ISI array
        isi_sigma_ratio: ratio of sigma to mean ISI (higher = wider kernel)

    returns:
        kern_array: kernel array
    """
    kern_array = np.zeros((len(isi_mean), len(isi_mean)))
    for i, this_isi in enumerate(isi_mean):
        this_kern = half_gaussian(
            np.arange(len(isi_mean)), i, this_isi*isi_sigma_ratio)
        # Normalize
        this_kern /= this_kern.sum()
        kern_array[i] = this_kern
    return kern_array


kern_array = gen_half_gaussian_kern_array(isi_mean, isi_sigma_ratio=1)


def calc_half_gauss_rates(spike_array):
    """
    Calculate the half-gaussian convolved rates

    args:
        cat_spikes: concatenated spikes array
        kern_array: kernel array

    returns:
        half_gauss_rates: convolved rates array
    """
    isi_array = calc_isi_array(spike_array)
    mean_isi = isi_array.mean(axis=0)
    # Generate kernel array
    kern_array = gen_half_gaussian_kern_array(mean_isi, isi_sigma_ratio=1)

    half_gauss_rates = spike_array @ (kern_array.T)
    # Fill nans with 0
    half_gauss_rates = np.nan_to_num(half_gauss_rates)
    return half_gauss_rates

# rates = calc_half_gauss_rates(spike_array)
#
# fig, ax = plt.subplots(2, 1, sharex=True)
# ax[0] = vz.raster(ax[0], spike_array, marker='|', color='k')
# ax[1].imshow(rates, aspect='auto', origin='lower', vmax = 0.1)
# plt.show()
#
# plt.imshow(kern_array, aspect='equal', origin='lower')
# plt.show()


half_gauss_rates = cat_spikes @ (kern_array.T)
# Fill nans with 0
half_gauss_rates = np.nan_to_num(half_gauss_rates)

plt.imshow(half_gauss_rates, aspect='auto', origin='lower')
plt.show()

fig, ax = plt.subplots(3, 1, sharex=True)
ax[0] = vz.raster(ax[0], cat_spikes, marker='|', color='k')
ax[1].imshow(half_gauss_rates, aspect='auto', origin='lower', vmax=0.1)
ax[2].plot(half_gauss_rates.T, alpha=0.05, color='k')
ax[2].set_ylim(0, 0.1)
ax[2].set_xlim(1500, 4000)
plt.show()

taste_half_gauss_rates = np.tensordot(spikes, kern_array, axes=([-1], [1]))
mean_taste_half_gauss_rates = taste_half_gauss_rates.mean(axis=1)

sigmoid_fit = np.stack(this_frame.sigmoid_fit.to_numpy())
mean_sigmoid_fit = sigmoid_fit.mean(axis=1)

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(mean_sigmoid_fit.T)
ax[1].plot(mean_taste_half_gauss_rates.T[stim_t - pre_stim:stim_t+post_stim])
plt.show()


############################################################
############################################################


def parallelize(func, iterable, n_jobs=-1):
    """
    Parallelize a function over an iterable

    args:
        func: function to parallelize
        iterable: iterable to apply function to
        n_jobs: number of jobs to run in parallel

    returns:
        results: list of results
    """
    results = Parallel(n_jobs=n_jobs)(delayed(func)(i) for i in tqdm(iterable))
    return results


# Add half-gaussian convolution to the dat_frame
spikes_list = dat_frame.spikes.to_list()
# Cut down to just the relevant part
half_gauss_rates_list = parallelize(calc_half_gauss_rates, spikes_list)
half_gauss_rates_list = [this_rates[:, stim_t-pre_stim:stim_t+post_stim]
                         for this_rates in half_gauss_rates_list]

# for ind, this_row in tqdm(dat_frame.iterrows()):
#     wanted_dat = this_row['spikes']
#     half_gauss_rates = calc_half_gauss_rates(wanted_dat)
#     half_gauss_rates_list.append(half_gauss_rates)
#
dat_frame['half_gauss_rates'] = half_gauss_rates_list

# dat_frame.to_pickle(basis_data_path)
# dat_frame.to_hdf(basis_data_path.replace('.pkl', '.h5'), key='df')

############################################################
############################################################

this_plot_dir = os.path.join(data_base_dir, 'taste_plots')
if not os.path.exists(this_plot_dir):
    os.makedirs(this_plot_dir)

wanted_conv_x = conv_x[stim_t-pre_stim:stim_t+post_stim] - stim_t
basis_x = np.arange(pre_stim + post_stim) - pre_stim
for (session, neuron), this_frame in tqdm(dat_frame.groupby(['session', 'neuron'])):
    save_path = os.path.join(
        this_plot_dir,
        f'{session}_{neuron}.png')
    # if os.path.exists(save_path):
    #     continue

    plot_dat = [
        this_frame.conv_psth.to_numpy(),
        this_frame.linear_fit.to_numpy(),
        this_frame.log_fit.to_numpy(),
        this_frame.sigmoid_fit.to_numpy(),
        this_frame.half_gauss_rates.to_numpy()
    ]
    plot_dat = [np.stack(this_dat) for this_dat in plot_dat]
    dat_names = ['Convolved PSTH', 'Linear fit', 'Log fit',
                 'Sigmoid fit', 'ISI-modulated Half-Gaussian fit']
    fig, ax = plt.subplots(len(plot_dat)+1, 1, sharex=True, figsize=(10, 20))
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
