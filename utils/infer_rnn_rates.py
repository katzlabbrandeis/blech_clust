"""
This module uses an Auto-regressive Recurrent Neural Network (RNN) to infer firing rates from electrophysiological data. It processes data for each taste separately, trains an RNN model, and saves the predicted firing rates and latent factors.

- Parses command-line arguments to configure the RNN model, including data directory, training steps, hidden size, bin size, train-test split, PCA usage, retraining option, and time limits.
- Loads configuration from a JSON file if not overridden by command-line arguments.
- Loads spike data using the `ephys_data` class and preprocesses it, including binning and optional PCA.
- Trains an RNN model for each taste, using a specified loss function (MSE) and saves the model and training artifacts.
- Generates and saves various plots, including firing rate overviews, latent factors, and mean firing rates.
- Writes the predicted firing rates and latent outputs to an HDF5 file for each taste.
- Handles file paths and directories for saving models, plots, and outputs, ensuring necessary directories exist.
"""
import argparse  # noqa: E402
parser = argparse.ArgumentParser(description='Infer firing rates using RNN')
parser.add_argument('data_dir', help='Path to data directory')
parser.add_argument('--train_steps', type=int,
                    help='Number of training steps (default: %(default)s)')
# Hidden size of 8 was tested to be optimal across multiple datasets
parser.add_argument('--hidden_size', type=int,
                    help='Hidden size of RNN (default: %(default)s)')
parser.add_argument('--bin_size', type=int,
                    help='Bin size for binning spikes (default: %(default)s)')
parser.add_argument('--train_test_split', type=float,
                    help='Fraction of data to use for training (default: %(default)s)')
parser.add_argument('--no_pca', action='store_true',
                    help='Do not use PCA for preprocessing (default: %(default)s)')
parser.add_argument('--retrain', action='store_true',
                    help='Force retraining of model. Will overwrite existing model' +
                    ' (default: %(default)s)')
parser.add_argument('--time_lims', type=int, nargs=2,
                    help='Time limits inferred firing rates (default: %(default)s)')
parser.add_argument('--separate_regions', action='store_true',
                    help='Fit RNNs for each region separately (default: %(default)s)')

args = parser.parse_args()

import tables  # noqa
from scipy.stats import zscore  # noqa
import matplotlib.pyplot as plt  # noqa
import torch  # noqa
from sklearn.decomposition import PCA  # noqa
from sklearn.preprocessing import StandardScaler  # noqa
import numpy as np  # noqa
import sys  # noqa
import os  # noqa
from pprint import pprint  # noqa
import json  # noqa
from itertools import product  # noqa
import pandas as pd  # noqa

data_dir = args.data_dir
# data_dir = '/media/fastdata/Thomas_Data/data/sorted_new/EB13/Day3Exp120trl_230529_110345'
script_path = os.path.abspath(__file__)
blech_clust_path = os.path.dirname(os.path.dirname(script_path))
sys.path.append(blech_clust_path)  # noqa
from utils.blech_utils import entry_checker, imp_metadata, pipeline_graph_check  # noqa
# blech_clust_path = '/home/abuzarmahmood/Desktop/blech_clust'

metadata_handler = imp_metadata([[], args.data_dir])
# Perform pipeline graph check
this_pipeline_check = pipeline_graph_check(args.data_dir)
this_pipeline_check.check_previous(script_path)
this_pipeline_check.write_to_log(script_path, 'attempted')

config_path = os.path.join(
    blech_clust_path, 'params', 'blechrnn_params.json')
if not os.path.exists(config_path):
    raise FileNotFoundError(
        f'BlechRNN Config file not found @ {config_path}')
with open(config_path, 'r') as f:
    config = json.load(f)
print('Loaded config file\n')
train_steps = config['train_steps']
hidden_size = config['hidden_size']
bin_size = config['bin_size']
train_test_split = config['train_test_split']
use_pca = config['use_pca']
time_lims = config['time_lims']

# If any argument provided, use those instead
if args.train_steps:
    print(f'Using provided train_steps: {args.train_steps}')
    train_steps = args.train_steps
if args.hidden_size:
    print(f'Using provided hidden_size: {args.hidden_size}')
    hidden_size = args.hidden_size
if args.bin_size:
    print(f'Using provided bin_size: {args.bin_size}')
    bin_size = args.bin_size
if args.train_test_split:
    print(f'Using provided train_test_split: {args.train_test_split}')
    train_test_split = args.train_test_split
if args.no_pca:
    print(f'Using provided pca setting: {not args.no_pca}')
    use_pca = not args.no_pca
if args.time_lims:
    print(f'Using provided time_lims: {args.time_lims}')
    time_lims = args.time_lims

params_dict = dict(
    train_steps=train_steps,
    hidden_size=hidden_size,
    bin_size=bin_size,
    train_test_split=train_test_split,
    use_pca=use_pca,
    time_lims=time_lims,
)
pprint(params_dict)

##############################

# Check that blechRNN is on the Desktop, if so, add to path

blechRNN_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'blechRNN')
if os.path.exists(blechRNN_path):
    sys.path.append(blechRNN_path)
else:
    raise FileNotFoundError('blechRNN not found on Desktop')

# script_path = '/home/abuzarmahmood/Desktop/blech_clust/utils/infer_rnn_rates.py'
sys.path.append(blech_clust_path)

from utils.ephys_data import visualize as vz  # noqa
from utils.ephys_data import ephys_data  # noqa
from src.train import train_model  # noqa
from src.model import autoencoderRNN  # noqa

# mse loss performs better than poisson loss
loss_name = 'mse'

output_path = os.path.join(data_dir, 'rnn_output')
artifacts_dir = os.path.join(output_path, 'artifacts')
plots_dir = os.path.join(output_path, 'plots')

for dir_path in [output_path, artifacts_dir, plots_dir]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


print(f'Processing data from {data_dir}')

data = ephys_data.ephys_data(data_dir)
data.get_spikes()

############################################################

if args.separate_regions:
    print('Processing regions separately')
    data.get_region_units()
    print(data.region_units)
    region_names = data.region_names
    # Get spikes for each region
    # Shape : (tastes, trials, neurons, time)
    spike_arrays = [data.return_region_spikes(
        region) for region in region_names]
    # Remove None
    keep_inds = [i for i, x in enumerate(spike_arrays) if x is not None]
    region_names = [region_names[i] for i in keep_inds]
    spike_arrays = [spike_arrays[i] for i in keep_inds]
    print(f'Processing regions: {region_names}')
    # Product of region and taste indices
else:
    print('Processing all regions together')
    region_names = ['all']
    spike_arrays = [np.stack(data.spikes)]

processing_inds = list(
    product(range(len(region_names)), range(len(spike_arrays[0]))))
processing_items = [(taste_ind, (region_names[region_ind], spike_arrays[region_ind][taste_ind]))
                    for region_ind, taste_ind in processing_inds]
processing_str = [[f'Taste {i}, Region {j[0]}'] for i, j in processing_items]
print('Processing the following items:')
pprint(processing_str)

pred_firing_list = []
pred_x_list = []
conv_rate_list = []
conv_x_list = []
binned_spikes_list = []
latent_out_list = []
region_name_list = []
taste_ind_list = []

# Train model for each taste/region combination
for idx, (name, spike_data) in processing_items:

    iden_str = f'{name}_taste_{idx}'
    region_name_list.append(name)
    taste_ind_list.append(idx)

    print(f'Processing region {name}, taste {idx}')
    model_name = f'region_{name}_taste_{idx}_hidden_{hidden_size}_loss_{loss_name}'
    model_save_path = os.path.join(artifacts_dir, f'{model_name}.pt')

    # taste_spikes = np.concatenate(spike_array)
    # Cut taste_spikes to time limits
    # Shape: (trials, neurons, time)
    spike_data = spike_data[..., time_lims[0]:time_lims[1]]

    trial_num = np.arange(spike_data.shape[0])

    # Bin spikes
    # (tastes x trials, neurons, time)
    # for example : (120, 35, 280)
    binned_spikes = np.reshape(spike_data,
                               (*spike_data.shape[:2], -1, bin_size)).sum(-1)
    binned_spikes_list.append(binned_spikes)

    # ** The naming of inputs / labels throughouts is confusing as hell

    # Reshape to (seq_len, batch, input_size)
    # seq_len = time
    # batch = trials
    # input_size = neurons
    inputs = binned_spikes.copy()
    inputs = np.moveaxis(inputs, -1, 0)

    ##############################
    # Perform PCA on data
    # If PCA is performed on raw data, higher firing neurons will dominate
    # the latent space
    # Therefore, perform PCA on zscored data

    inputs_long = inputs.reshape(-1, inputs.shape[-1])

    # Perform standard scaling
    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    inputs_long = scaler.fit_transform(inputs_long)

    if use_pca:
        print('Performing PCA')
        # Perform PCA and get 95% explained variance
        pca_obj = PCA(n_components=0.95)
        inputs_pca = pca_obj.fit_transform(inputs_long)
        n_components = inputs_pca.shape[-1]

        # # Scale the PCA outputs
        # pca_scaler = StandardScaler()
        # inputs_pca = pca_scaler.fit_transform(inputs_pca)

        inputs_trial_pca = inputs_pca.reshape(
            inputs.shape[0], -1, n_components)

        # shape: (time, trials, pca_components)
        inputs = inputs_trial_pca.copy()
    else:
        inputs_trial_scaled = inputs_long.reshape(inputs.shape)
        inputs = inputs_trial_scaled.copy()

    ##############################

    # Add stim time as external input
    # Shape: (time, trials, 1)
    stim_time_val = 2000 - time_lims[0]
    if stim_time_val < 0:
        raise ValueError('Stim time is before time limits')
    stim_time = np.zeros((inputs.shape[0], inputs.shape[1]))
    stim_time[stim_time_val//bin_size, :] = 1
    # Don't use taste_num as external input
    # Network tends to read too much into it
    # stim_time[2000//bin_size, :] = taste_num / taste_num.max()

    # Also add trial number as external input
    # Scale trial number to be between 0 and 1
    trial_num_scaled = trial_num / trial_num.max()
    trial_num_broad = np.broadcast_to(trial_num_scaled, inputs.shape[:2])

    # Shape: (time, trials, pca_components + 1)
    inputs_plus_context = np.concatenate(
        [
            inputs,
            stim_time[:, :, None],
            trial_num_broad[:, :, None]
        ],
        axis=-1)

    stim_t_input = inputs_plus_context[..., -2]
    plt.imshow(stim_t_input.T, aspect='auto')
    plt.title('Stim Time Input')
    plt.savefig(os.path.join(plots_dir, 'input_stim_time.png'))
    plt.close()

    # Plot inputs for sanity check
    # vz.firing_overview(inputs.swapaxes(0,1).swapaxes(1,2),
    vz.firing_overview(inputs_plus_context.T,
                       figsize=(10, 10),
                       cmap='viridis',
                       backend='imshow',
                       zscore_bool=False,)
    fig = plt.gcf()
    plt.suptitle('RNN Inputs')
    fig.savefig(os.path.join(plots_dir, f'inputs_{iden_str}.png'))
    plt.close(fig)

    ############################################################
    # Train model
    ############################################################
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    input_size = inputs_plus_context.shape[-1]
    # We don't want to predict the stim time or trial number
    output_size = inputs_plus_context.shape[-1] - 2

    # Instead of predicting activity in the SAME time-bin,
    # predict activity in the NEXT time-bin
    # Forcing the model to learn temporal dependencies
    inputs_plus_context = inputs_plus_context[:-1]
    inputs = inputs[1:]

    # (seq_len * batch, output_size)
    labels = torch.from_numpy(inputs).type(torch.float32)
    # (seq_len, batch, input_size)
    inputs = torch.from_numpy(inputs_plus_context).type(torch.float)

    # Split into train and test
    train_inds = np.random.choice(
        np.arange(inputs.shape[1]),
        int(train_test_split * inputs.shape[1]),
        replace=False)
    test_inds = np.setdiff1d(np.arange(inputs.shape[1]), train_inds)

    train_inputs = inputs[:, train_inds]
    train_labels = labels[:, train_inds]
    test_inputs = inputs[:, test_inds]
    test_labels = labels[:, test_inds]

    train_inputs = train_inputs.to(device)
    train_labels = train_labels.to(device)
    test_inputs = test_inputs.to(device)
    test_labels = test_labels.to(device)

    ##############################
    # Train
    ##############################
    net = autoencoderRNN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        rnn_layers=2,
        dropout=0.2,
    )

    loss_path = os.path.join(artifacts_dir, f'loss_taste_{idx}.json')
    cross_val_loss_path = os.path.join(
        artifacts_dir, f'cross_val_loss_taste_{idx}.json')
    if (not os.path.exists(model_save_path)) or args.retrain:
        if args.retrain:
            print('Retraining model')
        net.to(device)
        net, loss, cross_val_loss = train_model(
            net,
            train_inputs,
            train_labels,
            output_size=output_size,
            lr=0.001,
            train_steps=train_steps,
            loss=loss_name,
            test_inputs=test_inputs,
            test_labels=test_labels,
        )
        # Save artifacts and plots
        torch.save(net, model_save_path)
        # np.save(loss_path, loss)
        # np.save(cross_val_loss_path, cross_val_loss)
        with open(loss_path, 'w') as f:
            json.dump(loss, f)
        with open(cross_val_loss_path, 'w') as f:
            json.dump(cross_val_loss, f)
        with open(os.path.join(artifacts_dir, 'params.json'), 'w') as f:
            json.dump(params_dict, f)
    else:
        print('Model already exists. Loading model')
        net = torch.load(model_save_path)
        # loss = np.load(loss_path, allow_pickle = True)
        # cross_val_loss = np.load(cross_val_loss_path, allow_pickle = True)
        with open(loss_path, 'r') as f:
            loss = json.load(f)
        with open(cross_val_loss_path, 'r') as f:
            cross_val_loss = json.load(f)
        with open(os.path.join(artifacts_dir, 'params.json'), 'r') as f:
            params_dict = json.load(f)

    # If final train loss > cross val loss, issue warning
    if loss[-1] > cross_val_loss[max(cross_val_loss.keys())]:
        warning_file_path = os.path.join(artifacts_dir, 'warning.txt')
        warning_str = """
        Final training loss is greater than cross validation loss.
        This indicates something weird is going on (maybe with train-test split or PCA).
        Try retraining the model (to get a new train-test split) or using the --no-pca flag.
        """
        with open(warning_file_path, 'w') as f:
            f.write(warning_str)
        print(warning_str)

    ############################################################
    # Reconstruction
    outs, latent_outs = net(inputs.to(device))
    outs = outs.cpu().detach().numpy()
    latent_outs = latent_outs.cpu().detach().numpy()
    pred_firing = np.moveaxis(outs, 0, -1)

    latent_out_list.append(latent_outs)

    ##############################
    # Convert back into neuron space
    pred_firing = np.moveaxis(pred_firing, 0, -1).T
    pred_firing_long = pred_firing.reshape(-1, pred_firing.shape[-1])

    # If pca was performed, first reverse PCA, then reverse pca standard scaling
    if use_pca:
        # # Reverse NMF scaling
        # pred_firing_long = nmf_scaler.inverse_transform(pred_firing_long)
        # pred_firing_long = pca_scaler.inverse_transform(pred_firing_long)

        # Reverse NMF transform
        # pred_firing_long = nmf_obj.inverse_transform(pred_firing_long)
        pred_firing_long = pca_obj.inverse_transform(pred_firing_long)

    # Reverse standard scaling
    pred_firing_long = scaler.inverse_transform(pred_firing_long)

    pred_firing = pred_firing_long.reshape((*pred_firing.shape[:2], -1))
    # shape: (trials, neurons, time)
    pred_firing = np.moveaxis(pred_firing, 1, 2)

    pred_firing_list.append(pred_firing)

    ##############################

    ############################################################

    # Loss plot
    fig, ax = plt.subplots()
    ax.plot(loss, label='Train Loss')
    ax.plot(cross_val_loss.keys(), cross_val_loss.values(), label='Test Loss')
    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc='upper left', borderaxespad=0.)
    ax.set_title(f'Losses')
    fig.savefig(os.path.join(plots_dir, f'run_loss_{iden_str}.png'),
                bbox_inches='tight')
    plt.close(fig)

    # Firing rate plots
    vz.firing_overview(pred_firing.swapaxes(0, 1))
    fig = plt.gcf()
    plt.suptitle('RNN Predicted Firing Rates')
    fig.savefig(os.path.join(plots_dir, f'firing_pred_{iden_str}.png'))
    plt.close(fig)
    vz.firing_overview(binned_spikes.swapaxes(0, 1))
    fig = plt.gcf()
    plt.suptitle('Binned Firing Rates')
    fig.savefig(os.path.join(
        plots_dir, f'firing_binned_{iden_str}.png'))
    plt.close(fig)

    # Latent factors
    fig, ax = plt.subplots(latent_outs.shape[-1], 1, figsize=(5, 10),
                           sharex=True, sharey=True)
    for i in range(latent_outs.shape[-1]):
        ax[i].imshow(latent_outs[..., i].T, aspect='auto')
    plt.suptitle('Latent Factors')
    fig.savefig(os.path.join(
        plots_dir, f'latent_factors_{iden_str}.png'))
    plt.close(fig)

    # Mean firing rates
    pred_firing_mean = pred_firing.mean(axis=0)
    binned_spikes_mean = binned_spikes.mean(axis=0)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(pred_firing_mean, aspect='auto', interpolation='none')
    ax[1].imshow(binned_spikes_mean, aspect='auto', interpolation='none')
    ax[0].set_title('Pred')
    ax[1].set_title('True')
    fig.savefig(os.path.join(plots_dir, f'mean_firing_{iden_str}.png'))
    plt.close(fig)
    # plt.show()

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(zscore(pred_firing_mean, axis=-1),
                 aspect='auto', interpolation='none')
    ax[1].imshow(zscore(binned_spikes_mean, axis=-1),
                 aspect='auto', interpolation='none')
    ax[0].set_title('Pred')
    ax[1].set_title('True')
    fig.savefig(os.path.join(
        plots_dir, f'mean_firing_zscored_{iden_str}.png'))
    plt.close(fig)

    # For every neuron, plot 1) spike raster, 2) convolved firing rate ,
    # 3) RNN predicted firing rate
    ind_plot_dir = os.path.join(plots_dir, 'individual_neurons')
    if not os.path.exists(ind_plot_dir):
        os.makedirs(ind_plot_dir)

    binned_x = np.arange(0, binned_spikes.shape[-1]*bin_size, bin_size)
    pred_x_list.append(binned_x)

    conv_kern = np.ones(250) / 250
    conv_rate = np.apply_along_axis(
        lambda m: np.convolve(m, conv_kern, mode='valid'),
        axis=-1, arr=spike_data)*bin_size
    conv_x = np.convolve(
        np.arange(spike_data.shape[-1]), conv_kern, mode='valid')
    conv_rate_list.append(conv_rate)
    conv_x_list.append(conv_x)

    for i in range(binned_spikes.shape[1]):
        fig, ax = plt.subplots(3, 1, figsize=(10, 10),
                               sharex=True, sharey=False)
        ax[0] = vz.raster(ax[0], spike_data[:, i], marker='|')
        ax[1].plot(conv_x, conv_rate[:, i].T, c='k', alpha=0.1)
        # ax[2].plot(binned_x, binned_spikes[:,i].T, label = 'True')
        ax[2].plot(binned_x[1:], pred_firing[:, i].T,
                   c='k', alpha=0.1)
        # ax[2].sharey(ax[1])
        for this_ax in ax:
            # this_ax.set_xlim([1500, 4000])
            this_ax.axvline(stim_time_val, c='r', linestyle='--')
        ax[1].set_title(
            f'Convolved Firing Rate : Kernel Size {len(conv_kern)}')
        ax[2].set_title('RNN Predicted Firing Rate')
        fig.savefig(
            os.path.join(ind_plot_dir,
                         f'neuron_{i}_{iden_str}_raster_conv_pred.png')
        )
        plt.close(fig)

    # Plot single-trial latent factors
    trial_latent_dir = os.path.join(plots_dir, 'trial_latent')
    if not os.path.exists(trial_latent_dir):
        os.makedirs(trial_latent_dir)

    for i in range(latent_outs.shape[1]):
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(latent_outs[1:, i], alpha=0.5)
        ax[0].set_title(f'Latent factors for trial {i}')
        ax[1].plot(zscore(latent_outs[1:, i], axis=0), alpha=0.5)
        fig.savefig(os.path.join(trial_latent_dir,
                    f'{iden_str}_trial_{i}_latent.png'))
        plt.close(fig)

############################################################
############################################################

pred_frame = pd.DataFrame(
    dict(
        region_name=region_name_list,
        taste_ind=taste_ind_list,
        pred_firing=pred_firing_list,
        latent_out=latent_out_list,
        pred_x=pred_x_list,
        conv_rate=conv_rate_list,
        conv_x=conv_x_list,
        binned_spikes=binned_spikes_list,
    )
)

all_pred_firing_taste_mean = [
    [x.mean(axis=0) for x in pred_frame.loc[pred_frame.region_name ==
                                            region_name, 'pred_firing'].to_list()]
    for region_name in region_names
]

all_binned_spikes_taste_mean = [
    [x.mean(axis=0) for x in pred_frame.loc[pred_frame.region_name ==
                                            region_name, 'binned_spikes'].to_list()]
    for region_name in region_names
]

# Mean neuron firing
# pred_firing_taste_mean = np.stack(
#     [pred_firing_list[i].mean(axis=0) for i in range(len(pred_firing_list))])
# binned_spikes_taste_mean = np.stack(
#     [bin_spikes.mean(axis=0) for bin_spikes in binned_spikes_list])

for pred_firing_taste_mean, binned_spikes_taste_mean, region_name in zip(
        all_pred_firing_taste_mean, all_binned_spikes_taste_mean, region_names
):
    cmap = plt.get_cmap('tab10')
    region_nrn_count = pred_firing_taste_mean[0].shape[0]
    fig, ax = vz.gen_square_subplots(region_nrn_count,
                                     figsize=(10, 10),
                                     sharex=True,)
    for nrn_ind in range(region_nrn_count):
        for taste_ind, (pred, bin) in enumerate(
                zip(pred_firing_taste_mean, binned_spikes_taste_mean)
        ):
            ax.flatten()[nrn_ind].plot(
                pred[nrn_ind], alpha=1, c=cmap(taste_ind))
            ax.flatten()[nrn_ind].plot(
                bin[nrn_ind], alpha=0.3, c=cmap(taste_ind))
        ax.flatten()[nrn_ind].set_ylabel(str(nrn_ind))
    fig.savefig(os.path.join(
        plots_dir, f'mean_neuron_firing_{region_name}.png'))
    plt.close(fig)

# Make another plot with taste_mean firing rates
for spike_array, region_name in zip(spike_arrays, region_names):
    region_nrn_count = spike_array[0].shape[1]
    region_conv_rate_list = pred_frame.loc[pred_frame.region_name ==
                                           region_name, 'conv_rate'].to_list()
    region_pred_firing_list = pred_frame.loc[pred_frame.region_name ==
                                             region_name, 'pred_firing'].to_list()
    cmap = plt.get_cmap('tab10')
    # Iterate over neurons
    for i in range(region_nrn_count):
        fig, ax = plt.subplots(3, 1, figsize=(10, 10),
                               sharex=True, sharey=False)
        # Get spikes from all tastes for this neuron
        this_spikes_list = [x[:, i] for x in spike_array]
        trial_counts = [len(x) for x in this_spikes_list]
        cum_trial_counts = np.cumsum([0, *trial_counts])
        this_cat_spikes = np.concatenate(
            this_spikes_list, axis=0)[..., time_lims[0]:time_lims[1]]

        ax[0] = vz.raster(ax[0], this_cat_spikes, marker='|', color='k')
        # Plot colors behind raster traces
        for j in range(len(cum_trial_counts)-1):
            ax[0].axhspan(cum_trial_counts[j], cum_trial_counts[j+1],
                          color=cmap(j), alpha=0.1, zorder=0)

        # this_conv_rate = np.stack([x[:, i] for x in region_conv_rate_list])
        # this_pred_firing = np.stack([x[:, i] for x in region_pred_firing_list])

        # mean_conv_rate = this_conv_rate.mean(axis=1)
        # mean_pred_firing = this_pred_firing.mean(axis=1)
        # sd_conv_rate = this_conv_rate.std(axis=1)
        # sd_pred_firing = this_pred_firing.std(axis=1)
        mean_conv_rate = np.stack([x[:, i].mean(axis=0)
                                  for x in region_conv_rate_list])
        mean_pred_firing = np.stack([x[:, i].mean(axis=0)
                                    for x in region_pred_firing_list])
        sd_conv_rate = np.stack([x[:, i].std(axis=0)
                                for x in region_conv_rate_list])
        sd_pred_firing = np.stack([x[:, i].std(axis=0)
                                  for x in region_pred_firing_list])
        for j in range(mean_conv_rate.shape[0]):
            ax[1].plot(conv_x, mean_conv_rate[j].T, c=cmap(j),
                       linewidth=2)
            ax[1].fill_between(
                conv_x,
                mean_conv_rate[j] - sd_conv_rate[j],
                mean_conv_rate[j] + sd_conv_rate[j],
                color=cmap(j), alpha=0.1)
            # ax[2].plot(binned_x, binned_spikes[:,i].T, label = 'True')
            ax[2].plot(binned_x[1:], mean_pred_firing[j].T,
                       c=cmap(j), linewidth=2)
            ax[2].fill_between(
                binned_x[1:],
                mean_pred_firing[j] - sd_pred_firing[j],
                mean_pred_firing[j] + sd_pred_firing[j],
                color=cmap(j), alpha=0.1)
            # ax[2].sharey(ax[1])
        for this_ax in ax:
            # this_ax.set_xlim([1500, 4000])
            this_ax.axvline(stim_time_val, c='r', linestyle='--')
        ax[1].set_title(
            f'Convolved Firing Rate : Kernel Size {len(conv_kern)}')
        ax[2].set_title('RNN Predicted Firing Rate')
        fig.savefig(
            os.path.join(
                ind_plot_dir,
                f'neuron_{i}_region_{region_name}_mean_raster_conv_pred.png'))
        plt.close(fig)


# Plot predicted activity vs true activity for every neuron
for i in range(pred_firing.shape[1]):
    cat_pred_firing = np.concatenate([x[:, i] for x in pred_firing_list])
    cat_binned_spikes = np.concatenate([x[:, i] for x in binned_spikes_list])

    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    min_val = min(cat_pred_firing.min(), cat_binned_spikes.min())
    max_val = max(cat_pred_firing.max(), cat_binned_spikes.max())
    img_kwargs = {'aspect': 'auto', 'interpolation': 'none', 'cmap': 'viridis',
                  }
    # 'vmin':min_val, 'vmax':max_val}
    im0 = ax[0].imshow(cat_pred_firing, **img_kwargs)
    im1 = ax[1].imshow(cat_binned_spikes[:, 1:], **img_kwargs)
    ax[0].set_title('Pred')
    ax[1].set_title('True')
    # Colorbars under each subplot
    cbar0 = fig.colorbar(im0, ax=ax[0], orientation='horizontal')
    cbar1 = fig.colorbar(im1, ax=ax[1], orientation='horizontal')
    cbar0.set_label('Firing Rate (Hz)')
    cbar1.set_label('Firing Rate (Hz)')
    fig.savefig(os.path.join(ind_plot_dir, f'neuron_{i}_firing.png'))
    plt.close(fig)


############################################################
############################################################
# Write out firing rates to file
hdf5_path = data.hdf5_path
with tables.open_file(hdf5_path, 'r+') as hf5:
    # Create directory for rnn output
    if '/rnn_output' not in hf5:
        hf5.create_group('/', 'rnn_output', 'RNN Output')
    rnn_output = hf5.get_node('/rnn_output')

    if '/rnn_output/regions' not in hf5:
        hf5.create_group('/rnn_output', 'regions',
                         'Region-specific RNN Output')
    rnn_output = hf5.get_node('/rnn_output/regions')

    # Write out latents and predicted firing rates
    for idx, (name, _) in processing_items:
        group_name = f'region_{name}_taste_{idx}'
        group_desc = f'Region {name} Taste {idx}'

        taste_grp = hf5.create_group(rnn_output, group_name, group_desc)
        hf5.create_array(taste_grp, 'pred_firing', pred_firing_list[idx])
        hf5.create_array(taste_grp, 'latent_out', latent_out_list[idx])
        hf5.create_array(taste_grp, 'pred_x', pred_x_list[idx])

# Write successful execution to log
this_pipeline_check.write_to_log(script_path, 'completed')
