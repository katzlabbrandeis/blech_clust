"""
Use Auto-regressive RNN to infer firing rates from a given data set.
"""

# Check that blechRNN is on the Desktop, if so, add to path
import os
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch
import matplotlib.pyplot as plt
from scipy.stats import zscore
import json

blechRNN_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'blechRNN')
if os.path.exists(blechRNN_path):
	sys.path.append(blechRNN_path)
else:
	raise FileNotFoundError('blechRNN not found on Desktop')
from src.model import autoencoderRNN
from src.train import train_model

# script_path = '/home/abuzarmahmood/Desktop/blech_clust/utils/infer_rnn_rates.py'
script_path = os.path.abspath(__file__)
blech_clust_path = os.path.dirname(os.path.dirname(script_path))
sys.path.append(blech_clust_path)
from utils.ephys_data import ephys_data
from utils.ephys_data import visualize as vz

import argparse
parser = argparse.ArgumentParser(description = 'Infer firing rates using RNN')
parser.add_argument('data_dir', help = 'Path to data directory')
parser.add_argument('--train_steps', type = int, default = 15000,
                    help = 'Number of training steps')
# Hidden size of 8 was tested to be optimal across multiple datasets
parser.add_argument('--hidden_size', type = int, default = 8,
                    help = 'Hidden size of RNN')
parser.add_argument('--bin_size', type = int, default = 25,
                    help = 'Bin size for binning spikes')
parser.add_argument('--no_pca', action = 'store_true', 
                    help = 'Do not use PCA for preprocessing')
parser.add_argument('--retrain', action = 'store_true', 
                    help = 'Force retraining of model. Will overwrite existing model')

# data_dir = '/media/bigdata/Abuzar_Data/bla_gc/AM11/AM11_4Tastes_191030_114043_copy'
# train_steps = 100
# hidden_size = 8
# bin_size = 25

args = parser.parse_args()
data_dir = args.data_dir
train_steps = args.train_steps
hidden_size = args.hidden_size
bin_size = args.bin_size
# data_dir = sys.argv[1]

# mse loss performs better than poisson loss
loss_name = 'mse'

output_path = os.path.join(data_dir, 'rnn_output')
artifacts_dir = os.path.join(output_path, 'artifacts')
plots_dir = os.path.join(output_path, 'plots')

if not os.path.exists(output_path):
	os.mkdir(output_path)
if not os.path.exists(artifacts_dir):
	os.mkdir(artifacts_dir)
if not os.path.exists(plots_dir):
	os.mkdir(plots_dir)

model_name = f'hidden_{hidden_size}_loss_{loss_name}'
model_save_path = os.path.join(artifacts_dir, f'{model_name}.pt')


print(f'Processing data from {data_dir}')

data = ephys_data.ephys_data(data_dir)
data.get_spikes()

############################################################

spike_array = np.stack(data.spikes)
cat_spikes = np.concatenate(spike_array)

# Trial number
trial_num = np.stack([np.arange(spike_array.shape[1]) for i in range(spike_array.shape[0])])
trial_num = np.concatenate(trial_num)

# Bin spikes
# (tastes x trials, neurons, time)
# for example : (120, 35, 280)
binned_spikes = np.reshape(cat_spikes, 
                           (*cat_spikes.shape[:2], -1, bin_size)).sum(-1)

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

if not args.no_pca:
    print('Performing PCA')
    # Perform PCA and get 95% explained variance
    pca_obj = PCA(n_components=0.95)
    inputs_pca = pca_obj.fit_transform(inputs_long)
    n_components = inputs_pca.shape[-1]

    # Scale the PCA outputs
    pca_scaler = StandardScaler()
    inputs_pca = pca_scaler.fit_transform(inputs_pca)

    inputs_trial_pca = inputs_pca.reshape(inputs.shape[0], -1, n_components)

    # shape: (time, trials, pca_components)
    inputs = inputs_trial_pca.copy()
else:
    inputs_trial_scaled = inputs_long.reshape(inputs.shape)
    inputs = inputs_trial_scaled.copy()

##############################

# Add stim time as external input
# Shape: (time, trials, 1)
stim_time = np.zeros((inputs.shape[0], inputs.shape[1]))
stim_time[2000//bin_size, :] = 1

# Also add trial number as external input
# Scale trial number to be between 0 and 1
trial_num_scaled = trial_num / trial_num.max()
trial_num_broad = np.broadcast_to(trial_num_scaled, inputs.shape[:2])

# Shape: (time, trials, pca_components + 1)
inputs_plus_context = np.concatenate(
        [
            inputs, 
            stim_time[:,:,None],
            trial_num_broad[:,:,None]
            ], 
        axis = -1)

stim_t_input = inputs_plus_context[..., -2]
plt.imshow(stim_t_input.T, aspect = 'auto')
plt.title('Stim Time Input')
plt.savefig(os.path.join(plots_dir, 'input_stim_time.png'))
plt.close()

# Plot inputs for sanity check
# vz.firing_overview(inputs.swapaxes(0,1).swapaxes(1,2),
vz.firing_overview(inputs_plus_context.T,
                   figsize = (10,10),
                   cmap = 'viridis',
                   backend = 'imshow',
                   zscore_bool = True,)
fig = plt.gcf()
plt.suptitle('RNN Inputs')
fig.savefig(os.path.join(plots_dir, 'inputs.png'))
plt.close(fig)

############################################################
# Train model
############################################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

input_size = inputs_plus_context.shape[-1] 
output_size = inputs_plus_context.shape[-1] -2 # We don't want to predict the stim time

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
train_test_split = 0.75
train_inds = np.random.choice(
        np.arange(inputs.shape[1]), 
        int(train_test_split * inputs.shape[1]), 
        replace = False)
test_inds = np.setdiff1d(np.arange(inputs.shape[1]), train_inds)

train_inputs = inputs[:,train_inds]
train_labels = labels[:,train_inds]
test_inputs = inputs[:,test_inds]
test_labels = labels[:,test_inds]


train_inputs = train_inputs.to(device)
train_labels = train_labels.to(device)
test_inputs = test_inputs.to(device)
test_labels = test_labels.to(device)

##############################
# Train 
##############################
net = autoencoderRNN( 
        input_size=input_size,
        hidden_size= hidden_size, 
        output_size=output_size,
        rnn_layers = 2,
        dropout = 0.2,
        )

loss_path = os.path.join(artifacts_dir, 'loss.json')
cross_val_loss_path = os.path.join(artifacts_dir, 'cross_val_loss.json')
if (not os.path.exists(model_save_path)) or args.retrain:
	net.to(device)
	net, loss, cross_val_loss = train_model(
			net, 
			train_inputs, 
			train_labels, 
			output_size = output_size,
			lr = 0.001, 
			train_steps = train_steps,
			loss = loss_name, 
			test_inputs = test_inputs,
			test_labels = test_labels,
			)
	# Save artifacts and plots
	torch.save(net, model_save_path)
	# np.save(loss_path, loss)
	# np.save(cross_val_loss_path, cross_val_loss)
	with open(loss_path, 'w') as f:
		json.dump(loss, f)
	with open(cross_val_loss_path, 'w') as f:
		json.dump(cross_val_loss, f)
else:
	net = torch.load(model_save_path)
	# loss = np.load(loss_path, allow_pickle = True)
	# cross_val_loss = np.load(cross_val_loss_path, allow_pickle = True)
	with open(loss_path, 'r') as f:
		loss = json.load(f)
	with open(cross_val_loss_path, 'r') as f:
		cross_val_loss = json.load(f)


############################################################
# Reconstruction
outs, latent_outs = net(inputs.to(device))
outs = outs.cpu().detach().numpy()
latent_outs = latent_outs.cpu().detach().numpy()
pred_firing = np.moveaxis(outs, 0, -1)

##############################
# Convert back into neuron space
pred_firing = np.moveaxis(pred_firing, 0, -1).T
pred_firing_long = pred_firing.reshape(-1, pred_firing.shape[-1])

# If pca was performed, first reverse PCA, then reverse pca standard scaling
if not args.no_pca:
    # # Reverse NMF scaling
    # pred_firing_long = nmf_scaler.inverse_transform(pred_firing_long)
    pred_firing_long = pca_scaler.inverse_transform(pred_firing_long)

    # Reverse NMF transform
    # pred_firing_long = nmf_obj.inverse_transform(pred_firing_long)
    pred_firing_long = pca_obj.inverse_transform(pred_firing_long)

# Reverse standard scaling
pred_firing_long = scaler.inverse_transform(pred_firing_long)

pred_firing = pred_firing_long.reshape((*pred_firing.shape[:2], -1))
pred_firing = np.moveaxis(pred_firing, 1,2)

##############################

############################################################

# Loss plot
fig, ax = plt.subplots()
ax.plot(loss, label = 'Train Loss') 
ax.plot(cross_val_loss.keys(), cross_val_loss.values(), label = 'Test Loss')
ax.legend(
        bbox_to_anchor=(1.05, 1), 
        loc='upper left', borderaxespad=0.)
ax.set_title(f'Losses') 
fig.savefig(os.path.join(plots_dir,'run_loss.png'),
            bbox_inches = 'tight')
plt.close(fig)

# Firing rate plots
vz.firing_overview(pred_firing.swapaxes(0,1))
fig = plt.gcf()
plt.suptitle('RNN Predicted Firing Rates')
fig.savefig(os.path.join(plots_dir, 'firing_pred.png'))
plt.close(fig)
vz.firing_overview(binned_spikes.swapaxes(0,1))
fig = plt.gcf()
plt.suptitle('Binned Firing Rates')
fig.savefig(os.path.join(plots_dir, 'firing_binned.png'))
plt.close(fig)

# Latent factors
fig, ax = plt.subplots(latent_outs.shape[-1], 1, figsize = (5,10),
					   sharex = True, sharey = True)
for i in range(latent_outs.shape[-1]):
    ax[i].imshow(latent_outs[...,i].T, aspect = 'auto')
plt.suptitle('Latent Factors')
fig.savefig(os.path.join(plots_dir, 'latent_factors.png'))
plt.close(fig)

# Mean firing rates
pred_firing_mean = pred_firing.mean(axis = 0)
binned_spikes_mean = binned_spikes.mean(axis = 0)

fig, ax = plt.subplots(1,2)
ax[0].imshow(pred_firing_mean, aspect = 'auto', interpolation = 'none')
ax[1].imshow(binned_spikes_mean, aspect = 'auto', interpolation = 'none')
ax[0].set_title('Pred')
ax[1].set_title('True')
fig.savefig(os.path.join(plots_dir, 'mean_firing.png'))
plt.close(fig)
# plt.show()

fig, ax = plt.subplots(1,2)
ax[0].imshow(zscore(pred_firing_mean,axis=-1), aspect = 'auto', interpolation = 'none')
ax[1].imshow(zscore(binned_spikes_mean,axis=-1), aspect = 'auto', interpolation = 'none')
ax[0].set_title('Pred')
ax[1].set_title('True')
fig.savefig(os.path.join(plots_dir, 'mean_firing_zscored.png'))
plt.close(fig)

# Mean neuron firing
fig, ax = vz.gen_square_subplots(len(pred_firing_mean),
                                 figsize = (10,10),
                                 sharex = True,)
for i in range(pred_firing.shape[1]):
    ax.flatten()[i].plot(pred_firing_mean[i], 
                         alpha = 0.7, label = 'pred')
    ax.flatten()[i].plot(binned_spikes_mean[i], 
                         alpha = 0.7, label = 'true')
    ax.flatten()[i].set_ylabel(str(i))
fig.savefig(os.path.join(plots_dir, 'mean_neuron_firing.png'))
plt.close(fig)

# For every neuron, plot 1) spike raster, 2) convolved firing rate , 
# 3) RNN predicted firing rate
ind_plot_dir = os.path.join(plots_dir, 'individual_neurons')
if not os.path.exists(ind_plot_dir):
    os.makedirs(ind_plot_dir)

binned_x = np.arange(0, binned_spikes.shape[-1]*bin_size, bin_size)

conv_kern = np.ones(250) / 250
conv_rate = np.apply_along_axis(
        lambda m: np.convolve(m, conv_kern, mode = 'valid'),
                            axis = -1, arr = cat_spikes)*bin_size
conv_x = np.convolve(
        np.arange(cat_spikes.shape[-1]), conv_kern, mode = 'valid')

for i in range(binned_spikes.shape[1]):
    fig, ax = plt.subplots(3,1, figsize = (10,10),
                           sharex = True, sharey = False)
    ax[0] = vz.raster(ax[0], cat_spikes[:, i], marker = '|')
    ax[1].plot(conv_x, conv_rate[:,i].T, c = 'k', alpha = 0.1)
    # ax[2].plot(binned_x, binned_spikes[:,i].T, label = 'True')
    ax[2].plot(binned_x[1:], pred_firing[:,i].T, 
               c = 'k', alpha = 0.1)
    # ax[2].sharey(ax[1])
    for this_ax in ax:
        this_ax.set_xlim([1500, 4000])
        this_ax.axvline(2000, c = 'r', linestyle = '--')
    ax[1].set_title(f'Convolved Firing Rate : Kernel Size {len(conv_kern)}')
    ax[2].set_title('RNN Predicted Firing Rate')
    fig.savefig(
            os.path.join(ind_plot_dir, f'neuron_{i}_raster_conv_pred.png'))
    plt.close(fig)

# Make another plot with taste_mean firing rates
cmap = plt.get_cmap('tab10')
for i in range(binned_spikes.shape[1]):
    fig, ax = plt.subplots(3,1, figsize = (10,10),
                           sharex = True, sharey = False)
    ax[0] = vz.raster(ax[0], cat_spikes[:, i], marker = '|', color = 'k')
    # Plot colors behind raster traces
    for j in range(4):
        ax[0].axhspan(len(cat_spikes)*j/4, len(cat_spikes)*(j+1)/4,
                      color = cmap(j), alpha = 0.1, zorder = 0)
    this_conv_rate = conv_rate[:,i]
    this_pred_firing = pred_firing[:,i]
    this_conv_rate = np.stack(np.split(this_conv_rate, 4))
    this_pred_firing = np.stack(np.split(this_pred_firing, 4))
    mean_conv_rate = this_conv_rate.mean(axis = 1)
    mean_pred_firing = this_pred_firing.mean(axis = 1)
    sd_conv_rate = this_conv_rate.std(axis = 1)
    sd_pred_firing = this_pred_firing.std(axis = 1)
    for j in range(mean_conv_rate.shape[0]):
        ax[1].plot(conv_x, mean_conv_rate[j].T, c = cmap(j),
                   linewidth = 2)
        ax[1].fill_between(
                conv_x, 
                mean_conv_rate[j] - sd_conv_rate[j],
                mean_conv_rate[j] + sd_conv_rate[j],
                color = cmap(j), alpha = 0.1)
        # ax[2].plot(binned_x, binned_spikes[:,i].T, label = 'True')
        ax[2].plot(binned_x[1:], mean_pred_firing[j].T,
                   c = cmap(j), linewidth = 2)
        ax[2].fill_between(
                binned_x[1:], 
                mean_pred_firing[j] - sd_pred_firing[j],
                mean_pred_firing[j] + sd_pred_firing[j],
                color = cmap(j), alpha = 0.1)
        # ax[2].sharey(ax[1])
    for this_ax in ax:
        this_ax.set_xlim([1500, 4000])
        this_ax.axvline(2000, c = 'r', linestyle = '--')
    ax[1].set_title(f'Convolved Firing Rate : Kernel Size {len(conv_kern)}')
    ax[2].set_title('RNN Predicted Firing Rate')
    fig.savefig(
            os.path.join(
                ind_plot_dir, 
                f'neuron_{i}_mean_raster_conv_pred.png'))
    plt.close(fig)

# Plot single-trial latent factors
trial_latent_dir = os.path.join(plots_dir, 'trial_latent')
if not os.path.exists(trial_latent_dir):
    os.makedirs(trial_latent_dir)

for i in range(latent_outs.shape[1]):
    fig, ax = plt.subplots(2,1)
    ax[0].plot(latent_outs[1:,i], alpha = 0.5)
    ax[0].set_title(f'Latent factors for trial {i}')
    ax[1].plot(zscore(latent_outs[1:,i], axis = 0), alpha = 0.5)
    fig.savefig(os.path.join(trial_latent_dir, f'trial_{i}_latent.png'))
    plt.close(fig)

# Plot predicted activity vs true activity for every neuron

for i in range(pred_firing.shape[1]):
    fig, ax = plt.subplots(1,2, sharex = True, sharey = True)
    min_val = min(pred_firing[:,i].min(), binned_spikes[:,i].min())
    max_val = max(pred_firing[:,i].max(), binned_spikes[:,i].max())
    img_kwargs = {'aspect':'auto', 'interpolation':'none', 'cmap':'viridis',
                  }
                  #'vmin':min_val, 'vmax':max_val}
    im0 = ax[0].imshow(pred_firing[:,i], **img_kwargs) 
    im1 = ax[1].imshow(binned_spikes[:,i, 1:], **img_kwargs) 
    ax[0].set_title('Pred')
    ax[1].set_title('True')
    # Colorbars under each subplot
    cbar0 = fig.colorbar(im0, ax = ax[0], orientation = 'horizontal')
    cbar1 = fig.colorbar(im1, ax = ax[1], orientation = 'horizontal')
    cbar0.set_label('Firing Rate (Hz)')
    cbar1.set_label('Firing Rate (Hz)')
    fig.savefig(os.path.join(ind_plot_dir, f'neuron_{i}_firing.png'))
    plt.close(fig)

