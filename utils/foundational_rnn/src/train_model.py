from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import torch.nn as nn
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import pandas as pd
import torch  # noqa

plt.ion()


blechRNN_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'blechRNN')
if os.path.exists(blechRNN_path):
    sys.path.append(blechRNN_path)
else:
    raise FileNotFoundError('blechRNN not found on Desktop')

from src.train import train_model, MSELoss  # noqa
from src.model import autoencoderRNN  # noqa


##############################
__file__ = '/home/abuzarmahmood/Desktop/blech_clust/utils/foundational_rnn/src/gather_data.py'
script_path = Path(__file__).resolve()
blech_clust_path = script_path.parents[3]

sys.path.append(str(blech_clust_path))

artifacts_dir = script_path.parents[1] / 'data' / 'artifacts'

out_frame = pd.read_pickle(artifacts_dir / 'spike_trains.pkl')

##############################
##############################


class RNN_Wrapper(nn.Module):
    """
    Wrap additional dataset specific layers around a shared RNN
    """

    def __init__(
            self,
            n_io_layers=3,
            shared_rnn=None,
            input_size=None,
            output_size=None,
            **kwargs):
        """
        Parameters
        ----------
        io_layer_size : int
        n_io_layers : int
        shared_rnn : nn.Module

        """
        super(RNN_Wrapper, self).__init__()
        self.input_transform = nn.Sequential(
            *[nn.Linear(input_size, input_size) for _ in range(n_io_layers-1)],
            nn.Linear(input_size, input_size)
        )
        self.shared_rnn = shared_rnn
        self.output_transform = nn.Sequential(
            nn.Linear(output_size, output_size),
            *[nn.Linear(output_size, output_size) for _ in range(n_io_layers-1)]
        )

    def forward(self, x):
        x = self.input_transform(x)
        x, latents = self.shared_rnn(x)
        x = self.output_transform(x)
        return x, latents


##############################
##############################
# Assert a minimum number of neurons
neuron_counts = [this_row.spike_train.shape[1]
                 for i, this_row in out_frame.iterrows()]
out_frame['neuron_count'] = neuron_counts

min_neuron_count = 8
out_frame = out_frame[out_frame.neuron_count > min_neuron_count]

stim_ind = 500 // 25

############################################################
############################################################

binned_spikes_list = []
mean_binned_spikes_list = []

# Perform PCA on data and perform alignment

for i in range(len(out_frame)):
    this_row = out_frame.iloc[i]
    stim_dur = this_row.stimulus_duration[this_row.taste]
    stim_dur_inds = int(stim_dur // 25)

    binned_spikes = this_row.spike_train.copy()
    mean_binned_spikes = binned_spikes.mean(axis=0)

    binned_spikes_list.append(binned_spikes)
    mean_binned_spikes_list.append(mean_binned_spikes)


cat_mean_binned_spikes = np.concatenate(mean_binned_spikes_list, axis=0)

pca_obj = PCA(n_components=8, whiten=True)
cat_mean_pca = pca_obj.fit_transform(cat_mean_binned_spikes.T).T

# vz.imshow(cat_mean_binned_spikes)
# plt.show()
# vz.imshow(cat_mean_pca)
# plt.show()

transform_list = []
# Learn transformation matrices
for this_mean in mean_binned_spikes_list:
    lr = LinearRegression()
    # this_mean shape: (neurons, time)
    # cat_mean_pca shape: (pca, time)
    lr.fit(this_mean.T, cat_mean_pca.T)
    transform_list.append(lr)

binned_spikes_long_list = []
aligned_spikes_long_list = []
aligned_spikes_trials_list = []
for i, binned_spikes in enumerate(binned_spikes_list):
    binned_spikes_long = binned_spikes.swapaxes(0, 1)
    binned_spikes_long = binned_spikes_long.reshape(
        binned_spikes_long.shape[0], -1)
    binned_spikes_long_list.append(binned_spikes_long)

    # Perform alignment
    aligned_spikes = transform_list[i].predict(binned_spikes_long.T).T
    aligned_spikes_long_list.append(aligned_spikes)

    # Convert back to trials
    aligned_spikes_trials = aligned_spikes.reshape(
        aligned_spikes.shape[0], binned_spikes.shape[0], binned_spikes.shape[-1])
    aligned_spikes_trials = aligned_spikes_trials.swapaxes(0, 1)

    aligned_spikes_trials_list.append(aligned_spikes_trials)

shared_rnn = autoencoderRNN(
    input_size=input_size,
    hidden_size=8,
    output_size=output_size,
    rnn_layers=2,
    dropout=0.2,
)

net_list = []
train_inputs_list = []
train_labels_list = []
test_inputs_list = []
test_labels_list = []

for i, input_array in enumerate(aligned_spikes_trials_list):

    ##############################
    # Begin copying from infer_rnn_rates.py
    ##############################
    # Reshape to (seq_len, batch, input_size)
    # seq_len = time
    # batch = trials
    # input_size = neurons
    inputs = input_array.copy()
    # inputs = binned_spikes.copy()
    inputs = np.moveaxis(inputs, -1, 0)

    ##############################
    # Perform PCA on data
    # If PCA is performed on raw data, higher firing neurons will dominate
    # the latent space
    # Therefore, perform PCA on zscored data

    # inputs_long = inputs.reshape(-1, inputs.shape[-1])
    #
    # # Perform standard scaling
    # scaler = StandardScaler()
    # # scaler = MinMaxScaler()
    # inputs_long = scaler.fit_transform(inputs_long)
    #
    # print('Performing PCA')
    # # Perform PCA and get 95% explained variance
    # pca_obj = PCA(n_components=8)
    # inputs_pca = pca_obj.fit_transform(inputs_long)
    # n_components = inputs_pca.shape[-1]
    #
    # # # Scale the PCA outputs
    # # pca_scaler = StandardScaler()
    # # inputs_pca = pca_scaler.fit_transform(inputs_pca)
    #
    # inputs_trial_pca = inputs_pca.reshape(
    #     inputs.shape[0], -1, n_components)
    #
    # # shape: (time, trials, pca_components)
    # inputs = inputs_trial_pca.copy()

    ##############################

    # Add stim time as external input
    # Shape: (time, trials, 1)
    stim_time = np.zeros((inputs.shape[0], inputs.shape[1]))
    stim_time[stim_ind:stim_ind + stim_dur_inds, :] = 1

    # Also add trial number as external input
    # Scale trial number to be between 0 and 1
    # trial_num = np.arange(binned_spikes.shape[0])
    trial_num = np.arange(input_array.shape[0])
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
    forecast_bins = 1
    inputs_plus_context = inputs_plus_context[:-forecast_bins]
    inputs = inputs[forecast_bins:]

    # (seq_len * batch, output_size)
    labels = torch.from_numpy(inputs).type(torch.float32)
    # (seq_len, batch, input_size)
    inputs = torch.from_numpy(inputs_plus_context).type(torch.float)

    # Split into train and test
    train_inds = np.random.choice(
        np.arange(inputs.shape[1]),
        int(0.9 * inputs.shape[1]),
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
    net = RNN_Wrapper(
        n_io_layers=3,
        input_size=input_size,
        output_size=output_size,
        shared_rnn=shared_rnn,
    )

    net.to(device)

    # Append to list
    net_list.append(net)
    train_inputs_list.append(train_inputs)
    train_labels_list.append(train_labels)
    test_inputs_list.append(test_inputs)
    test_labels_list.append(test_labels)


train_steps = 500_000
update_steps = 1000

fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
ax[0].set_title('Training Loss')
ax[1].set_title('Cross Validation Loss')
full_loss = []
full_cross_val_loss = {}
for epochs in range(train_steps // update_steps):
    for i in tqdm(range(len(net_list))):
        net = net_list[i]
        train_inputs = train_inputs_list[i]
        train_labels = train_labels_list[i]
        test_inputs = test_inputs_list[i]
        test_labels = test_labels_list[i]

        # Train model
        net, loss, cross_val_loss = train_model(
            net,
            train_inputs,
            train_labels,
            output_size=output_size,
            lr=0.001,
            train_steps=update_steps,
            criterion=MSELoss(),
            test_inputs=test_inputs,
            test_labels=test_labels,
        )
        full_loss.extend(loss)
        # Update keys in cross_val_loss
        cross_val_loss = {k + epochs * update_steps: v for k,
                          v in cross_val_loss.items()}
        full_cross_val_loss.update(cross_val_loss)

        # Create plot
        if i == 0:
            ax[0].plot(full_loss[::10])
            ax[1].plot(list(full_cross_val_loss.keys())[::10],
                       list(full_cross_val_loss.values())[::10])
            plt.pause(0.01)
