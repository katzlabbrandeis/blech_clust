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

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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
            io_layer_size=8,
            n_io_layers=3,
            shared_rnn=None,
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
            *[nn.Linear(io_layer_size, io_layer_size) for _ in range(n_io_layers)]
        )
        self.shared_rnn = shared_rnn
        self.output_transform = nn.Sequential(
            *[nn.Linear(io_layer_size, io_layer_size) for _ in range(n_io_layers)]
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

i = 0
this_row = out_frame.iloc[i]
stim_dur = this_row.stimulus_duration[this_row.taste]
stim_dur_inds = int(stim_dur // 25)

binned_spikes = this_row.spike_train.copy()

##############################
# Begin copying from infer_rnn_rates.py
##############################

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

print('Performing PCA')
# Perform PCA and get 95% explained variance
pca_obj = PCA(n_components=8)
inputs_pca = pca_obj.fit_transform(inputs_long)
n_components = inputs_pca.shape[-1]

# # Scale the PCA outputs
# pca_scaler = StandardScaler()
# inputs_pca = pca_scaler.fit_transform(inputs_pca)

inputs_trial_pca = inputs_pca.reshape(
    inputs.shape[0], -1, n_components)

# shape: (time, trials, pca_components)
inputs = inputs_trial_pca.copy()

##############################

# Add stim time as external input
# Shape: (time, trials, 1)
stim_time = np.zeros((inputs.shape[0], inputs.shape[1]))
stim_time[stim_ind:stim_ind + stim_dur_inds, :] = 1

# Also add trial number as external input
# Scale trial number to be between 0 and 1
trial_num = np.arange(binned_spikes.shape[0])
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
shared_rnn = autoencoderRNN(
    input_size=input_size,
    hidden_size=8,
    output_size=output_size,
    rnn_layers=2,
    dropout=0.2,
)

net = RNN_Wrapper(
    io_layer_size=8,
    n_io_layers=3,
    shared_rnn=shared_rnn,
)

net.to(device)
net, loss, cross_val_loss = train_model(
    net,
    train_inputs,
    train_labels,
    output_size=output_size,
    lr=0.001,
    train_steps=50_000,
    criterion=MSELoss(),
    test_inputs=test_inputs,
    test_labels=test_labels,
)
