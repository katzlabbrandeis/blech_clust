"""
Gather spike-trains from datasets
Only keep "stable" neurons
"""

from utils.ephys_data import visualize as vz
from utils.ephys_data.ephys_data import ephys_data
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import pandas as pd

__file__ = '/home/abuzarmahmood/Desktop/blech_clust/utils/foundational_rnn/src/gather_data.py'
script_path = Path(__file__).resolve()
blech_clust_path = script_path.parents[3]

sys.path.append(str(blech_clust_path))


##############################

data_list_path = script_path.parents[1] / 'data' / 'training_datasets.txt'
data_list = open(data_list_path).read().splitlines()

artifacts_dir = script_path.parents[1] / 'data' / 'artifacts'

##############################

time_lims = [2000, 4000]
bin_width = 25

# For each dataset, load the data and save the spike-trains
spike_train_list = []
stimulus_duration_list = []
basename_list = []
taste_list = []
for this_dir in tqdm(data_list):
    try:
        this_dir = this_dir.strip()
        print('Loading dataset: ', this_dir)
        data = ephys_data(this_dir)
        basename = os.path.basename(this_dir)
        data.get_spikes()
        data.get_stable_units()
        spike_array = np.stack(data.spikes)
        stable_spike_array = spike_array[:, :, data.stable_units]

        stable_spike_array = stable_spike_array[..., time_lims[0]:time_lims[1]]

        binned_spikes = stable_spike_array.reshape(
            (*stable_spike_array.shape[:3], -1, bin_width)).sum(axis=-1)

        # data.get_firing_rates()
        # vz.firing_overview(data.all_normalized_firing[data.stable_units]);plt.show()

        # Get duration of stimulus pulses for each taste
        data.get_trial_info_frame()
        data.trial_info_frame['stimulus_duration'] = \
            data.trial_info_frame['end_taste_ms'] - \
            data.trial_info_frame['start_taste_ms']

        stimulus_durations = data.trial_info_frame.groupby(
            'taste')['stimulus_duration'].mean()

        for taste, taste_data in enumerate(binned_spikes):
            # Save the data
            spike_train_list.append(taste_data)
            stimulus_duration_list.append(stimulus_durations)
            basename_list.append(basename)
            taste_list.append(taste)
    except Exception as e:
        print('Error loading dataset: ', this_dir)
        print(e)

# Save as a pickled pandas dataframe
out_frame = pd.DataFrame(
    dict(
        spike_train=spike_train_list,
        stimulus_duration=stimulus_duration_list,
        basename=basename_list,
        taste=taste_list
    )
)
out_frame.to_pickle(artifacts_dir / 'spike_trains.pkl')
