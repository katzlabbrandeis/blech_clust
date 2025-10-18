"""
This module processes and visualizes EMG data related to gapes and licking in response to different taste and laser conditions. It reads data from CSV and NPY files, processes it, and generates plots for analysis.

- Imports necessary libraries and modules, including custom utilities for metadata handling.
- Reads metadata and changes the working directory to the location of the data files.
- Loads EMG data from CSV and NPY files, filling missing laser data with `False`.
- Extracts and processes time indices for plotting based on metadata parameters.
- Constructs a long-format DataFrame by manually adding time, gapes, and licking data.
- Melts the DataFrame to long format for easier plotting, grouping by relevant categories.
- Creates a directory for saving plots if it doesn't exist.
- Generates and saves plots for gapes and licking data, showing single trials and averages for each combination of CAR, taste, and laser condition.
- Produces overlay plots for taste and laser conditions using seaborn, saving them to the specified directory.
"""
# Import stuff!
import numpy as np
import tables
import easygui
import sys
import os
import matplotlib.pyplot as plt
import glob
import json
import pandas as pd
import seaborn as sns
from tqdm import tqdm, trange
# Necessary blech_clust modules
sys.path.append('..')
from blech_clust.utils.blech_utils import imp_metadata  # noqa: E402

############################################################

# Ask for the directory where the hdf5 file sits, and change to that directory
# Get name of directory with the data files
test_bool = False

if test_bool:
    data_dir = '/media/storage/NM_resorted_data/NM43/NM43_500ms_160510_125413'
    metadata_handler = imp_metadata([[], data_dir])
else:
    metadata_handler = imp_metadata(sys.argv)

dir_name = metadata_handler.dir_name
info_dict = metadata_handler.info_dict
params_dict = metadata_handler.params_dict
os.chdir(dir_name)

emg_merge_df = pd.read_csv('emg_output/emg_env_merge_df.csv', index_col=0)
emg_merge_df['laser_tuple'] = emg_merge_df.apply(
    lambda x: (x['laser_duration_ms'], x['laser_lag_ms']), axis=1)

gapes = np.load('emg_output/gape_array.npy')
licking = np.load('emg_output/licking_array.npy')

# Reading single values from the hdf5 file seems hard,
# needs the read() method to be called
pre_stim = params_dict["spike_array_durations"][0]
time_limits = [int(x) for x in params_dict['psth_params']['durations']]
x = np.arange(gapes.shape[-1]) - pre_stim
plot_indices = np.where((x >= -time_limits[0])*(x <= time_limits[1]))[0]

plot_gapes = gapes[:, plot_indices]
plot_licking = licking[:, plot_indices]
plot_x = np.stack([x[plot_indices]]*len(gapes))

print('Gathering EMG data...')

# Multi-column explode is not available in current pandas
# Add time, gapes, and licking to the dataframe manually
fin_frame_list = []
for i in trange(len(emg_merge_df)):
    this_frame = pd.DataFrame(
        [emg_merge_df.iloc[i]]*len(plot_gapes[i]))
    this_frame['time'] = plot_x[i]
    this_frame['gapes'] = plot_gapes[i]
    this_frame['licking'] = plot_licking[i]
    fin_frame_list.append(this_frame)

emg_merge_df_long = pd.concat(fin_frame_list, axis=0)

# Melt gapes and licking into long format
emg_merge_df_long = pd.melt(
    emg_merge_df_long,
    id_vars=[x for x in emg_merge_df_long.columns if x not in [
        'gapes', 'licking']],
    value_vars=['gapes', 'licking'],
    var_name='emg_type',
    value_name='emg_value')

mean_emg_long = emg_merge_df_long.groupby(
    ['car', 'taste', 'laser_tuple', 'emg_type', 'time']).mean().reset_index()

############################################################
# Plotting
############################################################
# For each [CAR, Taste, Laser Condition],
# Plot both Gapes and Licking
# Single trials and Averages


plot_dir = os.path.join(
    dir_name,
    'emg_output',
    'plots')
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

print('Plotting Gape and Licking data...')

# plot_data = gapes.copy()
# Plot Grid
for plot_name, plot_data in zip(['gapes', 'licking'], [gapes, licking]):
    car_list = [x[1] for x in list(emg_merge_df.groupby('car'))]
    for this_car in car_list:
        taste_laser_groups = list(this_car.groupby(['taste', 'laser_tuple']))
        taste_laser_inds = [x[0] for x in taste_laser_groups]
        taste_laser_data = [x[1] for x in taste_laser_groups]
        n_tastes = this_car['taste'].nunique()
        n_lasers = this_car['laser'].nunique()
        fig, ax = plt.subplots(
            n_tastes, n_lasers,
            sharex=True, sharey=True,
            figsize=(4*n_lasers, 4*n_tastes))
        if n_tastes == 1:
            ax = np.expand_dims(ax, axis=0)
        if n_lasers == 1:
            ax = np.expand_dims(ax, axis=1)
        for this_ind, this_dat, this_ax in \
                zip(taste_laser_inds, taste_laser_data, ax.flatten()):
            this_data_inds = this_dat.index.values
            this_plot_data = plot_data[this_data_inds]
            this_ax.set_title(f"Taste: {this_ind[0]}, Laser: {this_ind[1]}")
            this_ax.pcolormesh(
                x[plot_indices],
                np.arange(this_plot_data.shape[0]),
                this_plot_data[:, plot_indices])
            this_ax.axvline(0,
                            color='red', linestyle='--',
                            linewidth=2, alpha=0.7)
            # this_ax.set_yticks(np.arange(this_plot_data.shape[0])+0.5)
            # this_ax.set_yticklabels(this_data_inds)
        for this_ax in ax[-1, :]:
            this_ax.set_xlabel('Time post-stim (ms)')
        fig.suptitle(f'{this_car.car.unique()[0]} : {plot_name}')
        fig.savefig(
            os.path.join(
                plot_dir,
                f'{this_car.car.unique()[0]}_{plot_name}.png'),
            bbox_inches='tight')
        plt.close(fig)


# Plot taste overlay per laser condition and CAR
print('Plotting Taste and Laser Overlay...')

# Downsample
bin_size = 25
mean_emg_long['bin'] = mean_emg_long['time'] // bin_size

# Average over bins
mean_emg_long = mean_emg_long.groupby(
    ['car', 'taste', 'laser_tuple', 'emg_type', 'bin']).mean().reset_index()

for car_name, car_dat in list(mean_emg_long.groupby('car')):
    g = sns.relplot(
        data=mean_emg_long,
        x='time',
        y='emg_value',
        hue='taste',
        row='laser_tuple',
        col='emg_type',
        # style = 'emg_type',
        kind='line',
        linewidth=2,
        # alpha = 0.7,
    )
    g.fig.suptitle('Taste Overlay')
    # Plot dashed line as x=0
    for ax in g.axes.flatten():
        ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    g.savefig(
        os.path.join(
            plot_dir,
            f'{car_name}_taste_overlay.png'),
        bbox_inches='tight')
    plt.close(g.fig)

    # Plot laser overlay per taste and CAR
    g = sns.relplot(
        data=mean_emg_long,
        x='time',
        y='emg_value',
        hue='laser_tuple',
        row='emg_type',
        col='taste',
        # style = 'emg_type',
        kind='line',
        linewidth=2,
        # alpha = 0.7,
    )
    g.fig.suptitle('Laser Overlay')
    # Plot dashed line as x=0
    for ax in g.axes.flatten():
        ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    g.savefig(
        os.path.join(
            plot_dir,
            f'{car_name}_laser_overlay.png'),
        bbox_inches='tight')
    plt.close(g.fig)
