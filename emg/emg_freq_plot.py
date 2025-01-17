"""
This module processes and visualizes EMG data related to gapes and LTPS (licking tongue protrusions) in response to different taste and laser conditions. It reads data from CSV and NPY files, processes it, and generates plots for analysis.

- Imports necessary libraries and modules, including custom utilities for metadata handling.
- Reads metadata and changes the working directory to the location of the data files.
- Loads EMG data from CSV and NPY files, filling missing laser data with `False`.
- Extracts and processes time indices for plotting based on metadata parameters.
- Constructs a long-format DataFrame by manually adding time, gapes, and LTPS data.
- Melts the DataFrame to long format for easier plotting, grouping by relevant categories.
- Creates a directory for saving plots if it doesn't exist.
- Generates and saves plots for gapes and LTPS data, showing single trials and averages for each combination of CAR, taste, and laser condition.
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
from utils.blech_utils import imp_metadata

############################################################

# Ask for the directory where the hdf5 file sits, and change to that directory
# Get name of directory with the data files
metadata_handler = imp_metadata(sys.argv)
dir_name = metadata_handler.dir_name
info_dict = metadata_handler.info_dict
params_dict = metadata_handler.params_dict
os.chdir(dir_name)

emg_merge_df = pd.read_csv('emg_output/emg_env_merge_df.csv', index_col = 0)
emg_merge_df['laser'].fillna(False, inplace = True)

gapes = np.load('emg_output/gape_array.npy')
ltps = np.load('emg_output/ltp_array.npy')

# Reading single values from the hdf5 file seems hard, 
# needs the read() method to be called
pre_stim = params_dict["spike_array_durations"][0]
time_limits = [int(x) for x in params_dict['psth_params']['durations']]
x = np.arange(gapes.shape[-1]) - pre_stim
plot_indices = np.where((x >= -time_limits[0])*(x <= time_limits[1]))[0]

plot_gapes = gapes[:,plot_indices]
plot_ltps = ltps[:,plot_indices]
plot_x = [x[plot_indices]]*len(gapes)

# Multi-column explode is not available in current pandas
# Add time, gapes, and ltps to the dataframe manually
fin_frame_list = []
for i in trange(len(emg_merge_df)):
    this_frame = emg_merge_df.iloc[i]
    this_dict = this_frame.to_dict()
    this_dict['time'] = plot_x[i]
    this_dict['gapes'] = plot_gapes[i]
    this_dict['ltps'] = plot_ltps[i]
    fin_frame_list.append(pd.DataFrame(this_dict))

emg_merge_df_long = pd.concat(fin_frame_list, axis = 0) 

# Melt gapes and ltps into long format
emg_merge_df_long = pd.melt(
        emg_merge_df_long, 
        id_vars = [x for x in emg_merge_df_long.columns if x not in ['gapes', 'ltps']], 
        value_vars = ['gapes', 'ltps'],
        var_name = 'emg_type',
        value_name = 'emg_value')

mean_emg_long = emg_merge_df_long.groupby(
        ['car','taste','laser','emg_type','time']).mean().reset_index()

############################################################
# Plotting
############################################################
# For each [CAR, Taste, Laser Condition],
# Plot both Gapes and LTPS
# Single trials and Averages


plot_dir = os.path.join(
        dir_name,
        'emg_output',
        'plots')
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# plot_data = gapes.copy()
# Plot Grid
for plot_name, plot_data in zip(['gapes', 'ltps'], [gapes, ltps]):
    car_list = [x[1] for x in list(emg_merge_df.groupby('car'))]
    for this_car in car_list:
        taste_laser_groups = list(this_car.groupby(['taste', 'laser']))
        taste_laser_inds = [x[0] for x in taste_laser_groups]
        taste_laser_data = [x[1] for x in taste_laser_groups]
        n_tastes = this_car['taste'].nunique()
        n_lasers = this_car['laser'].nunique()
        fig, ax = plt.subplots(
                n_tastes, n_lasers,
                sharex = True, sharey = True,
                figsize = (4*n_lasers, 4*n_tastes))
        if n_tastes == 1:
            ax = np.expand_dims(ax, axis = 0)
        if n_lasers == 1:
            ax = np.expand_dims(ax, axis = 1)
        for this_ind, this_dat, this_ax in \
                zip(taste_laser_inds, taste_laser_data, ax.flatten()):
            this_data_inds = this_dat.index.values
            this_plot_data = plot_data[this_data_inds]
            this_ax.set_title(f"Taste: {this_ind[0]}, Laser: {this_ind[1]}")
            this_ax.pcolormesh(
                    x[plot_indices], 
                    np.arange(this_plot_data.shape[0]), 
                    this_plot_data[:,plot_indices])
            this_ax.axvline(0, 
                    color = 'red', linestyle = '--', 
                    linewidth = 2, alpha = 0.7)
            # this_ax.set_yticks(np.arange(this_plot_data.shape[0])+0.5)
            # this_ax.set_yticklabels(this_data_inds)
        for this_ax in ax[-1,:]:
            this_ax.set_xlabel('Time post-stim (ms)')
        fig.suptitle(f'{this_car.car.unique()[0]} : {plot_name}')
        fig.savefig(
                os.path.join(
                    plot_dir,
                    f'{this_car.car.unique()[0]}_{plot_name}.png'),
                bbox_inches = 'tight')
        plt.close(fig)

# Plot taste overlay per laser condition and CAR
for car_name, car_dat in list(mean_emg_long.groupby('car')):
    g = sns.relplot(
            data = mean_emg_long,
            x = 'time',
            y = 'emg_value',
            hue = 'taste',
            row = 'laser',
            col = 'emg_type',
            # style = 'emg_type',
            kind = 'line',
            linewidth = 2,
            # alpha = 0.7,
            )
    g.fig.suptitle('Taste Overlay')
    # Plot dashed line as x=0
    for ax in g.axes.flatten():
        ax.axvline(0, color = 'red', linestyle = '--', linewidth = 2, alpha = 0.7)
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    g.savefig(
            os.path.join(
                plot_dir,
                f'{car_name}_taste_overlay.png'),
            bbox_inches = 'tight')
    plt.close(g.fig)


    # Plot laser overlay per taste and CAR
    g = sns.relplot(
            data = mean_emg_long,
            x = 'time',
            y = 'emg_value',
            hue = 'laser',
            row = 'emg_type',
            col = 'taste',
            # style = 'emg_type',
            kind = 'line',
            linewidth = 2,
            # alpha = 0.7,
            )
    g.fig.suptitle('Laser Overlay')
    # Plot dashed line as x=0
    for ax in g.axes.flatten():
        ax.axvline(0, color = 'red', linestyle = '--', linewidth = 2, alpha = 0.7)
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    g.savefig(
            os.path.join(
                plot_dir,
                f'{car_name}_laser_overlay.png'),
            bbox_inches = 'tight')
    plt.close(g.fig)

