from visualize import firing_overview, imshow
from ephys_data import ephys_data
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from scipy.stats import zscore
from tqdm import tqdm

# os.chdir('/media/bigdata/firing_space_plot/ephys_data')
sys.path.append('/media/bigdata/firing_space_plot/ephys_data')

# data_dir = '/media/bigdata/Abuzar_Data/AM28'
# dir_list = sorted(os.listdir(data_dir))

dir_list = open('/media/bigdata/Abuzar_Data/dir_list.txt', 'r').readlines()
dir_list = [x.strip() for x in dir_list]

for this_dir in tqdm(dir_list):

    # this_dir = [x for x in dir_list if 'bla_only' in x][0]
    dat = ephys_data(this_dir)

    # dat = ephys_data('/media/bigdata/Abuzar_Data/bla_only/AM28/AM28_4Tastes_201006_095803')

    dat.get_unit_descriptors()
    dat.get_lfps()
    dat.get_region_units()
    dat.get_lfp_electrodes()

    dat.stft_params["signal_window"] = 1000
    dat.stft_params["window_overlap"] = 990
    dat.get_stft()  # recalculate = True)

    electrode_group = np.concatenate([np.zeros(len(x))+num
                                      for num, x in enumerate(dat.lfp_region_electrodes)])
    sorted_electrode_group = \
        electrode_group[np.argsort(np.concatenate(dat.lfp_region_electrodes))]

    median_amplitude = np.median(dat.amplitude_array, axis=(0, 2))
    sorted_median_amp = [[median_amplitude[x] for x in this_region_inds]
                         for this_region_inds in dat.lfp_region_electrodes]

    zscore_med_amp = np.array([zscore(x, axis=-1) for x in median_amplitude])
    sorted_zscore_amp = [[zscore_med_amp[x] for x in this_region_inds]
                         for this_region_inds in dat.lfp_region_electrodes]

    for num, region_name in enumerate(dat.region_names):
        firing_overview(
            data=np.array(sorted_median_amp[num]),
            t_vec=dat.time_vec,
            y_values_vec=dat.freq_vec,
            subplot_labels=dat.parsed_lfp_channels[dat.lfp_region_electrodes[num]])
        title_str = os.path.basename(dat.data_dir)+'\nRaw median spectrogram : ' +\
            region_name.upper()
        plt.suptitle(title_str)
        plt.tight_layout()
        plt.subplots_adjust(top=0.8)
        fig = plt.gcf()
        fig.savefig(os.path.join(dat.data_dir, title_str), dpi=300)

    for num, region_name in enumerate(dat.region_names):
        firing_overview(
            data=np.array(sorted_zscore_amp[num]),
            t_vec=dat.time_vec,
            y_values_vec=dat.freq_vec,
            subplot_labels=dat.parsed_lfp_channels[dat.lfp_region_electrodes[num]])
        title_str = os.path.basename(dat.data_dir)+'\nZscored median spectrogram : ' +\
            region_name.upper()
        plt.suptitle(title_str)
        plt.tight_layout()
        plt.subplots_adjust(top=0.8)
        fig = plt.gcf()
        fig.savefig(os.path.join(dat.data_dir, title_str), dpi=300)
    # plt.show()

    # firing_overview(zscore_med_amp,t_vec = dat.time_vec, y_values_vec = dat.freq_vec,
    #                        subplot_labels = sorted_electrode_group)
    # title_str = os.path.basename(dat.data_dir)+'\nZscored median spectrogram'
    # plt.suptitle(title_str)
    # plt.show()

    raw_region_median_amp = np.array([np.median(np.array(x), axis=0)
                                      for x in sorted_median_amp])
    zscore_region_median_amp = np.array([np.median(np.array(x), axis=0)
                                         for x in sorted_zscore_amp])

    title_str = os.path.basename(dat.data_dir) + \
        '\nRaw median region spectrogram'
    fig, ax = plt.subplots(1, len(zscore_region_median_amp), figsize=(15, 5))
    if len(zscore_region_median_amp) < 2:
        ax = [ax]
    for num, (this_data, this_ax) in enumerate(zip(raw_region_median_amp, ax)):
        this_ax.imshow(this_data, aspect='auto', origin='lower',
                       cmap='jet', interpolation='gaussian')
        this_ax.set_title(dat.region_names[num].upper())
    plt.subplots_adjust(top=0.8)
    plt.suptitle(title_str)
    # plt.show()
    fig.savefig(os.path.join(dat.data_dir, title_str), dpi=300)

    title_str = os.path.basename(dat.data_dir) + \
        '\nZscored median region spectrogram'
    fig, ax = plt.subplots(1, len(zscore_region_median_amp), figsize=(15, 5))
    if len(zscore_region_median_amp) < 2:
        ax = [ax]
    for num, (this_data, this_ax) in enumerate(zip(zscore_region_median_amp, ax)):
        this_ax.imshow(this_data, aspect='auto', origin='lower',
                       cmap='jet', interpolation='gaussian')
        this_ax.set_title(dat.region_names[num].upper())
    plt.subplots_adjust(top=0.8)
    plt.suptitle(title_str)
    # plt.show()
    fig.savefig(os.path.join(dat.data_dir, title_str), dpi=300)

    plt.close('all')
