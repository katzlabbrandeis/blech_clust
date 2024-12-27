"""
Edits:
    X Clean file loading code
    X Move code for affective analysis to different script
    - Fix legends in seaborn plots and add file name to all plots over files
    X For pairwise t-test over bands and trial bins, use chronological
        to make new trial bins which don't rely on tastes
    X Save output plots
    X Save extracted ITIs to HDF5 file
    X Plots resized to be large
"""

# ___                            _
#|_ _|_ __ ___  _ __   ___  _ __| |_
# | || '_ ` _ \| '_ \ / _ \| '__| __|
# | || | | | | | |_) | (_) | |  | |_
#|___|_| |_| |_| .__/ \___/|_|   \__|
#              |_|

## Import required modules
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import tables
import h5py
import easygui
import scipy
from scipy.signal import spectrogram
import numpy as np
from scipy.signal import hilbert, butter, filtfilt,freqs
from tqdm import tqdm, trange
from itertools import product
import pandas as pd
import pingouin as pg
import seaborn as sns
from scipy.stats import zscore
import glob
from collections import namedtuple
from scipy.signal import convolve
from itertools import chain
import shutil


############################
# Define functions to extract data
############################

def get_whole_session_lfp(hdf5_name):
    with tables.open_file(hdf5_name, 'r+') as hf5:
        whole_lfp = hf5.get_node('/Whole_session_raw_LFP/WS_LFP')[:]
    return whole_lfp

def get_delivery_times(hdf5_name):
    delivery_times = \
            pd.read_hdf(hdf5_name,'/Whole_session_spikes/delivery_times')
    delivery_times['taste'] = delivery_times.index
    delivery_times = \
            pd.melt(delivery_times,
                    id_vars = 'taste',
                    var_name ='trial',
                    value_name='delivery_time')
    delivery_times.sort_values(by='delivery_time',inplace=True)
    # Delivery times are in 30kHz samples, convert to ms
    delivery_times['delivery_time'] = delivery_times['delivery_time'] // 30
    delivery_times['chronological'] = np.argsort(delivery_times.delivery_time)
    return delivery_times

# _                    _   ____        _
#| |    ___   __ _  __| | |  _ \  __ _| |_ __ _
#| |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
#| |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
#|_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
#

# All HDF5 files need to be in the same folder
# Load files and make sure the order is right

# Ask user for all relevant files
file_list = []
last_dir = None
while True:
    if last_dir is not None:
        file_name = easygui.fileopenbox(msg = 'Please select files to extract'\
                ' data from, CANCEL to stop', default = last_dir)
    else:
        file_name = easygui.fileopenbox(msg = 'Please select files to extract'\
                ' data from, CANCEL to stop')
    if file_name is not None:
        file_list.append(file_name)
        last_dir = os.path.dirname(file_name)
    else:
        break

# Ask user for directory to save plots in
output_dir = easygui.diropenbox(msg = \
        'Please select PARENT directory to save plots in \n'\
        'Another directory will be created for the analysis within', default = last_dir)
animal_names = [os.path.basename(x).split('_')[0] for x in file_list]
if animal_names[0] == animal_names[1]:
    fin_animal_name = animal_names[0]
else:
    fin_animal_name = easygui.multenterbox('Please enter animal name' \
            '("_ITI_analysis_output" will be concatenated automatically',
            'Enter animal name',['Animal Name'])
fin_output_dir = os.path.join(output_dir, fin_animal_name + '_ITI_analysis_output')

# If plot dir exists, delete and recreate, otherwise just make
if os.path.exists(fin_output_dir):
    shutil.rmtree(fin_output_dir)
os.makedirs(fin_output_dir)

# Generate name for the output HDF5 file
output_hf5_name = os.path.join(fin_output_dir,fin_animal_name + '_iti_analysis.h5')

# Extract LFP from all sessions
taste_whole_lfp = [get_whole_session_lfp(file_name) \
        for recording_num, file_name in tqdm(enumerate(file_list))]

# Extract ITI's from taste sessions
trial_time_data = [get_delivery_times(file_name) \
        for file_name in file_list]
delivery_time_list = [x.delivery_time for x in trial_time_data]

# Define parameters to extract ITI data
time_after_delivery = 10 #seconds
iti_duration = 9 #second before taste delivery won't be extracted
Fs = 1000 # Sampling frequency

# (trials x channels x time)
# Determine start and end times of ITIs
iti_intervals = [[(x+(time_after_delivery*Fs), x+((time_after_delivery+iti_duration)*Fs)) \
                    for x in delivery_times] for delivery_times in delivery_time_list]

iti_array_list = \
        [np.asarray(
            [lfp_array[:,interval[0]:interval[1]] for interval in interval_list]) \
                for interval_list,lfp_array in zip(iti_intervals,taste_whole_lfp)]

# Write ITI arrays back to files
for file_num, this_file in enumerate(file_list):
    with tables.open_file(this_file, 'r+') as h5:
        if '/Whole_session_raw_LFP/ITI_array' in h5:
            h5.remove_node('/Whole_session_raw_LFP/ITI_array')
        h5.create_array('/Whole_session_raw_LFP','ITI_array',iti_array_list[file_num])
        h5.flush()

#    _                _           _
#   / \   _ __   __ _| |_   _ ___(_)___
#  / _ \ | '_ \ / _` | | | | / __| / __|
# / ___ \| | | | (_| | | |_| \__ \ \__ \
#/_/   \_\_| |_|\__,_|_|\__, |___/_|___/
#                       |___/

# Bandpass filter lfp into relevant bands

#define bandpass filter parameters to parse out frequencies
def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

band_freqs = [(1,4),
                (4,7),
                (7,12),
                (12,25),
                (25,50)]


iti_lfp_bandpassed  = \
            [np.asarray([
                    butter_bandpass_filter(
                        data = data,
                        lowcut = band[0],
                        highcut = band[1],
                        fs = Fs) \
                                for band in tqdm(band_freqs)])\
                for data in iti_array_list]

# Remove to preserve memory
del iti_array_list, taste_whole_lfp

# Calculate Hilbert and amplitude
iti_lfp_hilbert = [hilbert(data) for data in tqdm(iti_lfp_bandpassed)]
iti_lfp_amplitude = [np.abs(data) for data in tqdm(iti_lfp_hilbert)]

# ____                                             _
#|  _ \ _ __ ___ _ __  _ __ ___   ___ ___  ___ ___(_)_ __   __ _
#| |_) | '__/ _ \ '_ \| '__/ _ \ / __/ _ \/ __/ __| | '_ \ / _` |
#|  __/| | |  __/ |_) | | | (_) | (_|  __/\__ \__ \ | | | | (_| |
#|_|   |_|  \___| .__/|_|  \___/ \___\___||___/___/_|_| |_|\__, |
#               |_|                                        |___/

# Create array index identifiers
# Used to convert array to pandas dataframe
def make_array_identifiers(array):
    nd_idx_objs = []
    for dim in range(array.ndim):
        this_shape = np.ones(len(array.shape))
        this_shape[dim] = array.shape[dim]
        nd_idx_objs.append(
                np.broadcast_to(
                    np.reshape(
                        np.arange(array.shape[dim]),
                                this_shape.astype('int')),
                    array.shape).flatten())
    return nd_idx_objs

# Take average of power across channels
mean_channel_iti_lfp_amplitude = [np.mean(data,axis=(2)) for data in iti_lfp_amplitude]
# Take average across every ITI
mean_iti_lfp_amplitude = [np.mean(data,axis=(2)) for data in mean_channel_iti_lfp_amplitude]

########################################
# Check for outliers
########################################
zscore_trials_power = np.asarray([\
        [zscore(band,axis = None) for band in data] for data in mean_iti_lfp_amplitude])

# Plot zscore trial averaged power so
fig, ax = plt.subplots(len(zscore_trials_power),1,sharex=True)
for num, this_ax in enumerate(ax):
   this_ax.plot(zscore_trials_power[num].T,'x')
   this_ax.title.set_text(os.path.basename(file_list[num]))
   this_ax.set_xlabel('Trial num');this_ax.set_ylabel('Zscore mean power')
plt.tight_layout()
plt.show()

# Ask user to input threshold zscore to remove bad trials from ANOVA analysis
file_str = ("".join(['{}) {} \n'.format(file_num,os.path.basename(file_name)) \
        for file_num,file_name in enumerate(file_list)]))
thresh_string = 'Please enter threshold zscore for the following files'\
        '\n(separated by commas e.g. 0.6,0.3):\n{}'.format(file_str)
user_check = 'n'
while 'y' not in user_check:
    try:
        bad_trial_threshes = [float(x) for x in input(thresh_string).split(',')]
        user_check = input('Please confirm (y/n): {}\n'.format(bad_trial_threshes))
    except:
        raise Exception('Please check the formatting of your string')

# Mark trials which violate the threshold
bad_trials = [np.unique(np.where(zscore_trials_power[num] > bad_trial_threshes[num])[1]) \
            for num in range(len(bad_trial_threshes))]

# Replot trials with threshold and marked trials
# Plot zscore trial averaged power
fig, ax = plt.subplots(len(zscore_trials_power),1,sharex=True, figsize= (10,8))
for num, this_ax in enumerate(ax):
   this_ax.plot(zscore_trials_power[num].T,'x')
   this_ax.plot(np.arange(zscore_trials_power[num].shape[-1])[bad_trials[num]],
                zscore_trials_power[num][:,bad_trials[num]].T,'o',c='r')
   this_ax.hlines(bad_trial_threshes[num],0,zscore_trials_power[num].shape[-1],color='r')
   this_ax.title.set_text('Thresh {} , {}'.\
           format(bad_trial_threshes[num],os.path.basename(file_list[num])))
   this_ax.set_xlabel('Trial num');this_ax.set_ylabel('Zscore mean power')
plt.tight_layout()
fig.savefig(os.path.join(fin_output_dir,'{}_trial_removal.png'.format(fin_animal_name)))
plt.show()

#    _    _   _  _____     ___
#   / \  | \ | |/ _ \ \   / / \
#  / _ \ |  \| | | | \ \ / / _ \
# / ___ \| |\  | |_| |\ V / ___ \
#/_/   \_\_| \_|\___/  \_/_/   \_\
#

# This comparison can be done without using zscored power but it is used here
# to be consistent with the trials that are removed above

# Break raw data into tastes
mean_iti_taste_lfp_amplitude = [\
        np.array([\
        mean_iti_lfp_amplitude[num][:,\
            trial_time_data[num].loc[\
                trial_time_data[num].taste == taste].chronological]\
                    for taste in np.sort(trial_time_data[num].taste.unique())])\
                    for num in range(len(trial_time_data))]

# Break zscored data into tastes
zscore_iti_taste_power = [\
        np.array([\
        zscore_trials_power[num][:,\
            trial_time_data[num].loc[\
                trial_time_data[num].taste == taste].chronological]\
                    for taste in np.sort(trial_time_data[num].taste.unique())])\
                    for num in range(len(trial_time_data))]

# Add both to dataframe
nd_idx = [make_array_identifiers(data) for data in zscore_iti_taste_power]
mean_band_df = [pd.DataFrame({\
        'taste' : nd_idx[num][0],
        'band' : nd_idx[num][1],
        'trial' : nd_idx[num][2],
        'raw_power' : data[0].flatten(),
        'zscore_power' : data[1].flatten()}) \
                for num,data in enumerate(zip(mean_iti_taste_lfp_amplitude,zscore_iti_taste_power))]

# Merge with trial_time_data to have chronological values for each trial
mean_band_df = [mean_band_df[num].merge(trial_time_data[num],'inner') \
                    for num in range(len(mean_band_df))]

# Check "bad trials" still show up bad in dataframe
# Plot zscore trial averaged power
fig, ax = plt.subplots(len(mean_band_df),1,sharex=True)
for num, this_ax in enumerate(ax):
    this_ax.plot(mean_band_df[num].chronological,mean_band_df[num].zscore_power,'x')
    bad_inds = mean_band_df[num].chronological.isin(bad_trials[num])
    this_ax.plot(mean_band_df[num].chronological[bad_inds],
           mean_band_df[num].zscore_power[bad_inds],'o',c='r')
    this_ax.hlines(bad_trial_threshes[num],0,zscore_trials_power[num].shape[-1],color='r')
    this_ax.title.set_text('Thresh {} , {}'.\
           format(bad_trial_threshes[num],os.path.basename(file_list[num])))
    this_ax.set_xlabel('Trial num');this_ax.set_ylabel('Zscore mean power')
plt.tight_layout()
plt.show()

# Remove "bad trials" from dataframe
mean_band_df = [mean_band_df[num].loc[~mean_band_df[num].chronological.isin(bad_trials[num])] \
        for num in range(len(mean_band_df))]

# Cluster trials for every taste into bins for anova
# Not to be confused with trial_bin_total which cuts trials according to overall
# chronological order regardless of tastant
trial_bin_taste_num = 5
for dat in mean_band_df:
    dat['trial_bin_taste'] = pd.cut(dat.trial,
            bins = trial_bin_taste_num, include_lowest = True, labels = range(trial_bin_taste_num))

trial_bin_total_num = 5
for dat in mean_band_df:
    dat['trial_bin_total'] = pd.cut(dat.chronological,
            bins = trial_bin_total_num, include_lowest = True, labels = range(trial_bin_total_num))

########################################
## 2 Way ANOVA
########################################

# Perform 2-way ANOVA to look at differences in taste and trial_bin_taste
# for each dataset individually
taste_trial_anova = [\
    [dat.loc[dat.band == band_num].anova(dv = 'zscore_power', \
        between= ['trial_bin_taste','taste'])[['Source','p-unc','np2']][:3] \
            for band_num in np.sort(dat.band.unique())] \
        for dat in mean_band_df]

# Transform ANOVA output into dataframe
taste_trial_anova_df = [ [\
        pd.DataFrame({  'File' : os.path.basename(file_list[file_num]),
                        'Band' : band_num,
                        'Source' : band['Source'],
                        'p-unc' : band['p-unc'],
                        'np2' : band['np2']})
            for band_num,band in enumerate(file)]\
            for file_num, file in enumerate(taste_trial_anova)]

taste_trial_anova_df = pd.concat(list(chain(*taste_trial_anova_df)))
# Write output to HDF5 file
taste_trial_anova_df.to_hdf(output_hf5_name, '/taste_trial_two_way_anova')

# Show statistically significant values
#taste_trial_anova_df.loc[taste_trial_anova_df['p-unc'] < 0.01]

########################################
### Perform comparison across days
########################################

# Concatenate datasets to easify further processing
mean_band_df_cat_dataset = pd.concat([x.assign(dataset=num) \
                                for num,x in enumerate(mean_band_df)])

##### Comparison must be done on RAW POWER #####
# Since Zscoring is performed on datasets individually, they can't be compared #
# One way ANOVA to look at differences in LFP power between both datasets
# for each band separately

session_trial_anova =  [mean_band_df_cat_dataset.loc[mean_band_df_cat_dataset.band == band_num]\
        .anova(dv = 'raw_power', between= ['dataset'])[['Source','p-unc','np2']][:3] \
            for band_num in np.sort(mean_band_df_cat_dataset.band.unique())]

session_trial_anova_df = pd.concat([\
        pd.DataFrame({  'Band' : band_num,
                        'Source' : band['Source'],
                        'p-unc' : band['p-unc'],
                        'np2' : band['np2']})
            for band_num,band in enumerate(session_trial_anova)])

# Write output to HDF5 file
session_trial_anova_df.to_hdf(output_hf5_name, '/across_session_one_way_anova')

# Check which bands have significant differences
#session_trial_anova_df.loc[session_trial_anova_df['p-unc'] < 0.01]

########################################
### Pairwise comparisons
########################################

# Perform pairwise comparisons for for each combination of band and trial-bin
pairwise_comparisons_list = list(mean_band_df_cat_dataset.groupby(['band','trial_bin_total']))
pairwise_session_trial_ttests = [ pg.pairwise_ttests(
                                    dv = 'raw_power',
                                    between = ['dataset'],
                                    padjust = 'holm',
                                    parametric = 'False',
                                    data = data[1]) \
                            for data in pairwise_comparisons_list]

pairwise_session_ttest_frame = pd.concat(
        [pd.DataFrame(
            {   'band' : ind[0][0],
                'trial_bin_total' : ind[0][1],
                'p_val' : dat['p-unc']}
            )
        for ind,dat in zip(pairwise_comparisons_list,pairwise_session_trial_ttests)])
# Write output to HDF5 file
pairwise_session_ttest_frame.to_hdf(output_hf5_name, '/band_trial_pairwise_ttests')

# Report significant values
#pairwise_session_ttest_frame.loc[pairwise_session_ttest_frame.p_val < 0.01]

# ____  _       _
#|  _ \| | ___ | |_ ___
#| |_) | |/ _ \| __/ __|
#|  __/| | (_) | |_\__ \
#|_|   |_|\___/ \__|___/
#

########################################
# Test plot to confirm correct intervals are being selected from lfp
########################################
#mean_taste_whole_lfp = [np.mean(dat,axis=0) for dat in taste_whole_lfp]
#fig,ax = plt.subplots(2,1)
#ax[0].plot(mean_taste_whole_lfp[0])
#ax[0].vlines(delivery_time_list[0], np.min(mean_taste_whole_lfp[0]),np.max(mean_taste_whole_lfp[0]))
#for interval in iti_intervals[0]:
#    ax[0].axvspan(interval[0],interval[1],alpha=0.5,color='r')
#ax[1].plot(mean_taste_whole_lfp[1])
#ax[1].vlines(delivery_time_list[1], np.min(mean_taste_whole_lfp[1]),np.max(mean_taste_whole_lfp[1]))
#for interval in iti_intervals[1]:
#    ax[1].axvspan(interval[0],interval[1],alpha=0.5,color='r')
#plt.show()


########################################
# Plot mean power for trial bins for every band in both datasets
########################################
g = sns.FacetGrid(data = mean_band_df_cat_dataset,
            row = 'dataset', col = 'band', hue = 'taste', sharey = 'col')
g.map(sns.pointplot, 'trial_bin_taste','raw_power', ci='sd').add_legend()
plt.subplots_adjust(top=0.8)
title_str = ['{})'.format(num) + os.path.basename(name) + '\n' for num,name in enumerate(file_list)]
title_str = ''.join(title_str)
g.fig.suptitle(title_str)
g.savefig(os.path.join(fin_output_dir,'{}_trialbin_iti_power.png'.format(fin_animal_name)))

########################################
# Plot mean power chronologically for both days overlayed
########################################
g = sns.FacetGrid(data = mean_band_df_cat_dataset,
        row = 'band', hue='dataset',sharey=False, size = 4, aspect = 3)
g.map(sns.pointplot, 'chronological', 'raw_power')
g.savefig(os.path.join(fin_output_dir,'{}_chronological_iti_power.png'.format(fin_animal_name)))

########################################
# Plot Average power for ITI for every band
########################################

zscore_iti_lfp_amplitude = [np.array([zscore(dat,axis=None) for dat in this_file]) \
        for this_file in mean_channel_iti_lfp_amplitude]
for dat_num, dat in enumerate(zscore_iti_lfp_amplitude):
    dat[:,bad_trials[dat_num]] = 0

for num, data in enumerate(zscore_iti_lfp_amplitude):
    fig,ax = plt.subplots(1,data.shape[0], sharey=True, figsize = (10,10))
    for band in range(len(data)):
        plt.sca(ax[band])
        im = plt.imshow(data[band],
                interpolation='nearest',aspect='auto',
                cmap = 'jet',vmin=-3,vmax=3, origin = 'lower')
    plt.suptitle(os.path.basename(file_list[num]) + '\nZscoring for each band individually'\
            '\n Each plot: x = Time (ms), y = Trials')
    fig.subplots_adjust(bottom = 0.2)
    cbar_ax = fig.add_axes([0.15,0.05,0.7,0.02])
    plt.colorbar(im, cax = cbar_ax,orientation = 'horizontal', pad = 0.2)
    fig.text(0.5, 0.1, 'Bands', ha='center')
    fig.savefig(
            os.path.join(
                fin_output_dir,
                '{}_band_iti_power.png'.format(fin_animal_name + '_' + str(num))))

########################################
# Plot Average power for ITI (taste x band)
########################################
taste_band_power = [np.asarray([\
        data[:,trial_time_data[num].taste == taste,:] \
        for taste in np.sort(trial_time_data[num].taste.unique())])\
        for num,data in enumerate(mean_channel_iti_lfp_amplitude)]

# Zscore power to have a consistent colormap across subplots
zscore_taste_band_power = [np.asarray(\
        [zscore(data[:,band],axis=None) \
        for band in range(data.shape[1])]).swapaxes(0,1)\
        for data in taste_band_power]

# Blank out bad trials in data
# First pull out bad trials
bad_taste_trials = [dat.loc[dat.chronological.isin(bad_trials[dat_num])][['taste','trial']] \
                            for dat_num,dat in enumerate(trial_time_data)]
for dat_num, dat in enumerate(zscore_taste_band_power):
    for row in range(bad_taste_trials[dat_num].shape[0]):
        dat[bad_taste_trials[dat_num].iloc[row][0],:,bad_taste_trials[dat_num].iloc[row][1]] = 0

# Plot shit
for num,data in enumerate(zscore_taste_band_power):
    fig,ax = plt.subplots(data.shape[0],data.shape[1],sharex=True,sharey=True, figsize = (10,10))
    for taste in range(data.shape[0]):
        for band in range(data.shape[1]):
            plt.sca(ax[taste,band])
            im = plt.imshow(data[taste,band],
                    interpolation='nearest',aspect='auto',
                    cmap = 'jet',vmin=-3,vmax=3, origin = 'lower')
    plt.suptitle(os.path.basename(file_list[num]) + '\nZscoring for each band individually'\
            '\n Each plot: x = Time (ms), y = Trials')
    fig.subplots_adjust(bottom = 0.2)
    cbar_ax = fig.add_axes([0.15,0.05,0.7,0.02])
    plt.colorbar(im, cax = cbar_ax,orientation = 'horizontal', pad = 0.2)
    fig.text(0.5, 0.1, 'Bands', ha='center')
    fig.text(0.04, 0.5, 'Tastes', va='center', rotation='vertical')
    fig.savefig(
            os.path.join(
                fin_output_dir,
                '{}_taste_band_iti_power.png'.format(fin_animal_name + '_' + str(num))))
