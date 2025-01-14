"""
Check for drift in firing rate across the session.

1) Check using ANOVA across quarters of the session
2) Generate plots of firing rate across the session for each unit

**NOTE**: Do not worry about opto trials for now. Assuming opto trials
are well ditributed across the session, they should not affect the
mean firing rate.
"""

# Import stuff!
from utils.ephys_data import ephys_data
from utils.blech_utils import imp_metadata, pipeline_graph_check
import numpy as np
import tables
import sys
import os
import matplotlib
import pylab as plt
from scipy.stats import zscore
import pandas as pd
import pingouin as pg
import seaborn as sns
import glob
from sklearn.decomposition import PCA
from umap import UMAP
# Get script path
script_path = os.path.realpath(__file__)
script_dir_path = os.path.dirname(script_path)
blech_path = os.path.dirname(os.path.dirname(script_dir_path))
sys.path.append(blech_path)


def get_spike_trains(hf5_path):
    """
    Get spike trains from hdf5 file

    Inputs:
        hf5_path: path to hdf5 file

    Outputs:
        spike_trains: list of spike trains (trials, units, time)
    """
    with tables.open_file(hf5_path, 'r') as hf5:
        # Get the spike trains
        dig_ins = hf5.list_nodes('/spike_trains')
        dig_in_names = [dig_in._v_name for dig_in in dig_ins]
        spike_trains = [x.spike_array[:] for x in dig_ins]
    return spike_trains


def array_to_df(array, dim_names):
    """
    Convert array to dataframe with dimensions as columns

    Inputs:
        array: array to convert
        dim_names: list of names for each dimension

    Outputs:
        df: dataframe with dimensions as columns
    """
    assert len(
        dim_names) == array.ndim, 'Number of dimensions does not match number of names'

    inds = np.array(list(np.ndindex(array.shape)))
    df = pd.DataFrame(inds, columns=dim_names)
    df['value'] = array.flatten()
    return df


############################################################
# Initialize
############################################################
# Get name of directory with the data files
metadata_handler = imp_metadata(sys.argv)
dir_name = metadata_handler.dir_name

# Perform pipeline graph check
this_pipeline_check = pipeline_graph_check(dir_name)
this_pipeline_check.check_previous(script_path)
this_pipeline_check.write_to_log(script_path, 'attempted')

os.chdir(dir_name)
print(f'Processing : {dir_name}')

basename = os.path.basename(dir_name[:-1])

output_dir = os.path.join(dir_name, 'QA_output')
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

warnings_file_path = os.path.join(output_dir, 'warnings.txt')

############################################################
# Load Data
############################################################
# Open the hdf5 file
# A list of length [# of stimuli], containing arrays with dimensions = [trials, units, samples/trial]
spike_trains = get_spike_trains(metadata_handler.hdf5_name)

############################################################
# Perform Processing
############################################################
# Load params
params_dict = metadata_handler.params_dict
drift_params = params_dict['qa_params']['drift_params']

alpha = drift_params['alpha']
n_trial_bins = drift_params['n_trial_bins']
stim_t = params_dict['spike_array_durations'][0]
# Take period from stim_t - baseline_duration to stim_t
baseline_duration = drift_params['baseline_duration']
# Take period from stim_t to stim_t + trial_duration
trial_duration = drift_params['post_stim_duration']
bin_size = drift_params['plot_bin_size']

##############################
# Plot firing rate across session
##############################
# Flatten out spike trains for each taste
# A list of length [# of stimuli], containing arrays with dimensions = [units, trials, samples/trial]
unit_spike_trains = [np.swapaxes(x, 0, 1) for x in spike_trains]
# For sessions with uneven trials/stim, fill shorter stims with NAs
maxTrials = np.max([UnitN.shape[1] for UnitN in unit_spike_trains])
# Pad arrays with fewer trials to match maxTrials
for i, UnitN in enumerate(unit_spike_trains):
    num_trials = UnitN.shape[1]
    if num_trials < maxTrials:
        print('Uneven # of stimulus presentations: filling with NaN')
        # Calculate how much padding is needed
        pad_width = ((0, 0), (0, maxTrials - num_trials), (0, 0))
        unit_spike_trains[i] = np.pad(
            UnitN, pad_width, mode='constant', constant_values=np.nan)


# A list of length [# of stimuli], containing arrays with dimensions = [units, trials x samples/trial], with all trials/unit concatenated into one vector
long_spike_trains = [x.reshape(x.shape[0], -1) for x in unit_spike_trains]

# Bin data to plot
# A list of length [stimuli], containing arrays of concatenated trials as long_spike_trains, binned per sorting_params
binned_spike_trains = [np.reshape(
    x, (x.shape[0], -1, bin_size)).sum(axis=2) for x in long_spike_trains]

# Group by neuron across tastes
# A list of length [units], containing tuples of length [stimuli], containing the concatenated, binned, spike train vectors.
plot_spike_trains = list(zip(*binned_spike_trains))
zscore_binned_spike_trains = [
    zscore(x, axis=-1, nan_policy='omit') for x in plot_spike_trains]
if len(plot_spike_trains) > 9:
    plotHeight = len(plot_spike_trains)+1
else:
    plotHeight = 10


# Plot heatmaps of all tastes, both raw data and zscored
fig, ax = plt.subplots(len(plot_spike_trains), 2, figsize=(10, plotHeight))
for i in range(len(plot_spike_trains)):
    ax[i, 0].imshow(plot_spike_trains[i], aspect='auto', interpolation='none')
    ax[i, 1].imshow(zscore_binned_spike_trains[i],
                    aspect='auto', interpolation='none')
    ax[i, 0].set_title(f'Unit {i} Raw')
    ax[i, 1].set_title(f'Unit {i} Zscored')
    ax[i, 0].set_ylabel('Taste')
fig.suptitle('Binned Spike Heatmaps \n' + basename +
             '\n' + 'Bin Size: ' + str(bin_size) + ' ms')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'binned_spike_heatmaps.png'))
plt.close()

# Plot timeseries of above data as well
fig, ax = plt.subplots(len(plot_spike_trains), 2, figsize=(10, plotHeight))
for i in range(len(plot_spike_trains)):
    ax[i, 0].plot(np.array(plot_spike_trains[i]).T, alpha=0.7)
    ax[i, 1].plot(zscore_binned_spike_trains[i].T, alpha=0.7)
    ax[i, 0].set_title(f'Unit {i} Raw')
    ax[i, 1].set_title(f'Unit {i} Zscored')
    ax[i, 0].set_ylabel('Taste')
fig.suptitle('Binned Spike Timeseries \n' + basename +
             '\n' + 'Bin Size: ' + str(bin_size) + ' ms')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'binned_spike_timeseries.png'))
plt.close()

##############################
# Perform ANOVA on baseline and post-stimulus firing rates separately
##############################

##############################
# Baseline

# For baseline, check across trials and tastes both
baseline_spike_trains = [
    x[..., stim_t-baseline_duration:stim_t] for x in spike_trains]
baseline_counts = [x.sum(axis=-1) for x in baseline_spike_trains]
baseline_counts_df_list = [array_to_df(
    x, ['trial', 'unit']) for x in baseline_counts]
# Add taste column
for i in range(len(baseline_counts_df_list)):
    baseline_counts_df_list[i]['taste'] = i
# Add indicator for trial bins
for i in range(len(baseline_counts_df_list)):
    baseline_counts_df_list[i]['trial_bin'] = pd.cut(
        baseline_counts_df_list[i]['trial'], n_trial_bins, labels=False)
baseline_counts_df = pd.concat(baseline_counts_df_list, axis=0)

# Plot baseline firing rates
g = sns.catplot(data=baseline_counts_df,
                x='trial_bin', y='value',
                row='taste', col='unit',
                kind='bar', sharey=False,
                )
fig = plt.gcf()
fig.suptitle('Baseline Firing Rates \n' +
             basename + '\n' +
             'Trial Bin Count: ' + str(n_trial_bins) + '\n' +
             'Baseline limits: ' + str(stim_t-baseline_duration) + ' to ' + str(stim_t) + ' ms')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'baseline_firing_rates.png'))
plt.close()

# Perform 2-way ANOVA on baseline firing rates across trial bins and tastes
grouped_list = baseline_counts_df.groupby('unit')
group_ids = [x[0] for x in grouped_list]
group_dat = [x[1] for x in grouped_list]
anova_out = [pg.anova(data=x, dv='value', between=[
                      'trial_bin', 'taste'], detailed=True) for x in group_dat]

p_vals = [x[['Source', 'p-unc']] for x in anova_out]
# Set source as index
for i in range(len(p_vals)):
    p_vals[i].set_index('Source', inplace=True)
# Transpose and add unit column
p_vals = [x.T for x in p_vals]
for i in range(len(p_vals)):
    p_vals[i]['unit'] = group_ids[i]
# Concatenate into single dataframe
p_val_frame = pd.concat(p_vals, axis=0)
p_val_frame.reset_index(inplace=True, drop=True)

# Output p-values
p_val_frame.to_csv(os.path.join(output_dir, 'baseline_drift_p_vals.csv'))

# Generate a plot of the above array marking significant p-values
# First, get the p-values
wanted_cols = ['trial_bin', 'taste', 'trial_bin * taste']
p_val_mat = p_val_frame[wanted_cols].values
# Then, get the significant p-values
sig_p_val_mat = p_val_mat < alpha
# Plot the significant p-values
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(sig_p_val_mat, aspect='auto', interpolation='none', cmap='gray')
ax.set_xticks(np.arange(len(wanted_cols)))
ax.set_yticks(np.arange(len(group_ids)))
ax.set_xticklabels(wanted_cols)
ax.set_yticklabels(group_ids)
ax.set_title('Significant p-values for 2-way ANOVA on Baseline Firing Rates')
ax.set_xlabel('Comparison Type')
ax.set_ylabel('Unit')
# Add p-values to plot
for i in range(len(group_ids)):
    for j in range(len(wanted_cols)):
        this_str_color = 'k' if sig_p_val_mat[i, j] else 'w'
        ax.text(j, i, str(round(p_val_mat[i, j], 3)),
                ha='center', va='center', color=this_str_color)
plt.savefig(os.path.join(output_dir, 'baseline_drift_p_val_heatmap.png'))
plt.close()

# If any significant p-values, write to warning file
if np.any(sig_p_val_mat):
    out_rows_inds = np.any(sig_p_val_mat, axis=1)
    out_rows = p_val_frame.iloc[out_rows_inds]

    with open(warnings_file_path, 'a') as f:
        print('=== Baseline Drift Warning ===', file=f)
        print('2-way ANOVA on baseline firing rates across trial bins and tastes', file=f)
        print('Baseline limits: ' + str(stim_t-baseline_duration) +
              ' to ' + str(stim_t) + ' ms', file=f)
        print('Trial Bin Count: ' + str(n_trial_bins), file=f)
        print('alpha: ' + str(alpha), file=f)
        print('\n', file=f)
        print(out_rows, file=f)
        print('\n', file=f)
        print('=== End Baseline Drift Warning ===', file=f)
        print('\n', file=f)


##############################
# Post-stimulus

# For post-stimulus, check across trials only
post_spike_trains = [x[..., stim_t:stim_t+trial_duration]
                     for x in spike_trains]
post_counts = [x.sum(axis=-1) for x in post_spike_trains]
post_counts_df_list = [array_to_df(x, ['trial', 'unit']) for x in post_counts]
# Add taste column
for i in range(len(post_counts_df_list)):
    post_counts_df_list[i]['taste'] = i
# Add indicator for trial bins
for i in range(len(post_counts_df_list)):
    post_counts_df_list[i]['trial_bin'] = pd.cut(
        post_counts_df_list[i]['trial'], n_trial_bins, labels=False)
post_counts_df = pd.concat(post_counts_df_list, axis=0)

# Plot post-stimulus firing rates
g = sns.catplot(data=post_counts_df,
                x='trial_bin', y='value',
                col='unit', hue='taste',
                kind='bar', sharey=False,
                )
fig = plt.gcf()
fig.suptitle('Post-stimulus Firing Rates \n' +
             basename + '\n' +
             'Trial Bin Count: ' + str(n_trial_bins) + '\n' +
             'Post-stimulus limits: ' + str(stim_t) + ' to ' + str(stim_t+trial_duration) + ' ms')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'post_firing_rates.png'))
plt.close()

# Perform repeated measures ANOVA on post-stimulus firing rates across trial bins
# Taste is the repeated measure

grouped_list = post_counts_df.groupby('unit')
group_ids = [x[0] for x in grouped_list]
group_dat = [x[1] for x in grouped_list]
anova_out = [pg.rm_anova(data=x, dv='value', within='trial_bin',
                         subject='taste', detailed=True) for x in group_dat]

p_vals = [x[['Source', 'p-unc']] for x in anova_out]
# Set source as index
for i in range(len(p_vals)):
    p_vals[i].set_index('Source', inplace=True)
# Transpose and add unit column
p_vals = [x.T for x in p_vals]
for i in range(len(p_vals)):
    p_vals[i]['unit'] = group_ids[i]
# Concatenate into single dataframe
p_val_frame = pd.concat(p_vals, axis=0)
p_val_frame.reset_index(inplace=True, drop=True)

# Output p-values
p_val_frame.to_csv(os.path.join(output_dir, 'post_drift_p_vals.csv'))

# Generate a plot of the above array marking significant p-values
# First, get the p-values
p_val_vec = p_val_frame['trial_bin'].values
# Then, get the significant p-values
sig_p_val_vec = p_val_vec < 0.05
# Then generate figure
plt.figure()
plt.imshow(sig_p_val_vec.reshape((1, -1)), cmap='gray')
plt.title('Significant p-values')
plt.xlabel('Unit')
plt.ylabel('Trial Bin')
# Add unit labels
plt.xticks(np.arange(len(group_ids)), group_ids)
# Add p-values to plot
for i in range(len(p_val_vec)):
    this_str_color = 'black' if sig_p_val_vec[i] else 'white'
    plt.text(i, 0, str(round(p_val_vec[i], 3)),
             horizontalalignment='center', verticalalignment='center',
             color=this_str_color)
plt.savefig(os.path.join(output_dir, 'post_drift_p_vals.png'))
plt.close()

# If any significant p-values, write to warning file
if np.any(sig_p_val_vec):
    out_rows_inds = sig_p_val_vec
    out_rows = p_val_frame.iloc[out_rows_inds]

    with open(warnings_file_path, 'a') as f:
        print('=== Post-stimulus Drift Warning ===', file=f)
        print('Repeated measures ANOVA on post-stimulus firing rates across trial bins and tastes', file=f)
        print('Post-stimulus limits: ' + str(stim_t) + ' to ' +
              str(stim_t+trial_duration) + ' ms', file=f)
        print('Trial Bin Count: ' + str(n_trial_bins), file=f)
        print('alpha: ' + str(alpha), file=f)
        # print('\n', file=f)
        print(out_rows, file=f)
        print('\n', file=f)
        print('=== End Post-stimulus Drift Warning ===', file=f)
        print('\n', file=f)

############################################################
# Perform PCA on firing rates across trials
############################################################
dat = ephys_data.ephys_data(dir_name)
dat.get_firing_rates()
# each element is a 3D array of shape (n_trials, n_neurons, n_timepoints)
firing_list = dat.firing_list
# Normalize for each neuron
n_neurons = firing_list[0].shape[1]
norm_firing_list = []
for i in range(len(firing_list)):
    this_firing = firing_list[i]
    norm_firing = np.zeros_like(this_firing)
    for j in range(n_neurons):
        norm_firing[:, j, :] = zscore(this_firing[:, j, :], axis=None)
    norm_firing_list.append(norm_firing)

# shape: (n_trials, n_neurons * n_timepoints)
long_firing_list = [x.reshape(x.shape[0], -1) for x in norm_firing_list]

# Perform PCA on long_firing_list
pca_firing_list = [PCA(n_components=1, whiten=True).fit_transform(x)
                   for x in long_firing_list]
umap_firing_list = [UMAP(n_components=1).fit_transform(x)
                    for x in long_firing_list]
umap_zscore = [zscore(x, axis=None) for x in umap_firing_list]

# Plot PCA and UMAP results
fig, ax = plt.subplots(2, 1, figsize=(5, 5), sharex=True)
for i in range(len(pca_firing_list)):
    ax[0].plot(pca_firing_list[i], alpha=0.7)
    ax[1].plot(umap_zscore[i], alpha=0.7)
    ax[0].set_title('PCA')
    ax[1].set_title('UMAP')
ax[-1].set_xlabel('Trial num')
fig.suptitle('PCA and UMAP of Firing Rates \n' + basename)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'pca_umap_firing_rates.png'))
plt.close()

# Write successful execution to log
this_pipeline_check.write_to_log(script_path, 'completed')
