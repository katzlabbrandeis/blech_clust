"""
NO OPTO HANDLING AS OF YET!!!

Analyses to include

- Firing rate calculation (scaled and unscaled)
- PSTH + Raster plots
- Taste responsive single neurons (magnitude and fraction)
- Taste discriminatory single neurons (ANOVA and cross-validated classifier) (magnitude and fraction)
- Single palatability single neurons and population (average and correlation with PCA of population) (magnitude and fraction)
- Dynamic neurons (ANOVA over time) (magnitude and fraction)
- Dynamic population (ANOVA over time on PCA/other latents)
"""

import numpy as np
import tables
import easygui
import sys
import os
import json
import glob
import itertools
from utils.blech_utils import entry_checker, imp_metadata, pipeline_graph_check
from utils.ephys_data import ephys_data
from utils.ephys_data import visualize as vz
import pandas as pd
from itertools import product
from scipy.stats import ttest_rel, zscore
import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg

# Ask for the directory where the hdf5 file sits, and change to that directory
# Get name of directory with the data files
dir_name = '/home/abuzarmahmood/projects/blech_clust/pipeline_testing/test_data_handling/test_data/KM45_5tastes_210620_113227_new'
metadata_handler = imp_metadata([[], dir_name])
# metadata_handler = imp_metadata(sys.argv)
dir_name = metadata_handler.dir_name

plot_dir = os.path.join(dir_name, 'unit_characteristic_plots')
if not os.path.exists(plot_dir):
	os.makedirs(plot_dir)

agg_plot_dir = os.path.join(plot_dir, 'aggregated')
if not os.path.exists(agg_plot_dir):
	os.makedirs(agg_plot_dir)

# Perform pipeline graph check
script_path = '/home/abuzarmahmood/projects/blech_clust/blech_unit_characteristics.py'
# script_path = os.path.realpath(__file__)
this_pipeline_check = pipeline_graph_check(dir_name)
this_pipeline_check.check_previous(script_path)
this_pipeline_check.write_to_log(script_path, 'attempted')

os.chdir(dir_name)

# Open the hdf5 file
# hf5 = tables.open_file(metadata_handler.hdf5_name, 'r+')
this_dat = ephys_data.ephys_data(dir_name)

# Extract taste dig-ins from experimental info file
info_dict = metadata_handler.info_dict
params_dict = metadata_handler.params_dict

# # Get the digital inputs/tastes available, 
# # then ask the user to rank them in order of palatability
# trains_dig_in = hf5.list_nodes('/spike_trains')
# palatability_rank = info_dict['taste_params']['pal_rankings']
# print(f'Palatability ranks : {palatability_rank}')
# 
taste_names = info_dict['taste_params']['tastes']
# tastes_set = set(taste_names)
# identities = [int(dict(zip(tastes_set,range(len(tastes_set))))[x]) for x in taste_names]
# print(f'Taste identities : {identities}')

##############################
# Get firing rates
##############################
psth_params = params_dict['psth_params']
this_dat.firing_rate_params = this_dat.default_firing_params
this_dat.firing_rate_params['window_size'] = psth_params['window_size']
this_dat.firing_rate_params['step_size'] = psth_params['step_size']
stim_time = params_dict['spike_array_durations'][0]
this_dat.get_firing_rates()
# List of len = n_tastes
# Each element is array of shape: n_trials x n_neurons x n_time_bins
spike_array = this_dat.spikes

# Plot rasters and psths
trial_lens = [x.shape[0] for x in spike_array]
cat_spikes = np.concatenate(spike_array, axis=0)
cat_firing = np.concatenate(this_dat.firing_list, axis=0)
mean_firing = np.stack([x.mean(axis=0) for x in this_dat.firing_list], axis=0)
zscore_cat_firing = zscore(cat_firing, axis=(0,2))

firing_t_vec = np.arange(cat_firing.shape[-1]) * this_dat.firing_rate_params['step_size']
firing_t_vec += psth_params['window_size']
cmap = plt.cm.get_cmap('tab10')
colors = [cmap(i) for i in range(len(spike_array))]
taste_blocks = np.concatenate([[0], np.cumsum(trial_lens)])
for nrn_ind in range(cat_spikes.shape[1]):
	fig, ax = plt.subplots(1,3, figsize=(20,5), sharex=True)
	vz.raster(ax[0], cat_spikes[:,nrn_ind,:], marker = '|', color = 'k')
	ax[0].set_title(f'Raster plot for neuron {nrn_ind}')
	for block_i in range(len(taste_blocks)-1):
		block_start = taste_blocks[block_i]
		block_end = taste_blocks[block_i+1]
		ax[0].axhspan(block_start, block_end, alpha=0.2, zorder=-1,
				color=colors[block_i])
		ax[1].axhline(block_start, color='r', linestyle='--', linewidth=2)
		ax[2].plot(firing_t_vec,
					mean_firing[block_i,nrn_ind],
					color=colors[block_i],
					label=taste_names[block_i],
					linewidth=2)
	ax[1].pcolorfast(
			firing_t_vec,
			np.arange(cat_firing.shape[0]),
			zscore_cat_firing[:,nrn_ind],
			cmap='jet',
			)
	ax[1].set_title(f'PSTH for neuron {nrn_ind}')
	ax[2].set_title(f'Mean PSTH for neuron {nrn_ind}')
	ax[2].legend()
	for this_ax in ax:
		this_ax.set_xlabel('Time (s)')
		this_ax.set_ylabel('Trials')
		this_ax.axvline(stim_time, color='r', linestyle='--', linewidth=2)
	plt.tight_layout()
	plt.savefig(os.path.join(plot_dir, f'neuron_{nrn_ind}_raster_psth.png'),
				bbox_inches='tight')
	plt.close()

# Plot firing overview
fig, ax = vz.firing_overview(
		data = cat_firing.swapaxes(0,1),
		t_vec = firing_t_vec,
		backend = 'pcolormesh',
		)
for this_ax in ax[:,0]:
	this_ax.set_ylabel('Trials')
for this_ax in ax[-1,:]:
	this_ax.set_xlabel('Time (ms)')
for this_ax in ax.flatten():
	this_ax.axvline(stim_time, color='k', linestyle='--', linewidth=2)
	for block_i in range(len(taste_blocks)-1):
		block_start = taste_blocks[block_i]
		block_end = taste_blocks[block_i+1]
		this_ax.axhline(block_start, color='r', linestyle='--', linewidth=2)
fig.suptitle('Firing overview')
plt.savefig(os.path.join(agg_plot_dir, 'firing_overview.png'),
			bbox_inches='tight')
plt.close()


##############################
# Responsiveness 
##############################
# Neuron counts as responsive if there is a difference from baseline
# for any taste
responsive_window = params_dict['responsiveness_pre_post_durations']

# Generate dataframe
n_tastes = len(spike_array)
n_neurons = spike_array[0].shape[1]
inds = list(product(np.arange(n_tastes), np.arange(n_neurons)))
# spike_list = []
post_spikes_list = []
pre_count_list = []
post_count_list = []
raw_spikes = []
for this_ind in inds:
	this_spike = spike_array[this_ind[0]][:,this_ind[1]]
	# spike_list.append(this_spike)
	pre_spikes = this_spike[..., stim_time-responsive_window[0]:stim_time]  
	post_spikes = this_spike[..., stim_time:stim_time+responsive_window[1]] 
	pre_count = pre_spikes.mean(axis=-1)
	post_count = post_spikes.mean(axis=-1)
	raw_spikes.append(this_spike)
	post_spikes_list.append(post_spikes)
	pre_count_list.append(pre_count)
	post_count_list.append(post_count)

spike_frame = pd.DataFrame(
		inds,
		columns = ['taste','neuron']
		)
spike_frame['raw_spikes'] = raw_spikes
spike_frame['post_spikes'] = post_spikes_list
spike_frame['pre_count'] = pre_count_list
spike_frame['post_count'] = post_count_list

# Calculate responsivess
alpha = 0.05
resp_test = ttest_rel
pval_list = []
for i, this_row in spike_frame.iterrows():
	pre_count = this_row.pre_count
	post_count = this_row.post_count
	_, pval = resp_test(pre_count, post_count)
	pval_list.append(pval)

spike_frame['resp_pval'] = pval_list
spike_frame['resp_sig'] = spike_frame.resp_pval < alpha 

# Mark neurons as responsive if any taste is responsive
resp_neurons = spike_frame.groupby('neuron').apply(
		lambda x: x.resp_sig.any()
	)

# Plot pvalues for all neurons across tastes and fraction of significant neurons
fig, ax = plt.subplots(1,2)
sns.stripplot(data=spike_frame, x='neuron', y='resp_pval', ax=ax[0])
ax[0].set_title('Pvalues for responsiveness')
ax[0].set_xlabel('Neuron')
ax[0].set_ylabel('Pvalue')
ax[0].axhline(alpha, color='r', linestyle='--')
ax[0].text(0, alpha, f'alpha={alpha}', color='r')
ax[1].bar(['Fraction responsive'], [resp_neurons.mean()])
ax[1].set_title('Fraction of responsive neurons\n'+\
		f'{resp_neurons.mean()} ({resp_neurons.sum()}/{resp_neurons.size})')
ax[1].set_ylabel('Fraction')
ax[1].set_ylim([0,1])
fig.supxlabel('Taste responsive neurons')
plt.tight_layout()
plt.savefig(os.path.join(agg_plot_dir, 'responsiveness.png'),
			bbox_inches='tight')
plt.close()

##############################
# Discriminability 
##############################
# 2 way ANOVA for each neuron over time and tastes
anova_bin_width = params_dict['discrim_analysis_params']['bin_width']
anova_bin_num = params_dict['discrim_analysis_params']['bin_num']
bin_lims = np.vectorize(int)(np.linspace(
	stim_time, 
	stim_time + (anova_bin_num*anova_bin_width), 
	anova_bin_num+1)) 

# For each row in spike_frame, chop post_spikes into bins
taste_list = []
neuron_list = []
bin_spike_list = []
bin_num_list = []
trial_num_list = []
for i, this_row in spike_frame.iterrows():
	this_spikes = this_row.raw_spikes
	taste_num = this_row.taste
	neuron_num = this_row.neuron
	for bin_ind in range(len(bin_lims)-1):
		bin_spikes = this_spikes[..., bin_lims[bin_ind]:bin_lims[bin_ind+1]]
		sum_bin_spikes = bin_spikes.sum(axis=-1)
		n_trials = len(sum_bin_spikes)
		taste_list.extend([taste_num]*n_trials)
		neuron_list.extend([neuron_num]*n_trials)
		bin_spike_list.extend(sum_bin_spikes)
		bin_num_list.extend([bin_ind]*n_trials)
		trial_num_list.extend(np.arange(n_trials))

discrim_frame = pd.DataFrame(
		dict(
			taste = taste_list,
			neuron = neuron_list,
			spike_count = bin_spike_list,
			bin_num = bin_num_list,
			trial_num = trial_num_list,
			)
		)

anova_list = []
for nrn_num in discrim_frame.neuron.unique():
	this_frame = discrim_frame.loc[discrim_frame.neuron == nrn_num]
	this_anova = pg.anova(
			data = this_frame,
			dv = 'spike_count',
			between = ['taste','bin_num'],
			)
	anova_list.append(this_anova)
p_val_list = []
for i, x in enumerate(anova_list):
	this_p = x[['Source','p-unc']] 
	this_p['neuron'] = i
	p_val_list.append(this_p)
p_val_frame = pd.concat(p_val_list)
p_val_frame = p_val_frame.loc[~p_val_frame.Source.str.contains('Residual')]
p_val_frame['sig'] = (p_val_frame['p-unc'] < alpha)*1

# Aggregate significance
# Drop interaction
p_val_frame = p_val_frame.loc[~p_val_frame.Source.isin(["taste * bin_num"])] 
taste_sig = p_val_frame.loc[p_val_frame.Source.str.contains('taste')].sig
bin_sig = p_val_frame.loc[p_val_frame.Source.str.contains('bin')].sig
sig_frame = pd.DataFrame(
		dict(
			taste_sig = taste_sig.values,
			bin_sig = bin_sig.values,
			)
		)
# sig_frame['neuron'] = p_val_frame.neuron.unique()

plt.figure()
g = sns.heatmap(sig_frame, cmap='coolwarm', cbar=True)
# cbar label
cbar = g.collections[0].colorbar
cbar.set_label('Significant (1 = yes, 0 = no)')
g.set_title('Significance of taste and time bins\n'+\
		f'alpha={alpha}')
ax = g.get_axes()
ax.set_xlabel('Variable')
ax.set_ylabel('Neuron')
plt.tight_layout()
plt.savefig(os.path.join(agg_plot_dir, 'discriminability.png'),
			bbox_inches='tight')
plt.close()
