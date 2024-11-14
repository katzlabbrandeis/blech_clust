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
from tqdm import tqdm

# Ask for the directory where the hdf5 file sits, and change to that directory
# Get name of directory with the data files
dir_name = '/media/fastdata/NM_sorted_data/NM51/NM51_2500ms_161030_130155_copy'
# dir_name = '/home/abuzarmahmood/projects/blech_clust/pipeline_testing/test_data_handling/test_data/KM45_5tastes_210620_113227_new'
# dir_name = '/media/bigdata/Abuzar_Data/bla_gc/AM11/AM11_4Tastes_191030_114043_copy'
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
script_path = '/home/abuzarmahmood/Desktop/blech_clust/blech_unit_characteristics.py'
# script_path = os.path.realpath(__file__)
this_pipeline_check = pipeline_graph_check(dir_name)
this_pipeline_check.check_previous(script_path)
this_pipeline_check.write_to_log(script_path, 'attempted')

os.chdir(dir_name)

# Open the hdf5 file
# hf5 = tables.open_file(metadata_handler.hdf5_name, 'r+')
# from importlib import reload
# reload(ephys_data)
this_dat = ephys_data.ephys_data(dir_name)
# this_dat.get_trial_info_frame()
# this_dat.check_laser()
# this_dat.separate_laser_data()

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
this_dat.get_sequestered_data()

mean_seq_firing = this_dat.sequestered_firing_frame.groupby(
		['taste_num','laser_tuple','neuron_num','time_num']).mean()
mean_seq_firing = mean_seq_firing.reset_index()
mean_seq_firing.drop(columns=['trial_num'], inplace=True)

# firing_t_vec = np.arange(cat_firing.shape[-1]) * this_dat.firing_rate_params['step_size']
firing_t_vec = np.arange(mean_seq_firing.time_num.max()+1) * this_dat.firing_rate_params['step_size']
firing_t_vec += psth_params['window_size']
firing_t_vec -= stim_time

mean_seq_firing['time_val'] = [firing_t_vec[x] for x in mean_seq_firing.time_num]
mean_seq_firing['taste'] = [taste_names[x] for x in mean_seq_firing.taste_num]

# Plot firing rates
laser_conditions = mean_seq_firing.laser_tuple.unique()
n_laser_conditions = len(laser_conditions)

# List of len = n_tastes
# Each element is array of shape: n_trials x n_neurons x n_time_bins
spike_array = this_dat.spikes
cmap = plt.cm.get_cmap('tab10')
colors = [cmap(i) for i in range(len(spike_array))]
for nrn_ind in tqdm(mean_seq_firing.neuron_num.unique()):
	fig, ax = plt.subplots(n_laser_conditions,2, figsize=(10,5*n_laser_conditions),
						sharex=True, sharey='col')
	for i, laser_cond in enumerate(laser_conditions):
		this_firing = mean_seq_firing.loc[
				(mean_seq_firing.neuron_num == nrn_ind) &
				(mean_seq_firing.laser_tuple == laser_cond)
				]
		this_spikes = this_dat.sequestered_spikes_frame.loc[
				(this_dat.sequestered_spikes_frame.neuron_num == nrn_ind) &
				(this_dat.sequestered_spikes_frame.laser_tuple == laser_cond)
				]
		this_spikes.sort_values(['taste_num','trial_num'], inplace=True)
		this_spikes['cum_trial_num'] = \
				this_spikes['taste_num']*(this_spikes['trial_num'].max()+1) + this_spikes['trial_num'] 
		this_spikes['cum_trial_num'] += 0.5
		this_spikes['time_num'] -= stim_time
		trial_lens = this_spikes.groupby('taste_num').trial_num.max() + 1
		taste_blocks = np.concatenate([[0], np.cumsum(trial_lens)])
		sns.lineplot(
				data = this_firing, 
				x = 'time_val',
				y = 'firing',
				hue = 'taste',
				ax = ax[i,0],
				)
		sns.scatterplot(
				data = this_spikes,
				x = 'time_num',
				y = 'cum_trial_num',
				color = 'k',
				marker = '|',
				ax = ax[i,1],
				legend = False,
				)
		for block_i in range(len(taste_blocks)-1):
			block_start = taste_blocks[block_i]
			block_end = taste_blocks[block_i+1]
			ax[i,1].axhspan(block_start, block_end, alpha=0.2, zorder=-1,
					color=colors[block_i])
		for this_ax in ax[i,:]:
			if i == len(ax)-1:
				this_ax.set_xlabel('Time (ms)')
			this_ax.axvline(0, color='r', linestyle='--', linewidth=2)
			this_ax.set_title(f'Laser condition {laser_cond}')
			this_ax.legend()
	# g.set_title(f'Neuron {nrn_ind}')
	fig.suptitle(f'Neuron {nrn_ind}')
	plt.savefig(os.path.join(plot_dir, f'neuron_{nrn_ind}_firing_rate.png'),
			 bbox_inches='tight')
	plt.close()

# If more than one laser condition, plot firing rates for each taste separately
# with laser conditions as hue
if n_laser_conditions > 1:
	for nrn_ind in tqdm(mean_seq_firing.neuron_num.unique()): 
		this_firing = mean_seq_firing.loc[
				mean_seq_firing.neuron_num == nrn_ind
				]
		# this_firing = this_dat.sequestered_firing_frame.loc[
		# 		this_dat.sequestered_firing_frame.neuron_num == nrn_ind
		# 		]
		this_firing.reset_index(inplace=True)
		g = sns.relplot(
				data = this_firing, 
				x = 'time_val',
				y = 'firing',
				hue = 'laser_tuple',
				col = 'taste',
				kind = 'line',
				linewidth = 3,
				)
		# leg = g._legend
		# leg.set_bbox_to_anchor([0.5, 0.5])
		# Remove figure legend
		g._legend.remove()
		for i, this_ax in enumerate(g.axes.flatten()):
			this_ax.set_xlabel('Time (ms)')
			this_ax.axvline(0, color='r', linestyle='--', linewidth=2)
			if i == 0:
				this_ax.set_ylabel('Firing rate (Hz)')
				this_ax.legend(title='Laser condition (lag, duration)')
		fig.suptitle(f'Neuron {nrn_ind}')
		plt.savefig(os.path.join(plot_dir, f'neuron_{nrn_ind}_opto_overlay.png'),
				 bbox_inches='tight')
		plt.close()

# this_dat.get_firing_rates()

# # Plot rasters and psths
# trial_lens = [x.shape[0] for x in spike_array]
# cat_spikes = np.concatenate(spike_array, axis=0)
# cat_firing = np.concatenate(this_dat.firing_list, axis=0)
# mean_firing = np.stack([x.mean(axis=0) for x in this_dat.firing_list], axis=0)
# zscore_cat_firing = zscore(cat_firing, axis=(0,2))
# 
# cmap = plt.cm.get_cmap('tab10')
# colors = [cmap(i) for i in range(len(spike_array))]
# taste_blocks = np.concatenate([[0], np.cumsum(trial_lens)])
# for nrn_ind in range(cat_spikes.shape[1]):
# 	fig, ax = plt.subplots(1,3, figsize=(20,5), sharex=True)
# 	vz.raster(ax[0], cat_spikes[:,nrn_ind,:], marker = '|', color = 'k')
# 	ax[0].set_title(f'Raster plot for neuron {nrn_ind}')
# 	for block_i in range(len(taste_blocks)-1):
# 		block_start = taste_blocks[block_i]
# 		block_end = taste_blocks[block_i+1]
# 		ax[0].axhspan(block_start, block_end, alpha=0.2, zorder=-1,
# 				color=colors[block_i])
# 		ax[1].axhline(block_start, color='r', linestyle='--', linewidth=2)
# 		ax[2].plot(firing_t_vec,
# 					mean_firing[block_i,nrn_ind],
# 					color=colors[block_i],
# 					label=taste_names[block_i],
# 					linewidth=2)
# 	ax[1].pcolorfast(
# 			firing_t_vec,
# 			np.arange(cat_firing.shape[0]),
# 			zscore_cat_firing[:,nrn_ind],
# 			cmap='jet',
# 			)
# 	ax[1].set_title(f'PSTH for neuron {nrn_ind}')
# 	ax[2].set_title(f'Mean PSTH for neuron {nrn_ind}')
# 	ax[2].legend()
# 	for this_ax in ax:
# 		this_ax.set_xlabel('Time (s)')
# 		this_ax.set_ylabel('Trials')
# 		this_ax.axvline(stim_time, color='r', linestyle='--', linewidth=2)
# 	plt.tight_layout()
# 	plt.savefig(os.path.join(plot_dir, f'neuron_{nrn_ind}_raster_psth.png'),
# 				bbox_inches='tight')
# 	plt.close()

trial_lens = [x.shape[0] for x in spike_array]
taste_blocks = np.concatenate([[0], np.cumsum(trial_lens)])
# Plot firing overview
fig, ax = vz.firing_overview(
		data = cat_firing.swapaxes(0,1),
		t_vec = firing_t_vec,
		backend = 'pcolormesh',
		figsize = (10,10),
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
# (ms of pres-stim, ms of post-stim)
responsive_window = params_dict['responsiveness_pre_post_durations']
responsive_inds = ((stim_time-responsive_window[0], stim_time),
				   (stim_time, stim_time+responsive_window[1]))
min_ind, max_ind = min(responsive_inds[0]), max(responsive_inds[1])
seq_spikes_frame = this_dat.sequestered_spikes_frame.copy()
# Cut to responsive window
seq_spikes_frame = seq_spikes_frame.loc[
		(seq_spikes_frame.time_num >= min_ind) &
		(seq_spikes_frame.time_num < max_ind)
		]
seq_spikes_frame['spikes'] = 1
# mark pre and post stim periods
seq_spikes_frame['post_stim'] = seq_spikes_frame['time_num'] >= stim_time
## NOTE: DON'T SUM SPIKES...NOT VALID UNLESS PRE-STIM and POST-STIM PERIODS ARE 
# OF EQUAL LENGTH
seq_spike_counts = seq_spikes_frame.groupby(
		['trial_num','neuron_num','taste_num','laser_tuple','post_stim']).mean()
seq_spike_counts.reset_index(inplace=True)
seq_spike_counts.drop(columns = ['time_num'], inplace=True)

# Have to add zeros where no spikes were seen
firing_frame = this_dat.sequestered_firing_frame.copy()
index_cols = ['trial_num','neuron_num','taste_num','laser_tuple']
firing_frame_group_list = list(firing_frame.groupby(index_cols))
firing_frame_group_inds = [x[0] for x in firing_frame_group_list]
# For each of group_inds, check if it is in seq_spike_counts
# If not add 0
seq_spike_counts.set_index(index_cols+['post_stim'], inplace=True)
for this_ind in tqdm(firing_frame_group_inds):
	# Iterate of post_stim
	for this_post_stim in [False, True]:
		fin_ind = tuple((*this_ind, this_post_stim))
		if fin_ind not in seq_spike_counts.index:
			this_row = pd.Series(
					dict(
						spikes = 0
						),
					name = fin_ind
					)
			# seq_spike_counts.loc[this_ind] = 0
			seq_spike_counts = seq_spike_counts.append(this_row)
seq_spike_counts.reset_index(inplace=True)

# For each neuron_num, taste_num, laser_tuple
# calculate significance of difference between post_stim groups
group_cols = ['neuron_num','taste_num','laser_tuple']
group_list = list(seq_spike_counts.groupby(group_cols))
group_inds = [x[0] for x in group_list]
group_frames = [x[1] for x in group_list]
pval_list = []
for this_frame in tqdm(group_frames):
	this_pval = ttest_rel(
			this_frame.loc[this_frame.post_stim].spikes,
			this_frame.loc[~this_frame.post_stim].spikes,
			)[1]
	pval_list.append(this_pval)

resp_frame = pd.DataFrame(
		columns = group_cols,
		data = group_inds
		)
resp_frame['resp_pval'] = pval_list
# Fillna with 1
resp_frame.fillna(1, inplace=True)
alpha = 0.05
resp_frame['resp_sig'] = resp_frame.resp_pval < alpha
n_comparisons = resp_frame.groupby('neuron_num')['resp_sig'].count().unique()
if len(n_comparisons) > 1:
	raise ValueError('Number of comparisons not consistent')
n_comparisons = n_comparisons[0]
corrected_alpha = alpha # / n_comparisons
resp_neurons = resp_frame.groupby(['neuron_num','laser_tuple']).agg(
		{'resp_pval': lambda x: any([y<corrected_alpha for y in x])})
resp_neurons.reset_index(inplace=True) 
resp_neurons['resp_pval'] *= 1

# Convert resp_neurons to pivot table
row_var = 'neuron_num'
col_var = 'laser_tuple'
val_var = 'resp_pval'

resp_neurons_pivot = resp_neurons.pivot(
		index = row_var,
		columns = col_var,
		values = val_var
		)

plt.figure(figsize=(5,10))
g = sns.heatmap(resp_neurons_pivot, 
			cmap='coolwarm', cbar=True,
			linewidth = 0.5,
			cbar_kws = {'label' : 'Significant (1 = yes, 0 = no)'},
				)
plt.title('Responsiveness of neurons\n'+\
		f'alpha={alpha}, corrected alpha={corrected_alpha}')
plt.xlabel('Laser condition\n(lag, duration)')
plt.ylabel('Neuron')
plt.tight_layout()
plt.savefig(os.path.join(agg_plot_dir, 'responsiveness_heatmap.png'),
			bbox_inches='tight')
plt.close()


# # Generate dataframe
# n_tastes = len(spike_array)
# n_neurons = spike_array[0].shape[1]
# inds = list(product(np.arange(n_tastes), np.arange(n_neurons)))
# # spike_list = []
# post_spikes_list = []
# pre_count_list = []
# post_count_list = []
# raw_spikes = []
# for this_ind in inds:
# 	this_spike = spike_array[this_ind[0]][:,this_ind[1]]
# 	# spike_list.append(this_spike)
# 	pre_spikes = this_spike[..., stim_time-responsive_window[0]:stim_time]  
# 	post_spikes = this_spike[..., stim_time:stim_time+responsive_window[1]] 
# 	pre_count = pre_spikes.mean(axis=-1)
# 	post_count = post_spikes.mean(axis=-1)
# 	raw_spikes.append(this_spike)
# 	post_spikes_list.append(post_spikes)
# 	pre_count_list.append(pre_count)
# 	post_count_list.append(post_count)
# 
# spike_frame = pd.DataFrame(
# 		inds,
# 		columns = ['taste','neuron']
# 		)
# spike_frame['raw_spikes'] = raw_spikes
# spike_frame['post_spikes'] = post_spikes_list
# spike_frame['pre_count'] = pre_count_list
# spike_frame['post_count'] = post_count_list

# # Calculate responsivess
# alpha = 0.05
# resp_test = ttest_rel
# pval_list = []
# for i, this_row in spike_frame.iterrows():
# 	pre_count = this_row.pre_count
# 	post_count = this_row.post_count
# 	_, pval = resp_test(pre_count, post_count)
# 	pval_list.append(pval)
# 
# spike_frame['resp_pval'] = pval_list
# spike_frame['resp_sig'] = spike_frame.resp_pval < alpha 

# Mark neurons as responsive if any taste is responsive
# resp_neurons = spike_frame.groupby('neuron').apply(
# 		lambda x: x.resp_sig.any()
# 	)

# Plot pvalues for all neurons across tastes and fraction of significant neurons
resp_num_laser = pd.DataFrame(resp_neurons.groupby('laser_tuple')['resp_pval'].sum())
resp_num_laser.reset_index(inplace=True)
resp_num_laser['total_count'] = resp_neurons.neuron_num.nunique()
resp_num_laser.rename(columns = {'resp_pval' : 'sig_count'}, inplace=True)
resp_frac_laser = pd.DataFrame(resp_neurons.groupby('laser_tuple')['resp_pval'].mean())
resp_frac_laser.reset_index(inplace=True)
fig, ax = plt.subplots(2,1)
sns.stripplot(
		data=resp_frame, 
		x='neuron_num', 
		y='resp_pval', 
		ax=ax[0])
ax[0].set_title('Pvalues for responsiveness')
ax[0].set_xlabel('Neuron')
ax[0].set_ylabel('Pvalue')
ax[0].axhline(alpha, color='r', linestyle='--')
ax[0].text(0, corrected_alpha, f'bonf alpha={alpha}', color='r')
# ax[1].bar(['Fraction responsive'], [(resp_neurons*1).mean()][0])
sns.barplot(
		data = resp_frac_laser,
		x = 'laser_tuple',
		y = 'resp_pval',
		ax = ax[1]
		)
ax[1].set_title('Fraction of responsive neurons\n'+\
		str(resp_num_laser))
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

seq_spikes_frame = this_dat.sequestered_spikes_frame.copy()
min_lim, max_lim = min(bin_lims), max(bin_lims)
# Cut frame to lims
seq_spikes_frame = seq_spikes_frame.loc[
		(seq_spikes_frame.time_num >= min_lim) &
		(seq_spikes_frame.time_num < max_lim)
		]
# Mark bins
seq_spikes_frame['bin_num'] = pd.cut(
		seq_spikes_frame.time_num,
		bin_lims,
		labels = np.arange(anova_bin_num),
		include_lowest = True,
		)
# Get counts per bin
seq_spikes_frame['spikes'] = 1
seq_spike_counts = seq_spikes_frame.groupby(
		['trial_num','neuron_num','taste_num','laser_tuple','bin_num']).mean()
seq_spike_counts.reset_index(inplace=True)
seq_spike_counts.drop(columns = ['time_num'], inplace=True)
seq_spike_counts.fillna(0, inplace=True)

# Make sure all inds are present
seq_spike_counts.set_index(index_cols+['bin_num'], inplace=True)
for this_ind in tqdm(firing_frame_group_inds):
	# Iterate of post_stim
	for bin_num in range(anova_bin_num): 
		fin_ind = tuple((*this_ind, bin_num))
		if fin_ind not in seq_spike_counts.index:
			this_row = pd.Series(
					dict(
						spikes = 0
						),
					name = fin_ind
					)
			# seq_spike_counts.loc[this_ind] = 0
			seq_spike_counts = seq_spike_counts.append(this_row)
seq_spike_counts.reset_index(inplace=True)

# For each neuron_num and laser_tuple, run 2-way ANOVA with taste_num and bin_num  
# as factors
group_cols = ['neuron_num','laser_tuple']
group_list = list(seq_spike_counts.groupby(group_cols))
group_inds = [x[0] for x in group_list]
group_frames = [x[1] for x in group_list]
pval_list = []
for this_frame in tqdm(group_frames):
	anova_out = pg.anova(
			data = this_frame,
			dv = 'spikes',
			between = ['taste_num','bin_num'],
			)
	anova_out = anova_out.loc[anova_out.Source != 'Residual']
	this_pval = anova_out[['Source','p-unc']]
	nrn_num = this_frame.neuron_num.unique()[0]
	laser_tuple = this_frame.laser_tuple.unique()[0]
	this_pval['neuron_num'] = nrn_num
	this_pval['laser_tuple'] = laser_tuple
	pval_list.append(this_pval)

p_val_frame = pd.concat(pval_list)
# Drop interaction
p_val_frame = p_val_frame.loc[~p_val_frame.Source.isin(["taste_num * bin_num"])] 
alpha = 0.05
p_val_frame['sig'] = (p_val_frame['p-unc'] < alpha)*1
taste_sig = pd.DataFrame(p_val_frame.loc[p_val_frame.Source.str.contains('taste')].sig)
bin_sig = pd.DataFrame(p_val_frame.loc[p_val_frame.Source.str.contains('bin')].sig)
index_col_names = ['neuron_num','laser_tuple']
index_cols = p_val_frame.loc[p_val_frame.Source.str.contains('bin'), index_col_names]
for this_col in index_col_names:
	taste_sig[this_col] = index_cols[this_col].values
	bin_sig[this_col] = index_cols[this_col].values
taste_sig.reset_index(inplace=True, drop=True)
bin_sig.reset_index(inplace=True, drop=True)

row_var = 'neuron_num'
col_var = 'laser_tuple'
val_var = 'sig'
taste_sig_pivot = taste_sig.pivot(
		index = row_var,
		columns = col_var,
		values = val_var
		)
bin_sig_pivot = bin_sig.pivot(
		index = row_var,
		columns = col_var,
		values = val_var
		)

fig, ax = plt.subplots(1,2, sharex=True, sharey=False,
					   figsize = (10,10))
sns.heatmap(taste_sig_pivot, 
			cmap='coolwarm', cbar=True,
			linewidth = 0.5,
			cbar_kws = {'label' : 'Significant (1 = yes, 0 = no)'},
			ax = ax[0]
				)
sns.heatmap(bin_sig_pivot, 
			cmap='coolwarm', cbar=True,
			linewidth = 0.5,
			cbar_kws = {'label' : 'Significant (1 = yes, 0 = no)'},
			ax = ax[1]
				)
ax[0].set_title('Discriminability ANOVA')
ax[1].set_title('Dynamic ANOVA')
plt.suptitle('Discriminability + Dynamicity of neurons\n'+\
		f'alpha={alpha}, corrected alpha={corrected_alpha}')
for this_ax in ax:
	this_ax.set_xlabel('Laser condition\n(lag, duration)')
	this_ax.set_ylabel('Neuron')
plt.tight_layout()
plt.savefig(os.path.join(agg_plot_dir, 'discrim_dynamic_heatmap.png'),
			bbox_inches='tight')
plt.close()

# # For each row in spike_frame, chop post_spikes into bins
# taste_list = []
# neuron_list = []
# bin_spike_list = []
# bin_num_list = []
# trial_num_list = []
# for i, this_row in spike_frame.iterrows():
# 	this_spikes = this_row.raw_spikes
# 	taste_num = this_row.taste
# 	neuron_num = this_row.neuron
# 	for bin_ind in range(len(bin_lims)-1):
# 		bin_spikes = this_spikes[..., bin_lims[bin_ind]:bin_lims[bin_ind+1]]
# 		sum_bin_spikes = bin_spikes.sum(axis=-1)
# 		n_trials = len(sum_bin_spikes)
# 		taste_list.extend([taste_num]*n_trials)
# 		neuron_list.extend([neuron_num]*n_trials)
# 		bin_spike_list.extend(sum_bin_spikes)
# 		bin_num_list.extend([bin_ind]*n_trials)
# 		trial_num_list.extend(np.arange(n_trials))
# 
# discrim_frame = pd.DataFrame(
# 		dict(
# 			taste = taste_list,
# 			neuron = neuron_list,
# 			spike_count = bin_spike_list,
# 			bin_num = bin_num_list,
# 			trial_num = trial_num_list,
# 			)
# 		)

# anova_list = []
# for nrn_num in discrim_frame.neuron.unique():
# 	this_frame = discrim_frame.loc[discrim_frame.neuron == nrn_num]
# 	this_anova = pg.anova(
# 			data = this_frame,
# 			dv = 'spike_count',
# 			between = ['taste','bin_num'],
# 			)
# 	anova_list.append(this_anova)
# p_val_list = []
# for i, x in enumerate(anova_list):
# 	this_p = x[['Source','p-unc']] 
# 	this_p['neuron'] = i
# 	p_val_list.append(this_p)
# p_val_frame = pd.concat(p_val_list)
# p_val_frame = p_val_frame.loc[~p_val_frame.Source.str.contains('Residual')]
# p_val_frame['sig'] = (p_val_frame['p-unc'] < alpha)*1
# 
# # Aggregate significance
# # Drop interaction
# p_val_frame = p_val_frame.loc[~p_val_frame.Source.isin(["taste * bin_num"])] 
# taste_sig = p_val_frame.loc[p_val_frame.Source.str.contains('taste')].sig
# bin_sig = p_val_frame.loc[p_val_frame.Source.str.contains('bin')].sig
# sig_frame = pd.DataFrame(
# 		dict(
# 			taste_sig = taste_sig.values,
# 			bin_sig = bin_sig.values,
# 			)
# 		)
# # sig_frame['neuron'] = p_val_frame.neuron.unique()
# 
# plt.figure()
# g = sns.heatmap(sig_frame, cmap='coolwarm', cbar=True)
# # cbar label
# cbar = g.collections[0].colorbar
# cbar.set_label('Significant (1 = yes, 0 = no)')
# g.set_title('Significance of taste and time bins\n'+\
# 		f'alpha={alpha}')
# ax = g.get_axes()
# ax.set_xlabel('Variable')
# ax.set_ylabel('Neuron')
# plt.tight_layout()
# plt.savefig(os.path.join(agg_plot_dir, 'discriminability.png'),
# 			bbox_inches='tight')
# plt.close()
