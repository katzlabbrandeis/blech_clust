"""

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
import glob
import itertools
from utils.blech_utils import entry_checker, imp_metadata, pipeline_graph_check
from utils.ephys_data import ephys_data
from utils.ephys_data import visualize as vz
import pandas as pd
pd.options.mode.chained_assignment = None
from itertools import product
from scipy.stats import ttest_rel, zscore, spearmanr
import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg
from tqdm import tqdm
tqdm.pandas()

# Ask for the directory where the hdf5 file sits, and change to that directory
# Get name of directory with the data files
metadata_handler = imp_metadata(sys.argv)
dir_name = metadata_handler.dir_name

plot_dir = os.path.join(dir_name, 'unit_characteristic_plots')
if not os.path.exists(plot_dir):
	os.makedirs(plot_dir)

agg_plot_dir = os.path.join(plot_dir, 'aggregated')
if not os.path.exists(agg_plot_dir):
	os.makedirs(agg_plot_dir)

# Perform pipeline graph check
script_path = os.path.realpath(__file__)
this_pipeline_check = pipeline_graph_check(dir_name)
this_pipeline_check.check_previous(script_path)
this_pipeline_check.write_to_log(script_path, 'attempted')

os.chdir(dir_name)

this_dat = ephys_data.ephys_data(dir_name)

# Extract taste dig-ins from experimental info file
info_dict = metadata_handler.info_dict
params_dict = metadata_handler.params_dict

taste_names = info_dict['taste_params']['tastes']


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
waveform_dir = os.path.join(dir_name, 'unit_waveforms_plots', 'waveforms_only')
spike_array = this_dat.spikes
cmap = plt.cm.get_cmap('tab10')
colors = [cmap(i) for i in range(len(spike_array))]
for nrn_ind in tqdm(mean_seq_firing.neuron_num.unique()):
	n_rows = np.max([n_laser_conditions, 2])
	fig, ax = plt.subplots(n_rows,3, figsize=(15,5*n_laser_conditions),
						# sharex=True, sharey='col')
	)
	# Remove axis for lower row if only one laser condition
	if n_laser_conditions == 1:
		for this_ax in ax[-1]:
			this_ax.axis('off')
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
		ax[i,0].legend()
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
		for this_ax in ax[i,:-1]:
			if i == len(ax)-1:
				this_ax.set_xlabel('Time (ms)')
			this_ax.axvline(0, color='r', linestyle='--', linewidth=2)
			this_ax.set_title(f'Laser condition {laser_cond}')
		# Plot waveforms
		datashader_img_path = os.path.join(waveform_dir, f'Unit{nrn_ind}_datashader.png')
		mean_sd_img_path = os.path.join(waveform_dir, f'Unit{nrn_ind}_mean_sd.png') 
		datashader_img = plt.imread(datashader_img_path)
		mean_sd_img = plt.imread(mean_sd_img_path)
		ax[0, -1].imshow(datashader_img)
		ax[1, -1].imshow(mean_sd_img)
		for this_ax in ax[:,-1]:
			this_ax.axis('off')
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

cat_firing = np.concatenate(this_dat.firing_list)
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
	this_ax.axvline(0, color='k', linestyle='--', linewidth=2)
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
# fig.supxlabel('Taste responsive neurons')
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
	if 'p-unc' in anova_out.columns:
		this_pval = anova_out[['Source','p-unc']]
	else:
		this_pval = anova_out[['Source']]
		this_pval['p-unc'] = 1
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
taste_bin_pval_frame = p_val_frame.copy()
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

##############################
# Palatability 
##############################
seq_firing_frame = this_dat.sequestered_firing_frame.copy()
seq_firing_frame['time_val'] = [firing_t_vec[x] for x in seq_firing_frame.time_num]
pal_ranks = info_dict['taste_params']['pal_rankings']
# Also print pal_dict
pal_dict = dict(
		zip(
			info_dict['taste_params']['tastes'],
			pal_ranks
			)
		)
seq_firing_frame['pal_rank'] = [pal_ranks[i] for i in seq_firing_frame.taste_num]

group_cols = ['neuron_num','time_val','laser_tuple']
group_list = list(seq_firing_frame.groupby(group_cols))
group_inds = [x[0] for x in group_list]
group_frames = [x[1] for x in group_list]

def palatability_corr(x):
	rho, p = spearmanr(x.firing, x.pal_rank)
	return pd.Series({'rho': rho, 'pval' : p})
pal_frame = seq_firing_frame.groupby(group_cols).progress_apply(palatability_corr)
pal_frame.reset_index(inplace=True)
pal_frame['abs_rho'] = pal_frame.rho.abs()
pal_frame['pal_sig'] = (pal_frame['pval'] < alpha)*1

# For each laser_tuple, generate a pivot frame
index_var = 'neuron_num'
col_var = 'time_val'
value_var_list = ['abs_rho','pal_sig']

pal_len = 250 # ms
pal_kern_len = pal_len // this_dat.firing_rate_params['step_size']
pal_kern = np.ones(pal_kern_len) / pal_kern_len
pal_pivot_list = []
frame_ind_list = []
pal_sig_vec_list = []
laser_conds = pal_frame.laser_tuple.unique()
for this_value_var in value_var_list:
	for this_laser in laser_conds:
		this_pivot = pal_frame.loc[pal_frame.laser_tuple == this_laser]
		this_pivot = this_pivot.pivot(
				index = index_var,
				columns = col_var,
				values = this_value_var
				)
		pal_pivot_list.append(this_pivot)
		frame_ind_list.append(
				[this_laser, this_value_var]
				)
		if this_value_var == 'pal_sig':
			pal_conv = np.apply_along_axis(
				lambda x:np.convolve(x, pal_kern, mode='same'),
				axis = -1,
				arr = this_pivot.values
				)
			# Cut to after stim
			stim_ind = firing_t_vec > 0
			pal_conv = pal_conv[:, stim_ind]
			pal_sig_vec = np.isclose(pal_conv, 1).sum(axis=-1) > 0
			pal_sig_vec_list.append(pal_sig_vec*1)

pal_sig_vec_frame = pd.DataFrame(
		index = pal_frame.neuron_num.unique(),
		columns = laser_conds,
		data = np.array(pal_sig_vec_list).T
		)

# plt.imshow(pal_conv, aspect='auto', interpolation='None')
# plt.colorbar()
# plt.show()

# Plot palatability
min_rho, max_rho = pal_frame.abs_rho.min(), pal_frame.abs_rho.max()
fig, ax = plt.subplots(
		len(laser_conds),
		len(frame_ind_list)//len(laser_conds) + 1,
		figsize = (15,5*len(laser_conds)),
		sharey=True
		)
# If only one laser cond, convert to 2D array
if len(laser_conds) == 1:
	ax = ax.reshape(1,-1)
stim_ind = np.argmin(np.abs(pal_pivot_list[0].columns - 0))
comb_inds = list(itertools.product(laser_conds, value_var_list))
for this_laser, this_var in comb_inds:
	row_ind = np.where(laser_conds == this_laser)[0][0] 
	col_ind = value_var_list.index(this_var)
	pivot_ind = frame_ind_list.index([this_laser, this_var]) 
	this_pivot = pal_pivot_list[pivot_ind]
	if this_var == 'abs_rho':
		cbar_label = 'Abs Spearman rho'
		vmin, vmax = min_rho, max_rho
	else:
		cbar_label = 'Significant (1 = yes, 0 = no)'
		vmin, vmax = None,None
	sns.heatmap(
			this_pivot,
			vmin = vmin,
			vmax = vmax,
			cmap='coolwarm', cbar=True,
			# linewidth = 0.5,
			cbar_kws = {'label' : cbar_label},
			ax = ax[row_ind, col_ind],
				)
	ax[row_ind, col_ind].set_title(f'{this_var} {this_laser}')
	ax[row_ind, col_ind].axvline(stim_ind, c = 'r', linewidth = 2, linestyle = '--',
							  zorder = 10)
	cbar_label = 'Significant (1 = yes, 0 = no)'
	if this_var == 'pal_sig':
		sns.heatmap(
				pal_sig_vec_list[row_ind].reshape(-1,1),
				cmap='coolwarm', cbar=True,
				cbar_kws = {'label' : cbar_label},
				linewidth = 0.5,
				ax = ax[row_ind, -1], 
				)
	ax[row_ind,-1].set_title(f'{this_laser} Post-stim Palatability significance')
plt.suptitle('Palatability of neurons\n'+\
		f'alpha={alpha}, corrected alpha={corrected_alpha}')
for this_ax in ax.flatten():
	this_ax.set_xlabel('Time (ms)')
	this_ax.set_ylabel('Neuron')
plt.tight_layout()
plt.savefig(os.path.join(agg_plot_dir, 'palatability_heatmap.png'),
			bbox_inches='tight')
plt.close()

############################################################
# Aggregate plot
############################################################
# Plot of significance for responsiveness, discriminability, palatability, and dynamicity
# For each opto condition

fig, ax = plt.subplots(len(laser_conds),4, figsize=(10,10))
# If only one laser cond, convert to 2D array
if len(laser_conds) == 1:
	ax = ax.reshape(1,-1)
for i, this_cond in enumerate(laser_conds):
	resp_vec = resp_neurons_pivot[this_cond].values.reshape(-1,1)
	discrim_vec = taste_sig_pivot[this_cond].values.reshape(-1,1)
	pal_vec = pal_sig_vec_frame[this_cond].values.reshape(-1,1)
	bin_vec = bin_sig_pivot[this_cond].values.reshape(-1,1)
	sns.heatmap(
			resp_vec,
			cmap='coolwarm', cbar=False,
			linewidth = 0.5,
			ax = ax[i,0]
				)
	sns.heatmap(
			discrim_vec,
			cmap='coolwarm', cbar=False,
			linewidth = 0.5,
			ax = ax[i,1]
				)
	sns.heatmap(
			pal_vec,
			cmap='coolwarm', cbar=False,
			linewidth = 0.5,
			ax = ax[i,2]
				)
	sns.heatmap(
			bin_vec,
			cmap='coolwarm', cbar=True,
			linewidth = 0.5,
			ax = ax[i,3],
			cbar_kws = {'label' : 'Significant (1 = yes, 0 = no)'},
				)
	ax[i,0].set_title(f'Responsiveness {resp_vec.mean():.2f}\nLaser {this_cond}')
	ax[i,1].set_title(f'Discriminability {discrim_vec.mean():.2f}\nLaser {this_cond}')
	ax[i,2].set_title(f'Palatability {pal_vec.mean():.2f}\nLaser {this_cond}')
	ax[i,3].set_title(f'Dynamicity {bin_vec.mean():.2f}\nLaser {this_cond}')
plt.suptitle('Aggregated plot of neuron characteristics')
plt.tight_layout()
plt.savefig(os.path.join(agg_plot_dir, 'aggregated_characteristics.png'),
			bbox_inches='tight')
plt.close()

############################################################
# Aggregate plot
############################################################
# Merge all frames and write out to disk and HDF5
resp_neurons
taste_bin_pval_frame
pal_sig_vec_frame # meelted

this_resp_neurons = resp_neurons.copy()
this_resp_neurons['Source'] = 'responsiveness'
this_resp_neurons.rename(columns = {'resp_pval' : 'sig'}, inplace=True)

this_pal_sig_frame = pal_sig_vec_frame.copy()
this_pal_sig_frame['neuron_num'] = this_pal_sig_frame.index
this_pal_sig_frame = this_pal_sig_frame.melt(
		id_vars = 'neuron_num',
		value_vars = laser_conds,
		var_name = 'laser_tuple',
		value_name = 'sig',
		)
this_pal_sig_frame['Source'] = 'palatability'

out_frame = pd.concat([this_resp_neurons, taste_bin_pval_frame, this_pal_sig_frame])
out_frame.drop(columns = ['p-unc'], inplace=True)

out_frame.to_csv(os.path.join(dir_name, 'aggregated_characteristics.csv'),
				 index=False)
out_frame.to_hdf(
		metadata_handler.hdf5_name,
		key = 'ancillary_analysis/unit_characteristics',
		mode = 'a',
		)
