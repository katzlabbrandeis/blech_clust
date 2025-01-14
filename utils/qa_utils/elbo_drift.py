"""
Using pymc change point model to detect drift
Model selection using ELBO
"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dir_name', type=str, help='Directory containing data')
parser.add_argument('--force', action='store_true', help='Force re-fitting')
args = parser.parse_args()

from tqdm import tqdm, trange
import sys
import os
from pymc.variational.callbacks import CheckParametersConvergence
import pymc as pm
import pytensor.tensor as tt
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import time
import seaborn as sns
import tables
from scipy.stats import zscore

script_path = os.path.realpath(__file__)
script_dir_path = os.path.dirname(script_path)
blech_path = os.path.dirname(os.path.dirname(script_dir_path))
sys.path.append(blech_path)
from utils.blech_utils import imp_metadata, pipeline_graph_check
from utils.ephys_data import ephys_data

def gaussian_changepoint_mean_var_2d(data_array, n_states, **kwargs):
    """Model for gaussian data on 2D array detecting changes in both
    mean and variance.

    Args:
        data_array (2D Numpy array): <dimension> x time
        n_states (int): Number of states to model

    Returns:
        pymc3 model: Model class containing graph to run inference on
    """
    mean_vals = np.array([np.mean(x, axis=-1)
                          for x in np.array_split(data_array, n_states, axis=-1)]).T
    mean_vals += 0.01  # To avoid zero starting prob

    y_dim = data_array.shape[0]
    idx = np.arange(data_array.shape[-1])
    length = idx.max() + 1

    with pm.Model() as model:
        mu = pm.Normal('mu', mu=mean_vals, sigma=1, shape=(y_dim, n_states))
        sigma = pm.HalfCauchy('sigma', 3., shape=(y_dim, n_states))

        if n_states > 1:
            a_tau = pm.HalfCauchy('a_tau', 3., shape=n_states - 1)
            b_tau = pm.HalfCauchy('b_tau', 3., shape=n_states - 1)

            even_switches = np.linspace(0, 1, n_states+1)[1:-1]
            tau_latent = pm.Beta('tau_latent', a_tau, b_tau,
                                 initval=even_switches,
                                 shape=(n_states-1)).sort(axis=-1)

            tau = pm.Deterministic('tau',
                                   idx.min() + (idx.max() - idx.min()) * tau_latent)

            weight_stack = tt.math.sigmoid(idx[np.newaxis, :]-tau[:, np.newaxis])
            weight_stack = tt.concatenate(
                [np.ones((1, length)), weight_stack], axis=0)
            inverse_stack = 1 - weight_stack[1:]
            inverse_stack = tt.concatenate(
                [inverse_stack, np.ones((1, length))], axis=0)
            weight_stack = np.multiply(weight_stack, inverse_stack)

            mu_latent = mu.dot(weight_stack)
            sigma_latent = sigma.dot(weight_stack)
        else:
            tau = pm.Uniform('tau', lower=0, upper=0.1, shape=1)
            mu_latent = np.ones_like(data_array) * mu
            sigma_latent = np.ones_like(data_array) * sigma

        observation = pm.Normal("obs", mu=mu_latent, sigma=sigma_latent,
                                observed=data_array)

    return model

############################################################
## Initialize 
############################################################
# Get name of directory with the data files

# dir_name = '/media/storage/ABU_GC-EMG_Data/updated/sorted/KM50_5tastes_EMG_210913_100710_copy'
metadata_handler = imp_metadata([[], args.dir_name])
# metadata_handler = imp_metadata(sys.argv)
dir_name = metadata_handler.dir_name

# Perform pipeline graph check
this_pipeline_check = pipeline_graph_check(dir_name)
this_pipeline_check.check_previous(script_path)
this_pipeline_check.write_to_log(script_path, 'attempted')

basename = os.path.basename(dir_name[:-1])

output_dir = os.path.join(dir_name, 'QA_output')
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
artifact_dir = os.path.join(output_dir, 'artifacts')
if not os.path.isdir(artifact_dir):
    os.mkdir(artifact_dir)

warnings_file_path = os.path.join(output_dir, 'warnings.txt')

############################################################
## Load Data
############################################################
# Load trial times from trial_info_frame.csv
trial_info_path = os.path.join(dir_name, 'trial_info_frame.csv')
trial_info_frame = pd.read_csv(trial_info_path)

# Open the hdf5 file
this_dat = ephys_data.ephys_data(dir_name)

# Get spike-times from the hdf5 file
hdf5_path = this_dat.hdf5_path
with tables.open_file(hdf5_path, 'r') as hf5:
    units = hf5.get_node('/sorted_units')
    unit_names = [unit._v_name for unit in units]
    spike_times = [unit.times.read() for unit in units]

# Convert to histograms
max_time = int(np.ceil(max([x[-1] for x in spike_times])))
bins = np.linspace(0, max_time, 150)
spiketime_hists = np.stack([np.histogram(x, bins=bins)[0] for x in spike_times])
zscored_hists = zscore(spiketime_hists, axis=1)

# Perform PCA and keep 5 components
pca = PCA(n_components=5, whiten=True)
pca.fit(zscored_hists.T)
zscored_hists_pca = pca.transform(zscored_hists.T)

fig, ax = plt.subplots(3,1, sharex=True)
ax[0].scatter(trial_info_frame['start_taste'], trial_info_frame['dig_in_num_taste'],
              marker = '|', color='black')
ax[0].set_title('Taste Trials')
ax[0].set_ylabel('Dig In #')
ax[1].pcolorfast(bins, np.arange(zscored_hists.shape[0]), zscored_hists, cmap='viridis')
ax[1].set_title('Spike Histograms')
ax[1].set_ylabel('Neuron #')
ax[2].pcolorfast(bins, np.arange(zscored_hists_pca.shape[0]), zscored_hists_pca.T, cmap='viridis')
ax[2].set_title('PCA of Spike Histograms')
ax[2].set_ylabel('Component #')
plt.tight_layout()
fig.savefig(f'{output_dir}/{basename}_spike_histograms.png')
plt.close()

# Can also sample number of changepoints using skopt
# That way we can define a very high max_changepoints and let the model decide

plot_save_path = f'{output_dir}/{basename}_taste_trial_change_elbo.png'
artifact_save_path = f'{artifact_dir}/{basename}_taste_trial_change_elbo.pkl'

max_changepoints = 4
n_repeats = 10
changes_vec = np.arange(max_changepoints+1)

if args.force:
    print('=== Force re-fitting ===')

if not os.path.exists(artifact_save_path) or args.force:

    elbo_list = []
    tau_list = []
    ppc_list = []
    changes_list = []
    repeat_list = []
    for n_changes in tqdm(changes_vec):
        for repeat_ind in range(n_repeats):
            print(f'Running {n_changes} changes, repeat {repeat_ind}')
            model = gaussian_changepoint_mean_var_2d(
                    zscored_hists_pca.T,
                    n_states=int(n_changes+1), # It doesn't like it being numpy.int64
                    )
            with model:
                inference = pm.ADVI('full-rank')
                # Note: Stick to very high fit for now
                # lower values like 1e4 don't converge / don't work well (i.e. no change comes out best)
                approx = pm.fit(
                        n=int(1e5), 
                        method=inference,
                        callbacks=[
                            CheckParametersConvergence(
                                diff='absolute', 
                                tolerance=1e-2, # This will change based on model + data size
                                # As is, this is a very high (coarse) tolerance
                                )
                            ],
                        )
                trace = approx.sample(draws=int(2e3))
                ppc = pm.sample_posterior_predictive(trace)

            tau_samples = trace.posterior['tau'].values
            tau_hists = np.stack([np.histogram(
                tau.flatten(), bins=np.arange(len(bins)+1))[0] \
                        for tau in tau_samples.T]) 
            ppc_samples = ppc.posterior_predictive.obs.values
            mean_ppc = np.squeeze(ppc_samples.mean(axis=1))

            elbo_list.append(approx.hist[-1])
            tau_list.append(tau_hists)
            ppc_list.append(mean_ppc)
            changes_list.append(n_changes)
            repeat_list.append(repeat_ind)

    mode_list = [[np.argmax(x) for x in y] for y in tau_list]

    run_frame = pd.DataFrame(
            dict(
                elbo=elbo_list,
                changes=changes_list,
                repeat=repeat_list,
                mode=mode_list,
                ppc=ppc_list,
                ),
            )
    run_frame['basename'] = basename
    run_frame['time_bins'] = [bins] * len(run_frame)
    run_frame.to_pickle(artifact_save_path)

else:
    print(f'{os.path.basename(artifact_save_path)} already exists, skipping')

##############################
# Aggregate all data
##############################
run_frame = pd.read_pickle(artifact_save_path)
median_elbo_df = run_frame.groupby(['changes']).elbo.median()
median_elbo_df = median_elbo_df.reset_index()

# Plot everything
ppc_list = run_frame.ppc.values
vmin = min([x.min() for x in ppc_list] + [zscored_hists_pca.min()])
vmax = max([x.max() for x in ppc_list] + [zscored_hists_pca.max()])
img_kwargs = {'aspect':'auto', 'interpolation':'none', 'cmap':'viridis', 
              'vmin':vmin, 'vmax':vmax}
change_colors = plt.cm.tab10(np.linspace(0,1,max_changepoints))
fig, ax = plt.subplots(len(changes_vec) + 1, 2, figsize=(7,3*len(changes_vec)),
                       sharex=False)
ax[0,0].imshow(zscored_hists_pca.T, **img_kwargs) 
ax[0,0].set_title('Actual Data PCA')
ax[0,1].scatter(run_frame.changes, run_frame.elbo, alpha=0.5,
             linewidth = 1, facecolor='none', edgecolor='black')
ax[0,1].plot(median_elbo_df.changes, median_elbo_df.elbo, 'r', label='Median ELBO')
ax[0,1].legend()
ax[0,1].set_xlabel('n_changes')
ax[0,1].set_ylabel('ELBO')
for i, n_change in enumerate(changes_vec): 
    this_frame = run_frame[run_frame.changes == n_change]
    mean_mean_ppc = np.stack(this_frame.ppc.values).mean(axis=0)
    median_elbo = this_frame.elbo.median()
    ax[i+1,0].imshow(mean_mean_ppc, **img_kwargs)
    ax[i+1,0].set_title(
            f'n_changes: {changes_vec[i]}, Median ELBO: {median_elbo:.2f}')
    for row_ind, this_row in this_frame.iterrows():
        for c_i, this_mode in enumerate(this_row['mode']):
            ax[i+1,1].scatter(this_mode, this_row['repeat'],
                              c = change_colors[c_i], cmap = 'tab10')
    ax[i+1,0].set_xlim(0, len(bins)) 
    ax[i+1,1].set_xlim(0, len(bins))
    ax[i+1,1].set_ylabel('Repeat #')
    ax[i+1,0].set_ylabel('Component #')
ax[0,1].set_title('Changepoint Samples')
ax[-1,0].set_xlabel('Bin #')
ax[-1,1].set_xlabel('Bin #')
plt.tight_layout()
plt.savefig(plot_save_path)
plt.close()

best_change = median_elbo_df.sort_values('elbo', ascending=True).changes.values[0]

if best_change:  
    with open(warnings_file_path, 'a') as f:
        print('=== Post-stimulus POPULATION Drift Warning ===', file=f)
        print('Ranks for ELBO calculated across tastes', file=f)
        print(f'Best change: {best_change}', file=f)
        print('\n', file=f)
        print('=== End Post-stimulus POPULATION Drift Warning ===', file=f)
        print('\n', file=f)

this_pipeline_check.write_to_log(script_path, 'completed')
