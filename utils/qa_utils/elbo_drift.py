"""
This module uses a PyMC change point model to detect drift in neural data, performing model selection using the Evidence Lower Bound (ELBO). It processes electrophysiological data to identify changes in mean and variance over time.

- `gaussian_changepoint_mean_var_2d(data_array, n_states, **kwargs)`: Constructs a PyMC model for detecting change points in a 2D Gaussian data array, modeling changes in both mean and variance across specified states.
- Initializes the environment by setting up directories for output and artifacts, and performs a pipeline graph check.
- Loads electrophysiological data, extracts spike trains, and processes them using PCA for dimensionality reduction.
- Iteratively fits a change point model to the PCA-transformed data, evaluating different numbers of change points and repeats, and records ELBO values for model selection.
- Aggregates results across different tastes, ranks the number of change points using median ELBO, and generates visualizations of the data and model fits.
- Logs warnings if significant post-stimulus population drift is detected based on ELBO rankings.
"""

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

script_path = os.path.realpath(__file__)
script_dir_path = os.path.dirname(script_path)
blech_path = os.path.dirname(os.path.dirname(script_dir_path))
sys.path.append(blech_path)
from utils.ephys_data import ephys_data  # noqa: E402
from utils.blech_utils import imp_metadata, pipeline_graph_check  # noqa: E402


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

            weight_stack = tt.math.sigmoid(
                idx[np.newaxis, :]-tau[:, np.newaxis])
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
# Initialize
############################################################
# Get name of directory with the data files
metadata_handler = imp_metadata(sys.argv)
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
# Load Data
############################################################
# Open the hdf5 file
this_dat = ephys_data.ephys_data(dir_name)
this_dat.get_spikes()
spike_trains = this_dat.spikes

max_components = 5
max_changepoints = 3
n_repeats = 10
time_lims = [2000, 4000]
bin_width = 50

n_tastes = len(spike_trains)
for taste_ind in range(n_tastes):
    save_path = f'{output_dir}/{basename}_taste_{taste_ind}_trial_change_elbo.png'

    if os.path.exists(save_path):
        print(f'{os.path.basename(save_path)} already exists, skipping')
        continue

    print(f'Processing {basename}, Taste {taste_ind}')
    this_taste = spike_trains[taste_ind]

    # 1) Cut by time_lims
    this_taste_cut = this_taste[:, :, time_lims[0]:time_lims[1]]

    # 2) Bin spikes
    n_trials, n_neurons, n_time = this_taste_cut.shape
    n_bins = n_time // bin_width
    this_taste_binned = this_taste_cut.reshape(
        n_trials, n_neurons, n_bins, bin_width).sum(axis=-1)

    this_taste_long = this_taste_binned.reshape(n_trials, -1)
    this_taste_pca = PCA(n_components=max_components,
                         whiten=True).fit_transform(this_taste_long)

    # 3) Fit changepoint model
    elbo_list = []
    tau_list = []
    ppc_list = []
    changes_list = []
    repeat_list = []
    changes_vec = np.arange(max_changepoints+1)
    for n_changes in tqdm(changes_vec):
        for repeat_ind in range(n_repeats):
            print(f'Running {n_changes} changes, repeat {repeat_ind}')
            model = gaussian_changepoint_mean_var_2d(
                this_taste_pca.T,
                # It doesn't like it being numpy.int64
                n_states=int(n_changes+1),
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
                            tolerance=1e-2,  # This will change based on model + data size
                            # As is, this is a very high (coarse) tolerance
                        )
                    ],
                )
                trace = approx.sample(draws=int(2e3))
                ppc = pm.sample_posterior_predictive(trace)

            tau_samples = trace.posterior['tau'].values
            tau_hists = np.stack([np.histogram(
                tau.flatten(), bins=np.arange(n_bins+1))[0]
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
    run_frame['taste_ind'] = taste_ind
    run_frame['basename'] = basename
    run_frame.to_pickle(
        f'{artifact_dir}/{basename}_taste_{taste_ind}_trial_change_elbo.pkl')

    median_elbo_df = run_frame.groupby('changes').elbo.median()
    median_elbo_df = median_elbo_df.reset_index()

    # Plot everything
    vmin = min([x.min() for x in ppc_list] + [this_taste_pca.min()])
    vmax = max([x.max() for x in ppc_list] + [this_taste_pca.max()])
    img_kwargs = {'aspect': 'auto', 'interpolation': 'none', 'cmap': 'viridis',
                  'vmin': vmin, 'vmax': vmax}
    change_colors = plt.cm.tab10(np.linspace(0, 1, max_changepoints))
    fig, ax = plt.subplots(len(changes_vec) + 1, 2, figsize=(7, 3*len(changes_vec)),
                           sharex=False)
    ax[0, 0].imshow(this_taste_pca.T, **img_kwargs)
    ax[0, 0].set_title('Actual Data PCA')
    ax[0, 1].scatter(run_frame.changes, run_frame.elbo, alpha=0.5,
                     linewidth=1, facecolor='none', edgecolor='black')
    ax[0, 1].plot(median_elbo_df.changes, median_elbo_df.elbo,
                  'r', label='Median ELBO')
    ax[0, 1].legend()
    ax[0, 1].set_xlabel('n_changes')
    ax[0, 1].set_ylabel('ELBO')
    for i, n_change in enumerate(changes_vec):
        this_frame = run_frame[run_frame.changes == n_change]
        mean_mean_ppc = np.stack(this_frame.ppc.values).mean(axis=0)
        # mean_elbo = this_frame.elbo.mean()
        median_elbo = this_frame.elbo.median()
        ax[i+1, 0].imshow(mean_mean_ppc, **img_kwargs)
        ax[i+1, 0].set_title(
            f'n_changes: {changes_vec[i]}, Median ELBO: {median_elbo:.2f}')
        for row_ind, this_row in this_frame.iterrows():
            for c_i, this_mode in enumerate(this_row['mode']):
                ax[i+1, 1].scatter(this_mode, this_row['repeat'],
                                   c=change_colors[c_i], cmap='tab10')
        ax[i+1, 0].set_xlim(0, n_trials)
        ax[i+1, 1].set_xlim(0, n_trials)
        ax[i+1, 1].set_ylabel('Repeat #')
        ax[i+1, 0].set_ylabel('Component #')
    ax[0, 1].set_title('Changepoint Samples')
    ax[-1, 0].set_xlabel('Trial #')
    ax[-1, 1].set_xlabel('Trial #')
    # plt.show()
    fig.suptitle(f'{basename} Taste {taste_ind}')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

##############################
# Aggregate all data
##############################
frame_list = []
for taste_ind in range(n_tastes):
    run_frame = pd.read_pickle(
        f'{artifact_dir}/{basename}_taste_{taste_ind}_trial_change_elbo.pkl')
    frame_list.append(run_frame)
full_frame = pd.concat(frame_list)
full_frame.reset_index(drop=True, inplace=True)
median_elbo_df = full_frame.groupby(['changes', 'taste_ind']).elbo.median()
median_elbo_df = median_elbo_df.reset_index()

sns.lineplot(
    data=full_frame,
    x='changes',
    y='elbo',
    hue='taste_ind',
    style='taste_ind',
    markers=True,
    dashes=False,
)
plt.title(f'{basename} ELBO vs. n_changes')
plt.savefig(f'{output_dir}/{basename}_all_tastes_trial_change_elbo.png')
plt.close()

# For each taste, rank the n_changes using median ELBO
ranked_changes = median_elbo_df.groupby('taste_ind').apply(
    lambda x: x.sort_values('elbo', ascending=True).changes.values)
best_change_per_taste = np.stack(ranked_changes.values)[:, 0]
best_change_per_taste_map = {
    taste_ind: best_change for taste_ind, best_change in enumerate(best_change_per_taste)}
best_change_per_taste_df = pd.DataFrame(
    dict(
        taste_ind=list(best_change_per_taste_map.keys()),
        best_change=list(best_change_per_taste_map.values()),
    ),
)

if any(best_change_per_taste > 0):
    with open(warnings_file_path, 'a') as f:
        print('=== Post-stimulus POPULATION Drift Warning ===', file=f)
        print('Ranks for ELBO calculated across tastes', file=f)
        print(best_change_per_taste_df, file=f)
        print('\n', file=f)
        print('=== End Post-stimulus POPULATION Drift Warning ===', file=f)
        print('\n', file=f)

this_pipeline_check.write_to_log(script_path, 'completed')
