from visualize import firing_overview
from ephys_data import ephys_data
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d as gf1d
import numpy as np

os.chdir('/media/bigdata/firing_space_plot/ephys_data')

dat = \
    ephys_data('/media/bigdata/Abuzar_Data/AM34/AM34_4Tastes_201215_115133')
# ephys_data('/media/bigdata/Abuzar_Data/AM37/AM37_4Tastes_210112_121908')

dat.firing_rate_params = dat.default_firing_params

# dat.extract_and_process()
dat.get_spikes()
dat.get_firing_rates()
# firing_overview(dat.all_normalized_firing);plt.show()
# dat.firing_overview(dat.all_firing_array);plt.show()

mean_firing = np.mean(dat.firing_array, axis=2)
# firing_overview(mean_firing.swapaxes(0,1));plt.show()

plot_dir = os.path.join(dat.data_dir, 'pretty_overlay_PSTH')
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

time_lims = np.array([1500, 3500])
time_lim_inds = time_lims//dat.firing_rate_params['step_size']
xlabels = np.arange(time_lims[0], time_lims[1])[
    ::dat.firing_rate_params['step_size']]
xlabels -= 2000
wanted_xlabels = np.where(xlabels % 250 == 0)[0]


def this_filter(x): return gf1d(x, 1)


for num, this_unit in enumerate(mean_firing.swapaxes(0, 1)):

    # this_unit = mean_firing[:,0]
    filtered_unit = this_filter(this_unit)
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].plot(this_unit[..., time_lim_inds[0]:time_lim_inds[1]].T,
               linewidth=3)  # , alpha = 0.7)
    ax[1].plot(filtered_unit[..., time_lim_inds[0]:time_lim_inds[1]].T,
               linewidth=3)  # , alpha = 0.7)
    ax[0].set_title(f'Unit #{num}')
    ax[0].set_xticks(wanted_xlabels)
    ax[0].set_xticklabels(xlabels[wanted_xlabels])
    ax[1].set_xticks(wanted_xlabels)
    ax[1].set_xticklabels(xlabels[wanted_xlabels])
    # plt.show()
    fig.savefig(os.path.join(plot_dir, f'unit_{num}'), dpi=300)
    plt.close(fig)
