
from utils.ephys_data import ephys_data
from utils.ephys_data import visualize as vz
import matplotlib.pyplot as plt

data_dir = '/media/storage/for_transfer/bla_gc/AM11_4Tastes_191030_114043_copy'

dat = ephys_data.ephys_data(data_dir)

dat.get_spikes()
dat.default_firing_params['type'] = 'basis'

dat.get_firing_rates()

vz.firing_overview(dat.all_normalized_firing)
plt.show()

mean_rates = dat.firing_array.mean(axis=2).swapaxes(0, 1)
fig, ax = vz.gen_square_subplots(len(mean_rates))

for this_rate, this_ax in zip(mean_rates, ax.flatten()):
    this_ax.plot(this_rate.T)
plt.show()
