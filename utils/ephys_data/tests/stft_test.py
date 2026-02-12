# from ephys_data import ephys_data
from blech_clust.utils.ephys_data.ephys_data import ephys_data
import os
import matplotlib.pyplot as plt

# os.chdir('/media/bigdata/firing_space_plot/ephys_data')

data_dir = '/media/fastdata/CM74_CTATest2_h2o_nacl_lowqhcl_highqhcl_250614_120546'

# dat = ephys_data('/media/bigdata/Abuzar_Data/AM11/AM11_4Tastes_191030_114043_copy')
dat = ephys_data(data_dir)
dat.firing_rate_params = dat.default_firing_params

dat.get_unit_descriptors()
dat.get_spikes()
dat.get_firing_rates()
dat.get_lfps()

dat.get_region_units()

dat.get_lfps(re_extract=True)
dat.get_stft(recalculate=True)
