import os
import matplotlib.pyplot as plt

os.chdir('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data

dat = \
ephys_data('/media/bigdata/Abuzar_Data/AM11/AM11_4Tastes_191030_114043_copy')
dat.firing_rate_params = dat.default_firing_params 

dat.get_unit_descriptors()
dat.get_spikes()
dat.get_firing_rates()
dat.get_lfps()

dat.get_region_units()

dat.get_stft(recalculate = True)
