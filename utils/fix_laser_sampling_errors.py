#
# Since the digital inputs are being sampled at 30kHz, sometimes the laser durations (or onsets)
# are recorded for a few ms more or less than the intended length of the pulse.
# We scale the length of the pulse by factors of 10 - so
# sometimes a 2500ms pulse can become 2510 or 2490ms. This gives errors in later steps.


import tables
import numpy as np
import easygui
import os
import sys
from blech_utils import (
    imp_metadata,
)

# Ask for the directory where the hdf5 file sits, and change to that directory
# Get name of directory with the data files
metadata_handler = imp_metadata(sys.argv)
dir_name = metadata_handler.dir_name
os.chdir(dir_name)

# Open the hdf5 file
hf5 = tables.open_file(metadata_handler.hdf5_name, 'r+')

# Get laser info from info_dict
info_dict = metadata_handler.info_dict
latencies = [0, info_dict['laser_params']['onset']]
durations = [0, info_dict['laser_params']['duration']]

print('Params from info file')
print(f'Latencies : {latencies}')
print(f'Durations : {durations}')

trains_dig_in = hf5.list_nodes('/spike_trains')

# Checking the laser-duration array to find sampling errors and correct them
for dig_in in range(len(trains_dig_in)):
    for duration in range(len(trains_dig_in[dig_in].laser_durations)):
        if trains_dig_in[dig_in].laser_durations[duration] not in durations:
            diff = np.absolute(np.array(durations) -
                               trains_dig_in[dig_in].laser_durations[duration])
            trains_dig_in[dig_in].laser_durations[duration] = durations[np.argmin(
                diff)]
hf5.flush()

# Checking the laser onset latency array to find sampling errors and correct them
for dig_in in range(len(trains_dig_in)):
    for latency in range(len(trains_dig_in[dig_in].laser_onset_lag)):
        if trains_dig_in[dig_in].laser_onset_lag[latency] not in latencies:
            diff = np.absolute(np.array(latencies) -
                               trains_dig_in[dig_in].laser_onset_lag[latency])
            trains_dig_in[dig_in].laser_onset_lag[latency] = latencies[np.argmin(
                diff)]
hf5.flush()


hf5.close()
