# Import stuff!
import numpy as np
import tables
import sys
import os
import pandas as pd
# Necessary blech_clust modules
sys.path.append('..')
from utils.blech_utils import imp_metadata
from blech_make_arrays import get_dig_in_data,

# Ask for the directory where the hdf5 file sits, and change to that directory
# Get name of directory with the data files
metadata_handler = imp_metadata(sys.argv)
dir_name = metadata_handler.dir_name
os.chdir(dir_name)

# Open the hdf5 file
hf5 = tables.open_file(metadata_handler.hdf5_name, 'r+')

# Extract taste dig-ins from experimental info file
info_dict = metadata_handler.info_dict
params_dict = metadata_handler.params_dict
sampling_rate = params_dict['sampling_rate']
sampling_rate_ms = sampling_rate/1000

# Grab the names of the arrays containing digital inputs, 
# and pull the data into a numpy array
dig_in_pathname, dig_in_basename, dig_in_data = get_dig_in_data(hf5)
dig_in_diff = np.diff(dig_in_data,axis=-1)
# Calculate start and end points of pulses
start_points = [np.where(x==1)[0] for x in dig_in_diff]
end_points = [np.where(x==-1)[0] for x in dig_in_diff]

# Pull out taste dig-ins
taste_digin_inds = info_dict['taste_params']['dig_ins']
taste_digin_channels = [dig_in_basename[x] for x in taste_digin_inds]
taste_str = "\n".join(taste_digin_channels)

# Extract laser dig-in from params file
laser_digin_inds = [info_dict['laser_params']['dig_in']][0]

# Pull laser digin from hdf5 file
if len(laser_digin_inds) == 0:
    laser_digin_channels = []
    laser_str = 'None'
else:
    laser_digin_channels = [dig_in_basename[x] for x in laser_digin_inds]
    laser_str = "\n".join(laser_digin_channels)

print(f'Taste dig_ins ::: \n{taste_str}\n')
print(f'Laser dig_in ::: \n{laser_str}\n')

#============================================================#
hf5.close()
elec_cutoff_frame = pd.read_hdf(metadata_handler.hdf5_name, '/cutoff_frame') 
hf5 = tables.open_file(metadata_handler.hdf5_name, 'r+')
expt_end_time = elec_cutoff_frame['recording_cutoff'].min()*sampling_rate
#============================================================#

#============================================================
# Correct for laser sampling errors before moving on to next step
#============================================================
# Take "correct" values from info file
# NOTE: This will likely need to be corrected later on as info file
#       currently only allows single values for onset and duration
info_laser_params = info_dict['laser_params']
info_laser_onset = info_laser_params['onset'] 
info_laser_duration = info_laser_params['duration']

if len(laser_digin_inds):
    info_laser_data = [(0,0), (info_laser_duration, info_laser_onset)]
else:
    info_laser_data = [(0,0)]
info_laser_data = [np.array(x) for x in info_laser_data]

dig_in_list = hf5.get_node('/','spike_trains')
dig_in_list = [x for x in dig_in_list if 'dig_in' in x._v_pathname]
durations = [x.laser_durations[:] for x in dig_in_list]
lags = [x.laser_onset_lag[:] for x in dig_in_list]

# Compare actual laser data to calculate onsets and durations
# And correct as needed
data_tuples = [np.vstack([x,y]).T for x,y in zip(durations, lags)] 

corrected_tuples = []
for this_dig_in in data_tuples:
    deviations = np.stack(
            [np.linalg.norm(this_dig_in - x,axis=-1) for x in info_laser_data])
    min_ind = np.argmin(deviations, axis=0)
    corrected_tuples.append(np.stack([info_laser_data[x] for x in min_ind]))

for this_dig_in, this_corrected_dat in zip(dig_in_list, corrected_tuples):
    this_dig_in.laser_durations[:] = this_corrected_dat[:,0] 
    this_dig_in.laser_onset_lag[:] = this_corrected_dat[:,1] 
hf5.flush()

orig_unique_tuples = set([tuple(x) for x in np.concatenate(data_tuples)])
fin_unique_tuples = set([tuple(x) for x in np.concatenate(corrected_tuples)])

print()
print("Laser timings corrected")
print("============================================================")
print("Original data")
for x in orig_unique_tuples:
    print(x)
print("")
print("Corrected data")
for x in fin_unique_tuples:
    print(x)

#============================================================
# Create an ancillary_analysis group in the hdf5 file, 
# and write these arrays to that group
if '/ancillary_analysis' in hf5:
    hf5.remove_node('/ancillary_analysis', recursive = True)
hf5.create_group('/', 'ancillary_analysis')

# First pull out the unique laser(duration,lag) combinations - 
# these are the same irrespective of the unit and time
unique_lasers = np.array(list(fin_unique_tuples))

# Now get the sets of trials with these unique duration and lag combinations
concat_laser_tuples = np.concatenate(corrected_tuples)
trials = np.stack(
        [
            [
                i for i, dat in enumerate(concat_laser_tuples) \
                        if np.array_equal(dat,this_cond)
                ]
            for this_cond in unique_lasers
            ]
        )


# Save the trials and unique laser combos to the hdf5 file as well
hf5.create_array('/ancillary_analysis', 'trials', trials)
hf5.create_array('/ancillary_analysis', 'laser_combination_d_l', unique_lasers)
hf5.flush()
hf5.close()
