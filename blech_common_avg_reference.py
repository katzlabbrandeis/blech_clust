# Run through all the raw electrode data,
# and subtract a common average reference from every electrode's recording
# The user specifies the electrodes to be used as a common average group

# Import stuff!
import tables
import numpy as np
import os
import easygui
import sys
from tqdm import tqdm
import glob
import json
from utils.blech_utils import imp_metadata
from sklearn.decomposition import IncrementalPCA as IPCA

def truncated_ipca(
        n_components = 10, 
        n_iter = 1000,
        tolerance = 1e-3,
        data_generator = None
        ):
    """
    Truncated IPCA, stops when the change in components is less than tolerance
    """
    if data_generator is None:
        raise ValueError('Data generator must be specified')
    print('Starting IPCA')
    print('Threshold : {:f}'.format(tolerance))

    delta_list = []
    components_list = []
    ipca = IPCA(n_components=n_components)
    ipca.partial_fit(data_generator())
    components_list.append(ipca.components_)
    for i in range(n_iter):
        ipca.partial_fit(data_generator())
        delta = np.mean(np.abs(components_list[-1] - ipca.components_))
        delta_list.append(delta)
        components_list.append(ipca.components_)
        print(f'Iteration : {i}, Delta : {delta}')
        if delta <= tolerance:
            print('Converged')
            break
    return ipca, delta_list

def return_subset(dat_len, electrode_inds, batch_size, raw_electrodes):
    """
    Return a random subset of the data for pca training
    """
    idx = np.random.choice(dat_len, batch_size, replace=False)
    dat = np.stack([get_electrode_by_name(raw_electrodes, electrode_ind)[idx] \
            for electrode_ind in electrode_inds], axis=1)
    return dat 

def get_electrode_by_name(raw_electrodes, name):
    """
    Get the electrode data from the list of raw electrodes
    by the name of the electrode
    """
    str_name = f"electrode{name:02}"
    wanted_electrode_ind = [
        x for x in raw_electrodes if str_name in x._v_pathname][0]
    return wanted_electrode_ind

def return_data_chunk(electrode_inds, time_bin):
    """
    Return a chunk of data from the HDF5 file
    Used for subtraction step
    """
    dat = np.stack(
            [
                get_electrode_by_name(
                    raw_electrodes, this_electrode_ind)[time_bin[0]:time_bin[1]] \
            for this_electrode_ind in electrode_inds
            ], axis=1
        )
    return dat

def subtract_ipca(transformer, data):
    """
    Subtract the components from the data
    """
    ipca_dat = data - transformer.inverse_transform(transformer.transform(data))
    return ipca_dat.astype(np.float32)

def append_data_to_arrays(data, electrode_names):
    """
    Append data to the arrays in the hdf5 file
    """
    for this_name, this_data in zip(electrode_names, data.T):
        hf5.get_node('/raw', this_name).append(this_data)
    return


############################################################
############################################################


# Get name of directory with the data files
#dir_name = '/media/storage/gc_only/AS18/AS18_4Tastes_200228_151511/'
#metadata_handler = imp_metadata([[], dir_name])
metadata_handler = imp_metadata(sys.argv)
dir_name = metadata_handler.dir_name
os.chdir(dir_name)
print(f'Processing : {dir_name}')

# Open the hdf5 file
hf5 = tables.open_file(metadata_handler.hdf5_name, 'r+')

# Read CAR groups from info file
# Every region is a separate group, multiple ports under single region is a separate group,
# emg is a separate group
info_dict = metadata_handler.info_dict
params_dict = metadata_handler.params_dict
transform_type = params_dict['reference_transform_type']
electrode_layout_frame = metadata_handler.layout
# Remove emg and none channels from the electrode layout frame
emg_bool = ~electrode_layout_frame.CAR_group.str.contains('emg')
none_bool = ~electrode_layout_frame.CAR_group.str.contains('none')
fin_bool = np.logical_and(emg_bool, none_bool)
electrode_layout_frame = electrode_layout_frame[fin_bool]

# Since electrodes are already in monotonic numbers (A : 0-31, B: 32-63)
# we can directly pull them
grouped_layout = list(electrode_layout_frame.groupby('CAR_group'))
all_car_group_names = [x[0] for x in grouped_layout]
# Note: electrodes in HDF5 are also saved according to inds
# specified in the layout file
all_car_group_vals = [x[1].electrode_ind.values for x in grouped_layout]

num_groups = len(all_car_group_vals)
print(f" Number of groups : {num_groups}")
for region, vals in zip(all_car_group_names, all_car_group_vals):
    print(f" {region} :: {vals}")

# Pull out the raw electrode nodes of the HDF5 file
raw_electrodes = hf5.list_nodes('/raw')

# Note: Order of error reduction
# Raw data > CAR > PCA on downsampled data > PCA on full data
# IncrementalPCA (both full and truncated) works just as well as PCA 
# but is more memory efficient AND faster (by about 2-20x) than PCA

if transform_type == 'pca':
    dat_len = raw_electrodes[0].shape[0]
    batch_size = 10000

    for group in range(num_groups):
        print(f'Processing Group {group} : {all_car_group_names[group]}')

        car_electrode_inds = all_car_group_vals[group]

        data_generator = lambda : return_subset(
                dat_len, car_electrode_inds, batch_size, raw_electrodes)
        n_iter = dat_len // batch_size
        ipca, delta_list = truncated_ipca(
                n_components = 3,
                n_iter = n_iter,
                tolerance = 1e-2,
                data_generator = data_generator
                )

        # Iterate through data in time (since PCA needs all electrodes together)
        # Perform subtraction and write back to the hdf5 file, appending to array
        # Step by sampling rate

        ## Test type after subtraction
        #raw_dat = data_generator()
        #car_dat = raw_dat - np.mean(raw_dat, axis=1, keepdims=True)
        #ipca_dat = subtract_ipca(ipca, raw_dat)

        #fig, ax = plt.subplots(3, 1, sharex=True, sharey=True)
        #ax[0].plot(raw_dat, alpha=0.5)
        #ax[1].plot(car_dat, alpha=0.5)
        #ax[2].plot(processed_dat, alpha=0.5)
        #plt.show()

        # First, generate extendable arrays for the new data
        electrode_names = [f"electrode_pca{electrode_name:02}" \
                for electrode_name in all_car_group_vals[group]]
        for this_name in electrode_names: 
            if os.path.join('/raw', this_name) in hf5:
                hf5.remove_node('/raw', this_name)
            hf5.create_earray('/raw', this_name, tables.Float32Atom(), (0,))
        time_steps = np.arange(0, dat_len, params_dict['sampling_rate'])
        time_bins = list(zip(time_steps[:-1], time_steps[1:]))
        for this_bin in tqdm(time_bins):
            raw_dat = return_data_chunk(all_car_group_vals[group], this_bin)
            ipca_sub_dat = subtract_ipca(ipca, raw_dat)
            append_data_to_arrays(ipca_sub_dat, electrode_names)

        # Delete corresponding raw data and rename pca channels to raw
        for electrode_name in all_car_group_vals[group]:
            hf5.remove_node('/raw', f"electrode{electrode_name:02}")
        for electrode_name in all_car_group_vals[group]:
            hf5.rename_node(
                    '/raw', 
                    f"electrode{electrode_name:02}",
                    f"electrode_pca{electrode_name:02}"
                    ) 
        hf5.flush()

if transform_type == 'car':
    # First get the common average references by averaging across
    # the electrodes picked for each group
    print(
        "Calculating common average reference for {:d} groups".format(num_groups))
    common_average_reference = np.zeros(
        (num_groups, raw_electrodes[0][:].shape[0]))
    print('Calculating mean values')
    for group in range(num_groups):
        print('Processing Group {}'.format(group))
        # First add up the voltage values from each electrode to the same array
        # then divide by number of electrodes to get the average
        # This is more memory efficient than loading all the electrode data into
        # a single array and then averaging
        for electrode_name in tqdm(all_car_group_vals[group]):
            common_average_reference[group, :] += \
                get_electrode_by_name(raw_electrodes, electrode_name)[:]
        common_average_reference[group, :] /= float(len(all_car_group_vals[group]))

    # Now run through the raw electrode data and
    # subtract the common average reference from each of them
    print('Performing background subtraction')
    for group_num, group_name in tqdm(enumerate(all_car_group_names)):
        print(f"Processing group {group_name}")
        for electrode_num in tqdm(all_car_group_vals[group_num]):
            # Subtract the common average reference for that group from the
            # voltage data of the electrode
            raw_electrodes = hf5.list_nodes('/raw')
            referenced_data = get_electrode_by_name(raw_electrodes, electrode_num)[:] - \
                common_average_reference[group_num]
            # First remove the node with this electrode's data
            # This closes the node in the raw_electrodes list, therefore
            # we have to redefine the list each iteration
            hf5.remove_node(f"/raw/electrode{electrode_num:02}")
            # Now make a new array replacing the node removed above with the referenced data
            hf5.create_array(
                "/raw", f"electrode{electrode_num:02}", referenced_data)
            del referenced_data
            hf5.flush()

hf5.close()
print("Modified electrode arrays written to HDF5 file after "
      "subtracting the common average reference")
