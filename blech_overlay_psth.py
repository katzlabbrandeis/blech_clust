# Import stuff!
import numpy as np
import tables
import pylab as plt
import easygui
import sys
import os
import json
import glob
from utils.blech_utils import imp_metadata, pipeline_graph_check

# Get name of directory with the data files
metadata_handler = imp_metadata(sys.argv)
dir_name = metadata_handler.dir_name

# Perform pipeline graph check
script_path = os.path.realpath(__file__)
this_pipeline_check = pipeline_graph_check(dir_name)
this_pipeline_check.check_previous(script_path)

os.chdir(dir_name)
print(f'Processing : {dir_name}')

params_dict = metadata_handler.params_dict
info_dict = metadata_handler.info_dict

# Open the hdf5 file
hf5 = tables.open_file(metadata_handler.hdf5_name, 'r+')

# Make directory to store the PSTH plots. Delete and remake the directory if it exists
if os.path.exists('./overlay_PSTH'):
    os.system('rm -r '+'./overlay_PSTH')
os.mkdir('./overlay_PSTH')

# Now ask the user to put in the identities of the digital inputs
trains_dig_in = hf5.list_nodes('/spike_trains')
# Pull identities from the json file
identities = info_dict['taste_params']['tastes'] 

# Plot all tastes
plot_tastes_dig_in = np.arange(len(identities))

pre_stim, post_stim = params_dict['spike_array_durations']

window_size = params_dict['psth_params']['window_size'] 
step_size = params_dict['psth_params']['step_size']
psth_dur = params_dict['psth_params']['durations']

spike_array_durations = params_dict['spike_array_durations']
spike_array_x = np.arange(-pre_stim, post_stim)
binned_x = spike_array_x[:-window_size+step_size][::step_size]

# Ask the user about the type of units they want to do the calculations on (single or all units)
chosen_units = np.arange(trains_dig_in[0].spike_array.shape[1])

# Extract neural response data from hdf5 file
# response : neurons x trials x tastes x time
response = hf5.root.ancillary_analysis.unscaled_neural_response[:]
num_units = len(chosen_units)
num_tastes = len(trains_dig_in)
plot_places = np.where((binned_x>=-psth_dur[0])*(binned_x<=psth_dur[1]))[0]

for i in range(num_units):
    fig, ax = plt.subplots(1,2,figsize=(18, 6))

    # First plot
    ax[0].set_title('Unit: %i, Window size: %i ms, Step size: %i ms' % \
            (chosen_units[i], window_size, step_size)) 
    for j in plot_tastes_dig_in:
            ax[0].plot(binned_x[plot_places], 
                    np.nanmean(response[j, :, i, plot_places], 
                        axis = 1), label = identities[j])
    ax[0].legend()
    ax[0].set_xlabel('Time from taste delivery (ms)')
    ax[0].set_ylabel('Firing rate (Hz)')

    # Second plot
    max_waveforms = 10000
    waveforms = hf5.get_node(f'/sorted_units/unit{chosen_units[i]:03}','waveforms')[:]
    orig_count = len(waveforms)
    if orig_count > max_waveforms:
        wav_inds = np.random.choice(np.arange(len(waveforms)), max_waveforms, replace=False)
    else:
        wav_inds = np.arange(len(waveforms))
    waveforms = waveforms[wav_inds]
    #exec('waveforms = hf5.root.sorted_units.unit%03d.waveforms[:]' % (chosen_units[i]))
    t = np.arange(waveforms.shape[1])
    ax[1].plot(t - 15, waveforms.T, linewidth = 0.01, color = 'red')
    ax[1].set_xlabel('Time (samples (30 per ms))')
    ax[1].set_ylabel('Voltage (microvolts)')
    title_str = f"Unit {chosen_units[i]}," \
                f"Total waveforms = {orig_count}\n"\
                f"Electrode: {hf5.root.unit_descriptor[chosen_units[i]]['electrode_number']},"\
                f"Single Unit: {hf5.root.unit_descriptor[chosen_units[i]]['single_unit']},"\
                f"RSU: {hf5.root.unit_descriptor[chosen_units[i]]['regular_spiking']},"\
                f"FS: {hf5.root.unit_descriptor[chosen_units[i]]['fast_spiking']}"
    ax[1].set_title(title_str)
    fig.savefig('./overlay_PSTH/' + '/Unit%i.png' % (chosen_units[i]), bbox_inches = 'tight')
    plt.close("all")
    print(f'Completed Unit {i}')

    # Also plot heatmaps of the PSTH for all tastes
    fig, ax = plt.subplots(1, len(identities), figsize=(18, 6),
                           sharex=True, sharey=True)
    neuron_response = response[:, :, i, :]
    min_val, max_val = np.nanmin(neuron_response), np.nanmax(neuron_response)
    for j in plot_tastes_dig_in:
        ax[j].set_title(identities[j])
        ax[j].pcolormesh(binned_x[plot_places], np.arange(response.shape[1]),
                neuron_response[j,...,plot_places].T,
                cmap = 'jet', vmin = min_val, vmax = max_val)
        ax[j].set_xlabel('Time from taste delivery (ms)')
        ax[j].set_ylabel('Trial number')
        ax[j].axvline(0, color = 'k', linestyle = '--', linewidth = 5)

    # Create a common colorbar below the subplots
    # Min and Max will automatically be set to the min and max of the data
    cax = fig.add_axes([0.2, -0.05, 0.6, 0.05])
    cbar = fig.colorbar(ax[0].collections[0], cax=cax, orientation='horizontal')
    cbar.ax.set_xlabel('Firing rate (Hz)')

    fig.suptitle('Unit: %i, Window size: %i ms, Step size: %i ms' % \
            (chosen_units[i], window_size, step_size))
    fig.savefig('./overlay_PSTH/' + '/Unit%i_heatmap.png' % (chosen_units[i]), bbox_inches = 'tight')
    plt.close(fig)

# Close hdf5 file
hf5.close()

# Write successful execution to log
this_pipeline_check.write_to_log(script_path)
