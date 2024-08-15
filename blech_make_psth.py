# Import stuff!
import numpy as np
import tables
import easygui
import sys
import os
import ast
import json
import matplotlib
import shutil
import pylab as plt
from scipy.stats import ttest_ind
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

# Open the hdf5 file
hf5 = tables.open_file(metadata_handler.hdf5_name, 'r+')

params_dict = metadata_handler.params_dict
pre_stim = params_dict['spike_array_durations'][0]
print(f'pre-stim time :: {pre_stim}')

params = [params_dict['psth_params']['window_size'], 
            params_dict['psth_params']['step_size']]
print(f'PSTH :: window_size : {params[0]}, step_size : {params[1]}')

# Make directory to store the PSTH plots. 
# Delete and remake the directory if it exists
try:
    shutil.rmtree("PSTH")
except:
	pass
os.mkdir("PSTH")

# Make directory to store the raster plots. 
# Delete and remake the directory if it exists
try:
    shutil.rmtree("raster")
except:
	pass
os.mkdir("raster")

# Get the list of spike trains by digital input channels
trains_dig_in = hf5.list_nodes('/spike_trains')

# Taste responsiveness calculation parameters
r_pre_stim = 500
r_post_stim = 2500

# Plot PSTHs and rasters by digital input channels
for dig_in in trains_dig_in:
	os.mkdir("./PSTH/" + str.split(dig_in._v_pathname, '/')[-1])
	os.mkdir("./raster/" + str.split(dig_in._v_pathname, '/')[-1])
	trial_avg_spike_array = np.mean(dig_in.spike_array[:], axis = 0)
	for unit in range(trial_avg_spike_array.shape[0]):
		time = []
		spike_rate = []
		for i in range(0, trial_avg_spike_array.shape[1] - params[0], params[1]):
			time.append(i - pre_stim)
			spike_rate.append(\
                                1000.0*np.sum(\
                                    trial_avg_spike_array[unit, i:i+params[0]])/float(params[0]))
		taste_responsiveness_t, taste_responsiveness_p = \
                        ttest_ind(\
                            np.mean(\
                                dig_in.spike_array[:, unit, pre_stim:pre_stim + r_post_stim], 
                                    axis = 1), 
                            np.mean(\
                                dig_in.spike_array[:, unit, pre_stim - r_pre_stim:pre_stim], 
                                    axis = 1))   
		fig = plt.figure() 
		plt.plot(time, spike_rate)
		plt.xlabel('Time from taste delivery (ms)')
		plt.ylabel('Firing rate (Hz)')
		plt.title('Window: %i ms, Step: %i ms, Taste responsive: %s' % (params[0], params[1], 
            str(bool(taste_responsiveness_p<0.001))) + \
                    '\n' + 'Electrode: %i, Single Unit: %i, RSU: %i, FS: %i' \
                    % (hf5.root.unit_descriptor[unit]['electrode_number'], 
                        hf5.root.unit_descriptor[unit]['single_unit'], 
                        hf5.root.unit_descriptor[unit]['regular_spiking'], 
                        hf5.root.unit_descriptor[unit]['fast_spiking']))
		#plt.tight_layout()
		fig.savefig('./PSTH/'+str.split(dig_in._v_pathname, '/')[-1]+'/Unit%i.png' % (unit))
		plt.close("all")

		# Now plot the rasters for this digital input channel and unit
		# Run through the trials
		time = np.arange(dig_in.spike_array[:].shape[2] + 1) - pre_stim
		fig = plt.figure()
		for trial in range(dig_in.spike_array[:].shape[0]):
			x = np.where(dig_in.spike_array[trial, unit, :] > 0.0)[0]
			plt.vlines(x, trial, trial + 1, colors = 'black')
		plt.xticks(np.arange(0, dig_in.spike_array[:].shape[2] + 1, 1000), time[::1000])
		#plt.yticks(np.arange(0, dig_in.spike_array[:].shape[0] + 1, 5))
		plt.title('Unit: %i raster plot' % (unit) + '\n' + \
                'Electrode: %i, Single Unit: %i, RSU: %i, FS: %i' \
                % (hf5.root.unit_descriptor[unit]['electrode_number'], 
                    hf5.root.unit_descriptor[unit]['single_unit'], 
                    hf5.root.unit_descriptor[unit]['regular_spiking'], 
                    hf5.root.unit_descriptor[unit]['fast_spiking']))	
		plt.xlabel('Time from taste delivery (ms)')
		plt.ylabel('Trial number')
		#plt.tight_layout()
		fig.savefig('./raster/'+str.split(dig_in._v_pathname, '/')[-1]+'/Unit%i.png' % (unit))
		plt.close("all")
		
		# Check if the laser_array exists, and plot laser PSTH if it does
		laser_exists = []		
		try:
			laser_exists = dig_in.laser_durations[:]
		except:
			pass
		if len(laser_exists) > 0:
			# First get the unique laser onset times (from end of taste delivery) in this dataset
			onset_lags = np.unique(dig_in.laser_onset_lag[:])
			# Then get the unique laser onset durations
			durations = np.unique(dig_in.laser_durations[:])

			# Then go through the combinations of the durations and onset lags and get and plot an averaged spike_rate array for each set of trials
			fig = plt.figure()
			for onset in onset_lags:
				for duration in durations:
					spike_rate = []
					time = []
					these_trials = np.where((dig_in.laser_durations[:] == duration)*(dig_in.laser_onset_lag[:] == onset) > 0)[0]
					# If no trials have this combination of onset lag and duration (can happen when duration = 0, laser off), break out of the loop
					if len(these_trials) == 0:
						continue
					trial_avg_array = np.mean(dig_in.spike_array[these_trials, :, :], axis = 0)
					for i in range(0, trial_avg_array.shape[1] - params[0], params[1]):
						time.append(i - pre_stim)
						spike_rate.append(1000.0*np.sum(trial_avg_array[unit, i:i+params[0]])/float(params[0]))
					# Now plot the PSTH for this combination of duration and onset lag
					plt.plot(time, spike_rate, linewidth = 3.0, label = 'Dur: %i ms, Lag: %i ms' % (int(duration), int(onset)))

			plt.title('Unit: %i laser PSTH, Window: %i ms, Step: %i ms' % \
                    (unit, params[0], params[1]) + '\n' + \
                    'Electrode: %i, Single Unit: %i, RSU: %i, FS: %i' \
                    % (hf5.root.unit_descriptor[unit]['electrode_number'], 
                        hf5.root.unit_descriptor[unit]['single_unit'], 
                        hf5.root.unit_descriptor[unit]['regular_spiking'], 
                        hf5.root.unit_descriptor[unit]['fast_spiking']))
			plt.xlabel('Time from taste delivery (ms)')
			plt.ylabel('Firing rate (Hz)')
			plt.legend(loc = 'upper left', fontsize = 15)
			#plt.tight_layout()
			fig.savefig('./PSTH/'+str.split(dig_in._v_pathname, '/')[-1]+'/Unit%i_laser_psth.png' % (unit))
			plt.close("all")

			# And do the same to get the rasters
			for onset in onset_lags:
				for duration in durations:
					time = np.arange(dig_in.spike_array[:].shape[2] + 1) - pre_stim
					these_trials = np.where((dig_in.laser_durations[:] == duration)\
                            *(dig_in.laser_onset_lag[:] == onset) > 0)[0]
					# If no trials have this combination of onset lag and duration 
                    # (can happen when duration = 0, laser off), break out of the loop
					if len(these_trials) == 0:
						continue
					fig = plt.figure()
					# Run through the trials
					for i in range(len(these_trials)):
						x = np.where(dig_in.spike_array[these_trials[i], unit, :] > 0.0)[0]
						plt.vlines(x, i, i + 1, colors = 'black')	
					plt.xticks(np.arange(0, dig_in.spike_array[:].shape[2] + 1, 1000), time[::1000])
					#plt.yticks(np.arange(0, len(these_trials) + 1, 5))
					plt.title('Unit: %i Dur: %i ms, Lag: %i ms' % \
                            (unit, int(duration), int(onset)) + '\n' + \
                            'Electrode: %i, Single Unit: %i, RSU: %i, FS: %i' % \
                            (hf5.root.unit_descriptor[unit]['electrode_number'], 
                                hf5.root.unit_descriptor[unit]['single_unit'], 
                                hf5.root.unit_descriptor[unit]['regular_spiking'], 
                                hf5.root.unit_descriptor[unit]['fast_spiking']))	
					plt.xlabel('Time from taste delivery (ms)')
					plt.ylabel('Trial number')
					#plt.tight_layout()
					fig.savefig('./raster/'+str.split(dig_in._v_pathname, '/')[-1] + \
                            '/Unit%i_Dur%ims_Lag%ims.png' % (unit, int(duration), int(onset)))
					plt.close("all")	

# Also plot PSTHs for all the digital inputs/tastes together, on the same scale, to help in comparison

# First ask the user for the time limits of plotting
#plot_lim = easygui.multenterbox(msg = 'Enter the time limits for plotting combined PSTHs', 
#                        fields = ['Start time (ms)', 'End time (ms)'])
#for i in range(len(plot_lim)):
#	plot_lim[i] = int(plot_lim[i])
plot_lim = params_dict['psth_params']['durations']
print(f'PSTH ::: start : {plot_lim[0]}, end : {plot_lim[1]}')

# Get number of units
num_units = trains_dig_in[0].spike_array[:].shape[1]

# Run through the units
for unit in range(num_units):
	# Load up the data from all the digital inputs
	data = []
	for dig_in in trains_dig_in:
		data.append(np.mean(dig_in.spike_array[:, unit, :], axis = 0))
	# Convert into a big numpy array of all the data for this unit, from all the digital inputs
	data = np.array(data)

	# Now get ready for the plotting by first making the axes (both x and y axis will be shared across plots)
	fig, ax = plt.subplots(len(trains_dig_in), sharex=True, sharey=True)

	# Now run through the tastes and make the plots
	for taste in range(len(trains_dig_in)):
		time = []
		spike_rate = []
		for i in range(0, data.shape[1] - params[0], params[1]):
			time.append(i - pre_stim)
			spike_rate.append(1000.0*np.sum(data[taste, i:i+params[0]])/float(params[0]))

		# Get the points to be plotted
		spike_rate = np.array(spike_rate)
		time = np.array(time)
		plot_points = np.where((time >= plot_lim[0])*(time <= plot_lim[1]))[0]

		ax[taste].plot(time[plot_points], spike_rate[plot_points], label = 'Taste {:d}'.format(taste+1))
		ax[taste].legend(loc = 'upper right', fontsize = 15)
	ax[0].set_title("Unit: {:d}, Window: {:d} ms, Step: {:d} ms".format(unit, params[0], params[1]))	
	# Bring the plots closer together
	fig.subplots_adjust(hspace=0)
	# Remove xticks from all but the last plot
	plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
	fig.text(0.5, 0.02, 'Time from taste delivery (ms)', ha='center')
	fig.text(0.03, 0.5, 'Firing rate (Hz)', va='center', rotation='vertical')	
	#plt.tight_layout()
	
	# Save the combined plot
	fig.savefig("./PSTH/Unit{:d}_combined_PSTH.png".format(unit))
	plt.close("all")

	# Check if the laser_array exists, and plot laser PSTH if it does
	laser_exists = []		
	try:
		laser_exists = dig_in.laser_durations[:]
	except:
		pass
	if len(laser_exists) > 0:
		# Now get ready for the plotting by first making the axes (both x and y axis will be shared across plots)
		fig, ax = plt.subplots(len(trains_dig_in), sharex=True, sharey=True)
		# Run through the tastes
		for taste in range(len(trains_dig_in)):
			# First get the unique laser onset times (from end of taste delivery) in this dataset
			onset_lags = np.unique(trains_dig_in[taste].laser_onset_lag[:])
			# Then get the unique laser onset durations
			durations = np.unique(trains_dig_in[taste].laser_durations[:])

			# Then go through the combinations of the durations and onset lags and 
            # get and plot an averaged spike_rate array for each set of trials
			for onset in onset_lags:
				for duration in durations:
					spike_rate = []
					time = []
					these_trials = np.where((trains_dig_in[taste].laser_durations[:] == duration)\
                            *(trains_dig_in[taste].laser_onset_lag[:] == onset) > 0)[0]
					# If no trials have this combination of onset lag and duration 
                    # (can happen when duration = 0, laser off), break out of the loop
					if len(these_trials) == 0:
						continue
					trial_avg_array = np.mean(trains_dig_in[taste].spike_array[these_trials, :, :], axis = 0)
					for i in range(0, trial_avg_array.shape[1] - params[0], params[1]):
						time.append(i - pre_stim)
						spike_rate.append(1000.0*np.sum(trial_avg_array[unit, i:i+params[0]])/float(params[0]))

					# Get the points to be plotted
					spike_rate = np.array(spike_rate)
					time = np.array(time)
					plot_points = np.where((time >= plot_lim[0])*(time <= plot_lim[1]))[0]
					# Now plot the PSTH for this combination of duration and onset lag
					ax[taste].plot(time[plot_points], spike_rate[plot_points], \
                            label = "Taste:{:d}, Dur:{:d} ms, Lag:{:d} ms".format(taste+1, int(duration), int(onset)))

			ax[taste].legend(loc = 'upper right', fontsize = 10)
		
		# Bring the plots closer together
		fig.subplots_adjust(hspace=0)
		# Remove xticks from all but the last plot
		plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
		fig.text(0.5, 0.02, 'Time from taste delivery (ms)', ha='center')
		fig.text(0.03, 0.5, 'Firing rate (Hz)', va='center', rotation='vertical')
		#plt.tight_layout()		

		# Save the combined plot
		ax[0].set_title("Unit: {:d}, Window size: {:d} ms, Step size: {:d} ms".format(unit, params[0], params[1]))
		fig.savefig("./PSTH/Unit{:d}_combined_laser_PSTH.png".format(unit))
		plt.close("all")

hf5.close()

# Write successful execution to log
this_pipeline_check.write_to_log(script_path)
