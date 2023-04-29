# Import stuff!
import numpy as np
import tables
import easygui
import sys
import os
import matplotlib.pyplot as plt
import pymc3 as pm
import seaborn as sns
import theano
sns.set(context="poster")

# Ask for the directory where the hdf5 file sits, and change to that directory
dir_name = easygui.diropenbox()
os.chdir(dir_name)

# Look for the hdf5 file in the directory
file_list = os.listdir('./')
hdf5_name = ''
for files in file_list:
	if files[-2:] == 'h5':
		hdf5_name = files

# Open the hdf5 file
hf5 = tables.open_file(hdf5_name, 'r+')

# Get the digital inputs/tastes available
trains_dig_in = hf5.list_nodes('/spike_trains')

# Get the pre-stimulus time from the user
pre_stim = easygui.multenterbox(msg = 'Enter the pre-stimulus time for the spike trains', fields = ['Pre stim (ms)'])
pre_stim = int(pre_stim[0])

# Ask the user about the type of units they want to do the calculations on (single or all units)
unit_type = easygui.multchoicebox(msg = 'Which type of units do you want to use?', choices = ('All units', 'Single units', 'Multi units', 'Custom choice'))
all_units = np.arange(trains_dig_in[0].spike_array.shape[1])
single_units = np.array([i for i in range(len(all_units)) if hf5.root.unit_descriptor[i]["single_unit"] == 1])
multi_units = np.array([i for i in range(len(all_units)) if hf5.root.unit_descriptor[i]["single_unit"] == 0])
chosen_units = []
if unit_type[0] == 'All units':
	chosen_units = all_units
elif unit_type[0] == 'Single units':
	chosen_units = single_units
elif unit_type[0] == 'Multi units':
	chosen_units = multi_units
else:
	chosen_units = easygui.multchoicebox(msg = 'Which units do you want to choose?', choices = ([i for i in all_units]))
	for i in range(len(chosen_units)):
		chosen_units[i] = int(chosen_units[i])
	chosen_units = np.array(chosen_units)

# Get the laser duration/lag combos
lasers = hf5.root.ancillary_analysis.laser_combination_d_l[:]
# Now sort the durations/lags in ascending order - the first combo is the control/laser off condition (0, 0) after sorting
lasers = lasers[lasers[:, 0].argsort(), :]
lasers = lasers[lasers[:, 1].argsort(), :]

# Ask the user for the number of MCMC samples wanted for estimation of firing rates
num_samples = easygui.multenterbox(msg = 'Enter the number of MCMC samples you want to estimate firing rates (10000 is reasonable)', fields = ['Number of MCMC samples'])
num_samples = int(num_samples[0])

# Ask the user to specify the significance level to use for finding laser effects
# Sig level here corresponds to mass of posterior density overlapping zero
sig_level = easygui.multenterbox(msg = "Enter the significance level you want to use to detect firing suppression/enhancement effects due to lasers", fields = ["Significance level"])
sig_level = float(sig_level[0])

# Make an array to store the results of the laser effect analysis of dimensions units X laser_conditions X tastes X MCMC samples X 2 (laser on/off)
#results = np.zeros((len(chosen_units), len(lasers) - 1, len(trains_dig_in), num_samples, 2))
results = []

# Make a file to store the eventual results for all units in this dataset
# Reports the number of units under each column
# f = open('laser_effects_results.txt', 'w') 
# print("Taste" + '\t' + "Laser condition" + '\t' + "Decrease firing" + '\t' + "Increase firing" + '\t' + , file = f)

# We will make 2 tables in the HDF5 file to store the results from this analysis
# 1.) A table for each unit showing suppresion, enhancement or no effect of laser being on
# 2.) A table adding all these results across units, showing number of units suppresed, enhanced or unchanged in every laser condition
# Both these sets of tables can be made from the same class definition
class laser_effects(tables.IsDescription):
	laser = tables.Int32Col()	
	taste = tables.Int32Col()
	suppressed = tables.Int32Col()
	enhanced = tables.Int32Col()
	unchanged = tables.Int32Col()
	# Add a column that is 1 if the control condition shows firing that is not different from zero - suppression effects of the laser cannot be observed in this case
	control_zero = tables.Int32Col() 

# Make a folder to store kdeplots for the units from the analysis
try:
	os.system('rm -r '+'./laser_response_plots')
except:
	pass
os.mkdir('./laser_response_plots')

# Store the results in the hdf5 file in /root/laser_effects_bayesian
# Check if the node already exists, delete it if it does, and remake it
try:
	hf5.remove_node('/laser_effects_bayesian', recursive = True)
except:
	pass
hf5.create_group('/', 'laser_effects_bayesian')
# Make another node under /root/laser_effects_bayesian to save the tables for every unit's results
hf5.create_group('/laser_effects_bayesian', 'unit_summaries')

# Make an array to tally up these results across units - the number of rows = num of tastes x num of laser conditions and num of columns = 4 (suppressed, enhanced, unchanged, control_zero) - same 
# as in the laser_effects class above 
combined_units_effects = np.zeros(((lasers.shape[0] - 1) * len(trains_dig_in), 4))

# Run through the chosen units
# Include a progress counter
print("====================================")
print("Starting Bayesian analysis of the effects of laser on firing rate of units")
for unit in range(len(chosen_units)):
	print("Looking at Unit {} of {}".format(unit + 1, len(chosen_units)))
	# And run through the laser conditions
	spikes = []
	tastes = []
	laser_condition = []
	bayesian_results = np.zeros((len(lasers) - 1, len(trains_dig_in), num_samples, 2))
	for laser_status in range(lasers.shape[0] - 1):
		condition = lasers[laser_status + 1, :]
		# And finally, run through the tastes and pull all the data into respective arrays
		for stimulus in range(len(trains_dig_in)):
			# Get the correct trial numbers and controls for this laser condition
			trials = np.where((trains_dig_in[stimulus].laser_durations[:] == condition[0])*(trains_dig_in[stimulus].laser_onset_lag[:] == condition[1]))[0]	
			controls = np.where((trains_dig_in[stimulus].laser_durations[:] == 0.0)*(trains_dig_in[stimulus].laser_onset_lag[:] == 0.0))[0]

			# Append the data from these trials to spikes
			spikes.append(np.sum(trains_dig_in[stimulus].spike_array[trials, chosen_units[unit], pre_stim + condition[1]:pre_stim + condition[1] + condition[0]], axis = 1))
			# And append the appropriate control data
			spikes.append(np.sum(trains_dig_in[stimulus].spike_array[controls, chosen_units[unit], pre_stim + condition[1]:pre_stim + condition[1] + condition[0]], axis = 1))

			# Append the condition markers - if the laser_status number is 0, its control is 1 (so they go in pairs)
			laser_condition.append([laser_status*2 for i in range(len(trials))])
			laser_condition.append([(laser_status*2 + 1) for i in range(len(controls))])

			# Append the taste markers
			tastes.append([stimulus for i in range(len(trials) + len(controls))])

	# Convert all the data to numpy arrays
	spikes = np.array(spikes).flatten()
	tastes = np.array(tastes).flatten()
	laser_condition = np.array(laser_condition).flatten()

	# Make pymc3 model for these data - assume: 1.) Poisson emissions, 2.) log link, 3.) Hierarchies for a) taste, b) laser, c)interaction. Estimate main effects for tastes and laser conditions, plus interactions
	with pm.Model() as model:
		mu_b_t = pm.Normal('mu_b_t', mu = 0, sd = 10)
		sigma_b_t = pm.HalfCauchy('sigma_b_t', 1)
		mu_b_l = pm.Normal('mu_b_l', mu = 0, sd = 10)
		sigma_b_l = pm.HalfCauchy('sigma_b_l', 1)
		mu_b_t_l = pm.Normal('mu_b_t_l', mu = 0, sd = 10)
		sigma_b_t_l = pm.HalfCauchy('sigma_b_t_l', 1)
		
		b_t_offset = pm.Normal('b_t_offset', mu = 0, sd = 1, shape = len(trains_dig_in))
		b_t = pm.Deterministic('b_t', mu_b_t + b_t_offset*sigma_b_t)

		b_l_offset = pm.Normal('b_l_offset', mu = 0, sd = 1, shape = 2*(lasers.shape[0] - 1))
		b_l = pm.Deterministic('b_l', mu_b_l + b_l_offset*sigma_b_l)

		b_t_l_offset = pm.Normal('b_t_l_offset', mu = 0, sd = 1, shape = (len(trains_dig_in), 2*(lasers.shape[0] - 1)))
		b_t_l = pm.Deterministic('b_t_l', mu_b_t_l + b_t_l_offset*sigma_b_t_l)

		p = np.exp(b_t[tastes] + b_l[laser_condition] + b_t_l[tastes, laser_condition])

		output = pm.Poisson('spikes', mu = p, observed = spikes)

	# Sample from the model - using 2 chains in parallel (minimum to compare traceplots and rhat values)
	# Eventually variational inference with advi seems a better prospect - NUTS is too slow/finicky to sample
	# The logic of using ADVI here follows from: https://pymc-devs.github.io/pymc3/notebooks/bayesian_neural_network_opvi-advi.html
	# And also from: https://pymc-devs.github.io/pymc3/notebooks/variational_api_quickstart.html
	# Here we aren't scaling the variance of the gradient as it doesn't seem to give much improvement in simple models (look at the first link above)
	
	# We also use callbacks similar to those used in the 'init' portion of pm.sample - this stops ADVI once it has converged/ELBO doesn't change beyond a threshold
	cb = [pm.callbacks.CheckParametersConvergence(diff='absolute', tolerance = 1e-4), pm.callbacks.CheckParametersConvergence(diff='relative', tolerance = 1e-4),]
	with model:
		#trace = pm.sample(num_samples + 1000, tune = 1000)[1000:]
		#v_params = pm.variational.advi(n = 200000)
		#trace = pm.variational.sample_vp(v_params, draws=num_samples)
		inference = pm.fit(n=200000, method = 'fullrank_advi', callbacks = cb)
		trace = inference.sample(num_samples)

	# Print the Gelman-Rubin statistics for this model to file
	#print('\n', file = f)
	#print("======================== Unit {} ============================", file = f)
	#print(pm.diagnostics.gelman_rubin(trace), file = f)
	#print("=============================================================", file = f)

	# Run through the laser conditions and tastes again, and save the model results in results
	# The strategy now is to run through the MCMC samples, and calculate the difference in the Poisson mean between the laser on and off conditions for the different tastes
	for laser_status in range(lasers.shape[0] - 1):
		for stimulus in range(len(trains_dig_in)):
			# First calculate the mean firing rate for the laser off (control) condition for this taste
			bayesian_results[laser_status, stimulus, :, 0] = np.exp(trace['b_t'][:, stimulus] + trace['b_l'][:, 2*laser_status + 1] + trace['b_t_l'][:, stimulus, 2*laser_status + 1])

			# Then calculate the mean firing rate for the laser on condition for this taste
			bayesian_results[laser_status, stimulus, :, 1] = np.exp(trace['b_t'][:, stimulus] + trace['b_l'][:, 2*laser_status] + trace['b_t_l'][:, stimulus, 2*laser_status])

	# Append everything to results
	results.append(bayesian_results)

	# We cannot simply subtract the interaction effect coefficients in the two conditions and conclude significance by comparing them to zero
	# The interaction effects are calculated separately for every condition, we can compare them to each other, but cannot subtract them (esp because of the log link in the equation)
	# So ignore the analysis below
	# Also check if this unit has significant taste-laser interaction (by checking if the b_t_l offsets (for control minus condition) are significantly different from zero). We will check if the HPD overlaps zero for them
#	for laser_status in range(lasers.shape[0] - 1):
#		for stimulus in range(len(trains_dig_in)):
#			interaction_diff = trace['b_t_l_offset'][:, stimulus, 2*laser_status + 1] - trace['b_t_l_offset'][:, stimulus, 2*laser_status]
#			if pm.hpd(interaction_diff)[0] * pm.hpd(interaction_diff)[1] > 0:
#				print("{}".format(chosen_units[unit]) + '\t' + "{}".format(stimulus) + '\t' + "Dur:{}ms, Lag:{}ms".format(lasers[laser_status + 1, 0], lasers[laser_status + 1, 1]), file = f)

	# Make a directory for this unit under laser_response_plots, and save kdeplots there
	os.mkdir('./laser_response_plots/Unit{:d}'.format(chosen_units[unit]))
	# Get the difference in the mean firing rate between the control and laser on conditions
	diff = bayesian_results[:, :, :, 0] - bayesian_results[:, :, :, 1]
	# First generate plots by laser conditions
	for laser_status in range(lasers.shape[0] - 1):
		fig = plt.figure()
		for stimulus in range(len(trains_dig_in)):
			sns.kdeplot(diff[laser_status, stimulus, :]*(1000.0/lasers[laser_status + 1, 0]), cumulative = True, label = 'Taste {}'.format(stimulus))
		plt.legend(loc = 'upper left')
		fig.set_size_inches(18.5, 10.5)
		plt.xlabel('Difference in the mean firing of control and laser conditions (Hz)')
		plt.ylabel('Cumulative Probability')
		fig.savefig("./laser_response_plots/Unit{:d}/Dur:{}ms,Lag:{}ms.png".format(chosen_units[unit], lasers[laser_status + 1, 0], lasers[laser_status + 1, 1]), bbox_inches = 'tight')
		plt.close("all")

	# Then generate plots by tastes
	for stimulus in range(len(trains_dig_in)):
		fig = plt.figure()
		for laser_status in range(lasers.shape[0] - 1):
			sns.kdeplot(diff[laser_status, stimulus, :]*(1000.0/lasers[laser_status + 1, 0]), cumulative = True, label = 'Dur:{}ms,Lag:{}ms'.format(lasers[laser_status + 1, 0], lasers[laser_status + 1, 1]))
		plt.legend(loc = 'upper left')
		fig.set_size_inches(18.5, 10.5)
		plt.xlabel('Difference in the mean firing of control and laser conditions (Hz)')
		plt.ylabel('Cumulative Probability')
		fig.savefig("./laser_response_plots/Unit{:d}/Taste{}.png".format(chosen_units[unit], stimulus), bbox_inches = 'tight')
		plt.close("all")

	# Store the traceplot from this analysis
	#fig, axis = plt.subplots(12, 2)
	#pm.traceplot(trace, ax = axis)
	#fig.set_size_inches(18.5, 10.5)
	#fig.savefig('laser_traceplots/Unit{}.png'.format(chosen_units[unit]), bbox_inches = 'tight')
	#plt.close('all')

	# Now use the array of differences plotted above to find if the laser had an effect on this unit's firing
	# Save these findings in a table specific to this unit
	unit_table = hf5.create_table('/laser_effects_bayesian/unit_summaries', 'unit{:d}'.format(chosen_units[unit]), description = laser_effects)

	# Now run through the tastes and laser conditions
	for laser in range(diff.shape[0]):
		for taste in range(diff.shape[1]):	
			# Get a new row for this taste and laser condition		
			this_condition_results = unit_table.row

			# Fill in the taste and laser conditions
			this_condition_results['laser'] = laser + 1		
			this_condition_results['taste'] = taste + 1
		
		
			# First check if the control firing was close to zero for this taste/laser combo (comparing it to a sufficiently small number because the control firing rate is always > 0 by definition)
			if pm.hpd(bayesian_results[laser, taste, :, 0], alpha = sig_level)[0] <= 1e-4:
				this_condition_results['control_zero'] = 1.0
			else:
				this_condition_results['control_zero'] = 0.0
			
			# Then check if the laser condition has no effect on firing (the diff HPD will overlap zero)
			diff_hpd = pm.hpd(diff[laser, taste, :], alpha = sig_level)
			if diff_hpd[0] * diff_hpd[1] < 0:
				this_condition_results['unchanged'] = 1.0
				this_condition_results['enhanced'] = 0.0
				this_condition_results['suppressed'] = 0.0
			# Firing is enhanced if the diff (control-laser) lies consistently below zero
			elif diff_hpd[0] < 0 and diff_hpd[1] < 0:
				this_condition_results['unchanged'] = 0.0
				this_condition_results['enhanced'] = 1.0
				this_condition_results['suppressed'] = 0.0
			# Firing is suppressed if the diff (control-laser) lies consistently above zero 
			else:
				this_condition_results['unchanged'] = 0.0
				this_condition_results['enhanced'] = 0.0
				this_condition_results['suppressed'] = 1.0

			# Also add to the count of units showing the different effects on firing
			combined_units_effects[diff.shape[1] * laser + taste, 0] += this_condition_results['suppressed']
			combined_units_effects[diff.shape[1] * laser + taste, 1] += this_condition_results['enhanced']
			combined_units_effects[diff.shape[1] * laser + taste, 2] += this_condition_results['unchanged']
			combined_units_effects[diff.shape[1] * laser + taste, 3] += this_condition_results['control_zero'] 

			# Append this row to the table
			this_condition_results.append()
			unit_table.flush()
			hf5.flush()

# Convert results to numpy array
results = np.array(results)
results.resize((len(chosen_units), len(lasers) - 1, len(trains_dig_in), num_samples, 2))

# Close the file that stores the identities of units with significant interaction effects
# f.close()

print("Bayesian analysis of the effects of laser on firing rate of units finished")
print("====================================")

# Then save the data to the file
hf5.create_array('/laser_effects_bayesian', 'laser_combination_d_l', lasers)
hf5.create_array('/laser_effects_bayesian', 'mean_firing_rates', results)

# Make a table to save the tally of units showing the different effects on firing
combined_units_table = hf5.create_table('/laser_effects_bayesian', 'combined_units_tally', description = laser_effects)

# Run through the laser and taste conditions
for laser in range(diff.shape[0]):
	for taste in range(diff.shape[1]):	
		# Get a new row for this taste and laser condition		
		this_condition_results = combined_units_table.row

		# Fill in the taste and laser conditions
		this_condition_results['laser'] = laser + 1		
		this_condition_results['taste'] = taste + 1

		# Fill up the data
		this_condition_results['suppressed'] = combined_units_effects[diff.shape[1] * laser + taste, 0]
		this_condition_results['enhanced'] = combined_units_effects[diff.shape[1] * laser + taste, 1]
		this_condition_results['unchanged'] = combined_units_effects[diff.shape[1] * laser + taste, 2]
		this_condition_results['control_zero'] = combined_units_effects[diff.shape[1] * laser + taste, 3]

		# Append this row to the table
		this_condition_results.append()
		combined_units_table.flush()
		hf5.flush()

hf5.close()
	
	
