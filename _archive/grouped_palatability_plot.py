#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 12:04:44 2020

@author: bradly
"""
# import Libraries
# Built-in Python libraries
import os # functions for interacting w operating system

# 3rd-party libraries
import numpy as np # module for low-level scientific computing
import easygui
import tables
import pandas as pd
from pandas.plotting import table
from datetime import date
import sys

#Import plotting tools
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_color_codes(palette = 'colorblind')

#Have user indicate what directory the hdf5 file resides in
dir_folder = easygui.diropenbox(msg = 'Choose where the hdf5 you want to edit is located.')
os.chdir(dir_folder)

#Look for the hdf5 files in the directory
file_list = os.listdir('./')
file_names = []
for files in file_list:
    if files[-2:] == 'h5':
        file_names.append(files)

hf5_files = easygui.multchoicebox(
        msg = 'Which file do you want to work from?',
        choices = ([x for x in file_names]))

#Open hdf5
hf5 = tables.open_file(hf5_files[0], 'r+')

#Extract animal/node names
animals = [node._v_name for node in hf5.list_nodes('/')]

#Ask user to choose which nodes need to have work done
rm_animals = easygui.multchoicebox(
        msg = 'Click on animals that you wish to EXCLUDE from analyses',
        choices = ([x for x in animals]))
try:
	val = len(rm_animals)
	analysis_nodes = [node for node in animals if node not in rm_animals]
except:
	analysis_nodes = animals

#Check if there are multiple conditions per animal
cond_check = []; cond_detail = []
for animal in range(len(analysis_nodes)):
	groups = [node._v_name for node in hf5.list_nodes('/%s' %(analysis_nodes[animal]))]
	cond_check.append([analysis_nodes[animal],len(groups)])
	cond_detail.append(groups)

if all([element[1] > 1 for element in cond_check]) is True:

	#Add identifiers for table creation
	cond_check.insert(0, ['Animal','Recordings'])

	#Create table
	df = pd.DataFrame(cond_check)

	#Format table for viewing
	fig, ax = plt.subplots(figsize=(3, 2)) # set size frame
	ax.xaxis.set_visible(False)  # hide the x axis
	ax.yaxis.set_visible(False)  # hide the y axis
	ax.set_frame_on(False)  # no visible frame, uncomment if size is ok
	tabla = table(ax, df, loc='upper right', colWidths=[0.5]*len(df.columns))  # where df is your data frame
	tabla.auto_set_font_size(False) # Activate set fontsize manually
	tabla.set_fontsize(12) # if ++fontsize is necessary ++colWidths
	tabla.scale(1, 1) # change size table
	plt.savefig('table.tif', transparent=True)

	image = 'table.tif'
	msg   = "It appears the following animals have several recordings,\
			treat them separate conditions or can they be merged?"
	condition_check = easygui.buttonbox(msg,image=image, choices = ["Separate","Merged"])

else:
	print('All animals only have one condition, and will be analyzed together.')
	condition_check = 'Merged'

#Ask user about what units (and how) to perform analyses on (performs All units by default)
msg   = "Reffering to units, what kind of analysis do you wish to perform?"
unit_check = easygui.buttonbox(msg,choices = ["All units","Held units","Descriminitive_Responsive","Custom"])

if unit_check == "Held units":
	held_check = "Yes"
	discrimresp_check = "No"
if unit_check == "Descriminitive_Responsive":
	held_check = "No"
	discrimresp_check = "Yes"
	custom_check = "No"
	msg   = "Do you want to look at Descriminitive or Responsive units?"
	drtype_check = easygui.buttonbox(msg,choices = ["Descriminitive","Responsive"])
if unit_check == "Custom":
	custom_check = "Yes"
	discrimresp_check = "No"
	held_check = "No"

if held_check == "Yes":
	#Ask if held cells were detected using CAR or Raw?
	msg   = "Were held cells detected using Raw data or Common Average Referenced?"
	type_check = easygui.buttonbox(msg,choices = ["Raw","C.A.R."])

	held_dir_name = easygui.diropenbox(msg = 'Choose directory with all "_held_units.txt" files in it.',default = dir_folder)

	#Flip through all files and create a held_units list
	for file in held_dir_name:
		#Change to the directory
		os.chdir(held_dir_name)
		#Locate the hdf5 file
		file_list = os.listdir('./')
		held_file_name = ''

		day1 =[]; day2=[]; #Create arrays for each day
		all_day1=[]; all_day2=[] #Create arrays for combined days
		for files in sorted(file_list):
			if files[-3:] == 'txt' and files[:4] in analysis_nodes:

				if type_check == "C.A.R.":
					with open(files,'r') as splitfile:
						for columns in [line.split() for line in splitfile]:
							day1.append(columns[0]);	day2.append(columns[1])
						all_day1.append(day1); all_day2.append(day2)
						day1 = []; day2 = []    #Clear day array

				if type_check == "Raw":

					#Check to see if animal had held cells, if not store blank
					#Read in table, skipping unnecessary headers
					try:
						frame = pd.read_table(files, sep='\s+',skiprows=2)

						#Determine order of days to store in correct output files
						date_1=frame.columns[4][(frame.columns[4]).rfind('_')-6:-7]
						date_2=frame.columns[5][(frame.columns[5]).rfind('_')-6:-7]

						#Store units into array based on condition order
						if date_1<date_2:
							day_1_units = list(frame[str(frame.columns[4])])
							day_2_units = list(frame[str(frame.columns[5])])

							for x,y in zip(day_1_units,day_2_units):
							    day1.append(int(''.join(c for c in x if c.isdigit())))
							    day2.append(int(''.join(c for c in y if c.isdigit())))

							#append larger sets
							all_day1.append(day1); all_day2.append(day2)
							day1 = []; day2 = []    #Clear day array

						if date_1>date_2:
							day_1_units = list(frame[str(frame.columns[5])])
							day_2_units = list(frame[str(frame.columns[4])])

							for x,y in zip(day_1_units,day_2_units):
							    day1.append(int(''.join(c for c in x if c.isdigit())))
							    day2.append(int(''.join(c for c in y if c.isdigit())))

							#append larger sets
							all_day1.append(day1); all_day2.append(day2)
							day1 = []; day2 = []    #Clear day array

					except:
						#append larger sets
						all_day1.append([]); all_day2.append([])
						print('%s has no held units, sorry buddy!' %(files[:4]))

	if type_check == "C.A.R.":
		#Remove 'Day' from all lists
		for sublist in all_day1:
		    del sublist[0]

		for sublist in all_day2:
		    del sublist[0]

	#Account for out of order units
	sorted_day1 = [];sorted_day2=[]
	for animal in range(len(all_day1)):
		sorted_day1.append(sorted(all_day1[animal], key=int))
		sorted_day2.append(sorted(all_day2[animal], key=int))

if condition_check == 'Merged':

	#Flag node for dig_in removals
	rm_digin = []

	#Initiate empty arrays for merged data storage
	unique_lasers = []
	r_pearson = []
	r_spearman = []
	r_isotonic = []
	p_pearson = []
	p_spearman = []
	p_identity = []
	lda_palatability = []
	lda_identity = []
	taste_cosine_similarity = []
	taste_euclidean_distance = []
	pairwise_NB_identity = []
	p_discriminability = []
	pre_stim = []
	params = []
	id_pal_regress = []
	num_units = 0

	for node in range(len(analysis_nodes)):
		node_groups = [node._v_name for node in hf5.list_nodes('/%s' %(analysis_nodes[node]))]
		for day in range(len(node_groups)):

			#set path for ease of reading
			node_path = os.path.join('/',analysis_nodes[node],node_groups[day])

			# Pull the data from the /ancillary_analysis node
			unique_lasers.append(hf5.get_node(node_path,'laser_combination_d_l')[:])
			r_pearson.append(hf5.get_node(node_path,'r_pearson')[:])
			r_spearman.append(hf5.get_node(node_path,'r_spearman')[:])
			r_isotonic.append(hf5.get_node(node_path,'r_isotonic')[:])
			p_pearson.append(hf5.get_node(node_path,'p_pearson')[:])
			p_spearman.append(hf5.get_node(node_path,'p_spearman')[:])
			p_identity.append(hf5.get_node(node_path,'p_identity')[:])
			lda_palatability.append(hf5.get_node(node_path,'lda_palatability')[:])
			lda_identity.append(hf5.get_node(node_path,'lda_identity')[:])
			taste_cosine_similarity.append(hf5.get_node(node_path,'taste_cosine_similarity')[:])
			taste_euclidean_distance.append(hf5.get_node(node_path,'taste_euclidean_distance')[:])
			pairwise_NB_identity.append(hf5.get_node(node_path,'pairwise_NB_identity')[:])
			p_discriminability.append(hf5.get_node(node_path,'p_discriminability')[:])
			#id_pal_regress.append(hf5.get_node(node_path,'id_pal_regress')[:])

			# Reading single values from the hdf5 file seems hard, needs the read() method to be called
			if os.path.join(node_path, 'pre_stim') in hf5:
				pre_stim.append(hf5.get_node(node_path,'pre_stim').read())
			else:
				# Get the pre-stimulus time from the user
				pre_stim_input = easygui.multenterbox(msg = 'Enter the pre-stimulus time for the spike trains',\
									fields = ['Pre stim (ms)'], values = [2000])
				pre_stim_input = int(pre_stim_input[0])
				pre_stim.append(pre_stim_input)
			params.append(hf5.get_node(node_path,'params')[:])

			#Pull in data from /ancillary_analysis mode that don't pertain to cell number
			unique_lasers.append(hf5.get_node(node_path,'laser_combination_d_l')[:])
			lda_identity.append(hf5.get_node(node_path,'lda_identity')[:])
			taste_cosine_similarity.append(hf5.get_node(node_path,'taste_cosine_similarity')[:])
			taste_euclidean_distance.append(hf5.get_node(node_path,'taste_euclidean_distance')[:])
			pairwise_NB_identity.append(hf5.get_node(node_path,'pairwise_NB_identity')[:])

			if held_check == "No":
				# Pull the data from the /ancillary_analysis node
				r_pearson.append(hf5.get_node(node_path,'r_pearson')[:])
				r_spearman.append(hf5.get_node(node_path,'r_spearman')[:])
				r_isotonic.append(hf5.get_node(node_path,'r_isotonic')[:])
				p_pearson.append(hf5.get_node(node_path,'p_pearson')[:])
				p_spearman.append(hf5.get_node(node_path,'p_spearman')[:])
				p_identity.append(hf5.get_node(node_path,'p_identity')[:])
				lda_palatability.append(hf5.get_node(node_path,'lda_palatability')[:])
				p_discriminability.append(hf5.get_node(node_path,'p_discriminability')[:])
				#id_pal_regress.append(hf5.get_node(node_path,'id_pal_regress')[:])

			if held_check == "Yes":
				#indicate working held group
				if day ==0:
					held_set = sorted_day1[day]
				if day ==1:
					held_set = sorted_day2[day]

				if len(held_set) > 0:
					# Pull the data from the /ancillary_analysis node
					r_pearson.append(hf5.get_node(node_path,'r_pearson')[:,:,held_set])
					r_spearman.append(hf5.get_node(node_path,'r_spearman')[:,:,held_set])
					r_isotonic.append(hf5.get_node(node_path,'r_isotonic')[:,:,held_set])
					p_pearson.append(hf5.get_node(node_path,'p_pearson')[:,:,held_set])
					p_spearman.append(hf5.get_node(node_path,'p_spearman')[:,:,held_set])
					p_identity.append(hf5.get_node(node_path,'p_identity')[:,:,held_set])
					lda_palatability.append(hf5.get_node(node_path,'lda_palatability')[:])
					p_discriminability.append(hf5.get_node(node_path,'p_discriminability')[:,:,:,:,held_set])
					#id_pal_regress.append(hf5.get_node(node_path,'id_pal_regress')[:])

			#update counter of the number of units in the analysis
			if held_check == "No":
				num_units += hf5.get_node(node_path,'palatability').shape[1]
			if held_check == "Yes" and len(held_set) > 0:
				num_units += hf5.get_node(node_path,'palatability')[:,held_set,:].shape[1]

			#Flag node if animal had dig_ins removed during palatabiltiy set-up
			if os.path.join(node_path, 'removed_dig_in') in hf5:
				rm_digin.append([1])

			if os.path.join(node_path, 'removed_dig_in') not in hf5:
				rm_digin.append([0])

	# Check if the number of laser activation/inactivation windows is same across files, raise an error and quit if it isn't
	if all(unique_lasers[i].shape == unique_lasers[0].shape for i in range(len(unique_lasers))):
		pass
	else:
		print("Number of inactivation/activation windows doesn't seem to be the same across days. Please check and try again")
		sys.exit()

	# Now first set the ordering of laser trials straight across data files
	laser_order = []
	for i in range(len(unique_lasers)):
		# The first file defines the order
		if i == 0:
			laser_order.append(np.arange(unique_lasers[i].shape[0]))
		# And everyone else follows
		else:
			this_order = []
			for j in range(unique_lasers[i].shape[0]):
				for k in range(unique_lasers[i].shape[0]):
					if np.array_equal(unique_lasers[0][j, :], unique_lasers[i][k, :]):
						this_order.append(k)
			laser_order.append(np.array(this_order))

	# Now join up all the data into big numpy arrays, maintaining the laser order defined in laser_order
	# If there's only one data file, set the final arrays to the only array read in
	if len(laser_order) == 1:
		r_pearson = r_pearson[0]
		r_spearman = r_spearman[0]
		r_isotonic = r_isotonic[0]
		p_pearson = p_pearson[0]
		p_spearman = p_spearman[0]
		p_identity = p_identity[0]
		lda_palatability = lda_palatability[0]
		lda_identity = lda_identity[0]
		taste_cosine_similarity = taste_cosine_similarity[0]
		taste_euclidean_distance = taste_euclidean_distance[0]
		pairwise_NB_identity = pairwise_NB_identity[0]
		p_discriminability = p_discriminability[0]
	else:
		r_pearson = np.concatenate(tuple(r_pearson[i][laser_order[i], :, :] for i in range(len(r_pearson))), axis = 2)
		r_spearman = np.concatenate(tuple(r_spearman[i][laser_order[i], :, :] for i in range(len(r_spearman))), axis = 2)
		r_isotonic = np.concatenate(tuple(r_isotonic[i][laser_order[i], :, :] for i in range(len(r_isotonic))), axis = 2)
		p_pearson = np.concatenate(tuple(p_pearson[i][laser_order[i], :, :] for i in range(len(p_pearson))), axis = 2)
		p_spearman = np.concatenate(tuple(p_spearman[i][laser_order[i], :, :] for i in range(len(p_spearman))), axis = 2)
		p_identity = np.concatenate(tuple(p_identity[i][laser_order[i], :, :] for i in range(len(p_identity))), axis = 2)
		lda_palatability = np.stack(tuple(lda_palatability[i][laser_order[i], :] for i in range(len(lda_palatability))), axis = -1)
		lda_identity = np.stack(tuple(lda_identity[i][laser_order[i], :] for i in range(len(lda_identity))), axis = -1)

		# Now average the lda and distance results along the last axis (i.e across sessions)
		lda_palatability = np.mean(lda_palatability, axis = 2)
		lda_identity = np.mean(lda_identity, axis = 2)

		#Check if a file was had dig_in removed, if it did, remove from analyses below
		if all([element[0] == 0 for element in rm_digin]) is False:
			#Trim files based on those removed during processing
			taste_euclidean_distance.pop(int(np.where(np.array(rm_digin)==1)[0]))
			taste_cosine_similarity.pop(int(np.where(np.array(rm_digin)==1)[0]))
			pairwise_NB_identity.pop(int(np.where(np.array(rm_digin)==1)[0]))
			p_discriminability.pop(int(np.where(np.array(rm_digin)==1)[0]))

		#Take the mean
		taste_cosine_similarity = np.mean(taste_cosine_similarity, axis = -1)
		taste_euclidean_distance = np.mean(taste_euclidean_distance, axis = -1)
		pairwise_NB_identity = np.mean(pairwise_NB_identity, axis = -1)

# =============================================================================
# #CREATE DIRECTORY FOR OUTPUTS
# =============================================================================

	# remake the directory if it exists
	os.chdir(dir_folder)
	try:
		os.system('rm -r '+'./palID_plots_%s_%s' %(condition_check,date.today().strftime("%d_%m_%Y")))
	except:
	        pass
	os.mkdir('./palID_plots_%s_%s' %(condition_check,date.today().strftime("%d_%m_%Y")))
	os.chdir('./palID_plots_%s_%s' %(condition_check,date.today().strftime("%d_%m_%Y")))

# =============================================================================
# #SAVING TIME
# =============================================================================
	#Store animals used in analyses
	with open('Animals_in_Analysis.txt', 'w') as filehandle:
		   filehandle.writelines("%s\n" % animal for animal in analysis_nodes)

	# Save all these arrays in the output directory
	np.save('r_pearson.npy', r_pearson)
	np.save('r_spearman.npy', r_spearman)
	np.save('r_isotonic.npy', r_isotonic)
	np.save('p_pearson.npy', p_pearson)
	np.save('p_spearman.npy', p_spearman)
	np.save('p_identity.npy', p_identity)
	np.save('lda_palatability.npy', lda_palatability)
	np.save('lda_identity.npy', lda_identity)
	np.save('unique_lasers.npy', unique_lasers)
	np.save('taste_cosine_similarity.npy', taste_cosine_similarity)
	np.save('taste_euclidean_distance.npy', taste_euclidean_distance)

# =============================================================================
# #PLOTTING TIME
# =============================================================================

	# Ask the user for the significance level and # of significant windows to use for plotting the identity/palatability p-values
	p_values = easygui.multenterbox(msg = 'Enter the significance criteria for palatability/identity calculation',\
				 fields = ['Significance level (p value)', 'Number of consecutive significant windows'],\
				 values = [0.05,3])
	p_values[0] = float(p_values[0])
	p_values[1] = int(p_values[1])

	# Get the x array for all the plotting
	x = np.arange(0, r_pearson.shape[1]*params[0][1], params[0][1]) - pre_stim[0]

	# Ask the user for the time limits to be plotted
	time_limits = easygui.multenterbox(msg = 'Enter the time limits for plotting the results',\
				fields = ['Start time (ms)', 'End time (ms)'],\
				values = [-500,2000])
	plot_indices = np.where((x>=time_limits[0])*(x<=time_limits[1]))[0]

	# Ask the user for the standard deviation to be used in smoothing the palatability correlation curves using a Gaussian
	sigma = easygui.multenterbox(msg = "Enter the standard deviation to use while Gaussian smoothing the palatability correlation plots (5 is good for 250ms bins)",\
			fields = ['sigma'],values=[5])
	sigma = int(sigma[0])

	# Plot the r_squared values together first (for the different laser conditions)
	fig = plt.figure(figsize=(12,8))
	for i in range(r_pearson.shape[0]):
		plt.plot(x[plot_indices], np.mean(r_pearson[i, plot_indices, :]**2, axis = 1), linewidth = 3.0, label = 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
	plt.title('Pearson $r^2$ with palatability ranks' + '\n' + 'Units:%i, Window (ms):%i, Step (ms):%i' % (num_units, params[0][0], params[0][1]))
	plt.xlabel('Time from stimulus (ms)')
	plt.ylabel('Average Pearson $r^2$')
	plt.legend(loc = 'upper left', fontsize = 15)
	fig.savefig('Pearson correlation-palatability.png', bbox_inches = 'tight')
	plt.close('all')

	fig = plt.figure(figsize=(12,8))
	for i in range(r_spearman.shape[0]):
		plt.plot(x[plot_indices], np.mean(r_spearman[i, plot_indices, :]**2, axis = 1), linewidth = 3.0, label = 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
	plt.title('Spearman $rho^2$ with palatability ranks' + '\n' + 'Units:%i, Window (ms):%i, Step (ms):%i' % (num_units, params[0][0], params[0][1]))
	plt.xlabel('Time from stimulus (ms)')
	plt.ylabel('Average Spearman $rho^2$')
	plt.legend(loc = 'upper left', fontsize = 15)
	fig.savefig('Spearman correlation-palatability.png', bbox_inches = 'tight')
	plt.close('all')

	fig = plt.figure(figsize=(12,8))
	for i in range(r_isotonic.shape[0]):
		plt.plot(x[plot_indices], np.median(r_isotonic[i, plot_indices, :], axis = 1), linewidth = 3.0, label = 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
	plt.title('Isotonic $R^2$ with palatability ranks' + '\n' + 'Units:%i, Window (ms):%i, Step (ms):%i' % (num_units, params[0][0], params[0][1]))
	plt.xlabel('Time from stimulus (ms)')
	plt.ylabel('Median Isotonic $R^2$')
	plt.legend(loc = 'upper left', fontsize = 15)
	fig.savefig('Isotonic correlation-palatability.png', bbox_inches = 'tight')
	plt.close('all')

	# Plot a Gaussian-smoothed version of the r_squared values as well
	fig = plt.figure(figsize=(12,8))
	for i in range(r_pearson.shape[0]):
		plt.plot(x[plot_indices], gaussian_filter1d(np.mean(r_pearson[i, plot_indices, :]**2, axis = 1), sigma = sigma), linewidth = 3.0, label = 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
	plt.title('Pearson $r^2$ with palatability ranks, smoothing std:%1.1f' % sigma + '\n' + 'Units:%i, Window (ms):%i, Step (ms):%i' % (num_units, params[0][0], params[0][1]))
	plt.xlabel('Time from stimulus (ms)')
	plt.ylabel('Average Pearson $r^2$')
	plt.legend(loc = 'upper left', fontsize = 15)
	plt.tight_layout()
	fig.savefig('Pearson correlation-palatability-smoothed.png', bbox_inches = 'tight')
	plt.close('all')

	fig = plt.figure(figsize=(12,8))
	for i in range(r_spearman.shape[0]):
		plt.plot(x[plot_indices], gaussian_filter1d(np.mean(r_spearman[i, plot_indices, :]**2, axis = 1), sigma = sigma), linewidth = 3.0, label = 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
	plt.title('Spearman $rho^2$ with palatability ranks, smoothing std:%1.1f' % sigma + '\n' + 'Units:%i, Window (ms):%i, Step (ms):%i' % (num_units, params[0][0], params[0][1]))
	plt.xlabel('Time from stimulus (ms)')
	plt.ylabel('Average Spearman $rho^2$')
	plt.legend(loc = 'upper left', fontsize = 15)
	plt.tight_layout()
	fig.savefig('Spearman correlation-palatability-smoothed.png', bbox_inches = 'tight')
	plt.close('all')

	# Now plot the r_squared values separately for the different laser conditions
	for i in range(r_pearson.shape[0]):
		fig = plt.figure(figsize=(12,8))
		plt.errorbar(x[plot_indices], np.mean(r_pearson[i, plot_indices, :]**2, axis = 1), yerr = np.std(r_pearson[i, plot_indices, :]**2, axis = 1)/np.sqrt(r_pearson.shape[2]), linewidth = 3.0, elinewidth = 0.8, label = 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
		plt.title('Pearson $r^2$ with palatability ranks, laser condition %i' % (i+1) + '\n' + 'Units:%i, Window (ms):%i, Step (ms):%i' % (num_units, params[0][0], params[0][1]))
		plt.xlabel('Time from stimulus (ms)')
		plt.ylabel('Average Pearson $r^2$')
		plt.legend(loc = 'upper left', fontsize = 15)
		plt.tight_layout()
		fig.savefig('Pearson correlation-palatability,laser_condition%i.png' % (i+1), bbox_inches = 'tight')
		plt.close('all')

	for i in range(r_spearman.shape[0]):
		fig = plt.figure(figsize=(12,8))
		plt.errorbar(x[plot_indices], np.mean(r_spearman[i, plot_indices, :]**2, axis = 1), yerr = np.std(r_spearman[i, plot_indices, :]**2, axis = 1)/np.sqrt(r_spearman.shape[2]), linewidth = 3.0, elinewidth = 0.8, label = 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
		plt.title('Spearman $rho^2$ with palatability ranks, laser condition %i' % (i+1) + '\n' + 'Units:%i, Window (ms):%i, Step (ms):%i' % (num_units, params[0][0], params[0][1]))
		plt.xlabel('Time from stimulus (ms)')
		plt.ylabel('Average Spearman $rho^2$')
		plt.legend(loc = 'upper left', fontsize = 15)
		plt.tight_layout()
		fig.savefig('Spearman correlation-palatability,laser_condition%i.png' % (i+1), bbox_inches = 'tight')
		plt.close('all')

	# Now plot the p values together using the significance criterion specified by the user
	# Make a final p array - this will store 1s if x consecutive time bins have significant p values (these parameters are specified by the user)
	p_pearson_final = np.zeros(p_pearson.shape)
	p_spearman_final = np.zeros(p_spearman.shape)
	p_identity_final = np.zeros(p_identity.shape)
	for i in range(p_pearson_final.shape[0]):
		for j in range(p_pearson_final.shape[1]):
			for k in range(p_pearson_final.shape[2]):
				if (j < p_pearson_final.shape[1] - p_values[1]):
					if all(p_pearson[i, j:j + p_values[1], k] <= p_values[0]):
						p_pearson_final[i, j, k] = 1
					if all(p_spearman[i, j:j + p_values[1], k] <= p_values[0]):
						p_spearman_final[i, j, k] = 1
					if all(p_identity[i, j:j + p_values[1], k] <= p_values[0]):
						p_identity_final[i, j, k] = 1

	# Now first plot the p values together for the different laser conditions
	fig = plt.figure(figsize=(12,8))
	for i in range(p_pearson_final.shape[0]):
		plt.plot(x[plot_indices], np.mean(p_pearson_final[i, plot_indices, :], axis = 1), linewidth = 3.0, label = 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
	plt.title('Units:%i, Window (ms):%i, Step (ms):%i' % (num_units, params[0][0], params[0][1]) + '\n' + 'threshold:%.02f, consecutive windows:%i' % (p_values[0], p_values[1]))
	plt.xlabel('Time from stimulus (ms)')
	plt.ylabel('Fraction of significant neurons')
	plt.legend(loc = 'upper left', fontsize = 15)
	plt.tight_layout()
	fig.savefig('Pearson correlation p values-palatability.png', bbox_inches = 'tight')
	plt.close('all')

	fig = plt.figure(figsize=(12,8))
	for i in range(p_spearman_final.shape[0]):
		plt.plot(x[plot_indices], np.mean(p_spearman_final[i, plot_indices, :], axis = 1), linewidth = 3.0, label = 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
	plt.title('Units:%i, Window (ms):%i, Step (ms):%i' % (num_units, params[0][0], params[0][1]) + '\n' + 'threshold:%.02f, consecutive windows:%i' % (p_values[0], p_values[1]))
	plt.xlabel('Time from stimulus (ms)')
	plt.ylabel('Fraction of significant neurons')
	plt.legend(loc = 'upper left', fontsize = 15)
	plt.tight_layout()
	fig.savefig('Spearman correlation p values-palatability.png', bbox_inches = 'tight')
	plt.close('all')

	fig = plt.figure(figsize=(12,8))
	for i in range(p_identity_final.shape[0]):
		plt.plot(x[plot_indices], np.mean(p_identity_final[i, plot_indices, :], axis = 1), linewidth = 3.0, label = 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
	plt.title('Units:%i, Window (ms):%i, Step (ms):%i' % (num_units, params[0][0], params[0][1]) + '\n' + 'threshold:%.02f, consecutive windows:%i' % (p_values[0], p_values[1]))
	plt.xlabel('Time from stimulus (ms)')
	plt.ylabel('Fraction of significant neurons')
	plt.legend(loc = 'upper left', fontsize = 15)
	plt.tight_layout()
	fig.savefig('ANOVA p values-identity.png', bbox_inches = 'tight')
	plt.close('all')

	# Now plot them separately for every laser condition
	for i in range(p_pearson_final.shape[0]):
		fig = plt.figure(figsize=(12,8))
		plt.plot(x[plot_indices], np.mean(p_pearson_final[i, plot_indices, :], axis = 1), linewidth = 3.0, label = 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
		plt.title('Units:%i, Window (ms):%i, Step (ms):%i' % (num_units, params[0][0], params[0][1]) + '\n' + 'threshold:%.02f, consecutive windows:%i' % (p_values[0], p_values[1]))
		plt.xlabel('Time from stimulus (ms)')
		plt.ylabel('Fraction of significant neurons')
		plt.legend(loc = 'upper left', fontsize = 15)
		plt.tight_layout()
		fig.savefig('Pearson correlation p values-palatability,laser condition%i.png' % (i+1), bbox_inches = 'tight')
		plt.close('all')

	for i in range(p_spearman_final.shape[0]):
		fig = plt.figure(figsize=(12,8))
		plt.plot(x[plot_indices], np.mean(p_spearman_final[i, plot_indices, :], axis = 1), linewidth = 3.0, label = 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
		plt.title('Units:%i, Window (ms):%i, Step (ms):%i' % (num_units, params[0][0], params[0][1]) + '\n' + 'threshold:%.02f, consecutive windows:%i' % (p_values[0], p_values[1]))
		plt.xlabel('Time from stimulus (ms)')
		plt.ylabel('Fraction of significant neurons')
		plt.legend(loc = 'upper left', fontsize = 15)
		plt.tight_layout()
		fig.savefig('Spearman correlation p values-palatability,laser condition%i.png' % (i+1), bbox_inches = 'tight')
		plt.close('all')

	for i in range(p_identity_final.shape[0]):
		fig = plt.figure(figsize=(12,8))
		plt.plot(x[plot_indices], np.mean(p_identity_final[i, plot_indices, :], axis = 1), linewidth = 3.0, label = 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
		plt.title('Units:%i, Window (ms):%i, Step (ms):%i' % (num_units, params[0][0], params[0][1]) + '\n' + 'threshold:%.02f, consecutive windows:%i' % (p_values[0], p_values[1]))
		plt.xlabel('Time from stimulus (ms)')
		plt.ylabel('Fraction of significant neurons')
		plt.legend(loc = 'upper left', fontsize = 15)
		plt.tight_layout()
		fig.savefig('ANOVA p values-identity,laser condition%i.png' % (i+1), bbox_inches = 'tight')
		plt.close('all')

	# Now plot the LDA results for palatability and identity together for the different laser conditions
	fig = plt.figure(figsize=(12,8))
	for i in range(lda_palatability.shape[0]):
		plt.plot(x[plot_indices], lda_palatability[i, plot_indices], linewidth = 3.0, label = 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
	plt.title('Units:%i, Window (ms):%i, Step (ms):%i, palatability LDA' % (num_units, params[0][0], params[0][1]))
	plt.xlabel('Time from stimulus (ms)')
	plt.ylabel('Fraction correct trials')
	plt.legend(loc = 'upper left', fontsize = 15)
	plt.tight_layout()
	fig.savefig('Palatability_LDA.png', bbox_inches = 'tight')
	plt.close('all')

	fig = plt.figure(figsize=(12,8))
	for i in range(lda_identity.shape[0]):
		plt.plot(x[plot_indices], lda_identity[i, plot_indices], linewidth = 3.0, label = 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
	plt.title('Units:%i, Window (ms):%i, Step (ms):%i, identity LDA' % (num_units, params[0][0], params[0][1]))
	plt.xlabel('Time from stimulus (ms)')
	plt.ylabel('Fraction correct trials')
	plt.legend(loc = 'upper left', fontsize = 15)
	plt.tight_layout()
	fig.savefig('Identity_LDA.png', bbox_inches = 'tight')
	plt.close('all')

	#Save and close
	hf5.flush()
	hf5.close()

if condition_check == 'Separate':
	#Set up conditional variable names
	condition_params = easygui.multenterbox(
	            msg = 'Input Condition Names:',
	            fields = ['Condition 1:',
	                    'Condition 2:'],
	            values = ['LiCl','Saline'])
# =============================================================================
# #CREATE DIRECTORY FOR OUTPUTS
# =============================================================================
	#adjust label for directory
	if unit_check == 'Descriminitive_Responsive':
		# remake the directory if it exists
		os.chdir(dir_folder)
		try:
			os.system('rm -r '+'./%s_palID_plots_%s_%s' %(drtype_check,condition_check,date.today().strftime("%d_%m_%Y")))
		except:
		        pass
		os.mkdir('./%s_palID_plots_%s_%s' %(drtype_check,condition_check,date.today().strftime("%d_%m_%Y")))
		os.chdir('./%s_palID_plots_%s_%s' %(drtype_check,condition_check,date.today().strftime("%d_%m_%Y")))

	if unit_check != 'Descriminitive_Responsive':
		# remake the directory if it exists
		os.chdir(dir_folder)
		try:
			os.system('rm -r '+'./%s_palID_plots_%s_%s' %(unit_check,condition_check,date.today().strftime("%d_%m_%Y")))
		except:
		        pass
		os.mkdir('./%s_palID_plots_%s_%s' %(unit_check,condition_check,date.today().strftime("%d_%m_%Y")))
		os.chdir('./%s_palID_plots_%s_%s' %(unit_check,condition_check,date.today().strftime("%d_%m_%Y")))


	#Flip through each animal and ask user to choose which file data belongs to analysis group
	group_1 = []; group_2 = []
	for animal in range(len(analysis_nodes)):

		group_analyze = easygui.multchoicebox(
		        msg = 'Click the file date for %s that belongs in the grouped analysis (assumes files not chosen to be in other group)' %(analysis_nodes[animal]),
		        choices = ([x for x in cond_detail[animal]]))

		group_1.append(group_analyze)
		group_2.append(list(set([x for x in cond_detail[animal]]) - set(group_analyze)))

	#Create list to store each groups data
	outputs = [[],[]]

	#Flip through both groups and perform analyses for plotting
	g_count = 0
	for cond in range(len(cond_check[0])):

		#Flag node for dig_in removals
		rm_digin = []

		#Initiate empty arrays for merged data storage
		unique_lasers = []
		r_pearson = []
		r_spearman = []
		r_isotonic = []
		p_pearson = []
		p_spearman = []
		p_identity = []
		lda_palatability = []
		lda_identity = []
		taste_cosine_similarity = []
		taste_euclidean_distance = []
		pairwise_NB_identity = []
		p_discriminability = []
		pre_stim = []
		params = []
		id_pal_regress = []
		num_units = 0

		for node in range(len(analysis_nodes)):
			node_groups = [node._v_name for node in hf5.list_nodes('/%s' %(analysis_nodes[node]))]

			if cond ==0:
				if node_groups[cond] == group_1[node][0]:

					#set path for ease of reading
					node_path = os.path.join('/',analysis_nodes[node],node_groups[cond])


				elif node_groups[cond] != group_1[node][0]:

					double_check = easygui.multchoicebox(
				        msg = 'Something is wrong!! Click the file date for %s that belongs in the grouped analysis for Group %s' %(analysis_nodes[animal],cond),
				        choices = ([x for x in cond_detail[node]]))

					#set path for ease of reading
					node_path = os.path.join('/',analysis_nodes[node],double_check[0])

			if cond ==1:
				if node_groups[cond] == group_2[node][0]:

					#set path for ease of reading
					node_path = os.path.join('/',analysis_nodes[node],node_groups[cond])

				elif node_groups[cond] != group_2[node][0]:

					double_check = easygui.multchoicebox(
				        msg = 'Something is wrong!! Click the file date for %s that belongs in the grouped analysis for Group %s' %(analysis_nodes[animal],cond),
				        choices = ([x for x in cond_detail[node]]))

					#set path for ease of reading
					node_path = os.path.join('/',analysis_nodes[node],double_check[0])

			#Pull in data from /ancillary_analysis mode that don't pertain to cell number
			unique_lasers.append(hf5.get_node(node_path,'laser_combination_d_l')[:])
			lda_identity.append(hf5.get_node(node_path,'lda_identity')[:])
			taste_cosine_similarity.append(hf5.get_node(node_path,'taste_cosine_similarity')[:])
			taste_euclidean_distance.append(hf5.get_node(node_path,'taste_euclidean_distance')[:])
			pairwise_NB_identity.append(hf5.get_node(node_path,'pairwise_NB_identity')[:])

			if unit_check == "All units":
				# Pull the data from the /ancillary_analysis node
				r_pearson.append(hf5.get_node(node_path,'r_pearson')[:])
				r_spearman.append(hf5.get_node(node_path,'r_spearman')[:])
				r_isotonic.append(hf5.get_node(node_path,'r_isotonic')[:])
				p_pearson.append(hf5.get_node(node_path,'p_pearson')[:])
				p_spearman.append(hf5.get_node(node_path,'p_spearman')[:])
				p_identity.append(hf5.get_node(node_path,'p_identity')[:])
				lda_palatability.append(hf5.get_node(node_path,'lda_palatability')[:])
				p_discriminability.append(hf5.get_node(node_path,'p_discriminability')[:])
				#id_pal_regress.append(hf5.get_node(node_path,'id_pal_regress')[:])

			if held_check == "Yes":
				#indicate working held group
				if cond ==0:
					held_set = sorted_day1[node]
				if cond ==1:
					held_set = sorted_day2[node]
			if discrimresp_check == "Yes":
				if drtype_check == "Descriminitive":
					held_set = hf5.get_node(node_path,'taste_discriminating_neurons')[:]
				if drtype_check == "Responsive":
					held_set = hf5.get_node(node_path,'taste_responsive_neurons')[:]

			#extract only specific unit data
			if unit_check != "All units":
				if len(held_set) > 0:
					# Pull the data from the /ancillary_analysis node
					r_pearson.append(hf5.get_node(node_path,'r_pearson')[:,:,held_set])
					r_spearman.append(hf5.get_node(node_path,'r_spearman')[:,:,held_set])
					r_isotonic.append(hf5.get_node(node_path,'r_isotonic')[:,:,held_set])
					p_pearson.append(hf5.get_node(node_path,'p_pearson')[:,:,held_set])
					p_spearman.append(hf5.get_node(node_path,'p_spearman')[:,:,held_set])
					p_identity.append(hf5.get_node(node_path,'p_identity')[:,:,held_set])
					lda_palatability.append(hf5.get_node(node_path,'lda_palatability')[:])
					p_discriminability.append(hf5.get_node(node_path,'p_discriminability')[:,:,:,:,held_set])
					#id_pal_regress.append(hf5.get_node(node_path,'id_pal_regress')[:])

			# Reading single values from the hdf5 file seems hard, needs the read() method to be called
			if os.path.join(node_path, 'pre_stim') in hf5:
				pre_stim.append(hf5.get_node(node_path,'pre_stim').read())
			else:
				# Get the pre-stimulus time from the user
				pre_stim_input = easygui.multenterbox(msg = 'Enter the pre-stimulus time for the spike trains',\
									fields = ['Pre stim (ms)'], values = [2000])
				pre_stim_input = int(pre_stim_input[0])
				pre_stim.append(pre_stim_input)
			params.append(hf5.get_node(node_path,'params')[:])

			#update counter of the number of units in the analysis
			if unit_check == "All units":
				num_units += hf5.get_node(node_path,'palatability').shape[1]
			if unit_check != "All units" and len(held_set) > 0:
				num_units += hf5.get_node(node_path,'palatability')[:,held_set,:].shape[1]

			#Flag node if animal had dig_ins removed during palatabiltiy set-up
			if os.path.join(node_path, 'removed_dig_in') in hf5:
				rm_digin.append([1])

			if os.path.join(node_path, 'removed_dig_in') not in hf5:
				rm_digin.append([0])

		# Check if the number of laser activation/inactivation windows is same across files, raise an error and quit if it isn't
		if all(unique_lasers[i].shape == unique_lasers[0].shape for i in range(len(unique_lasers))):
			pass
		else:
			print("Number of inactivation/activation windows doesn't seem to be the same across days. Please check and try again")
			sys.exit()

		# Now first set the ordering of laser trials straight across data files
		laser_order = []
		for i in range(len(unique_lasers)):
			# The first file defines the order
			if i == 0:
				laser_order.append(np.arange(unique_lasers[i].shape[0]))
			# And everyone else follows
			else:
				this_order = []
				for j in range(unique_lasers[i].shape[0]):
					for k in range(unique_lasers[i].shape[0]):
						if np.array_equal(unique_lasers[0][j, :], unique_lasers[i][k, :]):
							this_order.append(k)
				laser_order.append(np.array(this_order))

		# Now join up all the data into big numpy arrays, maintaining the laser order defined in laser_order
		# If there's only one data file, set the final arrays to the only array read in
		if len(laser_order) == 1:
			r_pearson = r_pearson[0]
			r_spearman = r_spearman[0]
			r_isotonic = r_isotonic[0]
			p_pearson = p_pearson[0]
			p_spearman = p_spearman[0]
			p_identity = p_identity[0]
			lda_palatability = lda_palatability[0]
			lda_identity = lda_identity[0]
			taste_cosine_similarity = taste_cosine_similarity[0]
			taste_euclidean_distance = taste_euclidean_distance[0]
			pairwise_NB_identity = pairwise_NB_identity[0]
			p_discriminability = p_discriminability[0]
		else:
			r_pearson = np.concatenate(tuple(r_pearson[i][laser_order[i], :, :] for i in range(len(r_pearson))), axis = 2)
			r_spearman = np.concatenate(tuple(r_spearman[i][laser_order[i], :, :] for i in range(len(r_spearman))), axis = 2)
			r_isotonic = np.concatenate(tuple(r_isotonic[i][laser_order[i], :, :] for i in range(len(r_isotonic))), axis = 2)
			p_pearson = np.concatenate(tuple(p_pearson[i][laser_order[i], :, :] for i in range(len(p_pearson))), axis = 2)
			p_spearman = np.concatenate(tuple(p_spearman[i][laser_order[i], :, :] for i in range(len(p_spearman))), axis = 2)
			p_identity = np.concatenate(tuple(p_identity[i][laser_order[i], :, :] for i in range(len(p_identity))), axis = 2)
			lda_palatability = np.stack(tuple(lda_palatability[i][laser_order[i], :] for i in range(len(lda_palatability))), axis = -1)
			lda_identity = np.stack(tuple(lda_identity[i][laser_order[i], :] for i in range(len(lda_identity))), axis = -1)

			# Now average the lda and distance results along the last axis (i.e across sessions)
			lda_palatability = np.mean(lda_palatability, axis = 2)
			lda_identity = np.mean(lda_identity, axis = 2)

			#Check if a file was had dig_in removed, if it did, remove from analyses below
			if all([element[0] == 0 for element in rm_digin]) is False:
				#Trim files based on those removed during processing
				taste_euclidean_distance.pop(int(np.where(np.array(rm_digin)==1)[0]))
				taste_cosine_similarity.pop(int(np.where(np.array(rm_digin)==1)[0]))
				pairwise_NB_identity.pop(int(np.where(np.array(rm_digin)==1)[0]))
				p_discriminability.pop(int(np.where(np.array(rm_digin)==1)[0]))

			#Take the mean
			taste_cosine_similarity = np.mean(taste_cosine_similarity, axis = -1)
			taste_euclidean_distance = np.mean(taste_euclidean_distance, axis = -1)
			pairwise_NB_identity = np.mean(pairwise_NB_identity, axis = -1)

		#Append to appropriate list
		outputs[cond].append(unique_lasers)
		outputs[cond].append(r_pearson)
		outputs[cond].append(r_spearman)
		outputs[cond].append(r_isotonic)
		outputs[cond].append(p_pearson)
		outputs[cond].append(p_spearman)
		outputs[cond].append(p_identity)
		outputs[cond].append(lda_palatability)
		outputs[cond].append(lda_identity)
		outputs[cond].append(taste_cosine_similarity)
		outputs[cond].append(taste_euclidean_distance)
		outputs[cond].append(pairwise_NB_identity)
		outputs[cond].append(p_discriminability)
		outputs[cond].append(num_units)

	# =============================================================================
	# #SAVING TIME
	# =============================================================================
		#Store animals used in analyses
		with open('Animals_in_Analysis.txt', 'w') as filehandle:
		   filehandle.writelines("%s\n" % animal for animal in analysis_nodes)

		# Save all these arrays in the output directory
		np.save('r_pearson_%s.npy' %(condition_params[cond]), r_pearson)
		np.save('r_spearman_%s.npy' %(condition_params[cond]), r_spearman)
		np.save('r_isotonic_%s.npy' %(condition_params[cond]), r_isotonic)
		np.save('p_pearson_%s.npy' %(condition_params[cond]), p_pearson)
		np.save('p_spearman_%s.npy' %(condition_params[cond]), p_spearman)
		np.save('p_identity_%s.npy'%(condition_params[cond]), p_identity)
		np.save('lda_palatability_%s.npy' %(condition_params[cond]), lda_palatability)
		np.save('lda_identity_%s.npy' %(condition_params[cond]), lda_identity)
		np.save('unique_lasers_%s.npy' %(condition_params[cond]), unique_lasers)
		np.save('taste_cosine_similarity_%s.npy' %(condition_params[cond]), taste_cosine_similarity)
		np.save('taste_euclidean_distance_%s.npy' %(condition_params[cond]), taste_euclidean_distance)

	#Save and close
	hf5.flush()
	hf5.close()

# =============================================================================
# #PLOTTING TIME
# =============================================================================

	# Ask the user for the significance level and # of significant windows to use for plotting the identity/palatability p-values
	p_values = easygui.multenterbox(msg = 'Enter the significance criteria for palatability/identity calculation',\
				 fields = ['Significance level (p value)', 'Number of consecutive significant windows'],\
				 values = [0.05,3])
	p_values[0] = float(p_values[0])
	p_values[1] = int(p_values[1])

	# Get the x array for all the plotting
	x = np.arange(0, r_pearson.shape[1]*params[0][1], params[0][1]) - pre_stim[0]

	# Ask the user for the time limits to be plotted
	time_limits = easygui.multenterbox(msg = 'Enter the time limits for plotting the results',\
				fields = ['Start time (ms)', 'End time (ms)'],\
				values = [-500,2000])
	for i in range(len(time_limits)):
		time_limits[i] = int(time_limits[i])
	plot_indices = np.where((x>=time_limits[0])*(x<=time_limits[1]))[0]

	# Ask the user for the standard deviation to be used in smoothing the palatability correlation curves using a Gaussian
	sigma = easygui.multenterbox(msg = "Enter the standard deviation to use while Gaussian smoothing the palatability correlation plots (5 is good for 250ms bins)",\
			fields = ['sigma'],values=[5])
	sigma = int(sigma[0])

	#Lovely colors
	colors_line = [sns.color_palette("bright",10)[2],\
			  sns.color_palette("PuOr", 10)[-1]]

	# Plot the r_squared values together first (for the different laser conditions)
	unique_lasers_1 = outputs[0][0]; unique_lasers_2 = outputs[1][0]
	r_pearson_1 = outputs[0][1]; r_pearson_2 = outputs[1][1]

	fig = plt.figure(figsize=(12,8))
	for i in range(r_pearson_1.shape[0]):
		plt.plot(x[plot_indices], np.mean(r_pearson_1[i, plot_indices, :]**2, axis = 1),\
		   linewidth = 3.0, label = '%s - Dur:%ims, Lag:%ims' % (condition_params[0], unique_lasers_1[0][i, 0],\
		   unique_lasers_1[0][i, 1]), color = colors_line[0])
	for i in range(r_pearson_2.shape[0]):
		plt.plot(x[plot_indices], np.mean(r_pearson_2[i, plot_indices, :]**2, axis = 1),\
		   linewidth = 3.0, label = '%s - Dur:%ims, Lag:%ims' % (condition_params[1], unique_lasers_2[0][i, 0],\
		   unique_lasers_2[0][i, 1]), color = colors_line[1])

	plt.title('Pearson $r^2$ with palatability ranks' + '\n' + 'Units:%i & %i, Window (ms):%i, Step (ms):%i'\
		   %(outputs[0][13],outputs[1][13], params[0][0], params[0][1]))
	plt.xlabel('Time from stimulus (ms)')
	plt.ylabel('Average Pearson $r^2$')
	plt.legend(loc = 'upper left', fontsize = 15)
	fig.savefig('Pearson correlation-palatability.png', bbox_inches = 'tight')
	plt.close('all')

	r_spearman_1 = outputs[0][2]; r_spearman_2 = outputs[1][2]

	fig = plt.figure(figsize=(12,8))
	for i in range(r_spearman_1.shape[0]):
		plt.plot(x[plot_indices], np.mean(r_spearman_1[i, plot_indices, :]**2, axis = 1),\
		   linewidth = 3.0, label = '%s - Dur:%ims, Lag:%ims' % (condition_params[0], unique_lasers_1[0][i, 0],\
		   unique_lasers_1[0][i, 1]), color = colors_line[0])
	for i in range(r_spearman_2.shape[0]):
		plt.plot(x[plot_indices], np.mean(r_spearman_2[i, plot_indices, :]**2, axis = 1),\
		   linewidth = 3.0, label = '%s - Dur:%ims, Lag:%ims' % (condition_params[1], unique_lasers_2[0][i, 0],\
		   unique_lasers_2[0][i, 1]), color = colors_line[1])

	plt.title('Spearman $rho^2$ with palatability ranks' + '\n' + 'Units:%i & %i, Window (ms):%i, Step (ms):%i'\
		   %(outputs[0][13],outputs[1][13], params[0][0], params[0][1]))
	plt.xlabel('Time from stimulus (ms)')
	plt.ylabel('Average Spearman $rho^2$')
	plt.legend(loc = 'upper left', fontsize = 15)
	fig.savefig('Spearman correlation-palatability.png', bbox_inches = 'tight')
	plt.close('all')

	i_isotonic_1 = outputs[0][3]; i_isotonic_2 = outputs[1][3]

	fig = plt.figure(figsize=(12,8))
	for i in range(i_isotonic_1.shape[0]):
		plt.plot(x[plot_indices], np.mean(i_isotonic_1[i, plot_indices, :]**2, axis = 1),\
		   linewidth = 3.0, label = '%s - Dur:%ims, Lag:%ims' % (condition_params[0], unique_lasers_1[0][i, 0],\
		   unique_lasers_1[0][i, 1]), color = colors_line[0])
	for i in range(i_isotonic_2.shape[0]):
		plt.plot(x[plot_indices], np.mean(i_isotonic_2[i, plot_indices, :]**2, axis = 1),\
		   linewidth = 3.0, label = '%s - Dur:%ims, Lag:%ims' % (condition_params[1], unique_lasers_2[0][i, 0],\
		   unique_lasers_2[0][i, 1]), color = colors_line[1])

	plt.title('Isotonic $R^2$ with palatability ranks' + '\n' + 'Units:%i & %i, Window (ms):%i, Step (ms):%i'\
		   %(outputs[0][13],outputs[1][13], params[0][0], params[0][1]))
	plt.xlabel('Time from stimulus (ms)')
	plt.ylabel('Median Isotonic $R^2$')
	plt.legend(loc = 'upper left', fontsize = 15)
	fig.savefig('Isotonic correlation-palatability.png', bbox_inches = 'tight')
	plt.close('all')

	# Plot a Gaussian-smoothed version of the r_squared values as well
	fig = plt.figure(figsize=(12,8))
	for i in range(r_pearson_1.shape[0]):
		plt.plot(x[plot_indices], gaussian_filter1d(np.mean(r_pearson_1[i, plot_indices, :]**2, axis = 1),\
		   sigma = sigma), linewidth = 3.0, label = '%s - Dur:%ims, Lag:%ims' % (condition_params[0], unique_lasers_1[0][i, 0],\
		   unique_lasers_1[0][i, 1]), color = colors_line[0])
	for i in range(r_pearson_2.shape[0]):
		plt.plot(x[plot_indices], gaussian_filter1d(np.mean(r_pearson_2[i, plot_indices, :]**2, axis = 1),\
		   sigma = sigma), linewidth = 3.0, label = '%s - Dur:%ims, Lag:%ims' % (condition_params[1], unique_lasers_2[0][i, 0],\
		   unique_lasers_2[0][i, 1]), color = colors_line[1])

	plt.title('Pearson $r^2$ with palatability ranks, smoothing std:%1.1f' % sigma + '\n' + 'Units:%i & %i, Window (ms):%i, Step (ms):%i'\
		   %(outputs[0][13],outputs[1][13], params[0][0], params[0][1]))
	plt.xlabel('Time from stimulus (ms)')
	plt.ylabel('Average Pearson $r^2$')
	plt.legend(loc = 'upper left', fontsize = 15)
	plt.tight_layout()
	fig.savefig('Pearson correlation-palatability-smoothed.png', bbox_inches = 'tight')
	plt.close('all')

	fig = plt.figure(figsize=(12,8))
	for i in range(r_spearman_1.shape[0]):
		plt.plot(x[plot_indices], gaussian_filter1d(np.mean(r_spearman_1[i, plot_indices, :]**2, axis = 1),\
		   sigma = sigma), linewidth = 3.0, label = '%s - Dur:%ims, Lag:%ims' % (condition_params[0], unique_lasers_1[0][i, 0],\
		   unique_lasers_1[0][i, 1]), color = colors_line[0])
	for i in range(r_spearman_2.shape[0]):
		plt.plot(x[plot_indices], gaussian_filter1d(np.mean(r_spearman_2[i, plot_indices, :]**2, axis = 1),\
		   sigma = sigma), linewidth = 3.0, label = '%s - Dur:%ims, Lag:%ims' % (condition_params[1], unique_lasers_2[0][i, 0],\
		   unique_lasers_2[0][i, 1]), color = colors_line[1])

	plt.title('Spearman $rho^2$ with palatability ranks, smoothing std:%1.1f' % sigma + '\n' + 'Units:%i & %i, Window (ms):%i, Step (ms):%i'\
		   %(outputs[0][13],outputs[1][13], params[0][0], params[0][1]))
	plt.xlabel('Time from stimulus (ms)')
	plt.ylabel('Average Spearman $rho^2$')
	plt.legend(loc = 'upper left', fontsize = 15)
	plt.tight_layout()
	fig.savefig('Spearman correlation-palatability-smoothed.png', bbox_inches = 'tight')
	plt.close('all')

	# Now plot the r_squared values separately for the different laser conditions
	# Assumes same number of laser conditions across groups
	for i in range(r_pearson_1.shape[0]):
		fig = plt.figure(figsize=(12,8))
		plt.errorbar(x[plot_indices], np.mean(r_pearson_1[i, plot_indices, :]**2, axis = 1),\
			   yerr = np.std(r_pearson_1[i, plot_indices, :]**2, axis = 1)/np.sqrt(r_pearson_1.shape[2]),\
			   linewidth = 3.0, elinewidth = 0.8, label = '%s - Dur:%ims, Lag:%ims' % (condition_params[0], unique_lasers_1[0][i, 0],\
			   unique_lasers_1[0][i, 1]), color = colors_line[0])
		plt.errorbar(x[plot_indices], np.mean(r_pearson_2[i, plot_indices, :]**2, axis = 1),\
			   yerr = np.std(r_pearson_1[i, plot_indices, :]**2, axis = 1)/np.sqrt(r_pearson_2.shape[2]),\
			   linewidth = 3.0, elinewidth = 0.8, label = '%s - Dur:%ims, Lag:%ims' % (condition_params[1], unique_lasers_2[0][i, 0],\
			   unique_lasers_2[0][i, 1]), color = colors_line[1])

		plt.title('Pearson $r^2$ with palatability ranks, laser condition %i' % (i+1) + '\n' + 'Units:%i & %i, Window (ms):%i, Step (ms):%i'\
		   %(outputs[0][13],outputs[1][13], params[0][0], params[0][1]))
		plt.xlabel('Time from stimulus (ms)')
		plt.ylabel('Average Pearson $r^2$')
		plt.legend(loc = 'upper left', fontsize = 15)
		plt.tight_layout()
		fig.savefig('Pearson correlation-palatability,laser_condition%i.png' % (i+1), bbox_inches = 'tight')
		plt.close('all')

	for i in range(r_spearman_1.shape[0]):
		fig = plt.figure(figsize=(12,8))
		plt.errorbar(x[plot_indices], np.mean(r_spearman_1[i, plot_indices, :]**2, axis = 1),\
			   yerr = np.std(r_spearman_1[i, plot_indices, :]**2, axis = 1)/np.sqrt(r_spearman_1.shape[2]),\
			   linewidth = 3.0, elinewidth = 0.8, label = '%s - Dur:%ims, Lag:%ims' % (condition_params[0], unique_lasers_1[0][i, 0],\
			   unique_lasers_1[0][i, 1]), color = colors_line[0])
		plt.errorbar(x[plot_indices], np.mean(r_spearman_2[i, plot_indices, :]**2, axis = 1),\
			   yerr = np.std(r_spearman_2[i, plot_indices, :]**2, axis = 1)/np.sqrt(r_spearman_2.shape[2]),\
			   linewidth = 3.0, elinewidth = 0.8, label = '%s - Dur:%ims, Lag:%ims' % (condition_params[1], unique_lasers_2[0][i, 0],\
			   unique_lasers_2[0][i, 1]), color = colors_line[1])

		plt.title('Spearman $rho^2$ with palatability ranks, laser condition %i' % (i+1) + '\n' + 'Units:%i & %i, Window (ms):%i, Step (ms):%i'\
		   %(outputs[0][13],outputs[1][13], params[0][0], params[0][1]))
		plt.xlabel('Time from stimulus (ms)')
		plt.ylabel('Average Spearman $rho^2$')
		plt.legend(loc = 'upper left', fontsize = 15)
		plt.tight_layout()
		fig.savefig('Spearman correlation-palatability,laser_condition%i.png' % (i+1), bbox_inches = 'tight')
		plt.close('all')

	# Now plot the p values together using the significance criterion specified by the user
	# Make a final p array - this will store 1s if x consecutive time bins have significant p values (these parameters are specified by the user)
	p_pearson_1 = outputs[0][4]; p_pearson_2 = outputs[1][4]
	p_spearman_1 = outputs[0][5]; p_spearman_2 = outputs[1][5]
	p_identity_1 = outputs[0][6]; p_identity_2 = outputs[1][6]

	p_pearson_final_1 = np.zeros(p_pearson_1.shape); p_pearson_final_2 = np.zeros(p_pearson_2.shape)
	p_spearman_final_1 = np.zeros(p_spearman_1.shape); p_spearman_final_2 = np.zeros(p_spearman_2.shape)
	p_identity_final_1 = np.zeros(p_identity_1.shape); p_identity_final_2 = np.zeros(p_identity_2.shape)
	for i in range(p_pearson_final_1.shape[0]):
		for j in range(p_pearson_final_1.shape[1]):
			for k in range(p_pearson_final_1.shape[2]):
				if (j < p_pearson_final_1.shape[1] - p_values[1]):
					if all(p_pearson_1[i, j:j + p_values[1], k] <= p_values[0]):
						p_pearson_final_1[i, j, k] = 1
					if all(p_spearman_1[i, j:j + p_values[1], k] <= p_values[0]):
						p_spearman_final_1[i, j, k] = 1
					if all(p_identity_1[i, j:j + p_values[1], k] <= p_values[0]):
						p_identity_final_1[i, j, k] = 1

	for i in range(p_pearson_final_2.shape[0]):
		for j in range(p_pearson_final_2.shape[1]):
			for k in range(p_pearson_final_2.shape[2]):
				if (j < p_pearson_final_2.shape[1] - p_values[1]):
					if all(p_pearson_2[i, j:j + p_values[1], k] <= p_values[0]):
						p_pearson_final_2[i, j, k] = 1
					if all(p_spearman_2[i, j:j + p_values[1], k] <= p_values[0]):
						p_spearman_final_2[i, j, k] = 1
					if all(p_identity_2[i, j:j + p_values[1], k] <= p_values[0]):
						p_identity_final_2[i, j, k] = 1

	# Now first plot the p values together for the different laser conditions
	fig = plt.figure(figsize=(12,8))
	for i in range(p_pearson_final_1.shape[0]):
		plt.plot(x[plot_indices], np.mean(p_pearson_final_1[i, plot_indices, :], axis = 1),\
		   linewidth = 3.0, label = '%s - Dur:%ims, Lag:%ims' % (condition_params[0], unique_lasers_1[0][i, 0],\
		   unique_lasers_1[0][i, 1]), color = colors_line[0])
	for i in range(p_pearson_final_2.shape[0]):
		plt.plot(x[plot_indices], np.mean(p_pearson_final_2[i, plot_indices, :], axis = 1),\
		   linewidth = 3.0, label = '%s - Dur:%ims, Lag:%ims' % (condition_params[1], unique_lasers_2[0][i, 0],\
		   unique_lasers_2[0][i, 1]), color = colors_line[1])

	plt.title('Units:%i & %i, Window (ms):%i, Step (ms):%i' %(outputs[0][13],outputs[1][13], params[0][0], params[0][1]) + \
		   '\n' + 'threshold:%.02f, consecutive windows:%i' % (p_values[0], p_values[1]))
	plt.xlabel('Time from stimulus (ms)')
	plt.ylabel('Fraction of significant neurons')
	plt.legend(loc = 'upper left', fontsize = 15)
	plt.tight_layout()
	fig.savefig('Pearson correlation p values-palatability.png', bbox_inches = 'tight')
	plt.close('all')

	fig = plt.figure(figsize=(12,8))
	for i in range(p_spearman_final_1.shape[0]):
		plt.plot(x[plot_indices], np.mean(p_spearman_final_1[i, plot_indices, :], axis = 1),\
		   linewidth = 3.0, label = '%s - Dur:%ims, Lag:%ims' % (condition_params[0], unique_lasers_1[0][i, 0],\
		   unique_lasers_1[0][i, 1]), color = colors_line[0])
	for i in range(p_spearman_final_2.shape[0]):
		plt.plot(x[plot_indices], np.mean(p_spearman_final_2[i, plot_indices, :], axis = 1),\
		   linewidth = 3.0, label = '%s - Dur:%ims, Lag:%ims' % (condition_params[1], unique_lasers_2[0][i, 0],\
		   unique_lasers_2[0][i, 1]), color = colors_line[1])

	plt.title('Units:%i & %i, Window (ms):%i, Step (ms):%i' %(outputs[0][13],outputs[1][13], params[0][0], params[0][1]) + \
		   '\n' + 'threshold:%.02f, consecutive windows:%i' % (p_values[0], p_values[1]))
	plt.xlabel('Time from stimulus (ms)')
	plt.ylabel('Fraction of significant neurons')
	plt.legend(loc = 'upper left', fontsize = 15)
	plt.tight_layout()
	fig.savefig('Spearman correlation p values-palatability.png', bbox_inches = 'tight')
	plt.close('all')

	fig = plt.figure(figsize=(12,8))
	for i in range(p_identity_final_1.shape[0]):
		plt.plot(x[plot_indices], np.mean(p_identity_final_1[i, plot_indices, :], axis = 1),\
		   linewidth = 3.0, label = '%s - Dur:%ims, Lag:%ims' % (condition_params[0], unique_lasers_1[0][i, 0],\
		   unique_lasers_1[0][i, 1]), color = colors_line[0])
	for i in range(p_identity_final_2.shape[0]):
		plt.plot(x[plot_indices], np.mean(p_identity_final_2[i, plot_indices, :], axis = 1),\
		   linewidth = 3.0, label = '%s - Dur:%ims, Lag:%ims' % (condition_params[1], unique_lasers_2[0][i, 0],\
		   unique_lasers_2[0][i, 1]), color = colors_line[1])

	plt.title('Units:%i & %i, Window (ms):%i, Step (ms):%i' %(outputs[0][13],outputs[1][13], params[0][0], params[0][1]) + \
		   '\n' + 'threshold:%.02f, consecutive windows:%i' % (p_values[0], p_values[1]))
	plt.xlabel('Time from stimulus (ms)')
	plt.ylabel('Fraction of significant neurons')
	plt.legend(loc = 'upper left', fontsize = 15)
	plt.tight_layout()
	fig.savefig('ANOVA p values-identity.png', bbox_inches = 'tight')
	plt.close('all')

	# Now plot them separately for every laser condition
	# Assumes same number of laser conditions across groups
	for i in range(p_pearson_final_1.shape[0]):
		fig = plt.figure(figsize=(12,8))
		plt.plot(x[plot_indices], np.mean(p_pearson_final_1[i, plot_indices, :], axis = 1),\
		   linewidth = 3.0, label = '%s - Dur:%ims, Lag:%ims' % (condition_params[0], unique_lasers_1[0][i, 0],\
		   unique_lasers_1[0][i, 1]), color = colors_line[0])
		plt.plot(x[plot_indices], np.mean(p_pearson_final_2[i, plot_indices, :], axis = 1),\
		   linewidth = 3.0, label = '%s - Dur:%ims, Lag:%ims' % (condition_params[1], unique_lasers_2[0][i, 0],\
		   unique_lasers_2[0][i, 1]), color = colors_line[1])

		plt.title('Units:%i & %i, Window (ms):%i, Step (ms):%i' %(outputs[0][13],outputs[1][13], params[0][0], params[0][1]) + \
		   '\n' + 'threshold:%.02f, consecutive windows:%i' % (p_values[0], p_values[1]))
		plt.xlabel('Time from stimulus (ms)')
		plt.ylabel('Fraction of significant neurons')
		plt.legend(loc = 'upper left', fontsize = 15)
		plt.tight_layout()
		fig.savefig('Pearson correlation p values-palatability,laser condition%i.png' % (i+1), bbox_inches = 'tight')
		plt.close('all')

	for i in range(p_spearman_final_1.shape[0]):
		fig = plt.figure(figsize=(12,8))
		plt.plot(x[plot_indices], np.mean(p_spearman_final_1[i, plot_indices, :], axis = 1),\
		   linewidth = 3.0, label = '%s - Dur:%ims, Lag:%ims' % (condition_params[0], unique_lasers_1[0][i, 0],\
		   unique_lasers_1[0][i, 1]), color = colors_line[0])
		plt.plot(x[plot_indices], np.mean(p_spearman_final_2[i, plot_indices, :], axis = 1),\
		   linewidth = 3.0, label = '%s - Dur:%ims, Lag:%ims' % (condition_params[1], unique_lasers_2[0][i, 0],\
		   unique_lasers_2[0][i, 1]), color = colors_line[1])

		plt.title('Units:%i & %i, Window (ms):%i, Step (ms):%i' %(outputs[0][13],outputs[1][13], params[0][0], params[0][1]) + \
		   '\n' + 'threshold:%.02f, consecutive windows:%i' % (p_values[0], p_values[1]))
		plt.xlabel('Time from stimulus (ms)')
		plt.ylabel('Fraction of significant neurons')
		plt.legend(loc = 'upper left', fontsize = 15)
		plt.tight_layout()
		fig.savefig('Spearman correlation p values-palatability,laser condition%i.png' % (i+1), bbox_inches = 'tight')
		plt.close('all')

	for i in range(p_identity_final_1.shape[0]):
		fig = plt.figure(figsize=(12,8))
		plt.plot(x[plot_indices], np.mean(p_identity_final_1[i, plot_indices, :], axis = 1),\
		   linewidth = 3.0, label = '%s - Dur:%ims, Lag:%ims' % (condition_params[0], unique_lasers_1[0][i, 0],\
		   unique_lasers_1[0][i, 1]), color = colors_line[0])
		plt.plot(x[plot_indices], np.mean(p_identity_final_2[i, plot_indices, :], axis = 1),\
		   linewidth = 3.0, label = '%s - Dur:%ims, Lag:%ims' % (condition_params[1], unique_lasers_2[0][i, 0],\
		   unique_lasers_2[0][i, 1]), color = colors_line[1])


		plt.title('Units:%i & %i, Window (ms):%i, Step (ms):%i' %(outputs[0][13],outputs[1][13], params[0][0], params[0][1]) + \
		   '\n' + 'threshold:%.02f, consecutive windows:%i' % (p_values[0], p_values[1]))
		plt.xlabel('Time from stimulus (ms)')
		plt.ylabel('Fraction of significant neurons')
		plt.legend(loc = 'upper left', fontsize = 15)
		plt.tight_layout()
		fig.savefig('ANOVA p values-identity,laser condition%i.png' % (i+1), bbox_inches = 'tight')
		plt.close('all')

	# Now plot the LDA results for palatability and identity together for the different laser conditions
	lda_palatability_1 = outputs[0][7]; lda_palatability_2 = outputs[1][7]
	lda_identity_1 = outputs[0][8]; lda_identity_2 = outputs[1][7]

	fig = plt.figure(figsize=(12,8))
	for i in range(lda_palatability_1.shape[0]):
		plt.plot(x[plot_indices], lda_palatability_1[i, plot_indices],\
		   linewidth = 3.0, label = '%s - Dur:%ims, Lag:%ims' % (condition_params[0], unique_lasers_1[0][i, 0],\
		   unique_lasers_1[0][i, 1]), color = colors_line[0])
		plt.plot(x[plot_indices], lda_palatability_2[i, plot_indices],\
		   linewidth = 3.0, label = '%s - Dur:%ims, Lag:%ims' % (condition_params[1], unique_lasers_2[0][i, 0],\
		   unique_lasers_2[0][i, 1]), color = colors_line[1])

	plt.title('Units:%i & %i, Window (ms):%i, Step (ms):%i, palatability LDA' % (outputs[0][13],outputs[1][13], params[0][0], params[0][1]))
	plt.xlabel('Time from stimulus (ms)')
	plt.ylabel('Fraction correct trials')
	plt.legend(loc = 'upper left', fontsize = 15)
	plt.tight_layout()
	fig.savefig('Palatability_LDA.png', bbox_inches = 'tight')
	plt.close('all')

	fig = plt.figure(figsize=(12,8))
	for i in range(lda_identity_1.shape[0]):
		plt.plot(x[plot_indices], lda_identity_1[i, plot_indices],\
		   linewidth = 3.0, label = '%s - Dur:%ims, Lag:%ims' % (condition_params[0], unique_lasers_1[0][i, 0],\
		   unique_lasers_1[0][i, 1]), color = colors_line[0])
		plt.plot(x[plot_indices], lda_identity_2[i, plot_indices],\
		   linewidth = 3.0, label = '%s - Dur:%ims, Lag:%ims' % (condition_params[1], unique_lasers_2[0][i, 0],\
		   unique_lasers_2[0][i, 1]), color = colors_line[1])

	plt.title('Units:%i & %i, Window (ms):%i, Step (ms):%i, palatability LDA' % (outputs[0][13],outputs[1][13], params[0][0], params[0][1]))
	plt.xlabel('Time from stimulus (ms)')
	plt.ylabel('Fraction correct trials')
	plt.legend(loc = 'upper left', fontsize = 15)
	plt.tight_layout()
	fig.savefig('Identity_LDA.png', bbox_inches = 'tight')
