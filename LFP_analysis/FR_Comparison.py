#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 11:53:49 2019

@author: bradly
"""
#import Libraries
# Built-in Python libraries
import os # functions for interacting w operating system

#import tools for multiprocessing
import pandas as pd

# 3rd-party libraries
import numpy as np # module for low-level scientific computing
import matplotlib.pyplot as plt 
from scipy import stats
from scipy.stats import chisquare
from scipy.stats import chi2_contingency
import re
import math
import easygui
import tables
from pylab import text
import seaborn.apionly as sns
from itertools import product

# =============================================================================
# #Establish files to work on
# =============================================================================
#Get name of directory where you want to save output files to
save_name = easygui.diropenbox(msg = 'Choose directory you want output files' +\
							   'sent to (and/or where ".dir" files are)')

#Ask user if they have ran set-up before
msg   = "Have you performed directory set-up before (ie. do you have '.dir' files' in output folder) ?"
dir_check = easygui.buttonbox(msg,choices = ["Yes","No"])

if dir_check == "No":
    # Ask the user for the hdf5 files that need to be plotted together (fist condition)
    dirs_1 = []
    while True:
    	dir_name = easygui.diropenbox(msg = 'Choose first condition directory with a hdf5 file, hit cancel to stop choosing')
    	try:
    		if len(dir_name) > 0:	
    			dirs_1.append(dir_name)
    	except:
    		break   

    #Dump the directory names into chosen output location for each condition
    #condition 1
    completeName_1 = os.path.join(save_name, 'dirs_cond1.dir') 
    f_1 = open(completeName_1, 'w')
    for item in dirs_1:
        f_1.write("%s\n" % item)
    f_1.close()


if dir_check == "Yes":
    #establish directories to flip through
    #condition 1
    dirs_1_path = os.path.join(save_name, 'dirs_cond1.dir')
    dirs_1_file = open(dirs_1_path,'r')
    dirs_1 = dirs_1_file.read().splitlines()
    dirs_1_file.close()
	
	#condition 1
    dirs_2_path = os.path.join(save_name, 'dirs_cond2.dir')
    dirs_2_file = open(dirs_2_path,'r')
    dirs_2 = dirs_2_file.read().splitlines()
    dirs_2_file.close()
	
# Make directory to store all phaselocking plots. Delete and 
# remake the directory if it exists
os.chdir(save_name)

try:
    os.system('rm -r '+'./Grouped_FR_analyses')
except:
    pass
os.mkdir('./Grouped_FR_analyses')

#Grab details regarding conditions from user
conditions = easygui.multenterbox(msg = 'Input condition names:', fields = ['Condition 1:','Condition 2:'], values = ['LiCl','Saline'])

#Create blank Dataframe for storing all animals
grouped_df=pd.DataFrame()

#Flip through files and extract dataframes and store into large one
for dir_name in dirs_1:	
	#Change to the directory
	os.chdir(dir_name)
	#Locate the hdf5 file
	file_list = os.listdir('./')
	hdf5_name = ''
	for files in file_list:
		if files[-2:] == 'h5':
			hdf5_name = files

	#Open the hdf5 file
	hf5 = tables.open_file(hdf5_name, 'r+')	
	freq_dframe = pd.read_hdf(hdf5_name,'Spike_Phase_Dframe/freq_keys','r+')
	
	dframe = pd.read_hdf(hdf5_name,'Spike_Phase_Dframe/dframe','r+')
		
	#Add column name for Animal and Condition and fill in appropriately
	dframe.insert(0,'Animal',hdf5_name[0:4])
	dframe.insert(1,'Condition',conditions[0])
	
	#Stack dataframes onto eachother
	grouped_df = pd.concat([grouped_df,dframe],sort=False)
	
	#Close the hdf5 file
	hf5.close()

for dir_name in dirs_2:	
	#Change to the directory
	os.chdir(dir_name)
	#Locate the hdf5 file
	file_list = os.listdir('./')
	hdf5_name = ''
	for files in file_list:
		if files[-2:] == 'h5':
			hdf5_name = files

	#Open the hdf5 file
	hf5 = tables.open_file(hdf5_name, 'r+')	
	freq_dframe = pd.read_hdf(hdf5_name,'Spike_Phase_Dframe/freq_keys','r+')
	
	dframe = pd.read_hdf(hdf5_name,'Spike_Phase_Dframe/dframe','r+')
		
	#Add column name for Animal and Condition and fill in appropriately
	dframe.insert(0,'Animal',hdf5_name[0:4])
	dframe.insert(1,'Condition',conditions[1])
	
	#Stack dataframes onto eachother
	grouped_df = pd.concat([grouped_df,dframe],sort=False)
	
	#Close the hdf5 file
	hf5.close()

#Ask user to clean up dataframe if need be
msg   = "Would you like to clean the dataframe up further (e.g. remove low firing units)?"
clean_check = easygui.buttonbox(msg,choices = ["Yes","No"])

if clean_check == "Yes":
	clean_animal = easygui.multchoicebox(msg = 'Which Animals would you like to update (select as many as you want, or Cancel)?',
	choices = ([i for i in sorted(grouped_df.Animal.unique())]))
	
	if len(clean_animal)<0:
		print("I do not think you are understanding how this code works.")
		pass
	else:
		for animal in range(len(clean_animal)):
			clean_cond = easygui.multchoicebox(msg = 'Which condition would you like to update for %s?' %(clean_animal[animal]),
			choices = ([i for i in np.unique(sorted(grouped_df.Condition[grouped_df.Animal == clean_animal[animal]]))]))
			if len(clean_animal)<0:
				print("I do not think you are understanding how this code works.")
				pass
			else:
				#These are the real values to change/update
				for cond in  range(len(clean_cond)):
					remove_units = easygui.multchoicebox(msg = 'Which units would you like to remove for %s under the %s condition?' %(clean_animal[animal],clean_cond[cond]),
					choices = ([i for i in np.unique(sorted(grouped_df.unit[(grouped_df.Animal == clean_animal[animal]) & (grouped_df.Condition == clean_cond[cond])]))]))
					if len(remove_units)<0:
						print("I do not think you are understanding how this code works.")
						pass
					else:
						for unit_clear in range(len(remove_units)):
							grouped_df.drop(grouped_df[(grouped_df.Animal == clean_animal[animal]) & (grouped_df.Condition == clean_cond[cond]) & (grouped_df.unit == int(remove_units[unit_clear]))].index, inplace=True)
else:
	pass

held_dir_name = easygui.diropenbox(msg = 'Choose directory with all "_held_units.txt" files in it.',default = save_name)

#Create empty arrays for storing FRH later
#held_FRs_cond1 = []; held_FRs_cond2 = []; 
held_FRs_cond1_2 = []; held_FRs_cond2_2 = []

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
		if files[-3:] == 'txt':

			with open(files,'r') as splitfile:
				for columns in [line.split() for line in splitfile]:
					day1.append(columns[0]);	day2.append(columns[1])
				all_day1.append(day1); all_day2.append(day2)
				day1 = []; day2 = []    #Clear day array
				
#Remove 'Day' from all lists
for sublist in all_day1:
    del sublist[0]
	
for sublist in all_day2:
    del sublist[0]
	
#Account for out of order units
sorted_day1 = [];sorted_day2=[]
for animal in range(np.size(all_day1)):
	sorted_day1.append(sorted(all_day1[animal], key=int))
	sorted_day2.append(sorted(all_day2[animal], key=int))	
	

#Insert Pre/Post identifier
time_params = easygui.multenterbox(msg = 'Enter the parameters for grouped' +\
        'bars', fields = ['Pre-stimulus time (ms)','Post-stimulus time (ms)'],
        values = ['2000','2000'])	

grouped_df['comp_time'] = ''
grouped_df.loc[(grouped_df['time'] <= int(time_params[0])), 'comp_time'] = int(0)
grouped_df.loc[(grouped_df['time'] > int(time_params[0])) & (grouped_df['time'] <= int(time_params[0]) + int(time_params[1])), 'comp_time'] = int(1)

grouped_df.query('comp_time == 0 or comp_time == 1', inplace=True)

#Perform grouping
FR_df = grouped_df.query('band==0').groupby(['Animal','trial','unit','taste','Condition','comp_time']).count()['time']
FR_df = FR_df.reset_index()

#Create dataframe to store only held data information
held_df = pd.DataFrame()

for animal in range(len(dirs_1)):
	animal_name = sorted(grouped_df.Animal.unique())[animal]
	
	for unit in range(len(all_day1[animal])):
		#Extract appropriate held cell for further analyses
		held_unit_1 = all_day1[animal][unit]
		held_unit_2 = all_day2[animal][unit]

		FR_df_query1 = FR_df.query(
						'Animal == @animal_name and unit ==@held_unit_1 and '\
						'Condition == @conditions[0]')
		FR_df_query1['held_unit'] = unit
		FR_df_query2 = FR_df.query(
								'Animal == @animal_name and unit ==@held_unit_2 and '\
								'Condition == @conditions[1]')
		FR_df_query2['held_unit'] = unit
		
		
		held_df = pd.concat([held_df,FR_df_query1],sort=False)
		held_df = pd.concat([held_df,FR_df_query2],sort=False)

#Update dataframe
held_df = held_df.rename(columns={"time":"spike_count"})
held_df = held_df.replace({'comp_time': {0: 'Pre', 1: 'Post'}})		

#Set formating colors
pal = {conditions[0]:"seagreen", conditions[1]:"gray"}
for animal in range(len(held_df.Animal.unique())):
	animal_name = sorted(grouped_df.Animal.unique())[animal]
			
	g = sns.FacetGrid(held_df.query('Animal == @animal_name'), col = 'held_unit', row = 'taste',
             sharey=True)
	g = g.map(sns.swarmplot, 'comp_time', 'spike_count', 'Condition', order=['Pre','Post'],dodge=True,color = 'black',alpha=0.3)	
	g = (g.map(sns.barplot, 'comp_time', 'spike_count', 'Condition', palette=pal,order=['Pre','Post']).add_legend())
	
	plt.savefig('./Grouped_FR_analyses/' + \
            '/{}_Conditional_FRs.png'.format(animal_name))	
	plt.close()

#Create dataframe to store only held data information
Phase_df = pd.DataFrame()

for animal in range(len(dirs_1)):
	animal_name = sorted(grouped_df.Animal.unique())[animal]
	
	for unit in range(len(all_day1[animal])):
		#Extract appropriate held cell for further analyses
		held_unit_1 = all_day1[animal][unit]
		held_unit_2 = all_day2[animal][unit]

		Phase_df_query1 = grouped_df.query(
						'Animal == @animal_name and unit ==@held_unit_1 and '\
						'Condition == @conditions[0]')
		Phase_df_query1['held_unit'] = unit
		Phase_df_query2 = grouped_df.query(
								'Animal == @animal_name and unit ==@held_unit_2 and '\
								'Condition == @conditions[1]')
		Phase_df_query2['held_unit'] = unit
		
		
		Phase_df = pd.concat([Phase_df,Phase_df_query1],sort=False)
		Phase_df = pd.concat([Phase_df,Phase_df_query2],sort=False)

# =============================================================================
# #Perform grouping
# Phase_df = grouped_df.groupby(['Animal','trial','unit','taste','Condition','comp_time']).count()
# Phase_df = Phase_df.reset_index()
# 
# =============================================================================
for animal in range(len(held_df.Animal.unique())):
	animal_name = sorted(grouped_df.Animal.unique())[animal]
			
	g = sns.FacetGrid(Phase_df.query('Animal == @animal_name and comp_time == 1'), col = 'held_unit', row = 'band',
             sharey=True)
	g = g.map(sns.violinplot, 'taste', 'phase', 'Condition', dodge=True)	
	g = (g.map(sns.violinplot, 'taste', 'phase', 'Condition', palette=pal).add_legend())
		
	plt.savefig('./Grouped_FR_analyses/' + \
            '/{}_Conditional_Phasic_Hists.png'.format(animal_name))	
	plt.close()
	
# =============================================================================
# 	
# 	g = sns.FacetGrid(Phase_df.query('Animal == @animal_name and comp_time == 1 and taste == 0'), col = 'held_unit', row = 'band',
#              subplot_kws=dict(projection='polar'), hue="phase", despine=False,sharey=True)
# 	
# 	g = g.map(plt.scatter,'phase', 'time')
# 	
# 	df1 = pd.melt(Phase_df.query('Animal == @animal_name and comp_time == 1 and taste == 0'), id_vars=['phase'], var_name='phase', value_name='band')
# 	
# 
# 
# =============================================================================










