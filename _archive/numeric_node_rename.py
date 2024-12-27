#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 08:47:54 2020

@author: bradly
"""
# import Libraries
# Built-in Python libraries
import os # functions for interacting w operating system

# 3rd-party libraries
import numpy as np # module for low-level scientific computing
import easygui
import tables

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
animal_nodes = easygui.multchoicebox(
        msg = 'Which animal(s) need renaming?',
        choices = ([x for x in animals]))

#Process the renaming
for animal in range(len(animal_nodes)):
	groups = [node._v_name for node in hf5.list_nodes('/%s' %(animal_nodes[animal]))]

	node_groups = easygui.multchoicebox(
        msg = 'Which groups need renaming for %s?' %(animals[animal]),
        choices = ([x for x in groups]))

	rename_val = easygui.multenterbox(msg = 'What will the new group name be for:'
								   '%s, session %s' %(animal_nodes[animal],node_groups[0]),\
								   fields = [node_groups[0]], values = [node_groups[0][1:]])

	#Add group with updated name and flip through old group to grab each array to store
	hf5.create_group('/%s' %(animal_nodes[animal]),rename_val[0])

	old_datasets = [dset._v_name for dset in hf5.list_nodes('/%s/%s' %(animal_nodes[animal],node_groups[0]))]

	for dset in range(len(old_datasets)):
		try:
			hf5.create_array('/%s/%s' %(animal_nodes[animal],rename_val[0]), old_datasets[dset],\
					   np.array(hf5.list_nodes('/%s/%s' %(animal_nodes[animal],node_groups[0]))[dset]))
			hf5.flush()
		except:

			hf5.flush()

	#remove old group
	hf5.remove_node('/%s/%s' %(animal_nodes[animal],node_groups[0]), recursive = True)
	hf5.flush()
	hf5.close
