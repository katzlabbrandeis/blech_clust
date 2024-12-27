#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 15:35:53 2017

@author: bradly
"""

# Import stuff!
import numpy as np
import tables
import easygui
import os
import argparse
from pprint import pprint

# Ask for the directory where the hdf5 file sits, and change to that directory
parser = argparse.ArgumentParser(description = 'Spike extraction and sorting script')
parser.add_argument('dir_name', help = 'Directory containing data files')
args = parser.parse_args()

if args.dir_name is not None:
    dir_name = os.path.abspath(args.dir_name)
    if dir_name[-1] != '/':
        dir_name += '/'
else:
    dir_name = easygui.diropenbox(msg = 'Please select data directory')

#dir_name = easygui.diropenbox()
os.chdir(dir_name)

# Look for the hdf5 file in the directory
file_list = os.listdir('./')
hdf5_name = ''
for files in file_list:
    if files[-2:] == 'h5':
        hdf5_name = files

# Open the hdf5 file
hf5 = tables.open_file(hdf5_name, 'r+')

dig_in_nodes = hf5.list_nodes('/spike_trains')
dig_in_names = [x._v_name for x in dig_in_nodes]
#trial_counts = [len(np.where(np.diff(x[:])==1)[0]) for x in dig_in_nodes]
trial_counts = [x.spike_array.shape[0] for x in dig_in_nodes]

# Ask user if they want to delete any dig_in_nodes
delete_bool = easygui.ynbox(
        title = "Delete any dig-ins?" ,
        msg = "Would you like to delete any dig-ins from spike trains? \n" +\
                f"dig_ins = {dig_in_names}" + '\n' +\
                f"trial counts = {trial_counts}"
        )

if delete_bool:
    delete_dig_ins = easygui.multchoicebox(
            msg = 'Delete which dig-ins? \n' +\
                    'This will delete only spike trains related to these digins',
            title  = 'Delete which dig-ins?',
            choices = dig_in_names
            )
    # Delete from spike_trains
    for dig_in in delete_dig_ins:
        hf5.remove_node("/spike_trains", dig_in, recursive=True)

# Reasses dig_ins
dig_in_nodes = hf5.list_nodes('/spike_trains')
dig_in_names = [x._v_name for x in dig_in_nodes]
#trial_counts = [len(np.where(np.diff(x[:])==1)[0]) for x in dig_in_nodes]
trial_counts = [x.spike_array.shape[0] for x in dig_in_nodes]

trim_bool = easygui.ynbox(
    title = "Trim spike trains" ,
    msg = "Would you like to trim all spike trains to a consistent trial number?\n"+\
            f"dig_ins = {dig_in_names}" + '\n' +\
            f"trial counts = {trial_counts}"
    )

if trim_bool:
    new_trial_number = easygui.enterbox(
            msg = 'Please enter new trial number',
            title  = 'Trim spike trains',
            default = np.min(trial_counts)
            )
    new_trial_number = int(new_trial_number)

# Grab array information for each digital input channel,
# split into first and last sections,
# place in corresponding digitial input group array
for node in dig_in_nodes:
    spike_array = node.spike_array[:new_trial_number]
    node.spike_array._f_remove()
    hf5.create_array(node._v_pathname, 'spike_array', spike_array)

dig_in_nodes = hf5.list_nodes('/spike_trains')
dig_in_names = [x._v_name for x in dig_in_nodes]
#trial_counts = [len(np.where(np.diff(x[:])==1)[0]) for x in dig_in_nodes]
trial_counts = [x.spike_array.shape[0] for x in dig_in_nodes]

print('========================================')
print('Final Dig-Ins and Trial Counts')
pprint(dict(zip(dig_in_names, trial_counts)), indent = 4, width = 1)
print('========================================')

hf5.flush()
hf5.close()
