"""
For help with input arguments:
    python blech_exp_info.py -h


Code to generate file containing relevant experimental info:

X Animal name
X Exp Type
X Date
X Time Stamp
X Regions Recorded from
X Electrode Layout According to Regions
X Taste concentrations and dig_in order
X Taste Palatability Ranks
X Laser parameters and dig_in
X Misc Notes
"""

"""
Module for handling experimental information in the Blech dataset.

This module includes functions to manage and retrieve experimental metadata.
"""
import json
import numpy as np
import os
import re
import argparse
import pandas as pd
from tqdm import tqdm
# When running in Spyder, throws an error,
# so cd to utils folder and then back out
from utils.blech_utils import entry_checker, imp_metadata, pipeline_graph_check
from utils.importrhdutilities import load_file, read_header

# Get name of directory with the data files
# Create argument parser
parser = argparse.ArgumentParser(
    description='Creates files with experiment info')
parser.add_argument('dir_name',  help='Directory containing data files')
parser.add_argument('--template', '-t',
                    help='Template (.info) file to copy experimental details from')
parser.add_argument('--mode', '-m', default='legacy',
                    choices=['legacy', 'updated'])
args = parser.parse_args()

metadata_handler = imp_metadata([[], args.dir_name])
dir_path = metadata_handler.dir_name

dir_name = os.path.basename(dir_path[:-1])

script_path = os.path.abspath(__file__)
this_pipeline_check = pipeline_graph_check(dir_path)
this_pipeline_check.write_to_log(script_path, 'attempted')

# Extract details from name of folder
splits = dir_name.split("_")
# Date and Timestamp are given as 2 sets of 6 digits
# Extract using regex
time_pattern = re.compile(r'\d{6}')
time_match = time_pattern.findall(dir_name)
if len(time_match) != 2:
    # raise ValueError('Timestamp not found in folder name')
    print('Timestamp not found in folder name')
    time_match = ['NA', 'NA']

this_dict = {
    "name": splits[0],
    "exp_type": splits[1],
    # "date": splits[-2],
    # "timestamp": splits[-1]}
    "date": time_match[0], 
    "timestamp": time_match[1],
    } 

##################################################
# Brain Regions and Electrode Layout
##################################################

if args.template:
    with open(args.template, 'r') as file:
        template_dict = json.load(file)
        var_names = ['regions', 'ports', 'electrode_layout', 'taste_params',
                     'laser_params', 'notes']

        from_template = {key: template_dict[key] for key in var_names}
        fin_dict = {**this_dict, **from_template}

else:

    # Find all ports used
    file_list = os.listdir(dir_path)
    if 'auxiliary.dat' in file_list:
        file_type = ['one file per signal type']
    elif len(['rhd' in x for x in file_list]) > 1:
        file_type = ['traditional']
    else:
        file_type = ['one file per channel']

    if file_type == ['one file per signal type']:
        electrodes_list = ['amplifier.dat']
        dig_in_list = ['digitalin.dat']
    elif file_type == ['one file per channel']:
        electrodes_list = [
            name for name in file_list if name.startswith('amp-')]
        dig_in_list = [
            name for name in file_list if name.startswith('board-DI')]
    else:
        with open(os.path.join(dir_path , file_list[0]), 'rb') as f:
            header = read_header(f)
        # temp_file, data_present = importrhdutilities.load_file(file_list[0])
        ports = [x['port_prefix'] for x in header['amplifier_channels']]
        electrode_files = [x['native_channel_name'] for x in header['amplifier_channels']]
        dig_in_list = [x['native_channel_name'].lower() for x in header['board_dig_in_channels']]

    dig_in_list = sorted(dig_in_list)

    if file_type == ['one file per channel']:
        electrode_files = sorted(electrodes_list)
        ports = [x.split('-')[1] for x in electrode_files]
        electrode_num_list = [x.split('-')[2].split('.')[0]
                              for x in electrode_files]
        # Sort the ports in alphabetical order
        ports.sort()
    elif file_type == ['one file per signal type']:
        print("\tSingle Amplifier File Detected")
        # Import amplifier data and calculate the number of electrodes
        print("\t\tCalculating Number of Ports")
        num_recorded_samples = len(np.fromfile(
            dir_path + 'time.dat', dtype=np.dtype('float32')))
        amplifier_data = np.fromfile(
            dir_path + 'amplifier.dat', dtype=np.dtype('uint16'))
        num_electrodes = int(len(amplifier_data)/num_recorded_samples)
        electrode_files = ['amplifier.dat' for i in range(num_electrodes)]
        ports = ['A']*num_electrodes
        electrode_num_list = list(np.arange(num_electrodes))
        del amplifier_data, num_electrodes
    elif file_type == ['traditional']:
        print("\tTraditional Intan Data Detected")
        electrode_num_list = [x.split('-')[1] for x in electrode_files]
        # Port have already been extracted

    # Write out file and ask user to define regions in file
    layout_file_path = os.path.join(
        dir_path, dir_name + "_electrode_layout.csv")

    def yn_check(x):
        return x in ['y', 'yes', 'n', 'no']

    if os.path.exists(layout_file_path):

        use_csv_str, continue_bool = entry_checker(
            msg="Layout file detected...use what's there? (y/yes/no/n) :: ",
            check_func=yn_check,
            fail_response='Please [y, yes, n, no]')
    else:
        use_csv_str = 'n'

    if use_csv_str in ['n', 'no']:
        layout_frame = pd.DataFrame()
        layout_frame['filename'] = electrode_files
        layout_frame['port'] = ports
        layout_frame['electrode_num'] = electrode_num_list
        layout_frame['electrode_ind'] = layout_frame.index
        layout_frame['CAR_group'] = pd.Series()

        layout_frame = \
            layout_frame[['filename', 'electrode_ind',
                          'electrode_num', 'port', 'CAR_group']]

        layout_frame.to_csv(layout_file_path, index=False)

        prompt_str = 'Please fill in car groups / regions' + "\n" + \
            "emg and none are case-specific" + "\n" +\
            "Indicate different CARS from same region as GC1,GC2...etc"
        print(prompt_str)

        def confirm_check(x):
            this_bool = x in ['y', 'yes']
            return this_bool
        perm_str, continue_bool = entry_checker(
            msg='Lemme know when its done (y/yes) :: ',
            check_func=confirm_check,
            fail_response='Please say y or yes')
        if not continue_bool:
            print('Welp...')
            exit()

    layout_frame_filled = pd.read_csv(layout_file_path)
    layout_frame_filled['CAR_group'] = \
            layout_frame_filled['CAR_group'].str.lower()
    layout_frame_filled['CAR_group'] = [x.strip() for x in
                                        layout_frame_filled['CAR_group']]
    layout_dict = dict(
        list(layout_frame_filled.groupby('CAR_group').electrode_ind))
    for key, vals in layout_dict.items():
        layout_dict[key] = [layout_dict[key].to_list()]

    if any(['emg' in x for x in layout_dict.keys()]):
        orig_emg_electrodes = [layout_dict[x][0] for x in layout_dict.keys()
                               if 'emg' in x]
        orig_emg_electrodes = [x for y in orig_emg_electrodes for x in y]
        fin_emg_port = layout_frame_filled.port.loc[
            layout_frame_filled.electrode_ind.isin(orig_emg_electrodes)].\
            unique()
        fin_emg_port = list(fin_emg_port)
        # Ask for emg muscle
        emg_muscle_str, continue_bool = entry_checker(
            msg='Enter EMG muscle name :: ',
            check_func=lambda x: True,
            fail_response='Please enter a valid muscle name')
        if not continue_bool:
            exit()
    else:
        fin_emg_port = []
        orig_emg_electrodes = []
        emg_muscle_str = ''

    fin_perm = layout_dict

    ##################################################
    # Dig-Ins
    ##################################################
    def count_check(x):
        nums = re.findall('[0-9]+', x)
        return sum([x.isdigit() for x in nums]) == len(nums)

    # Calculate number of deliveries from recorded data
    if file_type == ['one file per channel']:
        dig_in_trials = []
        num_dig_ins = len(dig_in_list)
        bad_dig_in_inds = []
        for i in range(num_dig_ins):
            dig_inputs = np.array(np.fromfile(
                dir_path + dig_in_list[i], dtype=np.dtype('uint16')))
            d_diff = np.diff(dig_inputs)
            start_ind = np.where(d_diff == 1)[0]
            if len(start_ind) == 0:
                bad_dig_in_inds.append(i)
            else:
                dig_in_trials.append(int(len(start_ind)))
        if len(bad_dig_in_inds) > 0:
            bad_dig_in_str = '\n'.join([dig_in_list[i] for i in bad_dig_in_inds])
            print(f"== No deliveries detected for following dig-ins ==" + '\n') 
            print('\n'+f"== {bad_dig_in_str} ==" + '\n')
            print('== They will be REMOVED from the list of dig-ins ==')
            dig_in_list = [x for i,x in enumerate(dig_in_list) if i not in bad_dig_in_inds]
            print('== Remaining dig-ins ==' + '\n')
            print('\n'.join(dig_in_list))

    elif file_type == ['one file per signal type']:
        d_inputs = np.fromfile(
            dir_path + dig_in_list[0], dtype=np.dtype('uint16'))
        d_inputs_str = d_inputs.astype('str')
        del d_inputs
        d_in_str_int = d_inputs_str.astype('int64')
        del d_inputs_str
        d_diff = np.diff(d_in_str_int)
        del d_in_str_int
        dig_in = list(np.unique(np.abs(d_diff)) - 1)
        dig_in.remove(-1)
        num_dig_ins = len(dig_in)
        dig_in_trials = []
        for n_i in range(num_dig_ins):
            start_ind = np.where(d_diff == n_i + 1)[0]
            if len(start_ind) == 0:
                bad_dig_in_inds.append(i)
            else:
                dig_in_trials.append(int(len(start_ind)))
        if len(bad_dig_in_inds) > 0:
            bad_dig_in_str = '\n'.join([dig_in_list[i] for i in bad_dig_in_inds])
            print(f"== No deliveries detected for following dig-ins ==" + '\n') 
            print('\n'+f"== {bad_dig_in_str} ==" + '\n')
            print('== They will be REMOVED from the list of dig-ins ==')
            dig_in_list = [x for i,x in enumerate(dig_in_list) if i not in bad_dig_in_inds]
            print('== Remaining dig-ins ==' + '\n')
            print('\n'.join(dig_in_list))

    elif file_type == ['traditional']:
        num_dig_ins = len(dig_in_list)
        bad_dig_in_inds = []

        dig_in_trials = {}
        rhd_file_list = [x for x in file_list if 'rhd' in x]
        for this_file in tqdm(rhd_file_list):
            this_file_data, data_present = load_file(os.path.join(dir_path, this_file))
            dig_inputs = this_file_data['board_dig_in_data']
            d_diff = np.diff(dig_inputs, axis=-1)
            start_ind = np.where(d_diff == 1)

            for this_dig in start_ind[0]:
                if this_dig not in dig_in_trials.keys():
                    dig_in_trials[this_dig] = 0
                dig_in_trials[this_dig] += 1

        for i in range(num_dig_ins):
            if i not in dig_in_trials.keys():
                bad_dig_in_inds.append(i)

        if len(bad_dig_in_inds) > 0:
            bad_dig_in_str = '\n'.join([dig_in_list[i] for i in bad_dig_in_inds])
            print(f"== No deliveries detected for following dig-ins ==" + '\n') 
            print('\n'+f"== {bad_dig_in_str} ==" + '\n')
            print('== They will be REMOVED from the list of dig-ins ==')
            dig_in_list = [x for i,x in enumerate(dig_in_list) if i not in bad_dig_in_inds]
            print('== Remaining dig-ins ==' + '\n')
            print('\n'.join(dig_in_list))
            dig_in_trials = [dig_in_trials[i] for i in dig_in_trials.keys()]

    indexed_digin_list = list(
        zip(np.arange(len(dig_in_list)), dig_in_list))
    dig_in_print_str = "Dig-ins : \n" + \
        ",\n".join([str(x) for x in indexed_digin_list])
    dig_in_present_bool = len(dig_in_list) > 0

    # Ask for user input of which line index the dig in came from
    if dig_in_present_bool:
        print(dig_in_print_str + "\n were found. Please provide the indices.")
        taste_dig_in_str, continue_bool = entry_checker(
            msg=' INDEX of Taste dig_ins used (IN ORDER, anything separated) :: ',
            check_func=count_check,
            fail_response='Please enter integers only')
        if continue_bool:
            nums = re.findall('[0-9]+', taste_dig_in_str)
            taste_digins = [int(x) for x in nums]
            taste_digin_filenames = [dig_in_list[i] for i in taste_digins]
            print('Selected taste digins: \n' + "\n".join(taste_digin_filenames))
        else:
            exit()

        def float_check(x):
            global taste_digins
            return len(x.split(',')) == len(taste_digins)

        def taste_check(x):
            global taste_digins
            return len(re.findall('[A-Za-z]+', x)) == len(taste_digins)

        taste_str, continue_bool = entry_checker(
            msg=' Tastes names used (IN ORDER, anything separated)  :: ',
            check_func=taste_check,
            fail_response='Please enter as many tastes as digins')
        if continue_bool:
            tastes = re.findall('[A-Za-z]+', taste_str)
        else:
            exit()

        conc_str, continue_bool = entry_checker(
            msg='Corresponding concs used (in M, IN ORDER, COMMA separated)  :: ',
            check_func=float_check,
            fail_response='Please enter as many concentrations as digins')
        if continue_bool:
            concs = [float(x) for x in conc_str.split(",")]
        else:
            exit()

        # Ask user for palatability rankings
        def pal_check(x):
            global taste_digins
            nums = re.findall('[1-9]+', x)
            return sum([x.isdigit() for x in nums]) == len(nums) and \
                sum([1 <= int(x) <= len(taste_digins)
                    for x in nums]) == len(taste_digins)

        taste_fin = str(list(zip(taste_digins, list(zip(tastes, concs)))))
        palatability_str, continue_bool = \
            entry_checker(
                msg=f' {taste_fin} \n Enter palatability rankings used '
                '(anything separated), higher number = more palatable  :: ',
                check_func=pal_check,
                fail_response='Please enter numbers 1<=x<len(tastes)')
        if continue_bool:
            nums = re.findall('[1-9]+', palatability_str)
            pal_ranks = [int(x) for x in nums]
        else:
            exit()
    else:
        print('No dig-ins found. Please check your data.')
        taste_digins = []
        taste_digin_filenames = []
        tastes = []
        concs = []
        pal_ranks = []


    ########################################
    # Ask for laser info
    # TODO: Allow for (onset, duration) tuples to be entered
    laser_select_str, continue_bool = entry_checker(
        msg='Laser dig_in index, <BLANK> for none :: ',
        check_func=count_check,
        fail_response='Please enter a single, valid integer')
    if continue_bool:
        if len(laser_select_str) == 0:
            laser_digin = []
            laser_digin_filenames = []
        else:
            laser_digin = [int(laser_select_str)]
            laser_digin_filenames = [dig_in_list[i] for i in laser_digin]
            print('Selected laser digins: \n' +
                  "\n".join(laser_digin_filenames))
    else:
        exit()

    def laser_check(x):
        nums = re.findall('[0-9]+', x)
        return sum([x.isdigit() for x in nums]) == 2
    if laser_digin:
        laser_select_str, continue_bool = entry_checker(
            msg='Laser onset_time, duration (ms, IN ORDER, anything separated) :: ',
            check_func=laser_check,
            fail_response='Please enter two, valid integers')
        if continue_bool:
            nums = re.findall('[0-9]+', laser_select_str)
            onset_time, duration = [int(x) for x in nums]
        else:
            exit()
        # Ask for virus region
        virus_region_str, continue_bool = entry_checker(
            msg='Enter virus region :: ',
            check_func=lambda x: True,
            fail_response='Please enter a valid region')
        if not continue_bool:
            exit()
        # Ask for opto-fiber location
        opto_loc_str, continue_bool = entry_checker(
            msg='Enter opto-fiber location :: ',
            check_func=lambda x: True,
            fail_response='Please enter a valid location')
        if not continue_bool:
            exit()
    else:
        onset_time, duration = [None, None]
        virus_region_str = ''
        opto_loc_str = ''

    notes = input('Please enter any notes about the experiment. \n :: ')

    ########################################
    # Finalize dictionary
    ########################################

    taste_digin_trials = [dig_in_trials[x] for x in taste_digins]
    if laser_digin:
        laser_digin_trials = [dig_in_trials[x] for x in laser_digin]
    else:
        laser_digin_trials = []

    fin_dict = {**this_dict,
                'regions': list(layout_dict.keys()),
                'ports': list(np.unique(ports)),
                'dig_ins': {
                    'filenames': dig_in_list,
                    'count': len(dig_in_trials),
                    'trial_counts': dig_in_trials,
                },
                'emg': {
                    'port': fin_emg_port,
                    'electrodes': orig_emg_electrodes,
                    'muscle': emg_muscle_str},
                'electrode_layout': fin_perm,
                'taste_params': {
                    'dig_ins': taste_digins,
                    'filenames': taste_digin_filenames,
                    'trial_count': taste_digin_trials,
                    'tastes': tastes,
                    'concs': concs,
                    'pal_rankings': pal_ranks},
                'laser_params': {
                    'dig_in': laser_digin,
                    'filenames': laser_digin_filenames,
                    'trial_count': laser_digin_trials,
                    'onset': onset_time,
                    'duration': duration,
                    'virus_region': virus_region_str,
                    'opto_loc': opto_loc_str},
                'notes': notes}


json_file_name = os.path.join(dir_path, '.'.join([dir_name, 'info']))
with open(json_file_name, 'w') as file:
    json.dump(fin_dict, file, indent=4)

# Write success to log
this_pipeline_check.write_to_log(script_path, 'completed')
