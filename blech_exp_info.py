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

import json
import numpy as np
import os
import re
import argparse
import pandas as pd
from tqdm import tqdm
# When running in Spyder, throws an error,
# so cd to utils folder and then back out
from utils.blech_utils import (
        entry_checker, 
        imp_metadata, 
        pipeline_graph_check,
        )
from utils.importrhdutilities import load_file, read_header
from utils.read_file import DigInHandler

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Creates files with experiment info')
    parser.add_argument('dir_name',  help='Directory containing data files')
    parser.add_argument('--template', '-t',
                        help='Template (.info) file to copy experimental details from')
    parser.add_argument('--mode', '-m', default='legacy',
                        choices=['legacy', 'updated'])
    parser.add_argument('--programmatic', action='store_true',
                        help='Run in programmatic mode')
    parser.add_argument('--use-layout-file', action='store_true', 
                        help='Use existing electrode layout file')
    parser.add_argument('--car-groups', help='Comma-separated CAR groupings')
    parser.add_argument('--emg-muscle', help='Name of EMG muscle')
    parser.add_argument('--taste-digins', help='Comma-separated indices of taste digital inputs')
    parser.add_argument('--tastes', help='Comma-separated taste names')
    parser.add_argument('--concentrations', help='Comma-separated concentrations in M')
    parser.add_argument('--palatability', help='Comma-separated palatability rankings')
    parser.add_argument('--laser-digin', help='Laser digital input index')
    parser.add_argument('--laser-params', help='Laser onset,duration in ms')
    parser.add_argument('--virus-region', help='Virus region')
    parser.add_argument('--opto-loc', help='Opto-fiber location')
    parser.add_argument('--notes', help='Experiment notes')
    return parser.parse_args()

def process_files(dir_path):
    """Process data files and determine file type"""
    file_list = os.listdir(dir_path)
    
    # Determine file type
    if 'auxiliary.dat' in file_list:
        file_type = 'one file per signal type'
    elif sum(['rhd' in x for x in file_list]) > 1:
        file_type = 'traditional'
    else:
        file_type = 'one file per channel'

    # Get electrode files and ports based on file type
    if file_type == 'one file per signal type':
        electrode_files = ['amplifier.dat']
        ports = ['A']
        num_recorded_samples = len(np.fromfile(
            os.path.join(dir_path, 'time.dat'), dtype=np.dtype('float32')))
        amplifier_data = np.fromfile(
            os.path.join(dir_path, 'amplifier.dat'), dtype=np.dtype('uint16'))
        num_electrodes = int(len(amplifier_data)/num_recorded_samples)
        electrode_files = ['amplifier.dat' for _ in range(num_electrodes)]
        ports = ['A']*num_electrodes
        electrode_num_list = list(range(num_electrodes))
    elif file_type == 'one file per channel':
        electrodes_list = [name for name in file_list if name.startswith('amp-')]
        electrode_files = sorted(electrodes_list)
        ports = [x.split('-')[1] for x in electrode_files]
        electrode_num_list = [x.split('-')[2].split('.')[0] for x in electrode_files]
        ports.sort()
    else:  # traditional
        rhd_file_list = [x for x in file_list if 'rhd' in x]
        with open(os.path.join(dir_path, rhd_file_list[0]), 'rb') as f:
            header = read_header(f)
        ports = [x['port_prefix'] for x in header['amplifier_channels']]
        electrode_files = [x['native_channel_name'] for x in header['amplifier_channels']]
        electrode_num_list = [x.split('-')[1] for x in electrode_files]

    return file_type, ports, electrode_files, electrode_num_list

def process_template(args, exp_info):
    """Process template file if provided"""
    if not args.template:
        return exp_info
        
    with open(args.template, 'r') as file:
        template_dict = json.load(file)
        template_keys = list(template_dict.keys())
        from_template = {
            this_key:template_dict[this_key] for this_key in template_keys \
            if this_key not in exp_info.keys()
        }
        return {**exp_info, **from_template}

def process_dig_ins(dir_path, file_type, args):
    """Process digital inputs and taste information"""
    dig_handler = DigInHandler(dir_path, file_type)
    dig_handler.get_dig_in_files()
    dig_handler.get_trial_data()

    # Check if any dig-ins are present
    dig_in_present = any(dig_handler.dig_in_frame.trial_counts > 0)
    if not dig_in_present:
        print('No dig-ins found. Please check your data.')
        return (dig_handler, [], [], [], [], [], [], [])

    def count_check(x):
        nums = re.findall('[0-9]+', x)
        return sum([x.isdigit() for x in nums]) == len(nums)

    # Get taste dig-ins
    if not args.programmatic:
        taste_dig_in_str, continue_bool = entry_checker(
            msg=' INDEX of Taste dig_ins used (IN ORDER, anything separated) :: ',
            check_func=count_check,
            fail_response='Please enter integers only')
        if not continue_bool:
            exit()
        nums = re.findall('[0-9]+', taste_dig_in_str)
        taste_dig_inds = [int(x) for x in nums]
    else:
        if not args.taste_digins:
            raise ValueError('Taste dig-ins not provided, use --taste-digins')
        taste_dig_inds = parse_csv(args.taste_digins, int)

    # Process taste information
    dig_handler.dig_in_frame.loc[taste_dig_inds, 'taste_bool'] = True
    dig_handler.dig_in_frame.taste_bool.fillna(False, inplace=True)
    
    taste_digin_nums = dig_handler.dig_in_frame.loc[taste_dig_inds, 'dig_in_nums'].to_list()
    taste_digin_trials = dig_handler.dig_in_frame.loc[taste_dig_inds, 'trial_counts'].to_list()

    # Get tastes
    def taste_check(x):
        return len(re.findall('[A-Za-z]+', x)) == len(taste_dig_inds)

    if not args.programmatic:
        taste_str, continue_bool = entry_checker(
            msg=' Tastes names used (IN ORDER, anything separated [no punctuation in name])  :: ',
            check_func=taste_check,
            fail_response='Please enter as many tastes as digins')
        if not continue_bool:
            exit()
        tastes = re.findall('[A-Za-z]+', taste_str)
    else:
        if not args.tastes:
            raise ValueError('Tastes not provided, use --tastes')
        tastes = parse_csv(args.tastes)

    dig_handler.dig_in_frame.loc[taste_dig_inds, 'taste'] = tastes

    # Get concentrations
    def float_check(x):
        return len(x.split(',')) == len(taste_dig_inds)

    if not args.programmatic:
        conc_str, continue_bool = entry_checker(
            msg='Corresponding concs used (in M, IN ORDER, COMMA separated)  :: ',
            check_func=float_check,
            fail_response='Please enter as many concentrations as digins')
        if not continue_bool:
            exit()
        concs = [float(x) for x in conc_str.split(",")]
    else:
        if not args.concentrations:
            raise ValueError('Concentrations not provided, use --concentrations')
        concs = parse_csv(args.concentrations, float)

    dig_handler.dig_in_frame.loc[taste_dig_inds, 'concentration'] = concs

    # Get palatability rankings
    def pal_check(x):
        nums = re.findall('[1-9]+', x)
        if not nums:
            return False
        pal_nums = [int(n) for n in nums]
        return all(1 <= p <= len(tastes) for p in pal_nums) and len(pal_nums) == len(tastes)

    if not args.programmatic:
        palatability_str, continue_bool = entry_checker(
            msg=f'Enter palatability rankings (IN ORDER) used '
                '(anything separated), higher number = more palatable  :: ',
            check_func=pal_check,
            fail_response=f'Please enter numbers 1<=x<={len(tastes)}')
        if not continue_bool:
            exit()
        nums = re.findall('[1-9]+', palatability_str)
        pal_ranks = [int(x) for x in nums]
    else:
        if not args.palatability:
            raise ValueError('Palatability rankings not provided, use --palatability')
        pal_ranks = parse_csv(args.palatability, int)

    dig_handler.dig_in_frame.loc[taste_dig_inds, 'palatability'] = pal_ranks

    return (dig_handler, taste_dig_inds, taste_digin_nums, taste_digin_trials, 
            tastes, concs, pal_ranks)

def main():
    """Main function to orchestrate experiment info generation"""
    args = parse_args()
    
    # Set up metadata and paths
    metadata_handler = imp_metadata([[], args.dir_name])
    dir_path = metadata_handler.dir_name
    dir_name = os.path.basename(dir_path[:-1])
    
    # Set up pipeline check
    script_path = os.path.abspath(__file__)
    pipeline_check = pipeline_graph_check(dir_path)
    pipeline_check.write_to_log(script_path, 'attempted')
    
    # Get experiment info
    exp_info = get_experiment_info(dir_name)
    
    # Process template if provided
    exp_info = process_template(args, exp_info)
    
    # Process files
    file_type, ports, electrode_files, electrode_num_list = process_files(dir_path)
    
    # Process layout and get all parameters
    layout_info = process_layout(dir_path, dir_name, file_type, ports, 
                               electrode_files, electrode_num_list, args)
    
    # Create final dictionary
    fin_dict = {
        'version': '0.0.2',
        **exp_info,
        **layout_info
    }
    
    # Write to JSON file
    json_file_name = os.path.join(dir_path, '.'.join([dir_name, 'info']))
    with open(json_file_name, 'w') as file:
        json.dump(fin_dict, file, indent=4)
    
    # Write success to log
    pipeline_check.write_to_log(script_path, 'completed')

if __name__ == '__main__':
    main()

def parse_csv(s, convert=str):
    """Parse comma-separated values with optional type conversion"""
    if not s:
        return []
    return [convert(x.strip()) for x in s.split(',')]

def get_experiment_info(dir_name):
    """Extract experiment info from directory name"""
    splits = dir_name.split("_")
    # Date and Timestamp are given as 2 sets of 6 digits
    time_pattern = re.compile(r'\d{6}')
    time_match = time_pattern.findall(dir_name)
    if len(time_match) != 2:
        print('Timestamp not found in folder name')
        time_match = ['NA', 'NA']

    return {
        "name": splits[0],
        "exp_type": splits[1],
        "date": time_match[0],
        "timestamp": time_match[1],
    }

def process_layout(dir_path, dir_name, file_type, ports, electrode_files, electrode_num_list, args):
    """Process electrode layout information
    
    Args:
        dir_path (str): Path to data directory
        dir_name (str): Name of directory
        file_type (str): Type of data files
        ports (list): List of ports
        electrode_files (list): List of electrode files
        electrode_num_list (list): List of electrode numbers
        args (argparse.Namespace): Command line arguments
        
    Returns:
        dict: Dictionary containing:
            - layout_dict (dict): Dictionary of electrode layouts
            - emg_info (dict): EMG-related information
            - dig_ins (dict): Digital input information
            - taste_params (dict): Taste parameters
            - laser_params (dict): Laser parameters
            - notes (str): Experiment notes
    """

    # Find all ports used
    file_list = os.listdir(dir_path)
    if 'auxiliary.dat' in file_list:
        file_type = 'one file per signal type'
    elif sum(['rhd' in x for x in file_list]) > 1:
        file_type = 'traditional'
    else:
        file_type = 'one file per channel'

    if file_type == 'one file per signal type':
        electrodes_list = ['amplifier.dat']
    elif file_type == 'one file per channel':
        electrodes_list = [
            name for name in file_list if name.startswith('amp-')]
    else:
        rhd_file_list = [x for x in file_list if 'rhd' in x]
        with open(os.path.join(dir_path , rhd_file_list[0]), 'rb') as f:
            header = read_header(f)
        ports = [x['port_prefix'] for x in header['amplifier_channels']]
        electrode_files = [x['native_channel_name'] for x in header['amplifier_channels']]

    ##################################################
    # Dig-Ins
    ##################################################
    # Process dig-ins
    this_dig_handler = DigInHandler(dir_path, file_type)
    this_dig_handler.get_dig_in_files()
    dig_in_list_str = "All Dig-ins : \n" + ", ".join([str(x) for x in this_dig_handler.dig_in_num])
    this_dig_handler.get_trial_data()

    def count_check(x):
        nums = re.findall('[0-9]+', x)
        return sum([x.isdigit() for x in nums]) == len(nums)

    # Calculate number of deliveries from recorded data
    dig_in_present_bool = any(this_dig_handler.dig_in_frame.trial_counts > 0)

    # Ask for user input of which line index the dig in came from
    if dig_in_present_bool:
        if not args.programmatic:
            taste_dig_in_str, continue_bool = entry_checker(
                msg=' INDEX of Taste dig_ins used (IN ORDER, anything separated) :: ',
                check_func=count_check,
                fail_response='Please enter integers only')
            if continue_bool:
                nums = re.findall('[0-9]+', taste_dig_in_str)
                taste_dig_inds = [int(x) for x in nums]
            else:
                exit()
        else:
            if args.taste_digins:
                taste_dig_inds = parse_csv(args.taste_digins, int)
            else:
                raise ValueError('Taste dig-ins not provided, use --taste-digins')

        this_dig_handler.dig_in_frame.loc[taste_dig_inds, 'taste_bool'] = True
        this_dig_handler.dig_in_frame.taste_bool.fillna(False, inplace=True)
        print('Taste dig-in frame: \n')
        print_df = this_dig_handler.dig_in_frame.drop(columns = ['pulse_times'])
        print_df = print_df[print_df.taste_bool]
        print(print_df)

        def float_check(x):
            return len(x.split(',')) == len(taste_dig_inds)

        def taste_check(x):
            return len(re.findall('[A-Za-z]+', x)) == len(taste_dig_inds)

        def pal_check(x):
            nums = re.findall('[1-9]+', x)
            if not nums:
                return False
            pal_nums = [int(n) for n in nums]
            return all(1 <= p <= len(tastes) for p in pal_nums) and len(pal_nums) == len(tastes)

        if not args.programmatic:
            taste_str, continue_bool = entry_checker(
                msg=' Tastes names used (IN ORDER, anything separated [no punctuation in name])  :: ',
                check_func=taste_check,
                fail_response='Please enter as many tastes as digins')
            if continue_bool:
                tastes = re.findall('[A-Za-z]+', taste_str)
            else:
                exit()
        else:
            if args.tastes:
                tastes = parse_csv(args.tastes)
            else:
                raise ValueError('Tastes not provided, use --tastes')

        this_dig_handler.dig_in_frame.loc[taste_dig_inds, 'taste'] = tastes
        print('Taste dig-in frame: \n')
        print_df = this_dig_handler.dig_in_frame.drop(columns = ['pulse_times'])
        print_df = print_df[print_df.taste_bool]
        print(print_df)

        if not args.programmatic:
            conc_str, continue_bool = entry_checker(
                msg='Corresponding concs used (in M, IN ORDER, COMMA separated)  :: ',
                check_func=float_check,
                fail_response='Please enter as many concentrations as digins')
            if continue_bool:
                concs = [float(x) for x in conc_str.split(",")]
            else:
                exit()
        else:
            if args.concentrations:
                concs = parse_csv(args.concentrations, float)
            else:
                raise ValueError('Concentrations not provided, use --concentrations')

        this_dig_handler.dig_in_frame.loc[taste_dig_inds, 'concentration'] = concs
        print('Taste dig-in frame: \n')
        print_df = this_dig_handler.dig_in_frame.drop(columns = ['pulse_times'])
        print_df = print_df[print_df.taste_bool]
        print(print_df)

        # Ask user for palatability rankings
        if not args.programmatic:
            palatability_str, continue_bool = \
                entry_checker(
                    msg=f'Enter palatability rankings (IN ORDER) used '
                    '(anything separated), higher number = more palatable  :: ',
                    check_func=pal_check,
                    fail_response=f'Please enter numbers 1<=x<={len(print_df)}')
            if continue_bool:
                nums = re.findall('[1-9]+', palatability_str)
                pal_ranks = [int(x) for x in nums]
            else:
                exit()
        else:
            if args.palatability:
                pal_ranks = parse_csv(args.palatability, int)
            else:
                raise ValueError('Palatability rankings not provided, use --palatability')

        this_dig_handler.dig_in_frame.loc[taste_dig_inds, 'palatability'] = pal_ranks
        print('Taste dig-in frame: \n')
        print_df = this_dig_handler.dig_in_frame.drop(columns = ['pulse_times'])
        print_df = print_df[print_df.taste_bool]
        print(print_df)

        taste_digin_nums = this_dig_handler.dig_in_frame.loc[taste_dig_inds, 'dig_in_nums'].to_list()
        tastes = this_dig_handler.dig_in_frame.loc[taste_dig_inds, 'taste'].to_list()
        concs = this_dig_handler.dig_in_frame.loc[taste_dig_inds, 'concentration'].to_list()
        pal_ranks = this_dig_handler.dig_in_frame.loc[taste_dig_inds, 'palatability'].to_list()
        taste_digin_trials = this_dig_handler.dig_in_frame.loc[taste_dig_inds, 'trial_counts'].to_list()
    else:
        print('No dig-ins found. Please check your data.')
        taste_digins = []
        taste_digin_nums = []
        tastes = []
        concs = []
        pal_ranks = []
        taste_digin_trials = []


    ########################################
    # Ask for laser info
    # TODO: Allow for (onset, duration) tuples to be entered
    if not args.programmatic:
        laser_select_str, continue_bool = entry_checker(
            msg='Laser dig_in index, <BLANK> for none :: ',
            check_func=count_check,
            fail_response='Please enter a single, valid integer')
        if continue_bool:
            if len(laser_select_str) == 0:
                laser_digin_ind = []
            else:
                laser_digin_ind = [int(laser_select_str)]
        else:
            exit()
    else:
        if args.laser_digin:
            laser_digin_ind = parse_csv(args.laser_digin, int)
        else:
            laser_digin_ind = []

    if laser_digin_ind:
        laser_digin_nums = this_dig_handler.dig_in_frame.loc[laser_digin_ind, 'dig_in_nums'].to_list()
        this_dig_handler.dig_in_frame.loc[laser_digin_ind, 'laser_bool'] = True
        this_dig_handler.dig_in_frame.laser_bool.fillna(False, inplace=True)
        laser_digin_nums = this_dig_handler.dig_in_frame.loc[laser_digin_ind, 'dig_in_nums'].to_list()
        print('Selected laser digins: \n')
        print_df = this_dig_handler.dig_in_frame.drop(columns = ['pulse_times'])
        print_df = print_df[print_df.laser_bool]
        print(print_df)
    else:
        laser_digin_nums = []

    def laser_check(x):
        nums = re.findall('[0-9]+', x)
        return sum([x.isdigit() for x in nums]) == 2

    if laser_digin_ind:
        if not args.programmatic:
            # Ask for laser parameters
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
            if args.laser_params:
                laser_params = parse_csv(args.laser_params, int) 
                onset_time, duration = laser_params
            else:
                raise ValueError('Laser parameters not provided, use --laser-params')
            if args.virus_region:
                virus_region_str = args.virus_region
            else:
                raise ValueError('Virus region not provided, use --virus-region')
            if args.opto_loc:
                opto_loc_str = args.opto_loc
            else:
                raise ValueError('Opto-fiber location not provided, use --opto-loc')

        # Fill in laser parameters
        this_dig_handler.dig_in_frame.loc[laser_digin_ind, 'laser_params'] = str([onset_time, duration])
    else:
        onset_time, duration = [None, None]
        virus_region_str = ''
        opto_loc_str = ''

    # Write out dig-in frame
    col_names = this_dig_handler.dig_in_frame.columns
    # Move 'pulse_times' to the end
    this_dig_handler.dig_in_frame = this_dig_handler.dig_in_frame[
        [x for x in col_names if x != 'pulse_times'] + ['pulse_times']]
    this_dig_handler.write_out_frame()

    ##############################


    if file_type == 'one file per channel':
        electrode_files = sorted(electrodes_list)
        ports = [x.split('-')[1] for x in electrode_files]
        electrode_num_list = [x.split('-')[2].split('.')[0]
                              for x in electrode_files]
        # Sort the ports in alphabetical order
        ports.sort()
    elif file_type == 'one file per signal type':
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
    elif file_type == 'traditional':
        print("\tTraditional Intan Data Detected")
        electrode_num_list = [x.split('-')[1] for x in electrode_files]
        # Port have already been extracted

    # Write out file and ask user to define regions in file
    layout_file_path = os.path.join(
        dir_path, dir_name + "_electrode_layout.csv")

    def yn_check(x):
        return x in ['y', 'yes', 'n', 'no']

    if os.path.exists(layout_file_path):
        # If neither programmatic nor use_layout_file, ask user
        if not args.programmatic and not args.use_layout_file and not args.car_groups:
            use_csv_str, continue_bool = entry_checker(
                msg="Layout file detected...use what's there? (y/yes/no/n) :: ",
                check_func=yn_check,
                fail_response='Please [y, yes, n, no]')
        elif args.car_groups:
            use_csv_str = 'n'
        # If use_layout_file, use it
        elif args.use_layout_file:
            use_csv_str = 'y'
        # If programmatic, don't use
        else:
            use_csv_str = 'y' 
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

        if not args.programmatic:
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

    if not args.programmatic:
        layout_frame_filled['CAR_group'] = \
                layout_frame_filled['CAR_group'].str.lower()
        layout_frame_filled['CAR_group'] = [x.strip() for x in
                                            layout_frame_filled['CAR_group']]
    else:
        if args.car_groups:
            car_groups = parse_csv(args.car_groups)
            layout_frame_filled['CAR_group'] = [x.strip().lower() for x in car_groups] 
        else:
            raise ValueError('CAR groups not provided, use --car-groups')

    layout_dict = dict(
        list(layout_frame_filled.groupby('CAR_group').electrode_ind))
    for key, vals in layout_dict.items():
        layout_dict[key] = [layout_dict[key].to_list()]

    # Write out layout_frame_filled if programmatically filled
    layout_frame_filled.to_csv(layout_file_path, index=False)

    if any(['emg' in x for x in layout_dict.keys()]):
        orig_emg_electrodes = [layout_dict[x][0] for x in layout_dict.keys()
                               if 'emg' in x]
        orig_emg_electrodes = [x for y in orig_emg_electrodes for x in y]
        fin_emg_port = layout_frame_filled.port.loc[
            layout_frame_filled.electrode_ind.isin(orig_emg_electrodes)].\
            unique()
        fin_emg_port = list(fin_emg_port)
        # Get EMG muscle name
        if not args.programmatic:
            emg_muscle_str, continue_bool = entry_checker(
                msg='Enter EMG muscle name :: ',
                check_func=lambda x: True,
                fail_response='Please enter a valid muscle name')
            if not continue_bool:
                exit()
        else:
            if args.emg_muscle:
                emg_muscle_str = args.emg_muscle
            else:
                raise ValueError('EMG muscle name not provided, use --emg-muscle')
    else:
        fin_emg_port = []
        orig_emg_electrodes = []
        emg_muscle_str = ''

    fin_perm = layout_dict

    ########################################
    # Finalize dictionary
    ########################################

    if not args.programmatic:
        notes = input('Please enter any notes about the experiment. \n :: ')
    else:
        notes = args.notes or ''


    if laser_digin_ind:
        laser_digin_trials = this_dig_handler.dig_in_frame.loc[laser_digin_ind, 'trial_counts'].to_list() 
    else:
        laser_digin_trials = []

    return {
        'file_type': file_type,
        'regions': list(layout_dict.keys()),
        'ports': list(np.unique(ports)),
        'dig_ins': {
            'nums': this_dig_handler.dig_in_frame.dig_in_nums.to_list(),
            'trial_counts': this_dig_handler.dig_in_frame.trial_counts.to_list(),
        },
        'emg': {
            'port': fin_emg_port,
            'electrodes': orig_emg_electrodes,
            'muscle': emg_muscle_str
        },
        'electrode_layout': fin_perm,
        'taste_params': {
            'dig_in_nums': taste_digin_nums,
            'trial_count': taste_digin_trials,
            'tastes': tastes,
            'concs': concs,
            'pal_rankings': pal_ranks
        },
        'laser_params': {
            'dig_in_nums': laser_digin_nums,
            'trial_count': laser_digin_trials,
            'onset': onset_time,
            'duration': duration,
            'virus_region': virus_region_str,
            'opto_loc': opto_loc_str
        },
        'notes': notes
    }
