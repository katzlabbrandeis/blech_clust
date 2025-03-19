"""
This module generates a file containing relevant experimental information for a given dataset. It processes data files to extract and organize details such as animal name, experiment type, date, timestamp, regions recorded from, electrode layout, taste concentrations, palatability ranks, laser parameters, and miscellaneous notes.

- Parses command-line arguments to specify the directory containing data files and optional parameters like template files, mode, and various experimental details.
- `parse_csv(s, convert=str)`: Helper function to parse comma-separated values from a string and convert them to a specified type.
- Extracts metadata from the directory name and checks the pipeline status.
- Processes digital input (dig-in) data to determine taste dig-ins, concentrations, palatability rankings, and laser parameters.
- Handles different file types for electrode data and generates or uses an existing electrode layout file.
- Organizes and writes out the final experimental information into a JSON file.
- Logs the completion status of the pipeline process.
- Caches manual entries to reduce redundant input across sessions.
- Auto-populates defaults from existing info files when available.
"""

test_bool = False  # noqa
import argparse  # noqa
if test_bool:
    args = argparse.Namespace(
        dir_name='/media/storage/for_transfer/bla_gc/AM35_4Tastes_201228_124547',
        template=None,
        mode='legacy',
        programmatic=False,
        use_layout_file=True,
        car_groups=None,
        emg_muscle=None,
        taste_digins=None,
        tastes=None,
        concentrations=None,
        palatability=None,
        laser_digin=None,
        laser_params=None,
        virus_region=None,
        opto_loc=None,
        notes=None
    )

else:
    # Create argument parser
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
    parser.add_argument(
        '--taste-digins', help='Comma-separated indices of taste digital inputs')
    parser.add_argument('--tastes', help='Comma-separated taste names')
    parser.add_argument('--concentrations',
                        help='Comma-separated concentrations in M')
    parser.add_argument(
        '--palatability', help='Comma-separated palatability rankings')
    parser.add_argument('--laser-digin', help='Laser digital input index')
    parser.add_argument(
        '--laser-params', help='Multiple laser parameters as (onset,duration) pairs in ms, comma-separated: (100,500),(200,300)')
    parser.add_argument('--virus-region', help='Virus region')
    parser.add_argument(
        '--opto-loc', help='Multiple opto-fiber locations, comma-separated (must match number of laser parameter pairs)')
    parser.add_argument('--notes', help='Experiment notes')
    args = parser.parse_args()

import json  # noqa
import numpy as np  # noqa
import os  # noqa
import re  # noqa
import pandas as pd  # noqa
from tqdm import tqdm  # noqa
# When running in Spyder, throws an error,
# so cd to utils folder and then back out
from utils.blech_utils import (
    entry_checker,
    imp_metadata,
    pipeline_graph_check,
)  # noqa
from utils.importrhdutilities import load_file, read_header  # noqa
from utils.read_file import DigInHandler  # noqa

# Define the cache directory and ensure it exists
script_path = os.path.abspath(__file__) if not test_bool else "test_path"
blech_clust_dir = os.path.dirname(os.path.dirname(script_path))
cache_dir = os.path.join(blech_clust_dir, 'cache_and_logs')
os.makedirs(cache_dir, exist_ok=True)

# Define the cache file path
cache_file_path = os.path.join(cache_dir, 'manual_entries_cache.json')


def load_cache():
    """Load the cache of manual entries if it exists"""
    if os.path.exists(cache_file_path):
        try:
            with open(cache_file_path, 'r') as cache_file:
                return json.load(cache_file)
        except json.JSONDecodeError:
            print("Warning: Cache file exists but is not valid JSON. Creating new cache.")
            return {}
    return {}


def save_to_cache(cache_dict):
    """Save the updated cache to the cache file"""
    with open(cache_file_path, 'w') as cache_file:
        json.dump(cache_dict, cache_file, indent=4)


def load_existing_info(dir_path, dir_name):
    """Load existing info file if it exists"""
    info_file_path = os.path.join(dir_path, f"{dir_name}.info")
    if os.path.exists(info_file_path):
        try:
            with open(info_file_path, 'r') as info_file:
                return json.load(info_file)
        except json.JSONDecodeError:
            print("Warning: Info file exists but is not valid JSON.")
            return {}
    return {}

# Get name of directory with the data files
# Helper function to parse comma-separated values


def parse_csv(s, convert=str):
    if not s:
        return []
    return [convert(x.strip()) for x in s.split(',')]


def populate_field_with_defaults(field_name, entry_checker_msg, check_func, existing_info, cache,
                                 convert_func=None, fail_response=None, nested_field=None):
    """
    Handle the logic for checking existing info, cache, and prompting the user.

    Args:
        field_name: Name of the field in cache
        entry_checker_msg: Message to display to user
        check_func: Function to validate user input
        existing_info: Dictionary containing existing information
        cache: Dictionary containing cached values
        convert_func: Function to convert user input to desired format
        fail_response: Message to display on validation failure
        nested_field: If field is nested in existing_info, provide the parent key

    Returns:
        The value to use for the field
    """
    default_value = []

    # Check existing info first
    if nested_field and nested_field in existing_info:
        if field_name in existing_info[nested_field]:
            default_value = existing_info[nested_field][field_name]
    elif field_name in existing_info:
        default_value = existing_info[field_name]
    # Then check cache
    elif field_name in cache:
        default_value = cache[field_name]

    # Format default for display
    if isinstance(default_value, list):
        default_str = ', '.join(map(str, default_value)
                                ) if default_value else ""
    else:
        default_str = str(default_value) if default_value else ""

    if fail_response is None:
        fail_response = f'Please enter valid input for {field_name}'

    # Prompt user
    user_input, continue_bool = entry_checker(
        msg=f'{entry_checker_msg} [{default_str}] :: ',
        check_func=check_func,
        fail_response=fail_response
    )

    if continue_bool:
        if user_input.strip():
            # Convert input if conversion function provided
            if convert_func:
                return convert_func(user_input)
            return user_input
        else:
            # Use default if input is empty
            return default_value
    else:
        exit()


def parse_laser_params(s):
    """Parse laser parameters in format (onset,duration),(onset,duration)"""
    if not s:
        return []
    # Find all pairs of numbers in parentheses
    pattern = re.compile(r'\((\d+),(\d+)\)')
    matches = pattern.findall(s)
    if not matches:
        raise ValueError(
            "Invalid laser parameter format. Expected format: (onset1,duration1),(onset2,duration2)")
    # Convert to integers and return as list of tuples
    return [(int(onset), int(duration)) for onset, duration in matches]


if args.programmatic:
    print('================================')
    print('Running in programmatic mode')
    print('================================')

metadata_handler = imp_metadata([[], args.dir_name])
dir_path = metadata_handler.dir_name

dir_name = os.path.basename(dir_path[:-1])

if not test_bool:
    script_path = os.path.abspath(__file__)
    this_pipeline_check = pipeline_graph_check(dir_path)
    this_pipeline_check.write_to_log(script_path, 'attempted')

# Load cache and existing info
cache = load_cache()
existing_info = load_existing_info(dir_path, dir_name)

# Display existing info if available
if existing_info and not args.programmatic:
    print("\n=== Current values from existing info file (will be used as defaults) ===")
    if 'taste_params' in existing_info:
        taste_params = existing_info['taste_params']
        print(f"Taste dig-ins: {taste_params.get('dig_in_nums', [])}")
        print(f"Tastes: {taste_params.get('tastes', [])}")
        print(f"Concentrations: {taste_params.get('concs', [])}")
        print(f"Palatability rankings: {taste_params.get('pal_rankings', [])}")

    if 'laser_params' in existing_info:
        laser_params = existing_info['laser_params']
        print(f"Laser dig-ins: {laser_params.get('dig_in_nums', [])}")
        print(f"Laser parameters: {laser_params.get('onset_duration', [])}")
        print(f"Opto locations: {laser_params.get('opto_locs', [])}")
        print(f"Virus region: {laser_params.get('virus_region', '')}")

    if 'notes' in existing_info:
        print(f"Notes: {existing_info['notes']}")
    print("===================================================================\n")

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
    "date": time_match[0],
    "timestamp": time_match[1],
}

##################################################
# Brain Regions and Electrode Layout
##################################################


if args.template:
    with open(args.template, 'r') as file:
        template_dict = json.load(file)
        template_keys = list(template_dict.keys())
        from_template = {
            this_key: template_dict[this_key] for this_key in template_keys
            if this_key not in this_dict.keys()
        }
        fin_dict = {**this_dict, **from_template}

else:

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
        with open(os.path.join(dir_path, rhd_file_list[0]), 'rb') as f:
            header = read_header(f)
        ports = [x['port_prefix'] for x in header['amplifier_channels']]
        electrode_files = [x['native_channel_name']
                           for x in header['amplifier_channels']]

    ##################################################
    # Dig-Ins
    ##################################################
    # Process dig-ins
    this_dig_handler = DigInHandler(dir_path, file_type)
    this_dig_handler.get_dig_in_files()
    dig_in_list_str = "All Dig-ins : \n" + \
        ", ".join([str(x) for x in this_dig_handler.dig_in_num])
    this_dig_handler.get_trial_data()

    def count_check(x):
        nums = re.findall('[0-9]+', x)
        # return sum([x.isdigit() for x in nums]) == len(nums)
        return all([int(x) in this_dig_handler.dig_in_frame.index for x in nums])

    # Calculate number of deliveries from recorded data
    dig_in_present_bool = any(this_dig_handler.dig_in_frame.trial_counts > 0)

    # Ask for user input of which line index the dig in came from
    if dig_in_present_bool:
        if not args.programmatic:
            # Use the helper function to get taste dig-ins
            taste_dig_in_str = populate_field_with_defaults(
                field_name='dig_in_nums',
                entry_checker_msg=' INDEX of Taste dig_ins used (IN ORDER, anything separated)',
                check_func=count_check,
                existing_info=existing_info.get('taste_params', {}),
                cache=cache,
                fail_response='Please enter numbers in index of dataframe above'
            )

            # Convert to integers
            nums = re.findall(
                '[0-9]+', taste_dig_in_str) if isinstance(taste_dig_in_str, str) else taste_dig_in_str
            taste_dig_inds = [int(x) for x in nums] if isinstance(
                nums[0], str) else nums

            # Save to cache
            cache['taste_dig_inds'] = taste_dig_inds
            save_to_cache(cache)
        else:
            if args.taste_digins:
                taste_dig_inds = parse_csv(args.taste_digins, int)
            else:
                raise ValueError(
                    'Taste dig-ins not provided, use --taste-digins')

        this_dig_handler.dig_in_frame.loc[taste_dig_inds, 'taste_bool'] = True
        this_dig_handler.dig_in_frame.taste_bool.fillna(False, inplace=True)
        print('Taste dig-in frame: \n')
        print_df = this_dig_handler.dig_in_frame.drop(columns=['pulse_times'])
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
            # Use the helper function to get tastes
            taste_str = populate_field_with_defaults(
                field_name='tastes',
                entry_checker_msg=' Tastes names used (IN ORDER, anything separated [no punctuation in name])',
                check_func=taste_check,
                existing_info=existing_info.get('taste_params', {}),
                cache=cache,
                fail_response=f'Please enter as many ({len(taste_dig_inds)}) tastes as digins'
            )

            # Extract taste names
            tastes = re.findall(
                '[A-Za-z]+', taste_str) if isinstance(taste_str, str) else taste_str

            # Save to cache
            cache['tastes'] = tastes
            save_to_cache(cache)
        else:
            if args.tastes:
                tastes = parse_csv(args.tastes)
            else:
                raise ValueError('Tastes not provided, use --tastes')

        this_dig_handler.dig_in_frame.loc[taste_dig_inds, 'taste'] = tastes
        print('Taste dig-in frame: \n')
        print_df = this_dig_handler.dig_in_frame.drop(columns=['pulse_times'])
        print_df = print_df[print_df.taste_bool]
        print(print_df)

        if not args.programmatic:
            # Use the helper function to get concentrations
            def convert_concs(input_str):
                return [float(x) for x in input_str.split(",")]

            conc_str = populate_field_with_defaults(
                field_name='concs',
                entry_checker_msg='Corresponding concs used (in M, IN ORDER, COMMA separated)',
                check_func=float_check,
                existing_info=existing_info.get('taste_params', {}),
                cache=cache,
                convert_func=convert_concs,
                fail_response=f'Please enter as many ({len(taste_dig_inds)}) concentrations as digins'
            )

            # If we got a string, convert it, otherwise use as is
            concs = convert_concs(conc_str) if isinstance(
                conc_str, str) else conc_str

            # Save to cache
            cache['concs'] = concs
            save_to_cache(cache)
        else:
            if args.concentrations:
                concs = parse_csv(args.concentrations, float)
            else:
                raise ValueError(
                    'Concentrations not provided, use --concentrations')

        this_dig_handler.dig_in_frame.loc[taste_dig_inds,
                                          'concentration'] = concs
        print('Taste dig-in frame: \n')
        print_df = this_dig_handler.dig_in_frame.drop(columns=['pulse_times'])
        print_df = print_df[print_df.taste_bool]
        print(print_df)

        # Ask user for palatability rankings
        if not args.programmatic:
            # Use the helper function to get palatability rankings
            def convert_pal_ranks(input_str):
                nums = re.findall('[1-9]+', input_str)
                return [int(x) for x in nums]

            palatability_str = populate_field_with_defaults(
                field_name='pal_rankings',
                entry_checker_msg='Enter palatability rankings (IN ORDER) used (anything separated), higher number = more palatable',
                check_func=pal_check,
                existing_info=existing_info.get('taste_params', {}),
                cache=cache,
                convert_func=convert_pal_ranks,
                fail_response=f'Please enter numbers 1<=x<={len(print_df)}'
            )

            # If we got a string, convert it, otherwise use as is
            pal_ranks = convert_pal_ranks(palatability_str) if isinstance(
                palatability_str, str) else palatability_str

            # Save to cache
            cache['pal_ranks'] = pal_ranks
            save_to_cache(cache)
        else:
            if args.palatability:
                pal_ranks = parse_csv(args.palatability, int)
            else:
                raise ValueError(
                    'Palatability rankings not provided, use --palatability')

        this_dig_handler.dig_in_frame.loc[taste_dig_inds,
                                          'palatability'] = pal_ranks
        print('Taste dig-in frame: \n')
        print_df = this_dig_handler.dig_in_frame.drop(columns=['pulse_times'])
        print_df = print_df[print_df.taste_bool]
        print(print_df)

        taste_digin_nums = this_dig_handler.dig_in_frame.loc[taste_dig_inds, 'dig_in_nums'].to_list(
        )
        tastes = this_dig_handler.dig_in_frame.loc[taste_dig_inds, 'taste'].to_list(
        )
        concs = this_dig_handler.dig_in_frame.loc[taste_dig_inds, 'concentration'].to_list(
        )
        pal_ranks = this_dig_handler.dig_in_frame.loc[taste_dig_inds, 'palatability'].to_list(
        )
        taste_digin_trials = this_dig_handler.dig_in_frame.loc[taste_dig_inds, 'trial_counts'].to_list(
        )
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
    if not args.programmatic:
        # Get defaults from existing info for laser dig-ins
        default_laser_digin_ind = []
        if 'laser_params' in existing_info and existing_info['laser_params'].get('dig_in_nums'):
            # Convert to indices from dig_in_nums
            laser_nums = existing_info['laser_params']['dig_in_nums']
            if laser_nums:
                for num in laser_nums:
                    matching_indices = this_dig_handler.dig_in_frame[this_dig_handler.dig_in_frame.dig_in_nums == num].index.tolist(
                    )
                    if matching_indices:
                        default_laser_digin_ind.extend(matching_indices)

        # Custom conversion function for laser dig-ins
        def convert_laser_digin(input_str):
            if len(input_str) == 0:
                return []
            return [int(input_str)]

        # Use helper function with special handling for blank input
        laser_select_str = populate_field_with_defaults(
            field_name='laser_digin_ind',
            entry_checker_msg='Laser dig_in index, <BLANK> for none',
            check_func=count_check,
            existing_info={'laser_digin_ind': default_laser_digin_ind},
            cache=cache,
            convert_func=convert_laser_digin,
            fail_response='Please enter numbers in index of dataframe above'
        )

        # Handle the special case for laser dig-ins
        if isinstance(laser_select_str, str):
            if len(laser_select_str) == 0:
                laser_digin_ind = default_laser_digin_ind if default_laser_digin_ind else []
            else:
                laser_digin_ind = [int(laser_select_str)]
        else:
            laser_digin_ind = laser_select_str

        # Save to cache
        cache['laser_digin_ind'] = laser_digin_ind
        save_to_cache(cache)
    else:
        if args.laser_digin:
            laser_digin_ind = parse_csv(args.laser_digin, int)
        else:
            laser_digin_ind = []

    if laser_digin_ind:
        laser_digin_nums = this_dig_handler.dig_in_frame.loc[laser_digin_ind, 'dig_in_nums'].to_list(
        )
        this_dig_handler.dig_in_frame.loc[laser_digin_ind, 'laser_bool'] = True
        this_dig_handler.dig_in_frame.laser_bool.fillna(False, inplace=True)
        laser_digin_nums = this_dig_handler.dig_in_frame.loc[laser_digin_ind, 'dig_in_nums'].to_list(
        )
        print('Selected laser digins: \n')
        print_df = this_dig_handler.dig_in_frame.drop(columns=['pulse_times'])
        print_df = print_df[print_df.laser_bool]
        print(print_df)
    else:
        laser_digin_nums = []

    def laser_check(x):
        nums = re.findall('[0-9]+', x)
        return sum([x.isdigit() for x in nums]) == 2

    if laser_digin_ind:
        if not args.programmatic:
            # Get defaults from existing info or cache
            default_laser_params_list = []
            default_opto_loc_list = []
            default_virus_region = ""

            if 'laser_params' in existing_info:
                if 'onset_duration' in existing_info['laser_params']:
                    default_laser_params_list = existing_info['laser_params']['onset_duration']
                if 'opto_locs' in existing_info['laser_params']:
                    default_opto_loc_list = existing_info['laser_params']['opto_locs']
                if 'virus_region' in existing_info['laser_params']:
                    default_virus_region = existing_info['laser_params']['virus_region']
            elif 'laser_params_list' in cache:
                default_laser_params_list = cache['laser_params_list']
                default_opto_loc_list = cache.get('opto_loc_list', [])
                default_virus_region = cache.get('virus_region_str', '')

            # Display defaults
            if default_laser_params_list:
                print(f"Default laser parameters: {default_laser_params_list}")
            if default_opto_loc_list:
                print(f"Default opto locations: {default_opto_loc_list}")
            if default_virus_region:
                print(f"Default virus region: {default_virus_region}")

            # Ask if user wants to use defaults
            use_defaults = False
            if default_laser_params_list:
                use_defaults_str, continue_bool = entry_checker(
                    msg='Use default laser parameters? (y/n) :: ',
                    check_func=lambda x: x.lower() in ['y', 'n', 'yes', 'no'],
                    fail_response='Please enter y or n')
                if continue_bool:
                    use_defaults = use_defaults_str.lower() in ['y', 'yes']
                else:
                    exit()

            if use_defaults:
                laser_params_list = default_laser_params_list
                opto_loc_list = default_opto_loc_list
                virus_region_str = default_virus_region
            else:
                # Ask for laser parameters - allow multiple entries
                laser_params_list = []
                opto_loc_list = []

                while True:
                    # Ask for laser parameters
                    default_param_str = ""
                    if default_laser_params_list and len(laser_params_list) < len(default_laser_params_list):
                        default_param = default_laser_params_list[len(
                            laser_params_list)]
                        default_param_str = f" [{default_param[0]}, {default_param[1]}]"

                    laser_select_str, continue_bool = entry_checker(
                        msg=f'Laser onset_time, duration (ms, IN ORDER, anything separated){default_param_str} or "done" to finish :: ',
                        check_func=lambda x: laser_check(
                            x) or x.lower() == 'done' or x.strip() == '',
                        fail_response='Please enter two valid integers, press Enter for default, or type "done"')

                    if laser_select_str.lower() == 'done':
                        break

                    if continue_bool:
                        if laser_select_str.strip() == '' and default_laser_params_list and len(laser_params_list) < len(default_laser_params_list):
                            # Use default if input is empty
                            laser_params_list.append(
                                default_laser_params_list[len(laser_params_list)])
                        else:
                            nums = re.findall('[0-9]+', laser_select_str)
                            onset_time, duration = [int(x) for x in nums]
                            laser_params_list.append((onset_time, duration))
                    else:
                        exit()

                # Ask for opto-fiber location for this condition
                def opto_loc_check(x):
                    return len(re.findall('[A-Za-z]+', x)) == len(laser_params_list) or x.strip() == ''

                print(f'Parsed laser parameters: {laser_params_list}')

                default_opto_str = ', '.join(
                    default_opto_loc_list) if default_opto_loc_list else ""

                opto_loc_entry, continue_bool = entry_checker(
                    msg=f'Enter ({len(laser_params_list)}) opto-fiber locations for this condition [{default_opto_str}] :: ',
                    check_func=opto_loc_check,
                    fail_response='Please enter a valid location or press Enter for default')
                if continue_bool:
                    if opto_loc_entry.strip() == '' and default_opto_loc_list:
                        opto_loc_list = default_opto_loc_list
                    else:
                        opto_loc_list = re.findall('[A-Za-z]+', opto_loc_entry)
                else:
                    exit()

                # If no entries were made, exit
                if not laser_params_list:
                    print("No laser parameters entered.")
                    exit()

                # Ask for virus region (common for all conditions)
                virus_region_str, continue_bool = entry_checker(
                    msg=f'Enter virus region [{default_virus_region}] :: ',
                    check_func=lambda x: True,
                    fail_response='Please enter a valid region')
                if continue_bool:
                    if virus_region_str.strip() == '':
                        virus_region_str = default_virus_region
                else:
                    exit()

            # Save to cache
            cache['laser_params_list'] = laser_params_list
            cache['opto_loc_list'] = opto_loc_list
            cache['virus_region_str'] = virus_region_str
            save_to_cache(cache)

        else:
            # Programmatic mode
            if args.laser_params:
                print('Parsing laser parameters')
                laser_params_list = parse_laser_params(args.laser_params)
                if not laser_params_list:
                    raise ValueError(
                        'Invalid laser parameters format. Use format: (onset1,duration1),(onset2,duration2)')
            else:
                raise ValueError(
                    'Laser parameters not provided, use --laser-params')

            if args.virus_region:
                print('Parsing virus region')
                virus_region_str = args.virus_region
            else:
                raise ValueError(
                    'Virus region not provided, use --virus-region')

            if args.opto_loc:
                print('Parsing opto-fiber locations')
                opto_loc_list = parse_csv(args.opto_loc)
                if len(opto_loc_list) != len(laser_params_list):
                    raise ValueError(
                        f'Number of opto locations ({len(opto_loc_list)}) must match number of laser parameter pairs ({len(laser_params_list)})')
            else:
                raise ValueError(
                    'Opto-fiber locations not provided, use --opto-loc')

        # Fill in laser parameters - store as list of parameter pairs
        this_dig_handler.dig_in_frame.loc[laser_digin_ind, 'laser_params'] = str(
            laser_params_list)
    else:
        laser_params_list = []
        virus_region_str = ''
        opto_loc_list = []

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
            layout_frame_filled['CAR_group'] = [x.strip().lower()
                                                for x in car_groups]
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
                raise ValueError(
                    'EMG muscle name not provided, use --emg-muscle')
    else:
        fin_emg_port = []
        orig_emg_electrodes = []
        emg_muscle_str = ''

    fin_perm = layout_dict

    ########################################
    # Finalize dictionary
    ########################################

    if not args.programmatic:
        # Use a simpler approach for notes since we're using input() directly
        default_notes = existing_info.get(
            'notes', '') or cache.get('notes', '')
        notes = input(
            f'Please enter any notes about the experiment [{default_notes}]. \n :: ')
        if notes.strip() == '':
            notes = default_notes

        # Save to cache
        cache['notes'] = notes
        save_to_cache(cache)
    else:
        notes = args.notes or existing_info.get('notes', '') or ''

    if laser_digin_ind:
        laser_digin_trials = this_dig_handler.dig_in_frame.loc[laser_digin_ind, 'trial_counts'].to_list(
        )
    else:
        laser_digin_trials = []

    fin_dict = {'version': '0.0.3',
                **this_dict,
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
                    'muscle': emg_muscle_str},
                'electrode_layout': fin_perm,
                'taste_params': {
                    'dig_in_nums': taste_digin_nums,
                    'trial_count': taste_digin_trials,
                    'tastes': tastes,
                    'concs': concs,
                    'pal_rankings': pal_ranks},
                'laser_params': {
                    'dig_in_nums': laser_digin_nums,
                    'trial_count': laser_digin_trials,
                    'onset_duration': laser_params_list,
                    'opto_locs': opto_loc_list,
                    'virus_region': virus_region_str},
                'notes': notes}


json_file_name = os.path.join(dir_path, '.'.join([dir_name, 'info']))
with open(json_file_name, 'w') as file:
    json.dump(fin_dict, file, indent=4)

# Write success to log
this_pipeline_check.write_to_log(script_path, 'completed')
