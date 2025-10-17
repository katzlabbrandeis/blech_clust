"""
This module generates a file containing relevant experimental information for a given dataset.

It processes data files to extract and organize details such as:
- Animal name, experiment type, date, timestamp
- Regions recorded from and electrode layout
- Taste concentrations and palatability ranks
- Laser parameters
- Miscellaneous notes

Key functionality:
- Parses command-line arguments to specify the directory containing data files
- Extracts metadata from the directory name and checks the pipeline status
- Processes digital input (dig-in) data to determine taste dig-ins, concentrations,
  palatability rankings, and laser parameters
- Handles different file types for electrode data and generates or uses an existing electrode layout file
- Organizes and writes out the final experimental information into a JSON file
- Logs the completion status of the pipeline process
- Caches manual entries to reduce redundant input across sessions
- Auto-populates defaults from existing info files when available

The module supports two modes of operation:
1. Programmatic mode: All parameters are provided via command-line arguments
2. Manual mode: User is prompted for input with sensible defaults
"""

# Standard library imports
import argparse
import json
import os
import re

# Third-party imports
import numpy as np
import pandas as pd
from tqdm import tqdm

# Local imports
from utils.blech_utils import (
    entry_checker,
    imp_metadata,
    pipeline_graph_check,
)
from utils.importrhdutilities import load_file, read_header
from utils.read_file import DigInHandler

# Constants
test_bool = False  # noqa


def parse_arguments():
    """
    Parse command line arguments for the experiment info generator.

    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    if test_bool:
        return argparse.Namespace(
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
            notes=None,
            auto_defaults=False
        )
    else:
        # Create argument parser
        parser = argparse.ArgumentParser(
            description='Creates files with experiment info')

        # Basic arguments
        parser.add_argument(
            'dir_name',  help='Directory containing data files')
        parser.add_argument('--template', '-t',
                            help='Template (.info) file to copy experimental details from')
        parser.add_argument('--mode', '-m', default='legacy',
                            choices=['legacy', 'updated'])

        # Mode selection
        parser.add_argument('--programmatic', action='store_true',
                            help='Run in programmatic mode (all parameters must be provided via command line)')
        parser.add_argument('--auto-defaults', action='store_true',
                            help='Use auto defaults for all fields if available (in manual mode)')
        parser.add_argument('--use-layout-file', action='store_true',
                            help='Use existing electrode layout file')

        # Programmatic mode parameters - Electrode layout
        parser.add_argument(
            '--car-groups', help='Comma-separated CAR groupings')
        parser.add_argument('--emg-muscle', help='Name of EMG muscle')

        # Programmatic mode parameters - Taste information
        parser.add_argument(
            '--taste-digins', help='Comma-separated indices of taste digital inputs')
        parser.add_argument('--tastes', help='Comma-separated taste names')
        parser.add_argument('--concentrations',
                            help='Comma-separated concentrations in M')
        parser.add_argument(
            '--palatability', help='Comma-separated palatability rankings')

        # Programmatic mode parameters - Laser information
        parser.add_argument('--laser-digin', help='Laser digital input index')
        parser.add_argument(
            '--laser-params', help='Multiple laser parameters as (onset,duration) pairs in ms, comma-separated: (100,500),(200,300)')
        parser.add_argument('--virus-region', help='Virus region')
        parser.add_argument(
            '--opto-loc', help='Multiple opto-fiber locations, comma-separated (must match number of laser parameter pairs)')

        # Additional information
        parser.add_argument('--notes', help='Experiment notes')

        return parser.parse_args()


# Parse command line arguments
args = parse_arguments()


def setup_cache_directory():
    """Set up the cache directory and return the cache file path."""
    script_path = os.path.abspath(__file__) if not test_bool else "test_path"
    blech_clust_dir = os.path.dirname(script_path)
    cache_dir = os.path.join(blech_clust_dir, 'cache_and_logs')
    os.makedirs(cache_dir, exist_ok=True)

    # Define the cache file path
    cache_file_path = os.path.join(cache_dir, 'manual_entries_cache.json')
    print(f"Cache file path: {cache_file_path}")
    return cache_file_path


def load_cache(cache_file_path):
    """Load the cache of manual entries if it exists."""
    if os.path.exists(cache_file_path):
        try:
            with open(cache_file_path, 'r') as cache_file:
                return json.load(cache_file)
        except json.JSONDecodeError:
            print("Warning: Cache file exists but is not valid JSON. Creating new cache.")
            return {}
    return {}


def save_to_cache(cache_dict, cache_file_path):
    """Save the updated cache to the cache file."""
    with open(cache_file_path, 'w') as cache_file:
        json.dump(cache_dict, cache_file, indent=4)


def load_existing_info(dir_path, dir_name):
    """Load existing info file if it exists."""
    info_file_path = os.path.join(dir_path, f"{dir_name}.info")
    if os.path.exists(info_file_path):
        try:
            with open(info_file_path, 'r') as info_file:
                return json.load(info_file)
        except json.JSONDecodeError:
            print("Warning: Info file exists but is not valid JSON.")
            return {}
    return {}


def parse_csv(s, convert=str):
    """
    Parse comma-separated values from a string and convert them to a specified type.

    Args:
        s: String containing comma-separated values
        convert: Function to convert each value (default: str)

    Returns:
        List of converted values
    """
    if not s:
        return []
    return [convert(x.strip()) for x in s.split(',')]


def get_default_value(field_name, existing_info, cache, nested_field=None, default_value_override=None):
    """
    Get the default value for a field from existing info or cache.

    Args:
        field_name: Name of the field
        existing_info: Dictionary containing existing information
        cache: Dictionary containing cached values
        nested_field: If field is nested, provide the parent key
        default_value_override: Override the default value

    Returns:
        The default value for the field
    """
    if default_value_override is not None:
        return default_value_override

    # Check existing info first
    if nested_field and nested_field in existing_info:
        if field_name in existing_info[nested_field]:
            return existing_info[nested_field][field_name]
    elif field_name in existing_info:
        return existing_info[field_name]

    # Then check cache
    if nested_field and nested_field in cache:
        if field_name in cache[nested_field]:
            return cache[nested_field][field_name]
    elif field_name in cache:
        return cache[field_name]

    # Return empty list as default
    return []


def format_default_for_display(default_value):
    """Format default value for display in prompt."""
    if isinstance(default_value, list):
        return ', '.join(map(str, default_value)) if default_value else ""
    else:
        return str(default_value) if default_value else ""


def populate_field_with_defaults(
        field_name,
        entry_checker_msg,
        check_func,
        existing_info,
        cache,
        convert_func=None,
        fail_response=None,
        nested_field=None,
        default_value_override=None,
        force_default=False
):
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
        default_value_override: Override the default value
        force_default: Whether to use the default value without prompting

    Returns:
        The value to use for the field
    """
    # Get default value
    default_value = get_default_value(
        field_name, existing_info, cache, nested_field, default_value_override
    )

    # Format default for display
    default_str = format_default_for_display(default_value)

    if fail_response is None:
        fail_response = f'Please enter valid input for {field_name}'

    if force_default:
        user_input = default_str
        continue_bool = True
    else:
        # Prompt user
        user_input, continue_bool = entry_checker(
            msg=f'{entry_checker_msg}\nDefault values: [{default_str}] :: ',
            check_func=check_func,
            fail_response=fail_response,
            default_input=default_str,
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
    """
    Parse laser parameters in format (onset,duration),(onset,duration).

    Args:
        s: String containing laser parameters

    Returns:
        List of tuples (onset, duration)
    """
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


def extract_metadata_from_dir_name(dir_name):
    """
    Extract metadata such as name, experiment type, date, and timestamp from directory name.

    Args:
        dir_name: Name of the directory

    Returns:
        Dictionary containing extracted metadata
    """
    splits = dir_name.split("_")
    # Date and Timestamp are given as 2 sets of 6 digits
    # Extract using regex
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


def display_existing_info(existing_info):
    """Display existing info values that will be used as defaults."""
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


def process_dig_ins_programmatic(this_dig_handler, args):
    """
    Process dig-ins in programmatic mode where all parameters are provided via command-line arguments.

    This function handles the automated processing of digital inputs for taste information
    without requiring user interaction.

    Args:
        this_dig_handler: DigInHandler instance
        args: Command line arguments containing taste-related parameters

    Returns:
        Tuple containing taste_dig_inds, tastes, concs, pal_ranks, taste_digin_nums, taste_digin_trials
    """
    this_dig_handler.get_dig_in_files()
    this_dig_handler.get_trial_data()

    dig_in_present_bool = any(this_dig_handler.dig_in_frame.trial_counts > 0)

    if not dig_in_present_bool:
        print('No dig-ins found. Please check your data.')
        return [], [], [], [], [], []

    # Process taste dig-ins
    if args.taste_digins:
        taste_dig_inds = parse_csv(args.taste_digins, int)
    else:
        raise ValueError('Taste dig-ins not provided, use --taste-digins')

    this_dig_handler.dig_in_frame.loc[taste_dig_inds, 'taste_bool'] = True
    this_dig_handler.dig_in_frame.taste_bool.fillna(False, inplace=True)
    print('Taste dig-in frame: \n')
    print_df = this_dig_handler.dig_in_frame.drop(columns=['pulse_times'])
    print_df = print_df[print_df.taste_bool]
    print(print_df)

    # Process tastes
    if args.tastes:
        tastes = parse_csv(args.tastes)
    else:
        raise ValueError('Tastes not provided, use --tastes')

    this_dig_handler.dig_in_frame.loc[taste_dig_inds, 'taste'] = tastes
    print('Taste dig-in frame: \n')
    print_df = this_dig_handler.dig_in_frame.drop(columns=['pulse_times'])
    print_df = print_df[print_df.taste_bool]
    print(print_df)

    # Process concentrations
    if args.concentrations:
        concs = parse_csv(args.concentrations, float)
    else:
        raise ValueError('Concentrations not provided, use --concentrations')

    this_dig_handler.dig_in_frame.loc[taste_dig_inds, 'concentration'] = concs
    print('Taste dig-in frame: \n')
    print_df = this_dig_handler.dig_in_frame.drop(columns=['pulse_times'])
    print_df = print_df[print_df.taste_bool]
    print(print_df)

    # Process palatability rankings
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
    taste_digin_trials = this_dig_handler.dig_in_frame.loc[taste_dig_inds, 'trial_counts'].to_list(
    )

    return taste_dig_inds, tastes, concs, pal_ranks, taste_digin_nums, taste_digin_trials


def process_dig_ins_manual(this_dig_handler, args, existing_info, cache, cache_file_path):
    """
    Process dig-ins in manual mode with user interaction.

    This function guides the user through the process of selecting and configuring
    taste digital inputs, with intelligent defaults from existing info or cache.
    It handles user prompts, validation, and saving preferences to cache.

    Args:
        this_dig_handler: DigInHandler instance
        args: Command line arguments
        existing_info: Dictionary containing existing information from previous runs
        cache: Dictionary containing cached values from previous sessions
        cache_file_path: Path to cache file for saving user preferences

    Returns:
        Tuple containing taste_dig_inds, tastes, concs, pal_ranks, taste_digin_nums, taste_digin_trials
    """
    this_dig_handler.get_dig_in_files()
    dig_in_list_str = "All Dig-ins : \n" + \
        ", ".join([str(x) for x in this_dig_handler.dig_in_num])
    this_dig_handler.get_trial_data()

    def count_check(x):
        nums = re.findall('[0-9]+', x)
        return all([int(x) in this_dig_handler.dig_in_frame.index for x in nums])

    dig_in_present_bool = any(this_dig_handler.dig_in_frame.trial_counts > 0)

    if not dig_in_present_bool:
        print('No dig-ins found. Please check your data.')
        return [], [], [], [], [], []

    # Get taste dig-ins
    if existing_info:
        existing_dig_in_nums = existing_info.get(
            'taste_params', {}).get('dig_in_nums', [])
        existing_dig_in_nums = this_dig_handler.dig_in_frame[
            this_dig_handler.dig_in_frame.dig_in_nums.isin(
                existing_dig_in_nums)
        ].index.tolist()
    else:
        existing_dig_in_nums = None

    taste_dig_in_str = populate_field_with_defaults(
        field_name='dig_in_nums',
        nested_field='taste_params',
        entry_checker_msg=' INDEX of Taste dig_ins used (IN ORDER, anything separated)',
        check_func=count_check,
        existing_info=existing_info,
        cache=cache,
        fail_response='Please enter numbers in index of dataframe above',
        default_value_override=existing_dig_in_nums,
        force_default=args.auto_defaults
    )

    nums = re.findall(
        '[0-9]+', taste_dig_in_str) if isinstance(taste_dig_in_str, str) else taste_dig_in_str
    taste_dig_inds = [int(x)
                      for x in nums] if isinstance(nums[0], str) else nums

    if 'taste_params' not in cache:
        cache['taste_params'] = {}
    cache['taste_params']['dig_in_nums'] = taste_dig_inds
    save_to_cache(cache, cache_file_path)

    this_dig_handler.dig_in_frame.loc[taste_dig_inds, 'taste_bool'] = True
    this_dig_handler.dig_in_frame.taste_bool.fillna(False, inplace=True)
    print('Taste dig-in frame: \n')
    print_df = this_dig_handler.dig_in_frame.drop(columns=['pulse_times'])
    print_df = print_df[print_df.taste_bool]
    print(print_df)

    def taste_check(x):
        return len(re.findall('[A-Za-z]+', x)) == len(taste_dig_inds)

    # Get tastes
    taste_str = populate_field_with_defaults(
        field_name='tastes',
        nested_field='taste_params',
        entry_checker_msg=' Tastes names used (IN ORDER, anything separated [no punctuation in name])',
        check_func=taste_check,
        existing_info=existing_info,
        cache=cache,
        fail_response=f'Please enter as many ({len(taste_dig_inds)}) tastes as digins',
        force_default=args.auto_defaults
    )

    tastes = re.findall(
        '[A-Za-z]+', taste_str) if isinstance(taste_str, str) else taste_str
    cache['taste_params']['tastes'] = tastes
    save_to_cache(cache, cache_file_path)

    this_dig_handler.dig_in_frame.loc[taste_dig_inds, 'taste'] = tastes
    print('Taste dig-in frame: \n')
    print_df = this_dig_handler.dig_in_frame.drop(columns=['pulse_times'])
    print_df = print_df[print_df.taste_bool]
    print(print_df)

    def float_check(x):
        return len(x.split(',')) == len(taste_dig_inds)

    # Get concentrations
    def convert_concs(input_str):
        return [float(x) for x in input_str.split(",")]

    conc_str = populate_field_with_defaults(
        field_name='concs',
        nested_field='taste_params',
        entry_checker_msg='Corresponding concs used (in M, IN ORDER, COMMA separated)',
        check_func=float_check,
        existing_info=existing_info,
        cache=cache,
        convert_func=convert_concs,
        fail_response=f'Please enter as many ({len(taste_dig_inds)}) concentrations as digins',
        force_default=args.auto_defaults
    )

    concs = convert_concs(conc_str) if isinstance(conc_str, str) else conc_str
    cache['taste_params']['concs'] = concs
    save_to_cache(cache, cache_file_path)

    this_dig_handler.dig_in_frame.loc[taste_dig_inds, 'concentration'] = concs
    print('Taste dig-in frame: \n')
    print_df = this_dig_handler.dig_in_frame.drop(columns=['pulse_times'])
    print_df = print_df[print_df.taste_bool]
    print(print_df)

    def pal_check(x):
        nums = re.findall('[1-9]+', x)
        if not nums:
            return False
        pal_nums = [int(n) for n in nums]
        return all(1 <= p <= len(tastes) for p in pal_nums) and len(pal_nums) == len(tastes)

    # Get palatability rankings
    def convert_pal_ranks(input_str):
        nums = re.findall('[1-9]+', input_str)
        return [int(x) for x in nums]

    palatability_str = populate_field_with_defaults(
        field_name='pal_rankings',
        nested_field='taste_params',
        entry_checker_msg='Enter palatability rankings (IN ORDER) used (anything separated), higher number = more palatable',
        check_func=pal_check,
        existing_info=existing_info,
        cache=cache,
        convert_func=convert_pal_ranks,
        fail_response=f'Please enter numbers 1<=x<={len(print_df)}',
        force_default=args.auto_defaults
    )

    pal_ranks = convert_pal_ranks(palatability_str) if isinstance(
        palatability_str, str) else palatability_str
    cache['taste_params']['pal_rankings'] = pal_ranks
    save_to_cache(cache, cache_file_path)

    this_dig_handler.dig_in_frame.loc[taste_dig_inds,
                                      'palatability'] = pal_ranks
    print('Taste dig-in frame: \n')
    print_df = this_dig_handler.dig_in_frame.drop(columns=['pulse_times'])
    print_df = print_df[print_df.taste_bool]
    print(print_df)

    taste_digin_nums = this_dig_handler.dig_in_frame.loc[taste_dig_inds, 'dig_in_nums'].to_list(
    )
    taste_digin_trials = this_dig_handler.dig_in_frame.loc[taste_dig_inds, 'trial_counts'].to_list(
    )

    return taste_dig_inds, tastes, concs, pal_ranks, taste_digin_nums, taste_digin_trials


def setup_experiment_info():
    """
    Set up the experiment info generation process.

    This function initializes the environment for generating experiment info:
    - Determines the operating mode (programmatic or manual)
    - Sets up metadata and paths
    - Initializes pipeline checking
    - Sets up caching for user preferences
    - Loads existing information if available
    - Extracts metadata from directory name

    Returns:
        Tuple containing setup information needed for the rest of the process
    """
    # Determine operating mode
    if args.programmatic:
        print('================================')
        print('Running in programmatic mode')
        print('All parameters will be taken from command line arguments')
        print('================================')
    else:
        print('================================')
        print('Running in manual mode')
        print('You will be prompted for input with sensible defaults')
        print('================================')

    # Set up metadata and paths
    metadata_handler = imp_metadata([[], args.dir_name])
    dir_path = metadata_handler.dir_name
    dir_name = os.path.basename(dir_path[:-1])

    if not test_bool:
        script_path = os.path.abspath(__file__)
        this_pipeline_check = pipeline_graph_check(dir_path)
        this_pipeline_check.write_to_log(script_path, 'attempted')

    # Set up cache
    cache_file_path = setup_cache_directory()

    # Load cache and existing info
    cache = load_cache(cache_file_path)
    existing_info = load_existing_info(dir_path, dir_name)

    # Display existing info if available
    if existing_info and not args.programmatic:
        display_existing_info(existing_info)

    # Extract metadata from directory name
    metadata_dict = extract_metadata_from_dir_name(dir_name)

    return dir_path, dir_name, cache_file_path, cache, existing_info, metadata_dict, this_pipeline_check if not test_bool else None


def process_laser_params_programmatic(this_dig_handler, args):
    """
    Process laser parameters in programmatic mode where all parameters are provided via command-line arguments.

    This function handles the automated processing of digital inputs for laser stimulation
    without requiring user interaction.

    Args:
        this_dig_handler: DigInHandler instance
        args: Command line arguments containing laser-related parameters

    Returns:
        Tuple containing laser_digin_ind, laser_digin_nums, laser_params_list, virus_region_str, opto_loc_list
    """
    # Process laser dig-ins
    if args.laser_digin:
        laser_digin_ind = parse_csv(args.laser_digin, int)
    else:
        laser_digin_ind = []

    if not laser_digin_ind:
        return [], [], [], "", []

    laser_digin_nums = this_dig_handler.dig_in_frame.loc[laser_digin_ind, 'dig_in_nums'].to_list(
    )
    this_dig_handler.dig_in_frame.loc[laser_digin_ind, 'laser_bool'] = True
    this_dig_handler.dig_in_frame.laser_bool.fillna(False, inplace=True)

    print('Selected laser digins: \n')
    print_df = this_dig_handler.dig_in_frame.drop(columns=['pulse_times'])
    print_df = print_df[print_df.laser_bool]
    print(print_df)

    # Process laser parameters
    if args.laser_params:
        print('Parsing laser parameters')
        laser_params_list = parse_laser_params(args.laser_params)
        if not laser_params_list:
            raise ValueError(
                'Invalid laser parameters format. Use format: (onset1,duration1),(onset2,duration2)')
    else:
        raise ValueError('Laser parameters not provided, use --laser-params')

    # Process virus region
    if args.virus_region:
        print('Parsing virus region')
        virus_region_str = args.virus_region
    else:
        raise ValueError('Virus region not provided, use --virus-region')

    # Process opto-fiber locations
    if args.opto_loc:
        print('Parsing opto-fiber locations')
        opto_loc_list = parse_csv(args.opto_loc)
        if len(opto_loc_list) != len(laser_params_list):
            raise ValueError(
                f'Number of opto locations ({len(opto_loc_list)}) must match number of laser parameter pairs ({len(laser_params_list)})')
    else:
        raise ValueError('Opto-fiber locations not provided, use --opto-loc')

    # Fill in laser parameters
    this_dig_handler.dig_in_frame.loc[laser_digin_ind, 'laser_params'] = str(
        laser_params_list)

    return laser_digin_ind, laser_digin_nums, laser_params_list, virus_region_str, opto_loc_list


def process_notes(args, existing_info, cache, cache_file_path):
    """
    Process experiment notes.

    This function handles the collection of experiment notes, either from
    command-line arguments in programmatic mode or via user input in manual mode.

    Args:
        args: Command line arguments
        existing_info: Dictionary containing existing information
        cache: Dictionary containing cached values
        cache_file_path: Path to cache file

    Returns:
        String containing experiment notes
    """
    if not args.programmatic:
        # Use a simpler approach for notes since we're using input() directly
        default_notes = existing_info.get(
            'notes', '') or cache.get('notes', '')
        if args.auto_defaults:
            notes = default_notes
        else:
            notes = input(
                f'Please enter any notes about the experiment [Default: {default_notes}]. \n :: ')
            if notes.strip() == '':
                notes = default_notes

        # Save to cache
        cache['notes'] = notes
        save_to_cache(cache, cache_file_path)
    else:
        notes = args.notes or existing_info.get('notes', '') or ''

    return notes


def process_permanent_path(dir_path, dir_name, args, existing_info, cache, cache_file_path):
    """
    Process permanent path for metadata backup.

    This function handles the collection of a permanent path where metadata files
    should be copied. It validates that the data exists at the specified location.

    Args:
        dir_path: Path to the current data directory
        dir_name: Name of the directory
        args: Command line arguments
        existing_info: Dictionary containing existing information
        cache: Dictionary containing cached values
        cache_file_path: Path to cache file

    Returns:
        String containing the permanent path, or None if skipped
    """
    if args.programmatic:
        # In programmatic mode, skip permanent path prompt
        return None

    # Get default from existing info or cache
    default_path = existing_info.get('permanent_path', '') or cache.get('permanent_path', '')

    if args.auto_defaults and default_path:
        permanent_path = default_path
    else:
        print("\n=== Permanent Metadata Copy ===")
        print("Please specify where the permanent copy of the data is stored.")
        print("Metadata files will be copied to this location.")
        permanent_path = input(
            f'Enter permanent data path [Default: {default_path}] (or press ENTER to skip): ')
        if permanent_path.strip() == '':
            if default_path:
                permanent_path = default_path
            else:
                print("Skipping permanent metadata copy.")
                return None

    # Validate the path
    permanent_path = os.path.expanduser(permanent_path.strip())
    
    if not os.path.exists(permanent_path):
        print(f"Warning: Path does not exist: {permanent_path}")
        response = input("Do you want to create this directory? (y/n): ")
        if response.lower() in ['y', 'yes']:
            try:
                os.makedirs(permanent_path, exist_ok=True)
                print(f"Created directory: {permanent_path}")
            except Exception as e:
                print(f"Error creating directory: {e}")
                return None
        else:
            print("Skipping permanent metadata copy.")
            return None

    # Check if data exists at the permanent location
    # Look for common data files to verify this is a valid data directory
    data_indicators = ['info.rhd', 'time.dat', 'amplifier.dat']
    has_data = any(os.path.exists(os.path.join(permanent_path, indicator)) 
                   for indicator in data_indicators)
    
    if not has_data:
        print(f"Warning: No data files found at {permanent_path}")
        print("Expected to find files like: info.rhd, time.dat, or amplifier.dat")
        response = input("Continue anyway? (y/n): ")
        if response.lower() not in ['y', 'yes']:
            print("Skipping permanent metadata copy.")
            return None

    # Save to cache
    cache['permanent_path'] = permanent_path
    save_to_cache(cache, cache_file_path)

    return permanent_path


def copy_metadata_to_permanent_location(dir_path, dir_name, permanent_path, generated_files):
    """
    Copy metadata files to permanent location.

    This function copies all generated metadata files to the permanent storage location.
    If files already exist, it prompts the user for confirmation before overwriting.

    Args:
        dir_path: Path to the current data directory
        dir_name: Name of the directory
        permanent_path: Path to permanent storage location
        generated_files: List of generated file paths to copy

    Returns:
        Boolean indicating success
    """
    import shutil

    if not permanent_path or not generated_files:
        return False

    print(f"\n=== Copying Metadata to Permanent Location ===")
    print(f"Destination: {permanent_path}")

    # Check if any files already exist
    existing_files = []
    for file_path in generated_files:
        if not os.path.exists(file_path):
            continue
        filename = os.path.basename(file_path)
        dest_path = os.path.join(permanent_path, filename)
        if os.path.exists(dest_path):
            existing_files.append(filename)

    # If files exist, ask for confirmation
    if existing_files:
        print(f"\nThe following files already exist at the destination:")
        for filename in existing_files:
            print(f"  - {filename}")
        response = input("Do you want to overwrite them? (y/n): ")
        if response.lower() not in ['y', 'yes']:
            print("Skipping metadata copy.")
            return False

    # Copy files
    copied_count = 0
    for file_path in generated_files:
        if not os.path.exists(file_path):
            print(f"Warning: File not found, skipping: {file_path}")
            continue
        
        filename = os.path.basename(file_path)
        dest_path = os.path.join(permanent_path, filename)
        
        try:
            shutil.copy2(file_path, dest_path)
            print(f"Copied: {filename}")
            copied_count += 1
        except Exception as e:
            print(f"Error copying {filename}: {e}")

    print(f"\nSuccessfully copied {copied_count} file(s) to permanent location.")
    return copied_count > 0


def process_electrode_layout(dir_path, dir_name, electrode_files, ports, electrode_num_list,
                             args, existing_info, cache, cache_file_path):
    """
    Process electrode layout file.

    This function handles the creation or use of an existing electrode layout file,
    processes CAR groups, and extracts EMG information.

    Args:
        dir_path: Path to the directory containing data files
        dir_name: Name of the directory
        electrode_files: List of electrode files
        ports: List of ports
        electrode_num_list: List of electrode numbers
        args: Command line arguments
        existing_info: Dictionary containing existing information
        cache: Dictionary containing cached values
        cache_file_path: Path to cache file

    Returns:
        Tuple containing layout_dict, fin_emg_port, orig_emg_electrodes, emg_muscle_str
    """
    layout_file_path = os.path.join(
        dir_path, dir_name + "_electrode_layout.csv")

    def yn_check(x):
        return x in ['y', 'yes', 'n', 'no', '']

    # Determine whether to use existing layout file
    use_csv_str = 'n'
    if os.path.exists(layout_file_path):
        # If neither programmatic nor use_layout_file, ask user
        if not args.programmatic and not args.use_layout_file and not args.car_groups:
            if args.auto_defaults:
                use_csv_str = 'y'
            else:
                use_csv_str, continue_bool = entry_checker(
                    msg="Layout file detected...use what's there? (y/yes/no/n) [ENTER for y] :: ",
                    check_func=yn_check,
                    fail_response='Please [y, yes, n, no]')
            if use_csv_str == '':
                use_csv_str = 'y'
        elif args.car_groups:
            use_csv_str = 'n'
        # If use_layout_file, use it
        elif args.use_layout_file:
            use_csv_str = 'y'
        # If programmatic, use existing file
        else:
            use_csv_str = 'y'

    # Create new layout file if needed
    if use_csv_str in ['n', 'no']:
        layout_frame = pd.DataFrame()
        layout_frame['filename'] = electrode_files
        layout_frame['port'] = ports
        layout_frame['electrode_num'] = electrode_num_list
        layout_frame['electrode_ind'] = layout_frame.index
        layout_frame['CAR_group'] = pd.Series()

        layout_frame = layout_frame[['filename', 'electrode_ind',
                                    'electrode_num', 'port', 'CAR_group']]

        layout_frame.to_csv(layout_file_path, index=False)

        if not args.programmatic:
            prompt_str = 'Please fill in car groups / regions' + "\n" + \
                "emg and none are case-specific" + "\n" +\
                "Indicate different CARS from same region as GC1,GC2...etc"
            print(prompt_str)

            def confirm_check(x):
                return x in ['y', 'yes']

            perm_str, continue_bool = entry_checker(
                msg='Lemme know when its done (y/yes) :: ',
                check_func=confirm_check,
                fail_response='Please say y or yes')

            if not continue_bool:
                print('Welp...')
                exit()

    # Read and process the layout file
    layout_frame_filled = pd.read_csv(layout_file_path)

    if not args.programmatic:
        layout_frame_filled['CAR_group'] = layout_frame_filled['CAR_group'].str.lower(
        )
        layout_frame_filled['CAR_group'] = [x.strip()
                                            for x in layout_frame_filled['CAR_group']]
    else:
        if args.car_groups:
            car_groups = parse_csv(args.car_groups)
            layout_frame_filled['CAR_group'] = [
                x.strip().lower() for x in car_groups]
        else:
            raise ValueError('CAR groups not provided, use --car-groups')

    # Create layout dictionary
    layout_dict = dict(
        list(layout_frame_filled.groupby('CAR_group').electrode_ind))
    for key, vals in layout_dict.items():
        layout_dict[key] = [layout_dict[key].to_list()]

    # Write out layout_frame_filled if programmatically filled
    layout_frame_filled.to_csv(layout_file_path, index=False)

    # Process EMG information
    fin_emg_port = []
    orig_emg_electrodes = []
    emg_muscle_str = ''

    if any(['emg' in x for x in layout_dict.keys()]):
        orig_emg_electrodes = [layout_dict[x][0]
                               for x in layout_dict.keys() if 'emg' in x]
        orig_emg_electrodes = [x for y in orig_emg_electrodes for x in y]
        fin_emg_port = layout_frame_filled.port.loc[
            layout_frame_filled.electrode_ind.isin(orig_emg_electrodes)].unique()
        fin_emg_port = list(fin_emg_port)

        # Get EMG muscle name
        if not args.programmatic:
            emg_muscle_str = populate_field_with_defaults(
                field_name='muscle',
                nested_field='emg',
                entry_checker_msg='Enter EMG muscle name :: ',
                check_func=lambda x: True,
                existing_info=existing_info,
                cache=cache,
                fail_response='Please enter a valid muscle name',
                force_default=args.auto_defaults
            )

            if 'emg' not in cache:
                cache['emg'] = {}
            cache['emg']['muscle'] = emg_muscle_str
            save_to_cache(cache, cache_file_path)
        else:
            if args.emg_muscle:
                emg_muscle_str = args.emg_muscle
            else:
                raise ValueError(
                    'EMG muscle name not provided, use --emg-muscle')

    return layout_dict, fin_emg_port, orig_emg_electrodes, emg_muscle_str


def process_electrode_files(file_type, electrodes_list, dir_path):
    """
    Process electrode files based on file type.

    This function handles different file formats and extracts electrode information.

    Args:
        file_type: Type of file ('one file per channel', 'one file per signal type', or 'traditional')
        electrodes_list: List of electrode files
        dir_path: Path to the directory containing data files

    Returns:
        Tuple containing electrode_files, ports, electrode_num_list
    """
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
        rhd_file_list = [x for x in os.listdir(dir_path) if 'rhd' in x]
        with open(os.path.join(dir_path, rhd_file_list[0]), 'rb') as f:
            header = read_header(f)
        ports = [x['port_prefix'] for x in header['amplifier_channels']]
        electrode_files = [x['native_channel_name']
                           for x in header['amplifier_channels']]
        electrode_num_list = [x.split('-')[1] for x in electrode_files]

    return electrode_files, ports, electrode_num_list


def process_laser_params_manual(this_dig_handler, args, existing_info, cache, cache_file_path):
    """
    Process laser parameters in manual mode with user interaction.

    This function guides the user through the process of selecting and configuring
    laser stimulation parameters, with intelligent defaults from existing info or cache.
    It handles user prompts, validation, and saving preferences to cache.

    Args:
        this_dig_handler: DigInHandler instance
        args: Command line arguments
        existing_info: Dictionary containing existing information from previous runs
        cache: Dictionary containing cached values from previous sessions
        cache_file_path: Path to cache file for saving user preferences

    Returns:
        Tuple containing laser_digin_ind, laser_digin_nums, laser_params_list, virus_region_str, opto_loc_list
    """
    def count_check(x):
        nums = re.findall('[0-9]+', x)
        return all([int(x) in this_dig_handler.dig_in_frame.index for x in nums])

    # Get defaults from existing info for laser dig-ins
    default_laser_digin_ind = []
    if 'laser_params' in existing_info and existing_info['laser_params'].get('dig_in_nums'):
        # Convert to indices from dig_in_nums
        laser_nums = existing_info['laser_params']['dig_in_nums']
        if laser_nums:
            for num in laser_nums:
                matching_indices = this_dig_handler.dig_in_frame[
                    this_dig_handler.dig_in_frame.dig_in_nums == num
                ].index.tolist()
                if matching_indices:
                    default_laser_digin_ind.extend(matching_indices)
    else:
        default_laser_digin_ind = None

    # Custom conversion function for laser dig-ins
    def convert_laser_digin(input_str):
        if len(input_str) == 0:
            return []
        return [int(input_str)]

    # Use helper function with special handling for blank input
    laser_select_str = populate_field_with_defaults(
        field_name='dig_in_nums',
        nested_field='laser_params',
        entry_checker_msg='Laser dig_in INDEX, <BLANK> for none',
        check_func=count_check,
        existing_info=existing_info,
        cache=cache,
        convert_func=convert_laser_digin,
        fail_response='Please enter numbers in index of dataframe above',
        default_value_override=default_laser_digin_ind,
        force_default=args.auto_defaults
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
    if 'laser_params' not in cache:
        cache['laser_params'] = {}
    cache['laser_params']['dig_in_nums'] = laser_digin_ind
    save_to_cache(cache, cache_file_path)

    if not laser_digin_ind:
        return [], [], [], "", []

    laser_digin_nums = this_dig_handler.dig_in_frame.loc[laser_digin_ind, 'dig_in_nums'].to_list(
    )
    this_dig_handler.dig_in_frame.loc[laser_digin_ind, 'laser_bool'] = True
    this_dig_handler.dig_in_frame.laser_bool.fillna(False, inplace=True)

    print('Selected laser digins: \n')
    print_df = this_dig_handler.dig_in_frame.drop(columns=['pulse_times'])
    print_df = print_df[print_df.laser_bool]
    print(print_df)

    def laser_check(x):
        nums = re.findall('[0-9]+', x)
        return sum([x.isdigit() for x in nums]) == 2

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
    elif 'laser_params' in cache:
        default_laser_params_list = cache['laser_params'].get(
            'onset_duration', [])
        default_opto_loc_list = cache['laser_params'].get('opto_locs', [])
        default_virus_region = cache['laser_params'].get('virus_region', "")

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
        if args.auto_defaults:
            use_defaults_str = 'y'
            continue_bool = True
        else:
            use_defaults_str, continue_bool = entry_checker(
                msg='Use default laser parameters? (y/n) [ENTER for y] :: ',
                check_func=lambda x: x.lower() in ['y', 'n', 'yes', 'no', ''],
                fail_response='Please enter y or n')
        if continue_bool:
            use_defaults = use_defaults_str.lower() in ['y', 'yes', '']
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
    cache['laser_params']['onset_duration'] = laser_params_list
    cache['laser_params']['opto_locs'] = opto_loc_list
    cache['laser_params']['virus_region'] = virus_region_str
    save_to_cache(cache, cache_file_path)

    # Fill in laser parameters
    this_dig_handler.dig_in_frame.loc[laser_digin_ind, 'laser_params'] = str(
        laser_params_list)

    return laser_digin_ind, laser_digin_nums, laser_params_list, virus_region_str, opto_loc_list


##################################################
# Brain Regions and Electrode Layout
##################################################


def main():
    """
    Main function to run the experiment info generation process.

    This function orchestrates the entire process of generating experiment info:
    1. Sets up the environment and loads existing data
    2. Determines file types and electrode configurations
    3. Processes digital inputs for taste and laser parameters
    4. Handles electrode layout and EMG information
    5. Collects experiment notes
    6. Assembles and saves the final experiment info file
    """
    # Setup experiment info
    dir_path, dir_name, cache_file_path, cache, existing_info, metadata_dict, pipeline_check = setup_experiment_info()

    # Initialize the final dictionary with metadata
    fin_dict = {}

    # If template is provided, use it
    if args.template:
        with open(args.template, 'r') as file:
            template_dict = json.load(file)
            template_keys = list(template_dict.keys())
            from_template = {
                this_key: template_dict[this_key] for this_key in template_keys
                if this_key not in metadata_dict.keys()
            }
            fin_dict = {**metadata_dict, **from_template}
    else:
        # Initialize with just metadata if no template
        fin_dict = {**metadata_dict}

    # Find all ports used
    file_list = os.listdir(dir_path)
    if 'auxiliary.dat' in file_list:
        file_type = 'one file per signal type'
    elif sum(['rhd' in x for x in file_list]) > 1:
        file_type = 'traditional'
    else:
        file_type = 'one file per channel'

    # Initialize electrodes_list based on file type
    if file_type == 'one file per signal type':
        electrodes_list = ['amplifier.dat']
    elif file_type == 'one file per channel':
        electrodes_list = [
            name for name in file_list if name.startswith('amp-')]
    else:
        # For traditional file type, we'll initialize electrodes_list as empty
        # since we'll get electrode info directly from the header
        electrodes_list = []
        rhd_file_list = [x for x in file_list if 'rhd' in x]
        with open(os.path.join(dir_path, rhd_file_list[0]), 'rb') as f:
            header = read_header(f)
        ports = [x['port_prefix'] for x in header['amplifier_channels']]
        electrode_files = [x['native_channel_name']
                           for x in header['amplifier_channels']]

    ##################################################
    # Process Digital Inputs
    ##################################################
    print("\n=== Processing Digital Inputs ===")
    this_dig_handler = DigInHandler(dir_path, file_type)

    ##################################################
    # Process Taste Parameters
    ##################################################
    print("\n=== Processing Taste Parameters ===")
    # Process dig-ins based on mode
    if args.programmatic:
        taste_dig_inds, tastes, concs, pal_ranks, taste_digin_nums, taste_digin_trials = process_dig_ins_programmatic(
            this_dig_handler, args)
    else:
        taste_dig_inds, tastes, concs, pal_ranks, taste_digin_nums, taste_digin_trials = process_dig_ins_manual(
            this_dig_handler, args, existing_info, cache, cache_file_path)

    ##################################################
    # Process Laser Parameters
    ##################################################
    print("\n=== Processing Laser Parameters ===")
    if args.programmatic:
        laser_digin_ind, laser_digin_nums, laser_params_list, virus_region_str, opto_loc_list = process_laser_params_programmatic(
            this_dig_handler, args)
    else:
        laser_digin_ind, laser_digin_nums, laser_params_list, virus_region_str, opto_loc_list = process_laser_params_manual(
            this_dig_handler, args, existing_info, cache, cache_file_path)

    # Write out dig-in frame
    col_names = this_dig_handler.dig_in_frame.columns
    # Move 'pulse_times' to the end
    this_dig_handler.dig_in_frame = this_dig_handler.dig_in_frame[
        [x for x in col_names if x != 'pulse_times'] + ['pulse_times']]
    this_dig_handler.write_out_frame()

    ##################################################
    # Process Electrode Layout
    ##################################################
    print("\n=== Processing Electrode Layout ===")

    # Process electrode files
    electrode_files, ports, electrode_num_list = process_electrode_files(
        file_type, electrodes_list, dir_path)

    # Process electrode layout
    layout_dict, fin_emg_port, orig_emg_electrodes, emg_muscle_str = process_electrode_layout(
        dir_path, dir_name, electrode_files, ports, electrode_num_list,
        args, existing_info, cache, cache_file_path)

    ##################################################
    # Process Notes and Finalize Dictionary
    ##################################################
    print("\n=== Processing Notes and Finalizing ===")

    # Process notes
    notes = process_notes(args, existing_info, cache, cache_file_path)

    # Process permanent path for metadata backup
    permanent_path = process_permanent_path(
        dir_path, dir_name, args, existing_info, cache, cache_file_path)

    # Get laser trial counts if laser dig-ins exist
    if laser_digin_ind:
        laser_digin_trials = this_dig_handler.dig_in_frame.loc[laser_digin_ind, 'trial_counts'].to_list(
        )
    else:
        laser_digin_trials = []

    # Create final dictionary
    fin_dict = {
        'version': '0.0.3',
        **metadata_dict,
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
        'electrode_layout': layout_dict,
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
            'onset_duration': laser_params_list,
            'opto_locs': opto_loc_list,
            'virus_region': virus_region_str
        },
        'notes': notes
    }

    # Add permanent_path to the dictionary if provided
    if permanent_path:
        fin_dict['permanent_path'] = permanent_path

    # Write the final dictionary to a JSON file
    json_file_name = os.path.join(dir_path, '.'.join([dir_name, 'info']))
    with open(json_file_name, 'w') as file:
        json.dump(fin_dict, file, indent=4)

    # Collect all generated files for copying
    layout_file_path = os.path.join(dir_path, dir_name + "_electrode_layout.csv")
    dig_in_frame_path = os.path.join(dir_path, 'dig_in_channel_info.json')
    
    generated_files = [
        json_file_name,
        layout_file_path,
        dig_in_frame_path
    ]

    # Copy metadata to permanent location if specified
    if permanent_path:
        copy_metadata_to_permanent_location(
            dir_path, dir_name, permanent_path, generated_files)

    # Write success to log
    if pipeline_check:
        pipeline_check.write_to_log(os.path.abspath(__file__), 'completed')

    print(f"Successfully created experiment info file: {json_file_name}")
    return fin_dict


if __name__ == "__main__":
    main()
