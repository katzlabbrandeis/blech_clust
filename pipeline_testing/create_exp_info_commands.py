"""
usage: blech_exp_info.py [-h] [--template TEMPLATE] [--mode {legacy,updated}]
                         [--programmatic] [--use-layout-file]
                         [--car-groups CAR_GROUPS] [--emg-muscle EMG_MUSCLE]
                         [--taste-digins TASTE_DIGINS] [--tastes TASTES]
                         [--concentrations CONCENTRATIONS]
                         [--palatability PALATABILITY]
                         [--laser-digin LASER_DIGIN]
                         [--laser-params LASER_PARAMS]
                         [--virus-region VIRUS_REGION] [--opto-loc OPTO_LOC]
                         [--notes NOTES] [--permanent-path PERMANENT_PATH]
                         dir_name

Creates files with experiment info

positional arguments:
  dir_name              Directory containing data files

optional arguments:
  -h, --help            show this help message and exit
  --template TEMPLATE, -t TEMPLATE
                        Template (.info) file to copy experimental details
                        from
  --mode {legacy,updated}, -m {legacy,updated}
  --programmatic        Run in programmatic mode
  --use-layout-file     Use existing electrode layout file
  --car-groups CAR_GROUPS
                        Comma-separated CAR groupings
  --emg-muscle EMG_MUSCLE
                        Name of EMG muscle
  --taste-digins TASTE_DIGINS
                        Comma-separated indices of taste digital inputs
  --tastes TASTES       Comma-separated taste names
  --concentrations CONCENTRATIONS
                        Comma-separated concentrations in M
  --palatability PALATABILITY
                        Comma-separated palatability rankings
  --laser-digin LASER_DIGIN
                        Laser digital input index
  --laser-params LASER_PARAMS
                        Laser onset,duration in ms
  --virus-region VIRUS_REGION
                        Virus region
  --opto-loc OPTO_LOC   Opto-fiber location
  --notes NOTES         Experiment notes
  --permanent-path PERMANENT_PATH
                        Permanent path where metadata files should be copied
"""

import os
import sys
import argparse
import re


def parse_laser_params(s):
    """
    Parse laser parameters from string format like "(0,2500),(0,500),(300,500),(800,500)"
    Returns a list of tuples [(onset, duration), ...]
    """
    pattern = re.compile(r'\((\d+),(\d+)\)')
    matches = pattern.findall(s)
    if not matches:
        raise ValueError(
            'Invalid format for laser parameters. Expected format: (onset1,duration1),(onset2,duration2),...')
    return [(int(onset), int(duration)) for onset, duration in matches]


def validate_laser_opto_match(laser_params, opto_locs):
    """
    Validate that the number of laser parameters matches the number of opto locations
    """
    laser_count = len(parse_laser_params(laser_params))
    opto_count = len(opto_locs.split(','))

    if laser_count != opto_count:
        raise ValueError(
            f"Number of laser conditions ({laser_count}) must match number of opto locations ({opto_count})")
    return True


test_bool = False  # noqa
if not test_bool:
    parser = argparse.ArgumentParser(
        description='Creates files with experiment info')
    parser.add_argument('dir_name', type=str,
                        help='Directory containing data files')
    parser.add_argument('file_type', type=str, help='File type to create',
                        choices=['ofpc', 'trad'])
    parser.add_argument('key', type=str, help='Key for command to run',
                        choices=['emg_only', 'spike_only', 'emg_spike', 'laser'])
else:
    args = argparse.Namespace(
        dir_name='/media/storage/for_transfer/bla_gc/AM35_4Tastes_201228_124547',
        file_type='ofpc',
        key='emg_only'
    )

ofpc_command_dict = {}
ofpc_stem_str = \
    """python blech_exp_info.py $DIR \
--programmatic \
--emg-muscle ad \
--taste-digins 0,1,2,3 \
--tastes a,b,c,d \
--concentrations 1,1,1,1 \
--palatability 1,2,3,4 \
"""

wanted_emg_inds = [8, 9]
car_groups = ['none']*32
for ind in wanted_emg_inds:
    car_groups[ind] = 'emg'
car_groups = ','.join(car_groups)
ofpc_command_dict['emg_only'] = \
    ofpc_stem_str + \
    f"""--car-groups "{car_groups}" \
"""
# --programmatic \
# --emg-muscle ad \
# --taste-digins 0,1,2,3 \
# --tastes a,b,c,d \
# --concentrations 1,1,1,1 \
# --palatability 1,2,3,4 \
# --car-groups "{car_groups}" \
# """

wanted_gc_inds = [0, 1, 2, 29]
car_groups = ['none']*32
for ind in wanted_gc_inds:
    car_groups[ind] = 'gc'
car_groups = ','.join(car_groups)
ofpc_command_dict['spike_only'] = \
    ofpc_stem_str + \
    f"""--car-groups "{car_groups}" \
"""
# """python blech_exp_info.py $DIR \
# --programmatic \
# --emg-muscle ad \
# --taste-digins 0,1,2,3 \
# --tastes a,b,c,d \
# --concentrations 1,1,1,1 \
# --palatability 1,2,3,4 \
# --car-groups "{car_groups}" \
# """

car_groups = ['none']*32
for ind in wanted_gc_inds:
    car_groups[ind] = 'gc'
for ind in wanted_emg_inds:
    car_groups[ind] = 'emg'
car_groups = ','.join(car_groups)
ofpc_command_dict['emg_spike'] = \
    ofpc_stem_str + \
    f"""--car-groups "{car_groups}" \
"""
# """python blech_exp_info.py $DIR \
# --programmatic \
# --emg-muscle ad \
# --taste-digins 0,1,2,3 \
# --tastes a,b,c,d \
# --concentrations 1,1,1,1 \
# --palatability 1,2,3,4 \
# --car-groups "{car_groups}" \
# """

trad_command_dict = {}
trad_stem_str = \
    """python blech_exp_info.py $DIR \
--programmatic \
--emg-muscle ad \
--taste-digins 3,4 \
--tastes a,b \
--concentrations 1,1 \
--palatability 1,2 \
"""

wanted_gc_inds = [39, 44, 63]
car_groups = ['none']*64
for ind in wanted_gc_inds:
    car_groups[ind] = 'gc'
car_groups = ','.join(car_groups)
trad_command_dict['spike_only'] = \
    trad_stem_str + \
    f"""--car-groups "{car_groups}" \
"""
# f"""python blech_exp_info.py $DIR \
# --programmatic \
# --emg-muscle ad \
# --taste-digins 3,4 \
# --tastes a,b \
# --concentrations 1,1 \
# --palatability 1,2 \
# --car-groups "{car_groups}" \
# """

wanted_emg_inds = [8, 9]
car_groups = ['none']*64
for ind in wanted_emg_inds:
    car_groups[ind] = 'emg'
car_groups = ','.join(car_groups)
trad_command_dict['emg_only'] = \
    trad_stem_str + \
    f"""--car-groups "{car_groups}" \
"""
# f"""python blech_exp_info.py $DIR \
# --programmatic \
# --emg-muscle ad \
# --taste-digins 3,4 \
# --tastes a,b \
# --concentrations 1,1 \
# --palatability 1,2 \
# --car-groups "{car_groups}" \
# """

car_groups = ['none']*64
for ind in wanted_gc_inds:
    car_groups[ind] = 'gc'
for ind in wanted_emg_inds:
    car_groups[ind] = 'emg'
car_groups = ','.join(car_groups)
trad_command_dict['emg_spike'] = \
    trad_stem_str + \
    f"""--car-groups "{car_groups}" \
"""
# f"""python blech_exp_info.py $DIR \
# --programmatic \
# --emg-muscle ad \
# --taste-digins 3,4 \
# --tastes a,b \
# --concentrations 1,1 \
# --palatability 1,2 \
# --car-groups "{car_groups}" \
# """


command_dict = {}
command_dict['ofpc'] = ofpc_command_dict
command_dict['trad'] = trad_command_dict

# Test laser command
multi_laser_params = "(0,2500),(0,500),(300,500),(800,500)"
multi_opto_locs = "gc,gc,gc,gc"
laser_stem_str = \
    f"""python blech_exp_info.py $DIR \
--programmatic \
--emg-muscle ad \
--taste-digins 0,1,2 \
--tastes a,b,c \
--concentrations 1,1,1 \
--palatability 1,2,3 \
--laser-digin 3 \
--laser-params "{multi_laser_params}" \
--opto-loc {multi_opto_locs} \
--virus-region gc \
"""

# Validate that laser params and opto locations match
try:
    validate_laser_opto_match(
        multi_laser_params,
        multi_opto_locs
    )
    print("Laser command template validated successfully.")
except ValueError as e:
    print(f"Warning in laser command template: {e}")

wanted_gc_inds = [0, 1, 2, 29]
car_groups = ['none']*32
for ind in wanted_gc_inds:
    car_groups[ind] = 'gc'
for ind in wanted_emg_inds:
    car_groups[ind] = 'emg'
car_groups = ','.join(car_groups)
laser_command = \
    laser_stem_str + \
    f"""--car-groups "{car_groups}" \
"""

command_dict['ofpc']['laser'] = laser_command


if __name__ == '__main__':
    from pprint import pprint as pp
    args = parser.parse_args()
    data_dir = args.dir_name
    file_type = args.file_type
    key = args.key

    print(f'Running {key} command')
    this_command_dict = command_dict[file_type]
    this_command = this_command_dict[key]

    # Replace $DIR with the directory name
    this_command = this_command.replace('$DIR', data_dir)
    print('================================\n')
    pp(this_command)
    print('\n================================\n')
    os.system(this_command)
    print('\n')
