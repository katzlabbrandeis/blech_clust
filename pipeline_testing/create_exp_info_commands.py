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
                         [--notes NOTES]
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
"""

import os
import sys
import argparse
parser = argparse.ArgumentParser(description='Creates files with experiment info')
parser.add_argument('dir_name', type=str, help='Directory containing data files')
parser.add_argument('file_type', type=str, help='File type to create',
                    choices=['ofpc', 'trad'])
parser.add_argument('key', type=str, help='Key for command to run',
                    choices=['emg_only', 'spike_only', 'emg_spike'])

ofpc_command_dict = {}
ofpc_command_dict['emg_only'] = \
"""python blech_exp_info.py $DIR \
--programmatic \
--emg-muscle ad \
--taste-digins 0,1,2,3 \
--tastes a,b,c,d \
--concentrations 1,1,1,1 \
--palatability 1,2,3,4 \
--car-groups "none,none,none,none,none,none,none,none,emg,emg,none,none,none,none,none,none,none,none,none,none,none,none,none,none,none,none,none,none,none,none,none,none" \
"""

ofpc_command_dict['spike_only'] = \
"""python blech_exp_info.py $DIR \
--programmatic \
--emg-muscle ad \
--taste-digins 0,1,2,3 \
--tastes a,b,c,d \
--concentrations 1,1,1,1 \
--palatability 1,2,3,4 \
--car-groups "gc,gc,gc,none,none,none,none,none,none,none,none,none,none,none,none,none,none,none,none,none,none,none,none,none,none,none,none,none,none,gc,none,none" \
"""

ofpc_command_dict['emg_spike'] = \
"""python blech_exp_info.py $DIR \
--programmatic \
--emg-muscle ad \
--taste-digins 0,1,2,3 \
--tastes a,b,c,d \
--concentrations 1,1,1,1 \
--palatability 1,2,3,4 \
--car-groups "gc,gc,gc,none,none,none,none,none,emg,emg,none,none,none,none,none,none,none,none,none,none,none,none,none,none,none,none,none,none,none,gc,none,none" \
"""

trad_command_dict = {}
trad_command_dict['spike_only'] = \
"""python blech_exp_info.py $DIR \
--programmatic \
--emg-muscle ad \
--taste-digins 3,4 \
--tastes a,b \
--concentrations 1,1 \
--palatability 1,2 \
--car-groups "none,none,none,none,none,none,gc,none,none,none,none,none,none,none,none,none,none,none,none,none,none,none,none,none,none,none,none,none,none,none,gc,none,none,none,none,none,none,none,none,none,none,none,none,none,none,none,none,none,none,none,none,none,none,none,none,none,none,none,none,gc,none,none,none,none" \
"""

command_dict = {}
command_dict['ofpc'] = ofpc_command_dict
command_dict['trad'] = trad_command_dict

if __name__ == '__main__':
    args = parser.parse_args()
    data_dir =  args.dir_name
    file_type = args.file_type
    key = args.key

    print(f'Running {key} command')
    this_command_dict = command_dict[file_type]
    this_command = this_command_dict[key]
    # Replace $DIR with the directory name
    this_command = this_command.replace('$DIR', data_dir)
    os.system(this_command)
    print('\n')
