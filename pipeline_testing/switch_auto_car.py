from glob import glob
import argparse
import json
import os


def set_auto_car(data_dir, auto_car):
    """Set auto_car parameter in sorting params file"""
    # Get paths
    params_file = glob(os.path.join(data_dir, '*.params'))[0]

    # Read the current params
    with open(params_file, 'r') as f:
        params = json.load(f)

    # Set the value
    params['auto_CAR']['use_auto_CAR'] = bool(auto_car)

    # Write back the updated params
    with open(params_file, 'w') as f:
        json.dump(params, f, indent=4)

    # Print the new state
    print(f"auto_car is now: {params['auto_CAR']['use_auto_CAR']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Set auto_car parameter')
    parser.add_argument('data_dir', type=str,
                        help='Path to data directory')
    parser.add_argument('auto_car', type=int, choices=[0, 1],
                        help='Set auto_car to True (1) or False (0)')
    args = parser.parse_args()

    set_auto_car(args.data_dir, args.auto_car)
