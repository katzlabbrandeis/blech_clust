def switch_auto_car(enable: bool):
    """
    Function to switch auto_car on or off.

    Parameters:
    enable (bool): If True, turn auto_car on. If False, turn it off.

    Returns:
    None
    """
    if enable:
        print("auto_car is now ON")
        # Add logic to enable auto_car
    else:
        print("auto_car is now OFF")
        # Add logic to disable auto_car
import os
import json
import argparse
from glob import glob


def set_auto_car(data_dir, auto_car):
    """Set auto_car parameter in sorting params file"""
    # Get paths
    params_file = glob(os.path.join(data_dir, '*.params'))[0]

    # Read the current params
    with open(params_file, 'r') as f:
        params = json.load(f)

    # Set the value
    params['preprocessing_params']['auto_car'] = bool(auto_car)

    # Write back the updated params
    with open(params_file, 'w') as f:
        json.dump(params, f, indent=4)

    # Print the new state
    print(f"auto_car is now: {params['preprocessing_params']['auto_car']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Set auto_car parameter')
    parser.add_argument('data_dir', type=str,
                        help='Path to data directory')
    parser.add_argument('auto_car', type=int, choices=[0, 1],
                        help='Set auto_car to True (1) or False (0)')
    args = parser.parse_args()

    set_auto_car(args.data_dir, args.auto_car)
