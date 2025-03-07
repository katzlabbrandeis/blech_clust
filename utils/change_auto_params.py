import os
import json
import argparse
from glob import glob


def set_auto_params(data_dir, auto_cluster, auto_post):
    """Set auto_cluster and auto_post_process parameters in sorting params file"""
    # Get paths
    params_file = glob(os.path.join(data_dir, '*.params'))[0]

    # Read the current params
    with open(params_file, 'r') as f:
        params = json.load(f)

    # Set the values
    params['clustering_params']['auto_params']['auto_cluster'] = bool(
        auto_cluster)
    params['clustering_params']['auto_params']['auto_post_process'] = bool(
        auto_post)

    # Write back the updated params
    with open(params_file, 'w') as f:
        json.dump(params, f, indent=4)

    # Print the new states
    print(
        f"auto_cluster is now: {params['clustering_params']['auto_params']['auto_cluster']}")
    print(
        f"auto_post_process is now: {params['clustering_params']['auto_params']['auto_post_process']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Set auto clustering parameters')
    parser.add_argument('data_dir', type=str,
                        help='Path to data directory')
    parser.add_argument('auto_cluster', type=int, choices=[0, 1],
                        help='Set auto_cluster to True (1) or False (0)')
    parser.add_argument('auto_post', type=int, choices=[0, 1],
                        help='Set auto_post_process to True (1) or False (0)')
    args = parser.parse_args()

    set_auto_params(args.data_dir, args.auto_cluster, args.auto_post)
