import os
import json
import argparse

def set_auto_params(auto_cluster, auto_post):
    # Get paths
    script_path = os.path.realpath(__file__)
    blech_clust_dir = os.path.dirname(os.path.dirname(script_path))
    params_dir = os.path.join(blech_clust_dir, 'params')
    params_file = os.path.join(params_dir, 'sorting_params_template.json')
    """Set auto_cluster and auto_post_process parameters in sorting params file"""
   
    # Read the current params
    with open(params_file, 'r') as f:
        params = json.load(f)
    
    # Set the values
    params['clustering_params']['auto_params']['auto_cluster'] = bool(auto_cluster)
    params['clustering_params']['auto_params']['auto_post_process'] = bool(auto_post)
    
    # Write back the updated params
    with open(params_file, 'w') as f:
        json.dump(params, f, indent=4)
    
    # Print the new states
    print(f"auto_cluster is now: {params['clustering_params']['auto_params']['auto_cluster']}")
    print(f"auto_post_process is now: {params['clustering_params']['auto_params']['auto_post_process']}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Set auto clustering parameters')
    parser.add_argument('auto_cluster', type=int, choices=[0, 1],
                    help='Set auto_cluster to True (1) or False (0)')
    parser.add_argument('auto_post', type=int, choices=[0, 1],
                    help='Set auto_post_process to True (1) or False (0)')
    args = parser.parse_args()
    
    set_auto_params(args.auto_cluster, args.auto_post)
