import os
import json
import argparse

def toggle_auto_params():
    # Get paths
    script_path = os.path.realpath(__file__)
    blech_clust_dir = os.path.dirname(os.path.dirname(script_path))
    params_dir = os.path.join(blech_clust_dir, 'params')
    params_file = os.path.join(params_dir, 'sorting_params.json')
    """Toggle auto_cluster and auto_post_process parameters in sorting params file"""
    
    # Read the current params
    with open(params_file, 'r') as f:
        params = json.load(f)
    
    # Toggle the values
    params['clustering_params']['auto_params']['auto_cluster'] = \
        not params['clustering_params']['auto_params']['auto_cluster']
    
    params['clustering_params']['auto_params']['auto_post_process'] = \
        not params['clustering_params']['auto_params']['auto_post_process']
    
    # Write back the updated params
    with open(params_file, 'w') as f:
        json.dump(params, f, indent=4)
    
    # Print the new states
    print(f"auto_cluster is now: {params['clustering_params']['auto_params']['auto_cluster']}")
    print(f"auto_post_process is now: {params['clustering_params']['auto_params']['auto_post_process']}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Toggle auto clustering parameters')
    args = parser.parse_args()
    
    toggle_auto_params()
