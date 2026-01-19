"""
GLM-based firing rate estimation using nemos.

This script orchestrates GLM fitting across two Python environments:
1. blech_clust conda environment: Extracts spike data using ephys_data
2. nemos virtual environment: Fits GLM models

The script automatically detects environments and saves intermediate data files
for cross-environment communication.

Usage:
    python infer_glm_rates.py <data_dir> [options]

Options:
    --bin_size: Bin size in ms for spike binning (default: 25)
    --history_window: History window in ms for autoregressive effects (default: 250)
    --n_basis_funcs: Number of basis functions for history filter (default: 8)
    --time_lims: Time limits for analysis [start, end] in ms (default: [1500, 4500])
    --include_coupling: Include coupling between neurons (default: False)
    --separate_tastes: Fit separate models for each taste (default: False)
    --separate_regions: Fit separate models for each region (default: False)
    --retrain: Force retraining even if model exists (default: False)
    --blech_clust_env: Name of blech_clust conda environment (default: blech_clust)
    --nemos_env: Path to nemos venv Python interpreter (default: searches common locations)
"""

import argparse
import os
import sys
import subprocess
import json
import shutil

############################################################
# Argument parsing
############################################################

def parse_args(test_mode=False):
    if test_mode:
        print('====================')
        print('Running in test mode')
        print('====================')
        # data_dir = '/home/abuzarmahmood/projects/blech_clust/pipeline_testing/test_data_handling/test_data/KM45_5tastes_210620_113227_new'
        # data_dir = '/media/storage/abu_resorted/bla_gc/AM11_4Tastes_191030_114043_copy'
        data_dir = '/media/storage/abu_resorted/bla_gc/AM35_4Tastes_201228_124547' 
        args = argparse.Namespace(
            data_dir=data_dir,
            bin_size=25,
            history_window=250,
            n_basis_funcs=8,
            time_lims=[1500, 4500],
            include_coupling=True,
            separate_tastes=True,
            separate_regions=True,
            retrain=False,
            blech_clust_env='blech_clust',
            nemos_env='/home/abuzarmahmood/Desktop/blech_clust/nemos_venv/bin/python',
        )
        return args, test_mode
    else:
        parser = argparse.ArgumentParser(
            description='Infer firing rates using GLM (nemos)')
        parser.add_argument('data_dir', help='Path to data directory')
        parser.add_argument('--bin_size', type=int, default=25,
                            help='Bin size in ms for spike binning (default: %(default)s)')
        parser.add_argument('--history_window', type=int, default=250,
                            help='History window in ms for autoregressive effects (default: %(default)s)')
        parser.add_argument('--n_basis_funcs', type=int, default=8,
                            help='Number of basis functions for history filter (default: %(default)s)')
        parser.add_argument('--time_lims', type=int, nargs=2, default=[1500, 4500],
                            help='Time limits for analysis [start, end] in ms (default: %(default)s)')
        parser.add_argument('--include_coupling', action='store_false',
                            help='Include coupling between neurons (default: %(default)s)')
        parser.add_argument('--separate_tastes', action='store_true',
                            help='Fit separate models for each taste (default: %(default)s)')
        parser.add_argument('--separate_regions', action='store_true',
                            help='Fit separate models for each region (default: %(default)s)')
        parser.add_argument('--retrain', action='store_true',
                            help='Force retraining of model (default: %(default)s)')
        parser.add_argument('--blech_clust_env', type=str, default='blech_clust',
                            help='Name of blech_clust conda environment (default: %(default)s)')
        parser.add_argument('--nemos_env', type=str, default=None,
                            help='Path to nemos venv Python interpreter')
        return parser.parse_args(), test_mode


############################################################
# Environment detection
############################################################

def find_conda_env(env_name):
    """Find conda environment by name and return its Python path."""
    try:
        result = subprocess.run(
            # ['conda', 'run', '-n', env_name, 'which', 'python'],
            ['conda', 'env', 'list'],
            capture_output=True, text=True, timeout=30
        )
        # Parse output to find env path
        for line in result.stdout.splitlines():
            if line.startswith(env_name + ' '):
                env_path = line.split()[1]
                python_path = os.path.join(env_path, 'bin', 'python')
                return python_path
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    # Try common conda locations
    home = os.path.expanduser('~')
    common_paths = [
        f'{home}/anaconda3/envs/{env_name}/bin/python',
        f'{home}/miniconda3/envs/{env_name}/bin/python',
        f'{home}/miniforge3/envs/{env_name}/bin/python',
        f'/opt/conda/envs/{env_name}/bin/python',
    ]
    for path in common_paths:
        if os.path.exists(path):
            return path
    return None


def find_nemos_env():
    """Find nemos virtual environment Python interpreter."""
    home = os.path.expanduser('~')
    common_paths = [
        f'{home}/nemos_env/bin/python',
        f'{home}/venvs/nemos/bin/python',
        f'{home}/.venvs/nemos/bin/python',
        f'{home}/envs/nemos/bin/python',
        f'{home}/.local/share/virtualenvs/nemos/bin/python',
    ]
    for path in common_paths:
        if os.path.exists(path):
            # Verify nemos is installed
            try:
                result = subprocess.run(
                    [path, '-c', 'import nemos'],
                    capture_output=True, timeout=10
                )
                if result.returncode == 0:
                    return path
            except subprocess.TimeoutExpired:
                pass
    return None


def prompt_for_env(env_type, default_name=None):
    """Prompt user for environment path."""
    print(f"\n{env_type} environment not found automatically.")
    if default_name:
        print(f"Expected conda environment name: {default_name}")
    
    response = input(f"Enter path to {env_type} Python interpreter (or 'q' to quit): ").strip()
    if response.lower() == 'q':
        sys.exit(1)
    if os.path.exists(response):
        return response
    else:
        print(f"Path not found: {response}")
        return prompt_for_env(env_type, default_name)


############################################################
# Main orchestration
############################################################

def main():
    args, _ = parse_args()
    # args, test_mode = parse_args(test_mode=True)
    
    if not test_mode:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    else:
        script_dir = '/home/abuzarmahmood/Desktop/blech_clust/utils/glm'
    data_dir = os.path.abspath(args.data_dir)
    
    # Setup output directories
    output_path = os.path.join(data_dir, 'glm_output')
    # Use temp dir within the glm module directory
    temp_dir = os.path.join(script_dir, '_temp')
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    
    print("=" * 60)
    print("GLM-based Firing Rate Estimation")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    
    # Find blech_clust environment
    print("\n[1/4] Locating blech_clust environment...")
    blech_python = find_conda_env(args.blech_clust_env)
    if blech_python is None:
        blech_python = prompt_for_env('blech_clust', args.blech_clust_env)
    print(f"  Found: {blech_python}")
    
    # Find nemos environment
    print("\n[2/4] Locating nemos environment...")
    if args.nemos_env:
        nemos_python = args.nemos_env
    else:
        nemos_python = find_nemos_env()
    if nemos_python is None:
        nemos_python = prompt_for_env('nemos')
    print(f"  Found: {nemos_python}")
    
    # Save parameters for sub-scripts
    params = {
        'data_dir': data_dir,
        'bin_size': args.bin_size,
        'history_window': args.history_window,
        'n_basis_funcs': args.n_basis_funcs,
        'time_lims': args.time_lims,
        'include_coupling': args.include_coupling,
        'separate_tastes': args.separate_tastes,
        'separate_regions': args.separate_regions,
        'retrain': args.retrain,
        'temp_dir': temp_dir,
        'output_path': output_path,
    }
    params_path = os.path.join(temp_dir, 'glm_params.json')
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=2)
    
    # Step 1: Extract data using blech_clust environment
    print("\n[3/4] Extracting spike data (blech_clust environment)...")
    extract_script = os.path.join(script_dir, '_glm_extract_data.py')
    
    result = subprocess.run(
        [blech_python, extract_script, params_path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("ERROR: Data extraction failed")
        print(result.stderr)
        sys.exit(1)
    print(result.stdout)
    
    # Step 2: Fit GLM using nemos environment
    print("\n[4/4] Fitting GLM models (nemos environment)...")
    fit_script = os.path.join(script_dir, '_glm_fit_models.py')
    
    result = subprocess.run(
        [nemos_python, fit_script, params_path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("ERROR: GLM fitting failed")
        print(result.stderr)
        # sys.exit(1)
    print(result.stdout)
    
    # Cleanup temp files
    print("\nCleaning up temporary files...")
    shutil.rmtree(temp_dir)
    
    print("\n" + "=" * 60)
    print("GLM fitting complete!")
    print(f"Results saved to: {output_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
