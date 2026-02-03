"""
Creating a Prefect pipeline for running tests
Run python scripts using subprocess as prefect tasks

NOTE: Superflows will stop execution of that flow on the first subflow/task that fails,
    unless we catch exceptions within the subflow/task itself.
"""

############################################################
from blech_clust.pipeline_testing.s3_utils import (
    S3_BUCKET,
    dummy_upload_test_results,
    compress_image,
    upload_test_results,
)
import traceback
test_bool = False

import argparse  # noqa
import os  # noqa
if test_bool:
    # Run this script as a test
    # Set up test data directory
    script_path = os.path.expanduser(
        '~/Desktop/blech_clust/pipeline_testing/prefect_pipeline.py')

    args = argparse.Namespace(
        e=False,
        s=True,
        freq=False,
        bsa=False,
        stft=False,
        qda=False,
        all=False,
        spike_emg=False,
        fail_fast=False,
        file_type='ofpc',
        dummy_upload=False
    )
else:
    parser = argparse.ArgumentParser(
        description='Run tests, default = Run all tests')
    parser.add_argument('-e', action='store_true',
                        help='Run EMG test only')
    parser.add_argument('-s', action='store_true',
                        help='Run spike sorting test only')
    parser.add_argument('--freq', action='store_true',
                        help='Run freq test only')
    parser.add_argument('--bsa', action='store_true',
                        help='Run BSA test only')
    parser.add_argument('--stft', action='store_true',
                        help='Run STFT test only')
    parser.add_argument('--qda', action='store_true',
                        help='Run QDA test only')
    parser.add_argument('--all', action='store_true',
                        help='Run all tests')
    parser.add_argument('--spike-emg', action='store_true',
                        help='Run spike + emg in single test')
    parser.add_argument('--ephys_data', action='store_true',
                        help='Run ephys_data test')
    parser.add_argument('--fail-fast', action='store_true',
                        help='Stop execution on first error encountered')
    parser.add_argument('--file_type',
                        help='File types to run tests on',
                        choices=['ofpc', 'trad', 'all'],
                        default='all', type=str)
    parser.add_argument('--data_type',
                        help='Data type to run tests on',
                        choices=['emg', 'emg_spike', 'all'],
                        default='all', type=str)
    parser.add_argument('--dummy-upload', action='store_true',
                        help='Run dummy upload test')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose debug output')
    parser.add_argument('--fail-check', action='store_true',
                        help='Run tests expected to fail')
    parser.add_argument('--fail-popen', action='store_true',
                        help='Run fail check using Popen')
    args = parser.parse_args()
    script_path = os.path.realpath(__file__)

from subprocess import PIPE, Popen  # noqa
from prefect import flow, task  # noqa
from glob import glob  # noqa
import json  # noqa
import sys  # noqa
from create_exp_info_commands import command_dict  # noqa
from switch_auto_car import set_auto_car  # noqa
import pandas as pd  # noqa

blech_clust_dir = os.path.dirname(os.path.dirname(script_path))
sys.path.append(blech_clust_dir)
import blech_clust.utils.ephys_data.ephys_data as ephys_data  # noqa

print(args.fail_fast)
fail_fast = args.fail_fast

# Set file_types to run
if args.file_type == 'all':
    file_types = ['ofpc', 'trad']
else:
    file_types = [args.file_type]
print(f'Running tests for file types: {file_types}')

# Set data_types to run
if args.data_type == 'all':
    data_type_list = ['emg', 'emg_spike']
else:
    data_type_list = [args.data_type]
print(f'Running tests for data types: {data_type_list}')

if fail_fast:
    print('====================')
    print('Stopping execution on error')
    print('====================')

# Set verbose flag for debug prints
verbose = args.verbose
if verbose:
    print('====================')
    print('Verbose mode enabled')
    print('====================')


def raise_error_if_error(data_dir, process, stderr, stdout, fail_fast=True):
    # Print current data_type
    current_data_type_path = os.path.join(data_dir, 'current_data_type.txt')
    if os.path.exists(current_data_type_path):
        with open(os.path.join(data_dir, 'current_data_type.txt'), 'r') as f:
            current_data_type = f.readlines()
        print('=== Current data type: ', current_data_type, ' ===\n\n')
    print('=== Process stdout ===\n\n')
    print(stdout.decode('utf-8'))
    print('=== Process stderr ===\n\n')
    if process.returncode:
        decode_err = stderr.decode('utf-8')
        raise Exception(decode_err)
    if process.returncode and not fail_fast:
        print('Encountered error...fail-fast not enabled, continuing execution...\n\n')
    if fail_fast and process.returncode:
        exit(1)


############################################################
# Define paths
############################################################
# Define paths
# TODO: Replace with call to blech_process_utils.path_handler

# Read emg_env path
emg_params_path = os.path.join(blech_clust_dir, 'params', 'emg_params.json')
if not os.path.exists(emg_params_path):
    print('=== Environment params file not found. ===')
    print(
        '==> Please copy [[ blech_clust/params/_templates/emg_params.json ]] to [[ blech_clust/params/env_params.json ]] and update as needed.')
    exit()
with open(emg_params_path) as f:
    env_params = json.load(f)
emg_env_path = env_params['emg_env']

# Load test data configuration from test_config.json
from blech_clust.pipeline_testing.test_config_loader import (
    get_data_dirs_dict,
    get_test_data_dir,
)
data_dirs_dict = get_data_dirs_dict()
data_dir_base = get_test_data_dir()

############################################################
# Data Prep Scripts
############################################################


@task(log_prints=True)
def download_test_data(data_dir):
    if verbose:
        print(f'[DEBUG] download_test_data: Starting with data_dir={data_dir}')
    print('Checking for test data, and downloading if not found')
    script_name = './pipeline_testing/test_data_handling/download_test_data.sh'
    process = Popen(["bash", script_name],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(data_dir, process, stderr, stdout, fail_fast)


@task(log_prints=True)
def prep_data_info(
        file_type='ofpc',
        data_type='emg_spike'
):
    """
    Prepares data directory with info according to inputs

    Args:
        file_type (str): Type of file to prepare. Options are 'ofpc', 'trad'
        data_type (str): Type of data to prepare. Options are 'emg', 'spike', 'emg_spike'
    """
    if verbose:
        print(
            f'[DEBUG] prep_data_info: Starting with file_type={file_type}, data_type={data_type}')
    if data_type == 'emg':
        # flag_str = '-emg'
        data_key = 'emg_only'
    elif data_type == 'spike':
        # flag_str = '-spike'
        data_key = 'spike_only'
    elif data_type == 'emg_spike':
        # flag_str = '-emg_spike'
        data_key = 'emg_spike'

    # Write out data_type to file
    data_dir = data_dirs_dict[file_type]
    current_data_type_path = os.path.join(data_dir, 'current_data_type.txt')
    print(f"""
          Writing
          file_type: {file_type}
          data type: {data_type}
          to {current_data_type_path}
          """)
    with open(current_data_type_path, 'w') as f:
        f.write(f"{file_type} -- {data_type}")

    cmd_str = command_dict[file_type][data_key]
    # Replace $DIR with data_dir
    cmd_str = cmd_str.replace('$DIR', data_dir)
    process = Popen(cmd_str, shell=True, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(data_dir, process, stderr, stdout, fail_fast)


############################################################
# Common Scripts
############################################################
@task(log_prints=True)
def reset_blech_clust(data_dir):
    if verbose:
        print(f'[DEBUG] reset_blech_clust: Starting with data_dir={data_dir}')
    script_name = './pipeline_testing/reset_blech_clust.py'
    process = Popen(["python", script_name],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(data_dir, process, stderr, stdout, fail_fast)


@task(log_prints=True)
def run_clean_slate(data_dir):
    if verbose:
        print(f'[DEBUG] run_clean_slate: Starting with data_dir={data_dir}')
    script_name = 'blech_clean_slate.py'
    process = Popen(["python", script_name, data_dir],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(data_dir, process, stderr, stdout, fail_fast)


@task(log_prints=True)
def mark_exp_info_success(data_dir):
    if verbose:
        print(
            f'[DEBUG] mark_exp_info_success: Starting with data_dir={data_dir}')
    script_name = './pipeline_testing/mark_exp_info_success.py'
    process = Popen(["python", script_name, data_dir],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(data_dir, process, stderr, stdout, fail_fast)


@task(log_prints=True)
def run_blech_init(data_dir):
    if verbose:
        print(f'[DEBUG] run_blech_init: Starting with data_dir={data_dir}')
    script_name = 'blech_init.py'
    process = Popen(["python", script_name, data_dir],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(data_dir, process, stderr, stdout, fail_fast)


@task(log_prints=True)
def make_arrays(data_dir):
    if verbose:
        print(f'[DEBUG] make_arrays: Starting with data_dir={data_dir}')
    script_name = 'blech_make_arrays.py'
    process = Popen(["python", script_name, data_dir],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(data_dir, process, stderr, stdout, fail_fast)

############################################################
# Spike Only
############################################################


@task(log_prints=True)
def run_CAR(data_dir):
    if verbose:
        print(f'[DEBUG] run_CAR: Starting with data_dir={data_dir}')
    script_name = 'blech_common_avg_reference.py'
    process = Popen(["python", script_name, data_dir],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(data_dir, process, stderr, stdout, fail_fast)


@task(log_prints=True)
def change_waveform_classifier(data_dir, use_classifier=1):
    if verbose:
        print(
            f'[DEBUG] change_waveform_classifier: Starting with data_dir={data_dir}, use_classifier={use_classifier}')
    script_name = 'pipeline_testing/change_waveform_classifier.py'
    process = Popen(["python", script_name, str(use_classifier)],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(data_dir, process, stderr, stdout, fail_fast)


@task(log_prints=True)
def change_auto_params(data_dir, use_auto=1):
    if verbose:
        print(
            f'[DEBUG] change_auto_params: Starting with data_dir={data_dir}, use_auto={use_auto}')
    script_name = 'pipeline_testing/change_auto_params.py'
    process = Popen(["python", script_name, data_dir, str(use_auto), str(use_auto)],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(data_dir, process, stderr, stdout, fail_fast)


@task(log_prints=True)
def run_jetstream_bash(data_dir):
    if verbose:
        print(f'[DEBUG] run_jetstream_bash: Starting with data_dir={data_dir}')
    script_name = 'blech_run_process.sh'
    process = Popen(["bash", script_name, '--delete-log', data_dir],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(data_dir, process, stderr, stdout, fail_fast)


@task(log_prints=True)
def select_clusters(data_dir):
    if verbose:
        print(f'[DEBUG] select_clusters: Starting with data_dir={data_dir}')
    script_name = 'pipeline_testing/select_some_waveforms.py'
    process = Popen(["python", script_name, data_dir],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(data_dir, process, stderr, stdout, fail_fast)


@task(log_prints=True)
def post_process(data_dir, use_file=True, keep_raw=False, delete_existing=False):
    if verbose:
        print(
            f'[DEBUG] post_process: Starting with data_dir={data_dir}, use_file={use_file}, keep_raw={keep_raw}, delete_existing={delete_existing}')
    script_name = 'blech_post_process.py'
    if use_file:
        sorted_units_path = glob(os.path.join(
            data_dir, '*sorted_units.csv'))[0]
        file_flag = '-f' + sorted_units_path
        run_list = ["python", script_name, data_dir, file_flag]
    else:
        run_list = ["python", script_name, data_dir]
    if keep_raw:
        run_list.append('--keep-raw')
    if delete_existing:
        run_list.append('--delete-existing')
    print(f'Post-process: {run_list}')
    process = Popen(run_list, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(data_dir, process, stderr, stdout, fail_fast)


@task(log_prints=True)
def quality_assurance(data_dir):
    if verbose:
        print(f'[DEBUG] quality_assurance: Starting with data_dir={data_dir}')
    script_name = 'blech_run_QA.sh'
    process = Popen(["bash", script_name, data_dir],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(data_dir, process, stderr, stdout, fail_fast)


@task(log_prints=True)
def units_plot(data_dir):
    if verbose:
        print(f'[DEBUG] units_plot: Starting with data_dir={data_dir}')
    script_name = 'blech_units_plot.py'
    process = Popen(["python", script_name, data_dir],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(data_dir, process, stderr, stdout, fail_fast)


@task(log_prints=True)
def units_characteristics(data_dir):
    if verbose:
        print(
            f'[DEBUG] units_characteristics: Starting with data_dir={data_dir}')
    script_name = 'blech_units_characteristics.py'
    process = Popen(["python", script_name, data_dir],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(data_dir, process, stderr, stdout, fail_fast)

############################################################
# EMG Only
############################################################


@task(log_prints=True)
def change_emg_freq_method(data_dir, use_BSA=1):
    if verbose:
        print(
            f'[DEBUG] change_emg_freq_method: Starting with data_dir={data_dir}, use_BSA={use_BSA}')
    script_name = 'pipeline_testing/change_emg_freq_method.py'
    process = Popen(["python", script_name, str(use_BSA)],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(data_dir, process, stderr, stdout, fail_fast)


@task(log_prints=True)
def cut_emg_trials(data_dir):
    if verbose:
        print(f'[DEBUG] cut_emg_trials: Starting with data_dir={data_dir}')
    script_name = 'pipeline_testing/cut_emg_trials.py'
    process = Popen(["python", script_name, data_dir],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(data_dir, process, stderr, stdout, fail_fast)


@task(log_prints=True)
def emg_filter(data_dir):
    if verbose:
        print(f'[DEBUG] emg_filter: Starting with data_dir={data_dir}')
    script_name = 'emg_filter.py'
    process = Popen(["python", script_name, data_dir],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(data_dir, process, stderr, stdout, fail_fast)


@task(log_prints=True)
def emg_freq_setup(data_dir):
    if verbose:
        print(f'[DEBUG] emg_freq_setup: Starting with data_dir={data_dir}')
    script_name = 'emg_freq_setup.py'
    process = Popen(["python", script_name, data_dir],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(data_dir, process, stderr, stdout, fail_fast)


@task(log_prints=True)
def emg_jetstream_parallel(data_dir):
    if verbose:
        print(
            f'[DEBUG] emg_jetstream_parallel: Starting with data_dir={data_dir}')
    script_name = 'bash blech_emg_jetstream_parallel.sh'
    full_str = script_name
    process = Popen(full_str, shell=True, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(data_dir, process, stderr, stdout, fail_fast)


@task(log_prints=True)
def emg_freq_post_process(data_dir):
    if verbose:
        print(
            f'[DEBUG] emg_freq_post_process: Starting with data_dir={data_dir}')
    script_name = 'emg_freq_post_process.py'
    process = Popen(["python", script_name, data_dir],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(data_dir, process, stderr, stdout, fail_fast)


@task(log_prints=True)
def emg_freq_plot(data_dir):
    if verbose:
        print(f'[DEBUG] emg_freq_plot: Starting with data_dir={data_dir}')
    script_name = 'emg_freq_plot.py'
    process = Popen(["python", script_name, data_dir],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(data_dir, process, stderr, stdout, fail_fast)


@task(log_prints=True)
def run_gapes_Li(data_dir):
    if verbose:
        print(f'[DEBUG] run_gapes_Li: Starting with data_dir={data_dir}')
    script_name = 'get_gapes_Li.py'
    process = Popen(["python", script_name, data_dir],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(data_dir, process, stderr, stdout, fail_fast)


@task(log_prints=True)
def run_rnn(data_dir, separate_regions=False, separate_tastes=False):
    """Run RNN firing rate inference"""
    if verbose:
        print(
            f'[DEBUG] run_rnn: Starting with data_dir={data_dir}, separate_regions={separate_regions}, separate_tastes={separate_tastes}')
    script_name = 'utils/infer_rnn_rates.py'
    # Use 100 training steps for testing
    args = ["python", script_name, data_dir,
            "--train_steps", "100", "--retrain"]
    if separate_regions:
        args.append("--separate_regions")
    if separate_tastes:
        args.append("--separate_tastes")
    process = Popen(args,
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(data_dir, process, stderr, stdout)

############################################################
# Ephys Data Tests
############################################################


@task(log_prints=True)
def test_ephys_data(data_dir):
    """Test ephys_data functionality"""
    if verbose:
        print(f'[DEBUG] test_ephys_data: Starting with data_dir={data_dir}')
    print("Testing ephys_data with directory:", data_dir)

    dat = ephys_data.ephys_data(data_dir)
    dat.firing_rate_params = dat.default_firing_params

    test_methods = {
        'get_unit_descriptors': 'core',
        'get_spikes': 'core',
        'get_firing_rates': 'core',
        'get_lfps': 'core',
        'get_info_dict': 'core',
        'get_region_electrodes': 'electrode_handling',
        'get_region_units': 'electrode_handling',
        'get_lfp_electrodes': 'electrode_handling',
        'return_region_spikes': 'electrode_handling',
        'get_region_firing': 'electrode_handling',
        'return_region_lfps': 'electrode_handling',
        'return_representative_lfp_channels': 'electrode_handling',
        'get_stft': 'stft',
        'get_mean_stft_amplitude': 'stft',
        'get_trial_info_frame': 'trial_sequestering',
        'sequester_trial_inds': 'trial_sequestering',
        'get_sequestered_spikes': 'trial_sequestering',
        'get_sequestered_firing': 'trial_sequestering',
        'get_sequestered_data': 'trial_sequestering',
        'calc_palatability': 'palatability',
    }

    dat.check_laser()
    if dat.laser_exists:
        test_methods.update({
            'separate_laser_spikes': 'laser',
            'separate_laser_firing': 'laser',
            'separate_laser_lfp': 'laser',
        })
    else:
        print("No laser detected, skipping laser-specific methods")

    ephys_test_df = pd.DataFrame(columns=['method', 'category', 'result'])
    print("Starting ephys data tests...")

    # Run tests for each method and keep track of results
    for method, category in test_methods.items():
        try:
            print(f"Testing {method}...")
            getattr(dat, method)()
            result = 'Success'
        except Exception as e:
            print(f"Error in {method}: {str(e)}")
            # Return full traceback
            traceback.print_exc()
            result = 'Failed'

        # Append result to DataFrame
        ephys_test_df = ephys_test_df.append({
            'method': method,
            'category': category,
            'result': result
        }, ignore_index=True)

    print("Ephys data tests complete!")
    print("Test results:")
    print(ephys_test_df)
    print("Ephys data testing complete!")

    # If any tests fail, raise an error
    if 'Failed' in ephys_test_df['result'].values:
        raise Exception(
            "Some ephys data tests failed. Check the output above.")


@task(log_prints=True)
def fail_check_popen(data_dir):
    """
    Dummy task to raise an exception for testing error handling
    using raise_error_if_error function
    """
    if verbose:
        print(f'[DEBUG] fail_check_popen with data_dir={data_dir}')
    print("Running fail_check_popen to simulate an error...")
    process = Popen(["python", '-c', 'raise Exception("Simulated failure popen")'],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(data_dir, process, stderr, stdout, fail_fast)


@task(log_prints=True)
def fail_check_direct():
    """
    Dummy task to raise an exception for testing error handling
    directly
    """
    if verbose:
        print(f'[DEBUG] fail_check_direct')
    print("Running fail_check_direct to simulate an error...")
    raise Exception("Simulated direct failure")


@task(log_prints=True)
def pass_check_popen(data_dir):
    """
    Dummy task to simulate a successful process using Popen
    """
    if verbose:
        print(f'[DEBUG] pass_check_popen with data_dir={data_dir}')
    print("Running pass_check_popen to simulate success...")
    process = Popen(["python", '-c', 'print("Simulated success popen")'],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(data_dir, process, stderr, stdout, fail_fast)


@task(log_prints=True)
def pass_check_direct():
    """
    Dummy task to simulate a successful direct execution
    """
    if verbose:
        print(f'[DEBUG] pass_check_direct')
    print("Running pass_check_direct to simulate success...")
    return


############################################################
# Define Flows
############################################################
# Make a try-except decorator to catch errors in flows but allow Prefect to log and continue
def try_except_flow(flow_func):
    def wrapper(*args, **kwargs):
        print(f"Try-except wrapper for flow: {flow_func.__name__}")
        try:
            flow_func(*args, **kwargs)
        except Exception as e:
            print(f"Error in flow {flow_func.__name__}: {str(e)}")
    return wrapper


@try_except_flow
@flow(log_prints=True)
def fail_check_subflow(use_popen=False):
    """Flow to test error handling"""
    data_dir = data_dirs_dict['ofpc']
    if use_popen:
        fail_check_popen(data_dir)
    else:
        fail_check_direct()


@try_except_flow
@flow(log_prints=True)
def pass_check_subflow(use_popen=False):
    """Flow to test successful execution"""
    data_dir = data_dirs_dict['ofpc']
    if use_popen:
        pass_check_popen(data_dir)
    else:
        pass_check_direct()


@flow(log_prints=True)
def fail_check_flow(use_popen=False):
    """Flow to test error handling"""
    print('Running 2 sets of tests...')
    print('1- Running first test...')
    pass_check_subflow(use_popen=use_popen)
    fail_check_subflow(use_popen=use_popen)
    print('2- Running second test...')
    pass_check_subflow(use_popen=use_popen)
    fail_check_subflow(use_popen=use_popen)

##############################


@try_except_flow
@flow(log_prints=True)
def prep_data_flow(file_type, data_type='emg_spike'):
    data_dir = data_dirs_dict[file_type]
    os.chdir(blech_clust_dir)
    download_test_data(data_dir)
    prep_data_info(file_type, data_type)


@try_except_flow
@flow(log_prints=True)
def run_spike_test(data_dir):
    os.chdir(blech_clust_dir)
    reset_blech_clust(data_dir)
    run_clean_slate(data_dir)
    mark_exp_info_success(data_dir)
    run_blech_init(data_dir)

    # Test with auto_car enabled
    set_auto_car(data_dir, 1)
    run_CAR(data_dir)
    # Test with auto_car disabled
    set_auto_car(data_dir, 0)
    run_CAR(data_dir)

    # Run with classifier disabled and manual sorting
    change_waveform_classifier(data_dir, use_classifier=0)
    change_auto_params(data_dir, use_auto=0)
    run_jetstream_bash(data_dir)
    select_clusters(data_dir)
    post_process(data_dir, use_file=True, keep_raw=True, delete_existing=False)

    # Run with classifier enabled + autosorting
    # Run this second so that final plots are actually units
    change_waveform_classifier(data_dir, use_classifier=1)
    change_auto_params(data_dir, use_auto=1)
    run_jetstream_bash(data_dir)
    # Keep raw in the first pass so jetstream step can be rerun
    post_process(data_dir, use_file=False, keep_raw=True, delete_existing=True)
    post_process(data_dir, use_file=False,
                 keep_raw=False, delete_existing=True)

    make_arrays(data_dir)
    quality_assurance(data_dir)
    units_plot(data_dir)
    units_characteristics(data_dir)
    # Remove for now...needs pytorch installation which is optional
    # run_rnn(data_dir, separate_regions=False,   separate_tastes=False)
    # run_rnn(data_dir, separate_regions=True,    separate_tastes=False)
    # run_rnn(data_dir, separate_regions=False,   separate_tastes=True)
    # run_rnn(data_dir, separate_regions=True,    separate_tastes=True)

    # Run ephys_data tests as part of spike testing
    test_ephys_data(data_dir)


@try_except_flow
@flow(log_prints=True)
def run_emg_main_test(data_dir):
    os.chdir(blech_clust_dir)
    reset_blech_clust(data_dir)
    run_clean_slate(data_dir)
    mark_exp_info_success(data_dir)
    run_blech_init(data_dir)
    make_arrays(data_dir)
    # Chop number of trials down to preserve time
    cut_emg_trials(data_dir)
    os.chdir(os.path.join(blech_clust_dir, 'emg'))
    emg_filter(data_dir)
    emg_freq_setup(data_dir)


@try_except_flow
@flow(log_prints=True)
def spike_emg_flow(data_dir, file_type):
    # Set data type
    data_type = 'emg_spike'
    prep_data_flow(file_type, data_type=data_type)
    print(f'Running spike+emg test with data type : {data_type}')
    # Spike test
    run_spike_test(data_dir)
    # Switch to EMG test without resetting
    # Chop number of trials down to preserve time
    cut_emg_trials(data_dir)
    os.chdir(os.path.join(blech_clust_dir, 'emg'))
    emg_filter(data_dir)
    # Perform EMG tests
    # BSA
    os.chdir(blech_clust_dir)
    # change_freq_method is in pipeline_testing dir
    change_emg_freq_method(data_dir, use_BSA=1)
    os.chdir(os.path.join(blech_clust_dir, 'emg'))
    emg_freq_setup(data_dir)
    emg_jetstream_parallel(data_dir)
    emg_freq_post_process(data_dir)
    emg_freq_plot(data_dir)
    # STFT
    os.chdir(blech_clust_dir)
    # change_freq_method is in pipeline_testing dir
    change_emg_freq_method(data_dir, use_BSA=0)
    os.chdir(os.path.join(blech_clust_dir, 'emg'))
    # Freq setup needs to be rerun to recreate bash parallel file
    emg_freq_setup(data_dir)
    emg_jetstream_parallel(data_dir)
    emg_freq_post_process(data_dir)
    emg_freq_plot(data_dir)
    # QDA
    os.chdir(os.path.join(blech_clust_dir, 'emg', 'gape_QDA_classifier'))
    run_gapes_Li(data_dir)


@try_except_flow
@flow(log_prints=True)
def run_emg_freq_test(data_dir, use_BSA=1):
    os.chdir(blech_clust_dir)
    # change_emg_freq_method needs to be in blech_clust_dir
    change_emg_freq_method(data_dir, use_BSA=use_BSA)
    run_emg_main_test(data_dir)
    emg_jetstream_parallel(data_dir)
    emg_freq_post_process(data_dir)
    emg_freq_plot(data_dir)

##############################
# FINAL LEVEL FLOWS
##############################


@flow(log_prints=True)
def spike_only_test():
    for file_type in file_types:
        data_dir = data_dirs_dict[file_type]
        # for data_type in ['spike', 'emg_spike']:
        # spike+emg test is covered in spike_emg_test
        # don't need to run here
        for data_type in ['spike']:
            print(f"""Running spike test with
                  file type : {file_type}
                  data type : {data_type}""")
            prep_data_flow(file_type, data_type=data_type)
            run_spike_test(data_dir)

            # Upload results to S3
            # Get current data type from file
            current_data_type_path = os.path.join(
                data_dir, 'current_data_type.txt')
            if os.path.exists(current_data_type_path):
                with open(current_data_type_path, 'r') as f:
                    current_data_type = f.read().strip().split(' -- ')[-1]
            else:
                current_data_type = "spike"  # Default if file doesn't exist

            upload_test_results(data_dir, "spike",
                                file_type, data_type=current_data_type)


@flow(log_prints=True)
def ephys_data_only_flow():
    for file_type in file_types:
        data_dir = data_dirs_dict[file_type]
        test_ephys_data(data_dir)
    #     prep_data_flow(file_type, data_type='emg_spike')
    #     os.chdir(blech_clust_dir)
    #     reset_blech_clust(data_dir)
    #     run_clean_slate(data_dir)
    # mark_exp_info_success(data_dir)
    # run_blech_clust(data_dir)
    #
    # make_arrays(data_dir)
    # # Run ephys_data tests


@flow(log_prints=True)
def spike_emg_test():
    for file_type in file_types:
        data_dir = data_dirs_dict[file_type]
        spike_emg_flow(data_dir, file_type)

        # Upload results to S3 with data_type
        upload_test_results(data_dir, "spike_emg",
                            file_type, data_type="emg_spike")


@flow(log_prints=True)
def bsa_only_test():
    for file_type in file_types:
        data_dir = data_dirs_dict[file_type]
        for data_type in data_type_list:
            print(f"""Running BSA test with
                  file type : {file_type}
                  data type : {data_type}""")
            prep_data_flow(file_type, data_type=data_type)
            run_emg_freq_test(data_dir, use_BSA=1)
            upload_test_results(
                data_dir, "BSA", file_type, data_type=data_type)


@flow(log_prints=True)
def stft_only_test():
    for file_type in file_types:
        data_dir = data_dirs_dict[file_type]
        for data_type in data_type_list:
            print(f"""Running STFT test with
                  file type : {file_type}
                  data type : {data_type}""")
            prep_data_flow(file_type, data_type=data_type)
            run_emg_freq_test(data_dir, use_BSA=0)
            upload_test_results(
                data_dir, "STFT", file_type, data_type=data_type)


@flow(log_prints=True)
def run_EMG_QDA_test():
    for file_type in file_types:
        data_dir = data_dirs_dict[file_type]
        for data_type in data_type_list:
            print(f"""Running EMG QDA test with
                  file type : {file_type}
                  data type : {data_type}""")
            prep_data_flow(file_type, data_type=data_type)
            run_emg_main_test(data_dir)
            os.chdir(os.path.join(blech_clust_dir,
                     'emg', 'gape_QDA_classifier'))
            run_gapes_Li(data_dir)
            upload_test_results(
                data_dir, "QDA", file_type, data_type=data_type)


@flow(log_prints=True)
def run_emg_freq_only():
    bsa_only_test()
    stft_only_test()


@flow(log_prints=True)
def emg_only_test():
    run_emg_freq_only()
    run_EMG_QDA_test()


@flow(log_prints=True)
def full_test():
    spike_emg_test()
    emg_only_test()


############################################################
# Run Flows
############################################################
# If no individual tests are required, run both
if args.all:
    print('Running all tests')
    full_test(return_state=True)
elif args.e:
    print('Running emg tests only')
    emg_only_test(return_state=True)
elif args.s:
    print('Running spike-sorting tests only')
    spike_only_test(return_state=True)
elif args.freq:
    print('Running BSA tests only')
    run_emg_freq_only(return_state=True)
elif args.qda:
    print('Running QDA tests only')
    run_EMG_QDA_test(return_state=True)
elif args.bsa:
    print('Running BSA tests only')
    bsa_only_test(return_state=True)
elif args.stft:
    print('Running STFT tests only')
    stft_only_test(return_state=True)
elif args.spike_emg:
    print('Running spike then emg test')
    spike_emg_test(return_state=True)
elif args.dummy_upload:
    print('Running dummy upload test')
    dummy_upload_test_results()
elif args.ephys_data:
    print('Running ephys_data class tests only')
    ephys_data_only_flow(return_state=True)
elif args.fail_check:
    print('Running fail_check test')
    fail_check_flow(use_popen=args.fail_popen, return_state=True)
