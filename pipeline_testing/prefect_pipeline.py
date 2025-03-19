"""
Creating a Prefect pipeline for running tests
Run python scripts using subprocess as prefect tasks
"""

############################################################
import argparse  # noqa
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
parser.add_argument('--raise-exception', action='store_true',
                    help='Raise error if subprocess fails')
parser.add_argument('--file_type', 
                    help='File types to run tests on',
                    choices=['ofpc', 'trad', 'all'],
                    default= 'all', type=str)
args = parser.parse_args()

import os  # noqa
from subprocess import PIPE, Popen  # noqa
from prefect import flow, task  # noqa
from glob import glob  # noqa
import json  # noqa
import sys  # noqa
from create_exp_info_commands import command_dict  # noqa
from switch_auto_car import set_auto_car  # noqa

print(args.raise_exception)
break_bool = args.raise_exception

# Set file_types to run
if args.file_type == 'all':
    file_types = ['ofpc', 'trad']
else:
    file_types = [args.file_type]

if break_bool:
    print('====================')
    print('Raising error if subprocess fails')
    print('====================')


def raise_error_if_error(data_dir, process, stderr, stdout):
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


############################################################
# Define paths
############################################################
# Define paths
# TODO: Replace with call to blech_process_utils.path_handler
script_path = os.path.realpath(__file__)
blech_clust_dir = os.path.dirname(os.path.dirname(script_path))

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

# data_subdir = 'pipeline_testing/test_data_handling/test_data/KM45_5tastes_210620_113227_new'
# data_subdir = 'pipeline_testing/test_data_handling/eb24_behandephys_11_12_24_241112_114659_copy'
data_subdirs_dict = {
    'ofpc': 'KM45_5tastes_210620_113227_new',
    'trad': 'eb24_behandephys_11_12_24_241112_114659_copy'
}
data_dir_base = os.path.join(
    blech_clust_dir, 'pipeline_testing', 'test_data_handling', 'test_data')
data_dirs_dict = {key: os.path.join(data_dir_base, subdir)
                  for key, subdir in data_subdirs_dict.items()}

############################################################
# Data Prep Scripts
############################################################


@task(log_prints=True)
def download_test_data(data_dir):
    print('Checking for test data, and downloading if not found')
    script_name = './pipeline_testing/test_data_handling/download_test_data.sh'
    process = Popen(["bash", script_name],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(data_dir, process, stderr, stdout)


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
    raise_error_if_error(data_dir, process, stderr, stdout)


############################################################
# Common Scripts
############################################################
@task(log_prints=True)
def reset_blech_clust(data_dir):
    script_name = './pipeline_testing/reset_blech_clust.py'
    process = Popen(["python", script_name],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(data_dir, process, stderr, stdout)


@task(log_prints=True)
def run_clean_slate(data_dir):
    script_name = 'blech_clean_slate.py'
    process = Popen(["python", script_name, data_dir],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(data_dir, process, stderr, stdout)


@task(log_prints=True)
def mark_exp_info_success(data_dir):
    script_name = './pipeline_testing/mark_exp_info_success.py'
    process = Popen(["python", script_name, data_dir],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(data_dir, process, stderr, stdout)


@task(log_prints=True)
def run_blech_clust(data_dir):
    script_name = 'blech_clust.py'
    process = Popen(["python", script_name, data_dir],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(data_dir, process, stderr, stdout)


@task(log_prints=True)
def make_arrays(data_dir):
    script_name = 'blech_make_arrays.py'
    process = Popen(["python", script_name, data_dir],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(data_dir, process, stderr, stdout)

############################################################
# Spike Only
############################################################


@task(log_prints=True)
def run_CAR(data_dir):
    script_name = 'blech_common_avg_reference.py'
    process = Popen(["python", script_name, data_dir],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(data_dir, process, stderr, stdout)


@task(log_prints=True)
def change_waveform_classifier(data_dir, use_classifier=1):
    script_name = 'pipeline_testing/change_waveform_classifier.py'
    process = Popen(["python", script_name, str(use_classifier)],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(data_dir, process, stderr, stdout)


@task(log_prints=True)
def change_auto_params(data_dir, use_auto=1):
    script_name = 'pipeline_testing/change_auto_params.py'
    process = Popen(["python", script_name, data_dir, str(use_auto), str(use_auto)],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(data_dir, process, stderr, stdout)


@task(log_prints=True)
def run_jetstream_bash(data_dir):
    script_name = 'blech_run_process.sh'
    process = Popen(["bash", script_name, '--delete-log', data_dir],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(data_dir, process, stderr, stdout)


@task(log_prints=True)
def select_clusters(data_dir):
    script_name = 'pipeline_testing/select_some_waveforms.py'
    process = Popen(["python", script_name, data_dir],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(data_dir, process, stderr, stdout)


@task(log_prints=True)
def post_process(data_dir, use_file=True, keep_raw=False, delete_existing=False):
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
    raise_error_if_error(data_dir, process, stderr, stdout)


@task(log_prints=True)
def quality_assurance(data_dir):
    script_name = 'blech_run_QA.sh'
    process = Popen(["bash", script_name, data_dir],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(data_dir, process, stderr, stdout)


@task(log_prints=True)
def units_plot(data_dir):
    script_name = 'blech_units_plot.py'
    process = Popen(["python", script_name, data_dir],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(data_dir, process, stderr, stdout)


@task(log_prints=True)
def units_characteristics(data_dir):
    script_name = 'blech_units_characteristics.py'
    process = Popen(["python", script_name, data_dir],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(data_dir, process, stderr, stdout)

############################################################
# EMG Only
############################################################


@task(log_prints=True)
def change_emg_freq_method(data_dir, use_BSA=1):
    script_name = 'pipeline_testing/change_emg_freq_method.py'
    process = Popen(["python", script_name, str(use_BSA)],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(data_dir, process, stderr, stdout)


@task(log_prints=True)
def cut_emg_trials(data_dir):
    script_name = 'pipeline_testing/cut_emg_trials.py'
    process = Popen(["python", script_name, data_dir],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(data_dir, process, stderr, stdout)


@task(log_prints=True)
def emg_filter(data_dir):
    script_name = 'emg_filter.py'
    process = Popen(["python", script_name, data_dir],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(data_dir, process, stderr, stdout)


@task(log_prints=True)
def emg_freq_setup(data_dir):
    script_name = 'emg_freq_setup.py'
    process = Popen(["python", script_name, data_dir],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(data_dir, process, stderr, stdout)


@task(log_prints=True)
def emg_jetstream_parallel(data_dir):
    script_name = 'bash blech_emg_jetstream_parallel.sh'
    full_str = script_name
    process = Popen(full_str, shell=True, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(data_dir, process, stderr, stdout)


@task(log_prints=True)
def emg_freq_post_process(data_dir):
    script_name = 'emg_freq_post_process.py'
    process = Popen(["python", script_name, data_dir],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(data_dir, process, stderr, stdout)


@task(log_prints=True)
def emg_freq_plot(data_dir):
    script_name = 'emg_freq_plot.py'
    process = Popen(["python", script_name, data_dir],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(data_dir, process, stderr, stdout)


@task(log_prints=True)
def run_gapes_Li(data_dir):
    script_name = 'get_gapes_Li.py'
    process = Popen(["python", script_name, data_dir],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(data_dir, process, stderr, stdout)


@task(log_prints=True)
def run_rnn(data_dir, separate_regions=False, separate_tastes=False):
    """Run RNN firing rate inference"""
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
# Define Flows
############################################################


@flow(log_prints=True)
def prep_data_flow(file_type, data_type='emg_spike'):
    data_dir = data_dirs_dict[file_type]
    os.chdir(blech_clust_dir)
    download_test_data(data_dir)
    prep_data_info(file_type, data_type)


@flow(log_prints=True)
def run_spike_test(data_dir):
    os.chdir(blech_clust_dir)
    reset_blech_clust(data_dir)
    run_clean_slate(data_dir)
    mark_exp_info_success(data_dir)
    run_blech_clust(data_dir)

    # Test with auto_car enabled
    set_auto_car(data_dir, 1)
    run_CAR(data_dir)
    # Test with auto_car disabled
    set_auto_car(data_dir, 0)
    run_CAR(data_dir)

    # Run with classifier enabled + autosorting
    change_waveform_classifier(data_dir, use_classifier=1)
    change_auto_params(data_dir, use_auto=1)
    run_jetstream_bash(data_dir)
    # Keep raw in the first pass so jetstream step can be rerun
    post_process(data_dir, use_file=False, keep_raw=True)
    post_process(data_dir, use_file=False, keep_raw=True, delete_existing=True)

    # Run with classifier disabled and manual sorting
    change_waveform_classifier(data_dir, use_classifier=0)
    change_auto_params(data_dir, use_auto=0)
    run_jetstream_bash(data_dir)
    select_clusters(data_dir)
    post_process(data_dir, use_file=True, keep_raw=False, delete_existing=True)

    make_arrays(data_dir)
    quality_assurance(data_dir)
    units_plot(data_dir)
    units_characteristics(data_dir)
    run_rnn(data_dir, separate_regions=False,   separate_tastes=False)
    run_rnn(data_dir, separate_regions=True,    separate_tastes=False)
    run_rnn(data_dir, separate_regions=False,   separate_tastes=True)
    run_rnn(data_dir, separate_regions=True,    separate_tastes=True)


@flow(log_prints=True)
def run_emg_main_test(data_dir):
    os.chdir(blech_clust_dir)
    reset_blech_clust(data_dir)
    run_clean_slate(data_dir)
    mark_exp_info_success(data_dir)
    run_blech_clust(data_dir)
    make_arrays(data_dir)
    # Chop number of trials down to preserve time
    cut_emg_trials(data_dir)
    os.chdir(os.path.join(blech_clust_dir, 'emg'))
    emg_filter(data_dir)
    emg_freq_setup(data_dir)


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


@flow(log_prints=True)
def spike_only_test():
    if break_bool:
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
    else:
        for file_type in file_types:
            data_dir = data_dirs_dict[file_type]
            # for data_type in ['spike', 'emg_spike']:
            # spike+emg test is covered in spike_emg_test
            # don't need to run here
            for data_type in ['spike']:
                print(f"""Running spike test with
                      file type : {file_type}
                      data type : {data_type}""")
                try:
                    prep_data_flow(file_type, data_type=data_type)
                except:
                    print('Failed to prep data')
                try:
                    run_spike_test(data_dir)
                except:
                    print('Failed to run spike test')


@flow(log_prints=True)
def spike_emg_test():
    if break_bool:
        for file_type in file_types:
            data_dir = data_dirs_dict[file_type]
            spike_emg_flow(data_dir, file_type)
    else:
        for file_type in file_types:
            data_dir = data_dirs_dict[file_type]
            try:
                spike_emg_flow(data_dir, file_type)
            except:
                print('Failed to run spike+emg test')


@flow(log_prints=True)
def bsa_only_test():
    if break_bool:
        for file_type in file_types:
            data_dir = data_dirs_dict[file_type]
            for data_type in ['emg', 'emg_spike']:
                print(f"""Running BSA test with
                      file type : {file_type}
                      data type : {data_type}""")
                prep_data_flow(file_type, data_type=data_type)
                run_emg_freq_test(data_dir, use_BSA=1)
    else:
        for file_type in file_types:
            data_dir = data_dirs_dict[file_type]
            for data_type in ['emg', 'emg_spike']:
                print(f"""Running BSA test with
                      file type : {file_type}
                      data type : {data_type}""")
                try:
                    prep_data_flow(file_type, data_type=data_type)
                except:
                    print('Failed to prep data')
                try:
                    run_emg_freq_test(data_dir, use_BSA=1)
                except:
                    print('Failed to run emg BSA test')


@flow(log_prints=True)
def stft_only_test():
    if break_bool:
        for file_type in file_types:
            data_dir = data_dirs_dict[file_type]
            for data_type in ['emg', 'emg_spike']:
                print(f"""Running STFT test with
                      file type : {file_type}
                      data type : {data_type}""")
                prep_data_flow(file_type, data_type=data_type)
                run_emg_freq_test(data_dir, use_BSA=0)
    else:
        for file_type in file_types:
            data_dir = data_dirs_dict[file_type]
            for data_type in ['emg', 'emg_spike']:
                print(f"""Running STFT test with
                      file type : {file_type}
                      data type : {data_type}""")
                try:
                    prep_data_flow(file_type, data_type=data_type)
                except:
                    print('Failed to prep data')
                try:
                    run_emg_freq_test(data_dir, use_BSA=0)
                except:
                    print('Failed to run emg STFT test')


@flow(log_prints=True)
def run_EMG_QDA_test():
    if break_bool:
        for file_type in file_types:
            data_dir = data_dirs_dict[file_type]
            for data_type in ['emg', 'emg_spike']:
                print(f"""Running EMG QDA test with
                      file type : {file_type}
                      data type : {data_type}""")
                prep_data_flow(file_type, data_type=data_type)
                run_emg_main_test(data_dir)
                os.chdir(os.path.join(blech_clust_dir,
                         'emg', 'gape_QDA_classifier'))
                run_gapes_Li(data_dir)
    else:
        for file_type in file_types:
            data_dir = data_dirs_dict[file_type]
            for data_type in ['emg', 'emg_spike']:
                print(f"""Running EMG QDA test with
                      file type : {file_type}
                      data type : {data_type}""")
                try:
                    prep_data_flow(file_type, data_type=data_type)
                except:
                    print('Failed to prep data')
                try:
                    run_emg_main_test(data_dir)
                    os.chdir(os.path.join(blech_clust_dir,
                             'emg', 'gape_QDA_classifier'))
                    run_gapes_Li(data_dir)
                except:
                    print('Failed to run QDA test')


@flow(log_prints=True)
def run_emg_freq_only():
    if break_bool:
        bsa_only_test()
        stft_only_test()
    else:
        try:
            bsa_only_test()
        except:
            print('Failed to run BSA test')
        try:
            stft_only_test()
        except:
            print('Failed to run STFT test')


@flow(log_prints=True)
def emg_only_test():
    if break_bool:
        run_emg_freq_only()
        run_EMG_QDA_test()
    else:
        try:
            run_emg_freq_only()
        except:
            print('Failed to run emg freq test')
        try:
            run_EMG_QDA_test()
        except:
            print('Failed to run QDA test')


@flow(log_prints=True)
def full_test():
    if break_bool:
        spike_only_test()
        emg_only_test()
        spike_emg_test()
    else:
        try:
            spike_only_test()
        except:
            print('Failed to run spike test')
        try:
            emg_only_test()
        except:
            print('Failed to run emg test')
        try:
            spike_emg_test()
        except:
            print('Failed to run spike+emg test')


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
