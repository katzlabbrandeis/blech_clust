"""
Creating a Prefect pipeline for running tests
Run python scripts using subprocess as prefect tasks
"""
import os
from subprocess import PIPE, Popen
from prefect import flow, task
from glob import glob
import json
import argparse

############################################################
parser = argparse.ArgumentParser(description='Run tests, default = Run all tests')
parser.add_argument('-e', action = 'store_true',
                    help = 'Run EMG test only')
parser.add_argument('-s', action = 'store_true',
                    help = 'Run spike sorting test only')
parser.add_argument('--freq', action = 'store_true',
                    help = 'Run freq test only')
parser.add_argument('--bsa', action = 'store_true',
                    help = 'Run BSA test only')
parser.add_argument('--stft', action = 'store_true',
                    help = 'Run STFT test only')
parser.add_argument('--qda', action = 'store_true',
                    help = 'Run QDA test only')
parser.add_argument('--all', action = 'store_true',
                    help = 'Run all tests')
args = parser.parse_args()

def raise_error_if_error(process, stderr, stdout):
    # Print current data_type
    current_data_type_path = os.path.join(data_dir, 'current_data_type.txt')
    if os.path.exists(current_data_type_path):
        with open(os.path.join(data_dir, 'current_data_type.txt'), 'r') as f:
            current_data_type = f.read()
        print('=== Current data type: ', current_data_type, ' ===\n\n')
    print('=== Process stdout ===\n\n')
    print(stdout.decode('utf-8'))
    print('=== Process stderr ===\n\n')
    if process.returncode:
        decode_err = stderr.decode('utf-8')
        raise Exception(decode_err)

############################################################
## Define paths 
############################################################
# Define paths
# TODO: Replace with call to blech_process_utils.path_handler
script_path = os.path.realpath(__file__)
blech_clust_dir = os.path.dirname(os.path.dirname(script_path))

# Read emg_env path
emg_params_path = os.path.join(blech_clust_dir, 'params', 'emg_params.json')
if not os.path.exists(emg_params_path):
    print('=== Environment params file not found. ===')
    print('==> Please copy [[ blech_clust/params/_templates/emg_params.json ]] to [[ blech_clust/params/env_params.json ]] and update as needed.')
    exit()
with open(emg_params_path) as f:
    env_params = json.load(f)
emg_env_path = env_params['emg_env']

data_subdir = 'pipeline_testing/test_data_handling/test_data/KM45_5tastes_210620_113227_new'
data_dir = os.path.join(blech_clust_dir, data_subdir)

############################################################
## Data Prep Scripts 
############################################################
def check_data_present():
    full_data_path = os.path.join(blech_clust_dir, data_subdir)
    if os.path.isdir(full_data_path):
        return True
    else:
        return False

@task(log_prints=True)
def download_test_data():
    if check_data_present():
        print('Data already present')
        return
    else:
        print('Downloading data')
        script_name = './pipeline_testing/test_data_handling/download_test_data.sh'
        process = Popen(["bash", script_name],
                                   stdout = PIPE, stderr = PIPE)
        stdout, stderr = process.communicate()
        raise_error_if_error(process,stderr,stdout)

@task(log_prints=True)
def prep_data_info(data_type = 'emg_spike'):
    """
    Prepares data directory with info according to inputs

    Args:
        data_type (str): Type of data to prepare. Options are 'emg', 'spike', 'emg_spike'
    """
    if data_type == 'emg':
        flag_str = '-emg'
    elif data_type == 'spike':
        flag_str = '-spike'
    elif data_type == 'emg_spike':
        flag_str = '-emg_spike'

    # Write out data_type to file
    current_data_type_path = os.path.join(data_dir, 'current_data_type.txt')
    print(f'Writing data type: {data_type} to {current_data_type_path}')
    with open(current_data_type_path, 'w') as f:
        f.write(data_type)

    script_name = './pipeline_testing/test_data_handling/prep_data_info.py' 
    # cmd_str = 'python ' + script_name + ' ' + '-emg_spike' + ' ' + data_dir
    cmd_str = 'python ' + script_name + ' ' + flag_str + ' ' + data_dir
    process = Popen(cmd_str, shell=True, stdout = PIPE, stderr = PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process,stderr,stdout)

############################################################
## Common Scripts
############################################################
@task(log_prints=True)
def reset_blech_clust():
    script_name = './pipeline_testing/reset_blech_clust.py'
    process = Popen(["python", script_name],
                               stdout = PIPE, stderr = PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process,stderr,stdout)

@task(log_prints=True)
def run_clean_slate(data_dir):
    script_name = 'blech_clean_slate.py'
    process = Popen(["python", script_name, data_dir],
                               stdout = PIPE, stderr = PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process,stderr,stdout)

@task(log_prints=True)
def mark_exp_info_success(data_dir):
    script_name = './pipeline_testing/mark_exp_info_success.py'
    process = Popen(["python", script_name, data_dir],
                               stdout = PIPE, stderr = PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process,stderr,stdout)

@task(log_prints=True)
def run_blech_clust(data_dir):
    script_name = 'blech_clust.py'
    process = Popen(["python", script_name, data_dir],
                               stdout = PIPE, stderr = PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process,stderr,stdout)

@task(log_prints=True)
def make_arrays(data_dir):
    script_name = 'blech_make_arrays.py'
    process = Popen(["python", script_name, data_dir],
                               stdout = PIPE, stderr = PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process,stderr,stdout)

############################################################
## Spike Only 
############################################################

@task(log_prints=True)
def run_CAR(data_dir):
    script_name = 'blech_common_avg_reference.py'
    process = Popen(["python", script_name, data_dir],
                               stdout = PIPE, stderr = PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process,stderr,stdout)

@task(log_prints=True)
def run_jetstream_bash(data_dir):
    script_name = 'blech_run_process.sh'
    process = Popen(["bash", script_name, data_dir],
                               stdout = PIPE, stderr = PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process,stderr,stdout)

@task(log_prints=True)
def select_clusters(data_dir):
    script_name = 'pipeline_testing/select_some_waveforms.py'
    process = Popen(["python", script_name, data_dir],
                               stdout = PIPE, stderr = PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process,stderr,stdout)

@task(log_prints=True)
def post_process(data_dir):
    script_name = 'blech_post_process.py'
    dir_pos = data_dir
    plot_flag = '-p ' + 'False'
    sorted_units_path = glob(os.path.join(data_dir, '*sorted_units.csv'))[0]
    file_flag = '-f' + sorted_units_path
    process = Popen(["python", script_name, dir_pos, plot_flag, file_flag],
                               stdout = PIPE, stderr = PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process,stderr,stdout)

@task(log_prints=True)
def quality_assurance(data_dir):
    script_name = 'blech_run_QA.sh'
    process = Popen(["bash", script_name, data_dir],
                               stdout = PIPE, stderr = PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process,stderr,stdout)

@task(log_prints=True)
def units_plot(data_dir):
    script_name = 'blech_units_plot.py'
    process = Popen(["python", script_name, data_dir],
                               stdout = PIPE, stderr = PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process,stderr,stdout)


@task(log_prints=True)
def make_psth(data_dir):
    script_name = 'blech_make_psth.py'
    process = Popen(["python", script_name, data_dir],
                               stdout = PIPE, stderr = PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process,stderr,stdout)

@task(log_prints=True)
def pal_iden_setup(data_dir):
    script_name = 'blech_palatability_identity_setup.py'
    process = Popen(["python", script_name, data_dir],
                               stdout = PIPE, stderr = PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process,stderr,stdout)


@task(log_prints=True)
def overlay_psth(data_dir):
    script_name = 'blech_overlay_psth.py'
    process = Popen(["python", script_name, data_dir],
                               stdout = PIPE, stderr = PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process,stderr,stdout)

############################################################
## EMG Only
############################################################
@task(log_prints=True)
def change_emg_freq_method(use_BSA = 1):
    script_name = 'pipeline_testing/change_emg_freq_method.py'
    process = Popen(["python", script_name, str(use_BSA)],
                               stdout = PIPE, stderr = PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process,stderr,stdout)

@task(log_prints=True)
def cut_emg_trials(data_dir):
    script_name = 'pipeline_testing/cut_emg_trials.py'
    process = Popen(["python", script_name, data_dir],
                               stdout = PIPE, stderr = PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process,stderr,stdout)

@task(log_prints=True)
def emg_filter(data_dir):
    script_name = 'emg_filter.py'
    process = Popen(["python", script_name, data_dir],
                               stdout = PIPE, stderr = PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process,stderr,stdout)

@task(log_prints=True)
def emg_freq_setup(data_dir):
    script_name = 'emg_freq_setup.py'
    process = Popen(["python", script_name, data_dir],
                               stdout = PIPE, stderr = PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process,stderr,stdout)

@task(log_prints=True)
def emg_jetstream_parallel(data_dir): 
    script_name = 'bash blech_emg_jetstream_parallel.sh'
    full_str = script_name
    process = Popen(full_str, shell = True, stdout = PIPE, stderr = PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process,stderr,stdout)


@task(log_prints=True)
def emg_freq_post_process(data_dir):
    script_name = 'emg_freq_post_process.py'
    process = Popen(["python", script_name, data_dir],
                               stdout = PIPE, stderr = PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process,stderr,stdout)

@task(log_prints=True)
def emg_freq_plot(data_dir):
    script_name = 'emg_freq_plot.py'
    process = Popen(["python", script_name, data_dir],
                               stdout = PIPE, stderr = PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process,stderr,stdout)

@task(log_prints=True)
def run_gapes_Li(data_dir):
    script_name = 'get_gapes_Li.py'
    process = Popen(["python", script_name, data_dir],
                               stdout = PIPE, stderr = PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process,stderr,stdout)

############################################################
## Define Flows
############################################################
@flow(log_prints=True)
def prep_data_flow(data_type = 'emg_spike'):
    os.chdir(blech_clust_dir)
    download_test_data()
    prep_data_info(data_type)

@flow(log_prints=True)
def run_spike_test():
    os.chdir(blech_clust_dir)
    reset_blech_clust()
    run_clean_slate(data_dir)
    mark_exp_info_success(data_dir)
    run_blech_clust(data_dir)
    run_CAR(data_dir)
    run_jetstream_bash(data_dir)
    select_clusters(data_dir)
    post_process(data_dir)
    quality_assurance(data_dir)
    units_plot(data_dir)
    make_arrays(data_dir)
    make_psth(data_dir)
    pal_iden_setup(data_dir)
    overlay_psth(data_dir)

@flow(log_prints=True)
def run_emg_main_test():
    os.chdir(blech_clust_dir)
    reset_blech_clust()
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
def run_emg_freq_test(use_BSA = 1):
    os.chdir(blech_clust_dir)
    # change_emg_freq_method needs to be in blech_clust_dir
    change_emg_freq_method(use_BSA = use_BSA)
    run_emg_main_test()
    emg_jetstream_parallel(data_dir) 
    emg_freq_post_process(data_dir)
    emg_freq_plot(data_dir)

@flow(log_prints=True)
def run_EMG_QDA_test():
    for data_type in ['emg', 'emg_spike']:
        print(f'Running QDA test with data type : {data_type}')
        try:
            prep_data_flow(data_type = data_type)
        except:
            print('Failed to prep data')
        try:
            run_emg_main_test()
            os.chdir(os.path.join(blech_clust_dir, 'emg', 'gape_QDA_classifier'))
            run_gapes_Li(data_dir)
        except:
            print('Failed to run QDA test')

@flow(log_prints=True)
def spike_only_test():
    for data_type in ['spike', 'emg_spike']:
        print(f'Running spike test with data type : {data_type}')
        try:
            prep_data_flow(data_type = data_type)
        except:
            print('Failed to prep data')
        try:
            run_spike_test()
        except:
            print('Failed to run spike test')

@flow(log_prints=True)
def bsa_only_test():
    for data_type in ['emg', 'emg_spike']:
        print(f'Running BSA test with data type : {data_type}')
        try:
            prep_data_flow(data_type = data_type)
        except:
            print('Failed to prep data')
        try:
            run_emg_freq_test(use_BSA=1)
        except:
            print('Failed to run emg BSA test')

@flow(log_prints=True)
def stft_only_test():
    for data_type in ['emg', 'emg_spike']:
        print(f'Running STFT test with data type : {data_type}')
        try:
            prep_data_flow(data_type = data_type)
        except:
            print('Failed to prep data')
        try:
            run_emg_freq_test(use_BSA=0)
        except:
            print('Failed to run emg STFT test')

@flow(log_prints=True)
def run_emg_freq_only():
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
    try:
        run_emg_freq_only()
    except:
        print('Failed to run emg freq test')
    try:
        run_EMG_QDA_test()
    except:
        print('Failed to run EMG QDA test')

@flow(log_prints=True)
def full_test():
    try:
        spike_only_test()
    except:
        print('Failed to run spike test')
    try:
        emg_only_test()
    except:
        print('Failed to run emg test')

############################################################
## Run Flows
############################################################
# If no individual tests are required, run both
if args.all: 
    print('Running spike and emg tests')
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
