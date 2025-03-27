"""
Creating a Prefect pipeline for running tests
Run python scripts using subprocess as prefect tasks
"""

############################################################

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
        raise_exception=False,
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
    parser.add_argument('--raise-exception', action='store_true',
                        help='Raise error if subprocess fails')
    parser.add_argument('--file_type',
                        help='File types to run tests on',
                        choices=['ofpc', 'trad', 'all'],
                        default='all', type=str)
    parser.add_argument('--dummy-upload', action='store_true',
                        help='Run dummy upload test')
    args = parser.parse_args()
    script_path = os.path.realpath(__file__)

from subprocess import PIPE, Popen  # noqa
from prefect import flow, task  # noqa
from glob import glob  # noqa
import json  # noqa
import sys  # noqa
from PIL import Image  # noqa
from io import BytesIO  # noqa
from create_exp_info_commands import command_dict  # noqa
from switch_auto_car import set_auto_car  # noqa

blech_clust_dir = os.path.dirname(os.path.dirname(script_path))
sys.path.append(blech_clust_dir)
import utils.blech_utils as bu  # noqa

# S3 configuration
S3_BUCKET = os.getenv('BLECH_S3_BUCKET', 'blech-pipeline-outputs')

# GitHub Actions configuration
GITHUB_ACTIONS = os.environ.get('GITHUB_ACTIONS') == 'true'

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


def compress_image(image_path, max_size_kb=50):
    """Compress image to a maximum size in KB.

    Args:
        image_path (str): Path to the image file
        max_size_kb (int): Maximum size in KB

    Returns:
        bool: True if compression was successful, False otherwise
    """
    try:
        # Check if file exists and is an image
        if not os.path.exists(image_path):
            return False

        # Check current file size
        current_size = os.path.getsize(image_path)
        if current_size <= max_size_kb * 1024:
            return True  # Already small enough

        # Open the image
        img = Image.open(image_path)
        img_format = img.format if img.format else 'PNG'

        # If we get here, we couldn't compress enough with quality reduction alone
        # Try resizing the image
        width, height = img.size
        scale_factor = (max_size_kb * 1024) / current_size

        print(f'Scale factor: {scale_factor}')
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)

        temp_buffer = BytesIO()
        resized_img.save(temp_buffer, format=img_format,
                         quality=25, optimize=True)
        temp_size = temp_buffer.getbuffer().nbytes

        while temp_size > max_size_kb * 1024:
            # print(f'Scale factor: {scale_factor}')
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)

            temp_buffer = BytesIO()
            resized_img.save(temp_buffer, format=img_format,
                             quality=90, optimize=True)
            temp_size = temp_buffer.getbuffer().nbytes
            scale_factor *= 0.5  # Reduce scale factor for next iteration

        resized_img.save(image_path, format=img_format,
                         quality=90, optimize=True)
        # print(
        #     f"Compressed and resized {image_path} to {new_width}x{new_height} ({temp_size/1024:.1f}KB)")
        return True

    except Exception as e:
        print(f"Error compressing image {image_path}: {str(e)}")
        return False


def upload_test_results(data_dir, test_type, file_type, data_type=None):
    """Upload test results to S3 bucket and generate summary

    Args:
        data_dir (str): Directory containing results to upload
        test_type (str): Type of test (spike, emg, etc.)
        file_type (str): Type of file (ofpc, trad)
        data_type (str, optional): Type of data being tested (emg, spike, emg_spike)

    Returns:
        dict: Results from upload_to_s3 function
    """
    test_name = f"{test_type}_test"
    s3_dir = f"test_outputs/{os.path.basename(data_dir)}"

    # Compress all images before uploading
    print(f"Compressing images in {data_dir} before upload...")
    image_count = 0
    compressed_count = 0

    output_files = bu.find_output_files(data_dir)
    for file_list in output_files.values():
        for file in file_list:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                image_path = os.path.join(data_dir, file)
                image_count += 1
                if compress_image(image_path):
                    compressed_count += 1

    if image_count > 0:
        print(
            f"Compressed {compressed_count}/{image_count} images to max 50KB")

    try:
        # Upload files to S3
        upload_results = bu.upload_to_s3(data_dir, S3_BUCKET, s3_dir,
                                         add_timestamp=True, test_name=test_name, 
                                         data_type=data_type, file_type=file_type)

        # Generate summary
        summary_file = os.path.join(
            data_dir, f"{test_type}_{file_type}_s3_summary.md")
        # summary = bu.generate_github_summary(
        #     upload_results, summary_file, bucket_name=S3_BUCKET)

        # Add index.html link to summary if available
        if upload_results and upload_results.get('s3_directory'):
            index_url = f"https://{S3_BUCKET}.s3.amazonaws.com/{upload_results['s3_directory']}/index.html"
            # Include file_type and data_type in the summary
            data_type_str = f" ({data_type})" if data_type else ""
            index_summary = f"\n\n## {test_name} - {file_type}{data_type_str}\n\nView all files in this upload: [Index Page]({index_url})\n\n"

            # Append to summary file
            with open(summary_file, 'a') as f:
                f.write(index_summary)

            # # Append to summary string
            # summary += index_summary

        # If running in GitHub Actions, append to step summary
        if os.environ.get('GITHUB_STEP_SUMMARY'):
            with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as f:
                f.write(index_summary)

        return upload_results
    except Exception as e:
        print(f'Failed to upload results to S3: {str(e)}')
        return None


def dummy_upload_test_results():
    """Upload results without running tests"""
    file_type = 'ofpc'
    data_dir = data_dirs_dict[file_type]
    test_type = 'dummy'
    upload_test_results(data_dir, test_type, file_type, data_type='dummy_data')


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

                # Upload results to S3 even if test failed
                upload_test_results(data_dir, "spike", file_type)


@flow(log_prints=True)
def spike_emg_test():
    if break_bool:
        for file_type in file_types:
            data_dir = data_dirs_dict[file_type]
            spike_emg_flow(data_dir, file_type)

            # Upload results to S3 with data_type
            upload_test_results(data_dir, "spike_emg",
                                file_type, data_type="emg_spike")
    else:
        for file_type in file_types:
            data_dir = data_dirs_dict[file_type]
            try:
                spike_emg_flow(data_dir, file_type)
            except:
                print('Failed to run spike+emg test')

            # Upload results to S3 even if test failed
            upload_test_results(data_dir, "spike_emg", file_type)


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
                upload_test_results(
                    data_dir, "BSA", file_type, data_type=data_type)
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
                upload_test_results(data_dir, "BSA", file_type)


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
                upload_test_results(
                    data_dir, "STFT", file_type, data_type=data_type)
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
                upload_test_results(data_dir, "STFT", file_type)


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
                upload_test_results(
                    data_dir, "QDA", file_type, data_type=data_type)
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
                upload_test_results(data_dir, "QDA", file_type)


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
        spike_emg_test()
        emg_only_test()

    else:
        try:
            spike_emg_test()
        except:
            print('Failed to run spike+emg test')
        try:
            emg_only_test()
        except:
            print('Failed to run emg test')


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
