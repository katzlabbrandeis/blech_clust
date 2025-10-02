"""
This module provides utility functions and classes to support the `blech_clust` processing, including logging, path handling, and metadata management.

- `Tee` class: Redirects output to both stdout/stderr and a log file.
  - `__init__`: Initializes the Tee object with a log file.
  - `write`: Writes data to both the log file and stdout.
  - `flush`: Flushes the log file.
  - `close`: Restores original stdout/stderr and closes the log file.

- `path_handler` class: Manages paths related to the `blech_clust` directory.
  - `__init__`: Initializes the path handler with the home directory and `blech_clust` directory path.

- `log_wait` decorator: Implements log-waiting to handle file access issues in parallel processes.

- `pipeline_graph_check` class: Ensures proper execution order of scripts based on a computation graph.
  - `__init__`: Initializes the object, loads the computation graph, and checks it.
  - `get_git_info`: Retrieves and prints git branch and commit information.
  - `load_graph`: Loads the computation graph from a file.
  - `make_full_path`: Constructs full file paths.
  - `check_graph`: Verifies the presence of all scripts in the computation graph.
  - `check_previous`: Checks if the previous script was executed successfully.
  - `write_to_log`: Writes script execution attempts and completions to a log file.

- `entry_checker` function: Validates user input against a check function, allowing exit with "x".

- `imp_metadata` class: Manages metadata for a given directory, including file lists and parameters.
  - `__init__`: Initializes the object and loads metadata.
  - `get_dir_name`: Retrieves the directory name from arguments or user input.
  - `get_file_list`: Lists files in the directory.
  - `get_hdf5_name`: Finds the HDF5 file in the directory.
  - `get_params_path`: Finds the parameters file in the directory.
  - `get_layout_path`: Finds the layout CSV file in the directory.
  - `load_params`: Loads parameters from the parameters file.
  - `get_info_path`: Finds the info file in the directory.
  - `load_info`: Loads information from the info file.
  - `load_layout`: Loads layout data from the layout CSV file.
"""
import easygui
import os
import glob
import json
import pandas as pd
import sys
from datetime import datetime
import time
import boto3
from typing import Dict, List
import shutil
try:
    import git
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False


class Tee:
    """Tee output to both stdout/stderr and a log file"""

    def __init__(self, data_dir, name='output.log'):
        self.log_path = os.path.join(data_dir, name)
        self.file = open(self.log_path, 'a')
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        self.file.flush()
        self.stdout.flush()

    def flush(self):
        self.file.flush()

    def close(self):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.file.close()


class path_handler():

    def __init__(self):
        self.home_dir = os.getenv('HOME')
        file_path = os.path.abspath(__file__)
        blech_clust_dir = ('/').join(file_path.split('/')[:-2])
        self.blech_clust_dir = blech_clust_dir

# If multiple processes are running and trying to access the file (parallel steps)
# this will cause errors when trying to write to the file
# Use decorator to implement log-waiting if there are issues


def log_wait(func):
    wait_times = [0.1, 0.5, 1, 2, 5, 10]

    def wrapper(*args, **kwargs):
        for wait_time in wait_times:
            try:
                func(*args, **kwargs)
                break
            except Exception as e:
                print(f'Exception : {e}')
                print(
                    f'Unable to access log with func : {func.__name__}, waiting {wait_time} seconds')
                time.sleep(wait_time)
            else:
                raise Exception(
                    f'Unable to access log with func : {func.__name__}')
    return wrapper


class pipeline_graph_check():
    """
    Check that parent scripts executed properly before running child scripts

    1) Check that computation graph is present
    2) Check that all scripts mentioned in computation graph are present
    3) For current run script, check that previous run script is present and executed successfully
    4) If prior exeuction is not present or failed, generate warning, give user option to override, else exit
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.tee = Tee(data_dir)
        self.load_graph()
        self.get_git_info()
        self.check_graph()

    def get_git_info(self):
        """
        Get branch and commit info, and print
        If not in git repo, print warning
        """
        if not hasattr(self, 'blech_clust_dir'):
            print('Run load_graph() first')
            exit()
        pwd = os.getcwd()
        # Change to blech_clust directory
        os.chdir(self.blech_clust_dir)
        git_branch = os.popen('git rev-parse --abbrev-ref HEAD').read().strip()
        git_commit = os.popen('git rev-parse HEAD').read().strip()
        if git_branch == '' or git_commit == '':
            print(
                'Not in git repository, please clone blech_clust rather than downloading zip')
        self.git_str = f'Git branch: {git_branch}\nGit commit: {git_commit}'
        # change back to original directory
        os.chdir(pwd)

    @log_wait
    def load_graph(self):
        """
        Load computation graph from file, if file is present
        """
        this_path_handler = path_handler()
        self.blech_clust_dir = this_path_handler.blech_clust_dir
        graph_path = os.path.join(
            self.blech_clust_dir,
            'params',
            'dependency_graph.json')
        if os.path.exists(graph_path):
            with open(graph_path, 'r') as graph_file_connect:
                self.graph = json.load(graph_file_connect)['graph']
        else:
            raise FileNotFoundError(
                f'Dependency graph not found, looking for {graph_path}')

    def make_full_path(self, x, dir_name):
        return os.path.join(self.blech_clust_dir, dir_name, x)

    @log_wait
    def check_graph(self):
        """
        Check that all scripts mentioned in computation graph are present
        Also, flatten out the graph for easier checking downstream
        """
        # First, flatten out the graph and get all keys and values
        # First level = directories
        # Second level = parent scripts : child scripts
        directory_names = list(self.graph.keys())
        directories = [self.graph[x] for x in directory_names]
        flat_graph = {}
        all_files = []

        for dir_name, this_dir in zip(directory_names, directories):
            for this_parent, this_children in this_dir.items():
                this_parent_full = self.make_full_path(this_parent, dir_name)
                all_files.append(this_parent_full)
                if type(this_children) == list:
                    this_children_full = [self.make_full_path(
                        x, dir_name) for x in this_children]
                    for this_child in this_children_full:
                        all_files.append(this_child)
                else:
                    this_children_full = self.make_full_path(
                        this_children, dir_name)
                    all_files.append(this_children_full)
                flat_graph[this_parent_full] = this_children_full

        self.flat_graph = flat_graph

        # Now check that all files are present
        all_files_present = [os.path.exists(x) for x in all_files]
        if all(all_files_present):
            return True
        else:
            missing_files = [x for x, y in zip(
                all_files, all_files_present) if not y]
            raise FileNotFoundError(f'Missing files ::: {missing_files}')

    @log_wait
    def check_previous(self, script_path):
        """
        Check that previous run script is present and executed successfully
        """
        # Check that script_path is present in flat_graph
        if script_path in self.flat_graph.keys():
            # Check that parent script is present in log
            parent_script = self.flat_graph[script_path]
            # If parent_script is not list, convert to list
            if type(parent_script) != list:
                parent_script = [parent_script]
            # Check that parent script is present in log
            self.log_path = os.path.join(self.data_dir, 'execution_log.json')
            if os.path.exists(self.log_path):
                with open(self.log_path, 'r') as log_file_connect:
                    log_dict = json.load(log_file_connect)
                if any([x in log_dict['completed'].keys() for x in parent_script]):
                    return True
                else:
                    raise ValueError(
                        f'Parent script [{parent_script}] not found in log')
        else:
            raise ValueError(
                f'Script path [{script_path}] not found in flat graph')

    @log_wait
    def write_to_log(self, script_path, type='attempted'):
        """
        Write to log file

        type = 'attempted' : script was attempted
        type = 'completed' : script was completed
        """
        if not hasattr(self, 'git_str'):
            raise ValueError('Run get_git_info() first')
        self.log_path = os.path.join(self.data_dir, 'execution_log.json')
        current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if os.path.exists(self.log_path):
            with open(self.log_path, 'r') as log_file_connect:
                log_dict = json.load(log_file_connect)
        else:
            log_dict = {}
        if type == 'attempted':
            if 'attempted' not in log_dict.keys():
                log_dict['attempted'] = {}
            log_dict['attempted'][script_path] = current_datetime
            print('============================================================')
            print(
                f'Attempting {os.path.basename(script_path)}, started at {current_datetime}')
            print(self.git_str)
            print('============================================================')
        elif type == 'completed':
            if 'completed' not in log_dict.keys():
                log_dict['completed'] = {}
            log_dict['completed'][script_path] = current_datetime
            print('============================================================')
            print(
                f'Completed {os.path.basename(script_path)}, ended at {current_datetime}')
            print('============================================================')
            # Close tee when completing
            if hasattr(self, 'tee'):
                self.tee.close()
        # log_dict[script_path] = current_datetime
        with open(self.log_path, 'w') as log_file_connect:
            json.dump(log_dict, log_file_connect, indent=4)


def entry_checker(
        msg,
        check_func,
        fail_response,
        default_input='',
):
    check_bool = False
    continue_bool = True
    exit_str = '"x" to exit :: '
    while not check_bool:
        msg_input = input(msg.join([' ', exit_str]))
        if msg_input == 'x':
            continue_bool = False
            break
        elif msg_input == '' and default_input != '':
            msg_input = default_input
        check_bool = check_func(msg_input)
        if not check_bool:
            print(fail_response)
    return msg_input, continue_bool


def find_output_files(data_dir: str) -> Dict[str, List[str]]:
    """Find all output files that should be uploaded to S3.

    Args:
        data_dir (str): The directory to search for files

    Returns:
        dict: Dictionary mapping extensions to lists of file paths
    """
    file_types = ['*.png', '*.txt', '*.csv', '*.params',
                  '*.info', '*.log', '*.json', '*.html']
    found_files = {ext: [] for ext in file_types}

    for ext in file_types:
        found_files[ext] = glob.glob(os.path.join(
            data_dir, '**', ext), recursive=True)

    return found_files


def upload_to_s3(local_directory: str, bucket_name: str, s3_directory: str,
                 add_timestamp: bool, test_name: str, data_type: str, file_type: str = None) -> dict:
    """Upload files to S3 bucket preserving directory structure.

    Args:
        local_directory (str): Local directory containing files to upload
        bucket_name (str): Name of S3 bucket
        s3_directory (str): Directory prefix in S3 bucket
        add_timestamp (bool): Whether to add a timestamp to the S3 directory
        test_name (str): Name of the test to include in the S3 directory
        data_type (str): Type of data being tested (emg, spike, emg_spike)
        file_type (str, optional): Type of file (ofpc, trad)

    Returns:
        dict: Dictionary containing:
            - 's3_directory': The S3 directory path where files were uploaded
            - 'uploaded_files': List of dictionaries with file info (local_path, s3_path, s3_url)
    """
    try:
        s3_client = boto3.client('s3')
        uploaded_files = []

        # Add timestamp, test name, file type, and data type to S3 directory if requested
        if add_timestamp:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            if file_type:
                s3_directory = f"{s3_directory}/{timestamp}_{test_name}_{file_type}_{data_type}"
            else:
                s3_directory = f"{s3_directory}/{timestamp}_{test_name}_{data_type}"

        # Find all output files
        files_dict = find_output_files(local_directory)

        # Count total files to upload
        total_files = sum(len(files) for files in files_dict.values())
        uploaded_count = 0

        # Upload each file
        for ext, file_list in files_dict.items():
            for local_path in file_list:
                # Get path relative to local_directory
                relative_path = os.path.relpath(local_path, local_directory)
                # Create S3 path preserving structure
                s3_path = os.path.join(s3_directory, relative_path)
                # Replace backslashes with forward slashes for S3
                s3_path = s3_path.replace('\\', '/')

                # Upload the file
                uploaded_count += 1
                # print(
                #     f"Uploading {uploaded_count}/{total_files}: {local_path} to s3://{bucket_name}/{s3_path}")
                s3_client.upload_file(local_path, bucket_name, s3_path)

                # Generate S3 URL
                s3_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_path}"

                # Add file info to uploaded_files list
                uploaded_files.append({
                    'local_path': local_path,
                    'relative_path': relative_path,
                    's3_path': s3_path,
                    's3_url': s3_url
                })

        # Generate and upload index.html
        if uploaded_files:
            # Create index.html content with file_type and data_type info
            index_html_content = generate_index_html(
                uploaded_files, s3_directory, bucket_name, local_directory)

            # Create a temporary file for index.html
            index_html_path = os.path.join(local_directory, 'index.html')
            with open(index_html_path, 'w') as f:
                f.write(index_html_content)

            # Upload index.html to S3
            s3_path = f"{s3_directory}/index.html"
            print(f"Uploading index.html to s3://{bucket_name}/{s3_path}")
            s3_client.upload_file(index_html_path, bucket_name, s3_path, ExtraArgs={
                                  'ContentType': 'text/html'})

            # Add index.html to uploaded_files list
            s3_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_path}"
            uploaded_files.append({
                'local_path': index_html_path,
                'relative_path': 'index.html',
                's3_path': s3_path,
                's3_url': s3_url
            })

            # Print the URL to the index.html
            print(f"Index page available at: {s3_url}")

        print(
            f"Successfully uploaded {uploaded_count + 1} files to s3://{bucket_name}/{s3_directory}")

        return {
            's3_directory': s3_directory,
            'uploaded_files': uploaded_files
        }

    except Exception as e:
        print(f"Error uploading to S3: {str(e)}")
        return {'s3_directory': None, 'uploaded_files': []}


def generate_index_html(uploaded_files: list, s3_directory: str, bucket_name: str, local_directory: str) -> str:
    """Generate an index.html file for S3 directory listing.

    Args:
        uploaded_files (list): List of dictionaries with file info
        s3_directory (str): The S3 directory path
        bucket_name (str): Name of the S3 bucket
        local_directory (str): Local directory containing files

    Returns:
        str: HTML content as a string
    """
    # Skip index.html itself
    filtered_files = [f for f in uploaded_files if os.path.basename(
        f['local_path']) != 'index.html']

    # Group files by directory for better organization
    files_by_dir = {}
    for file_info in filtered_files:
        relative_path = file_info.get('relative_path', os.path.relpath(
            file_info['local_path'], local_directory))
        dir_path = os.path.dirname(relative_path)
        if dir_path == '':
            dir_path = 'root'

        if dir_path not in files_by_dir:
            files_by_dir[dir_path] = []
        files_by_dir[dir_path].append(file_info)

    # Create HTML content
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Index of {s3_directory}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ text-align: left; padding: 8px; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        th {{ background-color: #4CAF50; color: white; }}
        a {{ text-decoration: none; color: #0066cc; }}
        a:hover {{ text-decoration: underline; }}
        .directory {{ font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Index of s3://{bucket_name}/{s3_directory}</h1>
    <p>This directory contains files uploaded from a blech_clust pipeline run.</p>
"""

    # Add directory structure
    html += "    <h2>Directory Structure</h2>\n"

    # Sort directories to ensure root comes first, then alphabetical
    sorted_dirs = sorted(files_by_dir.keys(),
                         key=lambda x: (0 if x == 'root' else 1, x))

    for dir_path in sorted_dirs:
        display_path = dir_path if dir_path != 'root' else '/'
        html += f"    <h3 class='directory'>{display_path}</h3>\n"
        html += "    <table>\n"
        html += "        <tr><th>File</th><th>Size</th><th>Type</th></tr>\n"

        # Sort files by name
        files = sorted(
            files_by_dir[dir_path], key=lambda x: os.path.basename(x['local_path']))

        for file_info in files:
            filename = os.path.basename(file_info['local_path'])
            ext = os.path.splitext(filename)[1]
            if not ext:
                ext = 'no_extension'

            # Get file size if available
            try:
                size = os.path.getsize(file_info['local_path'])
                if size < 1024:
                    size_str = f"{size} B"
                elif size < 1024 * 1024:
                    size_str = f"{size/1024:.1f} KB"
                else:
                    size_str = f"{size/(1024*1024):.1f} MB"
            except:
                size_str = "Unknown"

            # Create relative URL for the file
            relative_url = file_info['relative_path']

            html += f"        <tr><td><a href=\"{relative_url}\">{filename}</a></td><td>{size_str}</td><td>{ext}</td></tr>\n"

        html += "    </table>\n"

    # Also group files by extension for alternative view
    html += "    <h2>Files by Type</h2>\n"
    files_by_ext = {}
    for file_info in filtered_files:
        ext = os.path.splitext(file_info['local_path'])[1]
        if not ext:
            ext = 'no_extension'
        if ext not in files_by_ext:
            files_by_ext[ext] = []
        files_by_ext[ext].append(file_info)

    # Sort extensions alphabetically
    for ext in sorted(files_by_ext.keys()):
        files = files_by_ext[ext]
        html += f"    <h3>{ext.upper()} Files</h3>\n"
        html += "    <table>\n"
        html += "        <tr><th>File</th><th>Path</th><th>Size</th></tr>\n"

        # Sort files by path
        files = sorted(files, key=lambda x: x['relative_path'])

        for file_info in files:
            filename = os.path.basename(file_info['local_path'])
            relative_path = file_info.get('relative_path', '')
            dir_path = os.path.dirname(relative_path)

            # Get file size if available
            try:
                size = os.path.getsize(file_info['local_path'])
                if size < 1024:
                    size_str = f"{size} B"
                elif size < 1024 * 1024:
                    size_str = f"{size/1024:.1f} KB"
                else:
                    size_str = f"{size/(1024*1024):.1f} MB"
            except:
                size_str = "Unknown"

            html += f"        <tr><td><a href=\"{relative_path}\">{filename}</a></td><td>{dir_path}</td><td>{size_str}</td></tr>\n"

        html += "    </table>\n"

    html += """    <p><small>Generated by blech_clust pipeline</small></p>
</body>
</html>
"""
    return html


class imp_metadata():
    def __init__(self, args):
        self.dir_name = self.get_dir_name(args)
        self.get_file_list()
        self.get_hdf5_name()
        self.load_params()
        self.load_info()
        self.load_layout()

    def get_dir_name(self, args):
        if len(args) > 1:
            dir_name = os.path.abspath(args[1])
            if dir_name[-1] != '/':
                dir_name += '/'
        else:
            dir_name = easygui.diropenbox(msg='Please select data directory')
        return dir_name

    def get_file_list(self,):
        self.file_list = os.listdir(self.dir_name)

    def get_hdf5_name(self,):
        file_list = glob.glob(os.path.join(self.dir_name, '**.h5'))
        if len(file_list) > 0:
            self.hdf5_name = file_list[0]
        else:
            print('No HDF5 file found')

    def get_params_path(self,):
        file_list = glob.glob(os.path.join(self.dir_name, '**.params'))
        if len(file_list) > 0:
            self.params_file_path = file_list[0]
        else:
            print('No PARAMS file found')

    def get_layout_path(self,):
        file_list = glob.glob(os.path.join(self.dir_name, '**layout.csv'))
        if len(file_list) > 0:
            self.layout_file_path = file_list[0]
        else:
            print('No LAYOUT file found')

    def load_params(self,):
        self.get_params_path()
        if 'params_file_path' in dir(self):
            with open(self.params_file_path, 'r') as params_file_connect:
                self.params_dict = json.load(params_file_connect)

    def get_info_path(self,):
        file_list = glob.glob(os.path.join(self.dir_name, '**.info'))
        if len(file_list) > 0:
            self.info_file_path = file_list[0]
        else:
            print('No INFO file found')

    def load_info(self,):
        self.get_info_path()
        if 'info_file_path' in dir(self):
            with open(self.info_file_path, 'r') as info_file_connect:
                self.info_dict = json.load(info_file_connect)

    def load_layout(self,):
        self.get_layout_path()
        if 'layout_file_path' in dir(self):
            self.layout = pd.read_csv(self.layout_file_path, index_col=0)


def ifisdir_rmdir(dir_name):
    """Remove directory if it exists"""
    if os.path.isdir(dir_name):
        shutil.rmtree(dir_name)


class RepoUpdater:
    """
    Auto-update functionality for blech_clust repository using GitPython.

    Checks for updates from the remote repository and optionally updates
    the local repository. Can be silenced via configuration.
    """

    def __init__(self, repo_path=None, config_path=None):
        """
        Initialize the RepoUpdater.

        Args:
            repo_path (str, optional): Path to the git repository.
                                     If None, uses the blech_clust directory.
            config_path (str, optional): Path to configuration file.
                                       If None, looks for config in standard locations.
        """
        self.repo_path = repo_path or self._get_blech_clust_dir()
        self.config_path = config_path
        self.config = self._load_config()
        self.repo = None

        if GIT_AVAILABLE:
            try:
                self.repo = git.Repo(self.repo_path)
            except (git.InvalidGitRepositoryError, git.NoSuchPathError):
                print(
                    f"Warning: {self.repo_path} is not a valid git repository")
                self.repo = None
        else:
            print("Warning: GitPython not available. Auto-update functionality disabled.")

    def _get_blech_clust_dir(self):
        """Get the blech_clust directory path."""
        file_path = os.path.abspath(__file__)
        return os.path.dirname(os.path.dirname(file_path))

    def _load_config(self):
        """Load configuration settings."""
        config = {
            'auto_update_enabled': True,
            'auto_update_silent': False,
            'check_frequency_hours': 24
        }

        # Look for config file in multiple locations
        config_locations = []
        if self.config_path:
            config_locations.append(self.config_path)

        # Add standard config locations
        config_locations.extend([
            os.path.join(self.repo_path, 'config.json'),
            os.path.join(self.repo_path, '.blech_config.json'),
            os.path.join(os.path.expanduser('~'), '.blech_config.json')
        ])

        for config_file in config_locations:
            if os.path.exists(config_file):
                try:
                    with open(config_file, 'r') as f:
                        user_config = json.load(f)
                        config.update(user_config)
                    break
                except (json.JSONDecodeError, IOError) as e:
                    print(
                        f"Warning: Could not load config from {config_file}: {e}")

        return config

    def is_update_check_needed(self):
        """
        Check if an update check is needed based on frequency settings.

        Returns:
            bool: True if update check is needed, False otherwise.
        """
        if not self.config.get('auto_update_enabled', True):
            return False

        last_check_file = os.path.join(self.repo_path, '.last_update_check')

        if not os.path.exists(last_check_file):
            return True

        try:
            with open(last_check_file, 'r') as f:
                last_check_time = float(f.read().strip())

            hours_since_check = (time.time() - last_check_time) / 3600
            return hours_since_check >= self.config.get('check_frequency_hours', 24)
        except (ValueError, IOError):
            return True

    def _update_last_check_time(self):
        """Update the last check time file."""
        last_check_file = os.path.join(self.repo_path, '.last_update_check')
        try:
            with open(last_check_file, 'w') as f:
                f.write(str(time.time()))
        except IOError as e:
            print(f"Warning: Could not update last check time: {e}")

    def check_for_updates(self):
        """
        Check if updates are available from the remote repository.

        Returns:
            dict: Dictionary containing update information:
                - 'updates_available': bool
                - 'current_commit': str
                - 'latest_commit': str
                - 'commits_behind': int
                - 'error': str (if any error occurred)
        """
        result = {
            'updates_available': False,
            'current_commit': None,
            'latest_commit': None,
            'commits_behind': 0,
            'error': None
        }

        if not self.repo:
            result['error'] = "Repository not available or GitPython not installed"
            return result

        try:
            # Get current commit
            current_commit = self.repo.head.commit.hexsha
            result['current_commit'] = current_commit

            # Fetch latest changes from remote
            origin = self.repo.remotes.origin
            origin.fetch()

            # Get the latest commit from the remote branch
            remote_branch = f"origin/{self.repo.active_branch.name}"
            if remote_branch in [ref.name for ref in self.repo.refs]:
                latest_commit = self.repo.refs[remote_branch].commit.hexsha
                result['latest_commit'] = latest_commit

                # Count commits behind
                commits_behind = list(self.repo.iter_commits(
                    f'{current_commit}..{latest_commit}'))
                result['commits_behind'] = len(commits_behind)
                result['updates_available'] = len(commits_behind) > 0
            else:
                result['error'] = f"Remote branch {remote_branch} not found"

        except Exception as e:
            result['error'] = str(e)

        self._update_last_check_time()
        return result

    def update_repository(self, force=False):
        """
        Update the repository to the latest version.

        Args:
            force (bool): If True, force update even if there are local changes.

        Returns:
            dict: Dictionary containing update result:
                - 'success': bool
                - 'message': str
                - 'error': str (if any error occurred)
        """
        result = {
            'success': False,
            'message': '',
            'error': None
        }

        if not self.repo:
            result['error'] = "Repository not available or GitPython not installed"
            return result

        try:
            # Check for local changes
            if self.repo.is_dirty() and not force:
                result['error'] = "Repository has local changes. Use force=True to override."
                return result

            # Get current branch
            current_branch = self.repo.active_branch.name

            # Pull latest changes
            origin = self.repo.remotes.origin
            pull_info = origin.pull(current_branch)

            if pull_info:
                result['success'] = True
                result['message'] = f"Successfully updated to latest version. {len(pull_info)} changes pulled."
            else:
                result['success'] = True
                result['message'] = "Repository is already up to date."

        except Exception as e:
            result['error'] = str(e)

        return result

    def auto_update_check(self):
        """
        Perform automatic update check and optionally update.
        This is the main method to be called during initialization.
        """
        if not self.config.get('auto_update_enabled', True):
            return

        if not self.is_update_check_needed():
            return

        if not self.config.get('auto_update_silent', False):
            print("Checking for blech_clust updates...")

        update_info = self.check_for_updates()

        if update_info.get('error'):
            if not self.config.get('auto_update_silent', False):
                print(f"Update check failed: {update_info['error']}")
            return

        if update_info.get('updates_available'):
            commits_behind = update_info.get('commits_behind', 0)
            if not self.config.get('auto_update_silent', False):
                print(
                    f"Updates available! Your blech_clust is {commits_behind} commits behind.")
                print("To update, run: git pull origin master")
                print(
                    "Or disable auto-update checks by setting 'auto_update_enabled': false in your config.")
        else:
            if not self.config.get('auto_update_silent', False):
                print("blech_clust is up to date.")


def check_for_repo_updates(config_path=None):
    """
    Convenience function to check for repository updates.

    Args:
        config_path (str, optional): Path to configuration file.
    """
    updater = RepoUpdater(config_path=config_path)
    updater.auto_update_check()
