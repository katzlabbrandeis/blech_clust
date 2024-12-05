"""
Utilities to support blech_clust processing
"""
import easygui
import os
import glob
import json
import pandas as pd
import sys
from datetime import datetime

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
        blech_clust_dir =  ('/').join(file_path.split('/')[:-2])
        self.blech_clust_dir = blech_clust_dir


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
        self.check_graph()

    def load_graph(self):
        """
        Load computation graph from file, if file is present
        """
        this_path_handler = path_handler()
        self.blech_clust_dir = this_path_handler.blech_clust_dir
        # self.blech_clust_dir = '/home/abuzarmahmood/Desktop/blech_clust'
        graph_path = os.path.join(
                self.blech_clust_dir, 
                'params', 
                'dependency_graph.json')
        if os.path.exists(graph_path):
            with open(graph_path, 'r') as graph_file_connect:
                self.graph = json.load(graph_file_connect)['graph']
        else:
            raise FileNotFoundError(f'Dependency graph not found, looking for {graph_path}')

    def make_full_path(self, x, dir_name):
        return os.path.join(self.blech_clust_dir, dir_name, x)

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
                    this_children_full = [self.make_full_path(x, dir_name) for x in this_children]
                    for this_child in this_children_full:
                        all_files.append(this_child)
                else:
                    this_children_full = self.make_full_path(this_children, dir_name)
                    all_files.append(this_children_full)
                flat_graph[this_parent_full] = this_children_full

        self.flat_graph = flat_graph

        # Now check that all files are present
        all_files_present = [os.path.exists(x) for x in all_files]
        if all(all_files_present):
            return True
        else:
            missing_files = [x for x, y in zip(all_files, all_files_present) if not y]
            raise FileNotFoundError(f'Missing files ::: {missing_files}')

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
                    raise ValueError(f'Parent script [{parent_script}] not found in log')
        else:
            raise ValueError(f'Script path [{script_path}] not found in flat graph')

    def write_to_log(self, script_path, type = 'attempted'):
        """
        Write to log file

        type = 'attempted' : script was attempted
        type = 'completed' : script was completed
        """
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
        elif type == 'completed':
            if 'completed' not in log_dict.keys():
                log_dict['completed'] = {}
            log_dict['completed'][script_path] = current_datetime
            # Close tee when completing
            if hasattr(self, 'tee'):
                self.tee.close()
        # log_dict[script_path] = current_datetime
        with open(self.log_path, 'w') as log_file_connect:
            json.dump(log_dict, log_file_connect, indent = 4)

def entry_checker(msg, check_func, fail_response):
    check_bool = False
    continue_bool = True
    exit_str = '"x" to exit :: '
    while not check_bool:
        msg_input = input(msg.join([' ',exit_str]))
        if msg_input == 'x':
            continue_bool = False
            break
        check_bool = check_func(msg_input)
        if not check_bool:
            print(fail_response)
    return msg_input, continue_bool


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
            dir_name = easygui.diropenbox(msg = 'Please select data directory')
        return dir_name

    def get_file_list(self,):
        self.file_list = os.listdir(self.dir_name)
        
    def get_hdf5_name(self,):
        file_list = glob.glob(os.path.join(self.dir_name,'**.h5'))
        if len(file_list) > 0:
            self.hdf5_name = file_list[0]
        else:
            print('No HDF5 file found')

    def get_params_path(self,):
        file_list = glob.glob(os.path.join(self.dir_name,'**.params'))
        if len(file_list) > 0:
            self.params_file_path = file_list[0]
        else:
            print('No PARAMS file found')

    def get_layout_path(self,):
        file_list = glob.glob(os.path.join(self.dir_name,'**layout.csv'))
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
        file_list = glob.glob(os.path.join(self.dir_name,'**.info'))
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
            self.layout = pd.read_csv(self.layout_file_path, index_col = 0)
