"""
Utilities to support blech_clust processing
"""
import easygui
import os
import glob
import json
import pandas as pd

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
