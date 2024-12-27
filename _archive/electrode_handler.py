"""
Class to handle merging and splitting during spike sorting
"""
import os
import tables
import numpy as np
import easygui
import ast
import sys
import re
import glob
import pylab as plt
import matplotlib.image as mpimg
from sklearn.mixture import GaussianMixture
import blech_waveforms_datashader

# Define a unit_descriptor class to be used to add things (anything!)
# about the sorted units to a pytables table
class unit_descriptor(tables.IsDescription):
        electrode_number = tables.Int32Col()
        single_unit = tables.Int32Col()
        regular_spiking = tables.Int32Col()
        fast_spiking = tables.Int32Col()
        waveform_count = tables.Int32Col()

class electrode_handler:
    def __init__(self, dir_name = None,
            electrode_num = None, solution_num = None, cluster_num = None):

        if dir_name is not None:
            self.dir_name = dir_name
        else:
            raise Exception ('dir_name is required')
        self.hf5_path = self.find_hf5_file(self.dir_name)
        if electrode_num == None:
            self.electrode_num = self.input_electrode_num()
        if solution_num == None:
            self.solution_num = self.input_solution_num()
        if cluster_num == None:
            self.cluster_num = self.input_cluster_num()

        # Print out selections
        print(f'<<< Electrode {self.electrode_num}, '\
                f'Solution {self.solution_num}, Cluster {self.cluster_num} >>>')

        # Define paths to load data
        key_names = ['waveforms','pca_slices','energy','amplitudes','spike_times','predictions']
        path_strings = [\
            f'./spike_waveforms/electrode{self.electrode_num}/spike_waveforms.npy',
            f'./spike_waveforms/electrode{self.electrode_num}/pca_waveforms.npy',
            f'./spike_waveforms/electrode{self.electrode_num}/energy.npy',
            f'./spike_waveforms/electrode{self.electrode_num}/spike_amplitudes.npy',
            f'./spike_times/electrode{self.electrode_num}/spike_times.npy',
            f'./clustering_results/electrode{self.electrode_num}'\
                                    f'/clusters{self.solution_num}/predictions.npy']
        path_strings_full = [os.path.join(self.dir_name, x) for x in path_strings]
        self.path_dict = dict(zip(key_names, path_strings_full))

    @staticmethod
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

    def input_electrode_num(self):
        # Get electrode number from user
        electrode_num_str, continue_bool = self.entry_checker(\
                msg = 'Electrode number :: ',
                check_func = str.isdigit,
                fail_response = 'Please enter an interger')
        if continue_bool:
            return int(electrode_num_str)

    def input_solution_num(self):
        num_solution_str, continue_bool = self.entry_checker(\
                msg = 'Solution number :: ',
                check_func = lambda x: (str.isdigit(x) and (1<int(x)<=7)),
                fail_response = 'Please enter an integer and 2<=x<=7')
        if continue_bool:
            return int(num_solution_str)

    def input_cluster_num(self):
        def cluster_check(x, max_val):
            clusters = re.findall('[0-9]+',x)
            all_digit_check = sum([i.isdigit() for i in clusters]) == len(clusters)
            bound_check = np.prod([int(x)<max_val for x in clusters])
            return all_digit_check and bound_check

        clusters_msg, continue_bool = self.entry_checker(\
                msg = 'Cluster numbers (anything separated) ::',
                check_func = lambda x: cluster_check(x,self.solution_num),
                fail_response = f'Please enter integers and numbers < {self.solution_num}')
        if continue_bool:
            clusters = re.findall('[0-9]+',clusters_msg)
            clusters = [int(x) for x in clusters]
        return clusters

    @staticmethod
    def find_hf5_file(dir_name):
        dir_name = os.path.abspath(dir_name)
        if dir_name[-1] != '/':
            dir_name += '/'
        files = glob.glob(os.path.join(dir_name,'*.h5'))
        if len(files) > 1:
            for num,x in enumerate(files):
                print(f'{num}) ' + x + '\n')
            file_select_msg, continue_bool = self.entry_checker(\
                    msg = 'Select file :: ',
                    check_func = cluster_check,
                    fail_response = 'Please enter integers')
            fin_file = files[int(file_select_msg)]
        elif len(files) == 0:
            raise Exception(f'No HDF5 files found in \n {dir_name}')
        else:
            fin_file = files[0]
        return fin_file

        # Load data from the chosen electrode and solution
    def load_data(self):
        self.spike_waveforms = np.load(self.path_dict['waveforms'])
        self.pca_slices = np.load(self.path_dict['pca_slices'])
        self.energy = np.load(self.path_dict['energy'])
        self.amplitudes = np.load(self.path_dict['amplitudes'])
        self.spike_times = np.load(self.path_dict['spike_times'])
        self.predictions = np.load(self.path_dict['predictions'])

        # Initiate indices for book-keeping
        self.ind_dict = dict(zip(np.sort(self.cluster_num),
            [np.where(self.predictions == x)[0] for x in self.cluster_num]))


    def data_display(self):
        """
        Displays selected clusters
        """
        # Re-show images of neurons so dumb people like Abu can make sure they
        # picked the right ones
        fig, ax = plt.subplots(len(self.cluster_num), 2)
        for cluster_num, cluster in enumerate(self.cluster_num):
            isi_plot = mpimg.imread(os.path.join(self.dir_name,
                    (f'Plots/{self.electrode_num}/{self.solution_num}_clusters_waveforms_ISIs/'\
                    f'Cluster{cluster}_ISIs.png')))
            waveform_plot =  mpimg.imread(os.path.join(self.dir_name,
                    f'Plots/{self.electrode_num}/{self.solution_num}_clusters_waveforms_ISIs/'\
                    f'Cluster{cluster}_waveforms.png'))
            if len(self.cluster_num) < 2:
                ax[0].imshow(isi_plot,aspect='auto');ax[0].axis('off')
                ax[1].imshow(waveform_plot,aspect='auto');ax[1].axis('off')
            else:
                ax[cluster_num, 0].imshow(isi_plot,aspect='auto');ax[cluster_num,0].axis('off')
                ax[cluster_num, 1].imshow(waveform_plot,aspect='auto');ax[cluster_num,1].axis('off')
        fig.suptitle('Are these the neurons you want to select?')
        fig.tight_layout()
        plt.show()
