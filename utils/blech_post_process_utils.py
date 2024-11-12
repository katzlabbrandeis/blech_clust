import os
import tables
import numpy as np
import easygui
import ast
import re
import pylab as plt
import matplotlib.image as mpimg
import argparse
import pandas as pd
import uuid
from utils.blech_utils import entry_checker, imp_metadata
from utils.blech_process_utils import gen_isi_hist
from utils import blech_waveforms_datashader
from datetime import datetime
from scipy.stats import chisquare
from tqdm import tqdm


class sort_file_handler():

    def __init__(self, sort_file_path):
        self.sort_file_path = sort_file_path
        if sort_file_path is not None:
            if not (sort_file_path[-3:] == 'csv'):
                raise Exception("Please provide CSV file")
            sort_table = pd.read_csv(sort_file_path)
            sort_table.fillna('',inplace=True)
            # Check when more than one cluster is specified
            sort_table['len_cluster'] = \
                    [len(re.findall('[0-9]+',str(x))) for x in sort_table.Cluster]
            
            # Get splits and merges out of the way first
            sort_table.sort_values(
                    ['len_cluster','Split'],
                    ascending=False, inplace=True)
            if 'level_0' in sort_table.columns:
                sort_table.drop(columns=['level_0'], inplace=True)
            sort_table.reset_index(inplace=True)
            sort_table['unit_saved'] = False
            self.sort_table = sort_table

            # Create generator for iterating through sort table
            self.sort_table_gen = self.sort_table.iterrows()
        else:
            self.sort_table = None

    def get_next_cluster(self):
        """
        Get the next cluster to process
        """
        try:
            counter, next_row = next(self.sort_table_gen)
        except StopIteration:
            return None, None, None, None 

        self.current_row = next_row

        electrode_num = int(self.current_row.Chan)
        num_clusters = int(self.current_row.Solution)
        clusters = re.findall('[0-9]+',str(self.current_row.Cluster))
        clusters = [int(x) for x in clusters]

        return counter, electrode_num, num_clusters, clusters

    def mark_current_unit_saved(self):
        self.sort_table.loc[self.current_row.name,'unit_saved'] = True
        # Write to disk
        self.sort_table.to_csv(self.sort_file_path, index=False)
        print('== Marked unit as saved ==')

def cluster_check(x):
    clusters = re.findall('[0-9]+',x)
    return sum([i.isdigit() for i in clusters]) == len(clusters)

def get_electrode_details(this_sort_file_handler):
    """
    Ask user for electrode number, number of clusters, and cluster numbers
    """

    if this_sort_file_handler.sort_table is not None:
        counter, electrode_num, num_clusters, clusters = \
                this_sort_file_handler.get_next_cluster()
        if counter is None:
            return False, None, None, None
        else:
            continue_bool = True
        print('== Got cluster number details from sort file ==')

    else:
        # Get electrode number from user
        electrode_num_str, continue_bool = entry_checker(\
                msg = 'Electrode number :: ',
                check_func = str.isdigit,
                fail_response = 'Please enter an interger')

        if continue_bool:
            electrode_num = int(electrode_num_str)
        else:
            return False, None, None, None

        num_clusters_str, continue_bool = entry_checker(\
                msg = 'Solution number :: ',
                check_func = str.isdigit, 
                fail_response = 'Please enter an interger')
        if continue_bool:
            num_clusters = int(num_clusters_str)
        else:
            return False, None, None, None


        clusters_msg, continue_bool = entry_checker(\
                msg = 'Cluster numbers (anything separated) ::',
                check_func = cluster_check,
                fail_response = 'Please enter integers')
        if continue_bool:
            clusters = re.findall('[0-9]+',clusters_msg)
            clusters = [int(x) for x in clusters]
        else:
            return False, None, None, None

    return continue_bool, electrode_num, num_clusters, clusters

def load_data_from_disk(electrode_num, num_clusters):
    """
    Load data from disk
    """

    loading_paths = [\
        f'./spike_waveforms/electrode{electrode_num:02}/spike_waveforms.npy',
        f'./spike_times/electrode{electrode_num:02}/spike_times.npy',
        f'./spike_waveforms/electrode{electrode_num:02}/pca_waveforms.npy',
        f'./spike_waveforms/electrode{electrode_num:02}/energy.npy',
        f'./spike_waveforms/electrode{electrode_num:02}/spike_amplitudes.npy',
        f'./clustering_results/electrode{electrode_num:02}/'\
                f'clusters{num_clusters}/predictions.npy',]

    loaded_dat = [np.load(x) for x in loading_paths]

    return loaded_dat


def gen_select_cluster_plot(electrode_num, num_clusters, clusters):
    """
    Generate plots for the clusters initially supplied by the user
    """
    fig, ax = plt.subplots(len(clusters), 2)
    for cluster_num, cluster in enumerate(clusters):
        isi_plot = mpimg.imread(
                './Plots/{:02}/clusters{}/'\
                                'Cluster{}_ISIs.png'\
                                .format(electrode_num, num_clusters, cluster)) 
        waveform_plot =  mpimg.imread(
                './Plots/{:02}/clusters{}/'\
                                'Cluster{}_waveforms.png'\
                                .format(electrode_num, num_clusters, cluster)) 
        if len(clusters) < 2:
            ax[0].imshow(isi_plot,aspect='auto');ax[0].axis('off')
            ax[1].imshow(waveform_plot,aspect='auto');ax[1].axis('off')
        else:
            ax[cluster_num, 0].imshow(isi_plot,aspect='auto');
            ax[cluster_num,0].axis('off')
            ax[cluster_num, 1].imshow(waveform_plot,aspect='auto');
            ax[cluster_num,1].axis('off')
    fig.suptitle('Are these the neurons you want to select? Press q to exit plot')
    fig.tight_layout()
    plt.show()

def generate_cluster_plots(
        split_predictions, 
        spike_waveforms, 
        spike_times, 
        n_clusters, 
        this_cluster,
        sampling_rate,
        ):
    """
    Generate grid of plots for each cluster

    Inputs:
        split_predictions: array of cluster numbers for each split 
        spike_waveforms: array of waveforms
        spike_times: array of spike times
        n_clusters: number of clusters
        this_cluster: cluster number to plot


    **NOTE**: This cluster specifies the cluster number in the original
    clustering, not the split clustering. Split cluster numbers are
    specified in split_predictions
    **NOTE**: This is a stupid way of doing this.
    """

    n_rows = int(np.ceil(np.sqrt(n_clusters)))
    n_cols = int(np.ceil(n_clusters/n_rows))
    fig, ax = plt.subplots(n_rows, n_cols, 
                           figsize = (10,10))

    for cluster in range(n_clusters):
        split_points = np.where(split_predictions == cluster)[0]
        # Waveforms and times from the chosen cluster
        slices_dejittered = spike_waveforms[this_cluster, :]            
        times_dejittered = spike_times[this_cluster]
        # Waveforms and times from the chosen split of the chosen cluster
        slices_dejittered = slices_dejittered[split_points, :]
        times_dejittered = times_dejittered[split_points]               

        generate_datashader_plot(
                slices_dejittered,
                times_dejittered,
                sampling_rate,
                title = f'Split Cluster {cluster}',
                ax = ax.flatten()[cluster],)

    for cluster in range(n_clusters, n_rows*n_cols):
        ax.flatten()[cluster].axis('off')

    plt.tight_layout()
    plt.show()

def get_clustering_params(this_sort_file_handler):
    """
    Ask user for clustering parameters
    """
    # Get clustering parameters from user
    if (this_sort_file_handler.sort_table is not None):
        dat_row = this_sort_file_handler.current_row
        split_val = re.findall('[0-9]+', str(dat_row.Split))
        if (len(split_val) > 0): 
            n_clusters = int(input(f'Number of clusters (default={split_val[0]}): ') or split_val[0])
        else:
            n_clusters = int(input('Number of clusters (default=5): ') or "5")
    else:    
        n_clusters = int(input('Number of clusters (default=5): ') or "5")
    
    fields = [
            'Max iterations',
            'Convergence criterion',
            'Number of random restarts']
    values = [100,0.001,10]
    fields_str = (
            f':: {fields[0]} (1000 is plenty) : {values[0]} \n' 
            f':: {fields[1]} (usually 0.0001) : {values[1]} \n' 
            f':: {fields[2]} (10 is plenty) : {values[2]}')
    print(fields_str) 
    edit_bool = 'a'
    edit_bool_msg, continue_bool = entry_checker(\
            msg = 'Use these parameters? (y/n)',
            check_func = lambda x: x in ['y','n'],
            fail_response = 'Please enter (y/n)')
    if continue_bool:
        if edit_bool_msg == 'y':
            n_iter = values[0] 
            thresh = values[1] 
            n_restarts = values[2] 

        elif edit_bool_msg == 'n': 
            clustering_params = easygui.multenterbox(msg = 'Fill in the'\
                    'parameters for re-clustering (using a GMM)', 
                    fields  = fields, values = values)
            n_iter = int(clustering_params[0])
            thresh = float(clustering_params[1])
            n_restarts = int(clustering_params[2]) 
    else:
        return False, None, None, None, None

    return continue_bool, n_clusters, n_iter, thresh, n_restarts


def get_split_cluster_choice(n_clusters):
    choice_list = tuple([str(i) for i in range(n_clusters)]) 

    chosen_msg, continue_bool = entry_checker(\
            msg = f'Please select from {choice_list} (anything separated) '\
            ':: "111" for all ::',
            check_func = cluster_check,
            fail_response = 'Please enter integers')
    if continue_bool:
        chosen_clusters = re.findall('[0-9]+|-[0-9]+',chosen_msg)
        chosen_split = [int(x) for x in chosen_clusters]
        negative_vals = re.findall('-[0-9]+',chosen_msg)
        # If 111, select all
        if 111 in chosen_split:
            chosen_split = range(n_clusters)
            # If any are negative, go into removal mode
        elif len(negative_vals) > 0:
            remove_these = [abs(int(x)) for x in negative_vals]
            chosen_split = [x for x in range(n_clusters) \
                    if x not in remove_these]
        print(f'Chosen splits {chosen_split}')
    else:
        return False, None

    return continue_bool, chosen_split


def prepare_data(
    this_cluster,
    pca_slices,
    energy,
    amplitudes,
    ):
    """
    Prepare data for clustering
    """

    n_pc = 3
    data = np.zeros((len(this_cluster), n_pc + 3))  
    data[:,3:] = pca_slices[this_cluster,:n_pc]
    data[:,0] = (energy[this_cluster]/np.max(energy[this_cluster])).flatten()
    data[:,1] = (np.abs(amplitudes[this_cluster])/\
            np.max(np.abs(amplitudes[this_cluster]))).flatten()

    return data

def clean_memory_monitor_data():
    """
    Clean memory monitor data
    """
    print('==============================')
    print('Cleaning memory monitor data')
    print()

    if not os.path.exists('./memory_monitor_clustering/memory_usage.txt'):
        file_list = os.listdir('./memory_monitor_clustering')
        f = open('./memory_monitor_clustering/memory_usage.txt', 'w')
        for files in file_list:
            try:
                mem_usage = np.loadtxt('./memory_monitor_clustering/' + files)
                print('electrode'+files[:-4], '\t', str(mem_usage)+'MB', file=f)
                os.system('rm ' + './memory_monitor_clustering/' + files)
            except:
                pass    
        f.close()
    print('==============================')

def get_ISI_violations(unit_times, sampling_rate):
    """
    Get ISI violations
    """

    ISIs = np.ediff1d(np.sort(unit_times))/(sampling_rate/1000)
    violations1 = 100.0*float(np.sum(ISIs < 1.0)/len(unit_times))
    violations2 = 100.0*float(np.sum(ISIs < 2.0)/len(unit_times))
    return violations1, violations2

def generate_datashader_plot(
        unit_waveforms,
        unit_times,
        sampling_rate,
        title = None,
        ax = None,
        ):
    """
    Generate datashader plot
    """
    violations1, violations2 = get_ISI_violations(unit_times, sampling_rate)

    # Show the merged cluster to the user, 
    # and ask if they still want to merge
    x = np.arange(len(unit_waveforms[0])) + 1
    if ax is None:
        fig, ax = blech_waveforms_datashader.\
                waveforms_datashader(unit_waveforms, x, downsample = False)
    else:
        fig, ax = blech_waveforms_datashader.\
                waveforms_datashader(unit_waveforms, x, 
                                     downsample = False, ax=ax)
    ax.set_xlabel('Sample (30 samples / ms)')
    ax.set_ylabel('Voltage (uV)')
    if title is not None:
        title_add = title
    else:
        title_add = ''
    print_str = (
        title_add + '\n' +\
        f'{violations2:.1f} % (<2ms), '
        f'{violations1:.1f} % (<1ms), '
        f'{len(unit_times)} total waveforms. \n') 
    ax.set_title(print_str)
    plt.tight_layout()

    return violations1, violations2, fig, ax

def plot_merged_units(
        cluster_waveforms,
        cluster_labels,
        cluster_times,
        sampling_rate,
        max_n_per_cluster = 1000,
        sd_bound = 1,
        ax = None,
        ):
    """
    Plot merged units

    Inputs:
        cluster_waveforms: list of arrays (n_waveforms, n_samples)
        cluster_labels: list of cluster labels
        max_n_per_cluster: maximum number of waveforms to plot per cluster
        sd_bound: number of standard deviations to plot for each cluster

    Outputs:
        fig, ax: figure and axis objects
    """

    violations1, violations2 = get_ISI_violations(cluster_times, sampling_rate)

    mean_waveforms = [x.mean(axis=0) for x in cluster_waveforms]
    sd_waveforms = [x.std(axis=0) for x in cluster_waveforms]

    n_clusters = len(cluster_waveforms)
    plot_inds = [
            np.random.choice(np.arange(len(x)), 
                             np.min([max_n_per_cluster, len(x)]),
                             replace = False) \
            for x in cluster_waveforms]

    cmap = plt.cm.get_cmap('Set1')

    if ax is None:
        fig, ax = plt.subplots(1,1, figsize = (7,7))
    else:
        fig = ax.get_figure()
    for i in range(n_clusters):
        inds = plot_inds[i]
        ax.plot(mean_waveforms[i], 
                color = cmap(i), 
                linewidth = 5,
                label = f'Cluster {cluster_labels[i]}',
                zorder = 11)
        ax.fill_between(np.arange(len(mean_waveforms[i])),
                mean_waveforms[i] - sd_bound*sd_waveforms[i],
                mean_waveforms[i] + sd_bound*sd_waveforms[i],
                color = cmap(i), alpha = 0.2,
                        zorder = 10)
        ax.plot(cluster_waveforms[i][plot_inds[i]].T, 
                color = cmap(i), alpha = 100/max_n_per_cluster)
    ax.set_xlabel('Sample (30 samples / ms)')
    ax.set_ylabel('Voltage (uV)')
    print_str = (
        f'{violations2:.1f} % (<2ms), '
        f'{violations1:.1f} % (<1ms), '
        f'{len(cluster_times)} total waveforms. \n') 
    ax.set_title('Merged units\n' + print_str)
    ax.legend()
    plt.tight_layout()
    
    return fig, ax

def gen_plot_auto_merged_clusters(
        spike_waveforms,
        spike_times,
        split_predictions,
        sampling_rate,
        final_merge_sets,
        new_clust_names,
        ):

    """
    Plot all merged clusters on sample plot
    **NOTE** This is different from plot_merged_units

    Inputs:
        spike_waveforms - (n_spikes, n_samples) array of spike waveforms
        spike_times - (n_spikes,) array of spike times
        split_predictions - (n_spikes,) array of cluster labels
        sampling_rate - sampling rate of the recording
        final_merge_sets - list of lists of clusters to merge

    Outputs:
        fig - matplotlib figure handle
        ax - matplotlib axis handle
    """

    # Plot merged clusters
    fig, ax = plt.subplots(1, len(final_merge_sets),
                           figsize = (len(final_merge_sets) * 5, 5))
    # Make sure ax is iterable
    if len(final_merge_sets) == 1:
        ax = [ax]
    for i, this_set in enumerate(final_merge_sets): 
        cluster_inds = [np.where(split_predictions == this_cluster)[0] \
                for this_cluster in this_set]
        cluster_waveforms = [spike_waveforms[this_inds] \
                for this_inds in cluster_inds]
        cluster_times = [spike_times[this_inds] \
                for this_inds in cluster_inds]


        fig, ax[i] = plot_merged_units(
                    cluster_waveforms,
                    this_set,
                    np.concatenate(cluster_times), 
                    sampling_rate,
                    max_n_per_cluster = 1000,
                    sd_bound = 1,
                    ax = ax[i],
                    )
    # Get titles for each ax
    ax_titles = [this_ax.get_title() for this_ax in ax]
    ax_titles = [f'New Cluster {new_name}'+'\n'+this_title \
            for new_name, this_title in zip(new_clust_names, ax_titles)] 

    # Set titles
    for this_ax, this_title in zip(ax, ax_titles):
        this_ax.set_title(this_title)

    # Add waveform counts to legend
    waveform_counts = [len(this_inds) for this_inds in cluster_inds]
    for this_ax in ax:
        current_legend_texts = this_ax.get_legend().get_texts() 
        new_legend_texts = [f'{this_text.get_text()} ({this_count})' \
                for this_text, this_count in zip(
                    current_legend_texts,
                    waveform_counts,
                    )]
        for this_text, new_text in zip(
                current_legend_texts,
                new_legend_texts,
                ):
            this_text.set_text(new_text)

    return fig, ax

def delete_raw_recordings(hdf5_name):
    """
    Delete raw recordings from hdf5 file

    Inputs:
        hf5: hdf5 file object
        hdf5_name: name of hdf5 file

    Outputs:
        hf5: new hdf5 file object
    """

    print('==============================')
    print("Removing raw recordings from hdf5 file")
    print()

    # Remove children node one a time so we can report progress
    with tables.open_file(hdf5_name, 'r+') as hf5:
        if '/raw' in hf5:
            removed_bool = False
            raw_children = hf5.list_nodes('/raw')
            len_raw_children = len(raw_children)
            # Get children nodes under /raw
            for i, child in enumerate(tqdm(raw_children)):
                child_name = child._v_name
                hf5.remove_node('/raw', child_name)
                tqdm.write(f"Removed {child_name}, {i+1}/{len_raw_children}")
            # Remove the raw recordings from the hdf5 file
            hf5.remove_node('/raw', recursive = 1)
            print("Raw recordings removed")
            print('==============================')
        else:
            removed_bool = True
            print("Raw recordings have already been removed, so moving on ..")
            print('==============================')

    if not removed_bool:
        os.system("ptrepack --chunkshape=auto --propindexes --complevel=9 "
            "--complib=blosc " + hdf5_name + " " + hdf5_name[:-3] + "_repacked.h5")
        # Delete the old (raw and big) hdf5 file
        os.system("rm " + hdf5_name)
        print("File repacked")
        return True
    else:
        return False

def generate_violations_warning(
        violations1,
        violations2,
        unit_times,
        ):
    print_str = (f':: Merged cluster \n'
        f':: {violations2:.1f} % (<2ms)\n'
        f':: {violations1:.1f} % (<1ms)\n'
        f':: {len(unit_times)} Total Waveforms \n' 
        ':: I want to still merge these clusters into one unit (y/n) :: ')
    proceed_msg, continue_bool = entry_checker(\
            msg = print_str, 
            check_func = lambda x: x in ['y','n'],
            fail_response = 'Please enter (y/n)')
    if continue_bool:
        if proceed_msg == 'y': 
            proceed = True
        elif proceed_msg == 'n': 
            proceed = False
    else:
        proceed = False
    return continue_bool, proceed

class unit_descriptor_handler():
    """
    Class to handle the unit_descriptor table in the hdf5 file

    Ops to handle mismatch between unit_descriptor and sorted_units:
        1- Resort units according to electrode number using
                unit metadata
        2- Recreate unit_descriptor table from scratch using
                metadata from sorted_units
    """
    def __init__(self, hf5, data_dir):
        self.hf5 = hf5
        # Make a table under /sorted_units describing the sorted units. 
        # If unit_descriptor already exists, 
        # just open it up in the variable table
        self.data_dir = data_dir
        self.hf5 = hf5

    def get_latest_unit_name(self,):
        """
        Get the name for the next unit to be saved 
        """

        # Get list of existing nodes/groups under /sorted_units
        saved_units_list = self.hf5.list_nodes('/sorted_units')

        # If saved_units_list is empty, start naming units from 000
        if saved_units_list == []:             
            unit_name = 'unit%03d' % 0
            max_unit = -1
        # Else name the new unit by incrementing the last unit by 1 
        else:
            unit_numbers = []
            for node in saved_units_list:
                    unit_numbers.append(node._v_pathname.split('/')[-1][-3:])
                    unit_numbers[-1] = int(unit_numbers[-1])
            unit_numbers = np.array(unit_numbers)
            max_unit = np.max(unit_numbers)
            unit_name = 'unit%03d' % int(max_unit + 1)

        return unit_name, max_unit

    def generate_hash(self,):
        """
        Generate a 10 character hash for the unit
        """
        return str(uuid.uuid4()).split('-')[0]

    def save_unit(
            self,
            unit_waveforms, 
            unit_times,
            electrode_num,
            this_sort_file_handler,
            split_or_merge,
            override_ask = False,
            ):
        """
        Save unit to hdf5 file
        """
        if not override_ask:
            continue_bool, unit_properties = \
                    self.get_unit_properties(
                            this_sort_file_handler,
                            split_or_merge,
                            )
            if not continue_bool:
                print(':: Unit not saved ::')
                return continue_bool, None
        else:
            unit_properties = {}
            unit_properties['single_unit'] = 0
            unit_properties['regular_spiking'] = 0
            unit_properties['fast_spiking'] = 0
            continue_bool = True

        if '/sorted_units' not in self.hf5:
            self.hf5.create_group('/', 'sorted_units')
        unit_name, max_unit = self.get_latest_unit_name()
        self.hf5.create_group('/sorted_units', unit_name)

        # Get a hash for the unit to compare stored data 
        # with unit_descriptor table
        unit_hash = self.generate_hash()

        # Add to HDF5
        waveforms = self.hf5.create_array('/sorted_units/%s' % unit_name, 
                        'waveforms', unit_waveforms)
        times = self.hf5.create_array('/sorted_units/%s' % unit_name, \
                                    'times', unit_times)

        unit_table = self.hf5.create_table(
                f'/sorted_units/{unit_name}',
                'unit_metadata',
                description = sorted_unit_metadata)

        # Get a new unit_descriptor table row for this new unit
        unit_description = unit_table.row    
        # Add to unit_descriptor table
        unit_description['waveform_count'] = int(len(unit_times))
        unit_description['electrode_number'] = electrode_num
        unit_description['hash'] = unit_hash
        unit_description['single_unit'] = unit_properties['single_unit'] 
        unit_description['regular_spiking'] = unit_properties['regular_spiking']
        unit_description['fast_spiking'] = unit_properties['fast_spiking']
        unit_description.append()

        # Flush table and hf5
        unit_table.flush()
        self.hf5.flush()
        return continue_bool, unit_name

    def get_saved_units_hashes(self,):
        """
        Get the hashes of the saved units
        Return both hahes and unit names
        """
        unit_list = self.hf5.list_nodes('/sorted_units')
        unit_hashes = []
        unit_names = []
        unit_numbers = []
        for unit in unit_list:
            metadata = unit.unit_metadata
            unit_hashes.append(metadata.col('hash')[0])
            unit_name = unit._v_pathname.split('/')[-1]
            unit_names.append(unit_name)
            unit_numbers.append(int(unit_name.split('unit')[-1]))
        saved_frame = pd.DataFrame({
            'hash': unit_hashes, 
            'unit_name': unit_names,
            'unit_number': unit_numbers,
            })
        return saved_frame

    def check_unit_descriptor_table(self,):
        if '/unit_descriptor' not in self.hf5:
            print(':: No unit_descriptor table found ::')
            return False
        else:
            return True

    def return_unit_descriptor_table(self,):
        """
        Return the unit descriptor table
        """
        if self.check_unit_descriptor_table():
            return self.hf5.root.unit_descriptor

    def table_to_frame(self,):
        """
        Convert the unit_descriptor table to a pandas dataframe
        """
        table = self.return_unit_descriptor_table() 
        table_cols = table.colnames
        dat_list = [table[i] for i in range(table.shape[0])]
        dict_list = [dict(zip(table_cols, dat)) for dat in dat_list]
        table_frame = pd.DataFrame(
                                data = dict_list,
                                    )
        return table_frame


    def check_table_matches_saved_units(self,): 
        """
        Check that the unit_descriptor table matches the saved units
        """
        table = self.return_unit_descriptor_table() 
        saved_frame = self.get_saved_units_hashes()
        table_frame = pd.DataFrame({
            'hash': table.col('hash')[:], 
            'unit_number': table.col('unit_number')[:]
            })

        saved_frame.sort_values(by = 'hash', inplace = True)
        table_frame.sort_values(by = 'hash', inplace = True)

        merged_frame = pd.merge(
                saved_frame, table_frame, on = 'unit_number', how = 'outer')
        merged_frame['match'] = merged_frame['hash_x'] == merged_frame['hash_y']

        if all(merged_frame['match']): 
            return True, merged_frame
        else:
            print('Unit descriptor table does not match saved units \n')
            return False, merged_frame

    def _rename_unit(self, hash, new_name):
        """
        Rename units in both unit_descriptor table and sorted_units directory
        in HDF5 file, using hash as identifier
        """
        # Rename saved unit
        unit_list = self.hf5.list_nodes('/sorted_units')
        wanted_unit_list = [unit for unit in unit_list \
                if unit.unit_metadata[:]['hash'][0] == hash]
        if not len(wanted_unit_list) > 0:
            print('Unit not found')
            return
        elif len(wanted_unit_list) > 1:
            print('Multiple units found')
            return
        wanted_unit = wanted_unit_list[0]
        wanted_unit._f_rename(new_name)

        # Flush table and hf5
        self.hf5.flush()


    def get_metadata_from_units(self,):
        """
        Extracts unit metadata from saved_units directory
        """
        if '/sorted_units' not in self.hf5:
            raise ValueError('No sorted_units directory found')

        unit_list = self.hf5.list_nodes('/sorted_units')

        if len(unit_list) == 0:
            raise ValueError('No units found in sorted_units directory')

        metadata_list = []
        for unit in unit_list:
            metadata_list.append(unit.unit_metadata[:])
        col_names = unit.unit_metadata.colnames
        saved_frame = pd.DataFrame(
                data = [dict(zip(col_names, row[0])) for row in metadata_list]
                )
        saved_frame['unit_name'] = [unit._v_pathname.split('/')[-1]
                for unit in unit_list]
        saved_frame['unit_number'] = [int(unit_name.split('unit')[-1])
                for unit_name in saved_frame['unit_name']]
        return saved_frame

    def resort_units(self,):
        """
        1) Get metadata from units
        2) Rename units sorted by electrode
        3) Update unit_descriptor table
        """
        metadata_table = self.get_metadata_from_units()
        metadata_table.sort_values(by = 'electrode_number', inplace = True)
        metadata_table['new_unit_number'] = np.arange(len(metadata_table))

        # Rename units
        for row in metadata_table.iterrows():
            this_hash = row[1]['hash']
            decoded_hash = abs(int(hash(this_hash)))
            new_name = f'unit{decoded_hash:03d}'
            self._rename_unit(this_hash, new_name)
        # This double step is necessary to avoid renaming conflicts
        for row in metadata_table.iterrows():
            this_hash = row[1]['hash']
            this_unit_number = row[1]['new_unit_number']
            this_unit_name = f'unit{this_unit_number:03d}'
            self._rename_unit(this_hash, this_unit_name)

        # Update unit_descriptor table
        self.write_unit_descriptor_from_sorted_units()

    def write_unit_descriptor_from_sorted_units(self,):
        """
        Generate unit descriptor table from metadata
        present in sorted units
        """
        metadata_table = self.get_metadata_from_units()

        if '/unit_descriptor' in self.hf5:
            self.hf5.remove_node('/unit_descriptor')
        table = self.hf5.create_table(
                '/','unit_descriptor',
                description = unit_descriptor)

        # Write from metadata table to unit_descriptor table
        for ind, this_row in metadata_table.iterrows():
            # Get a new unit_descriptor table row for this new unit
            unit_description = table.row    
            for col in table.colnames: 
                unit_description[col] = this_row[col]
            unit_description.append()

        table.flush()
        self.hf5.flush()

    def get_unit_properties(self, this_sort_file_handler, split_or_merge):
        """
        Ask user for unit properties and save in both unit_descriptor table
        and sorted_units directory in HDF5 file
        """

        # If unit has not been tampered with and sort_file is present
        if (not split_or_merge) and \
                (this_sort_file_handler.sort_table is not None):
            dat_row = this_sort_file_handler.current_row
            single_unit_msg = dat_row.single_unit
            if not (single_unit_msg.strip() == ''):
                single_unit = True

                # If single unit, check unit type
                unit_type_msg = dat_row.Type 
                if unit_type_msg == 'r': 
                    unit_type = 'regular_spiking'
                elif unit_type_msg == 'f': 
                    unit_type = 'fast_spiking'
                #unit_description[unit_type] = 1
            else:
                single_unit = False
                unit_type = 'none'
            continue_bool = True
            print('== Got unit property details from sort file ==')

        else:
            single_unit_msg, continue_bool = entry_checker(\
                    msg = 'Single-unit? (y/n)',
                    check_func = lambda x: x in ['y','n'],
                    fail_response = 'Please enter (y/n)')
            if continue_bool:
                if single_unit_msg == 'y': 
                    single_unit = True
                elif single_unit_msg == 'n': 
                    single_unit = False
            else:
                return continue_bool, None


            # If the user says that this is a single unit, 
            # ask them whether its regular or fast spiking
            if single_unit:
                unit_type_msg, continue_bool = entry_checker(\
                        msg = 'Regular or fast spiking? (r/f)',
                        check_func = lambda x: x in ['r','f'],
                        fail_response = 'Please enter (r/f)')
                if continue_bool:
                    if unit_type_msg == 'r': 
                        unit_type = 'regular_spiking'
                    elif unit_type_msg == 'f': 
                        unit_type = 'fast_spiking'
                else:
                    return continue_bool, None
                #unit_description[unit_type] = 1
            else:
                unit_type = 'none'
                continue_bool = True

        #unit_description['single_unit'] = int(single_unit)
        property_dict = dict(
                single_unit = int(single_unit),
                regular_spiking = int(unit_type == 'regular_spiking'),
                fast_spiking = int(unit_type == 'fast_spiking'),
                )

        return continue_bool, property_dict 


class sorted_unit_metadata(tables.IsDescription):
    electrode_number = tables.Int32Col()
    single_unit = tables.Int32Col()
    regular_spiking = tables.Int32Col()
    fast_spiking = tables.Int32Col()
    waveform_count = tables.Int32Col()
    hash = tables.StringCol(10)

# Define a unit_descriptor class to be used to add things (anything!) 
# about the sorted units to a pytables table
class unit_descriptor(tables.IsDescription):
    unit_number = tables.Int32Col(pos=0)
    electrode_number = tables.Int32Col()
    single_unit = tables.Int32Col()
    regular_spiking = tables.Int32Col()
    fast_spiking = tables.Int32Col()
    waveform_count = tables.Int32Col()
    hash = tables.StringCol(10)

class split_merge_signal:
    def __init__(self, clusters, this_sort_file_handler): 
        """
        First check whether there are multiple clusters to merge
        If not, check whether there is a split/sort file
        If not, ask whether to split
        """
        self.clusters = clusters
        self.this_sort_file_handler = this_sort_file_handler
        if not self.check_merge_clusters():
            if self.check_split_sort_file() is None:
                self.ask_split()


    def check_merge_clusters(self):
        if len(self.clusters) > 1:
            self.merge = True
            self.split = False
            return True
        else:
            self.merge = False
            return False

    def ask_split(self):
        msg, continue_bool = entry_checker(\
                msg = 'SPLIT this cluster? (y/n)',
                check_func = lambda x: x in ['y','n'],
                fail_response = 'Please enter (y/n)')
        if continue_bool:
            if msg == 'y':
                self.split = True
            elif msg == 'n': 
                self.split = False

    def check_split_sort_file(self):
        if self.this_sort_file_handler.sort_table is not None:
            dat_row = self.this_sort_file_handler.current_row
            split_val = re.findall('[0-9]+', str(dat_row.Split))
            if (len(str(dat_row.Split).strip()) > 0):
                self.split=True
            else:
                self.split=False
            print(f'== Got split details from sort file: {split_val} ==')
            return True
        else:
            return None


def gen_autosort_plot(
        subcluster_prob,
        subcluster_waveforms,
        chi_out,
        mean_waveforms,
        std_waveforms,
        subcluster_times,
        fin_bool,
        cluster_labels,
        electrode_num,
        sampling_rate,
        autosort_output_dir,
        n_max_plot=5000,
):
    """
    For each electrode, generate a summary of how each
    cluster was processed

    Inputs:
        subcluster_prob: list of probabilities of each subcluster 
        subcluster_waveforms: list of waveforms of each subcluster 
        chi_out: chi-square p-value of each subcluster's probability distribution
        mean_waveforms: list of mean waveforms of given subcluster
        std_waveforms: list of std of waveforms of given subcluster 
        subcluster_times: list of times of each subcluster 
        autosort_output_dir: absolute path of directory to save output
        n_max_plot: maximum number of waveforms to plot

    Outputs:
        Plots of following quantities
        1. prob distribution (indicate chi-square p-value in titles)
        2. ISI distribution
        3. Histogram of spikes over time
        4. Mean +/- std of waveform
        5. A finite amount of raw waveforms
    """

    if not os.path.exists(autosort_output_dir):
        os.makedirs(autosort_output_dir)
    print('== Generating autosort plot ==')
    print(f'== Saving to {autosort_output_dir} ==')

    fig, ax = plt.subplots(5, len(subcluster_waveforms),
                           figsize=(5*len(subcluster_prob), 20),
                           sharex=False, sharey=False)
    for i, (this_ax, this_waveforms) in \
            enumerate(zip(ax[:2,:].T, subcluster_waveforms)):
        waveform_count = len(this_waveforms)
        if waveform_count > n_max_plot:
            this_waveforms = this_waveforms[np.random.choice(
                len(this_waveforms), n_max_plot, replace=False)]
        this_ax[0].set_title(f'Cluster {cluster_labels[i]}' + '\n' +\
                'Waveform Count: {}'.format(waveform_count))
        for this_this_ax in this_ax:
            this_this_ax.plot(this_waveforms.T, color='k', alpha=0.01)
    for this_ax, this_dist, this_chi in zip(ax[2], subcluster_prob, chi_out):
        this_ax.hist(this_dist, bins=10, alpha=0.5, density=True)
        this_ax.hist(this_dist, bins=10, alpha=0.5, density=True,
                     histtype='step', color='k', linewidth=3)
        this_ax.set_title('Chi-Square p-value: {:.3f}'.format(this_chi.pvalue))
        this_ax.set_xlabel('Classifier Probability')
        this_ax.set_xlim([0, 1])
    # If chi-square p-value is less than alpha, 
    # create green border around subplots
    for i in range(len(subcluster_prob)):
        if fin_bool[i]:
            for this_ax in ax[:, i]:
                for this_spine in this_ax.spines.values():
                    this_spine.set_edgecolor('green')
                    this_spine.set_linewidth(5)
    ax[0, 0].set_ylabel('Waveform Amplitude')
    ax[2, 0].set_ylabel('Count')
    for this_ax, this_mean, this_std in zip(ax[1], mean_waveforms, std_waveforms):
        this_ax.plot(this_mean, color='k')
        this_ax.fill_between(np.arange(len(this_mean)),
                             y1=this_mean - this_std,
                             y2=this_mean + this_std,
                             color='k', alpha=0.5)
        this_ax.set_title('Mean +/- Std')
        this_ax.set_xlabel('Time (samples)')
        this_ax.set_ylabel('Amplitude')
    for this_ax, this_times in zip(ax[3], subcluster_times):
        this_ax.hist(this_times, bins=30, alpha=0.5, density=True)
        this_ax.set_title('Spike counts over time')
    for this_ax, this_times in zip(ax[4], subcluster_times):
        fig, this_ax = gen_isi_hist(
            this_times,
            np.arange(len(this_times)),
            sampling_rate,
            ax=this_ax,
        )
        this_ax.hist(np.diff(this_times), bins=30, alpha=0.5, density=True)
    # For first 2 rows, equalize y limits
    lims_list = [this_ax.get_ylim() for this_ax in ax[:2, :].flatten()]
    min_lim = np.min(lims_list)
    max_lim = np.max(lims_list)
    for this_ax in ax[0, :].flatten():
        this_ax.set_ylim([min_lim, max_lim])
    fig.suptitle(f'Electrode {electrode_num:02}', fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    fig.savefig(os.path.join(autosort_output_dir,
                f'{electrode_num:02}_subclusters.png'))
    plt.close()

def get_cluster_props(
        split_predictions,
        spike_waveforms,
        clf_prob,
        spike_times,
        chi_square_alpha,
        count_threshold,
        ):
    """
    Calculate the following properties for each cluster:
    - waveforms
    - times
    - probs
    - mean waveform
    - std waveform
    - chi_square p-value on classifier probability distribution

    Inputs:
        split_predictions: array of cluster labels
        spike_waveforms: array of spike waveforms
        clf_prob: array of classifier probabilities
        spike_times: array of spike times

    Outputs:
        subcluster_inds: list of indices for each cluster
        subcluster_waveforms: list of waveforms for each cluster
        subcluster_prob: list of probabilities for each cluster
        subcluster_times: list of times for each cluster
        mean_waveforms: list of mean waveforms for each cluster
        std_waveforms: list of std waveforms for each cluster
        fin_bool: list of booleans indicating whether the cluster is a wanted unit
        fin_bool_dict: dictionary of booleans indicating whether the cluster is a wanted unit
    """

    # Once selections have been made, save data
    # Waveforms of originally chosen cluster
    subcluster_inds = [np.where(split_predictions == this_split)[0] \
            for this_split in np.unique(split_predictions)]
    subcluster_waveforms = [spike_waveforms[this_inds] \
            for this_inds in subcluster_inds]
    subcluster_prob = [clf_prob[this_inds] \
            for this_inds in subcluster_inds]
    subcluster_times = [spike_times[this_inds] \
            for this_inds in subcluster_inds]
    mean_waveforms = [np.mean(this_waveform, axis = 0) for this_waveform in subcluster_waveforms]
    std_waveforms = [np.std(this_waveform, axis = 0) for this_waveform in subcluster_waveforms]

    # Check that the probability distributions are not uniform
    # That indicates that the cluster is likely generated by noise

    prob_dists = [np.histogram(this_prob, bins = 10, density = True)[0]
               for this_prob in subcluster_prob]

    chi_out = [chisquare(this_dist) for this_dist in prob_dists]

    chi_bool = [this_chi[1] < chi_square_alpha for this_chi in chi_out]
    count_bool = [len(this_waveform) > count_threshold for this_waveform in subcluster_waveforms]

    # To avoid confusion between index and cluster number,
    # change fin_bool to dictionary because it gets used later
    unique_clusters = np.unique(split_predictions)
    fin_bool = np.logical_and(chi_bool, count_bool)
    fin_bool_dict = dict(zip(unique_clusters, fin_bool))

    return (
        subcluster_inds,
        subcluster_waveforms,
        subcluster_prob,
        subcluster_times,
        mean_waveforms,
        std_waveforms,
        chi_out,
        fin_bool,
        fin_bool_dict,
        )
    
def calculate_merge_sets(
        mahal_mat,
        mahal_thresh,
        isi_threshs,
        split_predictions,
        spike_waveforms,
        spike_times,
        clf_prob,
        chi_square_alpha,
        count_threshold,
        sampling_rate,
        ):

    """
    Calculate which clusters to merge based on mahalanobis distance
    and ISI violations

    Inputs:
        mahal_mat: (n_clusters, n_clusters) matrix of mahalanobis distances
        mahal_thresh: float, threshold for mahalanobis distance
        isi_threshs: list of floats, thresholds for ISI violations
        split_predictions: (n_spikes,) array of cluster predictions
        spike_times: (n_spikes,) array of spike times
        sample_rate: float, sample rate of recording

    Outputs:
        final_merge_sets: list of tuples, 
                          each tuple is a pair of clusters to merge
    """

    unique_clusters = np.unique(split_predictions)

    # Set diagonal as nan
    np.fill_diagonal(mahal_mat, np.nan)

    # If -1 (outliers) in unique clusters, remove from
    # both unique clusters and mahalanobis matrix
    if -1 in unique_clusters:
        outlier_ind = np.where(unique_clusters == -1)[0][0]
        unique_clusters = np.delete(unique_clusters, outlier_ind)
        mahal_mat = np.delete(mahal_mat, outlier_ind, axis = 0)
        mahal_mat = np.delete(mahal_mat, outlier_ind, axis = 1)

    # Check mahal_mat against threshold
    merge_mat = mahal_mat < mahal_thresh

    # Get indices of clusters to merge
    merge_inds = np.array(np.where(merge_mat)).T
    merge_clusters = unique_clusters[merge_inds]
    
    # Make sure there are no sets of duplicates
    # i.e. (1, 2) and (2, 1)
    merge_sets = [tuple(set(this_pair)) for this_pair in merge_clusters]
    merge_sets = list(set(merge_sets)) 

    # At this stage, we are certain these need to be merged
    # Consolidate overlapping merge sets
    # Check for intersections between sets, if there is one,
    # merge sets
    # Repeat until no intersections
    final_merge_sets = [set(this_set) for this_set in merge_sets]
    while True:
        # Check for intersections
        # If there is one, merge sets and start over
        # If not, break
        intersect_bool = False
        for i, this_set in enumerate(final_merge_sets):
            for j, other_set in enumerate(final_merge_sets):
                if i == j:
                    continue
                if len(this_set.intersection(other_set)) > 0:
                    intersect_bool = True
                    final_merge_sets[i] = this_set.union(other_set)
                    final_merge_sets[j] = set()
                    break
            if intersect_bool:
                break
        # Remove empty sets
        final_merge_sets = [this_set for this_set in final_merge_sets \
                if len(this_set) > 0]
        if not intersect_bool:
            break

    # Convert back to tuples
    final_merge_sets = [tuple(this_set) for this_set in final_merge_sets]

    # Check ISI violations for each merge set
    violations_list = []
    for this_set in final_merge_sets:
        merged_inds = [i for i,val in enumerate(split_predictions) \
                if val in this_set] 
        merged_times = spike_times[merged_inds]
        violations = get_ISI_violations(
                merged_times, sampling_rate)
        violations_list.append(violations)

    violations_pass_bool = [all(
        np.array(this_violations) < np.array(isi_threshs)
        ) for this_violations in violations_list]

    final_merge_sets = [this_set for this_set, this_bool \
            in zip(final_merge_sets, violations_pass_bool) if this_bool]

    # Only keep merge sets if they contain at least one unit
    
    # Update split_predicitons so 1) waveform count, 2) distribution
    # of classifier probs can be tested

    new_clust_names = np.arange(len(final_merge_sets)) + \
            len(unique_clusters)

    new_name_dict = dict(zip(new_clust_names, final_merge_sets))

    merge_split_predictions = split_predictions.copy()

    for this_set, this_name in zip(final_merge_sets, new_clust_names):
        for this_cluster in this_set:
            inds = merge_split_predictions == this_cluster
            merge_split_predictions[inds] = this_name

    (
        subcluster_inds,
        subcluster_waveforms,
        subcluster_prob,
        subcluster_times,
        mean_waveforms,
        std_waveforms,
        chi_out,
        fin_bool,
        fin_bool_dict,
    ) = \
            get_cluster_props(
                merge_split_predictions,
                spike_waveforms,
                clf_prob,
                spike_times,
                chi_square_alpha,
                count_threshold,
                )

    # Check fin_merge_sets against fin_bool_dict
    fin_merge_sets = [val for key,val in new_name_dict.items() \
            if fin_bool_dict[key]]

    return fin_merge_sets, new_clust_names
