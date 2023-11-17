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
from utils import blech_waveforms_datashader

def cluster_check(x):
    clusters = re.findall('[0-9]+',x)
    return sum([i.isdigit() for i in clusters]) == len(clusters)

def get_electrode_details(args, counter):
    """
    Ask user for electrode number, number of clusters, and cluster numbers
    """

    if args.sort_file is not None:
        sort_table = pd.read_csv(args.sort_file)

        if counter == len(sort_table):
            return None, None, None

        electrode_num = int(sort_table.Chan[counter])
        num_clusters = int(sort_table.Solution[counter])
        clusters = re.findall('[0-9]+',str(sort_table.Cluster[counter]))
        clusters = [int(x) for x in clusters]

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
                check_func = lambda x: (str.isdigit(x) and (1<int(x)<=7)),
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
    fig.suptitle('Are these the neurons you want to select?')
    fig.tight_layout()
    plt.show()

def get_clustering_params():
    """
    Ask user for clustering parameters
    """
    # Get clustering parameters from user
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

def generate_cluster_plots(
        split_predictions, 
        spike_waveforms, 
        spike_times, 
        n_clusters, 
        this_cluster
        ):

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
                title = f'Split Cluster {cluster}',
                ax = ax.flatten()[cluster],)

    for cluster in range(n_clusters, n_rows*n_cols):
        ax.flatten()[cluster].axis('off')

    plt.tight_layout()
    plt.show()

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

def get_ISI_violations(unit_times):
    """
    Get ISI violations
    """

    ISIs = np.ediff1d(np.sort(unit_times))/30.0
    violations1 = 100.0*float(np.sum(ISIs < 1.0)/len(unit_times))
    violations2 = 100.0*float(np.sum(ISIs < 2.0)/len(unit_times))
    return violations1, violations2

def generate_datashader_plot(
        unit_waveforms,
        unit_times,
        title = None,
        ax = None,
        ):
    """
    Generate datashader plot
    """
    violations1, violations2 = get_ISI_violations(unit_times)

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
        f'{violations2:.1f} % (<2ms),'
        f'{violations1:.1f} % (<1ms),'
        f'{len(unit_times)} total waveforms. \n') 
    ax.set_title(print_str)

    return violations1, violations2, fig, ax

def delete_raw_recordings(hf5):
    """
    Delete raw recordings from hdf5 file
    """

    try:
        hf5.remove_node('/raw', recursive = 1)
        # And if successful, close the currently open hdf5 file and 
        # ptrepack the file
        hf5.close()
        print("Raw recordings removed")
        os.system("ptrepack --chunkshape=auto --propindexes --complevel=9 "
            "--complib=blosc " + hdf5_name + " " + hdf5_name[:-3] + "_repacked.h5")
        # Delete the old (raw and big) hdf5 file
        os.system("rm " + hdf5_name)
        # And open the new, repacked file
        hf5 = tables.open_file(hdf5_name[:-3] + "_repacked.h5", 'r+')
        print("File repacked")
    except:
        print("Raw recordings have already been removed, so moving on ..")

def generate_violations_warning(
        violations1,
        violations2,
        unit_times,
        ):
    print_str = (f':: Merged cluster \n'
        f':: {violations1:.1f} % (<2ms)\n'
        f':: {violations2:.1f} % (<1ms)\n'
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
    return continue_bool, proceed

class unit_descriptor_handler():
    """
    Class to handle the unit_descriptor table in the hdf5 file
    """
    def __init__(self, hf5):
        self.hf5 = hf5
        # Make a table under /sorted_units describing the sorted units. 
        # If unit_descriptor already exists, 
        # just open it up in the variable table
        self.hf5 = hf5
        if '/unit_descriptor' in hf5:
            self.table = self.hf5.root.unit_descriptor
        else:
            self.table = self.hf5.create_table('/', 'unit_descriptor', 
                    description = unit_descriptor)
        self.counter = len(self.hf5.root.unit_descriptor) - 1

    def get_latest_unit_name(self,):
        """
        Get the name for the next unit to be saved 
        """

        # Get list of existing nodes/groups under /sorted_units
        saved_units_list = self.hf5.list_nodes('/sorted_units')

        # if sort_table given, use that to name units
        #if args.sort_file is not None:
        #    unit_name = 'unit%03d' % int(true_index[counter])

        #else:
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
            ):
        """
        Save unit to hdf5 file
        """
        if '/sorted_units' not in self.hf5:
            self.hf5.create_group('/', 'sorted_units')
        unit_name, max_unit = self.get_latest_unit_name()
        self.hf5.create_group('/sorted_units', unit_name)
        # Put in unit number
        # if args.sort_file is not None:
        #     # If sort-file is given, use sort_table index 
        #     unit_description['unit_number'] = int(true_index[counter])
        # else:

        # Get a hash for the unit to compare stored data 
        # with unit_descriptor table
        unit_hash = self.generate_hash()

        # Add to HDF5
        waveforms = self.hf5.create_array('/sorted_units/%s' % unit_name, 
                        'waveforms', unit_waveforms)
        times = self.hf5.create_array('/sorted_units/%s' % unit_name, \
                                    'times', unit_times)

        # Get a new unit_descriptor table row for this new unit
        unit_description = self.table.row    
        # Add to unit_descriptor table
        unit_description['waveform_count'] = int(len(unit_times))
        unit_description['electrode_number'] = electrode_num
        unit_description['hash'] = unit_hash
        unit_description['unit_number'] = int(max_unit + 1)
        unit_description.append()

        self.counter += 1

        # Flush table and hf5
        self.table.flush()
        self.hf5.flush()

    def get_saved_units_hashes(self,):
        """
        Get the hashes of the saved units
        Return both hahes and unit names
        """
        unit_list = self.hf5.list_nodes('/sorted_units')
        unit_hashes = []
        unit_names = []
        for unit in unit_list:
            unit_hashes.append(unit.hash)
            unit_names.append(unit._v_pathname.split('/')[-1])
        saved_frame = pd.DataFrame({'hash': unit_hashes, 'unit_name': unit_names})
        return saved_frame

    def table_to_frame(self,):
        """
        Convert the unit_descriptor table to a pandas dataframe
        """
        table_cols = self.table.colnames
        dat_list = [self.table[i] for i in range(self.table.shape[0])]
        dict_list = [dict(zip(table_cols, dat)) for dat in dat_list]
        table_frame = pd.DataFrame(
                                data = dict_list,
                                    )
        return table_frame


    def check_table_matches_saved_units(self,): 
        """
        Check that the unit_descriptor table matches the saved units
        """
        saved_frame = self.get_saved_units_hashes()
        table_frame = pd.DataFrame({'hash': self.table.cols.hash[:], 
                                    'unit_name': self.table.cols.unit_name[:]})

        saved_frame.sort_values(by = 'hash', inplace = True)
        table_frame.sort_values(by = 'hash', inplace = True)

        merged_frame = pd.merge(
                saved_frame, table_frame, on = 'unit_name', how = 'outer')
        merged_frame['match'] = merged_frame['hash_x'] == merged_frame['hash_y']

        if saved_frame.equals(table_frame):
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
        wanted_unit = [unit for unit in unit_list if unit.hash == hash][0]
        wanted_unit._v_pathname = '/sorted_units/%s' % new_name

        # Rename unit in unit_descriptor table
        wanted_row = self.table.get_where_list(f'hash == {hash}')
        self.table.cols.unit_name[wanted_row] = new_name

        # Flush table and hf5
        self.table.flush()
        self.hf5.flush()


    def sort_table_and_saved_units(self,):
        """
        Sort both unit_descriptor table and sorted_units directory
        in HDF5 file
        """
        check_bool, merged_frame = self.check_table_matches_saved_units()
        if not check_bool:
            print('Please organize units before moving forward')
            print('The following units do not match: \n')
            print(merged_frame)
            exit()
        else:
            unit_descriptor_frame = pd.DataFrame(
                    {
                        'unit_number': self.table.cols.unit_number[:],
                        'electrode_number': self.table.cols.electrode_number[:],
                        'hash': self.table.cols.hash[:],
                        }
                    )
            unit_descriptor_frame.sort_values(
                    by = 'electrode_number', inplace = True)
            unit_descriptor_frame['new_unit_number'] = \
                    np.arange(len(unit_descriptor_frame))
            for row in unit_descriptor_frame.iterrows():
                this_hash = row[1]['hash']
                this_unit_number = row[1]['new_unit_number']
                self._rename_unit(this_hash, this_unit_number)
        table.flush()
        hf5.flush()

    def get_metadata_from_units(self,):
        """
        Extracts unit metadata from saved_units directory
        """
        unit_list = self.hf5.list_nodes('/sorted_units')
        metadata_list = []
        for unit in unit_list:
            metadata_list.append(unit.metadata)
        saved_frame = pd.DataFrame({'hash': unit_hashes, 
                                    'unit_name': unit_names,
                                    'electrode': unit_electrodes,
                                    'count': unit_counts})
        return saved_frame

    def clear_mismatches(self,):
        """
        If there are mismatches between the unit_descriptor table and the
        sorted_units directory, clear the mismatches from both
        """
        check_bool, merged_frame = self.check_table_matches_saved_units()
        if check_bool:
            print('No mismatches to clear')
            print()
            print(merged_frame)
            exit()
        else:
            # Delete unit descriptor and recreate from data
            # present in sorted_units directory
            # Generate log of old and new tables in data_dir
            old_table = self.table.copy()
            new_table = self.get_metadata_from_units() 

            # Delete old table and create new one
            hf5.remove_node('/', 'unit_descriptor')
            self.table = self.hf5.create_table('/', 'unit_descriptor', 
                    description = unit_descriptor)
            self.table.append(new_table.to_records(index = False))
            self.table.flush()
            hf5.flush()

            # Write log of old and new tables
            log_dir = os.path.join(self.data_dir, 'processing_log.txt')
            print(f'Unit descriptor table cleared {datetime.now()} \n')
            print(f'Changes in log: {log_dir}')

            with open(log_dir, 'a') as log_file:
                log_file.write(f'Unit descriptor table cleared {datetime.now()} \n')
                log_file.write('Old table: \n')
                log_file.write(old_table)
                log_file.write('New table: \n')
                log_file.write(new_table)

    def get_unit_properties(self, counter, args, unit_details_file_bool):
        """
        Ask user for unit properties and save in both unit_descriptor table
        and sorted_units directory in HDF5 file
        """

        unit_description = dict(
                zip(self.table.colnames, self.table[counter]))
        unit_description['regular_spiking'] = 0
        unit_description['fast_spiking'] = 0

        if unit_details_file_bool and (args.sort_file is not None):
            single_unit_msg = sort_table.single_unit[counter]
            if not (single_unit_msg.strip() == ''):
                single_unit = True

                # If single unit, check unit type
                unit_type_msg = sort_table.Type[counter] 
                if unit_type_msg == 'r': 
                    unit_type = 'regular_spiking'
                elif unit_type_msg == 'f': 
                    unit_type = 'fast_spiking'
                unit_description[unit_type] = 1
            else:
                single_unit = False

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
                unit_description[unit_type] = 1

        unit_description['single_unit'] = int(single_unit)

        return continue_bool, unit_description

    def modify_row(self, row_ind, new_data):
        self.table.modify_coordinates(row_ind, new_data)
        self.table.flush()
        
    def write_descriptor_to_saved_units(self, unit_number):
        """
        For the specified unit, write the unit descriptor row
        to the saved_units directory
        """
        wanted_row = self.table.read_where(\
                'unit_number == unit_number')
        self.hf5.create_table('/sorted_units',
                f'unit_{unit_number}',
                description = unit_descriptor)
        self.hf5.append_where('/sorted_units',
                f'unit_{unit_number}',
                wanted_row)
        self.hf5.flush()



            
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
