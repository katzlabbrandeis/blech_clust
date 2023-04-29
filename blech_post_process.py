import os
import tables
import numpy as np
import easygui
import ast
import re
import pylab as plt
import matplotlib.image as mpimg
from sklearn.mixture import GaussianMixture
import argparse
import pandas as pd

# Import 3rd party code
from utils import blech_waveforms_datashader
from utils.blech_utils import entry_checker, imp_metadata

# Set seed to allow inter-run reliability
# Also allows reusing the same sorting sheets across runs
np.random.seed(0)

def cluster_check(x):
    clusters = re.findall('[0-9]+',x)
    return sum([i.isdigit() for i in clusters]) == len(clusters)

# Get directory where the hdf5 file sits, and change to that directory
# Get name of directory with the data files
# Create argument parser
parser = argparse.ArgumentParser(description = 'Spike extraction and sorting script')
parser.add_argument('--dir-name',  '-d', help = 'Directory containing data files')
parser.add_argument('--show-plot', '-p', 
        help = 'Show waveforms while iterating (True/False)', default = 'True')
parser.add_argument('--sort-file', '-f', help = 'CSV with sorted units')
args = parser.parse_args()

if args.sort_file is not None:
    if not (args.sort_file[-3:] == 'csv'):
        raise Exception("Please provide CSV file")
    sort_table = pd.read_csv(args.sort_file)
    sort_table.fillna('',inplace=True)
    # Check when more than one cluster is specified
    sort_table['len_cluster'] = \
            [len(re.findall('[0-9]+',str(x))) for x in sort_table.Cluster]
    # Get splits and merges out of the way first
    sort_table.sort_values(['len_cluster','Split'],ascending=False, inplace=True)
    true_index = sort_table.index
    sort_table.reset_index(inplace=True)

if args.dir_name is not None: 
    metadata_handler = imp_metadata([[],args.dir_name])
else:
    metadata_handler = imp_metadata([])
dir_name = metadata_handler.dir_name
#dir_name = easygui.diropenbox()
os.chdir(dir_name)
file_list = metadata_handler.file_list
hdf5_name = metadata_handler.hdf5_name
# Open the hdf5 file
hf5 = tables.open_file(hdf5_name, 'r+')

# Clean up the memory monitor files, pass if clean up has been done already
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


# Delete the raw node, if it exists in the hdf5 file, to cut down on file size
try:
        hf5.remove_node('/raw', recursive = 1)
        # And if successful, close the currently open hdf5 file and ptrepack the file
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

# Make the sorted_units group in the hdf5 file if it doesn't already exist
try:
        hf5.create_group('/', 'sorted_units')
except:
        pass

# Define a unit_descriptor class to be used to add things (anything!) 
# about the sorted units to a pytables table
class unit_descriptor(tables.IsDescription):
        unit_number = tables.Int32Col(pos=0)
        electrode_number = tables.Int32Col()
        single_unit = tables.Int32Col()
        regular_spiking = tables.Int32Col()
        fast_spiking = tables.Int32Col()
        waveform_count = tables.Int32Col()
        #mean_amplitude = tables.Float32Col()

# Make a table under /sorted_units describing the sorted units. 
# If unit_descriptor already exists, just open it up in the variable table
try:
        table = hf5.create_table('/', 'unit_descriptor', 
                description = unit_descriptor)
except:
        table = hf5.root.unit_descriptor

# Run an infinite loop as long as the user wants to pick clusters from the electrodes   
counter = len(hf5.root.unit_descriptor) - 1
while True:

        unit_details_bool = 0
        counter += 1
        # If sort_file given, iterate through that, otherwise ask user
        if args.sort_file is not None:
            if counter == len(sort_table):
                break
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
                break

            num_clusters_str, continue_bool = entry_checker(\
                    msg = 'Solution number :: ',
                    check_func = lambda x: (str.isdigit(x) and (1<int(x)<=7)),
                    fail_response = 'Please enter an interger')
            if continue_bool:
                num_clusters = int(num_clusters_str)
            else:
                continue


            clusters_msg, continue_bool = entry_checker(\
                    msg = 'Cluster numbers (anything separated) ::',
                    check_func = cluster_check,
                    fail_response = 'Please enter integers')
            if continue_bool:
                clusters = re.findall('[0-9]+',clusters_msg)
                clusters = [int(x) for x in clusters]
            else:
                continue

        
        # Print out selections
        print('||| Electrode {}, Solution {}, Cluster {} |||'.\
                format(electrode_num, num_clusters, clusters))


        # Load data from the chosen electrode and solution
        loading_paths = [\
            f'./spike_waveforms/electrode{electrode_num:02}/spike_waveforms.npy',
            f'./spike_times/electrode{electrode_num:02}/spike_times.npy',
            f'./spike_waveforms/electrode{electrode_num:02}/pca_waveforms.npy',
            f'./spike_waveforms/electrode{electrode_num:02}/energy.npy',
            f'./spike_waveforms/electrode{electrode_num:02}/spike_amplitudes.npy',
            f'./clustering_results/electrode{electrode_num:02}/'\
                    f'clusters{num_clusters}/predictions.npy',]

        var_names = ['spike_waveforms','spike_times','pca_slices','energy',\
                'amplitudes','predictions',]

        for var, path in zip(var_names, loading_paths):
            globals()[var] = np.load(path)

        # Re-show images of neurons so dumb people like Abu can make sure they
        # picked the right ones
        if ast.literal_eval(args.show_plot): 
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

        # Check if the user wants to merge clusters if more than 1 cluster was chosen. 
        # Else ask if the user wants to split/re-cluster the chosen cluster
        merge = False
        re_cluster = False
        merge_msg = 'a'
        re_cluster_msg = 'a'
        if len(clusters) > 1:
            # Providing more than one cluster will AUTOMATICALLY merge
            merge = True

        else:
            # if sort_file present use that
            if args.sort_file is not None:
                split_element = sort_table.Split[counter]
                if not (split_element.strip() == ''):
                        re_cluster = True
                else:
                        re_cluster = False
            # Otherwise ask user
            else:
                split_msg, continue_bool = entry_checker(\
                        msg = 'SPLIT this cluster? (y/n)',
                        check_func = lambda x: x in ['y','n'],
                        fail_response = 'Please enter (y/n)')
                if continue_bool:
                    if split_msg == 'y': 
                        re_cluster = True
                    elif split_msg == 'n': 
                        re_cluster = False
                else:
                    continue


        # If the user asked to split/re-cluster, 
        # ask them for the clustering parameters and perform clustering
        split_predictions = []
        chosen_split = 0
        if re_cluster: 
            # Get clustering parameters from user
            n_clusters = int(input('Number of clusters (default=5): ') or "5")
            values = [100,0.001,10]
            fields_str = (
                    f':: Max iterations (1000 is plenty) : {values[0]} \n' 
                    f':: Convergence criterion (usually 0.0001) : {values[1]} \n' 
                    f':: Number of random restarts (10 is plenty) : {values[2]}')
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
                    n_iter = int(clustering_params[1])
                    thresh = float(clustering_params[2])
                    n_restarts = int(clustering_params[3]) 
            else:
                continue

            # Make data array to be put through the GMM - 5 components: 
            # 3 PCs, scaled energy, amplitude
            this_cluster = np.where(predictions == int(clusters[0]))[0]
            n_pc = 3
            data = np.zeros((len(this_cluster), n_pc + 3))  
            data[:,3:] = pca_slices[this_cluster,:n_pc]
            data[:,0] = energy[this_cluster]/np.max(energy[this_cluster])
            data[:,1] = np.abs(amplitudes[this_cluster])/\
                    np.max(np.abs(amplitudes[this_cluster]))

            # Cluster the data
            g = GaussianMixture(
                    n_components = n_clusters, 
                    covariance_type = 'full', 
                    tol = thresh, 
                    max_iter = n_iter, 
                    n_init = n_restarts)
            g.fit(data)
        
            # Show the cluster plots if the solution converged
            if g.converged_:
                split_predictions = g.predict(data)
                x = np.arange(len(spike_waveforms[0])) + 1
                #fig, ax = gen_square_subplots(n_clusters,sharex=True,sharey=True)
                for cluster in range(n_clusters):
                    split_points = np.where(split_predictions == cluster)[0]
                    # Waveforms and times from the chosen cluster
                    slices_dejittered = spike_waveforms[this_cluster, :]            
                    times_dejittered = spike_times[this_cluster]
                    # Waveforms and times from the chosen split of the chosen cluster
                    times_dejittered = times_dejittered[split_points]               
                    ISIs = np.ediff1d(np.sort(times_dejittered))/30.0
                    violations1 = 100.0*float(np.sum(ISIs < 1.0)/split_points.shape[0])
                    violations2 = 100.0*float(np.sum(ISIs < 2.0)/split_points.shape[0])
                    fig, ax = blech_waveforms_datashader.waveforms_datashader(\
                            slices_dejittered[split_points, :], x, downsample = False)
                    ax.set_xlabel('Sample (30 samples per ms)')
                    ax.set_ylabel('Voltage (microvolts)')
                    print_str = (f'\nSplit Cluster {cluster} \n'
                        f'{violations2:.1f} % (<2ms),'
                        f'{violations1:.1f} % (<1ms),'
                        f'{split_points.shape[0]} total waveforms. \n') 
                    ax.set_title(print_str)
            else:
                print("Solution did not converge "\
                        "- try again with higher number of iterations "\
                        "or lower convergence criterion")
                continue

            plt.show()

            # Ask the user for the split clusters they want to choose
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
                continue

        ##################################################

        # Get list of existing nodes/groups under /sorted_units
        node_list = hf5.list_nodes('/sorted_units')

        # if sort_table given, use that to name units
        if args.sort_file is not None:
            unit_name = 'unit%03d' % int(true_index[counter])

        else:
            # If node_list is empty, start naming units from 000
            unit_name = ''
            max_unit = 0
            if node_list == []:             
                    unit_name = 'unit%03d' % 0
            # Else name the new unit by incrementing the last unit by 1 
            else:
                    unit_numbers = []
                    for node in node_list:
                            unit_numbers.append(node._v_pathname.split('/')[-1][-3:])
                            unit_numbers[-1] = int(unit_numbers[-1])
                    unit_numbers = np.array(unit_numbers)
                    max_unit = np.max(unit_numbers)
                    unit_name = 'unit%03d' % int(max_unit + 1)

        # Get a new unit_descriptor table row for this new unit
        unit_description = table.row    
        # Put in unit number
        if args.sort_file is not None:
            unit_description['unit_number'] = int(true_index[counter])
        else:
            unit_description['unit_number'] = int(max_unit + 1)


        # If the user re-clustered/split clusters, 
        # add the chosen clusters in split_clusters
        if re_cluster:
                hf5.create_group('/sorted_units', unit_name)
                # Waveforms of originally chosen cluster
                cluster_inds = np.where(predictions == int(clusters[0]))[0] 
                fin_inds = np.concatenate(\
                        [np.where(split_predictions == this_split)[0] \
                                    for this_split in chosen_split])
                unit_waveforms = spike_waveforms[cluster_inds, :]    
                # Subsetting this set of waveforms to include only the chosen split
                unit_waveforms = unit_waveforms[fin_inds]

                # Do the same thing for the spike times
                unit_times = spike_times[cluster_inds]
                unit_times = unit_times[fin_inds] 
                # Add to HDF5
                waveforms = hf5.create_array('/sorted_units/%s' % unit_name, 
                                'waveforms', unit_waveforms)
                times = hf5.create_array('/sorted_units/%s' % unit_name, \
                                            'times', unit_times)
                unit_description['waveform_count'] = int(len(unit_times))
                unit_description['electrode_number'] = electrode_num

                # To consolidate asking for unit details (single unit vs multi,
                # regular vs fast), set bool and ask for details at the end
                unit_details_bool = 1
                unit_details_file_bool = 0
                
        ##################################################

        # If only 1 cluster was chosen (and it wasn't split), 
        # add that as a new unit in /sorted_units. 
        # Ask if the isolated unit is an almost-SURE single unit
        elif len(clusters) == 1:
                hf5.create_group('/sorted_units', unit_name)
                fin_inds = np.where(predictions == int(clusters[0]))[0]
                unit_waveforms = spike_waveforms[fin_inds, :]
                unit_times = spike_times[fin_inds]
                waveforms = hf5.create_array('/sorted_units/%s' % unit_name, 
                        'waveforms', unit_waveforms)
                times = hf5.create_array('/sorted_units/%s' % unit_name, \
                                        'times', unit_times)
                unit_description['waveform_count'] = int(len(unit_times))
                unit_description['electrode_number'] = electrode_num

                # To consolidate asking for unit details (single unit vs multi,
                # regular vs fast), set bool and ask for details at the end
                unit_details_bool = 1
                # If unit was not manipulated (merge/split), read unit details
                # from file if provided
                unit_details_file_bool = 1

        else:
            # If the chosen units are going to be merged, merge them
            if merge:
                fin_inds = np.concatenate(\
                        [np.where(predictions == int(cluster))[0] \
                        for cluster in clusters])
                unit_waveforms = spike_waveforms[fin_inds, :]
                unit_times = spike_times[fin_inds]

                # Show the merged cluster to the user, 
                # and ask if they still want to merge
                x = np.arange(len(unit_waveforms[0])) + 1
                fig, ax = blech_waveforms_datashader.\
                        waveforms_datashader(unit_waveforms, x, downsample = False)
                # plt.plot(x - 15, unit_waveforms[:, ::10].T, 
                #                   linewidth = 0.01, color = 'red')
                ax.set_xlabel('Sample (30 samples per ms)')
                ax.set_ylabel('Voltage (microvolts)')
                ax.set_title('Merged cluster, No. of waveforms={:d}'.\
                                    format(unit_waveforms.shape[0]))
                plt.show()
 
                # Warn the user about the frequency of ISI violations 
                # in the merged unit
                ISIs = np.ediff1d(np.sort(unit_times))/30.0
                violations1 = 100.0*float(np.sum(ISIs < 1.0)/len(unit_times))
                violations2 = 100.0*float(np.sum(ISIs < 2.0)/len(unit_times))
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
                else:
                    continue

                # Create unit if the user agrees to proceed, 
                # else abort and go back to start of the loop 
                if proceed:     
                    hf5.create_group('/sorted_units', unit_name)
                    waveforms = hf5.create_array('/sorted_units/%s' % unit_name, 
                            'waveforms', unit_waveforms)
                    times = hf5.create_array('/sorted_units/%s' % unit_name, 
                            'times', unit_times)
                    unit_description['waveform_count'] = int(len(unit_times))
                    unit_description['electrode_number'] = electrode_num

                    # To consolidate asking for unit details (single unit vs multi,
                    # regular vs fast), set bool and ask for details at the end
                    unit_details_bool = 1
                    unit_details_file_bool = 0

                else:
                    continue

        # Ask user for unit details, and ask for HDF5 file 
        if unit_details_bool:
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
                    continue


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
                        continue
                    unit_description[unit_type] = 1

            unit_description['single_unit'] = int(single_unit)

            unit_description.append()
            table.flush()
            hf5.flush()

        #is_nrn_path = f'./clustering_results/electrode{electrode_num:02}/'\
        #            f'clusters{num_clusters}/is_nrn.npy'
        #if not os.path.exists(is_nrn_path):
        #    is_nrn_list = [[fin_inds]]
        #else:
        #    is_nrn_list = np.load(is_nrn_path)
        #    is_nrn_list.append([fin_inds])
        #np.save(is_nrn_path, is_nrn_list)

        print('==== {} Complete ===\n'.format(unit_name))
        print('==== Iteration Ended ===\n')

# Sort unit_descriptor by unit_number
# This will be needed if sort_table is used, as using sort_table
# will add merge/split marked units first
temp_table = table[:]
temp_table.sort(order = 'unit_number')
table[:] = temp_table
table.flush()
hf5.flush()

print('== Post-processing exiting ==')
print(f'== {len(hf5.root.unit_descriptor)} total units')
if len(hf5.root.unit_descriptor) == hf5.root.sorted_units._g_getnchildren():
    print('== unit_descriptor and sorted units counts match ==')
    # If things match, renumber sorted_units and unit_num in unit_descriptor
    # to be continuously increasing (in case units from the sort-table
    # were deleted or not selected
    # This should not be a problem if units were enterred manually
    if args.sort_file is not None:
        # Since we're not indexing by name, we can rename directly
        base_dir = '/sorted_units'
        node_list = hf5.list_nodes(base_dir)
        final_nums = np.arange(len(node_list))
        for this_node,inter_num in zip(node_list,final_nums):
            this_node._f_rename('unit{:03d}'.format(inter_num))
        # Also rename unit_nums in unit_descriptor
        temp_table = table[:]
        for unit, this_unit_num in zip(temp_table, final_nums):
            unit[0] = this_unit_num
        table[:] = temp_table
else:
    print('== unit_descriptor and sorted units counts **DO NOT** match ==')
# Close the hdf5 file
hf5.close()
