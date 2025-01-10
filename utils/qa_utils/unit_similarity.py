# Import stuff!
from collections import Counter
import numpy as np
import tables
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
# Get script path
script_path = os.path.realpath(__file__)
script_dir_path = os.path.dirname(script_path)
blech_path = os.path.dirname(os.path.dirname(script_dir_path))
sys.path.append(blech_path)
from utils.blech_utils import imp_metadata, pipeline_graph_check
from tqdm import tqdm
from numba import jit


def unit_similarity_abu(all_spk_times):
    int_spike_times = [np.array(x, dtype=np.int32) for x in all_spk_times]
    spike_counts = [len(x) for x in int_spike_times]
    # Count duplicates generated dur to binning
    mat_len = max([max(x) for x in int_spike_times])

    spike_times_array = np.zeros((len(all_spk_times), mat_len+1))
    # Populate the matrix
    for i, x in enumerate(int_spike_times):
        for y in x:
            spike_times_array[i, y] = 1

    # For each unit, iterate over spiketimes and
    # find what other units have spikes
    # within 1 ms of the current unit's spike
    all_overlapping_units = []
    for unit_num, times in tqdm(enumerate(int_spike_times)):
        # Pull out array for all units but the current unit
        this_unit_overlaps = []
        for this_time in times:
            overlapping_units = np.where(
                spike_times_array[:, this_time-1:this_time+1].sum(axis=1))[0]
            this_unit_overlaps.append(overlapping_units)
        all_overlapping_units.append(this_unit_overlaps)

    # Flatten list
    all_overlapping_units = [[x for y in z for x in y]
                             for z in all_overlapping_units]

    # Count occurence of each value
    overlapping_counts = [Counter(x) for x in all_overlapping_units]

    unit_distances = np.zeros((len(units), len(units)))
    for unit_num, this_dict in enumerate(overlapping_counts):
        keys, vals = list(zip(*this_dict.items()))
        unit_distances[unit_num][np.array(keys)] = np.array(vals)

    # Divide each row by total count for that unit to get fraction
    unit_distances_frac = unit_distances / np.array(spike_counts)[:, None]
    unit_distances_frac = unit_distances / np.array(spike_counts)[None, :]
    unit_distances_perc = unit_distances_frac * 100
    return unit_distances_perc


@jit(nogil=True)
def unit_similarity(this_unit_times, other_unit_times):
    this_unit_counter = 0
    other_unit_counter = 0
    for this_time in this_unit_times:
        for other_time in other_unit_times:
            if abs(this_time - other_time) <= 1.0:
                this_unit_counter += 1
                other_unit_counter += 1
    return this_unit_counter, other_unit_counter


def unit_similarity_NM(all_spk_times):
    unit_distances = np.zeros((len(all_spk_times), len(all_spk_times)))
    for this_unit in range(len(all_spk_times)):
        this_unit_times = all_spk_times[this_unit]
        for other_unit in range(len(all_spk_times)):
            if other_unit < this_unit:
                continue
            other_unit_times = all_spk_times[other_unit]
            this_unit_counter, other_unit_counter = \
                unit_similarity(this_unit_times, other_unit_times)
            unit_distances[this_unit, other_unit] = \
                100.0*(float(this_unit_counter)/len(this_unit_times))
            unit_distances[other_unit, this_unit] = \
                100.0*(float(other_unit_counter)/len(other_unit_times))
        # Print the progress to the window
        print("Unit %i of %i completed" % (this_unit+1, len(units)))
    return unit_distances


def parse_collision_mat(unit_distances, similarity_cutoff):
    """
    Parses the unit similarity matrix to find units that are too similar

    Inputs:
    unit_distances: matrix of unit similarity values
    similarity_cutoff: similarity value above which units are considered

    Outputs:
    unique_pairs: list of tuples of unit numbers
    unique_pairs_collisions: list of similarity values
    """
    similar_units = list(zip(*np.where(unit_distances > similarity_cutoff)))
    # Remove same unit
    similar_units = [x for x in similar_units if x[0] != x[1]]
    # Remove flipped duplicates
    unique_pairs = []
    unique_pairs_collisions = []
    for this_pair in similar_units:
        if this_pair not in unique_pairs \
                and this_pair[::-1] not in unique_pairs:
            unique_pairs.append(this_pair)
            unique_pairs_collisions.append(unit_distances[this_pair])
    return unique_pairs, unique_pairs_collisions


def plot_similarity_matrix(unit_distances, similarity_cutoff, output_dir):
    """
    Creates a figure showing raw similarity matrix and thresholded values
    
    Inputs:
    unit_distances: matrix of unit similarity values
    similarity_cutoff: threshold for considering units similar
    output_dir: directory to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Raw similarity matrix
    im1 = ax1.matshow(unit_distances, cmap='viridis')
    plt.colorbar(im1, ax=ax1)
    ax1.set_title('Raw Similarity Matrix')
    ax1.set_xlabel('Unit #')
    ax1.set_ylabel('Unit #')
    
    # Thresholded matrix
    thresholded = np.where(unit_distances > similarity_cutoff, unit_distances, np.nan)
    im2 = ax2.matshow(thresholded, cmap='hot')
    plt.colorbar(im2, ax=ax2)
    ax2.set_title(f'Values > {similarity_cutoff}% (similarity cutoff)')
    ax2.set_xlabel('Unit #')
    ax2.set_ylabel('Unit #')

    fig.suptitle('Unit Similarity Matrix')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'similarity_matrix.png'))
    plt.close()

def write_out_similarties(unique_pairs, unique_pairs_collisions, waveform_counts, out_path, mode='w', waveform_count_cutoff=None):
    """
    Writes out the unit similarity violations to a file

    Inputs:
    unique_pairs: list of tuples of unit numbers
    unique_pairs_collisions: list of similarity values
    waveform_counts: list of waveform counts for each unit
    out_path: path to write file to
    mode: write mode (default is 'w')
    waveform_count_cutoff: percentage threshold for waveform count comparison
    """
    # Generate dataframe with unit numbers, similarity and waveform counts
    unit1_waveforms = [waveform_counts[x[0]] for x in unique_pairs]
    unit2_waveforms = [waveform_counts[x[1]] for x in unique_pairs]
    
    similarity_frame = pd.DataFrame({
        'unit1': [x[0] for x in unique_pairs],
        'unit2': [x[1] for x in unique_pairs],
        'similarity': unique_pairs_collisions,
        'unit1_waveforms': unit1_waveforms,
        'unit2_waveforms': unit2_waveforms
    })
    
    if waveform_count_cutoff is not None:
        # Calculate percentage difference between waveform counts
        max_counts = np.maximum(unit1_waveforms, unit2_waveforms)
        min_counts = np.minimum(unit1_waveforms, unit2_waveforms)
        waveform_diff_percent = ((max_counts - min_counts) / max_counts) * 100
        similarity_frame['within_waveform_cutoff'] = waveform_diff_percent <= waveform_count_cutoff
    # Write dataframe to file
    with open(out_path, mode) as unit_similarity_violations:
        print(similarity_frame.to_string(), file=unit_similarity_violations)

############################################################

if __name__ == '__main__':
    # Get name of directory with the data files
    metadata_handler = imp_metadata(sys.argv)
    dir_name = metadata_handler.dir_name

    # Perform pipeline graph check
    this_pipeline_check = pipeline_graph_check(dir_name)
    this_pipeline_check.check_previous(script_path)
    this_pipeline_check.write_to_log(script_path, 'attempted')

    os.chdir(dir_name)
    print(f'Processing : {dir_name}')

    output_dir = os.path.join(dir_name, 'QA_output')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    warnings_file_path = os.path.join(output_dir, 'warnings.txt')
    out_path = os.path.join(output_dir, 'unit_similarity_violations.txt')

    params_dict = metadata_handler.params_dict
    similarity_cutoff = params_dict['qa_params']['similarity_cutoff']
    similarity_waveform_count_cutoff = params_dict['qa_params']['similarity_waveform_count_cutoff']
    sampling_rate = params_dict['sampling_rate']
    sampling_rate_ms = sampling_rate/1000.0

    # Open the hdf5 file
    with tables.open_file(metadata_handler.hdf5_name, 'r+') as hf5:
        # Get all the units from the hdf5 file
        units = hf5.list_nodes('/sorted_units')
        # Get all spiketimes and waveform counts from all units
        all_spk_times = [x.times[:]/sampling_rate_ms for x in units]
        waveform_counts = [x.waveforms.shape[0] for x in units]

    # Open a file to write these unit distance violations to -
    # these units are likely the same and one of them
    # will need to be removed from the HDF5 file

    print("==================")
    print("Similarity calculation starting")
    print(f"Similarity cutoff ::: {similarity_cutoff}")
    unit_distances = unit_similarity_abu(all_spk_times)
    unique_pairs, unique_pairs_collisions = parse_collision_mat(unit_distances, similarity_cutoff)
    write_out_similarties(unique_pairs, unique_pairs_collisions, waveform_counts, out_path, mode='w', 
                         waveform_count_cutoff=similarity_waveform_count_cutoff)
    print("Similarity calculation complete, results being saved to file")
    print("==================")
    
    # Generate visualization
    plot_similarity_matrix(unit_distances, similarity_cutoff, output_dir)

    # If the similarity goes beyond the defined cutoff,
    # write these unit numbers to warnings file
    if len(unique_pairs) > 0:
        with open(warnings_file_path, 'a') as f:
            print("", file=f)
            print('=== Similarity cutoff warning ===', file=f)
            print("Similarity cutoff exceeded for the following pairs of units",
                  file=f)
            print('Similarity cutoff ::: %f' % similarity_cutoff, file=f)
            print("", file=f)
        write_out_similarties(unique_pairs, unique_pairs_collisions, waveform_counts, warnings_file_path, mode='a',
                            waveform_count_cutoff=similarity_waveform_count_cutoff)
        with open(warnings_file_path, 'a') as f:
            print("", file=f)
            print('=== End Similarity cutoff warning ===', file=f)

    # Make a node for storing unit distances under /sorted_units.
    # First try to delete it, and pass if it exists
    with tables.open_file(metadata_handler.hdf5_name, 'r+') as hf5:
        if '/unit_distances' in hf5:
            hf5.remove_node('/unit_distances')
        hf5.create_array('/', 'unit_distances', unit_distances)

    ############################################################
    # Perform comparison in output of algorithms if desired
    ############################################################
    perform_comparison = False
    if perform_comparison:
        import pylab as plt
        unit_distances_perc = unit_similarity_abu(all_spk_times)
        unit_distances = unit_similarity_NM(all_spk_times)

        fig, ax = plt.subplots(1, 2)
        im = ax[0].matshow(unit_distances_perc.T)
        plt.colorbar(im, ax=ax[0])
        ax[0].set_title('New')
        im = ax[1].matshow(unit_distances)
        plt.colorbar(im, ax=ax[1])
        ax[1].set_title('OG')
        plt.show()

        plt.hist(unit_distances_perc.flatten(), bins=np.linspace(0, 100, 100),
                 label='New', histtype='step', color='r',
                 linewidth=2, alpha=0.5)
        plt.hist(unit_distances.flatten(), bins=np.linspace(0, 100, 100),
                 label='OG', histtype='step', color='b',
                 linewidth=2, alpha=0.5)
        plt.yscale('log')
        plt.legend()
        plt.show()

        from scipy.stats import pearsonr
        print(pearsonr(unit_distances_perc.flatten(), unit_distances.flatten()))

    # Write successful execution to log
    this_pipeline_check.write_to_log(script_path, 'completed')
