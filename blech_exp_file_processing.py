"""Functions for processing experimental data files"""

import os
import numpy as np
from utils.importrhdutilities import read_header

def process_files(dir_path):
    """Process data files and determine file type
    
    Args:
        dir_path (str): Path to data directory
        
    Returns:
        tuple: (file_type, ports, electrode_files, electrode_num_list)
    """
    file_list = os.listdir(dir_path)
    
    # Determine file type
    if 'auxiliary.dat' in file_list:
        file_type = 'one file per signal type'
    elif sum(['rhd' in x for x in file_list]) > 1:
        file_type = 'traditional'
    else:
        file_type = 'one file per channel'

    # Get electrode files and ports based on file type
    if file_type == 'one file per signal type':
        num_recorded_samples = len(np.fromfile(
            os.path.join(dir_path, 'time.dat'), dtype=np.dtype('float32')))
        amplifier_data = np.fromfile(
            os.path.join(dir_path, 'amplifier.dat'), dtype=np.dtype('uint16'))
        num_electrodes = int(len(amplifier_data)/num_recorded_samples)
        electrode_files = ['amplifier.dat' for _ in range(num_electrodes)]
        ports = ['A']*num_electrodes
        electrode_num_list = list(range(num_electrodes))
    elif file_type == 'one file per channel':
        electrodes_list = [name for name in file_list if name.startswith('amp-')]
        electrode_files = sorted(electrodes_list)
        ports = [x.split('-')[1] for x in electrode_files]
        electrode_num_list = [x.split('-')[2].split('.')[0] for x in electrode_files]
        ports.sort()
    else:  # traditional
        rhd_file_list = [x for x in file_list if 'rhd' in x]
        with open(os.path.join(dir_path, rhd_file_list[0]), 'rb') as f:
            header = read_header(f)
        ports = [x['port_prefix'] for x in header['amplifier_channels']]
        electrode_files = [x['native_channel_name'] for x in header['amplifier_channels']]
        electrode_num_list = [x.split('-')[1] for x in electrode_files]

    return file_type, ports, electrode_files, electrode_num_list
