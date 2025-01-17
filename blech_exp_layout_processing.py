"""Functions for processing electrode layout information"""

import os
import pandas as pd
from utils.blech_utils import entry_checker

def process_layout(dir_path, dir_name, file_type, ports, electrode_files, 
                  electrode_num_list, args):
    """Process electrode layout information
    
    Args:
        dir_path (str): Path to data directory
        dir_name (str): Name of directory
        file_type (str): Type of data files
        ports (list): List of ports
        electrode_files (list): List of electrode files
        electrode_num_list (list): List of electrode numbers
        args (argparse.Namespace): Command line arguments
        
    Returns:
        dict: Layout information dictionary
    """
    layout_file_path = os.path.join(dir_path, f"{dir_name}_electrode_layout.csv")

    # Determine whether to use existing layout file
    use_existing = _should_use_existing_layout(layout_file_path, args)
    
    if not use_existing:
        _create_new_layout_file(layout_file_path, electrode_files, ports, 
                              electrode_num_list, args)

    # Process layout frame
    layout_frame = _process_layout_frame(layout_file_path, args)
    
    # Process EMG information
    emg_info = _process_emg_info(layout_frame, args)
    
    return {
        'file_type': file_type,
        'regions': list(layout_frame.CAR_group.unique()),
        'ports': list(pd.unique(ports)),
        'electrode_layout': dict(list(layout_frame.groupby('CAR_group').electrode_ind.apply(lambda x: [x.tolist()]))),
        'emg': emg_info
    }

def _should_use_existing_layout(layout_file_path, args):
    """Determine whether to use existing layout file"""
    if not os.path.exists(layout_file_path):
        return False
        
    if args.programmatic:
        return not bool(args.car_groups)
    
    if args.use_layout_file:
        return True
        
    if args.car_groups:
        return False
        
    use_csv_str, _ = entry_checker(
        msg="Layout file detected...use what's there? (y/yes/no/n) :: ",
        check_func=lambda x: x in ['y', 'yes', 'n', 'no'],
        fail_response='Please enter [y, yes, n, no]')
    return use_csv_str in ['y', 'yes']

def _create_new_layout_file(layout_file_path, electrode_files, ports, 
                          electrode_num_list, args):
    """Create new layout file"""
    layout_frame = pd.DataFrame({
        'filename': electrode_files,
        'port': ports,
        'electrode_num': electrode_num_list,
        'electrode_ind': range(len(electrode_files)),
        'CAR_group': pd.Series()
    })
    
    layout_frame = layout_frame[['filename', 'electrode_ind', 'electrode_num', 
                               'port', 'CAR_group']]
    layout_frame.to_csv(layout_file_path, index=False)
    
    if not args.programmatic:
        print('Please fill in car groups / regions\n'
              'emg and none are case-specific\n'
              'Indicate different CARS from same region as GC1,GC2...etc')
        entry_checker(
            msg='Let me know when its done (y/yes) :: ',
            check_func=lambda x: x in ['y', 'yes'],
            fail_response='Please say y or yes')

def _process_layout_frame(layout_file_path, args):
    """Process layout frame from file"""
    layout_frame = pd.read_csv(layout_file_path)
    
    if args.programmatic and args.car_groups:
        car_groups = [x.strip().lower() for x in args.car_groups.split(',')]
        layout_frame['CAR_group'] = car_groups
    else:
        layout_frame['CAR_group'] = layout_frame['CAR_group'].str.lower().str.strip()
        
    return layout_frame

def _process_emg_info(layout_frame, args):
    """Process EMG information from layout frame"""
    emg_groups = [x for x in layout_frame.CAR_group.unique() if 'emg' in str(x)]
    
    if not emg_groups:
        return {'port': [], 'electrodes': [], 'muscle': ''}
        
    emg_electrodes = layout_frame[layout_frame.CAR_group.isin(emg_groups)].electrode_ind.tolist()
    emg_ports = layout_frame[layout_frame.electrode_ind.isin(emg_electrodes)].port.unique().tolist()
    
    if args.programmatic:
        muscle = args.emg_muscle if args.emg_muscle else ''
    else:
        muscle, _ = entry_checker(
            msg='Enter EMG muscle name :: ',
            check_func=lambda x: True,
            fail_response='Please enter a valid muscle name')
            
    return {
        'port': emg_ports,
        'electrodes': emg_electrodes,
        'muscle': muscle
    }
