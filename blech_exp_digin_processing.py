"""Functions for processing digital input information"""

from utils.blech_utils import entry_checker
from utils.read_file import DigInHandler

def process_digins(dir_path, file_type, args):
    """Process digital input information
    
    Args:
        dir_path (str): Path to data directory
        file_type (str): Type of data files
        args (argparse.Namespace): Command line arguments
        
    Returns:
        dict: Digital input information dictionary
    """
    dig_handler = DigInHandler(dir_path, file_type)
    dig_handler.get_dig_in_files()
    dig_handler.get_trial_data()

    # Process taste information
    taste_info = _process_taste_info(dig_handler, args)
    
    # Process laser information  
    laser_info = _process_laser_info(dig_handler, args)
    
    # Write out dig-in frame
    _write_digin_frame(dig_handler)
    
    return {
        'dig_ins': {
            'nums': dig_handler.dig_in_frame.dig_in_nums.tolist(),
            'trial_counts': dig_handler.dig_in_frame.trial_counts.tolist(),
        },
        'taste_params': taste_info,
        'laser_params': laser_info
    }

def _process_taste_info(dig_handler, args):
    """Process taste-related digital inputs"""
    if not any(dig_handler.dig_in_frame.trial_counts > 0):
        return {
            'dig_in_nums': [],
            'trial_count': [],
            'tastes': [],
            'concs': [],
            'pal_rankings': []
        }
        
    # Get taste dig-ins
    if args.programmatic:
        if not args.taste_digins:
            raise ValueError('Taste dig-ins not provided, use --taste-digins')
        taste_dig_inds = [int(x) for x in args.taste_digins.split(',')]
    else:
        taste_dig_inds = _get_taste_digins()
        
    dig_handler.dig_in_frame.loc[taste_dig_inds, 'taste_bool'] = True
    dig_handler.dig_in_frame.taste_bool.fillna(False, inplace=True)
    
    # Get taste parameters
    tastes = _get_tastes(taste_dig_inds, args)
    concentrations = _get_concentrations(taste_dig_inds, args)
    palatability = _get_palatability(len(tastes), args)
    
    # Update dig_handler frame
    dig_handler.dig_in_frame.loc[taste_dig_inds, 'taste'] = tastes
    dig_handler.dig_in_frame.loc[taste_dig_inds, 'concentration'] = concentrations
    dig_handler.dig_in_frame.loc[taste_dig_inds, 'palatability'] = palatability
    
    return {
        'dig_in_nums': dig_handler.dig_in_frame.loc[taste_dig_inds, 'dig_in_nums'].tolist(),
        'trial_count': dig_handler.dig_in_frame.loc[taste_dig_inds, 'trial_counts'].tolist(),
        'tastes': tastes,
        'concs': concentrations,
        'pal_rankings': palatability
    }

def _process_laser_info(dig_handler, args):
    """Process laser-related digital inputs"""
    # Get laser dig-in
    if args.programmatic:
        laser_dig_ind = [int(x) for x in args.laser_digin.split(',')] if args.laser_digin else []
    else:
        laser_dig_ind = _get_laser_digin()
        
    if not laser_dig_ind:
        return {
            'dig_in_nums': [],
            'trial_count': [],
            'onset': None,
            'duration': None,
            'virus_region': '',
            'opto_loc': ''
        }
        
    # Update dig_handler frame
    dig_handler.dig_in_frame.loc[laser_dig_ind, 'laser_bool'] = True
    dig_handler.dig_in_frame.laser_bool.fillna(False, inplace=True)
    
    # Get laser parameters
    onset, duration = _get_laser_params(args)
    virus_region = _get_virus_region(args)
    opto_loc = _get_opto_location(args)
    
    dig_handler.dig_in_frame.loc[laser_dig_ind, 'laser_params'] = str([onset, duration])
    
    return {
        'dig_in_nums': dig_handler.dig_in_frame.loc[laser_dig_ind, 'dig_in_nums'].tolist(),
        'trial_count': dig_handler.dig_in_frame.loc[laser_dig_ind, 'trial_counts'].tolist(),
        'onset': onset,
        'duration': duration,
        'virus_region': virus_region,
        'opto_loc': opto_loc
    }

def _write_digin_frame(dig_handler):
    """Write digital input frame to file"""
    cols = dig_handler.dig_in_frame.columns.tolist()
    dig_handler.dig_in_frame = dig_handler.dig_in_frame[
        [x for x in cols if x != 'pulse_times'] + ['pulse_times']]
    dig_handler.write_out_frame()

# Helper functions for user input
def _get_taste_digins():
    """Get taste digital inputs from user"""
    dig_in_str, _ = entry_checker(
        msg=' INDEX of Taste dig_ins used (IN ORDER, anything separated) :: ',
        check_func=lambda x: all(n.isdigit() for n in re.findall('[0-9]+', x)),
        fail_response='Please enter integers only')
    return [int(x) for x in re.findall('[0-9]+', dig_in_str)]

def _get_tastes(taste_dig_inds, args):
    """Get taste names"""
    if args.programmatic:
        if not args.tastes:
            raise ValueError('Tastes not provided, use --tastes')
        return args.tastes.split(',')
    
    taste_str, _ = entry_checker(
        msg=' Tastes names used (IN ORDER, anything separated [no punctuation in name]) :: ',
        check_func=lambda x: len(re.findall('[A-Za-z]+', x)) == len(taste_dig_inds),
        fail_response='Please enter as many tastes as digins')
    return re.findall('[A-Za-z]+', taste_str)

def _get_concentrations(taste_dig_inds, args):
    """Get taste concentrations"""
    if args.programmatic:
        if not args.concentrations:
            raise ValueError('Concentrations not provided, use --concentrations')
        return [float(x) for x in args.concentrations.split(',')]
    
    conc_str, _ = entry_checker(
        msg='Corresponding concs used (in M, IN ORDER, COMMA separated) :: ',
        check_func=lambda x: len(x.split(',')) == len(taste_dig_inds),
        fail_response='Please enter as many concentrations as digins')
    return [float(x) for x in conc_str.split(',')]

def _get_palatability(num_tastes, args):
    """Get palatability rankings"""
    if args.programmatic:
        if not args.palatability:
            raise ValueError('Palatability rankings not provided, use --palatability')
        return [int(x) for x in args.palatability.split(',')]
    
    pal_str, _ = entry_checker(
        msg=f'Enter palatability rankings (IN ORDER, anything separated), higher number = more palatable :: ',
        check_func=lambda x: all(1 <= int(n) <= num_tastes for n in re.findall('[1-9]+', x)),
        fail_response=f'Please enter numbers 1<=x<={num_tastes}')
    return [int(x) for x in re.findall('[1-9]+', pal_str)]

def _get_laser_digin():
    """Get laser digital input"""
    laser_str, _ = entry_checker(
        msg='Laser dig_in index, <BLANK> for none :: ',
        check_func=lambda x: all(n.isdigit() for n in re.findall('[0-9]+', x)) if x else True,
        fail_response='Please enter a single, valid integer')
    return [int(laser_str)] if laser_str else []

def _get_laser_params(args):
    """Get laser parameters"""
    if args.programmatic:
        if not args.laser_params:
            raise ValueError('Laser parameters not provided, use --laser-params')
        params = [int(x) for x in args.laser_params.split(',')]
        return params[0], params[1]
    
    laser_str, _ = entry_checker(
        msg='Laser onset_time, duration (ms, IN ORDER, anything separated) :: ',
        check_func=lambda x: len(re.findall('[0-9]+', x)) == 2,
        fail_response='Please enter two, valid integers')
    nums = re.findall('[0-9]+', laser_str)
    return int(nums[0]), int(nums[1])

def _get_virus_region(args):
    """Get virus region"""
    if args.programmatic:
        return args.virus_region if args.virus_region else ''
    
    region, _ = entry_checker(
        msg='Enter virus region :: ',
        check_func=lambda x: True,
        fail_response='Please enter a valid region')
    return region

def _get_opto_location(args):
    """Get opto-fiber location"""
    if args.programmatic:
        return args.opto_loc if args.opto_loc else ''
    
    loc, _ = entry_checker(
        msg='Enter opto-fiber location :: ',
        check_func=lambda x: True,
        fail_response='Please enter a valid location')
    return loc
