import sys, struct, math, os, time
import numpy as np
import sys, struct

#from .intanutil.read_header import read_header
#from .intanutil.get_bytes_per_data_block import get_bytes_per_data_block

# constants
AMPLIFIER_BIT_MICROVOLTS = 0.195
UINT16_BIT_OFFSET = int(2**15)
AUX_BIT_VOLTS = 37.4e-6
SUPPLY_BIT_VOLTS = 74.8e-6
ADC_BIT_VOLTS_1 = 152.59e-6
ADC_BIT_VOLTS_0 = 50.353e-6
TEMP_BIT_CELCIUS = 0.01

def read_qstring(fid):
    """Read Qt style QString.  

    The first 32-bit unsigned number indicates the length of the string (in bytes).  
    If this number equals 0xFFFFFFFF, the string is null.

    Strings are stored as unicode.
    """

    length, = struct.unpack('<I', fid.read(4))
    if length == int('ffffffff', 16): return ""

    if length > (os.fstat(fid.fileno()).st_size - fid.tell() + 1) :
        print(length)
        raise Exception('Length too long.')

    # convert length from bytes to 16-bit Unicode words
    length = int(length / 2)

    data = []
    for i in range(0, length):
        c, = struct.unpack('<H', fid.read(2))
        data.append(c)

    if sys.version_info >= (3,0):
        a = ''.join([chr(c) for c in data])
    else:
        a = ''.join([unichr(c) for c in data])
    
    return a

def read_header(fid):
    """Reads the Intan File Format header from the given file."""

    # Check 'magic number' at beginning of file to make sure this is an Intan
    # Technologies RHD2000 data file.
    magic_number, = struct.unpack('<I', fid.read(4))
    if magic_number != int('c6912702', 16):
        raise Exception('Unrecognized file type.')

    header = {}
    # Read version number.
    version = {}
    (version['major'], version['minor']) = struct.unpack('<hh', fid.read(4))
    header['version'] = version

    print('')
    print('Reading Intan Technologies RHD2000 Data File, Version {}.{}'.format(
        version['major'], version['minor']))
    print('')

    freq = {}

    # Read information of sampling rate and amplifier frequency settings.
    header['sample_rate'], = struct.unpack('<f', fid.read(4))
    (freq['dsp_enabled'], freq['actual_dsp_cutoff_frequency'],
     freq['actual_lower_bandwidth'], freq['actual_upper_bandwidth'],
     freq['desired_dsp_cutoff_frequency'], freq['desired_lower_bandwidth'],
     freq['desired_upper_bandwidth']) = struct.unpack('<hffffff', fid.read(26))

    # This tells us if a software 50/60 Hz notch filter was enabled during
    # the data acquisition.
    notch_filter_mode, = struct.unpack('<h', fid.read(2))
    header['notch_filter_frequency'] = 0
    if notch_filter_mode == 1:
        header['notch_filter_frequency'] = 50
    elif notch_filter_mode == 2:
        header['notch_filter_frequency'] = 60
    freq['notch_filter_frequency'] = header['notch_filter_frequency']

    (freq['desired_impedance_test_frequency'],
     freq['actual_impedance_test_frequency']) = struct.unpack('<ff',
                                                              fid.read(8))

    note1 = read_qstring(fid)
    note2 = read_qstring(fid)
    note3 = read_qstring(fid)
    header['notes'] = {'note1': note1, 'note2': note2, 'note3': note3}

    # If data file is from GUI v1.1 or later, see if temperature sensor data was saved.
    header['num_temp_sensor_channels'] = 0
    if (version['major'] == 1 and version['minor'] >= 1) or (
            version['major'] > 1):
        header['num_temp_sensor_channels'], = struct.unpack('<h', fid.read(2))

    # If data file is from GUI v1.3 or later, load eval board mode.
    header['eval_board_mode'] = 0
    if ((version['major'] == 1) and
        (version['minor'] >= 3)) or (version['major'] > 1):
        header['eval_board_mode'], = struct.unpack('<h', fid.read(2))

    # Place frequency-related information in data structure. (Note: much of this structure is set above)
    freq['amplifier_sample_rate'] = header['sample_rate']
    freq['aux_input_sample_rate'] = header['sample_rate'] / 4
    freq['supply_voltage_sample_rate'] = header['sample_rate'] / 60
    freq['board_adc_sample_rate'] = header['sample_rate']
    freq['board_dig_in_sample_rate'] = header['sample_rate']

    header['frequency_parameters'] = freq

    # Create structure arrays for each type of data channel.
    header['spike_triggers'] = []
    header['amplifier_channels'] = []
    header['aux_input_channels'] = []
    header['supply_voltage_channels'] = []
    header['board_adc_channels'] = []
    header['board_dig_in_channels'] = []
    header['board_dig_out_channels'] = []

    # Read signal summary from data file header.

    number_of_signal_groups, = struct.unpack('<h', fid.read(2))

    for signal_group in range(0, number_of_signal_groups):
        signal_group_name = read_qstring(fid)
        signal_group_prefix = read_qstring(fid)
        (signal_group_enabled, signal_group_num_channels,
         signal_group_num_amp_channels) = struct.unpack('<hhh', fid.read(6))

        if (signal_group_num_channels > 0) and (signal_group_enabled > 0):
            for signal_channel in range(0, signal_group_num_channels):
                new_channel = {'port_name': signal_group_name,
                               'port_prefix': signal_group_prefix,
                               'port_number': signal_group}
                new_channel['native_channel_name'] = read_qstring(fid)
                new_channel['custom_channel_name'] = read_qstring(fid)
                (new_channel['native_order'], new_channel['custom_order'],
                 signal_type, channel_enabled, new_channel['chip_channel'],
                 new_channel['board_stream']) = struct.unpack('<hhhhhh',
                                                              fid.read(12))
                new_trigger_channel = {}
                (new_trigger_channel['voltage_trigger_mode'],
                 new_trigger_channel['voltage_threshold'],
                 new_trigger_channel['digital_trigger_channel'],
                 new_trigger_channel['digital_edge_polarity']) = struct.unpack(
                     '<hhhh', fid.read(8))
                (new_channel['electrode_impedance_magnitude'],
                 new_channel['electrode_impedance_phase']) = struct.unpack(
                     '<ff', fid.read(8))

                if channel_enabled:
                    if signal_type == 0:
                        header['amplifier_channels'].append(new_channel)
                        header['spike_triggers'].append(new_trigger_channel)
                    elif signal_type == 1:
                        header['aux_input_channels'].append(new_channel)
                    elif signal_type == 2:
                        header['supply_voltage_channels'].append(new_channel)
                    elif signal_type == 3:
                        header['board_adc_channels'].append(new_channel)
                    elif signal_type == 4:
                        header['board_dig_in_channels'].append(new_channel)
                    elif signal_type == 5:
                        header['board_dig_out_channels'].append(new_channel)
                    else:
                        raise Exception('Unknown channel type.')

    # Summarize contents of data file.
    header['num_amplifier_channels'] = len(header['amplifier_channels'])
    header['num_aux_input_channels'] = len(header['aux_input_channels'])
    header['num_supply_voltage_channels'] = len(header[
        'supply_voltage_channels'])
    header['num_board_adc_channels'] = len(header['board_adc_channels'])
    header['num_board_dig_in_channels'] = len(header['board_dig_in_channels'])
    header['num_board_dig_out_channels'] = len(header[
        'board_dig_out_channels'])

    return header

def get_bytes_per_data_block(header):
    """Calculates the number of bytes in each 60-sample datablock."""

    # Each data block contains 60 amplifier samples.
    bytes_per_block = 60 * 4  # timestamp data
    bytes_per_block = bytes_per_block + 60 * 2 * header['num_amplifier_channels']

    # Auxiliary inputs are sampled 4x slower than amplifiers
    bytes_per_block = bytes_per_block + 15 * 2 * header['num_aux_input_channels']

    # Supply voltage is sampled 60x slower than amplifiers
    bytes_per_block = bytes_per_block + 1 * 2 * header['num_supply_voltage_channels']

    # Board analog inputs are sampled at same rate as amplifiers
    bytes_per_block = bytes_per_block + 60 * 2 * header['num_board_adc_channels']

    # Board digital inputs are sampled at same rate as amplifiers
    if header['num_board_dig_in_channels'] > 0:
        bytes_per_block = bytes_per_block + 60 * 2

    # Board digital outputs are sampled at same rate as amplifiers
    if header['num_board_dig_out_channels'] > 0:
        bytes_per_block = bytes_per_block + 60 * 2

    # Temp sensor is sampled 60x slower than amplifiers
    if header['num_temp_sensor_channels'] > 0:
        bytes_per_block = bytes_per_block + 1 * 2 * header['num_temp_sensor_channels']

    return bytes_per_block
  
def get_file_info(filename, no_floats=False):
    """Reads Intan Technologies RHD2000 data file generated by evaluation board GUI.
    Data are returned in a dictionary, for future extensibility.
    """

    tic = time.time()
    fid = open(filename, 'rb')
    filesize = os.path.getsize(filename)

    header = read_header(fid)

    #print('Found {} amplifier channel{}.'.format(header[
    #    'num_amplifier_channels'], (header['num_amplifier_channels'])))
    #print('Found {} auxiliary input channel{}.'.format(header[
    #    'num_aux_input_channels'], (header['num_aux_input_channels'])))
    #print('Found {} supply voltage channel{}.'.format(header[
    #    'num_supply_voltage_channels'], (header[
    #        'num_supply_voltage_channels'])))
    #print('Found {} board ADC channel{}.'.format(header[
    #    'num_board_adc_channels'], (header['num_board_adc_channels'])))
    #print('Found {} board digital input channel{}.'.format(header[
    #    'num_board_dig_in_channels'], (header[
    #        'num_board_dig_in_channels'])))
    #print('Found {} board digital output channel{}.'.format(header[
    #    'num_board_dig_out_channels'], (header[
    #        'num_board_dig_out_channels'])))
    #print('Found {} temperature sensors channel{}.'.format(header[
    #    'num_temp_sensor_channels'], (header[
    #        'num_temp_sensor_channels'])))
    #print('')

    # Determine how many samples the data file contains.
    bytes_per_block = get_bytes_per_data_block(header)

    # How many data blocks remain in this file?
    data_present = False
    bytes_remaining = filesize - fid.tell()
    if bytes_remaining > 0:
        data_present = True

    if bytes_remaining % bytes_per_block != 0:
        raise Exception(
            'Something is wrong with file size : should have a whole number of data blocks')

    num_data_blocks = int(bytes_remaining / bytes_per_block)

    num_amplifier_samples = 60 * num_data_blocks

    return num_amplifier_samples
