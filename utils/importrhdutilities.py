# Adrian Foy September 2023

"""Imports a variety of Python functions used by 'LoadIntanRHD_Python.ipynb'
Jupyter Notebook.
"""

import struct
import math
import os
import time

import numpy as np

import matplotlib.pyplot as plt


def load_file(filename):
    """Loads .rhd file with provided filename, returning 'result' dict and
    'data_present' Boolean.
    """
    # Start timing
    tic = time.time()

    # Open file
    fid = open(filename, 'rb')
    filesize = os.path.getsize(filename)

    # Read file header
    header = read_header(fid)

    # Calculate how much data is present and summarize to console.
    data_present, filesize, num_blocks, num_samples = (
        calculate_data_size(header, filename, fid))

    # If .rhd file contains data, read all present data blocks into 'data'
    # dict, and verify the amount of data read.
    if data_present:
        data = read_all_data_blocks(header, num_samples, num_blocks, fid)
        check_end_of_file(filesize, fid)

    # Save information in 'header' to 'result' dict.
    result = {}
    header_to_result(header, result)

    # If .rhd file contains data, parse data into readable forms and, if
    # necessary, apply the same notch filter that was active during recording.
    if data_present:
        parse_data(header, data)
        apply_notch_filter(header, data)

        # Save recorded data in 'data' to 'result' dict.
        data_to_result(header, data, result)

    # Otherwise (.rhd file is just a header for One File Per Signal Type or
    # One File Per Channel data formats, in which actual data is saved in
    # separate .dat files), just return data as an empty list.
    else:
        data = []

    # Report how long read took.
    print('Done!  Elapsed time: {0:0.1f} seconds'.format(time.time() - tic))

    # Return 'result' dict.
    return result, data_present


def print_all_channel_names(result):
    """Searches through all present signal types in 'result' dict, and prints
    the names of these channels. Useful, for example, to determine names of
    channels that can be plotted.
    """
    if 'amplifier_channels' in result:
        print_names_in_group(result['amplifier_channels'])

    if 'aux_input_channels' in result:
        print_names_in_group(result['aux_input_channels'])

    if 'supply_voltage_channels' in result:
        print_names_in_group(result['supply_voltage_channels'])

    if 'board_adc_channels' in result:
        print_names_in_group(result['board_adc_channels'])

    if 'board_dig_in_channels' in result:
        print_names_in_group(result['board_dig_in_channels'])

    if 'board_dig_out_channels' in result:
        print_names_in_group(result['board_dig_out_channels'])


def print_names_in_group(signal_group):
    """Searches through all channels in this group and print them.
    """
    for this_channel in signal_group:
        print(this_channel['custom_channel_name'])


def find_channel_in_group(channel_name, signal_group):
    """Finds a channel with this name in this group, returning whether or not
    it's present and, if so, the position of this channel in signal_group.
    """
    for count, this_channel in enumerate(signal_group):
        if this_channel['custom_channel_name'] == channel_name:
            return True, count
    return False, 0


def find_channel_in_header(channel_name, header):
    """Looks through all present signal groups in header, searching for
    'channel_name'. If found, return the signal group and the index of that
    channel within the group.
    """
    signal_group_name = ''
    if 'amplifier_channels' in header:
        channel_found, channel_index = find_channel_in_group(
            channel_name, header['amplifier_channels'])
        if channel_found:
            signal_group_name = 'amplifier_channels'

    if not channel_found and 'aux_input_channels' in header:
        channel_found, channel_index = find_channel_in_group(
            channel_name, header['aux_input_channels'])
        if channel_found:
            signal_group_name = 'aux_input_channels'

    if not channel_found and 'supply_voltage_channels' in header:
        channel_found, channel_index = find_channel_in_group(
            channel_name, header['supply_voltage_channels'])
        if channel_found:
            signal_group_name = 'supply_voltage_channels'

    if not channel_found and 'board_adc_channels' in header:
        channel_found, channel_index = find_channel_in_group(
            channel_name, header['board_adc_channels'])
        if channel_found:
            signal_group_name = 'board_adc_channels'

    if not channel_found and 'board_dig_in_channels' in header:
        channel_found, channel_index = find_channel_in_group(
            channel_name, header['board_dig_in_channels'])
        if channel_found:
            signal_group_name = 'board_dig_in_channels'

    if not channel_found and 'board_dig_out_channels' in header:
        channel_found, channel_index = find_channel_in_group(
            channel_name, header['board_dig_out_channels'])
        if channel_found:
            signal_group_name = 'board_dig_out_channels'

    if channel_found:
        return True, signal_group_name, channel_index

    return False, '', 0


def read_header(fid):
    """Reads the Intan File Format header from the given file.
    """
    check_magic_number(fid)

    header = {}

    read_version_number(header, fid)
    set_num_samples_per_data_block(header)

    freq = {}

    read_sample_rate(header, fid)
    read_freq_settings(freq, fid)
    read_notch_filter_frequency(header, freq, fid)
    read_impedance_test_frequencies(freq, fid)
    read_notes(header, fid)
    read_num_temp_sensor_channels(header, fid)
    read_eval_board_mode(header, fid)
    read_reference_channel(header, fid)

    set_sample_rates(header, freq)
    set_frequency_parameters(header, freq)

    initialize_channels(header)

    read_signal_summary(header, fid)

    return header


def check_magic_number(fid):
    """Checks magic number at beginning of file to verify this is an Intan
    Technologies RHD data file.
    """
    magic_number, = struct.unpack('<I', fid.read(4))
    if magic_number != int('c6912702', 16):
        raise UnrecognizedFileError('Unrecognized file type.')


def read_version_number(header, fid):
    """Reads version number (major and minor) from fid. Stores them into
    header['version']['major'] and header['version']['minor'].
    """
    version = {}
    (version['major'], version['minor']) = struct.unpack('<hh', fid.read(4))
    header['version'] = version

    print('\nReading Intan Technologies RHD Data File, Version {}.{}\n'
          .format(version['major'], version['minor']))


def set_num_samples_per_data_block(header):
    """Determines how many samples are present per data block (60 or 128),
    depending on version. Data files v2.0 or later have 128 samples per block,
    otherwise 60.
    """
    header['num_samples_per_data_block'] = 60
    if header['version']['major'] > 1:
        header['num_samples_per_data_block'] = 128


def read_sample_rate(header, fid):
    """Reads sample rate from fid. Stores it into header['sample_rate'].
    """
    header['sample_rate'], = struct.unpack('<f', fid.read(4))


def read_freq_settings(freq, fid):
    """Reads amplifier frequency settings from fid. Stores them in 'freq' dict.
    """
    (freq['dsp_enabled'],
     freq['actual_dsp_cutoff_frequency'],
     freq['actual_lower_bandwidth'],
     freq['actual_upper_bandwidth'],
     freq['desired_dsp_cutoff_frequency'],
     freq['desired_lower_bandwidth'],
     freq['desired_upper_bandwidth']) = struct.unpack('<hffffff', fid.read(26))


def read_notch_filter_frequency(header, freq, fid):
    """Reads notch filter mode from fid, and stores frequency (in Hz) in
    'header' and 'freq' dicts.
    """
    notch_filter_mode, = struct.unpack('<h', fid.read(2))
    header['notch_filter_frequency'] = 0
    if notch_filter_mode == 1:
        header['notch_filter_frequency'] = 50
    elif notch_filter_mode == 2:
        header['notch_filter_frequency'] = 60
    freq['notch_filter_frequency'] = header['notch_filter_frequency']


def read_impedance_test_frequencies(freq, fid):
    """Reads desired and actual impedance test frequencies from fid, and stores
    them (in Hz) in 'freq' dicts.
    """
    (freq['desired_impedance_test_frequency'],
     freq['actual_impedance_test_frequency']) = (
         struct.unpack('<ff', fid.read(8)))


def read_notes(header, fid):
    """Reads notes as QStrings from fid, and stores them as strings in
    header['notes'] dict.
    """
    header['notes'] = {'note1': read_qstring(fid),
                       'note2': read_qstring(fid),
                       'note3': read_qstring(fid)}


def read_num_temp_sensor_channels(header, fid):
    """Stores number of temp sensor channels in
    header['num_temp_sensor_channels']. Temp sensor data may be saved from
    versions 1.1 and later.
    """
    header['num_temp_sensor_channels'] = 0
    if ((header['version']['major'] == 1 and header['version']['minor'] >= 1)
            or (header['version']['major'] > 1)):
        header['num_temp_sensor_channels'], = struct.unpack('<h', fid.read(2))


def read_eval_board_mode(header, fid):
    """Stores eval board mode in header['eval_board_mode']. Board mode is saved
    from versions 1.3 and later.
    """
    header['eval_board_mode'] = 0
    if ((header['version']['major'] == 1 and header['version']['minor'] >= 3)
            or (header['version']['major'] > 1)):
        header['eval_board_mode'], = struct.unpack('<h', fid.read(2))


def read_reference_channel(header, fid):
    """Reads name of reference channel as QString from fid, and stores it as
    a string in header['reference_channel']. Data files v2.0 or later include
    reference channel.
    """
    if header['version']['major'] > 1:
        header['reference_channel'] = read_qstring(fid)


def set_sample_rates(header, freq):
    """Determines what the sample rates are for various signal types, and
    stores them in 'freq' dict.
    """
    freq['amplifier_sample_rate'] = header['sample_rate']
    freq['aux_input_sample_rate'] = header['sample_rate'] / 4
    freq['supply_voltage_sample_rate'] = (header['sample_rate'] /
                                          header['num_samples_per_data_block'])
    freq['board_adc_sample_rate'] = header['sample_rate']
    freq['board_dig_in_sample_rate'] = header['sample_rate']


def set_frequency_parameters(header, freq):
    """Stores frequency parameters (set in other functions) in
    header['frequency_parameters']
    """
    header['frequency_parameters'] = freq


def initialize_channels(header):
    """Creates empty lists for each type of data channel and stores them in
    'header' dict.
    """
    header['spike_triggers'] = []
    header['amplifier_channels'] = []
    header['aux_input_channels'] = []
    header['supply_voltage_channels'] = []
    header['board_adc_channels'] = []
    header['board_dig_in_channels'] = []
    header['board_dig_out_channels'] = []


def read_signal_summary(header, fid):
    """Reads signal summary from data file header and stores information for
    all signal groups and their channels in 'header' dict.
    """
    number_of_signal_groups, = struct.unpack('<h', fid.read(2))
    for signal_group in range(1, number_of_signal_groups + 1):
        add_signal_group_information(header, fid, signal_group)
    add_num_channels(header)
    print_header_summary(header)


def add_signal_group_information(header, fid, signal_group):
    """Adds information for a signal group and all its channels to 'header'
    dict.
    """
    signal_group_name = read_qstring(fid)
    signal_group_prefix = read_qstring(fid)
    (signal_group_enabled, signal_group_num_channels, _) = struct.unpack(
        '<hhh', fid.read(6))

    if signal_group_num_channels > 0 and signal_group_enabled > 0:
        for _ in range(0, signal_group_num_channels):
            add_channel_information(header, fid, signal_group_name,
                                    signal_group_prefix, signal_group)


def add_channel_information(header, fid, signal_group_name,
                            signal_group_prefix, signal_group):
    """Reads a new channel's information from fid and appends it to 'header'
    dict.
    """
    (new_channel, new_trigger_channel, channel_enabled,
     signal_type) = read_new_channel(
         fid, signal_group_name, signal_group_prefix, signal_group)
    append_new_channel(header, new_channel, new_trigger_channel,
                       channel_enabled, signal_type)


def read_new_channel(fid, signal_group_name, signal_group_prefix,
                     signal_group):
    """Reads a new channel's information from fid.
    """
    new_channel = {'port_name': signal_group_name,
                   'port_prefix': signal_group_prefix,
                   'port_number': signal_group}
    new_channel['native_channel_name'] = read_qstring(fid)
    new_channel['custom_channel_name'] = read_qstring(fid)
    (new_channel['native_order'],
     new_channel['custom_order'],
     signal_type, channel_enabled,
     new_channel['chip_channel'],
     new_channel['board_stream']) = (
         struct.unpack('<hhhhhh', fid.read(12)))
    new_trigger_channel = {}
    (new_trigger_channel['voltage_trigger_mode'],
     new_trigger_channel['voltage_threshold'],
     new_trigger_channel['digital_trigger_channel'],
     new_trigger_channel['digital_edge_polarity']) = (
         struct.unpack('<hhhh', fid.read(8)))
    (new_channel['electrode_impedance_magnitude'],
     new_channel['electrode_impedance_phase']) = (
         struct.unpack('<ff', fid.read(8)))

    return new_channel, new_trigger_channel, channel_enabled, signal_type


def append_new_channel(header, new_channel, new_trigger_channel,
                       channel_enabled, signal_type):
    """"Appends 'new_channel' to 'header' dict depending on if channel is
    enabled and the signal type.
    """
    if not channel_enabled:
        return

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
        raise UnknownChannelTypeError('Unknown channel type.')


def add_num_channels(header):
    """Adds channel numbers for all signal types to 'header' dict.
    """
    header['num_amplifier_channels'] = len(header['amplifier_channels'])
    header['num_aux_input_channels'] = len(header['aux_input_channels'])
    header['num_supply_voltage_channels'] = len(
        header['supply_voltage_channels'])
    header['num_board_adc_channels'] = len(header['board_adc_channels'])
    header['num_board_dig_in_channels'] = len(header['board_dig_in_channels'])
    header['num_board_dig_out_channels'] = len(
        header['board_dig_out_channels'])


def header_to_result(header, result):
    """Merges header information from .rhd file into a common 'result' dict.
    If any fields have been allocated but aren't relevant (for example, no
    channels of this type exist), does not copy those entries into 'result'.
    """
    if header['num_amplifier_channels'] > 0:
        result['spike_triggers'] = header['spike_triggers']
        result['amplifier_channels'] = header['amplifier_channels']

    result['notes'] = header['notes']
    result['frequency_parameters'] = header['frequency_parameters']

    if header['version']['major'] > 1:
        result['reference_channel'] = header['reference_channel']

    if header['num_aux_input_channels'] > 0:
        result['aux_input_channels'] = header['aux_input_channels']

    if header['num_supply_voltage_channels'] > 0:
        result['supply_voltage_channels'] = header['supply_voltage_channels']

    if header['num_board_adc_channels'] > 0:
        result['board_adc_channels'] = header['board_adc_channels']

    if header['num_board_dig_in_channels'] > 0:
        result['board_dig_in_channels'] = header['board_dig_in_channels']

    if header['num_board_dig_out_channels'] > 0:
        result['board_dig_out_channels'] = header['board_dig_out_channels']

    return result


def print_header_summary(header):
    """Prints summary of contents of RHD header to console.
    """
    print('Found {} amplifier channel{}.'.format(
        header['num_amplifier_channels'],
        plural(header['num_amplifier_channels'])))
    print('Found {} auxiliary input channel{}.'.format(
        header['num_aux_input_channels'],
        plural(header['num_aux_input_channels'])))
    print('Found {} supply voltage channel{}.'.format(
        header['num_supply_voltage_channels'],
        plural(header['num_supply_voltage_channels'])))
    print('Found {} board ADC channel{}.'.format(
        header['num_board_adc_channels'],
        plural(header['num_board_adc_channels'])))
    print('Found {} board digital input channel{}.'.format(
        header['num_board_dig_in_channels'],
        plural(header['num_board_dig_in_channels'])))
    print('Found {} board digital output channel{}.'.format(
        header['num_board_dig_out_channels'],
        plural(header['num_board_dig_out_channels'])))
    print('Found {} temperature sensors channel{}.'.format(
        header['num_temp_sensor_channels'],
        plural(header['num_temp_sensor_channels'])))
    print('')


def get_timestamp_signed(header):
    """Checks version (major and minor) in 'header' to determine if data
    recorded from this version of Intan software saved timestamps as signed or
    unsigned integer. Returns True if signed, False if unsigned.
    """
    # All Intan software v1.2 and later saves timestamps as signed
    if header['version']['major'] > 1:
        return True

    if header['version']['major'] == 1 and header['version']['minor'] >= 2:
        return True

    # Intan software before v1.2 saves timestamps as unsigned
    return False


def plural(number_of_items):
    """Utility function to pluralize words based on the number of items.
    """
    if number_of_items == 1:
        return ''
    return 's'


def get_bytes_per_data_block(header):
    """Calculates the number of bytes in each 60 or 128 sample datablock."""
    # Depending on the system used to acquire the data,
    # 'num_samples_per_data_block' will be either 60 (USB Interface Board)
    # or 128 (Recording Controller).
    # Use this number along with numbers of channels to accrue a sum of how
    # many bytes each data block should contain.

    # Timestamps (one channel always present): Start with 4 bytes per sample.
    bytes_per_block = bytes_per_signal_type(
        header['num_samples_per_data_block'],
        1,
        4)

    # Amplifier data: Add 2 bytes per sample per enabled amplifier channel.
    bytes_per_block += bytes_per_signal_type(
        header['num_samples_per_data_block'],
        header['num_amplifier_channels'],
        2)

    # Auxiliary data: Add 2 bytes per sample per enabled aux input channel.
    # Note that aux inputs are sample 4x slower than amplifiers, so there
    # are 1/4 as many samples.
    bytes_per_block += bytes_per_signal_type(
        header['num_samples_per_data_block'] / 4,
        header['num_aux_input_channels'],
        2)

    # Supply voltage: Add 2 bytes per sample per enabled vdd channel.
    # Note that aux inputs are sampled once per data block
    # (60x or 128x slower than amplifiers), so there are
    # 1/60 or 1/128 as many samples.
    bytes_per_block += bytes_per_signal_type(
        1,
        header['num_supply_voltage_channels'],
        2)

    # Analog inputs: Add 2 bytes per sample per enabled analog input channel.
    bytes_per_block += bytes_per_signal_type(
        header['num_samples_per_data_block'],
        header['num_board_adc_channels'],
        2)

    # Digital inputs: Add 2 bytes per sample.
    # Note that if at least 1 channel is enabled, a single 16-bit sample
    # is saved, with each bit corresponding to an individual channel.
    if header['num_board_dig_in_channels'] > 0:
        bytes_per_block += bytes_per_signal_type(
            header['num_samples_per_data_block'],
            1,
            2)

    # Digital outputs: Add 2 bytes per sample.
    # Note that if at least 1 channel is enabled, a single 16-bit sample
    # is saved, with each bit corresponding to an individual channel.
    if header['num_board_dig_out_channels'] > 0:
        bytes_per_block += bytes_per_signal_type(
            header['num_samples_per_data_block'],
            1,
            2)

    # Temp sensor: Add 2 bytes per sample per enabled temp sensor channel.
    # Note that temp sensor inputs are sampled once per data block
    # (60x or 128x slower than amplifiers), so there are
    # 1/60 or 1/128 as many samples.
    if header['num_temp_sensor_channels'] > 0:
        bytes_per_block += bytes_per_signal_type(
            1,
            header['num_temp_sensor_channels'],
            2)

    return bytes_per_block


def bytes_per_signal_type(num_samples, num_channels, bytes_per_sample):
    """Calculates the number of bytes, per data block, for a signal type
    provided the number of samples (per data block), the number of enabled
    channels, and the size of each sample in bytes.
    """
    return num_samples * num_channels * bytes_per_sample


def read_one_data_block(data, header, indices, fid):
    """Reads one 60 or 128 sample data block from fid into data,
    at the location indicated by indices."""
    samples_per_block = header['num_samples_per_data_block']

    # In version 1.2, we moved from saving timestamps as unsigned
    # integers to signed integers to accommodate negative (adjusted)
    # timestamps for pretrigger data
    read_timestamps(fid,
                    data,
                    indices,
                    samples_per_block,
                    get_timestamp_signed(header))

    read_analog_signals(fid,
                        data,
                        indices,
                        samples_per_block,
                        header)

    read_digital_signals(fid,
                         data,
                         indices,
                         samples_per_block,
                         header)


def read_timestamps(fid, data, indices, num_samples, timestamp_signed):
    """Reads timestamps from binary file as a NumPy array, indexing them
    into 'data'.
    """
    start = indices['amplifier']
    end = start + num_samples
    format_sign = 'i' if timestamp_signed else 'I'
    format_expression = '<' + format_sign * num_samples
    read_length = 4 * num_samples
    data['t_amplifier'][start:end] = np.array(struct.unpack(
        format_expression, fid.read(read_length)))


def read_analog_signals(fid, data, indices, samples_per_block, header):
    """Reads all analog signal types present in RHD files: amplifier_data,
    aux_input_data, supply_voltage_data, temp_sensor_data, and board_adc_data,
    into 'data' dict.
    """

    read_analog_signal_type(fid,
                            data['amplifier_data'],
                            indices['amplifier'],
                            samples_per_block,
                            header['num_amplifier_channels'])

    read_analog_signal_type(fid,
                            data['aux_input_data'],
                            indices['aux_input'],
                            int(samples_per_block / 4),
                            header['num_aux_input_channels'])

    read_analog_signal_type(fid,
                            data['supply_voltage_data'],
                            indices['supply_voltage'],
                            1,
                            header['num_supply_voltage_channels'])

    read_analog_signal_type(fid,
                            data['temp_sensor_data'],
                            indices['supply_voltage'],
                            1,
                            header['num_temp_sensor_channels'])

    read_analog_signal_type(fid,
                            data['board_adc_data'],
                            indices['board_adc'],
                            samples_per_block,
                            header['num_board_adc_channels'])


def read_digital_signals(fid, data, indices, samples_per_block, header):
    """Reads all digital signal types present in RHD files: board_dig_in_raw
    and board_dig_out_raw, into 'data' dict.
    """

    read_digital_signal_type(fid,
                             data['board_dig_in_raw'],
                             indices['board_dig_in'],
                             samples_per_block,
                             header['num_board_dig_in_channels'])

    read_digital_signal_type(fid,
                             data['board_dig_out_raw'],
                             indices['board_dig_out'],
                             samples_per_block,
                             header['num_board_dig_out_channels'])


def read_analog_signal_type(fid, dest, start, num_samples, num_channels):
    """Reads data from binary file as a NumPy array, indexing them into
    'dest', which should be an analog signal type within 'data', for example
    data['amplifier_data'] or data['aux_input_data']. Each sample is assumed
    to be of dtype 'uint16'.
    """

    if num_channels < 1:
        return
    end = start + num_samples
    tmp = np.fromfile(fid, dtype='uint16', count=num_samples*num_channels)
    dest[range(num_channels), start:end] = (
        tmp.reshape(num_channels, num_samples))


def read_digital_signal_type(fid, dest, start, num_samples, num_channels):
    """Reads data from binary file as a NumPy array, indexing them into
    'dest', which should be a digital signal type within 'data', either
    data['board_dig_in_raw'] or data['board_dig_out_raw'].
    """

    if num_channels < 1:
        return
    end = start + num_samples
    dest[start:end] = np.array(struct.unpack(
        '<' + 'H' * num_samples, fid.read(2 * num_samples)))


def data_to_result(header, data, result):
    """Merges data from all present signals into a common 'result' dict. If
    any signal types have been allocated but aren't relevant (for example,
    no channels of this type exist), does not copy those entries into 'result'.
    """
    if header['num_amplifier_channels'] > 0:
        result['t_amplifier'] = data['t_amplifier']
        result['amplifier_data'] = data['amplifier_data']

    if header['num_aux_input_channels'] > 0:
        result['t_aux_input'] = data['t_aux_input']
        result['aux_input_data'] = data['aux_input_data']

    if header['num_supply_voltage_channels'] > 0:
        result['t_supply_voltage'] = data['t_supply_voltage']
        result['supply_voltage_data'] = data['supply_voltage_data']

    if header['num_temp_sensor_channels'] > 0:
        result['t_temp_sensor'] = data['t_temp_sensor']

    if header['num_board_adc_channels'] > 0:
        result['t_board_adc'] = data['t_board_adc']
        result['board_adc_data'] = data['board_adc_data']

    if (header['num_board_dig_in_channels'] > 0
            or header['num_board_dig_out_channels'] > 0):
        result['t_dig'] = data['t_dig']

    if header['num_board_dig_in_channels'] > 0:
        result['board_dig_in_data'] = data['board_dig_in_data']

    if header['num_board_dig_out_channels'] > 0:
        result['board_dig_out_data'] = data['board_dig_out_data']

    return result


def plot_channel(channel_name, result):
    """Plots all data associated with channel specified as 'channel_name' in
    'result' dict.
    """
    # Find channel that corresponds to this name
    channel_found, signal_type, signal_index = find_channel_in_header(
        channel_name, result)

    # Plot this channel
    if channel_found:
        _, ax = plt.subplots()
        # fig, ax = plt.subplots()
        ax.set_title(channel_name)
        ax.set_xlabel('Time (s)')

        if signal_type == 'amplifier_channels':
            ylabel = 'Voltage (microVolts)'
            signal_data_name = 'amplifier_data'
            t_vector = result['t_amplifier']

        elif signal_type == 'aux_input_channels':
            ylabel = 'Voltage (Volts)'
            signal_data_name = 'aux_input_data'
            t_vector = result['t_aux_input']

        elif signal_type == 'supply_voltage_channels':
            ylabel = 'Voltage (Volts)'
            signal_data_name = 'supply_voltage_data'
            t_vector = result['t_supply_voltage']

        elif signal_type == 'board_adc_channels':
            ylabel = 'Voltage (Volts)'
            signal_data_name = 'board_adc_data'
            t_vector = result['t_board_adc']

        elif signal_type == 'board_dig_in_channels':
            ylabel = 'Digital In Events (High or Low)'
            signal_data_name = 'board_dig_in_data'
            t_vector = result['t_dig']

        elif signal_type == 'board_dig_out_channels':
            ylabel = 'Digital Out Events (High or Low)'
            signal_data_name = 'board_dig_out_data'
            t_vector = result['t_dig']

        else:
            raise ChannelNotFoundError(
                'Plotting failed; signal type ', signal_type, ' not found')

        ax.set_ylabel(ylabel)

        ax.plot(t_vector, result[signal_data_name][signal_index, :])
        ax.margins(x=0, y=0)

    else:
        raise ChannelNotFoundError(
            'Plotting failed; channel ', channel_name, ' not found')


def read_qstring(fid):
    """Reads Qt style QString.

    The first 32-bit unsigned number indicates the length of the string
    (in bytes). If this number equals 0xFFFFFFFF, the string is null.

    Strings are stored as unicode.
    """

    length, = struct.unpack('<I', fid.read(4))
    if length == int('ffffffff', 16):
        return ""

    if length > (os.fstat(fid.fileno()).st_size - fid.tell() + 1):
        print(length)
        raise QStringError('Length too long.')

    # convert length from bytes to 16-bit Unicode words
    length = int(length / 2)

    data = []
    for _ in range(0, length):
        c, = struct.unpack('<H', fid.read(2))
        data.append(c)

    return ''.join([chr(c) for c in data])


def calculate_data_size(header, filename, fid):
    """Calculates how much data is present in this file. Returns:
    data_present: Bool, whether any data is present in file
    filesize: Int, size (in bytes) of file
    num_blocks: Int, number of 60 or 128-sample data blocks present
    num_samples: Int, number of samples present in file
    """
    bytes_per_block = get_bytes_per_data_block(header)

    # Determine filesize and if any data is present.
    filesize = os.path.getsize(filename)
    data_present = False
    bytes_remaining = filesize - fid.tell()
    if bytes_remaining > 0:
        data_present = True

    # If the file size is somehow different than expected, raise an error.
    if bytes_remaining % bytes_per_block != 0:
        raise FileSizeError(
            'Something is wrong with file size : '
            'should have a whole number of data blocks')

    # Calculate how many data blocks are present.
    num_blocks = int(bytes_remaining / bytes_per_block)

    num_samples = calculate_num_samples(header, num_blocks)

    print_record_time_summary(num_samples['amplifier'],
                              header['sample_rate'],
                              data_present)

    return data_present, filesize, num_blocks, num_samples


def calculate_num_samples(header, num_data_blocks):
    """Calculates number of samples for each signal type, storing the results
    in num_samples dict for later use.
    """
    samples_per_block = header['num_samples_per_data_block']
    num_samples = {}
    num_samples['amplifier'] = int(samples_per_block * num_data_blocks)
    num_samples['aux_input'] = int((samples_per_block / 4) * num_data_blocks)
    num_samples['supply_voltage'] = int(num_data_blocks)
    num_samples['board_adc'] = int(samples_per_block * num_data_blocks)
    num_samples['board_dig_in'] = int(samples_per_block * num_data_blocks)
    num_samples['board_dig_out'] = int(samples_per_block * num_data_blocks)
    return num_samples


def print_record_time_summary(num_amp_samples, sample_rate, data_present):
    """Prints summary of how much recorded data is present in RHD file
    to console.
    """
    record_time = num_amp_samples / sample_rate

    if data_present:
        print('File contains {:0.3f} seconds of data.  '
              'Amplifiers were sampled at {:0.2f} kS/s.'
              .format(record_time, sample_rate / 1000))
    else:
        print('Header file contains no data.  '
              'Amplifiers were sampled at {:0.2f} kS/s.'
              .format(sample_rate / 1000))


def read_all_data_blocks(header, num_samples, num_blocks, fid):
    """Reads all data blocks present in file, allocating memory for and
    returning 'data' dict containing all data.
    """
    data, indices = initialize_memory(header, num_samples)
    print("Reading data from file...")
    print_step = 10
    percent_done = print_step
    for i in range(num_blocks):
        read_one_data_block(data, header, indices, fid)
        advance_indices(indices, header['num_samples_per_data_block'])
        percent_done = print_progress(i, num_blocks, print_step, percent_done)
    return data


def initialize_memory(header, num_samples):
    """Pre-allocates NumPy arrays for each signal type that will be filled
    during this read, and initializes unique indices for data access to each
    signal type.
    """
    print('\nAllocating memory for data...')
    data = {}

    # Create zero array for amplifier timestamps.
    t_dtype = np.int_ if get_timestamp_signed(header) else np.uint
    data['t_amplifier'] = np.zeros(num_samples['amplifier'], t_dtype)

    # Create zero array for amplifier data.
    data['amplifier_data'] = np.zeros(
        [header['num_amplifier_channels'], num_samples['amplifier']],
        dtype=np.uint)

    # Create zero array for aux input data.
    data['aux_input_data'] = np.zeros(
        [header['num_aux_input_channels'], num_samples['aux_input']],
        dtype=np.uint)

    # Create zero array for supply voltage data.
    data['supply_voltage_data'] = np.zeros(
        [header['num_supply_voltage_channels'], num_samples['supply_voltage']],
        dtype=np.uint)

    # Create zero array for temp sensor data.
    data['temp_sensor_data'] = np.zeros(
        [header['num_temp_sensor_channels'], num_samples['supply_voltage']],
        dtype=np.uint)

    # Create zero array for board ADC data.
    data['board_adc_data'] = np.zeros(
        [header['num_board_adc_channels'], num_samples['board_adc']],
        dtype=np.uint)

    # By default, this script interprets digital events (digital inputs
    # and outputs) as booleans. if unsigned int values are preferred
    # (0 for False, 1 for True), replace the 'dtype=np.bool_' argument
    # with 'dtype=np.uint' as shown.
    # The commented lines below illustrate this for digital input data;
    # the same can be done for digital out.

    # data['board_dig_in_data'] = np.zeros(
    #     [header['num_board_dig_in_channels'], num_samples['board_dig_in']],
    #     dtype=np.uint)
    # Create 16-row zero array for digital in data, and 1-row zero array for
    # raw digital in data (each bit of 16-bit entry represents a different
    # digital input.)
    data['board_dig_in_data'] = np.zeros(
        [header['num_board_dig_in_channels'], num_samples['board_dig_in']],
        dtype=np.bool_)
    data['board_dig_in_raw'] = np.zeros(
        num_samples['board_dig_in'],
        dtype=np.uint)

    # Create 16-row zero array for digital out data, and 1-row zero array for
    # raw digital out data (each bit of 16-bit entry represents a different
    # digital output.)
    data['board_dig_out_data'] = np.zeros(
        [header['num_board_dig_out_channels'], num_samples['board_dig_out']],
        dtype=np.bool_)
    data['board_dig_out_raw'] = np.zeros(
        num_samples['board_dig_out'],
        dtype=np.uint)

    # Create dict containing each signal type's indices, and set all to zero.
    indices = {}
    indices['amplifier'] = 0
    indices['aux_input'] = 0
    indices['supply_voltage'] = 0
    indices['board_adc'] = 0
    indices['board_dig_in'] = 0
    indices['board_dig_out'] = 0

    return data, indices


def advance_indices(indices, samples_per_block):
    """Advances indices used for data access by suitable values per data block.
    """
    # Signal types sampled at the sample rate:
    # Index should be incremented by samples_per_block every data block.
    indices['amplifier'] += samples_per_block
    indices['board_adc'] += samples_per_block
    indices['board_dig_in'] += samples_per_block
    indices['board_dig_out'] += samples_per_block

    # Signal types sampled at 1/4 the sample rate:
    # Index should be incremented by samples_per_block / 4 every data block.
    indices['aux_input'] += int(samples_per_block / 4)

    # Signal types sampled once per data block:
    # Index should be incremented by 1 every data block.
    indices['supply_voltage'] += 1


def check_end_of_file(filesize, fid):
    """Checks that the end of the file was reached at the expected position.
    If not, raise FileSizeError.
    """
    bytes_remaining = filesize - fid.tell()
    if bytes_remaining != 0:
        raise FileSizeError('Error: End of file not reached.')


def parse_data(header, data):
    """Parses raw data into user readable and interactable forms (for example,
    extracting raw digital data to separate channels and scaling data to units
    like microVolts, degrees Celsius, or seconds.)
    """
    print('Parsing data...')
    extract_digital_data(header, data)
    scale_analog_data(header, data)
    scale_timestamps(header, data)


def scale_timestamps(header, data):
    """Verifies no timestamps are missing, and scales timestamps to seconds.
    """
    # Check for gaps in timestamps.
    num_gaps = np.sum(np.not_equal(
        data['t_amplifier'][1:]-data['t_amplifier'][:-1], 1))
    if num_gaps == 0:
        print('No missing timestamps in data.')
    else:
        print('Warning: {0} gaps in timestamp data found.  '
              'Time scale will not be uniform!'
              .format(num_gaps))

    # Scale time steps (units = seconds).
    data['t_amplifier'] = data['t_amplifier'] / header['sample_rate']
    data['t_aux_input'] = data['t_amplifier'][range(
        0, len(data['t_amplifier']), 4)]
    data['t_supply_voltage'] = data['t_amplifier'][range(
        0, len(data['t_amplifier']), header['num_samples_per_data_block'])]
    data['t_board_adc'] = data['t_amplifier']
    data['t_dig'] = data['t_amplifier']
    data['t_temp_sensor'] = data['t_supply_voltage']


def scale_analog_data(header, data):
    """Scales all analog data signal types (amplifier data, aux input data,
    supply voltage data, board ADC data, and temp sensor data) to suitable
    units (microVolts, Volts, deg C).
    """
    # Scale amplifier data (units = microVolts).
    data['amplifier_data'] = np.multiply(
        0.195, (data['amplifier_data'].astype(np.int32) - 32768))

    # Scale aux input data (units = Volts).
    data['aux_input_data'] = np.multiply(
        37.4e-6, data['aux_input_data'])

    # Scale supply voltage data (units = Volts).
    data['supply_voltage_data'] = np.multiply(
        74.8e-6, data['supply_voltage_data'])

    # Scale board ADC data (units = Volts).
    if header['eval_board_mode'] == 1:
        data['board_adc_data'] = np.multiply(
            152.59e-6, (data['board_adc_data'].astype(np.int32) - 32768))
    elif header['eval_board_mode'] == 13:
        data['board_adc_data'] = np.multiply(
            312.5e-6, (data['board_adc_data'].astype(np.int32) - 32768))
    else:
        data['board_adc_data'] = np.multiply(
            50.354e-6, data['board_adc_data'])

    # Scale temp sensor data (units = deg C).
    data['temp_sensor_data'] = np.multiply(
        0.01, data['temp_sensor_data'])


def extract_digital_data(header, data):
    """Extracts digital data from raw (a single 16-bit vector where each bit
    represents a separate digital input channel) to a more user-friendly 16-row
    list where each row represents a separate digital input channel. Applies to
    digital input and digital output data.
    """
    for i in range(header['num_board_dig_in_channels']):
        data['board_dig_in_data'][i, :] = np.not_equal(
            np.bitwise_and(
                data['board_dig_in_raw'],
                (1 << header['board_dig_in_channels'][i]['native_order'])
            ),
            0)

    for i in range(header['num_board_dig_out_channels']):
        data['board_dig_out_data'][i, :] = np.not_equal(
            np.bitwise_and(
                data['board_dig_out_raw'],
                (1 << header['board_dig_out_channels'][i]['native_order'])
            ),
            0)


def apply_notch_filter(header, data):
    """Checks header to determine if notch filter should be applied, and if so,
    apply notch filter to all signals in data['amplifier_data'].
    """
    # If data was not recorded with notch filter turned on, return without
    # applying notch filter. Similarly, if data was recorded from Intan RHX
    # software version 3.0 or later, any active notch filter was already
    # applied to the saved data, so it should not be re-applied.
    if (header['notch_filter_frequency'] == 0
            or header['version']['major'] >= 3):
        return

    # Apply notch filter individually to each channel in order
    print('Applying notch filter...')
    print_step = 10
    percent_done = print_step
    for i in range(header['num_amplifier_channels']):
        data['amplifier_data'][i, :] = notch_filter(
            data['amplifier_data'][i, :],
            header['sample_rate'],
            header['notch_filter_frequency'],
            10)

        percent_done = print_progress(i, header['num_amplifier_channels'],
                                      print_step, percent_done)


def notch_filter(signal_in, f_sample, f_notch, bandwidth):
    """Implements a notch filter (e.g., for 50 or 60 Hz) on vector 'signal_in'.

    f_sample = sample rate of data (input Hz or Samples/sec)
    f_notch = filter notch frequency (input Hz)
    bandwidth = notch 3-dB bandwidth (input Hz).  A bandwidth of 10 Hz is
    recommended for 50 or 60 Hz notch filters; narrower bandwidths lead to
    poor time-domain properties with an extended ringing response to
    transient disturbances.

    Example:  If neural data was sampled at 30 kSamples/sec
    and you wish to implement a 60 Hz notch filter:

    out = notch_filter(signal_in, 30000, 60, 10);
    """
    # Calculate parameters used to implement IIR filter
    t_step = 1.0/f_sample
    f_c = f_notch*t_step
    signal_length = len(signal_in)
    iir_parameters = calculate_iir_parameters(bandwidth, t_step, f_c)

    # Create empty signal_out NumPy array
    signal_out = np.zeros(signal_length)

    # Set the first 2 samples of signal_out to signal_in.
    # If filtering a continuous data stream, change signal_out[0:1] to the
    # previous final two values of signal_out
    signal_out[0] = signal_in[0]
    signal_out[1] = signal_in[1]

    # Run filter.
    for i in range(2, signal_length):
        signal_out[i] = calculate_iir(i, signal_in, signal_out, iir_parameters)

    return signal_out


def calculate_iir_parameters(bandwidth, t_step, f_c):
    """Calculates parameters d, b, a0, a1, a2, a, b0, b1, and b2 used for
    IIR filter and return them in a dict.
    """
    parameters = {}
    d = math.exp(-2.0*math.pi*(bandwidth/2.0)*t_step)
    b = (1.0 + d*d) * math.cos(2.0*math.pi*f_c)
    a0 = 1.0
    a1 = -b
    a2 = d*d
    a = (1.0 + d*d)/2.0
    b0 = 1.0
    b1 = -2.0 * math.cos(2.0*math.pi*f_c)
    b2 = 1.0

    parameters['d'] = d
    parameters['b'] = b
    parameters['a0'] = a0
    parameters['a1'] = a1
    parameters['a2'] = a2
    parameters['a'] = a
    parameters['b0'] = b0
    parameters['b1'] = b1
    parameters['b2'] = b2
    return parameters


def calculate_iir(i, signal_in, signal_out, iir_parameters):
    """Calculates a single sample of IIR filter passing signal_in through
    iir_parameters, resulting in signal_out.
    """
    sample = ((
        iir_parameters['a'] * iir_parameters['b2'] * signal_in[i - 2]
        + iir_parameters['a'] * iir_parameters['b1'] * signal_in[i - 1]
        + iir_parameters['a'] * iir_parameters['b0'] * signal_in[i]
        - iir_parameters['a2'] * signal_out[i - 2]
        - iir_parameters['a1'] * signal_out[i - 1])
        / iir_parameters['a0'])

    return sample


def print_progress(i, target, print_step, percent_done):
    """Prints progress of an arbitrary process based on position i / target,
    printing a line showing completion percentage for each print_step / 100.
    """
    fraction_done = 100 * (1.0 * i / target)
    if fraction_done >= percent_done:
        print('{}% done...'.format(percent_done))
        percent_done += print_step

    return percent_done


class UnrecognizedFileError(Exception):
    """Exception returned when reading a file as an RHD header yields an
    invalid magic number (indicating this is not an RHD header file).
    """


class UnknownChannelTypeError(Exception):
    """Exception returned when a channel field in RHD header does not have
    a recognized signal_type value. Accepted values are:
    0: amplifier channel
    1: aux input channel
    2: supply voltage channel
    3: board adc channel
    4: dig in channel
    5: dig out channel
    """


class FileSizeError(Exception):
    """Exception returned when file reading fails due to the file size
    being invalid or the calculated file size differing from the actual
    file size.
    """


class QStringError(Exception):
    """Exception returned when reading a QString fails because it is too long.
    """


class ChannelNotFoundError(Exception):
    """Exception returned when plotting fails due to the specified channel
    not being found.
    """
