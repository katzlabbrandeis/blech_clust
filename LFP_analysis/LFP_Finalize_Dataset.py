import easygui
import numpy as np
import glob
import tables
from itertools import product
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import sys
import os

# =============================================================================
# #Channel Check Processing
# =============================================================================

# If directory provided with script, use that otherwise ask
try:
    #dir_name = os.path.dirname(sys.argv[1])
    dir_name = sys.argv[1]
except:
    dir_name = easygui.diropenbox(msg = 'Select directory with HDF5 file')

hdf5_name = glob.glob(dir_name + '/*.h5')[0]

#Open the hdf5 file
hf5 = tables.open_file(hdf5_name, 'r+')
parsed_lfp_addr = '/Parsed_LFP'
taste_num = len([x for x in hf5.list_nodes(parsed_lfp_addr) \
        if 'dig_in' in str(x)])
channel_num = hf5.list_nodes(parsed_lfp_addr)[0].shape[0]

#Ask user to check LFP traces to ensure channels are not shorted/bad in order to remove said channel from further processing
try:
    channel_check =  list(map(int,easygui.multchoicebox(
            msg = 'Choose the channel numbers that you '\
                    'want to REMOVE from further analyses. '
                    'Click clear all and ok if all channels are good', 
                    choices = tuple([i for i in range(channel_num)]))))
except:
    channel_check = []

try:
    taste_check = list(map(int, easygui.multchoicebox(
            msg = 'Chose the taste numbers that you want to '\
                    'REMOVE from further analyses. Click clear all '\
                    'and ok if all tastes are good',
                    choices = tuple([i for i in range(taste_num)]))))
except:
    taste_check = []

# Create dataframe with all tastes and channels
flag_frame = pd.DataFrame(list(product(range(taste_num),range(channel_num))),
                columns = ['Dig_In','Channel'])

# Create list of all flagged rows in dataframe and mark
flagged_rows = np.isin(list(flag_frame['Dig_In']),taste_check) + \
                    np.isin(list(flag_frame['Channel']),channel_check)
flag_frame['Error_Flag'] = flagged_rows * 1 

#Write out to file
flag_frame.to_hdf(hdf5_name, parsed_lfp_addr + '/flagged_channels')
hf5.close()
