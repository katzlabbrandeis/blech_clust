# ==============================
# Setup
# ==============================

#Import necessary tools
import numpy as np
import tables
import easygui
import os
import glob
import matplotlib.pyplot as plt
import re
import sys
from tqdm import tqdm, trange
#Import specific functions in order to filter the data file
from scipy.signal import butter
from scipy.signal import filtfilt

# ==============================
# Define Functions 
# ==============================

def get_filtered_electrode(data, low_pass, high_pass, sampling_rate):
    el = 0.195*(data)
    m, n = butter(
            2, 
            [2.0*int(low_pass)/sampling_rate, 2.0*int(high_pass)/sampling_rate], 
            btype = 'bandpass')
    filt_el = filtfilt(m, n, el)
    return filt_el

# ==============================
# Collect user input needed for later processing 
# ==============================

#Get name of directory where the data files and hdf5 file sits, 
#and change to that directory for processing
if len(sys.argv)>1:
    dir_name = sys.argv[1]
else:
    dir_name = easygui.diropenbox()
os.chdir(dir_name)

#Specify filtering parameters (linear-phase finite impulse response filter) 
#to define filter specificity across electrodes
Boxes = ['low','high','Sampling Rate']
freqparam = list(map(int,easygui.multenterbox(
    'Specify LFP bandpass filtering paramters and sampling rate',
    'Low-Frequency Cut-off (Hz)', 
    Boxes, [1,300,30000]))) 

# Ask use whether they would like to extract LFPs from the
# start or end of the taste delivery signal
taste_signal_choice = \
                easygui.buttonbox(\
                'Should trials be marked using the START or END of the taste delivery pulse?', 
                'Please select', 
                choices = ['Start', 'End'], default_choice = 'Start')
print('Marking trials from {} of taste delivery pulse'.format(taste_signal_choice.upper()))

if taste_signal_choice is 'Start':
        diff_val = 1
elif taste_signal_choice is 'End':
        diff_val = -1

# ==============================
# Open HDF5 File 
# ==============================

#Look for the hdf5 file in the directory
hdf5_name = glob.glob('**.h5')[0]

#Open the hdf5 file
hf5 = tables.open_file(hdf5_name, 'r+')

# ==============================
# Select channels to read
# ==============================

#Create vector of electode numbers that have neurons 
#on them (from unit_descriptor table)
#Some electrodes may record from more than one neuron 
#(shown as repeated number in unit_descriptor); 
#Remove these duplicates within array

electrodegroup = np.unique(hf5.root.unit_descriptor[:]['electrode_number'])

## List all appropriate dat files
Raw_Electrodefiles = np.sort(glob.glob('*amp*dat*'))
Raw_Electrodefiles = Raw_Electrodefiles[electrodegroup]

# ==============================
# Extract Raw Data 
# ==============================

#Check if LFP data is already within file and remove node if so. 
#Create new raw LFP group within H5 file. 
try:
    hf5.remove_node('/raw_LFP', recursive = True)
except:
    pass
hf5.create_group('/', 'raw_LFP')

#Loop through each neuron-recording electrode (from .dat files), 
#filter data, and create array in new LFP node

final_sampling_rate = 1000
new_intersample_interval = freqparam[2]/final_sampling_rate

# Pull out signal for each electrode, down_sample, bandpass filter and store in HDF5
print('Extracting raw LFPs')
for i in trange(len(Raw_Electrodefiles)):
    data = np.fromfile(Raw_Electrodefiles[i], dtype = np.dtype('int16'))
    data_down = np.mean(data.reshape((-1, int(new_intersample_interval))), axis = -1)
    filt_el_down = get_filtered_electrode(data = data_down,
                                        low_pass = freqparam[0],
                                        high_pass = freqparam[1],
                                        sampling_rate = final_sampling_rate)

    # Zero padding to 3 digits because code get screwy with sorting electrodes
    # if that isn't done
    hf5.create_array('/raw_LFP','electrode{:0>3}'.format(electrodegroup[i]), filt_el_down)
    hf5.flush()
    del data, data_down, filt_el_down

# Grab the names of the arrays containing digital inputs, 
# and pull the data into a numpy array
dig_in_nodes = hf5.list_nodes('/digital_in')
dig_in = []
dig_in_pathname = []
for node in dig_in_nodes:
    dig_in_pathname.append(node._v_pathname)
    exec("dig_in.append(hf5.root.digital_in.%s[:])" \
                % dig_in_pathname[-1].split('/')[-1])
dig_in = np.array(dig_in)

# The tail end of the pulse generates a negative value when passed through diff
# This method removes the need for a "for" loop

diff_points = list(np.where(np.diff(dig_in) == diff_val))
diff_points[1] = diff_points[1]//30
change_points = [diff_points[1][diff_points[0]==this_dig_in] \
                for this_dig_in in range(len(dig_in))]

# Show the user the number of trials on each digital 
#input channel, and ask them to confirm
check = easygui.ynbox(
        msg = 'Digital input channels: ' + str(dig_in_pathname) + \
                '\n' + 'No. of trials: ' + str([len(changes) \
                for changes in change_points]), 
                title = 'Check and confirm the number of ' + \
                'trials detected on digital input channels', 
                default_choice = 'Yes')

# Go ahead only if the user approves by saying yes
if check:
    pass
else:
    print("Well, if you don't agree, blech_clust can't do much!")
    sys.exit()

# ==============================
# Write-Out Extracted LFP 
# ==============================

# Ask the user which digital input channels should be 
# used for slicing out LFP arrays, and convert the channel 
# numbers into integers for pulling stuff out of change_points
dig_in_channels = easygui.multchoicebox(
        msg = 'Which digital input channels should be used ' + \
        'to slice out LFP data trial-wise?', 
        choices = ([path for path in dig_in_pathname]))
dig_in_channel_nums = []
for i in range(len(dig_in_pathname)):
    if dig_in_pathname[i] in dig_in_channels:
        dig_in_channel_nums.append(i)

# Grab the names of the arrays containing LFP recordings
lfp_nodes = hf5.list_nodes('/raw_LFP')

# Make the Parsed_LFP node in the hdf5 file if it doesn't exist, else move on
try:
    hf5.remove_node('/Parsed_LFP', recursive = True)
except:
    pass
hf5.create_group('/', 'Parsed_LFP')

# Create array marking which channel were chosen for further analysis
# Made in root folder for backward compatibility of code
# Code further below simply enumerates arrays in Parsed_LFP
if "/Parsed_LFP_channels" in hf5:
        hf5.remove_node('/Parsed_LFP_channels')
hf5.create_array('/', 'Parsed_LFP_channels', electrodegroup)
hf5.flush()

# Ask if this analysis is looking at more than 1 trial and/or taste
#msg   = "Do you want to create LFPs for more than " + \
#        "ONE trial (i.e. did you do taste trials and not affective) ?"
#trial_check = easygui.buttonbox(msg,choices = ["Yes","No"])

trial_check = easygui.buttonbox('Please select the experimental paradigm',
        choices = ['Affective','Taste'], default_choice = 'Taste')

# Run through the tastes if user said there are more than 1 trial
if trial_check == "Taste":

    # Ask the user for the pre and post stimulus durations to 
    #be pulled out, and convert to integers
    durations = easygui.multenterbox(
            msg = 'What are the signal durations pre and post ' + \
            'stimulus that you want to pull out', 
            fields = ['Pre stimulus (ms)', 'Post stimulus (ms)'],
            values = [2000,5000])
    for i in range(len(durations)):
        durations[i] = int(durations[i])

    # Remove dig_ins which are not relevant
    change_points_fin = [change_points[x] for x in range(len(change_points))\
                    if x in dig_in_channel_nums]
    
    # Make markers to slice trials for every dig_on
    all_trial_markers = [[(x-durations[0],x+durations[1]) \
                    for x in this_dig_in_markers] \
                    for this_dig_in_markers in change_points_fin]
    
    # Extract trials for every channel for every dig_in
    print('Parsing LFPs')
    all_channel_trials = []
    for channel in tqdm(lfp_nodes):
            this_channel_trials = [ np.asarray([channel[marker_tuple[0]:marker_tuple[1]] \
                            for marker_tuple in this_dig_in]) \
                            for this_dig_in in all_trial_markers]
            all_channel_trials.append(this_channel_trials)
    
    # Resort data to have 4 arrays (one for every dig_in) 
    # with dims (channels , trials, time)
    for dig_in in dig_in_channel_nums:
            this_taste_LFP = np.asarray([\
                            channel[dig_in] for channel in all_channel_trials])
    
            # Put the LFP data for this taste in hdf5 file under /Parsed_LFP
            hf5.create_array('/Parsed_LFP', 'dig_in_%i_LFPs' \
                        % (dig_in), this_taste_LFP)
            hf5.flush()
        
if trial_check == "Affective":
    # There's no trials so just a big recording chunk
    num_electrodes = len(lfp_nodes) 
    this_taste_LFPs = np.array(
                        [lfp_nodes[electrode][change_points[dig_in_channel_nums[0]][0] \
                                :change_points[dig_in_channel_nums[0]][1]] \
                            for electrode in range(num_electrodes)])
    
    # Put the LFP data for this session in hdf5 file under /Parsed_LFP
    hf5.create_array('/Parsed_LFP', 'dig_in_%i_LFPs' \
            % (dig_in_channel_nums[0]), this_taste_LFPs)
    hf5.flush()

# Ask about subplotting
########################################
# Channel check plots are now made automatically (Abu 2/3/19)
########################################
#msg   = "Do you want channel check plots output?" 
#subplot_check = easygui.buttonbox(msg,choices = ["Yes","No"])

# Ask people if they want to delete rawLFPs or not, that 
#way we offer the option to run analyses in many different ways. 
#(ie. First half V back half)
########################################
# Raw LFP is now deleted automatically (Abu 2/3/19)
########################################
#msg   = "Do you want to delete the Raw LFP data?"
#rawLFPdelete = easygui.buttonbox(msg,choices = ["Yes","No"])
#if rawLFPdelete == "Yes":
#Delete data
hf5.remove_node('/raw_LFP', recursive = True)
hf5.flush()

# ================================================
# Make plots to visually check quality of channels 
# ================================================

# Code copied from LFP_Spectrogram_Stone.py
# Might need cleanup

# =============================================================================
# #Establish User inputs for Variables
# =============================================================================

#Ask if file needs to be split, if yes, split it
split_response = easygui.indexbox(
        msg='Do you need to split these trials? e.g. if you have uneven numbers of trials', 
        title='Split trials', choices=('Yes', 'No'), 
        image=None, default_choice='No', cancel_choice='No')

if trial_check == "Taste":
    total_trials = hf5.root.Parsed_LFP.dig_in_1_LFPs[:].shape[1]
elif trial_check == "Affective":
    total_trials = 1

dig_in_channels = hf5.list_nodes('/digital_in')
dig_in_LFP_nodes = hf5.list_nodes('/Parsed_LFP')

if split_response == 0:
    add_node_number = easygui.integerbox(msg='You have ' +\
            str(len(dig_in_nodes)) + \
            ' digital input channels, how many will you add?', \
            default='4',lowerbound=0,upperbound=10)
    trial_split = easygui.multenterbox(
            msg = "Put in the number of trials to parse from each of "\
                    "the LFP arrays (only integers)", 
                    fields = [node._v_name for node in dig_in_LFP_nodes], 
                    values = ['15' for node in dig_in_LFP_nodes])

    #Convert all values to integers
    trial_split = list(map(int,trial_split))
    
        # Grab array information for each digital input channel, 
        # split into first and last sections, 
        # place in corresponding digitial input group array
    for node in range(len(dig_in_nodes)):
        exec("full_array = hf5.root.Parsed_LFP.dig_in_%i_LFPs[:] " % node)
        hf5.remove_node('/Parsed_LFP/dig_in_%s_LFPs' % str(node), recursive = True)
        hf5.create_array('/Parsed_LFP', 'dig_in_%s_LFPs' % str(node), \
                        np.array(full_array[:,0:trial_split[node],:]))

    total_sessions = int(total_trials/int(trial_split[0]))
 
    #Reset nodes
    dig_in_LFP_nodes = hf5.list_nodes('/Parsed_LFP')

    #Create dictionary of all parsed LFP arrays
    LFP_data = [np.array(dig_in_LFP_nodes[node][:]) \
            for node in range(len(dig_in_LFP_nodes))]

else:    
    total_sessions = 1
    trial_split = list(map(int,[total_trials for node in dig_in_LFP_nodes]))
    #Create dictionary of all parsed LFP arrays
    
    LFP_data = [np.array(dig_in_LFP_nodes[node][:]) \
            for node in range(len(dig_in_LFP_nodes))]
    
#Establish timing parameters
if trial_check == "Affective":
    analysis_params = easygui.multenterbox(
            msg = 'Input analysis paramters:', 
            fields = ['Taste array start time (ms)', 
                    'Taste array end time (ms)', 
                    'Sampling Rate (samples per second)', 
                    'Signal Window (ms)', 
                    'Window Overlap (ms; default 90%)'], 
            values = ['0','1200000','1000','1000','900'])
        
    taste_params = easygui.buttonbox('Select condition:',\
                    choices = ["Experimental (e.g. LiCl)","Control (e.g. Saline)"])    
    if taste_params == 'Experimental (e.g. LiCl)':
        taste_params = 'Experimental'
    else:
        taste_params = 'Control'
    #create timing variables
    pre_stim = 0

else:    
    analysis_params = easygui.multenterbox(
            msg = 'Input analysis paramters:', 
            fields = ['Pre-stimulus signal duration (ms; from set-up)',
                    'Post-stimulus signal duration (ms; from set-up)',
                    'Pre-Taste array start time (ms)', 
                    'Taste array end time (ms)'], 
            values = ['2000','5000','0','2500'])
    
    #create timing variables
    pre_stim = int(durations[0])

    # Ask if this analysis is an average of normalization 
    taste_params = easygui.multenterbox(
            msg = 'Input taste identities:', 
            fields = ['Taste 1 (dig_in_1)', 
                    'Taste 2 (dig_in_2)',
                    'Taste 3 (dig_in_3)',
                    'Taste 4 (dig_in_4)'],
            values = ['NaCl','Sucrose','Citric Acid','QHCl'])


# =============================================================================
# #Channel Check
# =============================================================================
# Make directory to store the LFP trace plots. Delete and remake the directory if it exists
try:
        os.system('rm -r '+'./LFP_channel_check')
except:
        pass
os.mkdir('./LFP_channel_check')

#Check to make sure LFPs are "normal" and allow user to remove any that are not
########################################
# Channel check plots are now made automatically (Abu 2/3/19)
########################################
#if subplot_check is "Yes":
for taste in range(len(LFP_data)):

        #Set data
        if trial_check is 'Taste':
            channel_data = np.mean(LFP_data[taste],axis=1).T
            t=np.array(list(range(0,np.size(channel_data,axis=0))))
        else:
            channel_data = np.array(LFP_data)
            t = (np.arange(channel_data.shape[-1]))[np.newaxis,:]
        
        mean_val = np.mean(channel_data.flatten())
        std_val = np.std(channel_data.flatten())
        #Create figure
        fig,axes = plt.subplots(nrows=np.size(channel_data,axis=1), 
                ncols=1,sharex=True, sharey=True,figsize=(12, 8), squeeze=False)
        fig.text(0.5, 0.05, 'Milliseconds', ha='center',fontsize=15)
        axes_list = [item for sublist in axes for item in sublist]
        
        for ax, chan in zip(axes.flatten(),range(np.size(channel_data,axis=1))):
        
                ax = axes_list.pop(0)
                ax.set_yticks([])
                ax.plot(np.squeeze(t), np.squeeze(channel_data[:,chan]))
                ax.set_ylim([mean_val - 3*std_val, mean_val + 3*std_val])
                h = ax.set_ylabel('Channel %s' %(chan))
                h.set_rotation(0)
                ax.vlines(x=pre_stim, ymin=np.min(channel_data[:,chan]),
                        ymax=np.max(channel_data[:,chan]), linewidth=4, color='r')
                
        fig.subplots_adjust(hspace=0,wspace = -0.15)
        fig.suptitle('Dig in {} - '.format(taste) + \
                '%s - Channel Check: %s' %(taste_params[taste], 
                hdf5_name[0:4])+'\n' + 'Raw LFP Traces; Date: %s' \
                                %(re.findall(r'_(\d{6})', 
                hdf5_name)[0]),size=16,fontweight='bold')
        fig.savefig('./LFP_channel_check/' + hdf5_name[0:4] + \
                '_dig_in{}'.format(taste) + \
                '_ %s_%s' %(re.findall(r'_(\d{6})', hdf5_name)[0],
                    taste_params[taste]) + '_channelcheck.png')   

# ==============================
# Close Out 
# ==============================
print("If you want to compress the file to release disk space, " + \
        "run 'blech_hdf5_repack.py' upon completion.")
hf5.flush()
hf5.close()

