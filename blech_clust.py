# Necessary python modules
import easygui
import os
import tables
import sys
import numpy as np
import multiprocessing
import json
import glob
import pandas as pd
import shutil

# Necessary blech_clust modules
from utils import read_file
from utils.blech_utils import entry_checker, imp_metadata

# Get blech_clust path
blech_clust_path = ('/').join(os.path.abspath(__file__).split('/')[0:-1])

#First ask user if there are multiple files that need to be stitched together
m_loop = 1
while m_loop == 1:
	m_val = input("\nINPUT REQUESTED: Are there multiple recordings that need to be stitched together (y/n)? ")
	if m_val=='y' or m_val=='n':
		m_loop = 0
	else:
		print("Error, incorrect entry. Please try again.")
if m_val == 'y':
	m_loop = 1
	while m_loop == 1:
		m_count_input = input("\nINPUT REQUESTED: How many files need to be combined? ")
		try:
			m_count = int(m_count_input)
			m_loop = 0
		except:
			print("Error, non-integer entry. Please try again.")
else:
	m_count = 1

############################################################


if m_count > 1:
	print("\nWARNING: Combining multiple files requires a new storage directory.")
	print("Please create a storage folder, and select it in the next prompt.")
	save_dir_name = easygui.diropenbox(msg = 'Please select data directory') + '/'

#Store lists of information for each file to be concatenated
dir_name_list = []	
info_dict_list = []
file_list_list = []
file_type_list = []
for m_i in range(m_count):
	if m_count > 1:
		print('\nBeginning import of data from segment #' + str(m_i+1))
		metadata_handler = imp_metadata([])
		dir_name = metadata_handler.dir_name
	else:
		metadata_handler = imp_metadata(sys.argv)
		dir_name = metadata_handler.dir_name
		save_dir_name = dir_name + '/'
	
	dir_name_list.append(dir_name)
	
	print(f'Processing : {dir_name}')
	os.chdir(dir_name)
	
	info_file_path = metadata_handler.info_file_path
	
	#Check if save directory has an info file - if not copy first datasets info file
	info_file_list = glob.glob(os.path.join(save_dir_name,'**.info'))
	if len(info_file_list) < 1:
		info_file_name = save_dir_name.split('/')[-2] + '.info'
		shutil.copy(info_file_path,save_dir_name + info_file_name)
	
	info_dict = metadata_handler.info_dict
	info_dict_list.append(info_dict)
	file_list = metadata_handler.file_list
	file_list_list.append(file_list)

	# Get the type of data files (.rhd or .dat)
	try:
		file_list.index('auxiliary.dat')
		file_type = ['one file per signal type']
	except:
		file_type = ['one file per channel']
	file_type_list.append(file_type)

# Create hdf5 file, and make groups for raw data, raw emgs, 
# digital outputs and digital inputs, and close

# Use save directory name to create the name of the hdf5 file
# If HDF5 present, use that, otherwise, create new one
os.chdir(save_dir_name)
h5_search = glob.glob('*.h5')
if len(h5_search):
    hdf5_name = h5_search[0] 
    print(f'HDF5 file found...Using file {hdf5_name}')
    hf5 = tables.open_file(hdf5_name, 'r+')
else:
    hdf5_name = str(os.path.dirname(save_dir_name)).split('/')[-1]+'.h5'
    print(f'No HDF5 found...Creating file {hdf5_name}')
    hf5 = tables.open_file(hdf5_name, 'w', title = hdf5_name[-1])

group_list = ['raw','raw_emg','digital_in','digital_out']
for this_group in group_list:
    if '/'+this_group in hf5:
        hf5.remove_node('/', this_group, recursive=True)
    hf5.create_group('/',this_group)
hf5.close()
print('Created nodes in HF5')

# Create directories to store waveforms, spike times, clustering results, and plots
# And a directory for dumping files talking about memory usage in blech_process.py
# Check if dirs are already present, if they are, ask to delete and continue
# or abort
dir_list = ['spike_waveforms',
            'spike_times',
            'clustering_results',
            'Plots',
            'memory_monitor_clustering']
dir_exists = [x for x in dir_list if os.path.exists(x)]
recreate_msg = f'Following dirs are present :' + '\n' + f'{dir_exists}' + \
                '\n' + 'Overwrite dirs? (yes/y/n/no) ::: '

# If dirs exist, check with user
if len(dir_exists) > 0:
    recreate_str, continue_bool = entry_checker(\
            msg = recreate_msg,
            check_func = lambda x: x in ['y','yes','n','no'], 
            fail_response = 'Please enter (yes/y/n/no)')
# Otherwise, create all of them
else:
    continue_bool = True
    recreate_str = 'y'

# Break if user said n/no or gave exit signal
if continue_bool:
    if recreate_str in ['y','yes']:
        for x in dir_list:
            if os.path.exists(x):
                shutil.rmtree(x)
            os.makedirs(x)
else:
    quit()

print('Created dirs in data folder')
for m_i in range(m_count):
	print("Storing dataset # " + str(m_i + 1) + " to HDF5.")
	dir_name = dir_name_list[m_i]
	info_dict = info_dict_list[m_i]
	file_list = file_list_list[m_i]
	file_type = file_type_list[m_i]
	#Get lists of amplifier and digital input files
	if file_type == ['one file per signal type']:
		electrodes_list = ['amplifier.dat']
		dig_in_list = ['digitalin.dat']
	elif file_type == ['one file per channel']:
		electrodes_list = [name for name in file_list if name.startswith('amp-')]
		dig_in_list = [name for name in file_list if name.startswith('board-DI')]
	
	#Use info file for port list calculation
	info_file = np.fromfile(dir_name + '/info.rhd', dtype = np.dtype('float32'))
	sampling_rate = int(info_file[2])
	
	# Read the time.dat file for use in separating out the one file per signal type data
	num_recorded_samples = len(np.fromfile(dir_name + '/' + 'time.dat', dtype = np.dtype('float32')))
	total_recording_time = num_recorded_samples/sampling_rate #In seconds
	
	check_str = f'Amplifier files: {electrodes_list} \nSampling rate: {sampling_rate} Hz'\
	           f'\nDigital input files: {dig_in_list} \n ---------- \n \n'
	print(check_str)
	
	ports = info_dict['ports']
	
	if file_type == ['one file per channel']:
	    print("\tOne file per CHANNEL Detected")
	
	    # Read dig-in data
	    # Pull out the digital input channels used, 
	    # and convert them to integers
	    dig_in = [x.split('-')[-1].split('.')[0] for x in dig_in_list]
	    dig_in = sorted([int(x) for x in dig_in])
	
	elif file_type == ['one file per signal type']:
	
	    print("\tOne file per SIGNAL Detected")
	    dig_in = np.arange(info_dict['dig_ins']['count'])
	
	check_str = f'ports used: {ports} \n sampling rate: {sampling_rate} Hz'\
	            f'\n digital inputs on intan board: {dig_in}'
	
	print(check_str)
	
	all_car_group_vals = []
	for region_name, region_elecs in info_dict['electrode_layout'].items():
	    if not region_name == 'emg':
	        for group in region_elecs:
	            if len(group) > 0:
	                all_car_group_vals.append(group)
	all_electrodes = [electrode for region in all_car_group_vals \
	                        for electrode in region]
	
	emg_info = info_dict['emg']
	emg_port = emg_info['port']
	emg_channels = sorted(emg_info['electrodes'])
	
	
	layout_path = glob.glob(os.path.join(dir_name,"*layout.csv"))[0]
	electrode_layout_frame = pd.read_csv(layout_path) 
	
	
	# Read data files, and append to electrode arrays
	if file_type == ['one file per channel']:
		#read_file.read_files_abu(hdf5_name, dig_in, electrode_layout_frame) 
		read_file.read_digins(save_dir_name+hdf5_name, dir_name, dig_in, dig_in_list)
		read_file.read_electrode_channels(save_dir_name+hdf5_name, dir_name, electrode_layout_frame)
		if len(emg_channels) > 0:
		    read_file.read_emg_channels(save_dir_name+hdf5_name, dir_name, electrode_layout_frame)
	elif file_type == ['one file per signal type']:
		read_file.read_digins_single_file(save_dir_name+hdf5_name, dir_name, dig_in, dig_in_list)
		#This next line takes care of both electrodes and emgs
		read_file.read_electrode_emg_channels_single_file(save_dir_name+hdf5_name, dir_name, electrode_layout_frame, electrodes_list, num_recorded_samples, emg_channels)

# Write out template params file to directory if not present
#home_dir = os.getenv('HOME')
#blech_clust_path = os.path.join(home_dir,'Desktop','blech_clust')
print(blech_clust_path)
params_template_path = os.path.join(
        blech_clust_path,
        'params/sorting_params_template.json')
params_template = json.load(open(params_template_path,'r'))
# Info on taste digins and laser should be in exp_info file
all_params_dict = params_template.copy() 
all_params_dict['sampling_rate'] = sampling_rate

os.chdir(save_dir_name)
params_out_path = hdf5_name.split('.')[0] + '.params'
if not os.path.exists(params_out_path):
    print('No params file found...Creating new params file')
    with open(params_out_path, 'w') as params_file:
        json.dump(all_params_dict, params_file, indent = 4)
else:
    print("Params file already present...not writing a new one")

# Dump shell file(s) for running GNU parallel job on the user's blech_clust folder on the desktop
# First get number of CPUs - parallel be asked to run num_cpu-1 threads in parallel
num_cpu = multiprocessing.cpu_count()

electrode_bool = electrode_layout_frame.loc[
        electrode_layout_frame.electrode_ind.isin(all_electrodes)]
not_none_bool = electrode_bool.loc[~electrode_bool.CAR_group.isin(["none","None",'na'])]
not_emg_bool = not_none_bool.loc[
        ~not_none_bool.CAR_group.str.contains('emg')
        ]
bash_electrode_list = not_emg_bool.electrode_ind.values
job_count = np.min((len(bash_electrode_list), int(num_cpu-2)))
# todo: Account for electrodes labelled none when writing parallel command
runner_path = os.path.join(blech_clust_path,'blech_clust_jetstream_parallel1.sh') 
f = open(os.path.join(blech_clust_path,'blech_clust_jetstream_parallel.sh'), 'w')
print(f"parallel -k -j {job_count} --noswap --load 100% --progress " +\
        "--memfree 4G --retry-failed "+\
        f"--joblog {save_dir_name}/results.log "+\
        f"bash {runner_path} "+\
        #f"::: {{{','.join([str(x) for x in bash_electrode_list])}}}", 
        f"::: {' '.join([str(x) for x in bash_electrode_list])}", 
        file = f)
f.close()

# Then produce the file that runs blech_process.py
f = open(os.path.join(blech_clust_path,'blech_clust_jetstream_parallel1.sh'), 'w')
print("export OMP_NUM_THREADS=1", file = f)
blech_process_path = os.path.join(blech_clust_path,'blech_process.py')
print(f"python {blech_process_path} $1", file=f)
f.close()

# Dump the directory name where blech_process has to cd
f = open(os.path.join(blech_clust_path,'blech.dir'), 'w')
print(save_dir_name, file=f)
f.close()

print('blech_clust.py complete \n')
print('*** Please check params file to make sure all is good ***\n')
