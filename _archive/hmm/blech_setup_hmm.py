# Import stuff!
import numpy as np
import tables
import easygui
import sys
import os

# Ask for the directory where the hdf5 file sits
dir_name = easygui.diropenbox()

# Store the directory path to blech.dir
f = open('blech.dir', 'w')
print(dir_name, file=f)
f.close()

# Change to the directory
os.chdir(dir_name)

# Look for the hdf5 file in the directory
file_list = os.listdir('./')
hdf5_name = ''
for files in file_list:
	if files[-2:] == 'h5':
		hdf5_name = files

# Open the hdf5 file
hf5 = tables.open_file(hdf5_name, 'r')

# Ask the user for the HMM parameters
hmm_params = easygui.multenterbox(msg = 'Fill in the parameters for running a HMM (Poisson or Multinomial emissions) on your data', fields = ['Minimum number of states', 'Maximum number of states', 'Maximum number of iterations', 'Convergence criterion (usually 1e-9)', 'Number of random restarts for HMM (50-60 is more than enough)', 'Transition probability inertia (between 0 and 1)', 'Emission Distribution intertia (between 0 and 1)'])

# Ask the user for the taste to run the HMM on
tastes = hf5.list_nodes('/spike_trains')
hmm_taste = easygui.multchoicebox(msg = 'Which taste do you want to run the HMM on?', choices = ([str(taste).split('/')[-1] for taste in tastes]))
taste_num = 0
for i in range(len(tastes)):
	if str(tastes[i]).split('/')[-1] in hmm_taste:
		taste_num = i

# Ask the user to choose the units to run the HMM on
all_units = hf5.list_nodes('/sorted_units')
all_units = np.array([int(str(unit).split('/')[-1][4:7]) for unit in all_units])
single_units = np.array([i for i in range(len(all_units)) if hf5.root.unit_descriptor[i]["single_unit"] == 1])
chosen_units = []
units_choose = easygui.multchoicebox(msg = 'How do you want to choose units for the HMM?', choices = ('All units', 'Single units', 'Random choice over all units', 'Random choice over single units', 'Custom choice'))
if units_choose[0] == 'All units':
	chosen_units = all_units
elif units_choose[0] == 'Single units':
	chosen_units = single_units
elif units_choose[0] == 'Random choice over all units':
	num_units = easygui.multenterbox(msg = 'How many units do you want to choose?', fields = ['# of units (Total = %i)' % len(all_units)])
	num_units = int(num_units[0])
	chosen_units = np.random.choice(all_units, size = num_units, replace = True)
elif units_choose[0] == 'Random choice over single units':
	num_units = easygui.multenterbox(msg = 'How many single units do you want to choose?', fields = ['# of single units (Total = %i)' % len(single_units)])
	num_units = int(num_units[0])
	chosen_units = np.random.choice(single_units, size = num_units, replace = True)
else:
	chosen_units = easygui.multchoicebox(msg = 'Which units do you want to choose?', choices = ([i for i in all_units]))
	for i in range(len(chosen_units)):
		chosen_units[i] = int(chosen_units[i])

# Convert the chosen units into numpy array, and subtract 1 - this gives us the unit number to use in the spike arrays (because they run from 0 to n-1)
chosen_units = np.array(chosen_units)

# Create the folder for storing the plots coming in from HMM analysis of the data - pass if it exists
try:
	os.mkdir("HMM_plots")
	# Make folders for storing plots from each of the tastes within HMM_plots
	for i in range(len(tastes)):
		os.mkdir('HMM_plots/dig_in_%i' % i)
except:
	pass


# Ask the user for the parameters to process spike trains
spike_params = easygui.multenterbox(msg = 'Fill in the parameters for processing your spike trains', fields = ['Pre-stimulus time used for making spike trains (ms)', 'Bin size for HMM (ms) - usually 10', 'Pre-stimulus time for HMM (ms)', 'Post-stimulus time for HMM (ms)'])

# Ask the user to choose the type of HMM they want to fit - generic or feedforward
hmm_type = easygui.multchoicebox(msg = 'Which type of HMM do you want to fit?', choices = ('generic', 'feedforward'))
hmm_type = hmm_type[0]

# Print the paramaters to blech.hmm_params
f = open('blech.hmm_params', 'w')
for params in hmm_params:
	print(params, file=f)
print(taste_num, file=f)
for params in spike_params:
	print(params, file=f)
print(hmm_type, file=f)
f.close()

# Print the chosen units to blech.hmm_units
f = open('blech.hmm_units', 'w')
for unit in chosen_units:
	print(unit, file=f)
f.close()

# Grab Brandeis unet username
username = easygui.multenterbox(msg = 'Enter your Brandeis unet id', fields = ['unet username'])

# Dump shell file for running parallel job on the user's blech_clust folder on the desktop
os.chdir('/home/%s/Desktop/blech_clust' % username[0])
f = open('blech_multinomial_hmm.sh', 'w')
g = open('blech_poisson_hmm.sh', 'w')
print("module load PYTHON/ANACONDA-2.5.0", file=f)
print("module load PYTHON/ANACONDA-2.5.0", file=g)
print("cd /home/%s/Desktop/blech_clust" % username[0], file=f)
print("cd /home/%s/Desktop/blech_clust" % username[0], file=g)
print("python blech_multinomial_hmm.py", file=f)
print("python blech_poisson_hmm.py", file=g)
f.close()
g.close()

hf5.close()
