import numpy as np
import tables
import easygui
import sys
import os

from bokeh.plotting import Figure
from bokeh.models import ColumnDataSource
from bokeh.layouts import row, widgetbox
from bokeh.models.widgets import Slider, TextInput
from bokeh.io import curdoc
from bokeh.models.glyphs import MultiLine

# Ask for the directory where the hdf5 file sits, and change to that directory
dir_name = easygui.diropenbox()
os.chdir(dir_name)

# Look for the hdf5 file in the directory
file_list = os.listdir('./')
hdf5_name = ''
for files in file_list:
	if files[-2:] == 'h5':
		hdf5_name = files

# Open the hdf5 file
hf5 = tables.open_file(hdf5_name, 'r+')

# Get electrode number from user
electrode_num = easygui.multenterbox(msg = 'Which electrode do you want to choose? Hit cancel to exit', fields = ['Electrode #'])
electrode_num = int(electrode_num[0])

# Get the number of clusters in the chosen solution
num_clusters = easygui.multenterbox(msg = 'Which solution do you want to choose for electrode %i?' % electrode_num, fields = ['Number of clusters in the solution'])
num_clusters = int(num_clusters[0])

# Load data from the chosen electrode and solution
spike_waveforms = np.load('./spike_waveforms/electrode%i/spike_waveforms.npy' % electrode_num)
predictions = np.load('./clustering_results/electrode%i/clusters%i/predictions.npy' % (electrode_num, num_clusters))

# Get cluster choices from the chosen solution
clusters = easygui.multchoicebox(msg = 'Which clusters do you want to choose?', choices = tuple([str(i) for i in range(int(np.max(predictions) + 1))]))
for i in range(len(clusters)):
	clusters[i] = int(clusters[i])

# Get the current plot data
plot_data = []
for cluster in clusters:
	plot_data.append(np.where(predictions == cluster)[0])
plot_data = spike_waveforms[plot_data]

# Set up data
N = 0
x = np.arange(len(plot_data[N])/10)
y = spike_waveforms[0, ::10]
source = ColumnDataSource(data=dict(xs=[x for i in range(50)], ys = [plot_data[N + i, ::10] for i in range(50)]))

# Set up plot
plot = Figure(plot_height=400, plot_width=400, title="Unit waveforms",
              tools="crosshair,pan,reset,save,wheel_zoom",
              x_range=[0, 45], y_range=[-200, 200])

plot.multi_line('xs', 'ys', source=source, line_width=1, line_alpha=1.0)

# Set up widgets
# text = TextInput(title="title", value='my sine wave')
offset = Slider(title="offset", value=0, start=0, end=50000, step= 100) # put the end of the slider at a large enough value so that almost all cluster sizes will fit in
electrode = TextInput(title = 'Electrode Number', value = '0')
clusters = TextInput(title = 'Number of clusters', value = '2')
cluster_num = TextInput(title = 'Cluster Number', value = '0')
#amplitude = Slider(title="amplitude", value=1.0, start=-5.0, end=5.0)
#phase = Slider(title="phase", value=0.0, start=0.0, end=2*np.pi)
#freq = Slider(title="frequency", value=1.0, start=0.1, end=5.1)

def update_data(attrname, old, new):

    os.chdir(dir_name)

    # Get text values
    electrode_num = int(electrode.value)
    num_clusters = int(clusters.value)
    cluster = int(cluster_num.value)

    # Now get new data
    spike_waveforms = np.load('./spike_waveforms/electrode%i/spike_waveforms.npy' % electrode_num)
    predictions = np.load('./clustering_results/electrode%i/clusters%i/predictions.npy' % (electrode_num, num_clusters))
    plot_data = spike_waveforms[np.where(predictions == cluster)[0]]

    # Get the current slider values
    b = offset.value

    # Generate the new curve
    x = np.arange(len(plot_data[b])/10)
    #y = a*np.sin(k*x + w) + b

    source.data = dict(xs=[x for i in range(50)], ys= [plot_data[b + i, ::10] for i in range(50)])

for w in [offset]:
    w.on_change('value', update_data)

# Set up layouts and add to document
inputs = widgetbox(children=[offset, electrode, clusters, cluster_num])

curdoc().add_root(row(children=[inputs, plot], width=800))
