import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import numpy as np
import tables
from glob import glob

data_dir = '/media/storage/gc_only/AS18/AS18_4Tastes_200228_151511_old_car/'
hf5_path = glob(data_dir + '*.h5')[0] 	
hf5 = tables.open_file(hf5_path, 'r')
raw_electrodes_list = hf5.list_nodes('/raw')
dat_len = raw_electrodes_list[0].shape[0]
dat_sd = [np.std(node[::1000]) for node in raw_electrodes_list]
dat_mean_sd = np.mean(dat_sd)

def return_dat_cut(start, end):
	return [node[start:end] for node in raw_electrodes_list]

window_len = dat_len//10000

# Always start by initializing Qt (only once per application)
app = QtWidgets.QApplication([])

# Define a top-level widget to hold everything
w = QtWidgets.QWidget()

# Create some widgets to be placed inside
h_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
v_slider = QtWidgets.QSlider(QtCore.Qt.Vertical)
zoom_in_btn = QtWidgets.QPushButton('Zoom +')
zoom_out_btn = QtWidgets.QPushButton('Zoom -')
plot1 = pg.PlotWidget()
plot2 = pg.PlotWidget()
plot1.setYRange(-5, 30)

h_slider_position_label = QtWidgets.QLabel()
def update_h_slider_label(value):
    h_slider_position_label.setText(f'Slider position: {value}, {value+window_len}')
h_slider.valueChanged.connect(update_h_slider_label)

checkboxes = [QtWidgets.QCheckBox('Variable %d' % i) for i in range(len(raw_electrodes_list))]
for checkbox in checkboxes:
    checkbox.setChecked(True)

# Create a grid layout to manage the widgets size and position
layout = QtWidgets.QGridLayout()
w.setLayout(layout)
checkbox_layout = QtWidgets.QVBoxLayout()

layout.addWidget(v_slider, 1, 1, 3, 1)  # vertical slider goes next to the checkboxes, spanning 3 rows
layout.addWidget(zoom_in_btn, 0, 2)  # zoom in button goes above the plot
layout.addWidget(zoom_out_btn, 0, 3)  # zoom out button goes above the plot
layout.addWidget(plot1, 1, 2, 1, 2)  # first plot goes next to the vertical slider
layout.addWidget(plot2, 2, 2, 1, 2)  # second plot goes below the first one
layout.addWidget(h_slider, 3, 2, 1, 2)  # horizontal slider goes just below plots
for checkbox in checkboxes:
    checkbox_layout.addWidget(checkbox)  # add checkboxes to the vertical layout
layout.addLayout(checkbox_layout, 1, 0, 3, 1) 
layout.addWidget(h_slider_position_label, 0, 0)

# Set slider properties
h_slider.setMinimum(0)
h_slider.setMaximum(dat_len-window_len)
h_slider.setValue(0)

v_slider.setMinimum(0)
v_slider.setMaximum(10)
v_slider.setValue(5)

# Add an integer to each row in data
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
data = return_dat_cut(0, window_len)
plot_dat = [x+(i*dat_mean_sd*v_slider.value()) for i, x in enumerate(data)]
plot_mean = np.mean(plot_dat, axis=0)
plot_car = [x-plot_mean for x in plot_dat]
curve_list = [plot1.plot(plot_dat[i], pen = colors[i % len(colors)]) for i in range(len(plot_dat))]
car_curve_list = [plot2.plot(plot_car[i], pen = colors[i % len(colors)]) for i in range(len(plot_dat))]

# Update plot with scroll
#plotItem = plot.getPlotItem()
def updatePlotRange():
	i = h_slider.value() 
	data = return_dat_cut(i, i+window_len)
	dat_mean = np.mean(data, axis=0)
	dat_car = [x-dat_mean for x in data]
	check_bool = [x.isChecked() for x in checkboxes]
	plot_dat_raw = [data[i] for i, x in enumerate(check_bool) if x]
	plot_car_raw = [dat_car[i] for i, x in enumerate(check_bool) if x]
	plot_dat = \
			[x+(ind*dat_mean_sd*v_slider.value()) for ind, x in enumerate(plot_dat_raw)]
	plot_car = \
			[x+(ind*dat_mean_sd*v_slider.value()) for ind, x in enumerate(plot_car_raw)]
	plot1.clear()
	for ind,this_dat in enumerate(plot_dat):
		plot1.plot(this_dat, pen = colors[ind % len(colors)])
	plot2.clear()
	for ind, this_dat in enumerate(plot_car):
		plot2.plot(this_dat, pen = colors[ind % len(colors)])
	#for this_curve, this_dat, this_box in zip(curve_list, plot_dat, checkboxes):
	#	if this_box.isChecked():
	#		this_curve.setData(this_dat)
	plot1.setYRange(np.min(plot_dat), np.max(plot_dat))
	plot2.setYRange(np.min(plot_car), np.max(plot_car))

def zoomIn():
	temp_len = globals()['window_len'] * 1.1
	globals()['window_len'] = int(temp_len)  # Increase the zoom level by 10%
	updatePlotRange()  # Update the plot to reflect the new zoom level

def zoomOut():
	temp_len = globals()['window_len'] / 1.1
	globals()['window_len'] = int(temp_len)  # Increase the zoom level by 10%
	updatePlotRange()  # Update the plot to reflect the new zoom level

h_slider.valueChanged.connect(updatePlotRange)
v_slider.valueChanged.connect(updatePlotRange)
zoom_in_btn.clicked.connect(zoomIn)
zoom_out_btn.clicked.connect(zoomOut)
for checkbox in checkboxes:
    checkbox.stateChanged.connect(updatePlotRange)  # Update the plot when a checkbox state changes

# Display the widget as a new window
w.show()

# Start the Qt event loop
app.exec_()

