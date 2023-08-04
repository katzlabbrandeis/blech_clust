import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import numpy as np

# Always start by initializing Qt (only once per application)
app = QtWidgets.QApplication([])

# Define a top-level widget to hold everything
w = QtWidgets.QWidget()

# Create some widgets to be placed inside
h_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
v_slider = QtWidgets.QSlider(QtCore.Qt.Vertical)
plot = pg.PlotWidget()
plot.setYRange(-5, 30)

# Create a grid layout to manage the widgets size and position
layout = QtWidgets.QGridLayout()
w.setLayout(layout)

# Add widgets to the layout in their proper positions
layout.addWidget(plot, 2,1)  # plot goes in upper-middle, spanning 2 columns
layout.addWidget(h_slider, 3,1)
layout.addWidget(v_slider, 2, 0, 2, 1)

# Generate some random data
window_len = 1000
data = np.random.normal(size=(5,5000))
# Add an integer to each row in data
curve_list = [plot.plot(data[i,:]) for i in range(data.shape[0])]
#for this_dat in data:
#	plot.plot(this_dat[:window_len])

# Set slider properties
h_slider.setMinimum(0)
h_slider.setMaximum(data.shape[1]-window_len)
h_slider.setValue(0)

v_slider.setMinimum(0)
v_slider.setMaximum(10)
v_slider.setValue(5)

plot_dat = data + np.arange(len(data))[:,np.newaxis]*v_slider.value()
def updatePlotData(value):
	globals()['plot_dat'] = data + np.arange(len(data))[:,np.newaxis]*value
	plot.setYRange(np.min(plot_dat), np.max(plot_dat))

# Update plot with scroll
#plotItem = plot.getPlotItem()
def updatePlot(value):
	i = value
	#plot.clear()
	#for this_dat in data:
	#	plot.plot(this_dat[i:i+window_len])
	for this_curve, this_dat in zip(curve_list, plot_dat):
		this_curve.setData(this_dat[i:i+window_len])
	## Update xticks
	#x_range = range(i, i + window_len)
	#ticks = [list(zip(x_range, [str(x) for x in x_range]))]
	#plotItem.getAxis('bottom').setTicks(ticks)

h_slider.valueChanged.connect(updatePlot)
v_slider.valueChanged.connect(updatePlotData)

# Display the widget as a new window
w.show()

# Start the Qt event loop
app.exec_()

