import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import numpy as np

# Always start by initializing Qt (only once per application)
app = QtWidgets.QApplication([])

# Define a top-level widget to hold everything
w = QtWidgets.QWidget()

# Create some widgets to be placed inside
btn = QtWidgets.QPushButton('Push me!')
text = QtWidgets.QLineEdit('enter text')
slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
plot = pg.PlotWidget()

# Create a grid layout to manage the widgets size and position
layout = QtWidgets.QGridLayout()
w.setLayout(layout)

# Add widgets to the layout in their proper positions
layout.addWidget(btn, 0, 0)   # button goes in upper-left
layout.addWidget(text, 1, 0)   # text edit goes in middle-left
layout.addWidget(plot, 2, 0, 1, 2)  # plot goes in upper-middle, spanning 2 columns
layout.addWidget(slider, 3, 0, 1, 2)  # slider goes just below plot, spanning 2 columns

# Generate some random data
window_len = 100
data = np.random.normal(size=(1,5000))
curve = plot.plot(data[0,:window_len])

# Set slider properties
slider.setMinimum(0)
slider.setMaximum(data.shape[1]-window_len)
slider.setValue(0)

# Set Y range
plot.setYRange(-4, 4)


# Update plot with scroll
def updatePlot(value):
    i = value
    curve.setData(data[0,i:i+window_len])

slider.valueChanged.connect(updatePlot)

# Display the widget as a new window
w.show()

# Start the Qt event loop
app.exec_()

