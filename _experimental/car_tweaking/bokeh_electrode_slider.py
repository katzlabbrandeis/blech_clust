import numpy as np
import tables
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, Slider
from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.themes import Theme
from glob import glob
import pylab as plt
import pandas as pd

data_dir = '/media/storage/gc_only/AS18/AS18_4Tastes_200228_151511_old_car/'
hf5_path = glob(data_dir + '*.h5')[0] 	
hf5 = tables.open_file(hf5_path, 'r')
raw_electrodes_list = hf5.list_nodes('/raw')[:3]
dat_len = raw_electrodes_list[0].shape[0]

window_len = dat_len//100
step = window_len//10

def return_dat_cut(start, end):
	dat = [node[start:end] for node in raw_electrodes_list]
	dat_dict = {}
	dat_dict['time'] = [np.arange(start, end) for _ in range(len(dat))]
	dat_dict['data'] = dat
	return dat_dict

#df = pd.DataFrame(
#			data = return_dat_cut(0, window_len),
#			index = np.arange(window_len),
#		)
start_dat = return_dat_cut(0, window_len)
source = ColumnDataSource(data=start_dat)

# Create a new plot with a title and axis labels
p = figure(x_axis_type='datetime', title="Interactive data scroll", x_axis_label='Time', y_axis_label='Value')

# Add a line renderer with legend and line thickness
p.multi_line(xs='time', ys='data', source=source)

# The callback function to update the 'y' values
def update(attr, old, new):
    start = slider.value
    end = start + 10 if start + 10 < len(df) else len(df)
    new_data = df.iloc[start:end]
    source.data = new_data

# Slider to change the start index
slider = Slider(start=0, end=len(df), step=1, value=0)
slider.on_change('value', update)

curdoc().add_root(column(slider, p))
curdoc().theme = Theme(json={
    "attrs": {
        "Plot": { "toolbar_location": None },
        "Grid": { "grid_line_color": None },
        "Axis": {
            "axis_line_color": None,
            "major_label_text_color": None,
            "major_tick_line_color": None,
            "minor_tick_line_color": None
        }
    }
})


