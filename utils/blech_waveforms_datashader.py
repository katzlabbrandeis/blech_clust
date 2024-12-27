# Import stuff
import datashader as ds
import datashader.transfer_functions as tf
from functools import partial
from datashader.utils import export_image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imageio import imread
import shutil
import os

# A function that accepts a numpy array of waveforms and creates a datashader image from them
def waveforms_datashader(waveforms, x_values,
        downsample = True, threshold = None, dir_name = "datashader_temp",
                         ax = None):

    # Make a pandas dataframe with two columns, x and y,
    # holding all the data. The individual waveforms are separated by a row of NaNs

        # First downsample the waveforms 10 times
        # (to remove the effects of 10 times upsampling during de-jittering)
        if downsample:
            waveforms = waveforms[:, ::10]

        # Then make a new array of waveforms -
        # the last element of each waveform is a NaN
        new_waveforms = np.zeros((waveforms.shape[0], waveforms.shape[1] + 1))
        new_waveforms[:, -1] = np.nan
        new_waveforms[:, :-1] = waveforms

        # Now make an array of x's - the last element is a NaN
        x = np.zeros(x_values.shape[0] + 1)
        x[-1] = np.nan
        x[:-1] = x_values

        # Now make the dataframe
        df = pd.DataFrame({'x': np.tile(x, new_waveforms.shape[0]),
            'y': new_waveforms.flatten()})

        # Datashader function for exporting the temporary image with the waveforms
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        export = partial(export_image, background = "white", export_path=dir_name)

        # Produce a datashader canvas
        canvas = ds.Canvas(x_range = (np.min(x_values), np.max(x_values)),
                y_range = (df['y'].min() - 10, df['y'].max() + 10),
                plot_height=1200, plot_width=1600)
        # Aggregate the data
        agg = canvas.line(df, 'x', 'y', ds.count())

        # Transfer the aggregated data to image using log
        # transform and export the temporary image file
        export(tf.shade(agg, how='eq_hist'),'tempfile')

        # Read in the temporary image file
        img = imread(dir_name + "/tempfile.png")

        # Figure sizes chosen so that the resolution is 100 dpi
        if ax is None:
            fig,ax = plt.subplots(1, 1, figsize = (8,6), dpi = 200)
        else:
            fig = ax.get_figure()
        # Start plotting
        ax.imshow(img)
        # Set ticks/labels - 10 on each axis
        ax.set_xticks(np.linspace(0, 1600, 10))
        ax.set_xticklabels(\
                np.floor(np.linspace(np.min(x_values), np.max(x_values), 10)))
        ax.set_yticks(np.linspace(0, 1200, 10))
        ax.set_yticklabels(\
                np.floor(np.linspace(df['y'].max() + 10, df['y'].min() - 10, 10)))

        # Mark threshold for spike selection
        # This shit too confusing...just do it the hard way

        if threshold is not None:
            def y_transform(val):
                fin_line = np.linspace(0, 1200, 1000)
                orig_line = np.linspace(df['y'].max() + 10, df['y'].min() - 10, 1000)
                ind = np.argmin(np.abs(orig_line - val))
                return fin_line[ind]

            ax.axhline(y_transform(threshold), color ='red',
                    linewidth = 1, linestyle = '--', alpha = 0.5)
            ax.axhline(y_transform(-threshold), color ='red',
                    linewidth = 1, linestyle = '--', alpha = 0.5)

        # Delete the dataframe
        del df, waveforms, new_waveforms

        # Also remove the directory with the temporary image files
        shutil.rmtree(dir_name, ignore_errors = True)

        # Return and figure and axis for adding axis labels,
        # title and saving the file
        return fig, ax
