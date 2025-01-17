"""
This module provides functions for visualizing neural data, including raster plots, heatmaps, and firing rate overviews.

- `raster(ax, spike_array, marker='o', color=None)`: Creates a raster plot of spike data on the given axis. If no axis is provided, a new figure and axis are created.
- `imshow(x, cmap='viridis')`: Displays a heatmap of the input data using the specified colormap, with settings for better visualization.
- `gen_square_subplots(num, figsize=None, sharex=False, sharey=False)`: Generates a grid of subplots arranged in a square layout, returning the figure and axes.
- `firing_overview(data, t_vec=None, y_values_vec=None, interpolation='nearest', cmap='jet', cmap_lims='individual', subplot_labels=None, zscore_bool=False, figsize=None, backend='pcolormesh')`: Generates heatmaps of firing rates from a 3D numpy array, with options for z-scoring, colormap limits, and subplot labels. Returns the figure and axes.
"""
import numpy as np
import pylab as plt
from scipy.stats import zscore

def raster(ax, spike_array, marker = 'o', color = None):
    inds = np.where(spike_array)         
    if ax is None:                       
        fig, ax = plt.subplots()         
    ax.scatter(inds[1],inds[0], marker = marker, color = color)
    return ax

def imshow(x, cmap = 'viridis'):
    """
    Decorator function for more viewable firing rate heatmaps
    """
    plt.imshow(x,
            interpolation='nearest',aspect='auto', 
            origin = 'lower', cmap = cmap)

def gen_square_subplots(num, figsize = None, sharex = False, sharey = False):
    """
    number of subplots to generate
    """
    square_len = int(np.ceil(np.sqrt(num)))
    row_num = int(np.ceil(num / square_len))
    fig, ax = plt.subplots(row_num, square_len, 
            sharex = sharex, sharey = sharey, figsize = figsize)
    return fig, ax

def firing_overview(data, t_vec = None, y_values_vec = None,
                    interpolation = 'nearest',
                    cmap = 'jet',
                    #min_val = None, max_val=None, 
                    cmap_lims = 'individual',
                    subplot_labels = None,
                    zscore_bool = False,
                    figsize = None,
                    backend = 'pcolormesh'):
    """
    Takes 3D numpy array as input and rolls over first dimension
    to generate images over last 2 dimensions
    E.g. (neuron x trial x time) will generate heatmaps of firing
        for every neuron

    Inputs:
        data: 3D numpy array
        t_vec: time vector
        y_values_vec: y values vector
        cmap: colormap
        min_val: minimum value for colormap
        max_val: maximum value for colormap
        cmap_lims: 'individual' or 'shared'
        subplot_labels: labels for subplots
        zscore_bool: zscore data
        figsize: size of figure
        backend: 'pcolormesh' or 'imshow

    Outputs:
        fig: figure handle
        ax: axis handle
    """

    if zscore_bool:
        data = np.array([zscore(dat,axis=None) for dat in data])

    if cmap_lims == 'shared':
        min_val, max_val = np.repeat(np.min(data,axis=None),data.shape[0]),\
                                np.repeat(np.max(data,axis=None),data.shape[0])
    else:
        min_val,max_val = np.min(data,axis=tuple(list(np.arange(data.ndim)[1:]))),\
                            np.max(data,axis=tuple(list(np.arange(data.ndim)[1:])))
    if t_vec is None:
        t_vec = np.arange(data.shape[-1])
    if y_values_vec is None:
        y_values_vec = np.arange(data.shape[1])

    if data.shape[-1] != len(t_vec):
        raise Exception('Time dimension in data needs to be'\
            'equal to length of time_vec')

    num_nrns = data.shape[0]

    # Plot firing rates
    square_len = int(np.ceil(np.sqrt(num_nrns)))
    row_count = int(np.ceil(num_nrns/square_len))
    if figsize is None:
        fig, ax = plt.subplots(row_count, square_len, sharex='all',sharey='all')
    else:
        fig, ax = plt.subplots(row_count, square_len, 
                               sharex='all',sharey='all',figsize=figsize)
    # Account for case where row and cols are both 1
    if not isinstance(ax,np.array([]).__class__):
        ax = np.array(ax)[np.newaxis, np.newaxis]
    if ax.ndim < 2:
        ax = ax[:,np.newaxis]
    
    nd_idx_objs = list(np.ndindex(ax.shape))
    
    x,y = np.meshgrid(t_vec, y_values_vec)
    if subplot_labels is None:
        subplot_labels = np.zeros(num_nrns)
    if y_values_vec is None:
        y_values_vec = np.arange(data.shape[1])
    for this_ind,nrn in zip(nd_idx_objs,range(num_nrns)):
        plt.sca(ax[this_ind[0],this_ind[1]])
        plt.gca().set_title('{}:{}'.format(int(subplot_labels[nrn]),nrn))
        if backend == 'imshow':
            plt.gca().imshow(data[nrn], 
                interpolation=interpolation, aspect='auto', 
                origin='lower', cmap=cmap)
        elif backend == 'pcolormesh':
            plt.gca().pcolormesh(x, y,
                    data[nrn],cmap=cmap,
                    shading = 'nearest',
                    vmin = min_val[nrn], vmax = max_val[nrn])

    return fig,ax

