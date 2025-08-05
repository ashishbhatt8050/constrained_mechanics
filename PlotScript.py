#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 15:30:30 2020

@author: ashishbhatt

Copied from https://github.com/jbmouret/matplotlib_for_papers
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator
import brewer2mpl
from cycler import cycler
import os
import pickle
from functools import wraps
from time import time
import math
import numpy as np

# Set up plot parameters
bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)
colors = bmap.mpl_colors

params = {
    'axes.labelsize': 20,
    'axes.prop_cycle': (cycler(color=colors[:3]) + cycler(linestyle=['-','--','-.'])),
    'font.size': 20,
    'legend.fontsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'text.usetex': True,
    'figure.figsize': [14, 6],
    'figure.autolayout': True
}
plt.rcParams.update(params)


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        _ = f(*args, **kw)
        te = time()
        # print('func:%r args:[%r, %r] took: %2.4f sec' % \
        #   (f.__name__, args, kw, te-ts))
        return te-ts
    return wrap

def configure_axis(ax):
    """
    Configure the axis for a plot.

    Parameters:
    ax (matplotlib axis): The axis to configure.
    """
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.tick_params(axis='x', direction='out')
    ax.tick_params(axis='y', length=0)
    for spine in ax.spines.values():
        spine.set_position(('outward', 5))
    ax.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
    ax.set_axisbelow(True)

def plot_data(ax, x_data, y_data, xlims=None, ylabel=None, margins=None):
    """
    Plot data on an axis.

    Parameters:
    ax (matplotlib axis): The axis to plot on.
    x_data (numpy array): The x-coordinates of the data.
    y_data (numpy array): The y-coordinates of the data.
    xlims (list, optional): The x-axis limits. Defaults to None.
    ylabel (str, optional): The y-axis label. Defaults to None.
    margins (float, optional): The y-axis margins. Defaults to None.
    """
    configure_axis(ax)
    ax.plot(x_data, y_data, linewidth=2)
    if xlims is not None: ax.set_xlim(xlims)
    if ylabel is not None: ax.set_ylabel(ylabel)
    if margins is not None: ax.margins(y=margins)

def logplot(y_data, xlabel=None, xlims=None):
    """
    Create log plots for each element in y_data, arranged in a square grid.
    Backward compatible: if y_data is a 1D array, plot it as a single subplot.

    Parameters:
    y_data (list or numpy array): Each element is the y-coordinates for a subplot,
                                  or a single 1D array for one plot.
    xlabel (str, optional): The x-axis label. Defaults to None.
    xlims (list, optional): The x-axis limits. Defaults to None.

    Returns:
    fig (matplotlib figure): The figure.
    axes (numpy array of matplotlib axes): The axes.
    """

    # If y_data is a single 1D array, wrap it in a list for compatibility
    if isinstance(y_data, np.ndarray) and y_data.ndim == 1:
        y_data = [y_data]
        xlims = [xlims]
    elif isinstance(y_data, list) and (len(y_data) > 0) and isinstance(y_data[0], (np.ndarray, list)):
        # Already a list of arrays
        pass
    else:
        # Try to convert to list of arrays
        y_data = [np.array(y_data)]

    n = len(y_data)
    grid_size = math.ceil(math.sqrt(n))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(5*grid_size, 4*grid_size))
    axes = np.array(axes).flatten()
    for i, yd in enumerate(y_data):
        ax = axes[i]
        configure_axis(ax)
        ax.semilogy(yd, linewidth=2)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if xlims[i] is not None:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xticks([xlims[i][0]] + list(ax.get_xticks()) + [xlims[i][1]])
            ax.set_xlim(xlims[i])
        ax.margins(0.1)
    # Hide unused subplots
    for j in range(n, grid_size*grid_size):
        fig.delaxes(axes[j])
    return fig, axes[:n]

def save_figure(fig, filename):
    """
    Save a figure to a file.

    Parameters:
    fig (matplotlib figure): The figure to save.
    filename (str): The filename.
    """
    cwd = os.getcwd()
    data_folder = os.path.join(cwd, "data")
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    fig_path = os.path.join(data_folder, filename[:-4])
    fig.savefig(fig_path)
    pickle.dump(fig, open(fig_path+'.fig.pickle', 'wb'))    
    
#%% # edit the figure later
# import pickle

# with open('data/2024-10-04_13-39_osc_ConformalStormerVerlet_full.fig.pickle', 'rb') as file:
    
#     figx = pickle.load(file)

#     figx.show() # Show the figure, edit it, etc.!
#     ax = figx.axes
#     # ax[0].set_xlabel('time')
#     ax[0].set_ylim([-1e-15, 1e-15])
#     # ax[2].set_xlim([1,6])
#     figx.savefig('data/2024-10-04_13-39_osc_ConformalStormerVerlet_full_2.pdf')
#     figx.savefig('data/2024-10-04_13-39_osc_ConformalStormerVerlet_full_2.eps')

#%%
def tex_table(solver_name, array2print):
    """
    Print a table in LaTeX format.

    Parameters:
    solver_name (str): The name of the solver.
    array2print (list of lists): The data to print.
    """
    print(solver_name, "\n", " \\\\\n".join([" & ".join(map('{0:.6f}'.format, line)) \
                                             for line in array2print]))
    
    
def plot_3dsurface(fig, ax, xx, yy, zz):
    """
    Plot a 3D surface.

    Parameters:
    fig (matplotlib figure): The figure to plot on.
    ax (matplotlib axis): The axis to plot on.
    xx (numpy array): The x-coordinates of the surface.
    yy (numpy array): The y-coordinates of the surface.
    zz (numpy array): The z-coordinates of the surface.
    """
    surf = ax.plot_surface(xx, yy, zz,\
                           cmap = cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    _ = ax.contour(xx, yy, zz, zdir='z', offset=-1, cmap=cm.coolwarm)
    ax.view_init(30, -135)
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_zlim3d(-1, abs(zz).max())

#%% Testing
def test_PlotScript():
    raise NotImplementedError

if __name__ == '__main__':
    test_PlotScript()
