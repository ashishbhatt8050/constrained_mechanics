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
    Create a log plot.

    Parameters:
    y_data (numpy array): The y-coordinates of the data.
    xlabel (str, optional): The x-axis label. Defaults to None.
    xlims (list, optional): The x-axis limits. Defaults to None.

    Returns:
    fig (matplotlib figure): The figure.
    ax (matplotlib axis): The axis.
    """
    fig, ax = plt.subplots()
    configure_axis(ax)
    ax.semilogy(y_data, linewidth=2)
    if xlabel is not None: ax.set_xlabel(xlabel)
    if xlims is not None:
        # Set the x-axis ticks to include both min and max values
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xticks(xlims[0] + list(ax.get_xticks()) + [xlims[1]])
        ax.set_xlim(xlims)  # Extend the x-axis slightly beyond the max value
    ax.margins(0.1)
    return fig, ax

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
    print(solver_name, "\n", " \\\\\n".join([" & ".join(map('{0:.3f}'.format, line)) \
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
