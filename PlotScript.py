#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 15:30:30 2020

@author: ashishbhatt

Copied from https://github.com/jbmouret/matplotlib_for_papers
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator, LogLocator
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
    'text.usetex': False,
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
    ax.tick_params(axis='y', direction='out')
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
        ax.yaxis.set_major_locator(LogLocator())
        # Set margins
        ax.margins(0.1)
        # Set yticks to cover the whole range of yd, rounded to nearest powers of 10
        yd_nonzero = yd[np.isfinite(yd) & (yd > 0)]
        if yd_nonzero.size > 0:
            ymin = yd_nonzero.min()
            ymax = yd_nonzero.max()
            lower = 10 ** math.floor(math.log10(ymin))
            upper = 10 ** math.ceil(math.log10(ymax))
            # yticks = [10 ** exp for exp in range(int(math.floor(math.log10(lower))), int(math.ceil(math.log10(upper))) + 1)]
            # ax.set_yticks(yticks)
            ax.set_ylim([lower, upper])
            
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if xlims[i] is not None:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xlim(xlims[i])

        # # Remove penultimate xticks if too close to boundary xticks
        # xticks = list(ax.get_xticks())
        # if len(xticks) >= 3:
        #     # Find indices of boundary and penultimate ticks
        #     left, penult_left = xticks[0], xticks[1]
        #     penult_right, right = xticks[-2], xticks[-1]
        #     # Compute tolerance as max distance between consecutive xticks
        #     tol = max(np.diff(xticks))
        #     # Remove penultimate left if too close to left
        #     if abs(penult_left - left) < tol:
        #         xticks.pop(1)
        #     # Remove penultimate right if too close to right
        #     if len(xticks) >= 3 and abs(right - penult_right) < tol:
        #         xticks.pop(-2)
        #     ax.set_xticks(xticks)

    # Hide unused subplots
    for j in range(n, grid_size*grid_size):
        fig.delaxes(axes[j])
    return fig, axes[:n]

def save_figure(fig, filename, fig_data=None):
    """
    Save a figure and its data to files.

    Parameters:
    fig (matplotlib figure): The figure to save.
    filename (str): The filename.
    fig_data (dict, optional): A dictionary containing the data used to generate the figure.
    """
    # Ensure the directory exists
    dir_name = os.path.dirname(filename)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    # Save the figure with the specified extension
    fig.savefig(filename)

    # Save the figure object and data
    fig_path_base = os.path.splitext(filename)[0]
    with open(fig_path_base + '.fig.pickle', 'wb') as f:
        pickle.dump(fig, f)
    if fig_data is not None:
        with open(fig_path_base + '.data.pickle', 'wb') as f:
            pickle.dump(fig_data, f)

        # Save data for pgfplots
        pgf_data_path = os.path.join(dir_name, 'pgfplots_data')
        if not os.path.exists(pgf_data_path):
            os.makedirs(pgf_data_path)

        dat_filename_base = os.path.join(pgf_data_path, os.path.splitext(os.path.basename(filename))[0])

        if 'sv' in fig_data or 'sv_g' in fig_data:
            for label, array in fig_data.items():
                dat_filename = f"{dat_filename_base}_{label}.dat"
                np.savetxt(dat_filename, np.c_[np.arange(1, len(array) + 1), array], fmt='%f')
        else:
            if 't_points' in fig_data:
                for key in ['lim_momentum_err', 'angular_momentum_err', 'sym_error', 'g_norm', 'eng_error']:
                    if key in fig_data:
                        data_to_save = np.vstack((fig_data['t_points'], fig_data[key])).T
                        np.savetxt(f"{dat_filename_base}_{key}.dat", data_to_save, fmt='%f')

            if 'coords' in fig_data:
                coords_data = fig_data['coords']
                for i in range(coords_data.shape[1]):
                    particle_filename = f"{dat_filename_base}_coords_particle{i}.dat"
                    np.savetxt(particle_filename, coords_data[:, i, :], fmt='%f')

#%% # edit the figure later
# import pickle
# import matplotlib as mpl
# mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'

# with open('data/2025-09-11/osc_ConformalStormerVerletSolver_full.fig.pickle', 'rb') as file:
    
#     figx = pickle.load(file)

#     figx.show() # Show the figure, edit it, etc.!
#     ax = figx.axes
#     ax[0].set_xlabel('time')
#     ax[0].set_ylim([-1e-15, 1e-15])
#     # ax[2].set_xlim([1,6])
#     figx.savefig('data/2025-09-11/osc_ConformalStormerVerletSolver_full_2.pdf')
#     figx.savefig('data/2025-09-11/osc_ConformalStormerVerletSolver_full_2.eps')

#%%
def tex_table(solver_name, array2print):
    """
    Print a table in LaTeX format.

    Parameters:
    solver_name (str): The name of the solver.
    array2print (list of lists): The data to print.
    """
    print(solver_name, "\n", " \n".join([" & ".join(map('{0:.6f}'.format, line)) 
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
    surf = ax.plot_surface(xx, yy, zz,
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
