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

def plot_omega_distribution(train_set, test_set, filename=None):
    """
    Plot the distribution of Omega2 parameters for train and test sets.
    """
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    configure_axis(ax)
    
    ax.hist(train_set.flatten(), bins=20, alpha=0.5, label='Train', density=True, color=colors[0])
    if test_set is not None and test_set.size > 0:
        ax.hist(test_set.flatten(), bins=20, alpha=0.5, label='Test', density=True, color=colors[1])
    
    ax.set_xlabel(r'$\Omega^2$ value')
    ax.set_ylabel('Density')
    ax.set_title(r'Distribution of $\Omega^2$ parameters')
    ax.legend()
    
    if filename:
        save_figure(fig, filename)
    return fig

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

    # Check for 3D axes and generate Plotly HTML if applicable
    is_3d = any(getattr(ax, 'name', '') == '3d' for ax in fig.axes)
    
    if is_3d and fig_data is not None and 'coords' in fig_data:
        try:
            import plotly.graph_objects as go
            
            coords = fig_data['coords']
            t_points = fig_data.get('t_points', np.arange(coords.shape[0]))
            
            # Downsample for animation frames (limit to ~100 frames)
            n_steps = coords.shape[0]
            n_frames = 100
            step_size = max(1, n_steps // n_frames)
            frame_indices = np.arange(0, n_steps, step_size)
            
            fig_ply = go.Figure()
            
            # 1. Add static background trajectories (faint lines)
            for i in range(coords.shape[1]):
                fig_ply.add_trace(go.Scatter3d(
                    x=coords[:, i, 0], y=coords[:, i, 1], z=coords[:, i, 2],
                    mode='lines',
                    line=dict(color='gray', width=1),
                    opacity=0.3,
                    showlegend=False,
                    name=f'Path {i}'
                ))
            
            # 2. Add dynamic particles (markers) at initial position
            fig_ply.add_trace(go.Scatter3d(
                x=coords[0, :, 0], y=coords[0, :, 1], z=coords[0, :, 2],
                mode='markers',
                marker=dict(size=5, color='teal'),
                name='Particles'
            ))
            
            particle_trace_idx = len(fig_ply.data) - 1
            
            # 3. Create frames
            frames_fixed = []
            frames_rot = []
            
            for i, k in enumerate(frame_indices):
                # Rotate camera around Z axis
                angle = 2 * np.pi * i / len(frame_indices)
                radius = 1.5

                # Shared data for both frames
                frame_data = [go.Scatter3d(
                        x=coords[k, :, 0],
                        y=coords[k, :, 1],
                        z=coords[k, :, 2]
                    )]

                # Fixed camera frame
                frames_fixed.append(go.Frame(
                    data=frame_data,
                    traces=[particle_trace_idx],
                    name=f'fix_{k}'
                ))

                # Rotating camera frame
                frames_rot.append(go.Frame(
                    data=frame_data,
                    traces=[particle_trace_idx],
                    layout=dict(scene=dict(camera=dict(
                        eye=dict(x=radius*np.cos(angle), y=radius*np.sin(angle), z=0.8)
                    ))),
                    name=f'rot_{k}'
                ))
            
            fig_ply.frames = frames_fixed + frames_rot
            
            # 4. Create Sliders and Menus
            def create_slider(prefix, visible):
                return {
                    'pad': {'b': 10, 't': 50},
                    'len': 0.9, 'x': 0.1, 'y': 0,
                    'steps': [{
                        'args': [[f'{prefix}_{k}'], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}],
                        'label': f'{t_points[int(k)]:.2f}',
                        'method': 'animate'
                    } for k in frame_indices],
                    'currentvalue': {'prefix': 'Time: ', 'visible': True, 'xanchor': 'right'},
                    'visible': visible
                }

            def create_play_pause(prefix, visible):
                return {
                    'type': 'buttons', 'showactive': False,
                    'y': 0, 'x': 0, 'xanchor': 'right', 'yanchor': 'top', 'pad': {'t': 50, 'r': 10},
                    'buttons': [{
                        'label': 'Play', 'method': 'animate',
                        'args': [[f'{prefix}_{k}' for k in frame_indices], 
                                 {'frame': {'duration': 50, 'redraw': True}, 'fromcurrent': True}]
                    }, {
                        'label': 'Pause', 'method': 'animate',
                        'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}]
                    }],
                    'visible': visible
                }

            # Define components
            slider_fixed = create_slider('fix', True)
            slider_rot = create_slider('rot', False)
            
            menu_fixed = create_play_pause('fix', True)
            menu_rot = create_play_pause('rot', False)
            
            menu_toggle = {
                'type': 'buttons', 'direction': 'left', 'pad': {'r': 10, 't': 10},
                'showactive': True, 'x': 0.1, 'xanchor': 'right', 'y': 0.2, 'yanchor': 'top',
                'buttons': [
                    {'label': 'Fixed Camera', 'method': 'update',
                     'args': [{}, {'sliders': [create_slider('fix', True), create_slider('rot', False)],
                                   'updatemenus': [{'visible': True}, {'visible': False}, {'visible': True}]}]},
                    {'label': 'Rotating Camera', 'method': 'update',
                     'args': [{}, {'sliders': [create_slider('fix', False), create_slider('rot', True)],
                                   'updatemenus': [{'visible': False}, {'visible': True}, {'visible': True}]}]}
                ]
            }
            
            fig_ply.update_layout(
                title="3D Phase Portrait Animation",
                scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                updatemenus=[menu_fixed, menu_rot, menu_toggle],
                sliders=[slider_fixed, slider_rot]
            )
            
            html_filename = filename.replace('.pdf', '.html')
            fig_ply.write_html(html_filename)
            
        except ImportError:
            pass
        except Exception as e:
            print(f"Could not save interactive plot: {e}")

# %% edit the figure later
import pickle
import numpy as np

filename = "data/2025-12-26_12-32/osc_sv_rbDiscreteGradient.fig.pickle"
with open(filename, "rb") as file:

    figx = pickle.load(file)

    for ax in figx.axes:
        lines = ax.get_lines()
        if lines:
            x_data = np.concatenate([l.get_xdata() for l in lines])
            # print(f"x_data: min={x_data.min()}, max={x_data.max()}")
            if x_data.size > 0:
                xticks = np.append(ax.get_xticks(), [int(x_data.min())])
                ax.set_xticks(xticks)

            y_data = np.concatenate([l.get_ydata() for l in lines])
            if y_data.size > 0:
                y_max = y_data.max()
                if y_max > 0:
                    next_pow_10 = 10 ** (np.floor(np.log10(y_max)) + 1)
                    yticks = np.unique(np.append(ax.get_yticks(), [next_pow_10]))
                    ax.set_yticks(yticks)
        ax.set_xlabel("")

        for item in (
            [ax.title, ax.xaxis.label, ax.yaxis.label]
            + ax.get_xticklabels()
            + ax.get_yticklabels()
        ):
            item.set_fontsize(item.get_fontsize() - 6)

    figx.show()  # Show the figure, edit it, etc.!
    figx.savefig(filename.replace(".fig.pickle", ".pdf"), pad_inches=0.5, dpi=300)
    # figx.savefig("data/2025-12-20_12-43_/osc_ConformalStormerVerletSolver_full_2.eps")

# %%
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

# %% Testing
def test_PlotScript():
    raise NotImplementedError

if __name__ == '__main__':
    test_PlotScript()
