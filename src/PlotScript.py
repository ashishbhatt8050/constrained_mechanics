#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting utilities for constrained mechanics simulations.

This module provides plotting functions optimized for scientific visualization.
All color palettes are chosen for colorblind accessibility.

Color Accessibility:
- Discrete data: Uses the Okabe-Ito palette (http://jfly.uni-koeln.de/color/)
  This palette is specifically designed to be distinguishable by individuals with
  protanopia (red-blindness), deuteranopia (green-blindness), and tritanopia
  (blue-yellow-blindness).
- Continuous data: Uses viridis colormap, which is perceptually uniform and
  accessible to all types of colorblindness.

Created on Mon May 11 15:30:30 2020
@author: ashishbhatt

Adapted from https://github.com/jbmouret/matplotlib_for_papers
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, LogLocator
from cycler import cycler
import os
import pickle
from functools import wraps
from time import time
import math
import numpy as np

# Colorblind-safe palettes (Okabe-Ito palette for universal accessibility)
# Reference: https://jfly.uni-koeln.de/color/
OKABE_ITO_PALETTE = [
    '#E69F00',  # Orange
    '#56B4E9',  # Sky Blue
    '#009E73',  # Green
    '#F0E442',  # Yellow
    '#0072B2',  # Blue
    '#D55E00',  # Red-Orange
    '#CC79A7',  # Pink
]

# Viridis for continuous data (also colorblind-safe)
VIRIDIS_PALETTE = plt.cm.viridis

# Set up plot parameters with colorblind-safe colors
colors = OKABE_ITO_PALETTE

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

def get_colorblind_palette(palette_name='okabe-ito'):
    """
    Return a colorblind-safe color palette.
    
    Parameters:
    palette_name (str): Name of the palette. Options: 'okabe-ito' (default, universal),
                       'protanopia', 'deuteranopia', 'tritanopia'. Returns the Okabe-Ito
                       palette by default as it's optimized for all types of colorblindness.
    
    Returns:
    list: List of hex color codes in the selected palette.
    """
    palettes = {
        'okabe-ito': OKABE_ITO_PALETTE,  # Universal (works for all colorblindness types)
        'protanopia': OKABE_ITO_PALETTE,  # Red-blind (Okabe-Ito also optimized for this)
        'deuteranopia': OKABE_ITO_PALETTE,  # Green-blind (Okabe-Ito also optimized for this)
        'tritanopia': OKABE_ITO_PALETTE,  # Blue-yellow-blind (Okabe-Ito also optimized for this)
    }
    return palettes.get(palette_name.lower(), OKABE_ITO_PALETTE)

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
    # Local import to access global config without circular dependencies at module level
    from System import MechSystem

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
        
        # Conditionally save data for pgfplots based on global config flag
        if getattr(MechSystem, 'generate_pgfplots_data', True):
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
                    line=dict(color='rgba(100, 100, 100, 0.3)', width=1),
                    opacity=0.3,
                    showlegend=False,
                    name=f'Path {i}'
                ))
            
            # 2. Add dynamic particles (markers) at initial position
            # Use colorblind-safe color (from Okabe-Ito palette)
            fig_ply.add_trace(go.Scatter3d(
                x=coords[0, :, 0], y=coords[0, :, 1], z=coords[0, :, 2],
                mode='markers',
                marker=dict(size=5, color=OKABE_ITO_PALETTE[1]),  # Sky Blue
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
                    {'label': 'Fixed camera', 'method': 'update',
                     'args': [{}, {'sliders': [create_slider('fix', True), create_slider('rot', False)],
                                   'updatemenus': [{'visible': True}, {'visible': False}, {'visible': True}]}]},
                    {'label': 'Rotating camera', 'method': 'update',
                     'args': [{}, {'sliders': [create_slider('fix', False), create_slider('rot', True)],
                                   'updatemenus': [{'visible': False}, {'visible': True}, {'visible': True}]}]}
                ]
            }
            
            fig_ply.update_layout(
                title="3D phase portrait animation",
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

filename = "/home/bhattah/Documents/constrained_mechanics/data/2026-06-19/error_vs_basis_size.fig.pickle"

# # Configuration for custom labels
# X_LABEL_CUSTOM = r'QDEIM basis size ($\ell$)'
# Y_LABEL_LEFT_CUSTOM = 'Relative error'
# Y_LABEL_RIGHT_CUSTOM = 'Function evaluation time (ms)'

if os.path.exists(filename):
    with open(filename, "rb") as file:
        figx = pickle.load(file)

        # # Left Subplot (Relative Error)
        # ax_left = figx.axes[0]
        # ax_left.set_title("")  # Remove the title
        # ax_left.set_xlabel(X_LABEL_CUSTOM)
        # ax_left.set_ylabel(Y_LABEL_LEFT_CUSTOM)
        # ax_left.set_ylim([1e-5, 1e0])  # Set range from 10^-5 to 10^0
        # ax_left.set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0])
        # ax_left.set_xlim([0,55])

        # # Right Subplot (Online Time)
        # ax_right = figx.axes[1]
        # ax_right.set_title("")  # Remove the title
        # ax_right.set_xlabel(X_LABEL_CUSTOM)
        # ax_right.set_ylabel(Y_LABEL_RIGHT_CUSTOM)
        # ax_right.set_ylim([0, 6])  # Set range from 10^-5 to 10^0
        # ax_right.set_yticks([0, 1, 2, 3, 4, 5, 6])
        # ax_right.set_xlim([0,55])

        # Increase axis and legend text size for the loaded figure.
        for ax in figx.axes:
            ax.tick_params(axis='both', labelsize=24)
            ax.xaxis.label.set_size(26)
            ax.yaxis.label.set_size(26)
            ax.title.set_fontsize(26)
            
            # Increase marker sizes in all plot collections
            for collection in ax.collections:
                if hasattr(collection, 'set_sizes'):
                    current_sizes = collection.get_sizes()
                    if current_sizes is not None and len(current_sizes) > 0:
                        collection.set_sizes(current_sizes + 4)
                elif hasattr(collection, '_sizes'):
                    current_sizes = collection._sizes
                    if current_sizes is not None:
                        collection._sizes = current_sizes + 4
                # Ensure collections have a visible facecolor (fill) when hollow
                try:
                    # PathCollection: facecolors array may be empty for hollow markers
                    facecolors = collection.get_facecolors()
                    if facecolors is None or len(facecolors) == 0:
                        edgecolors = None
                        try:
                            edgecolors = collection.get_edgecolors()
                        except Exception:
                            pass
                        if edgecolors is not None and len(edgecolors) > 0:
                            collection.set_facecolor(edgecolors)
                        else:
                            # Fallback: try to set to the collection color or a default
                            try:
                                col = collection.get_color()
                                collection.set_facecolor(col)
                            except Exception:
                                pass
                except Exception:
                    pass
            
            # Also increase marker sizes for line objects
            for line in ax.get_lines():
                marker_size = line.get_markersize()
                if marker_size > 0:
                    line.set_markersize(marker_size + 4)
                # Ensure Line2D markers are filled instead of hollow
                try:
                    if hasattr(line, 'set_fillstyle'):
                        try:
                            line.set_fillstyle('full')
                        except Exception:
                            pass
                except Exception:
                    pass
                try:
                    mfc = None
                    try:
                        mfc = line.get_markerfacecolor()
                    except Exception:
                        mfc = None
                    if mfc is None or mfc == 'none':
                        try:
                            line.set_markerfacecolor(line.get_color())
                        except Exception:
                            pass
                except Exception:
                    pass

        # Ensure R2C1 subplot (third axis) has the correct ylabel
        try:
            if len(figx.axes) >= 3:
                ax_r2c1 = figx.axes[2]
                ax_r2c1.set_ylabel('Mean speedup factor')
                ax_r2c1.yaxis.label.set_size(26)
        except Exception:
            pass

        # Reposition existing legends to the top-right outside all subplots.
        existing_legends = getattr(figx, 'legends', [])
        if existing_legends:
            for i, legend in enumerate(existing_legends):
                legend.set_bbox_to_anchor((1.02, 0.98 - i * 0.18))
                # legend.set_bbox_transform(figx.transFigure)
                legend.set_loc('upper left')
                for text in legend.get_texts():
                    text.set_fontsize(24)
                # Collect legend handles in a backwards-compatible way
                if hasattr(legend, 'legendHandles'):
                    handles = legend.legendHandles
                else:
                    # Fall back to common accessors available on Legend
                    handles = []
                    try:
                        handles.extend(legend.get_lines())
                    except Exception:
                        pass
                    try:
                        handles.extend(legend.get_patches())
                    except Exception:
                        pass

                for handle in handles:
                    if hasattr(handle, 'set_fillstyle'):
                        try:
                            handle.set_fillstyle('full')
                        except Exception:
                            pass
                    if hasattr(handle, 'set_markerfacecolor'):
                        try:
                            facecolor = handle.get_markerfacecolor()
                        except Exception:
                            facecolor = None
                        if facecolor == 'none' or facecolor is None:
                            if hasattr(handle, 'get_color'):
                                try:
                                    handle.set_markerfacecolor(handle.get_color())
                                except Exception:
                                    pass
                title = legend.get_title()
                if title is not None:
                    title.set_fontsize(26)
            figx.subplots_adjust(right=0.70)
        else:
            # Fallback: create a combined figure legend if no legends were saved.
            all_handles = []
            all_labels = []
            for ax in figx.axes:
                handles, labels = ax.get_legend_handles_labels()
                for h, l in zip(handles, labels):
                    if l not in all_labels:
                        if hasattr(h, 'set_fillstyle'):
                            h.set_fillstyle('full')
                        if hasattr(h, 'set_markerfacecolor'):
                            facecolor = h.get_markerfacecolor()
                            if facecolor == 'none' or facecolor is None:
                                if hasattr(h, 'get_color'):
                                    h.set_markerfacecolor(h.get_color())
                        all_handles.append(h)
                        all_labels.append(l)
            if all_handles:
                figx.legend(all_handles, all_labels,
                            loc='upper left', bbox_to_anchor=(1.02, 0.98),
                            bbox_transform=figx.transFigure,
                            frameon=True, title='Legend', fontsize=24, title_fontsize=26)
                figx.subplots_adjust(right=0.70)

        pdf_path = filename.replace(".fig.pickle", ".pdf")
        if os.path.exists(pdf_path):
            try:
                os.remove(pdf_path)
            except Exception as e:
                print(f"Could not remove existing PDF {pdf_path}: {e}")
        figx.savefig(pdf_path, bbox_inches='tight', pad_inches=0.5, dpi=300)
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
    Plot a 3D surface using a colorblind-safe colormap (viridis).

    Parameters:
    fig (matplotlib figure): The figure to plot on.
    ax (matplotlib axis): The axis to plot on.
    xx (numpy array): The x-coordinates of the surface.
    yy (numpy array): The y-coordinates of the surface.
    zz (numpy array): The z-coordinates of the surface.
    """
    # Use viridis colormap (colorblind-safe) instead of coolwarm
    surf = ax.plot_surface(xx, yy, zz,
                           cmap='viridis', linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    _ = ax.contour(xx, yy, zz, zdir='z', offset=-1, cmap='viridis')
    ax.view_init(30, -135)
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_zlim3d(-1, abs(zz).max())

def plot_pareto(time_lapsed, errors_r, errors_dr, solver_names, dt_space, data_folder):
    """
    Generate Pareto plots (Error vs Time) for reduced and hyper-reduced models.
    
    Parameters:
    time_lapsed (list): List containing [full_time, reduced_percent, hyper_percent] arrays.
    errors_r (numpy array): Reduced model errors.
    errors_dr (numpy array): Hyper-reduced model errors.
    solver_names (list): List of solver names.
    dt_space (numpy array): Array of time step sizes.
    data_folder (str): Path to save the plot.
    """
    
    # Reconstruct raw times (seconds)
    # time_lapsed[0] is raw full time
    # time_lapsed[1] is reduced % of full
    # time_lapsed[2] is hyper % of full
    
    raw_time_full = time_lapsed[0]
    raw_time_red = time_lapsed[1] * raw_time_full / 100.0
    raw_time_hyper = time_lapsed[2] * raw_time_full / 100.0
    
    # Average over parameters (axis 2)
    avg_time_full = np.nanmean(raw_time_full, axis=2)
    avg_time_red = np.nanmean(raw_time_red, axis=2)
    avg_time_hyper = np.nanmean(raw_time_hyper, axis=2)
    
    # Std dev of time
    std_time_red = np.nanstd(raw_time_red, axis=2)
    std_time_hyper = np.nanstd(raw_time_hyper, axis=2)
    
    # Mean and Std errors over parameters (axis 2)
    mean_err_red = np.nanmean(errors_r, axis=2)
    std_err_red = np.nanstd(errors_r, axis=2)
    
    mean_err_hyper = np.nanmean(errors_dr, axis=2)
    std_err_hyper = np.nanstd(errors_dr, axis=2)
    
    n_solvers = len(solver_names)
    fig, axes = plt.subplots(1, n_solvers, figsize=(6 * n_solvers, 6), constrained_layout=False)
    if n_solvers == 1: axes = [axes]

    # Markers for different time steps
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '*']
    
    # For common legends
    line_handles = []

    for i, solver_name in enumerate(solver_names):
        ax = axes[i]
        configure_axis(ax)
        
        ax.set_xscale('log')
        ax.set_yscale('log')

        # Plot connecting lines with error bars (no markers)
        h_red, _, _ = ax.errorbar(avg_time_red[i], mean_err_red[i], xerr=std_time_red[i], yerr=std_err_red[i], fmt=':', color=colors[1], capsize=3)
        h_hyper, _, _ = ax.errorbar(avg_time_hyper[i], mean_err_hyper[i], xerr=std_time_hyper[i], yerr=std_err_hyper[i], fmt=':', color=colors[5], capsize=3)
        
        # Plot individual points with specific markers
        for j, dt in enumerate(dt_space):
            m = markers[j % len(markers)]
            # Reduced
            ax.plot(avg_time_red[i, j], mean_err_red[i, j], marker=m, color=colors[1], linestyle='None')
            # Hyper-reduced
            ax.plot(avg_time_hyper[i, j], mean_err_hyper[i, j], marker=m, color=colors[5], linestyle='None')
            
            # Annotate with speedup factors
            speedup_red = avg_time_full[i, j] / avg_time_red[i, j]
            ax.annotate(f'{speedup_red:.1f}x', (avg_time_red[i, j], mean_err_red[i, j]), textcoords="offset points", xytext=(5, 5), ha='left', fontsize=18)
            
            speedup_hyper = avg_time_full[i, j] / avg_time_hyper[i, j]
            ax.annotate(f'{speedup_hyper:.1f}x', (avg_time_hyper[i, j], mean_err_hyper[i, j]), textcoords="offset points", xytext=(5, 5), ha='left', fontsize=18)
        
        # Plot Full model reference times (vertical lines)
        h_full = None
        for j, dt in enumerate(dt_space):
            line = ax.axvline(x=avg_time_full[i, j], color=colors[2], linestyle='--', alpha=0.5)
            if j == 0:
                h_full = line

        ax.set_xlabel('Average wall time (s)')
        if i == 0:
            ax.set_ylabel('Mean global error')
        ax.set_title(f'{solver_name}')
        ax.grid(True, which="both", ls="-", alpha=0.3)
        
        # Collect handles for common legend on first pass
        if i == 0:
            line_handles.extend([h_red, h_hyper, h_full])

    # Create common legends below the subplots
    line_labels = ['Reduced', 'Hyper-reduced', 'Full model']
    marker_handles = [Line2D([0], [0], color='k', marker=markers[j % len(markers)], linestyle='None', label=f'dt={dt:.0e}') for j, dt in enumerate(dt_space)]
    
    # Adjust subplots to make room for legends at the bottom
    fig.tight_layout(rect=[0, 0.22, 1, 1])
    
    leg1 = fig.legend(handles=line_handles, labels=line_labels, loc='lower center', bbox_to_anchor=(0.5, 0.12), ncol=3, title='Model type', frameon=True)
    fig.add_artist(leg1) # Add the first legend manually to avoid it being overwritten
    
    fig.legend(handles=marker_handles, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=len(dt_space), title='Time step size', frameon=True)
        
    filename = os.path.join(data_folder, f"pareto_combined.pdf")
    fig.savefig(filename, bbox_inches='tight')
    print(f"Saved Pareto plot to {filename}")
    plt.close(fig)



def adjust_axis_limits(ax, data_points, axis='y', is_log_scale=False, max_steps=1):
    """
    Adjust axis limits based on data range and current tick intervals.
    Extends limits by tick steps if data overflows.
    """
    if not data_points:
        return

    ax.relim()
    ax.autoscale_view()

    data_min = np.min(data_points)
    data_max = np.max(data_points)
    plt.draw()

    if axis == 'x':
        get_ticks = ax.get_xticks
        set_lim = ax.set_xlim
    elif axis == 'y':
        get_ticks = ax.get_yticks
        set_lim = ax.set_ylim
    elif axis == 'z':
        get_ticks = ax.get_zticks
        set_lim = ax.set_zlim
    else:
        return

    tick_values = get_ticks()
    if len(tick_values) < 2: return

    if is_log_scale:
        log_ticks = np.log10(tick_values[tick_values > 0])
        if len(log_ticks) < 2: return
        
        log_step = log_ticks[-1] - log_ticks[-2]
        if log_step <= 1e-12: return 
        new_min, new_max = log_ticks[0], log_ticks[-1]
        
        steps = 0
        while new_max < np.log10(data_max) and steps < max_steps:
            new_max += log_step
            steps += 1
        
        if data_min > 0:
            steps = 0
            while new_min > np.log10(data_min) and steps < max_steps:
                new_min -= log_step
                steps += 1
        
        set_lim([10**new_min, 10**new_max])
    else:
        step = tick_values[1] - tick_values[0]
        if step <= 1e-12: return
        new_min, new_max = tick_values[0], tick_values[-1]
        
        steps = 0
        while new_max < data_max and steps < max_steps:
            new_max += step
            steps += 1
            
        steps = 0
        while new_min > data_min and steps < max_steps:
            new_min -= step
            steps += 1
        
        if axis == 'x' and 'dimension' in ax.get_xlabel().lower():
            new_min = max(0, new_min)
            
        set_lim([new_min, new_max])


def plot_error_vs_basis_size(basis_sizes, hr_basis_sizes, errors_r, errors_dr, solver_names, pod_tols, data_folder, **kwargs):
    """
    Plot ROM and HROM errors as a function of the reduced basis size.
    Also plots speedup factors and convergence orders as box plots against dt.
    Markers correspond to pod_tol values.
    """
    # Extract study data from kwargs
    dt_space = kwargs.get('dt_space', [])
    full_times_raw = kwargs.get('full_times_raw', {})
    full_errs_raw = kwargs.get('full_errs_raw', {})    
    times_r_raw = kwargs.get('times_r_raw', [])
    times_dr_raw = kwargs.get('times_dr_raw', [])

    fig, axes_grid = plt.subplots(2, 2, figsize=(22, 18.6))
    ax_err_r, ax_err_dr = axes_grid[0]
    ax_speed_r, ax_speed_dr = axes_grid[1]
    # ax_conv_r, ax_conv_dr = axes_grid[2]
    
    col_axes = [ax_err_r, ax_err_dr]

    # Define markers for pod_tols
    tol_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '*']
    marker_map = {tol: tol_markers[i % len(tol_markers)] for i, tol in enumerate(pod_tols)}

    # Define line styles and colors for solvers
    solver_linestyles = ['-', '--', ':', '-.']
    solver_colors = OKABE_ITO_PALETTE # Use the colorblind-safe palette

    # Store handles for legends
    marker_handles = []
    
    # To ensure marker legend elements are unique
    added_marker_labels = set()
    
    # Row 1: Max Error vs Dimension
    for ax_idx, ax in enumerate(col_axes):
        configure_axis(ax)
        ax.tick_params(axis='both', labelsize=24)
        all_x_col, all_y_col = [], []
        curr_max_x = (2 * np.nanmax(np.concatenate(basis_sizes)) if ax_idx == 0 else 2 * np.nanmax(np.concatenate(hr_basis_sizes))) if basis_sizes else 1

        for i, name in enumerate(solver_names):
            color = solver_colors[i % len(solver_colors)]
            x_sizes_raw = basis_sizes[i] if ax_idx == 0 else hr_basis_sizes[i]
            err_mats = errors_r[i] if ax_idx == 0 else errors_dr[i]

            for j, tol in enumerate(pod_tols):
                e_data = err_mats[j]
                if not np.isnan(x_sizes_raw[j]) and isinstance(e_data, np.ndarray) and e_data.size > 0:
                    x_plot_val = 2 * x_sizes_raw[j]
                    # Average over Omega2 (axis 1)
                    mean_errors = np.nanmean(e_data, axis=1)
                    alpha = 0.7
                    
                    for k in range(len(dt_space)):
                        if not np.isnan(mean_errors[k]):
                            ax.semilogy(x_plot_val, mean_errors[k], marker=marker_map[tol], 
                                        color=color, alpha=alpha, markersize=14 if ax_idx == 0 else 12,
                                        fillstyle='full', 
                                        markerfacecolor=color, linestyle='None')
                            all_y_col.append(mean_errors[k])
                    
                    all_x_col.append(x_plot_val)
                    if tol not in added_marker_labels:
                        marker_handles.append(Line2D([0], [0], marker=marker_map[tol], color='k', linestyle='None',
                                                     label=f'{tol:.0e}', markersize=8, alpha=0.7))
                        added_marker_labels.add(tol)

        ax.set_ylabel('Mean max relative error' if ax_idx == 0 else '')
        ax.set_xlabel(r'Reduced dimension $2r$' if ax_idx == 0 else r'Hyper-reduced dimension $2\ell$')
        ax.xaxis.label.set_size(26)
        ax.yaxis.label.set_size(26)
        ax.title.set_fontsize(26)
        adjust_axis_limits(ax, all_x_col, axis='x', is_log_scale=False, max_steps=5)
        adjust_axis_limits(ax, all_y_col, axis='y', is_log_scale=True, max_steps=5)

    # --- Row 2: Speedup Factor (Box plots vs dt) ---
    for ax_idx, (ax, data_type) in enumerate(zip([ax_speed_r, ax_speed_dr], ['times_r_raw', 'times_dr_raw'])):
        configure_axis(ax)
        ax.tick_params(axis='both', labelsize=24)
        ax.set_ylabel('Mean speedup factor' if ax_idx == 0 else '')
        ax.set_xlabel(r'Time step-size $\Delta t$')
        ax.xaxis.label.set_size(26)
        ax.yaxis.label.set_size(26)
        ax.title.set_fontsize(26)
        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        
        times_raw_list = kwargs.get(data_type, [])
        positions = np.arange(len(dt_space))
        n_solvers = len(solver_names)
        n_tols = len(pod_tols)
        width = 0.8 / (n_solvers * n_tols)
        dt_exponents = np.log2(dt_space)
        all_speedups = []

        for i, name in enumerate(solver_names):
            color = solver_colors[i % len(solver_colors)]
            for t_idx, tol in enumerate(pod_tols):
                alpha = 0.7
                offset = (i * n_tols + t_idx - (n_solvers * n_tols - 1) / 2) * width
                
                for d_idx in range(len(dt_space)):
                    t_red_mat = times_raw_list[i][t_idx]
                    t_full_mat = full_times_raw.get(name)
                    if isinstance(t_red_mat, np.ndarray) and t_full_mat is not None:
                        speedup_vals = t_full_mat[d_idx] / t_red_mat[d_idx]
                        mean_s = np.nanmean(speedup_vals)
                        if not np.isnan(mean_s):
                            ax.plot(positions[d_idx] + offset, mean_s, marker=marker_map[tol], 
                                    color=color, alpha=alpha, markersize=14, linestyle='None')
                            all_speedups.append(mean_s)

        adjust_axis_limits(ax, all_speedups, axis='y', is_log_scale=False, max_steps=10)
        ax.set_xticks(positions)
        ax.set_xticklabels([rf"$2^{{{int(exp)}}}$" for exp in dt_exponents])

        # Custom "bin" brackets for x-axis to group markers per step-size
        for pos in positions:
            ax.plot([pos - 0.4, pos - 0.4, pos + 0.4, pos + 0.4], 
                    [-0.03, 0, 0, -0.03], transform=ax.get_xaxis_transform(), 
                    color='black', linewidth=1.5, clip_on=False, linestyle='-')
        ax.tick_params(axis='x', which='both', length=0)

    solver_handles = [
        Line2D([0], [0], color=solver_colors[i], lw=4, alpha=0.5,
               label="Model 1" if "StormerVerlet" in name else "Model 2" if "DiscreteGradient" in name else name)
        for i, name in enumerate(solver_names)
    ]
    solver_handles.sort(key=lambda h: h.get_label())

    # Create combined tolerance legend (marker shape + transparency)
    combined_tol_handles = [
        Line2D([0], [0], marker=marker_map[tol], color='k', linestyle='None', 
               alpha=0.7,
               label=f'{tol:.0e}', markersize=10)
               for j, tol in enumerate(pod_tols)
    ]

    # Place legends outside the 2x2 axes in the top-right corner (stacked)
    legend1 = fig.legend(handles=solver_handles, title="Model", loc='upper left', bbox_to_anchor=(0.92, 0.98), bbox_transform=fig.transFigure, fontsize=20, title_fontsize=22, frameon=True)
    fig.add_artist(legend1)
    legend2 = fig.legend(handles=combined_tol_handles, title="POD Tolerance", loc='upper left', bbox_to_anchor=(0.92, 0.82), bbox_transform=fig.transFigure, fontsize=20, title_fontsize=22, frameon=True)
    fig.add_artist(legend2)

    # Increase right margin so legends don't overlap the subplots
    plt.subplots_adjust(right=0.70, hspace=0.4, wspace=0.25)


    filename = os.path.join(data_folder, "error_vs_basis_size.pdf")
    save_figure(fig, filename)
    print(f"Saved Error vs. Basis Size plot to {filename}")
    plt.close(fig)

# %% Testing
def test_PlotScript():
    raise NotImplementedError

if __name__ == '__main__':
    test_PlotScript()
