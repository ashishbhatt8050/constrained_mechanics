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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, LogLocator
from cycler import cycler
import os
import shutil
import pickle
from functools import wraps
from time import time
import math
import hashlib
import numpy as np
import matplotlib.cm as mcm
import matplotlib.colors as mcolors

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

has_latex = bool(shutil.which('latex'))
params = {
    'axes.labelsize': 20,
    'axes.prop_cycle': (cycler(color=colors[:3]) + cycler(linestyle=['-','--','-.'])),
    'font.size': 20,
    'legend.fontsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'text.usetex': has_latex,
    'font.family': 'serif' if has_latex else 'sans-serif',
    'figure.figsize': [14, 6],
    'figure.autolayout': False
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
    if xlims is not None:
        ax.set_xlim(xlims)
    else:
        adjust_axis_limits(ax, x_data, axis='x', is_log_scale=False)
    if ylabel is not None: ax.set_ylabel(ylabel)
    if margins is not None: ax.margins(y=margins)
    adjust_axis_limits(ax, y_data, axis='y', is_log_scale=False)

def logplot(y_data, xlabel=None, xlims=None, ylabel=None):
    """
    Create log plots for each element in y_data, arranged in a square grid.
    Backward compatible: if y_data is a 1D array, plot it as a single subplot.

    Parameters:
    y_data (list or numpy array): Each element is the y-coordinates for a subplot,
                                    or a single 1D array for one plot.
    xlabel (str, optional): The x-axis label. Defaults to None.
    xlims (list, optional): The x-axis limits. Defaults to None.
    ylabel (str, optional): The y-axis label. Defaults to None.

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
            adjust_axis_limits(ax, yd_nonzero, axis='y', is_log_scale=True, max_steps=5)
            
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if xlims[i] is not None:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xlim(xlims[i])
        else:
            adjust_axis_limits(ax, np.arange(len(yd)), axis='x', is_log_scale=False)

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
    
    # Adjust axis limits for Omega2 data
    all_omega = train_set.flatten()
    if test_set is not None and test_set.size > 0:
        all_omega = np.concatenate([all_omega, test_set.flatten()])
    adjust_axis_limits(ax, all_omega, axis='x', is_log_scale=False)

def plot_structured_rb_components(components_dict, filename=None, title=None):
    """
    Approach 1: Plot multi-block singular values and principal spectra of the structured RB assembly:
    [v0, V_dyn, V_nablaH, V_enrich(G)].

    Parameters
    ----------
    components_dict : dict
        Dictionary containing:
        - 'sv_dyn': 1D array of kinematic POD singular values
        - 'sv_hamz': 1D array of force residual POD singular values (optional)
        - 'sv_enrich': 1D array of constraint normal principal sines sin(theta_k) = sigma_k(R_G)
        - 'sv_Gr': 1D array of singular values of reduced constraint Jacobian G_0 V
        - 'tol_dyn': float, tolerance used for kinematic truncation (optional)
        - 'tol_hamz': float, tolerance used for force truncation (optional)
        - 'tau_tol': float, rank-filtering cutoff for enrichment (default: 1e-8)
        - 'cond_Gr': float, condition number of G_0 V (optional)
    filename : str, optional
        File path to save the figure.
    title : str, optional
        Overall figure title.

    Returns
    -------
    fig, axes : tuple
        Matplotlib Figure and array of Axes.
    """
    sv_dyn = components_dict.get('sv_dyn', None)
    sv_hamz = components_dict.get('sv_hamz', None)
    sv_enrich = components_dict.get('sv_enrich', None)
    sv_Gr = components_dict.get('sv_Gr', None)
    tau_tol = components_dict.get('tau_tol', 1e-8)
    cond_Gr = components_dict.get('cond_Gr', None)

    has_hamz = sv_hamz is not None and len(sv_hamz) > 0

    if has_hamz:
        fig, axes_grid = plt.subplots(2, 2, figsize=(15, 11))
        axes = axes_grid.flatten()
    else:
        fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
        axes = np.array(axes).flatten()
    
    panel_idx = 0

    # Panel 1: Kinematic POD Spectrum
    if sv_dyn is not None and len(sv_dyn) > 0:
        ax = axes[panel_idx]
        configure_axis(ax)
        idx_dyn = np.arange(1, len(sv_dyn) + 1)
        ax.semilogy(idx_dyn, sv_dyn, 'o-', color=OKABE_ITO_PALETTE[4], linewidth=2, markersize=5,
                    label=r'$\sigma_i(S_{\mathrm{kin}})$')
        ax.set_xlabel('Kinematic mode index $i$')
        ax.set_ylabel('Singular value magnitude')
        ax.set_title(r'(a) Kinematic POD modes ($V_{\mathrm{dyn}}$)')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, linestyle=':', alpha=0.6)
        adjust_axis_limits(ax, idx_dyn, axis='x', is_log_scale=False, max_steps=1)
        adjust_axis_limits(ax, sv_dyn[sv_dyn > 0], axis='y', is_log_scale=True, max_steps=1)
        panel_idx += 1

    # Panel 2 (if present): Hamiltonian Force Residual POD
    if has_hamz:
        ax = axes[panel_idx]
        configure_axis(ax)
        idx_hamz = np.arange(1, len(sv_hamz) + 1)
        ax.semilogy(idx_hamz, sv_hamz, 'd-', color=OKABE_ITO_PALETTE[2], linewidth=2, markersize=5,
                    label=r'$\sigma_i(R_{\nabla H})$')
        ax.set_xlabel('Force mode index $i$')
        ax.set_ylabel('Residual singular value')
        ax.set_title(r'(b) Force residual ($V_{\nabla H}$)')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, linestyle=':', alpha=0.6)
        adjust_axis_limits(ax, idx_hamz, axis='x', is_log_scale=False, max_steps=1)
        adjust_axis_limits(ax, sv_hamz[sv_hamz > 0], axis='y', is_log_scale=True, max_steps=1)
        panel_idx += 1

    # Panel 3: Constraint Normal Principal Sines
    if sv_enrich is not None and len(sv_enrich) > 0:
        ax = axes[panel_idx]
        configure_axis(ax)
        idx_enrich = np.arange(1, len(sv_enrich) + 1)
        panel_letter = chr(ord('a') + panel_idx)
        ax.semilogy(idx_enrich, sv_enrich, 's-', color=OKABE_ITO_PALETTE[5], linewidth=2, markersize=5,
                    label=r'$\sin\theta_k(R_G)$')
        ax.set_xlabel('Constraint normal index $k$')
        ax.set_ylabel(r'Principal sine $\sigma_k(R_G) = \sin\theta_k$')
        ax.set_title(f'({panel_letter}) Constraint normals ($V_{{\\mathrm{{enrich}}}}$)')
        min_enrich = np.min(sv_enrich[sv_enrich > 0])
        if min_enrich < 1e-2:
            y_pts_enrich = np.concatenate([sv_enrich[sv_enrich > 0], [tau_tol]])
        else:
            y_pts_enrich = sv_enrich[sv_enrich > 0]
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, linestyle=':', alpha=0.6)
        adjust_axis_limits(ax, idx_enrich, axis='x', is_log_scale=False, max_steps=1)
        adjust_axis_limits(ax, y_pts_enrich, axis='y', is_log_scale=True, max_steps=1)
        if tau_tol >= ax.get_ylim()[0]:
            ax.axhline(tau_tol, color='dimgray', linestyle='--', linewidth=1.5,
                       label=rf'Threshold $\tau = 10^{{{int(np.round(np.log10(tau_tol)))}}}$')
        else:
            ax.plot([], [], color='dimgray', linestyle='--', linewidth=1.5,
                    label=rf'Threshold $\tau = 10^{{{int(np.round(np.log10(tau_tol)))}}}$ ($<{min_enrich:.2f}$)')
        ax.legend(loc='lower left', framealpha=0.9, fontsize=15)
        panel_idx += 1

    # Panel 4: Reduced Constraint Jacobian Spectrum
    if sv_Gr is not None and len(sv_Gr) > 0:
        ax = axes[panel_idx]
        configure_axis(ax)
        idx_Gr = np.arange(1, len(sv_Gr) + 1)
        panel_letter = chr(ord('a') + panel_idx)
        ax.plot(idx_Gr, sv_Gr, '^-', color=OKABE_ITO_PALETTE[1], linewidth=2, markersize=5)
        ax.set_xlabel('Constraint index $k$')
        ax.set_ylabel(r'Singular value $\sigma_k(G_0 V)$')
        if cond_Gr is None:
            cond_Gr = sv_Gr.max() / max(1e-15, sv_Gr.min())
        ax.set_title(f'({panel_letter}) Reduced Jacobian $G_0 V$ (cond={cond_Gr:.2f})')
        min_g, max_g = sv_Gr.min(), sv_Gr.max()
        if abs(max_g - min_g) < 1e-6:
            ax.set_ylim([0.0, max_g * 1.5])
        else:
            ax.set_ylim([0.0, max_g * 1.2])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, linestyle=':', alpha=0.6)
        adjust_axis_limits(ax, idx_Gr, axis='x', is_log_scale=False, max_steps=1)
        y_pts_Gr = np.concatenate([[0.0], sv_Gr])
        adjust_axis_limits(ax, y_pts_Gr, axis='y', is_log_scale=False, max_steps=1)

    fig.tight_layout()
    if has_hamz:
        fig.subplots_adjust(top=0.92, wspace=0.38, hspace=0.38)
    else:
        fig.subplots_adjust(top=0.92, wspace=0.38)

    if title:
        fig.suptitle(title, fontsize=18, y=0.98)

    if filename:
        save_figure(fig, filename, fig_data=components_dict)

    return fig, axes


def plot_structured_rb_modal_energy(modal_energies, block_slices=None, s_projected=None, filename=None, title=None):
    """
    Approach 2: Plot modal snapshot projection energy waterfall across structured blocks:
    [v0, V_dyn, V_nablaH, V_enrich(G)], and subspace-restricted singular values.

    Parameters
    ----------
    modal_energies : 1D array
        Modal projection energy E_j = ||V_{:, j}^T S||_2 for each basis mode j.
    block_slices : dict
        Dict mapping block label to (start_idx, end_idx) 0-indexed half-open slice.
        e.g., {'Anchor (v0)': (0, 1), 'Kinematics': (1, 23), ...}
    s_projected : 1D array, optional
        Singular values of the projected snapshot data sigma_k(V^T S).
    filename : str, optional
        File path to save the figure.
    title : str, optional
        Overall figure title.

    Returns
    -------
    fig, axes : tuple
        Matplotlib Figure and Axes.
    """
    if isinstance(modal_energies, dict):
        d_modal = modal_energies
        modal_energies = d_modal['modal_energies']
        block_slices = d_modal['block_slices']
        s_projected = d_modal.get('s_projected', s_projected)

    has_proj = s_projected is not None and len(s_projected) > 0
    if has_proj:
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(18, 5.5), gridspec_kw={'width_ratios': [1.8, 1.2]})
        axes = [ax_l, ax_r]
    else:
        fig, ax_l = plt.subplots(1, 1, figsize=(12, 5.5))
        axes = [ax_l]

    configure_axis(ax_l)
    r_total = len(modal_energies)
    modes = np.arange(1, r_total + 1)
    safe_energies = np.maximum(modal_energies, 1e-15)

    # Semilogy plot of modal energies
    ax_l.semilogy(modes, safe_energies, '.-', color='black', linewidth=1.2, markersize=5, zorder=4)

    # Shaded colored background regions for each block
    palette_colors = [OKABE_ITO_PALETTE[0], OKABE_ITO_PALETTE[4], OKABE_ITO_PALETTE[2], OKABE_ITO_PALETTE[5], OKABE_ITO_PALETTE[6]]
    legend_elements = []

    for idx, (label, (start, end)) in enumerate(block_slices.items()):
        color = palette_colors[idx % len(palette_colors)]
        x_start = start + 0.5
        x_end = end + 0.5
        ax_l.axvspan(x_start, x_end, color=color, alpha=0.3, zorder=1)
        if end < r_total:
            ax_l.axvline(x_end, color='gray', linestyle=':', linewidth=1.2, zorder=2)
        n_modes = end - start
        legend_elements.append(Patch(facecolor=color, edgecolor='none', alpha=0.5, label=f'{label} ({n_modes})'))

    ax_l.set_xlabel('Reduced basis mode index $j$')
    ax_l.set_ylabel(r'Modal projection energy $\|V_{:, j}^T S\|_2$')
    ax_l.set_title(r'(a) Snapshot energy across structured basis blocks')
    ax_l.legend(handles=legend_elements, loc='upper right', framealpha=0.95, fontsize=15)
    ax_l.grid(True, linestyle=':', alpha=0.6)
    adjust_axis_limits(ax_l, modes, axis='x', is_log_scale=False, max_steps=1)
    adjust_axis_limits(ax_l, safe_energies[safe_energies > 0], axis='y', is_log_scale=True, max_steps=1)

    # Right panel: Subspace SVD
    if has_proj:
        configure_axis(ax_r)
        safe_s_proj = np.maximum(s_projected, 1e-15)
        idx_s = np.arange(1, len(safe_s_proj) + 1)
        ax_r.semilogy(idx_s, safe_s_proj, 'o-', color=OKABE_ITO_PALETTE[1], linewidth=2, markersize=4)

        # Highlight kinematic state rank if provided
        for label, (start, end) in block_slices.items():
            if 'Constraint' in label or 'enrich' in label:
                ax_r.axvline(start, color=OKABE_ITO_PALETTE[5], linestyle='--', linewidth=1.5,
                             label=f'Kinematic rank $r_{{\\mathrm{{state}}}}={start}$')
                break

        ax_r.set_xlabel('Projected mode index $k$')
        ax_r.set_ylabel(r'Singular value $\sigma_k(V^T S)$')
        ax_r.set_title(r'(b) Subspace SVD $\sigma(V^T S)$')
        ax_r.legend(loc='upper right', framealpha=0.95, fontsize=15)
        ax_r.grid(True, linestyle=':', alpha=0.6)
        adjust_axis_limits(ax_r, idx_s, axis='x', is_log_scale=False, max_steps=1)
        adjust_axis_limits(ax_r, safe_s_proj[safe_s_proj > 0], axis='y', is_log_scale=True, max_steps=1)

    fig.tight_layout()
    if has_proj:
        fig.subplots_adjust(top=0.88, wspace=0.28)
    else:
        fig.subplots_adjust(top=0.88)

    if title:
        fig.suptitle(title, fontsize=18, y=0.98)

    if filename:
        fig_data = {'modal_energies': modal_energies, 'block_slices': block_slices, 's_projected': s_projected}
        save_figure(fig, filename, fig_data=fig_data)

    return fig, axes


def plot_structured_rb_diagnostics(rb_dyn, nosc_r, sv_dyn, sv_enrich, sv_Gr,
                                   cond_val, tol, modal_energies,
                                   sv_projected=None, rb_hamz=None, sv_hamz=None,
                                   has_hamz=False, data_folder=None,
                                   suffix="H", system_label=None):
    """
    Orchestrate and save both the structured RB component spectra plot
    and modal energy waterfall plot for Hamiltonian or Discrete Gradient reduced systems.

    Parameters
    ----------
    rb_dyn : np.ndarray
        Kinematic POD basis matrix.
    nosc_r : int
        Total reduced system degrees of freedom.
    sv_dyn : np.ndarray
        Singular values of kinematic snapshots.
    sv_enrich : np.ndarray
        Principal sines of constraint normal enrichment.
    sv_Gr : np.ndarray
        Singular values of reduced constraint Jacobian.
    cond_val : float
        Condition number of reduced constraint Jacobian.
    tol : float
        POD truncation tolerance.
    modal_energies : np.ndarray
        Modal snapshot projection energies across basis modes.
    sv_projected : np.ndarray, optional
        Singular values of projected snapshots.
    rb_hamz : np.ndarray, optional
        Force residual POD basis matrix (Hamiltonian).
    sv_hamz : np.ndarray, optional
        Force residual singular values (Hamiltonian).
    has_hamz : bool, optional
        Whether force residual enrichment is active.
    data_folder : str, optional
        Destination directory for output plots. Defaults to current directory.
    suffix : str, optional
        Filename suffix (e.g. 'H' or 'DG'). Default is 'H'.
    system_label : str, optional
        Display name in titles (e.g. 'Hamiltonian' or 'Discrete Gradient').
    """
    if data_folder is None:
        data_folder = "."
    if system_label is None:
        system_label = "Hamiltonian" if suffix == "H" else "Discrete Gradient"

    r_dyn = rb_dyn.shape[1] if rb_dyn is not None else 0
    r_hamz = rb_hamz.shape[1] if (has_hamz and rb_hamz is not None and rb_hamz.shape[1] > 0) else 0
    r_enrich = nosc_r - (1 + r_dyn + r_hamz)

    # Approach 1: Multi-Block Component Spectra Plot
    components_dict = {
        'sv_dyn': sv_dyn,
        'sv_hamz': sv_hamz if (has_hamz and r_hamz > 0) else None,
        'sv_enrich': sv_enrich,
        'sv_Gr': sv_Gr,
        'tol_dyn': tol,
        'tol_hamz': tol if (has_hamz and r_hamz > 0) else None,
        'cond_Gr': cond_val,
    }
    filename_comp = os.path.join(data_folder, f"sv_structured_components_{suffix}.pdf")
    plot_structured_rb_components(
        components_dict,
        filename=filename_comp,
        title=f"Structured reduced basis component spectra ({system_label})"
    )

    # Approach 2: Modal Snapshot Projection Waterfall Plot
    block_slices = {
        r'Anchor ($v_0$)': (0, 1),
        r'Kinematics ($V_{\mathrm{dyn}}$)': (1, 1 + r_dyn),
    }
    if r_hamz > 0:
        block_slices[r'Force ($V_{\nabla H}$)'] = (1 + r_dyn, 1 + r_dyn + r_hamz)
    if r_enrich > 0:
        block_slices[r'Constraints ($V_{\mathrm{enrich}}$)'] = (1 + r_dyn + r_hamz, nosc_r)

    filename_modal = os.path.join(data_folder, f"sv_structured_modal_energy_{suffix}.pdf")
    plot_structured_rb_modal_energy(
        modal_energies,
        block_slices,
        s_projected=sv_projected,
        filename=filename_modal,
        title=f"Structured reduced basis modal energy waterfall ({system_label})"
    )


def save_figure(fig, filename, fig_data=None):
    """
    Save a figure and its data to files (.pdf and .data.pickle).

    Parameters:
    fig (matplotlib figure): The figure to save.
    filename (str): The filename.
    fig_data (dict, optional): A dictionary containing the data used to generate the figure.
    """
    # Ensure the directory exists
    dir_name = os.path.dirname(filename)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    # Save the figure with the specified extension (.pdf)
    fig.savefig(filename)

    # Save the figure data (.data.pickle)
    if fig_data is not None:
        fig_path_base = os.path.splitext(filename)[0]
        with open(fig_path_base + '.data.pickle', 'wb') as f:
            pickle.dump(fig_data, f)

# %% edit the figure later
def edit_figure_later(filename=None):
    """
    Load a saved matplotlib figure pickle file, customize formatting/layout, and save as PDF.

    Parameters
    ----------
    filename : str, optional
        Path to the `.fig.pickle` file. If None, defaults to the sample figure path.

    Returns
    -------
    matplotlib.figure.Figure or None
        The modified figure object if loaded and saved, else None.
    """
    if filename is None:
        filename = "/home/bhattah/Documents/constrained_mechanics/data/2026-06-19/error_vs_basis_size.fig.pickle"

    if os.path.exists(filename):
        with open(filename, "rb") as file:
            figx = pickle.load(file)

            # Increase axis and legend text size for the loaded figure.
            for ax in figx.axes:
                ax.tick_params(axis='both', labelsize=24)
                ax.xaxis.label.set_size(26)
                ax.yaxis.label.set_size(26)
                ax.title.set_fontsize(26)

                # Adjust axis limits dynamically based on line and collection data
                xdata, ydata = [], []
                for line in ax.get_lines():
                    xd, yd = line.get_xdata(), line.get_ydata()
                    if xd is not None and len(xd) > 0: xdata.extend(np.array(xd).flatten())
                    if yd is not None and len(yd) > 0: ydata.extend(np.array(yd).flatten())
                is_x_log = (ax.get_xscale() == 'log')
                is_y_log = (ax.get_yscale() == 'log')
                if xdata:
                    x_pts = np.array(xdata)
                    if is_x_log: x_pts = x_pts[np.isfinite(x_pts) & (x_pts > 0)]
                    adjust_axis_limits(ax, x_pts, axis='x', is_log_scale=is_x_log, max_steps=5)
                if ydata:
                    y_pts = np.array(ydata)
                    if is_y_log: y_pts = y_pts[np.isfinite(y_pts) & (y_pts > 0)]
                    adjust_axis_limits(ax, y_pts, axis='y', is_log_scale=is_y_log, max_steps=5)
                
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
                except FileNotFoundError:
                    print(f"File not found when trying to remove existing PDF {pdf_path}.")
                except Exception as e:
                    print(f"Could not remove existing PDF {pdf_path}: {e}")
            figx.savefig(pdf_path, bbox_inches='tight', pad_inches=0.5, dpi=300)
            return figx
    else:
        print(f"Figure file not found: {filename}")
        return None

# Convenience aliases
edit_figure = edit_figure_later
edit_saved_figure = edit_figure_later


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

    adjust_axis_limits(ax, xx.flatten(), axis='x', is_log_scale=False)
    adjust_axis_limits(ax, yy.flatten(), axis='y', is_log_scale=False)
    adjust_axis_limits(ax, zz.flatten(), axis='z', is_log_scale=False, max_steps=2)

def plot_pareto(time_lapsed, errors_r, errors_hr, solver_names, dt_space, data_folder, **kwargs):
    """
    Generate Pareto plots (Error vs Time) for reduced and hyper-reduced models.
    
    Parameters:
    time_lapsed (list): List containing [full_time, reduced_percent, hyper_percent] arrays.
    errors_r (numpy array): Reduced model errors.
    errors_hr (numpy array): Hyper-reduced model errors.
    solver_names (list): List of solver names.
    dt_space (numpy array): Array of time step sizes.
    data_folder (str): Path to save the plot.
    """
    if errors_hr is None:
        errors_hr = kwargs.get('errors_dr', None)

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
    
    mean_err_hyper = np.nanmean(errors_hr, axis=2)
    std_err_hyper = np.nanstd(errors_hr, axis=2)
    
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

        ax.set_xlabel('Average wall time')
        if i == 0:
            ax.set_ylabel('Mean global error')
        ax.set_title(f'{solver_name}')
        ax.grid(True, which="both", ls="-", alpha=0.3)

        # Adjust log-scale x and y axis limits
        x_points = np.concatenate([
            avg_time_red[i].flatten(), 
            avg_time_hyper[i].flatten(), 
            avg_time_full[i].flatten()
        ])
        y_points = np.concatenate([
            mean_err_red[i].flatten(), 
            mean_err_hyper[i].flatten()
        ])
        x_points_valid = x_points[np.isfinite(x_points) & (x_points > 0)]
        y_points_valid = y_points[np.isfinite(y_points) & (y_points > 0)]
        adjust_axis_limits(ax, x_points_valid, axis='x', is_log_scale=True, max_steps=5)
        adjust_axis_limits(ax, y_points_valid, axis='y', is_log_scale=True, max_steps=5)
        
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
        
    filename = os.path.join(data_folder, "pareto_combined.pdf")
    fig.savefig(filename, bbox_inches='tight')
    print(f"Saved Pareto plot to {filename}")
    plt.close(fig)



def adjust_axis_limits(ax, data_points, axis='y', is_log_scale=False, max_steps=1, margin_ratio=0.03):
    """
    Adjust axis limits based on data range and current tick intervals.
    Stops expanding ticks as soon as they span the entire data range.
    Provides margins around 0 when data touches boundary so the full solid line is visible.
    """
    if data_points is None:
        return
    data_points = np.asarray(data_points)
    if data_points.size == 0:
        return

    data_min = np.nanmin(data_points)
    data_max = np.nanmax(data_points)

    if axis == 'x':
        get_ticks = ax.get_xticks
        set_lim = ax.set_xlim
        set_ticks = ax.set_xticks
    elif axis == 'y':
        get_ticks = ax.get_yticks
        set_lim = ax.set_ylim
        set_ticks = ax.set_yticks
    elif axis == 'z':
        get_ticks = ax.get_zticks
        set_lim = ax.set_zlim
        set_ticks = ax.set_zticks
    else:
        return

    # When the y-data range is between +- 1e-14, it should simply be bounded +- 1e-14 ticks
    if axis == 'y' and not is_log_scale and -1e-14 <= data_min and data_max <= 1e-14:
        y_span = 2e-14
        margin_y = margin_ratio * y_span
        set_lim([-1e-14 - margin_y, 1e-14 + margin_y])
        set_ticks([-1e-14, 0.0, 1e-14])
        return

    ax.relim()
    ax.autoscale_view()
    plt.draw()

    tick_values = get_ticks()
    if len(tick_values) < 2: return

    if is_log_scale:
        log_ticks = np.log10(tick_values[tick_values > 0])
        if len(log_ticks) < 2: return
        
        log_step = log_ticks[-1] - log_ticks[-2]
        if log_step <= 1e-12: return 
        
        log_data_max = np.log10(data_max)
        log_data_min = np.log10(data_min) if data_min > 0 else log_ticks[0]
        
        # Stop expanding ticks as soon as they span the data range
        ticks_above = log_ticks[log_ticks >= log_data_max]
        if len(ticks_above) > 0:
            new_max = ticks_above[0]
        else:
            new_max = log_ticks[-1]
            steps = 0
            while new_max < log_data_max and steps < max_steps:
                new_max += log_step
                steps += 1
        
        ticks_below = log_ticks[log_ticks <= log_data_min]
        if len(ticks_below) > 0:
            new_min = ticks_below[-1]
        else:
            new_min = log_ticks[0]
            steps = 0
            while new_min > log_data_min and steps < max_steps:
                new_min -= log_step
                steps += 1
        
        if new_min >= new_max:
            new_min -= log_step
            new_max += log_step
        
        log_span = new_max - new_min
        log_margin = margin_ratio * log_span
        set_lim([10**(new_min - log_margin), 10**(new_max + log_margin)])
        if axis == 'y':
            ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
        elif axis == 'x':
            ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
    else:
        step = tick_values[1] - tick_values[0]
        if step <= 0 or not np.isfinite(step): return
        
        # Expand ticks as soon as they span the data range
        ticks_above = tick_values[tick_values >= data_max]
        if len(ticks_above) > 0:
            new_max = ticks_above[0]
        else:
            new_max = tick_values[-1]
            while new_max < data_max:
                new_max += step
            
        ticks_below = tick_values[tick_values <= data_min]
        if len(ticks_below) > 0:
            new_min = ticks_below[-1]
        else:
            new_min = tick_values[0]
            while new_min > data_min:
                new_min -= step
        
        if new_min >= new_max:
            new_min = max(0, new_min - step) if data_min >= 0 else new_min - step
            new_max += step
        
        if axis == 'x' and any(keyword in ax.get_xlabel().lower() for keyword in ['dimension', 'index', 'mode', 'size']):
            new_min = max(0, new_min)

        n_steps = int(round((new_max - new_min) / step))
        if 1 <= n_steps <= 12:
            actual_ticks = np.linspace(new_min, new_max, n_steps + 1)
        else:
            actual_ticks = tick_values[(tick_values >= new_min - 1e-15*abs(step)) & (tick_values <= new_max + 1e-15*abs(step))]

        span = new_max - new_min
        margin = margin_ratio * span

        lim_min = new_min - margin
        lim_max = new_max + margin

        set_lim([lim_min, lim_max])
        if len(actual_ticks) >= 2:
            set_ticks(actual_ticks)



# Convenience alias
adjust_axis = adjust_axis_limits


def plot_error_vs_basis_size(basis_sizes, hr_basis_sizes, errors_r, errors_hr, solver_names, pod_tols, data_folder, **kwargs):
    """
    Plot ROM and HROM errors as a function of the reduced basis size.
    Also plots speedup factors and convergence orders as box plots against dt.
    Markers correspond to pod_tol values.
    """
    if errors_hr is None:
        errors_hr = kwargs.get('errors_dr', None)

    # Extract study data from kwargs
    dt_space = kwargs.get('dt_space', [])
    full_times_raw = kwargs.get('full_times_raw', {})

    fig, axes_grid = plt.subplots(2, 2, figsize=(22, 18.6))
    ax_err_r, ax_err_hr = axes_grid[0]
    ax_speed_r, ax_speed_hr = axes_grid[1]
    # ax_conv_r, ax_conv_hr = axes_grid[2]
    
    col_axes = [ax_err_r, ax_err_hr]

    # Define markers for pod_tols
    tol_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '*']
    marker_map = {tol: tol_markers[i % len(tol_markers)] for i, tol in enumerate(pod_tols)}

    # Define colors for solvers
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

        for i, name in enumerate(solver_names):
            color = solver_colors[i % len(solver_colors)]
            x_sizes_raw = basis_sizes[i] if ax_idx == 0 else hr_basis_sizes[i]
            err_mats = errors_r[i] if ax_idx == 0 else errors_hr[i]

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
    for ax_idx, (ax, data_type) in enumerate(zip([ax_speed_r, ax_speed_hr], ['times_r_raw', 'times_hr_raw'])):
        configure_axis(ax)
        ax.tick_params(axis='both', labelsize=24)
        ax.set_ylabel('Mean speedup factor' if ax_idx == 0 else '')
        ax.set_xlabel(r'Time step-size $\Delta t$')
        ax.xaxis.label.set_size(26)
        ax.yaxis.label.set_size(26)
        ax.title.set_fontsize(26)
        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        
        times_raw_list = kwargs.get(data_type, kwargs.get('times_dr_raw', []))
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
    legend2 = fig.legend(handles=combined_tol_handles, title="POD tolerance", loc='upper left', bbox_to_anchor=(0.92, 0.82), bbox_transform=fig.transFigure, fontsize=20, title_fontsize=22, frameon=True)
    fig.add_artist(legend2)

    # Increase right margin so legends don't overlap the subplots
    plt.subplots_adjust(right=0.70, hspace=0.4, wspace=0.25)


    filename = os.path.join(data_folder, "error_vs_basis_size.pdf")
    save_figure(fig, filename)
    print(f"Saved Error vs. Basis Size plot to {filename}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Metric descriptors for plot_prediction_results
# ---------------------------------------------------------------------------
# Each entry: (key_suffix, marker_style, legend_label, markersize)
_PLOT_METRICS = [
    ('sym',       'o', r'$\Delta Sp$', 9),
    ('energy',    's', r'$\Delta H$',  8),
    ('phase',     '^', r'$\Delta \mathcal{S}$', 8),
    ('lin_mom',   'D', r'$\Delta L$',  8),
    ('ang_mom',   'v', r'$\Delta J$',  8),
]


def _plot_metric_panel(ax, metric_data, i, j, marker, color, ms, all_y_values, x_pos=None):
    """Plot a single metric marker on *ax* if *metric_data* is valid."""
    if x_pos is None:
        x_pos = i
    if metric_data is not None and len(metric_data) > i and len(metric_data[i]) > j:
        arr = metric_data[i][j]
        if isinstance(arr, np.ndarray) and arr.size > 0:
            val = np.nanmean(np.nanmean(arr, axis=1))
            val_plot = max(abs(val), 1e-16) if np.isfinite(val) else val
            if np.isfinite(val_plot):
                ax.semilogy([x_pos], [val_plot], marker=marker, color=color,
                            linestyle='None', markersize=ms,
                            markerfacecolor=color, markeredgecolor=color, zorder=3)
                all_y_values.append(val_plot)
                return True
    return False


def _plot_speedup_panel(ax, times_raw, i, j, marker, color, avg_full, all_y_values, x_pos=None, width=0.10):
    """Plot speedup boxplot and mean marker on *ax* if timing data is valid."""
    if x_pos is None:
        x_pos = i
    if len(times_raw) > i and len(times_raw[i]) > j:
        t_arr = times_raw[i][j]
        if isinstance(t_arr, np.ndarray) and t_arr.size > 0 and avg_full is not None:
            if getattr(avg_full, 'ndim', 0) > 0 and t_arr.ndim > 1:
                spd_arr = (avg_full[:, None] / t_arr).flatten()
            else:
                spd_arr = (np.nanmean(avg_full) / t_arr).flatten()
            spd_clean = spd_arr[np.isfinite(spd_arr)]
            if len(spd_clean) > 0:
                ax.boxplot([spd_clean], positions=[x_pos], widths=width,
                           patch_artist=True, showmeans=True,
                           meanprops=dict(marker='o', markerfacecolor=color, markeredgecolor='black', markersize=6, zorder=4),
                           medianprops=dict(color='black', linewidth=1.5),
                           boxprops=dict(facecolor=color, alpha=0.25, edgecolor=color, linewidth=1.5),
                           whiskerprops=dict(color=color, linewidth=1.2),
                           capprops=dict(color=color, linewidth=1.2),
                           flierprops=dict(marker='.', markerfacecolor=color, markeredgecolor=color, markersize=4),
                           zorder=3)
                rng = np.random.RandomState(42 + int(abs(x_pos) * 100))
                jit = rng.uniform(-width * 0.35, width * 0.35, size=len(spd_clean))
                ax.scatter(x_pos + jit, spd_clean, color=color, alpha=0.6, s=18, edgecolors='none', zorder=5)
                all_y_values.extend(spd_clean.tolist())
                return float(np.mean(spd_clean))
    return None


def plot_prediction_results(solver_names, pod_tols=None, dt_space=None, data_folder=None,
                             full_times_raw=None, times_r_raw=None, times_hr_raw=None,
                             sym_error_f=None, sym_error_r=None, sym_error_hr=None,
                             energy_error_f=None, energy_error_r=None, energy_error_hr=None,
                             phase_error_f=None, phase_error_r=None, phase_error_hr=None,
                             lin_mom_error_f=None, lin_mom_error_r=None, lin_mom_error_hr=None,
                             ang_mom_error_f=None, ang_mom_error_r=None, ang_mom_error_hr=None,
                             **kwargs):
    """
    Plot prediction-mode metrics for reduced and hyper-reduced models.

    Parameters
    ----------
    solver_names : list of str
        Names of solver classes.
    pod_tols : list of float
        POD tolerances used in the sweep.
    dt_space : numpy array
        Time step sizes.
    data_folder : str
        Directory for saving the plot.
    full_times_raw : dict
        Mapping solver name → full-order timing array of shape (n_dt, n_train_params).
    times_r_raw : list of list of numpy arrays
        Reduced timing arrays per solver and tolerance.
    times_hr_raw : list of list of numpy arrays
        Hyper-reduced timing arrays per solver and tolerance.
    sym_error_f, sym_error_r, sym_error_hr : list of list of numpy arrays, optional
    energy_error_f, energy_error_r, energy_error_hr : list of list of numpy arrays, optional
    phase_error_f, phase_error_r, phase_error_hr : list of list of numpy arrays, optional
    lin_mom_error_f, lin_mom_error_r, lin_mom_error_hr : list of list of numpy arrays, optional
    ang_mom_error_f, ang_mom_error_r, ang_mom_error_hr : list of list of numpy arrays, optional
    """
    if isinstance(solver_names, dict):
        d_pred = solver_names
        data_folder = data_folder or d_pred.get('data_folder', '.')
        full_times_raw = d_pred.get('full_times_raw', full_times_raw)
        times_r_raw = d_pred.get('times_r_raw', times_r_raw)
        times_hr_raw = d_pred.get('times_hr_raw', times_hr_raw)
        pod_tols = d_pred.get('pod_tols', pod_tols)
        dt_space = d_pred.get('dt_space', dt_space)
        sym_error_f = d_pred.get('sym_error_f', sym_error_f)
        sym_error_r = d_pred.get('sym_error_r', sym_error_r)
        sym_error_hr = d_pred.get('sym_error_hr', sym_error_hr)
        energy_error_f = d_pred.get('energy_error_f', energy_error_f)
        energy_error_r = d_pred.get('energy_error_r', energy_error_r)
        energy_error_hr = d_pred.get('energy_error_hr', energy_error_hr)
        phase_error_f = d_pred.get('phase_error_f', phase_error_f)
        phase_error_r = d_pred.get('phase_error_r', phase_error_r)
        phase_error_hr = d_pred.get('phase_error_hr', phase_error_hr)
        lin_mom_error_f = d_pred.get('lin_mom_error_f', lin_mom_error_f)
        lin_mom_error_r = d_pred.get('lin_mom_error_r', lin_mom_error_r)
        lin_mom_error_hr = d_pred.get('lin_mom_error_hr', lin_mom_error_hr)
        ang_mom_error_f = d_pred.get('ang_mom_error_f', ang_mom_error_f)
        ang_mom_error_r = d_pred.get('ang_mom_error_r', ang_mom_error_r)
        ang_mom_error_hr = d_pred.get('ang_mom_error_hr', ang_mom_error_hr)
        if 'metrics' in d_pred and not full_times_raw:
            m = d_pred['metrics']
            solver_names_list = d_pred['solver_names']
            full_times_raw = {s: np.array([[1.0]]) for s in solver_names_list}
            sp_r = m.get('ROM', {}).get('speedup', 1.0)
            sp_hr = m.get('HROM', {}).get('speedup', 1.0)
            times_r_raw = [[np.array([[1.0 / sp_r]])]]
            times_hr_raw = [[np.array([[1.0 / sp_hr]])]]
            sym_error_f = [[np.array([[m['FOM'].get('sym_error', 1e-16)]])]]
            sym_error_r = [[np.array([[m['ROM'].get('sym_error', 1e-16)]])]]
            sym_error_hr = [[np.array([[m['HROM'].get('sym_error', 1e-16)]])]]
            energy_error_f = [[np.array([[m['FOM'].get('energy_error', 1e-16)]])]]
            energy_error_r = [[np.array([[m['ROM'].get('energy_error', 1e-16)]])]]
            energy_error_hr = [[np.array([[m['HROM'].get('energy_error', 1e-16)]])]]
            phase_error_f = [[np.array([[m['FOM'].get('phase_error', 1e-16)]])]]
            phase_error_r = [[np.array([[m['ROM'].get('phase_error', 1e-16)]])]]
            phase_error_hr = [[np.array([[m['HROM'].get('phase_error', 1e-16)]])]]
            lin_mom_error_f = [[np.array([[m['FOM'].get('lin_mom_error', 1e-16)]])]]
            lin_mom_error_r = [[np.array([[m['ROM'].get('lin_mom_error', 1e-16)]])]]
            lin_mom_error_hr = [[np.array([[m['HROM'].get('lin_mom_error', 1e-16)]])]]
            ang_mom_error_f = [[np.array([[m['FOM'].get('ang_mom_error', 1e-16)]])]]
            ang_mom_error_r = [[np.array([[m['ROM'].get('ang_mom_error', 1e-16)]])]]
            ang_mom_error_hr = [[np.array([[m['HROM'].get('ang_mom_error', 1e-16)]])]]
        solver_names = d_pred['solver_names']

    if times_hr_raw is None:
        times_hr_raw = kwargs.get('times_dr_raw', [])
    if sym_error_hr is None:
        sym_error_hr = kwargs.get('sym_error_dr')
    if energy_error_hr is None:
        energy_error_hr = kwargs.get('energy_error_dr')
    if phase_error_hr is None:
        phase_error_hr = kwargs.get('phase_error_dr')
    if lin_mom_error_hr is None:
        lin_mom_error_hr = kwargs.get('lin_mom_error_dr')
    if ang_mom_error_hr is None:
        ang_mom_error_hr = kwargs.get('ang_mom_error_dr')

    # --- Reorder solvers so Model 1 (StormerVerlet) is on the left (x=0) ---
    # The input order may have Model 2 first; we want a consistent left-to-right
    # ordering: Model 1 (StormerVerlet) at x=0, Model 2 (DiscreteGradient) at x=1.
    def _model_rank(name):
        if 'StormerVerlet' in name:
            return 0
        if 'DiscreteGradient' in name:
            return 1
        return 2  # any other solver goes to the right

    order = sorted(range(len(solver_names)), key=lambda i: _model_rank(solver_names[i]))
    solver_names = [solver_names[i] for i in order]

    # Reorder all per-solver data lists to match the new solver order
    def _reorder(data_list):
        if data_list is None:
            return None
        return [data_list[i] for i in order]

    times_r_raw = _reorder(times_r_raw)
    times_hr_raw = _reorder(times_hr_raw)

    # Group all error metrics by model level for data-driven plotting
    _error_groups = {
        'f':  {'ax': None, 'data': {}, 'y_vals': []},
        'r':  {'ax': None, 'data': {}, 'y_vals': []},
        'hr': {'ax': None, 'data': {}, 'y_vals': []},
    }
    # Map keyword suffixes to the actual data passed in (reordered)
    _metric_data = {
        'sym':     {'f': _reorder(sym_error_f),     'r': _reorder(sym_error_r),     'hr': _reorder(sym_error_hr)},
        'energy':  {'f': _reorder(energy_error_f),  'r': _reorder(energy_error_r),  'hr': _reorder(energy_error_hr)},
        'phase':   {'f': _reorder(phase_error_f),   'r': _reorder(phase_error_r),   'hr': _reorder(phase_error_hr)},
        'lin_mom': {'f': _reorder(lin_mom_error_f), 'r': _reorder(lin_mom_error_r), 'hr': _reorder(lin_mom_error_hr)},
        'ang_mom': {'f': _reorder(ang_mom_error_f), 'r': _reorder(ang_mom_error_r), 'hr': _reorder(ang_mom_error_hr)},
    }

    fig, (ax_err, ax_spd) = plt.subplots(1, 2, figsize=(18, 7.5))

    tier_keys = ['f', 'r', 'hr']
    tier_labels = {'f': 'FOM', 'r': 'ROM', 'hr': 'HROM'}
    tier_offsets = {'f': -0.22, 'r': 0.0, 'hr': 0.22}

    # Group all error metrics by model level for data-driven plotting
    _error_groups = {
        'f':  {'data': {}},
        'r':  {'data': {}},
        'hr': {'data': {}},
    }
    for suffix, levels in _metric_data.items():
        for level in tier_keys:
            _error_groups[level]['data'][suffix] = levels[level]

    metric_marker_map = {suffix: (m, ms) for suffix, m, _, ms in _PLOT_METRICS}
    solver_colors = OKABE_ITO_PALETTE

    all_y_values_err = []
    all_y_values_spd = [1.0]
    plotted_suffixes = set()

    for i, name in enumerate(solver_names):
        color = solver_colors[i % len(solver_colors)]
        full_time = full_times_raw.get(name)
        avg_full = np.nanmean(full_time, axis=1) if isinstance(full_time, np.ndarray) else None

        # Draw vertical platform spines for all three tiers
        for level in tier_keys:
            x_pos = i + tier_offsets[level]
            ax_err.axvline(x=x_pos, color='0.88', linestyle='-', linewidth=1.5, zorder=1)
            ax_spd.axvline(x=x_pos, color='0.88', linestyle='-', linewidth=1.5, zorder=1)

        for j, tol in enumerate(pod_tols):
            # --- Error markers on vertical platforms ---
            for level in tier_keys:
                x_pos = i + tier_offsets[level]
                for suffix, (mkr, ms) in metric_marker_map.items():
                    if _plot_metric_panel(ax_err, _error_groups[level]['data'].get(suffix),
                                          i, j, mkr, color, ms, all_y_values_err, x_pos=x_pos):
                        plotted_suffixes.add(suffix)

            # --- Speedup boxplots and mean markers on vertical platforms ---
            x_fom = i + tier_offsets['f']
            x_rom = i + tier_offsets['r']
            x_hrom = i + tier_offsets['hr']

            n_tols = len(pod_tols)
            tol_offset = (j - (n_tols - 1) / 2) * (0.12 / max(1, n_tols)) if n_tols > 1 else 0.0
            box_width = 0.10 / max(1, n_tols) if n_tols > 1 else 0.10

            pos_r = x_rom + tol_offset
            pos_hr = x_hrom + tol_offset

            # FOM reference boxplot & mean
            if j == 0:
                if isinstance(full_time, np.ndarray) and full_time.size > 0 and avg_full is not None:
                    if getattr(avg_full, 'ndim', 0) > 0 and full_time.ndim > 1:
                        spd_f_arr = (avg_full[:, None] / full_time).flatten()
                    else:
                        spd_f_arr = (np.nanmean(avg_full) / full_time).flatten()
                    spd_f_clean = spd_f_arr[np.isfinite(spd_f_arr)]
                else:
                    spd_f_clean = np.array([1.0])

                if len(spd_f_clean) > 0:
                    ax_spd.boxplot([spd_f_clean], positions=[x_fom], widths=0.10,
                                   patch_artist=True, showmeans=True,
                                   meanprops=dict(marker='o', markerfacecolor=color, markeredgecolor='black', markersize=6, zorder=4),
                                   medianprops=dict(color='black', linewidth=1.5),
                                   boxprops=dict(facecolor=color, alpha=0.25, edgecolor=color, linewidth=1.5),
                                   whiskerprops=dict(color=color, linewidth=1.2),
                                   capprops=dict(color=color, linewidth=1.2),
                                   flierprops=dict(marker='.', markerfacecolor=color, markeredgecolor=color, markersize=4),
                                   zorder=3)
                    all_y_values_spd.extend(spd_f_clean.tolist())
                    rng_f = np.random.RandomState(42)
                    jit_f = rng_f.uniform(-0.10 * 0.35, 0.10 * 0.35, size=len(spd_f_clean))
                    ax_spd.scatter(x_fom + jit_f, spd_f_clean, color=color, alpha=0.6, s=18, edgecolors='none', zorder=5)
                    sp_f_mean = float(np.mean(spd_f_clean))
                else:
                    sp_f_mean = 1.0

            sp_r = _plot_speedup_panel(ax_spd, times_r_raw, i, j, 'o', color,
                                       avg_full, all_y_values_spd, x_pos=pos_r, width=box_width)
            sp_hr = _plot_speedup_panel(ax_spd, times_hr_raw, i, j, 'o', color,
                                        avg_full, all_y_values_spd, x_pos=pos_hr, width=box_width)

            # Connect mean speedups with dashed line
            pts_x = [x_fom]
            pts_y = [sp_f_mean]
            if sp_r is not None and np.isfinite(sp_r):
                pts_x.append(pos_r)
                pts_y.append(sp_r)
            if sp_hr is not None and np.isfinite(sp_hr):
                pts_x.append(pos_hr)
                pts_y.append(sp_hr)
            if len(pts_x) > 1:
                ax_spd.plot(pts_x, pts_y, color=color, linestyle='--', linewidth=1.5, alpha=0.7, zorder=2)

    # Filter legend handles strictly to metrics that were actually plotted
    error_legend_handles = [
        Line2D([0], [0], marker=m, color='black', label=label,
               markersize=ms, linestyle='None')
        for sfx, m, label, ms in _PLOT_METRICS
        if sfx in plotted_suffixes
    ]

    n_solvers = len(solver_names)
    for ax in [ax_err, ax_spd]:
        configure_axis(ax)
        ax.tick_params(axis='both', labelsize=12)
        ax.set_xlim(-0.55, max(0.55, n_solvers - 0.45))

        # Build sub-ticks for FOM, ROM, HROM
        sub_ticks = []
        sub_labels = []
        for i_s in range(n_solvers):
            for level in tier_keys:
                sub_ticks.append(i_s + tier_offsets[level])
                sub_labels.append(tier_labels[level])
        ax.set_xticks(sub_ticks)
        ax.set_xticklabels(sub_labels, fontsize=12)

        # Place model labels centered beneath each group
        for i_s, sn in enumerate(solver_names):
            if 'StormerVerlet' in sn:
                m_label = 'Model 1\n(Stormer-Verlet)' if n_solvers > 1 else 'Stormer-Verlet'
            elif 'DiscreteGradient' in sn:
                m_label = 'Model 2\n(Discrete Gradient)' if n_solvers > 1 else 'Discrete Gradient'
            else:
                m_label = sn
            ax.text(i_s, -0.15, m_label, ha='center', va='top', fontsize=14, fontweight='bold',
                    transform=ax.get_xaxis_transform())

    ax_err.set_title('Mean invariant error magnitude', fontsize=16, pad=12)
    ax_err.set_ylabel('Mean error magnitude', fontsize=14)

    ax_spd.set_title('Speedup factor', fontsize=16, pad=12)
    ax_spd.set_ylabel('Speedup factor', fontsize=14)
    ax_spd.axhline(1.0, color='black', linestyle=':', linewidth=1.2, alpha=0.7, zorder=1)

    iqr_label = r'IQR (25\%--75\%)' if plt.rcParams.get('text.usetex', False) else 'IQR (25%–75%)'
    speedup_legend_handles = [
        Line2D([0], [0], marker='o', color='black', label='Mean', markersize=6, linestyle='--'),
        Line2D([0], [0], color='black', label='Median', linewidth=1.5),
        Patch(facecolor='0.7', edgecolor='0.4', label=iqr_label, alpha=0.5),
        Line2D([0], [0], marker='o', color='0.5', label='Individual runs', markersize=4, linestyle='None', alpha=0.7),
    ]
    ax_spd.legend(handles=speedup_legend_handles, loc='upper left', fontsize=11, frameon=True)

    adjust_axis_limits(ax_err, all_y_values_err, axis='y', is_log_scale=True, max_steps=10)
    adjust_axis_limits(ax_spd, all_y_values_spd, axis='y', is_log_scale=False, max_steps=10)

    ax_err.legend(handles=error_legend_handles, title='Invariant error', loc='upper left',
                  fontsize=11, title_fontsize=12, frameon=True)

    plt.tight_layout()
    target_filename = kwargs.get('filename', None)
    if target_filename is None:
        filename = os.path.join(data_folder, 'prediction_results.pdf')
    elif os.path.isabs(target_filename) or (data_folder is None) or target_filename.startswith(data_folder):
        filename = target_filename
    else:
        filename = os.path.join(data_folder, target_filename)
    fig_data = {
        'solver_names': solver_names,
        'pod_tols': pod_tols,
        'dt_space': dt_space,
        'full_times_raw': full_times_raw,
        'times_r_raw': times_r_raw,
        'times_hr_raw': times_hr_raw,
        'sym_error_f': sym_error_f,
        'sym_error_r': sym_error_r,
        'sym_error_hr': sym_error_hr,
        'energy_error_f': energy_error_f,
        'energy_error_r': energy_error_r,
        'energy_error_hr': energy_error_hr,
        'phase_error_f': phase_error_f,
        'phase_error_r': phase_error_r,
        'phase_error_hr': phase_error_hr,
        'lin_mom_error_f': lin_mom_error_f,
        'lin_mom_error_r': lin_mom_error_r,
        'lin_mom_error_hr': lin_mom_error_hr,
        'ang_mom_error_f': ang_mom_error_f,
        'ang_mom_error_r': ang_mom_error_r,
        'ang_mom_error_hr': ang_mom_error_hr,
    }
    save_figure(fig, filename, fig_data=fig_data)
    print(f"Saved Prediction Results plot to {filename}")
    plt.close(fig)


def plot_prediction_sanity_check(s_f, s_r=None, s_hr=None, plot_path=None,
                                 coord_series=None,
                                 coord_label=None,
                                 coord_ylabel=None,
                                 system_label=None,
                                 target_idx=None,
                                 target_omega2=None,
                                 cls_name=None,
                                 dt=None,
                                 **kwargs):
    """
    Generate diagnostic 3x2 comparison plots for prediction sanity check against ground-truth FOM:
    (a) Max pointwise state error over time: ||y_FOM(t) - y_model(t)||_inf
    (b) Phase space error (constraint norm) |Delta S(t)|
    (c) Energy invariant drift |Delta H(t)|
    (d) Symplectic invariant error |Delta Sp(t)|
    (e) Linear momentum error |Delta L(t)|
    (f) Angular momentum error |Delta J(t)|

    Parameters
    ----------
    s_f : Solver instance or dict
        Ground-truth Full Order Model solver, or pre-extracted fig_data dictionary.
    s_r : Solver instance, optional
        Reduced Order Model (ROM) solver. Not needed if s_f is a dict.
    s_hr : Solver instance, optional
        Hyper-Reduced Order Model (HROM) solver. Not needed if s_f is a dict.
    plot_path : str, optional
        File path where the plot should be saved (.pdf).
    coord_series, coord_label, coord_ylabel : optional
        Deprecated parameters preserved for backward compatibility.
    system_label : str, optional
        Optional prefix for figure suptitle (e.g. 'Elastica').
    target_idx : int, optional
        Index of test parameter in Omega2_space_test.
    target_omega2 : np.ndarray or float, optional
        Test parameter value.
    cls_name : str, optional
        Name of solver class. If None, inferred.
    dt : float, optional
        Time step. If None, inferred.

    Returns
    -------
    fig, axes : tuple
        Matplotlib figure and axes array.
    """
    if isinstance(s_f, dict):
        fig_data = s_f
        t = fig_data['t']
        err_r_t = fig_data['err_r_t']
        err_hr_t = fig_data['err_hr_t']
        eng_error_f = fig_data.get('eng_error_fom')
        eng_error_r = fig_data.get('eng_error_rom')
        eng_error_hr = fig_data.get('eng_error_hrom')
        sym_error_f = fig_data.get('sym_error_fom')
        sym_error_r = fig_data.get('sym_error_rom')
        sym_error_hr = fig_data.get('sym_error_hrom')
        g_norm_f = fig_data.get('g_norm_fom', fig_data.get('phase_error_fom'))
        g_norm_r = fig_data.get('g_norm_rom', fig_data.get('phase_error_rom'))
        g_norm_hr = fig_data.get('g_norm_hrom', fig_data.get('phase_error_hrom'))
        lin_mom_f = fig_data.get('lin_mom_error_fom', fig_data.get('lin_momentum_err_fom'))
        lin_mom_r = fig_data.get('lin_mom_error_rom', fig_data.get('lin_momentum_err_rom'))
        lin_mom_hr = fig_data.get('lin_mom_error_hrom', fig_data.get('lin_momentum_err_hrom'))
        ang_mom_f = fig_data.get('ang_mom_error_fom', fig_data.get('angular_momentum_err_fom'))
        ang_mom_r = fig_data.get('ang_mom_error_rom', fig_data.get('angular_momentum_err_rom'))
        ang_mom_hr = fig_data.get('ang_mom_error_hrom', fig_data.get('angular_momentum_err_hrom'))
        cls_name = cls_name or fig_data.get('cls_name', '')
        dt = dt or fig_data.get('dt', None)
        target_idx = target_idx if target_idx is not None else fig_data.get('target_idx', None)
        target_omega2 = target_omega2 if target_omega2 is not None else fig_data.get('target_omega2', None)
    else:
        if cls_name is None:
            cls_name = getattr(s_f, '__name__', s_f.__class__.__name__)
        if dt is None:
            dt = getattr(s_f, 'dt', None)

        t = s_f.t_points
        err_r_t = np.amax(np.abs(s_f.y - s_r.y), axis=1)
        err_hr_t = np.amax(np.abs(s_f.y - s_hr.y), axis=1)

        eng_error_f = getattr(s_f, 'eng_error', None)
        eng_error_r = getattr(s_r, 'eng_error', None)
        eng_error_hr = getattr(s_hr, 'eng_error', None)
        sym_error_f = getattr(s_f, 'sym_error', None)
        sym_error_r = getattr(s_r, 'sym_error', None)
        sym_error_hr = getattr(s_hr, 'sym_error', None)
        g_norm_f = getattr(s_f, 'g_norm', None)
        g_norm_r = getattr(s_r, 'g_norm', None)
        g_norm_hr = getattr(s_hr, 'g_norm', None)
        lin_mom_f = getattr(s_f, 'lin_momentum_err', None)
        lin_mom_r = getattr(s_r, 'lin_momentum_err', None)
        lin_mom_hr = getattr(s_hr, 'lin_momentum_err', None)
        ang_mom_f = getattr(s_f, 'angular_momentum_err', None)
        ang_mom_r = getattr(s_r, 'angular_momentum_err', None)
        ang_mom_hr = getattr(s_hr, 'angular_momentum_err', None)

        fig_data = {
            't': t,
            'err_r_t': err_r_t,
            'err_hr_t': err_hr_t,
            'eng_error_fom': eng_error_f,
            'eng_error_rom': eng_error_r,
            'eng_error_hrom': eng_error_hr,
            'sym_error_fom': sym_error_f,
            'sym_error_rom': sym_error_r,
            'sym_error_hrom': sym_error_hr,
            'g_norm_fom': g_norm_f,
            'g_norm_rom': g_norm_r,
            'g_norm_hrom': g_norm_hr,
            'lin_mom_error_fom': lin_mom_f,
            'lin_mom_error_rom': lin_mom_r,
            'lin_mom_error_hrom': lin_mom_hr,
            'ang_mom_error_fom': ang_mom_f,
            'ang_mom_error_rom': ang_mom_r,
            'ang_mom_error_hrom': ang_mom_hr,
            'dt': dt,
            'target_idx': target_idx,
            'target_omega2': target_omega2,
            'cls_name': cls_name,
        }
        if hasattr(s_f, 'y') and s_f.y is not None:
            fig_data['q0_fom'] = s_f.y[:, 0]
        if s_r is not None and hasattr(s_r, 'y') and s_r.y is not None:
            fig_data['q0_rom'] = s_r.y[:, 0]
        if s_hr is not None and hasattr(s_hr, 'y') and s_hr.y is not None:
            fig_data['q0_hrom'] = s_hr.y[:, 0]

    if shutil.which('latex') and not plt.rcParams.get('text.usetex', False):
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

    has_momentum = any(d is not None and np.asarray(d).size > 0 and np.any(np.isfinite(d))
                       for d in [lin_mom_f, lin_mom_r, lin_mom_hr, ang_mom_f, ang_mom_r, ang_mom_hr])

    if has_momentum:
        fig, axes = plt.subplots(3, 2, figsize=(14, 13))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))

    for ax in axes.flatten():
        configure_axis(ax)
        ax.grid(True, which="both", linestyle=':', alpha=0.6)

    # (a) Pointwise Trajectory Error over time (log scale)
    axes[0, 0].semilogy(t, np.maximum(err_r_t, 1e-16), label='ROM', color=OKABE_ITO_PALETTE[0], ls='--', lw=1.8)
    axes[0, 0].semilogy(t, np.maximum(err_hr_t, 1e-16), label='HROM', color=OKABE_ITO_PALETTE[1], ls='-.', lw=1.8)
    axes[0, 0].set_title(r"Max state error $\|y_{\mathrm{FOM}}(t) - y_{\mathrm{model}}(t)\|_\infty$")
    axes[0, 0].set_xlabel("Time")
    adjust_axis_limits(axes[0, 0], t, axis='x', is_log_scale=False)
    y_err = np.concatenate([err_r_t, err_hr_t])
    y_err_pos = y_err[y_err > 0]
    if len(y_err_pos) > 0:
        adjust_axis_limits(axes[0, 0], y_err_pos, axis='y', is_log_scale=True, max_steps=1)

    def _plot_metric_panel(ax, title, f_data, r_data, hr_data):
        has_data = False
        pts = []
        if f_data is not None:
            y_f = np.abs(np.asarray(f_data, dtype=float)) + 1e-16
            ax.semilogy(t, y_f, label='FOM', color='black', ls='-', lw=1.6)
            pts.append(y_f)
            has_data = True
        if r_data is not None:
            y_r = np.abs(np.asarray(r_data, dtype=float)) + 1e-16
            ax.semilogy(t, y_r, label='ROM', color=OKABE_ITO_PALETTE[0], ls='--', lw=1.6)
            pts.append(y_r)
            has_data = True
        if hr_data is not None:
            y_hr = np.abs(np.asarray(hr_data, dtype=float)) + 1e-16
            ax.semilogy(t, y_hr, label='HROM', color=OKABE_ITO_PALETTE[1], ls='-.', lw=1.6)
            pts.append(y_hr)
            has_data = True
        if has_data:
            ax.set_title(title)
            ax.set_xlabel("Time")
            adjust_axis_limits(ax, t, axis='x', is_log_scale=False)
            all_pts = np.concatenate(pts)
            all_pos = all_pts[all_pts > 0]
            if len(all_pos) > 0:
                adjust_axis_limits(ax, all_pos, axis='y', is_log_scale=True, max_steps=1)
        else:
            ax.text(0.5, 0.5, f"{title} not available", ha='center', va='center', transform=ax.transAxes)

    # (b) Phase space error (constraint norm)
    _plot_metric_panel(axes[0, 1], r"Phase space error $|\Delta \mathcal{S}(t)|$", g_norm_f, g_norm_r, g_norm_hr)

    # (c) Energy Invariant Drift
    _plot_metric_panel(axes[1, 0], r"Energy drift $|\Delta H(t)|$", eng_error_f, eng_error_r, eng_error_hr)

    # (d) Symplectic Invariant Error
    _plot_metric_panel(axes[1, 1], r"Symplectic error $|\Delta \mathrm{Sp}(t)|$", sym_error_f, sym_error_r, sym_error_hr)

    # (e) Linear & (f) Angular Momentum Errors (only for systems possessing momentum invariants)
    if has_momentum:
        _plot_metric_panel(axes[2, 0], r"Linear momentum error $|\Delta L(t)|$", lin_mom_f, lin_mom_r, lin_mom_hr)
        _plot_metric_panel(axes[2, 1], r"Angular momentum error $|\Delta J(t)|$", ang_mom_f, ang_mom_r, ang_mom_hr)

    title_prefix = f"{system_label} " if system_label else ""
    dt_str = f", $\\Delta t={dt:.4f}$" if dt is not None else ""
    idx_str = f", $\\Omega^2$ (test[{target_idx}])" if target_idx is not None else ""
    safe_cls_name = cls_name.replace('_', r'\_') if cls_name else ""
    safe_prefix = title_prefix.replace('_', r'\_')
    fig.suptitle(f"Sanity check: {safe_prefix}{safe_cls_name}{dt_str}{idx_str}", fontsize=20)

    # Central legend at the bottom for all subplots
    legend_elements = [
        Line2D([0], [0], color='black', ls='-', lw=2.2, label='FOM (Ground truth)'),
        Line2D([0], [0], color=OKABE_ITO_PALETTE[0], ls='--', lw=2.2, label='ROM'),
        Line2D([0], [0], color=OKABE_ITO_PALETTE[1], ls='-.', lw=2.2, label='HROM'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=16,
               frameon=True, framealpha=0.95, facecolor='white', edgecolor='#cccccc',
               bbox_to_anchor=(0.5, 0.012))

    fig.tight_layout(rect=[0.0, 0.045, 1.0, 0.96])

    if plot_path:
        save_figure(fig, plot_path, fig_data=fig_data)
        plt.close(fig)

    return fig, axes


def plot_solver(solver, data_folder=None):
    """
    Generate and display plots for simulation results of a representative solver instance.

    Parameters:
        solver: Solver instance or dictionary containing simulation data.
        data_folder (str, optional): Directory to save output files. Defaults to MechSystem.data_folder.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    from System import MechSystem

    # Extract attributes whether solver is an object or dictionary
    if isinstance(solver, dict):
        t_points = solver['t_points']
        coords = solver['coords']
        n_particles = coords.shape[1]
        nosc = n_particles * 3
        dt = solver.get('dt', 0.001)
        omega2 = solver.get('Omega2', None)
        solver_name = solver.get('solver_name', 'Solver')
        solver_class_name = solver.get('solver_class_name', solver_name)
        sym_error = solver.get('sym_error')
        g_norm = solver.get('g_norm')
        eng_error = solver.get('eng_error')
        lin_momentum_err = solver.get('lin_momentum_err')
        angular_momentum_err = solver.get('angular_momentum_err')
        T_final = t_points[-1]
        is_beam = solver.get('is_beam', False)
    else:
        t_points = solver.t_points
        T_final = getattr(solver, 'T_final', t_points[-1])
        if hasattr(solver, 'coords') and solver.coords is not None:
            coords = solver.coords
            nosc = getattr(solver, 'nosc', coords.shape[1] * 3)
        else:
            nosc = solver.nosc
            coords = solver.y[:, :nosc].reshape(-1, nosc // 3, 3)
        n_particles = coords.shape[1]
        dt = getattr(solver, 'dt', None)
        omega2 = getattr(solver, 'Omega2', None)
        solver_class = getattr(solver, 'solver_class', getattr(solver, '__class__', None))
        solver_class_name = solver_class.__name__ if solver_class else 'Solver'
        sym_error = getattr(solver, 'sym_error', None)
        g_norm = getattr(solver, 'g_norm', None)
        eng_error = getattr(solver, 'eng_error', None)
        lin_momentum_err = getattr(solver, 'lin_momentum_err', None)
        angular_momentum_err = getattr(solver, 'angular_momentum_err', None)
        is_beam = (
            getattr(solver, 'is_elastica', False) or
            hasattr(solver, 'boundary') or
            'Elastica' in solver_class_name or
            (hasattr(solver, 'ds') and hasattr(solver, 'length'))
        )

    n_steps = coords.shape[0]

    fig = plt.figure(figsize=(13, 12), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.1, h_pad=0.1, hspace=0.1, wspace=0.1)

    if omega2 is not None:
        omega2_bytes = omega2.tobytes() if hasattr(omega2, 'tobytes') else str(omega2).encode()
        omega2_hash = hashlib.md5(omega2_bytes).hexdigest()[:8]
    else:
        omega2_hash = 'unknown'

    dt_str = f"{dt}" if dt is not None else "N/A"
    fig.suptitle(rf'integrator = {solver_class_name}, $\Delta t = {dt_str}$, $\Omega_2$ hash = {omega2_hash}')

    gs = fig.add_gridspec(5, 2)
    ax0, ax1, ax2, ax3, ax4 = [fig.add_subplot(gs[i, 0]) for i in [0, 1, 2, 3, 4]]

    if sym_error is not None:
        plot_data(ax0, t_points, sym_error, xlims=(0, T_final),
                  ylabel=r'$\Delta Sp$', margins=10)
        if max(abs(sym_error)) < 1e-15:
            ax0.set_ylim([-1e-15, 1e-15])

    if g_norm is not None:
        plot_data(ax1, t_points, g_norm, xlims=(0, T_final),
                  ylabel=r"$\Delta \mathcal{S}$", margins=10)

    if eng_error is not None:
        plot_data(ax2, t_points, eng_error, xlims=(0, T_final),
                  ylabel=r'$\Delta H$', margins=10)

    if lin_momentum_err is not None:
        plot_data(ax3, t_points, lin_momentum_err, xlims=(0, T_final),
                  ylabel=r'$\Delta L$', margins=10)

    if angular_momentum_err is not None:
        plot_data(ax4, t_points, angular_momentum_err, xlims=(0, T_final),
                  ylabel=r'$\Delta J$', margins=10)

    ax4.set_xlabel('time')

    # Hide inner labels
    for ax in [ax0, ax1, ax2, ax3, ax4]:
        ax.label_outer()

    # Plot particle positions / beam centerline over time
    sc = None
    if is_beam:
        ax_pp = fig.add_subplot(gs[:5, -1], projection='3d')
        ax_pp.set_box_aspect([1, 1, 1])
        # Pinned support marker at origin
        ax_pp.scatter([coords[0, 0, 0]], [coords[0, 0, 1]], [coords[0, 0, 2]],
                      s=80, c='black', marker='s', label='Pinned root', zorder=5)
        # Trace tip path
        ax_pp.plot(coords[:, -1, 0], coords[:, -1, 1], coords[:, -1, 2],
                   'r--', lw=1.2, alpha=0.7, label='Tip path')
        # Select 15 evenly-spaced snapshots
        n_snaps = min(15, n_steps)
        snap_indices = np.linspace(0, n_steps - 1, n_snaps, dtype=int)
        cmap = mcm.get_cmap('viridis')
        norm = mcolors.Normalize(vmin=t_points[0], vmax=t_points[-1])
        for s_idx in snap_indices:
            t_val = t_points[s_idx]
            color = cmap(norm(t_val))
            ax_pp.plot(coords[s_idx, :, 0], coords[s_idx, :, 1], coords[s_idx, :, 2],
                       color=color, lw=2.2, alpha=0.85)
            ax_pp.scatter(coords[s_idx, :, 0], coords[s_idx, :, 1], coords[s_idx, :, 2],
                          color=color, s=12, alpha=0.9)
        sc = mcm.ScalarMappable(norm=norm, cmap=cmap)
        sc.set_array(t_points)

        ax_pp.set_xlabel('x', labelpad=10)
        ax_pp.set_ylabel('y', labelpad=10)
        ax_pp.set_zlabel('z', labelpad=10)
        ax_pp.set_title('Phase portrait')
        ax_pp.view_init(elev=15, azim=45)
        cbar = fig.colorbar(sc, ax=ax_pp, shrink=0.5, pad=0.1)
        cbar.set_label('Time')
    else:
        cmap = plt.get_cmap('viridis')
        norm = mcolors.Normalize(vmin=t_points[0], vmax=t_points[-1])

        has_inset = n_particles >= 8
        if has_inset:
            ax_ins = fig.add_subplot(gs[:2, -1], projection='3d')
            ax_pp = fig.add_subplot(gs[2:, -1], projection='3d')
        else:
            ax_pp = fig.add_subplot(gs[:5, -1], projection='3d')

        strand1 = np.arange(0, n_particles, 2)
        strand2 = np.arange(1, n_particles, 2)
        n_rungs = n_particles // 2

        # Identify top region for the inset
        top_rungs_count = max(4, min(12, n_rungs // 4)) if has_inset else 0
        top_start_rung = n_rungs - top_rungs_count
        top_indices = np.arange(top_start_rung * 2, n_particles) if has_inset else np.array([])

        # Main phase portrait (ax_pp)
        # Double helix backbone at t=0
        ax_pp.plot(coords[0, strand1, 0], coords[0, strand1, 1], coords[0, strand1, 2],
                   color='teal', lw=1.6, alpha=0.85, label='Helix strand 1')
        ax_pp.plot(coords[0, strand2, 0], coords[0, strand2, 1], coords[0, strand2, 2],
                   color='darkcyan', lw=1.6, alpha=0.85, label='Helix strand 2')

        # Periodic cross-rungs at t=0
        rung_stride = max(1, n_rungs // 30)
        for r in range(0, n_rungs, rung_stride):
            p1, p2 = 2 * r, 2 * r + 1
            ax_pp.plot([coords[0, p1, 0], coords[0, p2, 0]],
                       [coords[0, p1, 1], coords[0, p2, 1]],
                       [coords[0, p1, 2], coords[0, p2, 2]],
                       color='teal', lw=0.8, alpha=0.45)

        # Initial configuration nodes
        ax_pp.scatter(coords[0, :, 0], coords[0, :, 1], coords[0, :, 2],
                      color='teal', s=8, alpha=0.6)

        # Subsampled particle evolution over time (both Strand 1 and Strand 2)
        step = max(1, n_steps // 1500)
        t_sub = t_points[::step]
        coords_sub = coords[::step]
        for i in range(n_particles):
            ax_pp.scatter(coords_sub[:, i, 0], coords_sub[:, i, 1], coords_sub[:, i, 2],
                          c=t_sub, cmap=cmap, s=0.6, alpha=0.06)

        # Highlight tip paths (strands 1 & 2 tips)
        if n_particles >= 2:
            ax_pp.plot(coords[:, -2, 0], coords[:, -2, 1], coords[:, -2, 2],
                       color='#D55E00', lw=1.8, alpha=0.85, label='Tip path 1')
            ax_pp.plot(coords[:, -1, 0], coords[:, -1, 1], coords[:, -1, 2],
                       color='#CC79A7', lw=1.8, alpha=0.85, label='Tip path 2')

            # Tip ground projection (clean single projection)
            ax_pp.plot(coords[:, -2, 0], coords[:, -2, 1], np.zeros(n_steps),
                       color='#D55E00', lw=1.2, alpha=0.5, ls='-', label='Tip xy-projection')

        # Declutter x, y, and z axes with integer ticks covering the whole data range
        all_x_min, all_x_max = coords[:, :, 0].min(), coords[:, :, 0].max()
        all_y_min, all_y_max = coords[:, :, 1].min(), coords[:, :, 1].max()
        all_z_min, all_z_max = coords[:, :, 2].min(), coords[:, :, 2].max()

        x_min_int = int(np.floor(all_x_min))
        x_max_int = int(np.ceil(all_x_max))
        y_min_int = int(np.floor(all_y_min))
        y_max_int = int(np.ceil(all_y_max))
        z_min_int = max(0, int(np.floor(all_z_min)))
        z_step = 2 if all_z_max > 5 else 1
        z_max_int = int(np.ceil(all_z_max / z_step)) * z_step

        ax_pp.set_xlim([x_min_int, x_max_int])
        ax_pp.set_ylim([y_min_int, y_max_int])
        ax_pp.set_zlim([z_min_int, z_max_int])
        ax_pp.set_xticks(np.arange(x_min_int, x_max_int + 1, 1))
        ax_pp.set_yticks(np.arange(y_min_int, y_max_int + 1, 1))
        ax_pp.set_zticks(np.arange(z_min_int, z_max_int + 1, z_step))

        ax_pp.set_xlabel('x', labelpad=10)
        ax_pp.set_ylabel('y', labelpad=10)
        ax_pp.set_zlabel('z', labelpad=10)
        ax_pp.set_title('Phase portrait (full lattice)')
        ax_pp.set_box_aspect([1, 1, 1.4] if has_inset else [1, 1, 1])
        ax_pp.view_init(elev=15, azim=45)

        # Top Lattice Inset (ax_ins)
        if has_inset and len(top_indices) > 0:
            z_min_traj = coords[:, top_indices, 2].min()
            z_max_traj = coords[:, top_indices, 2].max()
            x_min_traj, x_max_traj = coords[:, top_indices, 0].min(), coords[:, top_indices, 0].max()
            y_min_traj, y_max_traj = coords[:, top_indices, 1].min(), coords[:, top_indices, 1].max()

            x_pad = 0.08 * (x_max_traj - x_min_traj + 1e-6)
            y_pad = 0.08 * (y_max_traj - y_min_traj + 1e-6)
            z_pad = 0.08 * (z_max_traj - z_min_traj + 1e-6)

            x_box = [x_min_traj - x_pad, x_max_traj + x_pad]
            y_box = [y_min_traj - y_pad, y_max_traj + y_pad]
            z_box = [z_min_traj - z_pad, z_max_traj + z_pad]

            # Bounding box on main phase portrait
            box_edges = [
                ([x_box[0], x_box[1]], [y_box[0], y_box[0]], [z_box[0], z_box[0]]),
                ([x_box[0], x_box[1]], [y_box[1], y_box[1]], [z_box[0], z_box[0]]),
                ([x_box[0], x_box[1]], [y_box[0], y_box[0]], [z_box[1], z_box[1]]),
                ([x_box[0], x_box[1]], [y_box[1], y_box[1]], [z_box[1], z_box[1]]),
                ([x_box[0], x_box[0]], [y_box[0], y_box[1]], [z_box[0], z_box[0]]),
                ([x_box[1], x_box[1]], [y_box[0], y_box[1]], [z_box[0], z_box[0]]),
                ([x_box[0], x_box[0]], [y_box[0], y_box[1]], [z_box[1], z_box[1]]),
                ([x_box[1], x_box[1]], [y_box[0], y_box[1]], [z_box[1], z_box[1]]),
                ([x_box[0], x_box[0]], [y_box[0], y_box[0]], [z_box[0], z_box[1]]),
                ([x_box[1], x_box[1]], [y_box[0], y_box[0]], [z_box[0], z_box[1]]),
                ([x_box[0], x_box[0]], [y_box[1], y_box[1]], [z_box[0], z_box[1]]),
                ([x_box[1], x_box[1]], [y_box[1], y_box[1]], [z_box[0], z_box[1]]),
            ]
            for xs, ys, zs in box_edges:
                ax_pp.plot(xs, ys, zs, color='#D55E00', ls='--', lw=1.2, alpha=0.8)

            z_thresh_label = coords[0, top_indices[0], 2]
            ax_ins.set_title(rf'Top of lattice (inset zoom, $z_0 \geq {z_thresh_label:.1f}$)')

            top_s1 = [i for i in strand1 if i in top_indices]
            top_s2 = [i for i in strand2 if i in top_indices]

            # Initial top structure
            ax_ins.plot(coords[0, top_s1, 0], coords[0, top_s1, 1], coords[0, top_s1, 2],
                        color='teal', lw=2.2, alpha=0.9, label='Strand 1 (t=0)')
            ax_ins.plot(coords[0, top_s2, 0], coords[0, top_s2, 1], coords[0, top_s2, 2],
                        color='darkcyan', lw=2.2, alpha=0.9, ls='--', label='Strand 2 (t=0)')

            # Cross rungs at top
            for r in range(top_start_rung, n_rungs):
                p1, p2 = 2 * r, 2 * r + 1
                ax_ins.plot([coords[0, p1, 0], coords[0, p2, 0]],
                            [coords[0, p1, 1], coords[0, p2, 1]],
                            [coords[0, p1, 2], coords[0, p2, 2]],
                            color='gray', lw=1.2, alpha=0.7)

            ax_ins.scatter(coords[0, top_indices, 0], coords[0, top_indices, 1], coords[0, top_indices, 2],
                           color='teal', s=16, alpha=0.9)

            # Continuous trajectories of top rungs (both strands)
            for idx in top_indices:
                ax_ins.scatter(coords_sub[:, idx, 0], coords_sub[:, idx, 1], coords_sub[:, idx, 2],
                               c=t_sub, cmap=cmap, s=1.0, alpha=0.10)

            # Prominently trace tip trajectories in inset
            ax_ins.plot(coords[:, -2, 0], coords[:, -2, 1], coords[:, -2, 2],
                        color='#D55E00', lw=2.4, alpha=0.9, label='Tip strand 1')
            ax_ins.plot(coords[:, -1, 0], coords[:, -1, 1], coords[:, -1, 2],
                        color='#CC79A7', lw=2.4, alpha=0.9, label='Tip strand 2')

            # Declutter inset x, y, and z axes with integer ticks covering data range
            ins_x_min_int = int(np.floor(x_min_traj))
            ins_x_max_int = int(np.ceil(x_max_traj))
            ins_y_min_int = int(np.floor(y_min_traj))
            ins_y_max_int = int(np.ceil(y_max_traj))
            ins_z_min_int = int(np.floor(z_min_traj))
            ins_z_max_int = int(np.ceil(z_max_traj))

            ax_ins.set_xlim([ins_x_min_int, ins_x_max_int])
            ax_ins.set_ylim([ins_y_min_int, ins_y_max_int])
            ax_ins.set_zlim([ins_z_min_int, ins_z_max_int])
            ax_ins.set_xticks(np.arange(ins_x_min_int, ins_x_max_int + 1, 1))
            ax_ins.set_yticks(np.arange(ins_y_min_int, ins_y_max_int + 1, 1))
            ax_ins.set_zticks(np.arange(ins_z_min_int, ins_z_max_int + 1, 1))
            ax_ins.set_xlabel('x', labelpad=10)
            ax_ins.set_ylabel('y', labelpad=10)
            ax_ins.set_zlabel('z', labelpad=10)
            ax_ins.set_box_aspect([1, 1, 0.9])
            ax_ins.view_init(elev=15, azim=45)

        sc_map = mcm.ScalarMappable(norm=norm, cmap=cmap)
        sc_map.set_array(t_points)
        cbar_axes = [ax_ins, ax_pp] if has_inset else ax_pp
        cbar = fig.colorbar(sc_map, ax=cbar_axes, shrink=0.55 if has_inset else 0.5, pad=0.08)
        cbar.set_label('Time')

    fig.show()

    fig_data = {
        't_points': t_points,
        'lin_momentum_err': lin_momentum_err,
        'angular_momentum_err': angular_momentum_err,
        'coords': coords
    }
    if sym_error is not None:
        fig_data['sym_error'] = sym_error
    if g_norm is not None:
        fig_data['g_norm'] = g_norm
    if eng_error is not None:
        fig_data['eng_error'] = eng_error

    target_folder = data_folder or getattr(MechSystem, 'data_folder', 'data')
    solver_name_short = solver_class_name.replace('Conformal', 'C').replace('StormerVerlet', 'SV').replace('ImplicitMidpoint', 'IM').replace('DiscreteGradient', 'DG').replace('HyperReduced', 'HR').replace('Reduced', 'R')
    filename = os.path.join(target_folder, f"{solver_name_short}.pdf")
    save_figure(fig, filename, fig_data=fig_data)
    return fig


def plot_solvers(kwds, solvers, data_folder=None):
    """
    Generate and display plots for representative solver instances.

    Parameters:
        kwds (dict): Dictionary of keyword arguments containing registered_solver_classes.
        solvers (list): List of solver instances used to solve the system.
        data_folder (str, optional): Directory to save plots. Defaults to MechSystem.data_folder.
    """
    from ConcreteSolvers import _matches_solver_family
    for solver_class in kwds.get('registered_solver_classes', []):
        filtered_solvers = [s for s in solvers if s and _matches_solver_family(s, solver_class)]
        if filtered_solvers:
            plot_solver(filtered_solvers[-1], data_folder=data_folder)


def _resolve_solver_base(solver_repr):
    """
    Extract a normalized base solver short name (e.g. 'CSVSolver', 'DGSolver', 'CIMSolver')
    from a solver class, instance, or string name, removing any 'Reduced' or 'HyperReduced' prefixes.
    """
    if solver_repr is None:
        return 'CSVSolver'
    if hasattr(solver_repr, '__name__'):
        name = solver_repr.__name__
    elif isinstance(solver_repr, type):
        name = solver_repr.__name__
    elif hasattr(solver_repr, '__class__') and not isinstance(solver_repr, str):
        name = solver_repr.__class__.__name__
    else:
        name = str(solver_repr)

    short = (name
             .replace('Conformal', 'C')
             .replace('StormerVerlet', 'SV')
             .replace('ImplicitMidpoint', 'IM')
             .replace('DiscreteGradient', 'DG')
             .replace('HyperReduced', 'HR')
             .replace('Reduced', 'R'))
    # Strip leading HR or R
    if short.startswith('HR'):
        base = short[2:]
    elif short.startswith('R'):
        base = short[1:]
    else:
        base = short
    return base or 'CSVSolver'


def plot_combined_errors(data_folder=None, solver_name=None, registered_solver_classes=None,
                         filename=None, kwds=None, fom_data=None, rom_data=None, hrom_data=None,
                         solvers=None, show_fig=False, col_colors=None, title=None, **kwargs):
    """
    Generate and save a combined 5x3 grid of subplots for errors alone across
    Full-Order Model (FOM), Reduced-Order Model (ROM), and Hyper-Reduced Order Model (HROM):
    - Rows (5): Symplectic error (ΔSp), Constraint error (ΔS), Energy error (ΔH),
                Linear momentum error (ΔL), Angular momentum error (ΔJ).
    - Columns (3): FOM, ROM, HROM.

    The subplots are populated from the corresponding .data.pickle files (e.g., CSVSolver.data.pickle,
    RCSVSolver.data.pickle, HRCSVSolver.data.pickle) saved during the run, or directly from passed solvers.
    Fail-safe: will not crash if any or all of the solvers fail to produce valid results or data.

    Parameters
    ----------
    data_folder : str, optional
        Target directory where the .data.pickle files are stored. Defaults to MechSystem.data_folder or 'data'.
    solver_name : str or type, optional
        Name or class of the solver family to plot. If None, auto-detected from registered_solver_classes,
        kwds, or available files in data_folder.
    registered_solver_classes : list, optional
        List of solver classes to plot combined figures for.
    filename : str, optional
        Custom path for saving the combined plot (.pdf).
    kwds : dict, optional
        Dictionary of keyword arguments containing 'registered_solver_classes'.
    fom_data, rom_data, hrom_data : dict, optional
        Pre-loaded data dictionaries (skips loading from file if supplied).
    solvers : dict or list, optional
        Solver instances or dict of {'FOM': s_f, 'ROM': s_r, 'HROM': s_hr}.
    show_fig : bool, optional
        Whether to display the figure via fig.show(). Default is False.
    col_colors : list of 3 colors, optional
        Colors for FOM, ROM, and HROM columns. Defaults to Okabe-Ito Blue, Orange, Green.

    Returns
    -------
    matplotlib.figure.Figure or list of Figures or None
        The generated figure(s).
    """
    from System import MechSystem
    target_folder = data_folder or getattr(MechSystem, 'data_folder', None) or 'data'

    # Extract metrics from solver instances if supplied
    solvers = solvers or kwargs.get('solvers')
    if solvers is not None:
        s_f, s_r, s_hr = None, None, None
        if isinstance(solvers, dict):
            s_f = solvers.get('FOM', solvers.get('fom'))
            s_r = solvers.get('ROM', solvers.get('rom'))
            s_hr = solvers.get('HROM', solvers.get('hrom'))
        elif isinstance(solvers, (list, tuple)):
            if len(solvers) > 0:
                s_f = solvers[0]
            if len(solvers) > 1:
                s_r = solvers[1]
            if len(solvers) > 2:
                s_hr = solvers[2]

        def _extract_error_data(s):
            if s is None:
                return None
            if isinstance(s, dict):
                return s
            return {
                't_points': getattr(s, 't_points', None),
                'sym_error': getattr(s, 'sym_error', None),
                'g_norm': getattr(s, 'g_norm', getattr(s, 'constraint_error', getattr(s, 'phase_error', None))),
                'eng_error': getattr(s, 'eng_error', None),
                'lin_momentum_err': getattr(s, 'lin_momentum_err', getattr(s, 'lin_mom_error', None)),
                'angular_momentum_err': getattr(s, 'angular_momentum_err', getattr(s, 'ang_mom_error', None)),
            }

        if fom_data is None and s_f is not None:
            fom_data = _extract_error_data(s_f)
        if rom_data is None and s_r is not None:
            rom_data = _extract_error_data(s_r)
        if hrom_data is None and s_hr is not None:
            hrom_data = _extract_error_data(s_hr)

        if solver_name is None:
            for cand in [s_f, s_r, s_hr]:
                if cand is not None:
                    if hasattr(cand, 'solver_class') and cand.solver_class is not None:
                        solver_name = cand.solver_class.__name__
                        break
                    elif hasattr(cand, '__class__') and not isinstance(cand, dict):
                        solver_name = cand.__class__.__name__
                        break
                    elif isinstance(cand, dict) and 'solver_name' in cand:
                        solver_name = cand['solver_name']
                        break

    # Determine base solver names to process
    base_names = []
    if solver_name is not None:
        base_names.append(_resolve_solver_base(solver_name))
    elif registered_solver_classes is not None:
        for cls in registered_solver_classes:
            b = _resolve_solver_base(cls)
            if b not in base_names:
                base_names.append(b)
    elif kwds is not None and 'registered_solver_classes' in kwds:
        for cls in kwds.get('registered_solver_classes', []):
            b = _resolve_solver_base(cls)
            if b not in base_names:
                base_names.append(b)
    elif fom_data is not None or rom_data is not None or hrom_data is not None:
        base_names.append('CSVSolver')
    else:
        # Auto-discover from files in target_folder
        if os.path.exists(target_folder):
            try:
                for fname in os.listdir(target_folder):
                    if fname.endswith('Solver.data.pickle'):
                        clean_fname = fname[:-len('.data.pickle')]
                        b = _resolve_solver_base(clean_fname)
                        if b not in base_names:
                            base_names.append(b)
            except Exception:
                pass
        if not base_names:
            base_names.append('CSVSolver')

    # Color palette for columns: FOM (Blue), ROM (Orange), HROM (Green)
    if col_colors is None:
        col_colors = [OKABE_ITO_PALETTE[4], OKABE_ITO_PALETTE[0], OKABE_ITO_PALETTE[2]]

    metrics = [
        ('sym_error', r'$\Delta Sp$'),
        ('g_norm', r'$\Delta \mathcal{S}$'),
        ('eng_error', r'$\Delta H$'),
        ('lin_momentum_err', r'$\Delta L$'),
        ('angular_momentum_err', r'$\Delta J$'),
    ]

    metric_aliases = {
        'sym_error': ['sym_error', 'sym_err'],
        'g_norm': ['g_norm', 'constraint_error', 'phase_error'],
        'eng_error': ['eng_error', 'energy_error', 'eng_err'],
        'lin_momentum_err': ['lin_momentum_err', 'lin_mom_error'],
        'angular_momentum_err': ['angular_momentum_err', 'ang_mom_error'],
    }

    def _safe_load_pickle(fpath):
        if not fpath or not os.path.exists(fpath):
            return None
        try:
            with open(fpath, 'rb') as f:
                d = pickle.load(f)
                return d if isinstance(d, dict) else None
        except Exception as e:
            print(f"  [plot_combined_errors] Warning: Could not read {fpath}: {e}")
            return None

    figures = []

    for base_name in base_names:
        fom_file = os.path.join(target_folder, f"{base_name}.data.pickle")
        rom_file = os.path.join(target_folder, f"R{base_name}.data.pickle")
        hrom_file = os.path.join(target_folder, f"HR{base_name}.data.pickle")

        cur_fom = fom_data if fom_data is not None else _safe_load_pickle(fom_file)
        cur_rom = rom_data if rom_data is not None else _safe_load_pickle(rom_file)
        cur_hrom = hrom_data if hrom_data is not None else _safe_load_pickle(hrom_file)

        fig, axes = plt.subplots(5, 3, figsize=(16, 13), constrained_layout=True)
        try:
            fig.set_constrained_layout_pads(w_pad=0.08, h_pad=0.08, hspace=0.08, wspace=0.08)
        except Exception:
            pass

        models = [
            ('FOM', cur_fom, col_colors[0]),
            ('ROM', cur_rom, col_colors[1]),
            ('HROM', cur_hrom, col_colors[2]),
        ]

        for col_idx, (col_label, data_dict, color) in enumerate(models):
            has_data = data_dict is not None and isinstance(data_dict, dict)
            t_points = None
            T_final = None
            if has_data:
                raw_t = data_dict.get('t_points')
                if raw_t is not None:
                    try:
                        t_points = np.asarray(raw_t, dtype=float)
                        if t_points.size > 0:
                            T_final = float(t_points[-1])
                    except Exception:
                        t_points = None

            for row_idx, (canonical_key, ylabel) in enumerate(metrics):
                ax = axes[row_idx, col_idx]
                configure_axis(ax)

                # Column title on top row
                if row_idx == 0:
                    ax.set_title(col_label, fontsize=22, pad=18)

                # Row metric label on leftmost column
                if col_idx == 0:
                    ax.set_ylabel(ylabel, fontsize=20)

                # Find metric data
                y_raw = None
                if has_data:
                    for alias in metric_aliases.get(canonical_key, [canonical_key]):
                        if alias in data_dict and data_dict[alias] is not None:
                            y_raw = data_dict[alias]
                            break

                if y_raw is not None:
                    try:
                        y_arr = np.asarray(y_raw, dtype=float)
                        if y_arr.size > 0 and not np.all(np.isnan(y_arr)):
                            if t_points is None or len(t_points) != len(y_arr):
                                t_plot = np.arange(len(y_arr))
                            else:
                                t_plot = t_points

                            ax.plot(t_plot, y_arr, color=color, linewidth=2.0)
                            if T_final is not None and t_points is not None and len(t_points) == len(y_arr):
                                ax.set_xlim([0.0, T_final])
                                plt.draw()
                                ticks_x = [t for t in ax.get_xticks() if -1e-9 <= t <= T_final + 1e-9]
                                margin_x = 0.02 * (T_final - 0.0)
                                ax.set_xlim([-margin_x, T_final + margin_x])
                                if len(ticks_x) >= 2:
                                    ax.set_xticks(ticks_x)
                            else:
                                adjust_axis(ax, t_plot, axis='x', is_log_scale=False)

                            y_valid = y_arr[np.isfinite(y_arr)]
                            if y_valid.size > 0:
                                adjust_axis(ax, y_valid, axis='y', is_log_scale=False)
                        else:
                            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center',
                                    transform=ax.transAxes, color='gray', fontsize=14, fontstyle='italic')
                            ax.set_xticks([])
                            ax.set_yticks([])
                    except Exception:
                        ax.text(0.5, 0.5, 'Plotting error', ha='center', va='center',
                                transform=ax.transAxes, color='gray', fontsize=14, fontstyle='italic')
                        ax.set_xticks([])
                        ax.set_yticks([])
                else:
                    status_text = 'Solver failed / No data' if not has_data else 'Metric unavailable'
                    ax.text(0.5, 0.5, status_text, ha='center', va='center',
                            transform=ax.transAxes, color='gray', fontsize=14, fontstyle='italic')
                    ax.set_xticks([])
                    ax.set_yticks([])

                # Label x-axis on bottom row only
                if row_idx == 4:
                    ax.set_xlabel('time', fontsize=18)
                else:
                    ax.tick_params(labelbottom=False)

        if title is not None:
            fig.suptitle(title, fontsize=22)

        if filename is None:
            out_pdf = os.path.join(target_folder, f"{base_name}_combined_errors.pdf")
        elif os.path.isabs(filename) or (target_folder and filename.startswith(target_folder)):
            out_pdf = filename
        elif os.path.dirname(filename):
            out_pdf = filename
        else:
            out_pdf = os.path.join(target_folder, filename)

        combined_fig_data = {
            'FOM': cur_fom,
            'ROM': cur_rom,
            'HROM': cur_hrom,
            'solver_name': base_name,
            'metrics': [m[0] for m in metrics],
            'metric_labels': [m[1] for m in metrics],
        }

        try:
            save_figure(fig, out_pdf, fig_data=combined_fig_data)
            print(f"Saved combined error plot to {out_pdf}")

            # Also save canonical combined_errors.pdf for ease of reference
            if filename is None:
                default_out_pdf = os.path.join(target_folder, "combined_errors.pdf")
                if os.path.abspath(out_pdf) != os.path.abspath(default_out_pdf):
                    try:
                        shutil.copyfile(out_pdf, default_out_pdf)
                        default_data = os.path.join(target_folder, "combined_errors.data.pickle")
                        src_data = os.path.splitext(out_pdf)[0] + ".data.pickle"
                        if os.path.exists(src_data):
                            shutil.copyfile(src_data, default_data)
                    except Exception:
                        pass
        except Exception as e:
            print(f"  [plot_combined_errors] Warning: Could not save figure to {out_pdf}: {e}")

        if show_fig:
            fig.show()
        else:
            plt.close(fig)

        figures.append(fig)

    return figures[0] if len(figures) == 1 else figures


def plot_combined_phase_portraits(data_folder=None, solver_name=None, registered_solver_classes=None,
                                  filename=None, kwds=None, fom_data=None, rom_data=None, hrom_data=None,
                                  solvers=None,
                                  times=[1.0, 2.0, 3.0], radius=0.5, rung_color='#555555',
                                  strand1_color=OKABE_ITO_PALETTE[4],
                                  strand2_color=OKABE_ITO_PALETTE[5],
                                  circular_aspect_multiplier=3.0,
                                  show_base_circle=True,
                                  show_surface_circles=False,
                                  show_cross_rungs=False,
                                  azim_deg=36, elev_deg=16,
                                  show_fig=False, **kwargs):
    """
    Generate and save a combined 3x3 grid of phase portraits across
    Full-Order Model (FOM), Reduced-Order Model (ROM), and Hyper-Reduced Order Model (HROM)
    in cylindrical coordinates:
    - Columns (3): FOM, ROM, HROM.
    - Rows (3): Time instances t = 1, t = 2, t = 3.

    Visual style:
    - Side-on 3D orthographic projection without Cartesian frame clutter.
    - Translucent reference cylinder (r = radius) with solid base circle at z = 0.
    - Cylinders rendered visually twice as wide via box aspect ratio.
    - Centerline z-axis with top label z (tick at 8 and label O at origin removed).
    - Double-helix backbones and atoms colored in colorblind-accessible Okabe-Ito Blue
      and Vermillion (cross-rungs and intermediate surface circles removed).
    - Unified 3D coordinate limits across all subplots for faithful comparison.
    - Clean 2-item legend for Strand 1 and Strand 2.
    - Regular non-bold typography.
    - Fail-safe: handles missing or corrupt data gracefully without crashing.

    Parameters
    ----------
    data_folder : str, optional
        Target directory where the .data.pickle files are stored. Defaults to MechSystem.data_folder or 'data'.
    solver_name : str or type, optional
        Name or class of the solver family to plot. If None, auto-detected.
    registered_solver_classes : list, optional
        List of solver classes to plot combined figures for.
    filename : str, optional
        Custom path for saving the combined plot (.pdf).
    kwds : dict, optional
        Dictionary of keyword arguments containing 'registered_solver_classes'.
    fom_data, rom_data, hrom_data : dict, optional
        Pre-loaded data dictionaries (skips loading from file if supplied).
    solvers : dict or list, optional
        Solver instances or dict of {'FOM': s_f, 'ROM': s_r, 'HROM': s_hr}.
    times : list of float, optional
        Time instances to plot in rows. Default is [1.0, 2.0, 3.0].
    radius : float, optional
        Reference cylinder base radius. Default is 0.5.
    rung_color : str, optional
        Color for the cross-rungs. Default is '#555555'.
    strand1_color : str, optional
        Color for Helix Strand 1. Default is Okabe-Ito Blue ('#0072B2').
    strand2_color : str, optional
        Color for Helix Strand 2. Default is Okabe-Ito Vermillion ('#D55E00').
    circular_aspect_multiplier : float, optional
        Aspect multiplier for x and y axes relative to z. Default is 3.0 (twice as wide visually).
    show_base_circle : bool, optional
        Whether to display the cylinder base circle at z = 0. Default is True.
    show_surface_circles : bool, optional
        Whether to display latitude circles along the cylinder surface. Default is False.
    show_cross_rungs : bool, optional
        Whether to display cross-rungs of the lattice. Default is False.
    azim_deg : float, optional
        Azimuth view angle in degrees. Default is 36.
    elev_deg : float, optional
        Elevation view angle in degrees. Default is 16.
    show_fig : bool, optional
        Whether to display the figure via fig.show(). Default is False.

    Returns
    -------
    matplotlib.figure.Figure or list of Figures or None
        The generated figure(s).
    """
    from System import MechSystem
    target_folder = data_folder or getattr(MechSystem, 'data_folder', None) or 'data'

    # Extract coordinates from solver instances if supplied
    solvers = solvers or kwargs.get('solvers')
    if solvers is not None:
        s_f, s_r, s_hr = None, None, None
        if isinstance(solvers, dict):
            s_f = solvers.get('FOM', solvers.get('fom'))
            s_r = solvers.get('ROM', solvers.get('rom'))
            s_hr = solvers.get('HROM', solvers.get('hrom'))
        elif isinstance(solvers, (list, tuple)):
            if len(solvers) > 0:
                s_f = solvers[0]
            if len(solvers) > 1:
                s_r = solvers[1]
            if len(solvers) > 2:
                s_hr = solvers[2]

        def _extract_phase_data(s):
            if s is None:
                return None
            if isinstance(s, dict):
                return s
            t_pts = getattr(s, 't_points', None)
            if hasattr(s, 'coords') and s.coords is not None:
                c = s.coords
            elif hasattr(s, 'y') and s.y is not None:
                nosc = getattr(s, 'nosc', None)
                if nosc is None:
                    nosc = getattr(MechSystem, 'nosc', s.y.shape[1] // 2)
                c = s.y[:, :nosc].reshape(-1, nosc // 3, 3)
            else:
                c = None
            return {'coords': c, 't_points': t_pts}

        if fom_data is None and s_f is not None:
            fom_data = _extract_phase_data(s_f)
        if rom_data is None and s_r is not None:
            rom_data = _extract_phase_data(s_r)
        if hrom_data is None and s_hr is not None:
            hrom_data = _extract_phase_data(s_hr)

        if solver_name is None:
            for cand in [s_f, s_r, s_hr]:
                if cand is not None:
                    if hasattr(cand, 'solver_class') and cand.solver_class is not None:
                        solver_name = cand.solver_class.__name__
                        break
                    elif hasattr(cand, '__class__') and not isinstance(cand, dict):
                        solver_name = cand.__class__.__name__
                        break
                    elif isinstance(cand, dict) and 'solver_name' in cand:
                        solver_name = cand['solver_name']
                        break

    # Determine base solver names to process
    base_names = []
    if solver_name is not None:
        base_names.append(_resolve_solver_base(solver_name))
    elif registered_solver_classes is not None:
        for cls in registered_solver_classes:
            b = _resolve_solver_base(cls)
            if b not in base_names:
                base_names.append(b)
    elif kwds is not None and 'registered_solver_classes' in kwds:
        for cls in kwds.get('registered_solver_classes', []):
            b = _resolve_solver_base(cls)
            if b not in base_names:
                base_names.append(b)
    elif fom_data is not None or rom_data is not None or hrom_data is not None:
        base_names.append('CSVSolver')
    else:
        # Auto-discover from files in target_folder
        if os.path.exists(target_folder):
            try:
                for fname in os.listdir(target_folder):
                    if fname.endswith('Solver.data.pickle'):
                        clean_fname = fname[:-len('.data.pickle')]
                        b = _resolve_solver_base(clean_fname)
                        if b not in base_names:
                            base_names.append(b)
            except Exception:
                pass
        if not base_names:
            base_names.append('CSVSolver')

    def _safe_load_pickle(fpath):
        if not fpath or not os.path.exists(fpath):
            return None
        try:
            with open(fpath, 'rb') as f:
                d = pickle.load(f)
                return d if isinstance(d, dict) else None
        except Exception as e:
            print(f"  [plot_combined_phase_portraits] Warning: Could not read {fpath}: {e}")
            return None

    usetex = plt.rcParams.get('text.usetex', False) or (shutil.which('latex') is not None)
    if usetex and not plt.rcParams.get('text.usetex', False):
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

    figures = []

    for base_name in base_names:
        fom_file = os.path.join(target_folder, f"{base_name}.data.pickle")
        rom_file = os.path.join(target_folder, f"R{base_name}.data.pickle")
        hrom_file = os.path.join(target_folder, f"HR{base_name}.data.pickle")

        cur_fom = fom_data if fom_data is not None else _safe_load_pickle(fom_file)
        cur_rom = rom_data if rom_data is not None else _safe_load_pickle(rom_file)
        cur_hrom = hrom_data if hrom_data is not None else _safe_load_pickle(hrom_file)

        models = [
            ('FOM', cur_fom),
            ('ROM', cur_rom),
            ('HROM', cur_hrom),
        ]

        # Collect coords to determine unified limits
        all_coords = []
        for _, d in models:
            if d is not None and isinstance(d, dict) and 'coords' in d and d['coords'] is not None:
                try:
                    c_arr = np.asarray(d['coords'], dtype=float)
                    if c_arr.ndim == 3 and c_arr.size > 0:
                        all_coords.append(c_arr)
                except Exception:
                    pass

        if all_coords:
            all_c_concat = np.concatenate(all_coords, axis=0)
            max_r = float(np.max(np.sqrt(all_c_concat[:, :, 0]**2 + all_c_concat[:, :, 1]**2)))
            z_max_global = float(np.max(all_c_concat[:, :, 2]))
            lim_xy = max(1.6, 1.2 * max_r)
            z_axis_top = max(8.55, z_max_global + 0.6)
            n_particles_legend = all_coords[0].shape[1]
            n_rungs_legend = n_particles_legend // 2
        else:
            lim_xy = 1.6
            z_axis_top = 8.55
            n_particles_legend = 180
            n_rungs_legend = 90

        fig = plt.figure(figsize=(13.0, 14.0))
        gs = fig.add_gridspec(3, 3, left=0.10, right=0.98, bottom=0.075, top=0.94, hspace=0.02, wspace=0.02)

        theta_dense = np.linspace(0, 2 * np.pi, 80)
        z_dense = np.linspace(0, 8.0, 40)
        Theta_cyl, Z_cyl = np.meshgrid(theta_dense, z_dense)
        X_cyl = radius * np.cos(Theta_cyl)
        Y_cyl = radius * np.sin(Theta_cyl)

        for col_idx, (col_label, d) in enumerate(models):
            has_data = d is not None and isinstance(d, dict) and 'coords' in d and d['coords'] is not None
            coords = None
            t_points = None
            if has_data:
                try:
                    coords = np.asarray(d['coords'], dtype=float)
                    raw_t = d.get('t_points')
                    if raw_t is not None:
                        t_points = np.asarray(raw_t, dtype=float)
                    else:
                        t_points = np.linspace(0.0, 3.0, len(coords))
                    n_particles = coords.shape[1]
                    n_rungs = n_particles // 2
                    strand1 = np.arange(0, n_particles, 2)
                    strand2 = np.arange(1, n_particles, 2)
                except Exception:
                    has_data = False

            for row_idx, t_target in enumerate(times):
                ax = fig.add_subplot(gs[row_idx, col_idx], projection='3d')
                ax.set_proj_type('ortho')
                ax.set_axis_off()

                if not has_data or coords is None:
                    ax.text(0.5, 0.5, 0.5, 'Solver failed / No data', ha='center', va='center',
                            color='gray', fontsize=14, fontstyle='italic')
                    if row_idx == 0:
                        ax.set_title(col_label, fontsize=24, pad=18)
                    if col_idx == 0:
                        row_y = 0.94 - (row_idx + 0.5) * (0.865 / 3.0)
                        t_str = f"{int(t_target)}" if float(t_target).is_integer() else f"{t_target}"
                        fig.text(0.045, row_y, rf"$t = {t_str}$",
                                 fontsize=24, ha='center', va='center')
                    continue

                idx = int(np.argmin(np.abs(t_points - t_target)))
                c_t = coords[idx]

                # 1. Translucent reference cylinder
                ax.plot_surface(X_cyl, Y_cyl, Z_cyl, color='#eef4fb', alpha=0.18, edgecolor='none', zorder=1)

                # Optional surface circles (latitude rings and top rim)
                if show_surface_circles:
                    for zl in [2, 4, 6]:
                        ax.plot(radius * np.cos(theta_dense), radius * np.sin(theta_dense), np.full_like(theta_dense, zl),
                                color='#90a4ae', ls=':', lw=0.8, alpha=0.55, zorder=1)
                    ax.plot(radius * np.cos(theta_dense), radius * np.sin(theta_dense), np.full_like(theta_dense, 8.0),
                            color='#78909c', ls='--', lw=1.1, alpha=0.75, zorder=2)

                # Cylinder base circle at z = 0
                if show_base_circle:
                    ax.plot(radius * np.cos(theta_dense), radius * np.sin(theta_dense), np.zeros_like(theta_dense),
                            color='#37474f', ls='-', lw=1.4, alpha=0.85, zorder=2)

                # 2. Centerline z-axis (tick at 8 and label O removed)
                ax.plot([0, 0], [0, 0], [0, z_axis_top], color='#2c3e50', ls='-', lw=1.4, zorder=3)
                ax.text(0, 0, z_axis_top + 0.22, r'$z$', fontsize=18, color='#2c3e50', ha='center', va='bottom')

                # 3. Optional cross-rungs
                if show_cross_rungs:
                    for r in range(n_rungs):
                        p1, p2 = 2 * r, 2 * r + 1
                        ax.plot([c_t[p1, 0], c_t[p2, 0]],
                                [c_t[p1, 1], c_t[p2, 1]],
                                [c_t[p1, 2], c_t[p2, 2]],
                                color=rung_color, ls='--', lw=0.85, alpha=0.6, zorder=2)

                # 4. Helical backbones
                ax.plot(c_t[strand1, 0], c_t[strand1, 1], c_t[strand1, 2], color=strand1_color, ls='-', lw=2.1, alpha=0.95, zorder=4)
                ax.plot(c_t[strand2, 0], c_t[strand2, 1], c_t[strand2, 2], color=strand2_color, ls='-', lw=2.1, alpha=0.95, zorder=4)

                # 5. Atoms
                ax.scatter(c_t[strand1, 0], c_t[strand1, 1], c_t[strand1, 2], color=strand1_color,
                           edgecolor='black', linewidth=0.35, s=15, alpha=0.95, zorder=5)
                ax.scatter(c_t[strand2, 0], c_t[strand2, 1], c_t[strand2, 2], color=strand2_color,
                           edgecolor='black', linewidth=0.35, s=15, alpha=0.95, zorder=5)

                ax.set_box_aspect([circular_aspect_multiplier, circular_aspect_multiplier, 3.0])
                ax.view_init(elev=elev_deg, azim=azim_deg)
                ax.set_xlim([-lim_xy, lim_xy])
                ax.set_ylim([-lim_xy, lim_xy])
                ax.set_zlim([-0.2, z_axis_top + 0.35])

                # Column headers on top row
                if row_idx == 0:
                    ax.set_title(col_label, fontsize=24, pad=18)

                # Row labels on leftmost column
                if col_idx == 0:
                    row_y = 0.94 - (row_idx + 0.5) * (0.865 / 3.0)
                    t_str = f"{int(t_target)}" if float(t_target).is_integer() else f"{t_target}"
                    fig.text(0.045, row_y, rf"$t = {t_str}$",
                             fontsize=24, ha='center', va='center')

        # Legend: only Strand 1 and Strand 2
        atoms_per_strand = n_particles_legend // 2
        legend_elements = [
            Line2D([0], [0], color=strand1_color, ls='-', lw=2.2, marker='o', markeredgecolor='black',
                   markersize=6.5, label=f'Helix strand 1 ({atoms_per_strand} atoms)'),
            Line2D([0], [0], color=strand2_color, ls='-', lw=2.2, marker='o', markeredgecolor='black',
                   markersize=6.5, label=f'Helix strand 2 ({atoms_per_strand} atoms)'),
        ]
        if show_cross_rungs:
            legend_elements.append(
                Line2D([0], [0], color=rung_color, ls='--', lw=1.3, alpha=0.8,
                       label=f'Cross-rungs ({n_rungs_legend} rungs)')
            )

        fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.54, 0.02),
                   ncol=len(legend_elements), fontsize=15, framealpha=0.95, facecolor='white',
                   edgecolor='#cccccc', columnspacing=2.5, handletextpad=0.6)

        if filename is None:
            out_pdf = os.path.join(target_folder, f"{base_name}_combined_phase_portraits.pdf")
        elif os.path.isabs(filename) or (target_folder and filename.startswith(target_folder)):
            out_pdf = filename
        elif os.path.dirname(filename):
            out_pdf = filename
        else:
            out_pdf = os.path.join(target_folder, filename)

        combined_phase_data = {
            'FOM': cur_fom,
            'ROM': cur_rom,
            'HROM': cur_hrom,
            'solver_name': base_name,
            'times': times,
            'radius': radius,
            'columns': ['FOM', 'ROM', 'HROM'],
        }

        try:
            save_figure(fig, out_pdf, fig_data=combined_phase_data)
            print(f"Saved combined phase portraits to {out_pdf}")

            # Also save canonical combined_phase_portraits.pdf for ease of reference
            if filename is None:
                default_out_pdf = os.path.join(target_folder, "combined_phase_portraits.pdf")
                if os.path.abspath(out_pdf) != os.path.abspath(default_out_pdf):
                    try:
                        shutil.copyfile(out_pdf, default_out_pdf)
                        default_data = os.path.join(target_folder, "combined_phase_portraits.data.pickle")
                        src_data = os.path.splitext(out_pdf)[0] + ".data.pickle"
                        if os.path.exists(src_data):
                            shutil.copyfile(src_data, default_data)
                    except Exception:
                        pass
        except Exception as e:
            print(f"  [plot_combined_phase_portraits] Warning: Could not save figure to {out_pdf}: {e}")

        if show_fig:
            fig.show()
        else:
            plt.close(fig)

        figures.append(fig)

    return figures[0] if len(figures) == 1 else figures


def plot_initial_lattice_configuration(coords, filename=None, data_folder=None,
                                       rung_color='#555555',
                                       strand1_color=OKABE_ITO_PALETTE[4],
                                       strand2_color=OKABE_ITO_PALETTE[5]):
    """
    Generate and save a publication-quality figure of the initial lattice configuration (t = 0):
    (a) Side-on view with an elongated, tall vertical 3D axis showing individual atoms,
        helical backbones, horizontal cross-rungs, and an inset zoom of the top rungs.
    (b) Top view (axial projection) showing the circular cylindrical cross-section,
        radial aperture fan of the 90 rungs, and the two opposite helical strands.

    Parameters
    ----------
    coords : np.ndarray or Solver instance or dict
        Can be:
        - 3D array of shape (n_particles, 3) or (n_steps, n_particles, 3)
        - Solver instance with solver.coords or solver.y
        - Dictionary containing 'coords'
    filename : str, optional
        Target file path for saving the figure (.pdf).
    data_folder : str, optional
        Target folder if filename is not an absolute path.
    rung_color : str, optional
        Color for the cross-rungs. Default is '#555555'.
    strand1_color : str, optional
        Color for Helix Strand 1. Default is Okabe-Ito Blue ('#0072B2').
    strand2_color : str, optional
        Color for Helix Strand 2. Default is Okabe-Ito Vermillion ('#D55E00').

    Returns
    -------
    fig, axes : tuple
        Matplotlib Figure and list of Axes [ax1, ax2].
    """
    from System import MechSystem

    # Check LaTeX rendering capability
    usetex = plt.rcParams.get('text.usetex', False) or (shutil.which('latex') is not None)
    if usetex and not plt.rcParams.get('text.usetex', False):
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

    # Extract initial coordinates (n_particles, 3)
    if isinstance(coords, dict):
        c_all = coords.get('coords', coords.get('coords_t0'))
        if c_all is None:
            raise KeyError("Dictionary must contain 'coords' or 'coords_t0'")
        c0 = c_all[0] if c_all.ndim == 3 else c_all
    elif hasattr(coords, 'coords') and coords.coords is not None:
        c0 = coords.coords[0] if coords.coords.ndim == 3 else coords.coords
    elif hasattr(coords, 'y') and coords.y is not None:
        nosc = getattr(coords, 'nosc', coords.y.shape[1] // 2 * 3)
        y0 = coords.y[0, :nosc]
        c0 = y0.reshape(-1, 3)
    elif isinstance(coords, np.ndarray):
        c0 = coords[0] if coords.ndim == 3 else coords
    else:
        raise ValueError("Unsupported format for coords. Expected ndarray, Solver, or dict.")

    n_particles = c0.shape[0]
    n_rungs = n_particles // 2
    strand1 = np.arange(0, n_particles, 2)
    strand2 = np.arange(1, n_particles, 2)

    top_rungs_count = max(4, min(12, n_rungs // 4)) if n_rungs >= 8 else 0
    top_start_rung = n_rungs - top_rungs_count
    top_indices = np.arange(top_start_rung * 2, n_particles) if n_rungs >= 8 else np.array([], dtype=int)

    fig = plt.figure(figsize=(16, 11))
    fig.set_layout_engine(None)

    # Subplot 1: Side-on view (shifted left to eliminate whitespace and prevent overlap)
    ax1 = fig.add_axes([-0.02, 0.22, 0.38, 0.64], projection='3d')
    ax1.set_proj_type('ortho')
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    ax1.xaxis.pane.set_edgecolor('#d0d0d0')
    ax1.yaxis.pane.set_edgecolor('#d0d0d0')
    ax1.zaxis.pane.set_edgecolor('#d0d0d0')
    ax1.grid(True, linestyle=':', alpha=0.55)

    # Cross-rungs: dashed lines
    for r in range(n_rungs):
        p1, p2 = 2 * r, 2 * r + 1
        ax1.plot([c0[p1, 0], c0[p2, 0]], [c0[p1, 1], c0[p2, 1]], [c0[p1, 2], c0[p2, 2]],
                 color=rung_color, ls='--', lw=0.9, alpha=0.6, zorder=2)

    # Helical backbones: both continuous solid lines
    ax1.plot(c0[strand1, 0], c0[strand1, 1], c0[strand1, 2], color=strand1_color, ls='-', lw=2.2, alpha=0.95, zorder=3)
    ax1.plot(c0[strand2, 0], c0[strand2, 1], c0[strand2, 2], color=strand2_color, ls='-', lw=2.2, alpha=0.95, zorder=3)

    # Atoms / nodes
    ax1.scatter(c0[strand1, 0], c0[strand1, 1], c0[strand1, 2], color=strand1_color,
                edgecolor='black', linewidth=0.35, s=14, alpha=0.95, zorder=4)
    ax1.scatter(c0[strand2, 0], c0[strand2, 1], c0[strand2, 2], color=strand2_color,
                edgecolor='black', linewidth=0.35, s=14, alpha=0.95, zorder=4)

    # Stretched vertical aspect
    ax1.set_box_aspect([1, 1, 3.8])
    ax1.view_init(elev=15, azim=35)
    ax1.set_xlabel(r'$x$', labelpad=10, fontsize=20)
    ax1.set_ylabel(r'$y$', labelpad=10, fontsize=20)
    ax1.set_zlabel(r'$z$', labelpad=16, fontsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=16)

    ax1.set_xlim([-0.65, 0.65])
    ax1.set_ylim([-0.65, 0.65])
    ax1.set_zlim([0, 8.5])
    ax1.set_xticks([-0.5, 0, 0.5])
    ax1.set_yticks([-0.5, 0, 0.5])
    ax1.set_zticks([0, 2, 4, 6, 8])

    # Top Inset Zoom
    if top_rungs_count > 0:
        z_top_min = c0[top_indices, 2].min() - 0.08
        z_top_max = c0[top_indices, 2].max() + 0.08

        # Bounding box for zoom region in main view
        box_edges = [
            ([-0.55, 0.55], [-0.55, -0.55], [z_top_min, z_top_min]),
            ([-0.55, 0.55], [0.55, 0.55], [z_top_min, z_top_min]),
            ([-0.55, 0.55], [-0.55, -0.55], [z_top_max, z_top_max]),
            ([-0.55, 0.55], [0.55, 0.55], [z_top_max, z_top_max]),
            ([-0.55, -0.55], [-0.55, 0.55], [z_top_min, z_top_min]),
            ([0.55, 0.55], [-0.55, 0.55], [z_top_min, z_top_min]),
            ([-0.55, -0.55], [-0.55, 0.55], [z_top_max, z_top_max]),
            ([0.55, 0.55], [-0.55, 0.55], [z_top_max, z_top_max]),
            ([-0.55, -0.55], [-0.55, -0.55], [z_top_min, z_top_max]),
            ([0.55, 0.55], [-0.55, -0.55], [z_top_min, z_top_max]),
            ([-0.55, -0.55], [0.55, 0.55], [z_top_min, z_top_max]),
            ([0.55, 0.55], [0.55, 0.55], [z_top_min, z_top_max]),
        ]
        for xs, ys, zs in box_edges:
            ax1.plot(xs, ys, zs, color='#E69F00', ls='--', lw=1.2, alpha=0.9)

        # Inset 3D axes (positioned cleanly in the gap between ax1 and ax2 without overlap)
        ax_ins = fig.add_axes([0.315, 0.33, 0.18, 0.44], projection='3d')
        ax_ins.set_proj_type('ortho')
        ax_ins.xaxis.pane.fill = False
        ax_ins.yaxis.pane.fill = False
        ax_ins.zaxis.pane.fill = False
        ax_ins.xaxis.pane.set_edgecolor('#d0d0d0')
        ax_ins.yaxis.pane.set_edgecolor('#d0d0d0')
        ax_ins.zaxis.pane.set_edgecolor('#d0d0d0')
        ax_ins.grid(True, linestyle=':', alpha=0.55)

        for r in range(top_start_rung, n_rungs):
            p1, p2 = 2 * r, 2 * r + 1
            ax_ins.plot([c0[p1, 0], c0[p2, 0]], [c0[p1, 1], c0[p2, 1]], [c0[p1, 2], c0[p2, 2]],
                        color=rung_color, ls='--', lw=1.3, alpha=0.7, zorder=2)

        top_s1 = [i for i in strand1 if i in top_indices]
        top_s2 = [i for i in strand2 if i in top_indices]

        # Both strands continuous solid lines in inset
        ax_ins.plot(c0[top_s1, 0], c0[top_s1, 1], c0[top_s1, 2], color=strand1_color, ls='-', lw=2.6, alpha=0.95, zorder=3)
        ax_ins.plot(c0[top_s2, 0], c0[top_s2, 1], c0[top_s2, 2], color=strand2_color, ls='-', lw=2.6, alpha=0.95, zorder=3)

        ax_ins.scatter(c0[top_s1, 0], c0[top_s1, 1], c0[top_s1, 2], color=strand1_color,
                       edgecolor='black', linewidth=0.7, s=36, alpha=0.95, zorder=4)
        ax_ins.scatter(c0[top_s2, 0], c0[top_s2, 1], c0[top_s2, 2], color=strand2_color,
                       edgecolor='black', linewidth=0.7, s=36, alpha=0.95, zorder=4)

        ax_ins.set_box_aspect([1, 1, 2.4])
        ax_ins.view_init(elev=15, azim=35)
        ax_ins.set_xlim([-0.6, 0.6])
        ax_ins.set_ylim([-0.6, 0.6])
        ax_ins.set_zlim([z_top_min, z_top_max])
        ax_ins.set_xticks([])
        ax_ins.set_yticks([])
        ax_ins.set_zticks([])
        ax_ins.set_xlabel('')
        ax_ins.set_ylabel('')
        ax_ins.set_zlabel('')
        ins_title = r'\textbf{Top inset zoom}' if usetex else 'Top inset zoom'
        ax_ins.set_title(ins_title, fontsize=16, pad=6, weight='bold' if not usetex else None)

    # Subplot 2: Top View (Axial projection)
    ax2 = fig.add_axes([0.53, 0.22, 0.43, 0.64])
    ax2.set_aspect('equal')
    ax2.grid(True, linestyle=':', alpha=0.55)

    # Cross-rungs: dashed lines
    for r in range(n_rungs):
        p1, p2 = 2 * r, 2 * r + 1
        ax2.plot([c0[p1, 0], c0[p2, 0]], [c0[p1, 1], c0[p2, 1]],
                 color=rung_color, ls='--', lw=0.9, alpha=0.5, zorder=2)

    # Both strands continuous solid lines
    ax2.plot(c0[strand1, 0], c0[strand1, 1], color=strand1_color, ls='-', lw=2.4, alpha=0.95, zorder=3)
    ax2.plot(c0[strand2, 0], c0[strand2, 1], color=strand2_color, ls='-', lw=2.4, alpha=0.95, zorder=3)

    ax2.scatter(c0[strand1, 0], c0[strand1, 1], color=strand1_color,
                edgecolor='black', linewidth=0.6, s=24, alpha=0.95, zorder=4)
    ax2.scatter(c0[strand2, 0], c0[strand2, 1], color=strand2_color,
                edgecolor='black', linewidth=0.6, s=24, alpha=0.95, zorder=4)

    ax2.set_xlabel(r'$x$', labelpad=10, fontsize=20)
    ax2.set_ylabel(r'$y$', labelpad=10, fontsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=16)

    ax2.set_xlim([-0.65, 0.65])
    ax2.set_ylim([-0.65, 0.65])
    ax2.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
    ax2.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])

    for spine in ax2.spines.values():
        spine.set_color('#cccccc')

    # Subplot titles placed cleanly via fig.text
    title_a = r'\textbf{(a) Side-on view}' if usetex else '(a) Side-on view'
    title_b = r'\textbf{(b) Top view (axial projection)}' if usetex else '(b) Top view (axial projection)'
    fig.text(0.18, 0.89, title_a, ha='center', fontsize=20, weight='bold' if not usetex else None)
    fig.text(0.74, 0.89, title_b, ha='center', fontsize=20, weight='bold' if not usetex else None)

    # Shared horizontal legend across both panels, placed at the bottom below subplots
    legend_elements = [
        Line2D([0], [0], color=strand1_color, ls='-', lw=2.2, marker='o', markeredgecolor='black',
               markersize=6.5, label=f'Helix strand 1 ({len(strand1)} atoms)'),
        Line2D([0], [0], color=strand2_color, ls='-', lw=2.2, marker='o', markeredgecolor='black',
               markersize=6.5, label=f'Helix strand 2 ({len(strand2)} atoms)'),
        Line2D([0], [0], color=rung_color, ls='--', lw=1.4, alpha=0.8,
               label=f'Cross-rungs ({n_rungs} rungs)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.05),
               ncol=3, fontsize=18, framealpha=0.95, facecolor='white', edgecolor='#cccccc')

    sup_title = r'\textbf{Initial configuration of the constrained lattice grid ($t = 0$)}' if usetex else r'Initial configuration of the constrained lattice grid ($t = 0$)'
    fig.suptitle(sup_title, fontsize=22, y=0.96, weight='bold' if not usetex else None)

    if filename is None:
        target_folder = data_folder or getattr(MechSystem, 'data_folder', 'data')
        filename = os.path.join(target_folder, 'lattice_initial_configuration.pdf')
    elif data_folder is not None and not os.path.isabs(filename) and not os.path.dirname(filename):
        filename = os.path.join(data_folder, filename)

    fig_data = {
        'coords_t0': c0,
        'n_particles': n_particles,
        'n_rungs': n_rungs,
        'strand1_indices': strand1,
        'strand2_indices': strand2,
        'strand1_coords': c0[strand1],
        'strand2_coords': c0[strand2],
    }
    save_figure(fig, filename, fig_data=fig_data)
    plt.close(fig)
    return fig, [ax1, ax2]


def plot_initial_lattice_cylindrical(coords=None, radius=0.5, pitch=15.0, nosc=540,
                                     turns=0.5, filename=None, data_folder=None,
                                     rung_color='#555555',
                                     strand1_color=OKABE_ITO_PALETTE[4],
                                     strand2_color=OKABE_ITO_PALETTE[5],
                                     show_inset=True,
                                     circular_aspect_multiplier=1.5):
    """
    Generate and save a publication-quality figure of the initial lattice configuration (t = 0):
    - Side-on view with a translucent cylinder (r = 0.5), no x and y axes,
      centerline z-axis with origin O and tick mark at z = 8, cross-rungs, and helical backbones.
    - Top inset zoom highlighting nearest-neighbor bonds and rungs on the cylinder surface.
    - Sentence case titles, no physical units, LaTeX typography.

    Parameters
    ----------
    coords : np.ndarray or Solver instance or dict, optional
        Simulation coordinates or state. If None, generated using radius, pitch, nosc, turns.
    radius : float, optional
        Radius of the cylinder base. Default is 0.5.
    pitch : float, optional
        Pitch of the double helix. Default is 15.0 (scaled with nosc=540).
    nosc : int, optional
        Number of oscillators (default 540).
    turns : float, optional
        Angular turns of the helix strands around the cylinder.
        0.5 corresponds to half turn (Delta theta = pi, matching System.py physics).
        1.0 corresponds to full turn (Delta theta = 2*pi). Default is 0.5.
    filename : str, optional
        Target file path (.pdf).
    data_folder : str, optional
        Folder to save the output file if filename is not absolute.
    rung_color : str, optional
        Color for the cross-rungs. Default is '#555555'.
    strand1_color : str, optional
        Color for Helix Strand 1. Default is Okabe-Ito Blue ('#0072B2').
    strand2_color : str, optional
        Color for Helix Strand 2. Default is Okabe-Ito Vermillion ('#D55E00').
    show_inset : bool, optional
        Whether to display the top inset zoom. Default is True.

    Returns
    -------
    fig, axes : tuple
        Matplotlib Figure and list of Axes.
    """
    from System import MechSystem

    # Check LaTeX rendering capability
    usetex = plt.rcParams.get('text.usetex', False) or (shutil.which('latex') is not None)
    if usetex and not plt.rcParams.get('text.usetex', False):
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

    if coords is not None:
        if isinstance(coords, dict):
            c_all = coords.get('coords', coords.get('coords_t0'))
            if c_all is None:
                raise KeyError("Dictionary must contain 'coords' or 'coords_t0'")
            c0 = c_all[0] if c_all.ndim == 3 else c_all
        elif hasattr(coords, 'coords') and coords.coords is not None:
            c0 = coords.coords[0] if coords.coords.ndim == 3 else coords.coords
        elif hasattr(coords, 'y') and coords.y is not None:
            nosc_actual = getattr(coords, 'nosc', coords.y.shape[1] // 2 * 3)
            y0 = coords.y[0, :nosc_actual]
            c0 = y0.reshape(-1, 3)
        elif isinstance(coords, np.ndarray):
            c0 = coords[0] if coords.ndim == 3 else coords
        else:
            raise ValueError("Unsupported format for coords.")

        c0 = c0.copy()
        n_particles = c0.shape[0]
        n_rungs = n_particles // 2

        # Scale xy radius to specified radius if needed
        r_curr = np.sqrt(c0[:, 0]**2 + c0[:, 1]**2)
        mask = r_curr > 1e-12
        if not np.allclose(r_curr[mask], radius, atol=1e-3):
            c0[mask, 0] = c0[mask, 0] / r_curr[mask] * radius
            c0[mask, 1] = c0[mask, 1] / r_curr[mask] * radius

        pos1 = c0[::2]
        pos2 = c0[1::2]
        z_max = c0[:, 2].max()
    else:
        n_particles = nosc // 3
        n_rungs = n_particles // 2
        _i = np.arange(n_rungs)
        if np.isclose(turns, 0.5):
            t_arr = _i * 2 * np.pi / n_particles
        else:
            t_arr = _i * (2 * np.pi * turns) / n_rungs

        z_arr = pitch * t_arr / (2 * np.pi)
        z_max = z_arr[-1]

        pos1 = np.stack((radius * np.cos(t_arr), radius * np.sin(t_arr), z_arr), axis=1)
        pos2 = np.stack((radius * np.cos(t_arr + np.pi), radius * np.sin(t_arr + np.pi), z_arr), axis=1)

        c0 = np.zeros((n_particles, 3))
        c0[::2] = pos1
        c0[1::2] = pos2

    strand1 = np.arange(0, n_particles, 2)
    strand2 = np.arange(1, n_particles, 2)

    top_rungs_count = 10 if n_rungs >= 12 else max(4, n_rungs // 2)
    top_start_rung = n_rungs - top_rungs_count
    top_indices = np.arange(top_start_rung * 2, n_particles)
    z_top_min = c0[top_indices, 2].min() - 0.12
    z_top_max = c0[top_indices, 2].max() + 0.12

    fig = plt.figure(figsize=(15, 11))
    fig.set_layout_engine(None)

    # Subplot 1: Main side-on view (x and y axes removed, floating cylindrical frame)
    ax1 = fig.add_axes([0.04, 0.18, 0.48, 0.70], projection='3d')
    ax1.set_proj_type('ortho')
    ax1.set_axis_off()

    # 1. Cylinder surface
    theta_dense = np.linspace(0, 2 * np.pi, 80)
    z_dense = np.linspace(0, z_max, 40)
    Theta_cyl, Z_cyl = np.meshgrid(theta_dense, z_dense)
    X_cyl = radius * np.cos(Theta_cyl)
    Y_cyl = radius * np.sin(Theta_cyl)
    ax1.plot_surface(X_cyl, Y_cyl, Z_cyl, color='#eef4fb', alpha=0.18, edgecolor='none', zorder=1)

    # Latitude rings at z levels
    for zl in [2, 4, 6]:
        if zl <= z_max:
            ax1.plot(radius * np.cos(theta_dense), radius * np.sin(theta_dense), np.full_like(theta_dense, zl),
                     color='#90a4ae', ls=':', lw=0.9, alpha=0.6, zorder=1)

    # Base circle at z = 0 and top rim at z = z_max
    ax1.plot(radius * np.cos(theta_dense), radius * np.sin(theta_dense), np.zeros_like(theta_dense),
             color='#37474f', ls='-', lw=1.6, alpha=0.9, zorder=2)
    ax1.plot(radius * np.cos(theta_dense), radius * np.sin(theta_dense), np.full_like(theta_dense, z_max),
             color='#78909c', ls='--', lw=1.2, alpha=0.8, zorder=2)

    # 2. Centerline z-axis from z = 0 to z = 8.55
    z_axis_top = max(8.55, z_max + 1.1)
    ax1.plot([0, 0], [0, 0], [0, z_axis_top], color='#2c3e50', ls='-', lw=1.6, zorder=3)

    # Mark bottom of cylinder by O (for origin)
    ax1.scatter([0], [0], [0], color='#2c3e50', s=26, zorder=6)
    azim_rad = np.radians(36)
    tan_a = np.tan(azim_rad)

    # Position O cleanly to the right of origin on the cylinder base
    dx_O = -0.05
    dy_O = -dx_O / tan_a
    ax1.text(dx_O, dy_O, 0, r'$O$', fontsize=18, color='#2c3e50', weight='bold', ha='left', va='center')

    # Place a z-tick at z = 8 on the centerline z-axis
    dx_tick = 0.035
    dy_tick = -dx_tick / tan_a
    ax1.plot([-dx_tick, dx_tick], [-dy_tick, dy_tick], [8, 8], color='#2c3e50', ls='-', lw=1.8, zorder=5)

    # Label 8 to the left of the tick mark
    dx_8 = 0.08
    dy_8 = -dx_8 / tan_a
    ax1.text(dx_8, dy_8, 8, r'$8$', fontsize=17, color='#2c3e50', weight='bold', ha='right', va='center',
             bbox=dict(boxstyle='square,pad=0.08', facecolor='white', edgecolor='none', alpha=0.85))

    # Axis label z at the top
    ax1.text(0, 0, z_axis_top + 0.18, r'$z$', fontsize=20, color='#2c3e50', weight='bold', ha='center', va='bottom')

    # 3. Cross-rungs: dashed lines
    for r in range(n_rungs):
        p1, p2 = 2 * r, 2 * r + 1
        ax1.plot([c0[p1, 0], c0[p2, 0]],
                 [c0[p1, 1], c0[p2, 1]],
                 [c0[p1, 2], c0[p2, 2]],
                 color=rung_color, ls='--', lw=0.9, alpha=0.6, zorder=2)

    # 4. Helical backbones: continuous solid lines
    ax1.plot(pos1[:, 0], pos1[:, 1], pos1[:, 2], color=strand1_color, ls='-', lw=2.4, alpha=0.95, zorder=4)
    ax1.plot(pos2[:, 0], pos2[:, 1], pos2[:, 2], color=strand2_color, ls='-', lw=2.4, alpha=0.95, zorder=4)

    # 5. Atoms
    ax1.scatter(pos1[:, 0], pos1[:, 1], pos1[:, 2], color=strand1_color,
                edgecolor='black', linewidth=0.35, s=15, alpha=0.95, zorder=5)
    ax1.scatter(pos2[:, 0], pos2[:, 1], pos2[:, 2], color=strand2_color,
                edgecolor='black', linewidth=0.35, s=15, alpha=0.95, zorder=5)

    ax1.set_box_aspect([circular_aspect_multiplier, circular_aspect_multiplier, 3.2])
    ax1.view_init(elev=16, azim=36)
    lim_xy = 1.3 * radius
    ax1.set_xlim([-lim_xy, lim_xy])
    ax1.set_ylim([-lim_xy, lim_xy])
    ax1.set_zlim([-0.2, z_axis_top + 0.35])

    # Helper function to generate 3D spring geometry between two points
    def _make_3d_spring(p1, p2, n_coils=3.0, r_coil=0.018, n_pts=60):
        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        disp = p2 - p1
        length = np.linalg.norm(disp)
        if length < 1e-9:
            return np.repeat(p1[None, :], n_pts, axis=0)
        u = disp / length
        ref = np.array([0.0, 0.0, 1.0]) if abs(u[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        n1 = np.cross(u, ref)
        n1 /= np.linalg.norm(n1)
        n2 = np.cross(u, n1)

        t = np.linspace(0, 1, n_pts)
        lead = 0.10
        env = np.ones_like(t)
        mask1 = t < lead
        mask2 = t > (1 - lead)
        env[mask1] = np.sin(np.pi / 2 * t[mask1] / lead)
        env[mask2] = np.sin(np.pi / 2 * (1 - t[mask2]) / lead)

        phase = 2 * np.pi * n_coils * (t - lead) / (1 - 2 * lead)
        phase[mask1] = 0
        phase[mask2] = 2 * np.pi * n_coils

        pts = (p1[None, :] + t[:, None] * disp[None, :]
               + r_coil * env[:, None] * (np.cos(phase)[:, None] * n1[None, :] + np.sin(phase)[:, None] * n2[None, :]))
        return pts

    axes_list = [ax1]

    # Subplot 2: Top inset zoom
    if show_inset and top_rungs_count > 0:
        # Cylindrical inset volume indicator in main side-on view (replacing rectangular cube)
        r_box = 1.15 * radius
        ax1.plot(r_box * np.cos(theta_dense), r_box * np.sin(theta_dense), np.full_like(theta_dense, z_top_min),
                 color='#E69F00', ls='--', lw=1.3, alpha=0.95, zorder=6)
        ax1.plot(r_box * np.cos(theta_dense), r_box * np.sin(theta_dense), np.full_like(theta_dense, z_top_max),
                 color='#E69F00', ls='--', lw=1.3, alpha=0.95, zorder=6)

        for ang in [0, np.pi/2, np.pi, 3*np.pi/2]:
            ax1.plot([r_box * np.cos(ang), r_box * np.cos(ang)],
                     [r_box * np.sin(ang), r_box * np.sin(ang)],
                     [z_top_min, z_top_max],
                     color='#E69F00', ls='--', lw=1.2, alpha=0.95, zorder=6)

        z_box_mesh = np.linspace(z_top_min, z_top_max, 15)
        Th_b, Z_b = np.meshgrid(theta_dense, z_box_mesh)
        ax1.plot_surface(r_box * np.cos(Th_b), r_box * np.sin(Th_b), Z_b,
                         color='#FFF8E1', alpha=0.15, edgecolor='none', zorder=5)

        # Inset 3D axes (rectangular axes and panes removed)
        ax_ins = fig.add_axes([0.54, 0.22, 0.42, 0.62], projection='3d')
        ax_ins.set_proj_type('ortho')
        ax_ins.set_axis_off()

        z_grid_ins = np.linspace(z_top_min, z_top_max, 25)
        Theta_ins, Z_ins = np.meshgrid(theta_dense, z_grid_ins)
        ax_ins.plot_surface(radius * np.cos(Theta_ins), radius * np.sin(Theta_ins), Z_ins,
                            color='#eef4fb', alpha=0.18, edgecolor='none', zorder=1)

        ax_ins.plot(radius * np.cos(theta_dense), radius * np.sin(theta_dense), np.full_like(theta_dense, z_top_min),
                    color='#90a4ae', ls=':', lw=0.9, alpha=0.6, zorder=1)
        ax_ins.plot(radius * np.cos(theta_dense), radius * np.sin(theta_dense), np.full_like(theta_dense, z_top_max),
                    color='#90a4ae', ls=':', lw=0.9, alpha=0.6, zorder=1)

        # Cross-rungs in inset: dashed lines
        for r in range(top_start_rung, n_rungs):
            p1, p2 = 2 * r, 2 * r + 1
            ax_ins.plot([c0[p1, 0], c0[p2, 0]],
                        [c0[p1, 1], c0[p2, 1]],
                        [c0[p1, 2], c0[p2, 2]],
                        color=rung_color, ls='--', lw=1.3, alpha=0.7, zorder=2)

        top_s1 = [i for i in range(0, n_particles, 2) if i in top_indices]
        top_s2 = [i for i in range(1, n_particles, 2) if i in top_indices]

        # Strand bonds as tiny 3D springs in inset zoom
        for k in range(len(top_s1) - 1):
            pt_a = c0[top_s1[k]]
            pt_b = c0[top_s1[k + 1]]
            sp_pts = _make_3d_spring(pt_a, pt_b, n_coils=3.0, r_coil=0.018 * (radius / 0.5), n_pts=60)
            ax_ins.plot(sp_pts[:, 0], sp_pts[:, 1], sp_pts[:, 2],
                        color=strand1_color, ls='-', lw=2.0, alpha=0.95, zorder=3)

        for k in range(len(top_s2) - 1):
            pt_a = c0[top_s2[k]]
            pt_b = c0[top_s2[k + 1]]
            sp_pts = _make_3d_spring(pt_a, pt_b, n_coils=3.0, r_coil=0.018 * (radius / 0.5), n_pts=60)
            ax_ins.plot(sp_pts[:, 0], sp_pts[:, 1], sp_pts[:, 2],
                        color=strand2_color, ls='-', lw=2.0, alpha=0.95, zorder=3)

        # Atoms in inset
        ax_ins.scatter(c0[top_s1, 0], c0[top_s1, 1], c0[top_s1, 2], color=strand1_color,
                       edgecolor='black', linewidth=0.7, s=36, alpha=0.95, zorder=4)
        ax_ins.scatter(c0[top_s2, 0], c0[top_s2, 1], c0[top_s2, 2], color=strand2_color,
                       edgecolor='black', linewidth=0.7, s=36, alpha=0.95, zorder=4)

        ax_ins.set_box_aspect([circular_aspect_multiplier, circular_aspect_multiplier, 2.0])
        ax_ins.view_init(elev=16, azim=36)
        ins_lim = 1.15 * radius
        ax_ins.set_xlim([-ins_lim, ins_lim])
        ax_ins.set_ylim([-ins_lim, ins_lim])
        ax_ins.set_zlim([z_top_min, z_top_max])
        axes_list.append(ax_ins)

    # Panel titles
    title_main = r'\textbf{Side-on view}' if usetex else 'Side-on view'
    title_ins = r'\textbf{Top inset zoom}' if usetex else 'Top inset zoom'
    fig.text(0.28, 0.90, title_main, ha='center', fontsize=20, weight='bold' if not usetex else None)
    if show_inset and top_rungs_count > 0:
        fig.text(0.75, 0.90, title_ins, ha='center', fontsize=20, weight='bold' if not usetex else None)

    # Shared Legend
    legend_elements = [
        Line2D([0], [0], color=strand1_color, ls='-', lw=2.2, marker='o', markeredgecolor='black',
               markersize=6.5, label=f'Helix strand 1 ({len(pos1)} atoms)'),
        Line2D([0], [0], color=strand2_color, ls='-', lw=2.2, marker='o', markeredgecolor='black',
               markersize=6.5, label=f'Helix strand 2 ({len(pos2)} atoms)'),
        Line2D([0], [0], color=rung_color, ls='--', lw=1.4, alpha=0.8,
               label=f'Cross-rungs ({n_rungs} rungs)'),
        Line2D([0], [0], color='#37474f', ls='-', lw=1.5,
               label=rf'Cylinder base ($r = {radius}$)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.05),
               ncol=4, fontsize=15, framealpha=0.95, facecolor='white', edgecolor='#cccccc')

    # Figure title without "in cylindrical coordinates"
    sup_title = r'\textbf{Initial configuration of the constrained lattice grid ($t = 0$)}' if usetex else 'Initial configuration of the constrained lattice grid (t = 0)'
    fig.suptitle(sup_title, fontsize=22, y=0.96, weight='bold' if not usetex else None)

    if filename is None:
        target_folder = data_folder or getattr(MechSystem, 'data_folder', 'data')
        filename = os.path.join(target_folder, 'lattice_cylindrical_initial_configuration.pdf')
    elif data_folder is not None and not os.path.isabs(filename) and not os.path.dirname(filename):
        filename = os.path.join(data_folder, filename)

    fig_data = {
        'coords_t0': c0,
        'radius': radius,
        'turns': turns,
        'pitch': pitch,
        'circular_aspect_multiplier': circular_aspect_multiplier,
        'n_particles': n_particles,
        'n_rungs': n_rungs,
        'strand1_indices': strand1,
        'strand2_indices': strand2,
        'strand1_coords': c0[strand1],
        'strand2_coords': c0[strand2],
    }
    save_figure(fig, filename, fig_data=fig_data)
    plt.close(fig)
    return fig, axes_list


def plot_elastica_beam(solvers_f, solvers_r=None, solvers_hr=None, data_folder=None,
                       filename="elastica_beam_test_sanity_check.pdf", solver_idx=0, **kwargs):
    """
    Dedicated visualization for the Discrete Euler Elastica / Flexible Beam:
    - Panel (a): 3D perspective of dynamic beam centerline deformation over time,
      with FOM, ROM, and HROM compared at final state t = T.
    - Panel (b): 2D elevation profiles z(x) across time snapshots (t = T/2 and t = T).
    - Panel (c): Tip vertical deflection history z_tip(t) comparing FOM, ROM, and HROM.

    Color scheme & styling:
    - Standardized Okabe-Ito colorblind-accessible palette:
      FOM: Black solid ('-')
      ROM: Okabe-Ito Orange ('--')
      HROM: Okabe-Ito Sky Blue ('-.')
    - Tip trajectory: Okabe-Ito Vermillion ('--')
    - Enforced physical 3D lateral scaling to prevent spurious numerical roundoff magnification.
    - Saves publication-quality PDF and companion .data.pickle via save_figure.
    """
    if solvers_hr is None:
        solvers_hr = kwargs.get('solvers_dr', None)

    if data_folder is None:
        try:
            from System import MechSystem
            data_folder = getattr(MechSystem, 'data_folder', 'data')
        except ImportError:
            data_folder = 'data'
    os.makedirs(data_folder, exist_ok=True)

    if isinstance(solvers_f, (list, tuple)):
        sf = solvers_f[solver_idx] if len(solvers_f) > solver_idx else None
    else:
        sf = solvers_f

    if sf is None:
        print("[plot_elastica_beam] No FOM solver provided; skipping.")
        return

    if isinstance(solvers_r, (list, tuple)):
        sr = solvers_r[solver_idx] if len(solvers_r) > solver_idx else None
    else:
        sr = solvers_r

    if isinstance(solvers_hr, (list, tuple)):
        shr = solvers_hr[solver_idx] if len(solvers_hr) > solver_idx else None
    else:
        shr = solvers_hr

    t_points = sf.t_points
    nosc = sf.nosc
    n_nodes = nosc // 3
    coords_f = sf.y[:, :nosc].reshape(-1, n_nodes, 3)
    coords_r = sr.y[:, :nosc].reshape(-1, n_nodes, 3) if sr is not None and getattr(sr, 'y', None) is not None else None
    coords_hr = shr.y[:, :nosc].reshape(-1, n_nodes, 3) if shr is not None and getattr(shr, 'y', None) is not None else None

    n_steps = coords_f.shape[0]

    usetex = plt.rcParams.get('text.usetex', False)
    lbl_fom_T = r"FOM ($t = T$)" if usetex else "FOM (t = T)"
    lbl_rom_T = r"ROM ($t = T$)" if usetex else "ROM (t = T)"
    lbl_hrom_T = r"HROM ($t = T$)" if usetex else "HROM (t = T)"

    # --- Matplotlib Publication Figure ---
    fig = plt.figure(figsize=(18, 5.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 1], wspace=0.28)

    # Panel (a): 3D Dynamic Centerline Deformation
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    n_snaps = min(12, n_steps)
    snap_indices = np.linspace(0, n_steps - 1, n_snaps, dtype=int)

    # Pinned anchor marker at origin
    ax1.scatter([0], [0], [0], s=90, c='black', marker='s', label='Root pinned', zorder=10)

    # Intermediate snapshots of FOM centerline in subtle progression
    for s_idx in snap_indices[:-1]:
        ax1.plot(coords_f[s_idx, :, 0], coords_f[s_idx, :, 1], coords_f[s_idx, :, 2],
                 color='#999999', lw=1.2, ls='--', alpha=0.5)

    # Free tip trajectory in Okabe-Ito Vermillion
    ax1.plot(coords_f[:, -1, 0], coords_f[:, -1, 1], coords_f[:, -1, 2],
             color=OKABE_ITO_PALETTE[5], ls='--', lw=1.5, alpha=0.85, label='Tip trajectory')

    # Final beam state comparison at t = T
    ax1.plot(coords_f[-1, :, 0], coords_f[-1, :, 1], coords_f[-1, :, 2],
             color='black', lw=2.5, ls='-', label=lbl_fom_T)
    if coords_r is not None:
        ax1.plot(coords_r[-1, :, 0], coords_r[-1, :, 1], coords_r[-1, :, 2],
                 color=OKABE_ITO_PALETTE[0], lw=2.0, ls='--', label=lbl_rom_T)
    if coords_hr is not None:
        ax1.plot(coords_hr[-1, :, 0], coords_hr[-1, :, 1], coords_hr[-1, :, 2],
                 color=OKABE_ITO_PALETTE[1], lw=2.0, ls='-.', label=lbl_hrom_T)

    # Physical scaling limits for 3D box:
    # Ensure minimum physical lateral span of at least 0.4 m to avoid
    # numerical roundoff (~1e-18) in nominal in-plane motion being magnified.
    x_all = coords_f[:, :, 0]
    y_all = coords_f[:, :, 1]
    z_all = coords_f[:, :, 2]
    x_max = float(np.amax(x_all))
    z_min, z_max = float(np.amin(z_all)), float(np.amax(z_all))
    y_span = max(float(np.ptp(y_all)), 0.4)
    y_mid = float(np.mean(y_all))

    ax1.set_xlim([-0.05, max(x_max + 0.05, 1.05)])
    ax1.set_ylim([y_mid - y_span / 2.0, y_mid + y_span / 2.0])
    ax1.set_zlim([min(z_min - 0.05, -0.25), max(z_max + 0.05, 0.05)])
    ax1.set_box_aspect([1.1, 0.4, 0.3])

    ax1.set_xlabel('X (axial)', labelpad=8)
    ax1.set_ylabel('Y (lateral)', labelpad=8)
    ax1.set_zlabel('Z (vertical)', labelpad=8)
    ax1.set_title('(a) 3D Dynamic Centerline', fontsize=13, fontweight='bold')
    ax1.view_init(elev=20, azim=-60)
    ax1.legend(loc='upper right', fontsize=8)

    # Panel (b): 2D Elevation Profiles z(x)
    ax2 = fig.add_subplot(gs[0, 1])
    mid_idx = n_steps // 2
    t_mid = t_points[mid_idx]
    t_end = t_points[-1]

    lbl_fom_mid = rf"FOM ($t = {t_mid:.2f}$)" if usetex else f"FOM (t = {t_mid:.2f})"
    lbl_rom_mid = rf"ROM ($t = {t_mid:.2f}$)" if usetex else f"ROM (t = {t_mid:.2f})"
    lbl_hrom_mid = rf"HROM ($t = {t_mid:.2f}$)" if usetex else f"HROM (t = {t_mid:.2f})"
    lbl_fom_end = rf"FOM ($t = {t_end:.2f}$)" if usetex else f"FOM (t = {t_end:.2f})"
    lbl_rom_end = rf"ROM ($t = {t_end:.2f}$)" if usetex else f"ROM (t = {t_end:.2f})"
    lbl_hrom_end = rf"HROM ($t = {t_end:.2f}$)" if usetex else f"HROM (t = {t_end:.2f})"

    # Mid snapshot (t = T/2)
    ax2.plot(coords_f[mid_idx, :, 0], coords_f[mid_idx, :, 2],
             color='0.5', lw=1.6, ls='-', label=lbl_fom_mid)
    if coords_r is not None:
        ax2.plot(coords_r[mid_idx, :, 0], coords_r[mid_idx, :, 2],
                 color=OKABE_ITO_PALETTE[0], lw=1.4, ls='--', alpha=0.7, label=lbl_rom_mid)
    if coords_hr is not None:
        ax2.plot(coords_hr[mid_idx, :, 0], coords_hr[mid_idx, :, 2],
                 color=OKABE_ITO_PALETTE[1], lw=1.4, ls='-.', alpha=0.7, label=lbl_hrom_mid)

    # Final snapshot (t = T)
    ax2.plot(coords_f[-1, :, 0], coords_f[-1, :, 2],
             color='black', lw=2.2, ls='-', label=lbl_fom_end)
    if coords_r is not None:
        ax2.plot(coords_r[-1, :, 0], coords_r[-1, :, 2],
                 color=OKABE_ITO_PALETTE[0], lw=1.8, ls='--', label=lbl_rom_end)
    if coords_hr is not None:
        ax2.plot(coords_hr[-1, :, 0], coords_hr[-1, :, 2],
                 color=OKABE_ITO_PALETTE[1], lw=1.8, ls='-.', label=lbl_hrom_end)

    ax2.plot([0], [0], 'k^', markersize=10, label='Pinned base')
    ax2.axhline(0, color='gray', lw=0.8, ls=':')
    ax2.set_xlabel('Axial coordinate $x$', fontsize=11)
    ax2.set_ylabel('Vertical deflection $z$', fontsize=11)
    ax2.set_title('(b) Elevation Deflection Profiles $z(x)$', fontsize=13, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='lower left', fontsize=7.5)

    # Panel (c): Tip Deflection History z_tip(t)
    ax3 = fig.add_subplot(gs[0, 2])
    z_tip_f = coords_f[:, -1, 2]
    ax3.plot(t_points, z_tip_f, color='black', lw=2.2, ls='-', label='FOM')
    if coords_r is not None:
        z_tip_r = coords_r[:, -1, 2]
        ax3.plot(t_points, z_tip_r, color=OKABE_ITO_PALETTE[0], lw=1.8, ls='--', label='ROM')
    if coords_hr is not None:
        z_tip_hr = coords_hr[:, -1, 2]
        ax3.plot(t_points, z_tip_hr, color=OKABE_ITO_PALETTE[1], lw=1.8, ls='-.', label='HROM')

    ax3.set_xlabel('Time $t$', fontsize=11)
    ax3.set_ylabel(r'Tip vertical displacement $z_{\mathrm{tip}}$', fontsize=11)
    ax3.set_title('(c) Tip Displacement History', fontsize=13, fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend(loc='lower left', fontsize=9)

    err_text = []
    if coords_r is not None:
        err_tip_r = float(np.amax(np.abs(z_tip_f - z_tip_r)))
        err_text.append(f"ROM: {err_tip_r:.2e}")
    if coords_hr is not None:
        err_tip_hr = float(np.amax(np.abs(z_tip_f - z_tip_hr)))
        err_text.append(f"HROM: {err_tip_hr:.2e}")
    if err_text:
        box_str = "Max Tip Error:\n" + "\n".join(err_text)
        ax3.text(0.95, 0.95, box_str,
                 transform=ax3.transAxes, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='gray'),
                 fontsize=8)

    out_pdf = os.path.join(data_folder, filename) if not os.path.isabs(filename) else filename
    fig_data = {
        't_points': t_points,
        'coords_f': coords_f,
        'coords_r': coords_r,
        'coords_hr': coords_hr,
        'z_tip_f': z_tip_f,
        'z_tip_r': coords_r[:, -1, 2] if coords_r is not None else None,
        'z_tip_hr': coords_hr[:, -1, 2] if coords_hr is not None else None,
    }
    save_figure(fig, out_pdf, fig_data=fig_data)
    plt.close(fig)
    print(f"Saved dedicated Elastica beam plot to {out_pdf}")
    return fig



