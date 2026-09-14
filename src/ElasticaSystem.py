#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import hashlib
import json
import numpy as np
from datetime import datetime

from System import (SysConfig, MechSystem, get_data_dir, fast_load, fast_dump, 
                    check_checkpoint_exists)
from ConcreteSolvers import (BaseSolverMixin, ConformalStormerVerletFixedPointMixin,
                             ConformalStormerVerlet, REDUCED_SOLVER_MAPPING,
                             HYPERREDUCED_SOLVER_MAPPING)
from ReduceMechSystem import HamiltonianReducer

rng = np.random.default_rng(seed=267257368022227711484290921317604022527)


def get_symbolic_expressions_file_elastica(nosc):
    """Path to the cached symbolic expressions for Elastica."""
    filename = f"symbolic_expr_elastica_{nosc}_cse.pickle"
    scratch_file = get_data_dir(filename)
    persistent_file = os.path.join('data', filename)
    if not os.path.exists(scratch_file) and os.path.exists(persistent_file):
        return persistent_file
    return scratch_file


def load_symbolic_expressions_elastica(cls):
    """Class decorator to load cached elastica symbolic expressions."""
    try:
        expr_file = get_symbolic_expressions_file_elastica(cls.nosc)
        expressions = fast_load(expr_file)
    except Exception:
        return cls

    for k, v in expressions.items():
        if callable(v) and not isinstance(v, type):
            setattr(cls, k, staticmethod(v))
        else:
            setattr(cls, k, v)
    return cls


class ElasticaConfig(SysConfig):
    """
    Configuration parameters specifically for the Discrete Euler Elastica / Beam.
    """
    n_nodes = 21  # 21 nodes -> 20 segments
    nosc = n_nodes * 3  # 63 position DOFs in 3D
    length = 1.0
    mass_total = 1.0
    gravity = 9.81
    boundary = "pinned"  # 'pinned' or 'clamped'
    ds = length / (n_nodes - 1)

    # Time-stepping
    predict = True
    train_ratio = 0.7
    reducer = 'psd'
    hyperreducer = 'MDEIM'
    constraints_reduce = False

    tol = 1e-12
    tol_reduced = 1e-8
    M = 500
    store = False
    pod_tol_sweep = [1e-4]

    if predict:
        dt_space_dim = 1
        dt_space = np.array([0.001])
        T_final = dt_space[-1] * 3e3  # 3.0 seconds
        _Omega2_space_dim = 15  # 15 parameter samples
    else:
        dt_space_dim = 4
        dt_space = np.logspace(-4, -4 - dt_space_dim, num=dt_space_dim, base=2, endpoint=False)
        T_final = 0.1
        _Omega2_space_dim = 3

    # Number of bending joints: (n_nodes - 2)
    num_joints = n_nodes - 2
    # Bending stiffness parameter samples B_i in [2.0, 10.0]
    _Omega2_space = 2.0 + 8.0 * rng.random((_Omega2_space_dim, num_joints))
    _Omega2_space = np.sort(_Omega2_space, axis=1)

    if predict:
        split_idx = int(_Omega2_space_dim * train_ratio)
        Omega2_space = _Omega2_space[:split_idx]
        Omega2_space_test = _Omega2_space[split_idx:]
    else:
        Omega2_space = Omega2_space_test = _Omega2_space


    is_elastica = True


def make_elastica_initial_conditions(n_nodes, ds, boundary="pinned", transverse_velocity_scale=0.0):
    """
    Construct initial state for Euler Elastica satisfying position constraints
    g(q0) = 0 and velocity constraints d/dt g(q0, p0) = 0.
    """
    positions = np.zeros((n_nodes, 3))
    for i in range(n_nodes):
        positions[i, 0] = i * ds
        positions[i, 1] = 0.0
        positions[i, 2] = 0.0

    momenta = np.zeros((n_nodes, 3))
    if transverse_velocity_scale != 0.0:
        # Transverse perturbation in z-direction with zero axial velocity (p_x = 0)
        # This guarantees (q_i - q_{i-1}) . (p_i - p_{i-1}) = ds * (0 - 0) = 0 for all segments!
        for i in range(1, n_nodes):
            s_i = i / (n_nodes - 1)
            momenta[i, 2] = transverse_velocity_scale * np.sin(np.pi * s_i)
        if boundary == "clamped":
            momenta[1, 2] = 0.0

    q0 = positions.flatten()
    p0 = momenta.flatten()

    # 1. Assert position constraints: g(q0) = 0
    g_pos = []
    g_pos.extend(positions[0])  # Pinned root: q0 = [0, 0, 0]
    if boundary == "clamped":
        g_pos.extend([positions[1, 1], positions[1, 2]])  # Clamped: q1_y = 0, q1_z = 0
    for i in range(1, n_nodes):
        diff_q = positions[i] - positions[i - 1]
        g_pos.append(0.5 * (np.dot(diff_q, diff_q) - ds**2))
    g_pos = np.array(g_pos)
    assert np.allclose(g_pos, 0.0, atol=1e-12), (
        f"Position constraints g(q0) must be 0, max violation: {np.max(np.abs(g_pos)):.2e}"
    )

    # 2. Assert velocity constraints: d/dt g(q0, p0) = 0
    d_dt_g = []
    d_dt_g.extend(momenta[0])  # Root velocity: p0 = [0, 0, 0]
    if boundary == "clamped":
        d_dt_g.extend([momenta[1, 1], momenta[1, 2]])  # Clamped velocity: p1_y = 0, p1_z = 0
    for i in range(1, n_nodes):
        diff_q = positions[i] - positions[i - 1]
        diff_p = momenta[i] - momenta[i - 1]
        d_dt_g.append(np.dot(diff_q, diff_p))
    d_dt_g = np.array(d_dt_g)
    assert np.allclose(d_dt_g, 0.0, atol=1e-12), (
        f"Velocity constraints d/dt g(q0, p0) must be 0, max violation: {np.max(np.abs(d_dt_g)):.2e}"
    )

    # Axial velocity is zero, so q0 . p0 == 0 is also satisfied
    assert np.isclose(np.dot(q0, p0), 0.0, atol=1e-12), f"q0 . p0 must be 0, got {np.dot(q0, p0)}"
    return np.r_[q0, p0]


class ElasticaMechSystem(ElasticaConfig, MechSystem):
    """
    Mechanical system representing a 3D Discrete Euler Elastica / Flexible Beam.
    """
    is_elastica = True
    keep_time = os.environ.get("JOB_KEEP_TIME") or datetime.now().strftime("%Y-%m-%d")
    data_folder = get_data_dir(os.path.join(keep_time, "elastica"))
    os.makedirs(data_folder, exist_ok=True)

    # Initial state satisfying g(q0) = 0 and G(q0) p0 = 0 (q0 . p0 = 0)
    y_init = make_elastica_initial_conditions(ElasticaConfig.n_nodes, ElasticaConfig.ds, ElasticaConfig.boundary)

    def __init__(self, kwds):
        super().__init__(kwds)
        # Ensure JJ matches the elastica nosc
        self.JJ = self.compute_J(self.nosc_r if hasattr(self, 'RB') else self.nosc)


@load_symbolic_expressions_elastica
class HamiltonianElasticaMechSystem(ElasticaMechSystem):
    """Full-order Hamiltonian Elastica system."""
    def __init__(self, kwds):
        super().__init__(kwds)
        self.ham_z = self.ham_z_lambda
        self.ham_zz = self.ham_zz_lambda
        self.g__ = self.g__lambda
        self.g_prime__ = self.g_prime__lambda


class ReducedHamiltonianElasticaMechSystem(HamiltonianElasticaMechSystem):
    """Reduced-order Hamiltonian Elastica system."""
    def __init__(self, kwds):
        if 'solver_data' in kwds and self.__class__.__name__ in kwds['solver_data']:
            for k, v in kwds['solver_data'][self.__class__.__name__].items():
                setattr(self, k, v)
        super().__init__(kwds)
        self.tol = self.tol_reduced
        self.ham_z = self.ham_z_reduced
        self.ham_zz = self.ham_zz_reduced
        self.g__ = self.g_reduced
        self.g_prime__ = self.g_prime_reduced


class HyperReducedHamiltonianElasticaMechSystem(ReducedHamiltonianElasticaMechSystem):
    """Hyper-reduced order Hamiltonian Elastica system."""
    def __init__(self, kwds):
        super().__init__(kwds)
        self.ham_z = self.ham_z_hyperreduced
        self.ham_zz = self.ham_zz_hyperreduced
        if self.hyperreducer == 'MDEIM':
            self.ham_zz = self.ham_zz_mdeim_hyperreduced
        if self.constraints_reduce:
            self.g_prime__ = self.g_prime_hyperreduced


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------

class ElasticaStormerVerletSolver(BaseSolverMixin, ConformalStormerVerletFixedPointMixin,
                                 ConformalStormerVerlet, HamiltonianElasticaMechSystem):
    """Full-order Stormer-Verlet (RATTLE) solver for Elastica."""
    pass


class ReducedElasticaStormerVerletSolver(BaseSolverMixin, ConformalStormerVerletFixedPointMixin,
                                        ConformalStormerVerlet, ReducedHamiltonianElasticaMechSystem):
    """Reduced-order Stormer-Verlet (RATTLE) solver for Elastica."""
    pass


class HyperReducedElasticaStormerVerletSolver(BaseSolverMixin, ConformalStormerVerletFixedPointMixin,
                                             ConformalStormerVerlet, HyperReducedHamiltonianElasticaMechSystem):
    """Hyper-reduced order Stormer-Verlet (RATTLE) solver for Elastica."""
    pass


ELASTICA_REDUCED_SOLVER_MAPPING = {
    ElasticaStormerVerletSolver: ReducedElasticaStormerVerletSolver,
    ReducedElasticaStormerVerletSolver: ReducedElasticaStormerVerletSolver,
    HyperReducedElasticaStormerVerletSolver: ReducedElasticaStormerVerletSolver,
}

ELASTICA_HYPERREDUCED_SOLVER_MAPPING = {
    ElasticaStormerVerletSolver: HyperReducedElasticaStormerVerletSolver,
    ReducedElasticaStormerVerletSolver: HyperReducedElasticaStormerVerletSolver,
    HyperReducedElasticaStormerVerletSolver: HyperReducedElasticaStormerVerletSolver,
}

# Update global mapping dictionaries
REDUCED_SOLVER_MAPPING.update(ELASTICA_REDUCED_SOLVER_MAPPING)
HYPERREDUCED_SOLVER_MAPPING.update(ELASTICA_HYPERREDUCED_SOLVER_MAPPING)


class HamiltonianElasticaReducer(HamiltonianReducer, HamiltonianElasticaMechSystem):
    """Reducer tailored for Hamiltonian Elastica MechSystem."""
    pass


def plot_elastica_beam(solvers_f, solvers_r=None, solvers_dr=None, data_folder=None,
                       filename="elastica_beam_3d.pdf", solver_idx=0):
    """
    Dedicated visualization for the Discrete Euler Elastica / Flexible Beam.

    Generates:
      1. High-resolution publication PDF figure:
         - Panel (a): 3D perspective of dynamic beam centerline deformation over time
         - Panel (b): 2D elevation profiles z(x) across time snapshots
         - Panel (c): Tip vertical deflection history z_tip(t) comparing FOM, ROM, and HRM
      2. Interactive 3D HTML visualization via Plotly (elastica_beam_3d.html)
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as mcm
    import matplotlib.colors as mcolors

    if data_folder is None:
        data_folder = ElasticaMechSystem.data_folder
    os.makedirs(data_folder, exist_ok=True)

    sf = solvers_f[solver_idx] if solvers_f and len(solvers_f) > solver_idx else None
    if sf is None:
        print("[plot_elastica_beam] No FOM solver provided; skipping.")
        return

    sr = solvers_r[solver_idx] if solvers_r and len(solvers_r) > solver_idx else None
    sdr = solvers_dr[solver_idx] if solvers_dr and len(solvers_dr) > solver_idx else None

    t_points = sf.t_points
    nosc = sf.nosc
    n_nodes = nosc // 3
    coords_f = sf.y[:, :nosc].reshape(-1, n_nodes, 3)
    coords_r = sr.y[:, :nosc].reshape(-1, n_nodes, 3) if sr is not None and getattr(sr, 'y', None) is not None else None
    coords_dr = sdr.y[:, :nosc].reshape(-1, n_nodes, 3) if sdr is not None and getattr(sdr, 'y', None) is not None else None

    n_steps = coords_f.shape[0]

    # --- Matplotlib Publication Figure ---
    fig = plt.figure(figsize=(18, 5.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 1], wspace=0.28)

    # Panel (a): 3D Dynamic Centerline Deformation
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    n_snaps = min(12, n_steps)
    snap_indices = np.linspace(0, n_steps - 1, n_snaps, dtype=int)
    cmap = mcm.get_cmap('viridis')
    norm = mcolors.Normalize(vmin=t_points[0], vmax=t_points[-1])

    # Pinned anchor marker at origin
    ax1.scatter([0], [0], [0], s=90, c='black', marker='s', label='Root pinned', zorder=10)

    for s_idx in snap_indices:
        t_val = t_points[s_idx]
        color = cmap(norm(t_val))
        ax1.plot(coords_f[s_idx, :, 0], coords_f[s_idx, :, 1], coords_f[s_idx, :, 2],
                 color=color, lw=2.2, alpha=0.8)
        ax1.scatter(coords_f[s_idx, :, 0], coords_f[s_idx, :, 1], coords_f[s_idx, :, 2],
                    color=color, s=10, alpha=0.85)

    # Free tip trajectory
    ax1.plot(coords_f[:, -1, 0], coords_f[:, -1, 1], coords_f[:, -1, 2],
             'r--', lw=1.5, alpha=0.8, label='Tip trajectory')

    # Final beam state comparison at t = T_final
    ax1.plot(coords_f[-1, :, 0], coords_f[-1, :, 1], coords_f[-1, :, 2],
             color='navy', lw=3.0, label='FOM ($t=T$)')
    if coords_r is not None:
        ax1.plot(coords_r[-1, :, 0], coords_r[-1, :, 1], coords_r[-1, :, 2],
                 color='forestgreen', lw=2.2, ls='--', label='ROM ($t=T$)')
    if coords_dr is not None:
        ax1.plot(coords_dr[-1, :, 0], coords_dr[-1, :, 1], coords_dr[-1, :, 2],
                 color='crimson', lw=2.2, ls=':', label='HRM ($t=T$)')

    ax1.set_xlabel('X (axial, m)', labelpad=8)
    ax1.set_ylabel('Y (lateral, m)', labelpad=8)
    ax1.set_zlabel('Z (vertical, m)', labelpad=8)
    ax1.set_title('(a) 3D Dynamic Centerline', fontsize=13, fontweight='bold')
    ax1.view_init(elev=20, azim=-60)
    ax1.legend(loc='upper right', fontsize=8)

    sm = mcm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(t_points)
    cbar = fig.colorbar(sm, ax=ax1, shrink=0.6, pad=0.14)
    cbar.set_label('Time (s)', fontsize=9)

    # Panel (b): 2D Elevation Profiles z(x)
    ax2 = fig.add_subplot(gs[0, 1])
    eval_snaps = np.linspace(0, n_steps - 1, min(5, n_steps), dtype=int)
    colors_snaps = ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725']
    for idx_color, s_idx in enumerate(eval_snaps):
        t_val = t_points[s_idx]
        col = colors_snaps[idx_color % len(colors_snaps)]
        ax2.plot(coords_f[s_idx, :, 0], coords_f[s_idx, :, 2],
                 color=col, lw=2.0, label=f't = {t_val:.2f}s')
        if coords_dr is not None:
            ax2.plot(coords_dr[s_idx, :, 0], coords_dr[s_idx, :, 2],
                     color=col, lw=1.5, ls='--')

    ax2.plot([0], [0], 'k^', markersize=10, label='Pinned base')
    ax2.axhline(0, color='gray', lw=0.8, ls=':')
    ax2.set_xlabel('Axial coordinate $x$ (m)', fontsize=11)
    ax2.set_ylabel('Vertical deflection $z$ (m)', fontsize=11)
    ax2.set_title('(b) Elevation Deflection Profiles $z(x)$', fontsize=13, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='lower left', fontsize=8)

    # Panel (c): Tip Deflection History z_tip(t)
    ax3 = fig.add_subplot(gs[0, 2])
    z_tip_f = coords_f[:, -1, 2]
    ax3.plot(t_points, z_tip_f, color='navy', lw=2.2, label='FOM')
    if coords_r is not None:
        z_tip_r = coords_r[:, -1, 2]
        ax3.plot(t_points, z_tip_r, color='forestgreen', lw=1.8, ls='--', label='ROM')
    if coords_dr is not None:
        z_tip_dr = coords_dr[:, -1, 2]
        ax3.plot(t_points, z_tip_dr, color='crimson', lw=1.8, ls=':', label='HRM')

    ax3.set_xlabel('Time $t$ (s)', fontsize=11)
    ax3.set_ylabel('Tip vertical displacement $z_{\\mathrm{tip}}$ (m)', fontsize=11)
    ax3.set_title('(c) Tip Displacement History', fontsize=13, fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend(loc='lower left', fontsize=9)

    if coords_dr is not None:
        err_tip_hrm = np.amax(np.abs(z_tip_f - coords_dr[:, -1, 2]))
        ax3.text(0.95, 0.95, f'Max HRM Tip Error:\n{err_tip_hrm:.2e} m',
                 transform=ax3.transAxes, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='gray'),
                 fontsize=8)

    out_pdf = os.path.join(data_folder, filename)
    fig.savefig(out_pdf, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved dedicated Elastica beam plot to {out_pdf}")

    # --- Interactive Plotly 3D HTML Animation ---
    try:
        import plotly.graph_objects as go
        fig_ply = go.Figure()

        # Static tip trajectory trace
        fig_ply.add_trace(go.Scatter3d(
            x=coords_f[:, -1, 0], y=coords_f[:, -1, 1], z=coords_f[:, -1, 2],
            mode='lines', line=dict(color='red', width=3, dash='dash'),
            name='Tip path (FOM)'
        ))

        # Root support marker
        fig_ply.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode='markers', marker=dict(size=8, color='black', symbol='square'),
            name='Pinned Root'
        ))

        # Initial frame (t=0)
        fig_ply.add_trace(go.Scatter3d(
            x=coords_f[0, :, 0], y=coords_f[0, :, 1], z=coords_f[0, :, 2],
            mode='lines+markers', line=dict(color='#21918c', width=6),
            marker=dict(size=4, color='#3b528b'),
            name='Beam Centerline (FOM)'
        ))
        if coords_dr is not None:
            fig_ply.add_trace(go.Scatter3d(
                x=coords_dr[0, :, 0], y=coords_dr[0, :, 1], z=coords_dr[0, :, 2],
                mode='lines+markers', line=dict(color='#e7298a', width=4, dash='dot'),
                marker=dict(size=3, color='#e7298a'),
                name='Beam Centerline (HRM)'
            ))

        # Add time snapshot frames
        frames = []
        n_frames = min(25, n_steps)
        frame_indices = np.linspace(0, n_steps - 1, n_frames, dtype=int)

        for k in frame_indices:
            data_frame = [
                go.Scatter3d(
                    x=coords_f[:, -1, 0], y=coords_f[:, -1, 1], z=coords_f[:, -1, 2],
                    mode='lines', line=dict(color='red', width=3, dash='dash')
                ),
                go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(size=8, color='black', symbol='square')),
                go.Scatter3d(
                    x=coords_f[k, :, 0], y=coords_f[k, :, 1], z=coords_f[k, :, 2],
                    mode='lines+markers', line=dict(color='#21918c', width=6),
                    marker=dict(size=4, color='#3b528b')
                ),
            ]
            if coords_dr is not None:
                data_frame.append(go.Scatter3d(
                    x=coords_dr[k, :, 0], y=coords_dr[k, :, 1], z=coords_dr[k, :, 2],
                    mode='lines+markers', line=dict(color='#e7298a', width=4, dash='dot'),
                    marker=dict(size=3, color='#e7298a')
                ))
            frames.append(go.Frame(data=data_frame, name=f"t={t_points[k]:.3f}s"))

        fig_ply.frames = frames

        sliders = [{
            'steps': [{'args': [[f.name], {'frame': {'duration': 50, 'redraw': True}, 'mode': 'immediate'}],
                       'label': f.name, 'method': 'animate'} for f in frames],
            'transition': {'duration': 50},
            'x': 0.1, 'y': 0, 'len': 0.85
        }]
        updatemenus = [{
            'type': 'buttons',
            'buttons': [
                {'args': [None, {'frame': {'duration': 50, 'redraw': True}, 'fromcurrent': True}],
                 'label': 'Play', 'method': 'animate'},
                {'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}],
                 'label': 'Pause', 'method': 'animate'}
            ],
            'direction': 'left', 'pad': {'r': 10, 't': 10},
            'showactive': False, 'type': 'buttons', 'x': 0.1, 'y': 0.15
        }]

        fig_ply.update_layout(
            title=f"3D Euler Elastica Flexible Beam Dynamics ({n_nodes} nodes, L={ElasticaConfig.length}m)",
            scene=dict(
                xaxis_title='X (axial, m)',
                yaxis_title='Y (lateral, m)',
                zaxis_title='Z (elevation, m)',
                aspectmode='data'
            ),
            updatemenus=updatemenus,
            sliders=sliders
        )

        out_html = os.path.join(data_folder, filename.replace('.pdf', '.html'))
        fig_ply.write_html(out_html)
        print(f"Saved interactive 3D Elastica beam animation to {out_html}")
    except Exception as e:
        print(f"Could not create interactive Plotly HTML: {e}")
