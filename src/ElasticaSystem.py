#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from datetime import datetime

from System import (SysConfig, MechSystem, HamiltonianMechSystem,
                    ReducedHamiltonianMechSystem, HyperReducedHamiltonianMechSystem,
                    get_data_dir, fast_load)
from ConcreteSolvers import (BaseSolverMixin, ReducedSolverMixin, HyperReducedSolverMixin,
                             ConformalStormerVerletFixedPointMixin,
                             ConformalStormerVerlet, REDUCED_SOLVER_MAPPING,
                             HYPERREDUCED_SOLVER_MAPPING)
from ReduceMechSystem import HamiltonianReducer

np.random.seed(42)
rng = np.random.default_rng(seed=42)


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
    n_nodes = 201  # 201 nodes -> 200 segments
    nosc = n_nodes * 3  # 603 position DOFs in 3D
    length = 10.0
    mass_total = 1.0
    gravity = 9.81
    boundary = "pinned"  # 'pinned' or 'clamped'
    ds = length / (n_nodes - 1)

    # Time-stepping
    predict = True
    train_ratio = 0.7
    reducer = 'psd'
    hyperreducer = 'MDEIM'
    constraints_reduce = True

    tol = 1e-12
    tol_reduced = None
    M = 500
    pod_tol_sweep = [1e-2]

    if predict:
        dt_space_dim = 1
        dt_space = np.array([0.001])
        T_final = dt_space[-1] * 3e1  # 3.0 seconds

        # Dynamically align training split and test size to integer multiples of compute workers
        # to ensure 100% core occupancy and eliminate idle tail stragglers during parallel solves.
        _env_workers = os.environ.get('COMPUTE_WORKERS') or os.environ.get('DASK_NUM_WORKERS')
        _worker_count = int(_env_workers) if _env_workers else int(os.environ.get('SLURM_CPUS_PER_TASK', 30))

        omega2_train_waves = float(os.environ.get('OMEGA2_TRAIN_WAVES', 1))
        omega2_test_waves = float(os.environ.get('OMEGA2_TEST_WAVES', 0.3334))

        split_idx = max(1, int(round(float(os.environ.get('OMEGA2_SPLIT_IDX', omega2_train_waves * _worker_count)))))
        _test_dim = max(1, int(round(float(os.environ.get('OMEGA2_TEST_DIM', omega2_test_waves * _worker_count)))))
        _Omega2_space_dim = split_idx + _test_dim
        train_ratio = split_idx / _Omega2_space_dim
    else:
        dt_space_dim = 4
        dt_space = np.logspace(-4, -4 - dt_space_dim, num=dt_space_dim, base=2, endpoint=False)
        T_final = 0.1
        _Omega2_space_dim = 3
        split_idx = 3
        train_ratio = 1.0

    # Number of bending joints: (n_nodes - 2)
    num_joints = n_nodes - 2
    # Bending stiffness parameter samples B_i in [2.0, 10.0]
    _Omega2_space = 2.0 + 8.0 * rng.random((_Omega2_space_dim, num_joints))
    _Omega2_space = np.sort(_Omega2_space, axis=1)

    if predict:
        Omega2_space = _Omega2_space[:split_idx]
        Omega2_space_test = _Omega2_space[split_idx:]
    else:
        Omega2_space = Omega2_space_test = _Omega2_space


    is_elastica = True
    has_momentum_invariants = False


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
    return np.concatenate([q0, p0])


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

    def __init__(self, kwds=None, **kwargs):
        if kwds is None:
            kwds = {}
        if kwargs:
            kwds = {**kwds, **kwargs}
        super().__init__(kwds)
        # Ensure JJ matches the elastica nosc
        self.JJ = self.compute_J(self.nosc_r if hasattr(self, 'RB') else self.nosc)


@load_symbolic_expressions_elastica
class HamiltonianElasticaMechSystem(HamiltonianMechSystem, ElasticaMechSystem):
    """Full-order Hamiltonian Elastica system."""
    pass


class ReducedHamiltonianElasticaMechSystem(ReducedHamiltonianMechSystem, HamiltonianElasticaMechSystem):
    """Reduced-order Hamiltonian Elastica system."""
    pass


class HyperReducedHamiltonianElasticaMechSystem(HyperReducedHamiltonianMechSystem, ReducedHamiltonianElasticaMechSystem):
    """Hyper-reduced order Hamiltonian Elastica system."""
    pass


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------

class ElasticaStormerVerletSolver(BaseSolverMixin, ConformalStormerVerletFixedPointMixin,
                                 ConformalStormerVerlet, HamiltonianElasticaMechSystem):
    """Full-order Stormer-Verlet (RATTLE) solver for Elastica."""
    model_tier = 'fom'


class ReducedElasticaStormerVerletSolver(ReducedSolverMixin, ElasticaStormerVerletSolver,
                                        ReducedHamiltonianElasticaMechSystem):
    """Reduced-order Stormer-Verlet (RATTLE) solver for Elastica."""
    model_tier = 'rom'


class HyperReducedElasticaStormerVerletSolver(HyperReducedSolverMixin, ReducedElasticaStormerVerletSolver,
                                             HyperReducedHamiltonianElasticaMechSystem):
    """Hyper-reduced order Stormer-Verlet (RATTLE) solver for Elastica."""
    model_tier = 'hrom'


ElasticaStormerVerletSolver.reduced_solver = ReducedElasticaStormerVerletSolver
ElasticaStormerVerletSolver.hyperreduced_solver = HyperReducedElasticaStormerVerletSolver


ELASTICA_REDUCED_SOLVER_MAPPING = {
    ElasticaStormerVerletSolver: ReducedElasticaStormerVerletSolver,
    ReducedElasticaStormerVerletSolver: ReducedElasticaStormerVerletSolver,
}

ELASTICA_HYPERREDUCED_SOLVER_MAPPING = {
    ElasticaStormerVerletSolver: HyperReducedElasticaStormerVerletSolver,
    HyperReducedElasticaStormerVerletSolver: HyperReducedElasticaStormerVerletSolver,
}

# Update global mapping dictionaries
REDUCED_SOLVER_MAPPING.update(ELASTICA_REDUCED_SOLVER_MAPPING)
HYPERREDUCED_SOLVER_MAPPING.update(ELASTICA_HYPERREDUCED_SOLVER_MAPPING)


class HamiltonianElasticaReducer(HamiltonianReducer, HamiltonianElasticaMechSystem):
    """Reducer tailored for Hamiltonian Elastica MechSystem."""
    pass


# Re-export plot_elastica_beam from PlotScript for backwards compatibility
from PlotScript import plot_elastica_beam
__all__ = ['plot_elastica_beam']


