#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: bhattah
"""

import os
import sys
import cloudpickle as pickle
import pickle as std_pickle
import lz4.frame
from functools import wraps
from datetime import datetime
import hashlib
import json

# Third-party imports
import numpy as np
import sympy as smp
from pylab import r_, c_, zeros, eye, linspace

# Local imports
from PlotScript import timing

# get numpy random seed value
rng = np.random.default_rng(
    seed=267257368022227711484290921317604022527
)  # rng = np.random.default_rng(seed=408394104)  # Create RNG with fixed seed

import shutil

# High-performance compressed serialization helpers for large checkpoints
def fast_dump(obj, filename):
    """Save object using cloudpickle with LZ4 compression and a 16MB I/O buffer."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    tmp_filename = filename + ".tmp"
    with open(tmp_filename, 'wb', buffering=16 * 1024 * 1024) as raw_f:
        with lz4.frame.open(raw_f, 'wb') as f:
            pickle.dump(obj, f)
    os.replace(tmp_filename, filename)

    # Immediately mirror checkpoint to persistent master node storage if running in local scratch
    scratch_base = os.environ.get("SCRATCH_DIR") or os.environ.get("SLURM_TMPDIR")
    if scratch_base and os.path.abspath(filename).startswith(os.path.abspath(scratch_base)):
        rel_path = os.path.relpath(os.path.abspath(filename), os.path.abspath(scratch_base))
        master_file = os.path.abspath(rel_path)
        try:
            os.makedirs(os.path.dirname(master_file), exist_ok=True)
            shutil.copy2(filename, master_file)
            print(f"💾 Mirrored checkpoint to master storage: {master_file}", flush=True)
        except Exception as e:
            print(f"Warning: Could not mirror checkpoint to master storage ({e})", flush=True)

def fast_load(filename):
    """Load object using cloudpickle with LZ4 decompression and 16MB I/O buffer fallback."""
    if not os.path.exists(filename):
        scratch_base = os.environ.get("SCRATCH_DIR") or os.environ.get("SLURM_TMPDIR")
        if scratch_base and os.path.abspath(filename).startswith(os.path.abspath(scratch_base)):
            rel_path = os.path.relpath(os.path.abspath(filename), os.path.abspath(scratch_base))
            master_file = os.path.abspath(rel_path)
            if os.path.exists(master_file):
                print(f"📥 Restoring checkpoint file from master storage fallback ({master_file})...", flush=True)
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                shutil.copy2(master_file, filename)

    try:
        with open(filename, 'rb', buffering=16 * 1024 * 1024) as raw_f:
            with lz4.frame.open(raw_f, 'rb') as f:
                return pickle.load(f)
    except Exception:
        # Fallback for uncompressed legacy pickle files
        with open(filename, 'rb', buffering=16 * 1024 * 1024) as f:
            return pickle.load(f)

def get_data_dir(subpath=""):
    """
    Return data directory path. Prefers local scratch directory if set via SCRATCH_DIR or SLURM_TMPDIR,
    otherwise falls back to 'data' in current working directory.
    """
    scratch_base = os.environ.get("SCRATCH_DIR") or os.environ.get("SLURM_TMPDIR")
    if scratch_base:
        base = os.path.join(scratch_base, 'data')
    else:
        base = 'data'
    if subpath:
        return os.path.join(base, subpath)
    return base

def get_symbolic_expressions_file(nosc):
    """
    Locates symbolic expressions pickle file. Checks scratch data directory first,
    falling back to root 'data' directory if present there.
    """
    filename = f"symbolic_expr_{nosc}_cse.pickle"
    scratch_file = get_data_dir(filename)
    persistent_file = os.path.join('data', filename)
    
    if not os.path.exists(scratch_file) and os.path.exists(persistent_file):
        return persistent_file
    return scratch_file

def check_checkpoint_exists(checkpoint_path, *files):
    """
    Checks if checkpoint directory and specified files exist in local scratch or persistent master storage.
    If present in master storage but missing in scratch, automatically restores them to scratch.
    """
    scratch_base = os.environ.get("SCRATCH_DIR") or os.environ.get("SLURM_TMPDIR")

    # 1. Check if all files exist directly at target path
    if os.path.exists(checkpoint_path) and all(os.path.exists(f) for f in files):
        return True

    # 2. If target path is in scratch, check if files exist in persistent master storage
    if scratch_base and os.path.abspath(checkpoint_path).startswith(os.path.abspath(scratch_base)):
        rel_checkpoint = os.path.relpath(os.path.abspath(checkpoint_path), os.path.abspath(scratch_base))
        master_checkpoint = os.path.abspath(rel_checkpoint)

        master_files = [
            os.path.abspath(os.path.relpath(os.path.abspath(f), os.path.abspath(scratch_base)))
            for f in files
        ]

        if os.path.exists(master_checkpoint) and all(os.path.exists(mf) for mf in master_files):
            print(f"📥 Restoring checkpoint from master storage ({master_checkpoint}) to local scratch ({checkpoint_path})...", flush=True)
            os.makedirs(checkpoint_path, exist_ok=True)
            for mf, sf in zip(master_files, files):
                shutil.copy2(mf, sf)
            return True

    return False

# %%
def load_symbolic_expressions(cls, expressions_file=None):
    """Decorator or function to handle loading/saving of symbolic expressions"""
    try:
        if expressions_file is None:
            expressions_file = get_symbolic_expressions_file(cls.nosc)
        expressions = fast_load(expressions_file)

        # print("Loaded symbolic expressions from disk.")
    except: # (FileNotFoundError, std_pickle.UnpicklingError):
        # Fail silently; the driver script is responsible for generating/loading these.
        return cls

    # Update class attributes, wrapping lambdas to include self
    for k, v in expressions.items():
        if callable(v) and not isinstance(v, type):
            # Use staticmethod to make the function available as a class method without passing `self`
            setattr(cls, k, staticmethod(v))
        else:
            setattr(cls, k, v)

    return cls


class SysConfig:
    """
    Central configuration for simulation parameters.
    """

    @classmethod
    def get_config_hash(cls):
        """
        Generates a hash of the SysConfig parameters to detect changes.
        """
        config_params = {}
        # Iterate over class attributes, excluding methods and dunder attributes
        for key in SysConfig.__dict__:
            if key.startswith('__'): continue
            
            value = getattr(cls, key)
            
            if not callable(value):
                # Special handling for numpy arrays to make them JSON serializable
                if isinstance(value, np.ndarray):
                    config_params[key] = value.tolist()
                else:
                    config_params[key] = value
        
        # Create a sorted JSON string for a consistent hash
        # Using separators to remove whitespace and make it compact
        sorted_params = json.dumps(config_params, sort_keys=True, separators=(',', ':'))
        return hashlib.md5(sorted_params.encode('utf-8')).hexdigest()

    # Numerical solver and its properties
    w_values = [0.28, 0.62546642846767004501]
    w_values.append(1.0 - 2.0 * (sum(w_values)))
    w_values.append(w_values[1])
    w_values.append(w_values[0])
    w_values = [1]
    assert np.isclose(sum(w_values), 1), 'sum_i w_i must be 1'
    
    # Solver and reduction settings
    tol, M, var = 1.0E-12, 500, True
    tol_reduced = 1.0E-8
    # pod_tol_ham = 1e-10
    # pod_tol_dg = 1e-10
    reducer = 'psd'
    hyperreducer = 'MDEIM'
    constraint_type = 'spherical'
    constraints_reduce = True # True = hyper-reduce constraints, False = reduce constraints

    # MDEIM snapshot sampling settings (subsamples Hessian/Jacobian snapshots across FOM solves)
    mdeim_max_snapshots_per_solver = int(1e3)  # Maximum snapshots sampled per solver trajectory for MDEIM
    mdeim_target_total_snapshots = int(5e4)   # Global snapshot budget across all solves for MDEIM
    mdeim_min_snapshots_per_solver = 3        # Minimum snapshots per solver
    mdeim_sampling_method = 'uniform'         # 'uniform', 'stride', or 'random'
    mdeim_snapshot_stride = None              # Fixed stride if specified, else uniform spacing
    mdeim_max_solvers = None                  # Optional cap on number of solvers sampled
    
    # System parameters
    nosc = 54 * 10  # Number of oscillators
    assert nosc % 6 == 0, 'nosc must be divisible by 6'

    predict = True # False = reproduce results of the full model

    if predict: # prediction parameters
        pod_tol_sweep = [1e-3]
        train_ratio = 0.7

        # Time-stepping parameters
        dt_space_dim = 1
        dt_space = np.array([0.001])
        T_final = dt_space[-1] * 3e3

        # Parameter space for Omega^2
        _Omega2_space_dim = nosc // 3 - 2
    else: # reproduction parameters
        pod_tol_sweep = [1e-4, 1e-6, 1e-8, 1e-10] # must be in descending order

        # Time-stepping parameters
        dt_space_dim = 5
        dt_space = np.logspace(-4, -4 - dt_space_dim, num=dt_space_dim, base=2, endpoint=False)
        T_final = 0.1

        # Parameter space for Omega^2
        _Omega2_space_dim = 3

    assert pod_tol_sweep == sorted(pod_tol_sweep, reverse=True), "pod_tol_sweep must be in descending order"

    # Generate random parameter space for Omega^2 in range (0, 10]
    num_freqs = nosc // 3 - 2
    _Omega2_space = np.sort(3 * (1 - rng.random((_Omega2_space_dim, num_freqs))), axis=1)

    # Freeze higher frequencies across samples to match the first sample
    # freeze_idx = _Omega2_space_dim // 2
    # if freeze_idx < num_freqs:
    #     _Omega2_space[:, freeze_idx:] = _Omega2_space[0, freeze_idx:]

    # Train-test split
    if predict:
        split_idx = int(_Omega2_space_dim * train_ratio)
        Omega2_space, Omega2_space_test = _Omega2_space[:split_idx], _Omega2_space[split_idx:]
        
        if Omega2_space.shape[0] == 0 or Omega2_space_test.shape[0] == 0:
            raise ValueError(f"Train-test split resulted in empty sets. "
                             f"Total samples: {_Omega2_space_dim}, Split index: {split_idx}, Train ratio: {train_ratio}. "
                             "Adjust train_ratio or increase _Omega2_space_dim.")
    else:
        Omega2_space = Omega2_space_test = _Omega2_space

def make_lattice_initial_conditions(nosc):
    """
    Construct initial state for double-helix spring lattice satisfying
    inter-helix distance constraints and momentum orthogonality.
    """
    _i = np.arange(nosc // 3 // 2)
    _radius = 0.5
    _pitch = 0.5 * nosc / 18  # Scale pitch with number of particles
    _t = _i * 2 * np.pi / (nosc // 3)
    _phase = np.pi

    positions_1 = np.stack((
        _radius * np.cos(_t),
        _radius * np.sin(_t),
        _pitch * _t / (2 * np.pi)
    ), axis=1)

    positions_2 = np.stack((
        _radius * np.cos(_t + _phase),
        _radius * np.sin(_t + _phase),
        _pitch * _t / (2 * np.pi)
    ), axis=1)

    # Combine the two helices, interleaving their positions
    positions = np.zeros((nosc // 3, 3))
    positions[::2] = positions_1
    positions[1::2] = positions_2

    # Assert that the corresponding coordinates of positions_1 and positions_2 are distant 1 apart
    distances = np.linalg.norm(positions[::2] - positions[1::2], axis=1)
    assert np.allclose(distances, 1), "The corresponding coordinates of positions_1 and positions_2 are not distant 1 apart."

    positions = positions.flatten().reshape(-1, 1)

    momenta = np.zeros((nosc // 3, 3))
    momenta[::2] = rng.uniform(-0.001, 0.001, momenta[::2].shape)
    momenta[1::2] = momenta[::2]
    assert np.allclose(momenta[1::2], momenta[::2]), 'position and momenta are not orthogonal'
    momenta = momenta.flatten().reshape(-1, 1)

    return r_[positions, momenta].flatten()


# @load_symbolic_expressions # Moved decorator to concrete subclasses or base if needed
class MechSystem(SysConfig):
    """
    Represents the mechanical system, including parameters, initial conditions, 
    and methods for evaluating system dynamics (Hamiltonian, Lagrangian, constraints).
    """

    keep_time = os.environ.get("JOB_KEEP_TIME") or datetime.now().strftime("%Y-%m-%d") #_%H-%M-%S")
    data_folder = get_data_dir(keep_time)
    os.makedirs(data_folder, exist_ok=True)

    @staticmethod
    def compute_J(d):
        return r_[c_[zeros((d, d)), eye(d)], 
                    c_[-eye(d), zeros((d, d))]]

    y_init = make_lattice_initial_conditions(SysConfig.nosc)

    def ham(self, y, beta=0):
        return self.ham_(y, self.Omega2, beta)

    def g_vec(self, y):
        return self.g_(y)

    def g_prime(self, y):
        return self.g_prime_(y)

    # ── Projection operator property aliases (canonical <-> legacy) ──
    @property
    def deim_field(self):
        return self.__dict__.get('deim_field', self.__dict__.get('RBxUx_inv_PxU', None))

    @deim_field.setter
    def deim_field(self, val):
        self.__dict__['deim_field'] = val
        self.__dict__['RBxUx_inv_PxU'] = val

    @property
    def RBxUx_inv_PxU(self):
        return self.__dict__.get('deim_field', self.__dict__.get('RBxUx_inv_PxU', None))

    @RBxUx_inv_PxU.setter
    def RBxUx_inv_PxU(self, val):
        self.__dict__['deim_field'] = val
        self.__dict__['RBxUx_inv_PxU'] = val

    @property
    def mdeim_Hessian(self):
        return self.__dict__.get('mdeim_Hessian', self.__dict__.get('IP_Ux_inv_PxU', None))

    @mdeim_Hessian.setter
    def mdeim_Hessian(self, val):
        self.__dict__['mdeim_Hessian'] = val
        self.__dict__['IP_Ux_inv_PxU'] = val

    @property
    def IP_Ux_inv_PxU(self):
        return self.__dict__.get('mdeim_Hessian', self.__dict__.get('IP_Ux_inv_PxU', None))

    @IP_Ux_inv_PxU.setter
    def IP_Ux_inv_PxU(self, val):
        self.__dict__['mdeim_Hessian'] = val
        self.__dict__['IP_Ux_inv_PxU'] = val

    @property
    def mdeim_g_prime(self):
        return self.__dict__.get('mdeim_g_prime', self.__dict__.get('_IP_Ux_inv_PxU', None))

    @mdeim_g_prime.setter
    def mdeim_g_prime(self, val):
        self.__dict__['mdeim_g_prime'] = val
        self.__dict__['_IP_Ux_inv_PxU'] = val

    @property
    def _IP_Ux_inv_PxU(self):
        return self.__dict__.get('mdeim_g_prime', self.__dict__.get('_IP_Ux_inv_PxU', None))

    @_IP_Ux_inv_PxU.setter
    def _IP_Ux_inv_PxU(self, val):
        self.__dict__['mdeim_g_prime'] = val
        self.__dict__['_IP_Ux_inv_PxU'] = val

    @property
    def mdeim_g_var(self):
        return self.__dict__.get('mdeim_g_var', self.__dict__.get('IP_g_prime_x_lambda_y', None))

    @mdeim_g_var.setter
    def mdeim_g_var(self, val):
        self.__dict__['mdeim_g_var'] = val
        self.__dict__['IP_g_prime_x_lambda_y'] = val

    @property
    def IP_g_prime_x_lambda_y(self):
        return self.__dict__.get('mdeim_g_var', self.__dict__.get('IP_g_prime_x_lambda_y', None))

    @IP_g_prime_x_lambda_y.setter
    def IP_g_prime_x_lambda_y(self, val):
        self.__dict__['mdeim_g_var'] = val
        self.__dict__['IP_g_prime_x_lambda_y'] = val

    # Window lists
    @property
    def deim_field_windows(self):
        return self.__dict__.get('deim_field_windows', self.__dict__.get('RBxUx_inv_PxU_windows', None))

    @deim_field_windows.setter
    def deim_field_windows(self, val):
        self.__dict__['deim_field_windows'] = val
        self.__dict__['RBxUx_inv_PxU_windows'] = val

    @property
    def RBxUx_inv_PxU_windows(self):
        return self.__dict__.get('deim_field_windows', self.__dict__.get('RBxUx_inv_PxU_windows', None))

    @RBxUx_inv_PxU_windows.setter
    def RBxUx_inv_PxU_windows(self, val):
        self.__dict__['deim_field_windows'] = val
        self.__dict__['RBxUx_inv_PxU_windows'] = val

    @property
    def mdeim_Hessian_windows(self):
        return self.__dict__.get('mdeim_Hessian_windows', self.__dict__.get('IP_Ux_inv_PxU_windows', None))

    @mdeim_Hessian_windows.setter
    def mdeim_Hessian_windows(self, val):
        self.__dict__['mdeim_Hessian_windows'] = val
        self.__dict__['IP_Ux_inv_PxU_windows'] = val

    @property
    def IP_Ux_inv_PxU_windows(self):
        return self.__dict__.get('mdeim_Hessian_windows', self.__dict__.get('IP_Ux_inv_PxU_windows', None))

    @IP_Ux_inv_PxU_windows.setter
    def IP_Ux_inv_PxU_windows(self, val):
        self.__dict__['mdeim_Hessian_windows'] = val
        self.__dict__['IP_Ux_inv_PxU_windows'] = val

    @property
    def mdeim_g_prime_windows(self):
        return self.__dict__.get('mdeim_g_prime_windows', self.__dict__.get('_IP_Ux_inv_PxU_windows', None))

    @mdeim_g_prime_windows.setter
    def mdeim_g_prime_windows(self, val):
        self.__dict__['mdeim_g_prime_windows'] = val
        self.__dict__['_IP_Ux_inv_PxU_windows'] = val

    @property
    def _IP_Ux_inv_PxU_windows(self):
        return self.__dict__.get('mdeim_g_prime_windows', self.__dict__.get('_IP_Ux_inv_PxU_windows', None))

    @_IP_Ux_inv_PxU_windows.setter
    def _IP_Ux_inv_PxU_windows(self, val):
        self.__dict__['mdeim_g_prime_windows'] = val
        self.__dict__['_IP_Ux_inv_PxU_windows'] = val

    @property
    def mdeim_g_var_windows(self):
        return self.__dict__.get('mdeim_g_var_windows', self.__dict__.get('IP_g_prime_x_lambda_y_windows', None))

    @mdeim_g_var_windows.setter
    def mdeim_g_var_windows(self, val):
        self.__dict__['mdeim_g_var_windows'] = val
        self.__dict__['IP_g_prime_x_lambda_y_windows'] = val

    @property
    def IP_g_prime_x_lambda_y_windows(self):
        return self.__dict__.get('mdeim_g_var_windows', self.__dict__.get('IP_g_prime_x_lambda_y_windows', None))

    @IP_g_prime_x_lambda_y_windows.setter
    def IP_g_prime_x_lambda_y_windows(self, val):
        self.__dict__['mdeim_g_var_windows'] = val
        self.__dict__['IP_g_prime_x_lambda_y_windows'] = val

    def _apply_sparsification(self):
        """Redefines constraint methods to ensure the system is underconstrained."""
        sample_g = self.g_vec(self.y_init)
        m_h = sample_g.size // 2
        print(f"m_h = {m_h}, nosc_r = {self.nosc_r}")
        
        if m_h >= self.nosc_r:
            m_target = self.nosc_r - 1
            idx_h = np.linspace(0, m_h - 1, m_target, dtype=int)
            print(f"Sparsifying constraints: {2*m_h} -> {2*m_target} ({(1 - m_target/m_h):.1%} reduction) to satisfy LBB.")
            full_idx = np.concatenate([idx_h, idx_h + m_h])
            
            _orig_g = self.g_vec
            self.g_vec = lambda y: _orig_g(y)[full_idx]
            
            _orig_gp = self.g_prime
            self.g_prime = lambda y: _orig_gp(y)[full_idx, :]
            
            if hasattr(self, 'g_prime_x_lambda_y'):
                _orig_gpxy = self.g_prime_x_lambda_y
                def wrapped_gpxy(y, lm):
                    inflated = np.zeros(2 * m_h)
                    inflated[full_idx] = lm
                    return _orig_gpxy(y, inflated)
                self.g_prime_x_lambda_y = wrapped_gpxy

    def __init__(self, kwds=None, **kwargs):        
        if kwds is None:
            kwds = {}
        if kwargs:
            kwds = {**kwds, **kwargs}

        if 'pool' in kwds:
            self.__dict__.update(kwds['pool'])
        if 'nosc' in kwds:
            self.nosc = kwds['nosc']
        for k, v in kwds.items():
            if k not in ('pool',):
                setattr(self, k, v)

        self.beta = (max(1e-2, 0 * rng.random() / 10)) * 0 # Damping coefficient, obsolete
        self.time_lapsed = []

        if self.reducer == 'psd':
             self.JJ = self.compute_J(self.nosc_r if hasattr(self, 'RB') else self.nosc)
        else:
             self.JJ = self.compute_J(self.nosc)

        # Projection matrices
        if hasattr(self, 'RB'):
            self.y_init = self.RB.T @ self.y_init


@load_symbolic_expressions
class HamiltonianMechSystem(MechSystem):
    """System with Hamiltonian structure."""

    def ham_z(self, y, beta=0):
        return self.ham_z_(y, self.Omega2, beta)

    def ham_zz(self, y, beta=0):
        return self.ham_zz_(y, self.Omega2, beta)

    def __init__(self, kwds=None, **kwargs):
        super().__init__(kwds, **kwargs)


@load_symbolic_expressions
class LagrangianMechSystem(MechSystem):
    """System with Lagrangian structure."""

    def lag_dg(self, y):
        return self.lag_dg_(*y, self.Omega2)

    def lag_dg_z(self, y):
        return self.lag_dg_z_(*y, self.Omega2)

    def g_prime_x_lambda_y(self, y, lag_mult):
        """Compute g_prime_x_lambda_y for full order system"""
        return self.g_prime_x_lambda_y_(y, lag_mult)

    def __init__(self, kwds=None, **kwargs):
        super().__init__(kwds, **kwargs)


class ReducedHamiltonianMechSystem(HamiltonianMechSystem):
    """Reduced order Hamiltonian system."""

    def ham_z(self, y, beta=0):
        return self.RB.T @ self.ham_z_(y @ self.RB.T, self.Omega2, beta)

    def ham_zz(self, y, beta=0):
        return self.RB.T @ self.ham_zz_(y @ self.RB.T, self.Omega2, beta) @ self.RB

    def g_vec(self, y):
        return self.g_(y @ self.RB.T)

    def g_prime(self, y):
        return self.g_prime_(y @ self.RB.T) @ self.RB

    def __init__(self, kwds=None, **kwargs):
        # Load solver specific data
        if kwds and 'solver_data' in kwds:
            for cls in self.__class__.mro():
                if cls.__name__ in kwds['solver_data']:
                    for k, v in kwds['solver_data'][cls.__name__].items():
                        setattr(self, k, v)
                    break
        super().__init__(kwds, **kwargs)
        self.tol = self.tol_reduced


class ReducedLagrangianMechSystem(LagrangianMechSystem):
    """Reduced order Lagrangian system."""

    def lag_dg(self, y):
        return self.RB.T @ self.lag_dg_(*(y @ self.RB.T), self.Omega2)

    def lag_dg_z(self, y):
        return self.RB.T @ self.lag_dg_z_(*(y @ self.RB.T), self.Omega2) @ self.RB

    def g_vec(self, y):
        return self.g_(y @ self.RB.T)

    def g_prime(self, y):
        return self.g_prime_(y @ self.RB.T) @ self.RB

    def g_prime_x_lambda_y(self, y, lag_mult):
        """Compute g_prime_x_lambda_y for reduced order system"""
        return self.RB.T @ self.g_prime_x_lambda_y_(y @ self.RB.T, lag_mult) @ self.RB

    def __init__(self, kwds=None, **kwargs):
        # Load solver specific data
        if kwds and 'solver_data' in kwds:
            for cls in self.__class__.mro():
                if cls.__name__ in kwds['solver_data']:
                    for k, v in kwds['solver_data'][cls.__name__].items():
                        setattr(self, k, v)
                    break
        super().__init__(kwds, **kwargs)
        self.tol = self.tol_reduced


class HyperReducedHamiltonianMechSystem(ReducedHamiltonianMechSystem):
    """Hyper-reduced Hamiltonian system."""

    def ham_z(self, y, beta=0):
        return self.deim_field @ self.ham_z_deim(y @ self.RB.T, self.Omega2, beta)

    def ham_zz(self, y, beta=0):
        if self.hyperreducer == 'MDEIM':
            return self.RB.T @ np.reshape(
                self.mdeim_Hessian @ self.ham_zz_mdeim(y @ self.RB.T, self.Omega2, beta),
                (2*self.nosc, 2*self.nosc)
            ) @ self.RB
        return self.deim_field @ self.ham_zz_deim(y @ self.RB.T, self.Omega2, beta) @ self.RB

    def g_vec(self, y):
        return super().g_vec(y)

    def g_prime(self, y):
        if self.constraints_reduce and hasattr(self, 'mdeim_g_prime') and hasattr(self, 'g_prime_mdeim'):
            shape = getattr(self, 'g_prime_shape', None)
            if shape is not None:
                return np.reshape(
                    self.mdeim_g_prime @ self.g_prime_mdeim(y @ self.RB.T),
                    shape
                ) @ self.RB
        return super().g_prime(y)

    def __init__(self, kwds=None, **kwargs):
        super().__init__(kwds, **kwargs)


class HyperReducedLagrangianMechSystem(ReducedLagrangianMechSystem):
    """Hyper-reduced Lagrangian system."""

    def lag_dg(self, y):
        return self.deim_field @ self.lag_dg_deim(*(y @ self.RB.T), self.Omega2)

    def lag_dg_z(self, y):
        if self.hyperreducer == 'MDEIM':
            return self.RB.T @ np.reshape(
                self.mdeim_Hessian @ self.lag_dg_z_mdeim(*(y @ self.RB.T), self.Omega2),
                (2*self.nosc, 2*self.nosc)
            ) @ self.RB
        return self.deim_field @ self.lag_dg_z_deim(*(y @ self.RB.T), self.Omega2) @ self.RB

    def g_vec(self, y):
        return super().g_vec(y)

    def g_prime(self, y):
        if self.constraints_reduce and hasattr(self, 'mdeim_g_prime') and hasattr(self, 'g_prime_mdeim'):
            shape = getattr(self, 'g_prime_shape', None)
            if shape is not None:
                return np.reshape(
                    self.mdeim_g_prime @ self.g_prime_mdeim(y @ self.RB.T),
                    shape
                ) @ self.RB
        return super().g_prime(y)

    def g_prime_x_lambda_y(self, y, lag_mult):
        """Compute g_prime_x_lambda_y for hyperreduced system"""
        if self.constraints_reduce and hasattr(self, 'mdeim_g_var') and hasattr(self, 'g_prime_x_lambda_y_mdeim'):
            shape = getattr(self, 'g_prime_x_lambda_y_shape', None)
            if shape is not None:
                return (
                    self.RB.T
                    @ np.reshape(
                        self.mdeim_g_var
                        @ self.g_prime_x_lambda_y_mdeim(y @ self.RB.T, lag_mult),
                        shape,
                    )
                    @ self.RB
                )
        return super().g_prime_x_lambda_y(y, lag_mult)

    def __init__(self, kwds=None, **kwargs):
        super().__init__(kwds, **kwargs)


# ---------------------------------------------------------------------------
# Backward-compatibility aliases on MechSystem for unpickling legacy checkpoints
# ---------------------------------------------------------------------------
MechSystem.ham_lambda = MechSystem.ham
MechSystem.ham_z_lambda = HamiltonianMechSystem.ham_z
MechSystem.ham_zz_lambda = HamiltonianMechSystem.ham_zz
MechSystem.lag_dg_lambda = LagrangianMechSystem.lag_dg
MechSystem.lag_dg_z_lambda = LagrangianMechSystem.lag_dg_z
MechSystem.g_prime_x_lambda_y__ = LagrangianMechSystem.g_prime_x_lambda_y
MechSystem.ham_z_reduced = ReducedHamiltonianMechSystem.ham_z
MechSystem.ham_zz_reduced = ReducedHamiltonianMechSystem.ham_zz
MechSystem.g_reduced = ReducedHamiltonianMechSystem.g_vec
MechSystem.g_prime_reduced = ReducedHamiltonianMechSystem.g_prime
MechSystem.lag_dg_reduced = ReducedLagrangianMechSystem.lag_dg
MechSystem.lag_dg_z_reduced = ReducedLagrangianMechSystem.lag_dg_z
MechSystem.g_prime_x_lambda_y_reduced = ReducedLagrangianMechSystem.g_prime_x_lambda_y
MechSystem.ham_z_hyperreduced = HyperReducedHamiltonianMechSystem.ham_z
MechSystem.ham_zz_hyperreduced = HyperReducedHamiltonianMechSystem.ham_zz
MechSystem.ham_zz_mdeim_hyperreduced = HyperReducedHamiltonianMechSystem.ham_zz
MechSystem.lag_dg_hyperreduced = HyperReducedLagrangianMechSystem.lag_dg
MechSystem.lag_dg_z_hyperreduced = HyperReducedLagrangianMechSystem.lag_dg_z
MechSystem.lag_dg_z_mdeim_hyperreduced = HyperReducedLagrangianMechSystem.lag_dg_z
MechSystem.g_hyperreduced = HyperReducedHamiltonianMechSystem.g_vec
MechSystem.g_prime_hyperreduced = HyperReducedHamiltonianMechSystem.g_prime
MechSystem.g_prime_x_lambda_y_hyperreduced = HyperReducedLagrangianMechSystem.g_prime_x_lambda_y
MechSystem.g__lambda = lambda self, y: self.g_(y)
MechSystem.g_prime__lambda = lambda self, y: self.g_prime_(y)
MechSystem.g__ = MechSystem.g_vec
MechSystem.g_prime__ = MechSystem.g_prime

