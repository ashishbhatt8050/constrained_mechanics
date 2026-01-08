#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: bhattah
"""

import os
import sys
import cloudpickle as pickle
import pickle as std_pickle
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

# %%
def load_symbolic_expressions(cls):
    """Decorator to handle loading/saving of symbolic expressions"""
    try:
        # Try to load expressions
        expressions_file = os.path.join('data', f"symbolic_expr_{cls.nosc}_cse.pickle")
        with open(expressions_file, 'rb') as f:
            expressions = pickle.load(f)

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
    tol, M, var, store = 1.0E-12, 500, True, False
    
    predict = True  # False = reproduce results of the full model
    train_ratio = 0.7
    reducer = 'psd'
    hyperreducer = 'MDEIM'
    constraint_type = 'spherical'
    constraints_reduce = True
    
    # System parameters
    nosc = 54 * 10  # Number of oscillators
    assert nosc % 6 == 0, 'nosc must be divisible by 6'

    if predict: # prediction parameters
        # Time-stepping parameters
        dt_space_dim = 1
        dt_space = np.array([0.01])
        T_final = dt_space[-1] * 1e3

        # Parameter space for Omega^2
        _Omega2_space_dim = 20
    else: # reproduction parameters
        # Time-stepping parameters
        dt_space_dim = 3
        dt_space = np.round(np.logspace(-3, -2, num=dt_space_dim), 10)
        T_final = dt_space[-1] * 5e1

        # Parameter space for Omega^2
        _Omega2_space_dim = 3

    # Generate random parameter space for Omega^2 in range (0, 10]
    num_freqs = nosc // 3 - 2
    _Omega2_space = np.sort(10 * (1 - rng.random((_Omega2_space_dim, num_freqs))), axis=1)

    # Freeze higher frequencies across samples to match the first sample
    freeze_idx = _Omega2_space_dim - 1
    if freeze_idx < num_freqs:
        _Omega2_space[:, freeze_idx:] = _Omega2_space[0, freeze_idx:]

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

# @load_symbolic_expressions # Moved decorator to concrete subclasses or base if needed
class MechSystem(SysConfig):
    """
    Represents the mechanical system, including parameters, initial conditions, 
    and methods for evaluating system dynamics (Hamiltonian, Lagrangian, constraints).
    """

    keep_time = datetime.now().strftime("%Y-%m-%d")
    data_folder = os.path.join('data', keep_time)
    # if not os.path.exists(data_folder):
    #     os.makedirs(data_folder)
    os.makedirs(data_folder, exist_ok=True)

    @staticmethod
    def compute_J(d):
        return r_[c_[zeros((d, d)), eye(d)], c_[-eye(d), zeros((d, d))]]

    _i = np.arange(SysConfig.nosc//3//2)
    _radius = 0.5
    _pitch = 0.5 * SysConfig.nosc/18  # Scale pitch with number of particles
    _t = _i * 2 * np.pi / (SysConfig.nosc//3)
    _phase = np.pi

    positions_1 = np.stack((
        _radius * np.cos(_t),
        _radius * np.sin(_t),
        _pitch * _t / (2*np.pi)
    ), axis=1)

    positions_2 = np.stack((
        _radius * np.cos(_t + _phase),
        _radius * np.sin(_t + _phase),
        _pitch * _t / (2*np.pi)
    ), axis=1)

    # Combine the two helices, interleaving their positions
    positions = np.zeros((SysConfig.nosc//3, 3))
    positions[::2] = positions_1
    positions[1::2] = positions_2

    # Assert that the corresponding coordinates of positions_1 and positions_2 are distant 1 apart
    distances = np.linalg.norm(positions[::2] - positions[1::2], axis=1)
    assert np.allclose(distances, 1), "The corresponding coordinates of positions_1 and positions_2 are not distant 1 apart."

    positions = positions.flatten().reshape(-1, 1)

    momenta = np.zeros((SysConfig.nosc//3, 3))
    momenta[::2] = rng.uniform(-0.001, 0.001, momenta[::2].shape)
    momenta[1::2] = momenta[::2]
    assert np.allclose(momenta[1::2], momenta[::2]), 'position and momenta are not orthogonal'
    momenta = momenta.flatten().reshape(-1, 1)

    y_init = r_[positions, momenta].flatten()

    drag = lambda self, y: 0 #beta/2 * r_[x, u]
    drag_z = lambda self, y: 0 #beta/2 * eye(2*x.shape[0])

    non_quad = None

    "y_init alias"
    @property
    def u_init(self):
        return self.y_init

    @u_init.setter
    def u_init(self, value):
        self.y_init = value

    def ham_lambda(self, y, beta=0):
        return self.ham_(y, self.Omega2, beta)

    def ham_z_lambda(self, y, beta=0):
        return self.ham_z_(y, self.Omega2, beta).squeeze()

    def ham_zz_lambda(self, y, beta=0):
        return self.ham_zz_(y, self.Omega2, beta)

    def lag_dg_lambda(self, y):
        return self.lag_dg_(*y, self.Omega2).squeeze()

    def lag_dg_z_lambda(self, y):
        return self.lag_dg_z_(*y, self.Omega2)

    def g__lambda(self, y):
        return self.g_(y).squeeze()

    def g_prime__lambda(self, y):
        return self.g_prime_(y)

    def g_prime_x_lambda_y__(self, y, lag_mult):
        """Compute g_prime_x_lambda_y for full order system"""
        # Compute g_prime_x_lambda_y using the full order system
        return self.g_prime_x_lambda_y_(y, lag_mult)

    def g_prime_x_lambda_lambda__(self, y, lag_mult):
        """Compute g_prime_x_lambda_lambda for full order system"""
        # Compute g_prime_x_lambda_lambda using the full order system
        return self.g_prime_x_lambda_lambda_(y, lag_mult)

    def ham_z_reduced(self, y, beta=0):
        return self.RB.T @ self.ham_z_(y @ self.RB.T, self.Omega2, beta).squeeze()

    def ham_zz_reduced(self, y, beta=0):
        # print(f'Computing ham_zz inside ham_zz_reduced')
        return self.RB.T @ self.ham_zz_(y @ self.RB.T, self.Omega2, beta) @ self.RB

        """Break large matrix multiplication into smaller chunks"""
        y_projected = y @ self.RB.T
        ham_zz_result = self.ham_zz_(y_projected, self.Omega2, beta)

        # Use 32 as chunk size (96 % 32 = 0 and 32 < 36)
        chunk_size = 32  # Gives exactly 3 chunks of size 32
        n_chunks = ham_zz_result.shape[1] // chunk_size

        # First multiplication
        result = np.zeros((self.RB.T.shape[0], ham_zz_result.shape[1]))
        for i in range(n_chunks):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            result[:, start:end] = self.RB.T @ ham_zz_result[:, start:end]

        # Second multiplication
        final = np.zeros((result.shape[0], self.RB.shape[1]))
        for i in range(n_chunks):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            final += result[:, start:end] @ self.RB[start:end, :]

        return final

    def lag_dg_reduced(self, y):
        return self.RB.T @ self.lag_dg_(*(y @ self.RB.T), self.Omega2).squeeze()

    def lag_dg_z_reduced(self, y):
        # print(f'Computing lag_dg_z inside lag_dg_z_reduced')
        return self.RB.T @ self.lag_dg_z_(*(y @ self.RB.T), self.Omega2) @ self.RB

        """Break large matrix multiplication into smaller chunks"""
        # First compute the DG result
        result = self.lag_dg_z_(*(y @ self.RB.T), self.Omega2)

        # Use 32 as chunk size (96 % 32 = 0 and 32 < 36)
        chunk_size = 32  # Gives exactly 3 chunks of size 32
        n_chunks = result.shape[1] // chunk_size

        # First multiplication
        intermediate = np.zeros((self.RB.T.shape[0], result.shape[1]))
        for i in range(n_chunks):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            intermediate[:, start:end] = self.RB.T @ result[:, start:end]

        # Second multiplication
        final = np.zeros((intermediate.shape[0], self.RB.shape[1]))
        for i in range(n_chunks):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            final += intermediate[:, start:end] @ self.RB[start:end, :]

        return final

    def g_reduced(self, y):
        return self.g_(y @ self.RB.T).squeeze()

    def g_prime_reduced(self, y):
        return self.g_prime_(y @ self.RB.T) @ self.RB

        """Break large matrix multiplication into smaller chunks"""
        # First compute g_prime result
        y_projected = y @ self.RB.T
        g_prime_result = self.g_prime_(y_projected)

        # Use 32 as chunk size (96 % 32 = 0 and 32 < 36)
        chunk_size = 32  # Gives exactly 3 chunks of size 32
        n_chunks = g_prime_result.shape[1] // chunk_size

        # Process in chunks
        final = np.zeros((g_prime_result.shape[0], self.RB.shape[1]))
        for i in range(n_chunks):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            final += g_prime_result[:, start:end] @ self.RB[start:end, :]

        return final

    def g_prime_x_lambda_y_reduced(self, y, lag_mult):
        """Compute g_prime_x_lambda_y for reduced order system"""
        # Compute g_prime_x_lambda_y using the reduced order system
        return self.RB.T @ self.g_prime_x_lambda_y_(y @ self.RB.T, lag_mult) @ self.RB

    def g_prime_x_lambda_lambda_reduced(self, y, lag_mult):
        """Compute g_prime_x_lambda_lambda for reduced order system"""
        # Compute g_prime_x_lambda_lambda using the reduced order system
        return self.RB.T @ self.g_prime_x_lambda_lambda_(y @ self.RB.T, lag_mult)

    def ham_z_hyperreduced(self, y, beta=0):
        return self.RBxUx_inv_PxU @ np.squeeze(self.ham_z_deim(y @ self.RB.T, self.Omega2, beta))

    def ham_zz_hyperreduced(self, y, beta=0):
        return self.RBxUx_inv_PxU @ self.ham_zz_deim(y @ self.RB.T, self.Omega2, beta) @ self.RB

    def ham_zz_mdeim_hyperreduced(self, y, beta=0):
        return self.RB.T @ np.reshape(
            self.IP_Ux_inv_PxU @ self.ham_zz_mdeim(y @ self.RB.T, self.Omega2, beta),
            (2*self.nosc, 2*self.nosc)
        ) @ self.RB

    def lag_dg_hyperreduced(self, y):
        return self.RBxUx_inv_PxU @ np.squeeze(self.lag_dg_deim(*(y @ self.RB.T), self.Omega2))

    def lag_dg_z_hyperreduced(self, y):
        return self.RBxUx_inv_PxU @ self.lag_dg_z_deim(*(y @ self.RB.T), self.Omega2) @ self.RB

    def lag_dg_z_mdeim_hyperreduced(self, y):
        return self.RB.T @ np.reshape(
            self.IP_Ux_inv_PxU @ self.lag_dg_z_mdeim(*(y @ self.RB.T), self.Omega2),
            (2*self.nosc, 2*self.nosc)
        ) @ self.RB

    def g_hyperreduced(self, y):
        return (self._Ux_inv_PxU @ self.g_deim(y @ self.RB.T)).squeeze()

    def g_prime_hyperreduced(self, y):
        return np.reshape(
            self._IP_Ux_inv_PxU @ self.g_prime_mdeim(y @ self.RB.T),
            self.g_prime_shape
        ) @ self.RB

    def g_prime_x_lambda_y_hyperreduced(self, y, lag_mult):
        """Compute g_prime_x_lambda_y for hyperreduced system"""
        return (
            self.RB.T
            @ np.reshape(
                self.IP_g_prime_x_lambda_y
                @ self.g_prime_x_lambda_y_mdeim(y @ self.RB.T, lag_mult),
                self.g_prime_x_lambda_y_shape,
            )
            @ self.RB
        )

    def g_prime_x_lambda_lambda_hyperreduced(self, y, lag_mult):
        """Compute g_prime_x_lambda_lambda for hyperreduced system"""
        return self.RB.T @ np.reshape(
            self.IP_g_prime_x_lambda_lambda
            @ self.g_prime_x_lambda_lambda_mdeim(y @ self.RB.T, lag_mult),
            self.g_prime_x_lambda_lambda_shape,
        )

    def __init__(self, kwds):        
        if 'pool' in kwds:
            self.__dict__.update(kwds['pool'])

        self.ham = self.ham_lambda
        self.beta = (max(1e-2, 0 * rng.random() / 10)) * 0 # Damping coefficient, obsolete
        self.time_lapsed = []

        if self.reducer == 'psd':
             self.JJ = self.compute_J(self.nosc_r if hasattr(self, 'RB') else self.nosc)
        else:
             self.JJ = self.compute_J(self.nosc)

        # Projection matrices
        if hasattr(self, 'RB'):
            self.y_init = self.RB.T @ self.y_init

    def get_en_err(self):
        return np.array([self.ham(y) for y in self.y]) - self.ham(self.y[0])

@load_symbolic_expressions
class HamiltonianMechSystem(MechSystem):
    """System with Hamiltonian structure."""
    def __init__(self, kwds):
        super().__init__(kwds)
        # Map lambda methods to standard names
        self.ham_z = self.ham_z_lambda
        self.ham_zz = self.ham_zz_lambda
        self.g__ = self.g__lambda
        self.g_prime__ = self.g_prime__lambda

@load_symbolic_expressions
class LagrangianMechSystem(MechSystem):
    """System with Lagrangian structure."""
    def __init__(self, kwds):
        super().__init__(kwds)
        # Map lambda methods to standard names
        self.lag_dg = self.lag_dg_lambda
        self.lag_dg_z = self.lag_dg_z_lambda
        self.g__ = self.g__lambda
        self.g_prime__ = self.g_prime__lambda
        self.g_prime_x_lambda_y = self.g_prime_x_lambda_y__
        self.g_prime_x_lambda_lambda = self.g_prime_x_lambda_lambda__

class ReducedHamiltonianMechSystem(HamiltonianMechSystem):
    """Reduced order Hamiltonian system."""
    def __init__(self, kwds):
        # Load solver specific data
        if 'solver_data' in kwds and self.__class__.__name__ in kwds['solver_data']:
            for k, v in kwds['solver_data'][self.__class__.__name__].items():
                setattr(self, k, v)
        super().__init__(kwds)
        
        # Override with reduced methods
        self.ham_z = self.ham_z_reduced
        self.ham_zz = self.ham_zz_reduced
        self.g__ = self.g_reduced
        self.g_prime__ = self.g_prime_reduced

class ReducedLagrangianMechSystem(LagrangianMechSystem):
    """Reduced order Lagrangian system."""
    def __init__(self, kwds):
        # Load solver specific data
        if 'solver_data' in kwds and self.__class__.__name__ in kwds['solver_data']:
            for k, v in kwds['solver_data'][self.__class__.__name__].items():
                setattr(self, k, v)
        super().__init__(kwds)
        
        # Override with reduced methods
        self.lag_dg = self.lag_dg_reduced
        self.lag_dg_z = self.lag_dg_z_reduced
        self.g__ = self.g_reduced
        self.g_prime__ = self.g_prime_reduced
        self.g_prime_x_lambda_y = self.g_prime_x_lambda_y_reduced
        self.g_prime_x_lambda_lambda = self.g_prime_x_lambda_lambda_reduced

class HyperReducedHamiltonianMechSystem(ReducedHamiltonianMechSystem):
    """Hyper-reduced Hamiltonian system."""
    def __init__(self, kwds):
        super().__init__(kwds)
        
        self.ham_z = self.ham_z_hyperreduced
        self.ham_zz = self.ham_zz_hyperreduced
        self.g_prime__ = self.g_prime_hyperreduced
        
        if self.hyperreducer == 'MDEIM':
            self.ham_zz = self.ham_zz_mdeim_hyperreduced

class HyperReducedLagrangianMechSystem(ReducedLagrangianMechSystem):
    """Hyper-reduced Lagrangian system."""
    def __init__(self, kwds):
        super().__init__(kwds)
        
        self.lag_dg = self.lag_dg_hyperreduced
        self.lag_dg_z = self.lag_dg_z_hyperreduced
        self.g_prime__ = self.g_prime_hyperreduced
        
        if self.hyperreducer == 'MDEIM':
            self.lag_dg_z = self.lag_dg_z_mdeim_hyperreduced
            
        if self.constraints_reduce:
            
            self.g_prime_x_lambda_y = self.g_prime_x_lambda_y_hyperreduced
            self.g_prime_x_lambda_lambda = self.g_prime_x_lambda_lambda_hyperreduced
