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
class IndexedBaseSymbolicComputer:
    """
    Computes and stores symbolic expressions using IndexedBase for performance.
    
    Uses a hybrid approach: standard symbols for robust symbolic differentiation,
    and IndexedBase for efficient lambdified functions.
    """

    def __init__(self, nosc):
        self.nosc = nosc
        # 1. Create standard symbols for robust symbolic manipulation
        self.y = smp.Matrix(smp.symbols(f"y_:{nosc*2//3}_:{3}", real=True))
        self.y1 = smp.Matrix(
            smp.symbols("y1_:{}_:{}".format(nosc * 2 // 3, 3), real=True)
        )
        self.omega2 = smp.Matrix(
            smp.symbols("omega^2_:{}".format(self.nosc // 3 - 2), real=True)
        )
        self.beta = smp.symbols("beta", real=True)

        # 2. Create IndexedBase symbols for the lambdify step
        self.y_base = smp.IndexedBase("y")
        self.y1_base = smp.IndexedBase("y1")
        self.omega2_base = smp.IndexedBase("omega2")

        # 3. Create substitution rules
        self.y_subs = {s: self.y_base[i] for i, s in enumerate(self.y)}
        self.y1_subs = {s: self.y1_base[i] for i, s in enumerate(self.y1)}
        self.omega2_subs = {s: self.omega2_base[i] for i, s in enumerate(self.omega2)}

        # Create symbols for lag_mult
        self.lag_mult_syms = smp.Matrix(
            smp.symbols("gamma_:{}".format(self.nosc // 3), real=True)
        )
        self.lag_mult_base = smp.IndexedBase("gamma")
        self.lag_mult_subs = {
            s: self.lag_mult_base[i] for i, s in enumerate(self.lag_mult_syms)
        }

        # --- The rest is the same as the original SymbolicComputer ---
        self._q = smp.Matrix(self.y[:nosc])
        self.q = self._q.reshape(nosc // 3, 3)
        self._p = smp.Matrix(self.y[nosc:])
        self.p = self._p.reshape(nosc // 3, 3)
        self.y05_repl = dict(zip(self.y, (self.y + self.y1) / 2))
        self.y1_repl = dict(zip(self.y, self.y1))
        self.pi = smp.Matrix(
            [
                (self.q.row(i + 2) - self.q.row(i)).norm(2) ** 2
                for i in range(self.nosc // 3 - 2)
            ]
        )

    def _lambdify_hybrid(self, args, expr, custom_subs=None):
        """
        Helper to substitute, lambdify, and wrap for input shape compatibility.
        """
        subs_dict = self.y_subs.copy()
        subs_dict.update(self.y1_subs)
        subs_dict.update(self.omega2_subs)
        subs_dict.update(self.lag_mult_subs)
        if custom_subs:
            subs_dict.update(custom_subs)

        # Substitute standard symbols with IndexedBase symbols
        expr_indexed = expr.subs(subs_dict)

        # The actual lambdified function
        _func = smp.lambdify(args, expr_indexed, "numpy", cse=True)

        # Wrapper to handle input shapes
        @wraps(_func)
        def wrapper(*wrapper_args):
            new_args = []
            for arg in wrapper_args:
                if isinstance(arg, np.ndarray) and arg.ndim > 1:
                    new_args.append(arg.flatten())
                else:
                    new_args.append(arg)
            return _func(*new_args)

        return wrapper

    def compute_hamiltonian(self):
        """Compute Hamiltonian expressions"""
        print("Computing Hamiltonian expressions...")
        kin_expr = 0.5 * self._p.dot(self._p)
        pot_expr_vec = 0.5 * self.omega2.multiply_elementwise(
            (self.pi - smp.ones(*self.pi.shape)).applyfunc(lambda x: x**2)
        )
        pot_expr = sum(pot_expr_vec)
        ham_expr = kin_expr + pot_expr
        ham_z_expr = smp.Matrix([ham_expr]).jacobian(self.y).T
        ham_zz_expr = ham_z_expr.jacobian(self.y)
        ham_zz_nonzero_indices = np.where(
            np.array(ham_zz_expr.tolist()).flatten() != 0
        )[0]

        # Use the hybrid lambdify
        ham_ = self._lambdify_hybrid(
            (self.y_base, self.omega2_base, self.beta), ham_expr
        )
        ham_z_ = self._lambdify_hybrid(
            (self.y_base, self.omega2_base, self.beta), ham_z_expr
        )
        ham_zz_ = smp.lambdify(
            (self.y, self.omega2, self.beta), ham_zz_expr, modules=["numpy"], cse=True
        )

        return {
            "y": self.y,
            "y1": self.y1,
            "omega2": self.omega2,
            "beta": self.beta,
            "ham_expr": ham_expr,
            "ham_z_expr": ham_z_expr,
            "ham_zz_expr": ham_zz_expr,
            "ham_": ham_,
            "ham_z_": ham_z_,
            "ham_zz_": ham_zz_,
            "ham_zz_nonzero_indices": ham_zz_nonzero_indices,
            "pi": self.pi,
            "pot_expr_vec": pot_expr_vec,
        }

    def compute_lagrangian(self, ham_z_expr):
        """Compute Lagrangian expressions"""
        print("Computing Lagrangian expressions...")
        dpi_dq = self.pi.jacobian(self._q).subs(self.y05_repl)
        dV_dpi = 0.5 * self.omega2.multiply_elementwise(
            self.pi.subs(self.y1_repl) + self.pi - 2 * smp.ones(*self.pi.shape)
        )
        DG_V_expr = dpi_dq.T @ dV_dpi
        lag_dg_expr = smp.Matrix.vstack(
            ham_z_expr[self.nosc :, :].subs(self.y05_repl), -DG_V_expr
        )

        lag_dg_z_expr = lag_dg_expr.jacobian(self.y1)
        lag_dg_z_nonzero_indices = np.where(
            np.array(lag_dg_z_expr.tolist()).flatten() != 0
        )[0]

        # Hybrid lambdify
        lag_dg_ = self._lambdify_hybrid(
            (self.y_base, self.y1_base, self.omega2_base), lag_dg_expr
        )
        lag_dg_z_ = smp.lambdify(
            (self.y, self.y1, self.omega2), lag_dg_z_expr, modules=["numpy"], cse=True
        )

        return {
            "lag_dg_expr": lag_dg_expr,
            "lag_dg_": lag_dg_,
            "lag_dg_z_expr": lag_dg_z_expr,
            "lag_dg_z_": lag_dg_z_,
            "lag_dg_z_nonzero_indices": lag_dg_z_nonzero_indices,
        }

    def compute_constraints(self):
        """Compute constraint expressions"""
        print("Computing constraint expressions...")
        row_diffs_q = [
            self.q.row((i + 1)) - self.q.row(i) for i in range(0, self.nosc // 3, 2)
        ]
        row_diffs_p = [
            self.p.row((i + 1)) - self.p.row(i) for i in range(0, self.nosc // 3, 2)
        ]
        row_norms = [(row_diff.dot(row_diff) - 1) / 2 for row_diff in row_diffs_q]
        ddt_row_norms = [
            row_diffs_q[i].dot(row_diffs_p[i]) for i in range(len(row_diffs_q))
        ]
        g_expr = smp.Matrix(row_norms + ddt_row_norms)
        g_prime_expr = g_expr.jacobian(self.y)

        # Create symbols for lag_mult
        lag_mult_syms = self.lag_mult_syms

        g_prime_x_lambda_expr = g_prime_expr.T @ lag_mult_syms
        g_prime_x_lambda_y_expr = g_prime_x_lambda_expr.jacobian(self.y)
        g_prime_x_lambda_lambda_expr = g_prime_x_lambda_expr.jacobian(lag_mult_syms)
        g_prime_x_lambda_y_nonzero_indices = np.where(
            np.array(g_prime_x_lambda_y_expr.tolist()).flatten() != 0
        )[0]
        g_prime_x_lambda_lambda_nonzero_indices = np.where(
            np.array(g_prime_x_lambda_lambda_expr.tolist()).flatten() != 0
        )[0]

        # Lambdify for these expressions requires custom substitutions
        g_prime_x_lambda_y_lam = smp.lambdify(
            (self.y, lag_mult_syms), g_prime_x_lambda_y_expr, modules=["numpy"], cse=True
        )
        g_prime_x_lambda_lambda_lam = smp.lambdify(
            (self.y, lag_mult_syms), g_prime_x_lambda_lambda_expr, modules=["numpy"], cse=True
        )

        g_prime_nonzero_indices = np.where(
            np.array(g_prime_expr.tolist()).flatten() != 0
        )[0]

        g_lam = self._lambdify_hybrid((self.y_base,), g_expr)
        g_prime_lam = smp.lambdify((self.y,), g_prime_expr, modules=["numpy"], cse=True)

        g_prime_x_lambda_ = self._lambdify_hybrid(
            (self.y_base, self.lag_mult_base), g_prime_x_lambda_expr
        )

        return {
            "g_expr": g_expr,
            "g_prime_expr": g_prime_expr,
            "g_": g_lam,
            "g_prime_": g_prime_lam,
            "g_prime_nonzero_indices": g_prime_nonzero_indices,
            "lag_mult": lag_mult_syms,
            "g_prime_x_lambda_expr": g_prime_x_lambda_expr,
            "g_prime_x_lambda_": g_prime_x_lambda_,
            "g_prime_x_lambda_y_expr": g_prime_x_lambda_y_expr,
            "g_prime_x_lambda_lambda_expr": g_prime_x_lambda_lambda_expr,
            "g_prime_x_lambda_y_": g_prime_x_lambda_y_lam,
            "g_prime_x_lambda_lambda_": g_prime_x_lambda_lambda_lam,
            "g_prime_x_lambda_y_nonzero_indices": g_prime_x_lambda_y_nonzero_indices,
            "g_prime_x_lambda_lambda_nonzero_indices": g_prime_x_lambda_lambda_nonzero_indices,
        }

    @timing
    def compute_all(self, expressions):
        """Compute all symbolic expressions"""
        ham_exprs = self.compute_hamiltonian()
        expressions.update(ham_exprs)
        expressions.update(self.compute_lagrangian(ham_exprs["ham_z_expr"]))
        expressions.update(self.compute_constraints())


def load_symbolic_expressions(cls):
    """Decorator to handle loading/saving of symbolic expressions"""
    # Only run this on the main process, not on Dask workers.
    # Workers will receive the class with methods already attached via pickling.
    if 'DASK_WORKER_NAME' in os.environ:
        return cls

    try:
        # Try to load expressions
        expressions_file = os.path.join('data', f"symbolic_expr_{cls.nosc}_cse.pickle")
        with open(expressions_file, 'rb') as f:
            expressions = pickle.load(f)

        print("Loaded symbolic expressions from disk.")
    except (FileNotFoundError, std_pickle.UnpicklingError):
        # Compute and save if loading fails
        sys.setrecursionlimit(50000)
        computer = IndexedBaseSymbolicComputer(cls.nosc)
        expressions = {}
        tl = computer.compute_all(expressions)
        print(f'Computed symbolic expressions in {tl:.2f} seconds.')
        os.makedirs('data', exist_ok=True)
        with open(expressions_file, 'wb') as f:
            pickle.dump(expressions, f)
        print("Saved symbolic expressions to disk.")

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

    # Numerical solver and its properties
    w_values = [0.28, 0.62546642846767004501]
    w_values.append(1.0 - 2.0 * (sum(w_values)))
    w_values.append(w_values[1])
    w_values.append(w_values[0])
    # w_values = [1]
    assert np.isclose(sum(w_values), 1), 'sum_i w_i must be 1'
    
    # Solver and reduction settings
    tol, M, var, store = 1.0E-12, 500, True, False
    
    predict = False  # False = reproduce results of the full model
    reducer = 'psd'
    hyperreducer = 'MDEIM'
    constraint_type = None # 'spherical'
    constraints_reduce = False
    
    # System parameters
    nosc = 54  # Number of oscillators
    assert nosc % 6 == 0, 'nosc must be divisible by 6'

    if not predict: # reproduction parameters
        # Time-stepping parameters
        dt_space_dim = 3
        dt_space = np.round(np.logspace(-3, -2, num=dt_space_dim), 10)
        T_final = dt_space[-1] * 5e1

        # Parameter space for Omega^2
        _Omega2_space_dim = 3
    else: # prediction parameters
        # Time-stepping parameters
        dt_space_dim = 1
        dt_space = np.array([0.01])
        T_final = dt_space[-1] * 1e3

        # Parameter space for Omega^2
        _Omega2_space_dim = 10

    _Omega2_space = np.sort(10 * (1 - rng.random((_Omega2_space_dim, nosc // 3 - 2))))
    _Omega2_space[:, 1 * _Omega2_space_dim - 1:] = _Omega2_space[0, 1 * _Omega2_space_dim - 1:]

    # Train-test split
    Omega2_space = _Omega2_space[:-1] if predict else _Omega2_space
    Omega2_space_test = _Omega2_space[-1:] if predict else _Omega2_space

@load_symbolic_expressions
class MechSystem(SysConfig):
    """
    Represents the mechanical system, including parameters, initial conditions, 
    and methods for evaluating system dynamics (Hamiltonian, Lagrangian, constraints).
    """

    keep_time = datetime.now().strftime("%Y-%m-%d")
    data_folder = os.path.join('data', keep_time)
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

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
                self.IP_g_prime_x_lambda_y_dg
                @ self.g_prime_x_lambda_y_mdeim_dg(y @ self.RB.T, lag_mult),
                self.g_prime_x_lambda_y_shape_dg,
            )
            @ self.RB
        )

    def g_prime_x_lambda_lambda_hyperreduced(self, y, lag_mult):
        """Compute g_prime_x_lambda_lambda for hyperreduced system"""
        return self.RB.T @ np.reshape(
            self.IP_g_prime_x_lambda_lambda_dg
            @ self.g_prime_x_lambda_lambda_mdeim_dg(y @ self.RB.T, lag_mult),
            self.g_prime_x_lambda_lambda_shape_dg,
        )

    def __init__(self, kwds):        
        self.__dict__.update(kwds)        
        if 'pool' in kwds:
            self.__dict__.update(kwds['pool'])

        self.beta = (max(1e-2, 0 * rng.random() / 10)) * 0 # Damping coefficient, obsolete
        self.time_lapsed = []

        # Determine the model state and set methods accordingly
        is_hyperreduced = hasattr(self, 'RBxUx_inv_PxU') or hasattr(self, '_RBxUx_inv_PxU_')
        is_reduced = hasattr(self, 'RB') or hasattr(self, 'RB_dg')

        if is_hyperreduced:
            self._set_hyperreduced_methods()
        elif is_reduced:
            self._set_reduced_methods()
        else:
            self._set_full_order_methods()
            
        # Update functions based on solver type
        self.ham = self.ham_lambda

        # Projection matrices
        if hasattr(self, 'RB'):
            self.y_init = self.RB.T @ self.y_init
        
        # Initialize symplectic matrix J
        if hasattr(self, 'RB') and self.reducer == 'psd':
            self.JJ = self.compute_J(self.nosc_r)
        elif not hasattr(self, 'RB'):
            self.JJ = self.compute_J(self.nosc)

    def _set_full_order_methods(self):
        """Assigns methods for the full-order model."""
        print('Setting full order functions...')
        self.ham_z = self.ham_z_lambda
        self.ham_zz = self.ham_zz_lambda
        self.lag_dg = self.lag_dg_lambda
        self.lag_dg_z = self.lag_dg_z_lambda
        self.g__ = self.g__lambda
        self.g_prime__ = self.g_prime__lambda
        self.g_prime_x_lambda_y = self.g_prime_x_lambda_y__
        self.g_prime_x_lambda_lambda = self.g_prime_x_lambda_lambda__

    def _set_reduced_methods(self):
        """Assigns methods for the reduced-order model."""
        print('Setting reduced order functions...')
        if self.solver_class.__name__ == 'DiscreteGradientSolver':
            print('Setting RB to RB_dg...', flush=True)
            self.RB = self.RB_dg
            self.nosc_r = self.nosc_r_dg
            self.g_prime_x_lambda_y = self.g_prime_x_lambda_y_reduced
            self.g_prime_x_lambda_lambda = self.g_prime_x_lambda_lambda_reduced

        self.ham_z = self.ham_z_reduced
        self.ham_zz = self.ham_zz_reduced
        self.lag_dg = self.lag_dg_reduced
        self.lag_dg_z = self.lag_dg_z_reduced
        self.g__ = self.g_reduced
        self.g_prime__ = self.g_prime_reduced

    def _set_hyperreduced_methods(self):
        """Assigns methods for the hyper-reduced model."""
        print('Setting hyperreduced functions...')
        if self.solver_class.__name__ == 'DiscreteGradientSolver':
            print('Setting RB to RB_dg...', flush=True)
            self.RB = self.RB_dg
            self.nosc_r = self.nosc_r_dg
            self.RBxUx_inv_PxU = self._RBxUx_inv_PxU_
            if self.hyperreducer == 'MDEIM':
                self.IP_Ux_inv_PxU = self._IP_Ux_inv_PxU_
            if self.constraints_reduce:
                self._IP_Ux_inv_PxU = self._IP_Ux_inv_PxU_dg
                # self.g_deim = self.g_deim_dg
                self.g_prime_mdeim = self.g_prime_mdeim_dg
                self.g_prime_shape = self.g_prime_shape_dg
                self.g_prime_x_lambda_y = self.g_prime_x_lambda_y_hyperreduced
                self.g_prime_x_lambda_lambda = self.g_prime_x_lambda_lambda_hyperreduced

        self.ham_z = self.ham_z_hyperreduced
        self.ham_zz = self.ham_zz_hyperreduced
        self.lag_dg = self.lag_dg_hyperreduced
        self.lag_dg_z = self.lag_dg_z_hyperreduced
        self.g__ = self.g_reduced  # Note: g uses reduced, not hyperreduced
        self.g_prime__ = self.g_prime_hyperreduced

        if self.hyperreducer == 'MDEIM':
            self.ham_zz = self.ham_zz_mdeim_hyperreduced
            self.lag_dg_z = self.lag_dg_z_mdeim_hyperreduced

    def get_en_err(self):
        return np.array([self.ham(y) for y in self.y]) - self.ham(self.y[0])
