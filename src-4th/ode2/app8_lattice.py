#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

"""
Created on June 2024

@author: ashishbhatt

This app solves a full order and reduced-order model from the System class,
using solver classes in the ODESolver hierarchy of methods,
fixed point iterators from the Newton.py,
and reduced bases from podDEIM.

Classes:
    BaseSolverMixin: Base mixin class providing common solving capabilities for solvers.
    HamiltonianSolverMixin: Mixin for Hamiltonian-based solvers.
    DiscreteGradientMixin: Mixin for Discrete Gradient specific computations.
    DiscreteGradientSolver: DiscreteGradient solver with DG-specific capabilities.
    ConformalStormerVerletSolver: CSVSolver with Hamiltonian capabilities.
    ConformalImplicitMidpointSolver: CIMSolver with Hamiltonian capabilities.
Functions:
    compose_solver_solves(func): Decorator to compose solver solves.
    solve_mech_system(solver_class, dt, Omega2, kwds): Helper function to solve mechanical system.
    parallel_solve_mech_system(kwds, Omega2_space, MSsolvers): Parallelize the solve_mech_system calls.
    setup_and_solve_reduced_system(solver_type, solvers, kwds_base): Setup and solve reduced system for a specific solver type.
    setup_and_solve_hyperreduced_system(solver_type, solvers, kwds_base): Setup and solve hyper-reduced system for a specific solver type.
Main driver:
    Model order reduction of the MechSystem using concrete solvers.
"""

import numpy as np
import sympy as smp
from numpy import linalg as LA
from scipy import linalg as scipyLA
from pylab import  log, r_, c_, sqrt, reshape, linspace, roll, figure

import concurrent.futures
import multiprocessing
from dask_jobqueue import SLURMCluster
from dask.distributed import Client, wait

import sys
import shutil
import gc
from joblib import dump, load
from functools import partial, wraps
from tqdm.auto import tqdm  # Use tqdm.auto for better process handling

from matplotlib import rc

from ODESolver import (ConformalStormerVerlet, ConformalImplicitMidpoint, 
                      DiscreteGradient)
# from Newton import fixed_point
from System import MechSystem  # Keep for static properties
from podDEIM import POD, PSD, DEIM
from PlotScript import plot_data, tex_table, logplot, timing, save_figure

# Configure LaTeX rendering based on availability
if False and shutil.which('latex'):
    rc('text', usetex=True)
    rc('text.latex', preamble=r'\usepackage{amsfonts}')  # Load AMSFonts for Fraktur
else:
    rc('text', usetex=False)
    print("LaTeX disabled or not found. Using standard fonts for plots.", flush=True)

def worker(idx, *arg):
    # pin this process to a specific core (0–23 in this example)
    print(f"Worker {idx} starting...", flush=True)
    # try:
    #     cpus = sorted(os.sched_getaffinity(0))
    #     core_id = cpus[idx % len(cpus)]
    #     os.sched_setaffinity(0, {core_id})
    # except (AttributeError, OSError):
    #     pass
    # run your solver
    try:
        return BaseSolverMixin.solve_mech_system(*arg)
    except Exception as e:
        print(f"Worker {idx} failed with error: {e}", flush=True)
        raise

# %%
def compose_solver_solves(func):
    @wraps(func)
    def wrapper(self, y_, k):
        for w_val in self.w_values:
            y_, y_full_, Lambda = func(self, w_val, y_, k)
                
        return y_, y_full_, Lambda
    return wrapper

class ReduceMechSystem(MechSystem):
    """
    Reducer class providing common reduction and hyperreduction capabilities to MechSystem.

    Methods:
    - solve_for_w: Solves for a given w value and updates the solution.
    - handle_constraints: Enforces constraints on the solution.
    - solve_trajectory: Solves the entire trajectory of the system.
    - plot: Plots the results of the simulation.
    - measures: Computes various measurements based on the solution.
    - hyperreduce_constraints: Hyper-reduces constraints using MDEIM.
    """
    @classmethod
    def filter_solvers(cls, solvers, solver_type):
        """
        Filter solvers by type and return filtered solvers and indices.
        """
        # Filter solvers by type
        if solver_type == "Hamiltonian":
            filtered_solvers = [s for s in solvers if not isinstance(s, DiscreteGradient)]
        elif solver_type == "DiscreteGradient":
            filtered_solvers = [s for s in solvers if isinstance(s, DiscreteGradient)]
        else:
            raise ValueError(f"Invalid solver type: {solver_type}")

        # Create a mapping from solvers to their indices
        solver_to_indices = {solver: indices for solver, indices in zip(solvers, cls.indices_list)}

        # Filter indices_list based on filtered_solvers
        filtered_indices = [solver_to_indices[solver] for solver in filtered_solvers]
        return filtered_solvers, filtered_indices

    @classmethod
    def setup_reduced_model(cls, solvers):
        """Setup reduced basis"""
        print('Computing reduced basis...')

        def compute_batch_snapshots(solver, indices, expr, batch_size=100):
            """Process snapshots in batches"""
            result = []
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                if isinstance(solver, DiscreteGradient):  # For DiscreteGradient
                    batch_results = [expr(y) for y in zip(solver.y[batch_indices], 
                                                        solver.y[np.array(batch_indices)+1])]
                else:  # For Hamiltonian
                    batch_results = [expr(y) for y in solver.y[batch_indices]]
                result.extend(batch_results)
                if (i + batch_size) % 100 == 0:
                    gc.collect()

            return np.array(result).T

        def compute_reduced_basis(solver_type, expr, attr_name):
            filtered_solvers, filtered_indices = cls.filter_solvers(solvers, solver_type)

            # Create snapshot list
            y_list = np.hstack([solver.y[indices].T for solver, indices 
                                in zip(filtered_solvers, filtered_indices)])

            if True or solver_type == "Hamiltonian":
                # Create F2 in batches
                F2_list = []
                for solver, indices in zip(filtered_solvers, filtered_indices):
                    batch_result = compute_batch_snapshots(solver, indices, expr)
                    F2_list.append(batch_result)

                F2 = np.hstack(F2_list)
                del F2_list
                gc.collect()

                # Calculate weight ratio based on norms
                norm_y = LA.norm(y_list)
                norm_F2 = LA.norm(F2)
                weight_ratio = (norm_y / norm_F2) if norm_F2 > 1e-12 else 1.0
                
                # Create a diagonal weight matrix for the PSD inner product
                # This implicitly scales the F2 contribution during SVD without altering data
                weights = np.eye(cls.nosc) * weight_ratio

                RB, sv, nosc_r = PSD(F2, y_list, cls) #, weights=weights)

            else:
                RB, sv, nosc_r = PSD(y_list[:, 0, None], y_list, cls)

            del y_list
            gc.collect()

            setattr(MechSystem, attr_name[0], RB)
            setattr(MechSystem, attr_name[1], nosc_r)
            print(f'{RB.shape = }')

            fig, ax = logplot(sv, xlabel=f'index of singular values of [F2, y_list]', xlims=(1, len(sv)))
            filename = os.path.join(
                MechSystem.data_folder, "osc_sv_rb" + solver_type + ".pdf"
            )
            save_figure(fig, filename, fig_data=None)

        # Compute reduced basis for Hamiltonian solvers
        if ConformalStormerVerletSolver in kwds['registered_solver_classes'] \
            or ConformalImplicitMidpointSolver in kwds['registered_solver_classes']:
            compute_reduced_basis("Hamiltonian", solvers[0].ham_z, ["RB", "nosc_r"])

        # Compute reduced basis for Discrete Gradient solvers if applicable
        if DiscreteGradientSolver in kwds['registered_solver_classes']:
            compute_reduced_basis("DiscreteGradient", solvers[0].lag_dg, ["RB_dg", "nosc_r_dg"])

    @classmethod
    def _create_indexed_deim_func(cls, expr, P, solver_type):
        """
        Helper method to create a lambdified function with IndexedBase symbols.
        """
        # 1. Define IndexedBase symbols
        y_base = smp.IndexedBase("y")
        y1_base = smp.IndexedBase("y1")
        omega2_base = smp.IndexedBase("omega2")

        # 2. Create substitution dictionary from the class's standard symbols
        subs_dict = {s: y_base[i] for i, s in enumerate(cls.y)}
        subs_dict.update({s: y1_base[i] for i, s in enumerate(cls.y1)})
        subs_dict.update({s: omega2_base[i] for i, s in enumerate(cls.omega2)})

        # 3. Substitute the expression to use IndexedBase symbols
        expr_to_lambdify = P.T @ expr.flat()
        expr_indexed = smp.Matrix(expr_to_lambdify).subs(subs_dict)

        # 4. Define arguments for lambdify and create the function
        y_symbols = set(cls.y)
        is_y_only = all(s in y_symbols for s in expr.free_symbols)

        if is_y_only:
            args = (y_base,)
        else:
            args = (
                (y_base, omega2_base, cls.beta)
                if solver_type == "Hamiltonian"
                else (y_base, y1_base, omega2_base)
            )
        deim_func = smp.lambdify(args, expr_indexed, modules=["scipy"])

        return lambda *args, **kwargs: np.squeeze(deim_func(*args, **kwargs))

    @classmethod
    def setup_hyperreduction(cls):
        """Setup hyperreduction as classmethod"""
        print('Setting up hyperreduction...')

        def compute_hyperreduction_basis(solver_type, expr, attr, deim_attr_name, deim_func_name, expr_z=None, mdeim_func_name=None):
            attr_11 = attr[:cls.nosc, :cls.nosc_r] if solver_type == "Hamiltonian" else attr[:cls.nosc, :cls.nosc_r_dg]
            P11, _ = DEIM(attr_11, plot_deim=False)

            attr_22 = attr[cls.nosc:, cls.nosc_r:] if solver_type == "Hamiltonian" else attr[cls.nosc:, cls.nosc_r_dg:]
            P22, _ = DEIM(attr_22, plot_deim=False)

            P = np.block([
                [P11, np.zeros((P11.shape[0], P22.shape[1]))],
                [np.zeros((P22.shape[0], P11.shape[1])), P22]
            ])
            setattr(MechSystem, deim_attr_name, attr.T @ attr @ LA.inv(P.T @ attr))
            print(f'{P.shape = }')

            # if solver_type == "Hamiltonian":
            #     cls._IP_Ux_inv_PxU = np.block([
            #         [attr_11 @ LA.inv(P11.T @ attr_11), np.zeros((attr_11.shape[0], attr_22.shape[1]))],
            #         [np.zeros((attr_22.shape[0], attr_11.shape[1])), attr_22 @ LA.inv(P22.T @ attr_22)]
            #     ])

            deim_func = cls._create_indexed_deim_func(expr, P, solver_type)

            if callable(deim_func) and not isinstance(deim_func, type):
                # Wrap lambda functions to include self parameter
                wrapped = (lambda f: lambda self, *args, **kwargs: f(*args, **kwargs))(deim_func)
                setattr(MechSystem, deim_func_name, wrapped)
            else:
                print(f"Memory address of {deim_func_name}: {hex(id(deim_func))}")

            # Create and wrap extra deim function if provided
            if expr_z is not None:
                mdeim_func = smp.lambdify((cls.y, cls.omega2, cls.beta) if solver_type == "Hamiltonian" else (cls.y, cls.y1, cls.omega2),
                                        P.T @ expr_z,
                                        modules=['scipy'])

                if callable(mdeim_func) and not isinstance(mdeim_func, type):
                    wrapped = (lambda f: lambda self, *args, **kwargs: f(*args, **kwargs))(mdeim_func)
                    setattr(MechSystem, mdeim_func_name, wrapped)
                else:
                    print(f"Memory address of {mdeim_func_name}: {hex(id(mdeim_func))}")

        # Update the calls to compute_hyperreduction_basis
        if ConformalStormerVerletSolver in kwds['registered_solver_classes'] \
            or ConformalImplicitMidpointSolver in kwds['registered_solver_classes']:
            compute_hyperreduction_basis(
                "Hamiltonian", 
                cls.ham_z_expr, 
                cls.RB, 
                "RBxUx_inv_PxU", 
                "ham_z_deim",
                cls.ham_zz_expr if cls.hyperreducer == 'DEIM' else None,
                "ham_zz_deim" if cls.hyperreducer == 'DEIM' else None
            )

        if DiscreteGradientSolver in kwds['registered_solver_classes']:
            compute_hyperreduction_basis(
                "DiscreteGradient",
                cls.lag_dg_expr,
                cls.RB_dg,
                "_RBxUx_inv_PxU_",
                "lag_dg_deim",
                cls.lag_dg_z_expr if cls.hyperreducer == 'DEIM' else None,
                "lag_dg_z_deim" if cls.hyperreducer == 'DEIM' else None
            )

    @staticmethod
    def compute_snapshot(solver, indices, IP, is_dg=False, batch_size=32):
        """Memory efficient batch processing of matrix multiplications"""
        result = []
        batch_size = len(indices)

        # Process indices in batches
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]

            # Pre-allocate batch matrices
            if is_dg:
                batch_mats = np.vstack([
                    solver.lag_dg_z((solver.y[idx], solver.y[idx+1])).flatten().astype(np.float32)
                    for idx in batch_indices
                ])
            else:
                batch_mats = np.vstack([
                    solver.ham_zz(solver.y[idx]).flatten().astype(np.float32)
                    for idx in batch_indices
                ])

            # Compute batch matrix multiplication
            batch_results = IP @ batch_mats.T  # More efficient than multiple small multiplications
            result.extend(batch_results.T)

            # Force garbage collection after each batch
            if (i + batch_size) % 100 == 0:
                gc.collect()

        return result

    @classmethod
    def update_mdeim_hyperreduction(cls, solvers):
        """Update methods with MDEIM hyperreduction"""

        def compute_mdeim_basis(solver_type, func_name, expr, attr_name, deim_func_name):
            """Helper function to compute MDEIM basis and create lambdified functions"""
            # NOTE: symbolic matrix multiplication in high-precision arithmetic can become expensive
            # and may cause memory overflow. This can lead to a program crash.
            # Though _IP_Ux_inv_PxU_ @ lag_dg_z_col can be done here once and for all solvers.
            print(f'Computing {func_name} reduction...')

            # Collect snapshots, load non-zero indices stored in func_name_nonzero_indices
            # and create interpolation matrix IP
            non_zero_indices = getattr(cls, func_name + "_nonzero_indices")
            IP = np.zeros((len(non_zero_indices), (2*cls.nosc)**2), dtype=np.int8)
            IP[np.arange(len(non_zero_indices)), non_zero_indices] = 1

            # Filter solvers and collect snapshots
            filtered_solvers, filtered_indices = cls.filter_solvers(solvers, solver_type)

            # Create snapshot matrix sequentially
            if solver_type == "DiscreteGradient":
                worker_func = partial(ReduceMechSystem.compute_snapshot, IP=IP, is_dg=True)
            else:
                worker_func = partial(ReduceMechSystem.compute_snapshot, IP=IP, is_dg=False)

            results = []
            for solver, indices in zip(filtered_solvers, filtered_indices):
                result = worker_func(solver, indices)
                results.append(result)

            # # Use a ProcessPoolExecutor for parallelism
            # with concurrent.futures.ProcessPoolExecutor() as executor:
            #     futures = [
            #         executor.submit(worker_func, solver, indices)
            #         for solver, indices in zip(filtered_solvers, filtered_indices)
            #     ]
            #     results = [f.result() for f in concurrent.futures.as_completed(futures)]

            F3 = np.hstack([np.array(result).T for result in results])

            # Compute POD basis and plot singular values
            Uj, sv, _ = POD(F3, np.eye(F3.shape[0]), cls.tol)
            fig, ax = logplot(sv, xlabel=f'index of singular values of {func_name}', xlims=(1, len(sv)))
            filename = os.path.join(
                MechSystem.data_folder, "osc_sv_mdeim" + solver_type + ".pdf"
            )
            save_figure(fig, filename, fig_data=None)

            # Compute DEIM points and interpolation matrix
            Pj, _ = DEIM(Uj, plot_deim=False)
            setattr(MechSystem, attr_name, IP.T @ Uj @ LA.inv(Pj.T @ Uj))
            print(f'{getattr(MechSystem, attr_name).shape = }')

            # Create lambdified function
            mdeim_col = Pj.T @ IP @ expr.flat()
            mdeim_func = smp.lambdify((cls.y, cls.y1, cls.omega2) if solver_type == "DiscreteGradient" 
                                    else (cls.y, cls.omega2, cls.beta),
                                    mdeim_col, modules=['scipy'])

            if callable(mdeim_func) and not isinstance(mdeim_func, type):
                wrapped = (lambda f: lambda self, *args, **kwargs: f(*args, **kwargs))(mdeim_func)
                setattr(MechSystem, deim_func_name, wrapped)
            else:
                print(f"Memory address of {deim_func_name}: {hex(id(mdeim_func))}")

            print(f'{deim_func_name} has been updated for solver type: {solver_type}')

        # Compute MDEIM basis for DiscreteGradient solvers
        if DiscreteGradientSolver in kwds['registered_solver_classes']:
            compute_mdeim_basis(
                "DiscreteGradient",
                "lag_dg_z",
                cls.lag_dg_z_expr,
                "_IP_Ux_inv_PxU_",
                "lag_dg_z_mdeim"
            )

        # Compute MDEIM basis for Hamiltonian solvers
        if ConformalStormerVerletSolver in kwds['registered_solver_classes'] \
            or ConformalImplicitMidpointSolver in kwds['registered_solver_classes']:
            compute_mdeim_basis(
                "Hamiltonian",
                "ham_zz",
                cls.ham_zz_expr,
                "IP_Ux_inv_PxU",
                "ham_zz_mdeim"
            )

    @classmethod
    def hyperreduce_constraints(cls, solvers, indices_list, solver_type):
        """Hyper-reduce constraints using MDEIM."""
        print('Computing constraints reduction...')

        def compute_constraint_basis(func_name, expr, is_g_prime=False):
            """
            Helper function to compute MDEIM basis for constraints.
            Supports g, g_prime, g_prime_x_lambda_y, and g_prime_x_lambda_lambda.
            """
            print(f'Computing {func_name} reduction...')

            # Determine if this is a "prime" type (g_prime, g_prime_x_lambda_y, g_prime_x_lambda_lambda)
            prime_types = ['g_prime_', 'g_prime_x_lambda_y_', 'g_prime_x_lambda_lambda_']
            is_prime_type = is_g_prime or func_name in prime_types

            if is_prime_type:
                # Get shape for the prime constraint
                if func_name in ['g_prime_x_lambda_y_', 'g_prime_x_lambda_lambda_']:
                    g_prime_shape = solvers[0].__getattribute__(func_name)(solvers[0].y[0], solvers[0].Lambda[0]).shape
                else:
                    g_prime_shape = solvers[0].__getattribute__(func_name)(solvers[0].y[0]).shape
                setattr(MechSystem, func_name + 'shape' if solver_type == "Hamiltonian" else func_name + 'shape_dg', g_prime_shape)

                non_zero_indices = getattr(cls, func_name + "nonzero_indices")
                IP = np.zeros((len(non_zero_indices), np.prod(g_prime_shape)), dtype=np.int8)
                IP[np.arange(len(non_zero_indices)), non_zero_indices] = 1

                # Create snapshot matrix for prime constraints
                F = np.hstack([
                    np.array([
                    IP @ (
                        solver.__getattribute__(func_name)(y, Lambda).flatten()
                        if func_name in ['g_prime_x_lambda_y_', 'g_prime_x_lambda_lambda_']
                        else solver.g_prime__(y).flatten()
                    )
                    for y, Lambda in zip(solver.y[indices], solver.Lambda[indices])
                    ]).T
                    for solver, indices in zip(solvers, indices_list)
                ])
            else:
                # Create snapshot matrix for g
                F = np.hstack([
                    np.array([solver.g(y) for y in solver.y[indices]]).T
                    for solver, indices in zip(solvers, indices_list)
                ])

            # Compute POD basis and plot singular values
            Uj, sv, _ = POD(F, np.eye(F.shape[0]), cls.tol)
            del F

            # Compute DEIM points and interpolation matrix
            Pj, _ = DEIM(Uj, plot_deim=False)

            if is_prime_type:
                basis = IP.T @ Uj @ LA.inv(Pj.T @ Uj)
                # Create lambdified function
                col = Pj.T @ IP @ expr.reshape(np.prod(g_prime_shape), 1)
            else:
                basis = Uj @ LA.inv(Pj.T @ Uj)
                col = Pj.T @ expr

            print(f'{basis.shape = }')

            # Create and wrap lambdified function
            # Handle argument signature for g_prime_x_lambda_y and g_prime_x_lambda_lambda
            if func_name in ['g_prime_x_lambda_y_', 'g_prime_x_lambda_lambda_']:
                mdeim_func = smp.lambdify((cls.y, cls.lag_mult), col, modules=['scipy'])
            elif func_name in ["g_"]:
                # mdeim_func = smp.lambdify((cls.y,), col, modules=['scipy'])
                mdeim_func = cls._create_indexed_deim_func(expr, Pj, solver_type)
            else:
                mdeim_func = smp.lambdify((cls.y,), col, modules=['scipy'])

            if callable(mdeim_func) and not isinstance(mdeim_func, type):
                wrapped = (lambda f: lambda self, *args, **kwargs: f(*args, **kwargs))(mdeim_func)
                # Set attributes based on solver type
                if solver_type == "Hamiltonian":
                    setattr(MechSystem, f'{func_name}{"mdeim" if is_prime_type else "deim"}', wrapped)
                else:  # DiscreteGradient
                    setattr(MechSystem, f'{func_name}{"mdeim" if is_prime_type else "deim"}_dg', wrapped)
            else:
                print(f"Memory address of {func_name}{'mdeim' if is_prime_type else 'deim'}: {hex(id(mdeim_func))}")

            return basis, sv

        # Compute bases for g and g_prime
        _Ux_inv_PxU, sv_g = compute_constraint_basis('g_', cls.g_expr)
        _IP_Ux_inv_PxU, sv_g_prime = compute_constraint_basis('g_prime_', cls.g_prime_expr, is_g_prime=True)

        # Set attributes based on solver type
        if solver_type == "Hamiltonian":
            setattr(MechSystem, '_Ux_inv_PxU', _Ux_inv_PxU)
            # setattr(MechSystem, 'g_prime_mdeim', cls.g_prime_)
            setattr(MechSystem, '_IP_Ux_inv_PxU', _IP_Ux_inv_PxU)
            # setattr(MechSystem, 'IP_g_prime_x_lambda_y', IP_g_prime_x_lambda_y)
            # setattr(MechSystem, 'IP_g_prime_x_lambda_lambda', IP_g_prime_x_lambda_lambda)
        elif solver_type == "DiscreteGradient":
            # Also compute bases for cls.g_prime_x_lambda_y and cls.g_prime_x_lambda_lambda
            IP_g_prime_x_lambda_y, sv_g_prime_x_lambda_y = compute_constraint_basis(
                "g_prime_x_lambda_y_", cls.g_prime_x_lambda_y_expr, is_g_prime=True
            )
            IP_g_prime_x_lambda_lambda, sv_g_prime_x_lambda_lambda = (
                compute_constraint_basis(
                    "g_prime_x_lambda_lambda_",
                    cls.g_prime_x_lambda_lambda_expr,
                    is_g_prime=True,
                )
            )

            setattr(MechSystem, '_Ux_inv_PxU_dg', _Ux_inv_PxU)
            setattr(MechSystem, '_IP_Ux_inv_PxU_dg', _IP_Ux_inv_PxU)
            setattr(MechSystem, 'IP_g_prime_x_lambda_y_dg', IP_g_prime_x_lambda_y)
            setattr(MechSystem, 'IP_g_prime_x_lambda_lambda_dg', IP_g_prime_x_lambda_lambda)
        else:
            raise ValueError(f"Invalid solver type: {solver_type}")

        # Plot singular values for g, g_prime, g_prime_x_lambda_y and g_prime_x_lambda_lambda using logplot, pass the singular values and xlims as a list
        if "sv_g_prime" in locals():
            sv_list = [sv_g, sv_g_prime]
            xlims_list = [(1, len(sv_g)), (1, len(sv_g_prime))]
        else:
            sv_list = [sv_g]
            xlims_list = [(1, len(sv_g))]

        # Add additional singular values if solver type is DiscreteGradient
        if solver_type == "DiscreteGradient":
            sv_list.extend([sv_g_prime_x_lambda_y, sv_g_prime_x_lambda_lambda])
            xlims_list.extend(
                [(1, len(sv_g_prime_x_lambda_y)), (1, len(sv_g_prime_x_lambda_lambda))]
            )

        fig, ax = logplot(sv_list, xlabel=f"index of singular values", xlims=xlims_list)
        filename = os.path.join(
            MechSystem.data_folder, "osc_sv_constraints" + solver_type + ".pdf"
        )
        save_figure(fig, filename, fig_data=None)

        print('Constraints reduction complete.')

class BaseSolverMixin:
    """
    Base mixin class providing common solving capabilities for solvers.

    Capabilities:
    - solve_for_w: compose solutions for a given w value.
    - parallel_solve_mech_system: parallelize the solve_mech_system calls.
    """
    @compose_solver_solves
    def solve_for_w(self, w_val, y_, k):
        self.set_initial_condition(y_[1])
        y_, _, info_ = super().solve(w_val*self.t_points[k:k+2])
        if self.store: self.info.append(np.array(info_[0::1]))

        if k > 0 and hasattr(self, 'Lambda') and self.solver_class == ConformalStormerVerletSolver:
            Lambda = self.Lambda[k].copy()
            if k == 1:
                print(f'Setting initial guess for Lagrange multipliers')
        else:
            Lambda = np.zeros_like(self.g_(np.zeros(2*self.nosc))).squeeze()

        if self.constraint_type:
            self.fixed_point(y_, Lambda)

        return (y_, y_ @ self.RB.T, Lambda) if self.RB is not None else (y_, y_, Lambda)

    @timing
    def solve_trajectory(self):

        if self.RB is None:
            self.y = np.zeros((self.n+1, 2*self.nosc))
        else:
            self.y = np.zeros((self.n+1, 2*self.nosc_r))
            self.y_full = np.zeros((self.n+1, 2*self.nosc))
            self.y_full[0] = MechSystem.y_init

        self.y[0] = self.y_init
        if self.store: self.info = []
        self.Lambda = np.zeros((self.n+1, self.g_(np.zeros(2*self.nosc)).shape[0]))

        # Calculate update frequency (10% of iterations)
        update_freq = max(1, self.n // 10)  

        # Create progress bar that updates less frequently
        with tqdm(total=self.n, desc='Solving trajectory', 
                miniters=update_freq,  # Minimum iterations between updates
                maxinterval=5.0,  # Maximum seconds between updates
                position=0, 
                leave=True) as pbar:

            for k in range(self.n):

                if self.store: self.info.append(self.y[k])
                y_ = np.array([self.y[k], self.y[k]])

                y_, y_full_, self.Lambda[k+1] = self.solve_for_w(y_, k)

                self.y[k+1] = y_[-1]

                if self.RB is not None:
                    self.y_full[k+1] = y_full_[-1]

                # Update progress bar every update_freq iterations
                if (k + 1) % update_freq == 0:
                    pbar.update(update_freq)
                    pbar.set_postfix({'step': k+1, 'total': self.n})

            # Update any remaining iterations
            remaining = self.n % update_freq
            if remaining:
                pbar.update(remaining)

        if self.store:
            self.info.append(self.y[k+1])
            self.info = np.vstack(self.info)

        if self.RB is not None:
            self.y_red = self.y.copy()
            self.y = self.y_full.copy()
            del self.y_full
            gc.collect()

    @staticmethod
    def solve_mech_system(solver_class, dt, Omega2, kwds):
        """
        Helper function to solve mechanical system.

        Parameters:
        solver_class (class): The solver class to be used for solving the system.
        dt (float): The time step size.
        Omega2 (numpy.ndarray): The Omega2 parameter for the system.
        kwds (dict): A dictionary of keyword arguments for the solver.

        Returns:
        solver: An instance of the solver class with the solved trajectory.
        """
        kwds['pool'] = {
            'solver_class': solver_class, 
            'dt': dt, 
            'Omega2': Omega2,
            'n': int(round(MechSystem.T_final/dt))
        }
        kwds['pool'].update({
            't_points': linspace(0, MechSystem.T_final, kwds['pool']['n']+1)
        })

        try:
            # Create concrete solver instance instead of MechSystemSolver
            solver = solver_class(kwds)

            tl = solver.solve_trajectory()

            solver.time_lapsed.append(tl)

            if not solver.beta:
                solver.eng_error = solver.get_en_err()
                solver.en_error = sqrt(dt) * LA.norm(solver.eng_error)
            else:
                solver.en_error = None

            if solver.var:
                if hasattr(solver, 'y_red'):
                    y = solver.y_red
                else:
                    y = solver.y

                solver.sym_error = solver.var_solve(y)
                # solver.sym_error = solver.symplectic_error(dpsi)

            return solver

        except Exception as e:
            print(f"Error in solve_mech_system: {str(e)}")
            print(f"Error type: {type(e)}")
            raise

    @staticmethod
    def parallel_solve_mech_system(kwds, MSsolvers, client=None):
        """Solve mechanical system in parallel using a Dask client, or sequentially."""
        # Prepare the arguments for solve_mech_system
        args = []
        for x in kwds['registered_solver_classes']:
            for y in MechSystem.dt_space:
                for z in kwds["Omega2_space"]:
                    # Create a deep copy of kwds for each process
                    process_kwds = kwds.copy()
                    args.append((x, y, z, process_kwds))

        if client and len(args) > 0:
            print(f"Submitting {len(args)} tasks to existing Dask cluster...", flush=True)
            # Pre-allocate MSsolvers with None values
            MSsolvers.extend([None] * len(args))

            # Submit all tasks to the cluster
            futures = [client.submit(worker, i, *arg) for i, arg in enumerate(args)]
            
            # Gather results (blocks until all are done)
            # client.gather returns results in the same order as futures list
            results = client.gather(futures, errors='raise')
            
            # Store results in correct order
            for i, res in enumerate(results):
                MSsolvers[i] = res
                if (i + 1) % 10 == 0:
                    print(f"Retrieved {i+1}/{len(args)} results", flush=True)
            
            print("Parallel processing complete.", flush=True)
        else:
            # Sequentially solve the mechanical system
            print('Solving mechanical system sequentially...')
            for arg in args:
                MSsolvers.append(BaseSolverMixin.solve_mech_system(*arg))

    @staticmethod
    def setup_and_solve_reduced_system(kwds, solvers, client=None):
        """Setup and solve reduced system for a specific solver type"""
        print(f'Setting up reduced system...')

        # Setup reduced model and update methods
        ReduceMechSystem.setup_reduced_model(solvers)

        # Solve reduced system
        solvers_r = []

        # Try to load from checkpoint
        checkpoint_path = os.path.join('data', f'{MechSystem.keep_time}')
        kwds_file = os.path.join(checkpoint_path, 'kwds_r.joblib')
        solvers_file = os.path.join(checkpoint_path, 'solvers_r.joblib')

        if os.path.exists(checkpoint_path) and os.path.exists(kwds_file) and os.path.exists(solvers_file):
            try:
                print("Loading from checkpoint...")
                kwds = load(kwds_file)
                solvers_r = load(solvers_file)
                print("Checkpoint loaded successfully")
            except Exception as e:
                raise Exception(f"Error loading checkpoint: {str(e)}")
        else:
            # Execute parallel solve
            BaseSolverMixin.parallel_solve_mech_system(kwds, solvers_r, client=client)

            # Create checkpoint directory and save initial state
            os.makedirs(checkpoint_path, exist_ok=True)
            print("Creating new checkpoint...")
            dump(kwds, kwds_file)
            dump(solvers_r, solvers_file)

        BaseSolverMixin.measures(kwds, solvers_r)

        return solvers_r

    @staticmethod
    def setup_and_solve_hyperreduced_system(kwds, solvers, client=None):
        """Setup and solve hyper-reduced system for a specific solver type"""
        print(f'Setting up hyper-reduced system...')

        # Setup hyperreduction using classmethod
        ReduceMechSystem.setup_hyperreduction()

        if ReduceMechSystem.hyperreducer == 'MDEIM':
            ReduceMechSystem.update_mdeim_hyperreduction(solvers)

        if ReduceMechSystem.constraints_reduce:
            if ConformalStormerVerletSolver in kwds['registered_solver_classes'] \
                or ConformalImplicitMidpointSolver in kwds['registered_solver_classes']:
                filtered_solvers, filtered_indices = ReduceMechSystem.filter_solvers(solvers, "Hamiltonian")
                ReduceMechSystem.hyperreduce_constraints(filtered_solvers, filtered_indices, "Hamiltonian")

            if DiscreteGradientSolver in kwds['registered_solver_classes']:
                filtered_solvers, filtered_indices = ReduceMechSystem.filter_solvers(solvers, "DiscreteGradient")
                ReduceMechSystem.hyperreduce_constraints(filtered_solvers, filtered_indices, "DiscreteGradient")

        # Solve hyper-reduced system
        solvers_dr = []

        # Try to load from checkpoint
        checkpoint_path = os.path.join('data', f'{MechSystem.keep_time}')
        kwds_file = os.path.join(checkpoint_path, 'kwds_dr.joblib')
        solvers_file = os.path.join(checkpoint_path, 'solvers_dr.joblib')

        if os.path.exists(checkpoint_path) and os.path.exists(kwds_file) and os.path.exists(solvers_file):
            try:
                print("Loading from checkpoint...")
                kwds = load(kwds_file)
                solvers_dr = load(solvers_file)
                print("Checkpoint loaded successfully")
            except Exception as e:
                raise Exception(f"Error loading checkpoint: {str(e)}")
        else:
            # Execute parallel solve
            BaseSolverMixin.parallel_solve_mech_system(kwds, solvers_dr, client=client)

            # Create checkpoint directory and save initial state
            os.makedirs(checkpoint_path, exist_ok=True)
            print("Creating new checkpoint...")
            dump(kwds, kwds_file)
            dump(solvers_dr, solvers_file)

        BaseSolverMixin.measures(kwds, solvers_dr)

        return solvers_dr

    @staticmethod
    def measures(kwds, solvers):
        """
        Compute various measurements based on the solution and plot the results.

        Parameters:
            kwds (dict): Dictionary of keyword arguments for the solver.
            solvers (list): List of solver instances used to solve the system.

        Returns:
        None
        """
        r_form = lambda numer, denom: r_[float('nan'), 
                (log(roll(numer, -1)/numer)/log(roll(denom, -1)/denom))[:-1]]

        # Compute en_error using a list comprehension and reshape it into a 2D array: rows = dt values, columns = Omega2 values for each solver class in kwds['registered_solver_classes']
        # For each solver_class, filter solvers and compute en_error separately
        for solver_class in kwds['registered_solver_classes']:
            filtered_solvers = [solver for solver in solvers if solver.solver_class == solver_class]
            en_error = [
                [solver.en_error for solver in filtered_solvers
                 if np.isclose(solver.dt, dt) and
                    np.isclose(solver.Omega2, omega2).all()]
                for dt in MechSystem.dt_space
                for omega2 in kwds['Omega2_space']
            ]
            # Reshape to (dt_space_dim, Omega2_space_dim)
            en_error = np.array(en_error).reshape(MechSystem.dt_space_dim, kwds['Omega2_space_dim'])

            if en_error.shape[0] > 1 and en_error[0, 0] is not None and MechSystem.dt_space_dim > 1 and not MechSystem.predict:
                print(f"\nConvergence rates for {solver_class.__name__}:")
                header = f"{'dt':>12}"
                for col in range(en_error.shape[1]):
                    header += f" | {f'Omega2_{col}':>12}"
                print(header)
                print("-" * len(header))
                all_r_values = np.array([r_form(en_error[:, col], MechSystem.dt_space) for col in range(en_error.shape[1])]).T
                for i, dt_val in enumerate(MechSystem.dt_space):
                    print(f"{dt_val:12.6f}" + "".join([f" | {val:12.6f}" for val in all_r_values[i]]))

            filtered_solvers[-1].plot()

        # # recompute energy error of conformal stormer verlet solvers by substracting the solver Hamiltonian from the Hamiltonian of DiscreteGradientSolver

        # # initialize energy error array
        # eng_error = np.zeros((MechSystem.dt_space_dim * kwds['Omega2_space_dim']))

        # # Create a dictionary mapping solver classes to their respective solver lists
        # solver_lists = {
        #     solver_class: [
        #         solver for solver in solvers if solver.solver_class == solver_class
        #     ]
        #     for solver_class in kwds['registered_solver_classes']
        # }
        # for i, solver in enumerate(solver_lists[ConformalStormerVerletSolver], start=0):
        #     # Compute energy error by subtracting the Hamiltonian of DiscreteGradientSolver
        #     # Find the corresponding DiscreteGradientSolver instance with matching dt and Omega2
        #     dg_solver = next(
        #         (s for s in solver_lists[DiscreteGradientSolver]
        #             if np.isclose(s.dt, solver.dt)
        #             and np.isclose(s.Omega2, solver.Omega2).all()),
        #         None
        #     )
        #     if dg_solver is not None:
        #         _eng_error = np.array([solver.ham(y) for y in solver.y]) - dg_solver.ham(dg_solver.y[-1])
        #         eng_error[i] = sqrt(solver.dt) * LA.norm(_eng_error)
        #     else:
        #         print("Warning: No matching DiscreteGradientSolver found for energy error computation.")

        # eng_error = eng_error.reshape(MechSystem.dt_space_dim, kwds['Omega2_space_dim'])

        # if eng_error.shape[0] > 1 and eng_error[0, 0] is not None and MechSystem.dt_space_dim > 1 and not MechSystem.predict:
        #     for col in range(eng_error.shape[1]):
        #         r_values = r_form(eng_error[:, col], MechSystem.dt_space)
        #         temp = c_[MechSystem.dt_space, r_values].T
        #         tex_table(f'ConformalStormerVerletSolver (Omega2_{col})', temp)

    def plot(self):
        """
        Generate and display plots for the simulation results.

        Plot the results of the simulation.
        This method generates a figure with multiple subplots to visualize various
        aspects of the simulation results, including symmetry error, norm of g, 
        energy error, linear momentum error, and angular momentum error. It also 
        plots the particle positions over time in a 3D phase portrait.
        Subplots:
        - ax0: Symmetry error over time (if `self.sym_error` is available).
        - ax1: Norm of g over time (if `self.g` is available).
        - ax2: Energy error over time (if `self.eng_error` is available).
        - ax3: Linear momentum error over time.
        - ax4: Angular momentum error over time.
        - ax5: (Commented out) Matrix condition number over time.
        - ax6: 3D phase portrait of particle positions over time.
        The figure's title includes the integrator name and time step size.
        The x-axis of the last subplot (ax4) is labeled as 'time'.
        The 3D phase portrait (ax6) includes labels for x, y, and z axes and a title.
        Note:
        - The method assumes that `self.y` contains the simulation data.
        - The method uses `plot_data` to plot data on the subplots.
        - The method uses `LA.norm` to compute norms.
        - The method uses `np.vstack` and `np.cross` for data manipulation.
        - The method includes commented-out code for plotting matrix condition number.
        """

        fig = figure(figsize=(10, 10))  # Make figure taller
        # fig.tight_layout(pad=0)
        fig.suptitle(rf'integrator = {self.solver_class.__name__}, $\Delta t = {self.dt}$', y=1)

        gs = fig.add_gridspec(5, 2, hspace=1)
        ax0, ax1, ax2, ax3, ax4 = [fig.add_subplot(gs[i, 0]) for i in [0, 1, 2, 3, 4]]
        ax_pp = fig.add_subplot(gs[:5, -1], projection='3d')
        ax_pp.set_box_aspect([1, 1, 1])  # Set aspect ratio to be equal for all axes

        if hasattr(self, 'sym_error'):
            plot_data(ax0, self.t_points, self.sym_error, xlims=(0, self.T_final), \
                    ylabel=r'$\Delta Sp$', margins=10)

            if max(abs(self.sym_error)) < 1e-15:
                ax0.set_ylim([-1e-15, 1e-15])

        g_norm = None
        if hasattr(self, 'g__lambda'):
            g_norm = [LA.norm(self.g__lambda(y)) for y in self.y]

            plot_data(
                ax1,
                self.t_points,
                g_norm,
                xlims=(0, self.T_final),
                ylabel=r"$\Delta \mathcal{S}$",
                margins=10,
            )

        if hasattr(self, 'eng_error'):
            plot_data(ax2, self.t_points, self.eng_error, xlims=(0, self.T_final), \
                    ylabel=r'$\Delta H$', margins=10)

        lin_momentum = np.sum(self.y[:, self.nosc:].reshape(-1, self.nosc//3, 3), axis=1)
        lim_momentum_err = r_[0, LA.norm(lin_momentum[1:] - lin_momentum[0], axis=1)]

        angular_momentum = np.cross(self.y[:, :self.nosc].reshape(-1, self.nosc//3, 3), \
                                    self.y[:, self.nosc:].reshape(-1, self.nosc//3, 3))
        angular_momentum_sum = np.sum(angular_momentum, axis=1)
        angular_momentum_err = r_[0, \
                                LA.norm(angular_momentum_sum[1:] - angular_momentum_sum[0], axis=1)]

        plot_data(ax3, self.t_points, lim_momentum_err, xlims=(0, self.T_final), \
                ylabel=r'$\Delta L$', margins=10)

        plot_data(ax4, self.t_points, angular_momentum_err, xlims=(0, self.T_final), \
                ylabel=r'$\Delta J$', margins=10)

        # # Plot matrix condition number
        # if self.RB is not None:
        #     temp = lambda t: self.g_prime[0](self.y[t])[:,:self.nosc] @ self.RB[:self.nosc,:self.nosc_r]
        #     mat = lambda t: temp(t) @ self.ham_zz(*np.split(self.y[t],2))[self.nosc_r:,self.nosc_r:] @ temp(t).T
        # else:
        #     temp = lambda t: self.g_prime[0](self.y[t])[:,:self.nosc]
        #     mat = lambda t: temp(t) @ self.ham_zz(*np.split(self.y[t],2))[self.nosc:,self.nosc:] @ temp(t).T

        # # self.mat_cond = [LA.cond(mat(t), 2) for t in range(self.n)]
        # self.mat_cond = LA.cond(np.array([mat(t) for t in range(self.n)]), 2)

        # plot_data(ax5, self.t_points[2:], self.mat_cond[1:], xlims=(0, self.T_final), \
        #           ylabel=r'$\kappa$', margins=0.5)

        ax4.set_xlabel('time')
        # Plot the particle positions over time
        coords = self.y[:, :self.nosc].reshape(-1, self.nosc//3, 3)
        for i in range(coords.shape[1]):
            ax_pp.scatter(coords[0, i, 0], coords[0, i, 1], coords[0, i, 2], s=20, c='teal')  # plot initial configuration
            ax_pp.plot(coords[:, i, 0], coords[:, i, 1], coords[:, i, 2], 'k-')  # plot system evolution

            # Plot projection onto x-y plane (z=0)
            ax_pp.plot(coords[:, i, 0], coords[:, i, 1], np.zeros_like(coords[:, i, 2]), 'r-', alpha=0.7)

        # Set the axes' labels and title
        ax_pp.set_xlabel('x', labelpad=10)
        ax_pp.set_ylabel('y', labelpad=10)
        ax_pp.set_zlabel('z', labelpad=10)
        ax_pp.set_title('Phase portrait')
        # ax_pp.set_xlim([coords[:, :, 0].min(), coords[:, :, 0].max()])
        # ax_pp.set_ylim([coords[:, :, 1].min(), coords[:, :, 1].max()])
        # ax_pp.set_zlim([coords[:, :, 2].min(), coords[:, :, 2].max()])
        ax_pp.set_box_aspect([1, 1, 1])  # Set aspect ratio to be equal for all axes
        ax_pp.view_init(elev=15, azim=45)

        # Hide inner labels
        for ax in [ax0, ax1, ax2, ax3, ax4]:
            ax.label_outer()

        fig.show()            

        fig_data = {
            't_points': self.t_points,
            'lim_momentum_err': lim_momentum_err,
            'angular_momentum_err': angular_momentum_err,
            'coords': coords
        }
        if hasattr(self, 'sym_error'):
            fig_data['sym_error'] = self.sym_error
        if g_norm is not None:
            fig_data['g_norm'] = g_norm
        if hasattr(self, 'eng_error'):
            fig_data['eng_error'] = self.eng_error

        solver_name_short = self.solver_class.__name__.replace('Conformal', 'C').replace('StormerVerlet', 'SV').replace('ImplicitMidpoint', 'IM').replace('DiscreteGradient', 'DG')
        suffix = '_full' if self.RB is None else '_r' if not hasattr(MechSystem, 'RBxUx_inv_PxU') else '_dr'
        suffix += '_predict' if self.predict else '' if self.RB is None else '_repro'
        filename = os.path.join(MechSystem.data_folder, f"osc_{solver_name_short}{suffix}.pdf")
        save_figure(fig, filename, fig_data=None)

# %% Concrete solver classes
class DiscreteGradientSolver(BaseSolverMixin, DiscreteGradient):
    """DiscreteGradient solver with DG-specific capabilities"""
    def __init__(self, kwds):
        super().__init__(kwds)
        self.g = self.g__
        self.g_prime = (self.g_prime__, self._g_prime_with_JJ)

    def _g_prime_with_JJ(self, y): #New class method
        return self.g_prime__(y) @ self.JJ.T

    def residual(self, x, x1, Lambda):

        g_x1 = self.g(x1)
        resi = r_[x1 - x - self.dt*self.f(c_[x, x1].T, None) - self.dt*self._g_prime_with_JJ(0.5 *(x + x1)).T @ Lambda,\
                g_x1]
        tang = r_[c_[np.eye(x1.shape[0]) - self.dt * self.dfdu(c_[x, x1].T, None) - 0.5 * self.dt * self.JJ @ self.g_prime_x_lambda_y(0.5 *(x + x1), Lambda), -self.dt * self.JJ @ self.g_prime_x_lambda_lambda(0.5 *(x + x1), Lambda)],\
                  c_[self.g_prime__(x1), np.zeros((g_x1.shape[0],)*2)]]

        return resi, tang

    def fixed_point(self, x, Lambda):
        """
        Fixed point iteration method.

        Parameters:
        x (array): The initial guess.
        Lambda (array): The initial value for Lambda.
        """

        m = 0
        residual = self.tol * 10

        while residual > self.tol and m < self.M:

            # Update residual and tangent
            resi, tang = self.residual(x[0], x[1], Lambda)

            Delta_z = -LA.solve(tang, resi)
            x[1] = x[1] + Delta_z[:x[1].shape[0]]
            Lambda += Delta_z[x[1].shape[0]:]

            # Update iteration counter and residual
            m += 1
            residual = LA.norm(r_[resi, Delta_z], np.inf)

        # Check convergence
        if m >= self.M:
            raise RuntimeError("Nonlinear solver did not converge")


class ConformalStormerVerletSolver(BaseSolverMixin, ConformalStormerVerlet):
    """CSVSolver with Hamiltonian capabilities"""
    def __init__(self, kwds):
        super().__init__(kwds)
        self.g = self.g__
        self.g_prime = (self.g_prime_diag, self.block_diag_matrix)

    def block_diag_matrix(self, y):
        """Compute g_prime using the original method"""

        ham_zz_y3 = self.ham_zz(y[3])
        i0, i1 = ham_zz_y3.shape
        i0, i1 = i0//2, i1//2
        ham_zz_y3_22 = ham_zz_y3[i0:, i1:]
        
        i0, i1 = self.g_prime__(y[0]).shape
        i0, i1 = i0//2, i1//2

        result = np.block([
            [-0.25 * self.dt**2 * (self.g_prime__(y[2])[i0:, i1:] + self.g_prime__(y[0])[:i0, :i1] @ ham_zz_y3_22), np.zeros((i0, i1))],
            [np.zeros((i0, i1)), -0.5 * self.dt * self.g_prime__(y[1])[i0:, i1:]]
        ])
        return result
    
    def g_prime_diag(self, y):
        """Compute g_prime using the original method"""
        _g_prime = self.g_prime__(y)
        i0, i1 = _g_prime.shape
        _g_prime_11 = _g_prime[:i0//2, :i1//2]
        result = np.block([
            [_g_prime_11, np.zeros(_g_prime_11.shape)],
            [np.zeros(_g_prime_11.shape), _g_prime_11]
        ])
        return result
        
    def fixed_point(self, x, Lambda):
        """
        Fixed point iteration method.

        Parameters:
        x (array): The initial guess.
        Lambda (array): The initial value for Lambda.
        """

        # Setup constants
        f, neq, dt = self.f, self.neq//2, self.dt
        q0, p0 = np.split(x[0], 2)
        
        # G(q0)
        g_prime_x0 = self.g_prime[0](x[0])
        i0, i1 = g_prime_x0.shape
        i0, i1 = i0//2, i1//2
        G_q0 = g_prime_x0[:i0, :i1] # G(q0)
        
        # Split Lambda
        Lambda_1, Lambda_2 = np.split(Lambda, 2)
        
        # p_half (unconstrained)
        p_half_unconstrained = p0 + 0.5 * dt * f(x[0], None)[neq:]
        
        # 1. Position Constraints Loop (Solve for Lambda_1)
        m = 0
        while m < self.M:
            # p_{n+1/2}
            p_half1 = p_half_unconstrained - 0.5 * dt * G_q0.T @ Lambda_1
            
            # q_{n+1}
            q_next = q0 + dt * f(np.concatenate([q0, p_half1]), None)[:neq]
            x[1, :neq] = q_next
            
            # Check position constraints
            g_pos = self.g(x[1])[:i0]
            if LA.norm(g_pos) < self.tol:
                break
                
            # Jacobian R11 = G(q_{n+1}) @ (-0.5 * dt^2 * G(q0)^T)
            G_qnext = self.g_prime[0](x[1])[:i0, :i1]
            R11 = -0.5 * dt**2 * G_qnext @ G_q0.T
            
            Delta_Lambda_1 = LA.lstsq(R11, g_pos, rcond=None)[0]
            Lambda_1 -= Delta_Lambda_1
            m += 1
            
        if m >= self.M:
             if self.RB is not None and LA.norm(g_pos) < 1e-5 and not m%100:
                 print(f"Warning: Position constraints not fully satisfied (error={LA.norm(g_pos):.2e}) in reduced model. Continuing.", flush=True)
             else:
                 raise RuntimeError(f"Nonlinear solver (position) did not converge. m: {m} -- Error: {LA.norm(g_pos)}")

        # 2. Velocity Constraints Loop (Solve for Lambda_2)
        m = 0
        while m < self.M:
            # p_{n+1}
            G_qnext = self.g_prime[0](x[1])[:i0, :i1]
            force_next = f(np.concatenate([x[1, :neq], p_half1]), None)[neq:]
            p_next = p_half1 + 0.5 * dt * force_next - 0.5 * dt * G_qnext.T @ Lambda_2
            x[1, neq:] = p_next
            
            # Check velocity constraints
            g_vel = self.g(x[1])[i0:]
            if LA.norm(g_vel) < self.tol:
                break
                
            # Jacobian R22 = -0.5 * dt * G(p_{n+1}) @ G(q_{n+1})^T
            G_pnext = self.g_prime__(x[1])[i0:, i1:]
            R22 = -0.5 * dt * G_pnext @ G_qnext.T
            
            Delta_Lambda_2 = LA.lstsq(R22, g_vel, rcond=None)[0]
            Lambda_2 -= Delta_Lambda_2
            m += 1
            
        if m >= self.M:
             if self.RB is not None and LA.norm(g_vel) < 1e-5 and not m%100:
                 print(f"Warning: Velocity constraints not fully satisfied (error={LA.norm(g_vel):.2e}) in reduced model. Continuing.", flush=True)
             else:
                 raise RuntimeError(f"Nonlinear solver (velocity) did not converge. m: {m} -- Error: {LA.norm(g_vel)}")
             
        # Update Lambda in place
        Lambda[:i0] = Lambda_1
        Lambda[i0:] = Lambda_2


class ConformalImplicitMidpointSolver(BaseSolverMixin, ConformalImplicitMidpoint):
    """CIMSolver with Hamiltonian capabilities"""
    def __init__(self, kwds):
        super().__init__(kwds)
        self.g = self.g__
        self.g_prime = [self.g_prime__, self.block_diag_matrix]

    def block_diag_matrix(self, y):
        """Compute g_prime using the original method"""
        _g_prime_shape = self.g_prime__(y[0]).shape
        result = np.block([
            [self.g_prime__(y[0])[:_g_prime_shape[0]//2, :_g_prime_shape[1]//2], np.zeros((_g_prime_shape[0]//2, _g_prime_shape[1]//2))],
            [np.zeros((_g_prime_shape[0]//2, _g_prime_shape[1]//2)), self.g_prime__(y[1])[:_g_prime_shape[0]//2, :_g_prime_shape[1]//2]]
        ])
        return result

# %% Main driver
if __name__ == '__main__':
    """
    Model order reduction of the MechSystem using concrete solvers
    """
    
    # %% Full order solution
    kwds = {
        "registered_solver_classes": [
            # ConformalImplicitMidpointSolver,
            DiscreteGradientSolver,
            ConformalStormerVerletSolver,
        ]
    }

    Omega2_space = (MechSystem._Omega2_space[:-1] if MechSystem.predict
                else MechSystem._Omega2_space)        
    Omega2_space_dim = len(Omega2_space)
    kwds.update({
        'Omega2_space': Omega2_space,
        'Omega2_space_dim': Omega2_space_dim
        })
    
    # Dask SLURM Cluster Configuration
    cluster = SLURMCluster(
        queue='workq',
        project='cpu_users',
        cores=32,
        processes=32,
        memory='64GB',
        walltime='01:00:00',
        job_extra_directives=['--exclusive', '--output=dask_worker_%j.log', '--qos=cpu_users']
    )
    
    # Determine number of nodes needed for the full simulation
    num_full_tasks = len(kwds['registered_solver_classes']) * MechSystem.dt_space_dim * kwds['Omega2_space_dim']
    num_nodes = min(6, (num_full_tasks + 31) // 32) if num_full_tasks > 0 else 0
    if num_nodes > 0:
        print(f"Requesting {num_nodes} SLURM nodes for Dask workers...")
        cluster.scale(jobs=num_nodes)
    
    print(f"Dask Dashboard: {cluster.dashboard_link}", flush=True)

    client = None
    try:
        # Use a client if we have nodes, otherwise run sequentially
        if num_nodes > 0:
            client = Client(cluster)
            print("Waiting for workers to start...", flush=True)
            client.wait_for_workers(1)

        # --- Full order solution ---
        solvers = []
        print('Computing full solution ...')
        checkpoint_path = os.path.join('data', f'{MechSystem.keep_time}')
        kwds_file = os.path.join(checkpoint_path, 'kwds.joblib')
        solvers_file = os.path.join(checkpoint_path, 'solvers.joblib')

        if os.path.exists(checkpoint_path) and os.path.exists(kwds_file) and os.path.exists(solvers_file):
            print("Loading full solution from checkpoint...")
            kwds = load(kwds_file)
            solvers = load(solvers_file)
        else:
            BaseSolverMixin.parallel_solve_mech_system(kwds, solvers, client=client)
            os.makedirs(checkpoint_path, exist_ok=True)
            print("Creating new checkpoint for full model...")
            dump(kwds, kwds_file)
            dump(solvers, solvers_file)
        BaseSolverMixin.measures(kwds, solvers)

        # --- Reduced order solution ---
        print('Computing reduced bases...')
        n_samples_list = [int(solver.n // 1) for solver in solvers]
        MechSystem.indices_list = [
            np.concatenate(([0], np.random.choice(np.arange(1, solver.n), n_samples - 1, replace=False)))
            for solver, n_samples in zip(solvers, n_samples_list)
        ]
        Omega2_space_ = (MechSystem._Omega2_space[-1:] if MechSystem.predict else MechSystem._Omega2_space)
        kwds.update({
            'Omega2_space': Omega2_space_,
            'Omega2_space_dim': len(Omega2_space_),
        })
        solvers_r = BaseSolverMixin.setup_and_solve_reduced_system(kwds, solvers, client=None)

        # --- Hyper-reduced model ---
        print('Computing hyper-reduction bases...')
        solvers_dr = BaseSolverMixin.setup_and_solve_hyperreduced_system(kwds, solvers, client=None)

    finally:
        if client:
            client.close()
        cluster.close()

    # %% Compute and display metrics
    array_shape = (len(kwds['registered_solver_classes']), 
                  MechSystem.dt_space_dim, 
                  Omega2_space_dim)

    time_lapsed = [reshape([x.time_lapsed for x in solvers], array_shape)]

    if not MechSystem.predict:
        # Calculate errors
        errors_r = reshape([np.amax(abs(y1 - y2)) 
            for y1, y2 in zip([x.y for x in solvers], 
            [x.y for x in solvers_r])], 
            array_shape)
        errors_dr = reshape([np.amax(abs(y1 - y2)) 
            for y1, y2 in zip([x.y for x in solvers], 
            [x.y for x in solvers_dr])], 
            array_shape)

        print('\n--- Max Solution Errors (over all frequencies) ---')
        error_matrices = [np.amax(errors_r, axis=2), np.amax(errors_dr, axis=2)]
        model_names_err = ["Reduced Model Error", "Hyper-reduced Model Error"]

        for i, error_matrix in enumerate(error_matrices):
            print(f"\n{model_names_err[i]}:")
            header = "Solver".ljust(35) + "".join([f"dt={dt:<11.4f}" for dt in MechSystem.dt_space])
            print(header)
            print("-" * len(header))
            for j, solver_class in enumerate(kwds['registered_solver_classes']):
                row_str = solver_class.__name__.ljust(35)
                for k in range(MechSystem.dt_space_dim):
                    val = error_matrix[j, k]
                    row_str += f"{val:<11.2e}"
                print(row_str)

        time_lapsed.append(reshape([x.time_lapsed for x in solvers_r], array_shape))
        # time_lapsed[0].shape) / time_lapsed[0] * 100)
        time_lapsed.append(reshape([x.time_lapsed for x in solvers_dr], array_shape))
        # time_lapsed[0].shape) / time_lapsed[0] * 100)
        time_lapsed[1] = time_lapsed[1] / time_lapsed[0] * 100
        time_lapsed[2] = time_lapsed[2] / time_lapsed[0] * 100

    else:
        time_lapsed.extend([
            reshape([x.time_lapsed for x in solvers_r], (*array_shape[:2], 1)),
            reshape([x.time_lapsed for x in solvers_dr], (*array_shape[:2], 1))
        ])
        
        # Calculate average full order time over all training frequencies
        avg_full_time = np.mean(time_lapsed[0], axis=2, keepdims=True)
        
        # Normalize reduced and hyper-reduced times
        time_lapsed[1] = time_lapsed[1] / avg_full_time * 100
        time_lapsed[2] = time_lapsed[2] / avg_full_time * 100

    print("\n--- Average Time Lapsed ---")
    if MechSystem.predict:
        model_names = ["Full Order Model (s)", "Reduced Model (% of Avg Full)", "Hyper-reduced Model (% of Avg Full)"]
    else:
        model_names = ["Full Order Model (s)", "Reduced Model (% of Full)", "Hyper-reduced Model (% of Full)"]

    # Average over the Omega2 dimension (axis=2)
    avg_times = [np.mean(tl, axis=2) for tl in time_lapsed]

    for i, avg_time_matrix in enumerate(avg_times):
        print(f"\n{model_names[i]}:")
        
        # Header
        header = "Solver".ljust(35) + "".join([f"dt={dt:<11.4f}" for dt in MechSystem.dt_space])
        print(header)
        print("-" * len(header))
        
        for j, solver_class in enumerate(kwds['registered_solver_classes']):
            row_str = solver_class.__name__.ljust(35)
            for k in range(MechSystem.dt_space_dim):
                val = avg_time_matrix[j, k]
                row_str += f"{val:<11.2f}"
            print(row_str)

    # tex_table('', time_lapsed)
    '''
    Observations:
        - hyperreduced model is efficient
            - and symplecitc with sparse MDEIM
            - and not demonstrably symplectic with DEIM
            - order of numerical methods is not verified
        - All solvers show 2nd order convergence for constrained full and reduced models
        - Parallel computation is unpredictable for reduced and hyperreduced models, specially for larger nosc

    # TODO:
        # Implement elastic beam deformation
        # Pretty print the terminal output
    '''
