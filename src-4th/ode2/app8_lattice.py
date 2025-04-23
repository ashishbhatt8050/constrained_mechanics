#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from pylab import  log, r_, c_, sqrt, reshape, linspace, roll, figure

import concurrent.futures

import os
import gc
from joblib import dump, load
from functools import partial, wraps
from tqdm.auto import tqdm  # Use tqdm.auto for better process handling
import psutil

from matplotlib import rc
rc('text', usetex=True)  # Enable LaTeX rendering
rc('text.latex', preamble=r'\usepackage{amsfonts}')  # Load AMSFonts for Fraktur

from ODESolver import (ConformalStormerVerlet, ConformalImplicitMidpoint, 
                      DiscreteGradient)
from Newton import fixed_point
from System import MechSystem  # Keep for static properties
from podDEIM import POD, PSD, DEIM
from PlotScript import plot_data, tex_table, logplot, timing

#%%
def compose_solver_solves(func):
    @wraps(func)
    def wrapper(self, y_, k):
        for w_val in self.w_values:
            y_, y_full_ = func(self, w_val, y_, k)
                
        return y_, y_full_
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

            # Create F2 in batches
            F2_list = []
            for solver, indices in zip(filtered_solvers, filtered_indices):
                batch_result = compute_batch_snapshots(solver, indices, expr)
                F2_list.append(batch_result)
            
            F2 = np.hstack(F2_list)
            del F2_list
            gc.collect()

            # Create snapshot list
            y_list = np.hstack([solver.y[indices].T for solver, indices 
                                in zip(filtered_solvers, filtered_indices)])
            
            RB, sv, nosc_r = PSD(F2, y_list, cls)
            setattr(MechSystem, attr_name[0], RB)
            setattr(MechSystem, attr_name[1], nosc_r)
            print(f'{RB.shape = }')

            fig, ax = logplot(sv, xlabel=f'index of singular values of [F2, y_list]', xlims=(1, len(sv)))
            # filename = MechSystem.keep_time +'osc_sv' + '.pdf'
            # save_figure(fig, filename)

        # Compute reduced basis for Hamiltonian solvers
        if ConformalStormerVerletSolver in kwds['registered_solver_classes'] \
            or ConformalImplicitMidpointSolver in kwds['registered_solver_classes']:
            compute_reduced_basis("Hamiltonian", solvers[0].ham_z, ["RB", "nosc_r"])

        # Compute reduced basis for Discrete Gradient solvers if applicable
        if DiscreteGradientSolver in kwds['registered_solver_classes']:
            compute_reduced_basis("DiscreteGradient", solvers[0].lag_dg, ["RB_dg", "nosc_r_dg"])

    @classmethod
    def setup_hyperreduction(cls, solvers):
        """Setup hyperreduction as classmethod"""
        print('Setting up hyperreduction...')
        
        def compute_hyperreduction_basis(solver_type, expr, attr, deim_attr_name, deim_func_name, expr_z=None, mdeim_func_name=None):
            P, _ = DEIM(attr, plot_deim=False)
            setattr(MechSystem, deim_attr_name, attr.T @ attr @ LA.inv(P.T @ attr))
            print(f'{P.shape = }')
            
            deim_func = smp.lambdify((cls.y, cls.omega2, cls.beta) if solver_type == "Hamiltonian" else (cls.y, cls.y1, cls.omega2), P.T @ expr.flat(), modules=['scipy'])

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

            F3 = np.hstack([np.array(result).T for result in results])

            # Compute POD basis and plot singular values
            Uj, sv, _ = POD(F3, np.eye(F3.shape[0]), cls.tol)
            fig, ax = logplot(sv, xlabel=f'index of singular values of {func_name}', xlims=(1, len(sv)))
            
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
            """Helper function to compute MDEIM basis for constraints"""
            print(f'Computing {func_name} reduction...')
            
            # Handle g_prime specific setup
            if is_g_prime:
                g_prime_shape = solvers[0].g_prime__(solvers[0].y[0]).shape
                setattr(MechSystem, 'g_prime_shape' if solver_type == "Hamiltonian" else 'g_prime_shape_dg', g_prime_shape)

                non_zero_indices = getattr(cls, func_name + "_nonzero_indices")
                IP = np.zeros((len(non_zero_indices), np.prod(g_prime_shape)), dtype=np.int8)
                IP[np.arange(len(non_zero_indices)), non_zero_indices] = 1
                
                # Create snapshot matrix for g_prime
                F = np.hstack([np.array([IP @ solver.g_prime__(y).flatten() 
                            for y in solver.y[indices]]).T 
                            for solver, indices in zip(solvers, indices_list)])
            else:
                # Create snapshot matrix for g
                F = np.hstack([np.array([solver.g(y) + 0 
                            for y in solver.y[indices]]).T 
                            for solver, indices in zip(solvers, indices_list)])
            
            # Compute POD basis and plot singular values
            Uj, sv, _ = POD(F, np.eye(F.shape[0]), cls.tol)
            del F
            
            # Plot singular values
            fig, ax = logplot(sv, xlabel=f'index of singular values of {func_name}', xlims=(1, len(sv)))
            
            # Compute DEIM points and interpolation matrix
            Pj, _ = DEIM(Uj, plot_deim=False)
            
            if is_g_prime:
                basis = IP.T @ Uj @ LA.inv(Pj.T @ Uj)
                # Create lambdified function
                col = Pj.T @ IP @ expr.reshape(np.prod(g_prime_shape), 1)
            else:
                basis = Uj @ LA.inv(Pj.T @ Uj)
                # Create lambdified function
                col = Pj.T @ expr
                
            print(f'{basis.shape = }')
            
            # Create and wrap lambdified function
            mdeim_func = smp.lambdify((cls.y,), col, modules=['scipy'])
            
            if callable(mdeim_func) and not isinstance(mdeim_func, type):
                wrapped = (lambda f: lambda self, *args, **kwargs: f(*args, **kwargs))(mdeim_func)
                # Set attributes based on solver type
                if solver_type == "Hamiltonian":
                    setattr(MechSystem, f'{func_name}_{"mdeim" if is_g_prime else "deim"}', wrapped)
                else:  # DiscreteGradient
                    setattr(MechSystem, f'{func_name}_{"mdeim" if is_g_prime else "deim"}_dg', wrapped)
            else:
                print(f"Memory address of {func_name}_{'mdeim' if is_g_prime else 'deim'}: {hex(id(mdeim_func))}")
                
            return basis
        
        # Compute bases for g and g_prime
        _Ux_inv_PxU = compute_constraint_basis('g', cls.g_expr)
        _IP_Ux_inv_PxU = compute_constraint_basis('g_prime', cls.g_prime_expr, is_g_prime=True)
        
        # Set attributes based on solver type
        if solver_type == "Hamiltonian":
            setattr(MechSystem, '_Ux_inv_PxU', _Ux_inv_PxU)
            setattr(MechSystem, '_IP_Ux_inv_PxU', _IP_Ux_inv_PxU)
        elif solver_type == "DiscreteGradient":
            setattr(MechSystem, '_Ux_inv_PxU_dg', _Ux_inv_PxU)
            setattr(MechSystem, '_IP_Ux_inv_PxU_dg', _IP_Ux_inv_PxU)
        else:
            raise ValueError(f"Invalid solver type: {solver_type}")
    
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

        if self.constraint_type:
            fixed_point(self.g, y_, self.g_prime, self.tol, self.M, False)
            # temp = self.g(y_)

        return (y_, y_ @ self.RB.T) if self.RB is not None else (y_, y_)

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
    
        # Calculate update frequency (10% of iterations)
        update_freq = max(1, self.n // 10)  
        
        # Create progress bar that updates less frequently
        with tqdm(total=self.n, desc='Solving trajectory', 
                mininterval=1.0,  # Minimum time between updates in seconds
                maxinterval=10.0,  # Maximum time between updates
                position=0,  # Prevent multiple bars overlapping
                leave=True) as pbar:

            for k in range(self.n):

                if self.store: self.info.append(self.y[k])
                y_ = np.array([self.y[k], self.y[k]])
                
                y_, y_full_ = self.solve_for_w(y_, k)
                
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
            self.y_red = self.y
            self.y = self.y_full

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
                    
                dpsi, _ = solver.var_solve(y)
                solver.sym_error = solver.symplectic_error(dpsi)

            return solver
        
        except Exception as e:
            print(f"Error in solve_mech_system: {str(e)}")
            print(f"Error type: {type(e)}")
            raise

    @staticmethod
    def parallel_solve_mech_system(kwds, MSsolvers):

        # Prepare the arguments for solve_mech_system
        args = []
        for x in kwds['registered_solver_classes']:
          for y in MechSystem.dt_space:
            for z in kwds['Omega2_space']:
              # Create a deep copy of kwds for each process
              process_kwds = kwds.copy()
              args.append((x, y, z, process_kwds))

        # Check the length of args
        if False and len(args) > 1 and not hasattr(MechSystem, 'RB') and not hasattr(MechSystem, 'RB_dg'):
            # Use ProcessPoolExecutor to parallelize the solve_mech_system calls
            print('Solving mechanical system using parallel processing...')
            
            # Pre-allocate MSsolvers with None values
            MSsolvers.extend([None] * len(args))
            
            with concurrent.futures.ProcessPoolExecutor() as executor:
                try:
                    futures = [
                        executor.submit(BaseSolverMixin.solve_mech_system, *arg)
                        for arg in args
                    ]
                    
                    # Place results directly in their final position
                    for future in concurrent.futures.as_completed(futures, timeout=None):
                        idx = futures.index(future)
                        try:
                            result = future.result()
                            print(f"Completed task {idx+1}/{len(args)}")
                            MSsolvers[idx] = result  # Store directly in correct position
                        except Exception as e:
                            print(f"Error in task {idx}: {str(e)}")
                            executor.shutdown(wait=False)
                            raise
                            
                except (concurrent.futures.TimeoutError, KeyboardInterrupt, Exception) as e:
                    print(f"\nReceived {type(e).__name__}, cancelling tasks...")
                    for f in futures:
                        f.cancel()
                    executor.shutdown(wait=False)
                    raise
        else:
            # Sequentially solve the mechanical system
            print('Solving mechanical system sequentially...')
            for arg in args:
                MSsolvers.append(BaseSolverMixin.solve_mech_system(*arg))

    @staticmethod
    def setup_and_solve_reduced_system(kwds, solvers):
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
            BaseSolverMixin.parallel_solve_mech_system(kwds, solvers_r)

            # Create checkpoint directory and save initial state
            os.makedirs(checkpoint_path, exist_ok=True)
            print("Creating new checkpoint...")
            dump(kwds, kwds_file)
            dump(solvers_r, solvers_file)

        BaseSolverMixin.measures(kwds, solvers_r)
        
        return solvers_r

    @staticmethod
    def setup_and_solve_hyperreduced_system(kwds, solvers):
        """Setup and solve hyper-reduced system for a specific solver type"""
        print(f'Setting up hyper-reduced system...')
        
        # Setup hyperreduction using classmethod
        ReduceMechSystem.setup_hyperreduction(solvers)

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

        if False and os.path.exists(checkpoint_path) and os.path.exists(kwds_file) and os.path.exists(solvers_file):
            try:
                print("Loading from checkpoint...")
                kwds = load(kwds_file)
                solvers_dr = load(solvers_file)
                print("Checkpoint loaded successfully")
            except Exception as e:
                raise Exception(f"Error loading checkpoint: {str(e)}")
        else:
            # Execute parallel solve
            BaseSolverMixin.parallel_solve_mech_system(kwds, solvers_dr)

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
        en_error = [solver.en_error for solver in solvers]

        for i in range(0, len(solvers), MechSystem.dt_space_dim * kwds['Omega2_space_dim']):
            if len(en_error) > 1 and en_error[0] is not None and MechSystem.dt_space_dim > 1 and not MechSystem.predict:
                r_values = r_form(en_error[i:i+MechSystem.dt_space_dim*kwds['Omega2_space_dim']:kwds['Omega2_space_dim']], MechSystem.dt_space)
                temp = c_[MechSystem.dt_space, r_values].T
                tex_table(solvers[i].solver_class.__name__, temp)
            solvers[i].plot()

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

        fig = figure()
        fig.tight_layout(pad=0)
        fig.suptitle(f'integrator = {self.solver_class.__name__}, dt = {self.dt}')

        gs = fig.add_gridspec(5, 2, hspace=1)
        ax0, ax1, ax2, ax3, ax4 = [fig.add_subplot(gs[i, 0]) for i in [0, 1, 2, 3, 4]]
        ax_pp = fig.add_subplot(gs[:5, -1], projection='3d')

        if hasattr(self, 'sym_error'):
            plot_data(ax0, self.t_points, self.sym_error, xlims=(0, self.T_final), \
                    ylabel=r'$\Delta Sp$', margins=1)
                
            if max(abs(self.sym_error)) < 1e-15:
                ax0.set_ylim([-1e-15, 1e-15])

        if hasattr(self, 'g__lambda'):
            if self.RB is None:
                temp = np.vstack([self.g__lambda(y) for y in self.y])
            else:
                temp = np.vstack([self.g__lambda(y) for y in self.y_full])

            g_norm = LA.norm(temp, axis=1)
            plot_data(ax1, self.t_points, g_norm, xlims=(0, self.T_final), \
                    ylabel=r'$\Delta \mathfrak{P}$', margins=1)

        if hasattr(self, 'eng_error'):
            plot_data(ax2, self.t_points, self.eng_error, xlims=(0, self.T_final), \
                    ylabel=r'$\Delta H$', margins=1)

        lin_momentum = np.sum(self.y[:, self.nosc:].reshape(-1, self.nosc//3, 3), axis=1)
        lim_momentum_err = r_[0, LA.norm(lin_momentum[1:] - lin_momentum[0], axis=1)]

        angular_momentum = np.cross(self.y[:, :self.nosc].reshape(-1, self.nosc//3, 3), \
                                    self.y[:, self.nosc:].reshape(-1, self.nosc//3, 3))
        angular_momentum_sum = np.sum(angular_momentum, axis=1)
        angular_momentum_err = r_[0, \
                                LA.norm(angular_momentum_sum[1:] - angular_momentum_sum[0], axis=1)]

        plot_data(ax3, self.t_points, lim_momentum_err, xlims=(0, self.T_final), \
                ylabel=r'$\Delta L$', margins=0.5)

        plot_data(ax4, self.t_points, angular_momentum_err, xlims=(0, self.T_final), \
                ylabel=r'$\Delta J$', margins=1)

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

        # Set the axes' labels and title
        ax_pp.set_xlabel('x')
        ax_pp.set_ylabel('y')
        ax_pp.set_zlabel('z')
        ax_pp.set_title('Phase portrait')
        ax_pp.view_init(elev=0, azim=45)  # Front-ish view

        for ax in [ax0, ax1, ax2, ax3, ax4]:
            ax.label_outer()

        # fig.show()
            
        # string = '_full' if self.RB is None else '_predict' if self.predict else '_repro'
        # filename = os.path.join(MechSystem.data_folder, f"osc_{self.solver_class.__name__}{string}.pdf")
        # save_figure(fig, filename)

#%% Concrete solver classes
class DiscreteGradientSolver(BaseSolverMixin, DiscreteGradient):
    """DiscreteGradient solver with DG-specific capabilities"""
    def __init__(self, kwds):
        super().__init__(kwds)
        self.g = self.g__
        self.g_prime = (self.g_prime__, self._g_prime_with_JJ)

    def _g_prime_with_JJ(self, y): #New class method
        return self.g_prime__(y) @ self.JJ.T
        
class ConformalStormerVerletSolver(BaseSolverMixin, ConformalStormerVerlet):
    """CSVSolver with Hamiltonian capabilities"""
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

#%% Main driver
if __name__ == '__main__':
    """
    Model order reduction of the MechSystem using concrete solvers
    """

    #%% Full order solution
    kwds = {
        'registered_solver_classes': [
            ConformalImplicitMidpointSolver,
            DiscreteGradientSolver,
            ConformalStormerVerletSolver,
        ]
    }

    solvers = []
    Omega2_space = (MechSystem._Omega2_space[:-1] if MechSystem.predict
                else MechSystem._Omega2_space)        
    Omega2_space_dim = len(Omega2_space)
    kwds.update({
        'Omega2_space': Omega2_space,
        'Omega2_space_dim': Omega2_space_dim
        })

    print('Computing full solution ...')
    # Try to load from checkpoint
    checkpoint_path = os.path.join('data', f'{MechSystem.keep_time}')
    kwds_file = os.path.join(checkpoint_path, 'kwds.joblib')
    solvers_file = os.path.join(checkpoint_path, 'solvers.joblib')

    if os.path.exists(checkpoint_path) and os.path.exists(kwds_file) and os.path.exists(solvers_file):
        try:
            print("Loading from checkpoint...")
            kwds = load(kwds_file)
            solvers = load(solvers_file)
            print("Checkpoint loaded successfully")
        except Exception as e:
                raise Exception(f"Error loading checkpoint: {str(e)}")
    else:
        # Execute parallel solve
        BaseSolverMixin.parallel_solve_mech_system(kwds, solvers)

        # Create checkpoint directory and save initial state
        os.makedirs(checkpoint_path, exist_ok=True)
        print("Creating new checkpoint...")
        dump(kwds, kwds_file)
        dump(solvers, solvers_file)

    BaseSolverMixin.measures(kwds, solvers)

    #%% Reduced order solution
    print('Computing reduced bases...')

    # Calculate samples and indices
    n_samples_list = [int(solver.n // 1) for solver in solvers]
    MechSystem.indices_list = [
        np.concatenate(([0], np.random.choice(np.arange(1, solver.n), n_samples - 1, replace=False)))
        for solver, n_samples in zip(solvers, n_samples_list)
        ]
    
    Omega2_space_ = (MechSystem._Omega2_space[-1:] if MechSystem.predict
                else MechSystem._Omega2_space)        
    Omega2_space_dim_ = len(Omega2_space_)
    kwds.update({
        'Omega2_space': Omega2_space_,
        'Omega2_space_dim': Omega2_space_dim_,
        })
    
    # Solve for each solver type
    solvers_r = BaseSolverMixin.setup_and_solve_reduced_system(kwds, solvers)

    # Merge solver lists
    # solvers_r = solvers_r_dg + solvers_r_ham
                
    #%% Hyper-reduced model
    print('Computing hyper-reduction bases...')

    # Setup and solve for each solver type
    solvers_dr = BaseSolverMixin.setup_and_solve_hyperreduced_system(kwds, solvers)
            
    #%% Compute and display metrics
    array_shape = (len(kwds['registered_solver_classes']), 
                  MechSystem.dt_space_dim, 
                  Omega2_space_dim)
    
    time_lapsed = [reshape([x.time_lapsed for x in solvers], array_shape)]
    
    if not MechSystem.predict:
        print('Solution errors:')
        print(reshape([np.amax(abs(y1 - y2)) 
            for y1, y2 in zip([x.y for x in solvers], 
            [x.y for x in solvers_r])], 
            array_shape))
        print(reshape([np.amax(abs(y1 - y2)) 
            for y1, y2 in zip([x.y for x in solvers], 
            [x.y for x in solvers_dr])], 
            array_shape))
    
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

    print(f'{time_lapsed = }')

    # tex_table('', time_lapsed)
    '''
    Observations:
        - hyperreduced model is efficient
            - and symplecitc with sparse MDEIM
            - and not demonstrably symplectic with DEIM
            - order of numerical methods is not verified
        - All solvers shows 4th order convergence for constrained full and reduced models
        - Convergence order of methods not observable under constraints
        - Parallel computation is unpredictable for reduced and hyperreduced models, specially for larger nosc
        - DiscreteGradientSolver in kwds somehow causes other solvers to overflow during runtime.

    # TODO:
        # Implement elastic beam deformation
        # Something's messing up the order of ConformalImplicitMidpointSolver (isolate the method and retry)
        # Consolidate reduction and hyperreduction methods in BaseSolverMixin
        # Remove unnecessary kwds from function calls
        # Pretty print the terminal output
    '''
