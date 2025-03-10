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

import os
import numpy as np
import sympy as smp
from numpy import linalg as LA
from pylab import  log, r_, c_, sqrt, reshape, linspace, roll, figure
import warnings

import concurrent.futures
import cloudpickle

from functools import wraps

from matplotlib import rc
rc('text', usetex=True)  # Enable LaTeX rendering
rc('text.latex', preamble=r'\usepackage{amsfonts}')  # Load AMSFonts for Fraktur

from ODESolver import (ConformalStormerVerlet, ConformalImplicitMidpoint, 
                      DiscreteGradient)
from Newton import fixed_point
from System import MechSystem  # Keep for static properties
from podDEIM import POD, PSD, DEIM
from PlotScript import plot_data, tex_table, logplot, save_figure, timing

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
        
        def compute_reduced_basis(solver_type, expr, attr_name):
            filtered_solvers, filtered_indices = cls.filter_solvers(solvers, solver_type)
            
            if solver_type == "Hamiltonian":
                F2 = np.hstack([np.array([expr(y) 
                    for y in solver.y[indices]]).T 
                    for solver, indices in zip(filtered_solvers, filtered_indices)])
            elif solver_type == "DiscreteGradient":
                F2 = np.hstack([np.array([expr(y) 
                    for y in zip(solver.y[indices], solver.y[np.array(indices)+1])]).T 
                    for solver, indices in zip(filtered_solvers, filtered_indices)]
                )

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
        compute_reduced_basis("Hamiltonian", solvers[0].ham_z, ["RB", "nosc_r"])

        # Compute reduced basis for Discrete Gradient solvers if applicable
        if DiscreteGradientSolver in kwds['registered_solver_classes']:
            compute_reduced_basis("DiscreteGradient", solvers[0].lag_dg, ["RB_dg", "nosc_r_dg"])

        '''
        filtered_solvers, filtered_indices = cls.filter_solvers(solvers, cls.indices_list, "Hamiltonian")
        F2 = np.hstack([np.array([solver.ham_z(y) 
            for y in solver.y[indices]]).T 
            for solver, indices in zip(filtered_solvers, filtered_indices)])

        # Create snapshot list
        y_list = np.hstack([solver.y[indices].T for solver, indices 
                            in zip(filtered_solvers, filtered_indices)])
        
        MechSystem.RB, sv, MechSystem.nosc_r = PSD(F2, y_list, cls)
        print(f'{MechSystem.RB.shape = }')

        fig, ax = logplot(sv, xlabel='index of singular values of ham_z', xlims=(1, len(sv)))
        # filename = MechSystem.keep_time +'osc_sv' + '.pdf'
        # save_figure(fig, filename)

        if DiscreteGradientSolver in kwds['registered_solver_classes']:
            filtered_solvers, filtered_indices = cls.filter_solvers(solvers, cls.indices_list, "DiscreteGradient")
            F2 = np.hstack([np.array([solver.lag_dg(y) 
                for y in zip(solver.y[indices], solver.y[np.array(indices)+1])]).T 
                for solver, indices in zip(filtered_solvers, filtered_indices)])

            # Create snapshot list
            y_list = np.hstack([solver.y[indices].T for solver, indices 
                                in zip(filtered_solvers, filtered_indices)])
            
            MechSystem.RB_dg, sv, MechSystem.nosc_r_dg = PSD(F2, y_list, cls)
            print(f'{MechSystem.RB_dg.shape = }')

            fig, ax = logplot(sv, xlabel='index of singular values of lag_dg', xlims=(1, len(sv)))
        '''

    @classmethod
    def setup_hyperreduction(cls, solvers):
        """Setup hyperreduction as classmethod"""
        print('Setting up hyperreduction...')
        
        def compute_hyperreduction_basis(expr, attr, deim_attr_name, deim_func_name, expr_z=None, mdeim_func_name=None):
            P, _ = DEIM(attr, plot_deim=False)
            setattr(MechSystem, deim_attr_name, attr.T @ attr @ LA.inv(P.T @ attr))
            print(f'{P.shape = }')

            deim_func = smp.lambdify((cls.y, cls.omega2, cls.beta), P.T @ expr.flat(), modules=['scipy'])

            if callable(deim_func) and not isinstance(deim_func, type):
                # Wrap lambda functions to include self parameter
                wrapped = (lambda f: lambda self, *args, **kwargs: f(*args, **kwargs))(deim_func)
                setattr(MechSystem, deim_func_name, wrapped)
            else:
                print(f"Memory address of {deim_func_name}: {hex(id(deim_func))}")

            # Create and wrap extra deim function if provided
            if expr_z is not None:
                mdeim_func = smp.lambdify((cls.y, cls.omega2, cls.beta),
                                        P.T @ expr_z,
                                        modules=['scipy'])

                if callable(mdeim_func) and not isinstance(mdeim_func, type):
                    wrapped = (lambda f: lambda self, *args, **kwargs: f(*args, **kwargs))(mdeim_func)
                    setattr(MechSystem, mdeim_func_name, wrapped)
                else:
                    print(f"Memory address of {mdeim_func_name}: {hex(id(mdeim_func))}")

        # Update the calls to compute_hyperreduction_basis
        compute_hyperreduction_basis(
            cls.ham_z_expr, 
            cls.RB, 
            "RBxUx_inv_PxU", 
            "ham_z_deim",
            cls.ham_zz_expr if cls.hyperreducer == 'DEIM' else None,
            "ham_zz_deim" if cls.hyperreducer == 'DEIM' else None
        )

        if DiscreteGradientSolver in kwds['registered_solver_classes']:
            compute_hyperreduction_basis(
                cls.lag_dg_expr,
                cls.RB_dg,
                "_RBxUx_inv_PxU_",
                "lag_dg_deim",
                cls.lag_dg_z_expr if cls.hyperreducer == 'DEIM' else None,
                "lag_dg_z_deim" if cls.hyperreducer == 'DEIM' else None
            )

        ''' 

        P, _ = DEIM(cls.RB, plot_deim=False)
        MechSystem.RBxUx_inv_PxU = cls.RB.T @ cls.RB @ LA.inv(P.T @ cls.RB)
        print(f'{P.shape = }')

        ham_z_deim = smp.lambdify((cls.y, cls.omega2, cls.beta), 
                    P.T @ cls.ham_z_expr.flat(), 
                    modules=['scipy'])

        if callable(ham_z_deim) and not isinstance(ham_z_deim, type):
            # Wrap lambda functions to include self parameter
            wrapped = (lambda f: lambda self, *args, **kwargs: f(*args, **kwargs))(ham_z_deim)
            setattr(MechSystem, 'ham_z_deim', wrapped)
        else:
            print(f"Memory address of ham_z_deim: {hex(id(ham_z_deim))}")

        if DiscreteGradientSolver in kwds['registered_solver_classes']:
            P, _ = DEIM(cls.RB_dg, plot_deim=False)
            MechSystem._RBxUx_inv_PxU_ = cls.RB_dg.T @ cls.RB_dg @ LA.inv(P.T @ cls.RB_dg)
            print(f'{P.shape = }')

            lag_dg_deim = smp.lambdify((cls.y, cls.y1, cls.omega2), 
                                    P.T @ cls.lag_dg_expr.flat(), modules=['scipy'])
            
            if callable(lag_dg_deim) and not isinstance(lag_dg_deim, type):
                # Wrap lambda functions to include self parameter
                wrapped = (lambda f: lambda self, *args, **kwargs: f(*args, **kwargs))(lag_dg_deim)
                setattr(MechSystem, 'lag_dg_deim', wrapped)
            else:
                print(f"Memory address of lag_dg_deim: {hex(id(lag_dg_deim))}")

        if cls.hyperreducer == 'DEIM':
            ham_zz_deim = smp.lambdify((cls.y, cls.omega2, cls.beta), 
                                    P.T @ cls.ham_zz_expr,
                                    modules=['scipy'])
            
            if callable(ham_zz_deim) and not isinstance(ham_zz_deim, type):
                # Wrap lambda functions to include self parameter
                wrapped = (lambda f: lambda self, *args, **kwargs: f(*args, **kwargs))(ham_zz_deim)
                setattr(MechSystem, 'ham_zz_deim', wrapped)
            else:
                print(f"Memory address of ham_zz_deim: {hex(id(ham_zz_deim))}")
                
            if DiscreteGradientSolver in kwds['registered_solver_classes']:
                lag_dg_z_deim = smp.lambdify((cls.y, cls.y1, cls.omega2),
                                        P.T @ cls.lag_dg_z_expr,
                                        modules=['scipy'])

                if callable(lag_dg_z_deim) and not isinstance(lag_dg_z_deim, type):
                    # Wrap lambda functions to include self parameter
                    wrapped = (lambda f: lambda self, *args, **kwargs: f(*args, **kwargs))(lag_dg_z_deim)
                    setattr(MechSystem, 'lag_dg_z_deim', wrapped)
                else:
                    print(f"Memory address of lag_dg_z_deim: {hex(id(lag_dg_z_deim))}")
        '''
                    
        if cls.hyperreducer == 'MDEIM':
            cls.update_mdeim_hyperreduction(solvers)

        if cls.constraints_reduce:
            filtered_solvers, filtered_indices = cls.filter_solvers(solvers, "Hamiltonian")
            MechSystem._Ux_inv_PxU, MechSystem._IP_Ux_inv_PxU = cls.hyperreduce_constraints(filtered_solvers, filtered_indices)

            filtered_solvers, filtered_indices = cls.filter_solvers(solvers, "DiscreteGradient")
            MechSystem._Ux_inv_PxU_dg, MechSystem._IP_Ux_inv_PxU_dg = cls.hyperreduce_constraints(filtered_solvers, filtered_indices)
    
    @classmethod
    def update_mdeim_hyperreduction(cls, solvers):
        """Update methods with MDEIM hyperreduction"""
        
        def compute_mdeim_basis(solver_type, func_name, expr, attr_name, deim_func_name):
            """Helper function to compute MDEIM basis and create lambdified functions"""
            # NOTE: symbolic matrix multiplication in high-precision arithmetic can become expensive
            # and may cause memory overflow. This can lead to a program crash.
            # Though _IP_Ux_inv_PxU_ @ lag_dg_z_col can be done here once and for all solvers.
            print(f'Computing {func_name} reduction...')
            
            # Collect snapshots
            non_zero_indices = np.nonzero(getattr(solvers[0], func_name)(solvers[0].y[0:2] if solver_type == "DiscreteGradient" else solvers[0].y[0]).flatten())[0]
            IP = np.zeros((len(non_zero_indices), (2*cls.nosc)**2))
            IP[np.arange(len(non_zero_indices)), non_zero_indices] = 1
            
            # Filter solvers and collect snapshots
            filtered_solvers, filtered_indices = cls.filter_solvers(solvers, solver_type)
            
            # Create snapshot matrix
            if solver_type == "DiscreteGradient":
                F3 = np.hstack([np.array([IP @ solver.lag_dg_z(y).flatten()
                            for y in zip(solver.y[indices], solver.y[np.array(indices)+1])]).T 
                            for solver, indices in zip(filtered_solvers, filtered_indices)])
            else:
                F3 = np.hstack([np.array([IP @ solver.ham_zz(y).flatten() 
                        for y in solver.y[indices]]).T 
                        for solver, indices in zip(filtered_solvers, filtered_indices)])
            
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

            print(f'{deim_func_name} has been updated for class: {cls.__name__}')

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

        # Compute MDEIM basis for DiscreteGradient solvers
        if DiscreteGradientSolver in kwds['registered_solver_classes']:
            compute_mdeim_basis(
                "DiscreteGradient",
                "lag_dg_z",
                cls.lag_dg_z_expr,
                "_IP_Ux_inv_PxU_",
                "lag_dg_z_mdeim"
            )

        '''
        if ConformalStormerVerletSolver in kwds['registered_solver_classes'] \
            or ConformalImplicitMidpointSolver in kwds['registered_solver_classes']:
            print('Computing ham_zz reduction...')
            
            # Collect ham_zz snapshots
            non_zero_indices = np.nonzero(solvers[0].ham_zz(solvers[0].y[0]).flatten())[0]
            IP = np.zeros((len(non_zero_indices), (2*cls.nosc)**2))
            IP[np.arange(len(non_zero_indices)), non_zero_indices] = 1
            
            filtered_solvers, filtered_indices = cls.filter_solvers(solvers, cls.indices_list, "Hamiltonian")
            F3 = np.hstack([np.array([IP @ solver.ham_zz(y).flatten() 
                    for y in solver.y[indices]]).T 
                    for solver, indices in zip(filtered_solvers, filtered_indices)])
            
            # Compute POD basis and plot singular values
            Uj, sv, _ = POD(F3, np.eye(F3.shape[0]), cls.tol)
            fig, ax = logplot(sv, xlabel='index of singular values of ham_zz', xlims=(1, len(sv)))
            # filename = os.path.join(cls.data_folder, f'ham_zz_sv_{cls.__name__}.pdf')
            # save_figure(fig, filename)
            
            Pj, _ = DEIM(Uj, plot_deim=False)
            MechSystem.IP_Ux_inv_PxU = IP.T @ Uj @ LA.inv(Pj.T @ Uj)
            print(f'{MechSystem.IP_Ux_inv_PxU.shape = }')

            # Create lambdified function
            ham_zz_col = Pj.T @ IP @ cls.ham_zz_expr.flat() #reshape((2*cls.nosc)**2, 1)
            ham_zz_mdeim = smp.lambdify((cls.y, cls.omega2, cls.beta), 
                        ham_zz_col, modules=['scipy'])
            
            if callable(ham_zz_mdeim) and not isinstance(ham_zz_mdeim, type):
                # Wrap lambda functions to include self parameter
                wrapped = (lambda f: lambda self, *args, **kwargs: f(*args, **kwargs))(ham_zz_mdeim)
                setattr(MechSystem, 'ham_zz_mdeim', wrapped)
            else:
                print(f"Memory address of ham_zz_mdeim: {hex(id(ham_zz_mdeim))}")

            print(f'ham_zz_mdeim has been updated for class: {cls.__name__}')

        if DiscreteGradientSolver in kwds['registered_solver_classes']:
            # NOTE: symbolic matrix multiplication in high-precision arithmetic can become expensive
            # and may require more memory than available. This can lead to a program crash.
            # Though _IP_Ux_inv_PxU_ @ lag_dg_z_col can be done here once and for all solvers.
            print('Computing lag_dg_z reduction...')
            
            # Collect lag_dg_z snapshots
            non_zero_indices = np.nonzero(solvers[0].lag_dg_z(solvers[0].y[0:2]).flatten())[0]
            IP = np.zeros((len(non_zero_indices), (2*cls.nosc)**2))
            IP[np.arange(len(non_zero_indices)), non_zero_indices] = 1
            
            filtered_solvers, filtered_indices = cls.filter_solvers(solvers, cls.indices_list, "DiscreteGradient")
            F3 = np.hstack([np.array([IP @ solver.lag_dg_z(y).flatten()
                        for y in zip(solver.y[indices], solver.y[np.array(indices)+1])]).T 
                        for solver, indices in zip(filtered_solvers, filtered_indices)])
            
            # Compute POD basis and plot singular values
            Uj, sv, _ = POD(F3, np.eye(F3.shape[0]), cls.tol)
            fig, ax = logplot(sv, xlabel='index of singular values of lag_dg_z', xlims=(1, len(sv)))
            # filename = os.path.join(cls.data_folder, f'lag_dg_z_sv_{cls.__name__}.pdf')
            # save_figure(fig, filename)
            
            Pj, _ = DEIM(Uj, plot_deim=False)
            MechSystem._IP_Ux_inv_PxU_ = IP.T @ Uj @ LA.inv(Pj.T @ Uj)
            print(f'{MechSystem._IP_Ux_inv_PxU_.shape = }')

            # Create lambdified function
            lag_dg_z_col = Pj.T @ IP @ cls.lag_dg_z_expr.flat() #reshape((2*cls.nosc)**2, 1) # TODO: reshape in SymbolicComputer
            lag_dg_z_mdeim = smp.lambdify((cls.y, cls.y1, cls.omega2), 
                                        lag_dg_z_col, modules=['scipy'])

            if callable(lag_dg_z_mdeim) and not isinstance(lag_dg_z_mdeim, type):
                # Wrap lambda functions to include self parameter
                wrapped = (lambda f: lambda self, *args, **kwargs: f(*args, **kwargs))(lag_dg_z_mdeim)
                setattr(MechSystem, 'lag_dg_z_mdeim', wrapped)
            else:
                print(f"Memory address of lag_dg_z_mdeim: {hex(id(lag_dg_z_mdeim))}")

            print(f'lag_dg_z_mdeim has been updated for class: {cls.__name__}')
        '''
    
    @classmethod
    def hyperreduce_constraints(cls, solvers, indices_list):
        """Hyper-reduce constraints using MDEIM."""
        print('Computing constraints reduction...')
        
        def compute_constraint_basis(func_name, expr, is_g_prime=False):
            """Helper function to compute MDEIM basis for constraints"""
            print(f'Computing {func_name} reduction...')
            
            # Handle g_prime specific setup
            if is_g_prime:
                MechSystem.g_prime_shape = solvers[0].g_prime__(solvers[0].y[0]).shape
                non_zero_indices = np.nonzero(solvers[0].g_prime__(solvers[0].y[0]).flatten())[0]
                IP = np.zeros((len(non_zero_indices), np.prod(cls.g_prime_shape)))
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
                col = Pj.T @ IP @ expr.reshape(np.prod(cls.g_prime_shape), 1)
            else:
                basis = Uj @ LA.inv(Pj.T @ Uj)
                # Create lambdified function
                col = Pj.T @ expr
                
            print(f'{basis.shape = }')
            
            # Create and wrap lambdified function
            mdeim_func = smp.lambdify((cls.y,), col, modules=['scipy'])
            
            if callable(mdeim_func) and not isinstance(mdeim_func, type):
                wrapped = (lambda f: lambda self, *args, **kwargs: f(*args, **kwargs))(mdeim_func)
                setattr(MechSystem, f'{func_name}_{"mdeim" if is_g_prime else "deim"}', wrapped)
            else:
                print(f"Memory address of {func_name}_{'mdeim' if is_g_prime else 'deim'}: {hex(id(mdeim_func))}")
                
            return basis
        
        # Compute bases for g and g_prime
        _Ux_inv_PxU = compute_constraint_basis('g', cls.g_expr)
        _IP_Ux_inv_PxU = compute_constraint_basis('g_prime', cls.g_prime_expr, is_g_prime=True)
        
        print('Constraints reduction complete.')
        return _Ux_inv_PxU, _IP_Ux_inv_PxU

    @classmethod
    def _hyperreduce_constraints(cls, solvers, indices_list):
        """
        Hyper-reduce constraints using MDEIM.

        This method computes the hyper-reduction of constraints using the Matrix Discrete Empirical Interpolation Method (MDEIM).
        It updates cls with reduced versions of the constraint functions and their derivatives.

        Parameters:
        cls (class): The class that calls this method.
        solvers (list): A list of solver instances used to compute the reduction.

        Returns:
        None: The method updates cls.
        """
        print('Computing constraints reduction...')
        
        # Compute g reduction
        print('Computing g reduction...')
        F5 = np.hstack([np.array([solver.g(y) + 0 
                        for y in solver.y[indices]]).T 
                        for solver, indices in zip(solvers, indices_list)])
        
        # Compute POD basis and plot singular values for g
        Uj, sv, mj = POD(F5, np.eye(F5.shape[0]), cls.tol)
        del F5
        
        # Plot singular values
        fig, ax = logplot(sv, xlabel='index of singular values of g', xlims=(1, len(sv)))
        # filename = os.path.join(cls.data_folder, 'g_sv.pdf')
        # save_figure(fig, filename)
        
        # Compute DEIM points and interpolation matrix for g
        Pj, _ = DEIM(Uj, plot_deim=False)
        _Ux_inv_PxU = Uj @ LA.inv(Pj.T @ Uj)
        print(f'{_Ux_inv_PxU.shape = }')
        
        # Create lambdified function for g
        g_deim = smp.lambdify((cls.y,), Pj.T @ cls.g_expr, modules=['scipy'])
        
        # Compute g_prime reduction
        print('Computing g_prime reduction...')
        MechSystem.g_prime_shape = solvers[0].g_prime__(solvers[0].y[0]).shape
        non_zero_indices = np.nonzero(solvers[0].g_prime__(solvers[0].y[0]).flatten())[0]
        IP = np.zeros((len(non_zero_indices), np.prod(cls.g_prime_shape)))
        IP[np.arange(len(non_zero_indices)), non_zero_indices] = 1
        
        F4 = np.hstack([np.array([IP @ solver.g_prime__(y).flatten() 
                        for y in solver.y[indices]]).T 
                        for solver, indices in zip(solvers, indices_list)])
        
        # Compute POD basis and plot singular values for g_prime
        Uj, sv, mj = POD(F4, np.eye(F4.shape[0]), cls.tol)
        del F4
        
        # Plot singular values
        fig, ax = logplot(sv, xlabel='index of singular values of g_prime', xlims=(1, len(sv)))
        # filename = os.path.join(cls.data_folder, 'g_prime_sv.pdf')
        # save_figure(fig, filename)
        
        # Compute DEIM points and interpolation matrix for g_prime
        Pj, _ = DEIM(Uj, plot_deim=False)
        _IP_Ux_inv_PxU = IP.T @ Uj @ LA.inv(Pj.T @ Uj)
        print(f'{_IP_Ux_inv_PxU.shape = }')
        
        # Create lambdified function for g_prime
        # TODO: find a more efficient way to reshape g_prime
        g_prime_col = Pj.T @ IP @ cls.g_prime_expr.reshape(np.prod(cls.g_prime_shape), 1)
        g_prime_mdeim = smp.lambdify((cls.y,), g_prime_col, modules=['scipy'])

        if callable(g_deim) and not isinstance(g_deim, type):
            # Wrap lambda functions to include self parameter
            wrapped = (lambda f: lambda self, *args, **kwargs: f(*args, **kwargs))(g_deim)
            setattr(MechSystem, 'g_deim', wrapped)
        else:
            print(f"Memory address of g_deim: {hex(id(g_deim))}")

        if callable(g_prime_mdeim) and not isinstance(g_prime_mdeim, type):
            # Wrap lambda functions to include self parameter
            wrapped = (lambda f: lambda self, *args, **kwargs: f(*args, **kwargs))(g_prime_mdeim)
            setattr(MechSystem, 'g_prime_mdeim', wrapped)
        else:
            print(f"Memory address of g_prime_mdeim: {hex(id(g_prime_mdeim))}")

        print('Constraints reduction complete.')

        return _Ux_inv_PxU, _IP_Ux_inv_PxU


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

        for k in range(self.n):
            if self.store: self.info.append(self.y[k])
            y_ = np.array([self.y[k], self.y[k]])
            
            y_, y_full_ = self.solve_for_w(y_, k)
            
            self.y[k+1] = y_[-1]
            
            if self.RB is not None:
                self.y_full[k+1] = y_full_[-1]
        
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
        # args = [(x, y, z, kwds.copy()) for x in kwds['registered_solver_classes']
        #         for y in MechSystem.dt_space for z in Omega2_space]

        # Check the length of args
        if len(args) > 1:
            # Use ProcessPoolExecutor to parallelize the solve_mech_system calls
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures_and_args = []

                for arg in args:
                  futures_and_args.append((executor.submit(BaseSolverMixin.solve_mech_system, *arg), arg))
                
                for future, arg in futures_and_args:
                    MSsolvers.append(future.result())
        else:
            # Sequentially solve the mechanical system
            for arg in args:
                MSsolvers.append(BaseSolverMixin.solve_mech_system(*arg))

    @staticmethod
    def setup_and_solve_reduced_system(kwds, solvers):
        """Setup and solve reduced system for a specific solver type"""
        print(f'Setting up reduced system...')
        
        # Filter solvers by type
        # if solver_type == "DG":
        #     filtered_solvers = [s for s in solvers if isinstance(s, DiscreteGradient)]
        #     solver_class = DiscreteGradientSolver
        #     kwds = kwds_base.copy()
        #     kwds['registered_solver_classes'] = [DiscreteGradientSolver]
        # else:  # Hamiltonian
        #     filtered_solvers = [s for s in solvers if not isinstance(s, DiscreteGradient)]
        #     solver_class = ConformalStormerVerletSolver
        #     kwds = kwds_base.copy()
        #     kwds['registered_solver_classes'] = [ConformalStormerVerletSolver, ConformalImplicitMidpointSolver]

        # Setup reduced model and update methods
        ReduceMechSystem.setup_reduced_model(solvers)

        # Solve reduced system
        solvers_r = []

        BaseSolverMixin.parallel_solve_mech_system(kwds, solvers_r)
        BaseSolverMixin.measures(kwds, solvers_r)
        
        return solvers_r

    @staticmethod
    def setup_and_solve_hyperreduced_system(kwds, solvers):
        """Setup and solve hyper-reduced system for a specific solver type"""
        print(f'Setting up hyper-reduced system...')
        
        # Create solver-specific kwds dictionary
        # kwds = kwds_base.copy()
        # if solver_type == "DG":
        #     kwds['registered_solver_classes'] = [DiscreteGradientSolver]
        #     filtered_solvers = [s for s in solvers if isinstance(s, DiscreteGradientSolver)]
        #     solver_class = DiscreteGradientSolver
        # else:  # Hamiltonian
        #     kwds['registered_solver_classes'] = [ConformalStormerVerletSolver, ConformalImplicitMidpointSolver]
        #     filtered_solvers = [s for s in solvers if not isinstance(s, DiscreteGradientSolver)]
        #     solver_class = ConformalStormerVerletSolver

        # Calculate samples and indices, replace filtered_solvers with solvers
        # n_samples_list = [int(solver.n // 2) for solver in solvers]
        # MechSystem.indices_list = [np.random.choice(solver.n-1, n_samples, replace=False)
        #             for solver, n_samples in zip(solvers, n_samples_list)]
        
        # Setup hyperreduction using classmethod
        ReduceMechSystem.setup_hyperreduction(solvers)

        # Solve hyper-reduced system
        solvers_dr = []

        BaseSolverMixin.parallel_solve_mech_system(kwds, solvers_dr)
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
                r_values = r_form(en_error[i:i+MechSystem.dt_space_dim*kwds['Omega2_space_dim']:kwds['Omega2_space_dim']], 
                                MechSystem.dt_space)
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
            ax_pp.plot(coords[:, i, 0], coords[:, i, 1], coords[:, i, 2], 'k-')  # plot the trajectory of each particle
            ax_pp.scatter(coords[-1, i, 0], coords[-1, i, 1], coords[-1, i, 2], s=20)  # plot the final position of each particle

        # Set the axes' labels and title
        ax_pp.set_xlabel('x')
        ax_pp.set_ylabel('y')
        ax_pp.set_zlabel('z')
        ax_pp.set_title('Phase portrait')

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
        self.g_prime = self.g_prime__

class ConformalImplicitMidpointSolver(BaseSolverMixin, ConformalImplicitMidpoint):
    """CIMSolver with Hamiltonian capabilities"""
    def __init__(self, kwds):
        super().__init__(kwds)
        self.g = self.g__
        self.g_prime = self.g_prime__

#%% Main driver
if __name__ == '__main__':
    """
    Model order reduction of the MechSystem using concrete solvers
    """

    #%% Full order solution
    kwds = {
        'registered_solver_classes': [
            DiscreteGradientSolver,
            ConformalStormerVerletSolver,
            ConformalImplicitMidpointSolver
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
    BaseSolverMixin.parallel_solve_mech_system(kwds, solvers)
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
        'Omega2_space_dim': Omega2_space_dim_
        })
    
    # Solve for each solver type
    # solvers_r_dg = BaseSolverMixin.setup_and_solve_reduced_system("DG", solvers, kwds)
    solvers_r = BaseSolverMixin.setup_and_solve_reduced_system(kwds, solvers)

    # Merge solver lists
    # solvers_r = solvers_r_dg + solvers_r_ham
                
    #%% Hyper-reduced model
    print('Computing hyper-reduction bases...')

    # Setup and solve for each solver type
    solvers_dr = BaseSolverMixin.setup_and_solve_hyperreduced_system(kwds, solvers)
    # solvers_dr_ham = BaseSolverMixin.setup_and_solve_hyperreduced_system("Hamiltonian", solvers, kwds)
    
    # Merge solver lists 
    # solvers_dr = solvers_dr_dg + solvers_dr_ham
            
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
    
        time_lapsed.append(reshape([x.time_lapsed for x in solvers_r], 
                                        time_lapsed[0].shape) / time_lapsed[0] * 100)
        time_lapsed.append(reshape([x.time_lapsed for x in solvers_dr], 
                                        time_lapsed[0].shape) / time_lapsed[0] * 100)
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
        - DiscreteGradientSolver shows zero error in the reduced model
        - Convergence order of methods not observable under constraints

    # TODO:
        # Implement elastic beam deformation
        # Something's messing up the order of ConformalImplicitMidpointSolver (isolate the method and retry)
        # Investigate why the error in reduction and hyperreduction is always zero for both DiscreteGradientSolver and ConformalImplicitMidpointSolver (isolate and test)
        # Consolidate reduction and hyperreduction methods in BaseSolverMixin
        # Remove unnecessary kwds from function calls
        # Pretty print the terminal output
    '''
