#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 2024

@author: ashishbhatt

This app solves a full order and reduced-order model from the System class,
using solver classes in the ODESolver hierarchy of methods,
fixed point iterators from the Newton.py,
and bases from podDEIM.
"""

import os
import numpy as np
import sympy as smp
from numpy import linalg as LA
from pylab import  log, r_, c_, sqrt, reshape, linspace, roll, figure

import concurrent.futures

from functools import wraps

from matplotlib.ticker import MaxNLocator
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

class BaseSolverMixin:
    """Base mixin with common solving capabilities"""
    @compose_solver_solves
    def solve_for_w(self, w_val, y_, k):
        self.set_initial_condition(y_[1])
        y_, _, info_ = super().solve(w_val*self.t_points[k:k+2])
        if self.store: self.info.append(np.array(info_[0::1]))
        return self.handle_constraints(y_)

    def handle_constraints(self, y_):
        if self.constraint_type:
                fixed_point(self.g, y_, self.g_prime, self.tol, self.M, False)
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

        '''
        for k in range(self.n):
            if self.store: self.info.append(self.y[k])
            y_ = np.array([self.y[k], self.y[k]])

            for w_val in self.w_values:
                self.solver.set_initial_condition(y_[1])
                y_, _, info_ = self.solver.solve(w_val*self.t_points[k:k+2])
                if self.store: self.info.append(np.array(info_[0::1]))

                # enforce constraints
                if self.constraint_type:

                    if self.RB is not None:
                        y_ = y_ @ self.RB.T

                    fixed_point(self.g, y_, self.g_prime, self.tol, self.M, False)

            if self.RB is None or self.constraint_type is None:
                self.y[k+1] = y_[1]
            else:
                self.y_full[k+1] = y_[1]
                self.y[k+1] = self.RB.T @y_[1]
        '''
        
        if self.store:
            self.info.append(self.y[k+1])
            self.info = np.vstack(self.info)

        if self.RB is not None:
            self.y_red = self.y
            self.y = self.y_full

    def plot(self):
        """
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

        gs = fig.add_gridspec(6, 2, hspace=1)
        ax0, ax1, ax2, ax3, ax4, ax5 = [fig.add_subplot(gs[i, 0]) for i in [0, 1, 2, 3, 4, 5]]
        ax6 = fig.add_subplot(gs[:6, -1], projection='3d')

        if hasattr(self, 'sym_error'):
            plot_data(ax0, self.t_points, self.sym_error, xlims=(0, self.T_final), \
                      ylabel=r'$\Delta Sp$', margins=1)
                
            if max(abs(self.sym_error)) < 1e-15:
                ax0.set_ylim([-1e-15, 1e-15])

        if hasattr(self, 'g'):
            if self.RB is None:
                temp = np.vstack([self.g(y) for y in self.y])
            else:
                temp = np.vstack([self.g(y) for y in self.y_red])

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
            ax6.plot(coords[:, i, 0], coords[:, i, 1], coords[:, i, 2], 'k-')  # plot the trajectory of each particle
            ax6.scatter(coords[-1, i, 0], coords[-1, i, 1], coords[-1, i, 2], s=20)  # plot the final position of each particle

        # Set the axes' labels and title
        ax6.set_xlabel('x')
        ax6.set_ylabel('y')
        ax6.set_zlabel('z')
        ax6.set_title('Phase portrait')

        for ax in [ax0, ax1, ax2, ax3, ax4, ax5]:
            ax.label_outer()

        fig.show()
            
        # string = '_full' if self.RB is None else '_predict' if self.predict else '_repro'
        # filename = os.path.join(MechSystem.data_folder, f"osc_{self.solver_class.__name__}{string}.pdf")
        # save_figure(fig, filename)

    @staticmethod
    def measures(solvers, Omega2_space_dim):
        """Various measurements based on the solution"""
        r_form = lambda numer, denom: r_[float('nan'), 
                (log(roll(numer, -1)/numer)/log(roll(denom, -1)/denom))[:-1]]
        en_error = [solver.en_error for solver in solvers]

        for i in range(0, len(solvers), MechSystem.dt_space_dim * Omega2_space_dim):
            if len(en_error) > 1 and en_error[0] is not None and MechSystem.dt_space_dim > 1:
                r_values = r_form(en_error[i:i+MechSystem.dt_space_dim*Omega2_space_dim:Omega2_space_dim], 
                                MechSystem.dt_space)
                temp = c_[MechSystem.dt_space, r_values].T
                tex_table(solvers[i].solver_class.__name__, temp)
            solvers[i].plot()

class HamiltonianSolverMixin(BaseSolverMixin):
    """Mixin for Hamiltonian-based solvers"""
    def update_deim_hyperreduction(self, kwds, RB, P, PxU_inv_):
        """Update methods with DEIM hyperreduction"""
        y, omega2, beta = MechSystem.y, MechSystem.omega2, MechSystem.beta
        
        Pxham_zz_ = smp.lambdify((y, omega2, beta), 
                                P.T @ MechSystem.ham_zz_expr @ RB, 
                                modules=['scipy'])
        
        kwds.update({
            'ham_zz_': lambda y, omega2, beta: PxU_inv_ @ Pxham_zz_(y @ RB.T, omega2, beta)
        })

    def update_mdeim_hyperreduction(self, kwds, solvers, indices_list, RB, nosc):
        """Update methods with MDEIM hyperreduction"""
        print('Computing ham_zz reduction...')
        
        # Collect ham_zz snapshots
        non_zero_indices = np.nonzero(solvers[0].ham_zz(solvers[0].y[0]).flatten())[0]
        IP = np.zeros((len(non_zero_indices), (2*nosc)**2))
        IP[np.arange(len(non_zero_indices)), non_zero_indices] = 1
        
        F3 = np.hstack([np.array([IP @ solver.ham_zz(y).flatten() 
                                 for y in solver.y[indices]]).T 
                       for solver, indices in zip(solvers, indices_list)])
        
        # Compute POD basis and plot singular values
        Uj, sv, _ = POD(F3, np.eye(F3.shape[0]), self.tol)
        fig, ax = logplot(sv, xlabel='index of singular values', xlims=(1, len(sv)))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        filename = os.path.join(MechSystem.data_folder, f'ham_zz_sv_{self.__class__.__name__}.pdf')
        save_figure(fig, filename)
        
        Pj, _ = DEIM(Uj, plot_deim=False)
        _IP_UxPxU_inv = IP.T @ Uj @ LA.inv(Pj.T @ Uj)

        # Create lambdified function
        ham_zz_col = Pj.T @ IP @ MechSystem.ham_zz_expr.reshape((2*nosc)**2, 1)
        Pxham_zz_ = smp.lambdify((MechSystem.y, MechSystem.omega2, MechSystem.beta), 
                                ham_zz_col, modules=['scipy'])

        # Update dictionary
        kwds.update({
            'ham_zz_': lambda y, omega2, beta: RB.T @ np.reshape(
                _IP_UxPxU_inv @ Pxham_zz_(y @ RB.T, omega2, beta),
                (2*nosc, 2*nosc)
            ) @ RB
        })

class DiscreteGradientMixin(BaseSolverMixin):
    """Mixin for Discrete Gradient specific computations"""
    def update_deim_hyperreduction(self, kwds, RB_dg, P_dg, PxU_inv_dg):
        """Update methods with DEIM hyperreduction"""
        y, y1, omega2 = MechSystem.y, MechSystem.y1, MechSystem.omega2
        
        Pxlag_dg_z_ = smp.lambdify((y, y1, omega2), 
                                  P_dg.T @ MechSystem.lag_dg_z_expr @ RB_dg, 
                                  modules=['scipy'])
        
        kwds.update({
            'lag_dg_z_': lambda y, omega2: PxU_inv_dg @ Pxlag_dg_z_(*(y @ RB_dg.T), omega2)
        })

    def update_mdeim_hyperreduction(self, kwds, solvers, indices_list, RB_dg, nosc):
        """Update methods with MDEIM hyperreduction"""
        print('Computing lag_dg_z reduction...')
        
        # Collect lag_dg_z snapshots
        non_zero_indices = np.nonzero(solvers[0].lag_dg_z(solvers[0].y[0:2]).flatten())[0]
        IP = np.zeros((len(non_zero_indices), (2*nosc)**2))
        IP[np.arange(len(non_zero_indices)), non_zero_indices] = 1
        
        F3 = np.hstack([np.array([IP @ solver.lag_dg_z(y).flatten()
                       for y in zip(solver.y[indices], solver.y[np.array(indices)+1])]).T 
                       for solver, indices in zip(solvers, indices_list)])
        
        # Compute POD basis and plot singular values
        Uj, sv, _ = POD(F3, np.eye(F3.shape[0]), self.tol)
        fig, ax = logplot(sv, xlabel='index of singular values', xlims=(1, len(sv)))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        filename = os.path.join(MechSystem.data_folder, f'lag_dg_z_sv_{self.__class__.__name__}.pdf')
        save_figure(fig, filename)
        
        Pj, _ = DEIM(Uj, plot_deim=False)
        _IP_UxPxU_inv = IP.T @ Uj @ LA.inv(Pj.T @ Uj)

        # Create lambdified function
        y, y1, omega2 = MechSystem.y, MechSystem.y1, MechSystem.omega2
        lag_dg_z_col = Pj.T @ IP @ MechSystem.lag_dg_z_expr.reshape((2*nosc)**2, 1)
        Pxlag_dg_z_ = smp.lambdify((y, y1, omega2), lag_dg_z_col, modules=['scipy'])

        # Update dictionary
        kwds.update({
            'lag_dg_z_': lambda y, omega2: RB_dg.T @ np.reshape(
                _IP_UxPxU_inv @ Pxlag_dg_z_(*(y @ RB_dg.T), omega2),
                (2*nosc, 2*nosc)
            ) @ RB_dg
        })

#%% Concrete solver classes
class DiscreteGradientSolver(DiscreteGradientMixin, DiscreteGradient):
    """DiscreteGradient solver with DG-specific capabilities"""
    def __init__(self, kwds):
        super().__init__(kwds)
        self.g = self.g__
        self.g_prime = (self.g_prime__, lambda y: self.g_prime__(y) @ self.JJ.T)
        
    @classmethod
    def setup_reduced_model(cls, solvers, indices_list, y_list):
        """Setup DG-specific reduced model"""
        print('Computing DG reduced basis...')
        F2 = np.hstack([np.array([solver.lag_dg(y) 
            for y in zip(solver.y[indices], solver.y[np.array(indices)+1])]).T 
            for solver, indices in zip(solvers, indices_list)])
        print(f'{F2.shape = }')
        
        MechSystem.RB_dg, _, MechSystem.nosc_r_dg, _ = PSD(F2, y_list, MechSystem)

    def setup_hyperreduction(self, RB_dg, hyperreducer='DEIM'):
        """Setup DG-specific hyperreduction"""
        if hyperreducer == 'DEIM':
            P_dg, _ = DEIM(RB_dg, plot_deim=False)
            PxU_inv_dg = LA.inv(P_dg.T @ RB_dg)
            self.update_deim_hyperreduction(kwds, RB_dg, P_dg, PxU_inv_dg)
            
        elif hyperreducer == 'MDEIM':
            self.update_mdeim_hyperreduction(
                kwds, solvers, indices_list, RB_dg, nosc)

class ConformalStormerVerletSolver(HamiltonianSolverMixin, ConformalStormerVerlet):
    """CSVSolver with Hamiltonian capabilities"""
    def __init__(self, kwds):
        super().__init__(kwds)
        self.g = self.g__
        self.g_prime = self.g_prime__

    @classmethod
    def setup_reduced_model(cls, solvers, indices_list, y_list):
        """Setup Hamiltonian-specific reduced basis"""
        print('Computing Hamiltonian reduced basis...')
        F2 = np.hstack([np.array([solver.ham_z(y) 
            for y in solver.y[indices]]).T 
            for solver, indices in zip(solvers, indices_list)])
        print(f'{F2.shape = }')
        
        MechSystem.RB, _, MechSystem.nosc_r, _ = PSD(F2, y_list, MechSystem)
                
    def setup_hyperreduction(self, RB, hyperreducer='DEIM'):
        """Setup Hamiltonian-specific hyperreduction"""
        if hyperreducer == 'DEIM':
            P, _ = DEIM(RB, plot_deim=False)
            PxU_inv_ = LA.inv(P.T @ RB)
            self.update_deim_hyperreduction(kwds, RB, P, PxU_inv_)
            
        elif hyperreducer == 'MDEIM':
            self.update_mdeim_hyperreduction(
                kwds, solvers, indices_list, RB, nosc)

class ConformalImplicitMidpointSolver(HamiltonianSolverMixin, ConformalImplicitMidpoint):
    """CIMSolver with Hamiltonian capabilities"""
    def __init__(self, kwds):
        super().__init__(kwds)
        self.g = self.g__
        self.g_prime = self.g_prime__

    # Same implementation as ConformalStormerVerletSolver
    setup_reduced_model = ConformalStormerVerletSolver.setup_reduced_model
    setup_hyperreduction = ConformalStormerVerletSolver.setup_hyperreduction


#%% helper functions

def solve_mech_system(solver_class, dt, Omega2, kwds):
    """Helper function to solve mechanical system"""
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

def parallel_solve_mech_system(kwds, Omega2_space, MSsolvers):

    # Prepare the arguments for solve_mech_system
    args = [(x, y, z, kwds) for x in kwds['registered_solver_classes']
            for y in MechSystem.dt_space for z in Omega2_space]

    # Check the length of args
    if len(args) > 1:
        # Use ThreadPoolExecutor to parallelize the solve_mech_system calls
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda p: solve_mech_system(*p), args))
    else:
        # Sequentially solve the mechanical system
        results = [solve_mech_system(*args[0])]

    if 'pool' in kwds:
        del kwds['pool']
        print("kwds['pool'] deleted")

    # Sort results in the order of submitted jobs
    for arg in args:
        solver_class, dt, Omega2, _ = arg
        for MSsolver in results:
            if MSsolver.solver_class == solver_class and MSsolver.dt == dt and (MSsolver.Omega2 == Omega2).all():
                MSsolvers.append(MSsolver)
                break
            
def setup_and_solve_reduced_system(solver_type, solvers, kwds_base):
    """Setup and solve reduced system for a specific solver type"""
    print(f'Setting up {solver_type} reduced system...')
    
    # Filter solvers by type
    if solver_type == "DG":
        filtered_solvers = [s for s in solvers if isinstance(s, DiscreteGradient)]
        solver_class = DiscreteGradientSolver
        kwds = kwds_base.copy()
        kwds['registered_solver_classes'] = [DiscreteGradientSolver]
    else:  # Hamiltonian
        filtered_solvers = [s for s in solvers if not isinstance(s, DiscreteGradient)]
        solver_class = ConformalStormerVerletSolver
        kwds = kwds_base.copy()
        kwds['registered_solver_classes'] = [ConformalStormerVerletSolver, ConformalImplicitMidpointSolver]

    # Calculate samples and indices
    n_samples_list = [int(solver.n // 2) for solver in filtered_solvers]
    indices_list = [np.random.choice(solver.n-1, n_samples, replace=False)
                   for solver, n_samples in zip(filtered_solvers, n_samples_list)]

    # Create snapshot list
    y_list = np.hstack([solver.y[indices].T for solver, indices 
                        in zip(filtered_solvers, indices_list)])

    # Setup reduced model and update methods
    solver_class.setup_reduced_model(
        filtered_solvers, 
        indices_list, 
        y_list
    )

    # Solve reduced system
    solvers_r = []

    # Update parameters for prediction/reproduction
    if MechSystem.predict:
        Omega2_space = MechSystem._Omega2_space[-1:]
        Omega2_space_dim = len(Omega2_space)
    else:
        Omega2_space = MechSystem._Omega2_space
        Omega2_space_dim = len(Omega2_space)

    parallel_solve_mech_system(kwds, Omega2_space, solvers_r)
    BaseSolverMixin.measures(solvers_r, Omega2_space_dim)
    
    return solvers_r

#%% Main driver
if __name__ == '__main__':
    """Model order reduction of the MechSystem using concrete solvers"""

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

    print('Computing full solution ...')
    parallel_solve_mech_system(kwds, Omega2_space, solvers)
    BaseSolverMixin.measures(solvers, Omega2_space_dim)

    #%% Reduced order solution
    print('Computing reduced bases...')

    # Solve for each solver type
    solvers_r_dg = setup_and_solve_reduced_system("DG", solvers, kwds)
    solvers_r_ham = setup_and_solve_reduced_system("Hamiltonian", solvers, kwds)

    # Merge solver lists
    solvers_r = solvers_r_dg + solvers_r_ham
    raise SystemExit
                
    #%% Hyper-reduced model
    print('Setting up hyper-reduction...')
    
    # Setup hyperreduction for each solver type
    for solver in solvers:
        if isinstance(solver, DiscreteGradientSolver):
            solver.setup_hyperreduction(RB_dg, MechSystem.hyperreducer)
        elif isinstance(solver, (ConformalStormerVerletSolver, ConformalImplicitMidpointSolver)):
            solver.setup_hyperreduction(RB, MechSystem.hyperreducer)

    # Solve hyper-reduced system
    print('Solving hyper-reduced system ...')
    solvers_dr = []
    parallel_solve_mech_system(kwds, Omega2_space, solvers_dr)
    BaseSolverMixin.measures(solvers_dr, Omega2_space_dim)
            
    #%% Compute and display metrics
    array_shape = (len(kwds['registered_solver_classes']), 
                  MechSystem.dt_space_dim, 
                  Omega2_space_dim)
    
    time_lapsed = [reshape([x.time_lapsed for x in solvers], array_shape)]
    
    if not MechSystem.predict:
        print('Solution errors:')
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
            reshape([x.time_lapsed for x in solvers_r], time_lapsed[0].shape[:2]),
            reshape([x.time_lapsed for x in solvers_dr], time_lapsed[0].shape[:2])
        ])

    print(f'{time_lapsed = }')

    # tex_table('', time_lapsed)
    '''
    Observations:
        - hyperreduced model is efficient
            - and symplecitc with sparse MDEIM
            - and not demonstrably symplectic with DEIM
            - order of numerical methods is not verified.
        - The Jacobian calculation in the hyperreduced model
            - is also reduced with DEIM
        - higher order integrators are not verified
        - constrained Hamiltonian system should also show order of the integrator

    # Next steps:
        # Implement elastic beam deformation
        # Second-order of methods not observable under spherical constraints
    '''
