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

import gc
from datetime import datetime

import numpy as np
import sympy as smp
from numpy import linalg as LA
from pylab import  log, r_, c_, sqrt, reshape, linspace, roll, figure, zeros_like

from pathos.pools import _ProcessPool as Pool

from functools import wraps

from matplotlib.ticker import MaxNLocator
from matplotlib import rc
rc('text', usetex=True)  # Enable LaTeX rendering
rc('text.latex', preamble=r'\usepackage{amsfonts}')  # Load AMSFonts for Fraktur

from ODESolver import ConformalStormerVerlet, ConformalImplicitMidpoint, DiscreteGradient
from Newton import fixed_point
from System import MechSystem, kwds, _ham_z_, ham_z_, _g_lam, g_lam, ham_zz_
from System import nosc, q, p, y, y1, omega2, beta, ham_z_expr, DG_V_expr, ham_zz_expr, g_expr, g_prime_expr, g_prime_lam
from System import _lag_dg_, lag_dg_, lag_dg_z_, lag_dg_expr, lag_dg_z_expr
from System import _Omega2_space
from podDEIM import POD, PSD, DEIM
from PlotScript import plot_data, tex_table, logplot, save_figure, timing

def compose_solver_solves(func):
    @wraps(func)
    def wrapper(self, y_, k):
        for w_val in self.w_values:
            y_, y_full_ = func(self, w_val, y_, k)
                
        return y_, y_full_
    return wrapper
   
class MechSystemSolver(MechSystem):

    def __init__(self, kwds):
    
        super().__init__(kwds)
        self.solver = kwds['pool']['solver_class'](kwds)
        
        if self.solver_class == DiscreteGradient:
            # print(f"{self.JJ().shape =}")
            # print(f"{self.g_prime(self.y_init).shape =}")
            self.g_prime_ = (self.g_prime, lambda y: self.g_prime(y) @ self.JJ(self.y_init.size//2).T)

    @compose_solver_solves
    def solve_for_w(self, w_val, y_, k):
        self.solver.set_initial_condition(y_[1])
        y_, _, info_ = self.solver.solve(w_val*self.t_points[k:k+2])
        if self.store: self.info.append(np.array(info_[0::1]))

        # enforce constraints
        if self.constraint_type:
    
            # if self.RB is not None:
            #     y_full_ = y_ @ self.RB.T
            # else:
            #     y_full_ = y_

            # if self.RB is not None: print(f"pre {LA.norm(y_[1]) =}")
            fixed_point(self.g, y_, self.g_prime_, self.tol, self.M, False)
            # if self.RB is not None: print(f"post {LA.norm(y_[1]) =}")
            
            # if self.RB is not None:
            #     y_ = y_full_ @ self.RB

            if self.RB is None:
                return y_, y_ 
            else:
                return y_, y_ @ self.RB.T


    @timing
    def solve(self):

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
            
    def var_solve(self):
    
        if hasattr(self, 'y_red'):
            y = self.y_red
        else:
            y = self.y
            
        dpsi, _ = self.solver.var_solve(y, self.t_points)
        self.sym_error = self.solver.symplectic_error(dpsi, self.t_points)

    def plot(self):
        "plot the results"

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

        lin_momentum = np.sum(self.y[:, nosc:].reshape(-1, nosc//3, 3), axis=1)
        lim_momentum_err = r_[0, LA.norm(lin_momentum[1:] - lin_momentum[0], axis=1)]

        angular_momentum = np.cross(self.y[:, :nosc].reshape(-1, nosc//3, 3), \
                                    self.y[:, nosc:].reshape(-1, nosc//3, 3))
        angular_momentum_sum = np.sum(angular_momentum, axis=1)
        angular_momentum_err = r_[0, \
                                  LA.norm(angular_momentum_sum[1:] - angular_momentum_sum[0], axis=1)]

        plot_data(ax3, self.t_points, lim_momentum_err, xlims=(0, self.T_final), \
                  ylabel=r'$\Delta L$', margins=0.5)

        plot_data(ax4, self.t_points, angular_momentum_err, xlims=(0, self.T_final), \
                  ylabel=r'$\Delta J$', margins=1)

        # # Plot matrix condition number
        # if self.RB is not None:
        #     temp = lambda t: self.g_prime[0](self.y[t])[:,:nosc] @ self.RB[:nosc,:self.nosc_r]
        #     mat = lambda t: temp(t) @ self.ham_zz(*np.split(self.y[t],2))[self.nosc_r:,self.nosc_r:] @ temp(t).T
        # else:
        #     temp = lambda t: self.g_prime[0](self.y[t])[:,:nosc]
        #     mat = lambda t: temp(t) @ self.ham_zz(*np.split(self.y[t],2))[nosc:,nosc:] @ temp(t).T

        # # self.mat_cond = [LA.cond(mat(t), 2) for t in range(self.n)]
        # self.mat_cond = LA.cond(np.array([mat(t) for t in range(self.n)]), 2)

        # plot_data(ax5, self.t_points[2:], self.mat_cond[1:], xlims=(0, self.T_final), \
        #           ylabel=r'$\kappa$', margins=0.5)

        ax4.set_xlabel('time')

        # Plot the particle positions over time
        coords = self.y[:, :nosc].reshape(-1, nosc//3, 3)
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
            
        # string = '_full' if self.RB is None else '_predict' if self.predict else '_repro'
        # filename = f"{self.keep_time}osc_{self.solver_class.__name__}{string}.pdf"
        # save_figure(fig, filename)

    @staticmethod
    def measures(MSsolvers, Omega2_space_dim):
        "Various measurements based on the solution"

        r_form = lambda numer, denom: r_[float('nan'), (log(roll(numer, -1)/numer)/log(roll(denom, -1)/denom))[:-1]]
        en_error = [solver.en_error for solver in MSsolvers]

        for i in range(0, len(MSsolvers), MechSystem.dt_space_dim * Omega2_space_dim):
            
            if len(en_error) > 1 and en_error[0] is not None and MechSystem.dt_space_dim > 1:
                "Compute convergence rates from the error in Energy"
                r_values = r_form(en_error[i:i+MechSystem.dt_space_dim*Omega2_space_dim:Omega2_space_dim], MechSystem.dt_space)
                temp = c_[MechSystem.dt_space, r_values].T
                tex_table(MSsolvers[i].solver_class.__name__, temp)

            # Plot the measures
            MSsolvers[i].plot()
    
#%% helper functions

def solve_mech_system(solver_class, dt, Omega2, kwds):
    kwds['pool'] = {'solver_class': solver_class, 'dt': dt, 'Omega2': Omega2, \
                    'n': int(round(MechSystem.T_final/dt))}
    kwds['pool'].update({'t_points': linspace(0, MechSystem.T_final, kwds['pool']['n']+1)})

    MSsolver = MechSystemSolver(kwds)
    tl = MSsolver.solve()
    MSsolver.time_lapsed.append(tl)

    if not MSsolver.beta:
        MSsolver.eng_error = MSsolver.get_en_err()
        MSsolver.en_error = sqrt(dt)*LA.norm(MSsolver.eng_error)
    else:
        MSsolver.en_error = None

    if MSsolver.var:
        MSsolver.var_solve()

    return MSsolver


def parallel_solve_mech_system(kwds, Omega2_space, MSsolvers):

    with Pool() as pool:

        results = pool.starmap(solve_mech_system, \
                               [(x, y, z, kwds) for x in kwds['registered_solver_classes'] \
                                for y in MechSystem.dt_space for z in Omega2_space])

    if 'pool' in kwds:
        del kwds['pool']
        print("kwds['pool'] deleted")

    # rearrange results in the order of submitted jobs
    for x, y, z in [(x, y, z) for x in kwds['registered_solver_classes'] \
                    for y in MechSystem.dt_space for z in Omega2_space]:
        for MSsolver in results:
            if MSsolver.solver_class == x and MSsolver.dt == y and (MSsolver.Omega2 == z).all():
                MSsolvers.append(MSsolver)
                break
            

#%% Main driver

if __name__ == '__main__':
    "Model order reduction of the MechSystem using MechSystem"

#%% Full order solution
    
    kwds.update({'registered_solver_classes': [DiscreteGradient]})

    MSsolvers = []

    if MechSystem.predict: # prediction experiment
        # training parameters
        Omega2_space = _Omega2_space[:-1]
    else: # reproduction
        # training parameters
        Omega2_space = _Omega2_space
        
    Omega2_space_dim = len(Omega2_space)

    print('Computing full solution ...')
    parallel_solve_mech_system(kwds, Omega2_space, MSsolvers)

    MechSystemSolver.measures(MSsolvers, Omega2_space_dim)

    array_shape = (len(kwds['registered_solver_classes']), MechSystem.dt_space_dim, Omega2_space_dim)

    time_lapsed = [reshape([x.time_lapsed for x in MSsolvers], array_shape)]
    
#%% Reduced order solution
    
    print('Assembling snapshots ...')
    n_samples_list = [int(MSsolver.n // 2) for MSsolver in MSsolvers]
    indices_list = [np.random.choice(MSsolver.n-1, n_samples, replace=False) for MSsolver, n_samples in zip(MSsolvers, n_samples_list)]
    y_list = np.hstack([MSsolver.y[indices].T for MSsolver, indices in zip(MSsolvers, indices_list)])
    print(f'{y_list.shape = }')
    
    F2 = np.hstack([np.array([MSsolver.ham_z(*np.split(y, 2)) for y in MSsolver.y[indices]]).T for MSsolver, indices in zip(MSsolvers, indices_list)])
    print(f'{F2.shape = }')
    
    RB, _, nosc_r, _ = PSD(F2, y_list, nosc, MechSystem.tol)
    
    F2_dg = np.hstack([np.array([MSsolver.lag_dg(y) \
                                for y in zip(MSsolver.y[indices], MSsolver.y[np.array(indices)+1])]).T \
                                for MSsolver, indices in zip(MSsolvers, indices_list)])
    print(f'{F2_dg.shape = }')
    
    RB_dg, _, nosc_r_dg, _ = PSD(F2_dg, y_list, nosc, MechSystem.tol)

    MechSystem.RB = RB_dg
    MechSystem.nosc_r = nosc_r_dg

    # TODO: multiply ham_z_ by RB on the inside
    kwds.update({'ham_z_': lambda q, p, omega2, beta: RB.T @ ham_z_(q, p, omega2, beta),
                 'ham_zz_': lambda q, p, omega2, beta: RB.T @ ham_zz_(q, p, omega2, beta) @ RB,
                 # 'RB': RB,
                 # 'RB_dg': RB_dg,
                 # 'nosc_r': nosc_r,
                 # 'nosc_r_dg': nosc_r_dg,
                 })

    # TODO: change the pre-multiplier matrix to be the symplectic inverse
    if 'lag_dg_' in locals():
        kwds.update({'lag_dg_': lambda y, Omega2: RB_dg.T @ lag_dg_(y @ RB_dg.T, Omega2),
                     'lag_dg_z_': lambda y, Omega2: RB_dg.T @ lag_dg_z_(y @ RB_dg.T, Omega2) @ RB_dg,
                     'g': lambda y: g_lam( y @ RB_dg.T),
                     'g_prime': lambda y: g_prime_lam( y @ RB_dg.T) @ RB_dg,
                     })

    print('solving reduced system ...')
            
    MSsolvers_r = []
    
    if MechSystem.predict: # prediction experiment
        # testing parameters
        Omega2_space = _Omega2_space[-1:]
        Omega2_space_dim = len(Omega2_space)
    
    else: # reproduction
        Omega2_space = _Omega2_space
        
    Omega2_space_dim = len(Omega2_space)
    
    parallel_solve_mech_system(kwds, Omega2_space, MSsolvers_r)
    
    MechSystemSolver.measures(MSsolvers_r, Omega2_space_dim)
            
    if not MechSystem.predict:
        print('solution errors')
        print(reshape([np.amax(abs(y1 - y2)) for y1, y2 in zip([x.y for x in MSsolvers], [x.y for x in MSsolvers_r])], array_shape))
    
        time_lapsed.append(reshape([x.time_lapsed for x in MSsolvers_r], time_lapsed[0].shape) / time_lapsed[0] * 100)
    else:
        time_lapsed.append(reshape([x.time_lapsed for x in MSsolvers_r], time_lapsed[0].shape[:2]))

#%% Hyper-reduced model

    P, _ = DEIM(RB, plot_deim=False)

    # MechSystem.U = RB
    MechSystem.P = P

    PxU_inv_ = LA.inv(P.T @ RB)

    Pxham_z_ = smp.lambdify((q, p, omega2, beta), P.T @ ham_z_expr.flat(), modules=['scipy'])

    kwds.update({'ham_z_': lambda q, p, omega2, beta: PxU_inv_ @ Pxham_z_(q, p, omega2, beta)})

    if 'lag_dg_' in locals():
        P_dg, _ = DEIM(RB_dg, plot_deim=False)

        PxU_inv_dg = LA.inv(P_dg.T @ RB_dg)

        Pxlag_dg_ = smp.lambdify((y, y1, omega2), P_dg.T @ lag_dg_expr.flat(), modules=['scipy'])

        kwds.update({'lag_dg_': lambda y, Omega2: PxU_inv_dg @ Pxlag_dg_(*(y @ RB_dg.T), Omega2)})

    if MechSystem.hyperreducer == 'DEIM':
        Pxham_zz_ = smp.lambdify((q, p, omega2, beta), P.T @ ham_zz_expr @ RB, modules=['scipy'])
        kwds.update({'ham_zz_': lambda q, p, omega2, beta: PxU_inv_ @ Pxham_zz_(q, p, omega2, beta)})
        
        Pxlag_dg_z_ = smp.lambdify((y, y1, omega2), P_dg.T @ lag_dg_z_expr @ RB_dg, modules=['scipy'])
        kwds.update({'lag_dg_z_': lambda y, omega2: PxU_inv_dg @ Pxlag_dg_z_(*(y @ RB_dg.T), omega2)})

    elif MechSystem.hyperreducer == 'MDEIM':

        non_zero_indices = np.nonzero(MSsolvers[0].ham_zz(*np.split(MSsolvers[0].y[0], 2)).flatten())[0]

        IP = np.zeros((len(non_zero_indices), (2*nosc)**2))
            
        IP[np.arange(len(non_zero_indices)), non_zero_indices] = 1
        
        F3 = np.hstack([np.array([IP @ MSsolver.ham_zz(*np.split(y, 2)).flatten() \
                                  for y in MSsolver.y[indices]]).T \
                                    for MSsolver, indices in zip(MSsolvers, indices_list)])
        print(f'{F3.shape = }')

        Uj, sv, mj = POD(F3, np.eye(F3.shape[0]), MSsolvers[0].tol)
        del F3

        print(f'{mj = }')
    
        fig, ax = logplot(sv, xlabel='index of singular values', xlims=(1, len(sv)))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # filename = MechSystem.keep_time +'osc_mdeim_sv' + '.pdf'
        # save_figure(fig, filename)

        Pj, _ = DEIM(Uj, plot_deim=False)

        _IP_UxPxU_inv = IP.T @ Uj @ LA.inv(Pj.T @ Uj)

        ham_zz_col = Pj.T @ IP @ ham_zz_expr.reshape((2*nosc)**2, 1)

        Pxham_zz_ = smp.lambdify((q, p, omega2, beta), ham_zz_col, modules=['scipy'])

        kwds.update({'ham_zz_': lambda q, p, omega2, beta: RB.T @ np.reshape(_IP_UxPxU_inv @ Pxham_zz_(q, p, omega2, beta), (2*nosc, 2*nosc)) @ RB})

        if 'lag_dg_' in locals():

            non_zero_indices = np.nonzero(MSsolvers[0].lag_dg_z(MSsolvers[0].y[0:2]).flatten())[0]

            IP = np.zeros((len(non_zero_indices), (2*nosc)**2))
                
            IP[np.arange(len(non_zero_indices)), non_zero_indices] = 1
            
            F3 = np.hstack([np.array([IP @ MSsolver.lag_dg_z(y).flatten()\
                                        for y in zip(MSsolver.y[indices], MSsolver.y[np.array(indices)+1])]).T \
                                        for MSsolver, indices in zip(MSsolvers, indices_list)])
            print(f'{F3.shape = }')

            Uj, sv, mj = POD(F3, np.eye(F3.shape[0]), MSsolvers[0].tol)
            del F3

            print(f'{mj = }')
        
            fig, ax = logplot(sv, xlabel='index of singular values', xlims=(1, len(sv)))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            # filename = MechSystem.keep_time +'osc_mdeim_sv' + '.pdf'
            # save_figure(fig, filename)

            Pj, _ = DEIM(Uj, plot_deim=False)

            _IP_UxPxU_inv = IP.T @ Uj @ LA.inv(Pj.T @ Uj)

            lag_dg_z_col = Pj.T @ IP @ lag_dg_z_expr.reshape((2*nosc)**2, 1)

            Pxlag_dg_z_ = smp.lambdify((y, y1, omega2), lag_dg_z_col, modules=['scipy'])

            kwds.update({'lag_dg_z_': lambda y, omega2: RB_dg.T @ np.reshape(_IP_UxPxU_inv @ Pxlag_dg_z_(*(y @ RB_dg.T), omega2), (2*nosc, 2*nosc)) @ RB_dg})
        
        
    if MechSystem.constraint_type is not None and MechSystem.constraints_reduce:

        F5 = np.hstack([np.array([MSsolver.g(y) + 0 for y in MSsolver.y[indices]]).T for MSsolver, indices in zip(MSsolvers, indices_list)])
        print(f'{F5.shape = }')

        Uj, sv, mj = POD(F5, np.eye(F5.shape[0]), MechSystem.tol)
        del F5

        print(f'{mj = }')

        Pj, _ = DEIM(Uj, plot_deim=False)

        fig, ax = logplot(sv, xlabel='index of singular values', xlims=(1, len(sv)))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        filename = MechSystem.keep_time +'osc_mdeim_sv' + '.pdf'

        _UxPxU_inv = Uj @ LA.inv(Pj.T @ Uj)

        _g_col = Pj.T @ g_expr

        _Pxg = smp.lambdify((y,), _g_col, modules=['scipy'])

        kwds.update({'g': lambda y: (_UxPxU_inv @ _Pxg(y) -0).squeeze()})

        non_zero_indices = np.nonzero(MSsolvers[0].g_prime(MSsolvers[0].y[0]).flatten())[0]

        g_prime_shape= MSsolvers[0].g_prime(MSsolvers[0].y[0]).shape
        IP = np.zeros((len(non_zero_indices), np.prod(g_prime_shape)))
            
        IP[np.arange(len(non_zero_indices)), non_zero_indices] = 1
        
        F4 = np.hstack([np.array([IP @ MSsolver.g_prime(y).flatten() for y in MSsolver.y[indices]]).T for MSsolver, indices in zip(MSsolvers, indices_list)])
        print(f'{F4.shape = }')

        Uj, sv, mj = POD(F4, np.eye(F4.shape[0]), MechSystem.tol)
        del F4

        print(f'{mj = }')

        fig, ax = logplot(sv, xlabel='index of singular values', xlims=(1, len(sv)))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # filename = MechSystem.keep_time +'osc_mdeim_sv' + '.pdf'
        # save_figure(fig, filename)

        Pj, _ = DEIM(Uj, plot_deim=False)

        IP_UxPxU_inv_ = IP.T @ Uj @ LA.inv(Pj.T @ Uj)

        g_prime_col = Pj.T @ IP @ g_prime_expr.reshape(np.prod(g_prime_shape), 1)

        Pxg_prime_ = smp.lambdify((y,), g_prime_col, modules=['scipy'])

        kwds.update({'g_prime': lambda y: np.reshape(IP_UxPxU_inv_ @ Pxg_prime_(y), g_prime_shape) })
            
        
        if 'lag_dg_' in locals():
            kwds.update({'g': lambda y: (_UxPxU_inv @ _Pxg( y @ RB_dg.T ) -0).squeeze(),
                         'g_prime': lambda y: np.reshape(IP_UxPxU_inv_ @ Pxg_prime_( y @ RB_dg.T), g_prime_shape) @ RB_dg})

    gc.collect()
    print('solving hyper-reduced system ...')

    MSsolvers_dr =[]
    parallel_solve_mech_system(kwds, Omega2_space, MSsolvers_dr)
    
    MechSystemSolver.measures(MSsolvers_dr, Omega2_space_dim)
            
    if not MechSystem.predict:
        print('solution errors')
        print(reshape([np.amax(abs(y1 - y2)) for y1, y2 in zip([x.y for x in MSsolvers], [x.y for x in MSsolvers_dr])], array_shape))
    
        time_lapsed.append(reshape([x.time_lapsed for x in MSsolvers_dr], time_lapsed[0].shape) / time_lapsed[0] * 100)
    else:
        time_lapsed.append(reshape([x.time_lapsed for x in MSsolvers_dr], time_lapsed[0].shape[:2]))

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
