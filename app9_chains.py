#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 17, 2024

@author: ashishbhatt

This app solves a full order and reduced-order model from the System class,
using solver classes in the ODESolver hierarchy of methods,
fixed point iterators from the Newton.py,
and bases from podDEIM.
"""

import numpy as np
import sympy as smp
from numpy import linalg as LA
from pylab import  log, r_, c_, zeros, eye, sqrt, reshape, linspace, roll, figure, zeros_like

import gc
from datetime import datetime
from matplotlib import rc
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)  # Enable LaTeX rendering
rc('text.latex', preamble=r'\usepackage{amsfonts}')  # Load AMSFonts for Fraktur
from matplotlib.ticker import MaxNLocator

import ODESolver
from System import System
from podDEIM import POD, DEIM
from Newton import fixed_point
from PlotScript import plot_data, tex_table, logplot, save_figure, timing


class MechSystem(System):
    """ Class of MechSystem methods """
        
    "Numerical solver and its properties"
    w_values = [0.28, 0.62546642846767004501]
    w_values.append(1.0 -2.0*(sum(w_values)))
    w_values.append(w_values[1])
    w_values.append(w_values[0])
    w_values = [1]
    assert np.isclose(sum(w_values), 1), 'sum_i w_i must be 1'

    "Fixed-point nonliner equations solver properties"
    tol, M, var, store = 1.0E-12, 100, True, False
    
    "System parameters"
    nosc = 40*2
        
    # These two properties only have effect during reduction
    reducer = 'psd'
    predict = True # False = reproduce
    hyperreducer = 'MDEIM'
    
    registered_solver_classes = [ODESolver.ConformalImplicitMidpoint, ODESolver.ConformalStormerVerlet]
    dt_space_dim = 1
    omega_space_dim = 2
    beta = (max(1e-2, 0*np.random.rand()/10))*0
    JJ = lambda self, d=nosc: r_[c_[zeros((d,d)), eye(d)], c_[-eye(d), zeros((d,d))]]
    
    omega_space = np.random.uniform([1-0.9, 1-0.9, 0.01-0.009], [1+0.9, 1+0.9, 0.01+0.009], (omega_space_dim, 3))
        
    dt_space = linspace(0.01, 0.05, num=dt_space_dim)
    T_final = 5
    
    
    keep_time = datetime.now().strftime('%Y-%m-%d_%H-%M_')
    
    "Initial conditions"
    positions = np.zeros(nosc)
    positions[:nosc//2:2] = np.arange(1,nosc//4+1) +np.random.rand(nosc//4)*1e-1
    positions[1:nosc//2:2] = 1
    positions[nosc//2::2] = np.arange(1,nosc//4+1) +np.random.rand(nosc//4)*1e-1
    positions[nosc//2+1::2] = -1
    
    momenta = np.zeros(nosc)
    
    constraint_type = None
    
    y_init = r_[positions, momenta].flatten()
    
    drag = lambda self, x, u: 0 #beta/2 * r_[x, u]
    drag_z = lambda self, x, u: 0 #beta/2 * eye(2*x.shape[0])
            
    def __init__(self, kwds):
        "MechSystem properties"
        
        System.__init__(self, kwds)
        
        "Projection matrices"
        if hasattr(self, 'RB'):
            self.y_init = self.RB.T @ self.y_init
            
            if self.reducer == 'pod':
                self.JJ_r = self.RB.T @ self.JJ() @ self.RB
            elif self.reducer == 'psd':
                self.JJ_r = self.JJ(self.nosc_r)
                
            # if self.hyperreducer == 'MDEIM':
            #     self.JJ_rxRB = self.JJ_r @ self.RB.T
        else:
            self.RB = None
                
        if not hasattr(self, 'P'):
            self.P = None
            self.U = None
        
        if self.solver_class in [ODESolver.ConformalImplicitMidpoint, ODESolver.ConformalStormerVerlet, ODESolver.ImplicitMidpoint]:
            self.solver = self.solver_class(self)
        else:
            NameError('Unknown solver class - %s' % self.solver_class.__name__)
        
        "various measures"
        self.time_lapsed = []

    "y_init alias"
    @property
    def u_init(self):
        return self.y_init

    @u_init.setter
    def u_init(self, value):
        self.y_init = value
        
    @staticmethod
    def solver(kwds):
    
        MSsolvers = []
        errors = {'energy': [], 'spl': []}
        
        if MechSystem.predict: # prediction experiment
            
            if (not hasattr(MechSystem, 'RB')):
                # training parameters
                omega_space = MechSystem.omega_space[:-1]
                omega_space_dim = len(omega_space)
                
            else:
                # testing parameters
                omega_space = MechSystem.omega_space[-1:]
                omega_space_dim = len(omega_space)
                
        else: # reproduction
            omega_space = MechSystem.omega_space
            omega_space_dim = len(omega_space)
            
            
        errors.update({'omega_space_dim': omega_space_dim})
        
        for solver_class, dt, omega in [(x, y, z) for x in MechSystem.registered_solver_classes for y in MechSystem.dt_space for z in omega_space]:
        
            kwds.update({'solver_class': solver_class, \
                        'dt': dt, \
                        'Omega2': omega, \
                        'n': int(round(MechSystem.T_final/dt)),\
                        })
            
            kwds.update({'t_points': linspace(0, MechSystem.T_final, kwds['n']+1)})
            
            MSsolver = MechSystem(kwds)
            
            tl = MSsolver.solve()
            
            MSsolver.time_lapsed.append(tl)
            
            if (not MSsolver.beta) and (MSsolver.constraint_type is None):
                "Energy (Hamiltonian) is an invariant for unconstrained conservative system"
                MSsolver.eng_error = MSsolver.get_en_err()
                errors['energy'].append(sqrt(dt)*LA.norm(MSsolver.eng_error))
                
            if MSsolver.var:
                MSsolver.var_solve()
                # errors['spl'].append(sqrt(dt)*LA.norm(MSsolver.sym_error))
                
            MSsolvers.append(MSsolver)
            
        MechSystem.measures(MSsolvers, errors)
    
        return MSsolvers
    
        
    @timing
    def solve(self):
        
        if self.RB is None:
            self.y = np.zeros((self.n+1, 2*self.nosc))
        else:
            self.y = np.zeros((self.n+1, 2*self.nosc_r))
            self.y_full = np.zeros((self.n+1, 2*self.nosc))
            
            y_ = np.array([self.y_init, self.y_init]) @ self.RB.T
            
            if self.constraint_type:
                fixed_point(self.g, y_, self.g_prime, self.tol, self.M, False)
            
            self.y_full[0] = y_[1]
            self.y_init = self.RB.T @y_[1]
                
        self.y[0] = self.y_init
        if self.store: self.info = []

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
            
        if self.store:
            self.info.append(self.y[k+1])
            self.info = np.vstack(self.info)

        if self.RB is not None:
            self.y_red = self.y
            self.y = self.y_full

    def var_solve(self):
        nosc = self.nosc
        if hasattr(self, 'y_red'):
            nosc = self.y_red.shape[1]//2
            
        self.dpsi = np.zeros((2, 2*nosc, 2*nosc))
        self.dpsi[0] = np.eye(2*nosc)
        self.sym_error = np.zeros(self.n+1)

        for k in range(self.n):
            if hasattr(self, 'y_red'):
                dpsi_, _ = self.solver.var_solve(self.y_red[k:k+2], self.t_points[k:k+2])
            else:
                dpsi_, _ = self.solver.var_solve(self.y[k:k+2], self.t_points[k:k+2])
            self.dpsi[1] = dpsi_[-1]
            sym_error_ =self.solver.symplectic_error(self.dpsi, self.t_points[k:k+2])
            self.sym_error[k+1] = sym_error_[-1]

    def plot(self):
        "plot the results"
        
        fig = figure()
        
        gs = fig.add_gridspec(3, 2, hspace=1)
        # ax = gs.subplots()
        
        ax0 = fig.add_subplot(gs[0,0])
        ax1 = fig.add_subplot(gs[1,0])
        ax2 = fig.add_subplot(gs[2,0])
        ax3 = fig.add_subplot(gs[2,-1])
        ax4 = fig.add_subplot(gs[:2,-1])
        
        if hasattr(self, 'sym_error'):
            plot_data(ax0, self.t_points, self.sym_error)
            
            ax0.margins(y=0.5)
            
            ax0.set_xlim((0, self.T_final))
            ax0.set_ylabel(r'$\Delta Sp$')
            
        if hasattr(self, 'eng_error'):
            plot_data(ax1, self.t_points, self.eng_error)
            # ax1.set_ylim((-max(self.eng_error)*1e1, max(self.eng_error)*1e1))
            ax1.margins(y=0.5)
            ax1.set_xlim((0, self.T_final))
            ax1.set_ylabel(r'$\Delta H$')
            
        if hasattr(self, 'g'):
            temp = np.vstack([self.g(y) for y in self.y])
                
            g_norm = LA.norm(temp, axis=1)
            # temp = log(temp/np.roll(temp, 1))
            # temp = r_[0, temp[1:]]
            plot_data(ax1, self.t_points, g_norm)
            # ax1.set_ylim((-max(temp)*1e1, max(temp)*1e1))
            ax1.margins(y=0.5)
            ax1.set_xlim((0, self.T_final))
            # plot_data(ax2, self.t_points, temp[1])
            # ax2.set_ylim((min(temp[1])*1e-1, max(temp[1])*1e1))
            ax1.set_ylabel(r'$\Delta \mathfrak{P}$')
        
        
        lin_momentum = np.sum(self.y[:, nosc:].reshape(-1, nosc//2, 2), axis=1)
        lim_momentum_err = r_[0, LA.norm(lin_momentum[1:] - lin_momentum[0], axis=1)]
        
        angular_momentum = np.cross(self.y[:, :nosc].reshape(-1, nosc//2, 2), self.y[:, nosc:].reshape(-1, nosc//2, 2))
        angular_momentum_sum = np.sum(angular_momentum, axis=1)
        angular_momentum_err = r_[0, angular_momentum_sum[1:] - angular_momentum_sum[0]]
        
        plot_data(ax2, self.t_points, lim_momentum_err)
        # ax1.set_ylim((-max(temp)*1e1, max(temp)*1e1))
        ax2.margins(y=0.5)
        ax2.set_xlim((0, self.T_final))
        # plot_data(ax2, self.t_points, temp[1])
        # ax2.set_ylim((min(temp[1])*1e-1, max(temp[1])*1e1))
        ax2.set_ylabel(r'$\Delta L$')
            
        ax2.set_xlabel('time')
                
        
        plot_data(ax3, self.t_points, angular_momentum_err)
        # ax1.set_ylim((-max(temp)*1e1, max(temp)*1e1))
        ax3.margins(y=0.5)
        ax3.set_xlim((0, self.T_final))
        # plot_data(ax2, self.t_points, temp[1])
        # ax2.set_ylim((min(temp[1])*1e-1, max(temp[1])*1e1))
        ax3.set_ylabel(r'$\Delta J$')
            
        ax3.set_xlabel('time')
        
        # Plot the particle positions over time
        coords = self.y[:, :nosc].reshape(-1, nosc//2, 2)
        for i in range(coords.shape[1]):
            ax4.plot(coords[:, i, 0], coords[:, i, 1], 'k-')  # plot the trajectory of each particle
            ax4.scatter(coords[0, i, 0], coords[0, i, 1], s=20)  # plot the initial position of each particle
        
        # Set the ax4is labels and title
        ax4.set_xlabel('x')
        ax4.set_ylabel('y')
        ax4.set_title('Phase portrait')
        ax4.set_aspect('equal')

        ax4.margins(0.1, 1)
        # ax4.set_aspect(1.0/ax4.get_data_ratio(), adjustable='box')
        # ax4.grid(axis='x', color="0.9", linestyle='-', linewidth=1)
        
        for ax in [ax0, ax1, ax2]:
            ax.label_outer()
                
        if self.RB is not None:
            if self.predict: string = '_predict'
            else: string = '_repro'
        else:
            string = '_full'

        filename = self.keep_time +'osc_' +self.solver_class.__name__ +string +'.pdf'
        # save_figure(fig, filename)
        
    @staticmethod
    def measures(MSsolvers, errors):
        "Various measurements based on the solution"
        
        r_form = lambda numer, denom: (log(roll(numer, -1)/numer)/log(roll(denom, -1)/denom))[:-1]
        C_form = lambda numer, denom, r: numer[:-1]/(denom[:-1]**r)
            
        for i in np.arange(0, len(MSsolvers), MechSystem.dt_space_dim * errors['omega_space_dim']):
            MSsolver = MSsolvers[i]
            
            r_values = []
            C_values = []
            
            if errors['energy'] and MechSystem.dt_space_dim > 1 and MechSystem.predict is False:
                "Compute convergence rates from the error in Energy"
                r_values.append(r_form(errors['energy'][i:i+MechSystem.dt_space_dim], MechSystem.dt_space))
                C_values.append(C_form(errors['energy'][i:i+MechSystem.dt_space_dim], MechSystem.dt_space,r_values[-1]))
                
            if False:
                '''Compute convergence rate from the error in symplecticness
                Only applicable if the error is non-zero'''
                r_values.append(r_form(errors['spl'][i:i+MechSystem.dt_space_dim],MechSystem.dt_space))
                C_values.append(C_form(errors['spl'][i:i+MechSystem.dt_space_dim],MechSystem.dt_space,r_values[-1]))
                
            # Display convergence rates if available
            if r_values:
                temp = r_[ reshape(MechSystem.dt_space, (1,-1)), \
                          c_[np.reshape([float('nan')]*len(r_values), (len(r_values),1)), r_values]].T
                    
                tex_table(MSsolver.solver_class.__name__, temp.T)
        
            # Plot the measures
            MSsolver.plot()
            

#%% Find symbolic quantities
#TODO: save and reload from disc
if True:
    nosc = MechSystem.nosc
    q = smp.Matrix(smp.symbols('q_:{}_:{}'.format(nosc//2,2), real=True)).reshape(nosc//2,2)
    p = smp.Matrix(smp.symbols('p_:{}_:{}'.format(nosc//2,2), real=True))
    omega = smp.Matrix(smp.symbols('omega_0:{}'.format(3), real=True))
    _l = 1 #1/(nosc//4)
    beta = smp.symbols('beta')
    
    kin_expr = 0.5 * omega[0] * sum(x**2 for x in p)
    
    row_diffs = [(q.row(i+1) - q.row(i)).norm(2)**2 - _l for i in range(nosc//4-1)]
    row_diffs = row_diffs + [(q.row(i+1) - q.row(i)).norm(2)**2 - _l for i in range(nosc//4,nosc//2-1)]
    pot_expr_1 = 0.5 * omega[1] * sum([row_diff**2 for row_diff in row_diffs])
    # pot_expr_1 = 0.5 * omega[0] * sum(row_diffs**2)
    
    pot_expr_2 = omega[2] * sum([smp.exp(-1/2/omega[1]**2 * (q.row(i) -q.row(nosc//4+j)).norm(2)**2) \
                                       for i in range(nosc//4) for j in range(nosc//4)])
    
    q = q.reshape(nosc,1)
    y = list(q)+list(p)
    
    ham_expr = kin_expr + pot_expr_1 + pot_expr_2
    ham_z_expr = smp.Matrix([ham_expr]).jacobian(y).T
    ham_zz_expr = ham_z_expr.jacobian(y)
    
    ham_ = smp.lambdify((q, p, omega, beta), ham_expr)
    _ham_z_ = smp.lambdify((q, p, omega, beta), ham_z_expr)
    ham_zz_ = smp.lambdify((q, p, omega, beta), ham_zz_expr)
    
    ham_z_ = lambda q, p, omega, beta: _ham_z_(q, p, omega, beta).squeeze()

#%% Main driver
if __name__ == '__main__':
    "Model order reduction of the MechSystem using MechSystem"
        
#%% Full order solution
    nosc = MechSystem.nosc
    
    # ham_, ham_z_, ham_zz_ = get_symbols(nosc)

    kwds = {'ham_': ham_, \
            'ham_z_': ham_z_, \
            'ham_zz_': ham_zz_, \
            }

    MSsolvers = MechSystem.solver(kwds)
    
    if MSsolvers[-1].non_quad: # is not None
        assert MechSystem.omega_space_dim == 1
        
    if MechSystem.predict:
        array_shape = (len(MechSystem.registered_solver_classes), MechSystem.dt_space_dim, MechSystem.omega_space_dim-1)
    else:
        array_shape = (len(MechSystem.registered_solver_classes), MechSystem.dt_space_dim, MechSystem.omega_space_dim)        
    
    time_lapsed = [reshape([x.time_lapsed for x in MSsolvers], array_shape)]
    
    1/0
    
#%% Reduced order solution
    print('Assembling snapshots ...')
    y_list = np.hstack([MSsolver.y[np.unique(np.random.randint(0,MSsolver.n,int(MSsolver.n)//2))].T for MSsolver in MSsolvers])
    print(f'{y_list.shape = }')
    F2 = [np.array([MSsolver.ham_z(*np.split(y, 2)) for y in MSsolver.y]) for MSsolver in MSsolvers]
    F2 = np.hstack([F2[i][np.unique(np.random.randint(0,MSsolvers[i].n,int(MSsolvers[i].n)//2))].T for i in range(len(MSsolvers))])
    print(f'{F2.shape = }')
    
    X = {#'Q_half': MSsolvers[-1].Q_spd()[:nosc, :nosc], \
         # 'sqrt': sp.linalg.sqrtm(MSsolvers[-1].Q_spd()), \
         'eye': np.eye(2*nosc), \
         'eye_half': np.eye(nosc)}
    
    if MechSystem.reducer == 'pod':
        RBq, sv_pod_q, nosc_q = POD(c_[y_list[:nosc,:], F2[:nosc,:]], X['eye_half'], MechSystem.tol)
        RBp, sv_pod_p, nosc_p = POD(c_[y_list[nosc:,:], F2[nosc:,:]], X['eye_half'], MechSystem.tol)
    
        nosc_r = max(nosc_q, nosc_p)
        
        RBq = RBq[:, :nosc_r]
        RBp = RBp[:, :nosc_r]
        RB = r_[c_[RBq, zeros_like(RBq)],\
                c_[zeros_like(RBp), RBp]]
        
        sv = c_[sv_pod_q, sv_pod_p].T
        
    elif MechSystem.reducer == 'psd':
        RB, sv, nosc_r = POD(c_[y_list[:nosc,:], y_list[nosc:,:], F2[:nosc, :], F2[nosc:,:]], X['eye_half'], MechSystem.tol)
        
        RB = RB[:, :nosc_r]
        RB = r_[c_[RB, zeros_like(RB)],\
                c_[zeros_like(RB), RB]]
                    
    print(f'{2*nosc_r = }')
        
    fig, ax = logplot(sv, xlabel='index of singular values', xlims=(0, len(sv)))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    filename = MechSystem.keep_time +'osc_sv' + '.pdf'
    # save_figure(fig, filename)
            
    del y_list, F2
    gc.collect()
    
    MechSystem.RB = RB
    MechSystem.nosc_r = nosc_r

    kwds.update({'ham_z_': lambda q, p, omega, beta: RB.T @ ham_z_(q, p, omega, beta),\
                 'ham_zz_': lambda q, p, omega, beta: RB.T @ ham_zz_(q, p, omega, beta) @ RB})
        
    print('solving reduced system ...')
    MSsolvers_r = MechSystem.solver(kwds)
         
    if len(MSsolvers) == len(MSsolvers_r):
        print('solution errors')
        print(reshape([np.amax(abs(MSsolvers[i].y -MSsolvers_r[i].y)) for i in range(len(MSsolvers))], array_shape))
    
        time_lapsed.append(reshape([x.time_lapsed for x in MSsolvers_r],\
                                    time_lapsed[0].shape)/time_lapsed[0]*100)
    else:
        time_lapsed.append(reshape([x.time_lapsed for x in MSsolvers_r],\
                                    time_lapsed[0].shape[:2]))
            

#%% Hyper-reduced model
        
    P, _ = DEIM(RB, plot_deim=False)
    
    # MechSystem.U = RB
    MechSystem.P = P
        
    PxU_inv_ = LA.inv(P.T @ RB)

    Pxham_z_ = smp.lambdify((q, p, omega, beta), P.T @ ham_z_expr)

    kwds.update({'ham_z_': lambda q, p, omega, beta: PxU_inv_ @ Pxham_z_(q, p, omega, beta).squeeze()})
            
    if MechSystem.hyperreducer == 'DEIM':
        Pxham_zz_ = smp.lambdify((q, p, omega, beta), P.T @ ham_zz_expr @ RB)
        kwds.update({'ham_zz_': lambda q, p, omega, beta: PxU_inv_ @ Pxham_zz_(q, p, omega, beta)})
        
    elif MechSystem.hyperreducer == 'MDEIM':
        
        non_zero_indices = np.nonzero(MSsolvers[0].ham_zz(*np.split(MSsolvers[0].y[0], 2)).flatten())[0]
        
        IP = np.zeros((len(non_zero_indices), (2*nosc)**2))
        
        for i, val in enumerate(non_zero_indices):
            IP[i, val] = 1
        
        F3 = [np.array([IP @ MSsolver.ham_zz(*np.split(y, 2)).flatten() for y in MSsolver.y]) for MSsolver in MSsolvers]
        
        F3 = np.hstack([F3[i][np.unique(np.random.randint(0,MSsolvers[i].n,int(MSsolvers[i].n)))].T for i in range(len(MSsolvers))])
        print(f'{F3.shape = }')
        
        Uj, sv, mj = POD(F3, np.eye(F3.shape[0]), MSsolvers[0].tol)
        del F3
            
        fig, ax = logplot(sv, xlabel='index of singular values', xlims=(0, len(sv)))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        filename = MechSystem.keep_time +'osc_mdeim_sv' + '.pdf'
        # save_figure(fig, filename)
        
        Pj, _ = DEIM(Uj, plot_deim=False)
        # Sj = Pj.T @ RB
            
        _IP_UxPxU_inv = IP.T @ Uj @ LA.inv(Pj.T @ Uj)
        
        ham_zz_col = Pj.T @ IP @ ham_zz_expr.reshape((2*nosc)**2, 1)
        
        Pxham_zz_ = smp.lambdify((q, p, omega, beta), ham_zz_col)
        
        kwds.update({'ham_zz_': lambda q, p, omega, beta: RB.T @ np.reshape(_IP_UxPxU_inv @ Pxham_zz_(q, p, omega, beta), (2*nosc, 2*nosc)) @ RB})
                    
    gc.collect()
    print('solving hyper-reduced system ...')
    MSsolvers_dr = MechSystem.solver(kwds)
         
    if len(MSsolvers) == len(MSsolvers_dr):
        print('solution errors')
        print(reshape([np.amax(abs(MSsolvers[i].y -MSsolvers_dr[i].y)) for i in range(len(MSsolvers))], array_shape))
    
        time_lapsed.append(reshape([x.time_lapsed for x in MSsolvers_dr],\
                                    time_lapsed[0].shape)/time_lapsed[0]*100)
    else:
        time_lapsed.append(reshape([x.time_lapsed for x in MSsolvers_dr],\
                                    time_lapsed[0].shape[:2]))
    
    print(f'{time_lapsed = }')
    
    # tex_table('', time_lapsed)
    '''
    
    Observations:
        - hyperreduced model is efficient
            - and symplecitc with sparse MDEIM
            - and not demonstrable symplectic with DEIM
            - order of numerical methods is not verified.
        - The Jacobian calculation in the hyperreduced model
            - is also reduced with DEIM
        - Weighted psd basis gives wrong eigenvalues of the snapshot matrix
            - sort eigenvalues in increasing order
            - multiply Chi by Xh_half
        
    # Next steps:
        # Implement elastic beam deformation
    '''