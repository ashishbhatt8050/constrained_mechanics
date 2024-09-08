#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 11:49:50 2020

@author: ashishbhatt

This app solves a full order and reduced-order model from the System. class
using solver classes in the ODESolver hierarchy of methods,
fixed point iterators from the Newton.py,
and bases from podDEIM.
"""

import numpy as np
import sympy as smp
from numpy import linalg as LA
from pylab import  log, r_, c_, zeros, eye, sqrt, reshape, linspace, roll, figure, zeros_like
import ODESolver
from System import System
from podDEIM import POD, DEIM
from PlotScript import plot_data, tex_table, logplot, save_figure, timing
import gc
from datetime import datetime
from matplotlib import rc
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)  # Enable LaTeX rendering
rc('text.latex', preamble=r'\usepackage{amsfonts}')  # Load AMSFonts for Fraktur
from matplotlib.ticker import MaxNLocator

from Newton import fixed_point

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
    tol, M, var, store = 1.0E-12, 100, False, False
    
    "System parameters"
    nosc = 50
        
    # These two properties only have effect during reduction
    reducer = 'psd'
    predict = False # False = reproduce
    hyperreducer = 'MDEIM'
    
    registered_solver_classes = [ODESolver.ConformalImplicitMidpoint, ODESolver.ConformalStormerVerlet]
    dt_space_dim = 5
    Omega2_space_dim = 1
    beta = (max(1e-2, 0*np.random.rand()/10))*0
    JJ = lambda self, d=nosc: r_[c_[zeros((d,d)), eye(d)], c_[-eye(d), zeros((d,d))]]
    
    Omega2_space = np.sort(3*(1 -np.random.rand(Omega2_space_dim, nosc)))
        
    dt_space = linspace(0.01, 0.05, num=dt_space_dim)
    T_final = 2
    
    osc_idx = np.array([0, 1, 2, nosc//2-1, nosc//2, nosc//2+1, nosc-3, nosc-2, nosc-1])
    keep_time = datetime.now().strftime('%Y-%m-%d_%H-%M_')
    
    "Initial conditions"
    y_init = r_[np.linspace(1,5,nosc), np.zeros(nosc)]
    
    "Constraints"
    constraint_type = None
    constraints_reduce = True # Reduce constraint jacobian g_prime
    
    alpha = list(np.linspace(0.1,0.5,num=nosc))
    alpha /= sqrt(sum(np.array(alpha)**2))
    assert np.isclose(np.array(alpha).dot(alpha), 1), "alpha**2 must be equal to 1"
    
    if constraint_type == 'linear':
        y_init[nosc-1] = -(y_init[:nosc-1].dot(alpha[:nosc-1]))/alpha[nosc-1] # project on the manifold
        assert np.isclose(y_init[:nosc].dot(alpha[:nosc]), 0), "alpha:y should be 0" 
    elif constraint_type == 'spherical':
        y_init /= sqrt((y_init[:nosc]**2).dot(alpha[:nosc])) # project on the manifold
        assert np.isclose((y_init[:nosc]**2).dot(alpha[:nosc]), 1), "alpha:y^2 should be 1" 
    elif constraint_type == None:
        pass
    else:
        raise NameError
    
    drag = lambda self, x, u: beta/2 * r_[x, u]
    drag_z = lambda self, x, u: beta/2 * eye(2*x.shape[0])
            
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
                
            if self.hyperreducer == 'MDEIM':
                self.JJ_rxRB = self.JJ_r @ self.RB.T
        else:
            self.RB = None
            
        "hyper-reduction"
        if hasattr(self, 'P'):
                            
            _PxU_inv = LA.inv(self.P.T @ self.U)
            
            if self.P.shape == self.U.shape: #POD-DEIM
                _sys_solve = LA.solve
            else: #Gappy-POD
                _sys_solve = lambda A, b: LA.lstsq(A, b, rcond=None)[0]
                
            self.hat = lambda foRBxy: _PxU_inv @ foRBxy #_sys_solve(_PxU, foRBxy)
            
            if MechSystem.hyperreducer == 'MDEIM':    
                
                _IPxUjxPjxUj_inv = self.IP.T @ self.Uj @ LA.inv(self.Pj.T @ self.Uj)
                                    
                self.hat_jac = lambda PjxIPxJac: np.reshape(_IPxUjxPjxUj_inv @ PjxIPxJac, (2*nosc, 2*nosc))
                
                # self.hat_jac = lambda jac: np.sum( [self.theta(jac)[i] * self.Jn[i] for i in range(self.mj)], axis=0 )
                        
            if self.constraints_reduce:
                    
                self.g_prime = lambda y : c_[self.hat_gp0(y),\
                                             self.hat_gp1(y)].T
                
        else:
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
                Omega2_space = MechSystem.Omega2_space[:-1]
                Omega2_space_dim = len(Omega2_space)
                
            else:
                # testing parameters
                Omega2_space = MechSystem.Omega2_space[-1:]
                Omega2_space_dim = len(Omega2_space)
                
        else: # reproduction
            Omega2_space = MechSystem.Omega2_space
            Omega2_space_dim = len(Omega2_space)
            
            
        errors.update({'Omega2_space_dim': Omega2_space_dim})
        
        for solver_class, dt, Omega2 in [(x, y, z) for x in MechSystem.registered_solver_classes for y in MechSystem.dt_space for z in Omega2_space]:
        
            kwds.update({'solver_class': solver_class, \
                        'dt': dt, \
                        'Omega2': Omega2, \
                        'n': int(round(MechSystem.T_final/dt)),\
                        # 'ham_': ham_, \
                        # 'ham_z_': ham_z_, \
                        # 'ham_zz_': ham_zz_, \
                        })
            
            kwds.update({'t_points': linspace(0, MechSystem.T_final, kwds['n']+1)})
            
            MSsolver = MechSystem(kwds)
            
            tl = MSsolver.solve()
            
            MSsolver.time_lapsed.append(tl)
            
            if (not MSsolver.beta):
                "Energy (Hamiltonian) is an invariant of undamped system"
                MSsolver.eng_error = MSsolver.get_en_err()
                
                if MechSystem.dt_space_dim > 1 and MechSystem.Omega2_space_dim == 1:
                     # compute errors only for fixed Omega2_space of length 1
                     errors['energy'].append(sqrt(dt)*LA.norm(MSsolver.eng_error))
                
            if MSsolver.var:
                MSsolver.var_solve()
                # errors['spl'].append(sqrt(dt)*LA.norm(MSsolver.sym_error))
                
            if not hasattr(MechSystem, 'RB'):
                MSsolver.F2 = np.array([MSsolver.ham_z(*np.split(y, 2)) for y in MSsolver.y])
                
            MSsolvers.append(MSsolver)
            
        MechSystem.measures(MSsolvers, errors)
    
        return MSsolvers
    
        
    @timing
    def solve(self):
        
        self.g_arr = np.zeros((self.n+1, 2))
        
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
        
        gs = fig.add_gridspec(2, 2, hspace=1)
        # ax = gs.subplots()
        
        ax0 = fig.add_subplot(gs[0,0])
        ax1 = fig.add_subplot(gs[1,0])
        # ax2 = fig.add_subplot(gs[2,0])
        ax4 = fig.add_subplot(gs[:,-1])
        
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
            
        elif hasattr(self, 'g'):
            if hasattr(self, 'y_red') and False:
                temp = np.hstack([self.g(self.y_red[[i],:]) for i in range(self.n+1)])
            elif False:
                temp = np.hstack([self.g(self.y[[i],:]) for i in range(self.n+1)])
                
            temp = LA.norm(self.g_arr, axis=1)
            # temp = log(temp/np.roll(temp, 1))
            # temp = r_[0, temp[1:]]
            plot_data(ax1, self.t_points, temp)
            # ax1.set_ylim((-max(temp)*1e1, max(temp)*1e1))
            ax1.margins(y=0.5)
            ax1.set_xlim((0, self.T_final))
            # plot_data(ax2, self.t_points, temp[1])
            # ax2.set_ylim((min(temp[1])*1e-1, max(temp[1])*1e1))
            ax1.set_ylabel(r'$\Delta \mathfrak{P}$')
            
        ax1.set_xlabel('time')
                
        plot_data(ax4, self.y[:,self.osc_idx], self.y[:,self.nosc+self.osc_idx])
        ax4.margins(0.5, 0.5)
        ax4.set_aspect(1.0/ax4.get_data_ratio(), adjustable='box')
        ax4.set_xlabel('q')
        ax4.set_ylabel('p')
        ax4.grid(axis='x', color="0.9", linestyle='-', linewidth=1)
        
        for ax in [ax0, ax1]:
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
            
        for i in np.arange(0, len(MSsolvers), MechSystem.dt_space_dim * errors['Omega2_space_dim']):
            MSsolver = MSsolvers[i]
            
            r_values = []
            C_values = []
            
            if errors['energy']:
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

nosc = MechSystem.nosc
x = smp.Matrix(smp.symbols('x0:{}'.format(nosc)))
u = smp.Matrix(smp.symbols('u0:{}'.format(nosc)))
omega2 = smp.Matrix(smp.symbols('omega^2_0:{}'.format(nosc)))
beta = smp.symbols('beta')

kin_expr = sum((u[i]**2)/2.0 for i in range(nosc))
# kin = smp.lambdify((u,), kin_expr, modules='numpy')

pot_expr = 1 -sum(smp.cos(omega2[i]*x[i]) for i in range(nosc))
# pot = smp.lambdify((x, omega2), pot_expr)

ham_expr = kin_expr + pot_expr +beta/2 * sum(x[i]*u[i] for i in range(nosc))
ham_lam = smp.lambdify((x, u, omega2, beta), ham_expr, "sympy")
# print(ham_lam(x, u, omega2, beta))

ham_ = smp.lambdify((x, u, omega2, beta), ham_expr)
# self.ham = lambda x, u, omega2=self.omega2, beta=self.beta: ham_(x, u, omega2, beta)
# u_arr, v_arr, Om2_arr = np.random.rand(3,nosc)
# print(ham(u_arr, v_arr, Om2_arr, 0.1))

ham_z_expr = smp.Matrix([ham_lam(x, u, omega2, beta)]).jacobian(list(x)+list(u)).T
ham_z_lam = smp.lambdify((x, u, omega2, beta), ham_z_expr, "sympy")
# print(ham_z_lam(x, u, omega2, beta))

_ham_z_ = smp.lambdify((x, u, omega2, beta), ham_z_expr)

ham_z_ = lambda x, u, omega2, beta: _ham_z_(x, u, omega2, beta).squeeze()
# self.ham_z = lambda x, u, omega2=self.omega2, beta=self.beta: ham_z_(x, u, omega2, beta).squeeze()
# print(ham_z(u_arr, v_arr, Om2_arr, 0.1))

ham_zz_expr = ham_z_lam(x, u, omega2, beta).jacobian(list(x)+list(u))
# ham_zz_lam = smp.lambdify((x, u, omega2, beta), ham_zz_expr, "sympy")
# print(ham_zz_lam(x, u, omega2, beta))

ham_zz_ = smp.lambdify((x, u, omega2, beta), ham_zz_expr)
# self.ham_zz = lambda x, u, omega2=self.omega2, beta=self.beta: ham_zz_(x, u, omega2, beta).squeeze()

#%%
if MechSystem.constraint_type is not None:
    alpha = smp.Matrix(MechSystem.alpha)
    y = smp.Matrix(smp.symbols('y:{}'.format(2*nosc)))
    
    A_mat = smp.zeros(2*nosc, 2*nosc)
    A_mat[:nosc, :nosc] = smp.diag(*alpha)
    B_mat = smp.zeros(2*nosc, 2*nosc)
    B_mat[:nosc, nosc:] = smp.diag(*alpha)
    B_mat[nosc:, :nosc] = smp.diag(*alpha)
    
    g_expr = smp.Matrix([y.T * A_mat * y - smp.Matrix([1]), y.T * B_mat * y])
    g_prime_expr = g_expr.jacobian(y)
    _g_lam = smp.lambdify((y,), g_expr)
    _g_prime_lam = smp.lambdify((y,), g_prime_expr)
    
    g_lam = lambda x: _g_lam(x).squeeze()
    g_prime_lam = lambda x: _g_prime_lam(x)
    
    # del A_mat, B_mat, alpha, y, g, g_prime

#%% Main driver
if __name__ == '__main__':
    "Model order reduction of the MechSystem using MechSystem"
        
#% Full order solution
    nosc = MechSystem.nosc
    
    # ham_, ham_z_, ham_zz_ = get_symbols(nosc)

    kwds = {'ham_': ham_, \
            'ham_z_': ham_z_, \
            'ham_zz_': ham_zz_, \
            'g': g_lam, \
            'g_prime': g_prime_lam, \
            }

    MSsolvers = MechSystem.solver(kwds)
    
    if MSsolvers[-1].non_quad: # is not None
        assert MechSystem.Omega2_space_dim == 1
        
    if MechSystem.predict:
        array_shape = (len(MechSystem.registered_solver_classes), MechSystem.dt_space_dim, MechSystem.Omega2_space_dim-1)
    else:
        array_shape = (len(MechSystem.registered_solver_classes), MechSystem.dt_space_dim, MechSystem.Omega2_space_dim)        
    
    time_lapsed = [reshape([x.time_lapsed for x in MSsolvers], array_shape)]
    
#%% Reduced order solution
    print('Assembling snapshots ...')
    y_list = np.hstack([MSsolver.y[np.unique(np.random.randint(0,MSsolver.n,int(MSsolver.n)))].T for MSsolver in MSsolvers])
    print(y_list.shape)
    F2 = np.hstack([MSsolver.F2[np.unique(np.random.randint(0,MSsolver.n,int(MSsolver.n)))].T for MSsolver in MSsolvers])
    print(F2.shape)
    
    X = {'Q_half': MSsolvers[-1].Q_spd()[:nosc, :nosc], \
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
                    
    print('2*nosc_r = %s' %(2*nosc_r))
        
    fig, ax = logplot(sv, xlabel='index of singular values', xlims=(0, len(sv)))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    filename = MechSystem.keep_time +'osc_sv' + '.pdf'
    # save_figure(fig, filename)
            
    del y_list, F2
    gc.collect()
    
    MechSystem.RB = RB
    MechSystem.nosc_r = nosc_r

    kwds.update({'ham_z_': lambda q, p, omega2, beta: RB.T @ ham_z_(q, p, omega2, beta),\
                 'ham_zz_': lambda q, p, omega2, beta: RB.T @ ham_zz_(q, p, omega2, beta) @ RB})
        
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
            
    1/0

#%% Hyper-reduced model
        
    P, _ = DEIM(RB, plot_deim=False)
    
    MechSystem.U = RB
    MechSystem.P = P

    Pxham_z_ = smp.lambdify((x, u, omega2, beta), P.T @ ham_z_expr)
    Pxham_zz_ = smp.lambdify((x, u, omega2, beta), P.T @ ham_zz_expr)

    kwds = {'ham_': ham_, \
            'ham_z_': Pxham_z_, \
            'ham_zz_': Pxham_zz_, \
            'g': g_lam, \
            'g_prime': g_prime_lam, \
            }
            
    if MechSystem.hyperreducer == 'MDEIM':
        non_zero_indices = np.nonzero(MSsolvers[0].ham_zz(*np.split(MSsolvers[0].y[0], 2)).flatten())[0]
        
        IP = np.zeros((len(non_zero_indices), (2*nosc)**2))
        
        for i, val in enumerate(non_zero_indices):
            IP[i, val] = 1
        
        F3 = [np.array([IP @ MSsolver.ham_zz(*np.split(y, 2)).flatten() for y in MSsolver.y]) for MSsolver in MSsolvers]
        
        F3 = np.hstack([F3[i][np.unique(np.random.randint(0,MSsolvers[i].n,int(MSsolvers[i].n)))].T for i in range(len(MSsolvers))])
        print(F3.shape)
        
        Uj, sv, mj = POD(F3, np.eye(F3.shape[0]), MechSystem.tol)
        del F3
        # _J = [Uj[:,i].reshape(2*nosc, -1) for i in range(mj)]
        # Jn = [RB.T @ _J[i] @ RB for i in range(mj)]
            
        fig, ax = logplot(sv, xlabel='index of singular values', xlims=(0, len(sv)))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        filename = MechSystem.keep_time +'osc_mdeim_sv' + '.pdf'
        # save_figure(fig, filename)
        
        Pj, _ = DEIM(Uj, plot_deim=False)
        # Sj = Pj.T @ RB
        
        ham_zz_col = Pj.T @ IP @ ham_zz_expr.reshape((2*nosc)**2, 1)
        
        Pxham_zz_ = smp.lambdify((x, u, omega2, beta), ham_zz_col)
        
        kwds.update({'ham_zz_': Pxham_zz_, \
                    'Uj': Uj, \
                    'Pj': Pj, \
                    'IP': IP, \
                    })
                    
    if MechSystem.constraints_reduce:
        
        F4_0 = [np.array([MSsolver.g_prime(y)[0] for y in MSsolver.y]) for MSsolver in MSsolvers]
        F4_1 = [np.array([MSsolver.g_prime(y)[1] for y in MSsolver.y]) for MSsolver in MSsolvers]
        
        F4_0 = np.hstack([F4_0[i][np.unique(np.random.randint(0,MSsolvers[i].n,int(MSsolvers[i].n)))].T for i in range(len(MSsolvers))])
        print(F4_0.shape)
        
        Ugp0, sv_gp0, mgp0 = POD(F4_0, np.eye(F4_0.shape[0]), MechSystem.tol)
        del F4_0
            
        fig, ax = logplot(sv_gp0, xlabel='index of singular values', xlims=(0, len(sv_gp0)))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        filename = MechSystem.keep_time +'osc_mdeim_sv' + '.pdf'
        # save_figure(fig, filename)
        
        Pgp0, _ = DEIM(Ugp0, plot_deim=False)
        Pgp0xgp0 = smp.lambdify((y,), Pgp0.T @ g_prime_expr[0,:].T)
        # Sj = Pj.T @ RB
        
        F4_1= np.hstack([F4_1[i][np.unique(np.random.randint(0,MSsolvers[i].n,int(MSsolvers[i].n)))].T for i in range(len(MSsolvers))])
        print(F4_1.shape)
        
        Ugp1, sv_gp1, mgp1 = POD(F4_1, np.eye(F4_1.shape[0]), MechSystem.tol)
        del F4_1
            
        fig, ax = logplot(sv_gp1, xlabel='index of singular values', xlims=(0, len(sv_gp1)))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        filename = MechSystem.keep_time +'osc_mdeim_sv' + '.pdf'
        # save_figure(fig, filename)
        
        Pgp1, _ = DEIM(Ugp1, plot_deim=False)
        Pgp1xgp1 = smp.lambdify((y,), Pgp1.T @ g_prime_expr[1,:].T)
        # Sj = Pj.T @ RB
            
        _UxPxU_inv_gp0 = Ugp0 @ LA.inv(Pgp0.T @ Ugp0)
            
        _UxPxU_inv_gp1 = Ugp1 @ LA.inv(Pgp1.T @ Ugp1)
        
        kwds.update({'hat_gp0': lambda y: _UxPxU_inv_gp0 @ Pgp0xgp0(y), \
                     'hat_gp1': lambda y: _UxPxU_inv_gp1 @ Pgp1xgp1(y), \
                    })        
                
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
    
    print('time lapsed')    
    print(time_lapsed)
    
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