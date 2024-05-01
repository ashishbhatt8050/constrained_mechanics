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
import scipy as sp
from numpy import linalg as LA
from pylab import  log, r_, c_, zeros, eye, sqrt, reshape, linspace, roll, figure, argmin, norm, zeros_like
import ODESolver
from System import System
from podDEIM import orthogonalize, POD, DEIM
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
    tol, M, var, store = 1.0E-12, 100, True, False
    
    "System parameters"
    nosc = 100
        
    # These two properties only have effect during reduction
    reducer = 'psd'
    predict = False # False = reproduce
    
    registered_solver_classes = [ODESolver.ConformalImplicitMidpoint, ODESolver.ConformalStormerVerlet]
    dt_space_dim = 5
    Omega2_space_dim = 1
    beta = (max(1e-2, 0*np.random.rand()/10))*0
    JJ = lambda self, d=nosc: r_[c_[zeros((d,d)), eye(d)], c_[-eye(d), zeros((d,d))]]
    
    Omega2_space = np.sort(2*(1 -np.random.rand(Omega2_space_dim, nosc)))
        
    dt_space = linspace(0.01, 0.05, num=dt_space_dim)
    T_final = 2
    
    osc_idx = np.array([0, 1, 2, nosc//2-1, nosc//2, nosc//2+1, nosc-3, nosc-2, nosc-1])
    keep_time = datetime.now().strftime('%Y-%m-%d_%H-%M_')
    
    "Initial conditions"
    y_init = r_[np.linspace(1,5,nosc), np.zeros(nosc)]
    
    "Initial condition"
    constraint_type = None #'spherical'
    
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

    "Constraints"    
    if constraint_type == 'linear':
        g = lambda self, y, alpha=alpha: r_[y[:, :nosc].dot(alpha), y[:, nosc:].dot(alpha)]
        g_prime = lambda self, y, alpha=alpha: r_[c_[np.array(alpha, ndmin=2), zeros((1,nosc))],\
                                                    c_[zeros((1,nosc)), np.array(alpha, ndmin=2)]]
        raise NotImplementedError
        
    elif constraint_type == 'spherical':
        Alpha = np.diag(alpha)
        A_mat = r_[c_[Alpha, np.zeros_like(Alpha)], c_[np.zeros_like(Alpha), np.zeros_like(Alpha)]]
        B_mat = r_[c_[np.zeros_like(Alpha), Alpha], c_[Alpha, np.zeros_like(Alpha)]]
            
        g = lambda self, y, alpha=[A_mat, B_mat]: r_[y @ alpha[0] @ y.T -1.0,\
                                                      y @ alpha[1] @ y.T]
        g_prime = lambda self, y, alpha=[A_mat, B_mat]: c_[2*y @alpha[0],\
                                                            2*y @alpha[1]].T
    
    
    drag = lambda self, x, u: beta/2 * r_[x, u]
    drag_z = lambda self, x, u: beta/2 * eye(2*x.shape[0])
            
    def __init__(self, kwds):
        "MechSystem properties"
        
        System.__init__(self, kwds)
        
        "Projection matrices"
        if hasattr(self, 'W_r'):
            self.y_init = self.W_r.T @ self.y_init
            
            if self.reducer == 'pod':
                self.JJ_r = self.W_r.T @ self.JJ() @ self.W_r
            elif self.reducer == 'psd':
                self.JJ_r = self.JJ(self.nosc_r)
        else:
            self.W_r = None
            self.RB = None
            
        "hyper-reduction"
        if hasattr(self, 'P'):
            # self.RBxU = self.RB.T @ self.U
            # self.UxRB = self.RBxU.T
            # self.UxP = self.PxU.T
            # self.PxP = self.P.T @ self.P
            
            self.PxU = self.P.T @ self.U
            
            if self.P.shape == self.U.shape: #POD-DEIM
                sys_solve = LA.solve
            else: #Gappy-POD
                sys_solve = lambda A, b: LA.lstsq(A, b, rcond=None)[0]
                
            self.hat = lambda foRBxy: sys_solve(self.PxU, foRBxy)
            self.hhat = lambda ffoRBxy: self.hat(self.hat(ffoRBxy.T).T)
            
            # self.PxIPxRB = lambda y: self.PxP @ sys_solve(self.UxP, self.UxRB @ y)
            # self.PxIPxRBxI = self.PxP @ sys_solve(self.UxP, self.UxRB)
            # self.PxIPxRBxP = lambda y: self.PxP @ sys_solve(self.UxP, self.UxRB @ y) @self.P.T
            # self.IPxV = lambda y: self.U @ sys_solve(self.PxU, self.PxU @ y)
            # self.IPt = lambda y: self.P @ sys_solve(self.UxP, self.U.T @y)
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
            
            if MechSystem.dt_space_dim > 1 and MechSystem.Omega2_space_dim == 1:
                 # compute errors only for fixed Omega2_space of length 1
                if (not MSsolver.beta) and (MSsolver.constraint_type is None):
                    "Energy (Hamiltonian) is an invariant for unconstrained conservative system"
                    MSsolver.eng_error = MSsolver.get_en_err()
                    errors['energy'].append(sqrt(dt)*LA.norm(MSsolver.eng_error))
                
            if MSsolver.var:
                MSsolver.var_solve()
                # errors['spl'].append(sqrt(dt)*LA.norm(MSsolver.sym_error))
                
            if not hasattr(MechSystem, 'RB'):
                #TODO: try to do without list comprehension
                MSsolver.F2 = np.array([MSsolver.ham_z(*np.split(y, 2)) for y in MSsolver.y])
                    
                if MSsolver.non_quad:
                    MSsolver.F3 = np.array([MSsolver.non_quad_z(*np.split(y, 2)) for y in MSsolver.info]).T
                
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
            
            if self.constraint_type:
                X_U = np.array([self.y_init, self.y_init])
                X_U, _, self.g_arr[0] = fixed_point(self.g, X_U @ self.RB.T, self.g_prime, self.tol, self.M, False)
                self.y_init = self.RB.T @X_U[1]
            
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
                    
                    if self.RB is None:
                        pass
                    else:
                        y_ = y_ @ self.RB.T
                    
                    y_, _, self.g_arr[k+1] = fixed_point(self.g, y_, self.g_prime, self.tol, self.M, False)

            if self.RB is None or self.constraint_type is None:
                self.y[k+1] = y_[1]
            else:
                self.y[k+1] = self.RB.T @y_[1]
            
        if self.store:
            self.info.append(self.y[k+1])
            self.info = np.vstack(self.info)

        if self.RB is not None:
            self.y_red = self.y
            self.y = self.y @ self.RB.T

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
            

#%%    "Find symbolic quantities"

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

ham_z_ = smp.lambdify((x, u, omega2, beta), ham_z_expr)
# self.ham_z = lambda x, u, omega2=self.omega2, beta=self.beta: ham_z_(x, u, omega2, beta).squeeze()
# print(ham_z(u_arr, v_arr, Om2_arr, 0.1))

ham_zz_expr = ham_z_lam(x, u, omega2, beta).jacobian(list(x)+list(u))
# ham_zz_lam = smp.lambdify((x, u, omega2, beta), ham_zz_expr, "sympy")
# print(ham_zz_lam(x, u, omega2, beta))

ham_zz_ = smp.lambdify((x, u, omega2, beta), ham_zz_expr)
# self.ham_zz = lambda x, u, omega2=self.omega2, beta=self.beta: ham_zz_(x, u, omega2, beta).squeeze()

#%% Main driver
if __name__ == '__main__':
    "Model order reduction of the MechSystem using MechSystem"
        
#%%    "Full order solution"

    nosc = MechSystem.nosc
    
    # ham_, ham_z_, ham_zz_ = get_symbols(nosc)

    kwds = {'ham_': ham_, \
            'ham_z_': ham_z_, \
            'ham_zz_': ham_zz_, \
            }

    MSsolvers = MechSystem.solver(kwds)
    
    if MSsolvers[-1].non_quad: # is not None
        assert MechSystem.Omega2_space_dim == 1
        
    if MechSystem.predict:
        array_shape = (len(MechSystem.registered_solver_classes), MechSystem.dt_space_dim, MechSystem.Omega2_space_dim-1)
    else:
        array_shape = (len(MechSystem.registered_solver_classes), MechSystem.dt_space_dim, MechSystem.Omega2_space_dim)        
    
    time_lapsed = [reshape([x.time_lapsed for x in MSsolvers], array_shape)]
    
#%% Reduced order solution"
    print('Assembling snapshots ...')
    y_list = np.hstack([MSsolver.y[np.unique(np.random.randint(0,MSsolver.n,int(MSsolver.n)))].T for MSsolver in MSsolvers])
    print(y_list.shape)
    F2 = np.hstack([MSsolver.F2[np.unique(np.random.randint(0,MSsolver.n,int(MSsolver.n)))].T for MSsolver in MSsolvers])
    print(F2.shape)
    
    X = {'Q': MSsolvers[-1].Q_spd(), \
         'sqrt': sp.linalg.sqrtm(MSsolvers[-1].Q_spd()), \
         'eye': np.eye(2*nosc), \
         'eye_half': np.eye(nosc)}
        
    if MSsolvers[-1].non_quad:
        np.linalg.cholesky(X['Q'])
        RB, s = POD(X['sqrt'] @ y_list, X['eye'])
        RB = LA.solve(X['sqrt'], RB)
    
        nosc_r = argmin(abs(np.asarray([norm(s[:i])/norm(s) for i in range(len(s))]) -0.99))
        if nosc_r % 20 == 1:
            nosc_r = nosc_r + 1  # ensure nosc_r is even
            
        X.update({'eye_r': np.eye(2*nosc_r)})
            
        print('nosc_r = %s' %nosc_r)
        
        RB = RB[:, :2*nosc_r]
    
        # Orthogonalize RB wrt Q_spd
        # U = sp.linalg.cholesky(RB.T @ X['Q'] @ RB) # upper triangular Cholesky factor
        # RB = RB @ sp.linalg.solve(U, X['eye_r'])
        assert np.allclose(RB.T @ X['Q'] @ RB, X['eye_r'])
        
        W_r_, s = POD(F2, X['eye'])
        # W_r_ = LA.solve(X['sqrt'], W_r_)
        W_r_ = W_r_[:, :2*nosc_r]
        # W_r = RB
        
        # Orthognalize W_r_ wrt RB
        W_r = orthogonalize(W_r_, RB, X['eye'])
        assert (W_r.shape == RB.shape)
    
    elif MechSystem.reducer == 'pod':
        RBq, sv_pod_q = POD(c_[y_list[:nosc,:], F2[:nosc,:]], X['eye_half'])
        RBp, sv_pod_p = POD(c_[y_list[nosc:,:], F2[nosc:,:]], X['eye_half'])
    
        nosc_r = max(\
                     argmin(abs(np.asarray([norm(sv_pod_q[:i])/norm(sv_pod_q) for i in range(len(sv_pod_q))]) -0.99)),\
                     argmin(abs(np.asarray([norm(sv_pod_p[:i])/norm(sv_pod_p) for i in range(len(sv_pod_p))]) -0.99)),\
                    )
            
        if nosc_r <  20:
            nosc_r = 40  # ensure nosc_r is even
    
        X.update({'eye_r': np.eye(2*nosc_r)})
            
        print('nosc_r = %s' %nosc_r)
        
        RBq = RBq[:, :nosc_r]
        RBp = RBp[:, :nosc_r]
        RB = r_[c_[RBq, zeros_like(RBq)],\
                c_[zeros_like(RBp), RBp]]
        W_r = RB
        
        s = c_[sv_pod_q, sv_pod_p].T
        
    elif MechSystem.reducer == 'psd':
        RB, sv_sp = POD(c_[y_list[:nosc,:], y_list[nosc:,:], F2[:nosc, :], F2[nosc:,:]], X['eye_half'])
    
        nosc_r = argmin(abs(np.asarray([norm(sv_sp[:i]**2)/norm(sv_sp**2) for i in range(len(sv_sp))]) -1 +MechSystem.tol))
        if nosc_r %  2 == 1:
            nosc_r = nosc_r + 1  # ensure nosc_r is even
    
        X.update({'eye_r': np.eye(2*nosc_r)})
            
        print('2*nosc_r = %s' %(2*nosc_r))
        
        RB = RB[:, :nosc_r]
        RB = r_[c_[RB, zeros_like(RB)],\
                c_[zeros_like(RB), RB]]
        W_r = RB
        
    fig, ax = logplot(sv_sp, xlabel='index of singular values', xlims=(0, len(sv_sp)))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    filename = MechSystem.keep_time +'osc_sv' + '.pdf'
    # save_figure(fig, filename)
    
    # assert that W_r and RB are orthogonal
    M = W_r.T @ RB
    assert np.allclose(M, X['eye_r'])
            
    del y_list, F2
    gc.collect()
    
    MechSystem.RB = RB
    MechSystem.W_r = W_r
    MechSystem.nosc_r = nosc_r
        
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

    # if MSsolvers[-1].non_quad:
    #     F3 = np.hstack([MSsolver.F3 for MSsolver in MSsolvers])
    #     noise = np.random.normal(0, 0, F3.shape)
    #     U, s = POD(X['sqrt'] @(F3+noise), X['eye'])
    #     U_ = LA.solve(X['sqrt'], U)
    #     U = U_[:, :2*nosc_r]
    #     assert np.allclose(U.T @ X['Q'] @ U, X['eye_r'])
    # else:
    #     noise = np.random.normal(0, 0, F2.shape)
    #     U_, s = POD(F2+noise, X['eye'])
    #     U = U_[:, :2*nosc_r]
        
    # ax.semilogy(s)
        
    P, _ = DEIM(RB, plot_deim=False)
    
    # P = P[:, :2*nosc_r]
    
    MechSystem.U = RB
    MechSystem.P = P
        
    # ham_, ham_z_, ham_zz_ = get_symbols(MechSystem.nosc)
    
    Pxham_z_ = smp.lambdify((x, u, omega2, beta), P.T @ ham_z_expr)
    Pxham_zz_ = smp.lambdify((x, u, omega2, beta), P.T @ ham_zz_expr @ P)

    kwds = {'ham_': ham_, \
            'ham_z_': Pxham_z_, \
            'ham_zz_': Pxham_zz_, \
            }
        
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

    # What works:
        # Results are sensitive to parameters
        # Reduced model works perfectly: efficient and accurate
        # Hyper-reduced model is efficient and accurate only for the non-conservative case.
        # Structure-preserving hyper-reduced model: the solution either doesn't converge or is highly inaccurate
        # and inefficient when it converges but the symplectic error is zero.
        # Symplectic error is the same for methods of different orders.
        
    # Next steps:
        # Implement structure-preserving hyper-reduction
        
    # In case of nonlinear solver non-convergence, try the following:
        # reduce dt
        # reduce Omega2_space
        # Increase solution vectors into the snapshot matrix
        # Create a separate basis for each reproduction experiment involving
        # only solutions from that paramter space.
    '''