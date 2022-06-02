#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 11:49:50 2020

@author: ashishbhatt

This app solves constrained mechanical systems using solver
classes in the ODESolver hierarchy of methods.
"""

import numpy as np
from numpy import linalg as LA
from pylab import *
import ODESolver
from podDEIM import rb_svd, DEIM, orthogonalize, POD
from PlotScript import plot_data, tex_table
from timeit import default_timer as timer
from time import process_time
from scipy.sparse import block_diag, identity, bmat, diags
import scipy as sp
import gc

from Newton import fixed_point

#%% Parameters class
class Params(object):
    """ Class of parameters """
    def __init__(self, **kwds):
        "Oscillator properties"
        self.nosc = nosc = kwds['nosc']
            
        self.Omega2_space_dim = 1
        self.Omega2_space = [np.linspace(1,1.1,nosc)]
        # self.Omega2_space = [(1 +np.random.rand(nosc))**2 for i in range(self.Omega2_space_dim)]
        self.Omega2 = None
        alpha = list(linspace(0.1,0.5,num=nosc))
        alpha /= sqrt(sum(np.array(alpha)**2))
        assert np.isclose(np.array(alpha).dot(alpha), 1), "alpha**2 must be equal to 1"
        self.alpha = alpha
        self.beta = (max(1e-2, 0*np.random.rand()/10))*0
        self.constraint_type = None
        self.f2 = lambda y: sin(y)
        self.f2_jac = lambda y: diag(cos(y))

        "MechSystem constituents"
        kin = lambda u: sum((u**2), axis=1)/2.0
        pot = lambda x, Omega2: -sum(array([diag(self.f2_jac(Omega2*x[i])) for i in range(x.shape[0])]), axis=1) 
        # TODO: include beta in Hamiltonian
        self.ham = lambda x, u, Omega2: pot(x, Omega2) +kin(u) +self.beta/2 * sum(x*u, axis=1)
        self.ham_z = lambda x, u, Omega2: r_[Omega2*sin(Omega2*x), u] +self.beta/2 * r_[u, x]
        self.ham_zz = lambda x, u, Omega2: r_[c_[diag(Omega2**2*cos(Omega2*x)), zeros(x.shape*2)],\
                                              c_[zeros(x.shape*2), eye(x.shape[0])]] \
                                            + self.beta/2 * r_[c_[zeros(x.shape*2), eye(x.shape[0])],\
                                              c_[eye(x.shape[0]), zeros(x.shape*2)]]
        self.Q_spd = lambda Omega2: self.ham_zz(0*Omega2,0*Omega2,Omega2)
        self.JJ = lambda d=nosc: r_[c_[zeros((d,d)), eye(d)],\
                                      c_[-eye(d), zeros((d,d))]]
        self.drag = lambda x, u: self.beta/2 * r_[x, u]
        self.drag_z = lambda x, u: self.beta/2 * r_[c_[eye(x.shape[0]), zeros(x.shape*2)],\
                                              c_[zeros(x.shape*2), eye(x.shape[0])]]
        self.non_quad = lambda Q, x, u, Omega2: self.ham(x,u, Omega2) -1/2 *r_[x, u].T @ Q @ r_[x, u]
        self.non_quad_z = lambda Q, x, u, Omega2: self.ham_z(x,u, Omega2) - Q @ r_[x, u]
        self.non_quad_zz = lambda Q, x, u, Omega2: self.ham_zz(x,u, Omega2) - Q
        
        "Initial conditions"
        # temp = True
        # while temp:
        y_init = r_[np.linspace(1,5,nosc), 0*np.linspace(1,5,nosc)]
        # y_init = r_[1 +4*np.random.rand(nosc), np.zeros(nosc)]
        
        if self.constraint_type == 'linear':
            y_init[nosc-1] = -(y_init[:nosc-1].dot(alpha[:nosc-1]))/alpha[nosc-1] # project on the manifold
            assert np.isclose(y_init[:nosc].dot(alpha[:nosc]), 0), "alpha:y should be 0" 
        elif self.constraint_type == 'spherical':
            y_init /= sqrt((y_init[:nosc]**2).dot(alpha[:nosc])) # project on the manifold
            assert np.isclose((y_init[:nosc]**2).dot(alpha[:nosc]), 1), "alpha:y^2 should be 1" 
        elif self.constraint_type == None:
            pass
        else:
            raise NameError
                
            # for Omega2 in self.Omega2_space:
                
            #     if np.isclose(pot(y_init[:nosc].reshape(1,nosc), Omega2),0):
            #         print('Energy is zero')
            #         if np.array_equal(Omega2, self.Omega2_space[-1]):
            #             temp = False
            #         break
            
        self.y_init = y_init
        
        "Constraints"
        if self.constraint_type == 'linear':
            self._g = lambda y: r_[y[:, :nosc].dot(alpha), y[:, nosc:].dot(alpha)]
            # self._g_prime = lambda y: r_[c_[np.array(alpha, ndmin=2), zeros((1,nosc))],\
            #                             c_[zeros((1,nosc)), np.array(alpha, ndmin=2)]]
            self._g_prime = lambda y: block_diag((np.array(alpha, ndmin=2), np.array(alpha, ndmin=2)), format="csr")
        elif self.constraint_type == 'spherical':
            self._g = lambda y: r_[(y[:, :nosc]**2).dot(alpha) -1.0,\
                                   diag(y[:, nosc:].dot((2*y[:, :nosc]*alpha).T)).reshape(-1)]
            # self._g_prime = lambda y: r_[c_[2*y[:, :nosc]*alpha, zeros((1,nosc))],\
            #                              c_[zeros((1,nosc)), 2*y[:, :nosc]*alpha]]
            self._g_prime = lambda y: block_diag((2*y[:, :nosc]*alpha, 2*y[:, :nosc]*alpha), format="csr")

        "Projection matrices"
        if 'RB' in kwds:
            self.RB = kwds['RB']
            self.y_init = y_init.dot(self.RB.T)
        else:
            self.RB = None
            
        if 'W_r' in kwds:
            self.W_r = kwds['W_r']
        else:
            self.W_r = None
            
        if 'U' in kwds:
            self.U = kwds['U']
        else:
            self.U = None
            
        if 'P' in kwds:
            self.P = kwds['P']
        else:
            self.P = None
            
        "Reduced MechSystem constituents"
        if self.W_r is not None:
            self.JJ_r = self.W_r.T @ self.JJ() @ self.W_r
            # self.ham_r_z = lambda x, u, Omega2: self.ham_z(x, u, Omega2)
            # self.ham_r_zz = lambda x, u, Omega2: self.ham_zz(x, u, Omega2)

        
        "Numerical solver and its properties"
        self.registered_solver_classes = kwds['registered_solver_classes']
            
        self.solver_class = kwds['solver_class']
        
        self.dt_space_dim = 5
        self.dt_space = linspace(0.05, 0.1, num=self.dt_space_dim)
        
        self.dt = None
         
        self.T_final = 10
        
        w_values = [0.28, 0.62546642846767004501]
        w_values.append(1.0 -2.0*(sum(w_values)))
        w_values.append(w_values[1])
        w_values.append(w_values[0])
        w_values = [1]
        assert np.isclose(sum(w_values), 1), 'sum_i w_i must be 1'
        self.w_values = w_values
        
        if self.solver_class in [ODESolver.ConformalStormerVerlet, ODESolver.ForwardEuler]:
            self.solver = self.solver_class(self)
        elif self.solver_class in [ODESolver.ConformalImplicitMidpoint, ODESolver.ImplicitMidpoint]:
            self.solver = self.solver_class(self, self.jacobian)
        else:
            NameError('Unknown solver class - %s' % self.solver_class.__name__)

        "Fixed-point nonliner equations solver properties"
        self.tol, self.M, self.var, self.store = 1.0E-13, 100, True, False
        
        "various measures"
        self.r_values, self.C_values = [], []
        self.sym_error_, self.eng_error_ = [], []
        self.time_lapsed = []
        
        
        self.r_form = lambda numer, denom: (log(roll(numer, -1)/numer)/log(roll(denom, -1)/denom))[:-1]
        self.C_form = lambda numer, denom, r: numer[:-1]/(denom[:-1]**r)

    "y_init alias"
    @property
    def u_init(self):
        return self.y_init

    @u_init.setter
    def u_init(self, value):
        self.y_init = value
 
#%% Define the system
class MechSystem(Params):
    "Define the Mechanical system as a sub-class of Params"
        
    def __call__(self, y, t):
        solver_class, Omega2, beta, f2 = self.solver_class, self.Omega2, self.beta, self.f2
        P, U, RB, W_r = self.P, self.U, self.RB, self.W_r
        Q = self.Q_spd(Omega2)
        
        if RB is not None:
            x_r, u_r = np.split(y,2)
            y = RB.T @ y
            
        x, u = np.split(y, 2)
        
        if P is not None and self.W_r is None:
            f_hat = U @ LA.solve(P.T @ U, f2(P.T @ (Omega2 * x)))
        elif P is not None and self.W_r is not None:
            # IP = U @ LA.solve(P.T @  U, P.T)
            y_select = P.T @ P @ LA.solve(U.T @ P, U.T @ y)
            # y_select = IP.T @ y
            x_select, u_select = np.split(y_select, 2)
            Omega2_select = P.T @ r_[Omega2, Omega2]
            Omega2_select, _ = np.split(Omega2_select, 2)
            Q_select = P.T @ Q @ P
            ham_z = lambda *arg: RB @ Q @ y \
                +RB @ U @ LA.solve(P.T @U, self.non_quad_z(Q_select, x_select, u_select, Omega2_select))
        elif P is None and W_r is not None:
            ham_z = lambda x, u, Omega2: RB @ self.ham_z(x, u, Omega2)
        else:
            f_hat = f2(Omega2 * x)

        if self.W_r is None:
            if solver_class in [ODESolver.ConformalStormerVerlet]:
                f = [u, -Omega2 * f_hat]
            elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                f = reshape([u, -Omega2 * f_hat -beta*u], -1) +beta/2*y
            elif solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                f = r_[u, -Omega2 * f_hat -beta*u]
            elif solver_class is None:
                ValueError('Unsupported solver_class')
            else:
                NameError('Undefined solver_class - %s' % solver_class)
        else:
            if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                f = self.JJ_r @ ham_z(x, u, Omega2) - self.drag(x_r, u_r)
            elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                f = self.JJ_r @ ham_z(x, u, Omega2)
            
        if RB is not None and self.W_r is None:
            return RB @ f
        else:
            return f

    def jacobian(self, y, t, *arg):
        "Jacobian of the function f"
        Omega2, beta, f2_jac = self.Omega2, self.beta, self.f2_jac
        P, U, RB, W_r = self.P, self.U, self.RB, self.W_r
        Q = self.Q_spd(Omega2)
        
        if RB is not None:
            x_r, u_r = np.split(y,2)
            y = RB.T @ y
            
        x, u = np.split(y, 2)
        nosc = len(x)
        
        if P is not None and W_r is None:
            f_hat_jac = U @ LA.solve(P.T @ U, f2_jac(P.T @ (Omega2 * x)) @ P.T * Omega2)
        elif P is not None and self.W_r is not None:
            y_select = P.T @ P @ LA.solve(U.T @ P, U.T @ y)
            x_select, u_select = np.split(y_select, 2)
            Omega2_select = P.T @ r_[Omega2, Omega2]
            Omega2_select, _ = np.split(Omega2_select, 2)
            Q_select = P.T @ Q @ P
            ham_zz = lambda *arg: RB @ Q @ RB.T \
                +RB @ U @ LA.solve(P.T @U, self.non_quad_zz(Q_select, x_select, u_select, Omega2_select)) @ P.T @ P @ LA.solve(U.T @ P, U.T) @ RB.T
        elif P is None and W_r is not None:
            ham_zz = lambda x, u, Omega2: RB @ self.ham_zz(x, u, Omega2) @RB.T
        else:
            f_hat_jac = f2_jac(Omega2 * x) * Omega2
            
        if self.W_r is None:
            if self.solver_class in [ODESolver.ConformalImplicitMidpoint]:
                dfdy = np.concatenate([np.concatenate([beta/2*eye(nosc), eye(nosc)], axis=1), \
                                        np.concatenate([-Omega2*f_hat_jac, -beta/2*eye(nosc)], axis=1)])
                # dfdy = bmat([[beta/2*eye_nosc, eye_nosc], [-diags(Omega2*Omega2*f_hat_jac), -beta/2*eye_nosc]], format='csr')
            elif self.solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                dfdy = r_[c_[zeros((nosc, nosc)),   eye(nosc)], \
                          c_[-Omega2*f_hat_jac,     -beta*eye(nosc)]]
                # dfdy = bmat([[None, eye_nosc], [-diags(Omega2*Omega2*f_hat_jac), -beta*eye_nosc]], format='csr')
            else:
                NameError('Jacobian undefined for the solver_class - %s' % self.solver_class)
        else:
            if self.solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                dfdy = self.JJ_r @ ham_zz(x,u,Omega2) - self.drag_z(x_r, u_r)
            elif self.solver_class in [ODESolver.ConformalImplicitMidpoint]:
                dfdy = self.JJ_r @ ham_zz(x,u,Omega2)
            
        if RB is not None and self.W_r is None:
            return RB @ dfdy @ RB.T
        else:
            return dfdy

    def en_err(self):
        "Energy error"
        # TODO: define energy for hyper-reduced variables
        assert self.beta == 0, 'Energy is only defined for beta = 0'
        U, P, Omega2 = self.U, self.P, self.Omega2
        
        # if self.P is not None:
        #     y_ = P @ LA.solve(U.T @ P, U.T @ self.y.T)
        #     y = y_.T
        #     x, u = np.split(y, 2, axis=1)
        #     # Omega2 = self.P.T @ r_[self.Omega2, self.Omega2]
        #     # Omega2, _ = np.split(Omega2, 2)
        #     # ham = lambda x, u, Omega2: self.U @ LA.solve(self.P.T @ self.U, self.ham(x, u, Omega2))
        #     ham = lambda x, u, Omega2: 1/2 * self.y @ self.Q_spd(Omega2) @ self.y.T + self.non_quad(self.Q_spd(Omega2), x, u, Omega2)
        # else:
        ham = self.ham
        x, u = np.split(self.y, 2, axis=1)
        
        return log(ham(x, u, Omega2)/ham(x[0:1, :], u[None, 0, :], Omega2))

#%% Solve the system and find convergence rates
class MechSystemSolver(MechSystem):
    """
    Class for solving problems of the class MechSystem
    """
        
    def solve(self):
        nosc = self.nosc
            
        self.n = int(round(self.T_final/self.dt))

        self.t_points = linspace(0, self.T_final, self.n+1)
        
        if self.RB is None:
            self.y = np.zeros((self.n+1, 2*nosc))
        else:
            self.y = np.zeros((self.n+1, self.RB.shape[0]))
            
        self.y[0] = self.y_init
        if self.store: self.info = []

        for k in range(self.n):
            y_ = np.array([self.y[k], self.y[k]])
            for w_val in self.w_values:
                self.solver.set_initial_condition(y_[1])
                y_, tp = self.solver.solve(w_val*self.t_points[k:k+2])
                
                # Apply constraints
                if self.constraint_type in ['spherical','linear']:
                
                    if self.RB is None:
                        X_U = y_
                    else:
                        X_U = y_.dot(self.RB)
                    start = timer()
                    X_U, _, _ = fixed_point(self._g, X_U, self._g_prime, self.tol, self.M, self.store)
                    end = timer()
                    
                    y_[1] = X_U[1] if self.RB is None else X_U[1].dot(self.RB.T)
                elif self.constraint_type == None:
                    pass
                else:
                    print('Unknwon constraint type')

            self.y[k+1] = y_[1]

        if self.RB is not None:
            self.y_red = self.y
            self.y = self.y.dot(self.RB)
        
        if self.var == True:
            self.var_solve()

    def var_solve(self):
        nosc = self.nosc
        if hasattr(self, 'y_red'):
            nosc = self.y_red.shape[1]//2
            
        self.dpsi = np.zeros((2, 2*nosc, 2*nosc))
        self.dpsi[0] = np.eye(2*nosc)
        self.sym_error = np.zeros(self.n+1)

        for k in range(self.n):
            if hasattr(self, 'y_red'):
                # if hasattr(self, 'JJ_r'):
                #     self.solver.J_mat = self.JJ_r
                dpsi_, _ = self.solver.var_solve(self.y_red[k:k+2], self.t_points[k:k+2])
            else:
                dpsi_, _ = self.solver.var_solve(self.y[k:k+2], self.t_points[k:k+2])
            self.dpsi[1] = dpsi_[1]
            sym_error_ =self.solver.symplectic_error(self.dpsi, self.t_points[k:k+2])
            self.sym_error[k+1] = sym_error_[1]

    def measures(self):
        "Various measurements based on the solution"
        
        if self.eng_error_:
            "Compute convergence rates from the error in Energy"
            self.r_values.append(self.r_form(self.eng_error_,self.dt_space))
            self.C_values.append(self.C_form(self.eng_error_,self.dt_space,self.r_values[-1]))
            
        if self.solver_class == ODESolver.ImplicitMidpoint and self.beta != 0 and hasattr(self, 'sym_error_'):
            '''Compute convergence rate from the error in symplecticness
            Only applicable if the error is non-zero'''
            self.r_values.append(self.r_form(self.sym_error_,self.dt_space))
            self.C_values.append(self.C_form(self.sym_error_,self.dt_space,self.r_values[-1]))

        # Plot the measures
        fig = figure()
        self.plot(fig)

    def plot(self, fig):
        "plot the results"
        
        gs = fig.add_gridspec(4, 2, hspace=1)
        ax = gs.subplots()
        if hasattr(self, 'sym_error'):
            plot_data(ax[0,0], self.t_points, self.sym_error)
        if self.eng_error_:
            plot_data(ax[1,0], self.t_points, self.eng_error)
            
        if hasattr(self, '_g'):
            plot_data(ax[2,0], self.t_points, self._g(self.y).reshape(2,-1)[0])
            plot_data(ax[3,0], self.t_points, self._g(self.y).reshape(2,-1)[1])
        ax[3,0].set_xlabel('time')
            
        # plot_data(ax[0,1], self.t_points, )
        ax3 = fig.add_subplot(gs[:, -1])
        if self.RB is not None:
            i_range = range(self.RB.shape[0]//2 -1)
            print('irange = %s' %i_range)
        else:
            i_range = range(min(4, self.y.shape[1]//2))
            
        for i in i_range: #range(self.y.shape[0]):
            if hasattr(self, 'y_red'):
                plot_data(ax3, self.y_red[:,i], self.y_red[:,i_range[-1]+1+i])
            else:
                plot_data(ax3, self.y[:,i], self.y[:,self.nosc+i])
        # ax[3,1].set_xlabel('time')
        # fig.savefig('app5_err_inv.pdf', bbox_inches='tight')
        
        # for ax in fig.get_axes():
        #     ax.label_outer()
        
    def convergence_rates(self):
            
        if self.RB is None:
            self.y_list = self.y_init.reshape((1,len(self.y_init)))
            self.F2 = np.array([self.ham_z(self.y_list[i,:self.nosc], self.y_list[i,self.nosc:], self.Omega2_space[0]) for i in range(self.y_list.shape[0])])
            self.F3 = np.array([self.non_quad_z(self.Q_spd(self.Omega2_space[0]), self.y_list[i,:self.nosc], self.y_list[i,self.nosc:], self.Omega2_space[0]) for i in range(self.y_list.shape[0])])
        
        # else:
            # self.Omega2_selector = np.random.randint(0, self.Omega2_space_dim)
        self.Omega2_selector = self.Omega2_space_dim-1
        self.Omega2_space = [self.Omega2_space[self.Omega2_selector]]
        self.Omega2_space_dim = 1
                
        for dt, Omega2 in [(x,y) for x in self.dt_space for y in self.Omega2_space]:            
            self.dt, self.Omega2 = dt, Omega2
            start = process_time()
            self.solve()
            end = process_time()
            self.time_lapsed.append(end-start)
            
            if self.dt_space.size > 1 and np.allclose(self.Omega2, self.Omega2_space[self.Omega2_selector]):
                 # compute errors only for fixed Omega2
                if (not self.beta) and (self.constraint_type is None):
                    "Energy (Hamiltonian) is an invariant for unconstrained conservative system"
                    self.eng_error =self.en_err()
                    self.eng_error_.append(sqrt(dt)*LA.norm(self.eng_error))
                    
                self.sym_error_.append(sqrt(dt)*LA.norm(self.sym_error))
            
            if self.RB is None:
                self.y_list = np.append(self.y_list, self.y, axis=0)
                self.F2 = np.append(self.F2, np.array([self.ham_z(self.y_list[i,:self.nosc], self.y_list[i,self.nosc:], self.Omega2) for i in range(self.y_list.shape[0])]), axis=0)
                self.F3 = np.append(self.F3, np.array([self.non_quad_z(self.Q_spd(self.Omega2), self.y_list[i,:self.nosc], self.y_list[i,self.nosc:], self.Omega2) for i in range(self.y_list.shape[0])]), axis=0)
                
        self.measures()
  
#%% Main driver function
def mor_demo():
    "Model order reduction of the MechSystem using MechSystemSolver"
    
    registered_solver_classes, nosc = [ODESolver.ImplicitMidpoint, ODESolver.ConformalImplicitMidpoint], 20
    
    for solver_class in registered_solver_classes:
        
        kwds = {'nosc': nosc, 'registered_solver_classes': registered_solver_classes, 'solver_class': solver_class}
        
        MSsolver = MechSystemSolver(**kwds)
        MSsolver.convergence_rates()
        
        # Display convergence rates if available
        if MSsolver.r_values:
            temp = r_[ reshape(MSsolver.dt_space, (1,-1)), \
                      c_[np.reshape([float('nan')]*len(MSsolver.r_values), (len(MSsolver.r_values),1)), MSsolver.r_values]].T
                
            tex_table(solver_class.__name__, temp)
        
        # _, s, RB = rb_svd(MSsolver.y_list)
        # _, s_F2, W_r = rb_svd(MSsolver.F2)
        # X = MSsolver.Q_spd(MSsolver.Omega2)
        X = np.eye(2*nosc)
        RB, s = POD(MSsolver.y_list.T, X)
        RB = RB.T
        W_r, s_F2 = POD(MSsolver.F2.T, X)
        # del F2
        fig = figure()
        ax = fig.add_subplot(111)
        ax.semilogy(s)
        ax.semilogy(s_F2)
        
        nosc_r = argmin(abs(np.asarray([norm(s[:i])/norm(s) for i in range(len(s))]) -0.99))
        if nosc_r <  10:
            nosc_r = 10  # ensure nosc_r is even
            
        if not np.isclose(nosc_r%2, 0):
            nosc_r = nosc_r+1
            
        print('nosc_r = %s' %nosc_r)
        RB = RB[:nosc_r, :]
        W_r = W_r[:nosc_r, :]
        
        # Orthogonalize RB wrt Q_spd
        # U = sp.linalg.cholesky(RB @ MSsolver.Q_spd(MSsolver.Omega2) @ RB.T) # upper triangular Cholesky factor
        # RB = (RB.T @ sp.linalg.solve(U, eye(U.shape[0]))).T
        # assert np.allclose(RB @ MSsolver.Q_spd(MSsolver.Omega2) @ RB.T, np.eye(RB.shape[0]))
        
        # M = RB @ W_r.T
        # assert (M.shape[0] == M.shape[1]) and np.allclose(M, np.eye(M.shape[0]))
        
        # Orthognalize W_r wrt RB
        # W_r = orthogonalize(W_r.T, RB.T, X)
        # W_r = W_r.T
        # # W_r = (MSsolver.Q_spd(MSsolver.Omega2) @ RB.T).T
        # assert (W_r.shape == RB.shape)
        
        # M = RB @ W_r.T
        # assert (M.shape[0] == M.shape[1]) and np.allclose(M, np.eye(M.shape[0]))
        
        y = MSsolver.y
        time_lapsed = [reshape(MSsolver.time_lapsed, (MSsolver.dt_space_dim, MSsolver.Omega2_space_dim))]
        # del MSsolver
        gc.collect()
        
        kwds.update({'RB': RB, 'W_r': W_r.T})
    
        MSsolver_r = MechSystemSolver(**kwds)
        MSsolver_r.convergence_rates()
        
        # Display convergence rates if available
        if MSsolver_r.r_values:
            temp = r_[ reshape(MSsolver_r.dt_space, (1,-1)), \
                      c_[np.reshape([float('nan')]*len(MSsolver_r.r_values), (len(MSsolver_r.r_values),1)), MSsolver_r.r_values]].T
                
            tex_table(solver_class.__name__, temp)
            
        print(np.amax(abs(y -MSsolver_r.y)))
        
        time_lapsed.append(reshape(MSsolver_r.time_lapsed, (MSsolver_r.dt_space_dim, MSsolver_r.Omega2_space_dim)))
        time_lapsed[1] = (time_lapsed[1].reshape(-1)/time_lapsed[0][:,MSsolver_r.Omega2_selector]*100).reshape(MSsolver_r.dt_space_dim,1)
        
        '''# Hyper-reduced model
        # _, s, U = rb_svd(MSsolver.F3)
        U, s = POD(MSsolver.F3.T, X)
        U = U.T
        del MSsolver
        # del F3
        ax.semilogy(s)
        U = U.T
        
        # fig = figure()
        # ax = fig.add_subplot(111)
        # ax.plot(range(U.shape[0]), U[:,:nosc_r])
        
        
        P, idx_list = DEIM(U, plot_deim=False)
        
        U = U[:, :nosc_r]
        P = P[:, :nosc_r]
        
        # Orthogonalize U wrt Q_spd
        # U_ = sp.linalg.cholesky(U.T @ MSsolver_r.Q_spd(MSsolver_r.Omega2) @ U) # upper triangular Cholesky factor
        # U = U @ sp.linalg.solve(U_, eye(U_.shape[0]))
        # assert np.allclose(U.T @ MSsolver_r.Q_spd(MSsolver_r.Omega2) @ U, np.eye(U.shape[1]))
        
        
        del MSsolver_r
        gc.collect()
        
        # POD-DEIM reduced model
        kwds.update({'U': U, 'P': P, 'W_r': W_r.T})
        MSsolver_dr = MechSystemSolver(**kwds)
        MSsolver_dr.convergence_rates()
        
        # Display convergence rates if available
        if MSsolver_dr.r_values:
            temp = r_[ reshape(MSsolver_dr.dt_space, (1,-1)), \
                      c_[np.reshape([float('nan')]*len(MSsolver_dr.r_values), (len(MSsolver_dr.r_values),1)), MSsolver_dr.r_values]].T
                
            tex_table(solver_class.__name__, temp)
            
        print(np.amax(abs(y -MSsolver_dr.y)))
        
        time_lapsed.append(reshape(MSsolver_dr.time_lapsed, time_lapsed[1].shape))
        # time_lapsed = [np.average(time_lapsed[i], axis=1) for i in range(1)]
        time_lapsed[2] = (time_lapsed[2].reshape(-1)/time_lapsed[0][:,MSsolver_dr.Omega2_selector]*100).reshape(MSsolver_dr.dt_space_dim,1)
        del MSsolver_dr
        gc.collect()
        '''
        
        # tex_table('', time_lapsed)
        print(time_lapsed)
    
        # What works:
            # Results are sensitive to parameters
            # Unconstrained and conservative system give correct order and plots with Implicit midpoint
            # Method order cannot be tested with energy error for constrained and dissipative systems because energy is not preserved
            # Spherical constraints with dissipatation giver order ~3 for implicit midpoint for both full and reduced models. But not time gain.
            
        # Next steps:
            # Q-orthogonality of V and V'W=I cannot be maintained simultaneously. 
            # Implement structure-preserving reduction for unconstrained problem
            # Implement structure-preserving reduction for constrained problem
            
if __name__ == '__main__':
    mor_demo()
    