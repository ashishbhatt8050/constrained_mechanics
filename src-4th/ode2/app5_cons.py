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
        
        self.Omega2 = kwds['Omega2']
        Omega2 = self.Omega2
        self.Omega2_00 = lambda Omega2=Omega2: r_[Omega2, np.ones_like(Omega2)]
        
        alpha = list(linspace(0.1,0.5,num=nosc))
        alpha /= sqrt(sum(np.array(alpha)**2))
        assert np.isclose(np.array(alpha).dot(alpha), 1), "alpha**2 must be equal to 1"
        self.alpha = alpha
        self.beta = (max(1e-2, 0*np.random.rand()/10))*0
        self.f2 = lambda y: sin(y)
        self.f2_jac = lambda y: diag(cos(y))

        "MechSystem constituents"
        kin = lambda u: sum((u**2), axis=1)/2.0
        pot = lambda x: -sum(array([diag(self.f2_jac(Omega2*x[i])) for i in range(x.shape[0])]), axis=1) 
        self.ham = lambda x, u: pot(x) +kin(u) +self.beta/2 * sum(x*u, axis=1)
        # self.ham_z = lambda x, u, Omega2=Omega2: r_[Omega2*sin(Omega2*x), u] +self.beta/2 * r_[u, x]
        # self.ham_zz = lambda x, u, Omega2=Omega2: r_[c_[diag(Omega2**2*cos(Omega2*x)), zeros(x.shape*2)],\
        #                                       c_[zeros(x.shape*2), eye(x.shape[0])]] \
        #                                     + self.beta/2 * r_[c_[zeros(x.shape*2), eye(x.shape[0])],\
        #                                       c_[eye(x.shape[0]), zeros(x.shape*2)]]
                                                
        self.ham_z = lambda x, u, Omega2=Omega2: self.Omega2_00(Omega2) * r_[sin(Omega2*x), u] \
                                                +self.beta/2 * r_[u, x]
        self.ham_zz = lambda x, u, Omega2=Omega2: diag(self.Omega2_00(Omega2) * r_[cos(Omega2*x), np.ones_like(x)] *self.Omega2_00(Omega2)) \
                                            + self.beta/2 * r_[c_[zeros(x.shape*2), eye(x.shape[0])],\
                                                               c_[eye(x.shape[0]), zeros(x.shape*2)]]
            
                                                
        self.Q_spd = lambda Omega2=Omega2: self.ham_zz(0*Omega2,0*Omega2, Omega2)
        self.JJ = lambda d=nosc: r_[c_[zeros((d,d)), eye(d)],\
                                      c_[-eye(d), zeros((d,d))]]
        self.drag = lambda x, u: self.beta/2 * r_[x, u]
        self.drag_z = lambda x, u: self.beta/2 * r_[c_[eye(x.shape[0]), zeros(x.shape*2)],\
                                              c_[zeros(x.shape*2), eye(x.shape[0])]]
            
        self.non_quad = lambda x, u: self.ham(x,u) -1/2 *r_[x, u].T @ self.Q_spd() @ r_[x, u]
        self.non_quad_z = lambda x, u, Omega2=Omega2: self.ham_z(x,u,Omega2) - self.Q_spd(Omega2) @ r_[x, u]
        self.non_quad_zz = lambda x, u, Omega2=Omega2: self.ham_zz(x,u,Omega2) - self.Q_spd(Omega2)
        self.non_quad = None
        
        "Initial conditions"
        # temp = True
        # while temp:
        y_init = r_[np.linspace(1,5,nosc), 0*np.linspace(1,5,nosc)]
        # y_init = r_[1 +4*np.random.rand(nosc), np.zeros(nosc)]
        
        self.constraint_type = None
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
            
        self.y_init = y_init
        
        "Constraints"
        if self.constraint_type == 'linear':
            self._g = lambda y, alpha=alpha: r_[y[:, :nosc].dot(alpha), y[:, nosc:].dot(alpha)]
            self._g_prime = lambda y, alpha=alpha: r_[c_[np.array(alpha, ndmin=2), zeros((1,nosc))],\
                                        c_[zeros((1,nosc)), np.array(alpha, ndmin=2)]]
            # self._g_prime = lambda y: block_diag((np.array(alpha, ndmin=2), np.array(alpha, ndmin=2)), format="csr")
            raise NotImplementedError
        elif self.constraint_type == 'spherical':
            Alpha = np.diag(alpha)
            A_mat = r_[c_[Alpha, np.zeros_like(Alpha)],\
                       c_[np.zeros_like(Alpha), np.zeros_like(Alpha)]]
            B_mat = r_[c_[np.zeros_like(Alpha), Alpha],\
                       c_[Alpha, np.zeros_like(Alpha)]]
            # self._g = lambda y, alpha=Alpha: r_[np.split(y,2,axis=1)[0] @ alpha @ np.split(y,2,axis=1)[0].T -1.0,\
            #                                     2*np.split(y,2,axis=1)[1] @ alpha @ np.split(y,2,axis=1)[0].T]
            # self._g_prime = lambda y, alpha=Alpha: r_[c_[2*np.split(y,2,axis=1)[0] @alpha.T, zeros((1,alpha.shape[0]))],\
            #                               c_[2*np.split(y,2,axis=1)[1] @alpha.T, 2*np.split(y,2,axis=1)[0] @alpha.T]]
                
            # Alpha = np.diag(r_[alpha, 2*alpha])
            _g = lambda y, alpha=[A_mat, B_mat]: r_[y @ alpha[0] @ y.T -1.0,\
                                                y @ alpha[1] @ y.T]
            _g_prime = lambda y, alpha=[A_mat, B_mat]: r_[2*y @alpha[0],\
                                                      2*y @alpha[1]]
        "Projection matrices"
        if 'RB' in kwds:
            self.RB = kwds['RB']
            
            if self.constraint_type:
                A_mat_ = self.RB.T @A_mat @self.RB
                B_mat_ = self.RB.T @B_mat @self.RB
                
                self._g = lambda y, alpha=[A_mat_, B_mat_]: _g(y, alpha)
                
                self._g_prime = lambda y, alpha=[A_mat_, B_mat_]: _g_prime(y, alpha)
        else:
            self.RB = None
            
            if self.constraint_type:
                self._g = _g
                self._g_prime = _g_prime
            
        if 'W_r' in kwds:
            self.W_r = kwds['W_r']
            self.y_init = self.W_r.T @ y_init
        else:
            self.W_r = None
            
        if 'U' in kwds:
            self.U = kwds['U']
        else:
            self.U = None
            
        if 'P' in kwds:
            self.P = kwds['P']
            self.RBxU = self.RB.T @ self.U
            self.UxRB = self.RBxU.T
            self.PxU = self.P.T @ self.U
            self.UxP = self.PxU.T
            self.PxP = self.P.T @ self.P
            self.PxIPxRB = lambda y: self.PxP @ LA.solve(self.UxP, self.UxRB @ y)
            self.PxIPxRBxI = self.PxP @ LA.solve(self.UxP, self.UxRB)
            # self.PxIPxRBxP = lambda y: self.PxP @ LA.solve(self.UxP, self.UxRB @ y) @self.P.T
            self.hat = lambda foPxIPxRBxy: self.RBxU @ LA.solve(self.PxU, foPxIPxRBxy)
            self.hhat = lambda ffoPxIPxRBxy: self.hat(ffoPxIPxRBxy) @ self.PxIPxRBxI
            
            self.IP = self.U @ LA.solve(self.PxU, self.P.T)
        else:
            self.P = None
            
        "Reduced MechSystem constituents"
        if self.W_r is not None:
            self.JJ_r = self.W_r.T @ self.JJ() @ self.W_r

        
        "Numerical solver and its properties"
        self.registered_solver_classes = kwds['registered_solver_classes']
            
        self.solver_class = kwds['solver_class']
        
        self.dt = kwds['dt']
         
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
        self.tol, self.M, self.var, self.store = 1.0E-15, 100, True, False
        
        "various measures"
        self.time_lapsed = []

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
    #TODO: have the __call__ return jacobian as well.
        
    def __call__(self, y, t):
        solver_class, beta = self.solver_class, self.beta
        P, U, RB, W_r = self.P, self.U, self.RB, self.W_r
        
        if RB is None:
            x, u = np.split(y, 2)
            if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                f = self.JJ() @ self.ham_z(x, u) - self.drag(x, u)
            elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                f = self.JJ() @ self.ham_z(x, u)
                
        elif W_r is not None and P is None:
            y = RB @ y
            x, u = np.split(y, 2)
            if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                f = self.JJ_r @ RB.T @ self.ham_z(x, u) - self.W_r.T @self.drag(x, u)
            elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                f = self.JJ_r @ RB.T @ self.ham_z(x, u)
                
        elif P is not None and self.non_quad is None:
            # IP = self.IP
            # self.U @ LA.solve(self.P.T @ self.U, self.P.T)
            x_, u_ = np.split(RB @y, 2)
            # y = P @ LA.solve(U.T @ P, U.T @ RB @ y)
            
            # x, u = np.split(self.IP.T @ RB @y, 2)
            # # Omega2 = P.T @ r_[self.Omega2, self.Omega2]
            # # Omega2, _ = np.split(Omega2, 2)
            # # Omega2 = np.split(np.split(P, 2, axis=1)[0],2)[0].T @self.Omega2
            
            # if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
            #     f = self.JJ_r @ RB.T @self.IP @self.ham_z(x, u) - self.W_r.T @self.drag(x_, u_)
            # elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
            #     f = self.JJ_r @ RB.T @self.IP @self.ham_z(x, u)
            
            x, u = np.split(self.PxIPxRB(y), 2)
            
            if x.shape != self.Omega2.shape:
                Omega2 = P.T @ r_[self.Omega2, self.Omega2]
                Omega2, _ = np.split(Omega2, 2)
            else:
                Omega2 = self.Omega2
            
            if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                f = self.JJ_r @ self.hat(self.ham_z(x, u, Omega2)) - self.W_r.T @self.drag(x_, u_)
            elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                f = self.JJ_r @ self.hat(self.ham_z(x, u, Omega2))
                
        elif P is not None and self.non_quad is not None:
            # IP = self.IP
            # self.U @ LA.solve(self.P.T @ self.U, self.P.T)
            x_, u_ = np.split(RB @y, 2)
            # y = P @ LA.solve(U.T @ P, U.T @ RB @ y)
            x, u = np.split(self.PxIPxRB(y), 2)
            
            if x.shape != self.Omega2.shape:
                Omega2 = P.T @ r_[self.Omega2, self.Omega2]
                Omega2, _ = np.split(Omega2, 2)
            else:
                Omega2 = self.Omega2
            # Omega2 = np.split(np.split(P, 2, axis=1)[0],2)[0].T @self.Omega2
            
            
            if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                f = self.JJ_r @ (y + self.hat(self.non_quad_z(x, u, Omega2)))\
                    - self.W_r.T @self.drag(x_, u_)
            elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                f = self.JJ_r @ (y + self.hat(self.non_quad_z(x, u, Omega2)))
                
        else:
            raise NotImplementedError
            
        return f

    def jacobian(self, y, t, *arg):
        "Jacobian of the function f"
        solver_class, beta = self.solver_class, self.beta
        P, U, RB, W_r = self.P, self.U, self.RB, self.W_r
        
        if RB is None:
            x, u = np.split(y, 2)
            if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                dfdy = self.JJ() @ self.ham_zz(x, u) - self.drag_z(x, u)
            elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                dfdy = self.JJ() @ self.ham_zz(x, u)
                
        elif W_r is not None and P is None:
            y = RB @ y
            x, u = np.split(y, 2)
            if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                dfdy = self.JJ_r @ RB.T @ self.ham_zz(x, u) @ RB - self.W_r.T @ self.drag_z(x, u) @ RB
            elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                dfdy = self.JJ_r @ RB.T @ self.ham_zz(x, u) @ RB
                
        elif P is not None and self.non_quad is None:
            # IP = self.IP
            # y = IP.T @ RB @ y
            x_, u_ = np.split(RB @y, 2)
            # y = P @ LA.solve(U.T @ P, U.T @ RB @ y)
            
            # x, u = np.split(self.IP.T @ RB @y, 2)
            # # Omega2 = P.T @ r_[self.Omega2, self.Omega2]
            # # Omega2, _ = np.split(Omega2, 2)
            # # Omega2 = np.split(np.split(P, 2, axis=1)[0],2)[0].T @self.Omega2
            
            # if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
            #     dfdy = self.JJ_r @ RB.T @self.IP @ self.ham_zz(x, u) @self.IP.T @ RB - self.W_r.T @self.drag_z(x_, u_) @ RB
            # elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
            #     dfdy = self.JJ_r @ RB.T @self.IP @self.ham_zz(x, u) @ self.IP.T @ RB
            
            x, u = np.split(self.PxIPxRB(y), 2)
            
            if x.shape != self.Omega2.shape:
                Omega2 = P.T @ r_[self.Omega2, self.Omega2]
                Omega2, _ = np.split(Omega2, 2)
            else:
                Omega2 = self.Omega2
            # Omega2 = np.split(np.split(P, 2, axis=1)[0],2)[0].T @self.Omega2
            
            if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                dfdy = self.JJ_r @ self.hhat(self.ham_zz(x, u, Omega2)) - self.W_r.T @self.drag_z(x_, u_) @ RB
            elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                dfdy = self.JJ_r @ self.hhat(self.ham_zz(x, u, Omega2))
                
                
        elif P is not None and self.non_quad is not None:
            # IP = self.IP
            # y = IP.T @ RB @ y
            x_, u_ = np.split(RB @y, 2)
            # y = P @ LA.solve(U.T @ P, U.T @ RB @ y)
            x, u = np.split(self.PxIPxRB(y), 2)
            
            if x.shape != self.Omega2.shape:
                Omega2 = P.T @ r_[self.Omega2, self.Omega2]
                Omega2, _ = np.split(Omega2, 2)
            else:
                Omega2 = self.Omega2
            # Omega2 = np.split(np.split(P, 2, axis=1)[0],2)[0].T @self.Omega2
            
            if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                dfdy = self.JJ_r @ (np.eye(self.JJ_r.shape[0]) + self.hhat(self.non_quad_zz(x, u, Omega2)))\
                    - self.W_r.T @self.drag_z(x_, u_)@ RB
            elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                dfdy = self.JJ_r @ (np.eye(self.JJ_r.shape[0]) + self.hhat(self.non_quad_zz(x, u, Omega2)))
                
        else:
            raise NotImplementedError
            
        return dfdy

    def en_err(self):
        "Energy error"
        assert self.beta == 0, 'Energy is only defined for beta = 0'
        
        
        # # TODO: define energy for hyper-reduced variables
        # if self.P is not None:
        #     x, u = np.split((self.IP.T @self.y.T).T, 2, axis=1)
        # else:
            
        x, u = np.split(self.y, 2, axis=1)
        
        return log(self.ham(x, u)/self.ham(x[0:1, :], u[None, 0, :]))

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
            self.y = np.zeros((self.n+1, self.RB.shape[1]))
            
            if self.constraint_type:
                X_U = np.array([self.y_init, self.y_init])
                X_U, _, _ = fixed_point(self._g, X_U, self._g_prime, self.tol, self.M, self.store)
                self.y_init = X_U[1]
            
        self.y[0] = self.y_init
        if self.store: self.info = []

        for k in range(self.n):
            y_ = np.array([self.y[k], self.y[k]])
            for w_val in self.w_values:
                self.solver.set_initial_condition(y_[1])
                y_, _ = self.solver.solve(w_val*self.t_points[k:k+2])
                
                # enforce constraints
                if self.constraint_type:
                    
                    y_, _, _ = fixed_point(self._g, y_, self._g_prime, self.tol, self.M, self.store)
                    

            self.y[k+1] = y_[1]

        if self.RB is not None:
            self.y_red = self.y
            self.y = (self.RB @ self.y.T).T
            
            # if self.P:
            #     self.y = (self.IP.T @ self.y.T).T
        
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
                dpsi_, _ = self.solver.var_solve(self.y_red[k:k+2], self.t_points[k:k+2])
            else:
                dpsi_, _ = self.solver.var_solve(self.y[k:k+2], self.t_points[k:k+2])
            self.dpsi[1] = dpsi_[1]
            sym_error_ =self.solver.symplectic_error(self.dpsi, self.t_points[k:k+2])
            self.sym_error[k+1] = sym_error_[1]

    def plot(self, fig):
        "plot the results"
        
        gs = fig.add_gridspec(4, 2, hspace=1)
        ax = gs.subplots()
        if hasattr(self, 'sym_error'):
            plot_data(ax[0,0], self.t_points, self.sym_error)
        if hasattr(self, 'eng_error'):
            plot_data(ax[1,0], self.t_points, self.eng_error)
            
        if hasattr(self, '_g'):
            if hasattr(self, 'y_red'):
                temp = np.hstack([self._g(self.y_red[[i],:]) for i in range(self.n+1)])
            else:
                temp = np.hstack([self._g(self.y[[i],:]) for i in range(self.n+1)])
                
            plot_data(ax[2,0], self.t_points, temp[0])
            plot_data(ax[3,0], self.t_points, temp[1])
        ax[3,0].set_xlabel('time')
            
        # plot_data(ax[0,1], self.t_points, )
        ax3 = fig.add_subplot(gs[0:2, -1])
        ax4 = fig.add_subplot(gs[2:4, -1])
        if self.RB is not None:
            i_range = range(min(3, self.RB.shape[1]//2))
            print('irange = %s' %i_range)
        else:
            i_range = range(min(3, self.y.shape[1]//2))
            
        for i in i_range: #range(self.y.shape[0]):
            if hasattr(self, 'y_red'):
                plot_data(ax3, self.y_red[:,i], self.y_red[:,i_range[-1]+1+i])
            
            plot_data(ax4, self.y[:,i], self.y[:,self.nosc+i])
        # ax[3,1].set_xlabel('time')
        # fig.savefig('app5_err_inv.pdf', bbox_inches='tight')
        
        # for ax in fig.get_axes():
        #     ax.label_outer()
        

    def measures(self, eng_error_, sym_error_, dt_space):
        "Various measurements based on the solution"
        
        r_values = []
        C_values = []
        
        
        r_form = lambda numer, denom: (log(roll(numer, -1)/numer)/log(roll(denom, -1)/denom))[:-1]
        C_form = lambda numer, denom, r: numer[:-1]/(denom[:-1]**r)
        
        if eng_error_:
            "Compute convergence rates from the error in Energy"
            r_values.append(r_form(eng_error_,dt_space))
            C_values.append(C_form(eng_error_,dt_space,r_values[-1]))
            
        if self.solver_class == ODESolver.ImplicitMidpoint and self.beta != 0 and sym_error_:
            '''Compute convergence rate from the error in symplecticness
            Only applicable if the error is non-zero'''
            r_values.append(r_form(sym_error_,dt_space))
            C_values.append(C_form(sym_error_,dt_space,r_values[-1]))
            
        # Display convergence rates if available
        if r_values:
            temp = r_[ reshape(dt_space, (1,-1)), \
                      c_[np.reshape([float('nan')]*len(r_values), (len(r_values),1)), r_values]].T
                
            tex_table(self.solver_class.__name__, temp)
    
        # Plot the measures
        fig = figure()
        self.plot(fig)
  
#%% Main driver function
def mor_demo():
    "Model order reduction of the MechSystem using MechSystemSolver"
    
    registered_solver_classes, nosc = [ODESolver.ImplicitMidpoint], 200
        
    dt_space_dim = 3
    dt_space = linspace(0.05, 0.1, num=dt_space_dim)
            
    Omega2_space_dim = 1
    # Omega2_space = [np.linspace(1,2,nosc)]
    Omega2_space = [(1 +0*np.random.rand(nosc)/1000)**2 for i in range(Omega2_space_dim)]
    
    MSsolvers = []
    eng_error_ = []
    sym_error_ = []
    
    for solver_class, dt, Omega2 in [(x,y,z) for x in registered_solver_classes for y in dt_space for z in Omega2_space]:
        
        kwds = {'nosc': nosc, \
                'registered_solver_classes': registered_solver_classes, \
                'solver_class': solver_class, \
                'dt': dt, \
                'Omega2': Omega2}
        
        MSsolver = MechSystemSolver(**kwds)
        # MSsolver.convergence_rates()
                
        start = process_time()
        MSsolver.solve()
        end = process_time()
        MSsolver.time_lapsed.append(end-start)
        
        if dt_space.size > 1 and np.allclose(MSsolver.Omega2, Omega2_space[-1]):
             # compute errors only for fixed Omega2
            if (not MSsolver.beta) and (MSsolver.constraint_type is None):
                "Energy (Hamiltonian) is an invariant for unconstrained conservative system"
                MSsolver.eng_error =MSsolver.en_err()
                eng_error_.append(sqrt(dt)*LA.norm(MSsolver.eng_error))
                
            sym_error_.append(sqrt(dt)*LA.norm(MSsolver.sym_error))
            
        MSsolver.F2 = np.array([MSsolver.ham_z(MSsolver.y[k,:nosc], MSsolver.y[k,nosc:])\
                                                             for k in range(MSsolver.y.shape[0])]).T
            
        MSsolver.F3 = np.array([MSsolver.non_quad_z(MSsolver.y[k,:nosc],\
                                MSsolver.y[k,nosc:]) for k in range(MSsolver.y.shape[0])]).T
            
        MSsolvers.append(MSsolver)
        
    MSsolvers[-1].measures(eng_error_, sym_error_, dt_space)
    
    time_lapsed = [reshape([x.time_lapsed for x in MSsolvers],\
                           (dt_space_dim, Omega2_space_dim))]
        
    y_list = np.hstack([MSsolvers[i].y.T for i in range(len(MSsolvers))])
    F2 = np.hstack([MSsolvers[i].F2 for i in range(len(MSsolvers))])
    F3 = np.hstack([MSsolvers[i].F3 for i in range(len(MSsolvers))])
    
    X = {'Q': MSsolver.Q_spd(), \
         'sqrt': sp.linalg.sqrtm(MSsolver.Q_spd()), \
         'eye': np.eye(2*nosc)}
    # X = np.eye(2*nosc)
    # RB1, s1 = POD(c_[y_list[:nosc, :], F3[:nosc, :]], X)
    # RB2, s2 = POD(c_[y_list[-nosc:, :], F3[-nosc:, :]], X)
    # W_r_1, s_F2_1 = POD(F2[:nosc, :], X)
    # W_r_2, s_F2_2 = POD(F2[-nosc:, :], X)
    
    fig = figure()
    ax = fig.add_subplot(111)
    
    if MSsolvers[-1].non_quad:
        RB, s = POD(X['sqrt'] @ y_list, X['eye'])
        RB = LA.solve(X['sqrt'], RB)
    
        nosc_r = argmin(abs(np.asarray([norm(s[:i])/norm(s) for i in range(len(s))]) -0.99))
        if nosc_r <  10:
            nosc_r = 150  # ensure nosc_r is even
            
        if not np.isclose(nosc_r%2, 0):
            nosc_r = nosc_r+1
            
        X.update({'eye_r': np.eye(nosc_r)})
            
        print('nosc_r = %s' %nosc_r)
        # RB = RB[:, :nosc_r]
        # RB1 = RB1[:, :nosc_r//2]
        # RB2 = RB2[:, :nosc_r//2]
        # RB = r_[c_[RB1, np.zeros_like(RB1)],\
        #         c_[np.zeros_like(RB2), RB2]]
        RB = RB[:, :nosc_r]
    
        # Orthogonalize RB wrt Q_spd
        # U = sp.linalg.cholesky(RB.T @ X['Q'] @ RB) # upper triangular Cholesky factor
        # RB = RB @ sp.linalg.solve(U, X['eye_r'])
        assert np.allclose(RB.T @ X['Q'] @ RB, X['eye_r'])
        
        # W_r1 = W_r_1[:, :nosc_r//2]
        # W_r2 = W_r_2[:, :nosc_r//2]
        # W_r_ = r_[c_[W_r1, np.zeros_like(W_r1)],\
        #         c_[np.zeros_like(W_r2), W_r2]]
        W_r_, s = POD(F2, X['eye'])
        # W_r_ = LA.solve(X['sqrt'], W_r_)
        W_r_ = W_r_[:, :nosc_r]
        ax.semilogy(s)
        # W_r = RB
        
        # Orthognalize W_r_ wrt RB
        W_r = orthogonalize(W_r_, RB, X['eye'])
        assert (W_r.shape == RB.shape)
        
        M = W_r.T @ RB
        assert (M.shape[0] == M.shape[1]) and np.allclose(M, X['eye_r'])
        # ax.semilogy(s2)
        # ax.semilogy(s_F2_1)
        # ax.semilogy(s_F2_2)
    
    else:
        RB, s = POD(c_[y_list, F2], X['eye'])
    
        nosc_r = argmin(abs(np.asarray([norm(s[:i])/norm(s) for i in range(len(s))]) -0.99))
        if nosc_r <  10:
            nosc_r = 100  # ensure nosc_r is even
            
        if not np.isclose(nosc_r%2, 0):
            nosc_r = nosc_r+1
            
        X.update({'eye_r': np.eye(nosc_r)})
            
        print('nosc_r = %s' %nosc_r)
        
        RB = RB[:, :nosc_r]
        W_r = RB
        
        # assert that W_r and RB are orthogonal
        M = W_r.T @ RB
        assert (M.shape[0] == M.shape[1]) and np.allclose(M, X['eye_r'])
            
    del y_list
    gc.collect()
    
    ax.semilogy(s)

    # del MSsolver
    # gc.collect()
    MSsolvers_r = []
    eng_error_ = []
    sym_error_ = []
    
    for solver_class, dt, Omega2 in [(x,y,z) for x in registered_solver_classes for y in dt_space for z in Omega2_space]:
        
        kwds = {'nosc': nosc, \
                'registered_solver_classes': registered_solver_classes, \
                'solver_class': solver_class, \
                'dt': dt, \
                'Omega2': Omega2,\
                'RB': RB,\
                'W_r': W_r}

        MSsolver_r = MechSystemSolver(**kwds)
        # MSsolver_r.convergence_rates()
                
        start = process_time()
        MSsolver_r.solve()
        end = process_time()
        MSsolver_r.time_lapsed.append(end-start)
        
        if dt_space.size > 1 and np.allclose(MSsolver_r.Omega2, Omega2_space[-1]):
             # compute errors only for fixed Omega2
            if (not MSsolver_r.beta) and (MSsolver_r.constraint_type is None):
                "Energy (Hamiltonian) is an invariant for unconstrained conservative system"
                MSsolver_r.eng_error =MSsolver_r.en_err()
                eng_error_.append(sqrt(dt)*LA.norm(MSsolver_r.eng_error))
                
            sym_error_.append(sqrt(dt)*LA.norm(MSsolver_r.sym_error))
            
        MSsolvers_r.append(MSsolver_r)
        
    MSsolvers_r[-1].measures(eng_error_, sym_error_, dt_space)
        
    print(np.amax(abs(MSsolvers[-1].y -MSsolvers_r[-1].y)))
    
    time_lapsed.append(reshape([x.time_lapsed for x in MSsolvers_r],\
                               (dt_space_dim, Omega2_space_dim)))
    time_lapsed[1] = (time_lapsed[1]/time_lapsed[0]*100)
    
    '''
    time_lapsed.append(nan*time_lapsed[0])
    '''
    

    # Hyper-reduced model
    
    if MSsolvers[-1].non_quad:
        U, s = POD(X['sqrt'] @F3, X['eye'])
        U = LA.solve(X['sqrt'], U)
        U = U[:, :nosc_r]
    
        # Orthogonalize U wrt Q_spd
        # U_ = sp.linalg.cholesky(U.T @ X['Q'] @ U) # upper triangular Cholesky factor
        # U = U @ sp.linalg.solve(U_, X['eye_r'])
        assert np.allclose(U.T @ X['Q'] @ U, X['eye_r'])
    else:
        U, s = POD(F2, X['eye'])
        U = U[:, :nosc_r]
        
    
    
    # fig = figure()
    # ax.semilogy(s1)
    ax.semilogy(s)
        
    P, _ = DEIM(U, plot_deim=False)
    
    
    # POD-DEIM reduced model
    MSsolvers_dr = []
    eng_error_ = []
    sym_error_ = []
    
    for solver_class, dt, Omega2 in [(x,y,z) for x in registered_solver_classes for y in dt_space for z in Omega2_space]:
        
        kwds = {'nosc': nosc, \
                'registered_solver_classes': registered_solver_classes, \
                'solver_class': solver_class, \
                'dt': dt, \
                'Omega2': Omega2,\
                'RB': RB,\
                'W_r': W_r,\
                'U': U,\
                'P': P}
            
        MSsolver_dr = MechSystemSolver(**kwds)
        # MSsolver_dr.convergence_rates()
                
        start = process_time()
        MSsolver_dr.solve()
        end = process_time()
        MSsolver_dr.time_lapsed.append(end-start)
        
        if dt_space.size > 1 and np.allclose(MSsolver_dr.Omega2, Omega2_space[-1]):
             # compute errors only for fixed Omega2
            if (not MSsolver_dr.beta) and (MSsolver_dr.constraint_type is None):
                "Energy (Hamiltonian) is an invariant for unconstrained conservative system"
                MSsolver_dr.eng_error =MSsolver_dr.en_err()
                eng_error_.append(sqrt(dt)*LA.norm(MSsolver_dr.eng_error))
                
            sym_error_.append(sqrt(dt)*LA.norm(MSsolver_dr.sym_error))
            
        MSsolvers_dr.append(MSsolver_dr)
        
    if MSsolvers_dr[-1].non_quad: # is not None
        assert Omega2_space_dim == 1
        
    MSsolvers_dr[-1].measures(eng_error_, sym_error_, dt_space)
        
    print(np.amax(abs(MSsolvers[-1].y -MSsolvers_dr[-1].y)))
    
    time_lapsed.append(reshape([x.time_lapsed for x in MSsolvers_dr],\
                               (dt_space_dim, Omega2_space_dim)))
    time_lapsed[2] = (time_lapsed[2]/time_lapsed[0]*100)
    
    # tex_table('', time_lapsed)
    print(time_lapsed)

    # What works:
        # Results are sensitive to parameters
        # Reduced model works perfectly: efficient and accurate
        # Hyper-reduced model is efficient and accurate only for the non-conservative case.
        # Structure-preserving hyper-reduced model: the solution either doesn't converge or is highly inaccurate
        # and inefficient when it converges but the symplectic error is zero.
        
    # Next steps:
        # Q-orthogonality of V and V'W=I cannot be maintained simultaneously.
        # Implement structure-preserving hyper-reduction
        
    # TODOs:
        # Pass Omega as a parameter
        # Update H for DEIM-reduced system
        # Simulate with a simpler Hamiltonian
        # figure out if the weighted projection matrix RB is computed correctly
        
if __name__ == '__main__':
    mor_demo()
    