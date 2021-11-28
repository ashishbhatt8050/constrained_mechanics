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
from podDEIM import rb_svd, DEIM
from PlotScript import plot_data, tex_table
from timeit import default_timer as timer
from scipy import optimize
from scipy.sparse import block_diag, identity, bmat, diags

from Newton import fixed_point

#%% Parameters class
class Params(object):
    """ Class of parameters """
    def __init__(self, **kwds):
        "Oscillator properties"
        self.nosc = nosc = kwds['nosc']
            
        self.Omega2 = linspace(1,1.5,num=nosc)**2
        alpha = list(linspace(0.1,0.5,num=nosc))
        alpha /= sqrt(sum(np.array(alpha)**2))
        assert np.isclose(np.array(alpha).dot(alpha), 1), "alpha**2 must be equal to 1"
        self.alpha = alpha
        self.beta = 0.1
        self.constraint_type = 'spherical'
        self.f2 = lambda y: sin(y)
        self.f2_jac = lambda y: cos(y)
        
        "Initial conditions"
        y_init = r_[linspace(1,5,num=nosc), np.zeros(nosc)]
        
        if self.constraint_type == 'linear':
            y_init[nosc-1] = -(y_init[:nosc-1].dot(alpha[:nosc-1]))/alpha[nosc-1] # project on the manifold
            assert np.isclose(y_init[:nosc].dot(alpha[:nosc]), 0), "alpha:y should be 0" 
        elif self.constraint_type == 'spherical':
            y_init /= sqrt((y_init[:nosc]**2).dot(alpha[:nosc])) # project on the manifold
            assert np.isclose((y_init[:nosc]**2).dot(alpha[:nosc]), 1), "alpha:y^2 should be 1" 
            
        self.y_init = y_init
        
        "Constraints"
        if self.constraint_type == 'linear':
            self._g = lambda y: r_[y[:, :nosc].dot(alpha), y[:, nosc:].dot(alpha)]
            # self._g_prime = lambda y: r_[c_[np.array(alpha, ndmin=2), zeros((1,nosc))],\
            #                             c_[zeros((1,nosc)), np.array(alpha, ndmin=2)]]
            self._g_prime = lambda y: block_diag((np.array(alpha, ndmin=2), np.array(alpha, ndmin=2)), format="csr")
        else:
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
            
        if 'U' in kwds:
            self.U = kwds['U']
        else:
            self.U = None
            
        if 'P' in kwds:
            self.P = kwds['P']
        else:
            self.P = None
        
        "Numerical solver ant its properties"
        self.registered_solver_classes = kwds['registered_solver_classes']
            
        self.solver_class = kwds['solver_class']
        
        self.dt_space = linspace(0.05, 0.1, num=5)
        
        self.dt = None
         
        self.T_final = 2
        
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
        P, U, RB = self.P, self.U, self.RB
        
        if RB is not None:
            y = RB.T @ y
            
        x, u = np.split(y, 2)
        
        if P is not None:
            f_hat = U @ LA.solve(P.T @ U, f2(P.T @ (Omega2 * x)))
        else:
            f_hat = f2(Omega2 * x)

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
            
        if RB is not None:
            return RB @ f
        else:
            return f

    def jacobian(self, y, t, *arg):
        # TODO: what is purpose of dt=0?
        "Jacobian of the function f"
        Omega2, beta, f2_jac = self.Omega2, self.beta, self.f2_jac
        P, U, RB = self.P, self.U, self.RB
        
        if RB is not None: y = RB.T @ y
            
        x, u = np.split(y, 2)
        nosc = len(x)
        
        if P is not None:
            f_hat_jac = U @ LA.solve(P.T @ U, f2_jac(P.T @ (Omega2 * x)))
        else:
            f_hat_jac = f2_jac(Omega2 * x)
        
        # TODO: define sparse dfdy
        eye_nosc = identity(nosc, format='csr')
        if self.solver_class in [ODESolver.ConformalImplicitMidpoint]:
            dfdy = np.concatenate([np.concatenate([beta/2*eye(nosc), eye(nosc)], axis=1), \
                                    np.concatenate([-diag(Omega2*Omega2*f_hat_jac), -beta/2*eye(nosc)], axis=1)])
            # dfdy = bmat([[beta/2*eye_nosc, eye_nosc], [-diags(Omega2*Omega2*f_hat_jac), -beta/2*eye_nosc]], format='csr')
        elif self.solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
            dfdy = r_[c_[zeros((nosc, nosc)),             eye(nosc)], \
                      c_[-diag(Omega2*Omega2*f_hat_jac), -beta*eye(nosc)]]
            # dfdy = bmat([[None, eye_nosc], [-diags(Omega2*Omega2*f_hat_jac), -beta*eye_nosc]], format='csr')
        else:
            NameError('Jacobian undefined for the solver_class - %s' % self.solver_class)
            
        if RB is not None:
            return RB @ dfdy @ RB.T
        else:
            return dfdy

    def en_err(self):
        "Energy error"
        assert self.beta == 0, 'Energy is only defined for beta = 0'

        nosc = self.nosc
        # TODO: use split() instead
        x, u = np.split(self.y, 2, axis=1)

        T = lambda u: sum((u**2), axis=1)/2.0
        V = lambda x: -sum(self.f2_jac(Omega2*x), axis=1)
        E = lambda x, u: V(x) +T(u)
        return log(E(x, u)/E(x[0:1, :], u[None, 0, :]))

#%% Solve the system and find convergence rates
class MechSystemSolver(MechSystem):
    """
    Class for solving problems of the class MechSystem
    """
        
    def solve(self):
        nosc = self.nosc
            
        self.n = int(round(self.T_final/self.dt))

        self.t_points = linspace(0, self.T_final, self.n+1)
        # TODO: remove F
        
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
                
                if self.RB is None:
                    X_U = y_
                else:
                    X_U = y_.dot(self.RB)
                    
                # TODO: move to newton.py
                
                if self.constraint_type in ['spherical','linear']:
                    start = timer()
                    X_U, _, _ = fixed_point(self._g, X_U, self._g_prime, self.tol, self.M, self.store)
                    end = timer()
                else:
                    raise NotImplementedError
                
                y_[1] = X_U[1] if self.RB is None else X_U[1].dot(self.RB.T)

            self.y[k+1] = y_[1]

        if self.RB is not None: self.y = self.y.dot(self.RB)
        
        if self.var == True:
            # TODO: why set variables off
            RB, self.RB = self.RB, None # temporarily set RB to None
            P, self.P = self.P, None # temporarily set P to None
            self.var_solve()
            self.RB = RB # Turn RB back on
            self.P = P

        self.measures()

    def var_solve(self):
        if not self.var:
            warnings.warn('Variatinoal equation solve is set to False')
            return
        else:
            nosc = self.nosc
            # TODO: make this sparse
            self.dpsi = np.zeros((self.n+1, 2*nosc, 2*nosc))
            self.dpsi[0] = np.eye(2*nosc)

            for k in range(self.n):
                dpsi_, tp = self.solver.var_solve(self.y[k:k+2], self.t_points[k:k+2])
                self.dpsi[k+1] = dpsi_[1]
                # TODO: evaluate sym_error here and loose dpsi

    def measures(self):
        "Various measurements based on the solution"
        if self.var:
            self.sym_error =self.solver.symplectic_error(self.dpsi, self.t_points)

        if (not self.beta) and (self.constraint_type is None):
            "Energy (Hamiltonian) is an invariant for unconstrained conservative system"
            # TODO: no need to pass the arguments to inherited class methods
            self.eng_error =self.en_err()

    def plot(self, fig):
        # TODO: remove unnecessary input arguments
        "plot the results"
        
        gs = fig.add_gridspec(4, 2, hspace=1)
        ax = gs.subplots(sharex=True)
        if hasattr(self, 'sym_error_'):
            plot_data(ax[0,0], self.t_points, self.sym_error)
        if self.eng_error_:
            plot_data(ax[1,0], self.t_points, self.eng_error)
            
        plot_data(ax[2,0], self.t_points, self._g(self.y).reshape(2,-1)[0])
        plot_data(ax[3,0], self.t_points, self._g(self.y).reshape(2,-1)[1])
        ax[3,0].set_xlabel('time')
        plot_data(ax[0,1], self.t_points, self.y[:,:self.nosc])
        plot_data(ax[1,1], self.t_points, self.y[:,self.nosc:])
        ax[3,1].set_xlabel('time')
        # fig.savefig('app5_err_inv.pdf', bbox_inches='tight')
        
        # for ax in fig.get_axes():
        #     ax.label_outer()
        
    def demo1(self):
            
        for dt in self.dt_space:            
            self.dt = dt
            start = timer()
            self.solve()
            end = timer()
            print('Execution time: %s' %(end-start))
            
            if self.dt_space.size > 1:
                if hasattr(self, 'eng_error'):
                    self.eng_error_.append(sqrt(dt)*LA.norm(self.eng_error))
                self.sym_error_.append(sqrt(dt)*LA.norm(self.sym_error))

        # Estimate Convergence rate r and coefficient C
        r_form = lambda numer, denom: (log(roll(numer, -1)/numer)/log(roll(denom, -1)/denom))[:-1]
        C_form = lambda numer, denom, r: numer[:-1]/(denom[:-1]**r)
        if self.eng_error_:
            "Compute convergence rates from the error in Energy"
            self.r_values.append(r_form(self.eng_error_,self.dt_space))
            self.C_values.append(C_form(self.eng_error_,self.dt_space,self.r_values[-1]))
            
        if self.solver_class == ODESolver.ImplicitMidpoint and self.beta != 0 and hasattr(self, 'sym_error_'):
            '''Compute convergence rate from the error in symplecticness
            Only applicable if the error is non-zero'''
            self.r_values.append(r_form(self.sym_error_,self.dt_space))
            self.C_values.append(C_form(self.sym_error_,self.dt_space,self.r_values[-1]))

        # Plot the measures
        fig = figure()
        self.plot(fig)
    
  
#%% Convergence rates
def mor_demo():
    "Model order reduction of the MechSystem using MechSystemSolver"
    # TODO: make it a class
    # Full order model
    
    registered_solver_classes, nosc = [ODESolver.ImplicitMidpoint], 100
    
    for tile, solver_class in enumerate(registered_solver_classes, start=131):
        
        kwds = {'nosc': nosc, 'registered_solver_classes': registered_solver_classes, 'solver_class': solver_class}
        
        MSsolver = MechSystemSolver(**kwds)
        MSsolver.demo1()
        
        # Display convergence rates if available
        if MSsolver.r_values:
            temp = r_[ reshape(MSsolver.dt_space, (1,-1)), \
                      c_[np.reshape([float('nan')]*len(MSsolver.r_values), (len(MSsolver.r_values),1)), MSsolver.r_values]].T
                
            tex_table(solver_class.__name__, temp)
    
        # Reuced model
        _, s, RB = rb_svd(MSsolver.y)
        fig = figure()
        ax = fig.add_subplot(111)
        ax.semilogy(s)
        
        nosc_r = 2*5 # ensure nosc_r is even
        RB = RB[:nosc_r, :]
        
        y = MSsolver.y
        
        kwds.update({'RB': RB})
    
        MSsolver_r = MechSystemSolver(**kwds)
        del MSsolver
        MSsolver_r.demo1()
        
        # Display convergence rates if available
        if MSsolver_r.r_values:
            temp = r_[ reshape(MSsolver_r.dt_space, (1,-1)), \
                      c_[np.reshape([float('nan')]*len(MSsolver_r.r_values), (len(MSsolver_r.r_values),1)), MSsolver_r.r_values]].T
                
            tex_table(solver_class.__name__, temp)
            
        print(np.amax(abs(y -MSsolver_r.y)))
        
        # Hyper-reduced model
        # TODO define F only once
        F = MSsolver_r.f2(MSsolver_r.Omega2 * np.split(MSsolver_r.y, 2, axis=1)[0])
        _, s, U = rb_svd(F)
        del F
        ax.semilogy(s)
        U = U.T
        
        fig = figure()
        ax = fig.add_subplot(111)
        ax.plot(range(U.shape[0]), U[:,:6])
        
        P, idx_list = DEIM(U, plot_deim=True)
        
        U = U[:, :nosc_r]
        P = P[:, :nosc_r]
                
        # POD-DEIM reduced model
        kwds.update({'U': U, 'P': P})
        MSsolver_dr = MechSystemSolver(**kwds)
        del MSsolver_r
        MSsolver_dr.demo1()
        
        # Display convergence rates if available
        if MSsolver_dr.r_values:
            temp = r_[ reshape(MSsolver_dr.dt_space, (1,-1)), \
                      c_[np.reshape([float('nan')]*len(MSsolver_dr.r_values), (len(MSsolver_dr.r_values),1)), MSsolver_dr.r_values]].T
                
            tex_table(solver_class.__name__, temp)
            
        print(np.amax(abs(y -MSsolver_dr.y)))
    
        # What works:
            # Results are sensitive to parameters
            # Unconstrained and conservative system give correct order and plots with Implicit midpoint
            # Method order cannot be tested with energy error for constrained and dissipative systems because energy is not preserved
            # Spherical constraints with dissipatation giver order ~3 for implicit midpoint for both full and reduced models. But not time gain.
            
        # Next steps:
            # impletement hyper-reduction

if __name__ == '__main__':
    mor_demo()
    