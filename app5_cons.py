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

#%% Define the system
class MechSystem(object):
    def __init__(self, Params, method=None):

        self.Omega2 = Params.Omega**2
        # TODO: make Omega2 a vector
        self.alpha = Params.alpha
        self.beta = Params.beta
        self.nosc = Params.nosc
        self.method = method
        self.u_init = Params.y_init
        self.T = Params.T_final
        self.constraint_type = Params.constraint_type
        self.f2 = Params.f2
        self.f2_jac = Params.f2_jac
        
        self.RB, self.U, self.P = Params.RB, Params.U, Params.P
            
        # TODO: make sparse arrays
        if Params.constraint_type == 'linear':
            self._g = lambda y: r_[y[:, :self.nosc].dot(self.alpha), y[:, self.nosc:].dot(self.alpha)]
            # self._g_prime = lambda y: r_[c_[np.array(self.alpha, ndmin=2), zeros((1,self.nosc))],\
            #                             c_[zeros((1,self.nosc)), np.array(self.alpha, ndmin=2)]]
            self._g_prime = lambda y: block_diag((np.array(self.alpha, ndmin=2), np.array(self.alpha, ndmin=2)), format="csr")
        else:
            self._g = lambda y: r_[(y[:, :self.nosc]**2).dot(self.alpha) -1.0,\
                                   diag(y[:, self.nosc:].dot((2*y[:, :self.nosc]*self.alpha).T)).reshape(-1)]
            # self._g_prime = lambda y: r_[c_[2*y[:, :self.nosc]*self.alpha, zeros((1,self.nosc))],\
            #                              c_[zeros((1,self.nosc)), 2*y[:, :self.nosc]*self.alpha]]
            self._g_prime = lambda y: block_diag((2*y[:, :self.nosc]*self.alpha, 2*y[:, :self.nosc]*self.alpha), format="csr")
                    
    def __call__(self, y, t):
        method, Omega2, beta, f2 = self.method, self.Omega2, self.beta, self.f2
        P, U, RB = self.P, self.U, self.RB
        
        if RB is not None:
            y = RB.T @ y
            
        x, u = np.split(y, 2)
        
        if P is not None:
            f_hat = U @ LA.solve(P.T @ U, f2(P.T @ (Omega2 * x)))
        else:
            f_hat = f2(Omega2 * x)

        if method in [ODESolver.ConformalStormerVerlet]:
            f = [u, -Omega2 * f_hat]
        elif method in [ODESolver.ConformalImplicitMidpoint]:
            f = reshape([u, -Omega2 * f_hat -beta*u], -1) +beta/2*y
        elif method in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
            f = r_[u, -Omega2 * f_hat -beta*u]
        else:
            NameError('Undefined method - %s' % method)
            
        if RB is not None:
            return RB @ f
        else:
            return f

    def jacobian(self, y, t, dt=0):
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
        if self.method in [ODESolver.ConformalImplicitMidpoint]:
            dfdy = np.concatenate([np.concatenate([beta/2*eye(nosc), eye(nosc)], axis=1), \
                                    np.concatenate([-diag(Omega2*Omega2*f_hat_jac), -beta/2*eye(nosc)], axis=1)])
            # dfdy = bmat([[beta/2*eye_nosc, eye_nosc], [-diags(Omega2*Omega2*f_hat_jac), -beta/2*eye_nosc]], format='csr')
        elif self.method in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
            dfdy = r_[c_[zeros((nosc, nosc)),             eye(nosc)], \
                      c_[-diag(Omega2*Omega2*f_hat_jac), -beta*eye(nosc)]]
            # dfdy = bmat([[None, eye_nosc], [-diags(Omega2*Omega2*f_hat_jac), -beta*eye_nosc]], format='csr')
        else:
            NameError('Jacobian undefined for the method - %s' % self.method)
            
        if RB is not None:
            return RB @ dfdy @ RB.T
        else:
            return dfdy

    def en_err(self, y, t=0):
        "Energy error"
        assert self.beta == 0, 'Energy is only defined for beta = 0'

        nosc = self.nosc
        x, u = y[:, :nosc], y[:, nosc:]
        # TODO: use split() instead
        # x, u = np.split(y, 2, axis=1)

        T = lambda u: sum((u**2), axis=1)/2.0
        V = lambda x: -sum(self.f2_jac(Omega2*x), axis=1)
        E = lambda x, u: V(x) +T(u)
        return log(E(x, u)/E(x[0:1, :], u[None, 0, :]))

#%% Solve the system and find convergence rates
class MechSystemSolver(object):
    """
    Class for solving problems of the class MechSystem
    """
    def __init__(self, Params, problem, method):
        self.f, self.dt, self.solver_class = problem, Params.dt, method
        self.w_values = Params.w_values
        self.var = Params.var
        self.store = Params.store
        self.tol, self.M = Params.tol, Params.M
        self.RB = Params.RB

        self.eng_error, self.sym_error = [],[]

        if self.solver_class in [ODESolver.ConformalStormerVerlet, ODESolver.ForwardEuler]:
            self.solver = self.solver_class(self.f)
        elif self.solver_class in [ODESolver.ConformalImplicitMidpoint, ODESolver.ImplicitMidpoint]:
            self.solver = self.solver_class(self.f, self.f.jacobian)
        else:
            NameError('Unknown solver class - %s' % method.__name__)

    def solve(self):
        nosc = self.f.nosc
            
        self.n = int(round(self.f.T/self.dt))

        self.t_points = linspace(0, self.f.T, self.n+1)
        # TODO: remove F
        
        if self.RB is None:
            self.y = np.zeros((self.n+1, 2*nosc))
        else:
            self.y = np.zeros((self.n+1, self.RB.shape[0]))
            
        self.y[0] = self.f.u_init
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
                
                if self.f.constraint_type in ['spherical','linear']:
                    
                    from Newton import fixed_point
                    start = timer()
                    X_U, _, _ = fixed_point(self.f._g, X_U, self.f._g_prime, self.tol, self.M, self.store)
                    end = timer()
                    # print('%s' %(end-start))
                    # obsolete code block
                    # m, Delta_Lambda = 0, self.f._g(X_U[1:2])/sum(self.f._g_prime(X_U[1:2])*self.f._g_prime(X_U[0:1]), axis=1)
                    
                    # if self.store: self.info.append((m, Delta_Lambda, X_U[1]))
    
                    # while max(abs(self.f._g(X_U))) > self.tol and m < self.M:
                    #     X_U[1] = X_U[1] -self.f._g_prime(X_U[0:1]).T.dot(Delta_Lambda)
    
                    #     m, Delta_Lambda = m+1, self.f._g(X_U[1:2])/sum(self.f._g_prime(X_U[1:2])*self.f._g_prime(X_U[0:1]), axis=1)
                    #     if self.store: self.info.append((m, Delta_Lambda, X_U[1]))
                    # assert m <= self.M, "Constraints not satisfied"
                        
                # elif self.f.constraint_type is ['sp.optimize.root']:
                #     X = optimize.root(self.f._g, X_U[:, :nosc], method='broyden1')
                #     U = optimize.root(lambda y: self.f._G(c_[X,y]), X_U[:, nosc:], method='broyden1')
                #     X_U = c_[X, U]
                #     raise NotImplementedError
                else:
                    raise NotImplementedError
                
                y_[1] = X_U[1] if self.RB is None else X_U[1].dot(self.RB.T)

            self.y[k+1] = y_[1]

        if self.RB is not None: self.y = self.y.dot(self.RB)
        
        if self.var == True:
            RB, self.f.RB = self.f.RB, None # temporarily set RB to None
            P, self.f.P = self.f.P, None # temporarily set RB to None
            self.var_solve()
            self.f.RB = RB # Turn RB back on
            self.f.P = P

        self.measures()

    def var_solve(self):
        if not self.var:
            warnings.warn('Variatinoal equation solve is set to False')
            return
        else:
            nosc = self.f.nosc
            self.dpsi = np.zeros((self.n+1, 2*nosc, 2*nosc))
            self.dpsi[0] = np.eye(2*nosc)

            for k in range(self.n):
                dpsi_, tp = self.solver.var_solve(self.y[k:k+2], self.t_points[k:k+2])
                self.dpsi[k+1] = dpsi_[1]

    def measures(self):
        "Various measurements based on the solution"
        if self.var:
            self.sym_error.append(self.solver.symplectic_error(self.dpsi, self.t_points))

        if (not self.f.beta) and (self.f.constraint_type is None):
            "Energy (Hamiltonian) is an invariant for unconstrained conservative system"
            self.eng_error.append(self.f.en_err(self.y))

    def plot(self, fig):
        # TODO: remove unnecessary input arguments
        "plot the results"
        
        gs = fig.add_gridspec(4, 2, hspace=1)
        ax = gs.subplots(sharex=True)
        if hasattr(self, 'sym_error'):
            plot_data(ax[0,0], self.t_points, self.sym_error[-1])
        if self.eng_error:
            plot_data(ax[1,0], self.t_points, self.eng_error[-1])
            
        plot_data(ax[2,0], self.t_points, self.f._g(self.y).reshape(2,-1)[0])
        plot_data(ax[3,0], self.t_points, self.f._g(self.y).reshape(2,-1)[1])
        ax[3,0].set_xlabel('time')
        plot_data(ax[0,1], self.t_points, self.y[:,:self.f.nosc])
        plot_data(ax[1,1], self.t_points, self.y[:,self.f.nosc:])
        ax[3,1].set_xlabel('time')
        # fig.savefig('app5_err_inv.pdf', bbox_inches='tight')
        
        # for ax in fig.get_axes():
        #     ax.label_outer()

#%% Parameters class
class Params(object):
    """ Class of parameters """
    def __init__(self, nosc=100):
        self.nosc = nosc
        self.Omega = linspace(1,1.5,num=nosc)
        alpha = list(linspace(0.1,0.5,num=nosc))
        alpha /= sqrt(sum(np.array(alpha)**2))
        assert np.isclose(np.array(alpha).dot(alpha), 1), "alpha**2 must be equal to 1"
        self.alpha = alpha
        self.beta = 0.1
        self.constraint_type = 'spherical'
        self.f2 = lambda y: sin(y)
        self.f2_jac = lambda y: cos(y)
        
        y_init = r_[linspace(1,5,num=nosc), np.zeros(nosc)]
        
        if self.constraint_type == 'linear':
            y_init[nosc-1] = -(y_init[:nosc-1].dot(alpha[:nosc-1]))/alpha[nosc-1] # project on the manifold
            assert np.isclose(y_init[:nosc].dot(alpha[:nosc]), 0), "alpha:y should be 0" 
        elif self.constraint_type == 'spherical':
            y_init /= sqrt((y_init[:nosc]**2).dot(alpha[:nosc])) # project on the manifold
            assert np.isclose((y_init[:nosc]**2).dot(alpha[:nosc]), 1), "alpha:y^2 should be 1" 
            
        
        # if reduction_bases is not None:
        #     self.RB, self.U, self.P = reduction_bases().RB, reduction_bases().U, reduction_bases().P
        # else:
        self.RB, self.U, self.P = None, None, None
            
        self.y_init = y_init #if reduction_bases is None else y_init.dot(self.RB.T)
         
        self.T_final = 2
        self.dt_space = concatenate(([], linspace(0.05, 0.1, num=5)))
        
        w_values = [0.28, 0.62546642846767004501]
        w_values.append(1.0 -2.0*(sum(w_values)))
        w_values.append(w_values[1])
        w_values.append(w_values[0])
        w_values = [1]
        assert np.isclose(sum(w_values), 1), 'sum_i w_i must be 1'
        self.w_values = w_values
        
        self.tol, self.M, self.var, self.store = 1.0E-13, 100, True, False
   
#%% Convergence rates
class Solver(object):
    "Class to repeatedly solve the MechSystem using MechSystemSolver"
    def __init__(self, registered_solver_classes=[ODESolver.ImplicitMidpoint]):
        self.registered_solver_classes = registered_solver_classes
        # self.reduction_bases = reduction_bases
        self.params = Params()
        
    def demo1(self):
        self.r_values, self.C_values = [], []
        
        for tile, solver_class in enumerate(self.registered_solver_classes, start=131):
            self.eng_error, self.sym_error, fig = [], [], figure()
            
            for dt in self.params.dt_space:
                self.params.dt = dt
                problem = MechSystem(self.params, method=solver_class)
                
                self.MSsolver = MechSystemSolver(self.params, problem=problem, method=solver_class)
                start = timer()
                self.MSsolver.solve()
                end = timer()
                print('Execution time: %s' %(end-start))
                
                if self.params.dt_space.size > 1:
                    self.eng_error.append(sqrt(dt)*LA.norm(self.MSsolver.eng_error))
                    self.sym_error.append(sqrt(dt)*LA.norm(self.MSsolver.sym_error))
    
            # Estimate Convergence rate r and coefficient C
            r_form = lambda numer, denom: (log(roll(numer, -1)/numer)/log(roll(denom, -1)/denom))[:-1]
            C_form = lambda numer, denom, r: numer[:-1]/(denom[:-1]**r)
            if self.eng_error[-1]:
                "Compute convergence rates from the error in Energy"
                self.r_values.append(r_form(self.eng_error,self.params.dt_space))
                self.C_values.append(C_form(self.eng_error,self.params.dt_space,self.r_values[-1]))
                
            if solver_class == ODESolver.ImplicitMidpoint and self.params.beta != 0 and hasattr(self, 'sym_error'):
                '''Compute convergence rate from the error in symplecticness
                Only applicable if the error is non-zero'''
                self.r_values.append(r_form(self.sym_error,self.params.dt_space))
                self.C_values.append(C_form(self.sym_error,self.params.dt_space,self.r_values[-1]))
    
            # Plot the measures
            self.MSsolver.plot(fig)
    
        # Display convergence rates if available
        if self.r_values:
            temp = r_[ reshape(self.params.dt_space, (1,-1)), \
                      c_[np.reshape([float('nan')]*len(self.r_values), (len(self.r_values),1)), self.r_values]].T
            if self.params.P is not None:
                temp[:,1] += 1
            elif self.params.RB is not None:
                temp[:,1] += -1
                
            tex_table(solver_class.__name__, temp)

    def mor_demo(self):
        "Model order reduction of the MechSystem using MechSystemSolver"
        # TODO: make it a class
        # Full order model
        self.demo1()
        
        # Reuced model
        _, s, RB = rb_svd(self.MSsolver.y)
        fig = figure()
        ax = fig.add_subplot(111)
        ax.semilogy(s)
        
        nosc_r = 2*5 # ensure nosc_r is even
        # RB = RB[:nosc_r, :]
        
        # class reduction_bases(object):
        #     """ Class of reduction bases """
        #     def __init__(self):
        #         self.RB = RB
        #         self.U = None
        #         self.P = None
        
        # POD-reduced solution
        self.params.RB = RB[:nosc_r, :];
        self.params.y_init = self.params.y_init.dot(self.params.RB.T);
        y = self.MSsolver.y
        self.demo1()
        print(np.amax(abs(y -self.MSsolver.y)))
        
        # Hyper-reduced model
        # TODO define F only once
        F = self.MSsolver.f.f2(self.MSsolver.f.Omega2 * self.MSsolver.y[:,:self.MSsolver.f.nosc])
        _, s, U = rb_svd(F)
        ax.semilogy(s)
        U = U.T
        
        fig = figure()
        ax = fig.add_subplot(111)
        ax.plot(range(U.shape[0]), U[:,:6])
        
        P, idx_list = DEIM(U, plot_deim=True)
        
        # class reduction_bases(object):
        #     """ Class of reduction bases """
        #     def __init__(self):
        #         self.RB = RB
        #         self.U = U[:, :nosc_r]
        #         self.P = P[:, :nosc_r]
                
        # POD-DEIM reduced model
        self.params.U, self.params.P = U[:, :nosc_r], P[:, :nosc_r]
        self.demo1()
        print(np.amax(abs(y -self.MSsolver.y)))
        
        # What works:
            # Results are sensitive to parameters
            # Unconstrained and conservative system give correct order and plots with Implicit midpoint
            # Method order cannot be tested with energy error for constrained and dissipative systems because energy is not preserved
            # Spherical constraints with dissipatation giver order ~3 for implicit midpoint for both full and reduced models. But not time gain.
            
        # Next steps:
            # impletement hyper-reduction
            
        # nosc = 400
        # Execution time: 9.104326547996607
        # Execution time: 7.462360573001206
        # Execution time: 6.858906443987507
        # Execution time: 5.642066926026018
        # Execution time: 4.1812998189998325
        # ImplicitMidpoint 
        #  0.050 & nan \\
        # 0.062 & 2.993 \\
        # 0.075 & 2.821 \\
        # 0.088 & 3.088 \\
        # 0.100 & 3.102
        # Execution time: 7.627723084995523
        # Execution time: 6.077715628023725
        # Execution time: 4.883099256985588
        # Execution time: 3.705847985984292
        # Execution time: 3.588499525008956
        # ImplicitMidpoint 
        #  0.050 & nan \\
        # 0.062 & 2.993 \\
        # 0.075 & 2.821 \\
        # 0.088 & 3.089 \\
        # 0.100 & 3.102
        # 0.0031688887180278957
        # Execution time: 6.780311873997562
        # Execution time: 6.871341875987127
        # Execution time: 4.48258072999306
        # Execution time: 4.010657687991625
        # Execution time: 3.86312277399702
        # ImplicitMidpoint 
        #  0.050 & nan \\
        # 0.062 & 2.993 \\
        # 0.075 & 2.821 \\
        # 0.088 & 3.089 \\
        # 0.100 & 3.102
        # 0.006587031839689672

if __name__ == '__main__':
    solver = Solver()
    solver.mor_demo()
    