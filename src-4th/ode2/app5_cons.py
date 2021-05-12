#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 11:49:50 2020

@author: ashishbhatt

This app solves constrained mechanical systems using solver
classes in the ODESolver hierarchy of methods.
"""

import numpy as np
from numpy import linalg as LA
from scitools.std import *
from pylab import *
import ODESolver
from PlotScript import plot_data, tex_table
from timeit import default_timer as timer
#import scipy
#from scipy.integrate import solve_ivp

#%% Define the system
class MechSystem(object):
    def __init__(self, Params, method=None):
        if np.dot(Params.alpha, Params.alpha) != 1:
            ValueError('alpha''s norm must be 1')

        self.Omega2 = np.diag(Params.Omega)**2
        self.alpha = Params.alpha
        self.beta = Params.beta
        self.nosc = Params.nosc
        self.method = method
        self.u_init = Params.y_init
        self.T = Params.T_final
        
        self.RB = Params.RB
        # if self.RB is None:
        self._g = lambda y: y[:, :self.nosc].dot(Params.alpha)
        self._G = lambda y=None: np.array(Params.alpha) if y is None else y[:, self.nosc:].dot(Params.alpha)
        # else:
        #     self._g = lambda y: (y.dot(self.RB)[:, :self.nosc]).dot(Params.alpha)
        #     self._G = lambda y=None: np.array(Params.alpha) if y is None else y.dot(self.RB)[:, self.nosc:].dot(Params.alpha)

    def __call__(self, y, t):
        method, Omega2, nosc, beta = self.method, self.Omega2, self.nosc, self.beta
        RB = self.RB

        if RB is not None:
            y = y.dot(RB)
            
        x, u = y[:nosc], y[nosc:]

        if method in [ODESolver.ConformalStormerVerlet]:
            f = [u, -Omega2.dot(sin(x))]
        elif method in [ODESolver.ConformalImplicitMidpoint]:
            f = reshape([u, -Omega2.dot(sin(x)) -beta*u], (2*nosc,)) +beta/2*y
        elif method in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
            f = reshape([u, -Omega2.dot(sin(x)) -beta*u], (2*nosc,))
        else:
            NameError('Undefined method - %s' % method)

        # if RB is not None: f = f.dot(RB.T)
        return f if RB is None else f.dot(RB.T)

    def jacobian(self, y, t, dt=0):
        "Jacobian of the function f"
        Omega2, nosc, beta, RB = self.Omega2, self.nosc, self.beta, self.RB
        if RB is not None:
            y = y.dot(RB)
        x, u = y[:nosc], y[nosc:]
        if self.method in [ODESolver.ConformalImplicitMidpoint]:
            dfdu = np.concatenate([np.concatenate([beta/2*eye(nosc), eye(nosc)], axis=1), \
                                   np.concatenate([-Omega2.dot(diag(cos(x))), -beta/2*eye(nosc)], axis=1)])
        elif self.method in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
            dfdu = r_[c_[zeros((nosc, nosc)), eye(nosc)], \
                                   c_[-Omega2.dot(diag(cos(x))), -beta*eye(nosc)]]
        else:
            NameError('Jacobian undefined for the method - %s' % self.method)
            
        # if RB is not None: dfdu = dfdu.dot(RB.T)
        return dfdu if RB is None else RB.dot(dfdu.dot(RB.T))

    def en_err(self, y, t=0):
        "Energy error"
        if self.beta != 0.:
            raise ValueError('Energy is only defined for beta = 0')

        Omega2, nosc = self.Omega2, self.nosc
        # if self.RB is not None:
        #     y = self.RB.dot(y)
        x, u = y[:, :nosc], y[:, nosc:]

        T = lambda u: sum((u**2), axis=1)/2.0
        V = lambda x: -sum(Omega2.dot(cos(x).T), axis=0)
        E = lambda x, u: V(x) +T(u)
        return E(x, u) - E(reshape(x[0, :], (1, nosc)), reshape(u[0, :], (1, nosc)))

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
        if self.RB is None:
            nosc2 = 2*self.f.nosc
        else:
            nosc2 = self.RB.shape[0] # 2*nosc
            
        self.n = int(round(self.f.T/self.dt))

        self.t_points = linspace(0, self.f.T, self.n+1)
        self.y = np.zeros((self.n+1, nosc2))
        self.y[0] = self.f.u_init
        if self.store: self.info = []

        for k in range(self.n):
            y_ = np.array([self.y[k], self.y[k]])
            for w_val in self.w_values:
                self.solver.set_initial_condition(y_[1])
                y_, tp = self.solver.solve(w_val*self.t_points[k:k+2])

# =============================================================================
#                 Delta_LambdaX, X, m = self.f._g(reshape(y_[1], (1, 2*nosc))), y_[1, :nosc], 0
#                 if self.RB is None:
#                     X = y_[1, :nosc]
#                 else:
#                     X = y_.dot(self.RB)[1, :self.f.nosc]
#                 if self.store: self.info.append((X, Delta_LambdaX, m))
# 
#                 while abs(Delta_LambdaX) > self.tol and m <= self.M:
#                     X = X -np.transpose(self.f._G())*Delta_LambdaX
# 
#                     m += 1
#                     if self.RB is not None: X = X.dot(self.RB.T)
#                     Delta_LambdaX = self.f._g(reshape(X, (1, nosc)))
#                     if self.store: self.info.append((X, Delta_LambdaX, m))
# 
#                 y_[1, :nosc] = X
# 
#                 Delta_LambdaU, U, m = self.f._G(reshape(y_[1], (1, 2*nosc))), y_[1, nosc:], 0
#                 if self.RB is not None: U = U.dot(self.RB)
#                 if self.store: self.info.append((U, Delta_LambdaU, m))
# 
#                 while abs(Delta_LambdaU) > self.tol and m <= self.M:
#                     U = U -np.transpose(self.f._G())*Delta_LambdaU
# 
#                     m += 1
#                     if self.RB is not None: U = U.dot(self.RB.T)
#                     Delta_LambdaU = self.f._G(U)
#                     if self.store: self.info.append((U, Delta_LambdaU, m))
# 
#                 y_[1, nosc:] = U
# =============================================================================
                
                
                m = 0
                if self.RB is None:
                    X_U = y_[1]
                else:
                    X_U = y_.dot(self.RB)[1]
                X_U = reshape(X_U, (1, 2*self.f.nosc))
                Delta_LambdaX, Delta_LambdaU = self.f._g(X_U), self.f._G(X_U)
                
                if self.store: self.info.append((X_U, Delta_LambdaX, Delta_LambdaU, m))

                while abs(max((Delta_LambdaX, Delta_LambdaU))) > self.tol and m <= self.M:
                    X_U = X_U -r_[np.transpose(self.f._G())*Delta_LambdaX, np.transpose(self.f._G())*Delta_LambdaU]

                    m += 1
                    # if self.RB is not None:
                    #     X_U = X_U.dot(self.RB.T)
                        
                    Delta_LambdaX, Delta_LambdaU = self.f._g(X_U), self.f._G(X_U)
                    if self.store: self.info.append((X_U, Delta_LambdaX, Delta_LambdaU, m))

                y_[1] = X_U if self.RB is None else X_U.dot(self.RB.T)

            if self.RB is None:
                self.y[k+1] = reshape(X_U, (nosc2,))
            else:
                self.y[k+1] = reshape(X_U.dot(self.RB.T), (nosc2,))

        if self.RB is not None: self.y = self.y.dot(self.RB)
        
        if self.var == True:
            RB, self.f.RB = self.f.RB, None # temporarily set RB to None
            self.var_solve()
            self.f.RB = RB # Turn RB back on

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

        if not self.f.beta:
            self.eng_error = sqrt(self.dt)*LA.norm(self.f.en_err(self.y))

    def plot(self, fig, tile=131):
        "plot the results"
        ax = fig.add_subplot(tile)
        if not self.f.beta:
            plot_data(ax, self.t_points, np.array([reshape(self.sym_error[-1:],(self.n+1,)), self.f.en_err(self.y), self.f._g(self.y), self.f._G(self.y)]).T, True, True)
            # ax2.legend([r'$\boldmath{E}_{cs}$', r'$\boldmath{E}_I$', r'$\bolmath{g}(\boldmath{q}^{n+1})$', r'$\boldmath{G}^\top \boldmath{p}^{n+1}$'], loc=1)
            fig.legend([r'$\boldmath{E}_{cs}$', r'$\boldmath{E}_I$', r'$\boldmath{g}(\boldmath{q}^{n+1})$', r'$\boldmath{G}^\top \boldmath{p}^{n+1}$'], \
                       bbox_to_anchor=(0.5,-0.08), loc='lower center',ncol=2,bbox_transform=fig.transFigure)
        elif self.var:
            plot_data(ax, self.t_points, np.array([reshape(self.sym_error[-1:],(self.n+1,)), self.f._g(self.y), self.f._G(self.y)]).T, True, True)
            # ax2.legend([r'$\boldmath{E}_{cs}$', r'$\bolmath{g}(\boldmath{q}^{n+1})$', r'$\boldmath{G}^\top \boldmath{p}^{n+1}$'], loc=1)
            fig.legend([r'$\boldmath{E}_{cs}$', r'$\boldmath{g}(\boldmath{q}^{n+1})$', r'$\boldmath{G}^\top \boldmath{p}^{n+1}$'], \
                       bbox_to_anchor=(0.5,-0.08), loc='lower center',ncol=3,bbox_transform=fig.transFigure)
        ax.set_xlabel('time')

        ax = fig.add_subplot(tile+1)
        plot_data(ax, self.t_points, self.y[:,:self.f.nosc], True, True)
        ax = fig.add_subplot(tile+2)
        plot_data(ax, self.t_points, self.y[:,self.f.nosc:], True, True)

        # fig.savefig('app5_err_inv.pdf', bbox_inches='tight')

#%% Convergence analysis
def demo1(registered_solver_classes=[ODESolver.ConformalImplicitMidpoint], nosc=100, RB=None):
    r_values, C_values = [], []
    
    class Params(object):
        """ Class of parameters """
        def __init__(self, nosc=100, RB=None):
            self.nosc = nosc
            self.Omega = list(linspace(1,5,num=nosc))
            alpha = list(linspace(0.001,0.01,num=nosc))
            alpha[-1] = sqrt(1 -sum(np.array(alpha[:-1])**2))
            self.alpha = alpha
            self.beta = 0.1
            
            y_init = list(linspace(0.1,0.5,num=nosc)) +list(np.zeros(nosc))
            y_init[nosc-1] = -(np.array(y_init[:nosc-1]).dot(alpha[:nosc-1]))/alpha[nosc-1] # project on the manifold
            self.y_init = y_init
            
            self.RB = RB            
            if RB is not None:
                self.y_init = np.asarray(y_init).dot(RB.T)
            
            self.T_final = 1
            self.dt_space = concatenate(([], linspace(0.005, 0.05, num=5)))
            
            w_values = [0.28, 0.62546642846767004501]
            w_values.append(1.0 -2.0*(sum(w_values)))
            w_values.append(w_values[1])
            w_values.append(w_values[0])
            w_values = [1]
            self.w_values = w_values
            
            self.tol, self.M, self.var, self.store = 1.0E-14, 100, True, False
    
    for tile, solver_class in enumerate(registered_solver_classes, start=131):
        eng_error, sym_error, fig, params = [], [], figure(), Params(nosc, RB)
        for dt in params.dt_space:
            params.dt = dt
            problem = MechSystem(params, method=solver_class)
            MSsolver = MechSystemSolver(params, problem=problem, method=solver_class)
            start = timer()
            MSsolver.solve()
            end = timer()
            print(end -start)

            eng_error.append(MSsolver.eng_error)
            sym_error.append(sqrt(dt)*LA.norm(MSsolver.sym_error))

        # Estimate Convergence rate r and coefficient C
        r_form = lambda numer, denom: (log(roll(numer, -1)/numer)/log(roll(denom, -1)/denom))[:-1]
        C_form = lambda numer, denom, r: numer[:-1]/(denom[:-1]**r)
        if not params.beta and params.dt_space.size > 1:
            r_values.append(r_form(eng_error,params.dt_space))
            C_values.append(C_form(eng_error,params.dt_space,r_values[-1]))
            
        if params.dt_space.size > 1:
            r_values.append(r_form(sym_error,params.dt_space))
            C_values.append(C_form(sym_error,params.dt_space,r_values[-1]))

        # Plot the measures
        MSsolver.plot(fig)

    # Display convergence table
    if params.dt_space.size > 1:
        temp = r_[ reshape(params.dt_space, (1,params.dt_space.size)), c_[np.reshape([np.float('nan')]*len(r_values), (len(r_values),1)), r_values]].T
        # temp = np.array((params.dt_space, np.concatenate(([np.float('nan')],r_values[0])),\
        #                   np.concatenate(([np.float('nan')], r_values[1])))).T
        # temp = np.asarray([params.dt_space, [np.float('nan')]+r_values[0], [np.float('nan')]+r_values[1]]).T
        tex_table(solver_class.__name__, temp)

    return MSsolver.y


#%% MOR
def mor_demo():
    registered_solver_classes = [ODESolver.ConformalImplicitMidpoint]
    
    y = demo1(registered_solver_classes, nosc=100)
    
    u, s, vh = LA.svd(y, full_matrices=False)
    print np.allclose(y, np.dot(u * s, vh))
    smat = np.diag(s)
    print np.allclose(y, np.dot(u, np.dot(smat, vh)))
    
    fig = figure()
    ax = fig.add_subplot(111)
    ax.plot(s)
    
    nosc_r = 5
    RB = vh[:nosc_r, :]
    y_r = demo1(registered_solver_classes, nosc=100, RB=RB)
    print np.amax(abs(y-y_r))
    