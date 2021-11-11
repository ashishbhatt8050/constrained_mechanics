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
from scipy import optimize, linalg
#from scipy.integrate import solve_ivp

#%% Define the system
class MechSystem(object):
    def __init__(self, Params, method=None):

        self.Omega2 = np.diag(Params.Omega)**2
        self.alpha = Params.alpha
        self.beta = Params.beta
        self.nosc = Params.nosc
        self.method = method
        self.u_init = Params.y_init
        self.T = Params.T_final
        self.constraint_type = Params.constraint_type
        
        self.RB, self.U, self.P = Params.RB, Params.U, Params.P
        # self.Ug, self.Pg = Params.Ug, Params.Pg
        # if self.RB is None:
            
        if self.RB is not None and self.P is not None:
            self.Omega2_r = np.diag(self.P[self.nosc:,self.P.shape[1]//2:].T @ np.array(Params.Omega)**2) # Needs justification
            # self.Omega2_r = Omega2_[:self.P.shape[1]//2, :self.P.shape[1]//2]
            # nosc = P.shape[1]
        
            
        if Params.constraint_type == 'linear':
            if True:
                self._g = lambda y: r_[y[:, :self.nosc].dot(self.alpha), y[:, self.nosc:].dot(self.alpha)]
                self._g_prime = lambda y: r_[c_[np.array(self.alpha, ndmin=2), zeros((1,self.nosc))],\
                                            c_[zeros((1,self.nosc)), np.array(self.alpha, ndmin=2)]]
            else:
                pass
            #self._G = lambda y=None: np.array(self.alpha) if y is None else y[:, self.nosc:].dot(self.alpha)
        else:
            if True:
                self._g = lambda y: r_[(y[:, :self.nosc]**2).dot(self.alpha) -1.0,\
                                       diag(y[:, self.nosc:].dot((2*y[:, :self.nosc]*self.alpha).T)).reshape(-1)]
                self._g_prime = lambda y: r_[c_[2*y[:, :self.nosc]*self.alpha, zeros((1,self.nosc))],\
                                             c_[zeros((1,self.nosc)), 2*y[:, :self.nosc]*self.alpha]]
            else:
                pass
            #self._G = lambda y: sum(y[:, self.nosc:]*(2*y[:, :self.nosc]*np.array(self.alpha)), axis=1)
        
        # else:
        #     self._g = lambda y: (y.dot(self.RB)[:, :self.nosc]).dot(Params.alpha)
        #     self._G = lambda y=None: np.array(Params.alpha) if y is None else y.dot(self.RB)[:, self.nosc:].dot(Params.alpha)

    def __call__(self, y, t):
        method, Omega2, beta = self.method, self.Omega2, self.beta
        P, U, RB = self.P, self.U, self.RB

        if RB is not None and P is not None:
            y = RB.T @ y
            # Omega2 = self.Omega2_r
            # Omega2_ = P.T @ linalg.block_diag(Omega2, np.eye(Omega2.shape[0])) @ P # Needs justification
            # Omega2 = Omega2_[:P.shape[1]//2, :P.shape[1]//2]
            # nosc = P.shape[1]
        elif RB is not None:
            y = RB.T @ y
            
        x, u = np.split(y, 2)
        nosc = len(x)

        if method in [ODESolver.ConformalStormerVerlet]:
            f = [u, -Omega2.dot(sin(Omega2.dot(x)))]
        elif method in [ODESolver.ConformalImplicitMidpoint]:
            f = reshape([u, -Omega2.dot(sin(Omega2.dot(x))) -beta*u], -1) +beta/2*y
        elif method in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
            f = r_[u, -Omega2.dot(sin(Omega2.dot(x))) -beta*u]
            # f = r_[c_[eye(nosc), zeros((nosc,nosc))], c_[-beta*eye(nosc), -Omega2]] @ r_[u, sin(Omega2.dot(x))]
        else:
            NameError('Undefined method - %s' % method)

        if RB is not None and P is not None:
            return RB @ U @ np.linalg.solve(P.T @ U, P.T @ f)
        elif RB is not None:
            return RB @ f
        else:
            return f

    def jacobian(self, y, t, dt=0):
        "Jacobian of the function f"
        Omega2, beta = self.Omega2, self.beta
        P, U, RB = self.P, self.U, self.RB

        if RB is not None and P is not None:
            y = RB.T @ y
            # Omega2 = self.Omega2_r
            # Omega2_ = P.T @ linalg.block_diag(Omega2, np.eye(Omega2.shape[0])) @ P # Needs justification
            # Omega2 = Omega2_[:P.shape[1]//2, :P.shape[1]//2]
            # nosc = P.shape[1]
        elif RB is not None:
            y = RB.T @ y
            # nosc = RB.shape[1]
            
        x, u = np.split(y, 2)
        nosc = len(x)
        
        if self.method in [ODESolver.ConformalImplicitMidpoint]:
            dfdy = np.concatenate([np.concatenate([beta/2*eye(nosc), eye(nosc)], axis=1), \
                                   np.concatenate([-diag(Omega2.dot(Omega2.dot(cos(Omega2.dot(x))))), -beta/2*eye(nosc)], axis=1)])
        elif self.method in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
            dfdy = r_[c_[zeros((nosc, nosc)), eye(nosc)], \
                                    c_[-diag(Omega2.dot(Omega2.dot(cos(Omega2.dot(x))))), -beta*eye(nosc)]]
            # dfdy = r_[c_[eye(nosc), zeros((nosc,nosc))], c_[-beta*eye(nosc), -Omega2]] \
            #     @ r_[c_[zeros((nosc,nosc)), eye(nosc)], c_[Omega2 @ cos(Omega2 @ x), zeros((nosc, nosc))]]
        else:
            NameError('Jacobian undefined for the method - %s' % self.method)

        if P is not None:
            return RB @ U @ np.linalg.solve(P.T @ U, P.T @ dfdy) @ RB.T
        elif RB is not None:
            return RB @ dfdy @ RB.T
        else:
            return dfdy
            
        # if RB is not None: dfdy = dfdy.dot(RB.T)
        # return dfdy if RB is None else RB.dot(dfdy.dot(RB.T))

    def en_err(self, y, t=0):
        "Energy error"
        if self.beta != 0.:
            raise ValueError('Energy is only defined for beta = 0')

        Omega2, nosc = self.Omega2, self.nosc
        x, u = y[:, :nosc], y[:, nosc:]

        T = lambda u: sum((u**2), axis=1)/2.0
        V = lambda x: -sum(cos(x.dot(Omega2)), axis=1)
        E = lambda x, u: V(x) +T(u)
        return log(E(x, u)/E(x[0:1, :], u[None, 0, :]))
        #return E(x, u) -E(x[0:1, :], u[None, 0, :])

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
        self.F = []
        
        if self.RB is None:
            self.y = np.zeros((self.n+1, 2*nosc))
            self.F.append(self.f(self.y[0], self.t_points[0]))
        else:
            self.y = np.zeros((self.n+1, self.RB.shape[0]))
            
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
                
                if self.RB is None:
                    X_U = y_
                else:
                    X_U = y_.dot(self.RB)
                #X_U = reshape(X_U, (1, -1))
                    
                start = timer()
                if self.f.constraint_type in ['spherical','linear']:
                    m, Delta_Lambda = 0, self.f._g(X_U[1:2])/sum(self.f._g_prime(X_U[1:2])*self.f._g_prime(X_U[0:1]), axis=1)
                    
                    if self.store: self.info.append((m, Delta_Lambda, X_U[1]))
    
                    while max(abs(self.f._g(X_U))) > self.tol and m < self.M:
                        X_U[1] = X_U[1] -self.f._g_prime(X_U[0:1]).T.dot(Delta_Lambda)
    
                        m, Delta_Lambda = m+1, self.f._g(X_U[1:2])/sum(self.f._g_prime(X_U[1:2])*self.f._g_prime(X_U[0:1]), axis=1)
                        if self.store: self.info.append((m, Delta_Lambda, X_U[1]))
                    assert m <= self.M, "Constraints not satisfied"
                    
                elif self.f.constraint_type is ['sp.optimize.root']:
                    X = optimize.root(self.f._g, X_U[:, :nosc], method='broyden1')
                    U = optimize.root(lambda y: self.f._G(c_[X,y]), X_U[:, nosc:], method='broyden1')
                    X_U = c_[X, U]
                    raise NotImplementedError
                end = timer()
                # print('Execution time: %s' %(end-start))
                
                
                y_[1] = X_U[1] if self.RB is None else X_U[1].dot(self.RB.T)

            self.y[k+1] = y_[1]
            
            if self.RB is None:
                self.F.append(self.f(self.y[k+1], self.t_points[k+1]))

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

    def plot(self, fig, tile=131):
        "plot the results"
                
        # ax = fig.add_subplot(tile)
        
        # if hasattr(self, 'eng_error'):
        #     plot_data(ax, self.t_points, c_[self.sym_error[-1], self.eng_error[-1], self.f._g(self.y).reshape(2,-1).T])
        #     # ax2.legend([r'$\boldmath{E}_{cs}$', r'$\boldmath{E}_I$', r'$\bolmath{g}(\boldmath{q}^{n+1})$', r'$\boldmath{G}^\top \boldmath{p}^{n+1}$'], loc=1)
        #     fig.legend([r'$\boldmath{E}_{cs}$', r'$\boldmath{E}_I$', r'$\boldmath{g}(\boldmath{q}^{n+1})$', r'$\boldmath{G}^\top \boldmath{p}^{n+1}$'], \
        #                bbox_to_anchor=(0.5,-0.08), loc='lower center',ncol=2,bbox_transform=fig.transFigure)
                
        # elif self.var:
        #     plot_data(ax, self.t_points, c_[self.sym_error[-1], self.f._g(self.y).reshape(2,-1).T])
        #     # ax2.legend([r'$\boldmath{E}_{cs}$', r'$\bolmath{g}(\boldmath{q}^{n+1})$', r'$\boldmath{G}^\top \boldmath{p}^{n+1}$'], loc=1)
        #     fig.legend([r'$\boldmath{E}_{cs}$', r'$\boldmath{g}(\boldmath{q}^{n+1})$', r'$\boldmath{G}^\top \boldmath{p}^{n+1}$'], \
        #                bbox_to_anchor=(0.5,-0.08), loc='lower center',ncol=3,bbox_transform=fig.transFigure)
        # ax.set_xlabel('time')

        # ax = fig.add_subplot(tile+1)
        # plot_data(ax, self.t_points, self.y[:,:self.f.nosc])
        # ax = fig.add_subplot(tile+2)
        # plot_data(ax, self.t_points, self.y[:,self.f.nosc:])

        # fig.savefig('app5_err_inv.pdf', bbox_inches='tight')
        
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
        
        # for ax in fig.get_axes():
        #     ax.label_outer()

#%% Convergence analysis
def demo1(registered_solver_classes=[ODESolver.ConformalImplicitMidpoint], nosc=100, reduction_bases=None):
    r_values, C_values = [], []
    
    class Params(object):
        """ Class of parameters """
        def __init__(self, nosc=100, reduction_bases=None):
            self.nosc = nosc
            self.Omega = list(linspace(1,1.5,num=nosc))
            alpha = list(linspace(0.1,0.5,num=nosc))
            alpha /= sqrt(sum(np.array(alpha)**2))
            assert np.isclose(np.array(alpha).dot(alpha), 1), "alpha**2 must be equal to 1"
            self.alpha = alpha
            self.beta = 0.1
            self.constraint_type = None
            
            y_init = r_[linspace(1,5,num=nosc), np.zeros(nosc)]
            
            if self.constraint_type == 'linear':
                y_init[nosc-1] = -(y_init[:nosc-1].dot(alpha[:nosc-1]))/alpha[nosc-1] # project on the manifold
                assert np.isclose(y_init[:nosc].dot(alpha[:nosc]), 0), "alpha:y should be 0" 
            elif self.constraint_type == 'spherical':
                y_init /= sqrt((y_init[:nosc]**2).dot(alpha[:nosc])) # project on the manifold
                assert np.isclose((y_init[:nosc]**2).dot(alpha[:nosc]), 1), "alpha:y^2 should be 1" 
                
            
            if reduction_bases is not None:
                self.RB, self.U, self.P = reduction_bases().RB, reduction_bases().U, reduction_bases().P
            else:
                self.RB, self.U, self.P = None, None, None
                
            self.y_init = y_init if reduction_bases is None else y_init.dot(self.RB.T)
             
            self.T_final = 2
            self.dt_space = concatenate(([], linspace(0.05, 0.1, num=5)))
            
            w_values = [0.28, 0.62546642846767004501]
            w_values.append(1.0 -2.0*(sum(w_values)))
            w_values.append(w_values[1])
            w_values.append(w_values[0])
            w_values = [1]
            assert np.isclose(sum(w_values), 1), 'sum_i w_i must be 1'
            self.w_values = w_values
            
            self.tol, self.M, self.var, self.store = 1.0E-14, 100, True, False
    
    for tile, solver_class in enumerate(registered_solver_classes, start=131):
        eng_error, sym_error, fig, params = [], [], figure(), Params(nosc, reduction_bases)
        for dt in params.dt_space:
            params.dt = dt
            problem = MechSystem(params, method=solver_class)
            
            MSsolver = MechSystemSolver(params, problem=problem, method=solver_class)
            start = timer()
            MSsolver.solve()
            end = timer()
            print('Execution time: %s' %(end-start))
            
            if params.dt_space.size > 1:
                eng_error.append(sqrt(dt)*LA.norm(MSsolver.eng_error))
                sym_error.append(sqrt(dt)*LA.norm(MSsolver.sym_error))

        # Estimate Convergence rate r and coefficient C
        r_form = lambda numer, denom: (log(roll(numer, -1)/numer)/log(roll(denom, -1)/denom))[:-1]
        C_form = lambda numer, denom, r: numer[:-1]/(denom[:-1]**r)
        if eng_error[-1]:
            "Compute convergence rates from the error in Energy"
            r_values.append(r_form(eng_error,params.dt_space))
            C_values.append(C_form(eng_error,params.dt_space,r_values[-1]))
            
        if solver_class == ODESolver.ImplicitMidpoint and params.beta != 0 and 'sym_error' in locals():
            '''Compute convergence rate from the error in symplecticness
            Only applicable if the error is non-zero'''
            r_values.append(r_form(sym_error,params.dt_space))
            C_values.append(C_form(sym_error,params.dt_space,r_values[-1]))

        # Plot the measures
        MSsolver.plot(fig)

    # Display convergence rates if available
    if r_values:
        temp = r_[ reshape(params.dt_space, (1,-1)), c_[np.reshape([float('nan')]*len(r_values), (len(r_values),1)), r_values]].T
        # temp = np.array((params.dt_space, np.concatenate(([np.float('nan')],r_values[0])),\
        #                   np.concatenate(([np.float('nan')], r_values[1])))).T
        # temp = np.asarray([params.dt_space, [np.float('nan')]+r_values[0], [np.float('nan')]+r_values[1]]).T
        tex_table(solver_class.__name__, temp)

    return MSsolver.y, MSsolver.t_points, MSsolver.F


#%% MOR
def mor_demo():
    registered_solver_classes = [ODESolver.ImplicitMidpoint]
    
    # Full order model
    y, _, F = demo1(registered_solver_classes, nosc=100)
    
    
    _, s, vh = rb_svd(y)
    fig = figure()
    ax = fig.add_subplot(111)
    ax.semilogy(s)
    
    nosc_r = 2*2 # nosc_r must be even
    # assert nosc_r%2 == 0, 'nosc_r must be even'
    RB = vh[:nosc_r, :]
    
    _, s, U = rb_svd(np.array(F))
    # fig = pl.figure()
    # ax1 = fig.add_subplot(111)
    # ax1.set_prop_cycle(cycler('linestyle', ['-.','-*','-.o',':']))
    ax.semilogy(s)
    U = U.T
    
    fig = figure()
    ax = fig.add_subplot(111)
    ax.plot(range(U.shape[0]), U[:,:6])#,\
            # range(U.shape[0])[idx_list[:6]], 0*range(U.shape[0])[idx_list[:6]], '*')
    
    class reduction_bases(object):
        """ Class of reduction bases """
        def __init__(self):
            self.RB = RB
            # self.U = U[:, :nosc_r]
            # self.P = P[:, :nosc_r]
            self.U = None
            self.P = None
    
    # POD-reduced model
    y_r, _, _ = demo1(registered_solver_classes, nosc=100, reduction_bases=reduction_bases)
    print(np.amax(abs(y-y_r)))
    
    P, idx_list = DEIM(U, plot_deim=True)
    
    class reduction_bases(object):
        """ Class of reduction bases """
        def __init__(self):
            self.RB = RB
            self.U = U[:, :nosc_r]
            self.P = P[:, :nosc_r]
            # self.U = None
            # self.P = None
            
    # POD-DEIM reduced model
    y_r, _, _ = demo1(registered_solver_classes, nosc=100, reduction_bases=reduction_bases)
    print(np.amax(abs(y-y_r)))
    
    # What works:
        # Results are sensitive to parameters
        # Unconstrained and conservative system give correct order and plots with Implicit midpoint
        # Method order cannot be tested with energy error for constrained and dissipative systems because energy is not preserved
        # Spherical constraints with dissipatation giver order ~3 for implicit midpoint for both full and reduced models. But not time gain.
        
    # Next steps:
        # impletement hyper-reduction

if __name__ == '__main__':
    mor_demo()