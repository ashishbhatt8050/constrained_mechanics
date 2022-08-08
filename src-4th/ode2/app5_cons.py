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
from numpy import linalg as LA
from pylab import *
import ODESolver
from System import System
from podDEIM import DEIM, orthogonalize, POD
from PlotScript import plot_data, tex_table
from time import process_time
import scipy as sp
import gc

from Newton import fixed_point

#%% Parameters class
class Params(object):
    """ Class of parameters """
    def __init__(self, **kwds):
        "Oscillator properties"
        self.system_type = kwds['system_type']
        
        System.set_system(self, kwds)
        
        "Projection matrices"
        if 'RB' in kwds:
            self.RB = kwds['RB']
            self.nosc_r = kwds['nosc_r']
            self.i_range_r = kwds['i_range_r']
        else:
            self.RB = None
                
        System.set_constraints(self, kwds)
            
        self.i_range = kwds['i_range']
            
        if 'W_r' in kwds:
            self.W_r = kwds['W_r']
            self.y_init = self.W_r.T @ self.y_init
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
            
            if self.P.shape == self.U.shape: #POD-DEIM
                sys_solve = LA.solve
            else: #Gappy-POD
                sys_solve = lambda A, b: LA.lstsq(A, b, rcond=None)[0]
                
            self.PxIPxRB = lambda y: self.PxP @ sys_solve(self.UxP, self.UxRB @ y)
            self.PxIPxRBxI = self.PxP @ sys_solve(self.UxP, self.UxRB)
            # self.PxIPxRBxP = lambda y: self.PxP @ sys_solve(self.UxP, self.UxRB @ y) @self.P.T
            self.hat = lambda foPxIPxRBxy: self.RBxU @ sys_solve(self.PxU, foPxIPxRBxy)
            self.hhat = lambda ffoPxIPxRBxy: self.hat(ffoPxIPxRBxy) @ self.PxIPxRBxI
            
            self.IPt = lambda y: self.P @ sys_solve(self.UxP, self.U.T @y)
        else:
            self.P = None
            
        "Reduced MechSystem constituents"
        if self.W_r is not None:
            self.JJ_r = self.W_r.T @ self.JJ() @ self.W_r

        
        "Numerical solver and its properties"
        self.registered_solver_classes = kwds['registered_solver_classes']
            
        self.solver_class = kwds['solver_class']
        
        self.dt = kwds['dt']
         
        self.T_final = 20
        
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
        
        return System.get_func(self, y, t)
            

    def jacobian(self, y, t, *arg):
        "Jacobian of the function f"
        
        return System.get_jacobian(self, y, t)

    def en_err(self):
        "Energy error"
        assert self.beta == 0, 'Energy is only defined for beta = 0'
        
        return System.get_en_err(self)
                

#%% Solve the system and find convergence rates
class MechSystemSolver(MechSystem):
    """
    Class for solving problems of the class MechSystem
    """
        
    def solve(self):
            
        self.n = int(round(self.T_final/self.dt))

        self.t_points = linspace(0, self.T_final, self.n+1)
        
        if self.RB is None:
            self.y = np.zeros((self.n+1, 2*self.nosc))
        else:
            self.y = np.zeros((self.n+1, 2*self.nosc_r))
            
            if self.constraint_type and self.system_type == 'system':
                X_U = np.array([self.y_init, self.y_init])
                X_U, _, _ = fixed_point(self._g, X_U, self._g_prime, self.tol, self.M, False)
                self.y_init = X_U[1]
            
        self.y[0] = self.y_init
        if self.store: self.info = []

        for k in range(self.n):
            self.info.append(self.y[k])
            y_ = np.array([self.y[k], self.y[k]])
            
            for w_val in self.w_values:
                self.solver.set_initial_condition(y_[1])
                y_, _, info_ = self.solver.solve(w_val*self.t_points[k:k+2])
                if self.store: self.info.append(np.array(info_[0::1]))
                
                # enforce constraints
                if self.constraint_type and self.system_type == 'oscillator':
                    
                    y_, _, _ = fixed_point(self._g, y_, self._g_prime, self.tol, self.M, False)
                elif self.constraint_type:
                    y_[1] = np.exp(-2*self.beta*self.dt) *LA.norm(self.y[k])/LA.norm(y_[1]) *y_[1]
                    

            self.y[k+1] = y_[1]
            
        self.info.append(self.y[k+1])
        self.info = np.vstack(self.info)

        if self.RB is not None:
            self.y_red = self.y
            self.y = (self.RB @ self.y.T).T
        
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
        elif hasattr(self, 'norm_error'):
            plot_data(ax[1,0], self.t_points, self.norm_error)
            
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
            
        if self.system_type == 'oscillator':
            if self.RB is not None:
                plot_data(ax3, self.y_red[:,self.i_range_r], self.y_red[:,self.nosc_r+self.i_range_r])
                
            plot_data(ax4, self.y[:,self.i_range], self.y[:,self.nosc+self.i_range])
                
            # ax[3,1].set_xlabel('time')
            # fig.savefig('app5_err_inv.pdf', bbox_inches='tight')
            
            # for ax in fig.get_axes():
            #     ax.label_outer()
        else:
            # plot_data(ax[0,1], self.t_points, )
            ax4 = fig.add_subplot(gs[2:4, -1])
                
            plot_data(ax4, self.x_points, self.y[[0,-1]].T)
        

    def measures(self, errors, dt_space):
        "Various measurements based on the solution"
        
        r_values = []
        C_values = []
        
        r_form = lambda numer, denom: (log(roll(numer, -1)/numer)/log(roll(denom, -1)/denom))[:-1]
        C_form = lambda numer, denom, r: numer[:-1]/(denom[:-1]**r)
        
        if errors['energy']:
            "Compute convergence rates from the error in Energy"
            r_values.append(r_form(errors['energy'],dt_space))
            C_values.append(C_form(errors['energy'],dt_space,r_values[-1]))
            
        if self.solver_class == ODESolver.ImplicitMidpoint and self.beta != 0 and errors['spl']:
            '''Compute convergence rate from the error in symplecticness
            Only applicable if the error is non-zero'''
            r_values.append(r_form(errors['spl'],dt_space))
            C_values.append(C_form(errors['spl'],dt_space,r_values[-1]))
            
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
    num_solver_classes = len(registered_solver_classes)
        
    dt_space_dim = 3
    
    kwds = {'system_type': 'KdV'}
    
    if kwds['system_type'] == 'oscillator':
            
        Omega2_space_dim = 3
        Omega2_space = 1 +np.random.rand(Omega2_space_dim, nosc)/1000
        
        kwds.update({'nosc': nosc, \
                    'dt_space': linspace(0.05, 0.1, num=dt_space_dim), \
                    'Omega2_space': Omega2_space, \
                    'registered_solver_classes': registered_solver_classes, \
                    'i_range': np.random.randint(0, nosc, 3)})
    else:
        Omega2_space_dim = 1
        kwds.update({'dt_space': linspace(0.001, 0.009, num=dt_space_dim), \
                    'Omega2_space': np.ones((Omega2_space_dim,nosc)), \
                    'registered_solver_classes': registered_solver_classes, \
                    'i_range': np.random.randint(0, nosc, 3)})
        
    MSsolvers = solver(kwds)
    
    if MSsolvers[-1].non_quad: # is not None
        assert Omega2_space_dim == 1
    
    time_lapsed = [reshape([x.time_lapsed for x in MSsolvers],\
                           (num_solver_classes, dt_space_dim, Omega2_space_dim))]
        
    y_list = np.hstack([MSsolver.info.T for MSsolver in MSsolvers])
    F2 = np.hstack([MSsolver.F2 for MSsolver in MSsolvers])
    F3 = np.hstack([MSsolver.F3 for MSsolver in MSsolvers])
    
    X = {'Q': MSsolvers[-1].Q_spd(), \
         'sqrt': sp.linalg.sqrtm(MSsolvers[-1].Q_spd()), \
         'eye': np.eye(2*nosc)}
    
    fig = figure()
    ax = fig.add_subplot(111)
    
    if MSsolvers[-1].non_quad:
        np.linalg.cholesky(X['Q'])
        RB, s = POD(X['sqrt'] @ y_list, X['eye'])
        RB = LA.solve(X['sqrt'], RB)
    
        nosc_r = argmin(abs(np.asarray([norm(s[:i])/norm(s) for i in range(len(s))]) -0.99))
        if nosc_r <  20:
            nosc_r = 20  # ensure nosc_r is even
            
        X.update({'eye_r': np.eye(2*nosc_r)})
            
        print('nosc_r = %s' %nosc_r)
        
        RB = RB[:, :2*nosc_r]
        ax.semilogy(s)
    
        # Orthogonalize RB wrt Q_spd
        # U = sp.linalg.cholesky(RB.T @ X['Q'] @ RB) # upper triangular Cholesky factor
        # RB = RB @ sp.linalg.solve(U, X['eye_r'])
        assert np.allclose(RB.T @ X['Q'] @ RB, X['eye_r'])
        
        W_r_, s = POD(F2, X['eye'])
        # W_r_ = LA.solve(X['sqrt'], W_r_)
        W_r_ = W_r_[:, :2*nosc_r]
        ax.semilogy(s)
        # W_r = RB
        
        # Orthognalize W_r_ wrt RB
        W_r = orthogonalize(W_r_, RB, X['eye'])
        assert (W_r.shape == RB.shape)
    
    else:
        RB, s = POD(c_[y_list, F2], X['eye'])
    
        ax.semilogy(s)
    
        nosc_r = argmin(abs(np.asarray([norm(s[:i])/norm(s) for i in range(len(s))]) -0.99))
        if nosc_r <  20:
            nosc_r = 20  # ensure nosc_r is even
    
        X.update({'eye_r': np.eye(2*nosc_r)})
            
        print('nosc_r = %s' %nosc_r)
        
        RB = RB[:, :2*nosc_r]
        W_r = RB
            
    
    # assert that W_r and RB are orthogonal
    M = W_r.T @ RB
    assert np.allclose(M, X['eye_r'])
            
    del y_list
    gc.collect()
    
    kwds.update({'RB': RB,\
                'W_r': W_r,\
                'nosc_r': nosc_r,\
                'i_range_r': np.random.randint(0, nosc_r, 3)})
        
    MSsolvers_r = solver(kwds)
        
    print(np.amax(abs(MSsolvers[-1].y -MSsolvers_r[-1].y)))
    
    time_lapsed.append(reshape([x.time_lapsed for x in MSsolvers_r],\
                               time_lapsed[0].shape)/time_lapsed[0]*100)    

    # Hyper-reduced model
    
    if MSsolvers[-1].non_quad:
        noise = np.random.normal(0, 0, F3.shape)
        U, s = POD(X['sqrt'] @(F3+noise), X['eye'])
        U_ = LA.solve(X['sqrt'], U)
        U = U_[:, :nosc_r]
        assert np.allclose(U.T @ X['Q'] @ U, X['eye_r'])
    else:
        noise = np.random.normal(0, 0, F2.shape)
        U_, s = POD(F2+noise, X['eye'])
        U = U_[:, :nosc_r]
        
    ax.semilogy(s)
        
    P, _ = DEIM(U, plot_deim=False)
    
    # P = P[:, :2*nosc_r]
    
    kwds.update({'U': U,\
                'P': P})
    
    # POD-DEIM reduced model
    MSsolvers_dr = solver(kwds)
        
    print(np.amax(abs(MSsolvers[-1].y -MSsolvers_dr[-1].y)))
    
    time_lapsed.append(reshape([x.time_lapsed for x in MSsolvers_dr],\
                               time_lapsed[0].shape)/time_lapsed[0]*100)
    
    # tex_table('', time_lapsed)
    
    print(time_lapsed)

    # What works:
        # Results are sensitive to parameters
        # Reduced model works perfectly: efficient and accurate
        # Hyper-reduced model is efficient and accurate only for the non-conservative case.
        # Structure-preserving hyper-reduced model: the solution either doesn't converge or is highly inaccurate
        # and inefficient when it converges but the symplectic error is zero.
        # The inaccuracy in the solution of the hyper-reduced model might have to
        # do with the DEIM Hamiltonian not being close to the original Hamiltonian
        # Symplectic error is the same for methods of different orders.
        
    # Next steps:
        # Implement structure-preserving hyper-reduction
        
    # TODOs:
        # Pass Omega as a parameter
        # find derivatives symbolically 
        # Simulate with a simpler Hamiltonian
    
    
def solver(kwds):

    MSsolvers = []
    errors = {'energy': [], 'spl': []}

    for solver_class, dt, Omega2 in [(x,y,z) for x in kwds['registered_solver_classes'] for y in kwds['dt_space'] for z in kwds['Omega2_space']]:
        
        kwds.update({'solver_class': solver_class, \
                    'dt': dt, \
                    'Omega2': Omega2})
        
        MSsolver = MechSystemSolver(**kwds)
        nosc = MSsolver.nosc
                
        start = process_time()
        MSsolver.solve()
        end = process_time()
        MSsolver.time_lapsed.append(end-start)
        
        if kwds['dt_space'].size > 1 and np.allclose(MSsolver.Omega2, kwds['Omega2_space'][-1]):
             # compute errors only for fixed Omega2
            if (not MSsolver.beta) and (MSsolver.constraint_type is None):
                "Energy (Hamiltonian) is an invariant for unconstrained conservative system"
                MSsolver.eng_error =MSsolver.en_err()
                errors['energy'].append(sqrt(dt)*LA.norm(MSsolver.eng_error))
            
            if MSsolver.var:
                errors['spl'].append(sqrt(dt)*LA.norm(MSsolver.sym_error))
                
            if kwds['system_type'] == 'KdV':
                norm_y = LA.norm(MSsolver.y, axis=1)
                norm_error = log(norm_y/(np.exp(-2*MSsolver.beta*MSsolver.dt)*np.roll(norm_y, 1)))
                MSsolver.norm_error = r_[0, norm_error[1:]]
            
        if 'RB' not in kwds and MSsolver.system_type == 'oscillator':
            MSsolver.F2 = c_[np.array([MSsolver.ham_z(MSsolver.y[k,:nosc], MSsolver.y[k,nosc:])\
                                                                 for k in range(MSsolver.y.shape[0])]).T, \
                            np.array([MSsolver.ham_z(MSsolver.info[k,:nosc], MSsolver.info[k,nosc:])\
                                                                 for k in range(MSsolver.info.shape[0])]).T]
                
            MSsolver.F3 = c_[np.array([MSsolver.non_quad_z(MSsolver.y[k,:nosc],\
                                    MSsolver.y[k,nosc:]) for k in range(MSsolver.y.shape[0])]).T, \
                            np.array([MSsolver.non_quad_z(MSsolver.info[k,:nosc],\
                                    MSsolver.info[k,nosc:]) for k in range(MSsolver.info.shape[0])]).T]
        elif 'RB' not in kwds and MSsolver.system_type == 'KdV':
            MSsolver.F2 = np.array([MSsolver.ham_z(y) for y in MSsolver.info]).T
                
            MSsolver.F3 = np.array([MSsolver.non_quad_z(y) for y in MSsolver.info]).T
            
        MSsolvers.append(MSsolver)
        
    MSsolver.measures(errors, kwds['dt_space'])
        
    return MSsolvers
        
if __name__ == '__main__':
    mor_demo()
    