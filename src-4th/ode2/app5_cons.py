#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 11:49:50 2020

@author: ashishbhatt

This app solves a full order and reduced-order model from models.py
using solver classes in the ODESolver hierarchy of methods,
fixed point iterators from the Newton.py,
and bases from podDEIM.
"""

import numpy as np
from numpy import linalg as LA
from pylab import *
import ODESolver
from podDEIM import DEIM, orthogonalize, POD, cSVD
from PlotScript import tex_table, logplot
from time import process_time
import scipy as sp
from scipy.linalg import block_diag
from datetime import datetime
from models import sineGordon as model

from Newton import fixed_point

#%% Parameters class
class Params(object):
    """ Class of parameters """
    def __init__(self, **kwds):
        "Model properties"
        self.system_type = kwds['system_type']
        
        model.set_system(self, kwds)
        
        "Projection matrices"
        if 'RB' in kwds:
            self.RB = kwds['RB']
            self.nosc_r = kwds['nosc_r']
        else:
            self.RB = None

        model.set_constraints(self, kwds)
            
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
            if kwds['symplectic_mor']:
                self.JJ_r = self.JJ(self.nosc_r)
                self.reduced_model = 'symplectic_rom'
            else:
                self.JJ_r = self.W_r.T @ self.JJ() @ self.W_r
                self.reduced_model = 'nonsymplectic_rom'
        else:
            self.reduced_model = 'full'

        
        "Numerical solver and its properties"
        self.registered_solver_classes = kwds['registered_solver_classes']
            
        self.solver_class = kwds['solver_class']
        
        self.dt = kwds['dt']
         
        self.T_final = 1
        
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
    "Define the Mechanical system with parameters from the Params class"
    #TODO: have the __call__ return jacobian as well.
        
    def __call__(self, y, t):
        
        return model.get_func(self, y, t)
            

    def jacobian(self, y, t, *arg):
        "Jacobian of the function f"
        
        return model.get_jacobian(self, y, t)

    def en_err(self):
        "Energy error"
        assert self.beta == 0, 'Energy is only defined for beta = 0'
        
        return model.get_en_err(self)
                

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
            
            if self.constraint_type and self.system_type != 'KdV':
                X_U = np.array([self.y_init, self.y_init])
                X_U, _, _ = fixed_point(self._g, X_U, self._g_prime, self.tol, self.M, False)
                self.y_init = X_U[1]
            
        self.y[0] = self.y_init
        self.info = []

        for k in range(self.n):
            self.info.append(self.y[k])
            y_ = np.array([self.y[k], self.y[k]])
            
            for w_val in self.w_values:
                self.solver.set_initial_condition(y_[1])
                y_, _, info_ = self.solver.solve(w_val*self.t_points[k:k+2])
                if self.store: self.info.append(np.array(info_[0::1]))
                
                # enforce constraints
                if self.constraint_type and self.system_type == 'Oscillators':
                    
                    y_, _, _ = fixed_point(self._g, y_, self._g_prime, self.tol, self.M, False)
                elif self.constraint_type and self.system_type == 'KdV':
                    y_[1] = np.exp(-2*self.beta*self.dt) *LA.norm(self.y[k])/LA.norm(y_[1]) *y_[1]
                elif self.constraint_type and self.system_type == 'sine-Gordon':
                    y_, _, _ = fixed_point(self._g, y_, self._g_prime, self.tol, self.M, False)
                    

            self.y[k+1] = y_[1]
            
        self.info.append(self.y[k+1])
        self.info = np.vstack(self.info)

        if self.RB is not None:
            self.y_red = self.y
            self.y = (self.RB @ self.y.T).T

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
        model.plot(self, fig)


def solver(kwds):

    MSsolvers = []
    errors = {'energy': [], 'spl': []}

    for solver_class, dt, Omega2 in \
        [(x,y,z) for x in kwds['registered_solver_classes'] \
         for y in kwds['dt_space'] for z in kwds['Omega2_space'] \
             ]:
        
        kwds.update({'solver_class': solver_class, \
                    'dt': dt, \
                    'Omega2': Omega2})
        
        MSsolver = MechSystemSolver(**kwds)
        nosc = MSsolver.nosc
                
        start = process_time()
        MSsolver.solve()
        end = process_time()
        MSsolver.time_lapsed.append(end-start)
        
        if MSsolver.var == True:
            MSsolver.var_solve()
        
        if kwds['dt_space'].size > 1 and np.allclose(MSsolver.Omega2, kwds['Omega2_space'][-1]):
             # compute errors only for fixed Omega2
             
            model.get_errors(MSsolver, errors, dt, kwds)
            
            
        if 'RB' not in kwds and MSsolver.system_type in ['Oscillators', 'sine-Gordon']:
            MSsolver.F2 = np.array([MSsolver.ham_z(*np.split(y,2)) for y in MSsolver.info]).T
                
            if MSsolver.non_quad:
                MSsolver.F3 = np.array([MSsolver.non_quad_z(*np.split(y,2)) for y in MSsolver.info]).T
            
        elif 'RB' not in kwds:
            MSsolver.F2 = np.array([MSsolver.ham_z(y) for y in MSsolver.info]).T
                
            if MSsolver.non_quad:
                MSsolver.F3 = np.array([MSsolver.non_quad_z(y) for y in MSsolver.info]).T
            
        MSsolvers.append(MSsolver)
        
    MSsolver.measures(errors, kwds['dt_space'])
        
    return MSsolvers

#%% Main driver function
# def mor_demo():
"Model order reduction of the MechSystem using MechSystemSolver"

registered_solver_classes, nosc = [ODESolver.ConformalImplicitMidpoint], 200
num_solver_classes = len(registered_solver_classes)
    
dt_space_dim = 5
    
i_range = np.random.randint(0, nosc-3, 1)

kwds = {'model': model,\
        'system_type': 'sine-Gordon',\
        'symplectic_mor': False}

if kwds['system_type'] == 'Oscillators':
        
    Omega2_space_dim = 3
    Omega2_space = 1 +np.random.rand(Omega2_space_dim, nosc)/1000
    Omega2_space.sort()
    
    kwds.update({'nosc': nosc, \
                'dt_space': linspace(0.05, 0.1, num=dt_space_dim), \
                'Omega2_space': Omega2_space, \
                'registered_solver_classes': registered_solver_classes, \
                'i_range': np.append(i_range, [i_range+1, i_range+2])})

elif kwds['system_type'] == 'KdV':
    Omega2_space_dim = 1
    kwds.update({'dt_space': linspace(0.001, 0.009, num=dt_space_dim), \
                'Omega2_space': np.ones((Omega2_space_dim,nosc)), \
                'registered_solver_classes': registered_solver_classes, \
                'i_range': np.append(i_range, [i_range+1, i_range+2])})

elif kwds['system_type'] == 'sine-Gordon':
    Omega2_space_dim = 1
    kwds.update({'dt_space': linspace(0.01, 0.05, num=dt_space_dim), \
                'Omega2_space': np.ones((Omega2_space_dim,nosc)), \
                'registered_solver_classes': registered_solver_classes, \
                'i_range': np.append(i_range, [i_range+1, i_range+2]), \
                })
    
MSsolvers = solver(kwds)

if MSsolvers[-1].non_quad: # is not None
    assert Omega2_space_dim == 1

time_lapsed = [reshape([x.time_lapsed for x in MSsolvers],\
                       (num_solver_classes, dt_space_dim, Omega2_space_dim))]
    

#%%
y_list = np.hstack([MSsolver.info.T for MSsolver in MSsolvers])
F2 = np.hstack([MSsolver.F2 for MSsolver in MSsolvers])

nosc = MSsolvers[-1].nosc
X = {'eye': np.eye(2*nosc)}

fig = figure()
ax = fig.add_subplot(111)
solution_error = []

for nosc_r_ in [20]: #[10, 15, 20, 25, 30]:
        
    model.get_rb(y_list, F2, MSsolvers, X, kwds, nosc_r_, ax)
    
    # assert that W_r and RB are orthogonal
    M = kwds['W_r'].T @ kwds['RB']
    assert np.allclose(M, X['eye_r'])
    
    MSsolvers_r = solver(kwds)
        
    solution_error.append([np.amax(abs(MSsolvers[i].y -MSsolvers_r[i].y)) for i in range(len(MSsolvers_r))])
    
    time_lapsed.append(reshape([x.time_lapsed for x in MSsolvers_r],\
                               time_lapsed[0].shape)/time_lapsed[0]*100) 

print(time_lapsed)
print(solution_error)

#%% Hyper-reduced model
'''
if MSsolvers[-1].non_quad:
    F3 = np.hstack([MSsolver.F3 for MSsolver in MSsolvers])
    noise = np.random.normal(0, 0, F3.shape)
    U, s = POD(X['sqrt'] @(F3+noise), X['eye'])
    U_ = LA.solve(X['sqrt'], U)
    U = U_[:, :2*nosc_r]
    assert np.allclose(U.T @ X['Q'] @ U, X['eye_r'])
else:
    noise = np.random.normal(0, 0, F2.shape)
    U_, s = POD(F2+noise, X['eye'])
    U = U_[:, :2*nosc_r]
    
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
'''

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
        
# if __name__ == '__main__':
#     mor_demo()
    