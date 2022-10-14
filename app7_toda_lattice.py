#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 12:40:31 2022

@author: bhattah

This app solves the Toda lattice using symplectic and conformal symplectic integrators.
"""

import numpy as np
from numpy import linalg as LA
from pylab import r_, c_
from mpl_toolkits import mplot3d
#%matplotlib inline
import matplotlib.pyplot as plt
from scipy.sparse import block_diag, identity, bmat, diags
import ODESolver
from scipy import optimize

from Newton import Newton

#%% define the system

class TodaLattice(object):
    def __init__(self, **kwargs):
        self.N = 100
            
        self.beta = 0.1
        self.gamma = self.beta*np.ones((self.N,))
        
        self.dt = 0.1
        self.t_final = 100
        self.t_points = np.arange(0, self.t_final, self.dt)
        self.t_size = len(self.t_points)
        
        self.z0 = np.zeros((1,2*self.N))
        self.z = np.zeros((self.t_size+1, 2*self.N))
        self.z[0] = self.z0
        
        self.solver_class = kwargs['solver_class']
        self.solver = self.solver_class(self, self.jacobian)
        
        if self.solver_class == ODESolver.ImplicitMidpoint:
            beta = 0
        elif self.solver_class == ODESolver.ConformalImplicitMidpoint:
            beta = self.beta
        
        self.ham = lambda q, p: 1/2 *np.dot(p,p) +np.sum(np.exp(-np.diff(q))) +np.exp(q[-1]) -q[0] -self.N +beta/2*q.dot(p)
        self.ham_q = lambda q, p: np.exp(r_[-np.diff(q), q[-1]]) - np.exp(r_[0, -np.diff(q)]) +beta/2*p
        self.ham_p = lambda q, p: p +beta/2*q
        self.ham_z = lambda q, p: r_[self.ham_q(q, p), self.ham_p(q, p)]
        
        self.ham_qq = lambda q: diags([-np.exp(-np.diff(q)),\
                                       np.exp(r_[-np.diff(q), q[-1]]) +r_[0, np.exp(-np.diff(q))],\
                                      -np.exp(-np.diff(q))], [-1,0,1], shape=(self.N,)*2)
        self.ham_pp = lambda p: identity(self.N)
        self.ham_zz = lambda q, p: bmat([[self.ham_qq(q), beta/2*identity(self.N)],\
                                      [beta/2*identity(self.N), self.ham_pp(p)]]).toarray()
        # self.ham_zz = lambda q, p: block_diag((self.ham_qq(q), self.ham_pp(p)))
        
        self.JJ = lambda d=self.N: bmat([[None, identity(d)],[-identity(d), None]]).toarray()
        
        self.R = block_diag((np.zeros((self.N,)*2), diags(self.gamma))).toarray()
        
        self.B = np.zeros((2*self.N,))
        self.B[self.N] = 1
        # self.B = bmat(self.B)
        
        # self.u = lambda t: 0.1
        self.u = lambda t: 0.1 *np.sin(t)
        
        self.block_mat = bmat([[None, identity(self.N)],[identity(self.N), None]]).toarray()
        
        self.constraint_type = 'linear'
        self.tol_iter = 1e-10
        self.max_iter = 100
        self.store_iter = False

    "z0 alias"
    @property
    def u_init(self):
        return self.z0

    @u_init.setter
    def u_init(self, value):
        self.z0 = value
        
    def __call__(self, z, t):
        q, p = np.split(z.reshape(-1), 2)
        if self.solver_class == ODESolver.ImplicitMidpoint:
            return (self.JJ() -self.R) @self.ham_z(q, p) +self.B *self.u(t)
        elif self.solver_class == ODESolver.ConformalImplicitMidpoint:
            return self.JJ() @self.ham_z(q, p) +self.B *self.u(t)
    
    def jacobian(self, z, *arg):
        q, p = np.split(z.reshape(-1), 2)
        if self.solver_class == ODESolver.ImplicitMidpoint:
            return (self.JJ() -self.R) @self.ham_zz(q, p)
        elif self.solver_class == ODESolver.ConformalImplicitMidpoint:
            return self.JJ() @self.ham_zz(q, p)

    def output(self, z, t):
        q, p = np.split(z.reshape(-1), 2)
        if self.solver_class == ODESolver.ImplicitMidpoint:
            return r_[self.B.dot(self.ham_z(q, p)), self.B.dot(self.ham_zz(q, p)).dot(self(z,t))]\
                .dot(r_[self.B.dot(self.ham_z(q, p)), self.B.dot(self.ham_zz(q, p)).dot(self(z,t))])
        if self.solver_class == ODESolver.ConformalImplicitMidpoint:
            return r_[self.B.dot(self.ham_z(q, p) -self.beta/2 *r_[p, q]), \
                      self.B.dot(self.ham_zz(q, p) -self.beta/2 *self.block_mat)\
                        .dot((self.JJ() -self.R) @(self.ham_z(q, p) -self.beta/2 *r_[p, q]) +self.B *self.u(t))]

    def output_jacobian(self, z):
        q, p = np.split(z.reshape(-1), 2)
        if self.solver_class == ODESolver.ImplicitMidpoint:
            return 2*self.B.dot(self.ham_zz(q,p)) + 2*self.B.dot(self.ham_zz(q, p)).dot(self.jacobian(z))
        if self.solver_class == ODESolver.ConformalImplicitMidpoint:
            return block_diag((self.B.dot(self.ham_zz(q, p) -self.beta/2 *self.block_mat),\
                    self.B.dot(self.ham_zz(q, p) -self.beta/2 *self.block_mat)\
                        .dot((self.JJ() -self.R) @(self.ham_zz(q, p) -self.beta/2 *self.block_mat)))).toarray()
    
    def var_solve(self):
            
        self.dpsi = np.zeros((2, 2*self.N, 2*self.N))
        self.dpsi[0] = np.eye(2*self.N)
        self.sym_error = np.zeros(self.t_size)
        
        for k in range(self.t_size-1):
            self.solver.set_initial_condition(self.z[k])
            dpsi_, _ = self.solver.var_solve(self.z[k:k+2], self.t_points[k:k+2])
            self.dpsi[1] = dpsi_[1]
            
            sym_error_ =self.solver.symplectic_error(self.dpsi, self.t_points[k:k+2])
            self.sym_error[k+1] = sym_error_[1]
    
    def solve(self):
        
        for k in range(self.t_size-1):
            self.solver.set_initial_condition(self.z[k])
            z_, _ = self.solver.solve(self.t_points[k:k+2])
                
            # Apply constraints
            if self.constraint_type in ['linear']:
                
                # z_, _, _ = Newton(lambda z: self.output(z, self.t_points[k+1]), z_[1], self.output_jacobian, self.tol_iter, self.max_iter, self.store_iter)
                sol = optimize.minimize(lambda z: self.output(z, self.t_points[k+1]), z_[1], method='L-BFGS-B', jac=self.output_jacobian, tol=1e-10)
                z_[1] = sol.x
                
            elif self.constraint_type == None:
                pass
            else:
                raise ValueError('Unknown constraints')
                    
            self.z[k+1] = z_[1]
            
    def plot(self):
        out_ = [self.output(self.z[i], self.t_points[i]) for i in range(self.t_size)]
        fig = plt.figure()
        ax = plt.axes()
        ax.plot(out_)
        
        fig = plt.figure()
        ax = plt.axes()
        ax.plot(self.sym_error)
        
        # if np.allclose(self.B, np.zeros_like(self.B)) and np.allclose(self.R, np.zeros_like(self.R)):
        ham_ = [self.ham(self.z[i][:self.N], self.z[i][self.N:]) for i in range(self.t_size)]
        fig = plt.figure()
        ax = plt.axes()
        ax.plot(ham_)
            
#%% driver script
registered_solver_classes = [ODESolver.ImplicitMidpoint]

for solver_class in registered_solver_classes:
    kwargs = {'solver_class': solver_class}
    sys = TodaLattice(**kwargs)
    sys.solve()
    sys.var_solve()
    sys.plot()
    
    # The currently used constraint (output) is not being preserved, look for a another more suitable constraint, preferrably physically meanigful.
    # 
    