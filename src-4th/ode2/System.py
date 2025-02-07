#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 12:40:39 2022

@author: bhattah
"""

import numpy as np
from pylab import log, r_, c_, zeros, eye, linspace
import ODESolver #import ConformalImplicitMidpoint, ConformalStormerVerlet
from datetime import datetime
import os
import dill as pickle
import sympy as smp

#%% System definition
class MechSystem(object):
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
    nosc = 12*3
    assert nosc//3 % 2 == 0, 'nosc//3 must be even'

    # These two properties only have effect during reduction
    reducer = 'psd'
    predict = True # False = reproduce
    hyperreducer = 'MDEIM'

    dt_space_dim = 1
    beta = (max(1e-2, 0*np.random.rand()/10))*0
    JJ = lambda self, d=nosc: r_[c_[zeros((d,d)), eye(d)], c_[-eye(d), zeros((d,d))]]

    dt_space = linspace(0.01, 0.05, num=dt_space_dim)
    T_final = 5

    keep_time = datetime.now().strftime('%Y-%m-%d_%H-%M_')

    "Initial conditions satisfying the constraints"
    # Create a tensor to store the positions
    i = np.arange(nosc//3)
    positions = np.stack((i % 2 + 0*(i // 2 % 2) * 1e-1, i // 2 + 0*(-1)**(i // 2) * 1e-1, np.zeros_like(i)), axis=1)

    positions = positions.flatten().reshape(-1, 1)

    momenta = np.zeros((nosc//3, 3))
    momenta[::2] = np.random.uniform(-0.01, 0.01, momenta[::2].shape)
    momenta[1::2] = momenta[::2]
    assert np.allclose(momenta[1::2] - momenta[::2], 0), 'position and momenta are not orthogonal'
    momenta = momenta.flatten().reshape(-1, 1)

    constraint_type = 'spherical'
    constraints_reduce = True # True: Reduce constraint jacobian g_prime

    y_init = r_[positions, momenta].flatten()

    drag = lambda self, y: 0 #beta/2 * r_[x, u]
    drag_z = lambda self, y: 0 #beta/2 * eye(2*x.shape[0])


    def __init__(self, kwds):
        
        self.__dict__.update(kwds)
        
        if 'pool' in kwds:
            self.__dict__.update(kwds['pool'])
        
        # if hasattr(self, 'nosc_r') and self.solver_class in [ODESolver.DiscreteGradient]:
        #     self.RB = kwds['RB_dg']
        #     self.nosc_r = kwds['nosc_r_dg']

        "MechSystem constituents"
        self.ham = lambda y, Omega2=self.Omega2, beta=self.beta: self.ham_(y, Omega2, beta)
        self.ham_z = lambda y, Omega2=self.Omega2, beta=self.beta: self.ham_z_(y, Omega2, beta)
        self.ham_zz = lambda y, Omega2=self.Omega2, beta=self.beta: self.ham_zz_(y, Omega2, beta)

        self.non_quad = None
        
        if hasattr(self, 'lag_dg_'): # if Lagrangian is defined
            self.lag_dg = lambda y, Omega2=self.Omega2: self.lag_dg_(y, Omega2)
            self.lag_dg_z = lambda y, Omega2=self.Omega2: self.lag_dg_z_(y, Omega2)
            
        "Projection matrices"
        if hasattr(self, 'RB'):
            self.y_init = self.RB.T @ self.y_init
            
            if self.reducer == 'psd':
                self.JJ = self.JJ(self.nosc_r)
        else:
            self.RB = None
            self.JJ = self.JJ(self.nosc)

        if not hasattr(self, 'P'):
            self.P = None
            self.U = None
            
        "various measures"
        self.time_lapsed = []

    "y_init alias"
    @property
    def u_init(self):
        return self.y_init

    @u_init.setter
    def u_init(self, value):
        self.y_init = value
         
            
    def __call__(self, y, t, *y1, **kwargs):
            
        if self.solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
            f = self.JJ @ self.ham_z(y) - self.drag(y)
            dfdy = self.JJ @ self.ham_zz(y) - self.drag_z(y)
    
        elif self.solver_class in [ODESolver.ConformalImplicitMidpoint]:
            f = self.JJ @ self.ham_z(y)
            dfdy = self.JJ @ self.ham_zz(y)
            
        elif self.solver_class in [ODESolver.ConformalStormerVerlet]:
            f = self.JJ @ self.ham_z(y, self.Omega2, 0)
            dfdy = self.JJ @ self.ham_zz(y, self.Omega2, 0)
            
        elif self.solver_class in [ODESolver.DiscreteGradient]:
            f = self.lag_dg(y)
            dfdy = self.lag_dg_z(y)
            
        if kwargs['func']:
            return f
        else:
            return dfdy
                
    def get_en_err(self):
            
        return np.array([self.ham(y) for y in self.y]) - self.ham(self.y[0])
    
#%% Find symbolic quantities

# TODO: move inside the system class, profile it
nosc = MechSystem.nosc

# create the 'data' subfolder if it doesn't exist
data_folder = 'data'
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

# try to load the expressions from disk
filename = os.path.join(data_folder, f"ham_expr_{nosc}.pickle")
try:
    with open(filename, 'rb') as f:
        loaded_expressions = pickle.load(f)


    for key, value in loaded_expressions.items():
        exec(f"{key} = value")
    
    print("Loaded Hamiltonian expressions from disk.")

except: #FileNotFoundError or AttributeError:
    print("Hamiltonian expressions not found on disk. Computing and saving them...")

    y = smp.Matrix(smp.symbols('y_:{}_:{}'.format(nosc*2//3,3), real=True))
    q = smp.Matrix(y[:nosc]).reshape(nosc//3,3)
    p = smp.Matrix(y[nosc:])
    omega2 = smp.Matrix(smp.symbols('omega^2_:{}'.format(nosc//3-2), real=True))
    beta = smp.symbols('beta', real=True)

    kin_expr = 0.5 * p.dot(p)

    pi = smp.Matrix([(q.row(i+2) - q.row(i)).norm(2)**2 for i in range(nosc//3-2)])
    pot_expr_vec = 0.5 * omega2.multiply_elementwise((pi -smp.ones(nosc//3-2,1)).applyfunc(lambda x: x**2))
    pot_expr = sum(pot_expr_vec) #0.5 * omega2.dot((pi -smp.ones(nosc//3-2,1)).applyfunc(lambda x: x**2))

    ham_expr = kin_expr + pot_expr
    ham_z_expr = smp.Matrix([ham_expr]).jacobian(y).T
    ham_zz_expr = ham_z_expr.jacobian(y)

    ham_ = smp.lambdify((y, omega2, beta), ham_expr, 'numpy')
    _ham_z_ = smp.lambdify((y, omega2, beta), ham_z_expr, 'numpy')
    ham_zz_ = smp.lambdify((y, omega2, beta), ham_zz_expr, 'numpy')

    ham_z_ = lambda y, omega2, beta: _ham_z_(y, omega2, beta).squeeze()
    
    # Find discrete derivatives
    q = q.reshape(nosc,1)
    y1 = smp.Matrix(smp.symbols('y1_:{}_:{}'.format(nosc*2//3,3), real=True))
    
    # Update the replacement dictionaries
    y05_repl = dict(zip(y, (y + y1) / 2))
    y1_repl = dict(zip(y, y1))
    
    dpi_dq = pi.jacobian(q).subs(y05_repl)
    # dV_dpi = smp.Matrix([smp.factor((pot_expr_vec.subs(y1_repl) - pot_expr_vec)[i]/(pi.subs(y1_repl) - pi)[i]) for i in range(nosc//3-2)])
    dV_dpi = ((pot_expr_vec.subs(y1_repl) - pot_expr_vec).multiply_elementwise((pi.subs(y1_repl) - pi).applyfunc(lambda x: 1/x))).applyfunc(smp.factor)
    DG_V_expr = dpi_dq.T @ dV_dpi #smp.Matrix(np.sum([dV_dpi[i] * dpi_dq[i,:] for i in range(nosc//3-2)], axis=0)[0])
    
    lag_dg_expr = smp.Matrix.vstack(ham_z_expr[nosc:,:].subs(y05_repl), -DG_V_expr)
    _lag_dg_ = smp.lambdify((y, y1, omega2), lag_dg_expr, 'numpy')
    lag_dg_ = lambda y, omega2: _lag_dg_(*y, omega2).squeeze()
    
    # TODO: check math
    lag_dg_z_expr = lag_dg_expr.jacobian(y1)
    _lag_dg_z_ = smp.lambdify((y, y1, omega2), lag_dg_z_expr, 'numpy')
    lag_dg_z_ = lambda y, omega2: _lag_dg_z_(*y, omega2)

    expressions = {
        "ham_expr": ham_expr,
        "ham_z_expr": ham_z_expr,
        "ham_zz_expr": ham_zz_expr,
        "_ham_z_": _ham_z_,
        "ham_": ham_,
        "ham_z_": ham_z_,
        "ham_zz_": ham_zz_,
        "DG_V_expr": DG_V_expr,
        "lag_dg_expr": lag_dg_expr,
        "_lag_dg_": _lag_dg_,
        "lag_dg_": lag_dg_,
        "lag_dg_z_expr": lag_dg_z_expr,
        '_lag_dg_z_': _lag_dg_z_,
        "lag_dg_z_": lag_dg_z_,
        "q": q, "p": p, "omega2": omega2, "beta": beta,
        "y": y, "y1": y1,
        }

    # save the expressions to disk
    with open(filename, 'wb') as f:
        pickle.dump(expressions, f)
    print("Hamiltonian expressions saved to disk.")

#%%
if MechSystem.constraint_type is not None:

    # try to load the expressions from disk
    filename = os.path.join(data_folder, f"g_expr_{nosc}.pickle")
    try:
        with open(filename, 'rb') as f:
            loaded_expressions = pickle.load(f)

        for key, value in loaded_expressions.items():
            exec(f"{key} = value")
            
        print("Loaded constraints from disk.")

    except: #FileNotFoundError:
        print("Constraints not found on disk. Computing and saving them...")

        y = smp.Matrix(smp.symbols('y_:{}_:{}'.format(nosc*2//3,3), real=True))
        q = smp.Matrix(y[:nosc]).reshape(nosc//3,3)
        p = smp.Matrix(y[nosc:]).reshape(nosc//3,3)
        
        row_diffs_q = [q.row((i+1)) - q.row(i) for i in range(0, nosc//3, 2)]
        row_diffs_p = [p.row((i+1)) - p.row(i) for i in range(0, nosc//3, 2)]
        row_norms = [(row_diff.dot(row_diff) -1)/2 for row_diff in row_diffs_q]
        ddt_row_norms = [row_diffs_q[i].dot(row_diffs_p[i]) for i in range(len(row_diffs_q))]

        q = q.reshape(nosc,1)
        p = p.reshape(nosc,1)
        g_expr = smp.Matrix(row_norms+ ddt_row_norms)
        g_prime_expr = g_expr.jacobian(y)

        _g_lam = smp.lambdify((y,), g_expr, modules=['numpy'])
        g_prime_lam = smp.lambdify((y,), g_prime_expr, modules=['numpy'])
        
        g_lam = lambda x: _g_lam(x).squeeze()
        
        expressions = {
            "g_expr": g_expr,
            "g_prime_expr": g_prime_expr,
            "_g_lam": _g_lam,
            "g_lam": g_lam,
            "g_prime_lam": g_prime_lam,
        }

        # save the expressions to disk
        with open(filename, 'wb') as f:
            pickle.dump(expressions, f)
        print("Constraints saved to disk.")

kwds = {'ham_': ham_, \
        '_ham_z_': _ham_z_, \
        'ham_z_': ham_z_, \
        'ham_zz_': ham_zz_, \
        '_g_lam': _g_lam, \
        'g': g_lam, \
        'g_prime': g_prime_lam, \
        "_lag_dg_": _lag_dg_,
        "lag_dg_": lag_dg_,
        "lag_dg_z_": lag_dg_z_,
        }
    
_Omega2_space_dim = 5
_Omega2_space = np.sort(10*(1 -np.random.rand(_Omega2_space_dim, nosc//3-2)))
        
#%% main
if __name__ == '__main__':
    pass
