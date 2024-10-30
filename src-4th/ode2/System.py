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

#%%
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
    nosc = 50*3
    assert nosc//3 % 2 == 0, 'nosc//3 must be even'

    # These two properties only have effect during reduction
    reducer = 'psd'
    predict = False # False = reproduce
    hyperreducer = 'MDEIM'

    dt_space_dim = 5
    beta = (max(1e-2, 0*np.random.rand()/10))*0
    JJ = lambda self, d=nosc: r_[c_[zeros((d,d)), eye(d)], c_[-eye(d), zeros((d,d))]]

    dt_space = linspace(0.01, 0.05, num=dt_space_dim)
    T_final = 2

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

    drag = lambda self, x, u: 0 #beta/2 * r_[x, u]
    drag_z = lambda self, x, u: 0 #beta/2 * eye(2*x.shape[0])


    def __init__(self, kwds):
        
        self.__dict__.update(kwds)
        
        if 'pool' in kwds:
            self.__dict__.update(kwds['pool'])

        "MechSystem constituents"
        self.ham = lambda x, u, Omega2=self.Omega2, beta=self.beta: self.ham_(x, u, Omega2, beta)
        self.ham_z = lambda x, u, Omega2=self.Omega2, beta=self.beta: self.ham_z_(x, u, Omega2, beta)
        self.ham_zz = lambda x, u, Omega2=self.Omega2, beta=self.beta: self.ham_zz_(x, u, Omega2, beta)

        self.non_quad = None

        "Projection matrices"
        if hasattr(self, 'RB'):
            self.y_init = self.RB.T @ self.y_init

            if self.reducer == 'pod':
                self.JJ_r = self.RB.T @ self.JJ() @ self.RB
            elif self.reducer == 'psd':
                self.JJ_r = self.JJ(self.nosc_r)
        else:
            self.RB = None

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
         
            
    def __call__(self, y, t, *args, **kwargs):
        
        solver_class = self.solver_class
        
        if self.RB is None:
            x, u = np.split(y, 2)
            JJ = self.JJ()
        else:
            x, u = np.split(self.RB @ y, 2)
            JJ = self.JJ_r
            
        if kwargs['func']:
            if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                f = JJ @ self.ham_z(x, u) - self.drag(*np.split(y, 2))
        
            elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                f = JJ @ self.ham_z(x, u)
                
            elif solver_class in [ODESolver.ConformalStormerVerlet]:
                f = JJ @ self.ham_z(x, u, self.Omega2, 0)
                
            return f
                    
        if kwargs['jac']:
            if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                dfdy = JJ @ self.ham_zz(x, u) - self.drag_z(*np.split(y, 2))
        
            elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                dfdy = JJ @ self.ham_zz(x, u)
                
            elif solver_class in [ODESolver.ConformalStormerVerlet]:
                dfdy = JJ @ self.ham_zz(x, u, self.Omega2, 0)
                
            return dfdy
                
    def get_en_err(self):
            
        x, u = np.split(self.y.T, 2, axis=0)
            
        if self.non_quad and self.P is not None:
            return log(\
                        (self.non_quad(x, u) +1/2 *self.y @ self.Q_spd() @self.y.T)\
                        /(self.non_quad(x[0:1, :], u[None, 0, :]) +1/2 *self.y @ self.Q_spd() @self.y.T)\
                        )
        else:
            return self.ham(x, u) - self.ham(x[:, 0], u[:, 0])
    
#%% Find symbolic quantities

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
    print("Loaded Hamiltonian expressions from disk.")

    ham_expr = loaded_expressions['ham_expr']
    ham_z_expr = loaded_expressions['ham_z_expr']
    ham_zz_expr = loaded_expressions['ham_zz_expr']
    _ham_z_ = loaded_expressions['_ham_z_']
    ham_ = loaded_expressions['ham_']
    ham_z_ = loaded_expressions['ham_z_']
    ham_zz_ = loaded_expressions['ham_zz_']
    q = loaded_expressions['q']
    p = loaded_expressions['p']
    omega2 = loaded_expressions['omega2']
    beta = loaded_expressions['beta']
    y = loaded_expressions['y']

except: #FileNotFoundError or AttributeError:
    print("Hamiltonian expressions not found on disk. Computing and saving them...")

    q = smp.Matrix(smp.symbols('q_:{}_:{}'.format(nosc//3,3), real=True)).reshape(nosc//3,3)
    p = smp.Matrix(smp.symbols('p_:{}_:{}'.format(nosc//3,3), real=True))
    omega2 = smp.Matrix(smp.symbols('omega^2_0:{}'.format(nosc//3-2), real=True))
    beta = smp.symbols('beta', real=True)

    kin_expr = 0.5 * p.dot(p)

    pi = smp.Matrix([(q.row(i+2) - q.row(i)).norm(2)**2 for i in range(nosc//3-2)])
    pot_expr = 0.5 * omega2.dot((pi -smp.ones(nosc//3-2,1)).applyfunc(lambda x: x**2))

    q = q.reshape(nosc,1)
    y = list(q)+list(p)

    ham_expr = kin_expr + pot_expr
    ham_z_expr = smp.Matrix([ham_expr]).jacobian(y).T
    ham_zz_expr = ham_z_expr.jacobian(y)

    ham_ = smp.lambdify((q, p, omega2, beta), ham_expr, 'numpy')
    _ham_z_ = smp.lambdify((q, p, omega2, beta), ham_z_expr, 'numpy')
    ham_zz_ = smp.lambdify((q, p, omega2, beta), ham_zz_expr, 'numpy')

    ham_z_ = lambda q, p, omega2, beta: _ham_z_(q, p, omega2, beta).squeeze()

    expressions = {
        "ham_expr": ham_expr,
        "ham_z_expr": ham_z_expr,
        "ham_zz_expr": ham_zz_expr,
        "_ham_z_": _ham_z_,
        "ham_": ham_,
        "ham_z_": ham_z_,
        "ham_zz_": ham_zz_,
        "q": q, "p": p, "omega2": omega2, "beta": beta, "y": y,
    }

    # save the expressions to disk
    with open(filename, 'wb') as f:
        pickle.dump(expressions, f)
    print("Hamiltonian expressions saved to disk.")

if MechSystem.constraint_type is not None:

    # try to load the expressions from disk
    filename = os.path.join(data_folder, f"g_expr_{nosc}.pickle")
    try:
        with open(filename, 'rb') as f:
            loaded_expressions = pickle.load(f)
        print("Loaded constraints from disk.")

        g_expr = loaded_expressions['g_expr']
        g_prime_expr = loaded_expressions['g_prime_expr']
        _g_lam = loaded_expressions['_g_lam']
        g_lam = loaded_expressions['g_lam']
        g_prime_lam = loaded_expressions['g_prime_lam']

    except: #FileNotFoundError:
        print("Constraints not found on disk. Computing and saving them...")

        q = q.reshape(nosc//3,3)
        p = p.reshape(nosc//3,3)
        row_diffs_q = [q.row((i+1)) - q.row(i) for i in range(0, nosc//3, 2)]
        row_diffs_p = [p.row((i+1)) - p.row(i) for i in range(0, nosc//3, 2)]
        row_norms = [(row_diff.dot(row_diff) -1)/2 for row_diff in row_diffs_q]
        ddt_row_norms = [row_diffs_q[i].dot(row_diffs_p[i]) for i in range(len(row_diffs_q))]

        q = q.reshape(nosc,1)
        p = p.reshape(nosc,1)
        y = list(q)+list(p)
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
            "g_prime_lam": g_prime_lam
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
        }
    
_Omega2_space_dim = 1
_Omega2_space = np.sort(10*(1 -np.random.rand(_Omega2_space_dim, nosc//3-2)))
        
#%%
if __name__ == '__main__':
    pass
