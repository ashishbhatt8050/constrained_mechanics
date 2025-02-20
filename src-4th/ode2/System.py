#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 12:40:39 2022

@author: bhattah
"""

import numpy as np
from pylab import r_, c_, zeros, eye, linspace
from datetime import datetime
import os
import dill as pickle
import sympy as smp
from functools import wraps

class SymbolicComputer:
    """Class to compute and store symbolic expressions"""
    def __init__(self, nosc):
        self.nosc = nosc
        # Common symbolic variables
        self.y = smp.Matrix(smp.symbols('y_:{}_:{}'.format(nosc*2//3,3), real=True))
        self._q = smp.Matrix(self.y[:nosc])
        self.q = self._q.reshape(nosc//3,3)
        # Store both forms of p
        self._p = smp.Matrix(self.y[nosc:])
        self.p = self._p.reshape(nosc//3,3)
        # Additional variables for discrete derivatives
        self.y1 = smp.Matrix(smp.symbols('y1_:{}_:{}'.format(nosc*2//3,3), real=True))
        self.y05_repl = dict(zip(self.y, (self.y + self.y1) / 2))
        self.y1_repl = dict(zip(self.y, self.y1))
        
    def compute_hamiltonian(self):
        """Compute Hamiltonian expressions"""
        print("Computing Hamiltonian expressions...")
        
        omega2 = smp.Matrix(smp.symbols('omega^2_:{}'.format(self.nosc//3-2), real=True))
        beta = smp.symbols('beta', real=True)

        kin_expr = 0.5 * self._p.dot(self._p)
        
        pi = smp.Matrix([(self.q.row(i+2) - self.q.row(i)).norm(2)**2 
                         for i in range(self.nosc//3-2)])
        pot_expr_vec = 0.5 * omega2.multiply_elementwise(
            (pi - smp.ones(self.nosc//3-2,1)).applyfunc(lambda x: x**2))
        pot_expr = sum(pot_expr_vec)

        ham_expr = kin_expr + pot_expr
        ham_z_expr = smp.Matrix([ham_expr]).jacobian(self.y).T
        ham_zz_expr = ham_z_expr.jacobian(self.y)

        ham_ = smp.lambdify((self.y, omega2, beta), ham_expr, 'numpy')
        _ham_z_ = smp.lambdify((self.y, omega2, beta), ham_z_expr, 'numpy')
        ham_zz_ = smp.lambdify((self.y, omega2, beta), ham_zz_expr, 'numpy')
        ham_z_ = lambda y, omega2, beta: _ham_z_(y, omega2, beta).squeeze()

        return {
            'y': self.y, 'y1': self.y1,
            'omega2': omega2,
            'beta': beta,
            'ham_expr': ham_expr,
            'ham_z_expr': ham_z_expr,
            'ham_zz_expr': ham_zz_expr,
            'ham_': ham_,
            'ham_z_': ham_z_,
            'ham_zz_': ham_zz_,
            'pi': pi,
            'pot_expr_vec': pot_expr_vec
        }

    def compute_lagrangian(self, ham_exprs):
        """Compute Lagrangian expressions"""
        print("Computing Lagrangian expressions...")
        
        # Discrete derivatives
        dpi_dq = ham_exprs['pi'].jacobian(self._q).subs(self.y05_repl)
        dV_dpi = ((ham_exprs['pot_expr_vec'].subs(self.y1_repl) - ham_exprs['pot_expr_vec']).multiply_elementwise(
            (ham_exprs['pi'].subs(self.y1_repl) - ham_exprs['pi']).applyfunc(lambda x: 1/x)).applyfunc(smp.factor))
        DG_V_expr = dpi_dq.T @ dV_dpi
        
        lag_dg_expr = smp.Matrix.vstack(ham_exprs['ham_z_expr'][self.nosc:,:].subs(self.y05_repl), -DG_V_expr)
        _lag_dg_ = smp.lambdify((self.y, self.y1, ham_exprs['omega2']), lag_dg_expr, 'numpy')
        lag_dg_ = lambda y, omega2: _lag_dg_(*y, omega2).squeeze()
        
        lag_dg_z_expr = lag_dg_expr.jacobian(self.y1)
        _lag_dg_z_ = smp.lambdify((self.y, self.y1, ham_exprs['omega2']), lag_dg_z_expr, 'numpy')
        lag_dg_z_ = lambda y, omega2: _lag_dg_z_(*y, omega2)

        return {
            'lag_dg_expr': lag_dg_expr,
            'lag_dg_': lag_dg_,
            'lag_dg_z_expr': lag_dg_z_expr,
            'lag_dg_z_': lag_dg_z_
        }

    def compute_constraints(self):
        """Compute constraint expressions"""
        print("Computing constraint expressions...")
        
        row_diffs_q = [self.q.row((i+1)) - self.q.row(i) 
                       for i in range(0, self.nosc//3, 2)]
        row_diffs_p = [self.p.row((i+1)) - self.p.row(i) 
                       for i in range(0, self.nosc//3, 2)]
        row_norms = [(row_diff.dot(row_diff) -1)/2 for row_diff in row_diffs_q]
        ddt_row_norms = [row_diffs_q[i].dot(row_diffs_p[i]) 
                        for i in range(len(row_diffs_q))]

        g_expr = smp.Matrix(row_norms + ddt_row_norms)
        g_prime_expr = g_expr.jacobian(self.y)

        _g_lam = smp.lambdify((self.y,), g_expr, modules=['numpy'])
        g_prime_lam = smp.lambdify((self.y,), g_prime_expr, modules=['numpy'])
        g_lam = lambda x: _g_lam(x).squeeze()
        
        return {
            'g_expr': g_expr,
            'g_prime_expr': g_prime_expr,
            'g_': g_lam,
            'g_prime_': g_prime_lam
        }
    
    def compute_all(self):
        """Compute all symbolic expressions"""
        expressions = {}
        ham_exprs = self.compute_hamiltonian()
        expressions.update(ham_exprs)
        expressions.update(self.compute_lagrangian(ham_exprs))
        expressions.update(self.compute_constraints())
        return expressions


def load_symbolic_expressions(cls):
    """Decorator to handle loading/saving of symbolic expressions"""
    try:
        # Try to load expressions
        expressions_file = os.path.join('data', f"symbolic_expr_{cls.nosc}.pickle")
        with open(expressions_file, 'rb') as f:
            expressions = pickle.load(f)
        print("Loaded symbolic expressions from disk.")
    except:
        # Compute and save if loading fails
        computer = SymbolicComputer(cls.nosc)
        expressions = computer.compute_all()
        os.makedirs('data', exist_ok=True)
        with open(expressions_file, 'wb') as f:
            pickle.dump(expressions, f)
        print("Saved symbolic expressions to disk.")
    
    # Update class attributes, wrapping lambdas to include self
    for k, v in expressions.items():
        if callable(v) and not isinstance(v, type):
            # Wrap lambda functions to include self parameter
            wrapped = (lambda f: lambda self, *args, **kwargs: f(*args, **kwargs))(v)
            setattr(cls, k, wrapped)
        else:
            setattr(cls, k, v)
    
    return cls

@load_symbolic_expressions
class MechSystem:
    """Class of MechSystem methods"""
    
    # Load or compute expressions once at module level
    keep_time = datetime.now().strftime('%Y-%m-%d_')  #_%H-%M_')
    data_folder = os.path.join('data', keep_time)
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        
    "Numerical solver and its properties"
    w_values = [0.28, 0.62546642846767004501]
    w_values.append(1.0 - 2.0 * (sum(w_values)))
    w_values.append(w_values[1])
    w_values.append(w_values[0])
    w_values = [1]
    assert np.isclose(sum(w_values), 1), 'sum_i w_i must be 1'

    dt_space_dim = 1
    dt_space = linspace(0.01, 0.05, num=dt_space_dim)
    T_final = 5

    "Fixed-point nonliner equations solver properties"
    tol, M, var, store = 1.0E-12, 100, True, False

    # parameter space: frequency of the oscillators -- omega^2
    nosc = 6*3
    _Omega2_space_dim = 5
    _Omega2_space = np.sort(10 * (1 - np.random.rand(_Omega2_space_dim, nosc // 3 - 2)))
    JJ = lambda self, d=nosc: r_[c_[zeros((d, d)), eye(d)], c_[-eye(d), zeros((d, d))]]

    "Initial conditions satisfying the constraints"
    # Create a tensor to store the positions
    i = np.arange(nosc//3)
    positions = np.stack((i % 2 + 0*(i // 2 % 2) * 1e-1, 
                          i // 2 + 0*(-1)**(i // 2) * 1e-1, 
                          np.zeros_like(i)), axis=1)
    positions = positions.flatten().reshape(-1, 1)

    momenta = np.zeros((nosc//3, 3))
    momenta[::2] = np.random.uniform(-0.01, 0.01, momenta[::2].shape)
    momenta[1::2] = momenta[::2]
    assert np.allclose(momenta[1::2] - momenta[::2], 0), 'position and momenta are not orthogonal'
    momenta = momenta.flatten().reshape(-1, 1)

    y_init = r_[positions, momenta].flatten()

    # These two properties only have effect during reduction
    reducer = 'psd'
    predict = True # False = reproduce
    hyperreducer = 'MDEIM'

    # constraints
    constraint_type = 'spherical'
    constraints_reduce = True # True: Reduce constraint jacobian g_prime

    drag = lambda self, y: 0 #beta/2 * r_[x, u]
    drag_z = lambda self, y: 0 #beta/2 * eye(2*x.shape[0])

    non_quad = None
            
    "y_init alias"
    @property
    def u_init(self):
        return self.y_init

    @u_init.setter
    def u_init(self, value):
        self.y_init = value

    def __init__(self, kwds):        
        self.__dict__.update(kwds)        
        if 'pool' in kwds:
            self.__dict__.update(kwds['pool'])
            
        self.beta = (max(1e-2, 0 * np.random.rand() / 10)) * 0
        
        # Update functions based on solver type
        self.ham = lambda y, beta=self.beta: self.ham_(y, self.Omega2, beta)
        if not hasattr(self, 'RB'):
            self.ham_z = lambda y, beta=self.beta: self.ham_z_(y, self.Omega2, beta)
            self.ham_zz = lambda y, beta=self.beta: self.ham_zz_(y, self.Omega2, beta)
            self.lag_dg = lambda y: self.lag_dg_(y, self.Omega2)
            self.lag_dg_z = lambda y: self.lag_dg_z_(y, self.Omega2)

            self.g__ = lambda y: self.g_(y)
            self.g_prime__ = lambda y: self.g_prime_(y)
        else:
            if self.solver_class.__name__ == "DiscreteGradientSolver":
                self.RB = self.RB_dg  # Use RB_dg for DG solvers
                self.nosc_r = self.nosc_r_dg
                self.lag_dg = lambda y: self.RB.T @ self.lag_dg_(y @ self.RB.T, self.Omega2)
                self.lag_dg_z = lambda y: self.RB.T @ self.lag_dg_z_(y @ self.RB.T, self.Omega2) @ self.RB
            else:
                self.ham_z = lambda y, beta=self.beta: self.RB.T @ self.ham_z_(y @ self.RB.T, self.Omega2, beta)
                self.ham_zz = lambda y, beta=self.beta: self.RB.T @ self.ham_zz_(y @ self.RB.T, self.Omega2, beta) @ self.RB

            self.g__ = lambda y: self.g_(y @ self.RB.T)
            self.g_prime__ = lambda y: self.g_prime_(y @ self.RB.T) @ self.RB

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
                
    def get_en_err(self):
        return np.array([self.ham(y) for y in self.y]) - self.ham(self.y[0])

#%% main
if __name__ == '__main__':
    system = MechSystem({'nosc': 6*3})
