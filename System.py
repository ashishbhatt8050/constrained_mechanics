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
from PlotScript import timing

# Set and print random seed
rng = np.random.default_rng(seed=408394104)  # Create RNG with fixed seed

# Safely get and print seed
state = rng.__getstate__()
if isinstance(state, dict) and 'bit_generator' in state:
    bg_state = state['bit_generator']
    if hasattr(bg_state, '_seed_seq') and bg_state._seed_seq is not None:
        print(f"NumPy RNG seed: {bg_state._seed_seq.entropy}")
    else:
        print(f"NumPy RNG seed: {bg_state.state['seed']}")
else:
    ValueError("Unable to retrieve exact seed value")

# np.random.seed(208394104)  # Choose a fixed seed
# print(f"NumPy RNG seed: {np.random.get_state()[1][0]}")

#%%

class SymbolicComputer:
    """Class to compute and store symbolic expressions"""
    def __init__(self, nosc):
        self.nosc = nosc
        # Common symbolic variables
        self.y = smp.Matrix(smp.symbols(f'y_:{nosc*2//3}_:{3}', real=True))
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
            (pi - smp.ones(*pi.shape)).applyfunc(lambda x: x**2))
        pot_expr = sum(pot_expr_vec)

        ham_expr = kin_expr + pot_expr
        ham_z_expr = smp.Matrix([ham_expr]).jacobian(self.y).T
        ham_zz_expr = ham_z_expr.jacobian(self.y)
        ham_zz_nonzero_indices = np.where(np.array(ham_zz_expr.tolist()).flatten() != 0)[0]

        # Generate a numerical function for the Hamiltonian expression
        # This function takes the symbolic variables y, omega2, and beta as input
        # and returns the evaluated Hamiltonian expression as a numpy array.
        ham_ = smp.lambdify((self.y, omega2, beta), ham_expr, 'numpy')
        ham_z_ = smp.lambdify((self.y, omega2, beta), ham_z_expr, 'numpy')
        ham_zz_ = smp.lambdify((self.y, omega2, beta), ham_zz_expr, 'numpy')
        # ham_z_ = lambda y, omega2, beta: _ham_z_(y, omega2, beta).squeeze()

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
            'ham_zz_nonzero_indices': ham_zz_nonzero_indices,
            'pi': pi,
            'pot_expr_vec': pot_expr_vec
        }

    def compute_lagrangian(self, ham_exprs):
        """Compute Lagrangian expressions"""
        print("Computing Lagrangian expressions...")
        
        # Discrete derivatives
        dpi_dq = ham_exprs['pi'].jacobian(self._q).subs(self.y05_repl)


        dV_dpi = 0.5 * ham_exprs['omega2'].multiply_elementwise(
                ham_exprs['pi'].subs(self.y1_repl) + ham_exprs['pi'] - 2 * smp.ones(*ham_exprs['pi'].shape)
                )
        
        '''
        # First collect terms, then factor
        numerator = ham_exprs['pot_expr_vec'].subs(self.y1_repl) - ham_exprs['pot_expr_vec']
        denominator = ham_exprs['pi'].subs(self.y1_repl) - ham_exprs['pi']

        _numerator = 0.5 * ham_exprs['omega2'].multiply_elementwise(
                denominator.multiply_elementwise(
                ham_exprs['pi'].subs(self.y1_repl) + ham_exprs['pi'] - 2 * smp.ones(*ham_exprs['pi'].shape))
                )

        dV_dpi = _numerator.multiply_elementwise(
            denominator.applyfunc(lambda x: 1/x)
        ).applyfunc(smp.factor)

        def compare_expressions(expr1, expr2):
            """Compare two sympy expressions using multiple methods"""
            # Try direct comparison
            if expr1 == expr2:
                return True, "Direct comparison"
                
            # Try comparing simplified forms
            if smp.simplify(expr1 - expr2) == 0:
                return True, "After simplification"
                
            # Try comparing expanded forms
            if smp.expand(expr1) == smp.expand(expr2):
                return True, "After expansion"
                
            # Try comparing factored forms
            if smp.factor(expr1) == smp.factor(expr2):
                return True, "After factoring"
            
            return False, "Expressions are different"

        # Use in your code
        are_equal, method = compare_expressions(numerator, _numerator)
        print(f"Expressions are equal using {method}: {are_equal}")
        are_equal, method = compare_expressions(dV_dpi, _dV_dpi)
        print(f"Expressions are equal using {method}: {are_equal}")

        # dV_dpi = ((ham_exprs['pot_expr_vec'].subs(self.y1_repl) - ham_exprs['pot_expr_vec']).multiply_elementwise(
        #     (ham_exprs['pi'].subs(self.y1_repl) - ham_exprs['pi']).applyfunc(lambda x: 1/x)).applyfunc(smp.factor))'
        '''
        
        DG_V_expr = dpi_dq.T @ dV_dpi
        
        lag_dg_expr = smp.Matrix.vstack(ham_exprs['ham_z_expr'][self.nosc:,:].subs(self.y05_repl), -DG_V_expr)
        lag_dg_ = smp.lambdify((self.y, self.y1, ham_exprs['omega2']), lag_dg_expr, 'numpy')
        
        lag_dg_z_expr = lag_dg_expr.jacobian(self.y1)
        lag_dg_z_nonzero_indices = np.where(np.array(lag_dg_z_expr.tolist()).flatten() != 0)[0]
        lag_dg_z_ = smp.lambdify((self.y, self.y1, ham_exprs['omega2']), lag_dg_z_expr, 'numpy')

        return {
            'lag_dg_expr': lag_dg_expr,
            'lag_dg_': lag_dg_,
            'lag_dg_z_expr': lag_dg_z_expr,
            'lag_dg_z_': lag_dg_z_,
            'lag_dg_z_nonzero_indices': lag_dg_z_nonzero_indices
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
        g_prime_nonzero_indices = np.where(np.array(g_prime_expr.tolist()).flatten() != 0)[0]

        g_lam = smp.lambdify((self.y,), g_expr, modules=['numpy'])
        g_prime_lam = smp.lambdify((self.y,), g_prime_expr, modules=['numpy'])
        
        return {
            'g_expr': g_expr,
            'g_prime_expr': g_prime_expr,
            'g_': g_lam,
            'g_prime_': g_prime_lam,
            'g_prime_nonzero_indices': g_prime_nonzero_indices
        }
    
    @timing
    def compute_all(self, expressions):
        """Compute all symbolic expressions"""
        # expressions = {}
        ham_exprs = self.compute_hamiltonian()
        expressions.update(ham_exprs)
        expressions.update(self.compute_lagrangian(ham_exprs))
        expressions.update(self.compute_constraints())
        # return expressions

def load_symbolic_expressions(cls):
    """Decorator to handle loading/saving of symbolic expressions"""
    try:
        # Try to load expressions
        expressions_file = os.path.join('data', f"symbolic_expr_{cls.nosc}.pickle")
        with open(expressions_file, 'rb') as f:
            expressions = pickle.load(f)

        # if expressions does not contain nonzero indices, compute them
        if 'ham_zz_nonzero_indices' not in expressions:
            expressions['ham_zz_nonzero_indices'] = np.where(np.array(expressions['ham_zz_expr'].tolist()).flatten() != 0)[0]
        if 'lag_dg_z_nonzero_indices' not in expressions:
            expressions['lag_dg_z_nonzero_indices'] = np.where(np.array(expressions['lag_dg_z_expr'].tolist()).flatten() != 0)[0]
        if 'g_prime_nonzero_indices' not in expressions:
            expressions['g_prime_nonzero_indices'] = np.where(np.array(expressions['g_prime_expr'].tolist()).flatten() != 0)[0]
        print("Loaded symbolic expressions from disk.")
    except (FileNotFoundError, pickle.UnpicklingError):
        # Compute and save if loading fails
        computer = SymbolicComputer(cls.nosc)
        expressions = {}
        tl = computer.compute_all(expressions)
        print(f'Computed symbolic expressions in {tl:.2f} seconds.')
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
    keep_time = datetime.now().strftime('%Y-%m-%d')  #_%H-%M_')
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
    dt_space = np.round(np.logspace(-2.5, -2, num=dt_space_dim), 5)
    T_final = 1
    "Fixed-point nonliner equations solver properties"
    tol, M, var, store = 1.0E-12, 100, True, False

    # parameter space: frequency of the oscillators -- omega^2
    nosc = 16*3
    assert nosc%6 == 0, 'nosc is not exactly divisible by 6'
    _Omega2_space_dim = 3
    _Omega2_space = np.sort(10 * (1 - rng.random((_Omega2_space_dim, nosc // 3 - 2))))
    # _Omega2_space[:, 1*_Omega2_space_dim-1:] = _Omega2_space[0, 1*_Omega2_space_dim-1:] # Only first _Omega2_space_dim-1 columns are random

    JJ = lambda self, d=nosc: r_[c_[zeros((d, d)), eye(d)], c_[-eye(d), zeros((d, d))]]

    "Initial conditions satisfying the constraints"
    # Create a tensor to store the positions
    _i = np.arange(nosc//3//2)
    # positions = np.stack((_i % 2 + 0*(_i // 2 % 2) * 1e-1, 
    #                       _i // 2 + 0*(-1)**(_i // 2) * 1e-1, 
    #                       np.zeros_like(_i)), axis=1)
    
    # Calculate helix positions
    _radius = 0.5
    _pitch = 0.5 * nosc/18  # Scale pitch with number of particles
    _t = _i * 2 * np.pi / (nosc//3)
    _phase = np.pi

    positions_1 = np.stack((
        _radius * np.cos(_t),
        _radius * np.sin(_t),
        _pitch * _t / (2*np.pi)
    ), axis=1)

    positions_2 = np.stack((
        _radius * np.cos(_t + _phase),
        _radius * np.sin(_t + _phase),
        _pitch * _t / (2*np.pi)
    ), axis=1)

    # Combine the two helices, interleaving their positions
    positions = np.zeros((nosc//3, 3))
    positions[::2] = positions_1
    positions[1::2] = positions_2

    # Assert that the corresponding coordinates of positions_1 and positions_2 are distant 1 apart
    distances = np.linalg.norm(positions[::2] - positions[1::2], axis=1)
    assert np.allclose(distances, 1), "The corresponding coordinates of positions_1 and positions_2 are not distant 1 apart."

    positions = positions.flatten().reshape(-1, 1)

    momenta = np.zeros((nosc//3, 3))
    momenta[::2] = rng.uniform(-0.01, 0.01, momenta[::2].shape)
    momenta[1::2] = momenta[::2]
    assert np.allclose(momenta[1::2], momenta[::2]), 'position and momenta are not orthogonal'
    momenta = momenta.flatten().reshape(-1, 1)

    y_init = r_[positions, momenta].flatten()

    # These two properties only have effect during reduction
    predict = False  # False = reproduce results of the full model
    reducer = 'psd'
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

    def ham_lambda(self, y, beta=0):
        return self.ham_(y, self.Omega2, beta)

    def ham_z_lambda(self, y, beta=0):
        return self.ham_z_(y, self.Omega2, beta).squeeze()

    def ham_zz_lambda(self, y, beta=0):
        return self.ham_zz_(y, self.Omega2, beta)

    def lag_dg_lambda(self, y):
        return self.lag_dg_(*y, self.Omega2).squeeze()

    def lag_dg_z_lambda(self, y):
        return self.lag_dg_z_(*y, self.Omega2)

    def g__lambda(self, y):
        return self.g_(y).squeeze()

    def g_prime__lambda(self, y):
        return self.g_prime_(y)

    def ham_z_reduced(self, y, beta=0):
        return self.RB.T @ self.ham_z_(y @ self.RB.T, self.Omega2, beta).squeeze()

    def ham_zz_reduced(self, y, beta=0):
        # print(f'Computing ham_zz inside ham_zz_reduced')
        return self.RB.T @ self.ham_zz_(y @ self.RB.T, self.Omega2, beta) @ self.RB

        """Break large matrix multiplication into smaller chunks"""
        y_projected = y @ self.RB.T
        ham_zz_result = self.ham_zz_(y_projected, self.Omega2, beta)
        
        # Use 32 as chunk size (96 % 32 = 0 and 32 < 36)
        chunk_size = 32  # Gives exactly 3 chunks of size 32
        n_chunks = ham_zz_result.shape[1] // chunk_size
        
        # First multiplication
        result = np.zeros((self.RB.T.shape[0], ham_zz_result.shape[1]))
        for i in range(n_chunks):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            result[:, start:end] = self.RB.T @ ham_zz_result[:, start:end]
        
        # Second multiplication
        final = np.zeros((result.shape[0], self.RB.shape[1]))
        for i in range(n_chunks):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            final += result[:, start:end] @ self.RB[start:end, :]
            
        return final

    def lag_dg_reduced(self, y):
        return self.RB.T @ self.lag_dg_(*(y @ self.RB.T), self.Omega2).squeeze()

    def lag_dg_z_reduced(self, y):
        # print(f'Computing lag_dg_z inside lag_dg_z_reduced')
        return self.RB.T @ self.lag_dg_z_(*(y @ self.RB.T), self.Omega2) @ self.RB
    
        """Break large matrix multiplication into smaller chunks"""
        # First compute the DG result
        result = self.lag_dg_z_(*(y @ self.RB.T), self.Omega2)
        
        # Use 32 as chunk size (96 % 32 = 0 and 32 < 36)
        chunk_size = 32  # Gives exactly 3 chunks of size 32
        n_chunks = result.shape[1] // chunk_size
        
        # First multiplication
        intermediate = np.zeros((self.RB.T.shape[0], result.shape[1]))
        for i in range(n_chunks):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            intermediate[:, start:end] = self.RB.T @ result[:, start:end]
        
        # Second multiplication
        final = np.zeros((intermediate.shape[0], self.RB.shape[1]))
        for i in range(n_chunks):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            final += intermediate[:, start:end] @ self.RB[start:end, :]
            
        return final
    
    def g_reduced(self, y):
        return self.g_(y @ self.RB.T).squeeze()

    def g_prime_reduced(self, y):
        return self.g_prime_(y @ self.RB.T) @ self.RB
    
        """Break large matrix multiplication into smaller chunks"""
        # First compute g_prime result
        y_projected = y @ self.RB.T
        g_prime_result = self.g_prime_(y_projected)
        
        # Use 32 as chunk size (96 % 32 = 0 and 32 < 36)
        chunk_size = 32  # Gives exactly 3 chunks of size 32
        n_chunks = g_prime_result.shape[1] // chunk_size
        
        # Process in chunks
        final = np.zeros((g_prime_result.shape[0], self.RB.shape[1]))
        for i in range(n_chunks):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            final += g_prime_result[:, start:end] @ self.RB[start:end, :]
        
        return final

    def ham_z_hyperreduced(self, y, beta=0):
        return self.RBxUx_inv_PxU @ self.ham_z_deim(y @ self.RB.T, self.Omega2, beta)

    def ham_zz_hyperreduced(self, y, beta=0):
        return self.RBxUx_inv_PxU @ self.ham_zz_deim(y @ self.RB.T, self.Omega2, beta) @ self.RB
    
    def ham_zz_mdeim_hyperreduced(self, y, beta=0):
        return self.RB.T @ np.reshape(
            self.IP_Ux_inv_PxU @ self.ham_zz_mdeim(y @ self.RB.T, self.Omega2, beta),
            (2*self.nosc, 2*self.nosc)
        ) @ self.RB

    def lag_dg_hyperreduced(self, y):
        return self.RBxUx_inv_PxU @ self.lag_dg_deim(*(y @ self.RB.T), self.Omega2)

    def lag_dg_z_hyperreduced(self, y):
        return self.RBxUx_inv_PxU @ self.lag_dg_z_deim(*(y @ self.RB.T), self.Omega2) @ self.RB
    
    def lag_dg_z_mdeim_hyperreduced(self, y):
        return self.RB.T @ np.reshape(
            self.IP_Ux_inv_PxU @ self.lag_dg_z_mdeim(*(y @ self.RB.T), self.Omega2),
            (2*self.nosc, 2*self.nosc)
        ) @ self.RB
    
    def g_hyperreduced(self, y):
        return (self._Ux_inv_PxU @ self.g_deim(y @ self.RB.T)).squeeze()

    def g_prime_hyperreduced(self, y):
        return np.reshape(
            self._IP_Ux_inv_PxU @ self.g_prime_mdeim(y @ self.RB.T),
            self.g_prime_shape
        ) @ self.RB

    def __init__(self, kwds):        
        self.__dict__.update(kwds)        
        if 'pool' in kwds:
            self.__dict__.update(kwds['pool'])
            
        
        self.beta = (max(1e-2, 0 * rng.random() / 10)) * 0
        
        # Update functions based on solver type
        self.ham = self.ham_lambda
        if (not hasattr(self, 'RB')) and (not hasattr(self, 'RB_dg')):
            print('Setting full order functions...')
            self.ham_z = self.ham_z_lambda
            self.ham_zz = self.ham_zz_lambda
            self.lag_dg = self.lag_dg_lambda
            self.lag_dg_z = self.lag_dg_z_lambda

            self.g__ = self.g__lambda
            self.g_prime__ = self.g_prime__lambda

        elif (not hasattr(MechSystem, 'RBxUx_inv_PxU')) and (not hasattr(MechSystem, '_RBxUx_inv_PxU_')) and (not hasattr(MechSystem, '_Ux_inv_PxU_dg')) and (not hasattr(MechSystem, '_Ux_inv_PxU')):
            print('Setting reduced order functions...')
            if self.solver_class.__name__ == 'DiscreteGradientSolver':
                # TODO: move these inside the concrete solvers
                print('Setting RB to RB_dg...')
                self.RB = self.RB_dg
                self.nosc_r = self.nosc_r_dg

            self.lag_dg = self.lag_dg_reduced
            self.lag_dg_z = self.lag_dg_z_reduced
            self.ham_z = self.ham_z_reduced
            self.ham_zz = self.ham_zz_reduced

            self.g__ = self.g_reduced
            self.g_prime__ = self.g_prime_reduced
            
        else:
            print('Setting hyperreduced functions...')
            if self.solver_class.__name__ == 'DiscreteGradientSolver':
                print('Setting RB to RB_dg...')
                self.RB = self.RB_dg
                self.nosc_r = self.nosc_r_dg
                self.RBxUx_inv_PxU = self._RBxUx_inv_PxU_
                if self.hyperreducer == 'MDEIM':
                    self.IP_Ux_inv_PxU = self._IP_Ux_inv_PxU_

                self._Ux_inv_PxU, self._IP_Ux_inv_PxU = self._Ux_inv_PxU_dg, self._IP_Ux_inv_PxU_dg

                self.g_deim = self.g_deim_dg
                self.g_prime_mdeim = self.g_prime_mdeim_dg
                self.g_prime_shape = self.g_prime_shape_dg
                
            self.lag_dg = self.lag_dg_hyperreduced
            self.lag_dg_z = self.lag_dg_z_hyperreduced
            self.ham_z = self.ham_z_hyperreduced
            self.ham_zz = self.ham_zz_hyperreduced

            self.g__ = self.g_hyperreduced
            self.g_prime__ = self.g_prime_hyperreduced

            if self.hyperreducer == 'MDEIM':
                self.ham_zz = self.ham_zz_mdeim_hyperreduced
                self.lag_dg_z = self.lag_dg_z_mdeim_hyperreduced

        # Projection matrices
        if hasattr(self, 'RB'):
            self.y_init = self.RB.T @ self.y_init
            
            if self.reducer == 'psd':
                self.JJ = self.JJ(self.nosc_r)
        else:
            self.RB = None
            self.JJ = self.JJ(self.nosc)

        # various measures
        self.time_lapsed = []
                
    def get_en_err(self):
        return np.array([self.ham(y) for y in self.y]) - self.ham(self.y[0])

#%% main
if __name__ == '__main__':
    pass
    system = MechSystem({'nosc': 16*3})
    print(f"Shape of g_prime_expr: {system.g_prime_expr.tolist().flatten().shape}")
