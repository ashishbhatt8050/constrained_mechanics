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
import sys

import timeit

# get numpy random seed value
rng = np.random.default_rng(
    seed=267257368022227711484290921317604022527
)  # rng = np.random.default_rng(seed=408394104)  # Create RNG with fixed seed
print(f"NumPy RNG seed: {rng.bit_generator._seed_seq.entropy}")

# %%

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

        self.omega2 = smp.Matrix(smp.symbols('omega^2_:{}'.format(self.nosc//3-2), real=True))
        self.beta = smp.symbols('beta', real=True)

        self.pi = smp.Matrix([(self.q.row(i+2) - self.q.row(i)).norm(2)**2 
                         for i in range(self.nosc//3-2)])

    def compute_hamiltonian(self):
        """Compute Hamiltonian expressions"""
        print("Computing Hamiltonian expressions...")

        kin_expr = 0.5 * self._p.dot(self._p)

        pot_expr_vec = 0.5 * self.omega2.multiply_elementwise(
            (self.pi - smp.ones(*self.pi.shape)).applyfunc(lambda x: x**2))
        pot_expr = sum(pot_expr_vec)

        ham_expr = kin_expr + pot_expr
        ham_z_expr = smp.Matrix([ham_expr]).jacobian(self.y).T
        ham_zz_expr = ham_z_expr.jacobian(self.y)
        ham_zz_nonzero_indices = np.where(np.array(ham_zz_expr.tolist()).flatten() != 0)[0]

        # Generate a numerical function for the Hamiltonian expression
        # This function takes the symbolic variables y, omega2, and beta as input
        # and returns the evaluated Hamiltonian expression as a numpy array.
        ham_ = smp.lambdify((self.y, self.omega2, self.beta), ham_expr, 'numpy')
        ham_z_ = smp.lambdify((self.y, self.omega2, self.beta), ham_z_expr, 'numpy')
        ham_zz_ = smp.lambdify((self.y, self.omega2, self.beta), ham_zz_expr, 'numpy')
        # ham_z_ = lambda y, omega2, beta: _ham_z_(y, omega2, beta).squeeze()

        return {
            'y': self.y, 'y1': self.y1,
            'omega2': self.omega2,
            'beta': self.beta,
            'ham_expr': ham_expr,
            'ham_z_expr': ham_z_expr,
            'ham_zz_expr': ham_zz_expr,
            'ham_': ham_,
            'ham_z_': ham_z_,
            'ham_zz_': ham_zz_,
            'ham_zz_nonzero_indices': ham_zz_nonzero_indices,
            'pi': self.pi,
            'pot_expr_vec': pot_expr_vec
        }

    def compute_lagrangian(self, ham_z_expr):
        """Compute Lagrangian expressions"""
        print("Computing Lagrangian expressions...")

        # Discrete derivatives
        dpi_dq = self.pi.jacobian(self._q).subs(self.y05_repl)

        dV_dpi = 0.5 * self.omega2.multiply_elementwise(
                self.pi.subs(self.y1_repl) + self.pi - 2 * smp.ones(*self.pi.shape)
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

        lag_dg_expr = smp.Matrix.vstack(ham_z_expr[self.nosc:,:].subs(self.y05_repl), -DG_V_expr)
        lag_dg_ = smp.lambdify((self.y, self.y1, self.omega2), lag_dg_expr, 'numpy')

        lag_dg_z_expr = lag_dg_expr.jacobian(self.y1)
        lag_dg_z_nonzero_indices = np.where(np.array(lag_dg_z_expr.tolist()).flatten() != 0)[0]
        lag_dg_z_ = smp.lambdify((self.y, self.y1, self.omega2), lag_dg_z_expr, 'numpy')

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

        g_expr_len = g_expr.rows

        lag_mult = smp.Matrix(smp.symbols('gamma_:{}'.format(g_expr_len), real=True))

        g_prime_x_lambda_expr = g_prime_expr.T @ lag_mult
        g_prime_x_lambda_y_expr = g_prime_x_lambda_expr.jacobian(self.y)
        g_prime_x_lambda_lambda_expr = g_prime_x_lambda_expr.jacobian(lag_mult)
        g_prime_x_lambda_y_nonzero_indices = np.where(np.array(g_prime_x_lambda_y_expr.tolist()).flatten() != 0)[0]
        g_prime_x_lambda_lambda_nonzero_indices = np.where(np.array(g_prime_x_lambda_lambda_expr.tolist()).flatten() != 0)[0]

        g_prime_x_lambda_y_lam = smp.lambdify((self.y, lag_mult), g_prime_x_lambda_y_expr, modules=['numpy'])
        g_prime_x_lambda_lambda_lam = smp.lambdify((self.y, lag_mult), g_prime_x_lambda_lambda_expr, modules=['numpy'])

        g_prime_nonzero_indices = np.where(np.array(g_prime_expr.tolist()).flatten() != 0)[0]

        g_lam = smp.lambdify((self.y,), g_expr, modules=['numpy'])
        g_prime_lam = smp.lambdify((self.y,), g_prime_expr, modules=['numpy'])

        return {
            'g_expr': g_expr,
            'g_prime_expr': g_prime_expr,
            'g_': g_lam,
            'g_prime_': g_prime_lam,
            'g_prime_nonzero_indices': g_prime_nonzero_indices,
            'lag_mult': lag_mult,
            'g_prime_x_lambda_expr': g_prime_x_lambda_expr,
            'g_prime_x_lambda_y_expr': g_prime_x_lambda_y_expr,
            'g_prime_x_lambda_lambda_expr': g_prime_x_lambda_lambda_expr,
            'g_prime_x_lambda_y_': g_prime_x_lambda_y_lam,
            'g_prime_x_lambda_lambda_': g_prime_x_lambda_lambda_lam,
            'g_prime_x_lambda_y_nonzero_indices': g_prime_x_lambda_y_nonzero_indices,
            'g_prime_x_lambda_lambda_nonzero_indices': g_prime_x_lambda_lambda_nonzero_indices,
        }

    @timing
    def compute_all(self, expressions):
        """Compute all symbolic expressions"""
        # expressions = {}
        ham_exprs = self.compute_hamiltonian()
        expressions.update(ham_exprs)
        expressions.update(self.compute_lagrangian(ham_exprs['ham_z_expr']))
        expressions.update(self.compute_constraints())

        # """Parallel computation of symbolic expressions"""
        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #     # Compute Hamiltonian expressions
        #     ham_future = executor.submit(self.compute_hamiltonian)

        #     # While Hamiltonian computes, prepare other computations
        #     lag_future = executor.submit(self.compute_lagrangian)
        #     # Compute constraints
        #     con_future = executor.submit(self.compute_constraints)

        #     # Update expressions as results complete
        #     expressions.update(ham_future.result())
        #     expressions.update(lag_future.result())
        #     expressions.update(con_future.result())

        '''
        g_expr_len = expressions['g_expr'].rows

        lambda_expr = smp.Matrix(smp.symbols('lambda_:{}'.format(g_expr_len), real=True))
        lambda_expr[g_expr_len//2:, :] = smp.Matrix.zeros(g_expr_len//2, lambda_expr.shape[1])

        g_expr_x_lambda = expressions['g_expr'].T @ lambda_expr
        expressions['ham_aug_expr'] = expressions['ham_expr'] + (g_expr_x_lambda)[0,0]

        g_prime_expr_x_lambda = expressions['g_prime_expr'].T @ lambda_expr
        expressions['ham_aug_z_expr'] = expressions['ham_z_expr'] + g_prime_expr_x_lambda

        expressions['ham_aug_zz_expr'] = expressions['ham_zz_expr'] + g_prime_expr_x_lambda.jacobian(self.y)
        expressions['ham_aug_zz_nonzero_indices'] = np.where(np.array(expressions['ham_aug_zz_expr'].tolist()).flatten() != 0)[0]

        lambda_expr[g_expr_len//2:, :] = lambda_expr[:g_expr_len//2, :]
        expressions['lambda_expr'] = lambda_expr
        expressions['ham_aug_'] = smp.lambdify((self.y, lambda_expr, expressions['omega2']), expressions['ham_aug_expr'], 'numpy')
        expressions['ham_aug_z_'] = smp.lambdify((self.y, lambda_expr, expressions['omega2']), expressions['ham_aug_z_expr'], 'numpy')
        expressions['ham_aug_zz_'] = smp.lambdify((self.y, lambda_expr, expressions['omega2']), expressions['ham_aug_zz_expr'], 'numpy')
        # return expressions
        '''


# %%
class IndexedBaseSymbolicComputer:
    """
    Class to compute and store symbolic expressions.
    Uses a hybrid approach: standard symbols for robust symbolic differentiation,
    and IndexedBase for efficient lambdified functions.
    """

    def __init__(self, nosc):
        self.nosc = nosc
        # 1. Create standard symbols for robust symbolic manipulation
        self.y = smp.Matrix(smp.symbols(f"y_:{nosc*2//3}_:{3}", real=True))
        self.y1 = smp.Matrix(
            smp.symbols("y1_:{}_:{}".format(nosc * 2 // 3, 3), real=True)
        )
        self.omega2 = smp.Matrix(
            smp.symbols("omega^2_:{}".format(self.nosc // 3 - 2), real=True)
        )
        self.beta = smp.symbols("beta", real=True)

        # 2. Create IndexedBase symbols for the lambdify step
        self.y_base = smp.IndexedBase("y")
        self.y1_base = smp.IndexedBase("y1")
        self.omega2_base = smp.IndexedBase("omega2")

        # 3. Create substitution rules
        self.y_subs = {s: self.y_base[i] for i, s in enumerate(self.y)}
        self.y1_subs = {s: self.y1_base[i] for i, s in enumerate(self.y1)}
        self.omega2_subs = {s: self.omega2_base[i] for i, s in enumerate(self.omega2)}

        # Create symbols for lag_mult
        self.lag_mult_syms = smp.Matrix(
            smp.symbols("gamma_:{}".format(self.nosc // 3), real=True)
        )
        self.lag_mult_base = smp.IndexedBase("gamma")
        self.lag_mult_subs = {
            s: self.lag_mult_base[i] for i, s in enumerate(self.lag_mult_syms)
        }

        # --- The rest is the same as the original SymbolicComputer ---
        self._q = smp.Matrix(self.y[:nosc])
        self.q = self._q.reshape(nosc // 3, 3)
        self._p = smp.Matrix(self.y[nosc:])
        self.p = self._p.reshape(nosc // 3, 3)
        self.y05_repl = dict(zip(self.y, (self.y + self.y1) / 2))
        self.y1_repl = dict(zip(self.y, self.y1))
        self.pi = smp.Matrix(
            [
                (self.q.row(i + 2) - self.q.row(i)).norm(2) ** 2
                for i in range(self.nosc // 3 - 2)
            ]
        )

    def _lambdify_hybrid(self, args, expr, custom_subs=None):
        """
        Helper to substitute, lambdify, and wrap for input shape compatibility.
        """
        subs_dict = self.y_subs.copy()
        subs_dict.update(self.y1_subs)
        subs_dict.update(self.omega2_subs)
        subs_dict.update(self.lag_mult_subs)
        if custom_subs:
            subs_dict.update(custom_subs)

        # Substitute standard symbols with IndexedBase symbols
        expr_indexed = expr.subs(subs_dict)

        # The actual lambdified function
        _func = smp.lambdify(args, expr_indexed, "numpy", cse=True)

        # Wrapper to handle input shapes
        @wraps(_func)
        def wrapper(*wrapper_args):
            new_args = []
            for arg in wrapper_args:
                if isinstance(arg, np.ndarray) and arg.ndim > 1:
                    new_args.append(arg.flatten())
                else:
                    new_args.append(arg)
            return _func(*new_args)

        return wrapper

    def compute_hamiltonian(self):
        """Compute Hamiltonian expressions"""
        print("Computing Hamiltonian expressions...")
        kin_expr = 0.5 * self._p.dot(self._p)
        pot_expr_vec = 0.5 * self.omega2.multiply_elementwise(
            (self.pi - smp.ones(*self.pi.shape)).applyfunc(lambda x: x**2)
        )
        pot_expr = sum(pot_expr_vec)
        ham_expr = kin_expr + pot_expr
        ham_z_expr = smp.Matrix([ham_expr]).jacobian(self.y).T
        ham_zz_expr = ham_z_expr.jacobian(self.y)
        ham_zz_nonzero_indices = np.where(
            np.array(ham_zz_expr.tolist()).flatten() != 0
        )[0]

        # Use the hybrid lambdify
        ham_ = self._lambdify_hybrid(
            (self.y_base, self.omega2_base, self.beta), ham_expr
        )
        ham_z_ = self._lambdify_hybrid(
            (self.y_base, self.omega2_base, self.beta), ham_z_expr
        )
        ham_zz_ = smp.lambdify(
            (self.y, self.omega2, self.beta), ham_zz_expr, modules=["numpy"], cse=True
        )

        return {
            "y": self.y,
            "y1": self.y1,
            "omega2": self.omega2,
            "beta": self.beta,
            "ham_expr": ham_expr,
            "ham_z_expr": ham_z_expr,
            "ham_zz_expr": ham_zz_expr,
            "ham_": ham_,
            "ham_z_": ham_z_,
            "ham_zz_": ham_zz_,
            "ham_zz_nonzero_indices": ham_zz_nonzero_indices,
            "pi": self.pi,
            "pot_expr_vec": pot_expr_vec,
        }

    def compute_lagrangian(self, ham_z_expr):
        """Compute Lagrangian expressions"""
        print("Computing Lagrangian expressions...")
        dpi_dq = self.pi.jacobian(self._q).subs(self.y05_repl)
        dV_dpi = 0.5 * self.omega2.multiply_elementwise(
            self.pi.subs(self.y1_repl) + self.pi - 2 * smp.ones(*self.pi.shape)
        )
        DG_V_expr = dpi_dq.T @ dV_dpi
        lag_dg_expr = smp.Matrix.vstack(
            ham_z_expr[self.nosc :, :].subs(self.y05_repl), -DG_V_expr
        )

        # # Separate linear terms
        # expr_expanded = lag_dg_expr.expand()
        # linear_terms_y = smp.zeros(*lag_dg_expr.shape)
        # linear_terms_y1 = smp.zeros(*lag_dg_expr.shape)
        # other_terms = smp.zeros(*lag_dg_expr.shape)

        # # For each matrix element, collect terms
        # for idx in range(lag_dg_expr.rows):
        #     # Collect terms with y variables
        #     for yi in self.y:
        #         coeff = expr_expanded[idx].coeff(yi, 1)
        #         if coeff != 0:
        #             linear_terms_y[idx] += coeff * yi
        #             expr_expanded[idx] -= coeff * yi

        #     # Collect terms with y1 variables
        #     for y1i in self.y1:
        #         coeff = expr_expanded[idx].coeff(y1i, 1)
        #         if coeff != 0:
        #             linear_terms_y1[idx] += coeff * y1i
        #             expr_expanded[idx] -= coeff * y1i

        # other_terms = expr_expanded

        # # Verify the separation
        # print(lag_dg_expr.expand() - (linear_terms_y + linear_terms_y1 + other_terms))
        # print("Separation verified: lag_dg_expr = linear_terms_y + linear_terms_y1 + other_terms")

        lag_dg_z_expr = lag_dg_expr.jacobian(self.y1)
        lag_dg_z_nonzero_indices = np.where(
            np.array(lag_dg_z_expr.tolist()).flatten() != 0
        )[0]

        # Hybrid lambdify
        lag_dg_ = self._lambdify_hybrid(
            (self.y_base, self.y1_base, self.omega2_base), lag_dg_expr
        )
        lag_dg_z_ = smp.lambdify(
            (self.y, self.y1, self.omega2), lag_dg_z_expr, modules=["numpy"], cse=True
        )

        return {
            "lag_dg_expr": lag_dg_expr,
            "lag_dg_": lag_dg_,
            "lag_dg_z_expr": lag_dg_z_expr,
            "lag_dg_z_": lag_dg_z_,
            "lag_dg_z_nonzero_indices": lag_dg_z_nonzero_indices,
        }

    def compute_constraints(self):
        """Compute constraint expressions"""
        print("Computing constraint expressions...")
        row_diffs_q = [
            self.q.row((i + 1)) - self.q.row(i) for i in range(0, self.nosc // 3, 2)
        ]
        row_diffs_p = [
            self.p.row((i + 1)) - self.p.row(i) for i in range(0, self.nosc // 3, 2)
        ]
        row_norms = [(row_diff.dot(row_diff) - 1) / 2 for row_diff in row_diffs_q]
        ddt_row_norms = [
            row_diffs_q[i].dot(row_diffs_p[i]) for i in range(len(row_diffs_q))
        ]
        g_expr = smp.Matrix(row_norms + ddt_row_norms)
        g_prime_expr = g_expr.jacobian(self.y)

        # Create symbols for lag_mult
        lag_mult_syms = self.lag_mult_syms

        g_prime_x_lambda_expr = g_prime_expr.T @ lag_mult_syms
        g_prime_x_lambda_y_expr = g_prime_x_lambda_expr.jacobian(self.y)
        g_prime_x_lambda_lambda_expr = g_prime_x_lambda_expr.jacobian(lag_mult_syms)
        g_prime_x_lambda_y_nonzero_indices = np.where(
            np.array(g_prime_x_lambda_y_expr.tolist()).flatten() != 0
        )[0]
        g_prime_x_lambda_lambda_nonzero_indices = np.where(
            np.array(g_prime_x_lambda_lambda_expr.tolist()).flatten() != 0
        )[0]

        # Lambdify for these expressions requires custom substitutions
        g_prime_x_lambda_y_lam = smp.lambdify(
            (self.y, lag_mult_syms), g_prime_x_lambda_y_expr, modules=["numpy"], cse=True
        )
        g_prime_x_lambda_lambda_lam = smp.lambdify(
            (self.y, lag_mult_syms), g_prime_x_lambda_lambda_expr, modules=["numpy"], cse=True
        )

        g_prime_nonzero_indices = np.where(
            np.array(g_prime_expr.tolist()).flatten() != 0
        )[0]

        g_lam = self._lambdify_hybrid((self.y_base,), g_expr)
        g_prime_lam = smp.lambdify((self.y,), g_prime_expr, modules=["numpy"], cse=True)

        g_prime_x_lambda_ = self._lambdify_hybrid(
            (self.y_base, self.lag_mult_base), g_prime_x_lambda_expr
        )

        return {
            "g_expr": g_expr,
            "g_prime_expr": g_prime_expr,
            "g_": g_lam,
            "g_prime_": g_prime_lam,
            "g_prime_nonzero_indices": g_prime_nonzero_indices,
            "lag_mult": lag_mult_syms,
            "g_prime_x_lambda_expr": g_prime_x_lambda_expr,
            "g_prime_x_lambda_": g_prime_x_lambda_,
            "g_prime_x_lambda_y_expr": g_prime_x_lambda_y_expr,
            "g_prime_x_lambda_lambda_expr": g_prime_x_lambda_lambda_expr,
            "g_prime_x_lambda_y_": g_prime_x_lambda_y_lam,
            "g_prime_x_lambda_lambda_": g_prime_x_lambda_lambda_lam,
            "g_prime_x_lambda_y_nonzero_indices": g_prime_x_lambda_y_nonzero_indices,
            "g_prime_x_lambda_lambda_nonzero_indices": g_prime_x_lambda_lambda_nonzero_indices,
        }

    @timing
    def compute_all(self, expressions):
        """Compute all symbolic expressions"""
        ham_exprs = self.compute_hamiltonian()
        expressions.update(ham_exprs)
        expressions.update(self.compute_lagrangian(ham_exprs["ham_z_expr"]))
        expressions.update(self.compute_constraints())


def load_symbolic_expressions(cls):
    """Decorator to handle loading/saving of symbolic expressions"""
    try:
        # Try to load expressions
        expressions_file = os.path.join('data', f"symbolic_expr_{cls.nosc}_cse.pickle")
        with open(expressions_file, 'rb') as f:
            expressions = pickle.load(f)

        print("Loaded symbolic expressions from disk.")
    except (FileNotFoundError, pickle.UnpicklingError):
        # Compute and save if loading fails
        sys.setrecursionlimit(50000)
        computer = IndexedBaseSymbolicComputer(cls.nosc)
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
    if 'SLURM_JOB_ID' in os.environ and False:
        keep_time = f"{os.environ.get('SLURM_JOB_NAME', 'slurm')}-{os.environ['SLURM_JOB_ID']}"
        # keep_time = f"{os.environ.get('SLURM_JOB_NAME', 'slurm')}-1642"
    else:
        keep_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
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
    dt_space = np.round(np.logspace(-2, -1, num=dt_space_dim), 10)
    T_final = dt_space[-1] * 1e2
    "Fixed-point nonliner equations solver properties"
    tol, M, var, store = 1.0E-12, 500, True, False

    # parameter space: frequency of the oscillators -- omega^2
    nosc = 18 * 5 * 3
    assert nosc%6 == 0, 'nosc is not exactly divisible by 6'
    _Omega2_space_dim = 10
    _Omega2_space = np.sort(10 * (1 - rng.random((_Omega2_space_dim, nosc // 3 - 2))))
    _Omega2_space[:, 1 * _Omega2_space_dim - 1 :] = _Omega2_space[
        0, 1 * _Omega2_space_dim - 1 :
    ]  # Only first _Omega2_space_dim-1 columns are random

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
    momenta[::2] = rng.uniform(-0.001, 0.001, momenta[::2].shape)
    momenta[1::2] = momenta[::2]
    assert np.allclose(momenta[1::2], momenta[::2]), 'position and momenta are not orthogonal'
    momenta = momenta.flatten().reshape(-1, 1)

    y_init = r_[positions, momenta].flatten()

    # These two properties only have effect during reduction
    predict = True  # False = reproduce results of the full model
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

    '''
    def ham_aug_lambda(self, y):
        return self.ham_aug_(*y, self.Omega2)

    def ham_aug_z_lambda(self, y):
        return self.ham_aug_z_(*y, self.Omega2).squeeze()

    def ham_aug_zz_lambda(self, y):
        return self.ham_aug_zz_(*y, self.Omega2)
    '''

    def lag_dg_lambda(self, y):
        return self.lag_dg_(*y, self.Omega2).squeeze()

    def lag_dg_z_lambda(self, y):
        return self.lag_dg_z_(*y, self.Omega2)

    def g__lambda(self, y):
        return self.g_(y).squeeze()

    def g_prime__lambda(self, y):
        return self.g_prime_(y)

    def g_prime_x_lambda_y__(self, y, lag_mult):
        """Compute g_prime_x_lambda_y for full order system"""
        # Compute g_prime_x_lambda_y using the full order system
        return self.g_prime_x_lambda_y_(y, lag_mult)

    def g_prime_x_lambda_lambda__(self, y, lag_mult):
        """Compute g_prime_x_lambda_lambda for full order system"""
        # Compute g_prime_x_lambda_lambda using the full order system
        return self.g_prime_x_lambda_lambda_(y, lag_mult)

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

    def g_prime_x_lambda_y_reduced(self, y, lag_mult):
        """Compute g_prime_x_lambda_y for reduced order system"""
        # Compute g_prime_x_lambda_y using the reduced order system
        return self.RB.T @ self.g_prime_x_lambda_y_(y @ self.RB.T, lag_mult) @ self.RB

    def g_prime_x_lambda_lambda_reduced(self, y, lag_mult):
        """Compute g_prime_x_lambda_lambda for reduced order system"""
        # Compute g_prime_x_lambda_lambda using the reduced order system
        return self.RB.T @ self.g_prime_x_lambda_lambda_(y @ self.RB.T, lag_mult)

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

    def g_prime_x_lambda_y_hyperreduced(self, y, lag_mult):
        """Compute g_prime_x_lambda_y for hyperreduced system"""
        return (
            self.RB.T
            @ np.reshape(
                self.IP_g_prime_x_lambda_y_dg
                @ self.g_prime_x_lambda_y_mdeim_dg(y @ self.RB.T, lag_mult),
                self.g_prime_x_lambda_y_shape_dg,
            )
            @ self.RB
        )

    def g_prime_x_lambda_lambda_hyperreduced(self, y, lag_mult):
        """Compute g_prime_x_lambda_lambda for hyperreduced system"""
        return self.RB.T @ np.reshape(
            self.IP_g_prime_x_lambda_lambda_dg
            @ self.g_prime_x_lambda_lambda_mdeim_dg(y @ self.RB.T, lag_mult),
            self.g_prime_x_lambda_lambda_shape_dg,
        )

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

            # if self.solver_class.__name__ == 'DiscreteGradientSolver':
            self.g_prime_x_lambda_y = self.g_prime_x_lambda_y__
            self.g_prime_x_lambda_lambda = self.g_prime_x_lambda_lambda__

            '''
            self.ham_aug = self.ham_aug_lambda
            self.ham_aug_z = self.ham_aug_z_lambda
            self.ham_aug_zz = self.ham_aug_zz_lambda
            '''

        elif (not hasattr(MechSystem, 'RBxUx_inv_PxU')) and (not hasattr(MechSystem, '_RBxUx_inv_PxU_')) and (not hasattr(MechSystem, '_Ux_inv_PxU_dg')) and (not hasattr(MechSystem, '_Ux_inv_PxU')):
            print('Setting reduced order functions...')
            if self.solver_class.__name__ == 'DiscreteGradientSolver':
                # TODO: move these inside the concrete solvers
                print('Setting RB to RB_dg...')
                self.RB = self.RB_dg
                self.nosc_r = self.nosc_r_dg

                self.g_prime_x_lambda_y = self.g_prime_x_lambda_y_reduced
                self.g_prime_x_lambda_lambda = self.g_prime_x_lambda_lambda_reduced

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

                if self.constraints_reduce:
                    self._Ux_inv_PxU, self._IP_Ux_inv_PxU = self._Ux_inv_PxU_dg, self._IP_Ux_inv_PxU_dg

                    self.g_deim = self.g_deim_dg
                    self.g_prime_mdeim = self.g_prime_mdeim_dg
                    self.g_prime_shape = self.g_prime_shape_dg

                    self.g_prime_x_lambda_y = self.g_prime_x_lambda_y_hyperreduced
                    self.g_prime_x_lambda_lambda = self.g_prime_x_lambda_lambda_hyperreduced

            self.lag_dg = self.lag_dg_hyperreduced
            self.lag_dg_z = self.lag_dg_z_hyperreduced
            self.ham_z = self.ham_z_hyperreduced
            self.ham_zz = self.ham_zz_hyperreduced

            self.g__ = self.g_reduced
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


# %% main

if __name__ == '__main__':
    nosc = 18 * 1 * 3

    # initialize MechSystem instance
    system = MechSystem({"nosc": nosc})
    print(f"Shape of g_prime_expr: {system.g_prime_expr.shape}")

""" 
    print("--- Generating expressions for both classes ---")
    
    # Generate expressions from the original class
    print("--- SymbolicComputer ---")
    expressions_old = {}
    computer_old = SymbolicComputer(nosc)
    tl = computer_old.compute_all(expressions_old)
    print(f'Computed symbolic expressions in {tl:.2f} seconds.')

    # Generate expressions from the new hybrid class
    print("--- IndexedBaseSymbolicComputer ---")
    expressions_new = {}
    computer_new = IndexedBaseSymbolicComputer(nosc)
    tl = computer_new.compute_all(expressions_new)
    print(f'Computed symbolic expressions in {tl:.2f} seconds.')

    print("--- Comparing Evaluation Speed of Lambdified Functions ---")

    # --- Prepare numerical data for testing ---
    y_num = np.random.rand(2 * nosc)
    y1_num = np.random.rand(2 * nosc)
    omega2_num = np.random.rand(nosc // 3 - 2)
    beta_num = 0.5
    
    g_expr_len = expressions_old['g_expr'].rows
    lag_mult_num = np.random.rand(g_expr_len)

    # The old functions expect column vectors for matrix inputs
    y_num_mat = y_num.reshape(-1, 1)
    y1_num_mat = y1_num.reshape(-1, 1)
    omega2_num_mat = omega2_num.reshape(-1, 1)
    lag_mult_num_mat = lag_mult_num.reshape(-1, 1)

    number_of_runs = 1000
    
    # --- Comparison ---
    funcs_to_compare = [
        ('ham_', (y_num_mat, omega2_num_mat, beta_num), (y_num_mat, omega2_num_mat, beta_num)),
        ('ham_z_', (y_num_mat, omega2_num_mat, beta_num), (y_num_mat, omega2_num_mat, beta_num)),
        ('ham_zz_', (y_num, omega2_num, beta_num), (y_num, omega2_num, beta_num)),
        ('lag_dg_', (y_num_mat, y1_num_mat, omega2_num_mat), (y_num_mat, y1_num_mat, omega2_num_mat)),
        ('lag_dg_z_', (y_num, y1_num, omega2_num), (y_num, y1_num, omega2_num)),
        ('g_', (y_num_mat,), (y_num_mat,)),
        ('g_prime_', (y_num,), (y_num,)),
        ('g_prime_x_lambda_y_', (y_num, lag_mult_num), (y_num, lag_mult_num)),
        ('g_prime_x_lambda_lambda_', (y_num, lag_mult_num), (y_num, lag_mult_num)),
    ]

    print(f"Timing {number_of_runs} runs for each function...")
    print("-" * 70)
    print(f"{'Function':<28} | {'Original Time (ms)':<20} | {'New Time (ms)':<15} | Speedup")
    print("-" * 70)

    for func_name, args_old, args_new in funcs_to_compare:
        old_func = expressions_old[func_name]
        new_func = expressions_new[func_name]

        t_old = timeit.timeit(lambda: old_func(*args_old), number=number_of_runs) * 1000
        t_new = timeit.timeit(lambda: new_func(*args_new), number=number_of_runs) * 1000
        
        speedup = t_old / t_new if t_new > 0 else float('inf')
        print(f"{func_name:<28} | {t_old:<20.4f} | {t_new:<15.4f} | {speedup:.2f}x")
    
    print("-" * 70)
 """
