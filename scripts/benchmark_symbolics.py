#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script contains historical benchmarking code that was originally at the
end of the System.py module. It compares the performance of the obsolete
`SymbolicComputer` class with the `IndexedBaseSymbolicComputer`.
"""

import timeit
import numpy as np
import sympy as smp

from PlotScript import timing

# The following imports would be needed from the old System.py
# from System import IndexedBaseSymbolicComputer

class SymbolicComputer:
    """
    Computes and stores symbolic expressions for the mechanical system.
    
    Handles Hamiltonian, Lagrangian, and constraint expressions using SymPy.

    Obsolete: will be deleted in a future version.
    """
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


if __name__ == '__main__':
    # This block is for demonstration and will raise an error because
    # SymbolicComputer is no longer part of the project.
    try:
        from System import IndexedBaseSymbolicComputer

        nosc = 54

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
            ('ham_', (y_num_mat, omega2_num_mat, beta_num)),
            ('ham_z_', (y_num_mat, omega2_num_mat, beta_num)),
            ('ham_zz_', (y_num, omega2_num, beta_num)),
            ('lag_dg_', (y_num_mat, y1_num_mat, omega2_num_mat)),
            ('lag_dg_z_', (y_num, y1_num, omega2_num)),
            ('g_', (y_num_mat,)),
            ('g_prime_', (y_num,)),
            ('g_prime_x_lambda_y_', (y_num, lag_mult_num)),
            ('g_prime_x_lambda_lambda_', (y_num, lag_mult_num)),
        ]

        print(f"Timing {number_of_runs} runs for each function...")
        print("-" * 70)
        print(f"{'Function':<28} | {'Original Time (ms)':<20} | {'New Time (ms)':<15} | Speedup")
        print("-" * 70)

        for func_name, args in funcs_to_compare:
            old_func = expressions_old[func_name]
            new_func = expressions_new[func_name]

            t_old = timeit.timeit(lambda: old_func(*args), number=number_of_runs) * 1000
            t_new = timeit.timeit(lambda: new_func(*args), number=number_of_runs) * 1000

            speedup = t_old / t_new if t_new > 0 else float('inf')
            print(f"{func_name:<28} | {t_old:<20.4f} | {t_new:<15.4f} | {speedup:.2f}x")

        print("-" * 70)

    except ImportError:
        print("Could not run benchmark as `SymbolicComputer` is obsolete.")