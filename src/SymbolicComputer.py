import numpy as np
import sympy as smp
from functools import wraps
from PlotScript import timing
import concurrent.futures
import cloudpickle
from tqdm.auto import tqdm
import joblib
import os
import json
import shutil

CODE_VERSION = "1.0"

class ShapeWrapper:
    """Picklable wrapper for handling input shapes."""
    def __init__(self, func):
        self.func = func
    
    def __call__(self, *args):
        new_args = []
        for arg in args:
            if isinstance(arg, np.ndarray) and arg.ndim > 1:
                new_args.append(arg.flatten())
            else:
                new_args.append(arg)
        return self.func(*new_args)

def _run_method(computer, method_name, *args):
    """Helper to run a method and return cloudpickled result."""
    method = getattr(computer, method_name)
    result = method(*args)
    return cloudpickle.dumps(result)

# Setup persistent cache
verbose_level = int(os.environ.get('JOBLIB_VERBOSE', 0))
memory = joblib.Memory(os.path.join('data', 'joblib_cache'), verbose=verbose_level)

def manage_cache(nosc, config_hash, cache_dir, checkpoint_dir, expressions_file, 
                 clean_cache=False, clean_checkpoints=False, clean_symbolic=False):
    """
    Ensures cache consistency. Clears cache if nosc, code version, or config hash changes.
    """
    metadata_file = os.path.join(cache_dir, 'metadata.json')
    
    should_clear_cache = clean_cache
    should_clear_checkpoints = clean_checkpoints
    should_clear_symbolic = clean_symbolic
    
    if os.path.exists(metadata_file):
        try: 
            with open(metadata_file, 'r') as f:
                data = json.load(f)
            
            if data.get('nosc') != nosc:
                print(f"Parameter 'nosc' changed ({data.get('nosc')} -> {nosc}). Clearing joblib cache and checkpoints.")
                should_clear_cache = True
                should_clear_checkpoints = True
            elif data.get('code_version') != CODE_VERSION:
                print(f"Code version changed ({data.get('code_version')} -> {CODE_VERSION}). Clearing joblib cache and checkpoints.")
                should_clear_cache = True
                should_clear_checkpoints = True
            elif data.get('config_hash') != config_hash:
                print(f"SysConfig parameters changed. Clearing joblib cache and checkpoints.")
                should_clear_cache = True
                should_clear_checkpoints = True
        except Exception as e:
            print(f"Could not read cache metadata ({e}). Clearing joblib cache and checkpoints.")
            should_clear_cache = True
            should_clear_checkpoints = True

    if should_clear_cache:
        # This will delete the cache_dir, including metadata.json
        memory.clear(warn=False)
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            
    if should_clear_checkpoints and os.path.exists(checkpoint_dir):
        print(f"Cleaning checkpoint directory: {checkpoint_dir}")
        shutil.rmtree(checkpoint_dir)

    if should_clear_symbolic and os.path.exists(expressions_file):
        print(f"Clearing symbolic expressions cache: {expressions_file}")
        os.remove(expressions_file)
        
    # Ensure the directory exists before writing. memory.clear() removes it.
    os.makedirs(cache_dir, exist_ok=True)
    with open(metadata_file, 'w') as f:
        json.dump({'nosc': nosc, 'code_version': CODE_VERSION, 'config_hash': config_hash}, f)

@memory.cache
def _compile_hybrid_function_cached(args, expr, subs_items, code_version=None):
    """Cached implementation of compilation."""
    subs_dict = dict(subs_items)
    # Substitute standard symbols with IndexedBase symbols
    expr_indexed = expr.subs(subs_dict)

    # The actual lambdified function
    _func = smp.lambdify(args, expr_indexed, "numpy", cse=True)

    return cloudpickle.dumps(ShapeWrapper(_func))

def compile_hybrid_function(args, expr, subs_source):
    """
    Standalone helper to substitute, lambdify, and wrap.
    Can be used in parallel workers.
    """
    # Ensure expr is hashable (handle MutableDenseMatrix)
    if isinstance(expr, smp.Matrix):
        expr_key = smp.ImmutableMatrix(expr)
    else:
        expr_key = expr

    # Handle both dict (create sorted tuple) and tuple (use directly)
    if isinstance(subs_source, dict):
        subs_items = tuple(sorted(subs_source.items(), key=lambda x: x[0].name))
    else:
        subs_items = subs_source
    
    result_bytes = _compile_hybrid_function_cached(args, expr_key, subs_items, code_version=CODE_VERSION)
    
    return cloudpickle.loads(result_bytes)

class IndexedBaseSymbolicComputer:
    """
    Computes and stores symbolic expressions using IndexedBase for performance.
    
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
        
        # Precompute common substitution dictionary
        self.common_subs = self.y_subs.copy()
        self.common_subs.update(self.y1_subs)
        self.common_subs.update(self.omega2_subs)
        self.common_subs.update(self.lag_mult_subs)
        
        # Precompute common substitution dictionary items for caching
        self.common_subs_items = tuple(sorted(self.common_subs.items(), key=lambda x: x[0].name))

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
        if custom_subs:
            subs_dict = self.common_subs.copy()
            subs_dict.update(custom_subs)
            return compile_hybrid_function(args, expr, subs_dict)
        else:
            return compile_hybrid_function(args, expr, self.common_subs_items)

    def _compute_ham_z_expr(self):
        """Helper to compute ham_z_expr for internal use."""
        kin_expr = 0.5 * self._p.dot(self._p)
        pot_expr_vec = 0.5 * self.omega2.multiply_elementwise(
            (self.pi - smp.ones(*self.pi.shape)).applyfunc(lambda x: x**2)
        )
        pot_expr = sum(pot_expr_vec)
        ham_expr = kin_expr + pot_expr
        ham_z_expr = smp.Matrix([ham_expr]).jacobian(self.y).T
        return ham_expr, ham_z_expr, pot_expr_vec

    def compute_hamiltonian(self):
        """Compute Hamiltonian expressions"""
        print("Computing Hamiltonian expressions...")
        ham_expr, ham_z_expr, pot_expr_vec = self._compute_ham_z_expr()
        ham_zz_expr = ham_z_expr.jacobian(self.y)
        
        ham_ = self._lambdify_hybrid((self.y_base, self.omega2_base, self.beta), ham_expr)
        ham_z_ = self._lambdify_hybrid((self.y_base, self.omega2_base, self.beta), ham_z_expr)
        ham_zz_ = smp.lambdify((self.y, self.omega2, self.beta), ham_zz_expr, modules=["numpy"], cse=True)

        return {
            "y": self.y, "y1": self.y1, "omega2": self.omega2, "beta": self.beta,
            "ham_z_expr": ham_z_expr, "ham_zz_expr": ham_zz_expr,
            "ham_": ham_, "ham_z_": ham_z_, "ham_zz_": ham_zz_,
            "ham_zz_nonzero_indices": np.where(np.array(ham_zz_expr.tolist()).flatten() != 0)[0]
        }

    def compute_lagrangian(self):
        """Compute Lagrangian expressions"""
        print("Computing Lagrangian expressions...")
        _, ham_z_expr, _ = self._compute_ham_z_expr()
        dpi_dq = self.pi.jacobian(self._q).subs(self.y05_repl)
        dV_dpi = 0.5 * self.omega2.multiply_elementwise(
            self.pi.subs(self.y1_repl) + self.pi - 2 * smp.ones(*self.pi.shape)
        )
        DG_V_expr = dpi_dq.T @ dV_dpi
        lag_dg_expr = smp.Matrix.vstack(ham_z_expr[self.nosc :, :].subs(self.y05_repl), -DG_V_expr)
        lag_dg_z_expr = lag_dg_expr.jacobian(self.y1)

        lag_dg_ = self._lambdify_hybrid((self.y_base, self.y1_base, self.omega2_base), lag_dg_expr)
        lag_dg_z_ = smp.lambdify((self.y, self.y1, self.omega2), lag_dg_z_expr, modules=["numpy"], cse=True)

        return {
            "lag_dg_expr": lag_dg_expr, "lag_dg_z_expr": lag_dg_z_expr,
            "lag_dg_": lag_dg_, "lag_dg_z_": lag_dg_z_,
            "lag_dg_z_nonzero_indices": np.where(np.array(lag_dg_z_expr.tolist()).flatten() != 0)[0]
        }

    def compute_constraints(self):
        """Compute constraint expressions"""
        print("Computing constraint expressions...")
        row_diffs_q = [self.q.row((i + 1)) - self.q.row(i) for i in range(0, self.nosc // 3, 2)]
        row_diffs_p = [self.p.row((i + 1)) - self.p.row(i) for i in range(0, self.nosc // 3, 2)]
        row_norms = [(row_diff.dot(row_diff) - 1) / 2 for row_diff in row_diffs_q]
        ddt_row_norms = [row_diffs_q[i].dot(row_diffs_p[i]) for i in range(len(row_diffs_q))]
        g_expr = smp.Matrix(row_norms + ddt_row_norms)
        g_prime_expr = g_expr.jacobian(self.y)
        
        # Create symbols for lag_mult
        lag_mult = self.lag_mult_syms

        g_prime_x_lambda_expr = g_prime_expr.T @ lag_mult
        g_prime_x_lambda_y_expr = g_prime_x_lambda_expr.jacobian(self.y)
        g_prime_x_lambda_lambda_expr = g_prime_x_lambda_expr.jacobian(lag_mult)
        
        # Compute nonzero indices
        g_prime_nonzero_indices = np.where(np.array(g_prime_expr.tolist()).flatten() != 0)[0]
        g_prime_x_lambda_y_nonzero_indices = np.where(np.array(g_prime_x_lambda_y_expr.tolist()).flatten() != 0)[0]
        g_prime_x_lambda_lambda_nonzero_indices = np.where(np.array(g_prime_x_lambda_lambda_expr.tolist()).flatten() != 0)[0]

        return {
            "g_expr": g_expr, "g_prime_expr": g_prime_expr,
            "g_": self._lambdify_hybrid((self.y_base,), g_expr),
            "g_prime_": ShapeWrapper(smp.lambdify((self.y,), g_prime_expr, modules=["numpy"], cse=True)),
            "g_prime_nonzero_indices": g_prime_nonzero_indices,
            "lag_mult": lag_mult,
            "g_prime_x_lambda_expr": g_prime_x_lambda_expr,
            "g_prime_x_lambda_y_expr": g_prime_x_lambda_y_expr,
            "g_prime_x_lambda_lambda_expr": g_prime_x_lambda_lambda_expr,
            "g_prime_x_lambda_y_": self._lambdify_hybrid((self.y_base, self.lag_mult_base), g_prime_x_lambda_y_expr),
            "g_prime_x_lambda_lambda_": self._lambdify_hybrid((self.y_base, self.lag_mult_base), g_prime_x_lambda_lambda_expr),
            "g_prime_x_lambda_y_nonzero_indices": g_prime_x_lambda_y_nonzero_indices,
            "g_prime_x_lambda_lambda_nonzero_indices": g_prime_x_lambda_lambda_nonzero_indices,
        }

    @timing
    def compute_all(self, expressions):
        """Compute all symbolic expressions"""
        with concurrent.futures.ProcessPoolExecutor() as executor:
            future_map = {
                executor.submit(_run_method, self, 'compute_hamiltonian'): 'ham',
                executor.submit(_run_method, self, 'compute_constraints'): 'con',
                executor.submit(_run_method, self, 'compute_lagrangian'): 'lag'
            }
            
            for future in tqdm(concurrent.futures.as_completed(future_map), total=len(future_map), desc="Computing symbolic expressions"):
                result_bytes = future.result()
                result_exprs = cloudpickle.loads(result_bytes)
                expressions.update(result_exprs)