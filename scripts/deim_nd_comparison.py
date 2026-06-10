#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import sys
import cloudpickle as pickle
import time
from functools import partial
from itertools import product
import multiprocessing
import matplotlib.pyplot as plt
from scipy.linalg import qr
import sympy as sp

# Add project source to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from PlotScript import save_figure

try:
    import cupy as cp
    HAS_CUPY = False
except ImportError:
    HAS_CUPY = False

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = False
except ImportError:
    HAS_JAX = False

# Parameters
d = 6  # Increased number of dimensions
nx = 10  # Grid points per dimension
num_snapshots = 100
num_test_runs = 100
use_sparse_grid = False # Toggle: True for Smolyak sparse grid, False for regular Cartesian grid
USE_GPU_ONLINE = True  # Toggle: False keeps evaluation/reconstruction on CPU
FORCE_RECOMPUTE_SYM = False # Toggle: True forces re-generation of symbolic expressions

# Determine if JAX symbolic evaluation can truly use the GPU
USE_JAX_SYMBOLIC = False
if HAS_JAX and HAS_CUPY:
    if jax.default_backend() == 'gpu':
        USE_JAX_SYMBOLIC = True
    else:
        print("WARNING: JAX is installed but not configured for GPU. Falling back to CPU for symbolic evaluation.", flush=True)

# Set global font size for plots
plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 16,
    'legend.fontsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'axes.titlesize': 18 # Title might need to be slightly larger
})
def generate_sparse_grid(dim, level):
    """
    Generates a Smolyak-style sparse grid to avoid the curse of dimensionality.
    """
    def get_1d_pts(l):
        if l == 1:
            return np.array([0.5])
        n_pts = 2**(l-1) + 1
        return np.linspace(0.1, 0.9, n_pts)

    grid_points = []
    # Smolyak construction: sum of levels i_1 + ... + i_d <= d + level
    for levels in product(range(1, level + 2), repeat=dim):
        if sum(levels) <= dim + level:
            pts_1d = [get_1d_pts(l) for l in levels]
            for p in product(*pts_1d):
                grid_points.append(p)
    
    # Remove duplicates
    return np.unique(np.array(grid_points), axis=0)

print(f"Generating {'sparse' if use_sparse_grid else 'regular'} grid for d={d}...")
if use_sparse_grid:
    coords_np = generate_sparse_grid(d, level=2)
else:
    grid_axis = np.linspace(0.1, 0.9, nx)
    coords_np = np.array(list(product(grid_axis, repeat=d)))

n = coords_np.shape[0]
coords = coords_np  # Initialize on CPU for multiprocessing safety
print(f"Sparse grid constructed with n={n} points (vs {nx**d} for a full grid).")

# Nonlinear function (vectorized)
def f(mu, coords, indices=None):
    xp = cp.get_array_module(coords) if HAS_CUPY else np
    # Ensure parameter vector is on the same device as coords
    if HAS_CUPY and xp == cp and not isinstance(mu, cp.ndarray):
        mu = cp.asarray(mu)
        
    if indices is None:
        # Full evaluation
        dist_sq = xp.sum((coords - mu)**2, axis=1)
        return 1.0 / xp.sqrt(dist_sq + 0.1**2)
    else:
        # Evaluate only at specific indices
        dist_sq = xp.sum((coords[indices] - mu)**2, axis=1)
        return 1.0 / xp.sqrt(dist_sq + 0.1**2)

# QDEIM selection (pivoted QR)
def qdeim_indices(U):
    # QR with column pivoting on U^T
    # Note: CuPy linalg.qr does not support column pivoting; use CPU for index selection.
    if HAS_CUPY and hasattr(U, 'get'):
        U = U.get()
    _, _, piv = qr(U.T, pivoting=True)
    return piv[:U.shape[1]]

# --- Symbolic Offline Phase (Method A) ---
# Global symbolic variables
mu_sym = sp.IndexedBase('mu')
coords_sym = sp.IndexedBase('coords') # New: Symbolic IndexedBase for coordinates

def _create_full_symbolic_matrix_expr(total_n, d):
    print(f"Constructing FULL symbolic matrix expression for {d}D system over {total_n} grid points...")
    
    # Create a single Matrix expression where each element is the kernel at a grid point
    full_matrix_expr = sp.Matrix(total_n, 1, lambda idx, _: 
        1 / sp.sqrt(sp.Add(*[(coords_sym[idx, k] - mu_sym[k])**2 for k in range(d)]) + 0.1**2)
    )
    
    return full_matrix_expr

# --- Timing and Performance Comparison ---
def run_deim(U_deim, indices, mu_test, coords, deim_func_sym, method='numerical'):
    xp = cp.get_array_module(U_deim) if HAS_CUPY else np
    
    EU_inv = xp.linalg.inv(U_deim[indices, :])
    
    if method == 'symbolic':
        if USE_JAX_SYMBOLIC:
            # Evaluate directly on GPU using JAX
            # Ensure all inputs are JAX arrays for the JAX-lambdified function
            if not isinstance(mu_test, jnp.ndarray):
                mu_test = jnp.asarray(mu_test)
            if not isinstance(coords, jnp.ndarray):
                coords = jnp.asarray(coords)

            t1 = time.perf_counter()
            # Call the already sliced and lambdified function with full coords
            f_test_indices = deim_func_sym(mu_test, coords)
            # Convert JAX array to the current xp array type (CuPy or NumPy) and flatten to 1D
            if xp == cp:
                f_test_indices = cp.asarray(f_test_indices.flatten())
            else: # xp is numpy
                f_test_indices = np.asarray(f_test_indices.flatten())
            if HAS_CUPY and xp == cp: cp.cuda.Stream.null.synchronize() # Synchronize JAX/CuPy ops
        else:
            # Pre-transfer data to CPU (Not timed)
            mu_cpu = mu_test.get() if hasattr(mu_test, 'get') else mu_test
            # coords_np is already on CPU
            
            # Time Symbolic Evaluation
            t1 = time.perf_counter()
            # Call the already sliced and lambdified function with full coords_np
            f_test_indices_raw = deim_func_sym(mu_cpu, coords_np)
            
            f_test_indices = xp.asarray(f_test_indices_raw.flatten()) # Flatten as it's already the result for indices

        # Time Reconstruction (Always on Device)
        f_deim = U_deim @ (EU_inv @ f_test_indices)
        if HAS_CUPY and xp == cp: cp.cuda.Stream.null.synchronize()
        t4 = time.perf_counter()
        
        elapsed = t4 - t1
    else:
        # Time Numerical Evaluation and Reconstruction
        t1 = time.perf_counter()
        f_test_indices_raw = f(mu_test, coords, indices)
        # Ensure f_test_indices is on the correct array module for the current xp
        f_test_indices = xp.asarray(f_test_indices_raw)

        f_deim = U_deim @ (EU_inv @ f_test_indices)
        if HAS_CUPY and xp == cp: cp.cuda.Stream.null.synchronize()
        t2 = time.perf_counter()
        
        elapsed = t2 - t1
        
    return f_deim, elapsed, coords # Return coords in case it was converted to jnp.ndarray

# Test inputs
def benchmark(U_deim, indices, coords, deim_func_sym):
    xp = cp.get_array_module(U_deim) if HAS_CUPY else np
    mu_tests = np.random.uniform(-1, -0.01, size=(num_test_runs, d))
    results = {}
    for method in ['symbolic', 'numerical']:
        times = []
        errors = []
        for mu in mu_tests:
            f_deim, t, coords = run_deim(U_deim, indices, mu, coords, deim_func_sym, method) # Pass coords, update if converted
            f_true = f(mu, coords) # Pass coords. f_true might be a JAX array if USE_JAX_SYMBOLIC is True
            # Ensure f_true is on the same array module as xp for norm computation
            if xp == cp and not isinstance(f_true, cp.ndarray):
                f_true = cp.asarray(f_true)
            elif xp == np and not isinstance(f_true, np.ndarray):
                f_true = np.asarray(f_true)
            err = xp.linalg.norm(f_true - f_deim) / xp.linalg.norm(f_true)
            times.append(t)
            errors.append(float(err))
        results[method] = {
            'stats': (np.mean(errors), np.std(errors), np.mean(times), np.std(times)),
            'raw_err': errors,
            'raw_time': times
        }
    
    print("\n\\begin{tabular}{lllr}")
    print("\\hline")
    print("Method & Selector & Rel. Error (mean$\\pm$std) & Time (ms, mean$\\pm$std)\\\\")
    print("\\hline")
    for method in ['symbolic', 'numerical']:
        err_m, err_s, t_m, t_s = results[method]['stats']
        method_label = 'A' if method == 'symbolic' else 'B'
        print(f"{method_label} & QDEIM & {err_m:.2e} $\\pm$ {err_s:.1e} & {t_m*1e3:.2f} $\\pm$ {t_s*1e3:.2f}\\\\")
    print("\\hline")
    print("\\end{tabular}")
    print(f"\nAll times are averages over {num_test_runs} runs in {d} dimensions.")
    print("Method A: Symbolic offline-online (SymPy substitution)")
    print("Method B: Numerical CSR-like (only required entries evaluated)")
    return results

if __name__ == '__main__':
    # Define a range of ell values to test
    ell_values = [5, 10, 20, 30, 40, 50] # Ensure ell <= num_snapshots
    ell_max = max(ell_values)
    
    # Storage for boxplots
    history = {'ell': [], 'err_A': [], 'time_A': [], 'err_B': [], 'time_B': []}

    # --- Offline Phase (Single Initialization) ---
    np.random.seed(0)
    mu_snapshots = np.random.uniform(-1, -0.01, size=(num_snapshots, d))
    # Use functools.partial to bind coords_np to f for multiprocessing
    f_for_pool = partial(f, coords=coords_np)
    print(f"Generating {num_snapshots} snapshots in parallel using {multiprocessing.cpu_count()} cores...")
    with multiprocessing.Pool() as pool:
        snapshot_list = pool.map(f_for_pool, mu_snapshots)
    snapshots = np.column_stack(snapshot_list)

    # Move coordinates to GPU for benchmarking after parallel CPU snapshot generation
    if HAS_CUPY and USE_GPU_ONLINE:
        coords = cp.array(coords_np)

    # Compute SVD once for the maximum required dimension
    if HAS_CUPY:
        print(f"Using CuPy for GPU-accelerated SVD on snapshots {snapshots.shape}...")
        snapshots_gpu = cp.array(snapshots)
        U_gpu, S_gpu, _ = cp.linalg.svd(snapshots_gpu, full_matrices=False)
        if USE_GPU_ONLINE:
            # Keep U and S on GPU for online phase
            U, S = U_gpu, S_gpu
        else:
            # Transfer back to CPU immediately for online phase
            U, S = U_gpu.get(), S_gpu.get()
    else:
        U, S, _ = np.linalg.svd(snapshots, full_matrices=False)

    # Selection step (QR) always happens on CPU as it is small
    U_deim_max = U[:, :ell_max]
    indices_max = qdeim_indices(U_deim_max)
    
    # Create the full symbolic matrix expression once
    full_symbolic_matrix_expr = _create_full_symbolic_matrix_expr(n, d)

    # --- Online Comparison Phase ---
    for ell in ell_values:
        print(f"\n--- Running benchmark for ell = {ell} ---")
        
        # Slice the basis and indices for the current ell
        U_deim = U[:, :ell]
        indices = indices_max[:ell]
        
        # Slice the full symbolic matrix expression using the current DEIM indices
        sliced_symbolic_expr = full_symbolic_matrix_expr[indices.tolist(), :]

        # Create or load lambdified function for *this specific ell and its indices*
        backend_label = "jax" if USE_JAX_SYMBOLIC else "numpy"
        grid_label = "sparse" if use_sparse_grid else "reg"
        
        # Hash the actual indices to ensure cache uniqueness
        import hashlib
        indices_hash = hashlib.md5(indices.tobytes()).hexdigest()[:8]
        expr_filename_ell = os.path.join("data", f"sym_deim_{grid_label}_d{d}_ell{ell}_{indices_hash}_{backend_label}.pickle")

        deim_func_sym_sliced = None
        if not FORCE_RECOMPUTE_SYM and os.path.exists(expr_filename_ell):
            print(f"Loading cached symbolic function for ell={ell} from {expr_filename_ell}...")
            with open(expr_filename_ell, 'rb') as f_in:
                deim_func_sym_sliced = pickle.load(f_in)
        else:
            print(f"Lambdifying sliced symbolic expression for ell={ell}...")
            if USE_JAX_SYMBOLIC:
                deim_func_sym_sliced = sp.lambdify([mu_sym, coords_sym], sliced_symbolic_expr, 'jax', cse=True)
            else:
                deim_func_sym_sliced = sp.lambdify([mu_sym, coords_sym], sliced_symbolic_expr, 'numpy', cse=True)
            
            os.makedirs("data", exist_ok=True)
            with open(expr_filename_ell, 'wb') as f_out:
                pickle.dump(deim_func_sym_sliced, f_out)

        # Pass the newly created/loaded sliced function to benchmark
        res = benchmark(U_deim, indices, coords, deim_func_sym_sliced)
        
        history['ell'].append(ell)
        history['err_A'].append(res['symbolic']['raw_err'])
        history['time_A'].append(np.array(res['symbolic']['raw_time']) * 1000) # to ms
        history['err_B'].append(res['numerical']['raw_err'])
        history['time_B'].append(np.array(res['numerical']['raw_time']) * 1000) # to ms

    # --- Plotting Performance Metrics ---
    plt.figure(figsize=(14, 6))

    # Define flierprops for Method A (lightblue, circle marker)
    flierprops_A = dict(marker='o', markerfacecolor='lightblue', markeredgecolor='lightblue', markersize=5, linestyle='none', alpha=0.5)
    # Define flierprops for Method B (orange, triangle marker '^')
    flierprops_B = dict(marker='^', markerfacecolor='orange', markeredgecolor='orange', markersize=5, linestyle='none', alpha=0.5)

    # Plot 1: Relative Error (Boxplots)
    ax1 = plt.subplot(1, 2, 1)
    ax1.boxplot(history['err_A'], positions=np.array(ell_values)-0.5, widths=1, patch_artist=True, 
                boxprops=dict(facecolor="lightblue", alpha=0.5), flierprops=flierprops_A, label='Method A')
    ax1.boxplot(history['err_B'], positions=np.array(ell_values)+0.5, widths=1, patch_artist=True, 
                boxprops=dict(facecolor="orange", alpha=0.5), flierprops=flierprops_B, label='Method B')
    ax1.set_yscale('log')
    ax1.set_xlabel(r'Reduced Dimension ($\ell$)')
    ax1.set_ylabel('Mean Relative Error')
    ax1.set_title(f'Approximation Error vs Complexity ({d}D Grid)')
    ax1.set_xticks(ell_values)
    ax1.set_xticklabels(ell_values)
    ax1.grid(True, which="both", ls="-", alpha=0.3)

    # Plot 2: Computation Time (Boxplots)
    ax2 = plt.subplot(1, 2, 2)
    ax2.boxplot(history['time_A'], positions=np.array(ell_values)-0.5, widths=1, patch_artist=True, 
                boxprops=dict(facecolor="lightblue", alpha=0.5), flierprops=flierprops_A, label='Method A')
    ax2.boxplot(history['time_B'], positions=np.array(ell_values)+0.5, widths=1, patch_artist=True, 
                boxprops=dict(facecolor="orange", alpha=0.5), flierprops=flierprops_B, label='Method B')
    ax2.set_xlabel(r'Reduced Dimension ($\ell$)')
    ax2.set_ylabel('Mean Online Time (ms)')
    ax2.set_title(rf'Online Evaluation Time vs $\ell$ ({d}D Grid)')
    ax2.set_xticks(ell_values)
    ax2.set_xticklabels(ell_values)
    ax2.legend()
    ax2.grid(True, ls="-", alpha=0.3)

    plt.tight_layout()
    
    # Save the figure and its metadata
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join("data", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"deim_benchmark_d{d}_n{n}.pdf")
    
    # Pass the history dictionary as fig_data for later adjustment
    save_figure(plt.gcf(), filename, fig_data=history)
    plt.show() # Uncomment to display plot interactively