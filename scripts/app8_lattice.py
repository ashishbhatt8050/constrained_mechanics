#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import subprocess
import gc
import argparse
import concurrent.futures
import multiprocessing

# Set environment variables for thread control before importing numpy/scipy
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Increase recursion limit for deep symbolic trees
sys.setrecursionlimit(10000)

# Third-party imports
import numpy as np
import sympy as smp
from numpy import linalg as LA
from scipy import linalg as scipyLA
from pylab import log, r_, c_, sqrt, reshape, linspace, roll, figure
import cloudpickle as pickle
from tqdm.auto import tqdm
from matplotlib import rc

# Dask imports
from dask.distributed import Client, wait, LocalCluster

# Local application imports
from ConcreteSolvers import (BaseSolverMixin, DiscreteGradientSolver, 
                            ConformalStormerVerletSolver, ConformalImplicitMidpointSolver,
                            REDUCED_SOLVER_MAPPING, HYPERREDUCED_SOLVER_MAPPING)
from ReduceMechSystem import ReduceMechSystem
from System import MechSystem, HamiltonianMechSystem, LagrangianMechSystem, load_symbolic_expressions
from SymbolicComputer import IndexedBaseSymbolicComputer, manage_cache
from PlotScript import plot_omega_distribution, plot_pareto, plot_error_vs_basis_size

if __name__ == '__main__':
    """
    Model order reduction of the MechSystem using concrete solvers
    """
    # This print statement was moved from the top level of System.py
    # to prevent it from running on every Dask worker import.
    print(
        "NumPy RNG seed:",
        np.random.default_rng(seed=267257368022227711484290921317604022527).bit_generator._seed_seq.entropy,
    )

    parser = argparse.ArgumentParser(description="Constrained Mechanics Solver & Model Reduction")
    parser.add_argument("--no-latex", action="store_true", help="Disable LaTeX rendering for plots")
    parser.add_argument("--clean", action="store_true", help="Clean checkpoints before running")
    parser.add_argument("--clear-symbolic", action="store_true", help="Clear symbolic expressions cache before running")
    parser.add_argument("--clear-cache", action="store_true", help="Clear joblib cache before running")
    parser.add_argument("--no-pgfplots", action="store_true", help="Disable generation of data files for PGFPlots")
    
    args = parser.parse_args()

    # Set the pgfplots generation flag on the central config object
    MechSystem.generate_pgfplots_data = not args.no_pgfplots

    # Configure LaTeX rendering based on availability
    if args.no_latex or not shutil.which('latex'):
        rc('text', usetex=False)
        print("LaTeX disabled or not found. Using standard fonts for plots.", flush=True)
    elif shutil.which('latex'):
            rc('text', usetex=True)


    # Define paths and hash
    cache_dir = os.path.join('data', 'joblib_cache')
    checkpoint_path = os.path.join('data', f'{MechSystem.keep_time}')
    expressions_file = os.path.join('data', f"symbolic_expr_{MechSystem.nosc}_cse.pickle")
    config_hash = MechSystem.get_config_hash()

    # Manage joblib cache consistency
    manage_cache(MechSystem.nosc, config_hash, cache_dir, checkpoint_path,      expressions_file,
                 clean_cache=args.clear_cache,
                 clean_checkpoints=args.clean,
                 clean_symbolic=args.clear_symbolic)

    if not os.path.exists(expressions_file):
        print(f"Generating symbolic expressions for nosc={MechSystem.nosc}...")
        computer = IndexedBaseSymbolicComputer(MechSystem.nosc)
        expressions = {}
        tl = computer.compute_all(expressions)
        print(f'Computed symbolic expressions in {tl:.2f} seconds.')
        
        os.makedirs('data', exist_ok=True)
        with open(expressions_file, 'wb') as f:
            pickle.dump(expressions, f)
        
        # Load them now that they exist
        load_symbolic_expressions(HamiltonianMechSystem)
        load_symbolic_expressions(LagrangianMechSystem)

    if MechSystem.predict:
        print("Plotting Omega2 distribution...")
        plot_omega_distribution(MechSystem.Omega2_space, MechSystem.Omega2_space_test, 
                                filename=os.path.join(MechSystem.data_folder, "omega2_dist.pdf"))
    
    # %% Cluster set-up and system solution
    kwds = {
        "nosc": MechSystem.nosc,  # Read nosc from the central configuration
        "registered_solver_classes": [
            # ConformalImplicitMidpointSolver,
            DiscreteGradientSolver,
            ConformalStormerVerletSolver,
        ]
    }
    
    # Save original solver classes for reporting
    original_solver_classes = list(kwds['registered_solver_classes'])

    kwds.update({
        'Omega2_space': MechSystem.Omega2_space,
        'Omega2_space_dim': len(MechSystem.Omega2_space)
        })
    
    # Dask Cluster Configuration
    print("Initializing LocalCluster...", flush=True)
    
    n_workers = None
    threads_per_worker = 1
    
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        allocated_cpus = int(os.environ['SLURM_CPUS_PER_TASK'])
        n_workers = allocated_cpus # Start with 1 worker per core
        print(f"LocalCluster: Detected SLURM allocation of {allocated_cpus} CPUs.", flush=True)
        
        # Adjust workers based on memory to prevent OOM (aim for ~3GB/worker)
        min_mem_per_worker_mb = 3000
        total_mem_mb = None
        if 'SLURM_MEM_PER_NODE' in os.environ:
            total_mem_mb = int(os.environ['SLURM_MEM_PER_NODE'])
        elif 'SLURM_MEM_PER_CPU' in os.environ:
            total_mem_mb = int(os.environ['SLURM_MEM_PER_CPU']) * allocated_cpus
        
        if total_mem_mb:
            max_workers_mem = total_mem_mb // min_mem_per_worker_mb
            if max_workers_mem < n_workers:
                n_workers = max(1, max_workers_mem)
                print(f"Memory constrained ({total_mem_mb} MB). Reducing to {n_workers} workers.", flush=True)
        
        threads_per_worker = max(1, allocated_cpus // n_workers)
        
    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker)
    
    print(f"Dask Dashboard: {cluster.dashboard_link}", flush=True)
    
    if cluster.dashboard_link:
        try:
            from urllib.parse import urlparse
            parsed = urlparse(cluster.dashboard_link)
            port = parsed.port
            host = parsed.hostname
            print(f"\nTo access the dashboard from your local machine, run:")
            print(f"ssh -N -L {port}:{host}:{port} <your_username>@<cluster_login_node>")
            print(f"(If local port {port} is busy, try: ssh -N -L 8080:{host}:{port} ... and open http://localhost:8080/status)")
            print(f"Then open http://localhost:{port}/status in your browser.\n", flush=True)
        except Exception:
            pass

    client = Client(cluster)
    print("Waiting for workers to start...", flush=True)
    client.wait_for_workers(3)
    n_workers_actual = len(client.scheduler_info()['workers'])
    print(f"Dask cluster started with {n_workers_actual} workers.", flush=True)

    try:

        # --- Full order solution ---
        solvers = []
        print('Computing full solution ...')
        checkpoint_path = os.path.join('data', f'{MechSystem.keep_time}')
        kwds_file = os.path.join(checkpoint_path, 'kwds.joblib')
        solvers_file = os.path.join(checkpoint_path, 'solvers.joblib')

        if os.path.exists(checkpoint_path) and os.path.exists(kwds_file) and os.path.exists(solvers_file):
            print("Loading full solution from checkpoint...")
            with open(kwds_file, 'rb') as f: kwds = pickle.load(f)
            with open(solvers_file, 'rb') as f: solvers = pickle.load(f)
        else:
            BaseSolverMixin.parallel_solve_mech_system(kwds, solvers, client=client)
            os.makedirs(checkpoint_path, exist_ok=True)
            print("Creating new checkpoint for full model...")
            with open(kwds_file, 'wb') as f: pickle.dump(kwds, f)
            with open(solvers_file, 'wb') as f: pickle.dump(solvers, f)
        BaseSolverMixin.measures(kwds, solvers)

        # --- Reduced order solution ---
        print('Computing reduced bases...')
        MechSystem.indices_list = [np.arange(solver.n) for solver in solvers] # select solutions for POD (shuffle indices or implement a stride)
        kwds.update({
            'Omega2_space': MechSystem.Omega2_space_test,
            'Omega2_space_dim': len(MechSystem.Omega2_space_test),
        })

        # --- Basis Size Study ---
        study_data = {cls.__name__: {'sizes': [], 'sizes_dr': [], 'err_r': [], 'err_dr': [], 'times_r': [], 'times_dr': []} for cls in original_solver_classes}

        for tol in MechSystem.pod_tol_sweep:
            print(f"\n>>> Running MOR sweep for tolerance: {tol}")
            MechSystem.pod_tol_ham = MechSystem.pod_tol_dg = tol
            
            # Reset registered classes to original full-order versions at start of each sweep
            kwds['registered_solver_classes'] = list(original_solver_classes)

            # Solve Reduced and Hyper-reduced
            solvers_r = BaseSolverMixin.setup_and_solve_reduced_system(kwds, solvers, client=client)
            solvers_dr = BaseSolverMixin.setup_and_solve_hyperreduced_system(kwds, solvers, client=client)

            # Calculate errors for this tolerance
            for cls_idx, cls in enumerate(original_solver_classes):
                errs_r_for_tol = []
                errs_dr_for_tol = []
                nosc_r_for_tol = np.nan # Will be set by the first successful solver
                nosc_dr_for_tol = np.nan

                # Iterate through all solvers to ensure strict alignment between full, reduced, and hyper-reduced results
                for s_f, s_r, s_dr in zip(solvers, solvers_r, solvers_dr):
                    # Only calculate error if the full order reference corresponds to the current class
                    if s_f is not None and s_f.solver_class == cls:
                        # Convergence check: both reduced and hyper-reduced solvers must have succeeded
                        if s_r is not None and s_dr is not None:
                            current_err_r = np.amax(abs(s_f.y - s_r.y)) / np.amax(abs(s_f.y))
                            current_err_dr = np.amax(abs(s_f.y - s_dr.y)) / np.amax(abs(s_f.y))

                            errs_r_for_tol.append(current_err_r)
                            errs_dr_for_tol.append(current_err_dr)

                            if np.isnan(nosc_r_for_tol): # Capture the basis size for this tolerance level
                                nosc_r_for_tol = s_r.nosc_r                            
                            if np.isnan(nosc_dr_for_tol):
                                nosc_dr_for_tol = s_dr.RBxUx_inv_PxU.shape[1]//2

                # Capture raw timing and errors for box plots
                times_r_list = [s.time_lapsed[0] if s else np.nan for s_f, s in zip(solvers, solvers_r) if s_f is not None and s_f.solver_class == cls]
                times_dr_list = [s.time_lapsed[0] if s else np.nan for s_f, s in zip(solvers, solvers_dr) if s_f is not None and s_f.solver_class == cls]
                
                study_data[cls.__name__]['times_r'].append(np.array(times_r_list).reshape(MechSystem.dt_space_dim, -1))
                study_data[cls.__name__]['times_dr'].append(np.array(times_dr_list).reshape(MechSystem.dt_space_dim, -1))

                # Print success counts for monitoring
                num_success = len(errs_r_for_tol)
                num_expected = sum(1 for s in solvers if s and s.solver_class == cls)
                print(f"  [{cls.__name__}] Successful solvers for tol={tol}: {num_success}/{num_expected}")

                if not np.isnan(nosc_r_for_tol): # Only append if at least one solver succeeded for this tol
                    study_data[cls.__name__]['sizes'].append(nosc_r_for_tol)
                    study_data[cls.__name__]['sizes_dr'].append(nosc_dr_for_tol)
                    study_data[cls.__name__]['err_r'].append(np.array(errs_r_for_tol).reshape(MechSystem.dt_space_dim, -1))
                    study_data[cls.__name__]['err_dr'].append(np.array(errs_dr_for_tol).reshape(MechSystem.dt_space_dim, -1))
                else:
                    print(f"  [Warning] Solver {cls.__name__} failed for tol={tol}. Skipping error calculation.")

                    # Maintain consistent list lengths for plotting by appending NaNs
                    study_data[cls.__name__]['sizes'].append(np.nan)
                    study_data[cls.__name__]['sizes_dr'].append(np.nan)
                    study_data[cls.__name__]['err_r'].append(np.nan)
                    study_data[cls.__name__]['err_dr'].append(np.nan)

        # Debugging: Confirm data collection sizes
        print("\n>>> Debug: Timing data collection sizes:")
        for name in study_data:
            t_r = study_data[name]['times_r']
            print(f"  {name:40} | times_r: {len(t_r)} (tols) x {t_r[0].shape if len(t_r)>0 else 'N/A'} (dt x params)")

        # Prepare raw full order data
        f_times_raw = {cls.__name__: np.array([s.time_lapsed[0] for s in solvers if s.solver_class == cls]).reshape(MechSystem.dt_space_dim, -1) for cls in original_solver_classes}
        f_errs_raw = {cls.__name__: np.array([s.en_error if hasattr(s, 'en_error') else np.nan for s in solvers if s.solver_class == cls]).reshape(MechSystem.dt_space_dim, -1) for cls in original_solver_classes}

        plot_error_vs_basis_size(
            [study_data[n]['sizes'] for n in study_data],
            [study_data[n]['sizes_dr'] for n in study_data],
            [study_data[n]['err_r'] for n in study_data],
            [study_data[n]['err_dr'] for n in study_data],
            list(study_data.keys()), MechSystem.pod_tol_sweep, MechSystem.data_folder,
            dt_space=MechSystem.dt_space,
            full_times_raw=f_times_raw,
            full_errs_raw=f_errs_raw,
            times_r_raw=[study_data[n]['times_r'] for n in study_data],
            times_dr_raw=[study_data[n]['times_dr'] for n in study_data]
        )

    finally:
        if client:
            client.close()
        cluster.close()

    # %% Compute and display metrics
    print(f"\nFinal Reporting: Metrics correspond to tolerance {MechSystem.pod_tol_sweep[-1]}")
    array_shape_train = (len(kwds['registered_solver_classes']), 
                  MechSystem.dt_space_dim, 
                  len(MechSystem.Omega2_space))
    
    array_shape_test = (len(kwds['registered_solver_classes']), 
                  MechSystem.dt_space_dim, 
                  len(MechSystem.Omega2_space_test))

    time_lapsed = [reshape([x.time_lapsed[0] if x is not None else np.nan for x in solvers], array_shape_train)]

    if not MechSystem.predict:
        # Calculate errors
        errors_r = reshape([np.amax(abs(s1.y - s2.y)) if s1 is not None and s2 is not None else np.nan
            for s1, s2 in zip(solvers, solvers_r)], 
            array_shape_train)
        errors_dr = reshape([np.amax(abs(s1.y - s2.y)) if s1 is not None and s2 is not None else np.nan
            for s1, s2 in zip(solvers, solvers_dr)], 
            array_shape_train)

        print('\n--- Max Solution Errors (over all frequencies) ---')
        error_matrices = [np.nanmax(errors_r, axis=2), np.nanmax(errors_dr, axis=2)]
        model_names_err = ["Reduced Model Error", "Hyper-reduced Model Error"]

        for i, error_matrix in enumerate(error_matrices):
            print(f"\n{model_names_err[i]}:")
            header = "Solver".ljust(45) + "".join([f"dt={dt:<11.4f}" for dt in MechSystem.dt_space])
            print(header)
            print("-" * len(header))
            for j, solver_class in enumerate(original_solver_classes):
                row_str = solver_class.__name__.ljust(45)
                for k in range(MechSystem.dt_space_dim):
                    val = error_matrix[j, k]
                    row_str += f"{val:<11.2e}"
                print(row_str)

        time_lapsed.append(reshape([x.time_lapsed[0] if x is not None else np.nan for x in solvers_r], array_shape_train))
        # time_lapsed[0].shape) / time_lapsed[0] * 100)
        time_lapsed.append(reshape([x.time_lapsed[0] if x is not None else np.nan for x in solvers_dr], array_shape_train))
        # time_lapsed[0].shape) / time_lapsed[0] * 100)
        time_lapsed[1] = time_lapsed[1] / time_lapsed[0] * 100
        time_lapsed[2] = time_lapsed[2] / time_lapsed[0] * 100

    else:
        time_lapsed.extend([
            reshape([x.time_lapsed[0] if x is not None else np.nan for x in solvers_r], array_shape_test),
            reshape([x.time_lapsed[0] if x is not None else np.nan for x in solvers_dr], array_shape_test)
        ])
        
        # Calculate average full order time over all training frequencies
        avg_full_time = np.nanmean(time_lapsed[0], axis=2, keepdims=True)
        
        # Normalize reduced and hyper-reduced times
        time_lapsed[1] = time_lapsed[1] / avg_full_time * 100
        time_lapsed[2] = time_lapsed[2] / avg_full_time * 100

    print("\n--- Average Time Lapsed ---")
    if MechSystem.predict:
        model_names = ["Full Order Model (s)", "Reduced Model (% of Avg Full)", "Hyper-reduced Model (% of Avg Full)"]
    else:
        model_names = ["Full Order Model (s)", "Reduced Model (% of Full)", "Hyper-reduced Model (% of Full)"]

    # Average over the Omega2 dimension (axis=2)
    avg_times = [np.nanmean(tl, axis=2) for tl in time_lapsed]

    for i, avg_time_matrix in enumerate(avg_times):
        print(f"\n{model_names[i]}:")
        
        # Header
        header = "Solver".ljust(45) + "".join([f"dt={dt:<11.4f}" for dt in MechSystem.dt_space])
        print(header)
        print("-" * len(header))
        
        for j, solver_class in enumerate(original_solver_classes):
            row_str = solver_class.__name__.ljust(45)
            for k in range(MechSystem.dt_space_dim):
                val = avg_time_matrix[j, k]
                row_str += f"{val:<11.2f}"
            print(row_str)

    # %% Failure Report
    def print_failure_report(stage, solver_list, omega2_space):
        if not solver_list: return
        
        print(f"\n--- {stage} Failure Report ---")
        failures = []
        idx = 0
        # The iteration order must match ConcreteSolvers.parallel_solve_mech_system
        for solver_cls in original_solver_classes:
            for dt in MechSystem.dt_space:
                for omega2 in omega2_space:
                    if idx < len(solver_list):
                        if solver_list[idx] is None:
                            # Format omega2 for display (it's a vector)
                            omega2_str = f"[{omega2[0]:.2f}, ...]" if len(omega2) > 0 else "[]"
                            failures.append(f"{solver_cls.__name__:<40} | dt={dt:<8.4f} | Omega2={omega2_str}")
                    idx += 1
        
        if failures:
            print(f"Total Failures: {len(failures)}")
            print(f"{'Solver':<40} | {'Time Step':<11} | {'Parameter'}")
            print("-" * 80)
            for f in failures:
                print(f)
        else:
            print("No failures.")

    print_failure_report("Full Order Model", solvers, MechSystem.Omega2_space)
    if 'solvers_r' in locals():
        print_failure_report("Reduced Order Model", solvers_r, MechSystem.Omega2_space_test)
    if 'solvers_dr' in locals():
        print_failure_report("Hyper-reduced Model", solvers_dr, MechSystem.Omega2_space_test)

    # %% Pareto plots
    if not MechSystem.predict and 'errors_r' in locals() and 'errors_dr' in locals():
        print("\nGenerating Pareto plots...")
        solver_names = [cls.__name__ for cls in original_solver_classes]
        plot_pareto(time_lapsed, errors_r, errors_dr, solver_names, MechSystem.dt_space, MechSystem.data_folder)