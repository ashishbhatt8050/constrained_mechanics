#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import shutil
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
from dask_jobqueue import SLURMCluster
from dask.distributed import Client, wait

# Local application imports
from ConcreteSolvers import (BaseSolverMixin, DiscreteGradientSolver, 
                            ConformalStormerVerletSolver, ConformalImplicitMidpointSolver)
from ReduceMechSystem import ReduceMechSystem
from System import MechSystem, HamiltonianMechSystem, LagrangianMechSystem, load_symbolic_expressions
from SymbolicComputer import IndexedBaseSymbolicComputer, manage_cache
from PlotScript import plot_omega_distribution, plot_pareto

# Configure LaTeX rendering based on availability
if False and shutil.which('latex'):
    rc('text', usetex=True)
    rc('text.latex', preamble=r'\usepackage{amsfonts}')  # Load AMSFonts for Fraktur
else:
    rc('text', usetex=False)
    print("LaTeX disabled or not found. Using standard fonts for plots.", flush=True)

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
    
    args = parser.parse_args()

    # Apply CLI arguments
    if args.no_latex:
        rc('text', usetex=False)

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
    cluster = None
    if False and shutil.which('sbatch') and not "PYTEST_CURRENT_TEST" in os.environ:
        try:
            print("SLURM detected. Initializing SLURMCluster...", flush=True)
            
            # Define environment variables for workers
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            src_path = os.path.join(project_root, "src")
            
            # Shell commands to run in the job script before the worker starts.
            job_prologue = [
                'ulimit -s unlimited', # Prevent stack overflow with large symbolic expressions.
                'OMP_NUM_THREADS=1',
                'MKL_NUM_THREADS=1',
                'OPENBLAS_NUM_THREADS=1',
                'NUMEXPR_NUM_THREADS=1',
            ]

            cluster = SLURMCluster(
                queue='workq',
                account='cpu_users',
                cores=32,
                # processes=8, # Unset to let dask-jobqueue use its default (often 1 process per core)
                memory='99GB',
                walltime='1-00:00:00',
                job_extra_directives=['--exclusive', '--output=dask_worker_%j.log', '--qos=cpu_users'],
                job_script_prologue=job_prologue,
            )
        except Exception as e:
            print(f"Failed to initialize SLURMCluster: {e}", flush=True)
    
    # Determine number of nodes needed for the full simulation
    num_full_tasks = len(kwds['registered_solver_classes']) * MechSystem.dt_space_dim * kwds['Omega2_space_dim']

    if cluster is not None:
        num_nodes = min(6, (num_full_tasks + 31) // 32) if num_full_tasks > 0 else 0
        if num_nodes > 0:
            print(f"Requesting {num_nodes} SLURM nodes for Dask workers...")
            cluster.scale(jobs=num_nodes)
    else:
        print("Initializing LocalCluster...", flush=True)
        from dask.distributed import LocalCluster
        cluster = LocalCluster()
        num_nodes = 1  # Assume local resources are available
    
    print(f"Dask Dashboard: {cluster.dashboard_link}", flush=True)

    client = None
    try:
        # Use a client if we have nodes, otherwise run sequentially
        if num_nodes > 0:
            client = Client(cluster)
            print("Waiting for workers to start...", flush=True)
            client.wait_for_workers(1)

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
        n_samples_list = [int(solver.n // 1) for solver in solvers]
        MechSystem.indices_list = [
            np.concatenate(([0], np.random.choice(np.arange(1, solver.n), n_samples - 1, replace=False)))
            for solver, n_samples in zip(solvers, n_samples_list)
        ]
        kwds.update({
            'Omega2_space': MechSystem.Omega2_space_test,
            'Omega2_space_dim': len(MechSystem.Omega2_space_test),
        })
        solvers_r = BaseSolverMixin.setup_and_solve_reduced_system(kwds, solvers, client=client)

        # --- Hyper-reduced model ---
        print('Computing hyper-reduction bases...')
        solvers_dr = BaseSolverMixin.setup_and_solve_hyperreduced_system(kwds, solvers, client=client)

    finally:
        if client:
            client.close()
        cluster.close()

    # %% Compute and display metrics
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