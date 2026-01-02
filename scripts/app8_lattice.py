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
from System import MechSystem  # Keep for static properties

# Configure LaTeX rendering based on availability
if shutil.which('latex'):
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
    
    args = parser.parse_args()

    # Apply CLI arguments
    if args.no_latex:
        rc('text', usetex=False)

    if args.clean:
        checkpoint_path = os.path.join('data', f'{MechSystem.keep_time}')
        if os.path.exists(checkpoint_path):
            print(f"Cleaning checkpoint directory: {checkpoint_path}")
            shutil.rmtree(checkpoint_path)
    
    # %% Full order solution
    kwds = {
        "nosc": MechSystem.nosc,  # Read nosc from the central configuration
        "registered_solver_classes": [
            # ConformalImplicitMidpointSolver,
            DiscreteGradientSolver,
            ConformalStormerVerletSolver,
        ]
    }

    Omega2_space = (MechSystem._Omega2_space[:-1] if MechSystem.predict
                else MechSystem._Omega2_space)        
    Omega2_space_dim = len(Omega2_space)
    kwds.update({
        'Omega2_space': Omega2_space,
        'Omega2_space_dim': Omega2_space_dim
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
                memory='128GB',
                walltime='01:00:00',
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
        Omega2_space_ = (MechSystem._Omega2_space[-1:] if MechSystem.predict else MechSystem._Omega2_space)
        kwds.update({
            'Omega2_space': Omega2_space_,
            'Omega2_space_dim': len(Omega2_space_),
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
    array_shape = (len(kwds['registered_solver_classes']), 
                  MechSystem.dt_space_dim, 
                  Omega2_space_dim)

    time_lapsed = [reshape([x.time_lapsed for x in solvers], array_shape)]

    if not MechSystem.predict:
        # Calculate errors
        errors_r = reshape([np.amax(abs(y1 - y2)) 
            for y1, y2 in zip([x.y for x in solvers], 
            [x.y for x in solvers_r])], 
            array_shape)
        errors_dr = reshape([np.amax(abs(y1 - y2)) 
            for y1, y2 in zip([x.y for x in solvers], 
            [x.y for x in solvers_dr])], 
            array_shape)

        print('\n--- Max Solution Errors (over all frequencies) ---')
        error_matrices = [np.amax(errors_r, axis=2), np.amax(errors_dr, axis=2)]
        model_names_err = ["Reduced Model Error", "Hyper-reduced Model Error"]

        for i, error_matrix in enumerate(error_matrices):
            print(f"\n{model_names_err[i]}:")
            header = "Solver".ljust(35) + "".join([f"dt={dt:<11.4f}" for dt in MechSystem.dt_space])
            print(header)
            print("-" * len(header))
            for j, solver_class in enumerate(kwds['registered_solver_classes']):
                row_str = solver_class.__name__.ljust(35)
                for k in range(MechSystem.dt_space_dim):
                    val = error_matrix[j, k]
                    row_str += f"{val:<11.2e}"
                print(row_str)

        time_lapsed.append(reshape([x.time_lapsed for x in solvers_r], array_shape))
        # time_lapsed[0].shape) / time_lapsed[0] * 100)
        time_lapsed.append(reshape([x.time_lapsed for x in solvers_dr], array_shape))
        # time_lapsed[0].shape) / time_lapsed[0] * 100)
        time_lapsed[1] = time_lapsed[1] / time_lapsed[0] * 100
        time_lapsed[2] = time_lapsed[2] / time_lapsed[0] * 100

    else:
        time_lapsed.extend([
            reshape([x.time_lapsed for x in solvers_r], (*array_shape[:2], 1)),
            reshape([x.time_lapsed for x in solvers_dr], (*array_shape[:2], 1))
        ])
        
        # Calculate average full order time over all training frequencies
        avg_full_time = np.mean(time_lapsed[0], axis=2, keepdims=True)
        
        # Normalize reduced and hyper-reduced times
        time_lapsed[1] = time_lapsed[1] / avg_full_time * 100
        time_lapsed[2] = time_lapsed[2] / avg_full_time * 100

    print("\n--- Average Time Lapsed ---")
    if MechSystem.predict:
        model_names = ["Full Order Model (s)", "Reduced Model (% of Avg Full)", "Hyper-reduced Model (% of Avg Full)"]
    else:
        model_names = ["Full Order Model (s)", "Reduced Model (% of Full)", "Hyper-reduced Model (% of Full)"]

    # Average over the Omega2 dimension (axis=2)
    avg_times = [np.mean(tl, axis=2) for tl in time_lapsed]

    for i, avg_time_matrix in enumerate(avg_times):
        print(f"\n{model_names[i]}:")
        
        # Header
        header = "Solver".ljust(35) + "".join([f"dt={dt:<11.4f}" for dt in MechSystem.dt_space])
        print(header)
        print("-" * len(header))
        
        for j, solver_class in enumerate(kwds['registered_solver_classes']):
            row_str = solver_class.__name__.ljust(35)
            for k in range(MechSystem.dt_space_dim):
                val = avg_time_matrix[j, k]
                row_str += f"{val:<11.2f}"
            print(row_str)

    # tex_table('', time_lapsed)
    '''
    Observations:

    TODO:
    '''
