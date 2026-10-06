#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import copy
import shutil
import argparse
import numpy as np
from numpy import linalg as LA
from matplotlib import rc

# Set environment variables for single-thread control per worker
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

sys.setrecursionlimit(10000)

from dask.distributed import Client, LocalCluster

# Local application imports
from ConcreteSolvers import (BaseSolverMixin, ConformalStormerVerletSolver,
                            _matches_solver_family)
from System import (MechSystem, HamiltonianMechSystem, load_symbolic_expressions,
                    fast_dump, fast_load, get_data_dir, check_checkpoint_exists)
from SymbolicComputer import manage_cache
from ElasticaSymbolicComputer import ElasticaSymbolicComputer
from ElasticaSystem import (ElasticaConfig, ElasticaMechSystem,
                            get_symbolic_expressions_file_elastica,
                            make_elastica_initial_conditions,
                            plot_elastica_beam)
from PlotScript import (plot_omega_distribution, plot_error_vs_basis_size,
                        plot_prediction_results, plot_solvers,
                        plot_prediction_sanity_check, plot_combined_errors)


def _collect_metric(solver_list, cls, metric_fn):
    values = [metric_fn(s) if s and _matches_solver_family(s, cls) else np.nan
              for s in solver_list]
    return np.array(values).reshape(MechSystem.dt_space_dim, -1)


_PREDICTION_METRICS = [
    ('sym_error',      lambda s: np.amax(s.sym_error) if getattr(s, 'sym_error', None) is not None else np.nan,  lambda sl: any(s is not None and getattr(s, 'sym_error', None) is not None for s in sl)),
    ('energy_error',   lambda s: np.amax(s.eng_error) if getattr(s, 'eng_error', None) is not None else np.nan,  lambda sl: any(s is not None and getattr(s, 'eng_error', None) is not None for s in sl)),
    ('phase_error',    lambda s: getattr(s, 'phase_error', np.nan),  lambda sl: True),
    ('lin_mom_error',  lambda s: getattr(s, 'lin_mom_error', np.nan), lambda sl: True),
    ('ang_mom_error',  lambda s: getattr(s, 'ang_mom_error', np.nan), lambda sl: True),
]


def run_prediction_sanity_check(kwds, solvers_r, solvers_hr, original_solver_classes, **kwargs):
    """
    Sanity check for prediction experiments on Constrained Euler Elastica:
    Solves a FOM for the last unseen Omega2 test parameter (bending stiffnesses),
    directly compares trajectories, tip deflections, and physical invariants against
    ROM and HROM, prints a summary, and saves diagnostic comparison plots.
    """
    if not getattr(MechSystem, 'predict', False):
        return

    if solvers_hr is None:
        solvers_hr = kwargs.get('solvers_dr', None)

    if not solvers_r or not solvers_hr:
        print("\n[Sanity Check] No ROM/HROM solvers available to compare against. Skipping.")
        return

    last_omega2 = MechSystem.Omega2_space_test[-1]
    tol = kwds.get('pod_tol', MechSystem.pod_tol_sweep[-1])

    print("\n" + "=" * 88)
    print(">>> [SANITY CHECK] Evaluating Prediction Fidelity on Unseen Elastica Test Parameter")
    print(f"    Omega2 (test[-1]): {np.array2string(last_omega2, precision=3, separator=', ')}")
    print(f"    Basis Tolerance:   {tol}")
    print("=" * 88)

    for cls in original_solver_classes:
        cls_name = cls.__name__
        for dt in MechSystem.dt_space:
            # 1. Match ROM and HROM solvers from the test sweep
            s_r = next((s for s in solvers_r if s is not None 
                        and _matches_solver_family(s, cls) 
                        and np.isclose(s.dt, dt) 
                        and np.allclose(s.Omega2, last_omega2)), None)
            s_hr = next((s for s in solvers_hr if s is not None 
                         and _matches_solver_family(s, cls) 
                         and np.isclose(s.dt, dt) 
                         and np.allclose(s.Omega2, last_omega2)), None)

            if s_r is None or s_hr is None:
                print(f"  [Warning] Missing ROM or HROM result for {cls_name} (dt={dt:.4f}). Skipping.")
                continue

            # 2. Run the ground-truth FOM solve for this test parameter
            print(f"\n--> Running test FOM solve: {cls_name} (dt={dt:.4f}) ...", flush=True)
            kwds_fom = {
                'nosc': MechSystem.nosc,
                'registered_solver_classes': [cls],
            }
            s_f = BaseSolverMixin.solve_mech_system(cls, dt, last_omega2, kwds_fom)

            # 3. Compute trajectory difference metrics
            fom_norm_inf = np.amax(np.abs(s_f.y))
            fom_norm_frob = LA.norm(s_f.y)

            # Max absolute and relative errors
            err_r_inf = np.amax(np.abs(s_f.y - s_r.y))
            err_r_rel = err_r_inf / (fom_norm_inf + 1e-15)
            err_r_l2 = LA.norm(s_f.y - s_r.y) / (fom_norm_frob + 1e-15)

            err_hr_inf = np.amax(np.abs(s_f.y - s_hr.y))
            err_hr_rel = err_hr_inf / (fom_norm_inf + 1e-15)
            err_hr_l2 = LA.norm(s_f.y - s_hr.y) / (fom_norm_frob + 1e-15)

            # Position (q) and momentum (p) sub-state errors
            nosc = MechSystem.nosc
            n_nodes = nosc // 3
            err_r_q = np.amax(np.abs(s_f.y[:, :nosc] - s_r.y[:, :nosc]))
            err_r_p = np.amax(np.abs(s_f.y[:, nosc:] - s_r.y[:, nosc:]))
            err_hr_q = np.amax(np.abs(s_f.y[:, :nosc] - s_hr.y[:, :nosc]))
            err_hr_p = np.amax(np.abs(s_f.y[:, nosc:] - s_hr.y[:, nosc:]))

            # Tip deflection extraction (z_tip at node n_nodes-1)
            coords_f = s_f.y[:, :nosc].reshape(-1, n_nodes, 3)
            coords_r = s_r.y[:, :nosc].reshape(-1, n_nodes, 3)
            coords_hr = s_hr.y[:, :nosc].reshape(-1, n_nodes, 3)
            z_tip_f = coords_f[:, -1, 2]
            z_tip_r = coords_r[:, -1, 2]
            z_tip_hr = coords_hr[:, -1, 2]
            err_tip_r = np.amax(np.abs(z_tip_f - z_tip_r))
            err_tip_hr = np.amax(np.abs(z_tip_f - z_tip_hr))

            # Runtimes & Speedups
            t_f = s_f.time_lapsed[0]
            t_r = s_r.time_lapsed[0]
            t_hr = s_hr.time_lapsed[0]
            sp_r = t_f / t_r if t_r > 0 else np.nan
            sp_hr = t_f / t_hr if t_hr > 0 else np.nan

            # Invariant errors
            eng_f = np.amax(np.abs(s_f.eng_error)) if getattr(s_f, 'eng_error', None) is not None else np.nan
            eng_r = np.amax(np.abs(s_r.eng_error)) if getattr(s_r, 'eng_error', None) is not None else np.nan
            eng_hr = np.amax(np.abs(s_hr.eng_error)) if getattr(s_hr, 'eng_error', None) is not None else np.nan

            sym_f = np.amax(np.abs(s_f.sym_error)) if getattr(s_f, 'sym_error', None) is not None else np.nan
            sym_r = np.amax(np.abs(s_r.sym_error)) if getattr(s_r, 'sym_error', None) is not None else np.nan
            sym_hr = np.amax(np.abs(s_hr.sym_error)) if getattr(s_hr, 'sym_error', None) is not None else np.nan

            # Basis dimensions
            rom_dim = f"{2 * s_r.nosc_r}" if hasattr(s_r, 'nosc_r') else "N/A"
            hrom_dim = f"{s_hr.deim_field.shape[1]}" if hasattr(s_hr, 'deim_field') and s_hr.deim_field is not None else str(getattr(s_hr, 'nosc_r', 'N/A'))

            # 4. Print Summary Table
            print(f"\n--- Sanity Check Summary: {cls_name} (dt={dt:.4f}) ---")
            print(f"{'Model':<8} | {'Dim':<8} | {'Rel-Linf Err':<13} | {'Rel-L2 Err':<12} | {'Tip Err (m)':<12} | {'Max ΔH':<10} | {'Max ΔSp':<10} | {'Time (s)':<9} | {'Speedup'}")
            print("-" * 103)
            print(f"{'FOM':<8} | {2*nosc:<8} | {'Reference':<13} | {'Reference':<12} | {'Reference':<12} | {eng_f:<10.2e} | {sym_f:<10.2e} | {t_f:<9.3f} | 1.00x")
            print(f"{'ROM':<8} | {rom_dim:<8} | {err_r_rel:<13.2e} | {err_r_l2:<12.2e} | {err_tip_r:<12.2e} | {eng_r:<10.2e} | {sym_r:<10.2e} | {t_r:<9.3f} | {sp_r:.2f}x")
            print(f"{'HROM':<8} | {hrom_dim:<8} | {err_hr_rel:<13.2e} | {err_hr_l2:<12.2e} | {err_tip_hr:<12.2e} | {eng_hr:<10.2e} | {sym_hr:<10.2e} | {t_hr:<9.3f} | {sp_hr:.2f}x")
            print(f"  Component breakdown: ROM (q_err={err_r_q:.2e}, p_err={err_r_p:.2e}) | HROM (q_err={err_hr_q:.2e}, p_err={err_hr_p:.2e})")

            # 5. Generate Diagnostic Plot
            plot_path = os.path.join(MechSystem.data_folder, f"sanity_check_{cls_name}_dt_{dt:.4f}.pdf")
            plot_prediction_sanity_check(
                s_f, s_r, s_hr,
                plot_path=plot_path,
                coord_series=(z_tip_f, z_tip_r, z_tip_hr),
                coord_label=r"Tip deflection $z_{\mathrm{tip}}(t)$",
                coord_ylabel="Tip deflection",
                system_label="Elastica",
                target_idx=len(MechSystem.Omega2_space_test) - 1,
                target_omega2=last_omega2,
                cls_name=cls_name,
                dt=dt,
            )
            print(f"  Saved comparison plot to: {plot_path}")

            # 6. Generate dedicated 3D/2D Elastica beam plot for this test parameter
            beam_test_pdf = f"elastica_beam_test_sanity_check_{cls_name}_dt_{dt:.4f}.pdf"
            plot_elastica_beam([s_f], [s_r], [s_hr], data_folder=MechSystem.data_folder, filename=beam_test_pdf)
            print(f"  Saved Elastica 3D/2D beam sanity check plot to: {os.path.join(MechSystem.data_folder, beam_test_pdf)}")


if __name__ == '__main__':
    """
    Symplectic Model Order Reduction and Adaptive Traveling Bases for
    the Constrained Discrete Euler Elastica / Flexible Beam.
    """
    parser = argparse.ArgumentParser(description="Constrained Elastica Hamiltonian MOR")
    parser.add_argument("--no-latex", action="store_true", help="Disable LaTeX rendering for plots")
    parser.add_argument("--clean", action="store_true", help="Clean checkpoints before running")
    parser.add_argument("--clear-symbolic", action="store_true", help="Clear symbolic expressions cache")
    parser.add_argument("--clear-cache", action="store_true", help="Clear joblib cache")
    parser.add_argument("--no-pgfplots", action="store_true", help="Disable PGFPlots data generation")
    parser.add_argument("--n-nodes", type=int, default=None, help="Number of nodes in beam (default: 21, or 9 for --quick)")
    parser.add_argument("--t-final", type=float, default=None, help="Final simulation time (default: 3.0)")
    parser.add_argument("--n-samples", type=int, default=None, help="Number of parameter samples (default: 15)")
    parser.add_argument("--n-workers", type=int, default=None, help="Number of Dask workers (default: 2 on login node)")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test preset for login nodes (short time, few nodes & samples)")
    parser.add_argument("--reproduce", action="store_true", help="Run reproduction mode instead of prediction")
    args = parser.parse_args()

    # 1. Update config based on CLI
    if args.quick:
        n_nodes = 9 if args.n_nodes is None else args.n_nodes
        t_final = 0.2 if args.t_final is None else args.t_final
        n_samples = 4 if args.n_samples is None else args.n_samples
        args.no_latex = True
        args.no_pgfplots = True
        print(f"Quick smoke-test mode: n_nodes={n_nodes}, t_final={t_final}s, n_samples={n_samples}")
    else:
        n_nodes = 21 if args.n_nodes is None else args.n_nodes
        t_final = 3.0 if args.t_final is None else args.t_final
        n_samples = 15 if args.n_samples is None else args.n_samples

    nosc = n_nodes * 3
    ElasticaConfig.n_nodes = n_nodes
    ElasticaConfig.nosc = nosc
    ElasticaConfig.ds = ElasticaConfig.length / (n_nodes - 1)
    ElasticaConfig.num_joints = n_nodes - 2
    ElasticaConfig.T_final = t_final
    ElasticaConfig._Omega2_space_dim = n_samples

    if args.reproduce:
        ElasticaConfig.predict = False

    # 2. Configure MechSystem
    MechSystem.nosc = nosc
    MechSystem.predict = ElasticaConfig.predict
    MechSystem.dt_space = ElasticaConfig.dt_space
    MechSystem.dt_space_dim = len(ElasticaConfig.dt_space)
    MechSystem.T_final = ElasticaConfig.T_final
    MechSystem.pod_tol_sweep = ElasticaConfig.pod_tol_sweep
    MechSystem.tol = ElasticaConfig.tol
    MechSystem.tol_reduced = ElasticaConfig.tol_reduced
    MechSystem.M = ElasticaConfig.M
    MechSystem.data_folder = ElasticaMechSystem.data_folder
    os.makedirs(MechSystem.data_folder, exist_ok=True)

    # Recompute initial state satisfying g(q0) = 0 and d/dt g(q0, p0) = 0
    MechSystem.y_init = make_elastica_initial_conditions(n_nodes, ElasticaConfig.ds, ElasticaConfig.boundary)
    q0_dot_p0 = np.dot(MechSystem.y_init[:nosc], MechSystem.y_init[nosc:])
    print(f"Verified initial conditions: d/dt g(q0, p0) = 0 strictly asserted (q0 . p0 = {q0_dot_p0:.2e})")

    # Parameter space for bending stiffnesses
    rng = np.random.default_rng(seed=267257368022227711484290921317604022527)
    _Omega2_space = 2.0 + 8.0 * rng.random((ElasticaConfig._Omega2_space_dim, ElasticaConfig.num_joints))
    _Omega2_space = np.sort(_Omega2_space, axis=1)
    if MechSystem.predict:
        split_idx = max(1, int(ElasticaConfig._Omega2_space_dim * ElasticaConfig.train_ratio))
        MechSystem.Omega2_space = _Omega2_space[:split_idx]
        MechSystem.Omega2_space_test = _Omega2_space[split_idx:]
    else:
        MechSystem.Omega2_space = MechSystem.Omega2_space_test = _Omega2_space

    # LaTeX configuration
    if args.no_latex or not shutil.which('latex'):
        rc('text', usetex=False)
    elif shutil.which('latex'):
        rc('text', usetex=True)

    # Use isolated joblib cache for Elastica to prevent collisions with double helix lattice
    cache_dir = get_data_dir('joblib_cache_elastica')
    os.makedirs(cache_dir, exist_ok=True)
    checkpoint_path = MechSystem.data_folder
    expressions_file = get_symbolic_expressions_file_elastica(nosc)
    config_hash = MechSystem.get_config_hash()

    manage_cache(nosc, config_hash, cache_dir, checkpoint_path, expressions_file,
                 clean_cache=args.clear_cache,
                 clean_checkpoints=args.clean,
                 clean_symbolic=args.clear_symbolic)

    os.makedirs(checkpoint_path, exist_ok=True)

    # 3. Compute or load symbolic expressions
    if not os.path.exists(expressions_file):
        print(f"Generating Elastica symbolic expressions for nosc={nosc} (nodes={n_nodes})...")
        computer = ElasticaSymbolicComputer(nosc, length=ElasticaConfig.length,
                                            mass_total=float(n_nodes),
                                            gravity=ElasticaConfig.gravity,
                                            boundary=ElasticaConfig.boundary)
        expressions = {}
        tl = computer.compute_all(expressions)
        print(f"Computed Elastica symbolic expressions in {tl:.2f} seconds.")
        os.makedirs(os.path.dirname(expressions_file), exist_ok=True)
        fast_dump(expressions, expressions_file)
        persistent_file = os.path.join('data', f"symbolic_expr_elastica_{nosc}_cse.pickle")
        if os.path.abspath(expressions_file) != os.path.abspath(persistent_file):
            os.makedirs('data', exist_ok=True)
            fast_dump(expressions, persistent_file)

    load_symbolic_expressions(HamiltonianMechSystem, expressions_file=expressions_file)
    print(f"Loaded Elastica symbolic expressions from {expressions_file}")

    if MechSystem.predict:
        print("Plotting Omega2 bending stiffness distribution...")
        plot_omega_distribution(MechSystem.Omega2_space, MechSystem.Omega2_space_test,
                                filename=os.path.join(MechSystem.data_folder, "omega2_dist.pdf"))

    # 4. Cluster Setup
    original_solver_classes = [ConformalStormerVerletSolver]
    base_kwds = {
        "nosc": nosc,
        "registered_solver_classes": list(original_solver_classes),
        'Omega2_space': MechSystem.Omega2_space,
        'Omega2_space_dim': len(MechSystem.Omega2_space),
        'pod_tol_sweep': MechSystem.pod_tol_sweep
    }

    # Determine number of workers safely:
    if args.n_workers is not None:
        n_workers = args.n_workers
    elif 'SLURM_CPUS_PER_TASK' in os.environ:
        n_workers = int(os.environ['SLURM_CPUS_PER_TASK'])
    else:
        n_workers = 2  # Safe default on login node

    print(f"Initializing LocalCluster for Elastica with {n_workers} worker(s)...", flush=True)
    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
    client = Client(cluster)
    client.wait_for_workers(1)
    print(f"Dask cluster started with {len(client.scheduler_info()['workers'])} workers.", flush=True)

    # Initialize all Dask worker processes with the Elastica symbolic expressions and configuration
    def init_elastica_worker(expr_file, nosc_val, y_init_val, t_final_val):
        from System import MechSystem, HamiltonianMechSystem, load_symbolic_expressions
        MechSystem.nosc = nosc_val
        MechSystem.y_init = y_init_val
        MechSystem.T_final = t_final_val
        load_symbolic_expressions(HamiltonianMechSystem, expressions_file=expr_file)

    client.run(init_elastica_worker, expressions_file, nosc, MechSystem.y_init, MechSystem.T_final)

    try:
        # --- Full order solution ---
        solvers = []
        print("Computing full Elastica solution...")
        kwds_file = os.path.join(checkpoint_path, 'kwds.joblib')
        solvers_file = os.path.join(checkpoint_path, 'solvers.joblib')
        kwds_f = copy.deepcopy(base_kwds)

        if check_checkpoint_exists(checkpoint_path, kwds_file, solvers_file):
            print("Loading full Elastica solution from checkpoint...")
            kwds_f = fast_load(kwds_file)
            solvers = fast_load(solvers_file)
        else:
            BaseSolverMixin.parallel_solve_mech_system(kwds_f, solvers, client=client)
            os.makedirs(checkpoint_path, exist_ok=True)
            print("Creating new checkpoint for full Elastica model...")
            fast_dump(kwds_f, kwds_file)
            fast_dump(solvers, solvers_file)

        BaseSolverMixin.compute_convergence_rates(kwds_f, solvers)
        plot_solvers(kwds_f, solvers)

        # --- Reduced order setup ---
        print("Setting up snapshot indices for Elastica POD...")
        MechSystem.indices_list = [np.arange(solver.n) for solver in solvers]

        study_data = {cls.__name__: {
            'sizes': [], 'sizes_hr': [], 'err_r': [], 'err_hr': [],
            'times_f': [], 'times_r': [], 'times_hr': [],
            'sym_error_f': [], 'sym_error_r': [], 'sym_error_hr': [],
            'phase_error_f': [], 'phase_error_r': [], 'phase_error_hr': [],
            'energy_error_f': [], 'energy_error_r': [], 'energy_error_hr': [],
            'lin_mom_error_f': [], 'lin_mom_error_r': [], 'lin_mom_error_hr': [],
            'ang_mom_error_f': [], 'ang_mom_error_r': [], 'ang_mom_error_hr': []
        } for cls in original_solver_classes}

        for tol in base_kwds['pod_tol_sweep']:
            print(f"\n>>> Running MOR sweep for tolerance: {tol}")

            kwds_r = copy.deepcopy(base_kwds)
            kwds_r.update({
                'pod_tol': tol,
                'Omega2_space': MechSystem.Omega2_space_test,
                'Omega2_space_dim': len(MechSystem.Omega2_space_test),
            })

            kwds_hr = copy.deepcopy(base_kwds)
            kwds_hr.update({
                'pod_tol': tol,
                'Omega2_space': MechSystem.Omega2_space_test,
                'Omega2_space_dim': len(MechSystem.Omega2_space_test),
            })

            # Solve Reduced and Hyper-reduced (with adaptive traveling bases!)
            solvers_r = BaseSolverMixin.setup_and_solve_reduced_system(kwds_r, solvers, client=client)
            solvers_hr = BaseSolverMixin.setup_and_solve_hyperreduced_system(kwds_hr, solvers, client=client)

            if not MechSystem.predict:
                for cls_idx, cls in enumerate(original_solver_classes):
                    errs_r_for_tol = []
                    errs_hr_for_tol = []
                    nosc_r_for_tol = np.nan
                    nosc_hr_for_tol = np.nan

                    for s_f, s_r, s_hr in zip(solvers, solvers_r, solvers_hr):
                        if s_f is not None and _matches_solver_family(s_f, cls):
                            if s_r is not None and s_hr is not None:
                                current_err_r = np.amax(abs(s_f.y - s_r.y)) / np.amax(abs(s_f.y))
                                current_err_hr = np.amax(abs(s_f.y - s_hr.y)) / np.amax(abs(s_f.y))
                                errs_r_for_tol.append(current_err_r)
                                errs_hr_for_tol.append(current_err_hr)
                                if np.isnan(nosc_r_for_tol):
                                    nosc_r_for_tol = s_r.nosc_r
                                if np.isnan(nosc_hr_for_tol):
                                    nosc_hr_for_tol = s_hr.deim_field.shape[1] // 2

                    if not np.isnan(nosc_r_for_tol):
                        study_data[cls.__name__]['sizes'].append(nosc_r_for_tol)
                        study_data[cls.__name__]['sizes_hr'].append(nosc_hr_for_tol)
                        study_data[cls.__name__]['err_r'].append(np.array(errs_r_for_tol).reshape(MechSystem.dt_space_dim, -1))
                        study_data[cls.__name__]['err_hr'].append(np.array(errs_hr_for_tol).reshape(MechSystem.dt_space_dim, -1))
            else:
                for cls in original_solver_classes:
                    study_data[cls.__name__]['times_f'].append(_collect_metric(solvers, cls, lambda s: s.time_lapsed[0]))
                    study_data[cls.__name__]['times_r'].append(_collect_metric(solvers_r, cls, lambda s: s.time_lapsed[0]))
                    study_data[cls.__name__]['times_hr'].append(_collect_metric(solvers_hr, cls, lambda s: s.time_lapsed[0]))

                    for prefix, extractor, guard in _PREDICTION_METRICS:
                        if guard(solvers):
                            study_data[cls.__name__][f'{prefix}_f'].append(_collect_metric(solvers, cls, extractor))
                        if guard(solvers_r):
                            study_data[cls.__name__][f'{prefix}_r'].append(_collect_metric(solvers_r, cls, extractor))
                        if guard(solvers_hr):
                            study_data[cls.__name__][f'{prefix}_hr'].append(_collect_metric(solvers_hr, cls, extractor))

        if not MechSystem.predict:
            f_times_raw = {cls.__name__: np.array([s.time_lapsed[0] for s in solvers if s and _matches_solver_family(s, cls)]).reshape(MechSystem.dt_space_dim, -1) for cls in original_solver_classes}
            f_errs_raw = {cls.__name__: np.array([s.en_error if hasattr(s, 'en_error') else np.nan for s in solvers if s and _matches_solver_family(s, cls)]).reshape(MechSystem.dt_space_dim, -1) for cls in original_solver_classes}
            plot_error_vs_basis_size(
                [study_data[n]['sizes'] for n in study_data],
                [study_data[n]['sizes_hr'] for n in study_data],
                [study_data[n]['err_r'] for n in study_data],
                [study_data[n]['err_hr'] for n in study_data],
                list(study_data.keys()), MechSystem.pod_tol_sweep, MechSystem.data_folder,
                dt_space=MechSystem.dt_space,
                full_times_raw=f_times_raw,
                full_errs_raw=f_errs_raw,
                times_r_raw=[study_data[n]['times_r'] for n in study_data],
                times_hr_raw=[study_data[n]['times_hr'] for n in study_data]
            )
        else:
            full_time_map = {}
            for cls in original_solver_classes:
                key = cls.__name__
                t_full = np.array([s.time_lapsed[0] if s is not None else np.nan for s in solvers if s is not None and _matches_solver_family(s, cls)])
                full_time_map[key] = t_full.reshape(MechSystem.dt_space_dim, -1)

            plot_prediction_results(
                list(study_data.keys()), MechSystem.pod_tol_sweep, MechSystem.dt_space, MechSystem.data_folder,
                full_times_raw=full_time_map,
                times_r_raw=[study_data[n]['times_r'] for n in study_data],
                times_hr_raw=[study_data[n]['times_hr'] for n in study_data],
                sym_error_f=[study_data[n]['sym_error_f'] for n in study_data],
                sym_error_r=[study_data[n]['sym_error_r'] for n in study_data],
                sym_error_hr=[study_data[n]['sym_error_hr'] for n in study_data],
                energy_error_f=[study_data[n]['energy_error_f'] for n in study_data],
                energy_error_r=[study_data[n]['energy_error_r'] for n in study_data],
                energy_error_hr=[study_data[n]['energy_error_hr'] for n in study_data],
                phase_error_f=[study_data[n]['phase_error_f'] for n in study_data],
                phase_error_r=[study_data[n]['phase_error_r'] for n in study_data],
                phase_error_hr=[study_data[n]['phase_error_hr'] for n in study_data],
                lin_mom_error_f=[study_data[n]['lin_mom_error_f'] for n in study_data],
                lin_mom_error_r=[study_data[n]['lin_mom_error_r'] for n in study_data],
                lin_mom_error_hr=[study_data[n]['lin_mom_error_hr'] for n in study_data],
                ang_mom_error_f=[study_data[n]['ang_mom_error_f'] for n in study_data],
                ang_mom_error_r=[study_data[n]['ang_mom_error_r'] for n in study_data],
                ang_mom_error_hr=[study_data[n]['ang_mom_error_hr'] for n in study_data]
            )

            # Sanity check: evaluate ground-truth test FOM vs ROM and HROM predictions
            if 'solvers_r' in locals() and 'solvers_hr' in locals():
                sanity_kwds = kwds_hr if 'kwds_hr' in locals() else base_kwds
                run_prediction_sanity_check(sanity_kwds, solvers_r, solvers_hr, original_solver_classes)

        # Dedicated 3D and 2D Elastica beam visualization
        print("\nGenerating dedicated 3D and 2D Elastica beam plots...")
        plot_elastica_beam(solvers, solvers_r, solvers_hr, data_folder=MechSystem.data_folder)

        # Combined error plot for FOM, ROM, and HROM
        print("\nGenerating combined error plot across model reduction orders...")
        plot_combined_errors(
            data_folder=MechSystem.data_folder,
            registered_solver_classes=original_solver_classes
        )

        print("\nElastica Model Order Reduction study completed successfully!")

    finally:
        client.close()
        cluster.close()
