import os
import gc
import hashlib
import warnings
import numpy as np
from numpy import linalg as LA
from pylab import log, r_, c_, sqrt, roll, figure, linspace
import cloudpickle as pickle
from functools import wraps
from tqdm.auto import tqdm
import concurrent.futures
import matplotlib.pyplot as plt

from System import (MechSystem, HamiltonianMechSystem, LagrangianMechSystem,
                   ReducedHamiltonianMechSystem, ReducedLagrangianMechSystem, HyperReducedHamiltonianMechSystem, HyperReducedLagrangianMechSystem,
                   fast_dump, fast_load, check_checkpoint_exists)
from ODESolver import (ConformalStormerVerlet, ConformalImplicitMidpoint, 
                      DiscreteGradient)
from ReduceMechSystem import ReduceMechSystem, HamiltonianReducer, DiscreteGradientReducer
from PlotScript import plot_data, save_figure, timing

def _dask_worker(idx, *arg):
    """
    Top-level helper function for Dask parallel execution.
    It calls the static method for solving the system. This function must be at
    the top level of a module so that cloudpickle can serialize it by reference.
    """
    print(f"Worker {idx} starting...", flush=True)
    
    try:
        with warnings.catch_warnings(record=True) as w_list:
            warnings.simplefilter("always")
            warnings.filterwarnings("error", category=RuntimeWarning)
            # BaseSolverMixin is in the module's scope, so this is fine.
            result = BaseSolverMixin.solve_mech_system(*arg)
            
            return result
    except Exception as e:
        print(f"Worker {idx} failed with error: {e}", flush=True)
        return e

def compose_solver_solves(func):
    """Decorator to compose solver steps based on w_values."""
    @wraps(func)
    def wrapper(self, y_, k):
        for w_val in self.w_values:
            y_, y_full_, Lambda = func(self, w_val, y_, k)
                
        return y_, y_full_, Lambda
    return wrapper

def _matches_solver_family(solver, base_cls):
    """Check if a solver (instance or class) belongs to the same family as base_cls
    (handles Reduced/HyperReduced naming variants)."""
    if solver is None or base_cls is None:
        return False

    if isinstance(solver, type):
        solver_class = solver
    else:
        solver_class = getattr(solver, 'solver_class', getattr(solver, '__class__', None))

    if solver_class is None:
        return False

    if not isinstance(base_cls, type):
        base_class = getattr(base_cls, 'solver_class', getattr(base_cls, '__class__', None))
    else:
        base_class = base_cls

    if base_class is None:
        return False

    if solver_class is base_class:
        return True

    try:
        if issubclass(solver_class, base_class):
            return True
    except TypeError:
        pass

    try:
        if issubclass(base_class, solver_class):
            return True
    except TypeError:
        pass

    solver_name = getattr(solver_class, '__name__', '')
    base_name = getattr(base_class, '__name__', '')

    stripped_solver = solver_name.replace('HyperReduced', '').replace('Reduced', '')
    stripped_base = base_name.replace('HyperReduced', '').replace('Reduced', '')

    return bool(stripped_solver and stripped_base and stripped_solver == stripped_base)


class BaseSolverMixin:
    """
    Base mixin class providing common solving capabilities for solvers.
    """
    matches_solver_family = staticmethod(_matches_solver_family)

    @compose_solver_solves
    def solve_for_w(self, w_val, y_, k):
        self.set_initial_condition(y_[1])
        
        # Temporarily update dt for the substep
        original_dt = self.dt
        self.dt = w_val * original_dt
        try:
            if isinstance(self, DiscreteGradient) and self.constraint_type:
                # Heun's method predictor
                f_val = self.f(c_[y_[0], y_[0]].T, None)
                y_euler = y_[0] + self.dt * f_val
                f_val_next = self.f(c_[y_euler, y_euler].T, None)
                y_[1] = y_[0] + 0.5 * self.dt * (f_val + f_val_next)
                info_ = []
            elif isinstance(self, DiscreteGradient) and not self.constraint_type:
                y_, _, info_ = super().solve(w_val*self.t_points[k:k+2])
            else:
                y_, _, info_ = super().solve(w_val*self.t_points[k:k+2])
        finally:
            self.dt = original_dt

        if self.store: self.info.append(np.array(info_[0::1]))
        
        if k > 0 and hasattr(self, 'Lambda'): # and isinstance(self, ConformalStormerVerlet):
            Lambda = self.Lambda[k].copy()
        else:
            Lambda = np.zeros_like(self.g_(np.zeros(2*self.nosc)))

        if self.constraint_type:
            self.fixed_point(y_, Lambda)

        return (y_, y_ @ self.RB.T, Lambda) if hasattr(self, 'RB') else (y_, y_, Lambda)

    @timing
    def solve_trajectory(self):
        """Solve the system trajectory over the specified time range."""

        if not hasattr(self, 'RB'):
            self.y = np.zeros((self.n+1, 2*self.nosc))
        else:
            self.y = np.zeros((self.n+1, 2*self.nosc_r))
            self.y_full = np.zeros((self.n+1, 2*self.nosc))
            self.y_full[0] = MechSystem.y_init

        self.y[0] = self.y_init
        if self.store: self.info = []
        self.Lambda = np.zeros((self.n+1, self.g_(np.zeros(2*self.nosc)).shape[0]))

        # Calculate update frequency (10% of iterations)
        update_freq = max(1, self.n // 10)  

        # Create progress bar that updates less frequently
        with tqdm(total=self.n, desc='Solving trajectory', 
                miniters=update_freq,  # Minimum iterations between updates
                maxinterval=5.0,  # Maximum seconds between updates
                position=0, 
                leave=True) as pbar:

            for k in range(self.n):

                if self.store: self.info.append(self.y[k])
                y_ = np.array([self.y[k], self.y[k]])

                y_, y_full_, self.Lambda[k+1] = self.solve_for_w(y_, k)

                self.y[k+1] = y_[-1]

                if hasattr(self, 'RB'):
                    self.y_full[k+1] = y_full_[-1]

                # Update progress bar every update_freq iterations
                if (k + 1) % update_freq == 0:
                    pbar.update(update_freq)
                    pbar.set_postfix({'step': k+1, 'total': self.n})

            # Update any remaining iterations
            remaining = self.n % update_freq
            if remaining:
                pbar.update(remaining)

        if self.store:
            self.info.append(self.y[k+1])
            self.info = np.vstack(self.info)

        if hasattr(self, 'RB'):
            self.y_red = self.y.copy()
            self.y = self.y_full.copy()
            del self.y_full
            gc.collect()

    def compute_energy_error(self):
        """Compute time-series energy error (eng_error) and scalar RMS energy error (en_error)."""
        if hasattr(self, 'ham') and hasattr(self, 'y') and self.y is not None and not getattr(self, 'beta', False):
            self.eng_error = np.array([self.ham(y) for y in self.y]) - self.ham(self.y[0])
            self.en_error = sqrt(self.dt) * LA.norm(self.eng_error)
        else:
            self.eng_error = None
            self.en_error = None

    def compute_sym_error(self):
        """Compute time-series symplectic error."""
        if getattr(self, 'var', False):
            y = self.y_red if hasattr(self, 'y_red') else self.y
            self.sym_error = self.var_solve(y)
        else:
            self.sym_error = None

    def compute_constraint_error(self):
        """Compute time-series constraint norm (g_norm) and max constraint error (phase_error)."""
        if hasattr(self, 'g__lambda'):
            self.g_norm = [LA.norm(self.g__lambda(y)) for y in self.y]
            self.phase_error = np.amax(self.g_norm)
        else:
            self.g_norm = None
            self.phase_error = np.nan

    def compute_linear_momentum_error(self):
        """Compute time-series linear momentum error array and max error."""
        if hasattr(self, 'y') and self.y is not None:
            lin_mom = np.sum(self.y[:, self.nosc:].reshape(-1, self.nosc // 3, 3), axis=1)
            self.lin_momentum_err = r_[0, LA.norm(lin_mom[1:] - lin_mom[0], axis=1)]
            self.lin_mom_error = np.amax(self.lin_momentum_err)
        else:
            self.lin_momentum_err = None
            self.lin_mom_error = np.nan

    def compute_angular_momentum_error(self):
        """Compute time-series angular momentum error array and max error."""
        if hasattr(self, 'y') and self.y is not None:
            q = self.y[:, :self.nosc].reshape(-1, self.nosc // 3, 3)
            p = self.y[:, self.nosc:].reshape(-1, self.nosc // 3, 3)
            ang_mom = np.sum(np.cross(q, p), axis=1)
            self.angular_momentum_err = r_[0, LA.norm(ang_mom[1:] - ang_mom[0], axis=1)]
            self.ang_mom_error = np.amax(self.angular_momentum_err)
        else:
            self.angular_momentum_err = None
            self.ang_mom_error = np.nan

    def compute_all_metrics(self):
        """Compute all physical diagnostic metrics after trajectory solution."""
        self.compute_energy_error()
        self.compute_sym_error()
        self.compute_constraint_error()
        self.compute_linear_momentum_error()
        self.compute_angular_momentum_error()

    @staticmethod
    def solve_mech_system(solver_class, dt, Omega2, kwds):
        """
        Helper function to solve mechanical system.

        Parameters:
            solver_class (class): The solver class to be used for solving the system.
            dt (float): The time step size.
            Omega2 (numpy.ndarray): The Omega2 parameter for the system.
            kwds (dict): A dictionary of keyword arguments for the solver.

        Returns:
            solver: An instance of the solver class with the solved trajectory.
        """
        kwds['pool'] = {
            'solver_class': solver_class, 
            'dt': dt, 
            'Omega2': Omega2,
            'n': int(round(MechSystem.T_final/dt))
        }
        kwds['pool'].update({
            't_points': linspace(0, MechSystem.T_final, kwds['pool']['n']+1)
        })

        try:
            # Create concrete solver instance instead of MechSystemSolver
            solver = solver_class(kwds)

            tl = solver.solve_trajectory()

            solver.time_lapsed.append(tl)
            solver.compute_all_metrics()

            return solver

        except Exception as e:
            print(f"Error in solve_mech_system: {str(e)}")
            print(f"Error type: {type(e)}")
            raise

    @staticmethod
    def parallel_solve_mech_system(kwds, MSsolvers, client=None):
        """Solve mechanical system in parallel using a Dask client, or sequentially."""
        # Prepare the arguments for solve_mech_system
        args = []

        # The kwds dict can be large, especially after reduction.
        # Scatter it to workers once to avoid sending it with every task.
        if client:
            # Ensure at least one worker is available before scattering large data
            # This prevents deadlock when using adaptive scaling with min_workers=0
            if not client.scheduler_info()['workers']:
                print("Cluster has 0 workers. Triggering scaling...", flush=True)
                dummy = client.submit(lambda x: x, 0)
                client.wait_for_workers(1)

            kwds_future = client.scatter(kwds, broadcast=True)

        for x in kwds['registered_solver_classes']:
            # Check if this is a reduced solver class that failed setup (RB is None)
            is_reduced_type = any(sub in x.__name__ for sub in ['Reduced', 'HyperReduced'])
            rb_is_missing = is_reduced_type and getattr(x, 'RB', None) is None
            
            for y in MechSystem.dt_space:
                for z in kwds["Omega2_space"]:
                    if rb_is_missing:
                        args.append(None)
                    else:
                        # If using dask, pass the future to the data. Otherwise, pass the data itself.
                        kwds_arg = kwds_future if client else kwds
                        args.append((x, y, z, kwds_arg))

        if client and len(args) > 0:
            print(f"\nSubmitting {len(args)} tasks to existing Dask cluster...", flush=True)
            
            # Identify which arguments are valid tasks vs skipped ones
            valid_indices = [i for i, a in enumerate(args) if a is not None]
            valid_args = [a for a in args if a is not None]
            
            # Submit only valid tasks
            futures = [client.submit(_dask_worker, i, *a) for i, a in zip(valid_indices, valid_args)]
            
            # Gather results (blocks until all are done)
            # client.gather returns results in the same order as futures list
            gathered_results = client.gather(futures, errors='raise')
            
            # Align results back into the full MSsolvers list, preserving None for skips
            results = [None] * len(args)
            for idx, res in zip(valid_indices, gathered_results):
                 if not isinstance(res, Exception):
                     results[idx] = res

            MSsolvers.extend(results)
            
            print("Parallel processing complete.", flush=True)
        else:
            # Sequentially solve the mechanical system
            print('Solving mechanical system sequentially...')
            for arg in args:
                MSsolvers.append(BaseSolverMixin.solve_mech_system(*arg))

    @staticmethod
    def _transfer_reduction_attrs(kwds):
        """
        Transfers all dynamically set reduction/hyper-reduction attributes
        from the MechSystem class to the kwds dictionary to be passed to workers.
        This ensures that the state computed on the driver is available on the workers.
        """
        kwds['solver_data'] = {}
        
        # Check registered solver classes for attributes
        for cls in kwds.get('registered_solver_classes', []):
            data = {}
            for attr in ReduceMechSystem._REDUCTION_ATTRS:
                if hasattr(cls, attr):
                    data[attr] = getattr(cls, attr)
            kwds['solver_data'][cls.__name__] = data

        return kwds

    @staticmethod
    def _restore_reduction_attrs(kwds):
        """
        Restores reduction attributes from the loaded kwds dictionary back to the
        appropriate Reducer classes. This is crucial when loading from a checkpoint
        to ensure subsequent steps like hyper-reduction have access to the bases.
        """
        solver_data = kwds.get('solver_data', {})
        if not solver_data:
            return

        # Find the RB and nosc_r from the saved data. It should be consistent
        # across all solvers of a given type (Hamiltonian/DG).
        rb_ham, nosc_r_ham = None, None
        rb_dg, nosc_r_dg = None, None

        # The registered_solver_classes in the loaded kwds are the reduced ones.
        for cls in kwds.get('registered_solver_classes', []):
            cls_name = cls.__name__
            if cls_name in solver_data:
                data = solver_data[cls_name]
                if 'RB' in data and 'nosc_r' in data:
                    if issubclass(cls, ReducedHamiltonianMechSystem):
                        if rb_ham is None: # Store first one found
                            rb_ham, nosc_r_ham = data['RB'], data['nosc_r']
                    elif issubclass(cls, ReducedLagrangianMechSystem):
                        if rb_dg is None: # Store first one found
                            rb_dg, nosc_r_dg = data['RB'], data['nosc_r']

        # Now set the attributes on the Reducer classes
        if rb_ham is not None:
            setattr(HamiltonianReducer, 'RB', rb_ham)
            setattr(HamiltonianReducer, 'nosc_r', nosc_r_ham)
        if rb_dg is not None:
            setattr(DiscreteGradientReducer, 'RB', rb_dg)
            setattr(DiscreteGradientReducer, 'nosc_r', nosc_r_dg)

    @staticmethod
    def setup_and_solve_reduced_system(kwds, solvers, client=None):
        """Setup and solve the reduced-order system."""
        print(f'\nSetting up reduced system...')

        pod_tol = kwds.get('pod_tol', MechSystem.pod_tol_sweep[-1])

        # Solve reduced system
        solvers_r = []

        # Try to load from checkpoint
        checkpoint_path = MechSystem.data_folder
        tol_suffix = f"tol_{pod_tol:.1e}"
        kwds_file = os.path.join(checkpoint_path, f'kwds_r_{tol_suffix}.joblib')
        solvers_file = os.path.join(checkpoint_path, f'solvers_r_{tol_suffix}.joblib')

        if check_checkpoint_exists(checkpoint_path, kwds_file, solvers_file):
            try:
                print("Loading from checkpoint...")
                kwds = fast_load(kwds_file)
                solvers_r = fast_load(solvers_file)
                print("Checkpoint loaded successfully")

                # Restore class attributes needed for subsequent steps
                BaseSolverMixin._restore_reduction_attrs(kwds)
            except Exception as e:
                raise Exception(f"Error loading checkpoint: {str(e)}")
        else:
            # Setup reduced model and update methods
            ham_solvers = [s for s in solvers if not isinstance(s, DiscreteGradient)]
            dg_solvers = [s for s in solvers if isinstance(s, DiscreteGradient)]
            
            ham_classes = [REDUCED_SOLVER_MAPPING.get(c, c) for c in kwds['registered_solver_classes'] if issubclass(c, (ConformalStormerVerlet, ConformalImplicitMidpoint))]
            dg_classes = [REDUCED_SOLVER_MAPPING.get(c, c) for c in kwds['registered_solver_classes'] if issubclass(c, DiscreteGradient)]
            
            if ham_solvers:
                HamiltonianReducer.setup_reduced_model(ham_solvers, ham_classes)
            if dg_solvers:
                DiscreteGradientReducer.setup_reduced_model(dg_solvers, dg_classes)

            # Update registered classes to reduced versions for next steps
            kwds['registered_solver_classes'] = [REDUCED_SOLVER_MAPPING.get(cls, cls) for cls in kwds['registered_solver_classes']]

            # Pass the computed bases to the workers via the kwds dictionary.
            kwds = BaseSolverMixin._transfer_reduction_attrs(kwds)

            # Execute parallel solve
            BaseSolverMixin.parallel_solve_mech_system(kwds, solvers_r, client=client)

            # Create checkpoint directory and save initial state
            os.makedirs(checkpoint_path, exist_ok=True)
            print("Creating new checkpoint...")
            fast_dump(kwds, kwds_file)
            fast_dump(solvers_r, solvers_file)

        BaseSolverMixin.compute_convergence_rates(kwds, solvers_r)
        BaseSolverMixin.plot_solvers(kwds, solvers_r)

        return solvers_r

    @staticmethod
    def setup_and_solve_hyperreduced_system(kwds, solvers, client=None):
        """Setup and solve the hyper-reduced system."""
        print(f'\nSetting up hyper-reduced system...')

        pod_tol = kwds.get('pod_tol', MechSystem.pod_tol_sweep[-1])

        # Solve hyper-reduced system
        solvers_dr = []

        # Try to load from checkpoint
        checkpoint_path = MechSystem.data_folder
        tol_suffix = f"tol_{pod_tol:.1e}"
        kwds_file = os.path.join(checkpoint_path, f'kwds_dr_{tol_suffix}.joblib')
        solvers_file = os.path.join(checkpoint_path, f'solvers_dr_{tol_suffix}.joblib')

        if check_checkpoint_exists(checkpoint_path, kwds_file, solvers_file):
            try:
                print("Loading from checkpoint...")
                kwds = fast_load(kwds_file)
                solvers_dr = fast_load(solvers_file)
                print("Checkpoint loaded successfully")
            except Exception as e:
                raise Exception(f"Error loading checkpoint: {str(e)}")
        else:
            # Setup hyperreduction using classmethod
            ham_solvers = [s for s in solvers if not isinstance(s, DiscreteGradient)]
            dg_solvers = [s for s in solvers if isinstance(s, DiscreteGradient)]
            
            ham_classes = [HYPERREDUCED_SOLVER_MAPPING.get(c, c) for c in kwds['registered_solver_classes'] if issubclass(c, (ConformalStormerVerlet, ConformalImplicitMidpoint))]
            dg_classes = [HYPERREDUCED_SOLVER_MAPPING.get(c, c) for c in kwds['registered_solver_classes'] if issubclass(c, DiscreteGradient)]

            if ham_classes:
                HamiltonianReducer.setup_hyperreduction(ham_classes)
            if dg_classes:
                DiscreteGradientReducer.setup_hyperreduction(dg_solvers, dg_classes)

            if ReduceMechSystem.hyperreducer == 'MDEIM':
                if ham_solvers:
                    HamiltonianReducer.update_mdeim_hyperreduction(ham_solvers, ham_classes)
                if dg_solvers:
                    DiscreteGradientReducer.update_mdeim_hyperreduction(dg_solvers, dg_classes)

            # hyperreduce_constraints use parallel processing internally
            if ReduceMechSystem.constraints_reduce:
                if ham_solvers:
                    HamiltonianReducer.hyperreduce_constraints(ham_solvers, ham_classes)
                if dg_solvers:
                    DiscreteGradientReducer.hyperreduce_constraints(dg_solvers, dg_classes)

            # Update registered classes to hyper-reduced versions
            kwds['registered_solver_classes'] = [HYPERREDUCED_SOLVER_MAPPING.get(cls, cls) for cls in kwds['registered_solver_classes']]

            # Pass all computed bases and functions to workers
            kwds = BaseSolverMixin._transfer_reduction_attrs(kwds)

            # Execute parallel solve
            BaseSolverMixin.parallel_solve_mech_system(kwds, solvers_dr, client=client)

            # Create checkpoint directory and save initial state
            os.makedirs(checkpoint_path, exist_ok=True)
            print("Creating new checkpoint...")
            fast_dump(kwds, kwds_file)
            fast_dump(solvers_dr, solvers_file)

        BaseSolverMixin.compute_convergence_rates(kwds, solvers_dr)
        BaseSolverMixin.plot_solvers(kwds, solvers_dr)

        return solvers_dr

    @staticmethod
    def compute_convergence_rates(kwds, solvers):
        """
        Compute empirical convergence rates across step sizes for solved system instances.

        Parameters:
            kwds (dict): Dictionary of keyword arguments for the solver.
            solvers (list): List of solver instances used to solve the system. 
        """
        def empirical_orders(errors, dt):
            log_err = np.log(errors)
            log_dt  = np.log(dt)

            num = log_err[1:] - log_err[:-1]          # shape (n_dt-1, n_methods)
            den = log_dt[1:]  - log_dt[:-1]           # shape (n_dt-1,)

            s = num / den[:, None]                    # broadcast correctly

            return np.vstack([np.nan*np.ones((1, s.shape[1])), s])

        # Compute en_error using a list comprehension and reshape it into a 2D array: rows = dt values, columns = Omega2 values for each solver class in kwds['registered_solver_classes']
        # For each solver_class, filter solvers and compute en_error separately
        for solver_class in kwds['registered_solver_classes']:
            filtered_solvers = [s for s in solvers if s and _matches_solver_family(s, solver_class)]
            
            en_error_flat = []
            for dt in MechSystem.dt_space:
                for omega2 in kwds['Omega2_space']:
                    found_solver = next((s for s in filtered_solvers if np.isclose(s.dt, dt) and np.allclose(s.Omega2, omega2)), None)
                    
                    if found_solver is not None and np.any(np.isnan(found_solver.y)):
                        print(f"Warning: Solver {solver_class.__name__} (dt={dt:.4e}) contains NaNs.", flush=True)

                    if found_solver and hasattr(found_solver, 'en_error') and found_solver.en_error is not None:
                        en_error_flat.append(found_solver.en_error)
                    else:
                        en_error_flat.append(np.nan)
            
            en_error = np.array(en_error_flat).reshape(MechSystem.dt_space_dim, kwds['Omega2_space_dim'])

            if en_error.shape[0] > 1 and not np.all(np.isnan(en_error)) and MechSystem.dt_space_dim > 1 and not MechSystem.predict:
                print(f"\nConvergence rates for {solver_class.__name__}:")
                header = f"{'dt':>12}"
                for col in range(en_error.shape[1]):
                    header += f" | {f'Omega2_{col}':>12}"
                print(header)
                print("-" * len(header))
                all_r_values = empirical_orders(en_error, MechSystem.dt_space)
                for i, dt_val in enumerate(MechSystem.dt_space):
                    print(f"{dt_val:12.6f}" + "".join([f" | {val:12.6f}" for val in all_r_values[i]]))

    @staticmethod
    def plot_solvers(kwds, solvers):
        """
        Generate and display plots for representative solver instances.

        Parameters:
            kwds (dict): Dictionary of keyword arguments containing registered_solver_classes.
            solvers (list): List of solver instances used to solve the system.
        """
        for solver_class in kwds['registered_solver_classes']:
            filtered_solvers = [s for s in solvers if s and _matches_solver_family(s, solver_class)]
            if filtered_solvers:
                filtered_solvers[-1].plot()

    def plot(self):
        """
        Generate and display plots for the simulation results.
        """
        # ... (implementation same as original, omitted for brevity but assumed present in full file)
        # Since I cannot copy the full implementation here without making the response huge, 
        # I will assume the user copies the plot method from app8_lattice.py to here.
        # For the purpose of this diff, I will include the full method in the actual file creation if needed,
        # but here I will just copy it from the context provided.
        
        fig = figure(figsize=(12, 12), constrained_layout=True)  # Make figure taller
        fig.set_constrained_layout_pads(w_pad=0.1, h_pad=0.1, hspace=0.1, wspace=0.1)
        # fig.tight_layout(pad=0)
        omega2_hash = hashlib.md5(self.Omega2.tobytes()).hexdigest()[:8]
        fig.suptitle(rf'integrator = {self.solver_class.__name__}, $\Delta t = {self.dt}$, $\Omega_2$ hash = {omega2_hash}')

        gs = fig.add_gridspec(5, 2)
        ax0, ax1, ax2, ax3, ax4 = [fig.add_subplot(gs[i, 0]) for i in [0, 1, 2, 3, 4]]
        ax_pp = fig.add_subplot(gs[:5, -1], projection='3d')
        ax_pp.set_box_aspect([1, 1, 1])  # Set aspect ratio to be equal for all axes

        if hasattr(self, 'sym_error'):
            plot_data(ax0, self.t_points, self.sym_error, xlims=(0, self.T_final), \
                    ylabel=r'$\Delta Sp$', margins=10)

            if max(abs(self.sym_error)) < 1e-15:
                ax0.set_ylim([-1e-15, 1e-15])

        if getattr(self, 'g_norm', None) is not None:
            plot_data(
                ax1,
                self.t_points,
                self.g_norm,
                xlims=(0, self.T_final),
                ylabel=r"$\Delta \mathcal{S}$",
                margins=10,
            )

        if getattr(self, 'eng_error', None) is not None:
            plot_data(ax2, self.t_points, self.eng_error, xlims=(0, self.T_final), \
                    ylabel=r'$\Delta H$', margins=10)

        if getattr(self, 'lin_momentum_err', None) is not None:
            plot_data(ax3, self.t_points, self.lin_momentum_err, xlims=(0, self.T_final), \
                    ylabel=r'$\Delta L$', margins=10)

        if getattr(self, 'angular_momentum_err', None) is not None:
            plot_data(ax4, self.t_points, self.angular_momentum_err, xlims=(0, self.T_final), \
                    ylabel=r'$\Delta J$', margins=10)

        ax4.set_xlabel('time')
        # Plot the particle positions over time
        coords = self.y[:, :self.nosc].reshape(-1, self.nosc//3, 3)
        sc = None
        for i in range(coords.shape[1]):
            ax_pp.scatter(coords[0, i, 0], coords[0, i, 1], coords[0, i, 2], s=20, c='teal')  # plot initial configuration
            # ax_pp.plot(coords[:, i, 0], coords[:, i, 1], coords[:, i, 2], 'k-')  # plot system evolution
            sc = ax_pp.scatter(coords[:, i, 0], coords[:, i, 1], coords[:, i, 2], c=self.t_points, cmap='viridis', s=1, alpha=0.1)

            # Plot projection onto x-y plane (z=0)
            ax_pp.plot(coords[:, i, 0], coords[:, i, 1], np.zeros_like(coords[:, i, 2]), 'r-', alpha=0.7)

        # Set the axes' labels and title
        ax_pp.set_xlabel('x', labelpad=10)
        ax_pp.set_ylabel('y', labelpad=10)
        ax_pp.set_zlabel('z', labelpad=10)
        ax_pp.set_title('Phase portrait')
        ax_pp.set_box_aspect([1, 1, 1])  # Set aspect ratio to be equal for all axes
        ax_pp.view_init(elev=15, azim=45)

        if sc:
            cbar = fig.colorbar(sc, ax=ax_pp, shrink=0.5, pad=0.1)
            cbar.set_label('Time')

        # Hide inner labels
        for ax in [ax0, ax1, ax2, ax3, ax4]:
            ax.label_outer()

        fig.show()            

        fig_data = {
            't_points': self.t_points,
            'lin_momentum_err': getattr(self, 'lin_momentum_err', None),
            'angular_momentum_err': getattr(self, 'angular_momentum_err', None),
            'coords': coords
        }
        if getattr(self, 'sym_error', None) is not None:
            fig_data['sym_error'] = self.sym_error
        if getattr(self, 'g_norm', None) is not None:
            fig_data['g_norm'] = self.g_norm
        if getattr(self, 'eng_error', None) is not None:
            fig_data['eng_error'] = self.eng_error

        solver_name_short = self.solver_class.__name__.replace('Conformal', 'C').replace('StormerVerlet', 'SV').replace('ImplicitMidpoint', 'IM').replace('DiscreteGradient', 'DG').replace('HyperReduced', 'HR').replace('Reduced', 'R')
        # suffix = '_full' if not hasattr(self, 'RB') else '_r' if not hasattr(MechSystem, 'RBxUx_inv_PxU') else '_dr'
        suffix = '_predict' if self.predict else '' if not hasattr(self, 'RB') else '_repro'
        filename = os.path.join(MechSystem.data_folder, f"{solver_name_short}{suffix}.pdf")
        save_figure(fig, filename, fig_data=fig_data)

class DiscreteGradientFixedPointMixin:
    def __init__(self, kwds):
        super().__init__(kwds)
        self.g = self.g__
        self.g_prime = (self.g_prime__, self._g_prime_with_JJ)

    def residual(self, x, x1, Lambda):

        g_x1 = self.g(x1)
        resi = r_[x1 - x - self.dt*self.f(c_[x, x1].T, None) - self.dt*self._g_prime_with_JJ(0.5 *(x + x1)).T @ Lambda,\
                g_x1]
        tang = r_[c_[np.eye(x1.shape[0]) - self.dt * self.dfdu(c_[x, x1].T, None) - 0.5 * self.dt * self.JJ @ self.g_prime_x_lambda_y(0.5 *(x + x1), Lambda), -self.dt * self.JJ @ self.g_prime__(0.5 *(x + x1)).T],\
                  c_[self.g_prime__(x1), np.zeros((g_x1.shape[0],)*2)]]

        return resi, tang

    def fixed_point(self, x, Lambda):
        """
        Fixed point iteration method.

        Parameters:
        x (array): The initial guess.
        Lambda (array): The initial value for Lambda.
        """

        m = 0
        
        while m < self.M:

            # Update residual and tangent
            resi, tang = self.residual(x[0], x[1], Lambda) 
            assert resi.ndim == 1, f"DiscreteGradient FixedPoint: resi must be 1D, got {resi.shape}"

            Delta_z = -LA.lstsq(tang, resi, rcond=None)[0]
            x[1] = x[1] + Delta_z[:x[1].shape[0]]
            Lambda += Delta_z[x[1].shape[0]:]

            # Update iteration counter and residual
            m += 1
            residual = LA.norm(r_[resi, Delta_z], np.inf)
            
            if np.isnan(residual):
                raise RuntimeError("Nonlinear solver diverged: Residual is NaN")

            if residual < self.tol:
                return

        raise RuntimeError("Nonlinear solver did not converge")

    def _g_prime_with_JJ(self, y):
        return self.g_prime__(y) @ self.JJ.T

class ConformalStormerVerletFixedPointMixin:
    def __init__(self, kwds):
        super().__init__(kwds)
        self.g = self.g__
        self.g_prime = (self.g_prime_diag, self.block_diag_matrix)

    def fixed_point(self, x, Lambda):
        """
        Fixed point iteration method.

        Parameters:
        x (array): The initial guess.
        Lambda (array): The initial value for Lambda.
        """

        # Setup constants
        f, neq, dt = self.f, self.neq//2, self.dt
        q0, p0 = np.split(x[0], 2)
        
        # G(q0)
        g_prime_x0 = self.g_prime[0](x[0])
        i0, i1 = g_prime_x0.shape
        i0, i1 = i0//2, i1//2
        G_q0 = g_prime_x0[:i0, :i1] # G(q0)
        
        # Split Lambda
        Lambda_1, Lambda_2 = np.split(Lambda, 2)
        
        # p_half (unconstrained)
        p_half_unconstrained = p0 + 0.5 * dt * f(x[0], None)[neq:]
        
        # 1. Position Constraints Loop (Solve for Lambda_1)
        m = 0
        while m < self.M:
            # p_{n+1/2}
            p_half1 = p_half_unconstrained - 0.5 * dt * G_q0.T @ Lambda_1 # Lambda_1 is 1D
            
            # q_{n+1}
            q_next = q0 + dt * f(np.concatenate([q0, p_half1]), None)[:neq]
            x[1, :neq] = q_next
            
            # Check position constraints
            g_pos = self.g(x[1])[:i0]
            assert g_pos.ndim == 1, f"StormerVerlet FixedPoint: g_pos must be 1D, got {g_pos.shape}"
            norm_g_pos = LA.norm(g_pos)
            
            if np.isnan(norm_g_pos):
                raise RuntimeError("Nonlinear solver (position) diverged: Residual is NaN")
            if norm_g_pos < self.tol:
                break
                
            # Jacobian R11 = G(q_{n+1}) @ (-0.5 * dt^2 * G(q0)^T)
            G_qnext = self.g_prime[0](x[1])[:i0, :i1]
            R11 = -0.5 * dt**2 * G_qnext @ G_q0.T
            
            Delta_Lambda_1 = LA.lstsq(R11, g_pos, rcond=None)[0]
            Lambda_1 -= Delta_Lambda_1
            m += 1
            
        if m >= self.M:
             if getattr(self, 'RB', None) is not None and LA.norm(g_pos) < 1e-5 and not m%100:
                 print(f"Warning: Position constraints not fully satisfied (error={LA.norm(g_pos):.2e}) in reduced model. Continuing.", flush=True)
             else:
                 raise RuntimeError(f"Nonlinear solver (position) did not converge. m: {m} -- Error: {LA.norm(g_pos)}")

        # 2. Velocity Constraints Loop (Solve for Lambda_2)
        m = 0
        while m < self.M:
            # p_{n+1}
            G_qnext = self.g_prime[0](x[1])[:i0, :i1]
            force_next = f(np.concatenate([x[1, :neq], p_half1]), None)[neq:]
            p_next = p_half1 + 0.5 * dt * force_next - 0.5 * dt * G_qnext.T @ Lambda_2
            x[1, neq:] = p_next
            
            # Check velocity constraints
            g_vel = self.g(x[1])[i0:] # g_vel is (N,1)
            assert g_vel.ndim == 1, f"StormerVerlet FixedPoint: g_vel must be 1D, got {g_vel.shape}"
            norm_g_vel = LA.norm(g_vel)
            
            if np.isnan(norm_g_vel):
                raise RuntimeError("Nonlinear solver (velocity) diverged: Residual is NaN")
            if norm_g_vel < self.tol:
                break
                
            # Jacobian R22 = -0.5 * dt * G(p_{n+1}) @ G(q_{n+1})^T
            G_pnext = self.g_prime__(x[1])[i0:, i1:]
            R22 = -0.5 * dt * G_pnext @ G_qnext.T
            
            Delta_Lambda_2 = LA.lstsq(R22, g_vel, rcond=None)[0]
            Lambda_2 -= Delta_Lambda_2
            m += 1
            
        if m >= self.M:
             if getattr(self, 'RB', None) is not None and LA.norm(g_vel) < 1e-5 and not m%100:
                 print(f"Warning: Velocity constraints not fully satisfied (error={LA.norm(g_vel):.2e}) in reduced model. Continuing.", flush=True)
             else:
                 raise RuntimeError(f"Nonlinear solver (velocity) did not converge. m: {m} -- Error: {LA.norm(g_vel)}")
             
        # Update Lambda in place
        Lambda[:i0] = Lambda_1
        Lambda[i0:] = Lambda_2

    def block_diag_matrix(self, y):
        """Compute g_prime using the original method"""

        ham_zz_y3 = self.ham_zz(y[3])
        i0, i1 = ham_zz_y3.shape
        i0, i1 = i0//2, i1//2
        ham_zz_y3_22 = ham_zz_y3[i0:, i1:]
        
        i0, i1 = self.g_prime__(y[0]).shape
        i0, i1 = i0//2, i1//2

        result = np.block([
            [-0.25 * self.dt**2 * (self.g_prime__(y[2])[i0:, i1:] + self.g_prime__(y[0])[:i0, :i1] @ ham_zz_y3_22), np.zeros((i0, i1))],
            [np.zeros((i0, i1)), -0.5 * self.dt * self.g_prime__(y[1])[i0:, i1:]]
        ])
        return result
    
    def g_prime_diag(self, y):
        """Compute g_prime using the original method"""
        _g_prime = self.g_prime__(y)
        i0, i1 = _g_prime.shape
        _g_prime_11 = _g_prime[:i0//2, :i1//2]
        result = np.block([
            [_g_prime_11, np.zeros(_g_prime_11.shape)],
            [np.zeros(_g_prime_11.shape), _g_prime_11]
        ])
        return result

class DiscreteGradientSolver(BaseSolverMixin, DiscreteGradientFixedPointMixin, DiscreteGradient, LagrangianMechSystem):
    """DiscreteGradient solver with DG-specific capabilities."""
    pass

class ReducedDiscreteGradientSolver(BaseSolverMixin, DiscreteGradientFixedPointMixin, DiscreteGradient, ReducedLagrangianMechSystem):
    """Reduced order Discrete Gradient Solver."""
    pass

class HyperReducedDiscreteGradientSolver(BaseSolverMixin, DiscreteGradientFixedPointMixin, DiscreteGradient, HyperReducedLagrangianMechSystem):
    """Hyper-reduced order Discrete Gradient Solver."""
    pass


class ConformalStormerVerletSolver(BaseSolverMixin, ConformalStormerVerletFixedPointMixin, ConformalStormerVerlet, HamiltonianMechSystem):
    """Conformal Stormer-Verlet Solver with Hamiltonian capabilities."""
    pass

class ReducedConformalStormerVerletSolver(BaseSolverMixin, ConformalStormerVerletFixedPointMixin, ConformalStormerVerlet, ReducedHamiltonianMechSystem):
    """Reduced order Conformal Stormer-Verlet Solver."""
    pass

class HyperReducedConformalStormerVerletSolver(BaseSolverMixin, ConformalStormerVerletFixedPointMixin, ConformalStormerVerlet, HyperReducedHamiltonianMechSystem):
    """Hyper-reduced order Conformal Stormer-Verlet Solver."""
    pass

class ConformalImplicitMidpointSolver(BaseSolverMixin, ConformalImplicitMidpoint, HamiltonianMechSystem):
    """Conformal Implicit Midpoint Solver with Hamiltonian capabilities."""
    def __init__(self, kwds):
        super().__init__(kwds)
        self.g = self.g__
        self.g_prime = [self.g_prime__, self.block_diag_matrix]

    def block_diag_matrix(self, y):
        """Compute g_prime using the original method"""
        _g_prime_shape = self.g_prime__(y[0]).shape
        result = np.block([
            [self.g_prime__(y[0])[:_g_prime_shape[0]//2, :_g_prime_shape[1]//2], np.zeros((_g_prime_shape[0]//2, _g_prime_shape[1]//2))],
            [np.zeros((_g_prime_shape[0]//2, _g_prime_shape[1]//2)), self.g_prime__(y[1])[:_g_prime_shape[0]//2, :_g_prime_shape[1]//2]]
        ])
        return result

class ReducedConformalImplicitMidpointSolver(BaseSolverMixin, ConformalImplicitMidpoint, ReducedHamiltonianMechSystem):
    """Reduced order Conformal Implicit Midpoint Solver."""
    def __init__(self, kwds):
        super().__init__(kwds)
        self.g_prime = [self.g_prime__, self.block_diag_matrix]

    def block_diag_matrix(self, y):
        return ConformalImplicitMidpointSolver.block_diag_matrix(self, y)

class HyperReducedConformalImplicitMidpointSolver(BaseSolverMixin, ConformalImplicitMidpoint, HyperReducedHamiltonianMechSystem):
    """Hyper-reduced order Conformal Implicit Midpoint Solver."""
    def __init__(self, kwds):
        super().__init__(kwds)
        self.g_prime = [self.g_prime__, self.block_diag_matrix]

    def block_diag_matrix(self, y):
        return ConformalImplicitMidpointSolver.block_diag_matrix(self, y)

REDUCED_SOLVER_MAPPING = {
    DiscreteGradientSolver: ReducedDiscreteGradientSolver,
    ConformalStormerVerletSolver: ReducedConformalStormerVerletSolver,
    ConformalImplicitMidpointSolver: ReducedConformalImplicitMidpointSolver,
    # Map reduced to reduced (idempotent)
    ReducedDiscreteGradientSolver: ReducedDiscreteGradientSolver,
    ReducedConformalStormerVerletSolver: ReducedConformalStormerVerletSolver,
    ReducedConformalImplicitMidpointSolver: ReducedConformalImplicitMidpointSolver,
    # Map hyperreduced back to reduced (idempotent for tier lookup)
    HyperReducedDiscreteGradientSolver: ReducedDiscreteGradientSolver,
    HyperReducedConformalStormerVerletSolver: ReducedConformalStormerVerletSolver,
    HyperReducedConformalImplicitMidpointSolver: ReducedConformalImplicitMidpointSolver
}

HYPERREDUCED_SOLVER_MAPPING = {
    DiscreteGradientSolver: HyperReducedDiscreteGradientSolver,
    ConformalStormerVerletSolver: HyperReducedConformalStormerVerletSolver,
    ConformalImplicitMidpointSolver: HyperReducedConformalImplicitMidpointSolver,
    ReducedDiscreteGradientSolver: HyperReducedDiscreteGradientSolver,
    ReducedConformalStormerVerletSolver: HyperReducedConformalStormerVerletSolver,
    ReducedConformalImplicitMidpointSolver: HyperReducedConformalImplicitMidpointSolver,
    # Map hyperreduced to hyperreduced (idempotent)
    HyperReducedDiscreteGradientSolver: HyperReducedDiscreteGradientSolver,
    HyperReducedConformalStormerVerletSolver: HyperReducedConformalStormerVerletSolver,
    HyperReducedConformalImplicitMidpointSolver: HyperReducedConformalImplicitMidpointSolver
}
