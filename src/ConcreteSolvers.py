import os
import gc
import numpy as np
from numpy import linalg as LA
from pylab import log, r_, c_, sqrt, roll, figure, linspace
import cloudpickle as pickle
from functools import wraps
from tqdm.auto import tqdm

from System import MechSystem
from ODESolver import (ConformalStormerVerlet, ConformalImplicitMidpoint, 
                      DiscreteGradient)
from ReduceMechSystem import ReduceMechSystem
from PlotScript import plot_data, save_figure, timing

def _dask_worker(idx, *arg):
    """
    Top-level helper function for Dask parallel execution.
    It calls the static method for solving the system. This function must be at
    the top level of a module so that cloudpickle can serialize it by reference.
    """
    print(f"Worker {idx} starting...", flush=True)
    try:
        # BaseSolverMixin is in the module's scope, so this is fine.
        return BaseSolverMixin.solve_mech_system(*arg)
    except Exception as e:
        print(f"Worker {idx} failed with error: {e}", flush=True)
        raise

def compose_solver_solves(func):
    """Decorator to compose solver steps based on w_values."""
    @wraps(func)
    def wrapper(self, y_, k):
        for w_val in self.w_values:
            y_, y_full_, Lambda = func(self, w_val, y_, k)
                
        return y_, y_full_, Lambda
    return wrapper

class BaseSolverMixin:
    """
    Base mixin class providing common solving capabilities for solvers.
    """
    @compose_solver_solves
    def solve_for_w(self, w_val, y_, k):
        self.set_initial_condition(y_[1])
        
        # Temporarily update dt for the substep
        original_dt = self.dt
        self.dt = w_val * original_dt
        try:
            y_, _, info_ = super().solve(w_val*self.t_points[k:k+2])
        finally:
            self.dt = original_dt

        if self.store: self.info.append(np.array(info_[0::1]))

        if k > 0 and hasattr(self, 'Lambda') and self.solver_class == ConformalStormerVerletSolver:
            Lambda = self.Lambda[k].copy()
            if k == 1:
                print(f'Setting initial guess for Lagrange multipliers')
        else:
            Lambda = np.zeros_like(self.g_(np.zeros(2*self.nosc))).squeeze()

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

            if not solver.beta:
                solver.eng_error = solver.get_en_err()
                solver.en_error = sqrt(dt) * LA.norm(solver.eng_error)
            else:
                solver.en_error = None

            if solver.var:
                if hasattr(solver, 'y_red'):
                    y = solver.y_red
                else:
                    y = solver.y

                solver.sym_error = solver.var_solve(y)
                # solver.sym_error = solver.symplectic_error(dpsi)

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
            kwds_future = client.scatter(kwds, broadcast=True)

        for x in kwds['registered_solver_classes']:
            for y in MechSystem.dt_space:
                for z in kwds["Omega2_space"]:
                    # If using dask, pass the future to the data. Otherwise, pass the data itself.
                    kwds_arg = kwds_future if client else kwds
                    args.append((x, y, z, kwds_arg))

        if client and len(args) > 0:
            print(f"Submitting {len(args)} tasks to existing Dask cluster...", flush=True)
            # Pre-allocate MSsolvers with None values
            MSsolvers.extend([None] * len(args))

            futures = [client.submit(_dask_worker, i, *arg) for i, arg in enumerate(args)]
            
            # Gather results (blocks until all are done)
            # client.gather returns results in the same order as futures list
            results = client.gather(futures, errors='raise')
            
            # Store results in correct order
            for i, res in enumerate(results):
                MSsolvers[i] = res
                if (i + 1) % 10 == 0:
                    print(f"Retrieved {i+1}/{len(args)} results", flush=True)
            
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
        attrs_to_pass = [
            # Reduced Basis
            'RB', 'nosc_r', 'RB_dg', 'nosc_r_dg',
            # DEIM attributes
            'RBxUx_inv_PxU', '_RBxUx_inv_PxU_',
            # MDEIM attributes for system dynamics
            'IP_Ux_inv_PxU', '_IP_Ux_inv_PxU_',
            # MDEIM attributes for constraints
            '_IP_Ux_inv_PxU', '_IP_Ux_inv_PxU_dg',
            'IP_g_prime_x_lambda_y_dg', 'IP_g_prime_x_lambda_lambda_dg',
            # Lambdified functions (DEIM)
            'ham_z_deim', 'ham_zz_deim', 'lag_dg_deim', 'lag_dg_z_deim',
            # Lambdified functions (MDEIM)
            'ham_zz_mdeim', 'lag_dg_z_mdeim', 'g_prime_mdeim', 'g_prime_mdeim_dg',
            'g_prime_x_lambda_y_mdeim_dg', 'g_prime_x_lambda_lambda_mdeim_dg',
            # Shape attributes for reshaping
            'g_prime_shape', 'g_prime_shape_dg',
            'g_prime_x_lambda_y_shape_dg', 'g_prime_x_lambda_lambda_shape_dg'
        ]
        for attr in attrs_to_pass:
            if hasattr(MechSystem, attr):
                kwds[attr] = getattr(MechSystem, attr)
        return kwds

    @staticmethod
    def setup_and_solve_reduced_system(kwds, solvers, client=None):
        """Setup and solve the reduced-order system."""
        print(f'Setting up reduced system...')

        # Solve reduced system
        solvers_r = []

        # Try to load from checkpoint
        checkpoint_path = os.path.join('data', f'{MechSystem.keep_time}')
        kwds_file = os.path.join(checkpoint_path, 'kwds_r.joblib')
        solvers_file = os.path.join(checkpoint_path, 'solvers_r.joblib')

        if os.path.exists(checkpoint_path) and os.path.exists(kwds_file) and os.path.exists(solvers_file):
            try:
                print("Loading from checkpoint...")
                with open(kwds_file, 'rb') as f: kwds = pickle.load(f)
                with open(solvers_file, 'rb') as f: solvers_r = pickle.load(f)
                print("Checkpoint loaded successfully")
            except Exception as e:
                raise Exception(f"Error loading checkpoint: {str(e)}")
        else:
            # Setup reduced model and update methods
            ReduceMechSystem.setup_reduced_model(solvers, kwds)

            # Pass the computed bases to the workers via the kwds dictionary.
            kwds = BaseSolverMixin._transfer_reduction_attrs(kwds)

            # Execute parallel solve
            BaseSolverMixin.parallel_solve_mech_system(kwds, solvers_r, client=client)

            # Create checkpoint directory and save initial state
            os.makedirs(checkpoint_path, exist_ok=True)
            print("Creating new checkpoint...")
            with open(kwds_file, 'wb') as f: pickle.dump(kwds, f)
            with open(solvers_file, 'wb') as f: pickle.dump(solvers_r, f)

        BaseSolverMixin.measures(kwds, solvers_r)

        return solvers_r

    @staticmethod
    def setup_and_solve_hyperreduced_system(kwds, solvers, client=None):
        """Setup and solve the hyper-reduced system."""
        print(f'Setting up hyper-reduced system...')

        # Solve hyper-reduced system
        solvers_dr = []

        # Try to load from checkpoint
        checkpoint_path = os.path.join('data', f'{MechSystem.keep_time}')
        kwds_file = os.path.join(checkpoint_path, 'kwds_dr.joblib')
        solvers_file = os.path.join(checkpoint_path, 'solvers_dr.joblib')

        if os.path.exists(checkpoint_path) and os.path.exists(kwds_file) and os.path.exists(solvers_file):
            try:
                print("Loading from checkpoint...")
                with open(kwds_file, 'rb') as f: kwds = pickle.load(f)
                with open(solvers_file, 'rb') as f: solvers_dr = pickle.load(f)
                print("Checkpoint loaded successfully")
            except Exception as e:
                raise Exception(f"Error loading checkpoint: {str(e)}")
        else:
            # Setup hyperreduction using classmethod
            ReduceMechSystem.setup_hyperreduction(kwds)

            if ReduceMechSystem.hyperreducer == 'MDEIM':
                ReduceMechSystem.update_mdeim_hyperreduction(solvers, kwds)

            if ReduceMechSystem.constraints_reduce:
                solver_names = [s.__name__ for s in kwds['registered_solver_classes']]
                if 'ConformalStormerVerletSolver' in solver_names \
                    or 'ConformalImplicitMidpointSolver' in solver_names:
                    filtered_solvers, filtered_indices = ReduceMechSystem.filter_solvers(solvers, "Hamiltonian")
                    ReduceMechSystem.hyperreduce_constraints(filtered_solvers, filtered_indices, "Hamiltonian")

                if 'DiscreteGradientSolver' in solver_names:
                    filtered_solvers, filtered_indices = ReduceMechSystem.filter_solvers(solvers, "DiscreteGradient")
                    ReduceMechSystem.hyperreduce_constraints(filtered_solvers, filtered_indices, "DiscreteGradient")

            # Pass all computed bases and functions to workers
            kwds = BaseSolverMixin._transfer_reduction_attrs(kwds)

            # Execute parallel solve
            BaseSolverMixin.parallel_solve_mech_system(kwds, solvers_dr, client=client)

            # Create checkpoint directory and save initial state
            os.makedirs(checkpoint_path, exist_ok=True)
            print("Creating new checkpoint...")
            try:
                with open(kwds_file, 'wb') as f: pickle.dump(kwds, f)
            except Exception as e:
                print(f"Pickling failed for kwds. Tracing offending attribute...", flush=True)
                import tempfile
                with tempfile.NamedTemporaryFile() as tmp:
                    for k, v in kwds.items():
                        try:
                            with open(tmp.name, 'wb') as tmp_f: pickle.dump(v, tmp_f)
                        except Exception as ex:
                            print(f"FAIL: kwds['{k}'] ({type(v)}) could not be pickled: {ex}", flush=True)
                raise e
            with open(solvers_file, 'wb') as f: pickle.dump(solvers_dr, f)

        BaseSolverMixin.measures(kwds, solvers_dr)

        return solvers_dr

    @staticmethod
    def measures(kwds, solvers):
        """
        Compute various measurements based on the solution and plot the results.

        Parameters:
            kwds (dict): Dictionary of keyword arguments for the solver.
            solvers (list): List of solver instances used to solve the system. 
        """
        r_form = lambda numer, denom: r_[float('nan'), 
                (log(roll(numer, -1)/numer)/log(roll(denom, -1)/denom))[:-1]]

        # Compute en_error using a list comprehension and reshape it into a 2D array: rows = dt values, columns = Omega2 values for each solver class in kwds['registered_solver_classes']
        # For each solver_class, filter solvers and compute en_error separately
        for solver_class in kwds['registered_solver_classes']:
            filtered_solvers = [solver for solver in solvers if solver.solver_class == solver_class]
            en_error = [
                [solver.en_error for solver in filtered_solvers
                 if np.isclose(solver.dt, dt) and
                    np.isclose(solver.Omega2, omega2).all()]
                for dt in MechSystem.dt_space
                for omega2 in kwds['Omega2_space']
            ]
            # Reshape to (dt_space_dim, Omega2_space_dim)
            en_error = np.array(en_error).reshape(MechSystem.dt_space_dim, kwds['Omega2_space_dim'])

            if en_error.shape[0] > 1 and en_error[0, 0] is not None and MechSystem.dt_space_dim > 1 and not MechSystem.predict:
                print(f"\nConvergence rates for {solver_class.__name__}:")
                header = f"{'dt':>12}"
                for col in range(en_error.shape[1]):
                    header += f" | {f'Omega2_{col}':>12}"
                print(header)
                print("-" * len(header))
                all_r_values = np.array([r_form(en_error[:, col], MechSystem.dt_space) for col in range(en_error.shape[1])]).T
                for i, dt_val in enumerate(MechSystem.dt_space):
                    print(f"{dt_val:12.6f}" + "".join([f" | {val:12.6f}" for val in all_r_values[i]]))

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
        # fig.tight_layout(pad=0)
        fig.suptitle(rf'integrator = {self.solver_class.__name__}, $\Delta t = {self.dt}$', y=1)

        gs = fig.add_gridspec(5, 2)
        ax0, ax1, ax2, ax3, ax4 = [fig.add_subplot(gs[i, 0]) for i in [0, 1, 2, 3, 4]]
        ax_pp = fig.add_subplot(gs[:5, -1], projection='3d')
        ax_pp.set_box_aspect([1, 1, 1])  # Set aspect ratio to be equal for all axes

        if hasattr(self, 'sym_error'):
            plot_data(ax0, self.t_points, self.sym_error, xlims=(0, self.T_final), \
                    ylabel=r'$\Delta Sp$', margins=10)

            if max(abs(self.sym_error)) < 1e-15:
                ax0.set_ylim([-1e-15, 1e-15])

        g_norm = None
        if hasattr(self, 'g__lambda'):
            g_norm = [LA.norm(self.g__lambda(y)) for y in self.y]

            plot_data(
                ax1,
                self.t_points,
                g_norm,
                xlims=(0, self.T_final),
                ylabel=r"$\Delta \mathcal{S}$",
                margins=10,
            )

        if hasattr(self, 'eng_error'):
            plot_data(ax2, self.t_points, self.eng_error, xlims=(0, self.T_final), \
                    ylabel=r'$\Delta H$', margins=10)

        lin_momentum = np.sum(self.y[:, self.nosc:].reshape(-1, self.nosc//3, 3), axis=1)
        lim_momentum_err = r_[0, LA.norm(lin_momentum[1:] - lin_momentum[0], axis=1)]

        angular_momentum = np.cross(self.y[:, :self.nosc].reshape(-1, self.nosc//3, 3), \
                                    self.y[:, self.nosc:].reshape(-1, self.nosc//3, 3))
        angular_momentum_sum = np.sum(angular_momentum, axis=1)
        angular_momentum_err = r_[0, \
                                LA.norm(angular_momentum_sum[1:] - angular_momentum_sum[0], axis=1)]

        plot_data(ax3, self.t_points, lim_momentum_err, xlims=(0, self.T_final), \
                ylabel=r'$\Delta L$', margins=10)

        plot_data(ax4, self.t_points, angular_momentum_err, xlims=(0, self.T_final), \
                ylabel=r'$\Delta J$', margins=10)

        ax4.set_xlabel('time')
        # Plot the particle positions over time
        coords = self.y[:, :self.nosc].reshape(-1, self.nosc//3, 3)
        for i in range(coords.shape[1]):
            ax_pp.scatter(coords[0, i, 0], coords[0, i, 1], coords[0, i, 2], s=20, c='teal')  # plot initial configuration
            ax_pp.plot(coords[:, i, 0], coords[:, i, 1], coords[:, i, 2], 'k-')  # plot system evolution

            # Plot projection onto x-y plane (z=0)
            ax_pp.plot(coords[:, i, 0], coords[:, i, 1], np.zeros_like(coords[:, i, 2]), 'r-', alpha=0.7)

        # Set the axes' labels and title
        ax_pp.set_xlabel('x', labelpad=10)
        ax_pp.set_ylabel('y', labelpad=10)
        ax_pp.set_zlabel('z', labelpad=10)
        ax_pp.set_title('Phase portrait')
        ax_pp.set_box_aspect([1, 1, 1])  # Set aspect ratio to be equal for all axes
        ax_pp.view_init(elev=15, azim=45)

        # Hide inner labels
        for ax in [ax0, ax1, ax2, ax3, ax4]:
            ax.label_outer()

        fig.show()            

        fig_data = {
            't_points': self.t_points,
            'lim_momentum_err': lim_momentum_err,
            'angular_momentum_err': angular_momentum_err,
            'coords': coords
        }
        if hasattr(self, 'sym_error'):
            fig_data['sym_error'] = self.sym_error
        if g_norm is not None:
            fig_data['g_norm'] = g_norm
        if hasattr(self, 'eng_error'):
            fig_data['eng_error'] = self.eng_error

        solver_name_short = self.solver_class.__name__.replace('Conformal', 'C').replace('StormerVerlet', 'SV').replace('ImplicitMidpoint', 'IM').replace('DiscreteGradient', 'DG')
        suffix = '_full' if not hasattr(self, 'RB') else '_r' if not hasattr(MechSystem, 'RBxUx_inv_PxU') else '_dr'
        suffix += '_predict' if self.predict else '' if not hasattr(self, 'RB') else '_repro'
        filename = os.path.join(MechSystem.data_folder, f"osc_{solver_name_short}{suffix}.pdf")
        save_figure(fig, filename, fig_data=None)

class DiscreteGradientSolver(BaseSolverMixin, DiscreteGradient):
    """DiscreteGradient solver with DG-specific capabilities."""
    def __init__(self, kwds):
        super().__init__(kwds)
        self.g = self.g__
        self.g_prime = (self.g_prime__, self._g_prime_with_JJ)

    def _g_prime_with_JJ(self, y): #New class method
        return self.g_prime__(y) @ self.JJ.T

    def residual(self, x, x1, Lambda):

        g_x1 = self.g(x1)
        resi = r_[x1 - x - self.dt*self.f(c_[x, x1].T, None) - self.dt*self._g_prime_with_JJ(0.5 *(x + x1)).T @ Lambda,\
                g_x1]
        tang = r_[c_[np.eye(x1.shape[0]) - self.dt * self.dfdu(c_[x, x1].T, None) - 0.5 * self.dt * self.JJ @ self.g_prime_x_lambda_y(0.5 *(x + x1), Lambda), -self.dt * self.JJ @ self.g_prime_x_lambda_lambda(0.5 *(x + x1), Lambda)],\
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
        residual = self.tol * 10

        while residual > self.tol and m < self.M:

            # Update residual and tangent
            resi, tang = self.residual(x[0], x[1], Lambda)

            Delta_z = -LA.solve(tang, resi)
            x[1] = x[1] + Delta_z[:x[1].shape[0]]
            Lambda += Delta_z[x[1].shape[0]:]

            # Update iteration counter and residual
            m += 1
            residual = LA.norm(r_[resi, Delta_z], np.inf)

        # Check convergence
        if m >= self.M:
            raise RuntimeError("Nonlinear solver did not converge")


class ConformalStormerVerletSolver(BaseSolverMixin, ConformalStormerVerlet):
    """Conformal Stormer-Verlet Solver with Hamiltonian capabilities."""
    def __init__(self, kwds):
        super().__init__(kwds)
        self.g = self.g__
        self.g_prime = (self.g_prime_diag, self.block_diag_matrix)

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
            p_half1 = p_half_unconstrained - 0.5 * dt * G_q0.T @ Lambda_1
            
            # q_{n+1}
            q_next = q0 + dt * f(np.concatenate([q0, p_half1]), None)[:neq]
            x[1, :neq] = q_next
            
            # Check position constraints
            g_pos = self.g(x[1])[:i0]
            if LA.norm(g_pos) < self.tol:
                break
                
            # Jacobian R11 = G(q_{n+1}) @ (-0.5 * dt^2 * G(q0)^T)
            G_qnext = self.g_prime[0](x[1])[:i0, :i1]
            R11 = -0.5 * dt**2 * G_qnext @ G_q0.T
            
            Delta_Lambda_1 = LA.lstsq(R11, g_pos, rcond=None)[0]
            Lambda_1 -= Delta_Lambda_1
            m += 1
            
        if m >= self.M:
             if self.RB is not None and LA.norm(g_pos) < 1e-5 and not m%100:
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
            g_vel = self.g(x[1])[i0:]
            if LA.norm(g_vel) < self.tol:
                break
                
            # Jacobian R22 = -0.5 * dt * G(p_{n+1}) @ G(q_{n+1})^T
            G_pnext = self.g_prime__(x[1])[i0:, i1:]
            R22 = -0.5 * dt * G_pnext @ G_qnext.T
            
            Delta_Lambda_2 = LA.lstsq(R22, g_vel, rcond=None)[0]
            Lambda_2 -= Delta_Lambda_2
            m += 1
            
        if m >= self.M:
             if self.RB is not None and LA.norm(g_vel) < 1e-5 and not m%100:
                 print(f"Warning: Velocity constraints not fully satisfied (error={LA.norm(g_vel):.2e}) in reduced model. Continuing.", flush=True)
             else:
                 raise RuntimeError(f"Nonlinear solver (velocity) did not converge. m: {m} -- Error: {LA.norm(g_vel)}")
             
        # Update Lambda in place
        Lambda[:i0] = Lambda_1
        Lambda[i0:] = Lambda_2


class ConformalImplicitMidpointSolver(BaseSolverMixin, ConformalImplicitMidpoint):
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
