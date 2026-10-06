import os
import gc
import copy
import warnings
import numpy as np
from numpy import linalg as LA
from tqdm.auto import tqdm

from System import (MechSystem, HamiltonianMechSystem, LagrangianMechSystem,
                   ReducedHamiltonianMechSystem, ReducedLagrangianMechSystem, HyperReducedHamiltonianMechSystem, HyperReducedLagrangianMechSystem,
                   fast_dump, fast_load, check_checkpoint_exists)
from Reduction import Basis, IdentityBasis, as_basis
from ODESolver import (ConformalStormerVerlet, ConformalImplicitMidpoint, 
                      DiscreteGradient)
from ReduceMechSystem import ReduceMechSystem, HamiltonianReducer, DiscreteGradientReducer
from PlotScript import timing, plot_solvers, plot_combined_errors

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
        import traceback
        traceback.print_exc()
        return e

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


class ReducedSolverMixin:
    """Mixin providing polymorphic hooks for reduced-order and hyper-reduced solvers."""
    model_tier = 'rom'

    @property
    def basis(self):
        b = getattr(self, '_basis', None)
        if b is not None:
            return b
        rb = getattr(self, 'RB', None)
        if rb is not None:
            self._basis = Basis(rb)
            return self._basis
        return IdentityBasis(2 * getattr(self, 'nosc', 0))

    @basis.setter
    def basis(self, value):
        self._basis = as_basis(value, default_dim=2 * getattr(self, 'nosc', 0))
        if hasattr(self._basis, 'V'):
            self.RB = self._basis.V

    def to_full_state(self, y):
        """Reconstruct full-order state from reduced coordinates: y_full = basis.reconstruct(y)."""
        return self.basis.reconstruct(y)

    def reproject_to_manifold(self, y_red):
        """
        Reproject reduced state y_red = [q_r, p_r] onto the constraint manifold.
        Eliminates position and velocity constraint jumps at traveling window
        transitions caused by changing the projection basis subspace.
        """
        if not getattr(self, 'constraint_type', None):
            return y_red

        g_fn = getattr(self, 'g_full', None) or getattr(self, 'g_', None) or getattr(self, 'g_vec', None)
        g_prime_fn = getattr(self, 'g_prime_', None)
        if g_fn is None or g_prime_fn is None:
            return y_red

        nosc_r = getattr(self, 'nosc_r', y_red.size // 2)
        nosc = getattr(self, 'nosc', None)
        if nosc is None:
            return y_red

        if hasattr(self, 'RB') and self.RB is not None:
            V = self.RB[:nosc, :nosc_r]
        elif hasattr(self, 'basis') and hasattr(self.basis, 'V'):
            V = self.basis.V[:nosc, :nosc_r]
        else:
            return y_red

        q_r = y_red[:nosc_r].copy()
        p_r = y_red[nosc_r:].copy()

        # 1. Position Gauss-Newton reprojection: g(V q_r) = 0
        # ASSUMPTION: Standard Euclidean/identity kinetic metric (M = I) used for tangent space projection.
        for _ in range(5):
            q_full = V @ q_r
            y_eval = np.concatenate([q_full, np.zeros(nosc)])
            g_all = g_fn(y_eval)
            m = g_all.shape[0] // 2
            g_q = g_all[:m]
            if LA.norm(g_q) < 1e-12:
                break
            G_q = g_prime_fn(y_eval)[:m, :nosc]
            G_r = G_q @ V
            try:
                delta_lambda = LA.solve(G_r @ G_r.T, g_q)
            except LA.LinAlgError:
                delta_lambda = LA.lstsq(G_r @ G_r.T, g_q, rcond=None)[0]
            q_r -= G_r.T @ delta_lambda

        # 2. Velocity linear projection: G_r (p_r + delta_p_r) = 0
        # ASSUMPTION: Velocity projection assumes dq/dt = p (identity mass). For general mass matrix M,
        # the momentum projection onto the constraint nullspace uses M^{-1}: G_r M^{-1} G_r^T.
        q_full = V @ q_r
        y_eval = np.concatenate([q_full, np.zeros(nosc)])
        m = g_fn(y_eval).shape[0] // 2
        G_q = g_prime_fn(y_eval)[:m, :nosc]
        G_r = G_q @ V
        Gp = G_r @ p_r
        if LA.norm(Gp) > 1e-12:
            try:
                delta_lambda_p = LA.solve(G_r @ G_r.T, Gp)
            except LA.LinAlgError:
                delta_lambda_p = LA.lstsq(G_r @ G_r.T, Gp, rcond=None)[0]
            p_r -= G_r.T @ delta_lambda_p

        y_proj = np.empty_like(y_red)
        y_proj[:nosc_r] = q_r
        y_proj[nosc_r:] = p_r
        return y_proj

    def init_trajectory_arrays(self):
        """Allocate state and Lagrange multiplier arrays for reduced models."""
        from System import MechSystem
        if getattr(self, '_traveling', False):
            self.y_full = np.zeros((self.n+1, 2*self.nosc))
            self.y_full[0] = MechSystem.y_init
            self.y = np.zeros((self.n+1, 2*self.nosc))
            self.y[0] = MechSystem.y_init
        else:
            self.y = np.zeros((self.n+1, 2*self.nosc_r))
            self.y_full = np.zeros((self.n+1, 2*self.nosc))
            self.y_full[0] = MechSystem.y_init
            self.y[0] = self.y_init
        self.Lambda = np.zeros((self.n+1, self.g_(np.zeros(2*self.nosc)).shape[0]))

    def record_step(self, k, y_, y_full_):
        """Record reduced state and full-order projection at step k+1."""
        self.y_full[k+1] = y_full_[-1]
        if not getattr(self, '_traveling', False):
            self.y[k+1] = y_[-1]

    def finalize_trajectory(self):
        """Promote reconstructed full state to self.y and preserve reduced trajectory in self.y_red."""
        if not getattr(self, '_traveling', False):
            self.y_red = self.y.copy()
        else:
            self.y_red = self.basis.reduce(self.y_full) if hasattr(self, 'basis') else (self.y_full @ self.RB_windows[0] if hasattr(self, 'RB_windows') and self.RB_windows else None)
        self.y = self.y_full.copy()
        del self.y_full
        gc.collect()


class HyperReducedSolverMixin(ReducedSolverMixin):
    """Mixin providing tier tagging for hyper-reduced solvers."""
    model_tier = 'hrom'


class BaseSolverMixin:
    """
    Base mixin class providing common solving capabilities for solvers.
    """
    matches_solver_family = staticmethod(_matches_solver_family)
    model_tier = 'fom'

    @property
    def basis(self):
        b = getattr(self, '_basis', None)
        if b is not None:
            return b
        rb = getattr(self, 'RB', None)
        if rb is not None:
            self._basis = Basis(rb)
            return self._basis
        return IdentityBasis(2 * getattr(self, 'nosc', 0))

    @basis.setter
    def basis(self, value):
        self._basis = as_basis(value, default_dim=2 * getattr(self, 'nosc', 0))
        if hasattr(self._basis, 'V'):
            self.RB = self._basis.V

    def to_full_state(self, y):
        """Reconstruct full-order state from coordinates. FOM identity mapping."""
        return self.basis.reconstruct(y)

    def reproject_to_manifold(self, y):
        """No-op on BaseSolverMixin; overridden on ReducedSolverMixin."""
        return y

    def init_trajectory_arrays(self):
        """Allocate state and Lagrange multiplier arrays for FOM."""
        self.y = np.zeros((self.n+1, 2*self.nosc))
        self.y[0] = self.y_init
        self.Lambda = np.zeros((self.n+1, self.g_(np.zeros(2*self.nosc)).shape[0]))

    def record_step(self, k, y_, y_full_):
        """Record the solution at time step k+1 for FOM."""
        self.y[k+1] = y_[-1]

    def finalize_trajectory(self):
        """Finalize trajectory storage at completion of solve for FOM."""
        pass

    def step(self, y_, k):
        """Advance the solution by one time step (from t_k to t_{k+1}).

        If multi-stage splitting is configured via self.w_values, executes
        each substep sequentially.
        """
        w_values = getattr(self, 'w_values', [1])
        for w_val in w_values:
            self.set_initial_condition(y_[1], check=False)
            
            # Temporarily update dt for the substep
            original_dt = self.dt
            self.dt = w_val * original_dt
            try:
                if isinstance(self, DiscreteGradient) and self.constraint_type:
                    # Heun's method predictor
                    f_val = self.f(np.array([y_[0], y_[0]]), None)
                    y_euler = y_[0] + self.dt * f_val
                    f_val_next = self.f(np.array([y_euler, y_euler]), None)
                    y_[1] = y_[0] + 0.5 * self.dt * (f_val + f_val_next)
                else:
                    y_, _, _ = super().solve(w_val * self.t_points[k:k+2])
            finally:
                self.dt = original_dt
            
            if k > 0 and hasattr(self, 'Lambda'):
                Lambda = self.Lambda[k].copy()
            else:
                Lambda = np.zeros_like(self.g_(np.zeros(2*self.nosc)))

            if self.constraint_type:
                self.fixed_point(y_, Lambda)

        y_full_ = self.to_full_state(y_)
        return y_, y_full_, Lambda

    @timing
    def solve_trajectory(self):
        """Solve the system trajectory over the specified time range.

        When per-window bases are available (``RB_windows`` attribute is set on
        the instance), the solver automatically swaps the active reduced basis
        and hyper-reduction operators at every time-window boundary.
        """

        # ── Traveling-basis setup ────────────────────────────────────────────────
        # Detect whether per-window bases have been loaded onto this instance.
        # They are stored in lists indexed by window (0-based).
        _traveling = (
            hasattr(self, 'RB_windows') and
            self.RB_windows is not None and
            len(self.RB_windows) >= 1
        )
        self._traveling = _traveling

        self.init_trajectory_arrays()

        if _traveling:
            _n_windows = len(self.RB_windows)
            _current_window = 0

            # ── Build a step-to-window lookup from the adaptive boundary fracs ──
            # _boundary_fracs = [0.0, f1, f2, ..., 1.0] (length n_windows + 1)
            # stored on the instance via _REDUCTION_ATTRS.
            # Convert fractional positions to absolute step numbers once.
            _has_adaptive_boundaries = (
                hasattr(self, '_boundary_fracs') and
                self._boundary_fracs is not None and
                len(self._boundary_fracs) == _n_windows + 1
            )
            if _has_adaptive_boundaries:
                # Step boundaries: step k belongs to window w when
                #   _step_boundaries[w] <= k < _step_boundaries[w+1]
                _step_boundaries = [
                    int(f * self.n) for f in self._boundary_fracs
                ]
                import bisect as _bisect
                def _window_for_step(k):
                    # bisect_right gives insertion point → index of the
                    # boundary that k falls before = window index.
                    w = _bisect.bisect_right(_step_boundaries, k) - 1
                    return max(0, min(w, _n_windows - 1))
            else:
                # Fallback: equal-size windows (old checkpoints / N_WINDOWS=1 case)
                _window_size = self.n / _n_windows
                def _window_for_step(k):
                    return min(int(k / _window_size), _n_windows - 1)

            # Helper: set the per-window hyper-reduction operators on self so that
            # the methods in HyperReducedMechSystem (ham_z, etc.) pick them up.
            def _activate_window(w, k=0):
                self.RB = self.RB_windows[w]
                self.nosc_r = self.nosc_r_windows[w]
                self._basis = Basis(self.RB)

                val = getattr(self, 'deim_field_windows', None)
                if isinstance(val, (list, tuple)) and len(val) > w and val[w] is not None:
                    self.deim_field = val[w]

                val = getattr(self, 'mdeim_Hessian_windows', None)
                if isinstance(val, (list, tuple)) and len(val) > w and val[w] is not None:
                    self.mdeim_Hessian = val[w]

                # Hamiltonian DEIM/MDEIM functions
                val = getattr(self, 'ham_z_deim_windows', None)
                if isinstance(val, (list, tuple)) and len(val) > w and val[w] is not None:
                    self.ham_z_deim = val[w]
                    self.__class__.ham_z_deim = staticmethod(val[w])

                val = getattr(self, 'ham_zz_mdeim_windows', None)
                if isinstance(val, (list, tuple)) and len(val) > w and val[w] is not None:
                    self.ham_zz_mdeim = val[w]
                    self.__class__.ham_zz_mdeim = staticmethod(val[w])

                # Lagrangian DEIM/MDEIM functions
                val = getattr(self, 'lag_dg_deim_windows', None)
                if isinstance(val, (list, tuple)) and len(val) > w and val[w] is not None:
                    self.lag_dg_deim = val[w]
                    self.__class__.lag_dg_deim = staticmethod(val[w])

                val = getattr(self, 'lag_dg_z_mdeim_windows', None)
                if isinstance(val, (list, tuple)) and len(val) > w and val[w] is not None:
                    self.lag_dg_z_mdeim = val[w]
                    self.__class__.lag_dg_z_mdeim = staticmethod(val[w])

                # ── Constraint MDEIM operators ─────────────────────────────────────
                val = getattr(self, 'mdeim_g_prime_windows', None)
                if isinstance(val, (list, tuple)) and len(val) > w and val[w] is not None:
                    self.mdeim_g_prime = val[w]

                val = getattr(self, 'g_prime_mdeim_windows', None)
                if isinstance(val, (list, tuple)) and len(val) > w and val[w] is not None:
                    self.g_prime_mdeim = val[w]
                    self.__class__.g_prime_mdeim = staticmethod(val[w])

                val = getattr(self, 'mdeim_g_var_windows', None)
                if isinstance(val, (list, tuple)) and len(val) > w and val[w] is not None:
                    self.mdeim_g_var = val[w]

                val = getattr(self, 'g_prime_x_lambda_y_mdeim_windows', None)
                if isinstance(val, (list, tuple)) and len(val) > w and val[w] is not None:
                    self.g_prime_x_lambda_y_mdeim = val[w]
                    self.__class__.g_prime_x_lambda_y_mdeim = staticmethod(val[w])

                # Re-initialise JJ to match the potentially new nosc_r
                from System import MechSystem as _MS
                if self.reducer == 'psd':
                    self.JJ = _MS.compute_J(self.nosc_r)

                print(f'  [Traveling basis] Activated window {w+1}/{_n_windows} '
                      f'at step k={k} '
                      f'(nosc_r={self.nosc_r}, RB={self.RB.shape})', flush=True)

            _activate_window(0, k=0)

        # ── End traveling-basis setup ────────────────────────────────────────────

        # Calculate update frequency (10% of iterations)
        update_freq = max(1, self.n // 10)  

        # Initialize initial state representation for step 0
        if _traveling:
            y_k = self.basis.reduce(self.y_full[0])
            y_k = self.reproject_to_manifold(y_k)
            self.y_full[0] = self.to_full_state(y_k)
            y_ = np.array([y_k, y_k])
        else:
            y_ = np.array([self.y[0], self.y[0]])

        # Create progress bar that updates less frequently
        with tqdm(total=self.n, desc='Solving trajectory', 
                miniters=update_freq,  # Minimum iterations between updates
                maxinterval=5.0,  # Maximum seconds between updates
                position=0, 
                leave=True) as pbar:

            for k in range(self.n):

                # ── Traveling-basis: swap only when crossing a window boundary ───
                if _traveling:
                    new_window = _window_for_step(k)
                    if new_window != _current_window:
                        _current_window = new_window
                        _activate_window(_current_window, k=k)
                        y_k = self.basis.reduce(self.y_full[k])
                        y_k = self.reproject_to_manifold(y_k)
                        self.y_full[k] = self.to_full_state(y_k)
                        y_ = np.array([y_k, y_k])
                    elif k > 0:
                        y_ = np.array([y_[1], y_[1]])
                elif k > 0:
                    y_ = np.array([self.y[k], self.y[k]])
                # ── End swap ─────────────────────────────────────────────────────

                y_, y_full_, self.Lambda[k+1] = self.step(y_, k)

                self.record_step(k, y_, y_full_)

                # Update progress bar every update_freq iterations
                if (k + 1) % update_freq == 0:
                    pbar.update(update_freq)
                    pbar.set_postfix({'step': k+1, 'total': self.n})

            # Update any remaining iterations
            remaining = self.n % update_freq
            if remaining:
                pbar.update(remaining)

        self.finalize_trajectory()

    def compute_energy_error(self):
        """Compute time-series energy error (eng_error) and scalar RMS energy error (en_error)."""
        if hasattr(self, 'ham') and hasattr(self, 'y') and self.y is not None and not getattr(self, 'beta', False):
            self.eng_error = np.array([self.ham(y) for y in self.y]) - self.ham(self.y[0])
            self.en_error = np.sqrt(self.dt) * LA.norm(self.eng_error)
        else:
            self.eng_error = None
            self.en_error = None

    def compute_sym_error(self):
        """Compute time-series symplectic error."""
        if getattr(self, 'var', False):
            from System import MechSystem as _MS
            _traveling = (
                hasattr(self, 'RB_windows') and
                self.RB_windows is not None and
                len(self.RB_windows) >= 1
            )
            if _traveling and hasattr(self, 'y') and self.y is not None:
                try:
                    n_windows = len(self.RB_windows)
                    boundary_fracs = getattr(self, '_boundary_fracs', None)
                    if boundary_fracs is None or len(boundary_fracs) != n_windows + 1:
                        boundary_fracs = np.linspace(0.0, 1.0, n_windows + 1)
                    n_steps = len(self.y) - 1
                    step_boundaries = [int(round(f * n_steps)) for f in boundary_fracs]
                    step_boundaries[0] = 0
                    step_boundaries[-1] = n_steps

                    window_sym_errors = []
                    for w in range(min(n_windows, len(step_boundaries) - 1)):
                        k_start = step_boundaries[w]
                        k_end = step_boundaries[w + 1]
                        if k_end <= k_start:
                            continue

                        # Activate window basis and hyperreduction operators
                        self.basis = self.RB_windows[w]
                        self.nosc_r = self.nosc_r_windows[w] if hasattr(self, 'nosc_r_windows') else self.RB_windows[w].shape[1] // 2
                        self.neq = 2 * self.nosc_r
                        self.JJ = _MS.compute_J(self.nosc_r)
                        self.JJ_r = self.JJ
                        self.J_mat = self.JJ
                        self.I_mat = np.eye(self.neq)
                        self.I_mat_norm = LA.norm(self.I_mat)
                        self.Ecoeff_dt_inv = self.Ecoeff(self.dt)**4

                        for attr in ('deim_field', 'ham_z_deim', 'ham_zz_mdeim', 'mdeim_Hessian', 'lag_dg_deim', 'lag_dg_z_mdeim'):
                            w_list = getattr(self, attr + '_windows', None)
                            if isinstance(w_list, (list, tuple)) and len(w_list) > w and w_list[w] is not None:
                                setattr(self, attr, w_list[w])

                        u_w = self.basis.reduce(self.y[k_start:k_end + 1])
                        orig_t_points = getattr(self, 't_points', None)
                        self.t_points = np.arange(len(u_w)) * self.dt
                        try:
                            errs_w = self.var_solve(u_w)
                        finally:
                            if orig_t_points is not None:
                                self.t_points = orig_t_points

                        if w == 0:
                            window_sym_errors.append(errs_w)
                        else:
                            window_sym_errors.append(errs_w[1:])

                    self.sym_error = np.concatenate(window_sym_errors) if window_sym_errors else None

                    # Restore window 0 as default state
                    self.basis = self.RB_windows[0]
                    self.nosc_r = self.nosc_r_windows[0] if hasattr(self, 'nosc_r_windows') else self.RB_windows[0].shape[1] // 2
                    self.neq = 2 * self.nosc_r
                    self.JJ = _MS.compute_J(self.nosc_r)
                    self.JJ_r = self.JJ
                    self.J_mat = self.JJ
                    self.I_mat = np.eye(self.neq)
                    self.I_mat_norm = LA.norm(self.I_mat)
                    for attr in ('deim_field', 'ham_z_deim', 'ham_zz_mdeim', 'mdeim_Hessian', 'lag_dg_deim', 'lag_dg_z_mdeim'):
                        w_list = getattr(self, attr + '_windows', None)
                        if isinstance(w_list, (list, tuple)) and len(w_list) > 0 and w_list[0] is not None:
                            setattr(self, attr, w_list[0])
                except Exception:
                    self.sym_error = None
            else:
                if hasattr(self, 'RB') and self.RB is not None:
                    self.basis = self.RB
                    if getattr(self, 'nosc_r', None) is None:
                        self.nosc_r = self.RB.shape[1] // 2
                    self.neq = 2 * self.nosc_r
                    self.JJ = _MS.compute_J(self.nosc_r)
                    self.JJ_r = self.JJ
                    self.J_mat = self.JJ
                    self.I_mat = np.eye(self.neq)
                    self.I_mat_norm = LA.norm(self.I_mat)
                    self.Ecoeff_dt_inv = self.Ecoeff(self.dt)**4

                    y = getattr(self, 'y_red', None)
                    if y is None and hasattr(self, 'y') and self.y is not None:
                        y = self.basis.reduce(self.y)
                        self.y_red = y
                else:
                    y = getattr(self, 'y', None)

                if y is not None:
                    try:
                        self.sym_error = self.var_solve(y)
                    except Exception:
                        self.sym_error = None
                else:
                    self.sym_error = None
        else:
            self.sym_error = None

    def compute_constraint_error(self):
        """Compute time-series constraint norm (g_norm) and max constraint error (phase_error)."""
        g_fn = getattr(self, 'g_full', None) or getattr(self, 'g_', None) or getattr(self, 'g_vec', None)
        if g_fn is not None and hasattr(self, 'y') and self.y is not None:
            self.g_norm = [LA.norm(g_fn(y)) for y in self.y]
            self.phase_error = np.amax(self.g_norm)
        else:
            self.g_norm = None
            self.phase_error = np.nan

    def compute_linear_momentum_error(self):
        """Compute time-series linear momentum error array and max error."""
        if hasattr(self, 'y') and self.y is not None:
            lin_mom = np.sum(self.y[:, self.nosc:].reshape(-1, self.nosc // 3, 3), axis=1)
            self.lin_momentum_err = np.concatenate(([0], LA.norm(lin_mom[1:] - lin_mom[0], axis=1)))
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
            self.angular_momentum_err = np.concatenate(([0], LA.norm(ang_mom[1:] - ang_mom[0], axis=1)))
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
        T_final = kwds.get('T_final', MechSystem.T_final)
        kwds = kwds.copy()
        kwds['pool'] = {
            'solver_class': solver_class, 
            'dt': dt, 
            'Omega2': Omega2,
            'n': int(round(T_final/dt))
        }
        kwds['pool'].update({
            't_points': np.linspace(0, T_final, kwds['pool']['n']+1)
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
            import traceback
            traceback.print_exc()
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
                 else:
                     print(f"ERROR: Worker task {idx} failed with: {res}", flush=True)

            if any(s is None for s in results):
                failed_count = results.count(None)
                print(f"WARNING: {failed_count} of {len(results)} tasks failed or were skipped.", flush=True)

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
                    val = getattr(cls, attr)
                    if not isinstance(val, property):
                        data[attr] = val
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

        # Find the RB, nosc_r, and V_state from the saved data. It should be consistent
        # across all solvers of a given type (Hamiltonian/DG).
        rb_ham, nosc_r_ham, v_state_ham = None, None, None
        rb_dg, nosc_r_dg, v_state_dg = None, None, None

        for cls_name, data in solver_data.items():
            if not isinstance(data, dict):
                continue
            if 'RB' in data and 'nosc_r' in data:
                if any(tag in cls_name for tag in ('StormerVerlet', 'ImplicitMidpoint', 'Hamiltonian')):
                    if rb_ham is None:
                        rb_ham, nosc_r_ham = data['RB'], data['nosc_r']
                        v_state_ham = data.get('V_state', None)
                elif any(tag in cls_name for tag in ('DiscreteGradient', 'Lagrangian')):
                    if rb_dg is None:
                        rb_dg, nosc_r_dg = data['RB'], data['nosc_r']
                        v_state_dg = data.get('V_state', None)

        # Now set the attributes on the Reducer classes
        if rb_ham is not None:
            setattr(HamiltonianReducer, 'RB', rb_ham)
            setattr(HamiltonianReducer, 'nosc_r', nosc_r_ham)
        if v_state_ham is not None:
            setattr(HamiltonianReducer, 'V_state', v_state_ham)
        if rb_dg is not None:
            setattr(DiscreteGradientReducer, 'RB', rb_dg)
            setattr(DiscreteGradientReducer, 'nosc_r', nosc_r_dg)
        if v_state_dg is not None:
            setattr(DiscreteGradientReducer, 'V_state', v_state_dg)

    @staticmethod
    def setup_and_solve_reduced_system(kwds, solvers, client=None):
        """Setup and solve the reduced-order system."""
        print(f'\nSetting up reduced system...')
        kwds = copy.deepcopy(kwds)

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
                HamiltonianReducer.setup_reduced_model(ham_solvers, ham_classes, tol=pod_tol)
            if dg_solvers:
                DiscreteGradientReducer.setup_reduced_model(dg_solvers, dg_classes, tol=pod_tol)

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
        plot_solvers(kwds, solvers_r)

        return solvers_r

    @staticmethod
    def setup_and_solve_hyperreduced_system(kwds, solvers, client=None):
        """Setup and solve the hyper-reduced system."""
        print(f'\nSetting up hyper-reduced system...')
        kwds = copy.deepcopy(kwds)

        pod_tol = kwds.get('pod_tol', MechSystem.pod_tol_sweep[-1])

        # Solve hyper-reduced system
        solvers_hr = []

        # Try to load from checkpoint
        checkpoint_path = MechSystem.data_folder
        tol_suffix = f"tol_{pod_tol:.1e}"
        kwds_file = os.path.join(checkpoint_path, f'kwds_hr_{tol_suffix}.joblib')
        solvers_file = os.path.join(checkpoint_path, f'solvers_hr_{tol_suffix}.joblib')
        if not os.path.exists(kwds_file):
            legacy_kwds = os.path.join(checkpoint_path, f'kwds_dr_{tol_suffix}.joblib')
            legacy_solvers = os.path.join(checkpoint_path, f'solvers_dr_{tol_suffix}.joblib')
            if os.path.exists(legacy_kwds) and os.path.exists(legacy_solvers):
                kwds_file = legacy_kwds
                solvers_file = legacy_solvers

        if check_checkpoint_exists(checkpoint_path, kwds_file, solvers_file):
            try:
                print("Loading from checkpoint...")
                kwds = fast_load(kwds_file)
                solvers_hr = fast_load(solvers_file)
                print("Checkpoint loaded successfully")
            except Exception as e:
                raise Exception(f"Error loading checkpoint: {str(e)}")
        else:
            # Ensure reduction attributes (RB, nosc_r, V_state) are restored on Reducer classes
            BaseSolverMixin._restore_reduction_attrs(kwds)
            # If Reducer classes lack RB (e.g. clean kwds_hr without solver_data), try restoring from ROM checkpoint
            if getattr(HamiltonianReducer, 'RB', None) is None:
                rom_kwds_file = os.path.join(checkpoint_path, f'kwds_r_{tol_suffix}.joblib')
                if os.path.exists(rom_kwds_file):
                    rom_kwds = fast_load(rom_kwds_file)
                    BaseSolverMixin._restore_reduction_attrs(rom_kwds)

            # Setup hyperreduction using classmethod
            ham_solvers = [s for s in solvers if not isinstance(s, DiscreteGradient)]
            dg_solvers = [s for s in solvers if isinstance(s, DiscreteGradient)]
            
            ham_classes = [HYPERREDUCED_SOLVER_MAPPING.get(c, c) for c in kwds['registered_solver_classes'] if issubclass(c, (ConformalStormerVerlet, ConformalImplicitMidpoint))]
            dg_classes = [HYPERREDUCED_SOLVER_MAPPING.get(c, c) for c in kwds['registered_solver_classes'] if issubclass(c, DiscreteGradient)]

            if ham_classes:
                HamiltonianReducer.setup_hyperreduction(ham_classes, tol=pod_tol)
            if dg_classes:
                DiscreteGradientReducer.setup_hyperreduction(dg_solvers, dg_classes, tol=pod_tol)

            if ReduceMechSystem.hyperreducer == 'MDEIM' and getattr(ReduceMechSystem, 'use_mdeim_hessian', False):
                if ham_solvers:
                    HamiltonianReducer.update_mdeim_hyperreduction(ham_solvers, ham_classes, tol=pod_tol)
                if dg_solvers:
                    DiscreteGradientReducer.update_mdeim_hyperreduction(dg_solvers, dg_classes, tol=pod_tol)

            # hyperreduce_constraints use parallel processing internally
            if ReduceMechSystem.constraints_reduce:
                if ham_solvers:
                    HamiltonianReducer.hyperreduce_constraints(ham_solvers, ham_classes, tol=pod_tol)
                if dg_solvers:
                    DiscreteGradientReducer.hyperreduce_constraints(dg_solvers, dg_classes, tol=pod_tol)

            # When N_WINDOWS is None or > 1, build per-window local bases and hyper-reduction
            # operators.  These are appended to the class/target attributes as lists
            # and serialised to workers via _transfer_reduction_attrs, enabling the
            # online solver to swap bases at each window boundary.
            if ReduceMechSystem.N_WINDOWS is None or ReduceMechSystem.N_WINDOWS > 1:
                n_win_str = f"{ReduceMechSystem.N_WINDOWS}-window" if ReduceMechSystem.N_WINDOWS is not None else "adaptive (unbounded)"
                print(f'\n[setup_and_solve_hyperreduced_system] '
                      f'Building {n_win_str} traveling bases...')
                if ham_solvers and ham_classes:
                    HamiltonianReducer.setup_traveling_basis(ham_solvers, ham_classes, tol=pod_tol)
                    HamiltonianReducer.setup_traveling_hyperreduction(ham_solvers, ham_classes, tol=pod_tol)
                    if ReduceMechSystem.constraints_reduce:
                        HamiltonianReducer.setup_traveling_constraint_hyperreduction(
                            ham_solvers, ham_classes, tol=pod_tol)
                if dg_solvers and dg_classes:
                    DiscreteGradientReducer.setup_traveling_basis(dg_solvers, dg_classes, tol=pod_tol)
                    DiscreteGradientReducer.setup_traveling_hyperreduction(dg_solvers, dg_classes, tol=pod_tol)
                    if ReduceMechSystem.constraints_reduce:
                        DiscreteGradientReducer.setup_traveling_constraint_hyperreduction(
                            dg_solvers, dg_classes, tol=pod_tol)
            # ── End traveling-basis setup ─────────────────────────────────────────


            # Update registered classes to hyper-reduced versions
            kwds['registered_solver_classes'] = [HYPERREDUCED_SOLVER_MAPPING.get(cls, cls) for cls in kwds['registered_solver_classes']]

            # Pass all computed bases and functions to workers
            kwds = BaseSolverMixin._transfer_reduction_attrs(kwds)

            # Execute parallel solve
            BaseSolverMixin.parallel_solve_mech_system(kwds, solvers_hr, client=client)

            # Create checkpoint directory and save initial state
            os.makedirs(checkpoint_path, exist_ok=True)
            print("Creating new checkpoint...")
            kwds_file = os.path.join(checkpoint_path, f'kwds_hr_{tol_suffix}.joblib')
            solvers_file = os.path.join(checkpoint_path, f'solvers_hr_{tol_suffix}.joblib')
            fast_dump(kwds, kwds_file)
            fast_dump(solvers_hr, solvers_file)


        BaseSolverMixin.compute_convergence_rates(kwds, solvers_hr)
        plot_solvers(kwds, solvers_hr)
        try:
            plot_combined_errors(kwds=kwds, data_folder=checkpoint_path)
        except Exception as e:
            print(f"  [Warning] Could not generate combined error plot: {e}")

        return solvers_hr

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

class DiscreteGradientFixedPointMixin:
    def __init__(self, kwds):
        super().__init__(kwds)

    def residual(self, x, x1, Lambda):
        g_x1 = self.g_vec(x1)
        x_pair = np.array([x, x1])
        x_mid = 0.5 * (x + x1)
        g_prime_mid = self.g_prime(x_mid)
        dt_JJ_gp_T = self.dt * (self.JJ @ g_prime_mid.T)

        n = x1.shape[0]
        m_dim = g_x1.shape[0]

        resi = np.empty(n + m_dim)
        resi[:n] = x1 - x - self.dt * self.f(x_pair, None) - dt_JJ_gp_T @ Lambda
        resi[n:] = g_x1

        tang = np.empty((n + m_dim, n + m_dim))
        tang[:n, :n] = np.eye(n) - self.dt * self.dfdu(x_pair, None) - 0.5 * self.dt * (self.JJ @ self.g_prime_x_lambda_y(x_mid, Lambda))
        tang[:n, n:] = -dt_JJ_gp_T
        tang[n:, :n] = self.g_prime(x1)
        tang[n:, n:] = 0.0

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
            residual = max(LA.norm(resi, np.inf), LA.norm(Delta_z, np.inf))
            
            if np.isnan(residual):
                raise RuntimeError("Nonlinear solver diverged: Residual is NaN")

            if residual < self.tol:
                return

        raise RuntimeError("Nonlinear solver did not converge")

class ConformalStormerVerletFixedPointMixin:
    def __init__(self, kwds):
        super().__init__(kwds)

    def fixed_point(self, x, Lambda):
        """
        Fixed point iteration method.

        Parameters:
        x (array): The initial guess.
        Lambda (array): The initial value for Lambda.
        """

        # Setup constants
        f, neq, dt = self.f, self.neq//2, self.dt
        q0, p0 = x[0, :neq], x[0, neq:]
        
        # G(q0)
        g_prime_x0 = self.g_prime_diag(x[0])
        i0, i1 = g_prime_x0.shape
        i0, i1 = i0//2, i1//2
        G_q0 = g_prime_x0[:i0, :i1] # G(q0)
        
        # Split Lambda
        Lambda_1 = Lambda[:i0].copy()
        Lambda_2 = Lambda[i0:].copy()
        
        # p_half (unconstrained)
        p_half_unconstrained = p0 + 0.5 * dt * f(x[0], None)[neq:]

        # Precompute position Jacobian factor: (-0.5 * dt^2) * G(q0)^T
        half_dt2_G_q0_T = (-0.5 * dt**2) * G_q0.T
        half_dt_G_q0_T = 0.5 * dt * G_q0.T
        
        # 1. Position Constraints Loop (Solve for Lambda_1)
        m = 0
        while m < self.M:
            # p_{n+1/2}
            p_half1 = p_half_unconstrained - half_dt_G_q0_T @ Lambda_1 # Lambda_1 is 1D
            
            # ASSUMPTION: Separable Hamiltonian kinetics with identity mass (M = I):
            # \nabla_p H(q, p) = p, which yields dq/dt = p. The position update q_{n+1} = q0 + dt * p_half1
            # relies directly on \nabla_p H = p. If M != I, this becomes q_{n+1} = q0 + dt * M^{-1} p_half1.
            q_next = q0 + dt * p_half1
            x[1, :neq] = q_next
            
            # Check position constraints
            g_pos = self.g_vec(x[1])[:i0]
            assert g_pos.ndim == 1, f"StormerVerlet FixedPoint: g_pos must be 1D, got {g_pos.shape}"
            norm_g_pos = LA.norm(g_pos)
            
            if np.isnan(norm_g_pos):
                raise RuntimeError("Nonlinear solver (position) diverged: Residual is NaN")
            if norm_g_pos < self.tol:
                break
                
            # Jacobian R11 = G(q_{n+1}) @ (-0.5 * dt^2 * G(q0)^T)
            G_qnext = self.g_prime_diag(x[1])[:i0, :i1]
            R11 = G_qnext @ half_dt2_G_q0_T
            
            Delta_Lambda_1 = LA.lstsq(R11, g_pos, rcond=None)[0]
            Lambda_1 -= Delta_Lambda_1
            m += 1
            
        if m >= self.M:
             if getattr(self, 'RB', None) is not None and LA.norm(g_pos) < 1e-5 and not m%100:
                 print(f"Warning: Position constraints not fully satisfied (error={LA.norm(g_pos):.2e}) in reduced model. Continuing.", flush=True)
             else:
                 raise RuntimeError(f"Nonlinear solver (position) did not converge. m: {m} -- Error: {LA.norm(g_pos)}")

        # 2. Velocity Constraints Loop (Solve for Lambda_2)
        # ASSUMPTION: Partitioned Störmer-Verlet (RATTLE/SHAKE) splitting:
        # q_{n+1} and p_half1 are fixed during the velocity projection step. Therefore,
        # the coupling term (\partial g_vel / \partial q) does not enter the Newton tangent R22 (\Delta q = 0).
        G_qnext = self.g_prime_diag(x[1])[:i0, :i1]
        x[1, neq:] = p_half1
        force_next = f(x[1], None)[neq:]
        p_base = p_half1 + 0.5 * dt * force_next

        # Precompute velocity Jacobian factor: -0.5 * dt * G(q_{n+1})^T
        neg_half_dt_G_qnext_T = (-0.5 * dt) * G_qnext.T
        half_dt_G_qnext_T = 0.5 * dt * G_qnext.T

        m = 0
        while m < self.M:
            # p_{n+1}
            p_next = p_base - half_dt_G_qnext_T @ Lambda_2
            x[1, neq:] = p_next
            
            # Check velocity constraints
            g_vel = self.g_vec(x[1])[i0:] # g_vel is (N,1)
            assert g_vel.ndim == 1, f"StormerVerlet FixedPoint: g_vel must be 1D, got {g_vel.shape}"
            norm_g_vel = LA.norm(g_vel)
            
            if np.isnan(norm_g_vel):
                raise RuntimeError("Nonlinear solver (velocity) diverged: Residual is NaN")
            if norm_g_vel < self.tol:
                break
                
            # Jacobian R22 = -0.5 * dt * G(p_{n+1}) @ G(q_{n+1})^T
            # ASSUMPTION: Linear velocity constraint dependency on momentum:
            # g_vel(q, p) = G(q) @ p => \partial g_vel / \partial p = G(q).
            # Hence G_pnext = self.g_prime(x[1])[i0:, i1:] equals G(q_{n+1}) = \nabla_q g_pos(q_{n+1}).
            # This identity holds strictly for standard holonomic constraints with dq/dt = p,
            # meaning R22 relies on the same spatial constraint Jacobian G as R11.
            G_pnext = self.g_prime(x[1])[i0:, i1:]
            R22 = G_pnext @ neg_half_dt_G_qnext_T
            
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
        
        i0, i1 = self.g_prime(y[0]).shape
        i0, i1 = i0//2, i1//2

        result = np.block([
            [-0.25 * self.dt**2 * (self.g_prime(y[2])[i0:, i1:] + self.g_prime(y[0])[:i0, :i1] @ ham_zz_y3_22), np.zeros((i0, i1))],
            [np.zeros((i0, i1)), -0.5 * self.dt * self.g_prime(y[1])[i0:, i1:]]
        ])
        return result
    
    def g_prime_diag(self, y):
        """Compute g_prime using the original method"""
        # ASSUMPTION: The constraint Jacobian G(y) has identical diagonal blocks:
        # G_11 = \nabla_q g_pos(q) and G_22 = \nabla_p g_vel(q, p) \equiv \nabla_q g_pos(q).
        # Slicing [:i0//2, :i1//2] extracts \nabla_q g_pos(q).
        _g_prime = self.g_prime(y)
        i0, i1 = _g_prime.shape
        _g_prime_11 = _g_prime[:i0//2, :i1//2]
        result = np.block([
            [_g_prime_11, np.zeros(_g_prime_11.shape)],
            [np.zeros(_g_prime_11.shape), _g_prime_11]
        ])
        return result

class DiscreteGradientSolver(BaseSolverMixin, DiscreteGradientFixedPointMixin, DiscreteGradient, LagrangianMechSystem):
    """DiscreteGradient solver with DG-specific capabilities."""
    model_tier = 'fom'

class ReducedDiscreteGradientSolver(ReducedSolverMixin, DiscreteGradientSolver, ReducedLagrangianMechSystem):
    """Reduced order Discrete Gradient Solver."""
    model_tier = 'rom'

class HyperReducedDiscreteGradientSolver(HyperReducedSolverMixin, ReducedDiscreteGradientSolver, HyperReducedLagrangianMechSystem):
    """Hyper-reduced order Discrete Gradient Solver."""
    model_tier = 'hrom'

DiscreteGradientSolver.reduced_solver = ReducedDiscreteGradientSolver
DiscreteGradientSolver.hyperreduced_solver = HyperReducedDiscreteGradientSolver


class ConformalStormerVerletSolver(BaseSolverMixin, ConformalStormerVerletFixedPointMixin, ConformalStormerVerlet, HamiltonianMechSystem):
    """Conformal Stormer-Verlet Solver with Hamiltonian capabilities."""
    model_tier = 'fom'

class ReducedConformalStormerVerletSolver(ReducedSolverMixin, ConformalStormerVerletSolver, ReducedHamiltonianMechSystem):
    """Reduced order Conformal Stormer-Verlet Solver."""
    model_tier = 'rom'

class HyperReducedConformalStormerVerletSolver(HyperReducedSolverMixin, ReducedConformalStormerVerletSolver, HyperReducedHamiltonianMechSystem):
    """Hyper-reduced order Conformal Stormer-Verlet Solver."""
    model_tier = 'hrom'

ConformalStormerVerletSolver.reduced_solver = ReducedConformalStormerVerletSolver
ConformalStormerVerletSolver.hyperreduced_solver = HyperReducedConformalStormerVerletSolver


class ConformalImplicitMidpointSolver(BaseSolverMixin, ConformalImplicitMidpoint, HamiltonianMechSystem):
    """Conformal Implicit Midpoint Solver with Hamiltonian capabilities."""
    model_tier = 'fom'

    def __init__(self, kwds):
        super().__init__(kwds)

    def block_diag_matrix(self, y):
        """Compute g_prime using the original method"""
        _g_prime_shape = self.g_prime(y[0]).shape
        result = np.block([
            [self.g_prime(y[0])[:_g_prime_shape[0]//2, :_g_prime_shape[1]//2], np.zeros((_g_prime_shape[0]//2, _g_prime_shape[1]//2))],
            [np.zeros((_g_prime_shape[0]//2, _g_prime_shape[1]//2)), self.g_prime(y[1])[:_g_prime_shape[0]//2, :_g_prime_shape[1]//2]]
        ])
        return result

class ReducedConformalImplicitMidpointSolver(ReducedSolverMixin, ConformalImplicitMidpointSolver, ReducedHamiltonianMechSystem):
    """Reduced order Conformal Implicit Midpoint Solver."""
    model_tier = 'rom'
    def block_diag_matrix(self, y):
        return ConformalImplicitMidpointSolver.block_diag_matrix(self, y)

class HyperReducedConformalImplicitMidpointSolver(HyperReducedSolverMixin, ReducedConformalImplicitMidpointSolver, HyperReducedHamiltonianMechSystem):
    """Hyper-reduced order Conformal Implicit Midpoint Solver."""
    model_tier = 'hrom'
    def block_diag_matrix(self, y):
        return ConformalImplicitMidpointSolver.block_diag_matrix(self, y)

ConformalImplicitMidpointSolver.reduced_solver = ReducedConformalImplicitMidpointSolver
ConformalImplicitMidpointSolver.hyperreduced_solver = HyperReducedConformalImplicitMidpointSolver

REDUCED_SOLVER_MAPPING = {
    DiscreteGradientSolver: ReducedDiscreteGradientSolver,
    ConformalStormerVerletSolver: ReducedConformalStormerVerletSolver,
    ConformalImplicitMidpointSolver: ReducedConformalImplicitMidpointSolver,
    # Map reduced to reduced (idempotent)
    ReducedDiscreteGradientSolver: ReducedDiscreteGradientSolver,
    ReducedConformalStormerVerletSolver: ReducedConformalStormerVerletSolver,
    ReducedConformalImplicitMidpointSolver: ReducedConformalImplicitMidpointSolver,
}

HYPERREDUCED_SOLVER_MAPPING = {
    DiscreteGradientSolver: HyperReducedDiscreteGradientSolver,
    ConformalStormerVerletSolver: HyperReducedConformalStormerVerletSolver,
    ConformalImplicitMidpointSolver: HyperReducedConformalImplicitMidpointSolver,
    # Map hyperreduced to hyperreduced (idempotent)
    HyperReducedDiscreteGradientSolver: HyperReducedDiscreteGradientSolver,
    HyperReducedConformalStormerVerletSolver: HyperReducedConformalStormerVerletSolver,
    HyperReducedConformalImplicitMidpointSolver: HyperReducedConformalImplicitMidpointSolver
}
