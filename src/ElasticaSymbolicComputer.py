import numpy as np
import sympy as smp
import concurrent.futures
import cloudpickle
from tqdm.auto import tqdm
import os

from PlotScript import timing
from SymbolicComputer import compile_hybrid_function, ShapeWrapper, _run_method, CODE_VERSION


class ElasticaSymbolicComputer:
    """
    Computes and stores symbolic expressions for the Discrete Euler Elastica / Flexible Beam.
    
    Model:
    - Centerline discretized into N_nodes = nosc // 3 nodes in 3D.
    - N_seg = N_nodes - 1 segments with length ds = L / N_seg.
    - Inextensibility constraints: ||q_i - q_{i-1}||^2 = ds^2 for each segment.
    - Root boundary condition: Pinned at origin (q_0 = 0) or Clamped.
    - Bending potential: Discrete curvature energy based on normalized turning angle.
    - Gravity: mg * z_i along the vertical direction.
    """

    def __init__(self, nosc, length=1.0, mass_total=1.0, gravity=9.81, boundary="pinned"):
        assert nosc % 3 == 0, f"nosc must be divisible by 3 (got {nosc})"
        self.nosc = nosc
        self.n_nodes = nosc // 3
        assert self.n_nodes >= 3, f"Elastica requires at least 3 nodes (got {self.n_nodes})"
        self.n_seg = self.n_nodes - 1
        self.n_joints = self.n_seg - 1
        self.length = length
        self.ds = length / self.n_seg
        self.mass_node = mass_total / self.n_nodes
        self.gravity = gravity
        self.boundary = boundary

        # 1. Standard symbols (formatted as y_:{nosc*2//3}_:{3} to match IndexedBaseSymbolicComputer)
        self.y = smp.Matrix(smp.symbols(f"y_:{self.nosc*2//3}_:{3}", real=True))
        self.y1 = smp.Matrix(smp.symbols(f"y1_:{self.nosc*2//3}_:{3}", real=True))
        self.omega2 = smp.Matrix(smp.symbols(f"omega^2_:{self.n_joints}", real=True))
        self.beta = smp.symbols("beta", real=True)

        # 2. IndexedBase symbols for lambdify
        self.y_base = smp.IndexedBase("y")
        self.y1_base = smp.IndexedBase("y1")
        self.omega2_base = smp.IndexedBase("omega2")

        # 3. Substitution rules
        self.y_subs = {s: self.y_base[i] for i, s in enumerate(self.y)}
        self.y1_subs = {s: self.y1_base[i] for i, s in enumerate(self.y1)}
        self.omega2_subs = {s: self.omega2_base[i] for i, s in enumerate(self.omega2)}

        # Determine number of constraints:
        # Pinned: 3 (q0) + n_seg (distance)
        # Clamped: 3 (q0) + 2 (q1_y, q1_z) + n_seg (distance)
        if self.boundary == "pinned":
            self.n_pos_constraints = 3 + self.n_seg
        elif self.boundary == "clamped":
            self.n_pos_constraints = 5 + self.n_seg
        else:
            raise ValueError(f"Unknown boundary: {self.boundary}")

        self.n_total_constraints = 2 * self.n_pos_constraints

        # Symbols for Lagrange multipliers
        self.lag_mult_syms = smp.Matrix(
            smp.symbols(f"gamma_:{self.n_total_constraints}", real=True)
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
        self.common_subs_items = tuple(sorted(self.common_subs.items(), key=lambda x: x[0].name))

        # Position and momentum 3D vectors
        self._q = smp.Matrix(self.y[:nosc])
        self.q = self._q.reshape(self.n_nodes, 3)
        self._p = smp.Matrix(self.y[nosc:])
        self.p = self._p.reshape(self.n_nodes, 3)

    def _lambdify_hybrid(self, args, expr, custom_subs=None):
        if custom_subs:
            subs_dict = self.common_subs.copy()
            subs_dict.update(custom_subs)
            return compile_hybrid_function(args, expr, subs_dict)
        return compile_hybrid_function(args, expr, self.common_subs_items)

    def _compute_ham_z_expr(self):
        """Helper to compute Hamiltonian and its gradient."""
        # 1. Kinetic energy: T(p) = sum( ||p_i||^2 / (2 * m) )
        kin_expr = (0.5 / self.mass_node) * self._p.dot(self._p)

        # 2. Discrete bending potential:
        # Based on normalized turning angle: V = B / ds * (1 - e_prev . e_next / (|e_prev| |e_next|))
        pot_bend_terms = []
        for i in range(1, self.n_seg):
            e_prev = self.q.row(i) - self.q.row(i - 1)
            e_next = self.q.row(i + 1) - self.q.row(i)
            norm_prev = smp.sqrt(e_prev.dot(e_prev))
            norm_next = smp.sqrt(e_next.dot(e_next))
            cos_theta = e_prev.dot(e_next) / (norm_prev * norm_next)
            pot_bend_terms.append((self.omega2[i - 1] / self.ds) * (1 - cos_theta))

        pot_bend_vec = smp.Matrix(pot_bend_terms) if pot_bend_terms else smp.Matrix([0])
        pot_bend_expr = sum(pot_bend_terms) if pot_bend_terms else smp.Integer(0)

        # 3. Gravitational potential: V_grav = sum( m * g * z_i )
        pot_grav_expr = smp.Integer(0)
        if self.gravity != 0:
            for i in range(self.n_nodes):
                pot_grav_expr += self.mass_node * self.gravity * self.q[i, 2]

        ham_expr = kin_expr + pot_bend_expr + pot_grav_expr
        ham_z_expr = smp.Matrix([ham_expr]).jacobian(self.y).T
        return ham_expr, ham_z_expr, pot_bend_vec

    def compute_hamiltonian(self):
        """Compute Hamiltonian expressions and Hessians."""
        print("Computing Elastica Hamiltonian expressions...")
        ham_expr, ham_z_expr, _ = self._compute_ham_z_expr()
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

    def compute_constraints(self):
        """Compute constraint expressions g(y) and Jacobian g_prime(y)."""
        print("Computing Elastica constraint expressions...")
        pos_constraints = []
        vel_constraints = []
        ds2 = self.ds ** 2

        # 1. Root boundary condition: Node 0 fixed at origin
        pos_constraints.extend([self.q[0, 0], self.q[0, 1], self.q[0, 2]])
        vel_constraints.extend([self.p[0, 0], self.p[0, 1], self.p[0, 2]])

        if self.boundary == "clamped":
            pos_constraints.extend([self.q[1, 1], self.q[1, 2]])
            vel_constraints.extend([self.p[1, 1], self.p[1, 2]])

        # 2. Inextensibility of all segments:
        for i in range(1, self.n_nodes):
            diff_q = self.q.row(i) - self.q.row(i - 1)
            diff_p = self.p.row(i) - self.p.row(i - 1)
            pos_constraints.append((diff_q.dot(diff_q) - ds2) / 2)
            vel_constraints.append(diff_q.dot(diff_p))

        g_expr = smp.Matrix(pos_constraints + vel_constraints)
        g_prime_expr = g_expr.jacobian(self.y)

        lag_mult = self.lag_mult_syms
        g_prime_x_lambda_expr = g_prime_expr.T @ lag_mult
        g_prime_x_lambda_y_expr = g_prime_x_lambda_expr.jacobian(self.y)

        g_prime_nonzero_indices = np.where(np.array(g_prime_expr.tolist()).flatten() != 0)[0]
        g_prime_x_lambda_y_nonzero_indices = np.where(np.array(g_prime_x_lambda_y_expr.tolist()).flatten() != 0)[0]

        return {
            "g_expr": g_expr, "g_prime_expr": g_prime_expr,
            "g_": self._lambdify_hybrid((self.y_base,), g_expr),
            "g_prime_": ShapeWrapper(smp.lambdify((self.y,), g_prime_expr, modules=["numpy"], cse=True)),
            "g_prime_nonzero_indices": g_prime_nonzero_indices,
            "lag_mult": lag_mult,
            "g_prime_x_lambda_expr": g_prime_x_lambda_expr,
            "g_prime_x_lambda_y_expr": g_prime_x_lambda_y_expr,
            "g_prime_x_lambda_y_": self._lambdify_hybrid((self.y_base, self.lag_mult_base), g_prime_x_lambda_y_expr),
            "g_prime_x_lambda_y_nonzero_indices": g_prime_x_lambda_y_nonzero_indices,
        }

    @timing
    def compute_all(self, expressions):
        """Compute all symbolic expressions in parallel."""
        with concurrent.futures.ProcessPoolExecutor() as executor:
            future_map = {
                executor.submit(_run_method, self, 'compute_hamiltonian'): 'ham',
                executor.submit(_run_method, self, 'compute_constraints'): 'con',
            }

            for future in tqdm(concurrent.futures.as_completed(future_map), total=len(future_map), desc="Computing Elastica expressions"):
                result_bytes = future.result()
                result_exprs = cloudpickle.loads(result_bytes)
                expressions.update(result_exprs)
