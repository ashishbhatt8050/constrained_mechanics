#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reduction module providing projection operators for Model Order Reduction (MOR).

Defines structure-preserving, shape-aware basis classes following the
Pythonic trailing-axis convention:
- Full to Reduced:  z = x @ V       (reduction:   (..., N) @ (N, r) -> (..., r))
- Reduced to Full:  x = z @ V.T     (reconstruction: (..., r) @ (r, N) -> (..., N))
- Operator Galerkin: A_r = V.T @ A @ V (projection: (r, N) @ (N, N) @ (N, r) -> (r, r))
"""

from __future__ import annotations
from typing import Sequence, Any, Union
import numpy as np
from numpy import linalg as LA


class Basis:
    """
    Subspace projection operator encapsulating a reduced basis matrix V in R^{N x r}.

    Follows the trailing-axis convention:
    - reduction:      (..., N) @ (N, r) -> (..., r)
    - reconstruction: (..., r) @ (r, N) -> (..., N)
    - projection:     V^T @ A @ V       -> (r, r)

    Attributes
    ----------
    V : np.ndarray
        Full-to-reduced projection basis of shape (N, r).
    VT : np.ndarray
        Pre-cached transpose of V, shape (r, N).
    full_dim : int
        Full state dimension N.
    reduced_dim : int
        Reduced state dimension r.
    nosc : int
        Number of oscillators / configuration degrees of freedom (N // 2).
    nosc_r : int
        Reduced oscillators (r // 2).
    is_identity : bool
        False for reduced bases.
    """

    def __init__(self, V: np.ndarray | Sequence, name: str = "basis", metadata: dict | None = None):
        self.V = np.asarray(V)
        if self.V.ndim != 2:
            raise ValueError(f"Basis matrix V must be 2D, got shape {self.V.shape}")
        self.VT = self.V.T
        self.full_dim, self.reduced_dim = self.V.shape
        self.name = name
        self.metadata = metadata if metadata is not None else {}

    @property
    def N(self) -> int:
        """Full dimension N."""
        return self.full_dim

    @property
    def r(self) -> int:
        """Reduced dimension r."""
        return self.reduced_dim

    @property
    def nosc(self) -> int:
        """Full number of oscillators / configuration degrees of freedom (N // 2)."""
        return self.full_dim // 2

    @property
    def nosc_r(self) -> int:
        """Reduced number of oscillators (r // 2)."""
        return self.reduced_dim // 2

    @property
    def shape(self) -> tuple[int, int]:
        return self.V.shape

    @property
    def ndim(self) -> int:
        return 2

    @property
    def T(self) -> np.ndarray:
        """Transpose view VT of shape (r, N)."""
        return self.VT

    @property
    def is_identity(self) -> bool:
        return False

    def reduce(self, x: np.ndarray) -> np.ndarray:
        """
        Projects full-order state or trajectory to reduced coordinates: z = x @ V.
        Contracts across trailing dimension.
        Supports 1D state (N,), batch (..., N), or trajectory (T, N).
        """
        return x @ self.V

    def reconstruct(self, z: np.ndarray) -> np.ndarray:
        """
        Reconstructs full-order state or trajectory from reduced coordinates: x = z @ V.T.
        Contracts across trailing dimension.
        Supports 1D state (r,), batch (..., r), or trajectory (T, r).
        Output is natively C-contiguous.
        """
        return z @ self.VT

    def project_matrix(self, A: np.ndarray) -> np.ndarray:
        """
        Galerkin projection of an operator: A_r = V.T @ A @ V.
        Supports dense arrays, scipy.sparse matrices, and linear operators.
        """
        if hasattr(A, 'dot') and not isinstance(A, np.ndarray):
            return self.VT @ A.dot(self.V)
        return self.VT @ A @ self.V

    def project_dual_matrix(self, A: np.ndarray) -> np.ndarray:
        """Alias for project_matrix(A)."""
        return self.project_matrix(A)

    def project_vector(self, v: np.ndarray) -> np.ndarray:
        """Alias for reduce(v)."""
        return self.reduce(v)

    def __call__(self, z: np.ndarray) -> np.ndarray:
        """Calling basis on reduced state reconstructs the full-order state."""
        return self.reconstruct(z)

    def __getitem__(self, item: Any) -> Any:
        return self.V[item]

    def __len__(self) -> int:
        return len(self.V)

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self.V, dtype=dtype)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', N={self.full_dim}, r={self.reduced_dim})"

    def __getstate__(self) -> dict:
        return {
            'V': self.V,
            'name': self.name,
            'metadata': self.metadata
        }

    def __setstate__(self, state: dict) -> None:
        self.V = np.asarray(state['V'])
        self.VT = self.V.T
        self.full_dim, self.reduced_dim = self.V.shape
        self.name = state.get('name', 'basis')
        self.metadata = state.get('metadata', {})


class IdentityBasis:
    """
    Zero-overhead identity projection operator for Full-Order Models (FOM).
    Guarantees O(1) zero-copy passes for reduce, reconstruct, and matrix projection.
    """

    def __init__(self, dim: int, name: str = "identity_basis", metadata: dict | None = None):
        self.dim = int(dim)
        self.name = name
        self.metadata = metadata if metadata is not None else {}

    @property
    def full_dim(self) -> int:
        return self.dim

    @property
    def reduced_dim(self) -> int:
        return self.dim

    @property
    def N(self) -> int:
        return self.dim

    @property
    def r(self) -> int:
        return self.dim

    @property
    def nosc(self) -> int:
        return self.dim // 2

    @property
    def nosc_r(self) -> int:
        return self.dim // 2

    @property
    def shape(self) -> tuple[int, int]:
        return (self.dim, self.dim)

    @property
    def ndim(self) -> int:
        return 2

    @property
    def is_identity(self) -> bool:
        return True

    @property
    def V(self) -> np.ndarray:
        return np.eye(self.dim)

    @property
    def VT(self) -> np.ndarray:
        return np.eye(self.dim)

    @property
    def T(self) -> np.ndarray:
        return self.VT

    def reduce(self, x: np.ndarray) -> np.ndarray:
        """Identity mapping (zero-copy pass-through)."""
        return x

    def reconstruct(self, z: np.ndarray) -> np.ndarray:
        """Identity mapping (zero-copy pass-through)."""
        return z

    def project_matrix(self, A: np.ndarray) -> np.ndarray:
        """Identity mapping (zero-copy pass-through)."""
        return A

    def project_dual_matrix(self, A: np.ndarray) -> np.ndarray:
        return A

    def project_vector(self, v: np.ndarray) -> np.ndarray:
        return v

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return z

    def __getitem__(self, item: Any) -> Any:
        return np.eye(self.dim)[item]

    def __len__(self) -> int:
        return self.dim

    def __array__(self, dtype=None) -> np.ndarray:
        return np.eye(self.dim, dtype=dtype)

    def __repr__(self) -> str:
        return f"IdentityBasis(name='{self.name}', dim={self.dim})"

    def __getstate__(self) -> dict:
        return {'dim': self.dim, 'name': self.name, 'metadata': self.metadata}

    def __setstate__(self, state: dict) -> None:
        self.dim = state['dim']
        self.name = state.get('name', 'identity_basis')
        self.metadata = state.get('metadata', {})


class SymplecticBasis(Basis):
    """
    Symplectic (canonical structure-preserving) reduced basis for Hamiltonian systems:
        V = block_diag(V_state, V_state)
    where y = [q, p]^T in R^{2*nosc} and z = [q_r, p_r]^T in R^{2*nosc_r}.
    """

    def __init__(self, V: np.ndarray | Sequence, V_state: np.ndarray | None = None,
                 name: str = "symplectic_basis", metadata: dict | None = None):
        super().__init__(V, name=name, metadata=metadata)
        if V_state is not None:
            self._V_state = np.asarray(V_state)
        else:
            self._V_state = self.V[:self.nosc, :self.nosc_r]

    @property
    def V_state(self) -> np.ndarray:
        """Unaugmented position/momentum kinematic basis matrix V_state in R^{nosc x nosc_r}."""
        return self._V_state

    @property
    def V_q(self) -> np.ndarray:
        """Position block V_q in R^{nosc x nosc_r}."""
        return self.V[:self.nosc, :self.nosc_r]

    @property
    def V_p(self) -> np.ndarray:
        """Momentum block V_p in R^{nosc x nosc_r}."""
        return self.V[self.nosc:, self.nosc_r:]

    @classmethod
    def from_state_basis(cls, V_state: np.ndarray, name: str = "symplectic_basis",
                          metadata: dict | None = None) -> SymplecticBasis:
        """Construct a 2*nosc x 2*nosc_r block-diagonal symplectic basis from nosc x nosc_r V_state."""
        V_state = np.asarray(V_state)
        zeros = np.zeros_like(V_state)
        V = np.block([[V_state, zeros], [zeros, V_state]])
        return cls(V, V_state=V_state, name=name, metadata=metadata)

    def __getstate__(self) -> dict:
        state = super().__getstate__()
        state['V_state'] = self._V_state
        return state

    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)
        self._V_state = state.get('V_state', self.V[:self.nosc, :self.nosc_r])


class TravelingBasis:
    """
    Manager for adaptive / windowed reduced bases in time-dependent MOR.
    Encapsulates a sequence of window bases [B_0, B_1, ..., B_{W-1}].
    """

    def __init__(self, bases: Sequence[Basis | np.ndarray],
                 boundary_fracs: Sequence[float] | None = None,
                 name: str = "traveling_basis", metadata: dict | None = None):
        self.bases = [b if isinstance(b, (Basis, IdentityBasis)) else Basis(b) for b in bases]
        self.boundary_fracs = list(boundary_fracs) if boundary_fracs is not None else None
        self.name = name
        self.metadata = metadata if metadata is not None else {}
        self._active_idx = 0

    @property
    def n_windows(self) -> int:
        return len(self.bases)

    @property
    def active_basis(self) -> Basis:
        return self.bases[self._active_idx]

    @property
    def active_window(self) -> int:
        return self._active_idx

    def activate_window(self, w: int) -> Basis:
        if not (0 <= w < len(self.bases)):
            raise IndexError(f"Window index {w} out of range [0, {len(self.bases)})")
        self._active_idx = w
        return self.bases[w]

    def window_for_step(self, step: int, total_steps: int) -> int:
        """Return the window index corresponding to step index k out of total_steps."""
        if self.boundary_fracs is not None and len(self.boundary_fracs) == len(self.bases) + 1:
            step_boundaries = [int(f * total_steps) for f in self.boundary_fracs]
            import bisect
            w = bisect.bisect_right(step_boundaries, step) - 1
            return max(0, min(w, len(self.bases) - 1))
        window_size = total_steps / len(self.bases)
        return max(0, min(int(step / window_size), len(self.bases) - 1))

    def reduce(self, x: np.ndarray) -> np.ndarray:
        return self.active_basis.reduce(x)

    def reconstruct(self, z: np.ndarray) -> np.ndarray:
        return self.active_basis.reconstruct(z)

    def project_matrix(self, A: np.ndarray) -> np.ndarray:
        return self.active_basis.project_matrix(A)

    def __getitem__(self, idx: int) -> Basis:
        return self.bases[idx]

    def __len__(self) -> int:
        return len(self.bases)

    def __iter__(self):
        return iter(self.bases)

    def __repr__(self) -> str:
        return f"TravelingBasis(n_windows={len(self.bases)}, active={self._active_idx})"

    def __getstate__(self) -> dict:
        return {
            'bases': self.bases,
            'boundary_fracs': self.boundary_fracs,
            'name': self.name,
            'metadata': self.metadata,
            'active_idx': self._active_idx
        }

    def __setstate__(self, state: dict) -> None:
        self.bases = state['bases']
        self.boundary_fracs = state.get('boundary_fracs')
        self.name = state.get('name', 'traveling_basis')
        self.metadata = state.get('metadata', {})
        self._active_idx = state.get('active_idx', 0)


class DEIMProjector:
    """
    Hyper-reduction interpolation projector:
        f_r = deim_field @ f_sampled
    where deim_field in R^{r x m} and indices in R^m.
    """

    def __init__(self, deim_field: np.ndarray, indices: np.ndarray | Sequence[int] | None = None,
                 name: str = "deim_projector", metadata: dict | None = None):
        self.deim_field = np.asarray(deim_field)
        self.indices = np.asarray(indices) if indices is not None else None
        self.name = name
        self.metadata = metadata if metadata is not None else {}

    @property
    def r(self) -> int:
        return self.deim_field.shape[0]

    @property
    def m(self) -> int:
        return self.deim_field.shape[1]

    def project(self, sampled_values: np.ndarray) -> np.ndarray:
        """Project sampled nonlinear evaluation into reduced space: f_r = deim_field @ f_sampled."""
        return self.deim_field @ sampled_values

    def __call__(self, sampled_values: np.ndarray) -> np.ndarray:
        return self.project(sampled_values)

    def __repr__(self) -> str:
        return f"DEIMProjector(r={self.r}, m={self.m})"

    def __getstate__(self) -> dict:
        return {
            'deim_field': self.deim_field,
            'indices': self.indices,
            'name': self.name,
            'metadata': self.metadata
        }

    def __setstate__(self, state: dict) -> None:
        self.deim_field = np.asarray(state['deim_field'])
        self.indices = np.asarray(state['indices']) if state.get('indices') is not None else None
        self.name = state.get('name', 'deim_projector')
        self.metadata = state.get('metadata', {})


def as_basis(obj: Any, default_dim: int | None = None) -> Basis | IdentityBasis:
    """
    Convert an object to a Basis or IdentityBasis.
    If already a Basis/IdentityBasis, returns it unchanged.
    If a 2D numpy array, wraps it in Basis(obj).
    If None and default_dim is provided, returns IdentityBasis(default_dim).
    """
    if isinstance(obj, (Basis, IdentityBasis, TravelingBasis)):
        return obj
    if obj is not None and hasattr(obj, 'shape') and len(obj.shape) == 2:
        return Basis(obj)
    if obj is None and default_dim is not None:
        return IdentityBasis(default_dim)
    raise TypeError(f"Cannot convert object of type {type(obj)} to Basis")
