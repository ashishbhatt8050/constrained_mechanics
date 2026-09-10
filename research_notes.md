# Research Notes — Constrained Mechanics HRM Development

*Last updated: 2026-08-26*

---

## Problem Statement

The hyper-reduced models (HRMs) were failing to converge during time integration.
Root cause: a single global reduced basis (POD + DEIM) computed from all training
snapshots is forced to represent the entire solution manifold simultaneously. For
long or highly nonlinear trajectories the DEIM approximation becomes poor in
sub-intervals where the local dynamics differ significantly from the global average,
causing Newton / fixed-point iteration failure.

---

## Solution: Adaptive Traveling / Hierarchical Reduced Basis

Instead of one global basis for the full timeline, we compute a **family of local
bases** — one per adaptively-detected time window — and switch the active basis
(and its associated DEIM/MDEIM operators) at each window boundary during online
integration.

Each local basis is built **hierarchically** on top of the global basis, ensuring
a global-quality floor everywhere while adding window-specific enrichment.

---

## Implementation Overview

### Files Modified

| File | Role |
|---|---|
| `src/ReduceMechSystem.py` | Offline basis construction (all new logic lives here) |
| `src/ConcreteSolvers.py` | Online integration (basis-swap at window boundaries) |

---

### 1. Adaptive Window Detection  —  `_detect_adaptive_windows`

**Location:** `ReduceMechSystem` base class in `src/ReduceMechSystem.py`

**Purpose:** Determine *where* in the snapshot timeline the current basis becomes
inadequate and a new one should be opened.

**Algorithm (offline, snapshot-processing phase):**

1. Use the first training solver as the reference timeline.
2. Set `rolling_size = max(10, ROLLING_FRAC × n_snapshots)` (default: 5 %).
3. Seed the current-window basis `U_window` from the first rolling block via thin SVD.
4. At each subsequent rolling step:
   - Compute `U_roll` = thin SVD of the new block (fresh candidate basis).
   - Compute the **Grassmann distance** via the cross-Gram matrix:
     ```
     G       = U_window[:, :k].T @ U_roll[:, :k]
     σ_min   = minimum singular value of G
     ```
   - **Trigger** (new window): if `σ_min < GRASSMANN_COS_THRESHOLD` (= cos 26° ≈ 0.899)
     → record boundary, reset `U_window ← U_roll`.
   - **Extend** (same window): update `U_window` via incremental rank-updating SVD
     (QR of complement block + thin SVD of merged system), energy-truncated to `pod_tol`.
5. Apply the same fractional boundaries to all training solvers' index arrays.

**Tunable class attributes on `ReduceMechSystem`:**

| Attribute | Default | Meaning |
|---|---|---|
| `N_WINDOWS` | `5` | Maximum number of adaptive windows |
| `GRASSMANN_COS_THRESHOLD` | `0.8988` (cos 26°) | Misalignment trigger |
| `ROLLING_FRAC` | `0.05` | Fraction of snapshots per rolling step |

**Outputs stored on `cls`:**
- `_window_indices` — per-solver snapshot index arrays, one list per window
- `_boundary_fracs` — fractional timeline positions `[0, f1, …, 1]`

---

### 2. Hierarchical Basis Construction (per window)

**Location:** `HamiltonianReducer.setup_traveling_basis` and
`DiscreteGradientReducer.setup_traveling_basis`

**Per-window procedure:**

```
Step 1 — Local POD
  rb_local = incremental_POD(state snapshots in window w)
  Augmented with: ham_z or lag_dg snapshots (physics-specific)
  rb_local = energy-truncated to pod_tol

Step 2 — Hierarchical enrichment
  residual = rb_local − U_global @ (U_global.T @ rb_local)
           = component of rb_local orthogonal to the global basis

  Q_enrich = QR(residual)                       # orthonormal enrichment
  Q_enrich = Q_enrich[:, col_norm > 1e-8]       # drop near-zero columns

  rb_w = [U_global | Q_enrich]                  # global floor + local modes
  rb_w = QR(rb_w)                               # re-orthogonalise (safety)

Step 3 — Constraint-gradient augmentation
  while cond(G_constraint @ rb_w) > 10:
      append constraint-Jacobian columns from window snapshots
      update rb_w via incremental_POD + energy truncation

Step 4 — Block-diagonal assembly
  RB_w = [[rb_w,  0   ],
           [0,    rb_w ]]   # shape (2·nosc × 2·nosc_r_w)
```

**Stored as:** `cls.RB_windows[w]`, `cls.nosc_r_windows[w]`

---

### 3. Per-Window DEIM / MDEIM Hyper-Reduction

**Location:** `HamiltonianReducer.setup_traveling_hyperreduction` and
`DiscreteGradientReducer.setup_traveling_hyperreduction`

Window boundaries are read directly from `cls._window_indices` (set in step 2),
so basis and hyper-reduction always share the same partition.

**Per-window lists stored on `cls` and target classes:**

| Attribute | Content |
|---|---|
| `RBxUx_inv_PxU_windows[w]` | DEIM projection-interpolation matrix |
| `IP_Ux_inv_PxU_windows[w]` | MDEIM sparse projection matrix |
| `ham_z_deim_windows[w]` | Lambdified DEIM function for `ham_z` |
| `ham_zz_mdeim_windows[w]` | Lambdified MDEIM function for `ham_zz` |
| `lag_dg_deim_windows[w]` | Lambdified DEIM function for `lag_dg` |
| `lag_dg_z_mdeim_windows[w]` | Lambdified MDEIM function for `lag_dg_z` |

---

### 4. Online Basis-Switching  —  `solve_trajectory`

**Location:** `BaseSolverMixin.solve_trajectory` in `src/ConcreteSolvers.py`

```python
_window_size = self.n / n_windows          # steps per window (float)
new_window   = min(int(k / _window_size), n_windows - 1)

if new_window != _current_window:
    _activate_window(new_window)
```

`_activate_window(w)` swaps on `self` / `self.__class__`:
- `self.RB`, `self.nosc_r`
- `self.RBxUx_inv_PxU`, `self.IP_Ux_inv_PxU`
- DEIM/MDEIM lambdified functions (as `staticmethod`s)
- `self.JJ` (re-initialised for new `nosc_r` if PSD reducer)

**Overhead:** 2 integer comparisons per step; swap called ≤ `N_WINDOWS − 1` times
per full trajectory.

---

### 5. Orchestration (`setup_and_solve_hyperreduced_system`)

Extended call sequence when `N_WINDOWS > 1`:

```
setup_hyperreduction(...)            # global DEIM  (existing)
update_mdeim_hyperreduction(...)     # global MDEIM (existing)
hyperreduce_constraints(...)         # constraints  (existing)

# NEW:
setup_traveling_basis(...)           # detect windows + hierarchical bases
setup_traveling_hyperreduction(...)  # per-window DEIM/MDEIM operators

_transfer_reduction_attrs(kwds)      # serialise all *_windows lists to Dask
parallel_solve_mech_system(...)
```

`_REDUCTION_ATTRS` on `ReduceMechSystem` was extended with all `*_windows` list
attributes so they are pickled and scattered to workers correctly.

---

## Design Decisions & Rationale

| Decision | Rationale |
|---|---|
| Offline detection only | Window boundaries from FOM snapshots; no online monitoring overhead |
| Rolling-window alignment check | Block averaging is more robust than per-snapshot checks; avoids spurious triggers |
| Grassmann distance (σ_min) | Geometrically exact, rotation-invariant; cheap k×k SVD (k ≪ N_h) |
| Hierarchical enrichment | Global-quality floor in every window; prevents divergence in under-represented sub-intervals |
| Cap at 5 windows | Balances offline cost (≈ N× standard setup) vs. online benefit |
| cos 26° trigger | Moderate misalignment threshold; tunable via `GRASSMANN_COS_THRESHOLD` |
| `>= 1` activation guard in `solve_trajectory` | Single-window adaptive case still uses the hierarchical basis |

---

## Open Questions / Future Work

- [ ] Parallelise per-window basis construction (each window is independent offline).
- [ ] Expose `GRASSMANN_COS_THRESHOLD` and `ROLLING_FRAC` as config / CLI parameters in `app8_lattice.py`.
- [ ] Log the `σ_min` trace during detection so the threshold can be tuned post-hoc from data.
- [ ] Investigate constraint-specific window detection (constraint modes may misalign at different rates than state modes).
- [x] Evaluate replacing the inline incremental SVD update in `_detect_adaptive_windows` with a call to `incremental_POD` for pipeline consistency. *(Done 2026-09-10 — replaced; see update below)*
- [x] Remove `_reproduce/predict` subscript from resulting filenames.
---

## Update 2026-08-26 — Traveling Bases for Constraints

### Motivation

`g_prime` (constraint Jacobian) and `g_prime_x_lambda_y` (DG-only, constraint-force
matrix) are evaluated at every Newton iteration inside the integrator. Their MDEIM
approximation quality degrades across the timeline for the same reason as the
dynamics operators — a single global MDEIM basis is unable to represent the full
range of constraint-Jacobian variation. Adding per-window constraint MDEIM bases
eliminates this remaining source of approximation error.

### New components

#### `ReduceMechSystem.setup_traveling_constraint_hyperreduction`

Single classmethod on the base class — handles both Hamiltonian (g_prime only) and
DiscreteGradient (g_prime + g_prime_x_lambda_y) solver families.  Called after
`setup_traveling_basis` so it can reuse `cls._window_indices`.

**Per-window procedure:**
```
For each window w:
  g_prime MDEIM:
    1. Collect g_prime snapshots from all solvers in window w
       (projected onto non-zero indices of the symbolic expression)
    2. incremental_POD → energy-truncated basis Uj_gp
    3. DEIM(Uj_gp) → interpolation points Pj_gp
    4. IP_gp_w  = reconstruct_sparse_basis(Uj_gp @ inv(Pj_gp.T @ Uj_gp))
    5. col_gp_w = Pj_gp.T @ sym_selected_elements → lambdify

  g_prime_x_lambda_y MDEIM  [DG only — identical pipeline, different function]:
    snapshots via solver.g_prime_x_lambda_y_(y, Lambda)
    col lambdified with (cls.y, cls.lag_mult)
```

**Stored lists (on `cls` and each `target_class`):**

| Attribute | Content |
|---|---|
| `_IP_Ux_inv_PxU_windows[w]` | Sparse MDEIM projection for `g_prime` |
| `g_prime_mdeim_windows[w]` | Lambdified MDEIM function for `g_prime` |
| `IP_g_prime_x_lambda_y_windows[w]` | Sparse MDEIM projection for `g'·λ·y` (DG) |
| `g_prime_x_lambda_y_mdeim_windows[w]` | Lambdified MDEIM function for `g'·λ·y` (DG) |

All four added to `_REDUCTION_ATTRS` so they are serialised to Dask workers.

#### `_activate_window` (ConcreteSolvers.py)

Extended to also swap on window transition:
- `self._IP_Ux_inv_PxU` ← `self._IP_Ux_inv_PxU_windows[w]`
- `self.__class__.g_prime_mdeim` ← `self.g_prime_mdeim_windows[w]`
- `self.IP_g_prime_x_lambda_y` ← `self.IP_g_prime_x_lambda_y_windows[w]`  (DG)
- `self.__class__.g_prime_x_lambda_y_mdeim` ← `self.g_prime_x_lambda_y_mdeim_windows[w]`  (DG)

These feed directly into `g_prime_hyperreduced` and `g_prime_x_lambda_y_hyperreduced`
in `System.py`, which are the methods activated when `constraints_reduce = True`.

#### `setup_and_solve_hyperreduced_system` (ConcreteSolvers.py)

Adds two conditional calls (guarded by `constraints_reduce`):
```python
if ReduceMechSystem.constraints_reduce:
    HamiltonianReducer.setup_traveling_constraint_hyperreduction(ham_solvers, ham_classes)
    # or
    DiscreteGradientReducer.setup_traveling_constraint_hyperreduction(dg_solvers, dg_classes)
```


---

## Bug Fix 2026-09-10 — Online Window Lookup Was Using Equal Splits

### Finding (from run `data/2026-08-26/app8_lattice-7463.out`)

The adaptive window detector fired correctly offline, detecting 5 windows at
fractional boundaries `[0.000, 0.450, 0.550, 0.600, 0.700, 1.000]` from 3000 steps:

```
→ Window 2 at frac=0.450  (step 1350)
→ Window 3 at frac=0.550  (step 1650)
→ Window 4 at frac=0.600  (step 1800)
→ Window 5 at frac=0.700  (step 2100)
```

However, the online solver only ever printed "Activated window **1**/5 at step k=0"
and never switched. Root cause: the per-step window index was computed as

```python
new_window = min(int(k / _window_size), _n_windows - 1)
# where _window_size = self.n / n_windows = 3000 / 5 = 600  ← EQUAL SPLIT
```

This creates equal windows of 600 steps each, completely misaligned with the
adaptive boundaries. Window 2 would have been activated at step 600 (frac 0.20),
but the offline basis for window 2 covers frac 0.45–0.55. All 5 bases were built
for the right snapshot intervals, but the solver was using the wrong one at every
point in time.

Furthermore, `_boundary_fracs` was never in `_REDUCTION_ATTRS`, so it was never
serialised to Dask workers at all — worker instances had no access to the adaptive
boundaries.

### Fix

**`ReduceMechSystem._REDUCTION_ATTRS`**: added `'_boundary_fracs'` so the
fractional boundary list is serialised to every Dask worker alongside `RB_windows`.

**`BaseSolverMixin.solve_trajectory`**: replaced the equal-split formula with
a `bisect_right` lookup over the converted step boundaries:

```python
_step_boundaries = [int(f * self.n) for f in self._boundary_fracs]

def _window_for_step(k):
    w = bisect.bisect_right(_step_boundaries, k) - 1
    return max(0, min(w, n_windows - 1))
```

`_step_boundaries` is computed once before the time-stepping loop (O(n_windows)).
Each per-step call is O(log n_windows) ≈ O(3) — negligible.

A fallback to equal-size windows is retained for backward compatibility with
checkpoints saved before this fix.


---

## Update 2026-09-10 — Replaced Inline SVD with `incremental_POD` in `_detect_adaptive_windows`

### Evaluation

The inline update (lines 376–386 in the original) implemented Brand's incremental
SVD using `[[I, D], [0, R]]` as the small intermediate matrix `M`, where `I` is
the identity. This is correct only if the existing basis `U_window` has uniform
singular-value weighting, i.e. all columns are equally important. That is true for
the very first block (fresh SVD from seed data), but after any extend step the
columns of `U_window` are a mixture of high-energy and low-energy directions —
identity weighting will overcorrect toward the new block.

`incremental_POD` uses `[[diag(S), D], [0, R]]` as `M`, threading the current
singular values through every update. This correctly down-weights low-energy
accumulated modes relative to strong new data, exactly as the full SVD would.

**Additional issues with the old inline code:**

- `_rolling_basis` pre-compressed the block to a basis before passing it in.
  This discards within-block singular value information before the merge.
  The `incremental_POD` path passes the raw `(nosc, rolling_size)` block directly,
  so `diag(S_window)` is compared against *full-rank* new data.

- `_rolling_basis` called `_rolling_block` twice in the "else" branch
  (`D = U_window.T @ _rolling_basis(...)` and `E = _rolling_basis(...) - ...`)
  — two redundant identical matrix assemblies per loop iteration.

### What was changed (`_detect_adaptive_windows`)

1. **`_rolling_basis` → `_rolling_block`**: returns the raw `(nosc, rolling_size)`
   snapshot matrix instead of a pre-compressed basis.

2. **Seed**: `U_window, S_window` now come from a single `incremental_POD` call
   on the first block.

3. **Per-iteration `U_roll`**: computed via `incremental_POD([block])`, capturing
   `S_roll` in the same call (reused if a new window is triggered).

4. **Trigger branch**: `U_window, S_window = U_roll, S_roll` — no re-computation.

5. **Extend branch**: one `incremental_POD([raw_block], initial_basis=(U_window, S_window))`
   call, replacing the 12-line inline QR+SVD block.

### Call count per loop iteration

| Path | Old | New |
|---|---|---|
| Trigger (misaligned) | `_rolling_basis` ×1 + inline SVD skip | `incremental_POD` ×1 (shared with Grassmann check) |
| Extend (aligned) | `_rolling_basis` ×3 + inline QR + SVD | `incremental_POD` ×2 (one for Grassmann, one for update) |

The Grassmann check `incremental_POD([block])` and the update `incremental_POD([block], initial_basis=...)` both materialise the same raw block — they could be merged into one call, but that would require separating the Grassmann check from the basis update, reducing clarity for minimal gain since `rolling_size` ≪ `nosc`.
