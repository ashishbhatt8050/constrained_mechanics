# AI Agent Instructions for constrained_mechanics

## Project Overview
This is a computational physics research project implementing structure-preserving solvers for constrained mechanical systems using model order reduction (MOR). The codebase combines geometric integration methods (Discrete Gradient, Conformal Stormer-Verlet, Implicit Midpoint) with POD/DEIM hyper-reduction techniques.

**Key Goal:** Enable efficient simulation of constrained mechanical systems (e.g., helical lattice structures) through symbolic computation, reduced-order modeling, and parallel execution.

## Architecture Patterns

### 1. Hierarchical Class Structure (System.py)
The core architecture follows a strict inheritance hierarchy:
- **`SysConfig`**: Static configuration class with all simulation parameters (solvers, reduction settings, system properties)
  - `nosc`: number of oscillators (must be divisible by 6)
  - `w_values`: quadrature weights for splitting solver
  - `predict`/`train_ratio`: controls data generation vs. reproduction experiments
  - `pod_tol`, `hyperreducer`: MOR configuration
  - **Access pattern:** Parameters are class attributes, read at module load time for consistency across parallel workers

- **`MechSystem`** → **`HamiltonianMechSystem` / `LagrangianMechSystem`**
  - Inherits from `SysConfig` for config access
  - Implements system dynamics: Hamiltonian/Lagrangian, constraints, Jacobians
  - Static geometric data (helical positions/momenta) initialized once at class definition
  - **Key pattern:** Uses `@load_symbolic_expressions` decorator to lazy-load precomputed symbolic functions

- **`ReduceMechSystem`** → **`HamiltonianReducer` / `DiscreteGradientReducer`**
  - Extends MechSystem with POD/DEIM/MDEIM reduction methods
  - Stores reduction data: `RB` (reduced basis), `nosc_r` (reduced size), DEIM matrices
  - `_REDUCTION_ATTRS`: lists all attributes managed by reduction to ensure proper persistence

### 2. Solver Pattern (ConcreteSolvers.py & ODESolver.py)
- **Base class `ODESolver`**: Generic time-stepping framework with `solve()` loop
- **`BaseSolverMixin`**: Common solving logic with composition via `@compose_solver_solves` decorator
  - Handles multi-stage splitting using `w_values` (e.g., Stomer-Verlet substeps)
  - Manages constraint solving via Newton iteration (see `Newton.py`)
  - Pattern: `solve_for_w()` called multiple times per timestep for composition
  
- **Concrete solvers**: `DiscreteGradientSolver`, `ConformalStormerVerletSolver`, `ConformalImplicitMidpointSolver`
  - Each implements `advance()` or `__call__()` for one step
  - DiscreteGradient expects state pairs `(y_k, y_{k+1})`; others accept single state

### 3. Reduction Workflow (podDEIM.py & ReduceMechSystem.py)
Sequential stages (see `app8_lattice.py` main):
1. **Snapshot collection**: `compute_snapshots()` (static method with batch processing)
2. **POD**: `POD()` computes orthonormal reduced basis from weighted snapshots
3. **DEIM**: Builds interpolation matrix `P` and indices for nonlinear function reduction
4. **MDEIM**: Matrix version for reducing tensor operations (e.g., Jacobians)
- **Memory efficiency:** Dynamic batch sizing in `compute_snapshots()` to stay within 10% available RAM

### 4. Parallel Execution (app8_lattice.py)
- **SLURM detection**: Auto-detects `sbatch` availability; falls back to `dask.distributed.LocalCluster`
- **Top-level worker function `_dask_worker()`**: Must be module-level for cloudpickle serialization
- **Static method pattern:** `BaseSolverMixin.solve_mech_system()` enables pickling of solver objects
- **Environment isolation:** Thread count limited via `OMP_NUM_THREADS=1` to prevent nested parallelism

## Developer Workflows

### Running Simulations
```bash
# Standard (uses SLURM if available; LocalCluster on single machine)
python scripts/app8_lattice.py

# Force local execution (even on SLURM nodes)
python scripts/app8_lattice.py --no-latex

# Clear stale checkpoints before recomputing with new config
python scripts/app8_lattice.py --clean

# Regenerate symbolic expressions (slow, ~minutes for larger nosc)
python scripts/app8_lattice.py --clear-symbolic
```

### Configuration Workflow
**All changes to simulations start in [src/System.py](src/System.py) `SysConfig` class:**
- Modify `nosc`, `dt_space`, `T_final`, `pod_tol`, etc.
- `get_config_hash()` detects changes; triggers symbolic recomputation if needed
- Changes propagate automatically to all workers via class inheritance

### Adding New Solvers
1. Inherit from `ODESolver` + `BaseSolverMixin`
2. Implement `advance()` method (or override `__call__()`)
3. Register in `ConcreteSolvers.py` concrete class
4. Add to solver instantiation in `app8_lattice.py` main loop

## Project-Specific Conventions

### Symbolic Computation
- **Staged caching:** Symbolic expressions cached in `data/symbolic_expr_{nosc}_cse.pickle` using cloudpickle
- **On-demand loading:** `@load_symbolic_expressions` decorator loads cached expressions as static methods
- **CSE optimization:** SymPy Common Subexpression Elimination applied in `SymbolicComputer.py` for speed
- **Never modify** symbolic computation in classes during runtime—expressions are set at class definition time

### State Representation
- **Full system:** State `y` is flattened: `[q_1, ..., q_d, p_1, ..., p_d]` (positions, momenta)
- **Reduced system:** `z = y @ RB.T` (project onto reduced basis)
- **Discrete Gradient special case:** Works with state pairs `(y_k, y_{k+1})` for implicit stepping

### Data Persistence
- Solver objects pickled with cloudpickle (not standard pickle) due to lambda functions and nested classes
- Checkpoints stored in `data/{date}/` with per-parameter subdirectories
- `joblib_cache/` stores intermediate reduction computations (invalidated on config hash mismatch)

## Critical Integration Points

### Cross-Component Data Flow
```
app8_lattice.py (orchestrator)
  ↓
System.py (SysConfig, system dynamics, initial conditions)
  ↓
ReduceMechSystem.py (POD/DEIM basis computation)
  ↓
ConcreteSolvers.py (integrator + reduced solvers)
  ↓
SymbolicComputer.py (precomputed Hamiltonians, Jacobians)
```

### Multi-Oscillator Scaling
- System size grows as `2*nosc` (DOF for positions + momenta)
- Reduced basis size depends on `pod_tol` (typically `nosc_r ~ 10-20` for `nosc=54`)
- Helical geometry initialized once in `MechSystem` class vars; reused across all instances

### Constraint Handling
- Spherical constraints `g(q) = ||q_i - q_j||^2 - 1` enforced via Lagrange multipliers
- Multipliers solved iteratively in `fixed_point()` (Newton's method wrapper)
- Pattern: Each solver substep calls `self.fixed_point(y, Lambda)` to maintain constraints

## Common Pitfalls & Best Practices

1. **Configuration changes require symbolic recomputation:** Always use `--clear-symbolic` after changing `nosc` or system structure
2. **Nested parallelism kills performance:** Thread environment vars already set; don't override in worker code
3. **Pickling solver state:** Use `cloudpickle`, not `pickle`; lambda functions don't serialize with standard pickle
4. **Memory management:** Batch snapshot computation uses dynamic sizing—don't manually force large batches
5. **SLURM vs. local:** Script auto-detects; set `PYTEST_CURRENT_TEST` env var in tests to force LocalCluster
6. **Reduced basis orthonormality:** Always verify `RB @ RB.T ≈ I` after POD; use `np.allclose()` for numerical checks

## Quick Reference: Key File Locations
- **Configuration:** [src/System.py](src/System.py#L58-L120) (SysConfig class)
- **Solvers:** [src/ConcreteSolvers.py](src/ConcreteSolvers.py) + [src/ODESolver.py](src/ODESolver.py)
- **Reduction:** [src/ReduceMechSystem.py](src/ReduceMechSystem.py) + [src/podDEIM.py](src/podDEIM.py)
- **Symbolic computation:** [src/SymbolicComputer.py](src/SymbolicComputer.py)
- **Main script:** [scripts/app8_lattice.py](scripts/app8_lattice.py)
- **Plotting utilities:** [src/PlotScript.py](src/PlotScript.py)
