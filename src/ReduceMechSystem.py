import os
import gc
import numpy as np
import sympy as smp
import psutil
from scipy import sparse
from itertools import chain
from numpy import linalg as LA
from scipy.linalg import sqrtm
from functools import partial
from tqdm.auto import tqdm
from System import MechSystem, HamiltonianMechSystem, LagrangianMechSystem
from ODESolver import DiscreteGradient
from podDEIM import POD, PSD, DEIM, rb_svd
from PlotScript import logplot, save_figure
from SymbolicComputer import IndexedBaseSymbolicComputer

class ReduceMechSystem(MechSystem):
    """
    Reducer class extending MechSystem with reduction and hyper-reduction capabilities.
    
    Provides methods for computing reduced bases (POD), DEIM/MDEIM bases, 
    and setting up reduced/hyper-reduced models.
    """
    _REDUCTION_ATTRS = [
        # Reduced Basis
        'RB', 'nosc_r',
        # DEIM attributes
        'RBxUx_inv_PxU',
        # MDEIM attributes for system dynamics
        'IP_Ux_inv_PxU',
        # MDEIM attributes for constraints
        '_IP_Ux_inv_PxU',
        'IP_g_prime_x_lambda_y',
        # Lambdified functions (DEIM)
        'ham_z_deim', 'ham_zz_deim', 'lag_dg_deim', 'lag_dg_z_deim',
        # Lambdified functions (MDEIM)
        'ham_zz_mdeim', 'lag_dg_z_mdeim', 'g_prime_mdeim',
        'g_prime_x_lambda_y_mdeim',
        # Shape attributes for reshaping
        'g_prime_shape', 'g_prime_x_lambda_y_shape',
        # Cached Master SVD results for sweeps
        '_Full_rb', '_Full_sv',
        '_Full_U1', '_Full_S1', '_Full_U2', '_Full_S2',
        '_Full_Uj', '_Full_Sj'
    ]

    @staticmethod
    def compute_snapshots(solver, indices, func, is_dg=False, is_mdeim_gprime=False, projection_indices=None, batch_size=None, yield_batches=False):
        """
        Generic method to compute snapshots or projected snapshots in batches.
        """
        # Determine output dimensionality and memory usage per sample
        idx0 = indices[0]
        if is_dg:
            y_data = solver.y if hasattr(solver, 'y') else solver['y']
            sample_res = func((y_data[idx0], y_data[idx0+1]))
        elif is_mdeim_gprime:
            y_data = solver.y if hasattr(solver, 'y') else solver['y']
            lambda_data = solver.Lambda if hasattr(solver, 'Lambda') else solver['Lambda']
            sample_res = func(y_data[idx0], lambda_data[idx0])
        else:
            y_data = solver.y if hasattr(solver, 'y') else solver['y']
            sample_res = func(y_data[idx0])

        Nh_out = len(projection_indices) if projection_indices is not None else sample_res.size

        # Determine batch size dynamically if not provided
        if batch_size is None:
            # Estimate item size (raw result + flattened output buffer)
            item_size = sample_res.nbytes
            if projection_indices is not None:
                item_size += Nh_out * 4 # float32 buffer

            # Get available memory
            mem = psutil.virtual_memory()
            available_mem = mem.available

            # Check for CGroup memory limits (Docker/SLURM) to avoid OOM in constrained environments
            try:
                cgroup_limit = None
                # CGroup v2
                if os.path.exists('/sys/fs/cgroup/memory.max'):
                    with open('/sys/fs/cgroup/memory.max', 'r') as f:
                        val = f.read().strip()
                        if val != 'max':
                            cgroup_limit = int(val)
                            if os.path.exists('/sys/fs/cgroup/memory.current'):
                                with open('/sys/fs/cgroup/memory.current', 'r') as f:
                                    available_mem = min(available_mem, max(0, cgroup_limit - int(f.read().strip())))
                # CGroup v1
                elif os.path.exists('/sys/fs/cgroup/memory/memory.limit_in_bytes'):
                    with open('/sys/fs/cgroup/memory/memory.limit_in_bytes', 'r') as f:
                        cgroup_limit = int(f.read().strip())
                        # Filter out "unlimited" values (often very large integers)
                        if cgroup_limit < 1e15: 
                            if os.path.exists('/sys/fs/cgroup/memory/memory.usage_in_bytes'):
                                with open('/sys/fs/cgroup/memory/memory.usage_in_bytes', 'r') as f:
                                    available_mem = min(available_mem, max(0, cgroup_limit - int(f.read().strip())))
            except Exception:
                pass # Fallback to psutil
            
            # Use at most 10% of available memory for the batch buffer
            target_mem_limit = (available_mem * 0.1)
            
            batch_size = int(target_mem_limit / item_size)
            batch_size = max(1, min(batch_size, 10000)) # Cap for small data, memory limit handles large data

        def process_batch(batch_indices):
            # Pre-allocate output buffer to avoid list materialization and multiple large copies
            batch_buffer = np.empty((len(batch_indices), Nh_out), dtype=np.float32)
            
            for i, idx in enumerate(batch_indices):
                if is_dg:
                    res = func((solver.y[idx], solver.y[idx+1]))
                elif is_mdeim_gprime:
                    res = func(solver.y[idx], solver.Lambda[idx])
                else:
                    res = func(solver.y[idx])
                
                if projection_indices is not None:
                    # Project per result immediately to minimize peak memory
                    batch_buffer[i] = res.ravel()[projection_indices]
                else:
                    batch_buffer[i] = res.ravel()
                    
            return batch_buffer.T

        batches = [indices[i:i+batch_size] for i in range(0, len(indices), batch_size)]
        
        def _generator():
            # Process batches serially to align with incremental_POD's serial consumption
            for batch_indices in tqdm(batches, desc=f"Computing snapshots (serially)", leave=False):
                batch_res = process_batch(batch_indices)
                yield batch_res
                # Trigger GC if batch is substantial
                if batch_res.nbytes > 1e8: # ~100MB
                    del batch_res
                    gc.collect()

        if yield_batches:
            return _generator()

        # If not yielding batches, compute all and hstack (original behavior)
        return np.hstack(list(_generator()))

    @staticmethod
    def _reconstruct_sparse_basis(B_hat, non_zero_indices, full_rows):
        """
        Reconstructs a full sparse basis matrix from the DEIM reduced basis B_hat.
        Places rows of B_hat into the specified non_zero_indices of the full matrix.
        Using COO format for memory efficiency and speed.
        """
        n_rows_sub, n_cols = B_hat.shape
        
        # Construct COO coordinates
        # B_hat.flatten() is row-major: we repeat row indices and tile column indices
        rows = np.repeat(non_zero_indices, n_cols)
        cols = np.tile(np.arange(n_cols), n_rows_sub)
        data = B_hat.flatten()
        
        return sparse.coo_matrix((data, (rows, cols)), shape=(full_rows, n_cols)).tocsr()

    @classmethod
    def _create_indexed_deim_func(cls, expr, P, solver_type):
        """
        Helper method to create a lambdified function using IndexedBase symbols for efficiency.
        Uses IndexedBaseSymbolicComputer's hybrid lambdification.
        """
        computer = IndexedBaseSymbolicComputer(cls.nosc)
        
        # Project the expression: P.T @ expr
        projected_expr = smp.Matrix(P.T @ expr.flat())

        y_symbols = set(cls.y)
        is_y_only = all(s in y_symbols for s in expr.free_symbols)

        if is_y_only:
            args = (computer.y_base,)
        else:
            args = (
                (computer.y_base, computer.omega2_base, computer.beta)
                if solver_type == "Hamiltonian"
                else (computer.y_base, computer.y1_base, computer.omega2_base)
            )
            
        return computer._lambdify_hybrid(args, projected_expr)

    @staticmethod
    def _truncate_basis(U_full, S_full, tol):
        """Helper to truncate a basis based on singular value energy."""
        energy = np.cumsum(S_full**2) / np.sum(S_full**2)
        N = np.searchsorted(energy, 1 - tol, side='left') + 1
        
        # Ensure N is even (for complex-valued or paired DOF systems)
        if N % 2 == 1: N += 1
        
        # Bound N to max available
        N = min(N, U_full.shape[1])
        
        return U_full[:, :N], S_full[:N], N

    @classmethod
    def hyperreduce_constraints(cls, solvers, target_classes):
        solver_type = "Hamiltonian" if issubclass(cls, HamiltonianMechSystem) else "DiscreteGradient"
        pod_tol = cls.pod_tol_sweep[-1]
        print('Computing constraints reduction in parallel...')
        
        # Map solvers to indices
        solver_to_indices = {solver: indices for solver, indices in zip(solvers, cls.indices_list)}
        filtered_indices = [solver_to_indices[solver] for solver in solvers]

        def compute_constraint_basis(func_name, expr, is_g_prime=False):
            print(f'Computing {func_name} reduction...', flush=True)

            # Determine if this is a "prime" type (g_prime, g_prime_x_lambda_y)
            prime_types = ['g_prime_', 'g_prime_x_lambda_y_']
            is_prime_type = is_g_prime or func_name in prime_types

            if is_prime_type:
                # --- Get shape for the prime constraint ---
                if func_name == 'g_prime_x_lambda_y_':
                    actual_shape = solvers[0].g_prime_x_lambda_y(solvers[0].y[0], solvers[0].Lambda[0]).shape
                else: # This handles 'g_prime_'
                    actual_shape = solvers[0].g_prime__(solvers[0].y[0]).shape

                # Synchronize symbolic expression with sparsified solver output
                if actual_shape[0] != expr.shape[0] or actual_shape[1] != expr.shape[1]:
                    m_h_sym = (expr.shape[0] // 2)
                    m_h_act = (actual_shape[0] // 2)
                    
                    idx_h = np.linspace(0, m_h_sym - 1, m_h_act, dtype=int)
                    mask = np.concatenate([idx_h, idx_h + m_h_sym])
                    
                    expr = expr.extract(mask, range(expr.shape[1]))
                
                g_prime_shape = expr.shape
                for target_class in target_classes:
                    setattr(target_class, func_name + 'shape', g_prime_shape)

                # Recalculate non-zero indices based on the (potentially sliced) expression
                non_zero_indices = np.where(np.array(expr.tolist()).flatten() != 0)[0]

                # --- Create snapshot matrix for prime constraints ---
                results_generators = []
                for solver, indices in tqdm(zip(solvers, filtered_indices), total=len(solvers), desc=f"Snapshots for {func_name}"):
                    # Select the function to call based on func_name
                    if func_name == 'g_prime_x_lambda_y_':
                        snapshot_func = solver.g_prime_x_lambda_y_
                        gen = cls.compute_snapshots(solver, indices, snapshot_func,
                                                      is_mdeim_gprime=True,
                                                      projection_indices=non_zero_indices,
                                                      yield_batches=True)
                    else:  # This handles 'g_prime_'
                        snapshot_func = solver.g_prime__
                        gen = cls.compute_snapshots(solver, indices, snapshot_func,
                                                      projection_indices=non_zero_indices,
                                                      yield_batches=True)
                    results_generators.append(gen)
                
                all_snapshot_blocks = chain.from_iterable(results_generators)
                Uj, sv, _ = cls.incremental_POD(all_snapshot_blocks, tol=pod_tol)
                del results_generators, all_snapshot_blocks
                gc.collect()
            else:
                # Create snapshot matrix for g
                F = np.hstack([
                    np.array([solver.g(y) for y in solver.y[indices]]).T
                    for solver, indices in zip(solvers, filtered_indices)
                ])
                Uj, sv, _ = POD(F, np.eye(F.shape[0]), pod_tol)
                del F

            # Compute DEIM points and interpolation matrix
            Pj, _ = DEIM(Uj, plot_deim=False)

            if is_prime_type:
                # Reconstruct the full sparse basis matrix from the DEIM results
                B_hat = Uj @ LA.inv(Pj.T @ Uj)
                total_elements = np.prod(g_prime_shape)

                '''
                # --- Verification Test ---
                # Compare the new `_reconstruct_sparse_basis` (COO-based) against a
                # memory-safe equivalent of the original `IP.T @ B_hat` operation, which uses LIL format.
                # 1. Generate basis using the LIL matrix approach
                basis_lil = sparse.lil_matrix((total_elements, B_hat.shape[1]), dtype=np.float64)
                basis_lil[non_zero_indices, :] = B_hat
                
                # 2. Generate basis using the new COO-based method
                basis_coo = cls._reconstruct_sparse_basis(B_hat, non_zero_indices, total_elements)
                
                # 3. Compare the two sparse matrices. The difference should be negligible.
                diff = abs(basis_lil.tocsr() - basis_coo)
                assert diff.max() < 1e-14, "Verification failed: COO reconstruction does not match LIL reconstruction."
                print(f"Verification for '{func_name}' successful: _reconstruct_sparse_basis matches the LIL-based method.")
                # --- End Verification Test ---
                '''

                basis = cls._reconstruct_sparse_basis(B_hat, non_zero_indices, total_elements)

                # Create lambdified function
                flat_expr = expr.flat()
                selected_expr_elements = [flat_expr[i] for i in non_zero_indices]
                col = Pj.T @ smp.Matrix(selected_expr_elements)
            else:
                basis = Uj @ LA.inv(Pj.T @ Uj)
                col = Pj.T @ expr

            print(f'{basis.shape = }')

            # Create and wrap lambdified function
            # Handle argument signature for g_prime_x_lambda_y
            if func_name == 'g_prime_x_lambda_y_':
                mdeim_func = lambda *args: smp.lambdify((cls.y, cls.lag_mult), col, modules=['numpy', 'scipy'])(*args).flatten()
            elif func_name in ["g_"]:
                # mdeim_func = smp.lambdify((cls.y,), col, modules=['scipy'])
                mdeim_func = cls._create_indexed_deim_func(expr, Pj, solver_type)
            else:
                mdeim_func = lambda *args: smp.lambdify((cls.y,), col, modules=['numpy', 'scipy'])(*args).flatten()

            if callable(mdeim_func) and not isinstance(mdeim_func, type):
                # Set attributes based on solver type
                attr_name = f'{func_name}{"mdeim" if is_prime_type else "deim"}'
                if solver_type == "Hamiltonian":
                    for target_class in target_classes:
                        setattr(target_class, attr_name, staticmethod(mdeim_func))
                else:  # DiscreteGradient
                    for target_class in target_classes:
                        setattr(target_class, attr_name, staticmethod(mdeim_func))
            else:
                print(f"Memory address of {func_name}{'mdeim' if is_prime_type else 'deim'}: {hex(id(mdeim_func))}")

            print(f'Finished {func_name} reduction.', flush=True)
            return func_name, basis, sv

        # Prepare tasks for parallel execution
        tasks = [('g_prime_', cls.g_prime_expr, True)]
        if solver_type == "DiscreteGradient":
            tasks.append(("g_prime_x_lambda_y_", cls.g_prime_x_lambda_y_expr, True))

        results = {}
        for task in tasks:
            func_name, basis, sv = compute_constraint_basis(*task)
            results[func_name] = (basis, sv)

        # Retrieve results and set attributes
        _IP_Ux_inv_PxU, sv_g_prime = results['g_prime_']
        
        if solver_type == "Hamiltonian":
            for target_class in target_classes:
                setattr(target_class, '_IP_Ux_inv_PxU', _IP_Ux_inv_PxU)
        elif solver_type == "DiscreteGradient":
            IP_g_prime_x_lambda_y, sv_g_prime_x_lambda_y = results['g_prime_x_lambda_y_']

            for target_class in target_classes:
                setattr(target_class, '_IP_Ux_inv_PxU', _IP_Ux_inv_PxU)
                setattr(target_class, 'IP_g_prime_x_lambda_y', IP_g_prime_x_lambda_y)
        else:
            raise ValueError(f"Invalid solver type: {solver_type}")

        # Plot singular values for g_prime and g_prime_x_lambda_y using logplot
        sv_list = [sv_g_prime]
        xlims_list = [(1, len(sv_g_prime))]

        # Add additional singular values if solver type is DiscreteGradient
        if solver_type == "DiscreteGradient":
            sv_list.append(sv_g_prime_x_lambda_y)
            xlims_list.extend(
                [(1, len(sv_g_prime_x_lambda_y))]
            )

        fig, ax = logplot(sv_list, xlabel=f"index of singular values", xlims=xlims_list)
        filename = os.path.join(
            MechSystem.data_folder, "sv_constraints_" + solver_type.replace("Hamiltonian", "H").replace("DiscreteGradient", "DG") + ".pdf"
        )
        save_figure(fig, filename, fig_data=None)

        print('Constraints reduction complete.')

    @staticmethod
    def batched_POD(snapshot_blocks, tol):
        """
        Compute POD using batched method of snapshots to save memory.
        Avoids constructing the full snapshot matrix.
        """
        # Check for Dask arrays
        try:
            import dask.array as da
            has_dask = True
        except ImportError:
            has_dask = False

        is_dask_input = has_dask and any(isinstance(b, da.Array) for b in snapshot_blocks)

        if is_dask_input:
            # Concatenate into a single Dask array
            # Ensure all blocks are dask arrays
            S = da.concatenate([da.from_array(b) if not isinstance(b, da.Array) else b 
                                for b in snapshot_blocks], axis=1)
            
            Nh, total_ns = S.shape
            print(f"Computing POD with Dask for matrix of shape ({Nh}, {total_ns})...")

            # 1. Compute Gramian C = S.T @ S
            # This is a lazy operation. .compute() triggers execution.
            C = (S.T @ S).compute()
            
            # 2. Solve Eigenproblem (local numpy operation)
            Sigma2, Psi = LA.eigh(C)
            
            # Sort and filter
            Sigma2, Psi = np.flip(Sigma2), np.flip(Psi, axis=1)
            mask = Sigma2 > 1e-14
            Sigma2 = Sigma2[mask]
            Psi = Psi[:, mask]
            
            # Truncate
            if tol is not None:
                energy = np.cumsum(Sigma2) / np.sum(Sigma2)
                N = np.searchsorted(energy, 1 - tol) + 1
                if N % 2 == 1: N += 1
            else:
                N = len(Sigma2)
                
            Sigma2 = Sigma2[:N]
            Psi = Psi[:, :N]
            
            # 3. Compute Basis U = S @ Psi / sqrt(Sigma2)
            # Convert Psi to Dask array for distributed multiplication
            Psi_da = da.from_array(Psi, chunks=(Psi.shape[0], N))
            U_lazy = (S @ Psi_da) / np.sqrt(Sigma2)
            return U_lazy.compute(), np.sqrt(Sigma2), N

        # Calculate dimensions
        ns_list = [block.shape[1] for block in snapshot_blocks]
        total_ns = sum(ns_list)
        Nh = snapshot_blocks[0].shape[0]
        
        print(f"Computing batched POD for matrix of shape ({Nh}, {total_ns})...")

        # 1. Compute Gramian C = S.T @ S
        C = np.zeros((total_ns, total_ns))
        
        current_row = 0
        for i, block_i in enumerate(snapshot_blocks):
            current_col = 0
            for j, block_j in enumerate(snapshot_blocks):
                if j < i: 
                    current_col += ns_list[j]
                    continue # Symmetric
                
                # Compute block C_ij = S_i.T @ S_j
                sub_C = block_i.T @ block_j
                
                C[current_row:current_row+ns_list[i], current_col:current_col+ns_list[j]] = sub_C
                if i != j:
                    C[current_col:current_col+ns_list[j], current_row:current_row+ns_list[i]] = sub_C.T
                
                current_col += ns_list[j]
            current_row += ns_list[i]
            
        # 2. Solve Eigenproblem
        Sigma2, Psi = LA.eigh(C)
        
        # Sort and filter
        Sigma2, Psi = np.flip(Sigma2), np.flip(Psi, axis=1)
        mask = Sigma2 > 1e-14
        Sigma2 = Sigma2[mask]
        Psi = Psi[:, mask]
        
        # Truncate
        if tol is not None:
            energy = np.cumsum(Sigma2) / np.sum(Sigma2)
            N = np.searchsorted(energy, 1 - tol) + 1
            if N % 2 == 1: N += 1
        else:
            N = len(Sigma2)
            
        Sigma2 = Sigma2[:N]
        Psi = Psi[:, :N]
        
        # 3. Compute Basis U = S @ Psi / sqrt(Sigma2)
        U = np.zeros((Nh, N))
        
        current_idx = 0
        for i, block in enumerate(snapshot_blocks):
            ns_i = ns_list[i]
            Psi_block = Psi[current_idx:current_idx+ns_i, :]
            U += block @ Psi_block
            current_idx += ns_i
            
        U = U / np.sqrt(Sigma2)
        
        return U, np.sqrt(Sigma2), N

    @staticmethod
    def incremental_POD(snapshot_blocks, Xh=None, max_basis_size=None, tol=None, initial_basis=(None, None)):
        """
        Compute POD using Incremental SVD (uncentered) to handle extremely large datasets.
        
        Unlike sklearn.decomposition.IncrementalPCA, this does NOT center the data,
        preserving the physical state origin required for POD.
        
        Parameters:
        snapshot_blocks: Iterable of snapshot matrices (arrays of shape (Nh, batch_size)).
        Xh: Weighting matrix for the POD inner product.
        max_basis_size: Maximum rank of the basis to maintain.
        tol: Tolerance for singular value truncation (energy criteria).
        initial_basis (tuple, optional): A tuple (U, S) of a pre-existing orthonormal
                                         basis and its singular values to start from.
                                         Defaults to (None, None).
        
        Returns:
        U: POD modes (Nh, k)
        S: Singular values (k,)
        k: Rank of the basis
        """
        # Handle weighting matrix for consistency with POD()
        if Xh is not None:
            if np.allclose(Xh, np.eye(Xh.shape[0])):
                Xh_half = np.eye(Xh.shape[0])
            else:
                Xh_half = sqrtm(Xh)
        else:
            Xh_half = None

        U, S = initial_basis
        
        # Transform initial basis to weighted space if necessary
        if U is not None and Xh_half is not None:
            U = Xh_half @ U
        
        for i, block in enumerate(snapshot_blocks):
            # Handle Dask arrays
            if hasattr(block, 'compute'):
                block = block.compute()
            
            # Ensure block is 2D
            if block.ndim == 1:
                block = block.reshape(-1, 1)
                
            B = block
            
            if Xh_half is not None:
                B = Xh_half @ B
            
            if U is None:
                # Initialize with first batch
                U_batch, S_batch, _ = LA.svd(B, full_matrices=False)
                
                # Truncate
                k = len(S_batch)
                if max_basis_size is not None:
                    k = min(k, max_basis_size)
                
                if tol is not None:
                    energy = np.cumsum(S_batch**2) / np.sum(S_batch**2)
                    k_tol = np.searchsorted(energy, 1 - tol) + 1
                    k = min(k, k_tol)
                
                U = U_batch[:, :k]
                S = S_batch[:k]
            else:
                # Incremental update (Brand 2002 / Ross et al. 2008)
                # 1. Project new data onto current basis
                D = U.T @ B
                
                # 2. Compute residual
                E = B - U @ D
                
                # 3. QR decomposition of residual
                Q, R = LA.qr(E)
                
                # 4. Form the intermediate matrix M
                # M = [ diag(S)   D ]
                #     [    0      R ]
                k_curr = len(S)
                r_rows = R.shape[0]
                
                Z = np.zeros((r_rows, k_curr))
                S_mat = np.diag(S)
                
                M = np.block([
                    [S_mat, D],
                    [Z,     R]
                ])
                
                # 5. SVD of small matrix M
                Ub, Sb, _ = LA.svd(M, full_matrices=False)
                
                # 6. Truncate and Update
                k_new = len(Sb)
                if max_basis_size is not None:
                    k_new = min(k_new, max_basis_size)
                    
                if tol is not None:
                    energy = np.cumsum(Sb**2) / np.sum(Sb**2)
                    k_tol = np.searchsorted(energy, 1 - tol) + 1
                    k_new = min(k_new, k_tol)
                
                Sb = Sb[:k_new]
                Ub = Ub[:, :k_new]
                
                # Update U efficiently: U_new = U @ Ub[:k_curr, :] + Q @ Ub[k_curr:, :]
                U = U @ Ub[:k_curr, :] + Q @ Ub[k_curr:, :]
                S = Sb
            
        # Transform basis back to physical space: U = Xh^{-1/2} @ U_weighted
        if U is not None and Xh_half is not None:
            U = LA.solve(Xh_half, U)
            
        return U, S, len(S)

class HamiltonianReducer(ReduceMechSystem, HamiltonianMechSystem):
    """Reducer for Hamiltonian solvers."""

    @classmethod
    def setup_reduced_model(cls, solvers, target_classes):
        print(f'Setting up reduced model ({cls.__name__})...')
        tol = cls.pod_tol_sweep[-1]

        # Map solvers to indices (always needed)
        solver_to_indices = {solver: indices for solver, indices in zip(solvers, cls.indices_list)}
        filtered_indices = [solver_to_indices[solver] for solver in solvers]

        # Test condition on the first snapshot (needed for augmentation and final reporting)
        sample_y = solvers[0].y[0]
        g_prime_shape0 = cls.g_prime_(sample_y).shape[0] // 2
        test_q = sample_y[:cls.nosc]
        G_test = cls.g_prime_(np.concatenate([test_q, np.zeros(cls.nosc)]))[:g_prime_shape0, :cls.nosc]

        if not hasattr(cls, '_Full_rb') or cls._Full_rb is None:
            print(f"  [{cls.__name__}] Master SVD not found. Computing from snapshots...")
            # 1. Initial POD on state snapshots
            def get_state_snapshots():
                for solver, indices in zip(solvers, filtered_indices):
                    yield from cls.compute_snapshots(solver, indices, lambda y: y[:cls.nosc], yield_batches=True)
                for solver, indices in zip(solvers, filtered_indices):
                    yield from cls.compute_snapshots(solver, indices, lambda y: y[cls.nosc:], yield_batches=True)

            weights = np.eye(cls.nosc)
            # Use the final sweep tolerance to capture a "Master Basis"
            rb, _sv, _ = cls.incremental_POD(get_state_snapshots(), Xh=weights, tol=cls.pod_tol_sweep[-1])

            # 2. Increment with weighted snapshots of ham_z
            print("Incrementing basis with weighted ham_z snapshots...")
            def get_f2_snapshots():
                for solver, indices in zip(solvers, filtered_indices):
                    batch_f2 = cls.compute_snapshots(solver, indices, solver.ham_z, is_dg=False)
                    batch_f2_stacked = np.hstack([batch_f2[:cls.nosc, :], batch_f2[cls.nosc:, :]])
                    weight_ratio = (LA.norm(solver.y[indices]) / LA.norm(batch_f2)) if LA.norm(batch_f2) > 1e-12 else 1.0
                    yield batch_f2_stacked * weight_ratio

            # Combine initial POD and ham_z increment
            current_rb, current_sv, _ = cls.incremental_POD(get_f2_snapshots(), Xh=weights, tol=cls.pod_tol_sweep[-1], initial_basis=(rb, _sv))

            # 3. Iteratively augment with g_prime gradients until condition number is O(1)
            print("Iteratively augmenting with constraint gradients...")
            # Flatten all available time points for candidates
            candidates = []
            for solver, indices in zip(solvers, filtered_indices):
                for idx in indices:
                    candidates.append((solver, idx))
            
            current_cand_idx = 0
            cond_val = LA.cond(G_test @ current_rb)
            print(f"Initial condition number: {cond_val:.2e}")
            
            while cond_val > 10.0 and current_cand_idx < len(candidates):
                # Process gradients in small batches for efficiency
                batch_grads = []
                batch_norms_y = 0
                for _ in range(min(50, len(candidates) - current_cand_idx)):
                    s, i = candidates[current_cand_idx]
                    q = s.y[i][:cls.nosc]
                    G = cls.g_prime_(np.concatenate([q, np.zeros(cls.nosc)]))[:g_prime_shape0, :cls.nosc]
                    batch_grads.append(G.T)
                    batch_norms_y += LA.norm(s.y[i])**2
                    current_cand_idx += 1
                
                grad_matrix = np.hstack(batch_grads)
                weight = (np.sqrt(batch_norms_y) / LA.norm(grad_matrix)) if LA.norm(grad_matrix) > 1e-12 else 1.0
                
                current_rb, current_sv, _ = cls.incremental_POD([grad_matrix * weight], tol=cls.pod_tol_sweep[-1], initial_basis=(current_rb, current_sv))
                cond_val = LA.cond(G_test @ current_rb)
                print(f"Condition number at step {current_cand_idx}: {cond_val:.2e}")

            cls._Full_rb, cls._Full_sv = current_rb, current_sv
        else:
            print(f"  [{cls.__name__}] Reusing Master SVD (Rank: {cls._Full_rb.shape[1]})")

        # Perform energy-based truncation for the current tolerance
        rb, _sv, _ = cls._truncate_basis(cls._Full_rb, cls._Full_sv, tol)

        nosc_r = rb.shape[1]
        if nosc_r > cls.nosc:
            print(f"  [Warning] Basis rank ({nosc_r}) exceeds DOFs ({cls.nosc}). Basis is 'fat'. Skipping solver.")
            for target_class in target_classes:
                setattr(target_class, 'RB', None)
                setattr(target_class, 'nosc_r', None)
            setattr(cls, 'RB', None)
            setattr(cls, 'nosc_r', None)
            return

        sv = [_sv] # This should be the singular values of the truncated basis
        cond_val = LA.cond(G_test @ rb)
        print(f"Final condition number: {cond_val:.2e}")

        # Construct final RB and nosc_r
        RB = np.block([[rb, np.zeros_like(rb)], [np.zeros_like(rb), rb]])

        del rb
        gc.collect()

        print(f'Hamiltonian {RB.shape = }')

        fig, ax = logplot(sv, xlabel=f'index of singular values', xlims=[(1, len(s)) for s in sv])
        filename = os.path.join(MechSystem.data_folder, "sv_rb_H.pdf")
        save_figure(fig, filename, fig_data=None)

        # Set attributes on target classes
        for target_class in target_classes:
            setattr(target_class, 'RB', RB)
            setattr(target_class, 'nosc_r', nosc_r)
        # Also set on the class to make it available for hyper-reduction
        setattr(cls, 'RB', RB)
        setattr(cls, 'nosc_r', nosc_r)

    @classmethod
    def setup_hyperreduction(cls, target_classes):
        print('Setting up hyperreduction (Hamiltonian)...')
        
        if not hasattr(cls, 'RB') or cls.RB is None:
            print("  [Warning] Reduced basis not found or invalid. Skipping hyper-reduction setup.")
            for target_class in target_classes:
                setattr(target_class, 'RBxUx_inv_PxU', None)
                setattr(target_class, 'ham_z_deim', None)
                setattr(target_class, 'ham_zz_deim', None)
            return
        
        RB = cls.RB
        nosc_r = cls.nosc_r
        
        attr = cls.ham_z_expr
        attr_11 = RB[:cls.nosc, :nosc_r]
        P11, _ = DEIM(attr_11, plot_deim=False)

        attr_22 = RB[cls.nosc:, nosc_r:]
        P22, _ = DEIM(attr_22, plot_deim=False)

        P = np.block([
            [P11, np.zeros((P11.shape[0], P22.shape[1]))],
            [np.zeros((P22.shape[0], P11.shape[1])), P22]
        ])
        
        RBxUx_inv_PxU = RB.T @ RB @ LA.inv(P.T @ RB)
        print(f'{P.shape = }')

        deim_func = cls._create_indexed_deim_func(cls.ham_z_expr, P, "Hamiltonian")
        
        # Optional: ham_zz_deim
        ham_zz_deim = None
        if cls.hyperreducer == 'DEIM':
            ham_zz_deim = smp.lambdify((cls.y, cls.omega2, cls.beta),
                                        P.T @ cls.ham_zz_expr,
                                        modules=['numpy', 'scipy'])

        for target_class in target_classes:
            setattr(target_class, 'RB', RB)
            setattr(target_class, 'nosc_r', nosc_r)
            setattr(target_class, 'RBxUx_inv_PxU', RBxUx_inv_PxU)
            if callable(deim_func): # deim_func is already wrapped by ShapeWrapper
                setattr(target_class, 'ham_z_deim', staticmethod(deim_func))
            if ham_zz_deim and callable(ham_zz_deim):
                setattr(target_class, 'ham_zz_deim', staticmethod(ham_zz_deim))

    @classmethod
    def update_mdeim_hyperreduction(cls, solvers, target_classes):
        print(f'Updating MDEIM hyperreduction ({cls.__name__})...')
        tol = cls.pod_tol_sweep[-1]
        
        non_zero_indices = cls.ham_zz_nonzero_indices

        if not hasattr(cls, '_Full_Uj') or cls._Full_Uj is None:
            print(f"  [{cls.__name__}] Master MDEIM SVD not found. Computing from snapshots...")
            # Map solvers to indices
            solver_to_indices = {solver: indices for solver, indices in zip(solvers, cls.indices_list)}
            filtered_indices = [solver_to_indices[solver] for solver in solvers]

            results_generators = []
            for solver, indices in zip(solvers, filtered_indices):
                gen = cls.compute_snapshots(solver, indices, solver.ham_zz, is_dg=False, projection_indices=non_zero_indices, yield_batches=True)
                results_generators.append(gen)

            # Chain generators and use incremental_POD to capture Master basis
            all_snapshot_blocks = chain.from_iterable(results_generators)
            cls._Full_Uj, cls._Full_Sj, _ = cls.incremental_POD(all_snapshot_blocks, tol=cls.pod_tol_sweep[-1])
            del results_generators, all_snapshot_blocks
            gc.collect()
        else:
            print(f"  [{cls.__name__}] Reusing Master MDEIM SVD (Rank: {cls._Full_Uj.shape[1]})")

        Uj, sv, _ = cls._truncate_basis(cls._Full_Uj, cls._Full_Sj, tol)

        fig, ax = logplot(sv, xlabel=f'index of singular values of ham_zz', xlims=(1, len(sv)))
        filename = os.path.join(MechSystem.data_folder, "sv_mdeim_H.pdf")
        save_figure(fig, filename, fig_data=None)

        Pj, _ = DEIM(Uj, plot_deim=False)

        # Reconstruct the full sparse basis matrix from the DEIM results
        B_hat = Uj @ LA.inv(Pj.T @ Uj)
        total_elements = (2*cls.nosc)**2

        IP_Ux_inv_PxU = cls._reconstruct_sparse_basis(B_hat, non_zero_indices, total_elements)
        
        print(f'{IP_Ux_inv_PxU.shape = }')

        # Select non-zero elements from the symbolic expression for lambdification
        flat_expr = cls.ham_zz_expr.flat()
        selected_expr_elements = [flat_expr[i] for i in non_zero_indices]
        mdeim_col = Pj.T @ smp.Matrix(selected_expr_elements) # This is a column vector
        mdeim_func = lambda *args: smp.lambdify((cls.y, cls.omega2, cls.beta), mdeim_col, modules=['numpy', 'scipy'])(*args).flatten()

        for target_class in target_classes:
            setattr(target_class, 'IP_Ux_inv_PxU', IP_Ux_inv_PxU)
            if callable(mdeim_func):
                setattr(target_class, 'ham_zz_mdeim', staticmethod(mdeim_func))

class DiscreteGradientReducer(ReduceMechSystem, LagrangianMechSystem):
    """Reducer for Discrete Gradient solvers."""

    @classmethod
    def setup_reduced_model(cls, solvers, target_classes):
        print(f'Setting up reduced model ({cls.__name__})...')
        tol = cls.pod_tol_sweep[-1]

        # Map solvers to indices (always needed)
        solver_to_indices = {solver: indices for solver, indices in zip(solvers, cls.indices_list)}
        filtered_indices = [solver_to_indices[solver] for solver in solvers]

        # Test condition on the first snapshot (needed for augmentation and final reporting)
        sample_y = solvers[0].y[0]
        g_prime_shape0 = cls.g_prime_(sample_y).shape[0] // 2
        test_q = sample_y[:cls.nosc]
        G_test = cls.g_prime_(np.concatenate([test_q, np.zeros(cls.nosc)]))[:g_prime_shape0, :cls.nosc]

        if not hasattr(cls, '_Full_rb') or cls._Full_rb is None:
            print(f"  [{cls.__name__}] Master SVD not found. Computing from snapshots...")
            # Generator for q and p snapshots
            def get_state_snapshots():
                for solver, indices in zip(solvers, filtered_indices):
                    yield from cls.compute_snapshots(solver, indices, lambda y: y[:cls.nosc], yield_batches=True)
                for solver, indices in zip(solvers, filtered_indices):
                    yield from cls.compute_snapshots(solver, indices, lambda y: y[cls.nosc:], yield_batches=True)

            weights = np.eye(cls.nosc)
            # Capture a "Master Basis" with high precision
            current_rb, current_sv, _ = cls.incremental_POD(get_state_snapshots(), Xh=weights, tol=cls.pod_tol_sweep[-1])

            # 2. Iteratively augment with g_prime gradients until condition number is O(1)
            print("Iteratively augmenting with constraint gradients...")
            # Flatten all available time points for candidates
            candidates = []
            for solver, indices in zip(solvers, filtered_indices):
                for idx in indices:
                    candidates.append((solver, idx))
            
            current_cand_idx = 0
            cond_val = LA.cond(G_test @ current_rb)
            print(f"Initial condition number: {cond_val:.2e}")
            
            while cond_val > 10.0 and current_cand_idx < len(candidates):
                # Process gradients in small batches for efficiency
                batch_grads = []
                batch_norms_y = 0
                for _ in range(min(50, len(candidates) - current_cand_idx)):
                    s, i = candidates[current_cand_idx]
                    q = s.y[i][:cls.nosc]
                    G = cls.g_prime_(np.concatenate([q, np.zeros(cls.nosc)]))[:g_prime_shape0, :cls.nosc]
                    batch_grads.append(G.T)
                    batch_norms_y += LA.norm(s.y[i])**2
                    current_cand_idx += 1
                
                grad_matrix = np.hstack(batch_grads)
                weight = (np.sqrt(batch_norms_y) / LA.norm(grad_matrix)) if LA.norm(grad_matrix) > 1e-12 else 1.0
                
                current_rb, current_sv, _ = cls.incremental_POD([grad_matrix * weight], Xh=weights, tol=cls.pod_tol_sweep[-1], initial_basis=(current_rb, current_sv))
                cond_val = LA.cond(G_test @ current_rb)
                print(f"Condition number at step {current_cand_idx}: {cond_val:.2e}")

            cls._Full_rb, cls._Full_sv = current_rb, current_sv
        else:
            print(f"  [{cls.__name__}] Reusing Master SVD (Rank: {cls._Full_rb.shape[1]})")

        # Truncate based on current tol
        rb_1, sv_1, _ = cls._truncate_basis(cls._Full_rb, cls._Full_sv, tol)

        nosc_r = rb_1.shape[1]
        if nosc_r > cls.nosc:
            print(f"  [Warning] Basis rank ({nosc_r}) exceeds DOFs ({cls.nosc}). Basis is 'fat'. Skipping solver.")
            for target_class in target_classes:
                setattr(target_class, 'RB', None)
                setattr(target_class, 'nosc_r', None)
            setattr(cls, 'RB', None)
            setattr(cls, 'nosc_r', None)
            return

        sv = [sv_1] # This should be the singular values of the truncated basis
        cond_val = LA.cond(G_test @ rb_1)
        print(f"Final condition number: {cond_val:.2e}")

        # Construct final RB and nosc_r
        RB = np.block([[rb_1, np.zeros_like(rb_1)], [np.zeros_like(rb_1), rb_1]])

        del rb_1
        gc.collect()

        print(f'DiscreteGradient {RB.shape = }')

        fig, ax = logplot(sv, xlabel=f'index of singular values', xlims=[(1, len(s)) for s in sv])
        filename = os.path.join(MechSystem.data_folder, "sv_rb_DG.pdf")
        save_figure(fig, filename, fig_data=None)

        # Set attributes on target classes
        for target_class in target_classes:
            setattr(target_class, 'RB', RB)
            setattr(target_class, 'nosc_r', nosc_r)
        # Also set on the class to make it available for hyper-reduction
        setattr(cls, 'RB', RB)
        setattr(cls, 'nosc_r', nosc_r)

    @classmethod
    def setup_hyperreduction(cls, solvers, target_classes):
        print(f'Setting up hyperreduction ({cls.__name__})...')

        if not hasattr(cls, 'RB') or cls.RB is None:
            print("  [Warning] Reduced basis not found or invalid. Skipping hyper-reduction setup.")
            for target_class in target_classes:
                setattr(target_class, 'RBxUx_inv_PxU', None)
                setattr(target_class, 'lag_dg_deim', None)
                setattr(target_class, 'lag_dg_z_deim', None)
            return

        # Map solvers to indices
        solver_to_indices = {solver: indices for solver, indices in zip(solvers, cls.indices_list)}
        filtered_indices = [solver_to_indices[solver] for solver in solvers]

        # Use the final sweep tolerance for hyper-reduction
        hr_tol = cls.pod_tol_sweep[-1]

        if not hasattr(cls, '_Full_U1') or cls._Full_U1 is None:
            print(f"  [{cls.__name__}] Master Hyper-reduction SVDs not found. Computing from snapshots...")
            # Define a helper to recreate generators for dual-pass iPOD
            def get_snapshots():
                results_generators = []
                for solver, indices in zip(solvers, filtered_indices):
                    gen = cls.compute_snapshots(solver, indices, solver.lag_dg, is_dg=True, yield_batches=True)
                    results_generators.append(gen)
                return chain.from_iterable(results_generators)
            
            weights = np.eye(cls.nosc)
            cls._Full_U1, cls._Full_S1, _ = cls.incremental_POD((b[:cls.nosc, :] for b in get_snapshots()), Xh=weights, tol=cls.pod_tol_sweep[-1])
            cls._Full_U2, cls._Full_S2, _ = cls.incremental_POD((b[cls.nosc:, :] for b in get_snapshots()), Xh=weights, tol=cls.pod_tol_sweep[-1])
            gc.collect()
        else:
            print(f"  [{cls.__name__}] Reusing Master Hyper-reduction SVDs (Ranks: {cls._Full_U1.shape[1]}, {cls._Full_U2.shape[1]})")

        U1, S1, _ = cls._truncate_basis(cls._Full_U1, cls._Full_S1, hr_tol)
        U2, S2, _ = cls._truncate_basis(cls._Full_U2, cls._Full_S2, hr_tol)

        
        '''
        # Augment U1 and U2 with constraint gradients using incremental_POD
        print("Augmenting hyper-reduction bases with constraint gradients...")

        # Test condition number before augmentation
        J_test = cls.g_prime_(solvers[0].y[0])
        m_test = J_test.shape[0] // 2
        cond1_before = LA.cond(J_test[:, :cls.nosc] @ U1)
        cond2_before = LA.cond(J_test[m_test:, cls.nosc:] @ U2)
        print(f"Condition number of G @ U1 (before): {cond1_before:.2e}")
        print(f"Condition number of G_pv @ U2 (before): {cond2_before:.2e}")

        gs1, gs2 = [], []
        for solver, indices in zip(solvers, filtered_indices):
            for i in indices:
                J = cls.g_prime_(solver.y[i])
                m = J.shape[0] // 2
                gs1.append(J[:m, :cls.nosc].T)
                gs2.append(np.hstack([J[m:, :cls.nosc].T, J[m:, cls.nosc:].T]))
        
        gs1_all, gs2_all = np.hstack(gs1), np.hstack(gs2)

        # Weigh appropriately wrt full_snapshots
        norm_f1, norm_f2 = LA.norm(full_snapshots[:cls.nosc, :]), LA.norm(full_snapshots[cls.nosc:, :])
        if (n_gs1 := LA.norm(gs1_all)) > 1e-12: gs1_all *= (norm_f1 / n_gs1)
        if (n_gs2 := LA.norm(gs2_all)) > 1e-12: gs2_all *= (norm_f2 / n_gs2)

        U1, S1, _ = cls.incremental_POD([gs1_all], Xh=weights, tol=hr_tol, initial_basis=(U1, S1))
        U2, S2, _ = cls.incremental_POD([gs2_all], Xh=weights, tol=hr_tol, initial_basis=(U2, S2))

        # Test condition number after augmentation
        cond1_after = LA.cond(J_test[:, :cls.nosc] @ U1)
        cond2_after = LA.cond(J_test[m_test:, cls.nosc:] @ U2)
        print(f"Condition number of G @ U1 (after): {cond1_after:.2e}")
        print(f"Condition number of G_pv @ U2 (after): {cond2_after:.2e}")
        '''
        
        P1, _ = DEIM(U1, plot_deim=False)
        P2, _ = DEIM(U2, plot_deim=False)

        P = np.block([
            [P1, np.zeros((P1.shape[0], P2.shape[1]))],
            [np.zeros((P2.shape[0], P1.shape[1])), P2]
        ])

        # Construct nonlinear basis U for projection
        U_nonlinear = np.block([
            [U1, np.zeros((U1.shape[0], U2.shape[1]))],
            [np.zeros((U2.shape[0], U1.shape[1])), U2]
        ])
        
        RBxUx_inv_PxU = cls.RB.T @ U_nonlinear @ LA.inv(P.T @ U_nonlinear)
        print(f'{P.shape = }')

        deim_func = cls._create_indexed_deim_func(cls.lag_dg_expr, P, "DiscreteGradient")
        
        # Optional: lag_dg_z_deim
        lag_dg_z_deim = None
        if cls.hyperreducer == 'DEIM':
             lag_dg_z_deim = smp.lambdify((cls.y, cls.y1, cls.omega2),
                                        P.T @ cls.lag_dg_z_expr,
                                        modules=['numpy', 'scipy'])

        for target_class in target_classes:
            setattr(target_class, 'RB', cls.RB)
            setattr(target_class, 'nosc_r', cls.nosc_r)
            setattr(target_class, 'RBxUx_inv_PxU', RBxUx_inv_PxU)
            if callable(deim_func): # deim_func is already wrapped by ShapeWrapper
                setattr(target_class, 'lag_dg_deim', staticmethod(deim_func))
            if lag_dg_z_deim and callable(lag_dg_z_deim):
                setattr(target_class, 'lag_dg_z_deim', staticmethod(lag_dg_z_deim))

    @classmethod
    def update_mdeim_hyperreduction(cls, solvers, target_classes):
        print(f'Updating MDEIM hyperreduction ({cls.__name__})...')
        tol = cls.pod_tol_sweep[-1]
        
        non_zero_indices = cls.lag_dg_z_nonzero_indices

        if not hasattr(cls, '_Full_Uj') or cls._Full_Uj is None:
            print(f"  [{cls.__name__}] Master MDEIM SVD not found. Computing from snapshots...")
            # Map solvers to indices
            solver_to_indices = {solver: indices for solver, indices in zip(solvers, cls.indices_list)}
            filtered_indices = [solver_to_indices[solver] for solver in solvers]

            results_generators = []
            for solver, indices in zip(solvers, filtered_indices):
                gen = cls.compute_snapshots(solver, indices, solver.lag_dg_z, is_dg=True, projection_indices=non_zero_indices, yield_batches=True)
                results_generators.append(gen)
            
            all_snapshot_blocks = chain.from_iterable(results_generators)
            cls._Full_Uj, cls._Full_Sj, _ = cls.incremental_POD(all_snapshot_blocks, tol=cls.pod_tol_sweep[-1])
            del results_generators, all_snapshot_blocks
            gc.collect()
        else:
            print(f"  [{cls.__name__}] Reusing Master MDEIM SVD (Rank: {cls._Full_Uj.shape[1]})")

        Uj, sv, _ = cls._truncate_basis(cls._Full_Uj, cls._Full_Sj, tol)
        fig, ax = logplot(sv, xlabel=f'index of singular values of lag_dg_z', xlims=(1, len(sv)))
        filename = os.path.join(MechSystem.data_folder, "sv_mdeim_DG.pdf")
        save_figure(fig, filename, fig_data=None)

        Pj, _ = DEIM(Uj, plot_deim=False)
        
        # Reconstruct the full sparse basis matrix from the DEIM results
        B_hat = Uj @ LA.inv(Pj.T @ Uj)
        IP_Ux_inv_PxU = cls._reconstruct_sparse_basis(B_hat, non_zero_indices, (2*cls.nosc)**2)
        
        print(f'{IP_Ux_inv_PxU.shape = }')

        # Select non-zero elements from the symbolic expression for lambdification
        flat_expr = cls.lag_dg_z_expr.flat()
        selected_expr_elements = [flat_expr[i] for i in non_zero_indices]
        mdeim_col = Pj.T @ smp.Matrix(selected_expr_elements) # This is a column vector
        mdeim_func = lambda *args: smp.lambdify((cls.y, cls.y1, cls.omega2), mdeim_col, modules=['numpy', 'scipy'])(*args).flatten()

        for target_class in target_classes:
            setattr(target_class, 'IP_Ux_inv_PxU', IP_Ux_inv_PxU)
            if callable(mdeim_func):
                setattr(target_class, 'lag_dg_z_mdeim', staticmethod(mdeim_func))