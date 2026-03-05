import os
import gc
import numpy as np
import sympy as smp
import psutil
from itertools import chain
import concurrent.futures
from numpy import linalg as LA
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
        'IP_g_prime_x_lambda_y', 'IP_g_prime_x_lambda_lambda',
        # Lambdified functions (DEIM)
        'ham_z_deim', 'ham_zz_deim', 'lag_dg_deim', 'lag_dg_z_deim',
        # Lambdified functions (MDEIM)
        'ham_zz_mdeim', 'lag_dg_z_mdeim', 'g_prime_mdeim',
        'g_prime_x_lambda_y_mdeim', 'g_prime_x_lambda_lambda_mdeim',
        # Shape attributes for reshaping
        'g_prime_shape',
        'g_prime_x_lambda_y_shape', 'g_prime_x_lambda_lambda_shape'
    ]

    @staticmethod
    def compute_snapshots(solver, indices, func, is_dg=False, is_mdeim_gprime=False, projection_matrix=None, batch_size=None, yield_batches=False, max_workers=None):
        """
        Generic method to compute snapshots or projected snapshots in batches.
        """
        # Determine num_workers
        if max_workers is not None:
            num_workers = max_workers
        else:
            # If running in a SLURM job, use the allocated CPUs.
            # Otherwise, use the number of CPUs on the machine.
            # The dynamic batch sizing will adjust for memory constraints.
            if 'SLURM_CPUS_PER_TASK' in os.environ:
                num_workers = int(os.environ['SLURM_CPUS_PER_TASK'])
            else:
                num_workers = os.cpu_count() or 1

        # Determine batch size dynamically if not provided
        if batch_size is None:
            # Compute a single sample to estimate memory usage
            idx0 = indices[0]
            if is_dg:
                # DiscreteGradient expects pairs (y_k, y_{k+1})
                # Handle both object and dict access for solver
                y_data = solver.y if hasattr(solver, 'y') else solver['y']
                sample_inp = (y_data[idx0], y_data[idx0+1])
                sample_res = func(sample_inp)
            elif is_mdeim_gprime:
                y_data = solver.y if hasattr(solver, 'y') else solver['y']
                lambda_data = solver.Lambda if hasattr(solver, 'Lambda') else solver['Lambda']
                sample_res = func(y_data[idx0], lambda_data[idx0])
            else:
                y_data = solver.y if hasattr(solver, 'y') else solver['y']
                sample_inp = y_data[idx0]
                sample_res = func(sample_inp)
            
            # Estimate size in bytes
            item_size = sample_res.nbytes
            if projection_matrix is not None:
                # If projecting, we also store the flattened result temporarily
                item_size += sample_res.size * 4 # float32
            
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
            
            # Use at most 10% of available memory for the batch buffer across all threads
            target_mem_per_worker = (available_mem * 0.1) / num_workers
            
            batch_size = int(target_mem_per_worker / item_size)
            batch_size = max(1, min(batch_size, 2000)) # Clamp between 1 and 2000

        def process_batch(batch_indices):
            if is_dg:
                # DiscreteGradient expects pairs (y_k, y_{k+1})
                inputs = zip(solver.y[batch_indices], solver.y[np.array(batch_indices)+1])
                raw_results = [func(y) for y in inputs]
            elif is_mdeim_gprime:
                # MDEIM for g_prime_* needs (y, Lambda) pairs
                inputs = zip(solver.y[batch_indices], solver.Lambda[batch_indices])
                raw_results = [func(*y) for y in inputs]
            else:
                inputs = solver.y[batch_indices]
                raw_results = [func(y) for y in inputs]
            
            if projection_matrix is not None:
                # Flatten and stack for projection
                batch_mats = np.vstack([r.flatten().astype(np.float32) for r in raw_results])
                # Project: (IP @ batch_mats.T).T -> (batch_size, reduced_dim)
                return (projection_matrix @ batch_mats.T).T
            else:
                return raw_results

        batches = [indices[i:i+batch_size] for i in range(0, len(indices), batch_size)]
        
        def _generator():
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                for batch_res in tqdm(executor.map(process_batch, batches), total=len(batches), desc=f"Computing snapshots ({num_workers} workers)", leave=False):
                    if projection_matrix is not None:
                        yield batch_res.T
                    else:
                        # Flatten and stack to (batch_size, N_h), then transpose
                        yield np.vstack([r.flatten().astype(np.float32) for r in batch_res]).T

        if yield_batches:
            return _generator()

        return np.hstack(list(_generator()))

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

    @classmethod
    def hyperreduce_constraints(cls, solvers, target_classes):
        solver_type = "Hamiltonian" if issubclass(cls, HamiltonianMechSystem) else "DiscreteGradient"
        print('Computing constraints reduction in parallel...')
        
        # Map solvers to indices
        solver_to_indices = {solver: indices for solver, indices in zip(solvers, cls.indices_list)}
        filtered_indices = [solver_to_indices[solver] for solver in solvers]

        def compute_constraint_basis(func_name, expr, is_g_prime=False):
            print(f'Computing {func_name} reduction...', flush=True)

            # Determine if this is a "prime" type (g_prime, g_prime_x_lambda_y, g_prime_x_lambda_lambda)
            prime_types = ['g_prime_', 'g_prime_x_lambda_y_', 'g_prime_x_lambda_lambda_']
            is_prime_type = is_g_prime or func_name in prime_types

            if is_prime_type:
                # --- Get shape for the prime constraint ---
                # Use getattr for safer attribute access
                func_to_get_shape = getattr(solvers[0], func_name)
                if func_name in ['g_prime_x_lambda_y_', 'g_prime_x_lambda_lambda_']:
                    g_prime_shape = func_to_get_shape(solvers[0].y[0], solvers[0].Lambda[0]).shape
                else: # This handles 'g_prime_'
                    g_prime_shape = func_to_get_shape(solvers[0].y[0]).shape
                
                shape_attr = func_name + 'shape'
                for target_class in target_classes:
                    setattr(target_class, shape_attr, g_prime_shape)

                non_zero_indices = getattr(cls, func_name + "nonzero_indices")
                IP = np.zeros((len(non_zero_indices), np.prod(g_prime_shape)), dtype=np.int8)
                IP[np.arange(len(non_zero_indices)), non_zero_indices] = 1

                # --- Create snapshot matrix for prime constraints ---
                results_generators = []
                for solver, indices in tqdm(zip(solvers, filtered_indices), total=len(solvers), desc=f"Snapshots for {func_name}"):
                    # Select the function to call based on func_name
                    if func_name in ['g_prime_x_lambda_y_', 'g_prime_x_lambda_lambda_']:
                        snapshot_func = getattr(solver, func_name)
                        gen = cls.compute_snapshots(solver, indices, snapshot_func,
                                                          is_mdeim_gprime=True,
                                                          projection_matrix=IP,
                                                          yield_batches=True)
                    else:  # This handles 'g_prime_'
                        snapshot_func = solver.g_prime__
                        gen = cls.compute_snapshots(solver, indices, snapshot_func,
                                                          projection_matrix=IP,
                                                          yield_batches=True)
                    results_generators.append(gen)
                
                all_snapshot_blocks = chain.from_iterable(results_generators)
                Uj, sv, _ = cls.incremental_POD(all_snapshot_blocks, tol=cls.pod_tol)
                del results_generators, all_snapshot_blocks
                gc.collect()
            else:
                # Create snapshot matrix for g
                F = np.hstack([
                    np.array([solver.g(y) for y in solver.y[indices]]).T
                    for solver, indices in zip(solvers, filtered_indices)
                ])
                Uj, sv, _ = POD(F, np.eye(F.shape[0]), cls.pod_tol)
                del F

            # Compute DEIM points and interpolation matrix
            Pj, _ = DEIM(Uj, plot_deim=False)

            if is_prime_type:
                basis = IP.T @ Uj @ LA.inv(Pj.T @ Uj)
                # Create lambdified function
                col = Pj.T @ IP @ expr.reshape(np.prod(g_prime_shape), 1)
            else:
                basis = Uj @ LA.inv(Pj.T @ Uj)
                col = Pj.T @ expr

            print(f'{basis.shape = }')

            # Create and wrap lambdified function
            # Handle argument signature for g_prime_x_lambda_y and g_prime_x_lambda_lambda
            if func_name in ['g_prime_x_lambda_y_', 'g_prime_x_lambda_lambda_']:
                mdeim_func = smp.lambdify((cls.y, cls.lag_mult), col, modules=['numpy', 'scipy'])
            elif func_name in ["g_"]:
                # mdeim_func = smp.lambdify((cls.y,), col, modules=['scipy'])
                mdeim_func = cls._create_indexed_deim_func(expr, Pj, solver_type)
            else:
                mdeim_func = smp.lambdify((cls.y,), col, modules=['numpy', 'scipy'])

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
            tasks.append(("g_prime_x_lambda_lambda_", cls.g_prime_x_lambda_lambda_expr, True))

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
            IP_g_prime_x_lambda_lambda, sv_g_prime_x_lambda_lambda = results['g_prime_x_lambda_lambda_']

            for target_class in target_classes:
                setattr(target_class, '_IP_Ux_inv_PxU', _IP_Ux_inv_PxU)
                setattr(target_class, 'IP_g_prime_x_lambda_y', IP_g_prime_x_lambda_y)
                setattr(target_class, 'IP_g_prime_x_lambda_lambda', IP_g_prime_x_lambda_lambda)
        else:
            raise ValueError(f"Invalid solver type: {solver_type}")

        # Plot singular values for g_prime, g_prime_x_lambda_y and g_prime_x_lambda_lambda using logplot, pass the singular values and xlims as a list
        sv_list = [sv_g_prime]
        xlims_list = [(1, len(sv_g_prime))]

        # Add additional singular values if solver type is DiscreteGradient
        if solver_type == "DiscreteGradient":
            sv_list.extend([sv_g_prime_x_lambda_y, sv_g_prime_x_lambda_lambda])
            xlims_list.extend(
                [(1, len(sv_g_prime_x_lambda_y)), (1, len(sv_g_prime_x_lambda_lambda))]
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
    def incremental_POD(snapshot_blocks, max_basis_size=None, tol=None):
        """
        Compute POD using Incremental SVD (uncentered) to handle extremely large datasets.
        
        Unlike sklearn.decomposition.IncrementalPCA, this does NOT center the data,
        preserving the physical state origin required for POD.
        
        Parameters:
        snapshot_blocks: Iterable of snapshot matrices (arrays of shape (Nh, batch_size)).
        max_basis_size: Maximum rank of the basis to maintain.
        tol: Tolerance for singular value truncation (energy criteria).
        
        Returns:
        U: POD modes (Nh, k)
        S: Singular values (k,)
        k: Rank of the basis
        """
        U = None
        S = None
        
        for i, block in enumerate(snapshot_blocks):
            # Handle Dask arrays
            if hasattr(block, 'compute'):
                block = block.compute()
            
            # Ensure block is 2D
            if block.ndim == 1:
                block = block.reshape(-1, 1)
                
            B = block
            
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
            
        return U, S, len(S)

class HamiltonianReducer(ReduceMechSystem, HamiltonianMechSystem):
    """Reducer for Hamiltonian solvers."""

    @classmethod
    def setup_reduced_model(cls, solvers, target_classes):
        print('Computing reduced basis (Hamiltonian)...')
        
        # Map solvers to indices
        solver_to_indices = {solver: indices for solver, indices in zip(solvers, cls.indices_list)}
        filtered_indices = [solver_to_indices[solver] for solver in solvers]

        # Create snapshot list
        y_list = np.hstack([solver.y[indices].T for solver, indices 
                            in zip(solvers, filtered_indices)])

        # Create F2 in batches
        F2_list = []
        for solver, indices in zip(solvers, filtered_indices):
            batch_result = cls.compute_snapshots(solver, indices, solver.ham_z, is_dg=False)
            F2_list.append(batch_result)

        F2 = np.hstack(F2_list)
        del F2_list
        gc.collect()

        # Calculate weight ratio based on norms
        norm_y = LA.norm(y_list)
        norm_F2 = LA.norm(F2)
        weight_ratio = (norm_y / norm_F2) if norm_F2 > 1e-12 else 1.0
        
        weights = np.eye(cls.nosc)

        # --- Inlined PSD logic to get access to rb ---
        # 1. POD on state snapshots only
        snapshots = np.hstack([y_list[:cls.nosc, :], y_list[cls.nosc:, :]])
        rb, _sv, _ = POD(snapshots, weights, cls.pod_tol)

        # ---

        # Check for rank deficiency and augment basis
        print("Checking rank deficiency...")
        g_prime_shape0 = cls.g_prime_(y_list[:,0]).shape[0] // 2
        
        gradients_list = []
        ill_conditioned_indices = []

        # Subsample snapshots to check for efficiency
        num_check_snapshots = min(200, y_list.shape[1])
        check_indices = np.linspace(0, y_list.shape[1] - 1, num_check_snapshots, dtype=int)
        print(f"Checking rank deficiency on {num_check_snapshots} snapshots...")
        for i in check_indices:
            q_i = y_list[:cls.nosc, i]
            y_full = np.concatenate([q_i, np.zeros(cls.nosc)])
            G = cls.g_prime_(y_full)[:g_prime_shape0, :cls.nosc]
            
            if LA.matrix_rank(G @ rb) < g_prime_shape0 or LA.cond(G @ rb) > 1/cls.tol_reduced:
                gradients_list.append(G.T)
                ill_conditioned_indices.append(i)
        
        if gradients_list:
            print(f"Augmenting basis with {len(gradients_list)} constraint gradients and F2 snapshots...")
            
            # Collect constraint gradients and corresponding F2 snapshots
            constraint_gradients = np.hstack(gradients_list)
            f2_snapshots = np.hstack([F2[:cls.nosc, ill_conditioned_indices], F2[cls.nosc:, ill_conditioned_indices]]) * weight_ratio
            
            # Combine all augmentation snapshots
            augmentation_snapshots = np.hstack([constraint_gradients, f2_snapshots])

            # --- Check condition number before augmentation ---
            G_test = gradients_list[0].T 
            cond_before = LA.cond(G_test @ rb)
            print(f"Condition number of G @ rb (before): {cond_before:.2e}")
            # ---
            
            # Project gradients onto the orthogonal complement of rb
            aug_perp = augmentation_snapshots - rb @ (rb.T @ augmentation_snapshots)
            aug_perp -= rb @ (rb.T @ aug_perp) # re-orthogonalization
            
            # Compute SVD of the residuals
            U_aug, s_aug, _ = LA.svd(aug_perp, full_matrices=False)
            
            # Select modes
            energy_aug = np.cumsum(s_aug**2) / np.sum(s_aug**2)
            r_aug = np.searchsorted(energy_aug, 1 - cls.pod_tol, side='left') + 1

            if r_aug > 0:
                print(f"Adding {r_aug} modes from constraint gradients.")
                rb = np.hstack([rb, U_aug[:, :r_aug]])
                rb, _ = LA.qr(rb, mode='reduced') # Orthonormalize
                
                # --- Check condition number after augmentation ---
                cond_after = LA.cond(G_test @ rb)
                print(f"Condition number of G @ rb (after): {cond_after:.2e}")
                # ---
                
                # Update singular values for plotting
                sv = [_sv, s_aug[:r_aug]]

        # Construct final RB and nosc_r
        nosc_r = rb.shape[1]
        RB = np.block([[rb, np.zeros_like(rb)], [np.zeros_like(rb), rb]])

        del y_list, F2, snapshots, rb
        gc.collect()

        print(f'Hamiltonian {RB.shape = }')

        fig, ax = logplot(sv, xlabel=f'index of singular values', xlims=(1, len(sv)))
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
            if callable(deim_func):
                setattr(target_class, 'ham_z_deim', staticmethod(deim_func))
            if ham_zz_deim and callable(ham_zz_deim):
                setattr(target_class, 'ham_zz_deim', staticmethod(ham_zz_deim))

    @classmethod
    def update_mdeim_hyperreduction(cls, solvers, target_classes):
        print(f'Computing ham_zz reduction...')
        
        non_zero_indices = cls.ham_zz_nonzero_indices
        IP = np.zeros((len(non_zero_indices), (2*cls.nosc)**2), dtype=np.int8)
        IP[np.arange(len(non_zero_indices)), non_zero_indices] = 1

        # Map solvers to indices
        solver_to_indices = {solver: indices for solver, indices in zip(solvers, cls.indices_list)}
        filtered_indices = [solver_to_indices[solver] for solver in solvers]

        results_generators = []
        for solver, indices in zip(solvers, filtered_indices):
            gen = cls.compute_snapshots(solver, indices, solver.ham_zz, is_dg=False, projection_matrix=IP, yield_batches=True)
            results_generators.append(gen)

        print('Computing POD for ham_zz snapshots...')
        # Chain generators and use incremental_POD to avoid loading all snapshots into memory
        all_snapshot_blocks = chain.from_iterable(results_generators)
        Uj, sv, _ = cls.incremental_POD(all_snapshot_blocks, tol=cls.pod_tol)
        del results_generators, all_snapshot_blocks
        gc.collect()
        fig, ax = logplot(sv, xlabel=f'index of singular values of ham_zz', xlims=(1, len(sv)))
        filename = os.path.join(MechSystem.data_folder, "sv_mdeim_H.pdf")
        save_figure(fig, filename, fig_data=None)

        Pj, _ = DEIM(Uj, plot_deim=False)
        IP_Ux_inv_PxU = IP.T @ Uj @ LA.inv(Pj.T @ Uj)
        print(f'{IP_Ux_inv_PxU.shape = }')

        mdeim_col = Pj.T @ IP @ cls.ham_zz_expr.flat()
        mdeim_func = smp.lambdify((cls.y, cls.omega2, cls.beta), mdeim_col, modules=['numpy', 'scipy'])

        for target_class in target_classes:
            setattr(target_class, 'IP_Ux_inv_PxU', IP_Ux_inv_PxU)
            if callable(mdeim_func):
                setattr(target_class, 'ham_zz_mdeim', staticmethod(mdeim_func))

class DiscreteGradientReducer(ReduceMechSystem, LagrangianMechSystem):
    """Reducer for Discrete Gradient solvers."""

    @classmethod
    def setup_reduced_model(cls, solvers, target_classes):
        print('Computing reduced basis (DiscreteGradient)...')
        
        # Map solvers to indices
        solver_to_indices = {solver: indices for solver, indices in zip(solvers, cls.indices_list)}
        filtered_indices = [solver_to_indices[solver] for solver in solvers]

        # Create snapshot list
        y_list = np.hstack([solver.y[indices].T for solver, indices 
                            in zip(solvers, filtered_indices)])

        weights = np.eye(cls.nosc)

        # Use standard POD on state snapshots only. Including F2 snapshots,
        # while consistent with HamiltonianReducer, can interfere with the
        # subsequent constraint gradient augmentation for this solver type.
        snapshots = np.hstack([y_list[:cls.nosc, :], y_list[cls.nosc:, :]])
        rb_1, sv_1, _ = POD(snapshots, weights, cls.pod_tol)

        # Combine results in a block diagonal matrix RB
        RB = np.block([[rb_1, np.zeros_like(rb_1)], [np.zeros_like(rb_1), rb_1]])
        nosc_r = RB.shape[1] // 2
        sv = [sv_1]

        # Check for rank deficiency and augment basis to satisfy LBB condition
        print("Checking rank deficiency...")
        dummy_y = np.zeros(2 * cls.nosc)
        g_prime_shape0 = cls.g_prime_(dummy_y).shape[0] // 2
        
        gradients_list = []

        # Subsample snapshots to check for efficiency
        num_check_snapshots = min(200, y_list.shape[1])
        check_indices = np.linspace(0, y_list.shape[1] - 1, num_check_snapshots, dtype=int)
        print(f"Checking rank deficiency on {num_check_snapshots} snapshots...")
        for i in check_indices:
            q_i = y_list[:cls.nosc, i]
            y_full = np.concatenate([q_i, np.zeros(cls.nosc)])
            G = cls.g_prime_(y_full)[:g_prime_shape0, :cls.nosc]
            
            if LA.matrix_rank(G @ rb_1) < g_prime_shape0 or LA.cond(G @ rb_1) > 1/cls.tol_reduced:
                gradients_list.append(G.T)
        
        if gradients_list:
            print(f"Augmenting reduced basis with {len(gradients_list)} constraint gradients...")
            gradients = np.hstack(gradients_list)

            # --- Check condition number before augmentation ---
            # Use the first rank-deficient snapshot for comparison
            G_test = gradients_list[0].T 
            cond_before = LA.cond(G_test @ rb_1)
            print(f"Condition number of G @ rb_1 (before): {cond_before:.2e}")
            # ---
            
            # Project gradients onto the orthogonal complement of rb_1
            # Use re-orthogonalization (apply projection twice) for numerical stability
            gradients_perp = gradients - rb_1 @ (rb_1.T @ gradients)
            gradients_perp -= rb_1 @ (rb_1.T @ gradients_perp)
            
            # Compute SVD of the residuals to find dominant normal directions
            U_aug, s_aug, _ = LA.svd(gradients_perp, full_matrices=False)
            
            # Select modes with significant energy
            # tol_aug = 1e-8
            energy_aug = np.cumsum(s_aug**2) / np.sum(s_aug**2)
            r_aug = np.searchsorted(energy_aug, 1 - cls.pod_tol, side='left') + 1

            if r_aug > 0:
                print(f"Adding {r_aug} modes from constraint gradients.")
                rb_1 = np.hstack([rb_1, U_aug[:, :r_aug]])

                # Orthonormalize rb_1 using thin QR
                rb_1, _ = LA.qr(rb_1, mode='reduced')
                
                # Update RB and nosc_r
                RB = np.block([[rb_1, np.zeros_like(rb_1)], [np.zeros_like(rb_1), rb_1]])
                nosc_r = RB.shape[1] // 2

                # --- Check condition number after augmentation ---
                cond_after = LA.cond(G_test @ rb_1)
                print(f"Condition number of G @ rb_1 (after): {cond_after:.2e}")
                # ---
                
                # Update singular values for plotting
                # sv_1 = np.concatenate([sv_1, s_aug[:r_aug]])
                sv = [sv_1, s_aug[:r_aug]]

        del y_list, snapshots, rb_1
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
    def setup_hyperreduction(cls, target_classes):
        print('Setting up hyperreduction (DiscreteGradient)...')
        
        RB = cls.RB
        nosc_r = cls.nosc_r
        
        attr = cls.lag_dg_expr
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

        deim_func = cls._create_indexed_deim_func(cls.lag_dg_expr, P, "DiscreteGradient")
        
        # Optional: lag_dg_z_deim
        lag_dg_z_deim = None
        if cls.hyperreducer == 'DEIM':
             lag_dg_z_deim = smp.lambdify((cls.y, cls.y1, cls.omega2),
                                        P.T @ cls.lag_dg_z_expr,
                                        modules=['numpy', 'scipy'])

        for target_class in target_classes:
            setattr(target_class, 'RB', RB)
            setattr(target_class, 'nosc_r', nosc_r)
            setattr(target_class, 'RBxUx_inv_PxU', RBxUx_inv_PxU)
            if callable(deim_func):
                setattr(target_class, 'lag_dg_deim', staticmethod(deim_func))
            if lag_dg_z_deim and callable(lag_dg_z_deim):
                setattr(target_class, 'lag_dg_z_deim', staticmethod(lag_dg_z_deim))

    @classmethod
    def update_mdeim_hyperreduction(cls, solvers, target_classes):
        print(f'Computing lag_dg_z reduction...')
        
        non_zero_indices = cls.lag_dg_z_nonzero_indices
        IP = np.zeros((len(non_zero_indices), (2*cls.nosc)**2), dtype=np.int8)
        IP[np.arange(len(non_zero_indices)), non_zero_indices] = 1

        # Map solvers to indices
        solver_to_indices = {solver: indices for solver, indices in zip(solvers, cls.indices_list)}
        filtered_indices = [solver_to_indices[solver] for solver in solvers]

        results_generators = []
        for solver, indices in zip(solvers, filtered_indices):
            gen = cls.compute_snapshots(solver, indices, solver.lag_dg_z, is_dg=True, projection_matrix=IP, yield_batches=True)
            results_generators.append(gen)
        
        print('Computing POD for lag_dg_z snapshots...')
        # Chain generators and use incremental_POD to avoid loading all snapshots into memory
        all_snapshot_blocks = chain.from_iterable(results_generators)
        Uj, sv, _ = cls.incremental_POD(all_snapshot_blocks, tol=cls.pod_tol)
        del results_generators, all_snapshot_blocks
        gc.collect()
        fig, ax = logplot(sv, xlabel=f'index of singular values of lag_dg_z', xlims=(1, len(sv)))
        filename = os.path.join(MechSystem.data_folder, "sv_mdeim_DG.pdf")
        save_figure(fig, filename, fig_data=None)

        Pj, _ = DEIM(Uj, plot_deim=False)
        IP_Ux_inv_PxU = IP.T @ Uj @ LA.inv(Pj.T @ Uj)
        print(f'{IP_Ux_inv_PxU.shape = }')

        mdeim_col = Pj.T @ IP @ cls.lag_dg_z_expr.flat()
        mdeim_func = smp.lambdify((cls.y, cls.y1, cls.omega2), mdeim_col, modules=['numpy', 'scipy'])

        for target_class in target_classes:
            setattr(target_class, 'IP_Ux_inv_PxU', IP_Ux_inv_PxU)
            if callable(mdeim_func):
                setattr(target_class, 'lag_dg_z_mdeim', staticmethod(mdeim_func))