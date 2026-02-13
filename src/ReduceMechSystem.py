import os
import gc
import numpy as np
import sympy as smp
import psutil
import concurrent.futures
from numpy import linalg as LA
from functools import partial
from tqdm.auto import tqdm
from System import MechSystem, HamiltonianMechSystem, LagrangianMechSystem
from ODESolver import DiscreteGradient
from podDEIM import POD, PSD, DEIM
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
    def compute_snapshots(solver, indices, func, is_dg=False, projection_matrix=None, batch_size=None):
        """
        Generic method to compute snapshots or projected snapshots in batches.
        """
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
            
            # Use at most 10% of available memory for the batch buffer across all threads
            # Assuming ThreadPoolExecutor uses default max_workers (cpu_count + 4)
            num_workers = min(32, (os.cpu_count() or 1) + 4)
            target_mem_per_worker = (available_mem * 0.1) / num_workers
            
            batch_size = int(target_mem_per_worker / item_size)
            batch_size = max(1, min(batch_size, 2000)) # Clamp between 1 and 2000

        def process_batch(batch_indices):
            if is_dg:
                # DiscreteGradient expects pairs (y_k, y_{k+1})
                inputs = zip(solver.y[batch_indices], solver.y[np.array(batch_indices)+1])
                raw_results = [func(y) for y in inputs]
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
        result = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for batch_res in tqdm(executor.map(process_batch, batches), total=len(batches), desc="Computing snapshots", leave=False):
                result.extend(batch_res)

        return np.array(result).T

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
            print(f'Computing {func_name} reduction...')

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
                snapshot_list = []
                for solver, indices in zip(solvers, filtered_indices):
                    # Select the function to call based on func_name
                    if func_name in ['g_prime_x_lambda_y_', 'g_prime_x_lambda_lambda_']:
                        snapshot_func = getattr(solver, func_name)
                        vectors = [
                            IP @ snapshot_func(y, Lambda).flatten()
                            for y, Lambda in zip(solver.y[indices], solver.Lambda[indices])
                        ]
                    else:  # This handles 'g_prime_'
                        snapshot_func = solver.g_prime__
                        vectors = [
                            IP @ snapshot_func(y).flatten()
                            for y in solver.y[indices]  # g_prime__ only takes y
                        ]
                    snapshot_list.append(np.array(vectors).T)
                F = np.hstack(snapshot_list)
            else:
                # Create snapshot matrix for g
                F = np.hstack([
                    np.array([solver.g(y) for y in solver.y[indices]]).T
                    for solver, indices in zip(solvers, filtered_indices)
                ])

            # Compute POD basis and plot singular values
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

            return func_name, basis, sv

        # Prepare tasks for parallel execution
        tasks = [('g_prime_', cls.g_prime_expr, True)]
        if solver_type == "DiscreteGradient":
            tasks.append(("g_prime_x_lambda_y_", cls.g_prime_x_lambda_y_expr, True))
            tasks.append(("g_prime_x_lambda_lambda_", cls.g_prime_x_lambda_lambda_expr, True))

        results = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(compute_constraint_basis, *task) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                func_name, basis, sv = future.result()
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
        
        weights = np.eye(cls.nosc) * weight_ratio

        RB, sv, nosc_r = PSD(F2, y_list, cls)

        del y_list
        gc.collect()

        print(f'{RB.shape = }')

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

        results = []
        for solver, indices in zip(solvers, filtered_indices):
            result = cls.compute_snapshots(solver, indices, solver.ham_zz, is_dg=False, projection_matrix=IP)
            results.append(result)

        F3 = np.hstack(results)

        Uj, sv, _ = POD(F3, np.eye(F3.shape[0]), cls.pod_tol)
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

        # Create F2 in batches
        F2_list = []
        for solver, indices in zip(solvers, filtered_indices):
            batch_result = cls.compute_snapshots(solver, indices, solver.lag_dg, is_dg=True)
            F2_list.append(batch_result)

        F2 = np.hstack(F2_list)
        del F2_list
        gc.collect()

        # Calculate weight ratio based on norms
        norm_y = LA.norm(y_list)
        norm_F2 = LA.norm(F2)
        weight_ratio = (norm_y / norm_F2) if norm_F2 > 1e-12 else 1.0
        
        weights = np.eye(cls.nosc) * weight_ratio

        RB, sv, nosc_r = PSD(F2, y_list, cls)
        # POD(np.hstack([F2[:cls.nosc, :], y_list[:cls.nosc, :]]), np.eye(F2.shape[0]//2), cls.pod_tol)
        # RB, sv, _nosc_r = POD(np.hstack([F2, y_list]), np.eye(F2.shape[0]), cls.pod_tol)
        # nosc_r = _nosc_r//2

        del y_list
        gc.collect()

        print(f'{RB.shape = }')

        fig, ax = logplot(sv, xlabel=f'index of singular values', xlims=(1, len(sv)))
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

        results = []
        for solver, indices in zip(solvers, filtered_indices):
            result = cls.compute_snapshots(solver, indices, solver.lag_dg_z, is_dg=True, projection_matrix=IP)
            results.append(result)

        F3 = np.hstack(results)

        Uj, sv, _ = POD(F3, np.eye(F3.shape[0]), cls.pod_tol)
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