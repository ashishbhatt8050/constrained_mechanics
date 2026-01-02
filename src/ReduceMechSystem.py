import os
import gc
import numpy as np
import sympy as smp
from numpy import linalg as LA
from functools import partial
from System import MechSystem
from ODESolver import DiscreteGradient
from podDEIM import POD, PSD, DEIM
from PlotScript import logplot, save_figure

class ReduceMechSystem(MechSystem):
    """
    Reducer class extending MechSystem with reduction and hyper-reduction capabilities.
    
    Provides methods for computing reduced bases (POD), DEIM/MDEIM bases, 
    and setting up reduced/hyper-reduced models.
    """
    @classmethod
    def filter_solvers(cls, solvers, solver_type):
        """
        Filter solvers by type and return filtered solvers and indices.
        """
        # Filter solvers by type
        if solver_type == "Hamiltonian":
            filtered_solvers = [s for s in solvers if not isinstance(s, DiscreteGradient)]
        elif solver_type == "DiscreteGradient":
            filtered_solvers = [s for s in solvers if isinstance(s, DiscreteGradient)]
        else:
            raise ValueError(f"Invalid solver type: {solver_type}")

        # Create a mapping from solvers to their indices
        solver_to_indices = {solver: indices for solver, indices in zip(solvers, cls.indices_list)}

        # Filter indices_list based on filtered_solvers
        filtered_indices = [solver_to_indices[solver] for solver in filtered_solvers]
        return filtered_solvers, filtered_indices

    @classmethod
    def setup_reduced_model(cls, solvers, kwds):
        """
        Compute and setup the reduced basis (POD) for the system.

        Args:
            solvers (list): List of full-order solver instances.
            kwds (dict): Configuration dictionary containing registered solver classes.
        """
        print('Computing reduced basis...')

        solver_names = [s.__name__ for s in kwds['registered_solver_classes']]
        def compute_batch_snapshots(solver, indices, expr, batch_size=100):
            """Process snapshots in batches"""
            result = []
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                if isinstance(solver, DiscreteGradient):  # For DiscreteGradient
                    batch_results = [expr(y) for y in zip(solver.y[batch_indices], 
                                                        solver.y[np.array(batch_indices)+1])]
                else:  # For Hamiltonian
                    batch_results = [expr(y) for y in solver.y[batch_indices]]
                result.extend(batch_results)
                if (i + batch_size) % 100 == 0:
                    gc.collect()

            return np.array(result).T

        def compute_reduced_basis(solver_type, func_name, attr_name):
            filtered_solvers, filtered_indices = cls.filter_solvers(solvers, solver_type)

            # Create snapshot list
            y_list = np.hstack([solver.y[indices].T for solver, indices 
                                in zip(filtered_solvers, filtered_indices)])

            if True or solver_type == "Hamiltonian":
                # Create F2 in batches
                F2_list = []
                for solver, indices in zip(filtered_solvers, filtered_indices):
                    func = getattr(solver, func_name)
                    batch_result = compute_batch_snapshots(solver, indices, func)
                    F2_list.append(batch_result)

                F2 = np.hstack(F2_list)
                del F2_list
                gc.collect()

                # Calculate weight ratio based on norms
                norm_y = LA.norm(y_list)
                norm_F2 = LA.norm(F2)
                weight_ratio = (norm_y / norm_F2) if norm_F2 > 1e-12 else 1.0
                
                # Create a diagonal weight matrix for the PSD inner product
                # This implicitly scales the F2 contribution during SVD without altering data
                weights = np.eye(cls.nosc) * weight_ratio

                RB, sv, nosc_r = PSD(F2, y_list, cls) #, weights=weights)

            else:
                RB, sv, nosc_r = PSD(y_list[:, 0, None], y_list, cls)

            del y_list
            gc.collect()

            setattr(MechSystem, attr_name[0], RB)
            setattr(MechSystem, attr_name[1], nosc_r)
            print(f'{RB.shape = }')

            fig, ax = logplot(sv, xlabel=f'index of singular values of [F2, y_list]', xlims=(1, len(sv)))
            filename = os.path.join(
                MechSystem.data_folder, "osc_sv_rb" + solver_type + ".pdf"
            )
            save_figure(fig, filename, fig_data=None)

        # Compute reduced basis for Hamiltonian solvers
        if 'ConformalStormerVerletSolver' in solver_names \
            or 'ConformalImplicitMidpointSolver' in solver_names:
            compute_reduced_basis("Hamiltonian", "ham_z", ["RB", "nosc_r"])

        # Compute reduced basis for Discrete Gradient solvers if applicable
        if 'DiscreteGradientSolver' in solver_names:
            compute_reduced_basis("DiscreteGradient", "lag_dg", ["RB_dg", "nosc_r_dg"])

    @classmethod
    def _create_indexed_deim_func(cls, expr, P, solver_type):
        """
        Helper method to create a lambdified function using IndexedBase symbols for efficiency.
        """
        # 1. Define IndexedBase symbols
        y_base = smp.IndexedBase("y")
        y1_base = smp.IndexedBase("y1")
        omega2_base = smp.IndexedBase("omega2")

        # 2. Create substitution dictionary from the class's standard symbols
        subs_dict = {s: y_base[i] for i, s in enumerate(cls.y)}
        subs_dict.update({s: y1_base[i] for i, s in enumerate(cls.y1)})
        subs_dict.update({s: omega2_base[i] for i, s in enumerate(cls.omega2)})

        # 3. Substitute the expression to use IndexedBase symbols
        expr_to_lambdify = P.T @ expr.flat()
        expr_indexed = smp.Matrix(expr_to_lambdify).subs(subs_dict)

        # 4. Define arguments for lambdify and create the function
        y_symbols = set(cls.y)
        is_y_only = all(s in y_symbols for s in expr.free_symbols)

        if is_y_only:
            args = (y_base,)
        else:
            args = (
                (y_base, omega2_base, cls.beta)
                if solver_type == "Hamiltonian"
                else (y_base, y1_base, omega2_base)
            )
        deim_func = smp.lambdify(args, expr_indexed, modules=["numpy", "scipy"])

        return deim_func

    @classmethod
    def setup_hyperreduction(cls, kwds):
        """
        Compute and setup the hyper-reduction basis (DEIM/MDEIM).

        Args:
            kwds (dict): Configuration dictionary containing registered solver classes.
        """
        print('Setting up hyperreduction...')

        solver_names = [s.__name__ for s in kwds['registered_solver_classes']]
        def compute_hyperreduction_basis(solver_type, expr, attr, deim_attr_name, deim_func_name, expr_z=None, mdeim_func_name=None):
            attr_11 = attr[:cls.nosc, :cls.nosc_r] if solver_type == "Hamiltonian" else attr[:cls.nosc, :cls.nosc_r_dg]
            P11, _ = DEIM(attr_11, plot_deim=False)

            attr_22 = attr[cls.nosc:, cls.nosc_r:] if solver_type == "Hamiltonian" else attr[cls.nosc:, cls.nosc_r_dg:]
            P22, _ = DEIM(attr_22, plot_deim=False)

            P = np.block([
                [P11, np.zeros((P11.shape[0], P22.shape[1]))],
                [np.zeros((P22.shape[0], P11.shape[1])), P22]
            ])
            setattr(MechSystem, deim_attr_name, attr.T @ attr @ LA.inv(P.T @ attr))
            print(f'{P.shape = }')

            deim_func = cls._create_indexed_deim_func(expr, P, solver_type)

            if callable(deim_func) and not isinstance(deim_func, type):
                # Use staticmethod to avoid pickling issues with local lambdas
                setattr(MechSystem, deim_func_name, staticmethod(deim_func))
            else:
                print(f"Memory address of {deim_func_name}: {hex(id(deim_func))}")

            # Create and wrap extra deim function if provided
            if expr_z is not None:
                mdeim_func = smp.lambdify((cls.y, cls.omega2, cls.beta) if solver_type == "Hamiltonian" else (cls.y, cls.y1, cls.omega2),
                                        P.T @ expr_z,
                                        modules=['numpy', 'scipy'])

                if callable(mdeim_func) and not isinstance(mdeim_func, type):
                    setattr(MechSystem, mdeim_func_name, staticmethod(mdeim_func))
                else:
                    print(f"Memory address of {mdeim_func_name}: {hex(id(mdeim_func))}")

        # Update the calls to compute_hyperreduction_basis
        if 'ConformalStormerVerletSolver' in solver_names \
            or 'ConformalImplicitMidpointSolver' in solver_names:
            compute_hyperreduction_basis(
                "Hamiltonian", 
                cls.ham_z_expr, 
                cls.RB, 
                "RBxUx_inv_PxU", 
                "ham_z_deim",
                cls.ham_zz_expr if cls.hyperreducer == 'DEIM' else None,
                "ham_zz_deim" if cls.hyperreducer == 'DEIM' else None
            )

        if 'DiscreteGradientSolver' in solver_names:
            compute_hyperreduction_basis(
                "DiscreteGradient",
                cls.lag_dg_expr,
                cls.RB_dg,
                "_RBxUx_inv_PxU_",
                "lag_dg_deim",
                cls.lag_dg_z_expr if cls.hyperreducer == 'DEIM' else None,
                "lag_dg_z_deim" if cls.hyperreducer == 'DEIM' else None
            )

    @staticmethod
    def compute_snapshot(solver, indices, IP, is_dg=False, batch_size=32):
        """Memory efficient batch processing of matrix multiplications for snapshots."""
        result = []
        batch_size = len(indices)

        # Process indices in batches
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]

            # Pre-allocate batch matrices
            if is_dg:
                batch_mats = np.vstack([
                    solver.lag_dg_z((solver.y[idx], solver.y[idx+1])).flatten().astype(np.float32)
                    for idx in batch_indices
                ])
            else:
                batch_mats = np.vstack([
                    solver.ham_zz(solver.y[idx]).flatten().astype(np.float32)
                    for idx in batch_indices
                ])

            # Compute batch matrix multiplication
            batch_results = IP @ batch_mats.T  # More efficient than multiple small multiplications
            result.extend(batch_results.T)

            # Force garbage collection after each batch
            if (i + batch_size) % 100 == 0:
                gc.collect()

        return result

    @classmethod
    def update_mdeim_hyperreduction(cls, solvers, kwds):
        """Update system methods using MDEIM hyper-reduction."""

        solver_names = [s.__name__ for s in kwds['registered_solver_classes']]
        def compute_mdeim_basis(solver_type, func_name, expr, attr_name, deim_func_name):
            """Helper function to compute MDEIM basis and create lambdified functions"""
            # NOTE: symbolic matrix multiplication in high-precision arithmetic can become expensive
            # and may cause memory overflow. This can lead to a program crash.
            # Though _IP_Ux_inv_PxU_ @ lag_dg_z_col can be done here once and for all solvers.
            print(f'Computing {func_name} reduction...')

            # Collect snapshots, load non-zero indices stored in func_name_nonzero_indices
            # and create interpolation matrix IP
            non_zero_indices = getattr(cls, func_name + "_nonzero_indices")
            IP = np.zeros((len(non_zero_indices), (2*cls.nosc)**2), dtype=np.int8)
            IP[np.arange(len(non_zero_indices)), non_zero_indices] = 1

            # Filter solvers and collect snapshots
            filtered_solvers, filtered_indices = cls.filter_solvers(solvers, solver_type)

            # Create snapshot matrix sequentially
            if solver_type == "DiscreteGradient":
                worker_func = partial(ReduceMechSystem.compute_snapshot, IP=IP, is_dg=True)
            else:
                worker_func = partial(ReduceMechSystem.compute_snapshot, IP=IP, is_dg=False)

            results = []
            for solver, indices in zip(filtered_solvers, filtered_indices):
                result = worker_func(solver, indices)
                results.append(result)

            # # Use a ProcessPoolExecutor for parallelism
            # with concurrent.futures.ProcessPoolExecutor() as executor:
            #     futures = [
            #         executor.submit(worker_func, solver, indices)
            #         for solver, indices in zip(filtered_solvers, filtered_indices)
            #     ]
            #     results = [f.result() for f in concurrent.futures.as_completed(futures)]

            F3 = np.hstack([np.array(result).T for result in results])

            # Compute POD basis and plot singular values
            Uj, sv, _ = POD(F3, np.eye(F3.shape[0]), cls.tol)
            fig, ax = logplot(sv, xlabel=f'index of singular values of {func_name}', xlims=(1, len(sv)))
            filename = os.path.join(
                MechSystem.data_folder, "osc_sv_mdeim" + solver_type + ".pdf"
            )
            save_figure(fig, filename, fig_data=None)

            # Compute DEIM points and interpolation matrix
            Pj, _ = DEIM(Uj, plot_deim=False)
            setattr(MechSystem, attr_name, IP.T @ Uj @ LA.inv(Pj.T @ Uj))
            print(f'{getattr(MechSystem, attr_name).shape = }')

            # Create lambdified function
            mdeim_col = Pj.T @ IP @ expr.flat()
            mdeim_func = smp.lambdify((cls.y, cls.y1, cls.omega2) if solver_type == "DiscreteGradient" 
                                    else (cls.y, cls.omega2, cls.beta),
                                    mdeim_col, modules=['numpy', 'scipy'])

            if callable(mdeim_func) and not isinstance(mdeim_func, type):
                setattr(MechSystem, deim_func_name, staticmethod(mdeim_func))
            else:
                print(f"Memory address of {deim_func_name}: {hex(id(mdeim_func))}")

            print(f'{deim_func_name} has been updated for solver type: {solver_type}')

        # Compute MDEIM basis for DiscreteGradient solvers
        if 'DiscreteGradientSolver' in solver_names:
            compute_mdeim_basis(
                "DiscreteGradient",
                "lag_dg_z",
                cls.lag_dg_z_expr,
                "_IP_Ux_inv_PxU_",
                "lag_dg_z_mdeim"
            )

        # Compute MDEIM basis for Hamiltonian solvers
        if 'ConformalStormerVerletSolver' in solver_names \
            or 'ConformalImplicitMidpointSolver' in solver_names:
            compute_mdeim_basis(
                "Hamiltonian",
                "ham_zz",
                cls.ham_zz_expr,
                "IP_Ux_inv_PxU",
                "ham_zz_mdeim"
            )

    @classmethod
    def hyperreduce_constraints(cls, solvers, indices_list, solver_type):
        """Hyper-reduce system constraints using MDEIM."""
        print('Computing constraints reduction...')

        def compute_constraint_basis(func_name, expr, is_g_prime=False):
            """
            Helper function to compute MDEIM basis for constraints.
            Supports g, g_prime, g_prime_x_lambda_y, and g_prime_x_lambda_lambda.
            """
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
                setattr(MechSystem, func_name + 'shape' if solver_type == "Hamiltonian" else func_name + 'shape_dg', g_prime_shape)

                non_zero_indices = getattr(cls, func_name + "nonzero_indices")
                IP = np.zeros((len(non_zero_indices), np.prod(g_prime_shape)), dtype=np.int8)
                IP[np.arange(len(non_zero_indices)), non_zero_indices] = 1

                # --- Create snapshot matrix for prime constraints ---
                snapshot_list = []
                for solver, indices in zip(solvers, indices_list):
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
                    for solver, indices in zip(solvers, indices_list)
                ])

            # Compute POD basis and plot singular values
            Uj, sv, _ = POD(F, np.eye(F.shape[0]), cls.tol)
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
                if solver_type == "Hamiltonian":
                    setattr(MechSystem, f'{func_name}{"mdeim" if is_prime_type else "deim"}', staticmethod(mdeim_func))
                else:  # DiscreteGradient
                    setattr(MechSystem, f'{func_name}{"mdeim" if is_prime_type else "deim"}_dg', staticmethod(mdeim_func))
            else:
                print(f"Memory address of {func_name}{'mdeim' if is_prime_type else 'deim'}: {hex(id(mdeim_func))}")

            return basis, sv

        # Compute bases for g and g_prime
        # _Ux_inv_PxU, sv_g = compute_constraint_basis('g_', cls.g_expr)
        _IP_Ux_inv_PxU, sv_g_prime = compute_constraint_basis('g_prime_', cls.g_prime_expr, is_g_prime=True)

        # Set attributes based on solver type
        if solver_type == "Hamiltonian":
            # setattr(MechSystem, '_Ux_inv_PxU', _Ux_inv_PxU)
            # setattr(MechSystem, 'g_prime_mdeim', cls.g_prime_)
            setattr(MechSystem, '_IP_Ux_inv_PxU', _IP_Ux_inv_PxU)
            # setattr(MechSystem, 'IP_g_prime_x_lambda_y', IP_g_prime_x_lambda_y)
            # setattr(MechSystem, 'IP_g_prime_x_lambda_lambda', IP_g_prime_x_lambda_lambda)
        elif solver_type == "DiscreteGradient":
            # Also compute bases for cls.g_prime_x_lambda_y and cls.g_prime_x_lambda_lambda
            IP_g_prime_x_lambda_y, sv_g_prime_x_lambda_y = compute_constraint_basis(
                "g_prime_x_lambda_y_", cls.g_prime_x_lambda_y_expr, is_g_prime=True
            )
            IP_g_prime_x_lambda_lambda, sv_g_prime_x_lambda_lambda = (
                compute_constraint_basis(
                    "g_prime_x_lambda_lambda_",
                    cls.g_prime_x_lambda_lambda_expr,
                    is_g_prime=True,
                )
            )

            # setattr(MechSystem, '_Ux_inv_PxU_dg', _Ux_inv_PxU)
            setattr(MechSystem, '_IP_Ux_inv_PxU_dg', _IP_Ux_inv_PxU)
            setattr(MechSystem, 'IP_g_prime_x_lambda_y_dg', IP_g_prime_x_lambda_y)
            setattr(MechSystem, 'IP_g_prime_x_lambda_lambda_dg', IP_g_prime_x_lambda_lambda)
        else:
            raise ValueError(f"Invalid solver type: {solver_type}")

        # Plot singular values for g, g_prime, g_prime_x_lambda_y and g_prime_x_lambda_lambda using logplot, pass the singular values and xlims as a list
        if "sv_g" in locals():
            sv_list = [sv_g, sv_g_prime]
            xlims_list = [(1, len(sv_g)), (1, len(sv_g_prime))]
        else:
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
            MechSystem.data_folder, "osc_sv_constraints" + solver_type + ".pdf"
        )
        save_figure(fig, filename, fig_data=None)

        print('Constraints reduction complete.')