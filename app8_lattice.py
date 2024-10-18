#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 2024

@author: ashishbhatt

This app solves a full order and reduced-order model from the System class,
using solver classes in the ODESolver hierarchy of methods,
fixed point iterators from the Newton.py,
and bases from podDEIM.
"""

import os
import gc
from datetime import datetime

import numpy as np
import sympy as smp
from numpy import linalg as LA
from pylab import  log, r_, c_, zeros, eye, sqrt, reshape, linspace, roll, figure, zeros_like

from pathos.pools import _ProcessPool as Pool
import multiprocessing

import dill as pickle
from functools import wraps

import matplotlib.pyplot as plt
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator

# Configuration
rc('text', usetex=True)  # Enable LaTeX rendering
rc('text.latex', preamble=r'\usepackage{amsfonts}')  # Load AMSFonts for Fraktur

import ODESolver
from Newton import fixed_point
from System import System
from podDEIM import POD, DEIM
from PlotScript import plot_data, tex_table, logplot, save_figure, timing


def compose_solver_solves(func):
    @wraps(func)
    def wrapper(self, y_, k):
        for w_val in self.w_values:
            y_, y_full_ = func(self, w_val, y_, k)
                
        return y_, y_full_
    return wrapper

class MechSystem(System):
    """ Class of MechSystem methods """

    "Numerical solver and its properties"
    w_values = [0.28, 0.62546642846767004501]
    w_values.append(1.0 -2.0*(sum(w_values)))
    w_values.append(w_values[1])
    w_values.append(w_values[0])
    w_values = [1]
    assert np.isclose(sum(w_values), 1), 'sum_i w_i must be 1'

    "Fixed-point nonliner equations solver properties"
    tol, M, var, store = 1.0E-12, 100, True, False

    "System parameters"
    nosc = 50*3
    assert nosc//3 % 2 == 0, 'nosc//3 must be even'

    # These two properties only have effect during reduction
    reducer = 'psd'
    predict = True # False = reproduce
    hyperreducer = 'MDEIM'

    registered_solver_classes = [ODESolver.ConformalStormerVerlet]
    dt_space_dim = 1
    Omega2_space_dim = 10
    beta = (max(1e-2, 0*np.random.rand()/10))*0
    JJ = lambda self, d=nosc: r_[c_[zeros((d,d)), eye(d)], c_[-eye(d), zeros((d,d))]]

    Omega2_space = np.sort(10*(1 -np.random.rand(Omega2_space_dim, nosc//3-2)))
    # Omega2_space[1:] = Omega2_space[0]
    # Omega2_space[:,:3] = np.sort(np.random.uniform(Omega2_space[:,0], Omega2_space[:,2], (3, Omega2_space_dim)), axis=0).T

    dt_space = linspace(0.01, 0.05, num=dt_space_dim)
    T_final = 2

    keep_time = datetime.now().strftime('%Y-%m-%d_%H-%M_')

    "Initial conditions satisfying the constraints"
    # positions = np.zeros((nosc//3, 3))

    # for i in range(nosc//3):
    #     positions[i, 0] = i % 2 + 0*(i // 2 % 2) * 1e-1
    #     positions[i, 1] = i // 2 + 0*(-1)**(i // 2) * 1e-1
    #     positions[i, 2] = 0
        
    # Create a tensor to store the positions
    i = np.arange(nosc//3)
    positions = np.stack((i % 2 + 0*(i // 2 % 2) * 1e-1, i // 2 + 0*(-1)**(i // 2) * 1e-1, np.zeros_like(i)), axis=1)

    positions = positions.flatten().reshape(-1, 1)

    momenta = np.zeros((nosc//3, 3))
    momenta[::2] = np.random.uniform(-0.01, 0.01, momenta[::2].shape)
    momenta[1::2] = momenta[::2]
    assert np.allclose(momenta[1::2] - momenta[::2], 0), 'position and momenta are not orthogonal'
    momenta = momenta.flatten().reshape(-1, 1)

    constraint_type = 'spherical'
    constraints_reduce = True # True: Reduce constraint jacobian g_prime

    y_init = r_[positions, momenta].flatten()

    drag = lambda self, x, u: 0 #beta/2 * r_[x, u]
    drag_z = lambda self, x, u: 0 #beta/2 * eye(2*x.shape[0])

    def __init__(self, kwds):
        "MechSystem properties"

        System.__init__(self, kwds)

        "Projection matrices"
        if hasattr(self, 'RB'):
            self.y_init = self.RB.T @ self.y_init

            if self.reducer == 'pod':
                self.JJ_r = self.RB.T @ self.JJ() @ self.RB
            elif self.reducer == 'psd':
                self.JJ_r = self.JJ(self.nosc_r)
        else:
            self.RB = None

        if not hasattr(self, 'P'):
            self.P = None
            self.U = None

        if self.solver_class in MechSystem.registered_solver_classes:
            self.solver = self.solver_class(self)
        else:
            NameError(f'Unknown {self.solver_class.__name__ = }')

        "various measures"
        self.time_lapsed = []

    "y_init alias"
    @property
    def u_init(self):
        return self.y_init

    @u_init.setter
    def u_init(self, value):
        self.y_init = value

    @staticmethod
    def solve_mech_system(solver_class, dt, Omega2, kwds):
        kwds['pool'] = {'solver_class': solver_class, 'dt': dt, 'Omega2': Omega2, \
                        'n': int(round(MechSystem.T_final/dt))}
        kwds['pool'].update({'t_points': linspace(0, MechSystem.T_final, kwds['pool']['n']+1)})

        MSsolver = MechSystem(kwds)
        tl = MSsolver.solve()
        MSsolver.time_lapsed.append(tl)

        if not MSsolver.beta:
            MSsolver.eng_error = MSsolver.get_en_err()
            MSsolver.en_error = sqrt(dt)*LA.norm(MSsolver.eng_error)
        else:
            MSsolver.en_error = None

        if MSsolver.var:
            MSsolver.var_solve()
    
        # print("Number of CPUs:", multiprocessing.cpu_count())
        # print("Number of threads:", multiprocessing.active_children())

        return MSsolver

    @staticmethod
    def parallel_solve_mech_system(kwds, Omega2_space, MSsolvers):

        with Pool() as pool:

            results = pool.starmap(MechSystem.solve_mech_system, \
                                   [(x, y, z, kwds) for x in MechSystem.registered_solver_classes \
                                    for y in MechSystem.dt_space for z in Omega2_space])

        if 'pool' in kwds:
            del kwds['pool']
            print("kwds['pool'] deleted")

        # rearrange results in the order of submitted jobs
        for x, y, z in [(x, y, z) for x in MechSystem.registered_solver_classes \
                        for y in MechSystem.dt_space for z in Omega2_space]:
            for MSsolver in results:
                if MSsolver.solver_class == x and MSsolver.dt == y and (MSsolver.Omega2 == z).all():
                    MSsolvers.append(MSsolver)
                    break


    @staticmethod
    def solver(kwds):

        MSsolvers = []

        if MechSystem.predict: # prediction experiment

            if not hasattr(MechSystem, 'RB'):
                # training parameters
                Omega2_space = MechSystem.Omega2_space[:-1]
                Omega2_space_dim = len(Omega2_space)

            else:
                # testing parameters
                Omega2_space = MechSystem.Omega2_space[-1:]
                Omega2_space_dim = len(Omega2_space)

        else: # reproduction
            Omega2_space = MechSystem.Omega2_space
            Omega2_space_dim = len(Omega2_space)

        MechSystem.parallel_solve_mech_system(kwds, Omega2_space, MSsolvers)

        MechSystem.measures(MSsolvers, Omega2_space_dim)

        return MSsolvers

    @compose_solver_solves
    def solve_for_w(self, w_val, y_, k):
        self.solver.set_initial_condition(y_[1])
        y_, _, info_ = self.solver.solve(w_val*self.t_points[k:k+2])
        if self.store: self.info.append(np.array(info_[0::1]))

        # enforce constraints
        if self.constraint_type:
    
            if self.RB is not None:
                y_full_ = y_ @ self.RB.T
            else:
                y_full_ = y_
    
            fixed_point(self.g, y_full_, self.g_prime, self.tol, self.M, False)
            
            if self.RB is not None:
                y_ = y_full_ @ self.RB

        return y_, y_full_


    @timing
    def solve(self):

        if self.RB is None:
            self.y = np.zeros((self.n+1, 2*self.nosc))
        else:
            self.y = np.zeros((self.n+1, 2*self.nosc_r))
            self.y_full = np.zeros((self.n+1, 2*self.nosc))

            # y_ = np.array([self.y_init, self.y_init]) @ self.RB.T

            # if self.constraint_type:
            #     fixed_point(self.g, y_, self.g_prime, self.tol, self.M, False)

            self.y_full[0] = MechSystem.y_init
            # self.y_init = self.RB.T @y_[1]

        self.y[0] = self.y_init
        if self.store: self.info = []

        for k in range(self.n):
            if self.store: self.info.append(self.y[k])
            y_ = np.array([self.y[k], self.y[k]])
            
            # if self.RB is not None:
            #     y_full_ = np.array([self.y_full[k], self.y_full[k]])
            # else:
            #     y_full_ = y_

            y_, y_full_ = self.solve_for_w(y_, k)
            
            self.y[k+1] = y_[-1]

            # if self.RB is None:
            #     self.y[k+1] = y_[-1]
            # else:
            #     self.y[k+1] = y_[-1]
            if self.RB is not None:
                self.y_full[k+1] = y_full_[-1]

        '''
        for k in range(self.n):
            if self.store: self.info.append(self.y[k])
            y_ = np.array([self.y[k], self.y[k]])

            for w_val in self.w_values:
                self.solver.set_initial_condition(y_[1])
                y_, _, info_ = self.solver.solve(w_val*self.t_points[k:k+2])
                if self.store: self.info.append(np.array(info_[0::1]))

                # enforce constraints
                if self.constraint_type:

                    if self.RB is not None:
                        y_ = y_ @ self.RB.T

                    fixed_point(self.g, y_, self.g_prime, self.tol, self.M, False)

            if self.RB is None or self.constraint_type is None:
                self.y[k+1] = y_[1]
            else:
                self.y_full[k+1] = y_[1]
                self.y[k+1] = self.RB.T @y_[1]
        '''

        if self.store:
            self.info.append(self.y[k+1])
            self.info = np.vstack(self.info)

        if self.RB is not None:
            self.y_red = self.y
            self.y = self.y_full
            
    def var_solve(self):
    
        if hasattr(self, 'y_red'):
            y = self.y_red
            nosc = self.y_red.shape[1]//2
        else:
            y = self.y
            nosc = self.nosc

        dpsi = np.zeros((2, 2*nosc, 2*nosc))
        dpsi[0] = np.eye(2*nosc)
        self.sym_error = np.zeros(self.n+1)

        for k in range(self.n):
            dpsi_, _ = self.solver.var_solve(y[k:k+2], self.t_points[k:k+2])
            dpsi[1] = dpsi_[-1]
            sym_error_ =self.solver.symplectic_error(dpsi, self.t_points[k:k+2])
            self.sym_error[k+1] = sym_error_[-1]

    def plot(self):
        "plot the results"

        fig = figure()
        fig.tight_layout(pad=0)
        fig.suptitle(f'integrator = {self.solver_class.__name__}, dt = {self.dt}')

        gs = fig.add_gridspec(6, 2, hspace=1)
        ax0, ax1, ax2, ax3, ax4, ax5 = [fig.add_subplot(gs[i, 0]) for i in [0, 1, 2, 3, 4, 5]]
        ax6 = fig.add_subplot(gs[:6, -1], projection='3d')

        if hasattr(self, 'sym_error'):
            plot_data(ax0, self.t_points, self.sym_error, xlims=(0, self.T_final), \
                      ylabel=r'$\Delta Sp$', margins=1)
                
            if max(abs(self.sym_error)) < 1e-15:
                ax0.set_ylim([-1e-15, 1e-15])

        if hasattr(self, 'g'):
            temp = np.vstack([self.g(y) for y in self.y])

            g_norm = LA.norm(temp, axis=1)
            plot_data(ax1, self.t_points, g_norm, xlims=(0, self.T_final), \
                      ylabel=r'$\Delta \mathfrak{P}$', margins=1)

        if hasattr(self, 'eng_error'):
            plot_data(ax2, self.t_points, self.eng_error, xlims=(0, self.T_final), \
                      ylabel=r'$\Delta H$', margins=1)

        lin_momentum = np.sum(self.y[:, nosc:].reshape(-1, nosc//3, 3), axis=1)
        lim_momentum_err = r_[0, LA.norm(lin_momentum[1:] - lin_momentum[0], axis=1)]

        angular_momentum = np.cross(self.y[:, :nosc].reshape(-1, nosc//3, 3), \
                                    self.y[:, nosc:].reshape(-1, nosc//3, 3))
        angular_momentum_sum = np.sum(angular_momentum, axis=1)
        angular_momentum_err = r_[0, \
                                  LA.norm(angular_momentum_sum[1:] - angular_momentum_sum[0], axis=1)]

        plot_data(ax3, self.t_points, lim_momentum_err, xlims=(0, self.T_final), \
                  ylabel=r'$\Delta L$', margins=0.5)

        plot_data(ax4, self.t_points, angular_momentum_err, xlims=(0, self.T_final), \
                  ylabel=r'$\Delta J$', margins=1)

        # Plot matrix condition number
        if self.RB is not None:
            temp = lambda t: self.g_prime(self.y[t])[:,:nosc] @ self.RB[:nosc,:self.nosc_r]
            mat = lambda t: temp(t) @ self.ham_zz(*np.split(self.y[t],2))[self.nosc_r:,self.nosc_r:] @ temp(t).T
        else:
            temp = lambda t: self.g_prime(self.y[t])[:,:nosc]
            mat = lambda t: temp(t) @ self.ham_zz(*np.split(self.y[t],2))[nosc:,nosc:] @ temp(t).T

        # self.mat_cond = [LA.cond(mat(t), 2) for t in range(self.n)]
        self.mat_cond = LA.cond(np.array([mat(t) for t in range(self.n)]), 2)

        plot_data(ax5, self.t_points[2:], self.mat_cond[1:], xlims=(0, self.T_final), \
                  ylabel=r'$\kappa$', margins=0.5)

        ax5.set_xlabel('time')

        # Plot the particle positions over time
        coords = self.y[:, :nosc].reshape(-1, nosc//3, 3)
        for i in range(coords.shape[1]):
            ax6.plot(coords[:, i, 0], coords[:, i, 1], coords[:, i, 2], 'k-')  # plot the trajectory of each particle
            ax6.scatter(coords[-1, i, 0], coords[-1, i, 1], coords[-1, i, 2], s=20)  # plot the final position of each particle

        # Set the axes' labels and title
        ax6.set_xlabel('x')
        ax6.set_ylabel('y')
        ax6.set_zlabel('z')
        ax6.set_title('Phase portrait')

        for ax in [ax0, ax1, ax2, ax3, ax4, ax5]:
            ax.label_outer()

        if self.RB is not None:
            if self.predict: string = '_predict'
            else: string = '_repro'
        else:
            string = '_full'

        filename = self.keep_time +'osc_' +self.solver_class.__name__ +string +'.pdf'
        # save_figure(fig, filename)

    @staticmethod
    def measures(MSsolvers, Omega2_space_dim):
        "Various measurements based on the solution"

        r_form = lambda numer, denom: r_[float('nan'), (log(roll(numer, -1)/numer)/log(roll(denom, -1)/denom))[:-1]]
        # C_form = lambda numer, denom, r: numer[:-1]/(denom[:-1]**r)
        # en_error = [MSsolvers[i].en_error for i in range(len(MSsolvers))]
        en_error = [solver.en_error for solver in MSsolvers]

        for i in range(0, len(MSsolvers), MechSystem.dt_space_dim * Omega2_space_dim):
            
            if len(en_error) > 1 and en_error[0] is not None and MechSystem.dt_space_dim > 1:
                "Compute convergence rates from the error in Energy"
                r_values = r_form(en_error[i:i+MechSystem.dt_space_dim*Omega2_space_dim:Omega2_space_dim], MechSystem.dt_space)
                # C_values.append(C_form(en_error[i:i+MechSystem.dt_space_dim*Omega2_space_dim:Omega2_space_dim], MechSystem.dt_space, r_values[-1]))

                # print(f'{r_values =}')
                # Display convergence rates if available
                temp = c_[MechSystem.dt_space, r_values].T
                tex_table(MSsolvers[i].solver_class.__name__, temp)

            # Plot the measures
            MSsolvers[i].plot()


#%% Find symbolic quantities

nosc = MechSystem.nosc

# create the 'data' subfolder if it doesn't exist
data_folder = 'data'
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

# try to load the expressions from disk
filename = os.path.join(data_folder, f"ham_expr_{nosc}.pickle")
try:
    with open(filename, 'rb') as f:
        loaded_expressions = pickle.load(f)
    print("Loaded Hamiltonian expressions from disk.")

    ham_expr = loaded_expressions['ham_expr']
    ham_z_expr = loaded_expressions['ham_z_expr']
    ham_zz_expr = loaded_expressions['ham_zz_expr']
    _ham_z_ = loaded_expressions['_ham_z_']
    ham_ = loaded_expressions['ham_']
    ham_z_ = loaded_expressions['ham_z_']
    ham_zz_ = loaded_expressions['ham_zz_']
    q = loaded_expressions['q']
    p = loaded_expressions['p']
    omega2 = loaded_expressions['omega2']
    beta = loaded_expressions['beta']
    y = loaded_expressions['y']

except FileNotFoundError:
    print("Hamiltonian expressions not found on disk. Computing and saving them...")

    q = smp.Matrix(smp.symbols('q_:{}_:{}'.format(nosc//3,3), real=True)).reshape(nosc//3,3)
    p = smp.Matrix(smp.symbols('p_:{}_:{}'.format(nosc//3,3), real=True))
    omega2 = smp.Matrix(smp.symbols('omega^2_0:{}'.format(nosc//3-2), real=True))
    beta = smp.symbols('beta', real=True)

    kin_expr = 0.5 * p.dot(p)

    pi = smp.Matrix([(q.row(i+2) - q.row(i)).norm(2)**2 for i in range(nosc//3-2)])
    pot_expr = 0.5 * omega2.dot((pi -smp.ones(nosc//3-2,1)).applyfunc(lambda x: x**2))

    q = q.reshape(nosc,1)
    y = list(q)+list(p)

    ham_expr = kin_expr + pot_expr
    ham_z_expr = smp.Matrix([ham_expr]).jacobian(y).T
    ham_zz_expr = ham_z_expr.jacobian(y)

    ham_ = smp.lambdify((q, p, omega2, beta), ham_expr, 'numpy')
    _ham_z_ = smp.lambdify((q, p, omega2, beta), ham_z_expr, 'numpy')
    ham_zz_ = smp.lambdify((q, p, omega2, beta), ham_zz_expr, 'numpy')

    ham_z_ = lambda q, p, omega2, beta: _ham_z_(q, p, omega2, beta).squeeze()

    expressions = {
        "ham_expr": ham_expr,
        "ham_z_expr": ham_z_expr,
        "ham_zz_expr": ham_zz_expr,
        "_ham_z_": _ham_z_,
        "ham_": ham_,
        "ham_z_": ham_z_,
        "ham_zz_": ham_zz_,
        "q": q, "p": p, "omega2": omega2, "beta": beta, "y": y,
    }

    # save the expressions to disk
    with open(filename, 'wb') as f:
        pickle.dump(expressions, f)
    print("Hamiltonian expressions saved to disk.")

if MechSystem.constraint_type is not None:

    # try to load the expressions from disk
    filename = os.path.join(data_folder, f"g_expr_{nosc}.pickle")
    try:
        with open(filename, 'rb') as f:
            loaded_expressions = pickle.load(f)
        print("Loaded constraints from disk.")

        g_expr = loaded_expressions['g_expr']
        g_prime_expr = loaded_expressions['g_prime_expr']
        _g_lam = loaded_expressions['_g_lam']
        g_lam = loaded_expressions['g_lam']
        g_prime_lam = loaded_expressions['g_prime_lam']

    except FileNotFoundError:
        print("Constraints not found on disk. Computing and saving them...")

        q = q.reshape(nosc//3,3)
        p = p.reshape(nosc//3,3)
        row_diffs_q = [q.row((i+1)) - q.row(i) for i in range(0, nosc//3, 2)]
        row_diffs_p = [p.row((i+1)) - p.row(i) for i in range(0, nosc//3, 2)]
        row_norms = [(row_diff.dot(row_diff) -1)/2 for row_diff in row_diffs_q]
        ddt_row_norms = [row_diffs_q[i].dot(row_diffs_p[i]) for i in range(len(row_diffs_q))]

        q = q.reshape(nosc,1)
        p = p.reshape(nosc,1)
        y = list(q)+list(p)
        g_expr = smp.Matrix(row_norms+ ddt_row_norms)
        g_prime_expr = g_expr.jacobian(y)

        _g_lam = smp.lambdify((y,), g_expr, modules=['numpy'])
        g_prime_lam = smp.lambdify((y,), g_prime_expr, modules=['numpy'])

        g_lam = lambda x: _g_lam(x).squeeze()

        expressions = {
            "g_expr": g_expr,
            "g_prime_expr": g_prime_expr,
            "_g_lam": _g_lam,
            "g_lam": g_lam,
            "g_prime_lam": g_prime_lam
        }

        # save the expressions to disk
        with open(filename, 'wb') as f:
            pickle.dump(expressions, f)
        print("Constraints saved to disk.")

#%% Main driver
if __name__ == '__main__':
    "Model order reduction of the MechSystem using MechSystem"

#%% Full order solution
    print('Computing full solution ...')

    kwds = {'ham_': ham_, \
            '_ham_z_': _ham_z_, \
            'ham_z_': ham_z_, \
            'ham_zz_': ham_zz_, \
            '_g_lam': _g_lam, \
            'g': g_lam, \
            'g_prime': g_prime_lam, \
            }

    MSsolvers = MechSystem.solver(kwds)

    if MSsolvers[-1].non_quad: # is not None
        assert MechSystem.Omega2_space_dim == 1

    if MechSystem.predict:
        array_shape = (len(MechSystem.registered_solver_classes), MechSystem.dt_space_dim, MechSystem.Omega2_space_dim-1)
    else:
        array_shape = (len(MechSystem.registered_solver_classes), MechSystem.dt_space_dim, MechSystem.Omega2_space_dim)

    # time_lapsed = []
    # for x in MSsolvers:
    #     time_lapsed.append(x.time_lapsed)
    # reshape(time_lapsed, array_shape)

    time_lapsed = [reshape([x.time_lapsed for x in MSsolvers], array_shape)]

#%% Reduced order solution
    # print('Assembling snapshots ...')
    # y_list = np.hstack([MSsolver.y[np.unique(np.random.randint(0,MSsolver.n,int(MSsolver.n)//1))].T for MSsolver in MSsolvers])
    # print(f'{y_list.shape = }')
    # F2 = [np.array([MSsolver.ham_z(*np.split(y, 2)) for y in MSsolver.y]) for MSsolver in MSsolvers]
    # F2 = np.hstack([F2[i][np.unique(np.random.randint(0,MSsolvers[i].n,int(MSsolvers[i].n)//1))].T for i in range(len(MSsolvers))])
    # print(f'{F2.shape = }')
    
    print('Assembling snapshots ...')
    n_samples_list = [int(MSsolver.n // 1) for MSsolver in MSsolvers]
    indices_list = [np.random.choice(MSsolver.n, n_samples, replace=False) for MSsolver, n_samples in zip(MSsolvers, n_samples_list)]
    y_list = np.hstack([MSsolver.y[indices].T for MSsolver, indices in zip(MSsolvers, indices_list)])
    print(f'{y_list.shape = }')
    
    F2 = np.hstack([np.array([MSsolver.ham_z(*np.split(y, 2)) for y in MSsolver.y[indices]]).T for MSsolver, indices in zip(MSsolvers, indices_list)])
    print(f'{F2.shape = }')

    X = {#'Q_half': MSsolvers[-1].Q_spd()[:nosc, :nosc], \
         # 'sqrt': sp.linalg.sqrtm(MSsolvers[-1].Q_spd()), \
         'eye': np.eye(2*nosc), \
         'eye_half': np.eye(nosc)}

    if MechSystem.reducer == 'pod':
        RBq, sv_pod_q, nosc_q = POD(c_[y_list[:nosc,:], F2[:nosc,:]], X['eye_half'], MechSystem.tol)
        RBp, sv_pod_p, nosc_p = POD(c_[y_list[nosc:,:], F2[nosc:,:]], X['eye_half'], MechSystem.tol)

        nosc_r = max(nosc_q, nosc_p)

        RBq = RBq[:, :nosc_r]
        RBp = RBp[:, :nosc_r]
        RB = r_[c_[RBq, zeros_like(RBq)],\
                c_[zeros_like(RBp), RBp]]

        sv = c_[sv_pod_q, sv_pod_p].T

    elif MechSystem.reducer == 'psd':
        RB, sv, nosc_r = POD(c_[y_list[:nosc,:], y_list[nosc:,:], F2[:nosc, :], F2[nosc:,:]], X['eye_half'], MechSystem.tol)

        RB = RB[:, :nosc_r]
        RB = r_[c_[RB, zeros_like(RB)],\
                c_[zeros_like(RB), RB]]

    print(f'{2*nosc_r = }')

    fig, ax = logplot(sv, xlabel='index of singular values', xlims=(1, len(sv)))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    filename = MechSystem.keep_time +'osc_sv' + '.pdf'
    # save_figure(fig, filename)

    del y_list, F2
    gc.collect()

    MechSystem.RB = RB
    MechSystem.nosc_r = nosc_r

    kwds.update({'ham_z_': lambda q, p, omega2, beta: RB.T @ ham_z_(q, p, omega2, beta),\
                 'ham_zz_': lambda q, p, omega2, beta: RB.T @ ham_zz_(q, p, omega2, beta) @ RB})

    print('solving reduced system ...')
    MSsolvers_r = MechSystem.solver(kwds)

    # if len(MSsolvers) == len(MSsolvers_r):
    #     print('solution errors')
    #     print(reshape([np.amax(abs(MSsolvers[i].y -MSsolvers_r[i].y)) for i in range(len(MSsolvers))], array_shape))

    #     time_lapsed.append(reshape([x.time_lapsed for x in MSsolvers_r],\
    #                                 time_lapsed[0].shape)/time_lapsed[0]*100)
    # else:
    #     time_lapsed.append(reshape([x.time_lapsed for x in MSsolvers_r],\
    #                                 time_lapsed[0].shape[:2]))
            
    if len(MSsolvers) == len(MSsolvers_r):
        print('solution errors')
        print(reshape([np.amax(abs(y1 - y2)) for y1, y2 in zip(MSsolvers, MSsolvers_r)], array_shape))
    
        time_lapsed.append(reshape([x.time_lapsed for x in MSsolvers_r], time_lapsed[0].shape) / time_lapsed[0] * 100)
    else:
        time_lapsed.append(reshape([x.time_lapsed for x in MSsolvers_r], time_lapsed[0].shape[:2]))

            
    1/0

#%% Hyper-reduced model

    P, _ = DEIM(RB, plot_deim=False)

    # MechSystem.U = RB
    MechSystem.P = P

    PxU_inv_ = LA.inv(P.T @ RB)

    Pxham_z_ = smp.lambdify((q, p, omega2, beta), P.T @ ham_z_expr.flat(), modules=['scipy'])

    kwds.update({'ham_z_': lambda q, p, omega2, beta: PxU_inv_ @ Pxham_z_(q, p, omega2, beta)})

    if MechSystem.hyperreducer == 'DEIM':
        Pxham_zz_ = smp.lambdify((q, p, omega2, beta), P.T @ ham_zz_expr @ RB, modules=['scipy'])
        kwds.update({'ham_zz_': lambda q, p, omega2, beta: PxU_inv_ @ Pxham_zz_(q, p, omega2, beta)})

    elif MechSystem.hyperreducer == 'MDEIM':

        non_zero_indices = np.nonzero(MSsolvers[0].ham_zz(*np.split(MSsolvers[0].y[0], 2)).flatten())[0]

        IP = np.zeros((len(non_zero_indices), (2*nosc)**2))

        # for i, val in enumerate(non_zero_indices):
        #     IP[i, val] = 1
            
        IP[np.arange(len(non_zero_indices)), non_zero_indices] = 1

        # F3 = [np.array([IP @ MSsolver.ham_zz(*np.split(y, 2)).flatten() for y in MSsolver.y]) for MSsolver in MSsolvers]

        # F3 = np.hstack([F3[i][np.unique(np.random.randint(0,MSsolvers[i].n,int(MSsolvers[i].n)//1))].T for i in range(len(MSsolvers))])
        # print(f'{F3.shape = }')
        
        # n_samples_list = [int(MSsolver.n // 1) for MSsolver in MSsolvers]
        # indices_list = [np.random.choice(MSsolver.n, n_samples, replace=False) for MSsolver, n_samples in zip(MSsolvers, n_samples_list)]
        
        F3 = np.hstack([np.array([IP @ MSsolver.ham_zz(*np.split(y, 2)).flatten() for y in MSsolver.y[indices]]).T for MSsolver, indices in zip(MSsolvers, indices_list)])
        print(f'{F3.shape = }')

        Uj, sv, mj = POD(F3, np.eye(F3.shape[0]), MSsolvers[0].tol)
        del F3

        fig, ax = logplot(sv, xlabel='index of singular values', xlims=(1, len(sv)))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        filename = MechSystem.keep_time +'osc_mdeim_sv' + '.pdf'
        # save_figure(fig, filename)

        Pj, _ = DEIM(Uj, plot_deim=False)
        # Sj = Pj.T @ RB

        _IP_UxPxU_inv = IP.T @ Uj @ LA.inv(Pj.T @ Uj)

        ham_zz_col = Pj.T @ IP @ ham_zz_expr.reshape((2*nosc)**2, 1)

        Pxham_zz_ = smp.lambdify((q, p, omega2, beta), ham_zz_col, modules=['scipy'])

        kwds.update({'ham_zz_': lambda q, p, omega2, beta: RB.T @ np.reshape(_IP_UxPxU_inv @ Pxham_zz_(q, p, omega2, beta), (2*nosc, 2*nosc)) @ RB})

    if MechSystem.constraint_type is not None and MechSystem.constraints_reduce:

        # F5 = [np.array([MSsolver.g(y) +0.5 for y in MSsolver.y]) for MSsolver in MSsolvers]
        # F5 = np.hstack([F5[i][np.unique(np.random.randint(0,MSsolvers[i].n,int(MSsolvers[i].n)//1))].T for i in range(len(MSsolvers))])
        # print(f'{F5.shape = }')
        
        # n_samples_list = [int(MSsolver.n // 1) for MSsolver in MSsolvers]
        # indices_list = [np.random.choice(MSsolver.n, n_samples, replace=False) for MSsolver, n_samples in zip(MSsolvers, n_samples_list)]
        
        # g_vec = np.vectorize(lambda MSsolver, y: MSsolver.g(y) + 0.5)
        F5 = np.hstack([np.array([MSsolver.g(y) + 0 for y in MSsolver.y[indices]]).T for MSsolver, indices in zip(MSsolvers, indices_list)])
        print(f'{F5.shape = }')

        Uj, sv, mj = POD(F5, np.eye(F5.shape[0]), MechSystem.tol)
        del F5

        Pj, _ = DEIM(Uj, plot_deim=False)

        fig, ax = logplot(sv, xlabel='index of singular values', xlims=(1, len(sv)))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        filename = MechSystem.keep_time +'osc_mdeim_sv' + '.pdf'

        _UxPxU_inv = Uj @ LA.inv(Pj.T @ Uj)

        _g_col = Pj.T @ g_expr

        _Pxg = smp.lambdify((y,), _g_col, modules=['scipy'])

        kwds.update({'g': lambda y: (_UxPxU_inv @ _Pxg(y) -0).squeeze()})

        non_zero_indices = np.nonzero(MSsolvers[0].g_prime(MSsolvers[0].y[0]).flatten())[0]

        g_prime_shape= MSsolvers[0].g_prime(MSsolvers[0].y[0]).shape
        IP = np.zeros((len(non_zero_indices), np.prod(g_prime_shape)))

        # for i, val in enumerate(non_zero_indices):
        #     IP[i, val] = 1
            
        IP[np.arange(len(non_zero_indices)), non_zero_indices] = 1

        # F4 = [np.array([IP @ MSsolver.g_prime(y).flatten() for y in MSsolver.y]) for MSsolver in MSsolvers]

        # F4 = np.hstack([F4[i][np.unique(np.random.randint(0,MSsolvers[i].n,int(MSsolvers[i].n)//1))].T for i in range(len(MSsolvers))])
        # print(f'{F4.shape = }')
        
        # n_samples_list = [int(MSsolver.n // 1) for MSsolver in MSsolvers]
        # indices_list = [np.random.choice(MSsolver.n, n_samples, replace=False) for MSsolver, n_samples in zip(MSsolvers, n_samples_list)]
        
        F4 = np.hstack([np.array([IP @ MSsolver.g_prime(y).flatten() for y in MSsolver.y[indices]]).T for MSsolver, indices in zip(MSsolvers, indices_list)])
        print(f'{F4.shape = }')

        Uj, sv, mj = POD(F4, np.eye(F4.shape[0]), MechSystem.tol)
        del F4

        fig, ax = logplot(sv, xlabel='index of singular values', xlims=(1, len(sv)))
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        filename = MechSystem.keep_time +'osc_mdeim_sv' + '.pdf'
        # save_figure(fig, filename)

        Pj, _ = DEIM(Uj, plot_deim=False)
        # Sj = Pj.T @ RB

        IP_UxPxU_inv_ = IP.T @ Uj @ LA.inv(Pj.T @ Uj)

        g_prime_col = Pj.T @ IP @ g_prime_expr.reshape(np.prod(g_prime_shape), 1)

        Pxg_prime_ = smp.lambdify((y,), g_prime_col, modules=['scipy'])

        kwds.update({'g_prime': lambda y: np.reshape(IP_UxPxU_inv_ @ Pxg_prime_(y), g_prime_shape), \
                    })

    gc.collect()
    print('solving hyper-reduced system ...')
    MSsolvers_dr = MechSystem.solver(kwds)

    # if len(MSsolvers) == len(MSsolvers_dr):
    #     print('solution errors')
    #     print(reshape([np.amax(abs(MSsolvers[i].y -MSsolvers_dr[i].y)) for i in range(len(MSsolvers))], array_shape))

    #     time_lapsed.append(reshape([x.time_lapsed for x in MSsolvers_dr],\
    #                                 time_lapsed[0].shape)/time_lapsed[0]*100)
    # else:
    #     time_lapsed.append(reshape([x.time_lapsed for x in MSsolvers_dr],\
    #                                 time_lapsed[0].shape[:2]))
            
    if len(MSsolvers) == len(MSsolvers_dr):
        print('solution errors')
        print(reshape([np.amax(abs(y1 - y2)) for y1, y2 in zip(MSsolvers, MSsolvers_dr)], array_shape))
    
        time_lapsed.append(reshape([x.time_lapsed for x in MSsolvers_dr], time_lapsed[0].shape) / time_lapsed[0] * 100)
    else:
        time_lapsed.append(reshape([x.time_lapsed for x in MSsolvers_dr], time_lapsed[0].shape[:2]))

    print(f'{time_lapsed = }')

    # tex_table('', time_lapsed)
    '''

    Observations:
        - hyperreduced model is efficient
            - and symplecitc with sparse MDEIM
            - and not demonstrably symplectic with DEIM
            - order of numerical methods is not verified.
        - The Jacobian calculation in the hyperreduced model
            - is also reduced with DEIM
        - higher order integrators are not verified
        - constrained Hamiltonian system should also show order of the integrator

    # Next steps:
        # Implement elastic beam deformation
        # Second-order of methods not observable under spherical constraints
    '''
