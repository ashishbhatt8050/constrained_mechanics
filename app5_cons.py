#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 11:49:50 2020

@author: ashishbhatt

This app solves constrained mechanical systems using solver
classes in the ODESolver hierarchy of methods.
"""

import numpy as np
from numpy import linalg as LA
from scitools.std import *
from pylab import *
import ODESolver
from PlotScript import plot_data
#import scipy
#from scipy.integrate import solve_ivp

#%% Define the system
class MechSystem(object):
    def __init__(self, Omega, alpha, beta, method=None):
        if np.dot(alpha, alpha) != 1:
            ValueError('alpha''s norm must be 1')

        self.Omega2 = np.diag(Omega)**2
        self.alpha = alpha
        self.beta = beta
        self.nosc = max(size(Omega), size(alpha))
        self.method = method

    def __call__(self, y, t):
        Omega2, beta, nosc, method = self.Omega2, self.beta, self.nosc, self.method

        x, u = y[:nosc], y[nosc:]

        f = [u, -Omega2.dot(sin(x))]
        if method in ['ConformalStormerVerlet']:
            return f
        elif method in ['ConformalImplicitMidpoint', 'ForwardEuler', 'ImplicitMidpoint']:
            f = reshape([u, -Omega2.dot(sin(x)) -beta*u], (2*nosc,)) +beta/2*y
            return f
        else:
            NameError('Undefined method - %s' % method)

class Jacobian(MechSystem):
    def __call__(self, y, t, dt=0):

        x, u, beta = y[:nosc], y[nosc:], self.beta
        dfdu = np.concatenate([np.concatenate([beta/2*eye(nosc), eye(nosc)], axis=1), \
                               np.concatenate([-self.Omega2.dot(np.diag(cos(x))), -beta/2*eye(nosc)], axis=1)])
        return dfdu

class EnergyError(MechSystem):
    def __call__(self, y, t=0):
        Omega2, nosc = self.Omega2, self.nosc

        x, u = y[:, :nosc], y[:, nosc:]

        #V, T = lambda x: 1/2*x.dot(Omega2.dot(np.transpose(x))), \
        T = lambda u: sum((u**2), axis=1)/2.0
        V = lambda x: -sum(cos(x), axis=1)
        E = lambda x, u: V(x) +T(u)
        return E(x, u) - E(reshape(x[0, :], (1, nosc)), reshape(u[0, :], (1, nosc)))

#%% Initialize varialbles
Omega = [1.0, 1.0, 1.0] # do not change this value !!
alpha = [0.2, 0.4, float('nan')]
alpha[2] = sqrt(1 -alpha[0]**2 - alpha[1]**2)
beta = 0.1

en_err = EnergyError(Omega, alpha, beta)
r_form = lambda numer, denom: (log(roll(numer, -1)/numer)/log(roll(denom, -1)/denom))[:-1]
C_form = lambda numer, denom, r: numer[:-1]/(denom[:-1]**r)

y_init = [0.2, 0.4, float('nan'), 0., 0., 0.]
y_init[2] = -(np.array(y_init[:2]).dot(alpha[:2]))/alpha[2] # project on the manifold
nosc = int(size(y_init)/2)     # number of oscillators
_g = lambda y: y[:, :nosc].dot(alpha)
_G = lambda y: np.array(alpha)
alg = lambda solver_class: solver_class.__name__

T_final = 20
dt_space = concatenate(([], geomspace(0.01, 0.2, num=1)))
w_values = [0.28, 0.62546642846767004501]
w_values.append(1.0 -2.0*(sum(w_values)))
w_values.append(w_values[1])
w_values.append(w_values[0])
w_values = [1]

registered_solver_classes = [ODESolver.ConformalImplicitMidpoint]

#%% Solve the system and find convergence rates
for solver_class in registered_solver_classes:
    rel_error = []
    msr_error = []
    eng_error = []
    sym_error = []
    r_values = []
    C_values = []

    for dt in dt_space:
        n = int(round(T_final/dt))
        t_points = linspace(0, T_final, n+1)

        M, store, epsilon, var, var_store, plt_res = 100, False, 1.0E-10, True, False, True

        if store: info = []
        if var_store: var_info  = []

        if solver_class in [ODESolver.ConformalStormerVerlet, ODESolver.ForwardEuler]:
            f = MechSystem(Omega, alpha, beta, solver_class.__name__)
            solver = solver_class(f, beta)
        elif solver_class in [ODESolver.ConformalImplicitMidpoint, ODESolver.ImplicitMidpoint]:
            f, dfdu = MechSystem(Omega, alpha, beta, solver_class.__name__), Jacobian(Omega, alpha, beta)
            solver = solver_class(f, dfdu, 2*nosc, beta)
        else:
            NameError('Undefined solver class - %s' % solver_class.__name__)

        y = np.zeros((n+1, 2*nosc))
        y[0] = y_init

        if var:
            dpsi = np.zeros((n+1, 2*nosc, 2*nosc))
            dpsi[0] = np.eye(2*nosc)

        for k in range(n):
            y_ = np.array([y[k], y[k]])
            for w_val in w_values:
                solver.set_initial_condition(y_[1])
                y_, tp = solver.solve(w_val*t_points[k:k+2])

                Delta_LambdaX, X, m = _g(reshape(y_[1], (1, 2*nosc))), y_[1, :nosc], 0
                if store: info.append((X, Delta_LambdaX, m))

                '''
                lambda_ = np.array([[2, dt],[-2, dt]]).dot([y_[1,:nosc].dot(alpha), y_[1,nosc:].dot(alpha)])
                y_[1] = y[1] + np.concatenate([dt**2.0/4.0*_G(y_[0])*(lambda_[1] -lambda_[0]),\
                      -dt/2.0*_G(y_[0])*(lambda_[1] +lambda_[0])])
                '''

                while abs(Delta_LambdaX) > epsilon and m <= M:
                    X = X -np.transpose(_G(y_[0]))*Delta_LambdaX

                    m += 1
                    Delta_LambdaX = _g(reshape(X, (1, nosc)))
                    if store: info.append((X, Delta_LambdaX, m))

                y_[1, :nosc] = X

                Delta_LambdaU, U, m = _G(y_[1]).dot(y_[1, nosc:]), y_[1, nosc:], 0
                if store: info.append((U, Delta_LambdaU, m))

                while abs(Delta_LambdaU) > epsilon and m <= M:
                    U = U -np.transpose(_G(y_[1]))*Delta_LambdaU

                    m += 1
                    Delta_LambdaU = _G(y_[1]).dot(U)
                    if store: info.append((U, Delta_LambdaU, m))

                y_[1, nosc:] = U

            y[k+1] = y_[1]

        if var:
            dpsi, tp = solver.var_solve(y, t_points)
            '''
            d_Delta_Lambda = (dpsi_[1].T).dot(np.concatenate([alpha,alpha]))
            dY = dpsi_[1]
            m = 0

            while LA.norm(d_Delta_Lambda) > epsilon and m <= M:
                dY = dY - [[p*q for p in np.concatenate([alpha,alpha])] for q in d_Delta_Lambda]

                m += 1
                d_Delta_Lambda = (dY.T).dot(np.concatenate([alpha,alpha]))
                if var_store: var_info.append((dY, LA.norm(d_Delta_Lambda), m))
            if m > M:
                print "Solver failed to converge at t=%g "\
                      "(%d iterations, %d)" % (t_points[k+1], m, LA.norm(d_Delta_Lambda))
            dpsi_[1] = dY

            d_Delta_LambdaX = (dpsi_[1, :nosc, :nosc].T).dot(alpha)
            dX = dpsi_[1, :nosc, :nosc]
            m = 0

            while LA.norm(d_Delta_LambdaX) > epsilon and m <= M:
                dX = dX -[[p*q for p in _G(y_[0])] for q in d_Delta_LambdaX]

                m += 1
                d_Delta_LambdaX = _g(dX.T)
                if var_store: var_info.append((dX, LA.norm(d_Delta_LambdaX), m))

            dpsi_[1, :nosc, :nosc] = dX

            d_Delta_LambdaU = (dpsi_[1, nosc:, nosc:].T).dot(alpha)
            dU = dpsi_[1, nosc:, nosc:]
            m = 0

            while LA.norm(d_Delta_LambdaU) > epsilon and m <= M:
                dU = dU -[[p*q for p in _G(y_[0])] for q in d_Delta_LambdaU]

                m += 1
                d_Delta_LambdaU = _g(dU.T)
                if var_store: var_info.append((dU, LA.norm(d_Delta_LambdaU), m))

            dpsi_[1, nosc:, nosc:] = dU

            dpsi_[1, nosc:, nosc:] = (np.eye(nosc) -[[dt**2/4*p*q for p in alpha] for q in alpha]).dot(dpsi_[1, nosc:, nosc:])
            dpsi_[1, :nosc, :nosc] = dpsi_[1, nosc:, nosc:]
            '''
            #dpsi[k+1] = np.concatenate(((1 -dt**2)*dpsi_[1,:nosc,:],(1-dt)*dpsi_[1,nosc:,:]))

        if dt == dt_space[0]:
            y_interp = lambda t_, t_points, y: np.interp(t_, t_points, y)
            y_exact = lambda t_: [y_interp(t_, t_points, y[:, 0]), y_interp(t_, t_points, y[:, 1]), y_interp(t_, t_points, y[:, 2]), \
                                  y_interp(t_, t_points, y[:, 3]), y_interp(t_, t_points, y[:, 4]), y_interp(t_, t_points, y[:, 5])]
            msr_error.append(float('nan'))
            rel_error.append(float('nan'))
        else:
            msr_error.append(sqrt(dt)*LA.norm(y_exact(t_points) -y.T))
            rel_error.append(abs(y[-1,0] -y_exact(t_points)[0][-1])/abs(y_exact(t_points)[0][-1]))

        if var: sym_error.append(solver.symplectic_error(dpsi, t_points))

        if not beta:
            eng_error.append(sqrt(dt)*LA.norm(en_err(y)))

        # plot the results
        if plt_res:
            fig = figure()
            ax1 = fig.add_subplot(121)
            plot_data(ax1, t_points, y, True, False)
            ax1.legend(['$q_1$','$q_2$','$q_3$','$p_1$','$p_2$','$p_3$'],loc=1)
            ax1.set_xlabel('time')

            ax2 = fig.add_subplot(122)
            if not beta:
                plot_data(ax2, t_points, np.array([reshape(sym_error[-1:],(n+1,)), en_err(y), _g(y), y[:, nosc:].dot(_G(y))]).T, True, True)
                ax2.legend(['$E_{cs}$', '$E_Q$', '$g(q^{n+1})$', r'$G^\top p^{n+1}$'], loc=1)
            elif var:
                plot_data(ax2, t_points, np.array([reshape(sym_error[-1:],(n+1,)), _g(y), y[:, nosc:].dot(_G(y))]).T, True, True)
                ax2.legend(['$E_{cs}$', '$g(q^{n+1})$', r'$G^\top p^{n+1}$'], loc=1)
            ax2.set_xlabel('time')

            fig.savefig('app5_%s_%s.pdf' % (alg(solver_class), dt))

    # Estimate Convergence rate r and coefficient C
    if not beta:
        r_values.append(r_form(eng_error,dt_space))
        C_values.append(C_form(eng_error,dt_space,r_values[-1]))

    r_values.append(r_form(msr_error,dt_space))
    C_values.append(C_form(msr_error,dt_space,r_values[-1]))

    r_values.append(r_form(rel_error,dt_space))
    C_values.append(C_form(rel_error,dt_space,r_values[-1]))

    temp = np.array((dt_space, np.concatenate(([np.float('nan')],r_values[0])),\
                     np.concatenate(([np.float('nan')], C_values[0])))).T
    print alg(solver_class), " \\\\\n".join([" & ".join(map(str,line)) for line in temp])
    if var: print sym_error