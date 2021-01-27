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
from PlotScript import plot_data, tex_table
from scipy import optimize
#import scipy
#from scipy.integrate import solve_ivp

#%% Define the system
class MechSystem(object):
    def __init__(self, method=None):
        if np.dot(alpha, alpha) != 1:
            ValueError('alpha''s norm must be 1')

        self.Omega2 = np.diag(Omega)**2
        self.alpha = alpha
        self.beta = beta
        self.nosc = max(size(Omega), size(alpha))
        self.method = method
        self.u_init = y_init

    def __call__(self, y, t):
        method, Omega2 = self.method, self.Omega2

        x, u = y[:nosc], y[nosc:]

        if method in ['ConformalStormerVerlet']:
            f = [u, -Omega2.dot(sin(x))]
        elif method in ['ConformalImplicitMidpoint']:
            f = reshape([u, -Omega2.dot(sin(x)) -beta*u], (2*nosc,)) +beta/2*y
        elif method in ['ImplicitMidpoint', 'ForwardEuler','StormerVerlet']:
            f = reshape([u, -Omega2.dot(sin(x)) -beta*u], (2*nosc,))
        else:
            NameError('Undefined method - %s' % method)

        return f

class Jacobian(MechSystem):
    def __call__(self, y, t, dt=0):
        x, u = y[:nosc], y[nosc:]
        dfdu = np.concatenate([np.concatenate([beta/2*eye(nosc), eye(nosc)], axis=1), \
                               np.concatenate([-self.Omega2.dot(np.diag(cos(x))), -beta/2*eye(nosc)], axis=1)])
        return dfdu

class EnergyError(MechSystem):
    def __call__(self, y, t=0):
        x, u = y[:, :nosc], y[:, nosc:]

        T = lambda u: sum((u**2), axis=1)/2.0
        V = lambda x: -sum(cos(x), axis=1)
        E = lambda x, u: V(x) +T(u)
        return E(x, u) - E(reshape(x[0, :], (1, nosc)), reshape(u[0, :], (1, nosc)))

#%% Initialize varialbles
Omega = [1.0, 1.0, 1.0] # do not change this value !!
alpha = [0.2, 0.4, float('nan')]
alpha[2] = sqrt(1 -alpha[0]**2 - alpha[1]**2)
beta = 0.1

y_init = [0.2, 0.4, float('nan'), 0., 0., 0.]
# y_init[2] = -(np.array(y_init[:2]).dot(alpha[:2]))/alpha[2] # project on the manifold
y_init[2] = sqrt(1 -np.array(alpha[:2]).dot(np.array(y_init[:2])**2))/sqrt(alpha[2]) # spherical constraint

en_err = EnergyError()
r_form = lambda numer, denom: (log(roll(numer, -1)/numer)/log(roll(denom, -1)/denom))[:-1]
C_form = lambda numer, denom, r: numer[:-1]/(denom[:-1]**r)

nosc = int(size(y_init)/2)     # number of oscillators

# # linear constraints
# _g = lambda y: y[:, :nosc].dot(alpha)
# _G = lambda y: np.array(alpha)

# spherical constraints
_g = lambda y: (y[:nosc]**2).dot(alpha) -1.0
_G = lambda y: 2*np.array(y[:nosc])*np.array(alpha)

alg = lambda solver_class: solver_class.__name__
fig = figure()

T_final = 20
dt_space = concatenate(([], linspace(0.05, 0.2, num=1)))

# higher order composition coefficients
w_values = [0.28, 0.62546642846767004501]
w_values.append(1.0 -2.0*(sum(w_values)))
w_values.append(w_values[1])
w_values.append(w_values[0])
# w_values = [1]

registered_solver_classes = [ODESolver.ConformalImplicitMidpoint, ODESolver.ImplicitMidpoint]

#%% Solve the system and find convergence rates
r_values, C_values = [],[]
for solver_class in registered_solver_classes:
    rel_error, msr_error, eng_error, sym_error = [],[],[],[]

    for dt in dt_space:
        n = int(round(T_final/dt))
        t_points = linspace(0, T_final, n+1)

        M, store, epsilon, var, var_store, plt_res = 100, False, 1.0E-14, True, False, True

        if store: info = []
        if var_store: var_info  = []

        f, dfdu = MechSystem(alg(solver_class)), Jacobian(alg(solver_class))

        if solver_class in [ODESolver.ConformalStormerVerlet]:
            solver = solver_class(f)
        elif solver_class in [ODESolver.ConformalImplicitMidpoint, ODESolver.ImplicitMidpoint]:
            solver = solver_class(f, dfdu)
        else:
            NameError('Undefined solver class - %s' % alg(solver_class))

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
                
                y_[1,:nosc] = optimize.newton(_g, y_[1,:nosc], tol=1e-10)
                y_[1,nosc:] = optimize.newton(lambda Y: (_G(y_[1,:nosc]).dot(Y.T)), y_[1,nosc:])
                
# =============================================================================
#                 Y, Yp = reshape(y_[1], (1,2*nosc)), reshape(y[0], (1,2*nosc))
#                 Delta_Lambda, m = delta_lambda_g(Y, Yp), 0
#                 if store: info.append((Y[0,:nosc], Delta_Lambda, m))
# 
#                 while _g(Y) > epsilon and m <= M:
#                     Y[0, :nosc] = Y[0, :nosc] -_G(Yp)*Delta_Lambda
# 
#                     m += 1
#                     Delta_Lambda = delta_lambda_g(Y, Yp)
#                     if store: info.append((Y[0,:nosc], Delta_Lambda, m))
# 
#                 y_[1, :nosc] = Y[0,:nosc]
# 
#                 Delta_Lambda, m = delta_lambda_G(Y, Yp), 0
#                 if store: info.append((Y[0,nosc:], Delta_Lambda, m))
# 
#                 while _G(Y).dot(Y[0,nosc:]) > epsilon and m <= M:
#                     Y[0, nosc:] = Y[0, nosc:] -_G(Yp)*Delta_Lambda
# 
#                     m += 1
#                     Delta_Lambda = delta_lambda_G(Y, Yp)
#                     if store: info.append((Y[0, nosc:], Delta_Lambda, m))
# 
#                 y_[1, nosc:] = Y[0, nosc:]
# =============================================================================

            y[k+1] = y_[1]

            if var:
                dpsi_, tp = solver.var_solve(y_, t_points[k:k+2])
                dpsi[k+1] = dpsi_[1]

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
        if plt_res and dt == dt_space[-1]:
            g_list = lambda y: np.array([_g(x) for x in y])
            G_list = lambda y: np.array([_G(x) for x in y])
            temp = lambda y: sum(G_list(y)*y[:,nosc:], axis=1)
            
            if solver_class == ODESolver.ConformalImplicitMidpoint:
                ax = fig.add_subplot(321)
                ax.set_ylim(0,2.0E-15)
                ax.set_yticks([0,1.0E-15,2.0E-15])
                ax.set_ylabel(r'$\boldmath{E}_{cs}$')
                plot_data(ax, t_points, reshape(sym_error[-1:],(n+1,)).T)
                ax = fig.add_subplot(323)
                ax.set_ylabel(r'$\boldmath{g}(\boldmath{q}^{n+1})$')
                ax.set_ylim(-1.0E-15,1.0E-15)
                plot_data(ax, t_points, g_list(y).T)
                ax = fig.add_subplot(325)
                ax.set_ylabel(r'$\boldmath{G}^\top \boldmath{p}^{n+1}$')
                ax.set_ylim(-1E-15,1E-15)
                plot_data(ax, t_points, temp(y).T)
            elif solver_class == ODESolver.ImplicitMidpoint:
                ax = fig.add_subplot(322)
                ax.set_ylim(0,0.02)
                ax.set_yticks([0,0.01,0.02])
                plot_data(ax, t_points, reshape(sym_error[-1:],(n+1,)).T)
                ax = fig.add_subplot(324)
                ax.set_ylim(-1.0E-15,1.0E-15)
                plot_data(ax, t_points, g_list(y).T)
                ax = fig.add_subplot(326)
                ax.set_ylim(-1.0E-15,1.0E-15)
                plot_data(ax, t_points, temp(y).T)
                
            if not beta:
                plot_data(ax, t_points, np.array([reshape(sym_error[-1:],(n+1,)), en_err(y), g_list(y), temp(y)]).T, True, True)
                # ax2.legend([r'$\boldmath{E}_{cs}$', r'$\boldmath{E}_I$', r'$\bolmath{g}(\boldmath{q}^{n+1})$', r'$\boldmath{G}^\top \boldmath{p}^{n+1}$'], loc=1)
            elif not var:
                plot_data(ax, t_points, np.array([reshape(sym_error[-1:],(n+1,)), g_list(y), temp(y)]).T, True, True)
                # ax2.legend([r'$\boldmath{E}_{cs}$', r'$\bolmath{g}(\boldmath{q}^{n+1})$', r'$\boldmath{G}^\top \boldmath{p}^{n+1}$'], loc=1)
            # ax2.set_xlabel('time')
        
            # if not beta:
            #     plot_data(ax, t_points, np.array([reshape(sym_error[-1:],(n+1,)), en_err(y), g_list(y), temp(y)]).T, True, True)
            #     # ax2.legend([r'$\boldmath{E}_{cs}$', r'$\boldmath{E}_I$', r'$\bolmath{g}(\boldmath{q}^{n+1})$', r'$\boldmath{G}^\top \boldmath{p}^{n+1}$'], loc=1)
            # elif var:
            #     plot_data(ax, t_points, np.array([reshape(sym_error[-1:],(n+1,)), g_list(y), temp(y)]).T, True, True)
            #     # ax2.legend([r'$\boldmath{E}_{cs}$', r'$\bolmath{g}(\boldmath{q}^{n+1})$', r'$\boldmath{G}^\top \boldmath{p}^{n+1}$'], loc=1)
            
            ax.set_xlabel('time')
                
            if not beta:
                fig.legend([r'$\boldmath{E}_{cs}$', r'$\boldmath{E}_I$', r'$\boldmath{g}(\boldmath{q}^{n+1})$', r'$\boldmath{G}^\top \boldmath{p}^{n+1}$'], \
                           bbox_to_anchor=(0.5,-0.08), loc='lower center',ncol=2,bbox_transform=fig.transFigure)
            elif not var:
                fig.legend([r'$\boldmath{E}_{cs}$', r'$\boldmath{g}(\boldmath{q}^{n+1})$', r'$\boldmath{G}^\top \boldmath{p}^{n+1}$'], \
                           bbox_to_anchor=(0.5,-0.08), loc='lower center',ncol=3,bbox_transform=fig.transFigure)

            fig.savefig('app5_err_inv_sph.pdf', bbox_inches='tight')

    # Estimate Convergence rate r and coefficient C
    if not beta:
        r_values.append(r_form(eng_error,dt_space))
        C_values.append(C_form(eng_error,dt_space,r_values[-1]))

    r_values.append(r_form(msr_error,dt_space))
    C_values.append(C_form(msr_error,dt_space,r_values[-1]))

    r_values.append(r_form(rel_error,dt_space))
    C_values.append(C_form(rel_error,dt_space,r_values[-1]))

if not beta:
    temp = np.array((dt_space, np.concatenate(([np.float('nan')],r_values[0])),\
                     np.concatenate(([np.float('nan')], r_values[3])))).T
    tex_table(alg(solver_class), temp)