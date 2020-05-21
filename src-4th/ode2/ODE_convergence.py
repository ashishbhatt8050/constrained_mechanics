#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 12:26:19 2020

@author: ashishbhatt
"""
from scitools.std import *
from numpy import linalg as LA
from scipy.integrate import solve_ivp

def test_convergence_and_energy(u_init, u_exact, T, solver, high_order, f=None):
    """
    Find convergence rate r and coefficient C based on the formula
    error = C dt^r
    """

    if not callable(u_exact):
        raise TypeError('u_exact is %s, not a function' % type(u_exact))

    mean_error = []
    dt_space = geomspace(0.01, 0.2, num =5)

    if high_order:
        w_values = [0.28, 0.62546642846767004501]
        w_values.append(1.0 -2.0*(sum(w_values)))
        w_values.append(w_values[1])
        w_values.append(w_values[0])
    else:
        w_values = [1]

    for dt in dt_space:
        n = int(round(T/dt))
        t = linspace(0, T, n+1)
        u = zeros((n+1, size(u_init)))
        u[0] = u_init

        if type(solver) != str:
            for k in range(n):
                u_ = np.array([u[k], u[k]])
                for w_val in w_values:
                    solver.set_initial_condition(u_[1])
                    u_, tp = solver.solve(w_val*t[k:k+2])

                u[k+1] = u_[1]
        else:
            fun = lambda t, u: np.asarray(f(u, t), float)
            sol = solve_ivp(fun, [0, T], u_init, method='RK45', dense_output=True)
            u = sol.sol(t).T

        mean_error.append(sqrt(dt)*LA.norm(u[:, 0] -u_exact(t)))

    # Estimate Convergence rate r and coefficient C
    r = log(roll(mean_error, -1)[:4]/mean_error[:4])\
        /log(roll(dt_space, -1)[:4]/dt_space[:4])

    C = mean_error[:4]/(dt_space[:4]**r)

    return dt_space, r, C
