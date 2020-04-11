#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 12:26:19 2020

@author: ashishbhatt
"""
import ODESolver
from scitools.std import *
from numpy import linalg as LA

def test_convergence_and_energy(u_init, u_exact, en_err, T, solver):
    """
    Find convergence rate r and coefficient C based on the formula
    error = C dt^r
    """
    
    if not callable(u_exact):
        raise TypeError('u_exact is %s, not a function' % type(exact))

    figure()
        
    legends = []
    mean_error = []
    energy_err = lambda u, t: np.asarray(en_err(u, t), float)
    dt_space = geomspace(0.2, 0.001, num =5)

    for dt in dt_space:
        n = int(round(T/dt))
            
        solver.set_initial_condition(u_init)
        u, t = solver.solve(linspace(0, T, n+1))
            
        subplot(121), plot(t, u[:, 0])
        legends.append('dt=%g' % dt)
        hold('on')
        
        mean_error.append(sqrt(dt)*LA.norm(u[:, 0] -u_exact(t)))
        
        subplot(122), plot(t, energy_err(u, t), label = 'dt=%g' % dt)
        hold('on')
        legend()
        title('%s' % type(solver).__name__)
    
    subplot(121)
    plot(t, u_exact(t))
    legends.append('exact')
    legend(legends)
    savefig('tmp_%s.pdf' % type(solver).__name__)
    
    # Estimate Convergence rate r and coefficient C
    r = log(roll(mean_error, -1)[:4]/mean_error[:4])\
        /log(roll(dt_space, -1)[:4]/dt_space[:4])
        
    C = mean_error/dt_space    
        
    print type(solver).__name__, r, C