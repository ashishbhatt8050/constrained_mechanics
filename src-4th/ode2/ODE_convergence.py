#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 12:26:19 2020

@author: ashishbhatt
"""
import ODESolver
from scitools.std import *
from numpy import linalg as LA

def test_convergence_and_energy(u_init, u_exact, en_err, T, f, dfdu =None, beta =0, dt_space =None):
    """
    Find convergence rate r and coefficient C based on the formula
    error = C dt^r
    """
    
    if not callable(u_exact):
        raise TypeError('u_exact is %s, not a function' % type(exact))

    figure()
        
    registered_solver_classes = [ODESolver.ConformalImplicitMidpoint, ODESolver.ConformalStormerVerlet]
    
    for solver_class in registered_solver_classes:
        legends = []
        mean_error = []
        energy_err = lambda u, t: np.asarray(en_err(u, t), float)
        dt_space = geomspace(0.2, 0.001, num =5)
    
        for dt in dt_space:
            n = int(round(T/dt))
            if solver_class == registered_solver_classes[0]:
                solver = solver_class(f[0], dfdu[0], size(u_init), beta)
                subplot(221)
            elif solver_class == registered_solver_classes[1]:
                solver = solver_class(f[1], beta)
                subplot(223)
                
            solver.set_initial_condition(u_init)
            u, t = solver.solve(linspace(0, T, n+1))
                
            plot(t, u[:, 0])
            legends.append('dt=%g' % dt)
            hold('on')
            
            mean_error.append(sqrt(dt)*LA.norm(u[:, 0] -u_exact(t)))
            
            if solver_class == registered_solver_classes[0]:
                subplot(222)
                plot(t, energy_err(u, t), label ='dt=%g' % dt)
                hold('on')
                legend()
                title('%s' % solver_class.__name__)
            elif solver_class == registered_solver_classes[1]:
                subplot(224)
                plot(t, energy_err(u, t), label = 'dt=%g' % dt)
                hold('on')
                legend()
                title('%s' % solver_class.__name__)
        
        subplot(221)
        plot(t, u_exact(t))
        legends.append('exact')
        legend(legends)
        savefig('tmp_osc.pdf')
        
        # Estimate Convergence rate r and coefficient C
        r = log(roll(mean_error, -1)[:4]/mean_error[:4])\
            /log(roll(dt_space, -1)[:4]/dt_space[:4])
            
        C = mean_error/dt_space    
            
        print solver_class.__name__, r, C