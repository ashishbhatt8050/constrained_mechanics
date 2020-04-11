"""
Equation:

       m*u'' + beta*u' + k*u = m*w''(t) + m*g

written as a 2x2 first-order ODE system and solved
by classes in the ODESolver hierarchy of methods.
"""

import ODESolver
from scitools.std import *
#from matplotlib.pyplot import *

registered_solver_classes = [ODESolver.ForwardEuler, ODESolver.ImplicitMidpoint, ODESolver.ConformalImplicitMidpoint, ODESolver.ConformalStormerVerlet]

class OscSystem:
    def __init__(self, m, beta, k, g, w, method = None):
        self.m, self.beta, self.k, self.g, self.w, self.method = \
                float(m), float(beta), float(k), float(g), w, method

    def __call__(self, u, t):
        u0, u1 = u
        m, beta, k, g, w, method = \
           self.m, self.beta, self.k, self.g, self.w, self.method
        # Use a finite difference for w''(t)
        h = 1E-5
        ddw = (w(t+h) - 2*w(t) + w(t-h))/(h**2)
        if method in ['ForwardEuler', 'BackwardEuler', 'RungeKutta4', 'ImplicitMidpoint']:
            f = [u1, ddw  + g - beta/m*u1 - k/m*u0]
        elif method in ['ConformalImplicitMidpoint']:
            f = list([u1, ddw  + g - beta/m*u1 - k/m*u0] +beta/2*u)
        elif method in ['ConformalStormerVerlet']:
            f = [u1, ddw  + g - k/m*u0]
        else:
            NameError('f not defined for the given method %s' % method)
        return f

class Jacobian(OscSystem):
    def __call__(self, u, t, dt):
        if self.method in ['BackwardEuler','ImplicitMidpoint']:
            dfdu = [[0.0, 1.0],[-k/m, -beta/m]]
        elif self.method in ['ConformalImplicitMidpoint']:
            dfdu = [[beta/2, 1.0],[-k/m, beta/2-beta/m]]
        else:
            NameError('Jacobian not provided for %s' % self.method)
        return dfdu
    
class ExactSolution(OscSystem):
    def __call__(self, t):
        m, beta, k, g, w = \
           self.m, self.beta, self.k, self.g, self.w
        z = sqrt(beta**2 -4*k*m)/m
        
        return [exp(-beta/m*t/2)*(cosh(z/2*t) +beta/(m*z)*sinh(z/2*t)), \
                -2/(m*z)*exp(-beta/m*t/2)*sinh(z/2*t)]
        
class EnergyError(OscSystem):
    def __call__(self, u, t):
        m, beta, k, g, w = \
           self.m, self.beta, self.k, self.g, self.w
           
        def E(u):
            return k*u[:,0]**2 +m*u[:,1]**2 +beta*u[:,0]*u[:,1]
           
        #return E(u) -exp(-beta*dt)*E(roll(u,1))
        return abs(E(u) -exp(-beta*t)*E(resize(u[0,:],(1,2))))
        
# Test case: u = cos(t)

m, beta, k, g, w = 1.0, 0.1, 1.0, 0.0, lambda t: 0
u_exact, en_err = ExactSolution(m, beta, k, g, w), EnergyError(m, beta, k, g, w)
            
u_init = [1, 0]    # initial condition
nperiods = 3.5     # no of oscillation periods
T = 2*pi*nperiods

u0_exact = lambda t: u_exact(t)[0]
u1_exact = lambda t: u_exact(t)[1]
    

#%%
legends = []

for solver_class in registered_solver_classes:
    if solver_class == ODESolver.RungeKutta4:
        npoints_per_period = 20
    else:
        npoints_per_period = 100
        
    n = npoints_per_period*nperiods
    t_points = linspace(0.0, T, n+1)
    
    alg = lambda solver_class: solver_class.__name__  # (class) name of algorithm
    
    if solver_class == ODESolver.ConformalImplicitMidpoint:
        f, dfdu = OscSystem(m, beta, k, g, w, alg(solver_class)), \
            Jacobian(m, beta, k, g, w, alg(solver_class))
        solver = solver_class(f, dfdu, size(u_init), beta)
    elif solver_class == ODESolver.ConformalStormerVerlet:
        f = OscSystem(m, beta, k, g, w, alg(solver_class))
        solver = solver_class(f, beta)
    elif solver_class in [ODESolver.BackwardEuler, ODESolver.ImplicitMidpoint]:
        f, dfdu = OscSystem(m, beta, k, g, w, alg(solver_class)), \
            Jacobian(m, beta, k, g, w, alg(solver_class))
        solver = solver_class(f, dfdu, size(u_init))
    else:
        f = OscSystem(m, beta, k, g, w, alg(solver_class))
        solver = solver_class(f, beta)
    
    solver.set_initial_condition(u_init)
    u, t = solver.solve(t_points)

    # u is an array of [u0,u1] pairs for each time level,
    # get the u0 values from u for plotting
    u0_values = u[:, 0]
    u1_values = u[:, 1]
    figure()
    
    plot(t, u0_values, 'r-',
         t, u0_exact(t), 'b-')
    legend(['numerical', 'exact']),
    title('Oscillating system; position - %s' % alg(solver_class))
    savefig('tmp_oscsystem_pos_%s.pdf' % alg(solver_class))
    figure()
    plot(t, u1_values, 'r-',
         t, u1_exact(t), 'b-')
    legend(['numerical', 'exact'])
    title('Oscillating system; velocity - %s' % alg(solver_class))
    savefig('tmp_oscsystem_vel_%s.pdf' % alg(solver_class))

show()

#%% Convergence test
import sys, os

try:
    from ODE_convergence import test_convergence_and_energy
except ImportError:
    raise ImportError('''
    Could not import module "test_convergence_and_energy". Place ODE_convergence.py in this directory
    (%s)
    ''' % (os.path.dirname(os.path.abspath(__file__))))
    
for solver_class in registered_solver_classes:
    if solver_class == ODESolver.ConformalImplicitMidpoint:
        f, dfdu = OscSystem(m, beta, k, g, w, alg(solver_class)), \
            Jacobian(m, beta, k, g, w, alg(solver_class))
        solver = solver_class(f, dfdu, size(u_init), beta)
    elif solver_class == ODESolver.ConformalStormerVerlet:
        f = OscSystem(m, beta, k, g, w, alg(solver_class))
        solver = solver_class(f, beta)
    elif solver_class in [ODESolver.BackwardEuler, ODESolver.ImplicitMidpoint]:
        f, dfdu = OscSystem(m, beta, k, g, w, alg(solver_class)), \
            Jacobian(m, beta, k, g, w, alg(solver_class))
        solver = solver_class(f, dfdu, size(u_init))
    else:
        f = OscSystem(m, beta, k, g, w, alg(solver_class))
        solver = solver_class(f, beta)
        
    test_convergence_and_energy(u_init, u0_exact, en_err, T, solver)
