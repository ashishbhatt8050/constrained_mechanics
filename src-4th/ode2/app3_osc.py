"""
Equation:

       m*u'' + beta*u' + k*u = m*w''(t) + m*g

written as a 2x2 first-order ODE system and solved
by classes in the ODESolver hierarchy of methods.
"""

class OscSystem:
    def __init__(self, m, beta, k, g, w):
        self.m, self.beta, self.k, self.g, self.w = \
                float(m), float(beta), float(k), float(g), w

    def __call__(self, u, t):
        u0, u1 = u
        m, beta, k, g, w = \
           self.m, self.beta, self.k, self.g, self.w
        # Use a finite difference for w''(t)
        h = 1E-5
        ddw = (w(t+h) - 2*w(t) + w(t-h))/(h**2)
        f = [u1, ddw  + g - beta/m*u1 - k/m*u0]
        return f

class Jacobian(OscSystem):
    def __call__(self, u, t, dt):
        m, beta, k, g, w = \
           self.m, self.beta, self.k, self.g, self.w

        dfdu = [[0.0, 1.0],[-k/m, -beta/m]]
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
    
class Getf1(OscSystem):
    def __call__(self, u, t):
        u0, u1 = u
        m, beta, k, g, w = \
           self.m, self.beta, self.k, self.g, self.w
        # Use a finite difference for w''(t)
        h = 1E-5
        ddw = (w(t+h) - 2*w(t) + w(t-h))/(h**2)
        f1 = list([u1, ddw  + g - beta/m*u1 - k/m*u0] +beta/2*u)
        return f1
    
class Getdfdu1(OscSystem):
    def __call__(self, u, t, dt):
        m, beta, k, g, w = \
           self.m, self.beta, self.k, self.g, self.w

        dfdu1 = [[beta/2, 1.0],[-k/m, beta/2-beta/m]]
        return dfdu1
        
# Test case: u = cos(t)
import ODESolver
from scitools.std import *
#from matplotlib.pyplot import *
legends = []

m, beta, k, g, w = 1.0, 0.1, 1.0, 0.0, lambda t: 0
f, dfdu, u_exact, en_err, f1, dfdu1, f2 = OscSystem(m, beta, k, g, w), \
            Jacobian(m, beta, k, g, w), \
            ExactSolution(m, beta, k, g, w), \
            EnergyError(m, beta, k, g, w), \
            Getf1(m, beta, k, g, w), \
            Getdfdu1(m, beta, k, g, w), \
            OscSystem(m, 0.0, k, g, w)
            
u_init = [1, 0]    # initial condition
nperiods = 3.5     # no of oscillation periods
T = 2*pi*nperiods

registered_solver_classes = [ODESolver.ConformalImplicitMidpoint, ODESolver.ConformalStormerVerlet]

u0_exact = lambda t: u_exact(t)[0]
u1_exact = lambda t: u_exact(t)[1]
    

#%%
for solver_class in registered_solver_classes:
    if solver_class == ODESolver.RungeKutta4:
        npoints_per_period = 20
    else:
        npoints_per_period = 100
        
    n = npoints_per_period*nperiods
    t_points = linspace(0.0, T, n+1)
    
    if solver_class == ODESolver.ConformalImplicitMidpoint:
        solver = solver_class(f1, dfdu1, size(u_init), beta)
        solver.set_initial_condition(u_init)
        u, t = solver.solve(t_points)
    elif solver_class == ODESolver.ConformalStormerVerlet:
        solver = solver_class(f2, beta)
        solver.set_initial_condition(u_init)
        u, t = solver.solve(t_points)
    else:
        solver = solver_class(f, dfdu, size(u_init))
        solver.set_initial_condition(u_init)
        u, t = solver.solve(t_points)

    # u is an array of [u0,u1] pairs for each time level,
    # get the u0 values from u for plotting
    u0_values = u[:, 0]
    u1_values = u[:, 1]
    figure()
    alg = solver_class.__name__  # (class) name of algorithm
    plot(t, u0_values, 'r-',
         t, u0_exact(t), 'b-')
    legend(['numerical', 'exact']),
    title('Oscillating system; position - %s' % alg)
    savefig('tmp_oscsystem_pos_%s.pdf' % alg)
    figure()
    plot(t, u1_values, 'r-',
         t, u1_exact(t), 'b-')
    legend(['numerical', 'exact'])
    title('Oscillating system; velocity - %s' % alg)
    savefig('tmp_oscsystem_vel_%s.pdf' % alg)
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
test_convergence_and_energy(u_init, u0_exact, en_err, T, [f1,f2], [dfdu1], beta)
