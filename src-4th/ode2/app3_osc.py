"""
Equation:

       m*u'' + beta*u' + k*u = m*w''(t) + m*g

written as a 2x2 first-order ODE system and solved
by classes in the ODESolver hierarchy of methods.
"""

import ODESolver
from scitools.std import *
from numpy import linalg as LA
from pylab import *
from PlotScript import plot_data, tex_table
from scipy.integrate import solve_ivp

# class definitions
class OscSystem:
    def __init__(self, method=None):
        self.m, self.beta, self.k, self.g, self.w, self.method, self.eqn_type, self.u_init = \
                float(m), float(beta), float(k), float(g), w, method, eqn_type, u_init

    def __call__(self, u, t):
        u0, u1 = u
        method = self.method

        # Use a finite difference for w''(t)
        h = 1E-5
        ddw = (w(t+h) - 2*w(t) + w(t-h))/(h**2)

        if eqn_type == 'nonlinear':
            if method in ['ForwardEuler', 'BackwardEuler', 'RungeKutta4', 'ImplicitMidpoint', 'builtin']:
                f = [u1, ddw  + g - beta/m*u1 - k/m*sin(u0)]
            elif method in ['ConformalImplicitMidpoint']:
                f = list([u1, ddw  + g - beta/m*u1 - k/m*sin(u0)] +beta/2*u)
            elif method in ['ConformalStormerVerlet']:
                f = [u1, ddw  + g - k/m*sin(u0)]
            else:
                NameError('f not defined for the given method %s' % method)
        elif eqn_type == 'linear':
            if method in ['ForwardEuler', 'BackwardEuler', 'RungeKutta4', 'ImplicitMidpoint', 'builtin']:
                f = [u1, ddw  + g - beta/m*u1 - k/m*u0]
            elif method in ['ConformalImplicitMidpoint']:
                f = list([u1, ddw  + g - beta/m*u1 - k/m*u0] +beta/2*u)
            elif method in ['ConformalStormerVerlet']:
                f = [u1, ddw  + g - k/m*u0]
            else:
                NameError('f not defined for the given method %s' % method)
        else:
            NameError('f is not defined for the given type %s' % eqn_type)

        return f

class Jacobian(OscSystem):
    def __call__(self, u, t, dt=None):
        if self.eqn_type == 'nonlinear':
            if self.method in ['BackwardEuler','ImplicitMidpoint']:
                dfdu = [[0.0, 1.0],[-k/m*cos(u[0]), -beta/m]]
            elif self.method in ['ConformalImplicitMidpoint']:
                dfdu = [[beta/2, 1.0],[-k/m*cos(u[0]), beta/2-beta/m]]
            else:
                NameError('Jacobian not provided for %s' % self.method)
        elif self.eqn_type == 'linear':
            if self.method in ['BackwardEuler','ImplicitMidpoint']:
                dfdu = [[0.0, 1.0],[-k/m, -beta/m]]
            elif self.method in ['ConformalImplicitMidpoint']:
                dfdu = [[beta/2, 1.0],[-k/m, beta/2-beta/m]]
            else:
                NameError('Jacobian not provided for %s' % self.method)
        else:
            NameError('dfdu is not defined for the given type %s' % self.eqn_type)

        return dfdu

class ExactSolution(OscSystem):
    def __call__(self, t):
        z = numpy.lib.scimath.sqrt(beta**2 -4*k*m)/m

        if self.eqn_type == 'linear':
            return [exp(-beta/m*t/2)*(cosh(z/2*t) +beta/(m*z)*sinh(z/2*t)), \
                -2/(m*z)*exp(-beta/m*t/2)*sinh(z/2*t)]
        else:
            NameError('Exact solution is not defined for the equation type: %s' % self.eqn_type )

class EnergyError(OscSystem):
    def __call__(self, u, t):

        if self.eqn_type == 'linear':
            def E(u):
                return k*u[:,0]**2 +m*u[:,1]**2 +beta*u[:,0]*u[:,1]

            #return E(u) -exp(-beta*dt)*E(roll(u,1))
            return abs(E(u) -exp(-beta*t)*E(resize(u[0,:],(1,2))))
        else:
            NameError('Energy Error is not defined for the equation type: %s' % self.eqn_type)

# define system parameters and programming variables
registered_solver_classes = [ODESolver.ConformalImplicitMidpoint, ODESolver.ImplicitMidpoint] #, type('builtin', (object,),{})

m, beta, k, g, w, method, eqn_type, plt_res = 1.0, 0.1, 1.0, 0.0, lambda t: 0, None, 'linear', True
u_init = [1, 0]    # initial condition

u_exact, en_err = ExactSolution(), EnergyError()

nperiods = 15     # no of oscillation periods
# T = 2*pi*nperiods
T = 80

u0_exact = lambda t: u_exact(t)[0]
u1_exact = lambda t: u_exact(t)[1]

alg = lambda solver_class: solver_class.__name__  # (class) name of algorithm

#%%
'''
legends = []

for solver_class in registered_solver_classes:
    if solver_class == ODESolver.RungeKutta4:
        npoints_per_period = 20
    else:
        npoints_per_period = 100

    n = int(npoints_per_period*nperiods)
    t_points = linspace(0.0, T, n+1)

    if solver_class == ODESolver.ConformalImplicitMidpoint:
        f, dfdu = OscSystem(m, beta, k, g, w, alg(solver_class), eqn_type), \
            Jacobian(m, beta, k, g, w, alg(solver_class), eqn_type)
        solver = solver_class(f, dfdu, size(u_init), beta)
    elif solver_class == ODESolver.ConformalStormerVerlet:
        f = OscSystem(m, beta, k, g, w, alg(solver_class), eqn_type)
        solver = solver_class(f, beta)
    elif solver_class in [ODESolver.BackwardEuler, ODESolver.ImplicitMidpoint]:
        f, dfdu = OscSystem(m, beta, k, g, w, alg(solver_class), eqn_type), \
            Jacobian(m, beta, k, g, w, alg(solver_class), eqn_type)
        solver = solver_class(f, dfdu, size(u_init))
    else:
        f = OscSystem(m, beta, k, g, w, alg(solver_class), eqn_type)
        solver = solver_class(f, beta)

    solver.set_initial_condition(u_init)
    u, t = solver.solve(t_points)

    if plt_res:
        fig = figure()
        ax1 = fig.add_subplot(121)
        ax1.legend(["position", "momentum"], loc=4);
        plot_data(ax1, t, u, True, True)

        ax2 = fig.add_subplot(122)
        plot_data(ax2, t, en_err(u,t), True, True)
'''

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
    f, dfdu = OscSystem(alg(solver_class)), Jacobian(alg(solver_class))

    if solver_class == ODESolver.ConformalImplicitMidpoint:
        solver = solver_class(f, dfdu)
    elif solver_class == ODESolver.ConformalStormerVerlet:
        solver = solver_class(f)
    elif solver_class in [ODESolver.BackwardEuler, ODESolver.ImplicitMidpoint]:
        solver = solver_class(f, dfdu)

    if solver_class == ODESolver.ConformalImplicitMidpoint:
        dt_space, r1, C1 = test_convergence_and_energy(u_init, u0_exact, T, solver, False)
        dt_space, r2, C2 = test_convergence_and_energy(u_init, u0_exact, T, solver, True)
    elif solver_class == ODESolver.ImplicitMidpoint:
    #    r2 = nan*zeros(r1.size)
        dt_space, r3, C3 = test_convergence_and_energy(u_init, u0_exact, T, solver, False)
        dt_space, r4, C4 = test_convergence_and_energy(u_init, u0_exact, T, solver, True)

    if alg(solver_class) == 'builtin':
        dt_space, r1, C1 = test_convergence_and_energy(u_init, u0_exact, T, alg(solver_class), False, f)
        r2 = nan*zeros(r1.size)

array2print = np.array((dt_space, np.concatenate(([np.float('nan')],r1)), \
                                 np.concatenate(([np.float('nan')],r2)), \
                                np.concatenate(([np.float('nan')],r3)), \
                                np.concatenate(([np.float('nan')],r4)))).T

tex_table(alg(solver_class), array2print)

#    print type(solver).__name__, " \\\\\n".join([" & ".join(map(str,line)) \
#               for line in np.array((dt_space, np.concatenate(([np.float('nan')],r1)), \
#                                     np.concatenate(([np.float('nan')],r2)))).T])


#%% Compute conformal symplecticness error
fig = figure()

for solver_class in registered_solver_classes:
    if solver_class == ODESolver.RungeKutta4:
        npoints_per_period = 20
    else:
        npoints_per_period = 100

    n = int(npoints_per_period*nperiods)
    t_points = linspace(0.0, T, n+1)

    f, dfdu = OscSystem(alg(solver_class)), Jacobian(alg(solver_class))

    if solver_class == ODESolver.ConformalImplicitMidpoint:
        solver = solver_class(f, dfdu)
    elif solver_class == ODESolver.ImplicitMidpoint:
        solver = solver_class(f, dfdu)

    if alg(solver_class) != 'builtin':
        solver.set_initial_condition(u_init)
        u, t = solver.solve(t_points)

        dpsi, t = solver.var_solve(u, t_points)

        symp_error = solver.symplectic_error(dpsi, t)

    if alg(solver_class) == 'builtin':
        fun = lambda t, u: np.asarray(f(u, t), float)
        sol = solve_ivp(fun, [0, T], u_init, method='Radau', t_eval=t_points)
        u, t, symp_error = sol.y.T, sol.t, nan*np.zeros(t.size)

    # plot results
    if plt_res:
        if solver_class == ODESolver.ImplicitMidpoint:
            ax = fig.add_subplot(122)
            ax.set_ylim(0,0.00030)
        elif solver_class == ODESolver.ConformalImplicitMidpoint:
            ax = fig.add_subplot(121)
            # ax.set_ylim(0.0E-15,5.0E-15)
        # plot_data(ax1, t, u, True, False)
        # ax1.legend(["$q$", "$p$"])
        # ax1.set_ylim(-1,1)
        ax.set_xlabel('time')

        plot_data(ax, t, np.array([en_err(u,t), symp_error]).T, True, False)
        fig.legend([r'$\boldmath{E}_I$',r'$\boldmath{E}_{cs}$'], bbox_to_anchor=(0.5,-0.08), loc='lower center',ncol=2,bbox_transform=fig.transFigure)
        # fig.tight_layout(rect=[0,-0.2,1,1])
#        ax2.set_ylim(0.0E-15,3.5E-15)
        fig.savefig('app3_err_inv.pdf', bbox_inches='tight')
