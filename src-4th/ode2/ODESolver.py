import numpy as np
from pylab import *
from numpy import linalg as LA

class ODESolver(object):
    """
    Superclass for numerical methods solving scalar and vector ODEs

      du/dt = f(u, t)

    Attributes:
    t: array of time values
    u: array of solution values (at time points t)
    k: step number of the most recently computed solution
    f: callable object implementing f(u, t)
    """
    def __init__(self, f):
        if not callable(f):
            raise TypeError('f is %s, not a function' % type(f))
        # For ODE systems, f will often return a list, but
        # arithmetic operations with f in numerical methods
        # require that f is an array. Let self.f be a function
        # that first calls f(u,t) and then ensures that the
        # result is an array of floats.
        self.f = lambda u, t: np.asarray(f(u, t), float)

    def advance(self):
        """Advance solution one time step."""
        raise NotImplementedError

    def set_initial_condition(self, U0):
        if isinstance(U0, (float,int)):  # scalar ODE
            self.neq = 1
            U0 = float(U0)
        else:                            # system of ODEs
            U0 = np.asarray(U0)          # (assume U0 is sequence)
            self.neq = U0.size
        self.U0 = U0

        # Check that f returns correct length:
        try:
            f0 = self.f(self.U0, 0)
        except IndexError:
            raise IndexError('Index of u out of bounds in f(u,t) func. Legal indices are %s' % (str(list(range(self.neq)))))
        if f0.size != self.neq:
            raise ValueError('f(u,t) returns %d components, while u has %d components' % (f0.size, self.neq))

    def solve(self, time_points, terminate=None):
        """
        Compute solution u for t values in the list/array
        time_points, as long as terminate(u,t,step_no) is False.
        terminate(u,t,step_no) is a user-given function
        returning True or False. By default, a terminate
        function which always returns False is used.
        """
        if terminate is None:
            terminate = lambda u, t, step_no: False

        if isinstance(time_points, (float,int)):
            raise TypeError('solve: time_points is not a sequence')
        self.t = np.asarray(time_points)
        if self.t.size <= 1:
            raise ValueError('ODESolver.solve requires time_points array with at least 2 time points')

        n = self.t.size
        if self.neq == 1:  # scalar ODEs
            self.u = np.zeros(n)
        else:              # systems of ODEs
            self.u = np.zeros((n,self.neq))

        # Assume that self.t[0] corresponds to self.U0
        self.u[0] = self.U0

        # Time loop
        for k in range(n-1):
            self.k = k
            self.u[k+1] = self.advance()
            if terminate(self.u, self.t, self.k+1):
                break  # terminate loop over k
        return self.u[:k+2], self.t[:k+2]

    def var_solve(self, u, time_points, terminate=None):
        """
        Compute solution du for t values in the list/array
        time_points, as long as terminate(u,t,step_no) is False.
        terminate(u,t,step_no) is a user-given function
        returning True or False. By default, a terminate
        function which always returns False is used.
        """
        if terminate is None:
            terminate = lambda u, t, step_no: False

        self.u = u
        self.t = np.asarray(time_points)
        n = self.t.size
        if self.U0.shape != self.u[0].shape:
            self.neq = self.u[0].size
            
        if self.neq%2 == 1:  # odd number of equations
            raise ValueError('ODESolver.var_solve requires even number of equations')
        else:              # systems of ODEs
            # TODO: define sparse arrays
            self.du = np.zeros((n, self.neq, self.neq))
            self.I_mat = np.eye(self.neq)

        # Initialize du[0] with identity matrix
        self.du[0] = self.I_mat

        # Time loop
        for k in range(n-1):
            self.k = k
            self.du[k+1] = self.var_advance()
            if terminate(self.u, self.t, self.k+1):
                break  # terminate loop over k
        return self.du[:k+2], self.t[:k+2]

    def symplectic_error(self, du, t):
        '''
        Compute Symplectic error for the method
        '''
        n = t.size
        #du = self.du
        Ecoeff = self.Ecoeff
        neq = self.neq

        J_mat = np.concatenate([np.concatenate([np.zeros((neq//2,neq//2)), np.eye(neq//2)], axis=1)\
                          ,np.concatenate([-np.eye(neq//2), np.zeros((neq//2,neq//2))], axis=1)])
        J_mat_inv = J_mat.T
        symp_error = np.zeros(n)

        for k in range(n-1):
            dt = t[k+1] -t[k]
            symp_error[k+1] = LA.norm((du[k+1].T).dot(J_mat_inv.dot(du[k+1])) -Ecoeff(-dt)**4*J_mat_inv)
            
        return symp_error

class ForwardEuler(ODESolver):
    def advance(self):
        u, f, k, t = self.u, self.f, self.k, self.t
        dt = t[k+1] - t[k]
        u_new = u[k] + dt*f(u[k], t[k])
        return u_new

class RungeKutta4(ODESolver):
    def advance(self):
        u, f, k, t = self.u, self.f, self.k, self.t
        dt = t[k+1] - t[k]
        dt2 = dt/2.0
        K1 = dt*f(u[k], t[k])
        K2 = dt*f(u[k] + 0.5*K1, t[k] + dt2)
        K3 = dt*f(u[k] + 0.5*K2, t[k] + dt2)
        K4 = dt*f(u[k] + K3, t[k] + dt)
        u_new = u[k] + (1/6.0)*(K1 + 2*K2 + 2*K3 + K4)
        return u_new

class ConformalStormerVerlet(ODESolver):
    def __init__(self, f):
        ODESolver.__init__(self, f)

        self.Ecoeff = lambda dt: np.exp(f.beta*dt/2)

    def advance(self):
        u, f, k, t, Ecoeff, neq = self.u, self.f, self.k, self.t, self.Ecoeff, \
                                    self.neq
        dt = t[k+1] - t[k]
        u_new = np.zeros(neq)
        u_new[neq/2:] = Ecoeff(-dt)*u[k,neq/2:] +dt/2*f(u[k], t[k])[1]
        u_new[:neq/2] = u[k,:neq/2] +dt*f(np.reshape([u[k,:neq/2], u_new[neq/2:]],neq), t[k])[0]
        u_new[neq/2:] = Ecoeff(-dt)*(u_new[neq/2:] +dt/2*f(np.reshape([u_new[:neq/2], u_new[neq/2:]],neq), t[k])[1])
        return u_new

    def var_advance(self):
        u, var_u, dfdu, k, t, Ecoeff, neq = self.u, self.var_u, self.dfdu, self.k, \
                                            self.t, self.Ecoeff, self.neq
        dt = t[k+1] - t[k]
        var_u_new = np.zeros((neq,neq))
        u_new[neq/2:] = Ecoeff(-dt)*u[k,neq/2:] +dt/2*f(u[k], t[k])[1]

        var_u_new[neq/2:] = Ecoeff(-dt)*var_u[k][neq/2:] +dt/2*dfdu(u[k], t[k], dt)[neq/2:].dot(var_u[k])
        var_u_new[:neq/2] = var_u[k][:neq/2] +dt*dfdu(np.reshape([u[k,:neq/2], u_new[neq/2:]],neq), t[k], dt)[:neq/2].dot(np.concatenate((var_u[k][:neq/2], var_u_new[neq/2:]),axis=0))
        var_u_new[neq/2:] = Ecoeff(-dt)*(var_u_new[neq/2:] +dt/2*dfdu(np.reshape([u[k,:neq/2], u[k+1,neq/2:]],neq), t[k], dt)[neq/2:].dot(np.concatenate))
        raise NotImplementedError

import sys, os

class BackwardEuler(ODESolver):
    """Backward Euler solver for scalar or vector ODEs."""
    def __init__(self, f, dfdu=None):
        ODESolver.__init__(self, f)

        # BackwardEuler needs to import function Newton from Newton.py:
        try:
            from Newton import Newton
            self.Newton = Newton
        except ImportError:
            raise ImportError('''
Could not import module "Newton". Place Newton.py in this directory
(%s)
''' % (os.path.dirname(os.path.abspath(__file__))))

        # Select correct derivative
        if not callable(dfdu):
            try:
                value =f(np.array([1]), 1)
            except IndexError:
                raise ValueError('f(u,t) must return flaot/int')

            self.discrete_derivative =True
        else:
            neq = np.size(f.u_init)
            self.discrete_derivative = False
            self.dfdw = lambda u, t, dt: np.eye(neq)-dt*np.asarray(dfdu(u, t, dt), float)

    def advance(self):
        u, f, k, t = self.u, self.f, self.k, self.t
        dt = t[k+1] - t[k]

        def F(w):
            return w - dt*f(w, t[k+1]) - u[k]

        if self.discrete_derivative:
            dFdw = Derivative(F)
        else:
            def dFdw(w):
                dfdw = self.dfdw
                return dfdw(w, t[k+1], dt)

        w_start = u[k] + dt*f(u[k], t[k])  # Forward Euler step
        u_new, n, F_value = self.Newton(F, w_start, dFdw, N=30)
        if k == 0:
            self.Newton_iter = []
        self.Newton_iter.append(n)
        if n >= 30:
            print("Newton's failed to converge at t=%g "\
                  "(%d iterations)" % (t[k+1], n))
        return u_new

class ImplicitMidpoint(ODESolver):
    def __init__(self, f, dfdu=None):
        ODESolver.__init__(self, f)
        # try:
        #     self.dfdu = lambda u, t: np.asarray(dfdu(u,t), float)
        # except (TypeError, ValueError):
        self.dfdu = lambda u, t: dfdu(u,t)

        # Define Ecoeff for computing symplectic error
        self.Ecoeff = lambda dt: np.exp(f.beta*dt/4)

        # BackwardEuler needs to import function Newton from Newton.py:
        try:
            from Newton import Newton
            self.Newton = Newton
        except ImportError:
            raise ImportError('''
Could not import module "Newton". Place Newton.py in this directory
(%s)
''' % (os.path.dirname(os.path.abspath(__file__))))

        # Select correct derivative
        if not callable(dfdu):
            try:
                value =f(np.array([1]), 1)
            except IndexError: # must be scalar ODE
                raise ValueError('f(u,t) must return float/int')

            self.discrete_derivative =True
        else:
            neq = np.size(f.u_init)
            self.discrete_derivative = False
            # try:
            #     self.dfdw = lambda u, t, dt: \
            #                     np.eye(neq)-dt/2*np.asarray(dfdu((u[1]+u[0])/2, (t[1]+t[0])/2, dt), float)
            # except TypeError:
            self.dfdw = lambda u, t, dt: \
                            np.eye(neq)-dt/2*dfdu((u[1]+u[0])/2, (t[1]+t[0])/2, dt)

    def advance(self):
        u, f, k, t = self.u, self.f, self.k, self.t
        dt = t[k+1] - t[k]

        def F(w):
            return w - dt*f((w +u[k])/2, (t[k+1] +t[k])/2) - u[k]

        if self.discrete_derivative:
            dFdw = Derivative(F)
        else:
            def dFdw(w):
                dfdw = self.dfdw
                return dfdw([w, u[k]], [t[k+1], t[k]], dt)

        w_start = u[k] + dt*f(u[k], t[k])  # Forward Euler step
        u_new, n, F_value = self.Newton(F, w_start, dFdw, N=30)
        if k == 0:
            self.Newton_iter = []
        self.Newton_iter.append(n)
        if n >= 30:
            print("Newton's failed to converge at t=%g "\
                  "(%d iterations)" % (t[k+1], n))
        return u_new

    def var_advance(self):
        u, dfdu, k, t, I_mat = self.u, self.dfdu, self.k, self.t, self.I_mat
        dt = t[k+1] - t[k]

        temp = dt/2.0*dfdu((u[k+1] +u[k])/2.0, (t[k+1] +t[k])/2.0)
        du_new = LA.solve((I_mat -temp), (I_mat +temp))
        return du_new

class ConformalImplicitMidpoint(ODESolver):
    def __init__(self, f, dfdu=None):
        ODESolver.__init__(self, f)

        self.beta = f.beta
        self.Ecoeff = Ecoeff = lambda dt: np.exp(f.beta*dt/4)
        self.dfdu = lambda u, t: np.asarray(dfdu(u, t), float)

        # BackwardEuler needs to import function Newton from Newton.py:
        try:
            from Newton import Newton
            self.Newton = Newton
        except ImportError:
            raise ImportError('''
                Could not import module "Newton". Place Newton.py in this directory
                (%s)
                ''' % (os.path.dirname(os.path.abspath(__file__))))

        # Select correct derivative
        if not callable(dfdu):
            try:
                value =f(np.array([1]), 1)
            except IndexError: # must be scalar ODE
                raise ValueError('f(u,t) must return float/int')

            self.discrete_derivative =True
        else:
            self.discrete_derivative = False
            neq = np.size(f.u_init)
            self.dfdw = lambda u, t, dt: \
                            Ecoeff(dt)*(np.eye(neq)-dt/2*np.asarray(dfdu((Ecoeff(dt)*u[1] +Ecoeff(-dt)*u[0])/2, (Ecoeff(dt)*t[1] +Ecoeff(-dt)*t[0])/2, dt), float))

    def advance(self):
        u, f, k, t, beta, Ecoeff = self.u, self.f, self.k, self.t, self.beta, self.Ecoeff
        dt = t[k+1] - t[k]

        def F(w):
            return Ecoeff(dt)*w - dt*f((Ecoeff(dt)*w +Ecoeff(-dt)*u[k])/2, (Ecoeff(dt)*t[k+1] +Ecoeff(-dt)*t[k])/2) \
                    - Ecoeff(-dt)*u[k]

        if self.discrete_derivative:
            dFdw = Derivative(F)
        else:
            def dFdw(w):
                dfdw = self.dfdw
                return dfdw([w, u[k]], [t[k+1], t[k]], dt)

        w_start = u[k] + dt*(f(u[k], t[k]) -beta/2*u[k])  # Forward Euler step
        u_new, n, F_value = self.Newton(F, w_start, dFdw, N=30)
        if k == 0:
            self.Newton_iter = []
        self.Newton_iter.append(n)
        if n >= 30:
            print("Newton's failed to converge at t=%g "\
                  "(%d iterations)" % (t[k+1], n))
        return u_new

    def var_advance(self):
        u, dfdu, k, t, Ecoeff, I_mat = self.u, self.dfdu, self.k, self.t, \
                                        self.Ecoeff, self.I_mat
        dt = t[k+1] - t[k]

        temp = dt/2.0*dfdu((Ecoeff(dt)*u[k+1] +Ecoeff(-dt)*u[k])/2.0, (Ecoeff(dt)*t[k+1] +Ecoeff(-dt)*t[k])/2.0)
        du_new = LA.solve(Ecoeff(dt)*(I_mat -temp), Ecoeff(-dt)*(I_mat +temp))
        return du_new

class Derivative:
    def __init__(self, f, h=1E-9):
        self.f = f
        self.h = float(h)

    def __call__(self, x, discrete=True):
        f, h = self.f, self.h      # make short forms
        return (f(x+h) - f(x-h))/(2*h)

#%% Testing
def test_exact_numerical_solution():
    a = 0.2; b = 3
    alg = lambda solver_class: solver_class.__name__
    
    class fun(object):
        def __init__(self, method=None):
            self.u_init = u_exact(0)
            self.beta = 0
    
        def __call__(self, u, t):
            return a + (u - u_exact(t))**5
            
    class funJac(fun):
        def __call__(self, u, t, dt=0):
            return 5*(u - u_exact(t))**4

    def u_exact(t):
        """Exact u(t) corresponding to f above."""
        return a*t + b

    U0 = u_exact(0)
    T = 8
    n = 10
    tol = 1E-15
    t_points = np.linspace(0, T, n)
    registered_solver_classes = [ImplicitMidpoint, ConformalImplicitMidpoint]

    for solver_class in registered_solver_classes:
        f, dfdu = fun(alg(solver_class)), funJac(alg(solver_class))
        solver = solver_class(f)
        solver.set_initial_condition(U0)
        u, t = solver.solve(t_points)
        u_e = u_exact(t)
        max_error = (u_e - u).max()
        msg = '%s failed with max_error=%g' % \
              (solver.__class__.__name__, max_error)
        assert max_error < tol, msg

        solver = solver_class(f, dfdu)
        solver.set_initial_condition(U0)
        u, t = solver.solve(t_points)
        u_e = u_exact(t)
        max_error = (u_e - u).max()
        msg = '%s failed with max_error=%g' % \
              (solver.__class__.__name__, max_error)
        assert max_error < tol, msg

if __name__ == '__main__':
    test_exact_numerical_solution()
