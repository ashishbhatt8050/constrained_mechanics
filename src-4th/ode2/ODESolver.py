import numpy as np
from pylab import r_, c_
from numpy import linalg as LA
import os

import torch
import torch.linalg as torchLA
torch.set_grad_enabled(False)
# torch.cuda.empty_cache()

from System import MechSystem
from Newton import Newton

class ODESolver(MechSystem):
    """
    Subclass of numerical methods solving scalar and vector ODEs

      du/dt = f(u, t)
      
     defined in MechSystem parent class.

    Attributes:
    t: array of time values
    u: array of solution values (at time points t)
    k: step number of the most recently computed solution
    f: callable object implementing f(u, t)
    """
        
    def __init__(self, kwds):
        MechSystem.__init__(self, kwds)
        
        self.f = lambda u, t, *arg: self.__call__(u, t, arg, func=True)
        self.dfdu = lambda u, t, *arg: self.__call__(u, t, arg, func=False)
        self.Ecoeff = lambda dt: np.exp(self.beta*dt/2)
            
    def __call__(self, y, t, *y1, **kwargs):
        """Base implementation - must be overridden"""
        raise NotImplementedError("Subclasses must implement __call__")

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
        # TODO: get rid of this check
        try:
            # Determine function based on the solver class
            if self.solver_class.__name__ == "DiscreteGradientSolver":
                f0 = self.f(c_[self.U0, self.U0].T, 0)
            else:
                f0 = self.f(self.U0, 0, self.U0)
                
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
            self.dt = self.t[k+1] -self.t[k]
            self.u[k+1], info_ = self.advance()
            if terminate(self.u, self.t, self.k+1):
                break  # terminate loop over k
        return self.u[:k+2], self.t[:k+2], info_

    def var_solve(self, u, terminate=None):
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
        self.t = np.asarray(self.t_points)
        n = self.t.size - 1
        if self.U0.shape != self.u[0].shape:
            self.neq = self.u[0].size
            
        if self.neq % 2 == 1:  # odd number of equations
            raise ValueError('ODESolver.var_solve requires even number of equations')
        else:  # systems of ODEs
            self.du = np.zeros((n + 1, self.neq, self.neq))
            self.I_mat = np.eye(self.neq)

        # Initialize du[0] with identity matrix
        self.du[0] = self.I_mat

        # Time loop
        self.dt = self.t[1] - self.t[0]
        for k in range(n):
            self.k = k
            self.du[k + 1] = self.var_advance()
            if terminate(self.u, self.t, self.k + 1):
                break  # terminate loop over k
        return self.du, self.t

    def symplectic_error(self, du):
        '''
        Compute Symplectic error for the method
        '''
        # n = t.size
        #du = self.du
        Ecoeff_dt = self.Ecoeff(-(self.t_points[1] - self.t_points[0]))**4

        if hasattr(self, 'JJ_r'):
            J_mat = self.JJ_r
        elif hasattr(self, 'JJ'):
            J_mat = self.JJ
        else:
            raise ValueError
            
        # Convert the numpy arrays to PyTorch tensors
        du = torch.from_numpy(du)
        J_mat = torch.from_numpy(J_mat)
        
        # Move the tensors to the desired device (e.g. CUDA or CPU)
        device = torch.device('cpu') # if torch.cuda.is_available() else 'cpu')
        # print(f'{device = }')
        du = du.to(device)
        J_mat = J_mat.to(device)
        Ecoeff_dt = torch.tensor(Ecoeff_dt, device=device)
        
        # Perform the operation
        du_transpose = du.transpose(1, 2)
        du_solve = torchLA.solve(J_mat, du)
        du_solve_transpose = torchLA.solve(J_mat, torch.eye(J_mat.shape[0], device=device, dtype=J_mat.dtype))
        symp_error = torch.log(torchLA.norm(du_transpose @ du_solve, dim=(1, 2)) / torchLA.norm(Ecoeff_dt * du_solve_transpose))
        
        # Move the result back to CPU
        symp_error = symp_error.cpu().numpy()
        
        return symp_error

class ForwardEuler(ODESolver):
    def __call__(self, y, t, *y1, **kwargs):
        f = self.JJ @ self.ham_z(y) - self.drag(y)
        dfdy = self.JJ @ self.ham_zz(y) - self.drag_z(y)
        return f if kwargs['func'] else dfdy
    
    def advance(self):
        u, f, k, t = self.u, self.f, self.k, self.t
        dt = t[k+1] - t[k]
        u_new = u[k] + dt*f(u[k], t[k])
        return u_new

class BackwardEuler(ODESolver):
    """Backward Euler solver for scalar or vector ODEs."""
    def __init__(self, MechSys):
        ODESolver.__init__(self, MechSys)

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
        if not callable(MechSys):
            try:
                _ =MechSys(np.array([1]), 1)
            except IndexError:
                raise ValueError('MechSys(u,t) must return flaot/int')

            self.discrete_derivative =True
        else:
            neq = np.size(MechSys.u_init)
            self.discrete_derivative = False
            self.dfdw = lambda u, t, dt: np.eye(neq)-dt*np.asarray(self.dfdu(u, t, dt), float)

    def advance(self):
        u, f, k, t = self.u, self.f, self.k, self.t
        dt = self.dt

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
        if n >= 100:
            print("Newton's failed to converge at t=%g "\
                  "(%d iterations)" % (t[k+1], n))
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
    def __call__(self, y, t, *y1, **kwargs):
        f = self.JJ @ self.ham_z(y, 0)
        dfdy = self.JJ @ self.ham_zz(y, 0)
        return f if kwargs['func'] else dfdy
    
    def __init__(self, kwds):
        ODESolver.__init__(self, kwds)
        
        self.Gamma_p, self.Gamma_m = lambda dt: 1 + self.beta*dt/2, lambda dt: 1 - self.beta*dt/2

    def advance(self):
        u, f, k, t, neq = self.u, self.f, self.k, self.t, self.neq
        
        dt = self.dt
        Gamma_p, Gamma_m = self.Gamma_p(dt), self.Gamma_m(dt)
        
        # Split u into two parts for easier manipulation
        u_upper = u[k, :neq // 2]
        u_lower = u[k, neq // 2:]

        # Calculate intermediate values
        Ecoeff_dt = self.Ecoeff(-dt)
        f_lower = f(u[k], t[k])[neq // 2:]
        
        u_new = np.zeros(neq)
        u_new_lower = (Ecoeff_dt * u_lower + dt/2 * f_lower)/Gamma_p
        u_new_upper = (Gamma_p * Ecoeff_dt * u_upper + dt * f(np.concatenate([u_upper, u_new_lower]), t[k])[:neq//2]) * Ecoeff_dt / Gamma_m
        u_new_lower = Ecoeff_dt * (Gamma_m * u_new_lower + dt/2 * f(np.concatenate([u_new_upper, u_new_lower]), t[k])[neq//2:])
        
        # Combine u_new_lower and u_new_upper
        u_new = np.concatenate((u_new_upper, u_new_lower))
        
        return u_new, u_new

    def var_advance(self):
        u, f, dfdu, k, t, Ecoeff, neq = self.u, self.f, self.dfdu, self.k, \
                                            self.t, self.Ecoeff, self.neq
        dt = self.dt
        Gamma_p, Gamma_m = self.Gamma_p(dt), self.Gamma_m(dt)
                
        u_new = (Ecoeff(-dt)*u[k,neq//2:] +dt/2*f(u[k], t[k])[neq//2:])/Gamma_p
        dfdu_k = dfdu(np.concatenate([u[k,:neq//2], u_new]), t[k])
        
        H_qq_m = -dfdu_k[neq//2:,:neq//2]
        H_qq_p = -dfdu(np.concatenate([u[k+1,:neq//2], u_new]), t[k])[neq//2:,:neq//2]
        H_pp = dfdu_k[:neq//2, neq//2:]
        
        du_11 = (Gamma_p*np.eye(neq//2) -dt**2*H_pp @ H_qq_m/2/Gamma_p)/Gamma_m
        du_12 = dt*H_pp/Gamma_p/Gamma_m
        du_21 = -Gamma_m/Gamma_p * dt/2 * H_qq_m - dt/2 * H_qq_p @ du_11
        du_22 = Gamma_m/Gamma_p*np.eye(neq//2) - dt/2 * H_qq_p @ du_12
        
        du = r_[c_[du_11, du_12], c_[du_21, du_22]]*Ecoeff(-2*dt)
        
        return du
    
class ImplicitMidpoint(ODESolver):
    def __call__(self, y, t, *y1, **kwargs):
        f = self.JJ @ self.ham_z(y) - self.drag(y)
        dfdy = self.JJ @ self.ham_zz(y) - self.drag_z(y)
        return f if kwargs['func'] else dfdy
    
    def __init__(self, kwds):
        ODESolver.__init__(self, kwds)
            
        self.dfdw = lambda u, t: np.eye(self.neq) - self.dt/2*self.dfdu((u[1]+u[0])/2, (t[1]+t[0])/2)

    def advance(self, w_start=None):
        u, f, k, t = self.u, self.f, self.k, self.t
        dt = self.dt

        def F(w):
            return w - dt*f((w +u[k])/2, (t[k+1] +t[k])/2) - u[k]

        def dFdw(w):
            return self.dfdw([w, u[k]], [t[k+1], t[k]])

        if w_start is None: w_start = u[k] + dt*f(u[k], t[k])  # Forward Euler step
        u_new, n, info = Newton(F, w_start, dFdw, N=100, store=True)
        if k == 0:
            self.Newton_iter = []
        self.Newton_iter.append(n)
        if n >= 100:
            print("Newton's failed to converge at t=%g "\
                  "(%d iterations)" % (t[k+1], n))
        return u_new, info

    def var_advance(self):
        u, k, t, I_mat = self.u, self.k, self.t, self.I_mat
        dt = self.dt

        temp = dt/2.0*self.dfdu((u[k+1] +u[k])/2.0, (t[k+1] +t[k])/2.0)
        du_new = LA.solve((I_mat -temp), (I_mat +temp))
        return du_new

 
class DiscreteGradient(ODESolver):
    def __call__(self, y, t, *y1, **kwargs):
        f = self.lag_dg(y)
        dfdy = self.lag_dg_z(y)
        return f if kwargs['func'] else dfdy
    
    def __init__(self, kwds):
        ODESolver.__init__(self, kwds)
        
        self.dfdw = lambda u, t: np.eye(self.neq) - self.dt * self.dfdu(u, t)

    def advance(self, w_start=None):
        u, f, k, t = self.u, self.f, self.k, self.t
        dt = self.dt
        
        def F(w):
            return w - u[k] - dt*f(c_[u[k], w].T, t[k])

        def dFdw(w):
            return self.dfdw(c_[u[k], w].T, t[k])

        if w_start is None:
            w_start = u[k] + dt*f(c_[u[k], u[k]].T, t[k])  # Forward Euler step
            # w_start[self.neq//2:] = u[k, self.neq//2:] + dt*f(u[k], t[k], r_[w_start[:self.neq//2], u[k, self.neq//2:]])[self.neq//2:]
            
            # uk = u[k] if self.RB is None else u[k] @ self.RB.T
            # w_start = u[k] + dt * self.JJ(self.neq//2) @ self.ham_z(*np.split(uk, 2))
        
        u_new, n, info = Newton(F, w_start, dFdw, N=100, store=True)
        
        if n >= 100:
            print(f"Newton's failed to converge at t={t[k+1]:.6f} ({n} iterations)")

        return u_new, info

    def var_advance(self):
        u, k, t, I_mat = self.u, self.k, self.t, self.I_mat
        dt = self.dt

        temp = dt*self.dfdu(u[k:k+2], (t[k+1] +t[k])/2.0)
        du_new = LA.solve((I_mat -temp), (I_mat +temp))
        return du_new


class ConformalImplicitMidpoint(ImplicitMidpoint):
    """Conformal Implicit Midpoint solver with energy-preserving capabilities"""
    
    def __init__(self, kwds):
        ImplicitMidpoint.__init__(self, kwds)
        # self.solver = kwds['pool']['solver_class'](kwds)

    def __call__(self, y, t, *y1, **kwargs):
        """Compute vector field and its Jacobian"""
        f = self.JJ @ self.ham_z(y)  # Conservative part only
        dfdy = self.JJ @ self.ham_zz(y)  # Conservative part Jacobian
        return f if kwargs['func'] else dfdy

    def advance(self):
        """Advance solution one time step using conformal integration"""        
        k = self.k
        dt = self.dt

        # Scale initial condition by conformal factor
        self.u[k] = self.Ecoeff(-dt)*self.u[k]

        # Forward Euler predictor with conformal modification
        w_start = self.u[k] + dt*(self.f(self.u[k], self.t[k]) 
                                 - self.beta/2*self.u[k])
        
        # Implicit midpoint corrector 
        u_new, info = ImplicitMidpoint.advance(self, w_start=w_start)
        
        # Scale final solution by conformal factor
        u_new = self.Ecoeff(-dt)*u_new
        info = self.Ecoeff(-dt)*np.array(info)
        
        # Restore initial condition
        self.u[k] = self.Ecoeff(dt)*self.u[k]
                
        return u_new, info

    def var_advance(self):
        """Advance variational equation for error analysis"""
        u, k, t = self.u, self.k, self.t
        Ecoeff, I_mat = self.Ecoeff, self.I_mat
        dt = self.dt

        # Compute midpoint with conformal scaling
        midpoint = (Ecoeff(dt)*u[k+1] + Ecoeff(-dt)*u[k])/2.0
        midtime = (Ecoeff(dt)*t[k+1] + Ecoeff(-dt)*t[k])/2.0
        
        # Compute Jacobian at midpoint
        temp = dt/2.0*self.dfdu(midpoint, midtime)
        
        # Solve variational equation
        du_new = LA.solve(Ecoeff(dt)*(I_mat - temp), 
                            Ecoeff(-dt)*(I_mat + temp))
        return du_new

class Derivative:
    def __init__(self, f, h=1E-9):
        self.f = f
        self.h = float(h)

    def __call__(self, x, discrete=True):
        f, h = self.f, self.h      # make short forms
        return (f(x + h) - f(x - h)) / (2 * h)

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
        # solver = solver_class(f)
        # solver.set_initial_condition(U0)
        # u, t = solver.solve(t_points)
        # u_e = u_exact(t)
        # max_error = (u_e - u).max()
        # msg = '%s failed with max_error=%g' % \
        #       (solver.__class__.__name__, max_error)
        # assert max_error < tol, msg

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
