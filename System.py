#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 12:40:39 2022

@author: bhattah
"""

import numpy as np
from pylab import log, r_, c_, cos, sin, zeros, eye, sqrt, diag, sum
import ODESolver    

#%% symbolic computation
import sympy as smp
def get_symbols(nosc, P=None, **kw):
    "Find symbolic quantities"
    
    if P is None:
        x = smp.Matrix(smp.symbols('x0:{}'.format(nosc)))
        u = smp.Matrix(smp.symbols('u0:{}'.format(nosc)))
        omega2 = smp.Matrix(smp.symbols('omega^2_0:{}'.format(nosc)))
        beta = smp.symbols('beta')
        
        kin_expr = sum((u[i]**2)/2.0 for i in range(nosc))
        # kin = smp.lambdify((u,), kin_expr, modules='numpy')
        
        pot_expr = 1 -sum(smp.cos(omega2[i]*x[i]) for i in range(nosc))
        # pot = smp.lambdify((x, omega2), pot_expr)
        
        ham_expr = kin_expr + pot_expr +beta/2 * sum(x[i]*u[i] for i in range(nosc))
        ham_lam = smp.lambdify((x, u, omega2, beta), ham_expr, "sympy")
        # print(ham_lam(x, u, omega2, beta))
        
        ham_ = smp.lambdify((x, u, omega2, beta), ham_expr)
        # self.ham = lambda x, u, omega2=self.omega2, beta=self.beta: ham_(x, u, omega2, beta)
        # u_arr, v_arr, Om2_arr = np.random.rand(3,nosc)
        # print(ham(u_arr, v_arr, Om2_arr, 0.1))
        
        ham_z_expr = smp.Matrix([ham_lam(x, u, omega2, beta)]).jacobian(list(x)+list(u)).T
        ham_z_lam = smp.lambdify((x, u, omega2, beta), ham_z_expr, "sympy")
        # print(ham_z_lam(x, u, omega2, beta))
        
        ham_z_ = smp.lambdify((x, u, omega2, beta), ham_z_expr)
        # self.ham_z = lambda x, u, omega2=self.omega2, beta=self.beta: ham_z_(x, u, omega2, beta).squeeze()
        # print(ham_z(u_arr, v_arr, Om2_arr, 0.1))
        
        ham_zz_expr = ham_z_lam(x, u, omega2, beta).jacobian(list(x)+list(u))
        # ham_zz_lam = smp.lambdify((x, u, omega2, beta), ham_zz_expr, "sympy")
        # print(ham_zz_lam(x, u, omega2, beta))
        
        ham_zz_ = smp.lambdify((x, u, omega2, beta), ham_zz_expr)
        # self.ham_zz = lambda x, u, omega2=self.omega2, beta=self.beta: ham_zz_(x, u, omega2, beta).squeeze()
        
        return ham_, ham_z_, ham_zz_, ham_z_expr, ham_zz_expr
    
    else:
        Pxham_z_ = smp.lambdify((x, u, omega2, beta), P.T @ ham_z_expr)
        Pxham_zz_ = smp.lambdify((x, u, omega2, beta), P.T @ ham_zz_expr)
        
        return ham_, Pxham_z_, Pxham_zz_

#%% define the system
class System(object):
    
    def __init__(self, kwds):
        
        self.__dict__.update(kwds)

        "MechSystem constituents"
        
        self.ham = lambda x, u, Omega2=self.Omega2, beta=self.beta: self.ham_(x, u, Omega2, beta)
        self.ham_z = lambda x, u, Omega2=self.Omega2, beta=self.beta: self.ham_z_(x, u, Omega2, beta).squeeze()
        self.ham_zz = lambda x, u, Omega2=self.Omega2, beta=self.beta: self.ham_zz_(x, u, Omega2, beta).squeeze()
            
                                                
        self.Q_spd = lambda Omega2=self.Omega2: self.ham_zz(0*Omega2,0*Omega2, Omega2)            
        self.non_quad = lambda x, u, Omega2=self.Omega2: self.ham(x,u, Omega2) -1/2 *c_[x, u] @ self.Q_spd() @ c_[x, u].T
        self.non_quad_z = lambda x, u, Omega2=self.Omega2: self.ham_z(x,u,Omega2) - self.Q_spd(Omega2) @ r_[x, u]
        self.non_quad_zz = lambda x, u, Omega2=self.Omega2: self.ham_zz(x,u,Omega2) - self.Q_spd(Omega2)
        self.non_quad = None
            
    def __call__(self, y, t, *args, **kwargs):
        
        solver_class = self.solver_class
        P, RB, W_r = self.P, self.RB, self.W_r
        
        func = kwargs['func']
        jac = kwargs['jac']
        
        if RB is None:
            x, u = np.split(y, 2)
            
            if func:
                if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                    f = self.JJ() @ self.ham_z(x, u) - self.drag(x, u)
            
                elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                    f = self.JJ() @ self.ham_z(x, u)
                    
                elif solver_class in [ODESolver.ConformalStormerVerlet]:
                    f = self.JJ() @ self.ham_z(x, u, self.Omega2, 0)
                    
                return f
                        
            if jac:
                if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                    dfdy = self.JJ() @ self.ham_zz(x, u) - self.drag_z(x, u)
            
                elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                    dfdy = self.JJ() @ self.ham_zz(x, u)
                    
                elif solver_class in [ODESolver.ConformalStormerVerlet]:
                    dfdy = self.JJ() @ self.ham_zz(x, u, self.Omega2, 0)
                    
                return dfdy
                
        elif W_r is not None and P is None:
            y_ = RB @ y
            x, u = np.split(y_, 2)
            
            if func:
                if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                    f = self.JJ_r @ RB.T @ self.ham_z(x, u) -self.drag(*np.split(y, 2))
            
                elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                    f = self.JJ_r @ RB.T @ self.ham_z(x, u)
                    
                elif solver_class in [ODESolver.ConformalStormerVerlet]:
                    f = self.JJ_r @ RB.T @ self.ham_z(x, u, self.Omega2, 0)
                    
                return f
                        
            if jac:
                if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                    dfdy = self.JJ_r @ RB.T @ self.ham_zz(x, u) @ RB - self.drag_z(*np.split(y, 2))
            
                elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                    dfdy = self.JJ_r @ RB.T @ self.ham_zz(x, u) @ RB
                    
                elif solver_class in [ODESolver.ConformalStormerVerlet]:
                    dfdy = self.JJ_r @ RB.T @ self.ham_zz(x, u, self.Omega2, 0) @ RB
                    
                return dfdy
                
        elif P is not None and self.non_quad is None:
            y_ = RB @ y
            x, u = np.split(y_, 2)
                
            if func:
            
                if solver_class in [ODESolver.ImplicitMidpoint]:
                    f = self.JJ_r @ self.hat(self.ham_z(x, u)) - self.drag(*np.split(y, 2))
                    
                elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                    f = self.JJ_r @ self.hat(self.ham_z(x, u))
                    
                elif solver_class in [ODESolver.ConformalStormerVerlet]:
                    f = self.JJ_r @ self.hat(self.ham_z(x, u, self.Omega2, 0))
                    
                return f
            
            elif jac:
        
                if solver_class in [ODESolver.ImplicitMidpoint]:
                    dfdy = self.JJ_r @ self.hhat(self.ham_zz(x, u)) - self.drag_z(*np.split(y, 2))
                    
                elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                    dfdy = self.JJ_r @ self.hhat(self.ham_zz(x, u))
                    
                elif solver_class in [ODESolver.ConformalStormerVerlet]:
                    dfdy = self.JJ_r @ self.hhat(self.ham_zz(x, u, self.Omega2, 0))
                    
                return dfdy
                
        elif P is not None and self.non_quad is not None:
            x_, u_ = np.split(RB @y, 2)
            x, u = np.split(self.PxIPxRB(y), 2)
            
            if x.shape != self.Omega2.shape:
                Omega2 = P.T @ r_[self.Omega2, self.Omega2]
                Omega2, _ = np.split(Omega2, 2)
            else:
                Omega2 = self.Omega2
            
            if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                f = self.JJ_r @ (y + self.hat(self.non_quad_z(x, u, Omega2)))\
                    - self.W_r.T @self.drag(x_, u_)
            elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                f = self.JJ_r @ (y + self.hat(self.non_quad_z(x, u, Omega2)))
                
        else:
            raise NotImplementedError
        
    def get_jacobian(self, y, t):
        
        solver_class = self.solver_class
        P, RB = self.P, self.RB
                
        if P is not None and self.non_quad is None:
            x_, u_ = np.split(RB @y, 2)
            
            x, u = np.split(self.PxIPxRB(y), 2)
            
            if x.shape != self.Omega2.shape:
                Omega2 = P.T @ r_[self.Omega2, self.Omega2]
                Omega2, _ = np.split(Omega2, 2)
            else:
                Omega2 = self.Omega2
            
            if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                dfdy = self.JJ_r @ self.hhat(self.ham_zz(x, u, Omega2)) - self.W_r.T @self.drag_z(x_, u_) @ RB
            elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                dfdy = self.JJ_r @ self.hhat(self.ham_zz(x, u, Omega2))
                
                
        elif P is not None and self.non_quad is not None:
            x_, u_ = np.split(RB @y, 2)
            x, u = np.split(self.PxIPxRB(y), 2)
            
            if x.shape != self.Omega2.shape:
                Omega2 = P.T @ r_[self.Omega2, self.Omega2]
                Omega2, _ = np.split(Omega2, 2)
            else:
                Omega2 = self.Omega2
            
            if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                dfdy = self.JJ_r @ (np.eye(self.JJ_r.shape[0]) + self.hhat(self.non_quad_zz(x, u, Omega2)))\
                    - self.W_r.T @self.drag_z(x_, u_)@ RB
            elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                dfdy = self.JJ_r @ (np.eye(self.JJ_r.shape[0]) + self.hhat(self.non_quad_zz(x, u, Omega2)))
                
        else:
            raise NotImplementedError
            
        return dfdy
        
    def get_en_err(self):
        
        # if self.P is not None:
        #     x, u = np.split((self.IPt(self.y.T)).T, 2, axis=1)
        # else:
            
        x, u = np.split(self.y.T, 2, axis=0)
            
        if self.non_quad and self.P is not None:
            return log(\
                        (self.non_quad(x, u) +1/2 *self.y @ self.Q_spd() @self.y.T)\
                        /(self.non_quad(x[0:1, :], u[None, 0, :]) +1/2 *self.y @ self.Q_spd() @self.y.T)\
                        )
        else:
            return log(self.ham(x, u)/self.ham(x[:, 0], u[:, 0]))

            
if __name__ == '__main__':
    pass