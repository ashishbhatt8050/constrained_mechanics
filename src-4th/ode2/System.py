#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 12:40:39 2022

@author: bhattah
"""

import numpy as np
from pylab import log, r_, c_, cos, sin, zeros, eye, sqrt, diag, sum
import ODESolver

class System(object):
    
    def __init__(self):

        "MechSystem constituents"
        nosc = self.nosc
        Omega2 = self.Omega2        
        
        self.ham = lambda x, u, Omega2=self.Omega2, beta=self.beta: self.ham_(x, u, Omega2, beta)
        self.ham_z = lambda x, u, Omega2=self.Omega2, beta=self.beta: self.ham_z_(x, u, Omega2, beta).squeeze()
        self.ham_zz = lambda x, u, Omega2=self.Omega2, beta=self.beta: self.ham_zz_(x, u, Omega2, beta).squeeze()

        "MechSystem constituents"
        # self.Omega2_00 = lambda Omega2=Omega2: r_[Omega2, np.ones_like(Omega2)]        
        # kin = lambda u: sum((u**2), axis=1)/2.0
        # pot = lambda x, Omega2=Omega2: 1-sum(cos(Omega2*x), axis=1)
        # pot = lambda x, Omega2=Omega2: 1 -sum(1-(Omega2*x)**2/2+(Omega2*x)**4/24, axis=1)
        # print(ham_zz(u_arr, v_arr, Om2_arr, 0.1))
        
        # self.ham = lambda x, u, Omega2=Omega2, beta=self.beta: pot(x, Omega2) +kin(u) +beta/2 * sum(x*u, axis=1)
                                                
        # self.ham_z = lambda x, u, Omega2=Omega2, beta=self.beta: self.Omega2_00(Omega2) * r_[sin(Omega2*x), u] \
        #                                         +beta/2 * r_[u, x]
        # self.ham_zz = lambda x, u, Omega2=Omega2, beta=self.beta: diag(self.Omega2_00(Omega2) * r_[cos(Omega2*x), np.ones_like(x)] *self.Omega2_00(Omega2)) \
        #                                     + beta/2 * r_[c_[zeros(x.shape*2), eye(x.shape[0])],\
        #                                                         c_[eye(x.shape[0]), zeros(x.shape*2)]]
            
                                                
        self.Q_spd = lambda Omega2=self.Omega2: self.ham_zz(0*Omega2,0*Omega2, Omega2)            
        self.non_quad = lambda x, u, Omega2=self.Omega2: self.ham(x,u, Omega2) -1/2 *c_[x, u] @ self.Q_spd() @ c_[x, u].T
        self.non_quad_z = lambda x, u, Omega2=self.Omega2: self.ham_z(x,u,Omega2) - self.Q_spd(Omega2) @ r_[x, u]
        self.non_quad_zz = lambda x, u, Omega2=self.Omega2: self.ham_zz(x,u,Omega2) - self.Q_spd(Omega2)
        self.non_quad = None
            
    def get_func(self, y, t, *args, **kwargs):
        
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
            y = RB @ y
            x, u = np.split(y, 2)
            
            if func:
                if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                    f = self.JJ_r @ RB.T @ self.ham_z(x, u) - self.W_r.T @self.drag(x, u)
            
                elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                    f = self.JJ_r @ RB.T @ self.ham_z(x, u)
                    
                elif solver_class in [ODESolver.ConformalStormerVerlet]:
                    f = self.JJ_r @ RB.T @ self.ham_z(x, u, self.Omega2, 0)
                    
                return f
                        
            if jac:
                if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                    dfdy = self.JJ_r @ RB.T @ self.ham_zz(x, u) @ RB - self.W_r.T @ self.drag_z(x, u) @ RB
            
                elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                    dfdy = self.JJ_r @ RB.T @ self.ham_zz(x, u) @ RB
                    
                elif solver_class in [ODESolver.ConformalStormerVerlet]:
                    dfdy = self.JJ_r @ RB.T @ self.ham_zz(x, u, self.Omega2, 0) @ RB
                    
                return dfdy
                
        elif P is not None and self.non_quad is None:
            x_, u_ = np.split(RB @y, 2)
            
            x, u = np.split(self.PxIPxRB(y), 2)
            
            if x.shape != self.Omega2.shape:
                Omega2 = P.T @ r_[self.Omega2, self.Omega2]
                Omega2, _ = np.split(Omega2, 2)
            else:
                Omega2 = self.Omega2
            
            if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                f = self.JJ_r @ self.hat(self.ham_z(x, u, Omega2)) - self.W_r.T @self.drag(x_, u_)
            elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                f = self.JJ_r @ self.hat(self.ham_z(x, u, Omega2))
                
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
        
        if self.P is not None:
            x, u = np.split((self.IPt(self.y.T)).T, 2, axis=1)
        else:
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