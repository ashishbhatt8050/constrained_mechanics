#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 12:40:39 2022

@author: bhattah
"""

import numpy as np
from pylab import log, r_, c_
import ODESolver

#%% define the system

class System(object):
    
    def __init__(self, kwds):
        
        self.__dict__.update(kwds)
        
        if 'pool' in kwds:
            self.__dict__.update(kwds['pool'])

        "MechSystem constituents"
        self.ham = lambda x, u, Omega2=self.Omega2, beta=self.beta: self.ham_(x, u, Omega2, beta)
        self.ham_z = lambda x, u, Omega2=self.Omega2, beta=self.beta: self.ham_z_(x, u, Omega2, beta)
        self.ham_zz = lambda x, u, Omega2=self.Omega2, beta=self.beta: self.ham_zz_(x, u, Omega2, beta)
                                                
        self.Q_spd = lambda Omega2=self.Omega2: self.ham_zz(0*Omega2, 0*Omega2, Omega2, 0)
        self.non_quad = lambda x, u, Omega2=self.Omega2: self.ham(x,u, Omega2) -1/2 *c_[x, u] @ self.Q_spd() @ c_[x, u].T
        self.non_quad_z = lambda x, u, Omega2=self.Omega2: self.ham_z(x,u,Omega2) - self.Q_spd(Omega2) @ r_[x, u]
        self.non_quad_zz = lambda x, u, Omega2=self.Omega2: self.ham_zz(x,u,Omega2) - self.Q_spd(Omega2)
        self.non_quad = None
            
    def __call__(self, y, t, *args, **kwargs):
        
        solver_class = self.solver_class
        
        if self.RB is None:
            x, u = np.split(y, 2)
            JJ = self.JJ()
        else:
            y_ = self.RB @ y
            x, u = np.split(y_, 2)
            JJ = self.JJ_r
            
        if kwargs['func']:
            if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                f = JJ @ self.ham_z(x, u) - self.drag(*np.split(y, 2))
        
            elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                f = JJ @ self.ham_z(x, u)
                
            elif solver_class in [ODESolver.ConformalStormerVerlet]:
                f = JJ @ self.ham_z(x, u, self.Omega2, 0)
                
            return f
                    
        if kwargs['jac']:
            if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                dfdy = JJ @ self.ham_zz(x, u) - self.drag_z(*np.split(y, 2))
        
            elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                dfdy = JJ @ self.ham_zz(x, u)
                
            elif solver_class in [ODESolver.ConformalStormerVerlet]:
                dfdy = JJ @ self.ham_zz(x, u, self.Omega2, 0)
                
            return dfdy
                
        # else:
        #     y_ = RB @ y
        #     x, u = np.split(y_, 2)
            
        #     if func:
        #         if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
        #             f = self.JJ_r @ self.ham_z(x, u) -self.drag(*np.split(y, 2))
            
        #         elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
        #             f = self.JJ_r @ self.ham_z(x, u)
                    
        #         elif solver_class in [ODESolver.ConformalStormerVerlet]:
        #             f = self.JJ_r @ self.ham_z(x, u, self.Omega2, 0)
                    
        #         return f
                        
        #     if jac:
        #         if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
        #             dfdy = self.JJ_r @ self.ham_zz(x, u) - self.drag_z(*np.split(y, 2))
            
        #         elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
        #             dfdy = self.JJ_r @ self.ham_zz(x, u)
                    
        #         elif solver_class in [ODESolver.ConformalStormerVerlet]:
        #             dfdy = self.JJ_r @ self.ham_zz(x, u, self.Omega2, 0)
                    
        #         return dfdy
        
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
            return self.ham(x, u) - self.ham(x[:, 0], u[:, 0])

            
if __name__ == '__main__':
    pass