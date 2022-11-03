#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 16:41:09 2022

@author: bhattah
"""

import numpy as np
from numpy import linalg as LA
from pylab import *
import ODESolver
import scipy as sp
from scipy.linalg import block_diag
from PlotScript import plot_data
from datetime import datetime
from podDEIM import DEIM, orthogonalize, POD, cSVD
from PlotScript import tex_table, logplot

class KdV(object):
    
    def __init__(self, kwds):
        self.kwds = kwds
        
        self.Omega2 = self.kwds['Omega2']
    
        "MechSystem constituents"
        nu, alpha, rho, self.beta = -1e-5, -3/8, -1e-1, 0
        L, dx = 4, 0.0808
        self.nosc = int((2*L/dx +1)/2)
        self.x_points = np.array([-L +i*dx for i in range(2*self.nosc)])
        self.y_init = 2*np.exp(-2*self.x_points**2)/(sqrt(2)*np.pi)
        
        nosc = self.nosc
        
        FD = sp.linalg.toeplitz([-1]+[0]*(2*nosc-2)+[1], [-1]+[1]+[0]*(2*nosc-2))/dx # Forward difference
        self.ham = lambda u: sum(alpha/3*u**3 +rho/2*u**2 -nu/2*(u @FD.T)**2, axis=1)
        
        
        if 'P' in self.kwds:
            nosc = self.kwds['P'].shape[1]/2
            
        CD2 = sp.linalg.toeplitz([-2, 1]+[0]*(int(2*nosc)-3)+[1])/dx**2 # Second order central-difference
        nuxCD2 = nu*CD2
        alphax2 = alpha *2
        
        if 'RB' in self.kwds and 'P' not in self.kwds:
            rhoxRB = rho*self.kwds['RB']
            nuxCD2xRB = nuxCD2 @self.kwds['RB']
            alphaxRBx2 = 2 *alpha *self.kwds['RB']
            sqrt_abs_alphaxRB = sqrt(abs(alpha))*self.kwds['RB']
            self.ham_z = lambda u: (-(sqrt_abs_alphaxRB @u)**2 +rhoxRB @u +nuxCD2xRB @u)
            self.ham_zz = lambda u: (np.diag(alphaxRBx2 @u +rho) +nuxCD2)
        
            eye_2nosc = eye(2*nosc)
            betaxRB  = self.beta*self.kwds['RB']
            self.drag = lambda u: betaxRB @u
            self.drag_z = lambda u: self.beta*eye_2nosc
            
            nosc = self.nosc
            self.Q_spd = lambda u=self.y_init: self.ham_zz(np.zeros_like(u))
            CD1 = sp.linalg.toeplitz([0]+[-1]+[0]*(2*nosc-3)+[1], [0]+[1]+[0]*(2*nosc-3)+[-1])/(2*dx) # second order central-difference
            self.JJ = lambda d=nosc: CD1
        else:
            
            
            self.ham_z = lambda u: (alpha*u**2 +rho*u +nuxCD2 @u)
            self.ham_zz = lambda u: (np.diag(alphax2*u +rho) +nuxCD2)
        
            self.drag = lambda u: self.beta*u
            self.drag_z = lambda u: self.beta*np.eye(len(u))
            
            nosc = self.nosc
            self.Q_spd = lambda u=self.y_init: self.ham_zz(np.zeros_like(u))
            CD1 = sp.linalg.toeplitz([0]+[-1]+[0]*(2*nosc-3)+[1], [0]+[1]+[0]*(2*nosc-3)+[-1])/(2*dx) # second order central-difference
            self.JJ = lambda d=nosc: CD1
            
            self.non_quad = lambda u: sum(alpha/3*u**3 +rho/2*u**2 -nu/2*(u @FD.T)**2 -1/2 *u @self.Q_spd() @u.T, axis=1)
            self.non_quad_z = lambda u: (self.ham_z(u) -self.Q_spd(u) @u)
            self.non_quad_zz = lambda u: (self.ham_zz(u) -self.Q_spd(u))
            self.non_quad = None
    
        "Initial conditions"
        
        self.constraint_type = 'spherical'
        #FIXME: lift-up the constraints.
    
        "Constraints"
        # if self.constraint_type == 'spherical':                    
        #     self._g_ = lambda u: (u[1]**2 -np.exp(-2*self.beta*dt)*u[0]**2)*dx
        #     self._g_prime_ = lambda u: 2*u[1]
        # else:
        #     raise NameError

        "Fixed-point nonliner equations solver properties"
        self.tol, self.M, self.var, self.store = 1.0E-10, 100, False, True
    

    def __call__(self, y, t):
        
        solver_class, beta = self.solver_class, self.beta
        P, U, RB, W_r = self.P, self.U, self.RB, self.W_r
        
        if RB is None:
            if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                f = self.JJ() @ self.ham_z(y) - self.drag(y)
            elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                f = self.JJ() @ self.ham_z(y)
                
        elif W_r is not None and P is None:
            if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                f = self.JJ_r @ RB.T @ self.ham_z(y) - self.W_r.T @self.drag(y)
            elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                f = self.JJ_r @ RB.T @ self.ham_z(y)
                
        elif P is not None and self.non_quad is None:
            
            if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                f = self.JJ_r @ self.hat(self.ham_z(self.PxIPxRB(y))) - self.W_r.T @self.drag(RB @y)
            elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                f = self.JJ_r @ self.hat(self.ham_z(self.PxIPxRB(y)))
                
        elif P is not None and self.non_quad is not None:
            
            if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                f = self.JJ_r @ (y + self.hat(self.non_quad_z(self.PxIPxRB(y))))\
                    - self.W_r.T @self.drag(RB @y)
            elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                f = self.JJ_r @ (y + self.hat(self.non_quad_z(self.PxIPxRB(y))))
        else:
            raise ValueError
            
        return f
    

    def jacobian(self, y, t, *arg):
        
        solver_class, beta = self.solver_class, self.beta
        P, U, RB, W_r = self.P, self.U, self.RB, self.W_r
        
        if RB is None:
            if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                dfdy = self.JJ() @ self.ham_zz(y) - self.drag_z(y)
            elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                dfdy = self.JJ() @ self.ham_zz(y)
                
        elif W_r is not None and P is None:
            if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                dfdy = self.JJ_r @ RB.T @ self.ham_zz(y) @ RB - self.W_r.T @ self.drag_z(y) @ RB
            elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                dfdy = self.JJ_r @ RB.T @ self.ham_zz(y) @ RB
                
        elif P is not None and self.non_quad is None:
            
            if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                dfdy = self.JJ_r @ self.hhat(self.ham_zz(self.PxIPxRB(y))) - self.W_r.T @self.drag_z(RB @y) @ RB
            elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                dfdy = self.JJ_r @ self.hhat(self.ham_zz(self.PxIPxRB(y)))
                
                
        elif P is not None and self.non_quad is not None:
            
            if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                dfdy = self.JJ_r @ (np.eye(self.JJ_r.shape[0]) + self.hhat(self.non_quad_zz(self.PxIPxRB(y))))\
                    - self.W_r.T @self.drag_z(RB @y)@ RB
            elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                dfdy = self.JJ_r @ (np.eye(self.JJ_r.shape[0]) + self.hhat(self.non_quad_zz(self.PxIPxRB(y))))
        else:
            raise ValueError
            
        return dfdy
    

    def set_constraints(self):
        
        pass
    

    def en_err(self):
        "Energy error"
        assert self.beta == 0, 'Energy is only defined for beta = 0'
            
        if self.P is not None:
            u = (self.IPt(self.y.T)).T
        else:
            u = self.y
            
        return log(self.ham(u)/self.ham(u[None, 0, :]))
    

    def plot(self, fig):
        "plot the results"
        
        # fig = plt.figure(figsize=(5.5, 3.5), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, hspace=1)
        # ax = gs.subplots()
        
        ax0 = fig.add_subplot(gs[0,0])
        ax1 = fig.add_subplot(gs[1,0])
        
        if hasattr(self, 'sym_error'):
            plot_data(ax0, self.t_points, self.sym_error)
            ax0.set_ylim((-1e-14, max(10*max(abs(self.sym_error)), 1e-14)))
            # ax0.set_yticks([])
            # ax0.set_yticks([0, 2e-15, 4e-15, 6e-15, 8e-15, 10e-15])
        if hasattr(self, 'eng_error'):
            plot_data(ax1, self.t_points, self.eng_error)
        elif hasattr(self, 'norm_error'):
            plot_data(ax1, self.t_points, self.norm_error)
        elif hasattr(self, 'mom_error'):
            plot_data(ax1, self.t_points, self.mom_error)
            
        if hasattr(self, '_g'):
            if hasattr(self, 'y_red'):
                temp = np.hstack([self._g(self.y_red[i,:]) for i in range(self.n+1)])
            else:
                temp = np.hstack([self._g(self.y[i,:]) for i in range(self.n+1)])
                
            plot_data(ax1, self.t_points, temp)
            ax1.set_ylim((-1e-14, 10*max(temp)))
            # ax1.set_yticks([0, 2e-15, 4e-15, 6e-15, 8e-15, 10e-15])
            # plot_data(ax[3,0], self.t_points, temp[1])
        ax1.set_xlabel('time')
            
        
        # plot_data(ax[0,1], self.t_points, )
        # ax3 = fig.add_subplot(gs[0:2, -1])
        ax4 = fig.add_subplot(gs[:, -1])
                
        plot_data(ax4, self.x_points, self.y[[0,-1]].T)
                
        ax4.set_xlabel('x')
        ax4.set_ylim((-0.1, np.amax(self.y[::100, :self.nosc])+1))
        ax4.set_xlim((-30, 30))
        ax4.set_xticks([-30, -15, 0, 15, 30])
        # ax4.set_yticks([0, 2e-15, 4e-15, 6e-15, 8e-15, 10e-15])
        
        
        if self.RB is None:
            string = 'full'    
        else:
            string = self.reduced_model
                
        # fig.savefig('app5_' +self.system_type + '_' + string +'_.pdf', bbox_inches='tight')
        
        for ax in [ax0, ax1]:
            ax.label_outer()
          
    @staticmethod
    def get_rb(y_list, F2, F3, MSsolvers, kwds, ax):

        if MSsolvers[-1].non_quad:
            raise NotImplementedError
        else:
            RB, s = POD(c_[y_list, F2], kwds['eye'])
        
            nosc_r = argmin(abs(np.asarray([norm(s[:i])/norm(s) for i in range(len(s))]) -0.99))
            if nosc_r <  20:
                nosc_r = kwds['nosc_r_']  # ensure nosc_r is even
        
            kwds.update({'eye_r': np.eye(2*nosc_r)})
                
            print('nosc_r = %s' %nosc_r)
            
            RB = RB[:, :2*nosc_r]
            W_r = RB
        
            kwds.update({'RB': RB,\
                        'W_r': W_r,\
                        'nosc_r': nosc_r,\
                        # 'i_range_r': np.random.randint(0, nosc_r, 3),\
                            })
                
            logplot(ax, s)
            ax.set_ylim((1e-15, max(s)*1e3))
                
            ax.set_xlabel('index of the singular value')
            ax.set_xlim((0, len(s)+50))
    
# fig.savefig(datetime.now().strftime('%Y-%m-%d_%H-%M_')+'app5_' +MSsolvers_r[-1].system_type + '_' +str(MSsolvers_r[-1].constraint_type) + '_' + MSsolvers_r[-1].reduced_model +'_' + 'singular_values' +'_.pdf')
    

    def get_errors(self, errors, dt):
    
        if (not self.beta) and (self.constraint_type is None):
            "Energy (Hamiltonian) is an invariant for unconstrained conservative system"
            self.eng_error =self.en_err()
            errors['energy'].append(sqrt(dt)*LA.norm(self.eng_error))
        
        if self.var:
            errors['spl'].append(sqrt(dt)*LA.norm(self.sym_error))
            
        norm_y = LA.norm(self.y, axis=1)
        norm_error = log(norm_y/(np.exp(-2*self.beta*self.dt)*np.roll(norm_y, 1)))
        self.norm_error = r_[0, norm_error[1:]]
        