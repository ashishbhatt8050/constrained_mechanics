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
            
class sineGordon(object):
    
    def __init__(self, kwds):
        self.kwds = kwds
        
        self.Omega2 = self.kwds['Omega2']
        "MechSystem constituents"
        c, self.beta = 0.5, 0
        L, dx = 60, 0.187
        nosc = self.nosc = int((L/dx+1))
        self.x_points = np.array([-L/2 +i*dx for i in range(nosc)])
        
        c_sq = np.sqrt(1 -c**2)
        self.y_init = r_[4 *np.arctan(np.exp((self.x_points -L/4)/c_sq)) \
                        +4 *np.arctan(np.exp((-self.x_points -L/4)/c_sq)), \
                        4*c/c_sq *(np.exp((self.x_points -L/4)/c_sq)/(1 +np.exp(2*(self.x_points -L/4)/c_sq)) \
                                    +np.exp((-self.x_points -L/4)/c_sq)/(1 +np.exp(2*(-self.x_points -L/4)/c_sq)))]
        
        FD = sp.linalg.toeplitz([-1]+[0]*(nosc-2)+[1], [-1, 1]+[0]*(nosc-2))/dx # Forward difference
        BD = sp.linalg.toeplitz([1, -1]+[0]*(nosc-2), [1]+[0]*(nosc-2)+[-1])/dx # Backward difference
        
        
        if 'P' in self.kwds:
            nosc = self.kwds['P'].shape[1]/2
            
        CD2 = sp.linalg.toeplitz([-2, 1]+[0]*(int(nosc)-3)+[1])/dx**2 # Second order central-difference
        
    
        sigma = lambda u: u**2/2
        sigma_u = lambda u: u
        eye_nosc = np.eye(nosc)
        sigma_uu = lambda u: eye_nosc
        non_f = lambda u: 1 -np.cos(u)
        non_f_u = lambda u: np.sin(u)
        non_f_uu = lambda u: np.diag(np.cos(u))
        
        self.ham = lambda u: sum(1/2*u[:,nosc:]**2 +sigma(u[:,:nosc] @BD.T) +non_f(u[:,:nosc]), axis=1)*dx
            
        self.ham_u_ = lambda u: -CD2@u +non_f_u(u)
        self.ham_v_ = lambda v: v
            
        self.ham_z_ = lambda u, v: r_[self.ham_u_(u), self.ham_v_(v)].T
        
        if 'RB' in self.kwds and 'P' not in self.kwds:
            
            if self.kwds['symplectic_mor']:
                RB = self.kwds['RB']
                RBu = lambda z: np.split(RB @z, 2, axis=0)[0]
                RBv = lambda z: np.split(RB @z, 2, axis=0)[1]
            
                self.ham_u = lambda u: -(sigma_u(FD @u) -sigma_u(BD @u))/dx +non_f_u(u)
                self.ham_v = lambda v: v
                
                self.ham_z = lambda u, v: r_[self.ham_u(RBu(r_[u, v])), \
                                            self.ham_v(RBv(r_[u, v]))]
                
                self.JJ = lambda d=nosc: r_[c_[np.zeros((d,d)), np.eye(d)],\
                                           c_[-np.eye(d), np.zeros((d,d))]]
                
                # non_f_uu_ = lambda u, RB: np.diag(np.cos(u) @RB)
                self.ham_uu = lambda u: (-CD2 +non_f_uu(u))
                self.ham_vv = lambda v: eye_nosc
                
                self.ham_zz = lambda u, v: block_diag(self.ham_uu(RBu(r_[u, v])), self.ham_vv(RBv(r_[u, v])))
                
                self.non_quad = None
                self.Q_spd = None
                
            else:
                RBu = self.kwds['RBu']
                RBv = self.kwds['RBv']
                
                FDxRB = FD @RBu
                BDxRB = BD @RBu
            
                self.ham_u = lambda u: -(sigma_u(FDxRB @u) -sigma_u(BDxRB @u))/dx +non_f_u(RBu @u)
                self.ham_v = lambda v: RBv @v
                
                self.ham_z = lambda u, v: r_[self.ham_u(u), self.ham_v(v)]
                
                self.JJ = lambda d=nosc: r_[c_[np.zeros((d,d)), np.eye(d)],\
                                           c_[-np.eye(d), np.zeros((d,d))]]
                
                # non_f_uu_ = lambda u, RB: np.diag(np.cos(u) @RB)
                self.ham_uu = lambda u: (-CD2 +non_f_uu(RBu @u))
                self.ham_vv = lambda v: eye_nosc
                
                self.ham_zz = lambda u, v: block_diag(self.ham_uu(u), self.ham_vv(v))
                
                self.non_quad = None
                self.Q_spd = None
    
        else:
            # self.ham_u = lambda u: -(sigma_u(FD @u) -sigma_u(BD @u))/dx +non_f_u(u)
            self.ham_u = lambda u: -CD2@u +non_f_u(u)
            self.ham_v = lambda v: v
            
            # def ham_z(u, v):
            #     result = np.zeros(2*nosc)
            #     result[::2] = self.ham_u(u)
            #     result[1::2] = self.ham_v(v)
            #     return result
            
            self.ham_z = lambda u, v: r_[self.ham_u(u), self.ham_v(v)]
            
            # J2 = [[0, 1],[-1, 0]]
            # rep_J2 = (J2,)*nosc
            # JJ = block_diag(*rep_J2)
            # self.JJ = lambda d=nosc: JJ
            
            self.JJ = lambda d=nosc: r_[c_[np.zeros((d,d)), np.eye(d)],\
                                       c_[-np.eye(d), np.zeros((d,d))]]
            
            self.ham_uu = lambda u: -CD2 +non_f_uu(u)
            self.ham_vv = lambda v: eye_nosc
            
            # def ham_zz(u, v):
            #     result = np.zeros_like(JJ)
            #     result[::2, ::2] = self.ham_uu(u)
            #     result[1::2, 1::2] = self.ham_vv(v)
            #     return result
            
            self.ham_zz = lambda u, v: block_diag(self.ham_uu(u), self.ham_vv(v))
            
            self.non_quad = None
        
        "Initial conditions"
        
        self.constraint_type = None
        # CD1 = sp.linalg.toeplitz([0]+[-1]+[0]*(nosc-3)+[1], [0]+[1]+[0]*(nosc-3)+[-1])/(2*dx)
        # self._g_ = lambda z, z0=self.y_init: np.sum(z[nosc:] * (CD1 @z[:nosc]), axis=0)*dx -np.dot(z0[nosc:], CD1 @z0[:nosc])*dx
        # self._g_prime_ = lambda z: r_[z[nosc:] @CD1.T, CD1 @ z[:nosc]].T*dx
        
        self._g_ = lambda z, z0=self.y_init: self.ham(z[None, :]) -self.ham(z0[None, :])
        self._g_prime_ = lambda z: self.ham_z_(z[:nosc], z[nosc:])
        
        "Constraints"
        # if self.constraint_type == 'spherical':                    
        #     self._g_ = lambda u: (u[1]**2 -np.exp(-2*self.beta*dt)*u[0]**2)*dx
        #     self._g_prime_ = lambda u: 2*u[1]
        # else:
        #     raise NameError
        
        "Fixed-point nonliner equations solver properties"
        self.tol, self.M, self.var, self.store = 1.0E-10, 100, True, True
    

    def set_constraints(self):
        
        if 'RB' in self.kwds:
            self._g = lambda y, z0=self.y_init: self._g_(self.RB @y, z0)
            self._g_prime = lambda y: self.RB.T @self._g_prime_((y @self.RB.T))
        
        else:
            self._g = self._g_
            self._g_prime = self._g_prime_
    

    def __call__(self, y, t):
        
        solver_class, beta = self.solver_class, self.beta
        P, U, RB, W_r = self.P, self.U, self.RB, self.W_r
        
        if RB is None:
            if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                f = self.JJ() @ self.ham_z(*np.split(y, 2))
            elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                f = self.JJ() @ self.ham_z(*np.split(y, 2))
                
        elif W_r is not None and P is None:
            if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                f = self.JJ_r @ RB.T @ self.ham_z(*np.split(y, 2))
            elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                f = self.JJ_r @ RB.T @ self.ham_z(*np.split(y, 2))
                
        else:
            raise NotImplementedError
                
        return f
                
        if P is not None and self.non_quad is None:
            
            if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                f = self.JJ_r @ self.hat(self.ham_z(self.PxIPxRB(y)))
            elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                f = self.JJ_r @ self.hat(self.ham_z(self.PxIPxRB(y)))
                
        elif P is not None and self.non_quad is not None:
            
            if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                f = self.JJ_r @ (y + self.hat(self.non_quad_z(self.PxIPxRB(y))))
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
                dfdy = self.JJ() @ self.ham_zz(*np.split(y, 2))
            elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                dfdy = self.JJ() @ self.ham_zz(*np.split(y, 2))
                
        elif W_r is not None and P is None:
            if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                dfdy = self.JJ_r @ RB.T @ self.ham_zz(*np.split(y, 2)) @ RB
            elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                dfdy = self.JJ_r @ RB.T @ self.ham_zz(*np.split(y, 2)) @ RB
        else:
            raise NotImplementedError
                
        return dfdy
                
        if P is not None and self.non_quad is None:
            
            if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                dfdy = self.JJ_r @ self.hhat(self.ham_zz(self.PxIPxRB(y)))
            elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                dfdy = self.JJ_r @ self.hhat(self.ham_zz(self.PxIPxRB(y)))
                
                
        elif P is not None and self.non_quad is not None:
            
            if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                dfdy = self.JJ_r @ (np.eye(self.JJ_r.shape[0]) + self.hhat(self.non_quad_zz(self.PxIPxRB(y))))
            elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                dfdy = self.JJ_r @ (np.eye(self.JJ_r.shape[0]) + self.hhat(self.non_quad_zz(self.PxIPxRB(y))))
        else:
            raise ValueError
            
        return dfdy
    

    def en_err(self):
        "Energy error"
        assert self.beta == 0, 'Energy is only defined for beta = 0'
        
        if self.P is not None:
            u = (self.IPt(self.y.T)).T
        else:
            u = self.y
            
        return log(self.ham(u)/self.ham(u[None, 0, :]))
    

    def plot(self, fig):
        # fig = plt.figure(figsize=(5.5, 3.5), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, hspace=1)
        # ax = gs.subplots()
        
        ax0 = fig.add_subplot(gs[0,0])
        ax1 = fig.add_subplot(gs[1,0])
        
        if hasattr(self, 'sym_error'):
            plot_data(ax0, self.t_points, self.sym_error)
            ax0.set_ylim((-1e-14, max(10*max(abs(self.sym_error)), 1e-14)))
            ax0.set_xlim((0, self.T_final))
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
                mom = np.hstack([self._g(self.y_red[i,:], np.zeros_like(self.y[0,:])) for i in range(self.n+1)])
                temp = abs(mom)
                # temp = r_[0, mom_error[1:]]
            else:
                mom = np.hstack([self._g(self.y[i,:], np.zeros_like(self.y[0,:])) for i in range(self.n+1)])
                temp = abs(mom)
            temp = log(temp/np.roll(temp, 1))
            temp = r_[0, temp[1:]]
                
            plot_data(ax1, self.t_points, temp)
            ax1.set_xlim((0, self.T_final))
            ax1.set_ylim((-max((temp))*1e1, max((temp))*1e1))
            # ax1.set_yticks([0, 2e-15, 4e-15, 6e-15, 8e-15, 10e-15])
            # plot_data(ax[3,0], self.t_points, temp[1])
        ax1.set_xlabel('time')
        
        
        ax4 = fig.add_subplot(gs[:, -1])
            
        plot_data(ax4, self.x_points, self.y[::100, :self.nosc].T)
                
        ax4.set_xlabel('x')
        ax4.set_ylim((-0.1, np.amax(self.y[::100, :self.nosc])+2))
        ax4.set_xlim((-30, 30))
        ax4.set_xticks([-30, -15, 0, 15, 30])
        # ax4.set_yticks([0, 2e-15, 4e-15, 6e-15, 8e-15, 10e-15])
        
        
        if self.RB is None:
            string = 'full'    
        else:
            string = self.reduced_model
                
        # fig.savefig(datetime.now().strftime('%Y-%m-%d_%H-%M_')+'app5_' +self.system_type + '_' +str(self.constraint_type) + '_' + string +'_.pdf')
        
        for ax in [ax0, ax1]:
            ax.label_outer()
            
    @staticmethod
    def get_rb(y_list, F2, MSsolvers, X, kwds, nosc_r_, ax):

        if MSsolvers[-1].non_quad:
            raise NotImplementedError
        else:
            
            if kwds['symplectic_mor']:
                RB, s = cSVD(c_[y_list, F2], MSsolvers[-1].JJ())
            
                nosc_r = argmin(abs(np.asarray([norm(s[:i])/norm(s) for i in range(len(s))]) -0.99))
                if nosc_r <  20:
                    nosc_r = nosc_r_  # ensure nosc_r is even
            
                X.update({'eye_r': np.eye(2*nosc_r)})
                    
                print('nosc_r = %s' %nosc_r)
                nCol = RB.shape[1]
                RB = RB[:, r_[range(nosc_r), range(nCol//2,nCol//2+nosc_r)]]
                W_r = (MSsolvers[-1].JJ(nosc_r).T @RB.T @MSsolvers[-1].JJ()).T
            
            else:
                nosc = MSsolvers[-1].nosc
                RBu, s = POD(c_[y_list[:nosc,:], F2[:nosc,:]], np.eye(nosc))
                RBv, s2 = POD(c_[y_list[nosc:,:], F2[nosc:,:]], np.eye(nosc))
            
                nosc_r = argmin(abs(np.asarray([norm(s[:i])/norm(s) for i in range(len(s))]) -0.99))
                if nosc_r <  20:
                    nosc_r = nosc_r_  # ensure nosc_r is even
            
                X.update({'eye_r': np.eye(2*nosc_r)})
                    
                print('nosc_r = %s' %nosc_r)
                
                RBu, RBv = RBu[:, :nosc_r], RBv[:, :nosc_r]
                RB = block_diag(RBu, RBv)
                W_r = RB
            
                kwds.update({'RBu': RBu, 'RBv': RBv})
    
        kwds.update({'RB': RB,\
                    'W_r': W_r,\
                    'nosc_r': nosc_r,\
                    # 'i_range_r': np.random.randint(0, nosc_r, 3),\
                        })
            
        if kwds['symplectic_mor'] == False:
            logplot(ax, c_[s,s2])
            ax.set_ylim((1e-15, np.amax(c_[s,s2])*1e3))
        else:
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
            
        if hasattr(self, '_g') and False:
            mom = LA.norm(self._g(self.y.T, np.zeros_like(self.y[0])), axis=0)
            mom_error = log(mom/(np.roll(mom, 1)))
            self.mom_error = r_[0, mom_error[1:]]

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
    def get_rb(y_list, F2, MSsolvers, X, kwds, nosc_r_, ax):

        if MSsolvers[-1].non_quad:
            raise NotImplementedError
        else:
            RB, s = POD(c_[y_list, F2], X['eye'])
        
            nosc_r = argmin(abs(np.asarray([norm(s[:i])/norm(s) for i in range(len(s))]) -0.99))
            if nosc_r <  20:
                nosc_r = nosc_r_  # ensure nosc_r is even
        
            X.update({'eye_r': np.eye(2*nosc_r)})
                
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
        
class Oscillators(object):
    
    def __init__(self, kwds):
        self.kwds = kwds   
        
        self.Omega2 = self.kwds['Omega2']
        
        nosc = self.nosc = self.kwds['nosc']
        Omega2 = self.Omega2
        self.Omega2_00 = lambda Omega2=Omega2: r_[Omega2, np.ones_like(Omega2)]
        
        alpha = list(np.linspace(0.1,0.5,num=nosc))
        alpha /= sqrt(sum(np.array(alpha)**2))
        assert np.isclose(np.array(alpha).dot(alpha), 1), "alpha**2 must be equal to 1"
        self.alpha = alpha
        self.beta = (max(1e-2, 0*np.random.rand()/10))

        "MechSystem constituents"
        kin = lambda u: sum((u**2), axis=1)/2.0
        pot = lambda x, Omega2=Omega2: 1-sum(cos(Omega2*x), axis=1)
        # pot = lambda x, Omega2=Omega2: 1 -sum(1-(Omega2*x)**2/2+(Omega2*x)**4/24, axis=1)
        
        "Find symbolic quantities"
        # def get_equations(d=nosc):
        #     x_ = sympy.Matrix([sympy.symbols('x%d' % i) for i in range(d)])
        #     u_ = sympy.Matrix([sympy.symbols('u%d' % i) for i in range(d)])
        #     Omega2_ = sympy.Matrix([sympy.symbols('Omega2%d' % i) for i in range(d)])
            
        #     kin_ = u_.dot(u_)/2
        #     pot_ = 1 -sum(sympy.Matrix([sympy.cos(\
        #                             sympy.matrices.dense.matrix_multiply_elementwise(Omega2_, x_)[i])\
        #                             for i in range(d)]))
        #     ham_ = kin_ +pot_
        #     f_ham = sympy.lambdify((x_,u_,Omega2_), ham_, 'numpy')
            
        #     ham_z_ = sympy.Matrix([ham_]).jacobian([x_,u_]).T
        #     f_ham_z = sympy.lambdify((x_,u_,Omega2_), ham_z_, 'numpy')
            
        #     ham_zz_ = ham_z_.jacobian([x_,u_])
        #     f_ham_zz = sympy.lambdify((x_,u_,Omega2_), ham_zz_, 'numpy')
            
        #     return f_ham, f_ham_z, f_ham_zz
        
        # f_ham, f_ham_z, f_ham_zz = get_equations()
            
        # self.ham = lambda x, u, Omega2=Omega2: f_ham(x, u, Omega2)
        # self.ham_z = lambda x, u, Omega2=Omega2: f_ham_z(x, u, Omega2)
        # self.ham_zz = lambda x, u, Omega2=Omega2: f_ham_zz(x, u, Omega2)
        
        self.ham = lambda x, u, Omega2=Omega2: pot(x, Omega2) +kin(u) +self.beta/2 * sum(x*u, axis=1)
                                                
        self.ham_z = lambda x, u, Omega2=Omega2: self.Omega2_00(Omega2) * r_[sin(Omega2*x), u] \
                                                +self.beta/2 * r_[u, x]
        self.ham_zz = lambda x, u, Omega2=Omega2: diag(self.Omega2_00(Omega2) * r_[cos(Omega2*x), np.ones_like(x)] *self.Omega2_00(Omega2)) \
                                            + self.beta/2 * r_[c_[zeros(x.shape*2), eye(x.shape[0])],\
                                                                c_[eye(x.shape[0]), zeros(x.shape*2)]]
                                                
        # self.ham_z = lambda x, u, Omega2=Omega2: self.Omega2_00(Omega2) * r_[(Omega2*x)-(Omega2*x)**3/6, u] \
        #                                         +self.beta/2 * r_[u, x]
        # self.ham_zz = lambda x, u, Omega2=Omega2: diag(self.Omega2_00(Omega2) * r_[1-(Omega2*x)**2/2, np.ones_like(x)] *self.Omega2_00(Omega2)) \
        #                                     + self.beta/2 * r_[c_[zeros(x.shape*2), eye(x.shape[0])],\
        #                                                         c_[eye(x.shape[0]), zeros(x.shape*2)]]
            
                                                
        self.Q_spd = lambda Omega2=Omega2: self.ham_zz(0*Omega2,0*Omega2, Omega2)
        self.JJ = lambda d=nosc: r_[c_[zeros((d,d)), eye(d)],\
                                      c_[-eye(d), zeros((d,d))]]
        self.drag = lambda x, u: self.beta/2 * r_[x, u]
        self.drag_z = lambda x, u: self.beta/2 * eye(2*x.shape[0])
            
        self.non_quad = lambda x, u, Omega2=Omega2: self.ham(x,u, Omega2) -1/2 *c_[x, u] @ self.Q_spd() @ c_[x, u].T
        self.non_quad_z = lambda x, u, Omega2=Omega2: self.ham_z(x,u,Omega2) - self.Q_spd(Omega2) @ r_[x, u]
        self.non_quad_zz = lambda x, u, Omega2=Omega2: self.ham_zz(x,u,Omega2) - self.Q_spd(Omega2)
        self.non_quad = None
    
        "Initial conditions"
        y_init = r_[np.linspace(1,5,nosc), np.zeros(nosc)]
        
        self.constraint_type = 'spherical'
        if self.constraint_type == 'linear':
            y_init[nosc-1] = -(y_init[:nosc-1].dot(alpha[:nosc-1]))/alpha[nosc-1] # project on the manifold
            assert np.isclose(y_init[:nosc].dot(alpha[:nosc]), 0), "alpha:y should be 0" 
        elif self.constraint_type == 'spherical':
            y_init /= sqrt((y_init[:nosc]**2).dot(alpha[:nosc])) # project on the manifold
            assert np.isclose((y_init[:nosc]**2).dot(alpha[:nosc]), 1), "alpha:y^2 should be 1" 
        elif self.constraint_type == None:
            pass
        else:
            raise NameError
        self.y_init = y_init
    
        "Constraints"
        if self.constraint_type == 'linear':
            self._g = lambda y, alpha=alpha: r_[y[:, :nosc].dot(alpha), y[:, nosc:].dot(alpha)]
            self._g_prime = lambda y, alpha=alpha: r_[c_[np.array(alpha, ndmin=2), zeros((1,nosc))],\
                                        c_[zeros((1,nosc)), np.array(alpha, ndmin=2)]]
            # self._g_prime = lambda y: block_diag((np.array(alpha, ndmin=2), np.array(alpha, ndmin=2)), format="csr")
            raise NotImplementedError
        elif self.constraint_type == 'spherical':
            Alpha = np.diag(alpha)
            self.A_mat = r_[c_[Alpha, np.zeros_like(Alpha)],\
                       c_[np.zeros_like(Alpha), np.zeros_like(Alpha)]]
            self.B_mat = r_[c_[np.zeros_like(Alpha), Alpha],\
                       c_[Alpha, np.zeros_like(Alpha)]]
                
            self._g_ = lambda y, alpha=[self.A_mat, self.B_mat]: r_[y @ alpha[0] @ y.T -1.0,\
                                                y @ alpha[1] @ y.T]
            self._g_prime_ = lambda y, alpha=[self.A_mat, self.B_mat]: c_[2*y @alpha[0],\
                                                      2*y @alpha[1]].T

        "Fixed-point nonliner equations solver properties"
        self.tol, self.M, self.var, self.store = 1.0E-15, 100, True, True

    def en_err(self):
        "Energy error"
        assert self.beta == 0, 'Energy is only defined for beta = 0'
        
        if self.P is not None:
            x, u = np.split((self.IPt(self.y.T)).T, 2, axis=1)
        else:
            x, u = np.split(self.y, 2, axis=1)
            
        if self.non_quad and self.P is not None:
            return log(\
                        (self.non_quad(x, u) +1/2 *self.y @ self.Q_spd() @self.y.T)\
                        /(self.non_quad(x[0:1, :], u[None, 0, :]) +1/2 *self.y @ self.Q_spd() @self.y.T)\
                        )
        else:
            return log(self.ham(x, u)/self.ham(x[0:1, :], u[None, 0, :]))
    

    def jacobian(self, y, t, *arg):
        
        solver_class, beta = self.solver_class, self.beta
        P, U, RB, W_r = self.P, self.U, self.RB, self.W_r
        
        if RB is None:
            x, u = np.split(y, 2)
            if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                dfdy = self.JJ() @ self.ham_zz(x, u) - self.drag_z(x, u)
            elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                dfdy = self.JJ() @ self.ham_zz(x, u)
                
        elif W_r is not None and P is None:
            y = RB @ y
            x, u = np.split(y, 2)
            if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                dfdy = self.JJ_r @ RB.T @ self.ham_zz(x, u) @ RB - self.W_r.T @ self.drag_z(x, u) @ RB
            elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                dfdy = self.JJ_r @ RB.T @ self.ham_zz(x, u) @ RB
                
        elif P is not None and self.non_quad is None:
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
    

    def __call__(self, y, t):
        
        solver_class, beta = self.solver_class, self.beta
        P, U, RB, W_r = self.P, self.U, self.RB, self.W_r
        
        if RB is None:
            x, u = np.split(y, 2)
            if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                f = self.JJ() @ self.ham_z(x, u) - self.drag(x, u)
            elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                f = self.JJ() @ self.ham_z(x, u)
                
        elif W_r is not None and P is None:
            y = RB @ y
            x, u = np.split(y, 2)
            if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                f = self.JJ_r @ RB.T @ self.ham_z(x, u) - self.W_r.T @self.drag(x, u)
            elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                f = self.JJ_r @ RB.T @ self.ham_z(x, u)
                
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
            
        return f
    

    def set_constraints(self):
        
        if 'RB' in self.kwds:
    
            A_mat_ = self.RB.T @self.A_mat @self.RB
            B_mat_ = self.RB.T @self.B_mat @self.RB
            
            self._g = lambda y, alpha=[A_mat_, B_mat_]: self._g_(y, alpha)
            
            self._g_prime = lambda y, alpha=[A_mat_, B_mat_]: self._g_prime_(y, alpha)
                
        else:
            
            self._g = self._g_
            self._g_prime = self._g_prime_
            
            
    def plot(self, fig):
        "plot the results"
        
        gs = fig.add_gridspec(2, 2, hspace=1)
        # ax = gs.subplots()
        
        ax0 = fig.add_subplot(gs[0,0])
        ax1 = fig.add_subplot(gs[1,0])
        # ax2 = fig.add_subplot(gs[2,0])
        ax4 = fig.add_subplot(gs[:,-1])
        
        if hasattr(self, 'sym_error'):
            plot_data(ax0, self.t_points, self.sym_error)
            ax0.set_ylim((-max(self.sym_error)*1e1, max(self.sym_error)*1e1))
            ax0.set_xlim((0, self.T_final))
        if hasattr(self, 'eng_error'):
            plot_data(ax1, self.t_points, self.eng_error)
            ax1.set_ylim((-max(self.eng_error)*1e1, max(self.eng_error)*1e1))
            ax1.set_xlim((0, self.T_final))
        elif hasattr(self, 'norm_error'):
            plot_data(ax1, self.t_points, self.norm_error)
            
        elif hasattr(self, '_g'):
            if hasattr(self, 'y_red'):
                temp = np.hstack([self._g(self.y_red[[i],:]) for i in range(self.n+1)])
            else:
                temp = np.hstack([self._g(self.y[[i],:]) for i in range(self.n+1)])
                
            temp = LA.norm(temp, axis=0)
            # temp = log(temp/np.roll(temp, 1))
            # temp = r_[0, temp[1:]]
            plot_data(ax1, self.t_points, temp)
            ax1.set_ylim((-max(temp)*1e1, max(temp)*1e1))
            ax1.set_xlim((0, self.T_final))
            # plot_data(ax2, self.t_points, temp[1])
            # ax2.set_ylim((min(temp[1])*1e-1, max(temp[1])*1e1))
        ax1.set_xlabel('time')
            
        # plot_data(ax[0,1], self.t_points, )
        # ax3 = fig.add_subplot(gs[0:2, -1])
        # ax4 = fig.add_subplot(gs[2:4, -1])
            
        # if self.RB is not None:
        #     plot_data(ax3, self.y_red[:,self.i_range_r], self.y_red[:,self.nosc_r+self.i_range_r])
            
        plot_data(ax4, self.y[:,self.i_range], self.y[:,self.nosc+self.i_range])
        ax4.set_xlim((np.amin(self.y[:,self.i_range])-0.2, np.amax(self.y[:,self.i_range])+0.2))
        ax4.set_ylim((np.amin(self.y[:,self.nosc+self.i_range])-0.1, np.amax(self.y[:,self.nosc+self.i_range])+0.1))
        ax4.set_aspect(1.0/ax4.get_data_ratio(), adjustable='box')
        ax4.set_xlabel('q')
        ax4.set_ylabel('p')
        ax4.grid(axis='x', color="0.9", linestyle='-', linewidth=1)
        # ax4.set_ylim((min(temp[0])*1e-1, max(temp[0])*1e1))
        # ax[3,1].set_xlabel('time')
            
        if self.RB is not None:
            string = 'reduced'
        else:
            string = 'full'
            
        # curr_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M_')
        # fig.savefig(curr_datetime +'app5_oscillator_' +string +'_.pdf', bbox_inches='tight')
        
        # for ax in [ax0, ax1]:
        #     ax.label_outer()
            
    @staticmethod
    def get_rb(y_list, F2, MSsolvers, X, kwds, nosc_r_, ax):

        if MSsolvers[-1].non_quad:
            X.update({'Q': MSsolvers[-1].Q_spd(), \
             'sqrt': sp.linalg.sqrtm(MSsolvers[-1].Q_spd())})
            
            np.linalg.cholesky(X['Q'])
            RB, s = POD(X['sqrt'] @ y_list, X['eye'])
            RB = LA.solve(X['sqrt'], RB)
        
            nosc_r = argmin(abs(np.asarray([norm(s[:i])/norm(s) for i in range(len(s))]) -0.99))
            if nosc_r <  20:
                nosc_r = nosc_r_  # ensure nosc_r is even
                
            X.update({'eye_r': np.eye(2*nosc_r)})
                
            print('nosc_r = %s' %nosc_r)
            
            RB = RB[:, :2*nosc_r]
        
            # Orthogonalize RB wrt Q_spd
            # U = sp.linalg.cholesky(RB.T @ X['Q'] @ RB) # upper triangular Cholesky factor
            # RB = RB @ sp.linalg.solve(U, X['eye_r'])
            assert np.allclose(RB.T @ X['Q'] @ RB, X['eye_r'])
            
            W_r_, s = POD(F2, X['eye'])
            # W_r_ = LA.solve(X['sqrt'], W_r_)
            W_r_ = W_r_[:, :2*nosc_r]
            ax.semilogy(s)
            ax.set_xlabel('r')
            # W_r = RB
            
            # Orthognalize W_r_ wrt RB
            W_r = orthogonalize(W_r_, RB, X['eye'])
            assert (W_r.shape == RB.shape)
            
        else:
            RB, s = POD(c_[y_list, F2], X['eye'])
        
            nosc_r = argmin(abs(np.asarray([norm(s[:i])/norm(s) for i in range(len(s))]) -0.99))
            if nosc_r <  20:
                nosc_r = nosc_r_  # ensure nosc_r is even
        
            X.update({'eye_r': np.eye(2*nosc_r)})
                
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
        
