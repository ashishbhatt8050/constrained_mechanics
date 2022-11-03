#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 14:50:42 2022

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
    def get_rb(y_list, F2, F3, MSsolvers, kwds, ax):

        if MSsolvers[-1].non_quad:
            raise NotImplementedError
        else:
            
            if kwds['symplectic_mor']:
                RB, s = cSVD(c_[y_list, F2], MSsolvers[-1].JJ())
            
                nosc_r = argmin(abs(np.asarray([norm(s[:i])/norm(s) for i in range(len(s))]) -0.99))
                if nosc_r <  20:
                    nosc_r = kwds['nosc_r_']  # ensure nosc_r is even
            
                kwds.update({'eye_r': np.eye(2*nosc_r)})
                    
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
                    nosc_r = kwds['nosc_r_']  # ensure nosc_r is even
            
                kwds.update({'eye_r': np.eye(2*nosc_r)})
                    
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

