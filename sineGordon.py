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
        c, self.beta = 0.5, 0.01
        L, dx = 40, 0.187
        dt = kwds['dt']
        nosc = self.nosc = int((L/dx+1))
        self.x_points = np.array([-L/2 +i*dx for i in range(nosc)])
        
        c_sq = np.sqrt(1 -c**2)
        self.y_init = r_[4 *np.arctan(np.exp((self.x_points -L/4)/c_sq)) \
                        +4 *np.arctan(np.exp((-self.x_points -L/4)/c_sq)), \
                        4*c/c_sq *(np.exp((self.x_points -L/4)/c_sq)/(1 +np.exp(2*(self.x_points -L/4)/c_sq)) \
                                    +np.exp((-self.x_points -L/4)/c_sq)/(1 +np.exp(2*(-self.x_points -L/4)/c_sq)))]
        
        FD = sp.linalg.toeplitz([-1]+[0]*(nosc-2)+[1], [-1, 1]+[0]*(nosc-2))/dx # Forward difference
        BD = sp.linalg.toeplitz([1, -1]+[0]*(nosc-2), [1]+[0]*(nosc-2)+[-1])/dx # Backward difference
        Ax = sp.linalg.toeplitz([1]+[0]*(nosc-2)+[1], [1, 1]+[0]*(nosc-2))/2 # Forward average
        self.Ax = block_diag(Ax, Ax)
        
        if 'P' in self.kwds:
            nosc = self.kwds['P'].shape[1]/2
            
        CD2 = sp.linalg.toeplitz([-2, 1]+[0]*(int(nosc)-3)+[1])/dx**2 # Second order central-difference
        
    
        sigma = lambda u: u**2/2
        sigma_u = lambda u: u
        eye_nosc = np.eye(nosc)
        u_one = np.ones(nosc)
        sigma_uu = lambda u: eye_nosc
        non_f = lambda u, v=0: 1 -np.cos(u) +0*0.2*(1 -np.cos(2*u)) -0.01*u*np.cos(self.t_points)[:,None] +0.5*self.beta*u*v
        non_f_u = lambda u, v=0, t=0: np.sin(u) +0*0.2*(2*np.sin(2*u)) -0.01*np.cos(t)*u_one +0.5*self.beta*v
        non_f_uu = lambda u: np.diag(np.cos(u) +0*0.2*2**2*np.cos(2*u))
        
        ham_ = lambda u: 1/2*u[:,nosc:]**2 +sigma(u[:,:nosc] @BD.T) +non_f(*np.split(u,2,axis=1))
        # self.energy_residual = lambda u: np.diff(ham_(u), axis=0)/kwds['dt']
        
        
        if ODESolver.EulerBox in kwds['registered_solver_classes']:
            mom_ = lambda u: u[:,nosc:]*sigma_u(u[:,:nosc] @FD.T)
            mom_residual = lambda u: -np.vstack([np.mean(mom_(u[i:i+2]), axis=0) for i in range(self.n)])
            dudt = lambda u: ((np.roll(np.exp(0.25*self.beta*dt)*u, -1, axis=0) -np.exp(-0.25*self.beta*dt)*u)/dt)[:-1]
            energy_cons_res = lambda u: dudt(ham_(u)) +mom_residual(u) @BD.T
            self.energy_cons_res = lambda u: np.vstack((np.zeros((1,nosc)), energy_cons_res(u)))
            
            # assuming sigma = lambda u: u**2/2
            self.mom_cons_res = lambda u: np.vstack((np.zeros((1,nosc)), dudt(-(u[:,:nosc] @BD.T)*u[:,nosc:]) \
                                -(non_f(u[:,:nosc]) -0.5*self.beta*np.roll(u[:,:nosc]*u[:,nosc:], -1, axis=0) -0.5*(np.roll(u[:,nosc:], -1, axis=0))**2 - 0.5*(u[:,:nosc] @BD.T)**2)[:-1] @FD.T))
                
        elif ODESolver.PreissmanBox in kwds['registered_solver_classes']:
            S = lambda u: 1/2*u[:,nosc:]**2 -(u[:,:nosc] @FD.T)**2 +sigma(u[:,:nosc] @FD.T) +non_f(u[:,:nosc])
            IE = lambda u: S((u @self.Ax.T)) -((u[:,:nosc] @Ax.T @CD2)) *((u[:,:nosc] @FD.T))
            IF = lambda u: np.diff((u[:,:nosc] @Ax @FD.T), axis=0)/kwds['dt'] *(u[0:-1,:nosc] @Ax @FD.T)
            self.energy_cons_res = lambda u: np.vstack((np.zeros((1,nosc)), np.diff(IE(u), axis=0)/kwds['dt'])) \
                                +np.vstack((np.zeros((1,nosc)), [np.mean(IF(u)[i:i+2], axis=0) for i in range(self.n)])) @FD.T
            # self.energy_cons_res = lambda u: np.vstack((np.zeros((1,nosc)), energy_cons_res(u)))
            
            # assuming sigma = lambda u: u**2/2
            IM = lambda u: ((u[:, nosc:] @FD.T) * u[:, nosc:]) @Ax.T
            II = lambda u: S(np.vstack([np.mean(u[i:i+2], axis=0) for i in range(self.n)])) \
                            -np.diff(u[:, nosc:], axis=0)/kwds['dt'] * np.vstack([np.mean(u[i:i+2, nosc:], axis=0) for i in range(self.n)])
            self.mom_cons_res = lambda u: np.vstack((np.zeros((1,nosc)), np.diff(IM(u), axis=0)/kwds['dt'] +II(u) @Ax.T))
        
        self.ham = lambda u: sum(ham_(u), axis=1)*dx
            
        self.ham_u_ = lambda u, v=0, t=0: -CD2@u +non_f_u(u, v, t)
        self.ham_v_ = lambda u=0, v=0: v +0.5*self.beta*u
            
        self.ham_z_ = lambda u, v, t=0: r_[self.ham_u_(u, v, t), self.ham_v_(u, v)].T
        
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
            self.ham_u = lambda u, v, t=0: -CD2@u +non_f_u(u, v, t)
            self.ham_v = lambda u, v: v +0.5*self.beta*u
            
            # def ham_z(u, v):
            #     result = np.zeros(2*nosc)
            #     result[::2] = self.ham_u(u)
            #     result[1::2] = self.ham_v(v)
            #     return result
            
            self.ham_z = lambda u, v, t=0: r_[self.ham_u(u, v, t), self.ham_v(u, v)]
            
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
        
        "Constraints"
        self.constraint_type = None
        # CD1 = sp.linalg.toeplitz([0]+[-1]+[0]*(nosc-3)+[1], [0]+[1]+[0]*(nosc-3)+[-1])/(2*dx)
        # self._g_ = lambda z, z0=self.y_init: np.sum(z[nosc:] * (CD1 @z[:nosc]), axis=0)*dx -np.dot(z0[nosc:], CD1 @z0[:nosc])*dx
        # self._g_prime_ = lambda z: r_[z[nosc:] @CD1.T, CD1 @ z[:nosc]].T*dx
        
        self._g_ = lambda z, z0=self.y_init: self.ham(z) -self.ham(z0)
        self._g_prime_ = lambda z, t=0: self.ham_z_(z[:nosc], z[nosc:], t)
        
        "Fixed-point nonliner equations solver properties"
        self.tol, self.M, self.var, self.store = 1.0E-10, 100, False, False
    

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
            if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.EulerBox, ODESolver.PreissmanBox]:
                f = self.JJ() @ self.ham_z(*np.split(y, 2), t)
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
            if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.PreissmanBox]:
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
        # if hasattr(self, 'eng_error'):
        #     plot_data(ax1, self.t_points, self.eng_error)
        # elif hasattr(self, 'mom_error'):
        #     plot_data(ax1, self.t_points, self.mom_error)
            
        if hasattr(self, '_g'):
            if hasattr(self, 'y_red'):
                mom = np.hstack([self._g(self.y_red[i,:], np.zeros_like(self.y[0,:])) for i in range(self.n+1)])
                temp = abs(mom)
                # temp = r_[0, mom_error[1:]]
            else:
                # mom = np.hstack([self._g(self.y[i,:], np.zeros_like(self.y[0,:])) for i in range(self.n+1)])
                mom = self._g(self.y, np.zeros_like(self.y))
                temp = abs(mom)
            temp = log(temp/np.roll(temp, 1))
            temp = r_[0, temp[1:]]
                
            plot_data(ax1, self.t_points, temp)
            # ax1.set_xlim((0, self.T_final))
            # ax1.set_ylim((-max((temp))*1e1, max((temp))*1e1))
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
            
        from mpl_toolkits.mplot3d import axes3d
        import matplotlib.pyplot as plt
        
        # Set up a figure twice as tall as it is wide
        fig3d = plt.figure(figsize=(27,9)) #plt.figaspect(0.5))
        # fig3d.tight_layout(rect=[0,0,.8,.8])
        
        ax3d = fig3d.add_subplot(131, projection="3d")
        # fig3d, ax3d = plt.subplots(1,2,figsize=plt.figaspect(0.5), gridspec_kw={'width_ratios': [1, 1.5]}, projection="3d")
        X, Y = np.meshgrid(self.x_points, self.t_points)
        # Z = np.sin(np.pi*X)*np.sin(np.pi*Y)
        Z = self.energy_cons_res(self.y)
        ax3d.plot_wireframe(X, Y, Z)
        
        ax3d = fig3d.add_subplot(132, projection="3d")
        # top_offset = .07
        # left_offset = .15
        # right_offset = .2
        # bottom_offset = .13
        # hgap = .1
        # ax_width = 1-left_offset - right_offset
        # ax_height = (1-top_offset - bottom_offset - hgap)/2
        # ax3d.set_axes([left_offset, bottom_offset, ax_width, ax_height])
        # Z = np.sin(np.pi*X)*np.sin(np.pi*Y)
        Z = self.mom_cons_res(self.y)
        ax3d.plot_wireframe(X, Y, Z)
        # ax3d.legend(['plot'],loc=5)
        # ax1.contour(X, Y, Z, 10, lw=3, cmap="autumn_r", linestyles="solid", offset=-1)
        # ax1.contour(X, Y, Z, 10, lw=3, colors="k", linestyles="solid")
        # fig3d.tight_layout(pad=2)
        
        ax3d = fig3d.add_subplot(133, projection="3d")
        
        plt.show()
        
            
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

