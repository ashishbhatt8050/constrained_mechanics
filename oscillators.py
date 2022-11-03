#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 10:22:40 2022

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
        
        self.ham = lambda x, u: pot(x, Omega2) +kin(u) +self.beta/2 * sum(x*u, axis=1)
        
        if 'P' not in kwds:
                                                
            self.ham_z = lambda x, u: self.Omega2_00(Omega2) * r_[sin(Omega2*x), u] \
                                                    +self.beta/2 * r_[u, x]
            self.ham_zz = lambda x, u: diag(self.Omega2_00(Omega2) * r_[cos(Omega2*x), np.ones_like(x)] *self.Omega2_00(Omega2)) \
                                                + self.beta/2 * r_[c_[zeros(x.shape*2), eye(x.shape[0])],\
                                                                    c_[eye(x.shape[0]), zeros(x.shape*2)]]
            
            self.Q_spd = lambda Omega2=Omega2: diag(self.Omega2_00(Omega2) * r_[cos(Omega2*0), np.ones_like(Omega2)] *self.Omega2_00(Omega2)) \
                                                + 0*self.beta/2 * r_[c_[zeros(Omega2.shape*2), eye(Omega2.shape[0])],\
                                                                    c_[eye(Omega2.shape[0]), zeros(Omega2.shape*2)]]
            
            self.non_quad = lambda x, u: self.ham(x,u) -1/2 *c_[x, u] @ self.Q_spd() @ c_[x, u].T
            self.non_quad_z = lambda x, u: self.ham_z(x,u) - self.Q_spd(Omega2) @ r_[x, u]
            self.non_quad_zz = lambda x, u: self.ham_zz(x,u) - self.Q_spd(Omega2)
                                                  
            
        else:
            Omega2_ = kwds['P'].T @ r_[self.Omega2, self.Omega2]
            Omega2_, _ = np.split(Omega2_, 2)
            
            self.ham_z = lambda x, u: self.Omega2_00(Omega2_) * r_[sin(Omega2_*x), u] \
                                                    +self.beta/2 * r_[u, x]
            self.ham_zz = lambda x, u: diag(self.Omega2_00(Omega2_) * r_[cos(Omega2_*x), np.ones_like(x)] *self.Omega2_00(Omega2_)) \
                                                + self.beta/2 * r_[c_[zeros(x.shape*2), eye(x.shape[0])],\
                                                                    c_[eye(x.shape[0]), zeros(x.shape*2)]]                                              
            
            self.Q_spd = lambda Omega2=Omega2: diag(self.Omega2_00(Omega2) * r_[cos(Omega2*0), np.ones_like(Omega2)] *self.Omega2_00(Omega2)) \
                                                + 0*self.beta/2 * r_[c_[zeros(Omega2.shape*2), eye(Omega2.shape[0])],\
                                                                    c_[eye(Omega2.shape[0]), zeros(Omega2.shape*2)]]
            
            if kwds['non_quad']:
                self.non_quad = lambda x, u: self.ham(x,u) -1/2 *c_[x, u] @ self.Q_spd() @ c_[x, u].T
                self.non_quad_z = lambda x, u: self.ham_z(x,u) - self.Q_spd(Omega2_) @ r_[x, u]
                self.non_quad_zz = lambda x, u: self.ham_zz(x,u) - self.Q_spd(Omega2_)
            else:
                self.non_quad = None
            
        # self.non_quad = None     
            
        self.JJ = lambda d=nosc: r_[c_[zeros((d,d)), eye(d)],\
                                      c_[-eye(d), zeros((d,d))]]
        self.drag = lambda x, u: self.beta/2 * r_[x, u]
        self.drag_z = lambda x, u: self.beta/2 * eye(2*x.shape[0])
    
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
            
            # if x.shape != self.Omega2.shape:
            #     Omega2 = P.T @ r_[self.Omega2, self.Omega2]
            #     Omega2, _ = np.split(Omega2, 2)
            # else:
            #     Omega2 = self.Omega2
            
            if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                dfdy = self.JJ_r @ self.hhat(self.ham_zz(x, u)) - self.W_r.T @self.drag_z(x_, u_) @ RB
            elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                dfdy = self.JJ_r @ self.hhat(self.ham_zz(x, u))
                
                
        elif P is not None and self.non_quad is not None:
            x_, u_ = np.split(RB @y, 2)
            x, u = np.split(self.PxIPxRB(y), 2)
            
            # if x.shape != self.Omega2.shape:
            #     Omega2 = P.T @ r_[self.Omega2, self.Omega2]
            #     Omega2, _ = np.split(Omega2, 2)
            # else:
            #     Omega2 = self.Omega2
            
            if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                dfdy = self.JJ_r @ (np.eye(self.JJ_r.shape[0]) + self.hhat(self.non_quad_zz(x, u)))\
                    - self.W_r.T @self.drag_z(x_, u_)@ RB
            elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                dfdy = self.JJ_r @ (np.eye(self.JJ_r.shape[0]) + self.hhat(self.non_quad_zz(x, u)))
                
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
            
            # if x.shape != self.Omega2.shape:
            #     Omega2 = P.T @ r_[self.Omega2, self.Omega2]
            #     Omega2, _ = np.split(Omega2, 2)
            # else:
            #     Omega2 = self.Omega2
            
            if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                f = self.JJ_r @ self.hat(self.ham_z(x, u)) - self.W_r.T @self.drag(x_, u_)
            elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                f = self.JJ_r @ self.hat(self.ham_z(x, u))
                
        elif P is not None and self.non_quad is not None:
            x_, u_ = np.split(RB @y, 2)
            x, u = np.split(self.PxIPxRB(y), 2)
            
            # if x.shape != self.Omega2.shape:
            #     Omega2 = P.T @ r_[self.Omega2, self.Omega2]
            #     Omega2, _ = np.split(Omega2, 2)
            # else:
            #     Omega2 = self.Omega2
            
            if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                f = self.JJ_r @ (y + self.hat(self.non_quad_z(x, u)))\
                    - self.W_r.T @self.drag(x_, u_)
            elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                f = self.JJ_r @ (y + self.hat(self.non_quad_z(x, u)))
                
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
            ax0.set_ylim((-max(self.sym_error+[1e-16])*1e1, max(self.sym_error+[1e-16])*1e1))
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
    def get_rb(y_list, F2, F3, MSsolvers, kwds, ax):

        if kwds['non_quad']:
            kwds.update({'Q': MSsolvers[-1].Q_spd(), \
             'sqrt': sp.linalg.sqrtm(MSsolvers[-1].Q_spd())})
            
            np.linalg.cholesky(kwds['Q']) # gives error if Q is not positive-definite
            RB, s = POD(kwds['sqrt'] @ c_[y_list, F2], kwds['eye'])
            RB = LA.solve(kwds['sqrt'], RB)
        
            nosc_r = argmin(abs(np.asarray([norm(s[:i])/norm(s) for i in range(len(s))]) -0.99))
            if nosc_r <  20:
                nosc_r = kwds['nosc_r_']  # ensure nosc_r is even
                
            kwds.update({'eye_r': np.eye(2*nosc_r)})
                
            print('nosc_r = %s' %nosc_r)
            
            RB = RB[:, :2*nosc_r]
        
            # Orthogonalize RB wrt Q_spd
            # U = sp.linalg.cholesky(RB.T @ kwds['Q'] @ RB) # upper triangular Cholesky factor
            # RB = RB @ sp.linalg.solve(U, kwds['eye_r'])
            assert np.allclose(RB.T @ kwds['Q'] @ RB, kwds['eye_r'])
            
            W_r_, s_ = POD(F2, kwds['eye'])
            # W_r_ = LA.solve(kwds['sqrt'], W_r_)
            W_r_ = W_r_[:, :2*nosc_r]
            
            # Orthognalize W_r_ wrt RB
            W_r = orthogonalize(W_r_, RB, kwds['eye'])
            assert (W_r.shape == RB.shape)
        
            kwds.update({'RB': RB,\
                        'W_r': W_r,\
                        'nosc_r': nosc_r,\
                            })
                
            logplot(ax, s)
            logplot(ax, s_)
            ax.set_ylim((1e-15, np.amax(c_[s,s_])*1e3))
                
            ax.set_xlabel('index of the singular values')
            ax.set_xlim((0, max(len(s), len(s_))+50))
            
        else:
            RB, s = POD(c_[y_list, F2], kwds['eye'])
        
            nosc_r = argmin(abs(np.asarray([norm(s[:i])/norm(s) for i in range(len(s))]) -0.99))
            if nosc_r <  20:
                nosc_r = kwds['nosc_r_']  # ensure nosc_r is even
        
            kwds.update({'eye_r': np.eye(2*nosc_r)})
                
            print('nosc_r = %s' %nosc_r)
            
            RB = RB[:, :2*nosc_r]
            W_r = RB
        
            # assert that W_r and RB are orthogonal
            assert np.allclose(W_r.T @ RB, kwds['eye_r'])
        
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
        
