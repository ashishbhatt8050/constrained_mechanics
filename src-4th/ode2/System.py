#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 12:40:39 2022

@author: bhattah
"""

import numpy as np
from numpy import linalg as LA
from pylab import *
import ODESolver
import scipy as sp

class System(object):
    
    @staticmethod
    def get_en_err(obj):
        
        if obj.system_type == 'oscillator':
        
            if obj.P is not None:
                x, u = np.split((obj.IPt(obj.y.T)).T, 2, axis=1)
            else:
                x, u = np.split(obj.y, 2, axis=1)
                
            if obj.non_quad and obj.P is not None:
                return log(\
                            (obj.non_quad(x, u) +1/2 *obj.y @ obj.Q_spd() @obj.y.T)\
                            /(obj.non_quad(x[0:1, :], u[None, 0, :]) +1/2 *obj.y @ obj.Q_spd() @obj.y.T)\
                            )
            else:
                return log(obj.ham(x, u)/obj.ham(x[0:1, :], u[None, 0, :]))
            
        else:
            
        
            if obj.P is not None:
                u = (obj.IPt(obj.y.T)).T
            else:
                u = obj.y
                
            return log(obj.ham(u)/obj.ham(u[None, 0, :]))
    
    @staticmethod
    def get_jacobian(obj, y, t):
        
        solver_class, beta = obj.solver_class, obj.beta
        P, U, RB, W_r = obj.P, obj.U, obj.RB, obj.W_r
        
        if obj.system_type == 'oscillator':
        
            if RB is None:
                x, u = np.split(y, 2)
                if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                    dfdy = obj.JJ() @ obj.ham_zz(x, u) - obj.drag_z(x, u)
                elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                    dfdy = obj.JJ() @ obj.ham_zz(x, u)
                elif solver_class in [ODESolver.ConformalStormerVerlet]:
                    dfdy = obj.JJ() @ obj.ham_zz(x, u, obj.Omega2, 0)
                    
            elif W_r is not None and P is None:
                y = RB @ y
                x, u = np.split(y, 2)
                if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                    dfdy = obj.JJ_r @ RB.T @ obj.ham_zz(x, u) @ RB - obj.W_r.T @ obj.drag_z(x, u) @ RB
                elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                    dfdy = obj.JJ_r @ RB.T @ obj.ham_zz(x, u) @ RB
                elif solver_class in [ODESolver.ConformalStormerVerlet]:
                    dfdy = obj.JJ_r @ RB.T @ obj.ham_zz(x, u, obj.Omega2, 0) @ RB
                    
            elif P is not None and obj.non_quad is None:
                x_, u_ = np.split(RB @y, 2)
                
                x, u = np.split(obj.PxIPxRB(y), 2)
                
                if x.shape != obj.Omega2.shape:
                    Omega2 = P.T @ r_[obj.Omega2, obj.Omega2]
                    Omega2, _ = np.split(Omega2, 2)
                else:
                    Omega2 = obj.Omega2
                
                if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                    dfdy = obj.JJ_r @ obj.hhat(obj.ham_zz(x, u, Omega2)) - obj.W_r.T @obj.drag_z(x_, u_) @ RB
                elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                    dfdy = obj.JJ_r @ obj.hhat(obj.ham_zz(x, u, Omega2))
                    
                    
            elif P is not None and obj.non_quad is not None:
                x_, u_ = np.split(RB @y, 2)
                x, u = np.split(obj.PxIPxRB(y), 2)
                
                if x.shape != obj.Omega2.shape:
                    Omega2 = P.T @ r_[obj.Omega2, obj.Omega2]
                    Omega2, _ = np.split(Omega2, 2)
                else:
                    Omega2 = obj.Omega2
                
                if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                    dfdy = obj.JJ_r @ (np.eye(obj.JJ_r.shape[0]) + obj.hhat(obj.non_quad_zz(x, u, Omega2)))\
                        - obj.W_r.T @obj.drag_z(x_, u_)@ RB
                elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                    dfdy = obj.JJ_r @ (np.eye(obj.JJ_r.shape[0]) + obj.hhat(obj.non_quad_zz(x, u, Omega2)))
                    
            else:
                raise NotImplementedError
                
            return dfdy
        else:
        
            if RB is None:
                if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                    dfdy = obj.JJ() @ obj.ham_zz(y) - obj.drag_z(y)
                elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                    dfdy = obj.JJ() @ obj.ham_zz(y)
                    
            elif W_r is not None and P is None:
                if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                    dfdy = obj.JJ_r @ RB.T @ obj.ham_zz(y) @ RB - obj.W_r.T @ obj.drag_z(y) @ RB
                elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                    dfdy = obj.JJ_r @ RB.T @ obj.ham_zz(y) @ RB
                    
            elif P is not None and obj.non_quad is None:
                
                if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                    dfdy = obj.JJ_r @ obj.hhat(obj.ham_zz(obj.PxIPxRB(y))) - obj.W_r.T @obj.drag_z(RB @y) @ RB
                elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                    dfdy = obj.JJ_r @ obj.hhat(obj.ham_zz(obj.PxIPxRB(y)))
                    
                    
            elif P is not None and obj.non_quad is not None:
                
                if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                    dfdy = obj.JJ_r @ (np.eye(obj.JJ_r.shape[0]) + obj.hhat(obj.non_quad_zz(obj.PxIPxRB(y))))\
                        - obj.W_r.T @obj.drag_z(RB @y)@ RB
                elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                    dfdy = obj.JJ_r @ (np.eye(obj.JJ_r.shape[0]) + obj.hhat(obj.non_quad_zz(obj.PxIPxRB(y))))
            else:
                raise ValueError
                
            return dfdy
    
    @staticmethod
    def get_func(obj, y, t):
        
        solver_class, beta = obj.solver_class, obj.beta
        P, U, RB, W_r = obj.P, obj.U, obj.RB, obj.W_r
        
        if obj.system_type == 'oscillator':
        
            if RB is None:
                x, u = np.split(y, 2)
                if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                    f = obj.JJ() @ obj.ham_z(x, u) - obj.drag(x, u)
                elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                    f = obj.JJ() @ obj.ham_z(x, u)
                elif solver_class in [ODESolver.ConformalStormerVerlet]:
                    f = obj.JJ() @ obj.ham_z(x, u, obj.Omega2, 0)
                    
            elif W_r is not None and P is None:
                y = RB @ y
                x, u = np.split(y, 2)
                if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                    f = obj.JJ_r @ RB.T @ obj.ham_z(x, u) - obj.W_r.T @obj.drag(x, u)
                elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                    f = obj.JJ_r @ RB.T @ obj.ham_z(x, u)
                elif solver_class in [ODESolver.ConformalStormerVerlet]:
                    f = obj.JJ_r @ RB.T @ obj.ham_z(x, u, obj.Omega2, 0)
                    
            elif P is not None and obj.non_quad is None:
                x_, u_ = np.split(RB @y, 2)
                
                x, u = np.split(obj.PxIPxRB(y), 2)
                
                if x.shape != obj.Omega2.shape:
                    Omega2 = P.T @ r_[obj.Omega2, obj.Omega2]
                    Omega2, _ = np.split(Omega2, 2)
                else:
                    Omega2 = obj.Omega2
                
                if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                    f = obj.JJ_r @ obj.hat(obj.ham_z(x, u, Omega2)) - obj.W_r.T @obj.drag(x_, u_)
                elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                    f = obj.JJ_r @ obj.hat(obj.ham_z(x, u, Omega2))
                    
            elif P is not None and obj.non_quad is not None:
                x_, u_ = np.split(RB @y, 2)
                x, u = np.split(obj.PxIPxRB(y), 2)
                
                if x.shape != obj.Omega2.shape:
                    Omega2 = P.T @ r_[obj.Omega2, obj.Omega2]
                    Omega2, _ = np.split(Omega2, 2)
                else:
                    Omega2 = obj.Omega2
                
                if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                    f = obj.JJ_r @ (y + obj.hat(obj.non_quad_z(x, u, Omega2)))\
                        - obj.W_r.T @obj.drag(x_, u_)
                elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                    f = obj.JJ_r @ (y + obj.hat(obj.non_quad_z(x, u, Omega2)))
                    
            else:
                raise NotImplementedError
                
            return f
        
        else:
        
            if RB is None:
                if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                    f = obj.JJ() @ obj.ham_z(y) - obj.drag(y)
                elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                    f = obj.JJ() @ obj.ham_z(y)
                    
            elif W_r is not None and P is None:
                if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                    f = obj.JJ_r @ RB.T @ obj.ham_z(y) - obj.W_r.T @obj.drag(y)
                elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                    f = obj.JJ_r @ RB.T @ obj.ham_z(y)
                    
            elif P is not None and obj.non_quad is None:
                
                if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                    f = obj.JJ_r @ obj.hat(obj.ham_z(obj.PxIPxRB(y))) - obj.W_r.T @obj.drag(RB @y)
                elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                    f = obj.JJ_r @ obj.hat(obj.ham_z(obj.PxIPxRB(y)))
                    
            elif P is not None and obj.non_quad is not None:
                
                if solver_class in [ODESolver.ImplicitMidpoint, ODESolver.ForwardEuler]:
                    f = obj.JJ_r @ (y + obj.hat(obj.non_quad_z(obj.PxIPxRB(y))))\
                        - obj.W_r.T @obj.drag(RB @y)
                elif solver_class in [ODESolver.ConformalImplicitMidpoint]:
                    f = obj.JJ_r @ (y + obj.hat(obj.non_quad_z(obj.PxIPxRB(y))))
            else:
                raise ValueError
                
            return f
    
    @staticmethod 
    def set_constraints(obj, kwds):
        
        # if 'RB' in kwds:
            
        #     if obj.constraint_type and kwds['system_type'] == 'oscillator':
    
        #         A_mat_ = obj.RB.T @obj.A_mat @obj.RB
        #         B_mat_ = obj.RB.T @obj.B_mat @obj.RB
                
        #         obj._g = lambda y, alpha=[A_mat_, B_mat_]: obj._g_(y, alpha)
                
        #         obj._g_prime = lambda y, alpha=[A_mat_, B_mat_]: obj._g_prime_(y, alpha)
        
        # else:
        if obj.constraint_type and kwds['system_type'] == 'oscillator':
            obj._g = obj._g_
            obj._g_prime = obj._g_prime_

    @staticmethod
    def set_system(obj, kwds):
        obj.Omega2 = kwds['Omega2']
    
        if kwds['system_type'] == 'oscillator':
            nosc = obj.nosc = kwds['nosc']
            Omega2 = obj.Omega2
            obj.Omega2_00 = lambda Omega2=Omega2: r_[Omega2, np.ones_like(Omega2)]
            
            alpha = list(np.linspace(0.1,0.5,num=nosc))
            alpha /= sqrt(sum(np.array(alpha)**2))
            assert np.isclose(np.array(alpha).dot(alpha), 1), "alpha**2 must be equal to 1"
            obj.alpha = alpha
            obj.beta = (max(1e-2, 0*np.random.rand()/10))*0
    
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
                
            # obj.ham = lambda x, u, Omega2=Omega2: f_ham(x, u, Omega2)
            # obj.ham_z = lambda x, u, Omega2=Omega2: f_ham_z(x, u, Omega2)
            # obj.ham_zz = lambda x, u, Omega2=Omega2: f_ham_zz(x, u, Omega2)
            
            obj.ham = lambda x, u, Omega2=Omega2: pot(x, Omega2) +kin(u) +obj.beta/2 * sum(x*u, axis=1)
                                                    
            obj.ham_z = lambda x, u, Omega2=Omega2, beta=obj.beta: obj.Omega2_00(Omega2) * r_[sin(Omega2*x), u] \
                                                    +beta/2 * r_[u, x]
            obj.ham_zz = lambda x, u, Omega2=Omega2, beta=obj.beta: diag(obj.Omega2_00(Omega2) * r_[cos(Omega2*x), np.ones_like(x)] *obj.Omega2_00(Omega2)) \
                                                + beta/2 * r_[c_[zeros(x.shape*2), eye(x.shape[0])],\
                                                                    c_[eye(x.shape[0]), zeros(x.shape*2)]]
                                                    
            # obj.ham_z = lambda x, u, Omega2=Omega2: obj.Omega2_00(Omega2) * r_[(Omega2*x)-(Omega2*x)**3/6, u] \
            #                                         +obj.beta/2 * r_[u, x]
            # obj.ham_zz = lambda x, u, Omega2=Omega2: diag(obj.Omega2_00(Omega2) * r_[1-(Omega2*x)**2/2, np.ones_like(x)] *obj.Omega2_00(Omega2)) \
            #                                     + obj.beta/2 * r_[c_[zeros(x.shape*2), eye(x.shape[0])],\
            #                                                         c_[eye(x.shape[0]), zeros(x.shape*2)]]
                
                                                    
            obj.Q_spd = lambda Omega2=Omega2: obj.ham_zz(0*Omega2,0*Omega2, Omega2)
            obj.JJ = lambda d=nosc: r_[c_[zeros((d,d)), eye(d)],\
                                          c_[-eye(d), zeros((d,d))]]
            obj.drag = lambda x, u: obj.beta/2 * r_[x, u]
            obj.drag_z = lambda x, u: obj.beta/2 * eye(2*x.shape[0])
                
            obj.non_quad = lambda x, u, Omega2=Omega2: obj.ham(x,u, Omega2) -1/2 *c_[x, u] @ obj.Q_spd() @ c_[x, u].T
            obj.non_quad_z = lambda x, u, Omega2=Omega2: obj.ham_z(x,u,Omega2) - obj.Q_spd(Omega2) @ r_[x, u]
            obj.non_quad_zz = lambda x, u, Omega2=Omega2: obj.ham_zz(x,u,Omega2) - obj.Q_spd(Omega2)
            obj.non_quad = None
        
            "Initial conditions"
            y_init = r_[np.linspace(1,5,nosc), np.zeros(nosc)]
            
            obj.constraint_type = None
            if obj.constraint_type == 'linear':
                y_init[nosc-1] = -(y_init[:nosc-1].dot(alpha[:nosc-1]))/alpha[nosc-1] # project on the manifold
                assert np.isclose(y_init[:nosc].dot(alpha[:nosc]), 0), "alpha:y should be 0" 
            elif obj.constraint_type == 'spherical':
                y_init /= sqrt((y_init[:nosc]**2).dot(alpha[:nosc])) # project on the manifold
                assert np.isclose((y_init[:nosc]**2).dot(alpha[:nosc]), 1), "alpha:y^2 should be 1" 
            elif obj.constraint_type == None:
                pass
            else:
                raise NameError
            obj.y_init = y_init
        
            "Constraints"
            if obj.constraint_type == 'linear':
                obj._g = lambda y, alpha=alpha: r_[y[:, :nosc].dot(alpha), y[:, nosc:].dot(alpha)]
                obj._g_prime = lambda y, alpha=alpha: r_[c_[np.array(alpha, ndmin=2), zeros((1,nosc))],\
                                            c_[zeros((1,nosc)), np.array(alpha, ndmin=2)]]
                # obj._g_prime = lambda y: block_diag((np.array(alpha, ndmin=2), np.array(alpha, ndmin=2)), format="csr")
                raise NotImplementedError
            elif obj.constraint_type == 'spherical':
                Alpha = np.diag(alpha)
                obj.A_mat = r_[c_[Alpha, np.zeros_like(Alpha)],\
                           c_[np.zeros_like(Alpha), np.zeros_like(Alpha)]]
                obj.B_mat = r_[c_[np.zeros_like(Alpha), Alpha],\
                           c_[Alpha, np.zeros_like(Alpha)]]
                    
                obj._g_ = lambda y, alpha=[obj.A_mat, obj.B_mat]: r_[y @ alpha[0] @ y.T -1.0,\
                                                    y @ alpha[1] @ y.T]
                obj._g_prime_ = lambda y, alpha=[obj.A_mat, obj.B_mat]: c_[2*y @alpha[0],\
                                                          2*y @alpha[1]].T

            "Fixed-point nonliner equations solver properties"
            obj.tol, obj.M, obj.var, obj.store = 1.0E-12, 100, True, False
                
        else:
    
            "MechSystem constituents"
            nu, alpha, rho, obj.beta = -1e-5, -3/8, -1e-1, 0
            L, dx = 4, 0.0808
            obj.nosc = int((2*L/dx +1)/2)
            obj.x_points = np.array([-L +i*dx for i in range(2*obj.nosc)])
            obj.y_init = 2*np.exp(-2*obj.x_points**2)/(sqrt(2)*np.pi)
            
            nosc = obj.nosc
            
            FD = sp.linalg.toeplitz([-1]+[0]*(2*nosc-2)+[1], [-1]+[1]+[0]*(2*nosc-2))/dx # Forward difference
            obj.ham = lambda u: sum(alpha/3*u**3 +rho/2*u**2 -nu/2*(u @FD.T)**2, axis=1)
            
            
            if 'P' in kwds:
                nosc = kwds['P'].shape[1]/2
                
            CD2 = sp.linalg.toeplitz([-2, 1]+[0]*(int(2*nosc)-3)+[1])/dx**2 # Second order central-difference
            nuxCD2 = nu*CD2
            alphax2 = alpha *2
            
            if 'RB' in kwds and 'P' not in kwds:
                rhoxRB = rho*kwds['RB']
                nuxCD2xRB = nuxCD2 @kwds['RB']
                alphaxRBx2 = 2 *alpha *kwds['RB']
                sqrt_abs_alphaxRB = sqrt(abs(alpha))*kwds['RB']
                obj.ham_z = lambda u: (-(sqrt_abs_alphaxRB @u)**2 +rhoxRB @u +nuxCD2xRB @u)
                obj.ham_zz = lambda u: (np.diag(alphaxRBx2 @u +rho) +nuxCD2)
            
                eye_2nosc = eye(2*nosc)
                betaxRB  = obj.beta*kwds['RB']
                obj.drag = lambda u: betaxRB @u
                obj.drag_z = lambda u: obj.beta*eye_2nosc
                
                nosc = obj.nosc
                obj.Q_spd = lambda u=obj.y_init: obj.ham_zz(np.zeros_like(u))
                CD1 = sp.linalg.toeplitz([0]+[-1]+[0]*(2*nosc-3)+[1], [0]+[1]+[0]*(2*nosc-3)+[-1])/(2*dx) # second order central-difference
                obj.JJ = lambda d=nosc: CD1
            else:
                
                
                obj.ham_z = lambda u: (alpha*u**2 +rho*u +nuxCD2 @u)
                obj.ham_zz = lambda u: (np.diag(alphax2*u +rho) +nuxCD2)
            
                obj.drag = lambda u: obj.beta*u
                obj.drag_z = lambda u: obj.beta*np.eye(len(u))
                
                nosc = obj.nosc
                obj.Q_spd = lambda u=obj.y_init: obj.ham_zz(np.zeros_like(u))
                CD1 = sp.linalg.toeplitz([0]+[-1]+[0]*(2*nosc-3)+[1], [0]+[1]+[0]*(2*nosc-3)+[-1])/(2*dx) # second order central-difference
                obj.JJ = lambda d=nosc: CD1
                
                obj.non_quad = lambda u: sum(alpha/3*u**3 +rho/2*u**2 -nu/2*(u @FD.T)**2 -1/2 *u @obj.Q_spd() @u.T, axis=1)
                obj.non_quad_z = lambda u: (obj.ham_z(u) -obj.Q_spd(u) @u)
                obj.non_quad_zz = lambda u: (obj.ham_zz(u) -obj.Q_spd(u))
                obj.non_quad = None
        
            "Initial conditions"
            
            obj.constraint_type = 'spherical'
        
            "Constraints"
            # if obj.constraint_type == 'spherical':                    
            #     obj._g_ = lambda u: (u[1]**2 -np.exp(-2*obj.beta*dt)*u[0]**2)*dx
            #     obj._g_prime_ = lambda u: 2*u[1]
            # else:
            #     raise NameError

            "Fixed-point nonliner equations solver properties"
            obj.tol, obj.M, obj.var, obj.store = 1.0E-10, 100, False, True
            

if __name__ == '__main__':
    pass