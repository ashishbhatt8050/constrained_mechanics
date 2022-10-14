#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 15:48:13 2020

@author: ashishbhatt

This app solves a rigid body with its center of mass fixed in space,
so that only rotational degress of freedom are present in the system.
"""

import numpy as np
from numpy import linalg as LA
from pylab import r_, c_
from mpl_toolkits import mplot3d
#%matplotlib inline
import matplotlib.pyplot as plt

#%% define the system

class RigidBody(object):
    def __init__(self):
        self.k = 3
        self.r0 = np.array([[0.],[0.],[1.]])
        self.q0 = np.array([[0.],[0.],[-1.]])
        self.R = np.array([[0.5,0.,0.],
                           [0.,0.5,0.],
                           [0.,0.,3.5]])
        self.Q0 = np.eye(self.k)
        assert np.allclose(self.Q0.T @ self.Q0, np.eye(self.k)) and LA.det(self.Q0) ==1
        self.P0 = np.array([[0.,-1.,0.0875],
                            [1.,0.,0.],
                            [-0.0125,0.,0.]])
        self.K = 5
        self.gamma = 0.0
        # myfun = lambda x: 3*np.sin(x) -np.sin(x)**3
        # myfun = lambda x: np.heaviside(x-2500*dt, 1)
        # myfun = lambda x: 1/2 +1/2*np.tanh(20*(x -2500*dt))
        self.gamma_t = lambda t: -self.gamma*(t)
        # self.gamma_t = lambda t: -self.gamma*0
        # int_gamma_t = lambda t: self.gamma*np.cos(t)
        self.int_gamma_t = lambda t1, t0: -self.gamma*((t1) -(t0))
        self.exp_gamma = lambda dt, tn: np.exp(self.int_gamma_t(tn+dt, tn))

        self.Rinv = self.R - np.diag(np.diagonal(self.R)) +np.diag(1/np.diagonal(self.R))
        assert np.count_nonzero(self.R - np.diag(np.diagonal(self.R))) ==0
        
        self.Jinv = r_[c_[np.zeros((self.k,)*2), np.eye(self.k)],\
                       c_[-np.eye(self.k), np.zeros((self.k,)*2)]]

        self.Vext = lambda Q: self.K/2.0*LA.norm(Q.dot(self.r0) -self.q0)**2
        self.Ham = lambda Q, P, Lam: 1/2.0*np.trace(P.dot(self.Rinv).dot(P.T)) +self.Vext(Q) +Lam
        self.mom = lambda Q, P: P.dot(Q.T) -Q.dot(P.T)

    def __call__(self, Q):
        """ returns the potential Vext's gradient"""
        return self.K*(Q.dot(self.r0) - self.q0) @ self.r0.T

#%% Utility functions
def issymmetric(a, rtol=1e-12, atol=1e-12):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

#%% RATTLE implementation

def RATTLE(sys, dt, Tsize, tol, max_iter):
    R, Rinv, k, nab_V, exp_gamma = sys.R, sys.Rinv, sys.k, sys, sys.exp_gamma
    myeye = np.eye(k)
    M_ = lambda Q, n: (Q.T).dot(Q) -exp_gamma(0*dt,n*dt)*myeye
    Mb_= lambda Q,P,n: (Q.T).dot(P).dot(Rinv) +Rinv.dot(P.T).dot(Q) -sys.gamma_t((n+1)*dt*0)*exp_gamma(0*dt,n*dt)*myeye
    lam_ = lambda M: np.array([[(R[i,i]*R[j,j]/(R[i,i] +R[j,j]))*M[i,j] for j in range(k)] for i in range(k)])

    Q = np.zeros((Tsize,) +np.shape(sys.Q0))
    P = np.zeros((Tsize,) +np.shape(sys.Q0))
    Lam = np.zeros((Tsize,))
    Q[0], P[0] = sys.Q0, sys.P0

    store = False
    if store:
        info = []

    for n in range(Tsize-1):
        Qnp = Q[n] +exp_gamma(dt,n*dt)*dt*P[n].dot(Rinv) -dt**2/2.0*nab_V(Q[n]).dot(Rinv)

        m, M = 0, M_(Qnp,n)
        if store: info.append((m,LA.norm(M)))

        while LA.norm(M) > tol and m < max_iter:
            lam = lam_(M)
            assert np.allclose(lam, lam.T)
            Qnp = Qnp -Q[n].dot(lam).dot(Rinv)
            m, M = m+1, M_(Qnp,n)
            if store: info.append((m,LA.norm(M),issymmetric(lam)))
            
        Lam[n+1] = np.trace(M @ lam/dt**2)

        if m == max_iter:
            raise ValueError("Q did not converge, n=%d, norm(M)=%d" % (n,LA.norm(M)))

        # Pn1_2 = P[n] -dt/2.0*nab_V(Q[n]) -dt*Q[n].dot(lam)/dt**2
        Pn1_2 = (Qnp - Q[n])/dt @ R
        Q[n+1] = Qnp

        Pnp = exp_gamma(dt,n*dt)*(Pn1_2 -dt/2.0*nab_V(Q[n+1]))

        m, M = 0, Mb_(Q[n+1],Pnp,n)
        if store: info.append((m,LA.norm(M)))

        while LA.norm(M) > tol and m < max_iter:
            lam = lam_(M)
            assert np.allclose(lam, lam.T)
            Pnp = Pnp -Q[n+1].dot(lam)
            m, M = m+1, Mb_(Q[n+1],Pnp,n)
            if store: info.append((m,LA.norm(M),issymmetric(lam)))

        if m == max_iter:
            raise ValueError("P did not converge, n=%d, norm(M)=%d" % (n,LA.norm(M)))

        P[n+1] = Pnp

    return Q, P, Lam, False

def RATTLE_symplectic(sys, dt, Tsize):
    K, r0, Rinv, Jinv, exp_gamma = sys.K, sys.r0, sys.Rinv, sys.Jinv, sys.exp_gamma
    myeye = np.eye(sys.k)
    # Psi_p = LA.solve(r_[c_[myeye, dt/2*K*r0 @ r0.T],\
    #              c_[np.zeros((sys.k,)*2), 1/exp_gamma*myeye]].T,\
    #                   r_[c_[myeye -dt**2/2*K*r0 @ r0.T @ Rinv, -dt/2*K*r0 @ r0.T],\
    #            c_[exp_gamma*dt*Rinv, myeye]].T).T
    
    error = np.zeros(Tsize)
    Psi_p_ = np.eye(sys.k*2)
    for n in range(Tsize-1):
        Psi_p = r_[c_[myeye -dt**2/2*K*r0 @ r0.T @ Rinv, exp_gamma(dt,n*dt)*(-dt/2*K*r0 @ r0.T -dt/2*K*(myeye -dt**2/2*K*r0 @ r0.T @ Rinv) @ r0 @ r0.T)],\
                   c_[exp_gamma(dt,n*dt)*dt*Rinv, exp_gamma(dt,n*dt)*(myeye -dt/2*K*(exp_gamma(dt,n*dt)*dt*Rinv) @ r0 @ r0.T)]]
        error[n+1] = LA.norm((Psi_p @ Psi_p_).T @ Jinv @ (Psi_p @ Psi_p_) -exp_gamma((n+1)*dt,n*dt*0)**2*Jinv)
        Psi_p_ = Psi_p
    return error
    

#%% simulation
sys = RigidBody()
dt = 0.1
T = 500.0
Trange = np.arange(0,T,dt)
Tsize = np.size(Trange)
tol, max_iter = 1.0e-14, 50

Q, P, Lam, info = RATTLE(sys, dt, Tsize, tol, max_iter)

#%% symplectic error
error = RATTLE_symplectic(sys, dt, Tsize)
fig = plt.figure()
ax = plt.axes()
ax.plot(error[2:])
# ax.set_ylim([2.4470356400000001, 2.4470356600000001])

# Conformal symplectiness is not preserved.
# How to make this problem high-dimensional (for MOR) is not clear.

#%% plot Hamiltonian
Ham = [sys.Ham(Q[i],P[i], Lam[i]) for i in range(Tsize)]
fig = plt.figure()
ax = plt.axes()
ax.plot(Ham)

#%% plot 3d
fig = plt.figure()
ax = plt.axes(projection='3d')

# Data for a three-dimensional line
q = np.array([Q[i].dot(sys.r0) for i in range(Tsize)]).reshape(Tsize,sys.k)
x, y, z = q[:,0], q[:,1], q[:,2]
ax.plot3D(x[:1000],y[:1000],z[:1000], 'gray')
ax.plot3D(x[4000:],y[4000:],z[4000:], 'red')

#%% plot momentum, momentum is only conserved for V_ext = 0
if sys.Vext(Q[0]) < 1e-16:
    mom_error = [LA.norm(sys.mom(Q[i],P[i]) -sys.mom(Q[0],P[0])) for i in range(Tsize)]
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(mom_error)
