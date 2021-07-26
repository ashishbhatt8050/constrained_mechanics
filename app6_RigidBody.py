#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 15:48:13 2020

@author: ashishbhatt

This app solves a rigid body with its center of mass fixed in space,
so that only rotational degress of freedom are present in the system.
"""

import numpy as np
from numpy import linalg as LA
#from pylab import *
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
        self.P0 = np.array([[0.,-1.,0.0875],
                            [1.,0.,0.],
                            [-0.0125,0.,0.]])
        self.K = 5.

        self.Rinv = LA.inv(self.R)

        self.Vext = lambda Q: self.K/2.0*LA.norm(Q.dot(self.r0) -self.q0)**2
        self.Ham = lambda Q, P: 1/2.0*np.trace(P.dot(self.Rinv).dot(P.T)) +self.Vext(Q)
        self.mom = lambda Q, P: P.dot(Q.T) -Q.dot(P.T)

    def __call__(self, Q):
        """ returns the potential Vext's gradient"""
        return np.outer(self.r0, self.K*(Q.dot(self.r0) - self.q0))

#%% Utility functions
def issymmetric(a, rtol=1e-12, atol=1e-12):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

#%% RATTLE implementation

def RATTLE(sys, dt, Tsize, tol, max_iter):
    R, Rinv, k, nab_V = sys.R, sys.Rinv, sys.k, sys
    myeye = np.eye(k)
    M_ = lambda Q: (Q.T).dot(Q) -myeye
    Mb_= lambda Q,P: (Q.T).dot(P).dot(Rinv) +Rinv.dot(P.T).dot(Q)
    lam_ = lambda M: np.array([[(R[i,i]*R[j,j]/(R[i,i] +R[j,j]))*M[i,j] for j in range(k)] for i in range(k)])

    Q = np.zeros((Tsize,) +np.shape(sys.Q0))
    P = np.zeros((Tsize,) +np.shape(sys.Q0))
    Q[0], P[0] = sys.Q0, sys.P0

    store = True
    if store:
        info = []

    for n in range(Tsize-1):
        Qnp = Q[n] +dt*P[n].dot(Rinv) -dt**2/2.0*nab_V(Q[n]).dot(Rinv)

        m, M = 0, M_(Qnp)
        if store: info.append((m,LA.norm(M)))

        while LA.norm(M) > tol and m < max_iter:
            lam = lam_(M)
            Qnp = Qnp -Q[n].dot(lam).dot(Rinv)
            m, M = m+1, M_(Qnp)
            if store: info.append((m,LA.norm(M),issymmetric(lam)))

        if m == max_iter:
            raise ValueError("Q did not converge, norm(M)=%d" % LA.norm(M))

        Pn1_2 = P[n] -dt/2.0*nab_V(Q[n]) -dt*Q[n].dot(lam)/dt**2
        Q[n+1] = Qnp

        Pnp = Pn1_2 -dt/2.0*nab_V(Q[n+1])

        m, M = 0, Mb_(Q[n+1],Pnp)
        if store: info.append((m,LA.norm(M)))

        while LA.norm(M) > tol and m < max_iter:
            lam = lam_(M)
            Pnp = Pnp -Q[n+1].dot(lam)
            m, M = m+1, Mb_(Q[n+1],Pnp)
            if store: info.append((m,LA.norm(M),issymmetric(lam)))

        if m == max_iter:
            raise ValueError("P did not converge, norm(M)=%d" % LA.norm(M))

        P[n+1] = Pnp

    return Q, P, info

#%% simulation
sys = RigidBody()
dt = 0.1
T = 100.0
Trange = np.arange(0,T,dt)
Tsize = np.size(Trange)
tol, max_iter = 1.0e-15, 50

Q, P, info = RATTLE(sys, dt, Tsize, tol, max_iter)

#%% plot Hamiltonian
Ham = [sys.Ham(Q[i],P[i]) for i in range(Tsize)]
fig = plt.figure()
ax = plt.axes()
ax.plot(Ham)

#%% plot 3d
fig = plt.figure()
ax = plt.axes(projection='3d')

# Data for a three-dimensional line
q = np.array([Q[i].dot(sys.r0) for i in range(Tsize)]).reshape(Tsize,sys.k)
x, y, z = q[:,0], q[:,1], q[:,2]
ax.plot3D(x,y,z, 'gray')

#%% plot momentum
if sys.Vext(Q[0]) < 1e-16:
    mom = np.array([sys.mom(Q[i],P[i]) for i in range(Tsize)])
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(mom.reshape((Tsize,3*sys.k)))