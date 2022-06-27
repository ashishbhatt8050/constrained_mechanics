#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 11:38:09 2021

@author: bhattah


function DEIM to obtain Interpolating matrix P and a list of interpolation
indices P_list from DEIM basis Ub, a POD basis of nonlinear function

Tests for the 1D and 2D cases
"""


from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from itertools import product
from PlotScript import plot_3dsurface
from time import process_time
import numpy as np
import scipy as sp
import pylab as pl

# %% POD


def rb_svd(y):
    'Function for SVD decomposition of the input matrix y'
    u, s, vh = np.linalg.svd(y, full_matrices=False)
    assert np.allclose(y, np.dot(u * s, vh)), "SVD was unsuccessful"
    smat = np.diag(s)
    assert np.allclose(y, np.dot(u, np.dot(smat, vh))), "SVD was unsuccessful"

    return u, s, vh

def POD(S, Xh, eps=None):
    
    if np.allclose(Xh, np.eye(Xh.shape[0])):
        Chi, Sigma, _ = rb_svd(S)
        Sigma2 = Sigma**2
    else:
        Nh, ns = S.shape
        if ns <= Nh:
            C_ = S.T @ Xh @ S
            Sigma2, Psi = np.linalg.eigh(C_)
            Sigma2, Psi = np.flip(Sigma2), np.flip(Psi, axis=1)
            Chi = S @ Psi/np.sqrt(Sigma2)
        else:
            Xh_half = sp.linalg.sqrtm(Xh)
            K_ = Xh_half @ S @ S.T @ Xh_half
            Sigma2, Chi_ = np.linalg.eigh(K_)
            Sigma2, Chi_ = np.flip(Sigma2), np.flip(Chi_, axis=1)
            Chi = np.linalg.solve(Xh_half, Chi_)
        
        assert np.allclose(Chi.T @ Xh @ Chi, np.eye(Nh))
        
    if eps is not None:
        N = np.argmin([sum(Sigma2[:i])/sum(Sigma2) -1 +eps for i in range(Sigma2.size)])
    else:
        N = Sigma2.size
        
    return Chi[:, :N], np.sqrt(Sigma2)
# %%


def DEIM(Ub, plot_deim=False):
    'Function for discrete empirical interpolation of the basis Ub'
    p = abs(Ub[:, 0]).argmax()
    idx_list = [p]
    P = np.zeros((Ub.shape[0], 1), dtype=np.int8)
    P[p, 0] = 1
    U = Ub[:, 0:1]

    if plot_deim:
        fig = pl.figure()
        ax = fig.add_subplot(321)
        ax.plot(abs(U))
        ax.plot(p, max(abs(U)), '*')

    for i in range(1, Ub.shape[1]):
        u_i = Ub[:, i, None]
        c = np.linalg.solve(P.T @ U, P.T @ u_i)
        r = u_i - U @ c
        p = abs(r).argmax()
        U = pl.c_[U, u_i]
        P = pl.c_[P, np.zeros((Ub.shape[0], 1), dtype=np.int8)]
        P[p, i] = 1
        idx_list.append(p)

        if plot_deim and i < 6:
            ax = fig.add_subplot(3, 2, i+1)
            ax.plot(abs(r))
            ax.plot(p, max(abs(r)), '*')

    return P, idx_list

# %%


def orthogonalize(W, V, X=None):
    """Orthogonalize W with respect to V such that V.T @ W = I

    Arguments
      W: n x r array
      V: n x r array
      X: inner product matrix

    Returns
      Q: n x r array such that Q.T @ V = I
    """
    eps = 1e-12
    r = W.shape[1]
    Q = np.zeros(W.shape)
    
    if X is None:
        X = np.eye(W.shape[0])

    for k in range(r):
        w = W[:, k]  # a new candidate vector

        for j in range(r):  # Subtract the projections on V vectors
            if j != k:          # except the k-th vector
                h = np.dot(V[:, j].T, X @ w)/np.dot(V[:, j].T, X @ V[:, j])
                w = w - h * V[:, j]

        h = w.T @ X @ V[:, k]
        if abs(h) > eps:  # Add the produced vector to the list, unless
            Q[:, k] = w/h  # the k-th vector
        else:  # If that happens, stop iterating.
            return Q
    return Q

# %% Testing


def test_1D():
    'Test DEIM on a 1D function'
    N = 100  # full order
    x = np.linspace(-1, 1, N)
    def f(mu, x=x): return (1-x) * np.cos(3*np.pi*mu*(1+x)) * np.exp(-(1+x)*mu)

    F = np.array([f(mu) for mu in np.linspace(1, np.pi, 51)])

    pod_deim(N, x, f, F)


def pod_deim(N, x, f, F):
    'Helper function of test_1D'
    _, s, U = rb_svd(F)
    fig = pl.figure()
    ax1 = fig.add_subplot(111)
    # ax1.set_prop_cycle(cycler('linestyle', ['-.','-*','-.o',':']))
    ax1.semilogy(s)
    U = U.T

    P, idx_list = DEIM(U, plot_deim=True)

    fig = pl.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, U[:, :6], x[idx_list[:6]], 0*x[idx_list[:6]], '*')

    def f_h(P, U, mu): return U @ np.linalg.solve(P.T @ U, f(mu, x=P.T @ x))

    fig = pl.figure()
    ax = fig.add_subplot(111)
    mu = 3.1
    ax.plot(x, pl.c_[f(mu), f_h(P, U, mu)])

    error_deim_pod, error_pod = [], []
    for m in range(5, U.shape[1]):
        Pm, Um = P[:, :m], U[:, :m]  # reduced order m

        error_deim_pod.append(np.linalg.norm(f(mu) - f_h(Pm, Um, mu)))

        error_pod.append(np.linalg.norm((np.eye(N) - Um @ Um.T) @ f(mu)))

    ax1.semilogy(range(5, U.shape[1]), pl.c_[error_deim_pod, error_pod])

# %%


def test_2D():
    'Test DEIM on a 2D function'
    N = 40  # full order
    x = y = np.linspace(0.1, 0.9, N)
    x_vec = np.array([list(x_) for x_ in product(x, y)])
    mu1 = mu2 = np.linspace(-1, -0.01, 25)
    mu = [list(x_) for x_ in product(mu1, mu2)]

    def f(mu, xx, yy):
        return 1/np.sqrt((xx - mu[0])**2 + (yy - mu[1])**2 + 0.1**2)

    def f_vec(mu, x=x_vec): return 1 / \
        np.sqrt(np.sum((x - mu)**2, axis=1) + 0.1**2)

    F = np.array([f_vec(mu_) for mu_ in mu]).T

    U, s, _ = rb_svd(F)
    fig = pl.figure()
    ax1 = fig.add_subplot(111)
    ax1.semilogy(s)

    P, idx_list = DEIM(U, plot_deim=False)

    # Contour plot
    fig = plt.figure()
    ax = fig.add_subplot()
    xx, yy = pl.meshgrid(x, y)
    z = f(mu[-1], xx, yy)
    p = ax.pcolor(xx, yy, z, shading='auto',
                  vmin=abs(z).min(), vmax=abs(z).max())
    fig.colorbar(p, ax=ax)

    # function plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    plot_3dsurface(fig, ax, xx, yy, z)

    # POD bases plots
    fig = pl.figure()
    for i in range(1, 7):
        ax = fig.add_subplot(3, 2, i, projection='3d')
        plot_3dsurface(fig, ax, xx, yy, np.reshape(U[:, i-1], (N, N)))

    # DEIM points plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x_vec[idx_list[0:20]][:, 0], x_vec[idx_list[0:20]][:, 1])

    # %% Plot full and reduced functions

    fig = pl.figure()
    m = 6
    Pm = P[:, :m]
    Um = U[:, :m]

    f_list = [f_vec,
              lambda mu: Um @ Um.T @ f_vec(mu),
              lambda mu: Um @ np.linalg.solve(Pm.T @ Um, f_vec(mu, Pm.T @ x_vec))]

    for i in range(0, 3):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')

        plot_3dsurface(fig, ax, xx, yy, f_list[i](
            [-0.05, -0.05]).reshape((N, N)))

    # %% Errors and timing
    fv_list = [f_vec,
               lambda UUT, f_mu: UUT @ f_mu,
               lambda U, Um_P, f_mu_P: U @ np.linalg.solve(Um_P, f_mu_P)]

    deim_error = []
    pod_error = []
    deim_time = []
    pod_time = []
    for m in range(1, 21):
        Pm = P[:, :m]
        Um = U[:, :m]

        start = process_time()
        UmUmT = Um @ Um.T
        pod_time_ = process_time() - start

        start = process_time()
        Um_P = Pm.T @ Um
        deim_time_ = process_time() - start

        deim_error_mu = []
        pod_error_mu = []
        deim_time_mu = []
        pod_time_mu = []

        for mu_ in mu:
            f_temp = []
            f_temp.append(fv_list[0](mu_))
            f_temp.append(fv_list[0](mu_, Pm.T @ x_vec))

            start = process_time()
            f_temp.append(fv_list[1](UmUmT, f_temp[0]))
            pod_time_mu = process_time() - start
            pod_error_mu.append(np.linalg.norm(f_temp[0] - f_temp[-1], 2))

            start = process_time()
            f_temp.append(fv_list[2](Um, Um_P, f_temp[1]))
            deim_time_mu = process_time() - start
            deim_error_mu.append(np.linalg.norm(f_temp[0] - f_temp[-1], 2))

        deim_error.append(np.average(deim_error_mu))
        pod_error.append(np.average(pod_error_mu))
        deim_time.append(np.average(deim_time_mu) + deim_time_)
        pod_time.append(np.average(pod_time_mu) + pod_time_)

    fig = pl.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    # ax1.set_prop_cycle(cycler('linestyle', ['-*', '-.o', ':', '-.']))
    ax1.semilogy(pl.c_[deim_error, pod_error])
    # ax1.semilogy(pod_error,'-.o')
    ax2 = fig.add_subplot(1, 3, 2)
    # ax1.set_prop_cycle(cycler('linestyle', ['-*', '-.o', ':', '-.']))
    ax2.semilogy(pl.c_[deim_time, pod_time])


# %%
if __name__ == '__main__':
    test_1D()
    test_2D()
