#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 11:38:09 2021

@author: bhattah


function DEIM to obtain Interpolating matrix P and a list of interpolation
indices P_list from DEIM basis Ub, a POD basis of nonlinear function
"""


import numpy as np
from scipy.linalg import qr

# %% POD


def rb_svd(y, eps=None):
    'Function for SVD decomposition of the input matrix y'
    u, s, vh = np.linalg.svd(y, full_matrices=False)
    assert np.allclose(y, np.dot(u * s, vh)), "SVD was unsuccessful"
    smat = np.diag(s)
    assert np.allclose(y, np.dot(u, np.dot(smat, vh))), "SVD was unsuccessful"

    # Truncate the POD modes based on the tolerance
    if eps is not None:
        # Compute the cumulative sum of the singular values
        energy = np.cumsum(s**2) / np.sum(s**2)

        # Find the smallest index (>=2) where energy exceeds 1 - eps
        N = np.searchsorted(energy, 1 - eps, side='left') + 1

        # Ensure N is even
        if N % 2 == 1: N += 1

        u, s, vh = u[:, :N], s[:N], vh[:N, :]

    return u, s, vh

def POD(S, Xh, eps=None):
    """
    Compute Proper Orthogonal Decomposition (POD) of a matrix S.

    Parameters:
    S (numpy array): Input matrix
    Xh (numpy array): Weighting matrix
    eps (float, optional): Tolerance for truncation. Defaults to None.

    Returns:
    Chi (numpy array): POD modes
    Sigma2 (numpy array): Singular values
    N (int): Number of retained modes
    """

    # Get the dimensions of the input matrix
    Nh, ns = S.shape

    if ns <= Nh:
        C_ = S.T @ Xh @ S
        Sigma2, Psi = np.linalg.eigh(C_)
        Sigma2, Psi = np.flip(Sigma2), np.flip(Psi, axis=1)
        Sigma2, Psi = Sigma2[Sigma2>0], Psi.T[Sigma2>0].T
        
        assert np.allclose(C_ @ Psi, Sigma2 * Psi, atol=1e-5)
        
        Chi = S @ Psi/np.sqrt(Sigma2)
    else:
        # Compute the square root of the weighting matrix
        if np.allclose(Xh, np.eye(Xh.shape[0])):
            Xh_half = np.eye(Xh.shape[0])
        else:
            vals, vecs = np.linalg.eigh(Xh)
            Xh_half = (vecs * np.sqrt(np.maximum(vals, 0.0))) @ vecs.T

        K_ = Xh_half @ S @ S.T @ Xh_half
        Sigma2, Chi_ = np.linalg.eigh(K_)
        Sigma2, Chi_ = np.flip(Sigma2), np.flip(Chi_, axis=1)
        Sigma2, Chi_ = Sigma2[Sigma2>0], Chi_.T[Sigma2>0].T
        
        assert np.allclose(K_ @ Chi_, Sigma2 * Chi_, atol=1e-5)
        
        Chi = np.linalg.solve(Xh_half, Chi_)
        
    # Truncate the POD modes based on the tolerance
    if eps is not None:
        # Compute the cumulative sum of the singular values
        energy = np.cumsum(Sigma2) / np.sum(Sigma2)

        # Find the number of modes N required to capture (1 - eps) of the total energy.
        # We add 1 because searchsorted finds the insertion point.
        N = np.searchsorted(energy, 1 - eps, side='left') + 1

        # Ensure N is even
        if N % 2 == 1: N += 1

        if N > Chi.shape[1]:
            N -= 2
    else:
        N = Sigma2.size
        
    Chi = Chi[:, :N]
    assert np.allclose(Chi.T @ Xh @ Chi, np.eye(N), atol=1e-5)
    # Chi = Xh_half @ Chi
        
    return Chi, np.sqrt(Sigma2), N

def PSD(F2, y_list, MechSystem, weights=None):
    
    nosc, tol = MechSystem.nosc, MechSystem.pod_tol
    if weights is None:
        weights = np.eye(nosc)
        
    # Pass the weights (Xh) to POD instead of hardcoded identity
    rb, sv, nosc_r = POD(np.hstack([y_list[:nosc, :], y_list[nosc:, :], F2[:nosc, :], F2[nosc:, :]]), weights, tol)
    RB = np.block([[rb[:, :nosc_r], np.zeros_like(rb[:, :nosc_r])], [np.zeros_like(rb[:, :nosc_r]), rb[:, :nosc_r]]])
    
    return RB, sv, nosc_r

# %%

def DEIM(Ub, plot_deim=False):
    'Function for discrete empirical interpolation of the basis Ub'

    # QR with column pivoting on Ub^T
    _, _, qdeim_indices = qr(Ub.T, pivoting=True)

    # Construct permutation matrix E
    # P is shape (n, m) where n=Ub.shape[0], m=Ub.shape[1]
    n, m = Ub.shape
    P = np.zeros((n, m), dtype=np.int8)
    P[qdeim_indices[:m], np.arange(m)] = 1

    return P, qdeim_indices


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


