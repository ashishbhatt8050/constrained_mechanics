#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 11:43:44 2020

@author: ashishbhatt
"""

from sympy import Symbol, symbols, Matrix, pprint, simplify, trigsimp, diag, zeros, eye
from sympy.solvers.ode.systems import matrix_exp
t = Symbol('t')
alpha_1, alpha_2, alpha_3 = symbols('alpha_1 alpha_2 alpha_3')
omega_1, omega_2, omega_3 = symbols('omega_1 omega_2 omega_3')
beta = Symbol('beta')
alpha = [alpha_1, alpha_2, alpha_3]
omega = [omega_1, omega_2, omega_3]

import numpy as np

# alpha = np.array([0.2, 0.4, float('nan')])
# alpha[2] = np.sqrt(1 -alpha[0]**2 - alpha[1]**2)
# beta = 0.1
# omega = [1, 2, 3]
omega2 = [omega[i]**2 for i in range(len(omega))]
omegab2 = [alpha[i]*(omega2[i] -omega2[0]) for i in range(len(alpha))]

Omega2 = diag(omega2[1], omega2[2]) \
    +Matrix([[alpha[j]*omegab2[i] for i in  range(1,len(omega))] for j in range(1,len(alpha))])
# A = Matrix(np.concatenate((np.concatenate((np.zeros((2,2)), -Omega2)), \
#                           np.concatenate((np.eye(2), -beta*np.eye(2)))), axis=1))

A = zeros(2,2).col_insert(2,eye(2)).row_insert(2, -Omega2.col_insert(2, beta*eye(2)))

expA = matrix_exp(A, t)

pprint(A)

# pprint(matrix_exp(A, t))