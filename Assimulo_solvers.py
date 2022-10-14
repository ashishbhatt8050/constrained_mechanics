#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
Tutorial example showing how to use the implicit solver IDA. To run the example
simply type,

    run tutorialIDA.py (in IPython)
    
or,

    python tutorialIDA.py (in a command prompt)
"""
import numpy as np
from assimulo.problem import Implicit_Problem #Imports the problem formulation from Assimulo
from assimulo.solvers import IDA, Radau5DAE, GLIMDA, ODASSL       #Imports the solver IDA from Assimulo
from pylab import *
from numpy import linalg as LA

def run_example_1():

    def residual(t,y,yd):

        res_0, res_1, res_2 = yd[:3] -y[3:]
        res_3, res_4, res_5 = yd[3:] +np.sin(y[:3]) +beta*y[3:]
        res_6 = alpha.dot(y[:3])
        # res_0 = yd[0]-y[2]
        # res_1 = yd[1]-y[3]
        # res_2 = yd[2]+y[4]*y[0]
        # res_3 = yd[3]+y[4]*y[1]+9.82
        # res_4 = y[2]**2+y[3]**2-y[4]*(y[0]**2+y[1]**2)-y[1]*9.82

        return np.array([res_0,res_1,res_2,res_3,res_4,res_5,res_6])
    
    #The initial conditions
    alpha = np.array([0.2, 0.4, float('nan')])
    alpha[2] = np.sqrt(1 -alpha[0]**2 - alpha[1]**2)
    beta = 0.1
    t0  = 0.0 #Initial time
    y0 = [0.2, 0.4, float('nan'), 0.0, 0.0, 0.0]
    y0[2] = -(np.array(y0[:2]).dot(alpha[:2]))/alpha[2] # project on the manifold
    yd0 = [0.0, 0.0, 0.0, 0.1, 0.2, 0.3] #Initial conditions
    

    model = Implicit_Problem(residual, y0, yd0, t0)             #Create an Assimulo problem
    model.name = 'Pendulum'        #Specifies the name of problem (optional)

    sim = IDA(model) #Create the IDA solver
        
    tfinal = 20.0        #Specify the final time
    ncp = 500            #Number of communcation points (number of return points)
    sim.atol = 1e-10
    sim.rtol = 1e-10

    t,y,yd = sim.simulate(tfinal, ncp) #Use the .simulate method to simulate and provide the final time and ncp (optional)
    
    sim.plot()
    
    plot(alpha.dot(y[:,:3].T))

def run_example_2():

    def residual(t,y,yd):

        res_0 = np.reshape(yd[:9],(3,3)) -np.reshape(y[9:],(3,3))*Rinv
        res_1 = np.reshape(yd[9:],(3,3)) +np.outer(r0, K*(np.reshape(y[:9],(3,3)).dot(r0) - q0))
        res_2 = (y[:9].T).dot(y[:9]) -np.eye(k)

        return np.array(np.reshape(np.concatenate((res_0,res_1,res_2)),(27,)))
    
    #The initial conditions
    k = 3
    K = 5
    r0 = np.array([[0],[0],[1]])
    q0 = np.array([[0],[0],[-1]])
    t0 = 0.0
    R = np.array([[0.5,0,0],
                       [0,0.5,0],
                       [0,0,3.5]])
    Rinv = LA.inv(R)
    

    Q0 = np.eye(k)
    P0 = np.array([[0,-1,0.0875],
                        [1,0,0],
                        [-0.0125,0,0]])
    Q0d = P0*Rinv
    P0d = -np.outer(r0, K*(Q0.dot(r0) - q0))
    
    
    model = Implicit_Problem(residual, [Q0, P0], [Q0d, P0d], t0)             #Create an Assimulo problem
    model.name = 'Pendulum'        #Specifies the name of problem (optional)

    sim = IDA(model) #Create the IDA solver
        
    tfinal = 20.0        #Specify the final time
    ncp = 500            #Number of communcation points (number of return points)
    sim.atol = 1e-10
    sim.rtol = 1e-10

    t,y,yd = sim.simulate(tfinal, ncp) #Use the .simulate method to simulate and provide the final time and ncp (optional)
    
    sim.plot()
    
    #plot(alpha.dot(y[:,:3].T))
    
    #%% plot 3d
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    # Data for a three-dimensional line
    Q = np.reshape(y[:,:9],(501,3,3))
    q = np.array([Q[i].dot(r0) for i in range(501)]).reshape(501,k)
    x, y, z = q[:,0], q[:,1], q[:,2]
    ax.plot3D(x,y,z, 'gray')


if __name__=='__main__':
    run_example_2()
