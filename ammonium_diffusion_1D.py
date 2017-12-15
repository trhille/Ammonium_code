#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 17:40:57 2017

@author: trevorhillebrand
"""
import numpy as np
import matplotlib.pyplot as plt

#set up yo grid
N = 200  #number of nodes 
x0 = 0 #top surface (m)
xN = 100 #bottom surface (m)

x = np.linspace(x0, xN, N) #here's your x-nodes
dx = (xN-x0)/N#spatial step. Don't change this!
end_time = 1e3 #duration of simulation in years
dt = 0.1 #time step. If 1/2≤theta≤1, timestep can be basically anything (positive), but gets less accurate with larger values, especially near boundaries
s_per_yr = 3600*24*365.25 #seconds per year
#model parameters
D = np.linspace(9.8e-6, 5e-5, N)/100**2*s_per_yr #diffusivity from Krom and Berner (1980) in Limnology and Oceanography. Units are m^2 s^-1 (constant here, but doesn't have to be)
u_seawater = 1e-4 # 1 mM concentration, also from Krom and Berner (1980)
theta = 1/2 #Can be 0 (explicit) to 1(implicit), 0.5 is most accurate (Crank-Nicolson time-stepping). If theta<1/2, use dt < dx^2/(4D)
#initial conditions (Here, I just made up layer concentrations).
layers = 6 #number of stratigraphic layers (duh)
layer_thickness = np.floor(N/layers) #layer thickness. If there is a remainder, it's added onto the lowest layer

u_layer_1 = 0 #layer 1 is the top
u_layer_2 = 0
u_layer_3 = 0
u_layer_4 = 0
u_layer_5 = 0
u_layer_6 = 0 #layer 6 is the bottom
u_layer_list = (u_layer_1,u_layer_2, u_layer_3, u_layer_4, u_layer_5, u_layer_6)

u_t0 = np.zeros(N) #create initial condition vector 

#then populate that vector
for jj in range(0, layers):
    u_t0[int(jj*layer_thickness): int((jj+1)*layer_thickness)] = u_layer_list[jj]
    
u_t0[int(layers*layer_thickness): -1] = u_layer_list[layers-1] #fill in the remainder, so the lowest layer is the largest 
u_t0[-1] = u_layer_list[layers-1] #


f = np.zeros((N, int(end_time/dt))) #Initialize source term. Leave it at this, if you want it to be zero.  Some possible modifications are:
#f[:,int(end_time/dt/2)] = np.linspace(0, 1e-2,N)  # for example, you can just add a slug of ammonium everywhere halfway through the simulation

#or something fun that varies in space and time, like this:

#for tt in range(0, int(end_time/dt)):
#   f[:,tt] = np.exp(-x/10)*5e-6*(np.sin((x+tt*dt)/4/np.pi))

#or the source can be a function of concentration of seawater and time
f[0,:] = u_seawater*(1-np.sin(dt/4*np.linspace(0, end_time, end_time/dt)))

#set boundary conditions
u_x0 = u_seawater #upper boundary is constant at ammonium concentration of seawater
u_xN = u_t0[-1] #Constant initial concentration at the bottom (can easily change this later)
   
#create finite difference matrix
A_fd_diag = 2*np.ones(N)
A_fd_diag_1 = -np.ones(N-1)
A_fd_diag_minus1 = -np.ones(N-1)
A_fd = 1./dx**2*(np.diag(A_fd_diag, 0) + np.diag(A_fd_diag_1,1) + np.diag(A_fd_diag_minus1, -1))
A_fd[0, 0:2] = np.array([1, 0])
A_fd[-1, -2] = 0
A_fd[-1, -1] = 1


u_xt = np.zeros((N,int(end_time/dt))) #create solution array. The for-loop below will populate
u_xt[:,0] = u_t0 #initial condition in first column

#poopulate the solution array. Each column is a time step.
for tt in range(1, int(end_time/dt)):
    u_xt[:,tt] = np.linalg.solve((np.identity(N) + theta*dt*D*A_fd), np.matmul((np.identity(N)-dt*(1-theta)*D*A_fd), u_xt[:,tt-1]) + dt*(theta*f[:, tt] + (1-theta)*f[:,tt-1])) #solve Ax=b
    u_xt[0,tt] = u_x0 #ensure boundaries don't change 
    u_xt[-1,tt] = u_xN

#plot up initial profile
plt.plot(u_xt[:,0], -x)
plt.ylabel('depth (m)')
plt.xlabel('[NH4]')

#check out that final profile
plt.plot(u_xt[:,-1], -x)
plt.ylabel('depth (m)')
plt.xlabel('[NH4]')