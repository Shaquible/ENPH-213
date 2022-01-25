# -*- coding: utf-8 -*-
"""
Example of how to read in csv data file, and compute the numerical derivative
In this example we read in hysteresis data, we plot this and the derivative,
and extract the peak values of the derivative as well as the x point 
Yuu may find this useful for some of your lab experiments in ENPH 253

Last Modified:  Jan  9, 2022
@author: shugh
"""

import numpy as np
import matplotlib.pyplot as plt

# read in the data, 4 columns, skup first line (see data file)
InputArray = np.loadtxt("ENPH213_Hysteresis-Data.csv", delimiter=',', skiprows=1)
vx = InputArray[:, 0]
vy = InputArray[:, 1]
vx2 = InputArray[:, 2]
vy2 = InputArray[:, 3]

"""
Example derivative function.
We use use slicing here, shown below for a forward diff estimation 
 note: [1:] all points apart from first point
       [0:-1] all points apart from last point
"""


def vxDeriv(x, y):
    return (y[1:]-y[0:-1]) / (x[1:]-x[0:-1])


# %% Create arrays for plotting and obtain the numerical derivatives
# for plotting take off end end point to match arrays of derivative
plotvx = vx[0:-1]
plotvx2 = vx2[0:-1]
lowerDeriv = vxDeriv(vx, vy)
upperDeriv = vxDeriv(vx2, vy2)

# plot the hysteresis graph (voltage in x / voltage in y)
fig = plt.figure(figsize=(9, 5))
ax = fig.add_axes([.10, .2, .35, .35])
ax.plot(vx, vy)  # Plots the data in Vy vs Vx (upper)
ax.plot(vx2, vy2)  # Plots the data in Vy vs Vx (lower)
ax.set_xlabel(r'$V_x$')
# \r confuses python, so add r outside (carriage return)
ax.set_ylabel(r'$V_y$')
ax.set_xlim(-0.3, 0.3)


# plot the derivative
fig = plt.figure(figsize=(9, 5))
ax = fig.add_axes([.10, .2, .35, .35])
#plt.text(0.1,1.24,' (a) Pulse in time')
#plt.plot(t, E_time, 'r--',linewidth=lw)
ax.set_xlabel(r'$V_y$')
ax.set_ylabel(r'$dV_x/dV_y$')
ax.set_xlim(-0.3, 0.3)
# plot the derivatives
ax.plot(plotvx, lowerDeriv)  # Plots the data in Vy vs Vx
ax.plot(plotvx2, upperDeriv)
plt.show()
# Obtain V_x and dV_x/dV_y at the peaks of the derivatives
print("Lower curve (Vx,dV_x/dV_y): \n", vx[np.argmax(lowerDeriv)], lowerDeriv[np.argmax(lowerDeriv)])
print('---------------')
print("Upper curve: (Vx,dV_x/dV_y): \n", vx2[np.argmax(upperDeriv)], upperDeriv[np.argmax(upperDeriv)])
