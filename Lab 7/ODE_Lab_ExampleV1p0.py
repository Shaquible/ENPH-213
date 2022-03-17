# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 20:03:52 2022

last Edited: March 8, 2022

ODE Stepper Example - I am showing how to use with Euler, but you can add other
ODE solvers in exactly the same way. I am also using the Matlab-like "feval", so
you can can a function and teh variables for that function 

@author: shugh
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})  # keep those graph fonts readable!
plt.rcParams['figure.dpi'] = 120  # plot resolution

# similar to Matlab's fval function - allows one to pass a function


def feval(funcName, *args):
    return eval(funcName)(*args)

# vectorized forward Euler with 1d numpy arrays


def euler(f, y0, t, h):  # Vectorized forward Euler (so no need to loop)
    k1 = h*f(y0, t)
    y1 = y0+k1
    return y1

# stepper function for integrating ODE solver over some time array


def odestepper(odesolver, deriv, y0, t):
    # simple np array
    y0 = np.asarray(y0)  # convret just in case a numpy was not sent
    y = np.zeros((t.size, y0.size))
    y[0, :] = y0
    h = t[1]-t[0]
    y_next = y0  # initial conditions

    for i in range(1, len(t)):
        y_next = feval(odesolver, deriv, y_next, t[i-1], h)
        y[i, :] = y_next
    return y


def fun(y, t):
    return -np.exp(-t)


def funexact(t):
    return np.exp(-t)


def plotme():
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_axes([.12, .15, .4, .4])
    ax.plot(ts, y1, 'gs', label='Euler $n={}$'.format(n), markersize=4)
    ax.plot(ts, y2, '-r', label='Exact', linewidth=3)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$y$')
    ax.set_xlim(0, b)
    ax.set_ylim(0., 1.04)
    ax.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    a, b, n, y0 = 0., 2, 50, 1.
    ts = a+np.arange(n)/(n-1)*b
    y1 = odestepper('euler', fun, y0, ts)
    y2 = funexact(ts)
    plotme()
