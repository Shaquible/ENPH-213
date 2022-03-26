# Keegan Kelly
# 3/26/22

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as lg

# %% Q1


def heat_sol(u0, tn, dt, dx, xn, alpha, boundary):
    k = alpha*dt/dx**2
    if (k > 0.5):
        print("The stability condition is not met")
        return
    t = np.arange(0, tn+dt, dt)
    x = np.arange(0, xn+dx, dx)
    u = np.zeros((len(t), len(x)))
    u[0, :] = u0(x)
    u[:, 0] = boundary
    u[:, -1] = boundary
    for i in range(1, len(t)):
        u[i, 1:-1] = u[i-1, 1:-1] + k*(u[i-1, 2:] - 2*u[i-1, 1:-1] + u[i-1, :-2])
    return u


def u0(x): return 20+30*np.exp(-100*(x-0.5)**2)


us = heat_sol(u0, 61, 0.1, 0.01, 1, 2.3*10**-4, 20)
uPlot = np.array([us[0, :], us[50, :], us[100, :], us[200, :], us[300, :], us[600, :]])
for i in range(uPlot.shape[0]):
    plt.plot(uPlot[i, :], label="time = " + str(i*5) + "s")
plt.legend()
plt.xlabel("Rod length (m)")
plt.ylabel("Temperature (C)")
plt.show()

# %% Q2


def poissonJacobi(f, x0, xMax, Nx, y0, yMax, Ny, tol, boundary):
    xs = np.linspace(x0, xMax, Nx)
    ys = np.linspace(y0, yMax, Ny)
    Hx = xs[1]-xs[0]
    Hy = ys[1]-ys[0]
    x, y = np.meshgrid(xs, ys, indexing='ij')
    fpq = f(x, y)
    phiPQ = np.zeros((len(xs), len(ys)), dtype=float)
    phiPQ[0, :] = boundary
    phiPQ[-1, :] = boundary
    phiPQ[:, 0] = boundary
    phiPQ[:, -1] = boundary
    norm1 = lg.norm(phiPQ, ord=1)
    while True:
        phiPQ[1:-1, 1:-1] = (Hy**2*(phiPQ[2:, 1:-1]+phiPQ[:-2, 1:-1])+Hx**2*(phiPQ[1:-1, 2:]+phiPQ[1:-1, :-2])-(Hx*Hy)**2*fpq[1:-1, 1:-1])/(2*(Hx**2+Hy**2))
        norm2 = lg.norm(phiPQ, ord=1)
        if (norm2-norm1)/norm2 > tol:
            norm1 = norm2
        else:
            phiPQ = np.transpose(phiPQ)
            return phiPQ


def f(x, y):
    return np.cos(10*x)-np.sin(5*y-np.pi/4)


phi = poissonJacobi(f, 0, 2, 100, 0, 1, 50, 10**-5, 0)
print(phi.shape)
plt.imshow(phi, cmap="hot", extent=[0, 2, 0, 1], origin="lower", aspect="equal", interpolation="none")
plt.xlabel("x")
plt.ylabel("y")
cbar = plt.colorbar()
cbar.set_label("$\phi(x,y)$")
# this graph might be reversed ask jose what he got (now im reversing the graph)
plt.show()

# %% Q3
