# Keegan Kelly
# 3/26/22

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as lg
import time

# %% Q1


def heat_sol(u0, tmax, dt, dx, xmax, alpha, boundary):
    k = alpha*dt/dx**2
    # checking for stability
    if (k > 0.5):
        print("The stability condition is not met")
        return
    # building the time and x arrays
    t = np.arange(0, tmax+dt, dt)
    x = np.arange(0, xmax+dx, dx)
    # initializing the solution matrix
    u = np.zeros((len(t), len(x)))
    # setting the initial condition
    u[0, :] = u0(x)
    # setting the boundary conditions
    u[:, 0] = boundary
    u[:, -1] = boundary
    for i in range(1, len(t)):
        # solving for the next time step
        u[i, 1:-1] = u[i-1, 1:-1] + k*(u[i-1, 2:] - 2*u[i-1, 1:-1] + u[i-1, :-2])
    return u

# defining function for the initial condition


def u0(x): return 20+30*np.exp(-100*(x-0.5)**2)


# solving
us = heat_sol(u0, 61, 0.1, 0.01, 1, 2.3*10**-4, 20)
# selecting a few time steps to plot
uPlot = np.array([us[0, :], us[50, :], us[100, :], us[200, :], us[300, :], us[600, :]])
# plotting
for i in range(uPlot.shape[0]):
    plt.plot(uPlot[i, :], label="time = " + str(i*5) + "s")
plt.legend()
plt.xlabel("Rod length (m)")
plt.ylabel("Temperature (C)")
plt.show()

# %% Q2


def poissonJacobi(f, x0, xMax, Nx, y0, yMax, Ny, tol, boundary):
    # building the x and y arrays
    xs = np.linspace(x0, xMax, Nx)
    ys = np.linspace(y0, yMax, Ny)
    # finding step size for x and y
    Hx = xs[1]-xs[0]
    Hy = ys[1]-ys[0]
    # generating meshgrid to feed into f
    x, y = np.meshgrid(xs, ys)
    fpq = f(x, y)
    # initializing the solution matrix
    phiPQ = np.zeros((len(ys), len(xs)), dtype=float)
    # applying the boundary conditions
    phiPQ[0, :] = boundary
    phiPQ[-1, :] = boundary
    phiPQ[:, 0] = boundary
    phiPQ[:, -1] = boundary
    norm1 = lg.norm(phiPQ, ord=1)
    while True:
        # calculating the guess for phi for the inner points (non boundary points)
        phiPQ[1:-1, 1:-1] = (Hy**2*(phiPQ[2:, 1:-1]+phiPQ[:-2, 1:-1])+Hx**2*(phiPQ[1:-1, 2:]+phiPQ[1:-1, :-2])-(Hx*Hy)**2*fpq[1:-1, 1:-1])/(2*(Hx**2+Hy**2))
        # computing the norm
        norm2 = lg.norm(phiPQ, ord=1)
        # checking for convergence
        if (norm2-norm1)/norm2 > tol:
            norm1 = norm2
        else:
            return phiPQ

# deffining the function to solve with


def f(x, y):
    return np.cos(10*x)-np.sin(5*y-np.pi/4)


# calling the solution
phi = poissonJacobi(f, 0, 2, 100, 0, 1, 50, 10**-5, 0)
# plotting
plt.imshow(phi, cmap="hot", extent=[0, 2, 0, 1], origin='lower', aspect="equal", interpolation="none")
plt.xlabel("x")
plt.ylabel("y")
# plotting the colorbar
cbar = plt.colorbar()
cbar.set_label("$\phi(x,y)$")
plt.show()

# %% Q3


def poissonFourier(f, a, b, n):
    # building the x and y arrays
    x = np.arange(a, b, (b-a)/(n))
    # calculating the step size
    h = x[1]-x[0]
    # generating meshgrid to feed into f
    X, Y = np.meshgrid(x, x)
    # generating k and l arrays
    k = np.arange(n)
    kx, ky = np.meshgrid(k, k)
    # computing the FT of f
    fklFT = np.fft.fft2(f(X, Y))
    # solving for the denominator
    tmp = np.cos(2*np.pi*kx/n)+np.cos(2*np.pi*ky/n)-2
    # initializing the solution matrix
    phiFT = np.zeros((n, n), dtype=complex)
    # computing phi tilde for all but the first row to avoid division by zero
    phiFT[1:, :] = 0.5*h**2*fklFT[1:, :]/tmp[1:, :]
    # computing phi for the first row less the point 0,0
    phiFT[0, 1:] = 0.5*h**2*fklFT[0, 1:]/tmp[0, 1:]
    # inverting the FT
    return np.fft.ifft2(phiFT)

# deffining the function to solve with


def f2(x, y):
    return np.cos(3*x+4*y)-np.cos(5*x-2*y)


# calling the solution and timing its execution
start = time.time()
phi = poissonFourier(f2, 0, 2*np.pi, 800)
end = time.time()
print("Time taken: " + str(end-start))
# plotting a graph to match the assignment and slides
plt.imshow(phi.real, cmap="hot", extent=[0, 2, 0, 2], origin=None, aspect="equal", interpolation="none")
plt.xlabel("$x(\pi)$")
plt.ylabel("$y(\pi)$")
cbar = plt.colorbar()
cbar.set_label("$\phi(x,y)$")
plt.show()
# plotting a corrected graph (i believe in the slides and on the lab sheet origin='lower' was not included but should be since the index [0,0] is the bottom left corner not the top right). This graph orientation is consistent with the textbooks solution to the same problem.
plt.imshow(phi.real, cmap="hot", extent=[0, 2, 0, 2], origin='lower', aspect="equal", interpolation="none")
plt.xlabel("$x(\pi)$")
plt.ylabel("$y(\pi)$")
cbar = plt.colorbar()
cbar.set_label("$\phi(x,y)$")
plt.show()

# %%
