# Keegan Kelly 3/20/22
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
# %% Q1(a)


def feval(funcName, *args):
    return eval(funcName)(*args)


def euler(f, y0, t, h):  # Vectorized forward Euler (so no need to loop)
    k1 = h*f(y0, t)
    y1 = y0+k1
    return y1


def backwardEuler(f, y0, t, h):
    def back(y1):
        # defining the backward euler
        return y0-y1+h*f(y1, t)
    # running a solver to find the proper value of y1 through an iterative process
    return sp.fsolve(back, y0)


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


def fun(y, t):  # derivative of y
    return -10*y


def y(t):  # the actual value of the solution
    return np.exp(-10*t)


# conputing the excact solution
texact = np.linspace(0, 0.6, 100)
yexact = y(texact)
# for n=10
# defining the time array for the numerical solution
t = np.arange(10)/(9)*0.6
# using initial condition y0 to solve the ode with forward and backward euler
y0 = 1
yfor = odestepper('euler', fun, y0, t)
yback = odestepper('backwardEuler', fun, y0, t)
# plotting
figure, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].plot(texact, yexact, 'g', label='Exact', linewidth=3)
ax[0].plot(t, yfor, 'bo', label='Euler $n={}$'.format(10), markersize=4)
ax[0].plot(t, yback, 'ro', label='Backward Euler $n={}$'.format(10), markersize=4)
ax[0].set_xlabel('$t$')
ax[0].set_ylabel('$y$')
ax[0].legend(loc='best')
# for n=20
# time array for the numerical solution with n=20
t = np.arange(20)/(19)*0.6
# using initial condition y0 to solve the ode with forward and backward euler
y0 = 1
yfor = odestepper('euler', fun, y0, t)
yback = odestepper('backwardEuler', fun, y0, t)
# plotting
ax[1].plot(texact, yexact, 'g', label='Exact', linewidth=3)
ax[1].plot(t, yfor, 'bo', label='Euler $n={}$'.format(20), markersize=4)
ax[1].plot(t, yback, 'ro', label='Backward Euler $n={}$'.format(20), markersize=4)
ax[1].set_xlabel('$t$')
ax[1].set_ylabel('$y$')
ax[1].legend(loc='best')
plt.show()

# %% Q1(b)


def deriv(y, t):  # derivative of the y1 and y2
    return np.array([y[1], -y[0]])


def exact(t):  # excact solution
    return np.cos(t)


def RK4_step(f, y, t, h):
    # calculating the RK4 according to eqn 13
    k0 = h*f(y, t)
    k1 = h*f(y+k0/2, t+h/2)
    k2 = h*f(y+k1/2, t+h/2)
    k3 = h*f(y+k2, t+h)
    return y+(k0+2*k1+2*k2+k3)/6


y0 = np.array([1, 0])
# dt = 0.01
t = np.arange(0, 20*np.pi, 0.01, dtype=float)
# computing the solution with RK4
y = odestepper("RK4_step", deriv, y0, t)
# plotting
figure, ax = plt.subplots(2, 2, figsize=(10, 10))
ax[0, 0].plot(y[:, 1], y[:, 0], 'b', label='RK4')
ax[0, 0].set_xlabel('$v$')
ax[0, 0].set_ylabel('$x$')
ax[0, 0].set_title('$dt = 0.01$')
ax[0, 1].plot(t, y[:, 0], 'b', label="RK4")
ax[0, 1].set_xlabel('$t$')
ax[0, 1].set_ylabel('$x$')
# computing the solution with forward euler
y = odestepper("euler", deriv, y0, t)
ax[0, 0].plot(y[:, 1], y[:, 0], 'r--', label='F-Euler', linewidth=1)
ax[0, 1].plot(t, y[:, 0], 'r--', label='F-Euler', linewidth=1)
#dt = 0.005
t = np.arange(0, 20*np.pi, 0.005, dtype=float)
# computing the solution with RK4
y = odestepper("RK4_step", deriv, y0, t)
# plotting
ax[1, 0].plot(y[:, 1], y[:, 0], 'b', label='RK4')
ax[1, 0].set_xlabel('$v$')
ax[1, 0].set_ylabel('$x$')
ax[1, 0].set_title('$dt = 0.005$')
ax[1, 1].plot(t, y[:, 0], 'b', label='RK4')
ax[1, 1].set_xlabel('$t$')
ax[1, 1].set_ylabel('$x$')
# computing the solution with forward euler
y = odestepper("euler", deriv, y0, t)
# plotting
ax[1, 0].plot(y[:, 1], y[:, 0], 'r--', label='F-Euler', linewidth=1)
ax[1, 1].plot(t, y[:, 0], 'r--', label='F-Euler', linewidth=1)
# computing the exact solution
y = exact(t)
# plotting
ax[0, 1].plot(t, y, 'g--', label='Exact', linewidth=2)
ax[1, 1].plot(t, y, 'g--', label='Exact', linewidth=2)
ax[0, 0].legend(loc='best')
ax[0, 1].legend(loc='best')
ax[1, 0].legend(loc='best')
ax[1, 1].legend(loc='best')
plt.show()

# %% Q2(a)


def deriv2(y, t):  # derivative of the y1 and y2
    return np.array([y[1], -2*gamma*y[1]-alpha*y[0]-beta*y[0]**3+F*np.cos(t)])


# defining the initial condition and coefficients
omega = 1
alpha = 0
beta = 1
gamma = 0.04
F = 0.2
y0 = np.array([-0.1, 0.1])
# time array for the numerical solution with dt = 0.01
t = np.arange(0, 80*np.pi, 0.01, dtype=float)
# solving with RK4
y = odestepper("RK4_step", deriv2, y0, t)
# plotting
figure, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(y[round(len(t)/4):, 1], y[round(len(t)/4):, 0], 'r-', label='RK4')
ax[0].set_xlabel('$v$')
ax[0].set_ylabel('$x$')
ax[0].set_title(r"$\omega, \alpha, \beta, \gamma$, F = {}, {}, {}, {}, {}".format(omega, alpha, beta, gamma, F))
ax[1].plot(t/2/np.pi, y[:, 0], 'b-', label='RK4')
ax[1].set_xlabel('$t$')
ax[1].set_ylabel('$x$')
ax[0].plot(y[0, 1], y[0, 0], 'bo')
ax[0].plot(y[-1, 1], y[-1, 0], 'go')
plt.show()

# %% Q2(b)
# defining the initial condition and coefficients
alpha = 0.1
F = 7.5
y0 = np.array([-0.1, 0.1])
# time array for the numerical solution with dt = 0.01
t = np.arange(0, 80*np.pi, 0.01, dtype=float)
# solving with RK4
y = odestepper("RK4_step", deriv2, y0, t)
# plotting
figure, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(y[round(len(t)/4):, 1], y[round(len(t)/4):, 0], 'r-', label='RK4')
ax[0].set_xlabel('$v$')
ax[0].set_ylabel('$x$')
ax[0].set_title(r"$\omega, \alpha, \beta, \gamma$, F = {}, {}, {}, {}, {}".format(omega, alpha, beta, gamma, F))
ax[1].plot(t/2/np.pi, y[:, 0], 'b-', label='RK4')
ax[1].set_xlabel('$t$')
ax[1].set_ylabel('$x$')
ax[0].plot(y[0, 1], y[0, 0], 'bo')
ax[0].plot(y[-1, 1], y[-1, 0], 'go')
plt.show()

# %% Q3(a)


def lorenz(y, t):  # defining the lorenz equations
    rho = 10
    r = 28
    b = 8/3
    return np.array([rho*(y[1]-y[0]), r*y[0]-y[1]-y[0]*y[2], y[0]*y[1]-b*y[2]])


# defining the initial conditions, coefficients, and time array
y0 = np.array([1, 1, 1])
t = np.arange(0, 8*np.pi, 0.01, dtype=float)
# solving with RK4
y = odestepper("RK4_step", lorenz, y0, t)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=29, azim=20)
Axes3D.plot(ax, y[:, 0], y[:, 1], y[:, 2], 'b-', label='y0=[1,1,1]')
y0 = np.array([1, 1, 1.001])
# solving with RK4
y2 = odestepper("RK4_step", lorenz, y0, t)
# plotting
Axes3D.plot(ax, y2[:, 0], y2[:, 1], y2[:, 2], 'g-', label='y0=[1,1,1.001]')
Axes3D.set_xlabel(ax, '$x$')
Axes3D.set_ylabel(ax, '$y$')
Axes3D.set_zlabel(ax, '$z$')
plt.legend(loc='best')
plt.show()

# %% Q3(b)

# defining the xyz values
x11 = np.array(y[:, 0])
y11 = np.array(y[:, 1])
z11 = np.array(y[:, 2])
x22 = np.array(y2[:, 0])
y22 = np.array(y2[:, 1])
z22 = np.array(y2[:, 2])
# defining the time array
time_vals = t
# setting up the graph
plt.rcParams.update({'font.size': 18})
fig = plt.figure(dpi=180)
ax = fig.add_axes([0.1, 0.1, 0.85, 0.85], projection='3d')
# defining the two lines for the animation
line, = ax.plot3D(x11, y11, z11, 'r-', linewidth=0.8)
line2, = ax.plot3D(x22, y22, z22, 'b-', linewidth=0.8)
# setting the axis labels
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')


def init():
    # initializing the graph with axis limits
    line.set_data(np.array([]), np.array([]))
    line.set_3d_properties([])
    line.axes.axis([-25, 25, -25, 25])
    line2.set_data(np.array([]), np.array([]))
    line2.set_3d_properties([])
    line2.axes.axis([-25, 25, -25, 25])
    return line, line2


def update(num):
    # updating the lines on each frame
    line.set_data(x11[:num], y11[:num])
    line.set_3d_properties(z11[:num])
    line2.set_data(x22[:num], y22[:num])
    line2.set_3d_properties(z22[:num])
    fig.canvas.draw()
    return line, line2


# calling the animation
ani = animation.FuncAnimation(fig, update, init_func=init, interval=1, frames=len(time_vals), blit=True, repeat=True)
plt.show()
