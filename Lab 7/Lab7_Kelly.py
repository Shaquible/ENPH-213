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
        return y0-y1+h*f(y1, t)
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


def fun(y, t):
    return -10*y


def y(t):
    return np.exp(-10*t)


texact = np.linspace(0, 0.6, 100)
yexact = y(texact)
# for n=10
t = np.arange(10)/(9)*0.6
y0 = 1
yfor = odestepper('euler', fun, y0, t)
yback = odestepper('backwardEuler', fun, y0, t)
figure, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].plot(texact, yexact, 'g', label='Exact', linewidth=3)
ax[0].plot(t, yfor, 'bo', label='Euler $n={}$'.format(10), markersize=4)
ax[0].plot(t, yback, 'ro', label='Backward Euler $n={}$'.format(10), markersize=4)
ax[0].set_xlabel('$t$')
ax[0].set_ylabel('$y$')
ax[0].legend(loc='best')
# for n=20
t = np.arange(20)/(19)*0.6
y0 = 1
yfor = odestepper('euler', fun, y0, t)
yback = odestepper('backwardEuler', fun, y0, t)
ax[1].plot(texact, yexact, 'g', label='Exact', linewidth=3)
ax[1].plot(t, yfor, 'bo', label='Euler $n={}$'.format(20), markersize=4)
ax[1].plot(t, yback, 'ro', label='Backward Euler $n={}$'.format(20), markersize=4)
ax[1].set_xlabel('$t$')
ax[1].set_ylabel('$y$')
ax[1].legend(loc='best')
plt.show()

# %% Q1(b)


def deriv(y, t):
    return np.array([y[1], -y[0]])


def exact(t):
    return np.cos(t)


def RK4(f, y0, t, h):
    def RK4_step(y, t, h):
        k0 = h*f(y, t)
        k1 = h*f(y+k0/2, t+h/2)
        k2 = h*f(y+k1/2, t+h/2)
        k3 = h*f(y+k2, t+h)
        return (k0+2*k1+2*k2+k3)/6
    y = np.zeros((t.size, y0.size))
    y[0, :] = y0
    for i in range(1, len(t)):
        y[i, :] = y[i-1, :]+RK4_step(y[i-1, :], t[i-1], h)
    return y


y0 = np.array([1, 0])
# dt = 0.01
t = np.arange(0, 20*np.pi, 0.01, dtype=float)

y = RK4(deriv, y0, t, 0.01)
figure, ax = plt.subplots(2, 2, figsize=(10, 10))
ax[0, 0].plot(y[:, 1], y[:, 0], 'b', label='RK4')
ax[0, 0].set_xlabel('$v$')
ax[0, 0].set_ylabel('$x$')
ax[0, 0].set_title('$dt = 0.01$')
ax[0, 1].plot(t, y[:, 0], 'b', label="RK4")
ax[0, 1].set_xlabel('$t$')
ax[0, 1].set_ylabel('$x$')

y = odestepper("euler", deriv, y0, t)
ax[0, 0].plot(y[:, 1], y[:, 0], 'r--', label='F-Euler', linewidth=1)
ax[0, 1].plot(t, y[:, 0], 'r--', label='F-Euler', linewidth=1)
t = np.arange(0, 20*np.pi, 0.005, dtype=float)
y = RK4(deriv, y0, t, 0.005)
ax[1, 0].plot(y[:, 1], y[:, 0], 'b', label='RK4')
ax[1, 0].set_xlabel('$v$')
ax[1, 0].set_ylabel('$x$')
ax[1, 0].set_title('$dt = 0.005$')
ax[1, 1].plot(t, y[:, 0], 'b', label='RK4')
ax[1, 1].set_xlabel('$t$')
ax[1, 1].set_ylabel('$x$')
y = odestepper("euler", deriv, y0, t)
ax[1, 0].plot(y[:, 1], y[:, 0], 'r--', label='F-Euler', linewidth=1)
ax[1, 1].plot(t, y[:, 0], 'r--', label='F-Euler', linewidth=1)
y = exact(t)
ax[0, 1].plot(t, y, 'g--', label='Exact', linewidth=2)
ax[1, 1].plot(t, y, 'g--', label='Exact', linewidth=2)
ax[0, 0].legend(loc='best')
ax[0, 1].legend(loc='best')
ax[1, 0].legend(loc='best')
ax[1, 1].legend(loc='best')
plt.show()

# %% Q2(a)


def deriv2(y, t):
    return np.array([y[1], -0.08*y[1]-y[0]**3+0.2*np.cos(t)])


omega = 1
alpha = 0
beta = 1
gamma = 0.04
F = 0.2
y0 = np.array([-0.1, 0.1])
t = np.arange(0, 80*np.pi, 0.01, dtype=float)
y = RK4(deriv2, y0, t, 0.01)
figure, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(y[round(len(t)/4):, 1], y[round(len(t)/4):, 0], 'r-', label='RK4')
ax[0].set_xlabel('$v$')
ax[0].set_ylabel('$x$')
ax[0].set_title(r"$\omega, \alpha, \beta, \gamma$, F = {},{},{},{},{}".format(omega, alpha, beta, gamma, F))
ax[1].plot(t/2/np.pi, y[:, 0], 'b-', label='RK4')
ax[1].set_xlabel('$t$')
ax[1].set_ylabel('$x$')
ax[0].plot(y[0, 1], y[0, 0], 'bo')
ax[0].plot(y[-1, 1], y[-1, 0], 'go')
plt.show()

# %% Q2(b)


def deriv3(y, t):
    return np.array([y[1], -0.08*y[1]-0.1*y[0]-y[0]**3+7.5*np.cos(t)])


alpha = 0.1
F = 7.5
y0 = np.array([-0.1, 0.1])
t = np.arange(0, 80*np.pi, 0.01, dtype=float)
y = RK4(deriv3, y0, t, 0.01)
figure, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(y[round(len(t)/4):, 1], y[round(len(t)/4):, 0], 'r-', label='RK4')
ax[0].set_xlabel('$v$')
ax[0].set_ylabel('$x$')
ax[0].set_title(r"$\omega, \alpha, \beta, \gamma$, F = {},{},{},{},{}".format(omega, alpha, beta, gamma, F))
ax[1].plot(t/2/np.pi, y[:, 0], 'b-', label='RK4')
ax[1].set_xlabel('$t$')
ax[1].set_ylabel('$x$')
ax[0].plot(y[0, 1], y[0, 0], 'bo')
ax[0].plot(y[-1, 1], y[-1, 0], 'go')
plt.show()

# %% Q3(a)


def lorenz(y, t):
    rho = 10
    r = 28
    b = 8/3
    return np.array([rho*(y[1]-y[0]), r*y[0]-y[1]-y[0]*y[2], y[0]*y[1]-b*y[2]])


y0 = np.array([1, 1, 1])
t = np.arange(0, 8*np.pi, 0.01, dtype=float)
y = RK4(lorenz, y0, t, 0.01)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=29, azim=20)
Axes3D.plot(ax, y[:, 0], y[:, 1], y[:, 2], 'b-', label='y0=[1,1,1]')
y0 = np.array([1, 1, 1.001])
y2 = RK4(lorenz, y0, t, 0.01)
Axes3D.plot(ax, y2[:, 0], y2[:, 1], y2[:, 2], 'g-', label='y0=[1,1,1.001]')
Axes3D.set_xlabel(ax, '$x$')
Axes3D.set_ylabel(ax, '$y$')
Axes3D.set_zlabel(ax, '$z$')
plt.legend(loc='best')
plt.show()

# %% Q3(b)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


def update(num, data, line):
    line.set_data(data[:2, :num])
    line.set_3d_properties(data[2, :num])


N = len(t)
data = np.array(y).T
line, = ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1])

ax.set_xlim3d([-20, 20])
ax.set_ylim3d([-20, 20])
ax.set_zlim3d([0, 50])
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')

ani = animation.FuncAnimation(fig, update, N, fargs=(data, line), interval=10, blit=False)
plt.show()
