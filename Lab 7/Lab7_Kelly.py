import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sp

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
