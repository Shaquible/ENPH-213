# Keegan Kelly
#Date: 3/24/2022
# %%
import numpy as np
import matplotlib.pyplot as plt
# %% Q1(a)


def DFT(f, n):
    if n % 2 == 1:
        print("n must be even")
        return 0, 0

    k = np.linspace(0, n-1, n, endpoint=True)
    j = np.linspace(0, n-1, n, endpoint=True)
    yj = f(2*np.pi*j/n)

    ck = np.sum(yj*np.exp(-1j*2*k[:, None]*np.pi/n*j), axis=1)
    return k, ck, 2*np.pi*j/n, yj


def IDFT(ck, n):
    if n % 2 == 1:
        print("n must be even")
        return 0, 0

    k = np.linspace(0, n-1, n, endpoint=True)
    j = np.linspace(0, n-1, n, endpoint=True)
    yj = np.sum(ck*np.exp(1j*2*np.pi*k/n*j), axis=1)
    return 2*np.pi*j/n, yj


def y(t):
    return 3*np.sin(t)+np.sin(4*t)+0.5*np.sin(7*t)


k, ck, j, yj = DFT(y, 60)
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(j, yj, "r-")
ax[0].set_xlabel("t")
ax[0].set_ylabel("y(t)")
ax[1].bar(k, abs(ck), color="r")
ax[1].set_xlabel("$\omega$")
ax[1].set_ylabel("$|ck|$")
k, ck, j, yj = DFT(y, 30)
ax[0].plot(j, yj, "b--")
ax[1].bar(k, abs(ck), color="b")
plt.show()
