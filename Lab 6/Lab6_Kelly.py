# Keegan Kelly
#Date: 3/24/2022
# %%
import numpy as np
import matplotlib.pyplot as plt
# %% Q1(a)

# NEED TO MAKE THIS DO 2 CYCLES IN THE TIME ARRAY


def DFT(f, n, method="linspace"):
    if n % 2 == 1:
        print("n must be even")
        return 0, 0
    if method == "linspace":
        j = np.linspace(0, n-1, n, endpoint=True)
    if method == "arange":
        j = np.arange(0, n)
    k = np.linspace(0, n-1, n, endpoint=True)
    yj = f(2*np.pi*j/n)

    ck = np.sum(yj*np.exp(-1j*2*k[:, None]*np.pi/n*j), axis=1)
    return k, ck, 2*np.pi*j/n, yj


def IDFT(ck, n):
    if n % 2 == 1:
        print("n must be even")
        return 0, 0
    k = np.linspace(0, n-1, n, endpoint=True)
    j = np.linspace(0, n-1, n, endpoint=True)
    yj = 1/n*np.sum(ck*np.exp(1j*2*np.pi*k/n*j[:, None]), axis=1)
    return 2*np.pi*j/n, yj


def y(t):
    return 3*np.sin(t)+np.sin(4*t)+0.5*np.sin(7*t)


k, ck, j, yj = DFT(y, 60)
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(j, yj, "r-", label="n=60")
ax[0].set_xlabel("t")
ax[0].set_ylabel("y(t)")
ax[1].stem(k, abs(ck), linefmt='r', markerfmt=' ', basefmt='-r', label="n=60")
ax[1].set_xlabel("$\omega$")
ax[1].set_ylabel(r"$|\tilde{y}|$")
k, ck, j, yj = DFT(y, 30)
ax[0].plot(j, yj, "b--", label="n=30")
ax[1].stem(k, abs(ck), linefmt='b', markerfmt=' ', basefmt='-b', label="n=30")
ax[0].legend()
ax[1].legend()
ax[0].set_title("linspace")
plt.show()
# %% Q1(b)
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
k, ck, j, yj = DFT(y, 60)
ax[0].stem(k, abs(ck), linefmt='r', markerfmt=' ', basefmt='-r', label="n=60")
ax[0].set_xlabel("$\omega$")
ax[0].set_ylabel(r"$|\tilde{y}|$")
ck1 = ck
k, ck, j, yj = DFT(y, 30)
ax[0].stem(k, abs(ck), linefmt='b', markerfmt=' ', basefmt='-b', label="n=30")

k, ck, j, yj = DFT(y, 60, method="arange")
ax[1].stem(k, abs(ck), linefmt='r', markerfmt=' ', basefmt='-r', label="n=60")
ax[1].set_xlabel("$\omega$")
ax[1].set_ylabel(r"$|\tilde{y}|$")
ck2 = ck
ks = k
k, ck, j, yj = DFT(y, 30, method="arange")
ax[1].stem(k, abs(ck), linefmt='b', markerfmt=' ', basefmt='-b', label="n=30")
ax[0].legend()
ax[1].legend()
ax[0].set_title("linspace")
ax[1].set_title("arange")
plt.show()
plt.plot(ks, abs(ck1-ck2))
plt.title("Difference between linspace and arange")
plt.xlabel("$\omega$")
plt.ylabel(r"|$\Delta\tilde{y}$|")
plt.show()

# %% Q1(c)
k, ck, j, yj = DFT(y, 60)
ts, ys = IDFT(ck, 60)
plt.plot(np.real(ts), np.real(ys), "r-", label="IDFT")
plt.plot(np.real(j), np.real(yj), "b--", label="Original")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.show()

# %% Q2(a)
