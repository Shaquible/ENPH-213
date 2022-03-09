# Keegan Kelly
# Date: 3/24/2022
# %%
import numpy as np
import matplotlib.pyplot as plt
# %% Q1(a)


def DFT(f, n, method="linspace", tmin=0, tmax=2*np.pi):
    if n % 2 == 1:
        print("n must be even")
        return 0, 0
    if method == "linspace":
        j = np.linspace(0, n-1, n, endpoint=True)
    if method == "arange":
        j = np.arange(0, n)
    k = np.linspace(0, n-1, n, endpoint=True)
    yj = f(2*np.pi*j/n)
    ck = np.sum(yj*np.exp(-1j*k[:, None]*(2*np.pi/n*j)), axis=1)
    return k, ck


def IDFT(ck, n, tmin=0, tmax=2*np.pi):
    if n % 2 == 1:
        print("n must be even")
        return 0, 0
    k = np.linspace(0, n-1, n, endpoint=True)
    j = np.linspace(0, n-1, n, endpoint=True)
    yj = 1/n*np.sum(ck*np.exp(1j*k*((tmax-tmin)/n*j[:, None]+tmin)), axis=1)
    return (tmax-tmin)*j/n+tmin, yj


def y(t):
    return 3*np.sin(t)+np.sin(4*t)+0.5*np.sin(7*t)


k, ck = DFT(y, 60)
j = np.linspace(0, 2*np.pi, 60)
yj = y(j)
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(j, yj, "r-", label="n=60")
ax[0].set_xlabel("t")
ax[0].set_ylabel("y(t)")
ax[1].stem(k, abs(ck), linefmt='r', markerfmt=' ', basefmt='-r', label="n=60")
ax[1].set_xlabel("$\omega$")
ax[1].set_ylabel(r"$|\tilde{y}|$")
k, ck = DFT(y, 30)
j = np.linspace(0, 2*np.pi, 30)
yj = y(j)
ax[0].plot(j, yj, "b--", label="n=30")
ax[1].stem(k, abs(ck), linefmt='b', markerfmt=' ', basefmt='-b', label="n=30")
ax[0].legend()
ax[1].legend()
ax[0].set_title("linspace")
plt.show()
# %% Q1(b)
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
k, ck = DFT(y, 60)
j = np.linspace(0, 2*np.pi, 60)
yj = y(j)
ax[0].stem(k, abs(ck), linefmt='r', markerfmt=' ', basefmt='-r', label="n=60")
ax[0].set_xlabel("$\omega$")
ax[0].set_ylabel(r"$|\tilde{y}|$")
ck1 = ck
k, ck = DFT(y, 30)
j = np.linspace(0, 2*np.pi, 30)
yj = y(j)
ax[0].stem(k, abs(ck), linefmt='b', markerfmt=' ', basefmt='-b', label="n=30")

k, ck = DFT(y, 60, method="arange")
j = np.linspace(0, 2*np.pi, 60)
yj = y(j)
ax[1].stem(k, abs(ck), linefmt='r', markerfmt=' ', basefmt='-r', label="n=60")
ax[1].set_xlabel("$\omega$")
ax[1].set_ylabel(r"$|\tilde{y}|$")
ck2 = ck
ks = k
k, ck = DFT(y, 30, method="arange")
j = np.linspace(0, 2*np.pi, 30)
yj = y(j)
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
k, ck = DFT(y, 60)
j = np.linspace(0, 2*np.pi, 60)
yj = y(j)
ts, ys = IDFT(ck, 60)
plt.plot(np.real(ts), np.real(ys), "r-", label="IDFT")
plt.plot(np.real(j), np.real(yj), "b--", label="Original")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.show()

# %% Q2(a)
# NEEDS TO BE ADJUSTED FOR TO ACCEPT MANY TIME STEPS


def GaussianPulse(t, w=0, sigma=0.5):
    return np.exp(-t**2/sigma**2)*np.cos(w*t)


def DFTvariableTime(f, n, tmin, tmax, w, sigma):
    if n % 2 == 1:
        print("n must be even")
        return 0, 0
    deltat = (tmax-tmin)/n
    j = np.linspace(0, n-1, n, endpoint=True)
    tj = j*deltat+tmin

    yj = f(tj, w, sigma)
    ck = np.sum(yj*np.exp(-1j*j[:, None]*tj), axis=1)
    return j, ck


def DFTshift(f, n, tmin, tmax, w, sigma):
    if n % 2 == 1:
        print("n must be even")
        return 0, 0
    deltat = (tmax-tmin)/n
    j = np.linspace(0, n-1, n, endpoint=True)
    tj = j*deltat+tmin

    yj = f(tj, w, sigma)
    ck = np.sum(yj*np.exp(-1j*j[:, None]*tj), axis=1)
    wshift = np.fft.fftshift(np.fft.fftfreq(n, deltat)*2*np.pi)
    yshift = np.fft.fftshift(ck)
    return wshift, yshift


k, ck = DFTvariableTime(GaussianPulse, 60, -np.pi, np.pi, 0, 0.5)
t = np.linspace(-np.pi, np.pi, 60)
yj = GaussianPulse(t)
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(t, yj, "r-", label="n=60")
ax[0].set_xlabel("t")
ax[0].set_ylabel("y(t)")
ax[1].plot(k, abs(ck), "r-", label="no shift")
ax[1].set_xlabel("$\omega$")
ax[1].set_ylabel(r"$|\tilde{y}|$")
kshift, ckshift = DFTshift(GaussianPulse, 60, -np.pi, np.pi, 0, 0.5)
ax[1].plot(kshift, abs(ckshift), "b--", label="shift")
ax[1].legend()
plt.show()

# %% Q2(b)


wshift, yshift = DFTshift(GaussianPulse, 400, -np.pi, np.pi, 10, 1)
t = np.linspace(-np.pi, np.pi, 400)
yj = GaussianPulse(t, 10, 1)
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(t, yj, "r-", label="$\omega_p=10$")
ax[0].set_xlabel("t")
ax[0].set_ylabel("y(t)")
ax[1].plot(wshift, abs(yshift), "r-", label="$\omega_p=10$")
ax[1].set_xlabel("$\omega$")
ax[1].set_ylabel(r"$|\tilde{y}|$")
wshift, yshift = DFTshift(GaussianPulse, 400, -np.pi, np.pi, 20, 1)
yj = GaussianPulse(t, 20, 1)
ax[0].plot(t, yj, "b--", label="$\omega_p=20$")
ax[1].plot(wshift, abs(yshift), "b--", label="$\omega_p=20$")
ax[1].set_xlim(-40, 40)
ax[1].legend()
plt.show()

# %% Q3


def y1(t):
    return 3*np.sin(t)+np.sin(10*t)


def DFTvariableTimeY1(f, n, tmin, tmax):
    if n % 2 == 1:
        print("n must be even")
        return 0, 0
    deltat = (tmax-tmin)/n
    j = np.linspace(0, n-1, n, endpoint=True)
    tj = j*deltat+tmin

    yj = f(tj)
    ck = np.sum(yj*np.exp(-1j*j[:, None]*tj), axis=1)
    return j, ck


def freqFilter(w0, w, ck):
    max = abs(ck).max()
    found = False
    for i in range(len(w)):
        if round(abs(ck[i])) != round(max) or found:
            ck[i] = 0
        else:
            found = True
    return ck


k, ck = DFTvariableTimeY1(y1, 200, 0, 8*np.pi)
t = np.linspace(0, 8*np.pi, 200)
yj = y1(t)
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(t, yj, "r-", label="no filter")
ax[0].set_xlabel("t")
ax[0].set_ylabel("y(t)")
ax[1].plot(k, abs(ck), "r-", label="filter")
ax[1].set_xlabel("$\omega$")
ax[1].set_ylabel(r"$|\tilde{y}|$")
ax[1].set_xlim(0, 50)
ck = freqFilter(1, k, ck)
xs, ys = IDFT(ck, 200, 0, 8*np.pi)
ax[0].plot(xs, ys, "b--", label="filtered")
ax[1].plot(k, abs(ck), "b-", label="filtered")
plt.show()
