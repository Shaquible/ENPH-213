# Keegan Kelly
# Date: 3/24/2022
# %%
import numpy as np
import matplotlib.pyplot as plt
# %% Q1(a)


def DFT(f, n, method="linspace"):
    if n % 2 == 1:
        print("n must be even")
        return 0, 0
    # using linspace or arange to get the correct time array
    if method == "linspace":
        j = np.linspace(0, 2*np.pi, n, endpoint=False)
    if method == "arange":
        j = np.arange(0, 2*np.pi, 2*np.pi/n)
    # generating k array and y at the time array
    k = np.linspace(0, n-1, n, endpoint=True)
    yj = f(j)
    # summing across all time for each k
    ck = np.sum(yj*np.exp(-1j*k[:, None]*j), axis=1)
    return k, ck


def IDFT(ck, n, tmin=0, tmax=2*np.pi):
    if n % 2 == 1:
        print("n must be even")
        return 0, 0
    k = np.linspace(0, n-1, n, endpoint=True)
    j = np.linspace(0, n-1, n, endpoint=True)
    # calculating the IDFT according to equation 13
    yj = 1/n*np.sum(ck*np.exp(1j*k*((tmax-tmin)/n*j[:, None]+tmin)), axis=1)
    return (tmax-tmin)*j/n+tmin, yj


def y(t):  # defining the function for use in Question 1
    return 3*np.sin(t)+np.sin(4*t)+0.5*np.sin(7*t)


# plotting the dft and the given function for n=60 with a linspace
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
# plotting the dft and the given function for n=30 with a linspace
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
# plotting the dft and the given function for n=60 with a linspace
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
k, ck = DFT(y, 60)
j = np.linspace(0, 2*np.pi, 60)
yj = y(j)
ax[0].stem(k, abs(ck), linefmt='r', markerfmt=' ', basefmt='-r', label="n=60")
ax[0].set_xlabel("$\omega$")
ax[0].set_ylabel(r"$|\tilde{y}|$")
ck1 = ck
# plotting the dft and the given function for n=30 with a linspace
k, ck = DFT(y, 30)
j = np.linspace(0, 2*np.pi, 30)
yj = y(j)
ax[0].stem(k, abs(ck), linefmt='b', markerfmt=' ', basefmt='-b', label="n=30")
# plotting the dft and the given function for n=60 with arange
k, ck = DFT(y, 60, method="arange")
j = np.linspace(0, 2*np.pi, 60)
yj = y(j)
ax[1].stem(k, abs(ck), linefmt='r', markerfmt=' ', basefmt='-r', label="n=60")
ax[1].set_xlabel("$\omega$")
ax[1].set_ylabel(r"$|\tilde{y}|$")
ck2 = ck
ks = k
# plotting the dft and the given function for n=30 with arange
k, ck = DFT(y, 30, method="arange")
j = np.linspace(0, 2*np.pi, 30)
yj = y(j)
ax[1].stem(k, abs(ck), linefmt='b', markerfmt=' ', basefmt='-b', label="n=30")
ax[0].legend()
ax[1].legend()
ax[0].set_title("linspace")
ax[1].set_title("arange")
plt.show()
# plotting the difference between the arange and linspace dfts
plt.plot(ks, abs(ck1-ck2))
plt.title("Difference between linspace and arange")
plt.xlabel("$\omega$")
plt.ylabel(r"|$\Delta\tilde{y}$|")
plt.show()

# %% Q1(c)
# plotting the IDFT against the given function for n=60 with a linspace
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


def GaussianPulse(t, w=0, sigma=0.5):
    return np.exp(-t**2/sigma**2)*np.cos(w*t)

# defining a DFT function that will take an arbitrary time interval


def DFTvariableTime(f, n, tmin, tmax, w, sigma):
    if n % 2 == 1:
        print("n must be even")
        return 0, 0
    deltat = (tmax-tmin)/n
    # this j array will be used to generate the time values and then used as Ks in the sum
    j = np.linspace(0, n-1, n, endpoint=True)
    tj = j*deltat+tmin

    yj = f(tj, w, sigma)
    # summing across all time values for each k (now in the j array)
    ck = np.sum(yj*np.exp(-1j*j[:, None]*tj), axis=1)
    return j, ck


def DFTshift(f, n, tmin, tmax, w, sigma):
    # same code for the initial DTF but with a shift at the end
    if n % 2 == 1:
        print("n must be even")
        return 0, 0
    deltat = (tmax-tmin)/n
    j = np.linspace(0, n-1, n, endpoint=True)
    tj = j*deltat+tmin

    yj = f(tj, w, sigma)
    ck = np.sum(yj*np.exp(-1j*j[:, None]*tj), axis=1)
    # shifting according to the code given on the lecture slides
    wshift = np.fft.fftshift(np.fft.fftfreq(n, deltat)*2*np.pi)
    yshift = np.fft.fftshift(ck)
    return wshift, yshift


# plotting the dft for the pulse for n=60
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

# plotting the new function for n=400 with the shifted DFT
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

# repeating above but doubling the frequency resolution
wshift, yshift = DFTshift(GaussianPulse, 800, -2*np.pi, 2*np.pi, 10, 1)
t = np.linspace(-np.pi, np.pi, 800)
yj = GaussianPulse(t, 10, 1)
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(t, yj, "r-", label="$\omega_p=10$")
ax[0].set_xlabel("t")
ax[0].set_ylabel("y(t)")
ax[1].plot(wshift, abs(yshift), "r-", label="$\omega_p=10$")
ax[1].set_xlabel("$\omega$")
ax[1].set_ylabel(r"$|\tilde{y}|$")
wshift, yshift = DFTshift(GaussianPulse, 800, -2*np.pi, 2*np.pi, 20, 1)
yj = GaussianPulse(t, 20, 1)
ax[0].plot(t, yj, "b--", label="$\omega_p=20$")
ax[1].plot(wshift, abs(yshift), "b--", label="$\omega_p=20$")
ax[1].legend()
plt.show()

# %% Q3


def y1(t):
    return 3*np.sin(t)+np.sin(10*t)


def DFTvariableTimeY1(f, n, tmin, tmax):
    # same DFT variable time array as before but does have the other variables to call the pulse fn
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
    # searching for the most significant frequency and turning all others to 0 to eliminate noise
    # finding the most significant frequency component
    max = abs(ck).max()
    found = False
    # searching for the first index to have roughly the max magnitude so the frequency chosen is in the first cycle
    for i in range(len(w)):
        if round(abs(ck[i])) != round(max) or found:
            ck[i] = 0
        else:
            found = True
    return ck


# calculating the dft and plotting it and the original function
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
# filtering the frequencies and plotting them
ck = freqFilter(1, k, ck)
# calculating the IDFT with the filtered frequencies and plotting it
xs, ys = IDFT(ck, 200, 0, 8*np.pi)
ax[0].plot(xs, ys, "b--", label="filtered")
ax[1].plot(k, abs(ck), "b-", label="filtered")
plt.show()
