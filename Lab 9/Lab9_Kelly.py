# Keegan Kelly
# 4/5/22
# I regularly attended the zoom sessions with my webcam on
import numpy as np
import matplotlib.pyplot as plt

# %% Q1
n = 1000
# generating n (x,y) points
pi = np.zeros(4, dtype=np.float64)
fig, ax = plt.subplots(1, 4, figsize=(18, 4))
circ = plt.Circle((0, 0), 0.5, color='g', fill=True)
for i in range(4):
    xy = np.random.rand(n, 2)-0.5
    # calculating the rad of each xy points and turning it into a bool for if the radius is less than that of the circle
    r = (xy[:, 0]**2 + xy[:, 1]**2) <= 0.25
    # summing the number of true conditions
    inCircle = np.sum(r)
    # calculating pi
    pi[i] = inCircle*4/n
    # plotting the circle and the points
    circ = plt.Circle((0, 0), 0.5, color='g', fill=True)
    ax[i].plot(xy[:, 0], xy[:, 1], 'bo', markersize=2)
    ax[i].add_patch(circ)
    ax[i].set_xlabel("x")
    ax[i].set_ylabel("y")
    ax[i].set_title("Pi = " + str(pi[i]) + ", n="+str(n))
plt.show()
# %% Q2


def initialize(N):  # version of the initialize function from the lecture slides but with no for loops
    # initializing all spins to 1
    spin = np.ones(N)
    # greating a array of bools to update the spins
    spinTrue = np.random.rand(n) < p
    # setting the spins to -1 if the bool is true
    spin -= spinTrue*2
    # computing the energy and magnetization
    M = np.sum(spin)
    E = -(np.sum(spin[:-1]*spin[1:]) + spin[-1]*spin[0])
    return spin, E, M


def update(N, spin, kT, E, M):  # update function taken from the lecture slides
    num = np . random . randint(0, N - 1)
    flip = 0
    # periodic bc returns 0 if i + 1 == N , else no change :
    dE = 2 * spin[num] * (spin[num - 1] + spin[(num + 1) % N])
    # if dE is negative , accept flip :
    if dE < 0.0:
        flip = 1
    else:
        p = np . exp(- dE / kT)
        if np . random . rand(1) < p:
            flip = 1
    # otherwise , reject flip
    if flip == 1:
        E += dE
        M -= 2 * spin[num]
        spin[num] = - spin[num]
    return E, M, spin


def PlotSpinsEnergies(spins, Energies):
    # plotting the spins in a pcolormesh and plotting the energies in a line plot
    fig, ax = plt.subplots(2, 1)
    iterations = np.arange(nIteration)
    ax[1].plot(iterations/n, Energies/n, 'b-', label='E')
    ax[1].set_xlabel('Iteration/N')
    ax[1].set_ylabel('$Energy/N\epsilon$')
    ax[0].set_title("Spin Evolution with kbT = " + str(kT) + " and initial p = " + str(p))
    # transforming the spins so it can be plotted in a pcolormesh
    Spins = spins.T
    ax[0].pcolormesh(Spins)
    ax[0].set_ylabel('N Spins')
    ax[0].set_xticklabels([])
    ax[0].set_xticks([])
    Ean = -np.tanh(1/kT)
    ax[1].plot(iterations/n, Ean*np.ones(nIteration), 'r--', label='$<E> an$')
    ax[1].legend()
    plt.show()


# setting initial conditions and initiallizing the arrays
n = 50
nIteration = 100*n
spins = np.zeros((nIteration, n))
Energies = np.zeros((nIteration))
Magnetizations = np.zeros((nIteration))
kT = 0.1
p = 0.2
# simulating the system with kT = 0.1 and p = 0.2
spins[0, :], Energies[0], Magnetizations[0] = initialize(n)
for i in range(1, nIteration):
    Energies[i], Magnetizations[i], spins[i, :] = update(n, spins[i-1], kT, Energies[i-1], Magnetizations[i-1])
PlotSpinsEnergies(spins, Energies)
# simulating the system with kT = 0.1 and p = 0.6
p = 0.6
spins[0, :], Energies[0], Magnetizations[0] = initialize(n)
for i in range(1, nIteration):
    Energies[i], Magnetizations[i], spins[i, :] = update(n, spins[i-1, :], kT, Energies[i-1], Magnetizations[i-1])
PlotSpinsEnergies(spins, Energies)
# simulating the system with kT = 0.5 and p = 0.6
kT = 0.5
spins[0, :], Energies[0], Magnetizations[0] = initialize(n)
for i in range(1, nIteration):
    Energies[i], Magnetizations[i], spins[i, :] = update(n, spins[i-1], kT, Energies[i-1], Magnetizations[i-1])
PlotSpinsEnergies(spins, Energies)
# simulating the system with kT = 1 and p = 0.6
kT = 1
spins[0, :], Energies[0], Magnetizations[0] = initialize(n)
for i in range(1, nIteration):
    Energies[i], Magnetizations[i], spins[i, :] = update(n, spins[i-1], kT, Energies[i-1], Magnetizations[i-1])
PlotSpinsEnergies(spins, Energies)
print("By adjusting the p value to have a cold start, the equilibrium is reached faster, and with a warm start it takes longer to reach equilibrium.")
# %% Q3
# setting initial conditions and initiallizing the arrays
# n is the number of spins
n = 50
# N is the number of temperatures
N = 100
nIteration = 800*n
spins = np.zeros((nIteration, n))
Energies = np.zeros((nIteration))
Magnetizations = np.zeros((nIteration))
AvgEnergies = np.zeros((N))
AvgMag = np.zeros((N))
# generating temperature array
kTs = np.linspace(0.1, 6, N)
p = 0.6
for i in range(N):
    # simulating the system with initial p = 0.6 and each kT in the array
    kT = kTs[i]
    spins[0, :], Energies[0], Magnetizations[0] = initialize(n)
    for j in range(1, nIteration):
        Energies[j], Magnetizations[j], spins[j, :] = update(n, spins[j-1], kT, Energies[j-1], Magnetizations[j-1])
    # computing the average energy and magnetization for the last 400*n iterations
    AvgEnergies[i] = np.mean(Energies[400*n:])
    AvgMag[i] = np.mean(Magnetizations[40*n:])
# plotting the average energy and magnetization for each temperature and the theoretical curve
fig, ax = plt.subplots(2, 1, figsize=(8, 10))
M = np.zeros_like(kTs)
E = -np.tanh(1/kTs)
ax[0].plot(kTs, AvgEnergies/n, 'bo', label='$<E>$', markersize=2)
ax[0].plot(kTs, E, 'r-', label='$<E> an$')
ax[0].set_title("Average Energy vs Temperature for N=50 spin chain")
ax[1].plot(kTs, AvgMag/n, 'bo', label='$<M>$', markersize=2)
ax[1].plot(kTs, M, 'r--', label='$<M> an$')
ax[1].set_xlabel('$kT/\epsilon$')
ax[0].set_ylabel('$<E>/N\epsilon$')
ax[1].set_ylabel('$<M>/N$')
ax[1].set_title("Average Magnetization vs Temperature for N=50 spin chain")
plt.show()

# %%
