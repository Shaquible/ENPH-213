# Keegan Kelly
#Date: 3/6/2022
# %%
import numpy as np
import matplotlib.pyplot as plt
# %% Q1(a)


def generateData(n, f, spacing="equal"):
    # Generating Equal Spacing
    if spacing == "equal":
        x = np.linspace(-1, 1, n, endpoint=True)
    elif spacing == "cheb":
        # generating a linspace for j values
        j = np.linspace(0, n-1, n, endpoint=True)
        # calculting Chebyshev points
        x = -np.cos(j*np.pi/(n-1))
    # returning function at the points
    y = f(x)
    return x, y


def runge(x):  # defining the Runge function
    return 1/(1+25*x**2)


def polynomailInterpolation(xs, ys):  # xs, ys are the given points
    # filling array with the x values
    n = len(xs)
    A = np.ones((n, n), dtype=float)
    A = A*xs[:, np.newaxis]
    for i in range(n):
        # raising each x to the correct power
        A[:, i] = A[:, i]**i
    # computing the c values
    cs = np.linalg.solve(A, ys)

    def p(x):
        p = 0
        # summing according to equation 6
        for i in range(n):
            p += cs[i]*x**i
        return p
    # returning a function for the polynomial
    return p


def LagrangeBasis(xs, i, xInterp):  # function to define the Lagrange Basis polynomials, xs are given, i is the degree, xInterp is the points to interpolate
    n = len(xs)
    L = 1
    # multiplying through the series according to eqn 8
    for j in range(n):
        if j == i:
            continue
        L *= (xInterp - xs[j])/(xs[i] - xs[j])
    return L


def Lagrange(xs, ys, xInterp):  # xs, ys, are the given points, xInterp is the points to interpolate
    n = len(xs)
    # filling blank array
    yOut = np.zeros_like(xInterp)
    # summing for all ks according to equation 9
    for k in range(n):
        yOut += ys[k]*LagrangeBasis(xs, k, xInterp)
    return yOut


# plotting the polynomail vs lagrange for equal spacing
lw = 1
ms = 1
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
xEqual, yEqual = generateData(15, runge, spacing="equal")
axs[1].plot(xEqual, yEqual, "ko", label="Points", markersize=ms)
p = polynomailInterpolation(xEqual, yEqual)
xs = np.linspace(-1, 1, 100)
ys = p(xs)
axs[1].plot(xs, ys, "r-", label="Monomial", linewidth=lw)
ys = Lagrange(xEqual, yEqual, xs)
axs[1].plot(xs, ys, "b--", label="Lagrange", linewidth=lw)
axs[1].set_title("Equal Spacing")
axs[1].set_xlabel("x")
axs[1].set_ylabel("y")
axs[1].legend()

# plotting the polynomail vs lagrange for cheb spacing
xCheb, yCheb = generateData(15, runge, spacing="cheb")
axs[0].plot(xCheb, yCheb, "ko", label="Points", markersize=ms)
p = polynomailInterpolation(xCheb, yCheb)
ys = p(xs)
axs[0].plot(xs, ys, "r-", label="Monomial", linewidth=lw)
ys = Lagrange(xCheb, yCheb, xs)
axs[0].plot(xs, ys, "b--", label="Lagrange", linewidth=lw)
axs[0].set_title("Chebyshev")
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")
axs[0].legend()
plt.show()
# %% Q1(b)
# plotting 91 cheb points for polynomial and lagrange
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
xCheb, yCheb = generateData(91, runge, spacing="cheb")
axs[0].plot(xCheb, yCheb, "ko", label="Points", markersize=ms)
p = polynomailInterpolation(xCheb, yCheb)
ys = p(xs)
axs[0].plot(xs, ys, "r-", label="Monomial", linewidth=lw)
ys = Lagrange(xCheb, yCheb, xs)
axs[0].plot(xs, ys, "b--", label="Lagrange", linewidth=lw)
axs[0].set_title("n=91")
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")
axs[0].legend()

# plotting 101 cheb points for polynomial and lagrange
xCheb, yCheb = generateData(101, runge, spacing="cheb")
axs[1].plot(xCheb, yCheb, "ko", label="Points", markersize=ms)
p = polynomailInterpolation(xCheb, yCheb)
ys = p(xs)
axs[1].plot(xs, ys, "r-", label="Monomial", linewidth=lw)
ys = Lagrange(xCheb, yCheb, xs)
axs[1].plot(xs, ys, "b--", label="Lagrange", linewidth=lw)
axs[1].set_title("n=101")
axs[1].set_xlabel("x")
axs[1].set_ylabel("y")
axs[1].legend()
plt.show()
print("On my desktop the Monomial interpolation is better for n=101 around the boundaries but the Lagrange looks the same.")
print("I ran this on my laptop and the monomial broke down for n=91.")

# %% Q2


def cubicSpline(xs, ys, xInterp):  # xs, ys, are the given points, xInterp is the points to interpolate
    n = len(xs)
    # calculating bs according to equation 13
    bs = np.array(6*((ys[2:] - ys[1:-1])/(xs[2:] - xs[1:-1]) - (ys[1:-1] - ys[:-2])/(xs[1:-1] - xs[:-2])))
    A = np.zeros((n-2, n-2), dtype=float)
    # generatinge the middle upper and lower diagonal matrices
    diagMid = np.diagflat(np.ones_like(A[:, 0]))
    diagUP = np.diagflat(np.ones_like(A[1:, 0]), k=1)
    diagDOWN = np.diagflat(np.ones_like(A[1:, 0]), k=-1)
    # adding the diagonals times the correct difference of x values to one matrix
    A += 2*diagMid*(xs[2:, None]-xs[:-2, None])
    A += diagUP*(xs[2:, None]-xs[1:-1, None])
    A += diagDOWN*(xs[1:-1, None]-xs[:-2, None])
    # solving for cs and adding 0 to the front and back of the array
    cs = np.linalg.solve(A, bs)
    cs = np.insert(cs, 0, 0)
    cs = np.insert(cs, n-1, 0)
    # initializing k
    k = 1
    y = np.zeros_like(xInterp)
    for i in range(len(xInterp)):
        # changing K values as the array is looped through
        if xInterp[i] > xs[k]:
            k += 1
        # calculating the y with Equation 12
        y[i] = ys[k-1]*((xs[k]-xInterp[i])/(xs[k]-xs[k-1]))+ys[k]*((xInterp[i]-xs[k-1])/(xs[k]-xs[k-1]))+cs[k-1]/6*((xs[k]-xInterp[i])*(xs[k]-xs[k-1])-(xs[k]-xInterp[i])**3/(xs[k]-xs[k-1]))-cs[k]/6*((xInterp[i]-xs[k-1])*(xs[k]-xs[k-1])-(xInterp[i]-xs[k-1])**3/(xs[k]-xs[k-1]))
    # returning the array of y values
    return y


# plotting the cubic spline interpolation with n= 7 and n=15 for equal spaced and chebyshev spaced points
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
xCheb, yCheb = generateData(7, runge, spacing="cheb")
xs = np.linspace(-1, 1, 100)
ys = cubicSpline(xCheb, yCheb, xs)
axs[0].plot(xs, ys, "r-", label="n=7", linewidth=lw)
axs[0].plot(xCheb, yCheb, "bo", label="n=7 Points", markersize=ms+1)
xCheb, yCheb = generateData(15, runge, spacing="cheb")
ys = cubicSpline(xCheb, yCheb, xs)
axs[0].plot(xs, ys, "b--", label="n=15", linewidth=lw)
axs[0].plot(xCheb, yCheb, "ko", label="n=15 Points", markersize=ms)
axs[0].set_title("Chebyshev")
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")
axs[0].legend()
xEqual, yEqual = generateData(7, runge, spacing="equal")
ys = cubicSpline(xEqual, yEqual, xs)
axs[1].plot(xs, ys, "r-", label="n=7", linewidth=lw)
axs[1].plot(xEqual, yEqual, "bo", label="n=7 Points", markersize=ms+1)
xEqual, yEqual = generateData(15, runge, spacing="equal")
ys = cubicSpline(xEqual, yEqual, xs)
axs[1].plot(xs, ys, "b--", label="n=15", linewidth=lw)
axs[1].plot(xEqual, yEqual, "ko", label="n=15 Points", markersize=ms)
axs[1].set_title("Equal Spacing")
axs[1].set_xlabel("x")
axs[1].set_ylabel("y")
axs[1].legend()
plt.show()

# %% Q3


def trigonometricInterpolation(f, n, nInterp):  # f is the function, n is the number of points, nInterp is the number of points to interpolate
    m = (n-1)/2
    # generating k values and empty coefficients
    ks = np.linspace(0, m, int(m)+1, endpoint=True)
    aks = np.zeros_like(ks)
    bks = np.zeros_like(ks)
    # calculating the given x and y points
    xj = 2*np.pi/n*np.linspace(0, n, num=n, endpoint=False)
    yj = f(xj)
    # generating the x points to be interpreted
    xInterp = 2*np.pi/nInterp*np.linspace(0, nInterp, nInterp)
    # summing according to 19 for the sin and cos coefficients
    for i in range(0, int(m)+1):
        aks[i] = 1/m*np.sum(np.cos(ks[i]*xj)*yj)
        bks[i] = 1/m*np.sum(np.sin(ks[i]*xj)*yj)
    # getting a0 and cutting of k=0 from the coefficient and k arrays
    a0 = aks[0]
    aks = aks[1:]
    bks = bks[1:]
    ks = ks[1:]

    yInterp = np.zeros_like(xInterp)
    for i in range(len(xInterp)):
        # calculating the y values from equation 16
        yInterp[i] = 1/2*a0+np.sum(aks*np.cos(ks*xInterp[i])+bks*np.sin(ks*xInterp[i]))
    # returning the given and interpreted values
    return xInterp, yInterp, xj, yj


def f(x): return np.e**(np.sin(2*x))  # defining the function to be interpolated


# plotting the trigonometric interpolation with 11 and 51 points
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
xInterp, yInterp, xs, ys = trigonometricInterpolation(f, 11, 500)
axs[0].plot(xInterp, yInterp, "r-", label="Trig Interpolation", linewidth=lw)
axs[0].plot(xs, ys, "ko", markersize=ms, label="Points")
axs[0].set_title("n=11")
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")
axs[0].legend()
xInterp, yInterp, xs, ys = trigonometricInterpolation(f, 51, 500)
axs[1].plot(xInterp, yInterp, "r-", label="Trig Interpolation", linewidth=lw)
axs[1].plot(xs, ys, "ko", markersize=ms, label="Points")
axs[1].set_title("n=51")
axs[1].set_xlabel("x")
axs[1].set_ylabel("y")
axs[1].legend()
plt.show()
