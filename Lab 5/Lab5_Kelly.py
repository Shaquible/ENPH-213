# %%
import numpy as np
import matplotlib.pyplot as plt
# %% Q1(a)


def generateData(n, f, spacing="equal"):
    if spacing == "equal":
        x = np.linspace(-1, 1, n, endpoint=True)
    elif spacing == "cheb":
        j = np.linspace(0, n-1, n, endpoint=True)
        x = -np.cos(j*np.pi/(n-1))
    y = f(x)
    return x, y


def runge(x):
    return 1/(1+25*x**2)


def polynomailInterpolation(xs, ys):
    n = len(xs)
    A = np.ones((n, n), dtype=float)
    A = A*xs[:, np.newaxis]
    for i in range(n):
        A[:, i] = A[:, i]**i
    cs = np.linalg.solve(A, ys)

    def p(x):
        p = 0
        for i in range(n):
            p += cs[i]*x**i
        return p
    return p


def LagrangeBasis(xs, i, xInterp):
    n = len(xs)
    L = 1
    for j in range(n):
        if j == i:
            continue
        L *= (xInterp - xs[j])/(xs[i] - xs[j])
    return L


def Lagrange(xs, ys, xInterp):
    n = len(xs)
    yOut = np.zeros_like(xInterp)
    for k in range(n):
        yOut += ys[k]*LagrangeBasis(xs, k, xInterp)
    return yOut


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


def cubicSpline(xs, ys, xInterp):
    n = len(xs)
    bs = np.array(6*((ys[2:] - ys[1:-1])/(xs[2:] - xs[1:-1]) - (ys[1:-1] - ys[:-2])/(xs[1:-1] - xs[:-2])))
    A = np.zeros((n-2, n-2), dtype=float)
    diagMid = np.diagflat(np.ones_like(A[:, 0]))
    diagUP = np.diagflat(np.ones_like(A[1:, 0]), k=1)
    diagDOWN = np.diagflat(np.ones_like(A[1:, 0]), k=-1)
    A += 2*diagMid*(xs[2:, None]-xs[:-2, None])
    A += diagUP*(xs[2:, None]-xs[1:-1, None])
    A += diagDOWN*(xs[1:-1, None]-xs[:-2, None])
    cs = np.linalg.solve(A, bs)
    cs = np.insert(cs, 0, 0)
    cs = np.insert(cs, n-1, 0)
    k = 1
    y = np.zeros_like(xInterp)
    for i in range(len(xInterp)):
        if xInterp[i] > xs[k]:
            k += 1

        y[i] = ys[k-1]*((xs[k]-xInterp[i])/(xs[k]-xs[k-1]))+ys[k]*((xInterp[i]-xs[k-1])/(xs[k]-xs[k-1]))+cs[k-1]/6*((xs[k]-xInterp[i])*(xs[k]-xs[k-1])-(xs[k]-xInterp[i])**3/(xs[k]-xs[k-1]))-cs[k]/6*((xInterp[i]-xs[k-1])*(xs[k]-xs[k-1])-(xInterp[i]-xs[k-1])**3/(xs[k]-xs[k-1]))

    return y


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


def trigonometricInterpolation(f, n, nInterp):
    m = (n-1)/2
    ks = np.linspace(0, m, int(m)+1, endpoint=True)
    aks = np.zeros_like(ks)
    bks = np.zeros_like(ks)
    xj = 2*np.pi/n*np.linspace(0, n, num=n, endpoint=False)
    yj = f(xj)
    xInterp = 2*np.pi/nInterp*np.linspace(0, nInterp, nInterp)
    for i in range(0, int(m)+1):
        aks[i] = 1/m*np.sum(np.cos(ks[i]*xj)*yj)
        bks[i] = 1/m*np.sum(np.sin(ks[i]*xj)*yj)
    a0 = aks[0]
    aks = aks[1:]
    bks = bks[1:]
    ks = ks[1:]

    yInterp = np.zeros_like(xInterp)
    for i in range(len(xInterp)):
        yInterp[i] = 1/2*a0+np.sum(aks*np.cos(ks*xInterp[i])+bks*np.sin(ks*xInterp[i]))
    return xInterp, yInterp, xj, yj


def f(x): return np.e**(np.sin(2*x))


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
