# %%
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from math import e
from mpl_toolkits.mplot3d import Axes3D
# %% Q1(a)


def f(x):
    return 1/(x-3)


def bisection(f, x0, x1, toll):
    max = 2000
    x2 = (x0 + x1)/2
    for i in range(max):

        if (f(x0)*f(x2) < 0):
            x1 = x2
        else:
            x0 = x2
        xnew = (x0 + x1)/2
        xdiff = abs(xnew - x2)
        if abs(xdiff/xnew) < toll:
            break
        else:
            x2 = xnew

    return x2


print("Bisection for 1/(x-3):", bisection(f, 0, 5, 10**-8))
print("This answer is incorrect since it has no root")
# %% Q1(b)

# function taken from geeks for geeks to insert the superscript of i in brackets


def get_super(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans(''.join(normal), ''.join(super_s))
    return x.translate(res)


def newtowns(f, fp, x0, toll, a=0, b=0, show=False):
    max = 2000
    xinit = x0
    for i in range(max):

        x = x0-f(x0)/fp(x0)
        if show:
            xs = np.linspace(x0, x, 100)
            ys = fp(x0)*(xs-x0)+f(x0)
            plt.plot(xs, ys, 'r--')
            ys = np.linspace(f(x0), 0, 100)
            xs = np.ones(100)*x0
            plt.plot(xs, ys, 'k--')
            s = "("+str(i)+")"
            plt.text(x0, 0, "x"+get_super(s))
        if abs(x-x0) < toll:
            break
        else:
            x0 = x
    if show:
        xs = np.linspace(xinit, x, 100)
        ys = [f(x) for x in xs]
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.plot(xs, ys, 'b')
        plt.show()
    return x


x = sp.Symbol('x')
g = e**(x-x**0.5)-x
gp = g.diff(x)
glambda = sp.lambdify(x, g)
gplambda = sp.lambdify(x, gp)
print("Bisection for exp(x-sqrt(x))-x:", bisection(glambda, 0.5, 1.4, 10**-8))
print("Newtowns for exp(x-sqrt(x))-x:", newtowns(glambda, gplambda, 0.01, 10**-8, show=True))
# %% Q1(c)
a = 1
g = g/(x-a)
gp = g.diff(x)
glambda = sp.lambdify(x, g)
gplambda = sp.lambdify(x, gp)
print("Newtowns for exp(x-sqrt(x))-x, second root (guess 0.1):", newtowns(glambda, gplambda, 0.1, 10**-8))
print("Newtowns for exp(x-sqrt(x))-x, second root (guess 2):", newtowns(glambda, gplambda, 2, 10**-8))
print("Newtowns for exp(x-sqrt(x))-x, second root (guess 4):", newtowns(glambda, gplambda, 4, 10**-8))
print("Newtowns for exp(x-sqrt(x))-x, second root (guess 0.5):", newtowns(glambda, gplambda, 0.5, 10**-8))

# %% Q1(d)
p = 0.8
h = sp.Symbol('p')


def f(h):
    p = 0.8
    return 1/3*np.pi*(3*h**2-h**3) - (4/3)*np.pi*p


h = bisection(f, 0, 2, 10**-8)
print("Submerged depth", h)
(xplane, yplane) = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
zplane = np.zeros((100, 100))

u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:100j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)-h+1


fig = plt.figure()

ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.plot_surface(xplane, yplane, zplane, color='r')
ax.plot_wireframe(x, y, z, cmap='viridis')
ax.view_init(10, 45)
plt.show()
# %% Q2(a)


def fs1(xs):
    fs = np.zeros(2)
    fs[0] = xs[0]**2 - 2*xs[0]+xs[1]**4 - 2*xs[1]**2+xs[1]
    fs[1] = xs[0]**2 + xs[0]+2*xs[1]**3 - 2*xs[1]**2-1.5*xs[1]-0.05
    return fs


def fs2(xs):
    fs = np.zeros(3)
    fs[0] = 2*xs[0]-xs[1]*np.cos(xs[2])-3
    fs[1] = xs[0]**2-25*(xs[1]-2)**2+np.sin(xs[2])-np.pi/10
    fs[2] = 7*xs[0]*np.exp(xs[1])-17*xs[2]+8*np.pi
    return fs
# %% Q2(b)


def jacobian(f, xs, h=10**-4):
    xsPartial = np.array([xs]*len(xs))
    xsPartial += np.diag(np.ones_like(xs)*h)
    fsPartial = np.zeros_like(xsPartial)
    for i in range(len(xs)):
        fsPartial[:, i] = (f(xsPartial[i, :])-f(xs))/h
    return fsPartial, f(xs)

# %% Q2(c)


def newtonsN(f, xs, toll=10**-8):
    max = 200
    for i in range(max):
        js, fs = jacobian(f, xs)
        xsk = + np.linalg.solve(js, -fs)
        xs = xs + xsk
        if (abs(np.max(xsk)) < toll):
            break
    return xs


xs = np.array([1, 1], dtype=float)
ans = newtonsN(fs1, xs)
print("set 1 root", ans)
print("set 1 at root", fs1(ans))
xs = np.array([1, 1, 1], dtype=float)
ans = newtonsN(fs2, xs)
print("set 2 root", ans)
print("set 2 at root", fs2(ans))

# %% Q2(d)


def jacobianAnalytical(f, xs, xsubs):
    jacobain = np.zeros((len(xs), len(xs)))
    for i in range(len(xs)):
        for j in range(len(xs)):
            jacobain[i, j] = sp.diff(f[i], xs[j]).subs(xsubs)
    return jacobain


x0, x1, x2 = sp.symbols('x0 x1 x2')
f0 = 2*x0-x1*sp.cos(x2)-3
f1 = x0**2-25*(x1-2)**2+sp.sin(x2)-np.pi/10
f2 = 7*x0*sp.exp(x1)-17*x2+8*np.pi
set1 = [f0, f1, f2]
subs = {x0: 1, x1: 1, x2: 1}
xis = [x0, x1, x2]
numericalbad = jacobian(fs2, xs, h=2*10**-2)[0]
analyticalsolution = jacobianAnalytical(set1, xis, subs)
diffs = np.max((numericalbad - analyticalsolution)/analyticalsolution)*100
print("Analytical Jacobian for set 2\n", analyticalsolution)
print("Numerical Jacobian for set 2 (h=0.0001)\n", jacobian(fs2, xs)[0])
print("Maximum difference between analytical and numerical at h= 0.02:", diffs, "%")
