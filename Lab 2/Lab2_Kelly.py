import pandas as pd
import math as m
import numpy as np
import matplotlib.pyplot as plt
from mpmath import *
from scipy.integrate import dblquad
# %%%
# Q1(a)


def rectangle(f, a, b, n):
    h = (b - a) / (n)
    xs = np.linspace(a, b-h, n)
    return h * np.sum(f(xs))


# %%
# Q1(b)


def trapezoid(f, a, b, n):
    h = (b - a) / (n)
    xs = np.linspace(a, b, n+1)
    return h * (np.sum(f(xs)) - f(a)/2 - f(b)/2)


def simpsons(f, a, b, n):
    if n % 2 == 2:
        return "ERROR"
    h = (b - a) / (n-1)
    xs = np.linspace(a, b, n)
    cs = np.array([2+2*(x % 2) for x in range(n)])
    cs[0] = 1
    cs[-1] = 1
    return h/3 * np.sum(cs * f(xs))


# z is the z value i is the type of integration to be used, n is the number of steps

integrate = [rectangle, trapezoid, simpsons]


def erf(z, i, n):
    return 2/m.sqrt(m.pi)*integrate[i](np.vectorize(lambda x: m.exp(-x**2)), 0, z, n)


reference = m.erf(1)
errors = np.empty((2, 3))
for j in range(2):
    for i in range(3):
        errors[j][i] = abs((erf(1, i, 100+j)-reference)/reference*100)

print("Rectangle error at n = 100: ", errors[0][0], '%')
print("Trapezoid error at n = 100: ", errors[0][1], '%')
print("No simpson at 100 since it requires an odd n")
print("Rectangle error at n = 101: ", errors[1][0], '%')
print("Trapezoid error at n = 101: ", errors[1][1], '%')
print("Simpson error at n = 101: ", errors[1][2], '%')
# %%
# Q1(c)
# not sure if im calculating the relative error correctly


def adaptive_step(f, z):
    # simpson
    In = erf(z, 2, 3)
    i = 0
    n = 5
    while True:
        Inp = erf(z, 2, n)
        if (abs(Inp-In) > 10**-13):
            n = n*2 - 1
            In = Inp
            i += 1
        else:
            break
    print("Simpsons, n=", n, "iterations:", i)
    # trapezoid
    In = erf(z, 1, 3)
    i = 0
    n = 5
    while True:
        Inp = erf(z, 1, n)
        if (abs(Inp-In) > 10**-13):
            n = n*2 - 1
            In = Inp
            i += 1
        else:
            break
    print("Trapezoid, n=", n, "iterations:", i)


adaptive_step(erf, 1)

# %%
# Q2(a)
# insert file extension here
#InputArray = pd.read_csv(r"D:\Documents\2nd-year\ENPH-213\Lab 2\Hysteresis-Data.csv")
InputArray = pd.read_csv("Hysteresis-Data.csv")
InputArray = InputArray.to_numpy()
Vx = InputArray[:, 1]
Vy = InputArray[:, 2]
plt.plot(Vx, Vy)
plt.xlabel("Vx")
plt.ylabel("Vy")
plt.show()
# %%
# Q2(b)


def integral_2d(x, y):
    # taking area of a trapazoid with height delta_y and bases of length x_n and x_n+1
    area = (y[1:] - y[:-1])*(x[1:] + x[:-1])/2
    return np.sum(area)


print("Area in Hysteresis Curve =", integral_2d(Vx, Vy))
# %%
# Q3(a)


def simpsons2D(f, a, b, c, d, n, m):
    if n % 2 == 2:
        return "ERROR"
    h1 = (b - a) / (n-1)
    h2 = (d - c) / (m-1)
    # define xs and ys
    xs = np.linspace(a, b, n)
    ys = np.linspace(c, d, m)
    # creating the cs array
    csx = np.array([(2+2*(x % 2)) for x in range(n)])
    csx[0] = 1
    csx[-1] = 1
    csy = np.array([(2+2*(x % 2)) for x in range(m)])
    csy[0] = 1
    csy[-1] = 1
    cs = np.outer(h1/3*csx, h2/3*csy)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    return np.sum(cs * f(X, Y))

# %%
# Q3(b)


f2d = np.vectorize(lambda x, y: m.sqrt(x**2+y)*m.sin(x)*m.cos(y))
# n,m = 101,101
print("n,m = 101,101, A=", simpsons2D(f2d, 0, m.pi, 0, m.pi/2, 101, 101))
# n,m = 1001,1001
print("n,m = 1001,1001, A=", simpsons2D(f2d, 0, m.pi, 0, m.pi/2, 1001, 1001))
#n,m = 51,101
print("n,m = 51,101, A=", simpsons2D(f2d, 0, m.pi, 0, m.pi/2, 51, 101))

# %%
# Q3(c)


def f2d_nonvect(x, y): return m.sqrt(x**2+y)*m.sin(x)*m.cos(y)


print("Using Quad A =", quad(f2d_nonvect, [0, m.pi], [0, m.pi/2]))

# %%
# Q3(d)
# fucntion takes y first then x
print("Using dblquad A =", dblquad(f2d_nonvect, 0, m.pi/2, 0, m.pi)[0])
