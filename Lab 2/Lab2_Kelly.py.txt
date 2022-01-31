import pandas as pd
import math as m
import numpy as np
import matplotlib.pyplot as plt
from mpmath import *
from scipy.integrate import dblquad
# %%%
# Q1(a)


def rectangle(f, a, b, n):  # rectangle integration with n as the number of elements including the startpoint but no the endpoint
    h = (b - a) / (n)
    xs = np.linspace(a, b-h, n)
    return h * np.sum(f(xs))


# %%
# Q1(b)


def trapezoid(f, a, b, n):  # trapezoid method with n as the number of elements including the startpoint and endpoints
    h = (b - a) / (n)
    xs = np.linspace(a, b, n+1)
    # summing the whole array and then subtracting half the first and last element instead of summing the whole array but the first and last and adding half
    return h * (np.sum(f(xs)) - f(a)/2 - f(b)/2)


def simpsons(f, a, b, n):  # simpson's rule of 1/3 with n as the number of elements including the endpoint but not the startpoint
    if n % 2 == 0:
        return 0
    h = (b - a) / (n-1)
    xs = np.linspace(a, b, n)
    # generating the array for weights [1,4,2,......2,4,1]
    cs = np.array([2+2*(x % 2) for x in range(n)])
    cs[0] = 1
    cs[-1] = 1
    return h/3 * np.sum(cs * f(xs))


integrate = [rectangle, trapezoid, simpsons]


def erf(z, i, n):  # z is the z value, i is the integration function to be used (seen in the above array), and n is the number of steps
    # integrating a vectorized function of e^(-z^2) to make it compatible with the integration functions
    int = integrate[i](np.vectorize(lambda x: m.exp(-x**2)), 0, z, n)
    return (2/m.sqrt(m.pi))*int


# accepted value for comparison
reference = m.erf(1)
# generating an array to store the errors for each integration method, with each type of integration at the 2 n values
errors = np.empty((2, 3))
for j in range(2):
    for i in range(3):
        errors[j][i] = abs((erf(1, i, 100+j)-reference)/reference*100)

# error decreases as n increases
print("Rectangle error at n = 100: ", errors[0][0], '%')
print("Trapezoid error at n = 100: ", errors[0][1], '%')
print("No simpson at 100 since it requires an odd n")
print("Rectangle error at n = 101: ", errors[1][0], '%')
print("Trapezoid error at n = 101: ", errors[1][1], '%')
print("Simpson error at n = 101: ", errors[1][2], '%')
# %%
# Q1(c)


def adaptive_step(f, z):  # takes function and the value to evaluate at for comparison
    # simpson error
    # declaring the fist In to be used in the while loop and initial values for calculating In'
    In = erf(z, 2, 3)
    i = 0
    n = 5
    ref = m.erf(z)
    # calculating In' and then checking if the error is satisfactory at < 10^-13. if not then n is increased to 2n-1
    while True:
        Inp = erf(z, 2, n)
        if ((abs(Inp-In) / ref) > 10**-13):
            n = n*2 - 1
            In = Inp
            i += 1  # incrementing to the number of times that n has been increased from the initial
        else:
            break
    print("Simpsons, n=", n, "iterations:", i)
    # trapezoid error
    # same as simpson but with trapezoid instead of simpson
    In = erf(z, 1, 3)
    i = 0
    n = 5
    while True:
        Inp = erf(z, 1, n)
        if ((abs(Inp-In)/ref) > 10**-13):
            n = n*2 - 1
            In = Inp
            i += 1
        else:
            break
    print("Trapezoid, n=", n, "iterations:", i)


# calling function to calculate the number of steps to achieve error < 10^-13
adaptive_step(erf, 1)

# %%
# Q2(a)
# insert file extension here if the following line is not working
# InputArray = pd.read_csv(r"D:\Documents\2nd-year\ENPH-213\Lab 2\Hysteresis-Data.csv")
# reading the data into a dataframe and then converting to multiple lists for each of the columns
InputArray = pd.read_csv("Hysteresis-Data.csv")
InputArray = InputArray.to_numpy()
Vx = InputArray[:, 1]
Vy = InputArray[:, 2]
# plotting Vx vs Vy
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


# calling the function in a print statement to check the value
print("Area in Hysteresis Curve =", integral_2d(Vx, Vy))
# %%
# Q3(a)


def simpsons2D(f, a, b, c, d, n, m):
    if n % 2 == 2:
        return "ERROR"
    # setting up constants and linspaces needed for 1d simpsons for x and y
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
    # creating the weights array as the outer product of the csx and csy arrays
    cs = np.outer(h1/3*csx, h2/3*csy)
    # creating the array for the function input values
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    # summing the function output times the weights
    return np.sum(cs * f(X, Y))

# %%
# Q3(b)


# defining the function to be integrated
f2d = np.vectorize(lambda x, y: m.sqrt(x**2+y)*m.sin(x)*m.cos(y))
# n,m = 101,101
print("n,m = 101,101, A=", simpsons2D(f2d, 0, m.pi, 0, m.pi/2, 101, 101))
# n,m = 1001,1001
print("n,m = 1001,1001, A=", simpsons2D(f2d, 0, m.pi, 0, m.pi/2, 1001, 1001))
#n,m = 51,101
print("n,m = 51,101, A=", simpsons2D(f2d, 0, m.pi, 0, m.pi/2, 51, 101))

# %%
# Q3(c)


def f2d_nonvect(x, y): return m.sqrt(x**2+y)*m.sin(x)*m.cos(y)  # non vecotrized function to be integrated for compatibility with quad and dblquad


print("Using Quad A =", quad(f2d_nonvect, [0, m.pi], [0, m.pi/2]))

# %%
# Q3(d)
# function takes y first then x
print("Using dblquad A =", dblquad(f2d_nonvect, 0, m.pi/2, 0, m.pi)[0])
