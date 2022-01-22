import sympy as sm
import math as m
import numpy as np
import matplotlib.pyplot as plt
from legendre import legendre
# if you want to save figures to a file, set printPlot to true
printPlot = False
# %%
# Q1(a)
# defining x, f, and its derivatives as an array of functions for the normal and lambda functions
x = sm.Symbol('x')
f = [0]*4
f[0] = sm.exp(sm.sin(x))
for i in range(1, 4):
    f[i] = sm.diff(f[i-1], x)
f_lambda = [sm.lambdify(x, f[i]) for i in range(4)]
# printing the function and its derivatives
for i in range(4):
    print("f", "'"*i, "(x)=", f[i], sep='')
# declaring a linspace of x values from 0 to 2pi
xs = np.linspace(0, 2*m.pi, 200)
print(len(xs), xs[0], xs[len(xs)-1]-2*m.pi)
# mapping the functions to the x values as a 2d array for the function and its x values
fs = np.empty((4, len(xs)))
for i in range(4):
    fs[i] = [f_lambda[i](x) for x in xs]
lw = 1.5
plt.rcParams.update({'font.size': 10})
plt.figure(1)
for i in range(4):
    label = "f"+"'"*i+"(x)"
    plt.plot(xs, fs[i], label=label, linewidth=lw)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
if printPlot:

    plt.savefig("Q1(a).pdf", dpi=1200, bbox_inches='tight')
# %%
# Q1(b)


def calc_fd(f, x, h):
    return (f(x+h)-f(x))/(h)


def calc_cd(f, x, h):
    return (f(x+h/2)-f(x-h/2))/(h)


plt.figure(2)
plt.xlabel("x")
plt.ylabel("f'(x)")
plt.plot(xs, fs[1], 'b', linewidth=lw, label='analytical')
# for h = 0.15
h = 0.15
fd_fs = [calc_fd(f_lambda[0], x, h) for x in xs]
cd_fs = [calc_cd(f_lambda[0], x, h) for x in xs]
plt.plot(xs, fd_fs, 'r', linewidth=lw, label='f-d h=0.15')
plt.plot(xs, cd_fs, 'g', linewidth=lw, label='c-d h=0.15')
# for h = 0.5
h = 0.5
fd_fs = [calc_fd(f_lambda[0], x, h) for x in xs]
cd_fs = [calc_cd(f_lambda[0], x, h) for x in xs]
plt.plot(xs, fd_fs, 'y', linewidth=lw, label='f-d h=0.5')
plt.plot(xs, cd_fs, 'm', linewidth=lw, label='c-d h=0.5')
plt.legend()
if printPlot:
    plt.savefig("Q1(b).pdf", dpi=1200, bbox_inches='tight')
# %%
# Q1(c)
plt.figure(3)
# creating a linspace of h values by raising ten to the power of the original array
hs = np.linspace(-16, -1, 16)
hs = 10**hs
analytical_fp = f_lambda[1](1)
# machine precision gotten from class notes
round_error = 2**-52
# these errors are for the absolute errors
fd_error = [abs(calc_fd(f_lambda[0], 1, h) - analytical_fp) for h in hs]
cd_error = [abs(calc_cd(f_lambda[0], 1, h) - analytical_fp) for h in hs]
plt.plot(hs, fd_error, 'b', linewidth=lw, label="f-d error")
plt.plot(hs, cd_error, 'r', linewidth=lw, label="c-d error")
# finding analytical error approximations
fd_error_approx = [abs((h/2)*f_lambda[2](1)+2*f_lambda[0](1) * (round_error/h)) for h in hs]
cd_error_approx = [abs((h**2/24)*f_lambda[3](1)+2 * f_lambda[0](1)*(round_error/h)) for h in hs]
# plotting
plt.plot(hs, fd_error_approx, 'g', linewidth=lw, label="f-d error approx.")
plt.plot(hs, cd_error_approx, 'c', linewidth=lw, label="c-d error approx.")
plt.xlabel("h")
plt.ylabel("|abs. error|")
plt.yscale("log")
plt.xscale("log")
plt.legend()
if printPlot:
    plt.savefig("Q1(c).pdf", dpi=1200, bbox_inches='tight')
# %%
# Q1(d)
# calculating  the richardson errors
plt.figure(4)
fdrich_error = [abs((2*calc_fd(f_lambda[0], 1, h/2) - calc_fd(f_lambda[0], 1, h))-analytical_fp)for h in hs]
cdrich_error = [abs((4*calc_cd(f_lambda[0], 1, h/2) - calc_cd(f_lambda[0], 1, h))/3-analytical_fp) for h in hs]
# plotting the two sets of errors against each other
plt.plot(hs, fd_error_approx, 'b', linewidth=lw, label="f-d error approx.")
plt.plot(hs, cd_error_approx, 'r', linewidth=lw, label="c-d error approx.")
plt.plot(hs, fdrich_error, 'g', linewidth=lw, label="f-d richardson error")
plt.plot(hs, cdrich_error, 'm', linewidth=lw, label="c-d richardson error")
plt.xlabel("h")
plt.ylabel("|abs. error|")
plt.yscale("log")
plt.xscale("log")
plt.legend()
if printPlot:
    plt.savefig("Q1(d).pdf", dpi=1200, bbox_inches='tight')

# %%
# Q2(a)
# definitions for the first 4 derivatives of a function using cd


def calc_cd_1(f, n, x, h):
    cd = (f(n, x+h/2) - f(n, x-h/2))/h
    return cd


def calc_cd_2(f, n, x, h):
    cd = (calc_cd_1(f, n, x+h/2, h) - calc_cd_1(f, n, x-h/2, h))/h
    return cd


def calc_cd_3(f, n, x, h):
    cd = (calc_cd_2(f, n, x+h/2, h) - calc_cd_2(f, n, x-h/2, h))/h
    return cd


def calc_cd_4(f, n, x, h):
    cd = (calc_cd_3(f, n, x+h/2, h) - calc_cd_3(f, n, x-h/2, h))/h
    return cd

# legendre polynomail calculation function


def LP(x, n, h):
    # solving for the nth derivitive of f
    # if statements used to call the correct derivative function
    def f(n, x): return (x**2-1)**n
    if n == 1:
        diff = calc_cd_1(f, n, x, h)
    elif n == 2:
        diff = calc_cd_2(f, n, x, h)
    elif n == 3:
        diff = calc_cd_3(f, n, x, h)
    elif n == 4:
        diff = calc_cd_4(f, n, x, h)
    Pn = 1/((2**n)*m.factorial(n))*diff
    return Pn


# calculating and plotting the legendre polynomials
h = 0.01
# LPS is calculated with the function defined in this file, lps is calculated with the provided function
xs = np.linspace(-1, 1, 200)
lps = []
LPS = []
for i in range(4):
    lps.append([LP(x, i+1, h) for x in xs])
    LPS.append([legendre(i+1, x)[0] for x in xs])
# plotting the legendre polynomials
plt.rcParams.update({'font.size': 6})
fig1, ax = plt.subplots(4,  sharex=True)
for i in range(4):
    ax[i].plot(xs, lps[i], '-', c='b',  linewidth=lw, label="Rodrigues")
    ax[i].plot(xs, LPS[i], '--', c='r',  linewidth=lw, label="Reference")
    ax[i].set_xlabel("x")
    ax[i].set_ylabel("P"+str(i+1))
    ax[i].legend()
if printPlot:
    plt.savefig("Q2(a).pdf", dpi=1200, bbox_inches='tight')
# %%
# Q2(b)

# recursive cd function accepts the n as the original n from the expression and N as the order of the derivative to be calculated


def cd_n_recursive(f, n, N, x, h):
    if N == 1:
        return (f(n, x+h/2) - f(n, x-h/2))/h
    return (cd_n_recursive(f, n, N-1, x+h/2, h) - cd_n_recursive(f, n, N-1, x-h/2, h))/h

# save function as before but ifs for the function calls is replaced with the better function


def LP_improved(x, n, h):
    # solving for the nth derivitive of f
    def f(n, x): return (x**2-1)**n
    diff = cd_n_recursive(f, n, n, x, h)
    Pn = 1/((2**n)*m.factorial(n))*diff
    return Pn


h = 0.01
# empty lists for the legendre polynomials
lps.clear()
LPS.clear()
# filling the arrays with the first 8 LPs for the Rodrigues and the reference
for i in range(8):
    lps.append([LP_improved(x, i+1, h) for x in xs])
    LPS.append([legendre(i+1, x)[0] for x in xs])
# plotting the legendre polynomials (using a nested for loop to plot them in a 4*2 grid)
fig1, ax = plt.subplots(4, 2,  sharex=True)
for j in range(2):
    for i in range(4):
        ax[i][j].plot(xs, lps[i+j*4], '-', c='b',  linewidth=lw, label="Rodrigues")
        ax[i][j].plot(xs, LPS[i+j*4], '--', c='r',  linewidth=lw, label="Reference")
        ax[i][j].set_xlabel("x")
        ax[i][j].set_ylabel("P"+str(i+1+j*4))
        ax[i][j].legend()
if printPlot:
    plt.savefig("Q2(b).pdf", dpi=1200, bbox_inches='tight')
plt.show()
