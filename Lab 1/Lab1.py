import sympy as sm
import math as m
import numpy as np
import matplotlib.pyplot as plt
# %%
# Q1(a)
# defining x, f, and its derivatives
x = sm.Symbol('x')
f = sm.exp(sm.sin(x))
f_lambda = sm.lambdify(x, f)
fp = sm.diff(f, x)
fp_lambda = sm.lambdify(x, fp)
fpp = sm.diff(fp, x)
fpp_lambda = sm.lambdify(x, fpp)
fppp = sm.diff(fpp, x)
fppp_lambda = sm.lambdify(x, fppp)
# printing the function and its derivatives
print("f(x) =", f, "\nf'(x) =", fp, "\nf''(x) =", fpp, "\nf'''(x) =", fppp)
# declaring a linspace of x values from 0 to 2pi
xs = np.linspace(0, 2*m.pi, 200)
print(len(xs), xs[0], xs[len(xs)-1]-2*m.pi)
# mapping the functions to the x values
fs = [f_lambda(x) for x in xs]
fps = [fp_lambda(x) for x in xs]
fpps = [fpp_lambda(x) for x in xs]
fppps = [fppp_lambda(x) for x in xs]

# TO DO: Plots
# do I want to plot on the same graph or seprate graphs? if on the same graph what should the y label be
plt.rcParams.update({'font.size': 50})
lw = 2
fig = plt.figure(figsize=(8, 6))
ax = fig.add_axes([0, -4, 2*m.pi, 4])
plt.plot(xs, fs, 'b', linewidth=lw, label='f(x)')
plt.plot(xs, fps, 'r', linewidth=lw, label='f\'(x)')
plt.plot(xs, fpps, 'g', linewidth=lw, label='f\'\'(x)')
plt.plot(xs, fppps, 'y', linewidth=lw, label='f\'\'\'(x)')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend([r"$f(x)$", r"$f'(x)$", r"$f''(x)$", r"$f'''(x)$"], loc='upper left')
plt.savefig("f(x).pdf", dpi=1200, bbox_inches='tight')
plt.clf()
# %%
# Q1(b)


def calc_fd(f, x, h):
    return (f(x+h)-f(x))/(h)


def calc_cd(f, x, h):
    return (f(x+h/2)-f(x-h/2))/(h)


lw = 2
fig = plt.figure(figsize=(8, 6))
ax = fig.add_axes([0, -4, 2*m.pi, 4])
ax.set_xlabel("x")
ax.set_ylabel("f'(x)")
plt.plot(xs, fps, 'b', linewidth=lw, label='analytical')
# for h = 0.15
h = 0.15
fd_fs = [calc_fd(f_lambda, x, h) for x in xs]
cd_fs = [calc_cd(f_lambda, x, h) for x in xs]
plt.plot(xs, fd_fs, 'r', linewidth=lw, label='f-d h=0.15')
plt.plot(xs, cd_fs, 'g', linewidth=lw, label='c-d h=0.15')
# for h = 0.5
h = 0.5
fd_fs = [calc_fd(f_lambda, x, h) for x in xs]
cd_fs = [calc_cd(f_lambda, x, h) for x in xs]
plt.plot(xs, fd_fs, 'y', linewidth=lw, label='f-d h=0.5')
plt.plot(xs, cd_fs, 'm', linewidth=lw, label='c-d h=0.5')
ax.legend(["analytical", "f-d h=0.15", "c-d h=0.15",
          "f-d h=0.5", "c-d h=0.5"], loc='upper left')
plt.savefig("fd_comp.pdf", dpi=1200, bbox_inches='tight')
plt.clf()
# %%
# Q1(c)
hs = np.linspace(-16, -1, 16)
for i in range(len(hs)):
    hs[i] = 10**hs[i]
analytical_fp = fp_lambda(1)
# these errors are for the absolute errors
fd_error = [abs(calc_fd(f_lambda, 1, h) - analytical_fp) for h in hs]
cd_error = [abs(calc_cd(f_lambda, 1, h) - analytical_fp) for h in hs]
lw = 2
plt.rcParams.update({'font.size': 18})
fig = plt.figure(figsize=(8, 6))
ax = fig.add_axes([0, 0, 1, 1.3])
plt.plot(hs, fd_error, 'b', linewidth=lw, label="f-d error")
plt.plot(hs, cd_error, 'r', linewidth=lw, label="c-d error")
# TO DO: add plots for the error according to the lecture note formulas
ax.set_xlabel("h")
ax.set_ylabel("|abs. error|")
ax.set_yscale("log")
ax.set_xscale("log")
ax.legend(["f-d error", "c-d error"])
plt.savefig("error.pdf", dpi=1200, bbox_inches='tight')
plt.clf()
round_error = 2**-52
# finding analytical error approximations
fd_error_approx = [abs((h/2)*fpp_lambda(1)+2*f_lambda(1) * (round_error/h)) for h in hs]
cd_error_approx = [abs((h**2/24)*fppp_lambda(1)+2 * f_lambda(1)*(round_error/h)) for h in hs]
ax = fig.add_axes([0, 0, 1, 1.3])
plt.plot(hs, fd_error_approx, 'b', linewidth=lw, label="f-d error approx.")
plt.plot(hs, cd_error_approx, 'r', linewidth=lw, label="c-d error approx.")

ax.set_xlabel("h")
ax.set_ylabel("|abs. error|")
ax.set_yscale("log")
ax.set_xscale("log")
ax.legend(["f-d error approx.", "c-d error approx."])
plt.savefig("error_approx.pdf", dpi=1200, bbox_inches='tight')
plt.clf()
# %%
# Q1(d)
fdrich_error = [abs((2*calc_fd(f_lambda, 1, h/2) - calc_fd(f_lambda, 1, h))-analytical_fp)for h in hs]
cdrich_error = [abs((4*calc_cd(f_lambda, 1, h/2) - calc_cd(f_lambda, 1, h))/3-analytical_fp) for h in hs]
ax = fig.add_axes([0, 0, 1, 1.3])
plt.plot(hs, fd_error_approx, 'b', linewidth=lw, label="f-d error approx.")
plt.plot(hs, cd_error_approx, 'r', linewidth=lw, label="c-d error approx.")
plt.plot(hs, fdrich_error, 'g', linewidth=lw, label="f-d richardson error")
plt.plot(hs, cdrich_error, 'm', linewidth=lw, label="c-d richardson error")
ax.set_xlabel("h")
ax.set_ylabel("|abs. error|")
ax.set_yscale("log")
ax.set_xscale("log")
ax.legend(["f-d error approx.", "c-d error approx.", "f-d richardson error", "c-d richardson error"])
plt.savefig("RE_error_approx.pdf", dpi=1200, bbox_inches='tight')
# %%
# Q2(a)
# %%
# Q2(b)
