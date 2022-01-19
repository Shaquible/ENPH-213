# Author: Stephen Hughes
# Last modified: Jan 4, 2022
# For Lab 1 - Q2 - input example showing how to use and call legendre.py

# %% part (a)

from legendre import legendre  # import the LP function that uses recurrence
import matplotlib.pyplot as plt


def plotcomp(nsteps):

    xs = [i/nsteps for i in range(-nsteps+1, nsteps)]
    for n in range(1, 5):  # this will compute P1 to P4
        ys = [legendre(n, x)[0] for x in xs]
        plt.plot(xs, ys, 'k-', label='recurrence n={0}'.format(n), linewidth=3)
        plt.xlabel('$x$', fontsize=20)
        plt.ylabel("$P_n(x)$", fontsize=20)
        plt.legend(loc="best")
        plt.show()  # create a new graph for each n


"""
So after the y label is where you coul add your solution to part (a) by calling a function
that computes the Rodrigues formula, e.g.,    
        ys = [rodrigues(n,x,h) for x in xs]
        plt.plot(xs, ys, 'r--', label='Rodrigues n={0}'.format(n), linewidth=3)
"""

nsteps = 200  # number of x points (200 should be enough)
plotcomp(nsteps)


# %% part (b) - this is where you will do with recursion - good luck!
