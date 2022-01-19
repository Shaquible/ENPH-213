# Code to compute Legendre Polynomial using Recursion (Bonnet's Recursion Relation)
# Last Modified: Jan 4 , 2022 
# for use with Lab 1 - Q2

import matplotlib.pyplot as plt

# basic function to compute Pn and Pn'
def legendre(n,x):
    if n==0: # P
        val2 = 1. # P0
        dval2 = 0. # P0'
    elif n==1: # derivatives
        val2 = x # P1'
        dval2 = 1. # P1'
    else:
        val0 = 1.; val1 = x # sep P0 and P2 to start recurrence relation
        for j in range(1,n):
            val2 = ((2*j+1)*x*val1 - j*val0)/(j+1) 
            # P_j+1=(2j+1)xP_j(x)-jP_j-1(x)  / (j+1), starts from P2
            val0, val1 = val1, val2
        dval2 = n*(val0-x*val1)/(1.-x**2) # derivative
    return val2, dval2

# Example graph to plot solution vs x
def plotlegendre(der,nsteps):
    plt.xlabel('$x$', fontsize=20)

    dertostr = {0: "$P_n(x)$", 1: "$P_n'(x)$"}
    plt.ylabel(dertostr[der], fontsize=20)
        
    ntomarker = {1: 'k-', 2: 'r--', 3: 'b-.', 4: 'g:', 5: 'c^'}
    xs = [i/nsteps for i in range (-nsteps+1,nsteps)]
    for n,marker in ntomarker.items():
       # print(der,)
        ys = [legendre(n,x)[der] for x in xs] # der = 0 (P), der = 1 (P')
        labstr = 'n={0}'.format(n) 
        # The format code {0} is replaced by the first argument of format()
        plt.plot(xs, ys, marker, label=labstr, linewidth=3)

    plt.ylim(-3*der-1, 3*der+1) # +/-1, or +/-4
    plt.legend(loc="lower right")
    plt.show()

# Need this as we will be re-suing this function with an import legendre legendre
# else it will also run this part op the code which is not intended.

""" - in more detail:
When a Python interpreter reads a Python file, it first sets a few special variables. 
Then it executes the code from the file.

So when the interpreter runs a module, the __name__ variable will be set 
as  __main__ if the module that is being run is the main program.

But if the code is importing the module from another module, then the __name__  
variable will be set to that module’s name.

One of those variables is called __name__.
The global variable, __name__, in the module that is the entry point to your 
program, is '__main__'. Otherwise, it's the name you import the module by.

So, code under the if block will only run if the module is the entry point 
to your program.

It allows the code in the module to be importable by other modules, without 
executing the code block beneath on import.

This line checks to see if we are running teh present file as the main program
(which we are here); thus, in this case, it is unncessary - however, this can be important
if we wish to call this function without runnign teh rest of the code - 
if we wanted to import this .py as a module to another code)
                                                                                                                                        
"""
if __name__ == '__main__':
    nsteps = 200 # number of x points
    # plot Pn uo to n = 5
    plotlegendre(0,nsteps)
    # derivative
    plotlegendre(1,nsteps)

# replace without this and see what happens when you import and run this function    
# nsteps = 200 # number of x points
#     # plot Pn uo to n = 5
# plotlegendre(0,nsteps)
#     # derivative
# plotlegendre(1,nsteps)
    
