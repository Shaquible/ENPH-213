# -*- coding: utf-8 -*-
"""
@author: shugh
Last Modified: Jan 4, 2022
Simple tutorial code on some introductory Python Coding
"""

# so let's get started with some basic Python coding - this is a simple one line comment

""" this is a multi-line comment:

With all codes, it is a good idea to write comments while you are writing or updating a 
program, else it is easy to forget your thought process later, and comments written later 
may be less useful. Also this is good for others to read your code later on, as you may 
end up resusing bits of codes and modules multipel times, and code sharing.

You should generally aim to write good code: you will then need few comments. Instead 
of documenting bad code you should replace it with good code. Of course, it takes some 
experience to know what constitutes “good code”.
"""


#%% this starts a new section - which keeps the code modularized 
#ctl+return (or shft+return) will run the cell (or right click run cell) 
x = 2+3
print('x =',x)
print("------------")
"""
You can also do the same thing interactively on the Console "Interactive Python"
e.g., by typing the same thing in the Console (RHS of Spyder)
"""

#%% Example names 
    
#Integer:
ival = 17
print('ival is an', type(ival))
#Float: 
therm = 12.6
#Complex: 
z = 1.2 + 2.3j
#Boolean
myflag = True # (The other option is False).
#String: 
phrase = "Hello World" # (Note that we could have used single quotes instead, ‘My string’).
print(ival,',', therm,',', z,',', myflag,',', phrase)
print("------------")

# Some naming rules
"""
Variable names are made up of letters or numbers or the character _ (an underscore). 
Thus, how_do_you_do is an allowed variable name, but not.bad is not.

Variable names cannot start with a number, so se7en is an allowed variable name, but 122ven is not

Variable names are case sensitive, meaning that x and X are different variables, so be careful 
"""

#%% Modules and built-in functions  

# Python has many useful-built in functions, here are some examples:
    
a = abs(2 + 3j)
print(a)

# or import functions from the math module

from math import sqrt
from math import cos
from math import pi
print(sqrt(2**2 + 3**2))
print(cos(pi))
print("------------")

# could also do (better), and m. will show all the function options within Spyder
import math as m
print(m.sqrt(2**2 + 3**2))
print(m.cos(pi))
y = m.pow(5,2) # note this is the same thing as 5**2 (5^2 in Matlab)
print('y (5^2) is', y)
print("------------")

#%% We can import external functions and subroutines like this
import SimpleFunction as myfunc # comes from SimplyFunction.py
y_squared = myfunc.squared(5)
print(y_squared)
print(type(y_squared)) # just checking the type - float or integer (see function)?
print("------------")

"""
Here we first import a Python code (a module), e.g., if we saved a function inside another code
called code1a.py, with a function to compute x**2:

def squared(x):
    y = x**2
    return y

Note the indentations needed for Python - Spyder will indent these for you
"""

#%% Example function with multiple inputs and multiple outputs (returns)
#from math import cos, sin

def cartesian(r, theta):
    x = r*m.cos(theta)
    y = r*m.sin(theta)
    return (x, y)

a = cartesian(1.2, 0.1)
print(a)
x, y = cartesian(1.2, 0.1)
print(x, y)
print("------------")

#%% A lambda function
"""
A lambda function is a small anonymous functiion, which can take
any number of arguments, but can only have one expression.
"""

x = lambda a, n : a * n
print('a*n with 2*10 is', x(2, 10))

# or more powerful is to declare another function based on the value of a fixed n 
def myfunc(n):
  return lambda a : a * n

mydoubler = myfunc(2) # passing "n"
mytripler = myfunc(3) # same function, differnet instance

print(mydoubler(11)) # 2 times 11 - passing "a" now
print(mytripler(11))  # 3 times 11
print("------------")

def myQuad(a,b,c):
  return lambda x : a*x*2+b*x+c

# specific instance with a certain a,b,c
myQuad258 = myQuad(2,5,8)
myQuad122 = myQuad(1,2,2)

print(myQuad258(1)) # evaluates Q256 at x=1, passing "x" here
print(myQuad122(1)) # evaluates Q122 at x=1
print("------------")

# Jumping out of a function if something is not satisfied or will give an error
# this is one way of doing this
import sys
def myFunky(a,b):
   if b==0.:
       print('we have a problem with myFunky - exiting cleanly')
       sys.exit()
   else:   
       c = a/b
   return c

print('myFunky', myFunky(1.,1.))
# this will stop/exit the code - uncomment if you wnat to check
#c=myFunky(1.,0.)

#%% Some scientific constants

from scipy import constants as con # from scipy (https://docs.scipy.org/doc/scipy/reference/) 
print(con.epsilon_0)
print(con.c)
print(m.pi) # this is using the math library, so now we see why the "as m" is useful
print(con.pi)
print("------------")

#%% Variable type examples

"""
Python is a dynamically typed language, meaning that runtime objects (like the variable x above) 
get their type from their value. As a result, you can use the same variable to hold different 
types in the same interactive session (or in the same program). For example:
"""
x = 1
print(type(x)) # int
x = 3.5
print(type(x)) # float
x = "Hi"
print(type(x)) # string
x = 1
y = float(x)
print(type(y)) # converted to real
y = 1+1j*2
print(type(y)) # complex
x = 3.9
y = int(x) # int rounds down in Python, so 3.9 becomes 3
print(y)
print("------------")

#%% Simple Input 

# commenting out for now, as this stops the code from running and gets annoying 
#x = input("Enter an integer:  ")
#print("Twice that is:", 2*int(x))

"""
In Python 3, no matter what you input, x is saved as a string, so here we converted
to an int first.
There are better ways to do this, but it is not to practical to input from the screen anyway
"""

#%% Simple Arithmetic

x = 1
y = 2
print(x+y) # plus
print(x/y) # divide (converts to a float)
print(x**y) # to the power
print(x%y) # modulo - e.g/, 14%3 is 2. A useful application of this is that x%2 checks 
# if x is even or odd: if x is even, the result is 0,if x is odd the result is 1.
print("------------")

#%% Assigments

x = 7
x = x + 1 # x is now 8
x += 1 # x is now 9, same as x = x + 1
x, y, z = 1, 3.4, "Hello" # multiple assigments on one line
x, y = y, x # swap variables
x = 7
print(x)
z = x # same value and same id - returns a unique id for the specified object.
print(z)
print(x)
print(id(z))
print(id(x))
# if we change z, then it has a differnet id
print("------------")

# but watch when assigning arrays as equivalent 
x = [1, 2, 3]
y = x # just copies a reference to the original list
print(x)
print(y)
# these are equivelent, so if we change one, they bloth change
x.append(4)
print('y is now', y)
print(x)
print('id of y', id(y))
print('id of x', id(x)) # these are identical, and x == y !
print("------------")

# compare to this copy command - the id is now different
x = [1, 2, 3]
y = x.copy() # creates a new list
print(x)
print(y)
x.append(4)
print('now y is unchanged', y)
print(x)
print("------------")

# or you can also slice it
x = [1, 2, 3]
y = x[:] # slice from start to end or array
print(x)
print(y)
x.append(4)
print('again y is unchanged', y)
print(x)
print("------------")


#%% Simple for-loops and arrays

# useful for clearing out the variables from before
from IPython import get_ipython
get_ipython().magic('reset -sf')

#i loops from 1 to 10 here, python arrays start from [0]
powers = [0]*10; x=[0]*10
for i in range(1, 11):
    print(i)
    powers[i-1]=2**i
    x[i-1]=i
print(powers) # printing after loop has finished
print(x)
print("------------")

# or this is easier, and we do not need to know the size of arrays yet
get_ipython().magic('reset -sf')
powers = []; x=[]
for i in range(1, 11): # goes from 1 to 10 (can use any value now)
#for i in range(0, 10): # goes from 0 to 9
      print(i)
      powers.append(2**i)
      x.append(i)
print(powers)
print(x)
print("------------")

# we can look backwards with "reversed"
for i in reversed(range(len(x))): print(x[i])# i goes from 10 to 1 here
# you can check the data with the Variable Explorer on the right Spyder panel, and 
# also plot the data - right click on powers, then plot


#%% Simple graph, we will use matplotlib in this course

import matplotlib.pyplot as plt

#from matplotlib import rcParams - always a good idea to have minimum size fonts (defaults are to small)
plt.rcParams.update({'font.size': 18})

fig = plt.figure(figsize=(8,6))
lw=2
ax = fig.add_axes([.2, .6, .6, .3])
plt.plot(x, powers, 'b',linewidth=lw)
plt.plot(x, powers, 'bo',markersize=8) # or symbols
ax.set_xlim(0,10)
ax.set_ylabel(r'$powers$') # using math fonts and can use LaTeX style as well, e.g. \omega
# simple legend
ax.legend(['lines', 'markers'],loc='upper left') # best is the default


# could also call this one ax2 here
ax2 = fig.add_axes([.2, .2, .6, .3])
plt.semilogy(x, powers, 'r',linewidth=lw)
plt.semilogy(x, powers, 'ro',markersize=8)
ax2.set_xlim(0,10)
#ax.set_ylim(0,1100)
ax2.set_xlabel(r'$x$')
ax2.set_ylabel(r'$powers$')

# if you want to save a pdf of the figure
plt.savefig('graph1_example.pdf', dpi=1200, bbox_inches="tight")


#%% Another graph example using subplot (has less control, so we also use tight_layout, which helps)

plt.rcParams.update({'font.size': 10})
fig = plt.figure()
# with a handle
ax = fig.add_subplot(231) # grid of 2 rows and 3 colimns, graph 1
plt.plot(x, powers, 'b',linewidth=lw)
plt.plot(x, powers, 'o',linewidth=lw) 
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$powers$')

ax2=fig.add_subplot(232)
plt.semilogy(x, powers, 'r',linewidth=lw)
plt.semilogy(x, powers, 'o',linewidth=lw)
ax2.set_xlabel(r'$x$')
#ax2.set_ylabel(r'$powers$')

plt.tight_layout() # helps subplot layout
plt.show() # probably do not need under Spyder
#%% Debugger

"""
let's test the debugger - go to Debug Cell then step (or shift alt + enter, after clicking this cell, then
step through with ctl F10, note the printer value and also how the variables change within variable explorer).
Ctl F12 will continue through entire loop
    You can also type variables at the debugger prompt, e.g., x, and the array values will be listed.
"""

# clear out the variables
from IPython import get_ipython
get_ipython().magic('reset -f')
#get_ipython().magic('reset -sf')

# i goes from 1 to 10, python arrays start from [0]
powers = [0]*10; x=[0]*10
for i in range(1, 11):
    print(i)
    powers[i-1]=2**i
    x[i-1]=i
    
print(powers)    
print("------------")

"""
You can access and edit local and global variables at each breakpoint through the Variable Explorer. Left
click on the number at the left will set and delete a breakpoint (red dot)
Or debug run will run to each breakpoint, step to each point with ctl F12 (or double blue arrow above >>)

exit on IPdb prompt will come out, or just use Stop from debug or ctrl + shift + F12
"""
#%% Rounding fractions

import math as m
# integer
int(9/10) # rounds down to nearest integer, so this is 1
m.ceil(1/10) # round up
m.floor(1/10) # rounds down
# float
19/20


x=[0]*10
for i in range(0, len(x)):
    x[i]=i+0.1
print(x)

v = [m.floor(float(x)) for x in range(len(x))]  # last bit is same as range(0,len(x))
print(v)
print("------------")

# this loop is a bit more readable and transparant

for i in range(0, len(x)): 
    v[i] =  m.floor(float(x[i]))
print(v)
print("------------")
    

#%% Finding the largest value in an array (and its index)

#Initialize array     
arr = [25, 11, 7, 75, 56];     
     
#Initialize max with first element of array.    
max = arr[0];    
     
#Loop through the array    
for i in range(0, len(arr)):    
    #Compare elements of array with max    
   if(arr[i] > max):    
       max = arr[i];    
           
print("Largest element present in given array: " + str(max));   

# another example
numbers = [55, 4, 92, 1, 104, 64, 73, 99, 20]

max_value = None
max_idx = None

for idx, num in enumerate(numbers): # enumerate returns index and value
    print(idx,num)
    if (max_value is None or num > max_value):
        max_value = num
        max_idx = idx

print('Maximum value:', max_value, "At index: ", max_idx)
print("-------------------")

"""
However, NumPy can do this in a much cleaner way!
"""

#%% Default paramaters in a function call 

def defpar(n=2):
    val = 2**n
    return val

# first two are the same and third one changes the value of n from a default value
print(defpar())
print(defpar(2))
print(defpar(4)) # changes teh default paramater to 4

# or more generally

def cosder(x, h=0.01):
    return (m.cos(x+h) - m.cos(x))/h

print(cosder(0.))
print(cosder(0.,0.01))
print(cosder(0.,0.05))
print("------------")

#%% local and global variables 

def f(a):
    global c # if you want to modify a global variable in a function

# local unless stated otherwise
    a += 1
    b = 42
    c += 1 
    print('inside, a,b,c =', a, b, c)

# global variables (outside)
a = 1
b = 2
c = 10 # this one we want to change inside teh function for some reason
print('outside, a,b,c =', a, b, c)

f(7)
print('outside, a,b,c =', a, b, c) # c changes inside and outside, a changes inside (passed)

f(6)
print('outside, a,b,c =', a, b, c)

# the value outside is not changing, unless declared inside as a global variable

# however, it is generally not a good idea to change global variables inside a function

#%% Simple example of writing to a file and reading from a file

"""
Reading and writing data and variables is much easier with numpy which we will
use later, so here is just a simple example of writing and reading a dictionary
to and from a file (which contains some variables, in this case two variables a and b)
"""

# create a simple dictionary with two variables

a = 17
b = 9
# to test, call these a2 and b2
a_dictionary = {"a2" : a, "b2" : b}
# The repr() function returns the string representation of the value passed to eval function 
str_dictionary = repr(a_dictionary)

# open a file called example1.txt and write to it
file = open("example1.txt", "w")

# write the dictionary and start a new line in case you write something else later
file.write(str_dictionary + "\n")
# close the file
file.close()

    
# read in the file, convert the string back to a dictionary item and check some contents
file = open("example1.txt", "r")
import ast # this will use this to evaluate and convert contents back to a dictionary
contents = file.read()
aa = ast.literal_eval(contents)
file.close()
print(aa)
print(type(aa))
x = aa.get("a2")
print('a2 from aa dict=', x) # hopefully this is 17!
print("------------")


#%% Numpy example 

# numpy way - we will explore this in the next numpy tutorial (NumpyIntro), but listing here for now to compare
import numpy as np
np.savez('example1_test1', a1=a, b1=b)
#savez can save several arrays into a single file in uncompressed .npz format (zipped archive of files)
outfile='example1_test1.npz' # not readable (but who cares, like a .mat in Matlab!)

# load back in and check
z = np.load(outfile)
z.files
bb=z['a1']
delta=z['b1']
print(delta) # should be 9
print(bb) # should be 17
print("------------")

# can also create a readable txt file
# if using txt, needs 1d or 2d array, e.g., so we could do this
aa=(a,b)
#np.savetxt('example1_test.txt',aa, delimiter=', ', fmt='%1.4e')
np.savetxt('example1_test1.txt', aa, delimiter=', ', fmt='%10i')
X = open("example1_test1.txt", 'r')  #  open file in read mode
print("the file contains is:")
print(X.read())   # open file

print("-----------------------------")
print("Code is now complete. Goodbye!")
print("-----------------------------")

# but using savez is nicer








