# -*- coding: utf-8 -*-
"""
@author: shugh
Last Modified: Jan 4, 2022
Simple tutorial code on some introductory NumPy (https://numpy.org/)
- "The fundamental package for scientific computing"
"""

#%% Simple 1d arrays

import numpy as np

# python list
L = [11, 7, 19, 22, 55]
# convert to a numpy array
xs = np.array(L)
print('xs=', xs)
print('L=', L) # note this one has commas
print(xs.size) # total mnumber of elements
print(len(xs)) # no of elements in first dimension (same here)
print(xs.shape)
# xs has a fixed size, so cannot append like a list

# check its type - inferred from elements
print(xs.dtype) # int32

# quite different to lists, since numpy types directly map onto machine representations 
print(type(L))
print(type(L[0]))

xs = np.array([1., 2., 3., 4.])
print(xs.dtype) # float64
print("------------")

# we can also make the type explicit with the set up, e.g.
zs = np.array([5, 6, 7, 8], dtype=np.float32) # if use float == float64

# arrays of zeros or ones are often very useful
x = np.array([0.]*5)
print(x)
x = np.zeros(5) # float is teh default
print(x)
x = np.ones(5)
print(x)
x = np.arange(1.5,1.75,0.05)
print(x)  # note this stops at 1.7, as perfect precision is not used (change to 1.751 and it works better)
x = np.arange(1.5,1.751,0.05)
print(x) 
x = np.arange(1.5,1.8,0.05)
print(x)  # but this one is ok
# a better way might be to use linespace (depends on the application) - exact number of even elements
x = np.linspace(1.5,1.75,6)
print(x)
x = np.linspace(1.5,1.8,7) # 7 even elemnts, etc
print(x)
print("------------")


#%% Indexing and Slicing

# indexing is the same as lists

xs = np.arange(10,16)
print(xs)
xs[2]
# for fun, let's double every second x value (many ways to do this)
xs = np.array([x if i%2==0 else 2*x for i,x in enumerate(xs)])
print(xs)


# slicing is similar, but array slices are views (data is "shared") on the original array:
    
r = np.array([11,7,19,22])
print(r)
sli = r[1:3]
sli[0] = 55
print(sli)
print(r) # note the second element has changed
print("------------")

# if we want a true copy then can do this
r = np.array([11,7,19,22])
sli = np.copy(r[1:3]) # [start:end:step]
sli[0] = 55
print(sli)
print(r)
# or could just copy the first array of course (has no effect on firts one)
r2 = np.copy(r)
r2[1] = 0
print(r2)
print(r)

# other examples of slicing
print(xs)
print(xs[-1]) # last element
print(xs[1:5:2]) # first to fifth element in units  of 2, will not include last one
print(xs[1:5:1]) # first to fifth element in units of 2, will not include last one
print(xs[1:]) # first to last element (inclusive)
print(xs[:1]) # all to first 
print(xs[-3:-1]) # third last to last (will not include last)
print(xs[-3:]) # third last to last (includign last)
xs[:] = 10 # common example of setting all elemnts to a single value
print(xs)
xs = 7 # note this is entirely different and will change the xs array to a single integer!
print(xs)
print("------------")

#%% Vectorization

"""
Arrays allow us to avoid loops which can speed up the code significantly
We want to exploit vectorization where possible - some simple examples below:
"""

xs = np.arange(10,16)
ys = np.arange(20,26)
print(xs + ys) # does element by element
sum = np.sum(xs*ys) # can also do a scalar product like this
print(sum)

# this is not the same as 
xs = list(range(10,16))
ys = list(range(20,26))
print(xs+ys) # double size array!
# below would achieve the same sort of result (but not very nice)
news = [x+y for x,y in zip(xs,ys)] 
#The zip() function takes iterables (can be zero or more), aggregates them in a tuple, and return it
print(news)

# let's compare two different ways of using functions

# usual python way with elements of a list
import math
def f(x):
    return 1/math.sqrt(x**2+1)
ws = list(range(30,36))
xs = list(range(10,16))
tots = [w*f(x) for w,x in zip(ws,xs)]
print(tots)

# numpy way  - this "baby" is way nicer!
def fa(xs):
    return 1/np.sqrt(xs**2+1)
ws = np.arange(30,36)
xs = np.arange(10,16)
tots = ws*fa(xs)
print(tots)
print("------------")

#%% 2d arrays

# consider a list of lists
LL = [[11,12,13,14], [15,16,17,18], [19,20,21,22]]
print(LL)
nrow = len(LL)
ncol = len(LL[0])
print(nrow)
print(ncol)

# using numpy we can do this
A = np.array(LL)
print(A)
print('size of A', A.size)
print('shape of A', A.shape)
print('dimension of A', A.ndim)
print('length of A', len(A)) # first dim remember

# and like with 1d arrays, can specify a type:
B = np.array(LL,dtype=np.float64) 
print('B=', B)
C = np.zeros([3,3], dtype=complex)
C[:,:] = 1+ 1j*1
print('C=', C)
D = np.ones([3,3])
print('D=', D)
E = np.identity(3)
print('E=', E)
E = np.eye(3) # same thing
print('E=', E)
print("------------")

#%% Reshaping arrays
A = np.arange(11,23)
print(A)
print(A.shape)
A = A.reshape((3,4)) # indexing is 0,0 0,1 0,2 ... 1,0, 1,1 
print(A)
print('print first row of A',A[0,:])
print("------------")
print(A.shape)
h1 = np.array([[1,1,1,1]]) # double square brackets needed to keep dimensions correct with a 2d arrway
B = np.concatenate([A,h1]) # along zero is the default (extra row)
print(B)
# or can also do this
B = np.vstack([A,h1]) # stacks arrays in sequence vertically
# there is also a np.hstack function, that stacks array horizontally 
print(B)

#%%

print(A)
h2 = np.array([[1],[1],[1]])
C = np.concatenate([A,h2],1)
print(C)
print("------------")

A = A.reshape((12,1))
print(A)
print(A.shape)
A = A.reshape((1,12))
print(A)
print(A.shape)
B = np.transpose(A)
print(B)
print("------------")


#%% Indexing and slicing 2d arrays

LL = [[11,12,13,14], [15,16,17,18], [19,20,21,22]]

print(LL[2][1]) # not an easy way to index

A = np.arange(11,23).reshape(3,4)
print(A)
print(A[2,1])
print(A[1,1:3])
print(A[1,:]) # specific row
print(A[:,2]) # specific column
print("------------")

B = A[:2,1:]
print(B)
B[:,:]  = 0 # broadcast to all elements
print(B)

# remember this effects A
print(A)
# so if you do not want this, use copy
A = np.arange(11,23).reshape(3,4)
B = np.copy(A[:2,1:])
B[:,:]  = 0
print(A)

"""
Copy should be avoided as it uses more memroy and time, so as long
as you know what is going on, you can usually avoid having to do this.
However, sometimes we need ot use copy.
"""

A = np.arange(11,23)
print(A)
print(A.shape)
A = A.reshape((3,4)) # indexing is 0,0 0,1 0,2 ... 1,0, 1,1 
print(A)
print(A.shape)
print(A[1,1])
print("------------")
A = np.arange(11,23)
A = A.reshape((3, 4), order='F') # if you want the indexing: 0,0 1,0 2,0
#(means Fortran like ordering and should be avoided!)
print(A[1,1])
print("------------")


#%% Swapping rows or columns in an array
A=np.arange(0,5)

print(A)
A[3],A[2]=A[2],A[3]
print(A)
# 2d arrays are more tricky
A=np.ones([3,3])
A[0,1]= 2.; A[0,2]=3.
A[1,0]= 4.; A[2,0]=5.
A[1,1]= 9.; A[2,2]=20.
print(A)
A0 = A.copy() # save a copy to retest below
# if we swap arrays like this:
A[1,:], A[2,:] = A[2,:], A[1,:]    
print(A)
# it will not work as two rows are now identical!

print("------------") 
A=A0.copy()
# can do this
A[1,:], A[2,:] = A[2,:], A[1,:].copy()
print(A0)    
print(A)    
print("------------")  
A=A0
# or more efficiently
A[[1, 2],:] = A[[2, 1],:]
print(A0)
print(A)
print("------------")
# can swap columns in a similar way 

#%% Vectorization with 2d arrays

# consider adding two square matrices

A = np.arange(1,10).reshape(3,3)
B = np.arange(11,20).reshape(3,3)
print(A,B)
C=A+B
print(C)
C=A*B # this is also elemant wise
print(C)
print("------------")

# For matrix multiplication (these can also do 1d arrays)
C = np.dot(A,B)
print(C) 
# or this is better with modern python and looks cleaner
C = A@B
print(C)

# other examples and features
C = np.transpose(A) # or C = A.T
print(A)
print(C)
C = np.trace(A) # who doesn's love a trace function?
print(C)
print("------------")

# iterations
A = np.arange(1,10).reshape(3,3)

for row in A:
    print(row)
print("------------")
for column in A.T:
    print(column)

A = np.arange(1,10).reshape(3,3)
print(np.sum(A,axis=0)) # sum along rows
print(np.sum(A,axis=1)) # sum along columns

#%% Simple Input and Output - with Files

# simple text file example
xs = np.linspace(0,1.8,10)
ys = 4*xs**3 - 7
np.savetxt("compl1.out", (xs,ys))
print(xs)
print(ys)
print("------------")
xs2, ys2 = np.loadtxt("compl1.out")
print(xs2)
print(ys2)
print("------------")

# example using savez - a very useful function, as we can save anything
outfile='comple2_out.npz' # not readable, but who cares!
a = 10
b = 20
np.savez(outfile, xs2=xs, ys2=ys,a=a,b1=b)
#savez can save several arrays into a single file in uncompressed .npz format (zipped archive of files)

# load back in and check
z = np.load(outfile)
z.files
xs2=z['xs2']
ys2=z['ys2']
b2=z['b1']
print(xs2)
print(ys2)
print(b2)

print("-----------------------------")
print("Code is now complete. Goodbye!")
print("-----------------------------")





