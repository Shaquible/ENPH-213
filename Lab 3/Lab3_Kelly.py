from timeit import default_timer
import numpy as np
import matplotlib.pyplot as plt
# %%
# Q1(a)


def nxnfill(n):
    # generating a flat array with the numbers then reshaping to be n by n
    A = np.arange(21, 21+n**2, dtype=float).reshape(n, n)
    return A


def lowerTri(A):
    # creating a array of zeros and putting in elements of A into the lower triangle
    B = np.zeros_like(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if i >= j:
                B[i, j] = A[i, j]
    return B


def upperTri(A):
    # creating a array of zeros and putting in elements of A into the upper triangle
    B = np.zeros_like(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if i <= j:
                B[i, j] = A[i, j]
    return B


def euclideanNorm(A):
    # summing the squares of the elements in A then rooting it
    return np.sqrt(np.sum(A**2))


def infinityNorm(A):
    # summing the absolute values of each row then finding the max sum
    return np.max(np.sum(np.abs(A), axis=1))


# calling all of the created functions and printing the results
arr1 = nxnfill(4)
print("A\n", arr1)
print("Lower Triangle(A)\n", lowerTri(arr1))
print("Upper Triangle(A)\n", upperTri(arr1))
print("My function Euclidian norm", euclideanNorm(arr1))
print("NP Euclidian norm", np.linalg.norm(arr1))
print("My function infinity norm", infinityNorm(arr1))
print("NP infinity norm", np.linalg.norm(arr1, np.inf))
print("\n")
# %%
# Q1(b)


def matrixGenA(n):
    # generating a diagonal of 1s and a upper triange above the diagonal of -1s
    A = np.eye(n, n, dtype=float) - np.triu(np.ones((n, n), dtype=float), 1)
    return A


def matrixGenB(n):
    # generating alternating 1s and -1s
    B = [1-2*(i % 2) for i in range(n)]
    return B


def k(A):
    # calculating kappa
    return euclideanNorm(A)*euclideanNorm(np.linalg.inv(A))


def q1b(n):
    # printing all the statements required for Q1(b) ii and iii
    A = matrixGenA(n)
    B = matrixGenB(n)
    print("n=", n)
    print("Solution without perterbation", np.linalg.solve(A, B))
    print("k=", k(A))
    A[n-1, n-1] -= 0.001
    print("Solution with perterbation", np.linalg.solve(A, B))
    print("k=", k(A))


q1b(4)
print("\n")
q1b(16)
# %%
# Q2(a)


def backsub1(U, bs):  # given back subsitution
    n = bs.size
    xs = np.zeros_like(bs)  # assigns same dims as bs

    xs[n-1] = bs[n-1]/U[n-1, n-1]  # bb is not needed for this case
    for i in range(n-2, -1, -1):
        bb = 0
        for j in range(i+1, n):
            bb += U[i, j]*xs[j]
        xs[i] = (bs[i] - bb)/U[i, i]
    return xs


def backsub2(U, bs):
    # generaing xs as a list of zeros
    n = len(bs)
    xs = np.zeros_like(bs)
    xs[-1] = bs[-1]/U[-1, -1]
    # computing xs for each element backwards from second to last elements
    for i in range(n-2, -1, -1):
        # doing the sum by splicing the two arrays into the size required for the i value
        xs[i] = (bs[i] - np.sum(U[i, i+1:]*xs[i+1:]))/U[i, i]
    return xs


def solveCompare(f, g, A, bs):  # printing first 3 elements of the two solutions
    x1 = f(A, bs)
    x2 = g(A, bs)
    print("My solution first 3 elements", x2[0:3])
    print("Accepted Solution first 3 elements", x1[0:3])


print("\n")
U = nxnfill(5000)
bs = U[0, :]
timer_start = default_timer()
x1 = backsub1(U, bs)
timer_end = default_timer()
time1 = timer_end - timer_start
print('Backsub1 time:', time1)
timer_start = default_timer()
x2 = backsub2(U, bs)
timer_end = default_timer()
time2 = timer_end - timer_start
print('Backsub2 time:', time2)
solveCompare(backsub1, backsub2, U, bs)


# %%
# Q2(b)


def gausselim(A, bs):
    bs = bs.reshape(A.shape[0], 1)
    A = np.append(A, bs, axis=1)
    A = np.array(A, dtype=float)
    for i in range(1, len(A)):
        # filling an array with 0s then adding the ceofficients times the pivot row and subtracting, iterating for each pivot row
        coeff = np.zeros_like(A)
        coefftmp = A[i:, i-1]/A[i-1, i-1]
        coeff[i:, :] = coefftmp[:, None]
        coeff = coeff*A[i-1, :]
        A = A - coeff
    As = A[:, :-1]
    bs = A[:, -1]
    return backsub2(As, bs)


A = np.array([[2, 1, 1], [1, 1, -2], [1, 2, 1]])
B = np.array([8, -2, 2])

print("2(b) solution", gausselim(A, B))

# %%
# Q2(c)


def gausselimPartial(A, bs):
    bs = bs.reshape(A.shape[0], 1)
    A = np.append(A, bs, axis=1)
    A = np.array(A, dtype=float)
    for i in range(1, len(A)):
        # searching for the pivot row
        pivot = np.argmax(np.abs(A[i-1:, i-1])) + i-1
        # swapping the pivot row with the current row
        A[[i-1, pivot]] = A[[pivot, i-1]]
        # filling an array with 0s then adding the ceofficients times the pivot row and subtracting, iterating for each pivot row
        coeff = np.zeros_like(A)
        coefftmp = A[i:, i-1]/A[i-1, i-1]
        coeff[i:, :] = coefftmp[:, None]
        coeff = coeff*A[i-1, :]
        A = A - coeff
    As = A[:, :-1]
    bs = A[:, -1]
    return backsub2(As, bs)


A = np.array([[2, 1, 1], [2, 1, -4], [1, 2, 1]])
B = np.array([8, -2, 2])
print("2(c) solution", gausselimPartial(A, B))

# %%
# Q2(d)


def inverseMatrix(A):
    A = np.array(A, dtype=float)
    B1 = np.array([1, 0, 0])
    B2 = np.array([0, 1, 0])
    B3 = np.array([0, 0, 1])
    B1 = gausselimPartial(A, B1)
    B2 = gausselimPartial(A, B2)
    B3 = gausselimPartial(A, B3)
    inv = np.zeros_like(A)
    inv[:, 0] = B1
    inv[:, 1] = B2
    inv[:, 2] = B3
    return inv


A = np.array([[1, 2, 3], [0, 1, 4], [5, 6, 0]])
print("Inverse Matrix\n", inverseMatrix(A))
