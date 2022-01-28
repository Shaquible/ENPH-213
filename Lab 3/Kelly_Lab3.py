import numpy as np
import matplotlib.pyplot as plt
# %%
# Q1(a)


def nxnfill(n):
    A = np.arange(21, 21+n**2).reshape(n, n)
    return A


def euclideanNorm(A):
    return np.sqrt(np.sum(A**2))


def infinityNorm(A):
    return np.max([np.sum(np.abs(A[i, :])) for i in range(A.shape[0])])


# no idea what he wants for the lower and upper triangle part
arr1 = nxnfill(4)
print(arr1)
print("My function Euclidian norm", euclideanNorm(arr1))
print("NP Euclidian norm", np.linalg.norm(arr1))
print("My function infinity norm", infinityNorm(arr1))
print("NP infinity norm", np.linalg.norm(arr1, np.inf))

# %%
# Q1(b)


def matrixGenA(n):
    A = np.eye(n, n) - np.triu(np.ones((n, n)), 1)
    return A


def matrixGenB(n):
    B = [1-2*(i % 2) for i in range(n)]
    return B


def k(A):
    return euclideanNorm(A)*euclideanNorm(np.linalg.inv(A))


def q1c(n):
    A = matrixGenA(n)
    B = matrixGenB(n)
    print("n=", n)
    print("Solution without perterbation", np.linalg.solve(A, B))
    print("k=", k(A))
    A[n-1, n-1] -= 0.001
    print("Solution with perterbation", np.linalg.solve(A, B))
    print("k=", k(A))


q1c(4)
q1c(16)
