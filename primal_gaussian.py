import numpy as np
from numpy import sqrt
from numpy.linalg import inv
from numpy import log
from numpy.linalg import eig
from numpy.linalg import svd
from numpy.linalg import norm
from scipy.linalg import sqrtm
import pandas as pd
import matplotlib.pyplot as plt
import os
os.chdir('/Users/sifanliu/Dropbox/Dual Sketch/RidgeRegression')


# generate a uniformly distributed orthogonal matrix
def generate_haar_matrix(n, p):
    if n <= p:
        return np.linalg.qr(np.random.randn(p, n))[0].T
    else:
        return np.linalg.qr(np.random.randn(n, p))[0]


X = np.random.randn(500, 500)
A = X.T @ X / 500 + np.identity(500)
Y = np.random.randn(500, 500)
B = Y.T @ Y / 500 + np.identity(500)
a0 = np.trace(A @ B) / 500
a1 = np.trace(A @ B @ A @ B) / 500
a2 = np.trace(A @ A @ B @ B) / 500
eigval = eig(sqrtm(B) @ A @ sqrtm(B))[0]
singular_val = svd(A @ B)[1]
sum(eigval ** 2) / 500 - a1
sum(singular_val ** 2) / 500 - a2
plt.hist(eigval, 100)
plt.hist(singular_val, 100)
plt.plot(np.sort(eigval))
plt.plot(np.sort(singular_val))

sum(singular_val) - sum(eigval)

eigval1 = eig(A @ B)[0]
eigval2 = eig(B @ A)[0]
sum(eigval) - sum(eigval1)
sum(eigval1) - sum(eigval2)
