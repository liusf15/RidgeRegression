import numpy as np
import matplotlib as mpl

mpl.use('tkAgg')
import matplotlib.pyplot as plt
from numpy import sqrt, log
from numpy.linalg import norm
from numpy.linalg import inv
from numpy.linalg import eig
from scipy.integrate import quad
import os
import time
import pandas as pd
os.chdir('/Users/sifanliu/Dropbox/Dual Sketch/experiments')


def MP_Stieltjes(z, gamma):
    return (1 - gamma - z - sqrt((1 + gamma - z) ** 2 - 4 * gamma)) / (2 * gamma * z)


def inv_MP_Stieltjes(z, gamma):
    return -1 / z - 1 / z ** 2 * MP_Stieltjes(1 / z, gamma)


def MP_R(z, gamma):
    return 1 / (1 - gamma * z)


n = 800
p = 1000
gamma = p / n
X = np.random.randn(n, p) / sqrt(p)
gram = X @ X.T
eigval, _ = eig(gram)
# plt.hist(eigval, 100)
z_seq = np.linspace(-10, -6, 100)
a1 = np.zeros(len(z_seq))
a2 = np.zeros(len(z_seq))
for i in range(len(z_seq)):
    z = z_seq[i]
    a1[i] = np.mean(1 / (1 / eigval - z))
    # a2[i] = inv_MP_Stieltjes(z, 1 / gamma)
    a2[i] = inv_MP_Stieltjes(z, 1 / gamma)
plt.plot(z_seq, a1)
plt.plot(z_seq, a2, "--")


def inv_MP_R(z, gamma):
    a = (z + 1) * gamma
    b = -z * gamma * (1 + gamma)
    c = (z * gamma) ** 2
    w = (-b + sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    w2 = (-b - sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    # print(w2)
    # w2 = (z * (gamma + 1) - sqrt(z ** 2 * (gamma + 1) ** 2 - 4 * gamma * (z + 1) * z ** 2)) / (2 * (z + 1) * gamma)
    num = 8 * (z + 1) ** 2 * gamma ** 2
    den = z * (gamma + 1 + sqrt((gamma + 1) ** 2 - 4 * gamma * (z + 1)))
    # return num / den - 1 / z
    return 1 / w2 - 1 / z
# checked



a3 = np.zeros(20)
z_seq = np.linspace(-10, -6, 20)
for i in range(20):
    z = z_seq[i]
    R = MP_R(z, 1 / gamma)
    w = R + 1 / z
    a3[i] = (1 - 1 / gamma - w - sqrt((1 + 1 / gamma - w) ** 2 - 4 / gamma)) / (2 / gamma * w)
    # a3[i] = inv_MP_Stieltjes(inv_MP_R(z, 1 / gamma) + 1 / z, 1 / gamma)
    a3[i] = inv_MP_Stieltjes(inv_MP_R_2(z, gamma) + 1 / z, 1 / gamma)

max(abs(a3 + z_seq))


def inv_MP_R_2(z, gamma):
    a = (z + 1) / gamma
    b = -z / gamma * (1 + 1 / gamma)
    c = (z / gamma) ** 2
    w = (-b - sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    w = z / (2 * (z + 1) * gamma) * (gamma + 1 - sqrt((gamma + 1) ** 2 - 4 * gamma * (z + 1)))
    num = z * (gamma + 1) - sqrt(z ** 2 * (gamma + 1) ** 2 - 4 * gamma * (z + 1) * z ** 2)
    den = 2 * (z + 1) * gamma
    w = num / den
    num = z * ((gamma + 1) + sqrt((gamma + 1) ** 2 - 4 * gamma * (z + 1)))
    den = 2 * (z + 1) * gamma
    w = num / den
    num = 8 * (z + 1) ** 2 * gamma ** 2
    den = z * (gamma + 1 - sqrt((gamma + 1) ** 2 - 4 * gamma * (z + 1)))
    w = num / den
    w = (gamma + 1 - sqrt((gamma + 1) ** 2 - 4 * gamma * (z + 1))) / 2 / z
    return w - 1 / z


def inv_MP_Stieltjes_inv(z, gamma, xi, lbd):
    return 1 / (1 + z / xi) - (gamma - 1 - sqrt((gamma - 1) ** 2 + 4 * lbd * z)) / (2 * z) - 1 / z


def inv_MP_Stieltjes_0(gamma, xi, lbd):
    maxiter = 100
    tol = 1e-8
    low = 0.5
    high = 1
    if inv_MP_Stieltjes_inv(low, gamma, xi, lbd) * inv_MP_Stieltjes_inv(high, gamma, xi, lbd) > 0:
        print("no root")
        return
    err = 1
    old = 0
    for i in range(maxiter):
        mid = (low + high) / 2
        if inv_MP_Stieltjes_inv(low, gamma, xi, lbd) * inv_MP_Stieltjes_inv(mid, gamma, xi, lbd) < 0:
            high = mid
        else:
            low = mid
        if abs(old - inv_MP_Stieltjes_inv(mid, gamma, xi, lbd)) < tol:
            print("converged")
            break
        old = inv_MP_Stieltjes_inv(mid, gamma, xi, lbd)
    return mid




xi = 0.9
lbd = 1
y_seq = np.linspace(0.5, 1, 100)
a1 = np.zeros(100)
for i in range(100):
    y = y_seq[i]
    a1[i] = inv_MP_Stieltjes_inv(y, gamma, xi, lbd)


plt.plot(y_seq, a1)
plt.plot(y_seq, np.zeros(100))

inv_MP_Stieltjes_0(gamma, xi, lbd)
inv_MP_Stieltjes_inv(0.58052, gamma, xi, lbd)


