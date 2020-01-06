"""
    Codes for reproducing:
    Figure 5: MSE of ridge regression
"""

import numpy as np
import matplotlib as mpl

mpl.use('tkAgg')
import matplotlib.pyplot as plt
from numpy import sqrt
from numpy.linalg import norm
from numpy.linalg import inv
from scipy.integrate import quad
import os


def theta_1(lbd, gamma):
    z_2 = -0.5 * (sqrt(gamma) + (1 + lbd) / sqrt(gamma) + sqrt((sqrt(gamma) + (1 + lbd) / sqrt(gamma)) ** 2 - 4))
    return -(1 + lbd) / (gamma * lbd) - z_2 / (sqrt(gamma) * lbd)


def theta_2(lbd, gamma):
    delta = (sqrt(gamma) + (1 + lbd) / sqrt(gamma)) ** 2 - 4
    return -1 / (gamma * lbd ** 2) + (gamma + 1) / (2 * gamma * lbd ** 2) - 1 / (2 * sqrt(gamma)) * (
            (lbd + 1) / gamma + 1) / (lbd * sqrt(delta)) + 1 / (2 * sqrt(gamma)) * sqrt(delta) / (lbd ** 2)


def generate_haar_matrix(n, p):
    if n <= p:
        return np.linalg.qr(np.random.randn(p, n))[0].T
    else:
        return np.linalg.qr(np.random.randn(n, p))[0]


def MSE_original(lbd, gamma, alpha=1, sigma=1, verbose=0):
    the_1 = theta_1(lbd, gamma)
    the_2 = theta_2(lbd, gamma)
    bias = alpha ** 2 * lbd ** 2 * the_2
    variance = gamma * sigma ** 2 * (the_1 - lbd * the_2)
    if verbose == 0:
        return bias + variance
    else:
        return bias + variance, bias, variance


alpha = 3
sigma = 1
n = 1000
gamma = 0.3
p = np.int(n * gamma)
num_steps = 20
rep = 50
lbd_seq = np.linspace(0.001, 3, num_steps)
MSE_simu = np.zeros((rep, num_steps))
test_simu = np.zeros((rep, num_steps))
for k in range(rep):
    X = np.random.randn(n, p)
    beta = np.random.randn(p, 1) * alpha / sqrt(p)
    epsilon = np.random.randn(n, 1) * sigma
    Y = X @ beta + epsilon
    for i in range(num_steps):
        lbd = lbd_seq[i]
        beta_ridge = inv(X.T @ X / n + lbd * np.identity(p)) @ X.T @ Y / n
        MSE_simu[k, i] = norm(beta_ridge - beta) ** 2 / norm(beta) ** 2

MSE_theo = np.zeros(100)
lbd_seq_2 = np.linspace(0.001, 3, 100)
for i in range(100):
    MSE_theo[i] = MSE_original(lbd_seq_2[i], gamma, alpha, sigma) / (alpha ** 2)


xx = np.mean(MSE_simu, 0)
yerr = np.std(MSE_simu, 0)
# plt.figure(0, figsize=(12, 8))
plt.plot(lbd_seq_2, MSE_theo, lw=4, ls='-', label='Theory')
plt.errorbar(lbd_seq, xx, yerr, capsize=2, lw=3, ls='--', label='Simulation')
plt.grid(linestyle='dotted')
plt.xlabel(r'$\lambda$', fontsize=14)
plt.ylabel(r'$MSE(\hat\beta)/\Vert\beta\Vert^2$', fontsize=14)
plt.title(r'MSE of $\hat\beta$, $\gamma={},\alpha={},\sigma={}$'.format(gamma, alpha, sigma), fontsize=14)
plt.legend(fontsize=14)
plt.savefig('./Plots/gamma_{}_alpha_{}_sigma_{}.png'.format(gamma, alpha, sigma))
