"""
    Codes for reproducing Figure 1, 6
    Bias-variance tradeoff of ridge regression, fixed lambda and optimal lambda
"""


import numpy as np
import matplotlib as mpl

mpl.use('tkAgg')
import matplotlib.pyplot as plt
from numpy import sqrt
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


def residual_original(lbd, gamma, alpha=1, sigma=1, verbose=0):
    the_1 = theta_1(lbd, gamma)
    the_2 = theta_2(lbd, gamma)
    bias = alpha ** 2 * lbd ** 2 * (the_1 - lbd * the_2)
    variance = sigma ** 2 * (1 - 2 * gamma * (1 - lbd * the_1) + gamma * (1 - 2 * lbd * the_1 + lbd ** 2 * the_2))
    if verbose == 0:
        return bias + variance
    else:
        return bias + variance, bias, variance


alpha = 1
sigma = 1
gamma_1 = 0.2
lbd_seq_2 = np.linspace(0.001, 3, 100)
theory_1 = np.zeros((100, 3))
for i in range(100):
    lbd = lbd_seq_2[i]
    theory_1[i, :] = MSE_original(lbd, gamma_1, alpha, sigma, verbose=1)

gamma_2 = 2
theory_2 = np.zeros((100, 3))
lbd_seq_3 = np.linspace(0.001, 4, 100)
for i in range(100):
    lbd = lbd_seq_3[i]
    theory_2[i, :] = MSE_original(lbd, gamma_2, alpha, sigma, verbose=1)

plt.figure(0, figsize=(10, 4))
p1 = plt.subplot(121)
plt.plot(lbd_seq_2, theory_1[:, 0], label='test error', ls='-', lw=4)
p1.plot(lbd_seq_2, theory_1[:, 1], label=r'$Bias^2$', ls='--', lw=4)
p1.plot(lbd_seq_2, theory_1[:, 2], label='Var', ls=':', lw=4)
p1.set_xlabel(r'$\lambda$', fontsize=14)
p1.grid(linestyle='dotted')
p1.set_title(r'$\gamma={}$'.format(gamma_1), fontsize=14)

p2 = plt.subplot(122)
p2.plot(lbd_seq_3, theory_2[:, 0], label='test error', ls='-', lw=4)
p2.plot(lbd_seq_3, theory_2[:, 1], label=r'$Bias^2$', ls='--', lw=4)
p2.plot(lbd_seq_3, theory_2[:, 2], label='Var', ls=':', lw=4)
p2.set_xlabel(r'$\lambda$', fontsize=14)
p2.grid(linestyle='dotted')
p2.legend(fontsize=14)
p2.set_title(r'$\gamma={}$'.format(gamma_2), fontsize=14)
plt.subplots_adjust(wspace=0.1)
plt.savefig('./Plots/residual_bias_var_alpha_{}_sigma_{}.png'.format(alpha, sigma))

# at optimal lambda
alpha = 1
sigma = 1
gamma_seq = np.linspace(0.01, 5, 100)
theory_3 = np.zeros((100, 3))
for i in range(100):
    gamma = gamma_seq[i]
    lbd = gamma * sigma ** 2 / alpha ** 2
    theory_3[i, :] = MSE_original(lbd, gamma, alpha, sigma, verbose=1)

plt.figure(0)
plt.plot(gamma_seq, theory_3[:, 0], label='MSE', lw=4)
plt.plot(gamma_seq, theory_3[:, 1], label=r'$Bias^2$', ls='--', lw=4)
plt.plot(gamma_seq, theory_3[:, 2], label=r'$Var$', ls=':', lw=4)
plt.grid(linestyle='dotted')
plt.xlabel(r'$\gamma$', fontsize=14)
plt.legend(fontsize=14)
plt.title(r'Bias-variance at optimal $\lambda$')
plt.savefig('./Plots/bias_var_optimal_alpha_{}_sigma_{}.png'.format(alpha, sigma))
