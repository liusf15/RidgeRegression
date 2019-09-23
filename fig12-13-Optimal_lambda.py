"""
    Codes for reproducing Figure 12, 13
    Numerical ways to Find the optimal lambda of orthogonal sketching
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


def MSE_original(lbd, gamma, alpha=1, sigma=1, verbose=0):
    the_1 = theta_1(lbd, gamma)
    the_2 = theta_2(lbd, gamma)
    bias = alpha ** 2 * lbd ** 2 * the_2
    variance = gamma * sigma ** 2 * (the_1 - lbd * the_2)
    if verbose == 0:
        return bias + variance
    else:
        return bias + variance, bias, variance


def MSE_primal(lbd, gamma, xi, alpha=1, sigma=1, verbose=0):
    the_1 = 1 / xi * theta_1(lbd / xi, gamma / xi)
    the_2 = 1 / xi ** 2 * theta_2(lbd / xi, gamma / xi)
    bias = alpha ** 2 * ((lbd + xi - 1) ** 2 + gamma * (1 - xi)) * the_2
    variance = gamma * sigma ** 2 * (the_1 - (lbd + xi - 1) * the_2)
    if verbose == 0:
        return bias + variance
    else:
        return bias + variance, bias, variance


def MSE_dual(lbd, gamma, zeta, alpha=1, sigma=1, verbose=0):
    # the_1 = quad(MP_moment, (1 - sqrt(zeta)) ** 2, (1 + sqrt(zeta)) ** 2, args=(1, zeta, lbd, 1))[0]
    # the_2 = quad(MP_moment, (1 - sqrt(zeta)) ** 2, (1 + sqrt(zeta)) ** 2, args=(2, zeta, lbd, 1))[0]
    # barthe_1 = (1 - zeta) / lbd + zeta * the_1
    # barthe_2 = (1 - zeta) / lbd ** 2 + zeta * the_2
    the_1 = theta_1(lbd, zeta) * zeta + (1 - zeta) / lbd
    the_2 = theta_2(lbd, zeta) * zeta + (1 - zeta) / lbd ** 2
    bias_1 = alpha ** 2 / gamma * (
            1 + (lbd + zeta - gamma) ** 2 * the_2 + 2 * (gamma - zeta - lbd) * the_1 + (
            gamma - zeta) * the_1 ** 2)
    bias_2 = -2 * alpha ** 2 / gamma * (1 - (lbd + zeta - gamma) * the_1)
    bias = bias_1 + bias_2 + alpha ** 2
    variance = sigma ** 2 * (the_1 - (lbd + zeta - gamma) * the_2)
    if verbose == 0:
        return bias + variance
    else:
        return bias + variance, bias, variance


def MSE_marginal(lbd, gamma, alpha=1, sigma=1, verbose=0):
    bias = alpha**2 * ((lbd - 1)**2 + gamma) / lbd**2
    variance = gamma * sigma**2 / lbd**2
    if verbose == 0:
        return bias + variance
    else:
        return bias + variance, bias, variance


def MSE_full(lbd, gamma, xi, alpha=1, sigma=1, verbose=0):
    the_1 = theta_1(lbd/xi, gamma/xi) / xi
    the_2 = theta_2(lbd/xi, gamma/xi) / xi ** 2
    bias = alpha ** 2 * lbd ** 2 * the_2
    variance = gamma * sigma ** 2 * (the_1 - lbd * the_2)
    if verbose == 0:
        return bias + variance
    else:
        return bias + variance, bias, variance


def MSE(method, lbd, gamma, xi=1, alpha=1, sigma=1, verbose=0):
    if method == 'original':
        return MSE_original(lbd, gamma, alpha, sigma, verbose)
    if method == 'primal':
        return MSE_primal(lbd, gamma, xi, alpha, sigma, verbose)
    if method == 'dual':
        return MSE_dual(lbd, gamma, xi, alpha, sigma, verbose)
    if method == 'full':
        return MSE_full(lbd, gamma, xi, alpha, sigma, verbose)
    else:
        return 0


def optimal_lambda(method, gamma, lb=0.1, ub=3, xi=1, alpha=1, sigma=1):
    d = np.linspace(lb, ub, 300)
    mse = np.inf
    flag = 0
    for i in range(300):
        lbd = d[i]
        a = MSE(method, lbd, gamma, xi, alpha, sigma)
        if a < mse:
            mse = a
            flag = i
    return d[flag], mse

optimal_lambda('primal', 0.7, xi=0.000001)

lbd = 2
gamma = 0.5
xi = 0.0001
print(theta_1(lbd / xi, gamma / xi) / xi)
print(theta_2(lbd / xi, gamma / xi) / xi ** 2)


alpha = 1
sigma = 1
lbd_seq = np.linspace(1, 3, 100)
gamma = 0.7
verbose = 1

xi_seq = np.linspace(0.001, 1, 200)
mse_primal_optimal = np.zeros((200, 3))
lbd_primal_optimal = np.zeros(200)

mse_dual_optimal = np.zeros((200, 3))
lbd_dual_optimal = np.zeros(200)

mse_full_optimal = np.zeros((200, 3))
lbd_full_optimal = np.zeros(200)

lbd_original_optimal, mse_original_optimal = optimal_lambda('original', gamma)
_, bias_original_optimal, var_original_optimal = MSE('original', lbd_original_optimal, gamma, 1, alpha, sigma, verbose)
verbose = 1
for i in range(200):
    xi = xi_seq[i]
    lbd, _ = optimal_lambda('primal', gamma, xi, alpha, sigma)
    lbd_primal_optimal[i] = lbd
    mse_primal_optimal[i, :] = MSE('primal', lbd, gamma, xi, alpha, sigma, verbose)

    lbd = gamma * sigma ** 2 / alpha ** 2
    mse_full_optimal[i, :] = MSE('full', lbd, gamma, xi, alpha, sigma, verbose)

    zeta = xi_seq[i] * gamma
    lbd, _ = optimal_lambda('dual', gamma, zeta, alpha, sigma)
    lbd_dual_optimal[i] = lbd
    mse_dual_optimal[i, :] = MSE('dual', lbd, gamma, zeta, alpha, sigma, verbose)

# Figure 12
# MSE at optimal lambda, different xi
plt.plot(xi_seq, mse_primal_optimal[:, 0], label='Primal MSE', ls=':')
plt.plot(xi_seq, mse_dual_optimal[:, 0], label='Dual MSE', ls='--')
plt.plot(xi_seq, mse_full_optimal[:, 0], label='Full MSE', ls='-.')
plt.plot(xi_seq, mse_original_optimal * np.ones(200), label='Original')
plt.xlabel(r'$\xi$')
plt.ylabel('MSE')
plt.title(r'MSE at optimal $\lambda$, $\gamma={}$'.format(gamma))
plt.grid(linestyle='dotted')
plt.legend()
plt.savefig('primal_dual_mse_gamma={}.png'.format(gamma))

# bias at optimal lambda, different xi
plt.plot(xi_seq, mse_primal_optimal[:, 1], label='Primal bias', ls=':')
plt.plot(xi_seq, mse_dual_optimal[:, 1], label='Dual bias', ls='--')
plt.plot(xi_seq, mse_full_optimal[:, 1], label='Full bias', ls='-.')
plt.plot(xi_seq, bias_original_optimal * np.ones(200), label='Original')
plt.xlabel(r'$\xi$')
plt.ylabel('Bias')
plt.title(r'Bias at optimal $\lambda$, $\gamma={}$'.format(gamma))
plt.grid(linestyle='dotted')
plt.legend()
plt.savefig('primal_dual_bias_gamma={}.png'.format(gamma))

# variance at optimal lambda, different xi
plt.plot(xi_seq, mse_primal_optimal[:, 2], label='Primal var', ls=':')
plt.plot(xi_seq, mse_dual_optimal[:, 2], label='Dual var', ls='--')
plt.plot(xi_seq, mse_full_optimal[:, 2], label='Full bias', ls='-.')
plt.plot(xi_seq, var_original_optimal * np.ones(200), label='Original')
plt.xlabel(r'$\xi$')
plt.ylabel('Variance')
plt.title(r'Variance at optimal $\lambda$, $\gamma={}$'.format(gamma))
plt.grid(linestyle='dotted')
plt.legend()
plt.savefig('primal_dual_var_gamma={}.png'.format(gamma))


# optimal lambda, different xi
plt.plot(xi_seq, lbd_primal_optimal, label='Primal sketch', ls=':')
plt.plot(xi_seq, lbd_dual_optimal, label='Dual sketch', ls='--')
plt.plot(xi_seq, lbd_full_optimal, label='Full sketch', ls='-.')
plt.plot(xi_seq, lbd_original_optimal * np.ones(200), label='Original')
plt.xlabel(r'$\xi$')
plt.ylabel(r'Optimal $\lambda^*$')
plt.title(r'Optimal $\lambda^*$, $\gamma={}$'.format(gamma))
plt.grid(linestyle='dotted')
plt.legend()
plt.savefig('primal_dual_optimal_lbd_gamma={}.png'.format(gamma))


# bias variance trade-off
verbose = 1
original_mse_bias_var = np.zeros((100, 3))
primal_mse_bias_var = np.zeros((100, 3))
dual_mse_bias_var = np.zeros((100, 3))
lbd_seq = np.linspace(0.5, 2, 100)

xi = 0.5
for i in range(100):
    lbd = lbd_seq[i]
    original_mse_bias_var[i, :] = MSE('original', lbd, gamma, xi, alpha, sigma, verbose)
    primal_mse_bias_var[i, :] = MSE('primal', lbd, gamma, xi, alpha, sigma, verbose)
    dual_mse_bias_var[i, :] = MSE('dual', lbd, gamma, xi, alpha, sigma, verbose)


plt.plot(lbd_seq, original_mse_bias_var[:, 1], label='Bias', ls='--')
plt.plot(lbd_seq, original_mse_bias_var[:, 2], label='Variance', ls=':')
plt.plot(lbd_seq, original_mse_bias_var[:, 0], label='MSE')
plt.xlabel(r'$\lambda$')
plt.title(r'Bias & Variance, $\gamma$={}'.format(gamma))
plt.grid(linestyle='dotted')
plt.legend()
plt.savefig('bias_var_gamma={}.png'.format(gamma))


plt.plot(lbd_seq, primal_mse_bias_var[:, 1], label='Bias', ls='--')
plt.plot(lbd_seq, primal_mse_bias_var[:, 2], label='Variance', ls=':')
plt.plot(lbd_seq, primal_mse_bias_var[:, 0], label='MSE')

plt.plot(lbd_seq, dual_mse_bias_var[:, 1], label='Bias', ls='--')
plt.plot(lbd_seq, dual_mse_bias_var[:, 2], label='Variance', ls=':')
plt.plot(lbd_seq, dual_mse_bias_var[:, 0], label='MSE')

# compare bias for different lambda, fix xi
plt.plot(lbd_seq, original_mse_bias_var[:, 1], label='Original bias', ls='--')
plt.plot(lbd_seq, primal_mse_bias_var[:, 1], label='Primal bias', ls='--')
plt.plot(lbd_seq, dual_mse_bias_var[:, 1], label='Dual bias', ls='--')
plt.legend()

# compare variance for different lambda, fix xi
plt.plot(lbd_seq, original_mse_bias_var[:, 2], label='Original var', ls='--')
plt.plot(lbd_seq, primal_mse_bias_var[:, 2], label='Primal var', ls='--')
plt.plot(lbd_seq, dual_mse_bias_var[:, 2], label='Dual var', ls='--')
plt.legend()

verbose = 1
gamma = 0.7
xi_seq = np.linspace(0.001, 1, 100)
# zeta_seq = np.linspace(0.001, gamma, 100)
lbd = 0.7
original_mse, original_bias, original_var = MSE('original', lbd, gamma, 1, alpha, sigma, verbose)
for i in range(100):
    xi = xi_seq[i]
    primal_mse_bias_var[i, :] = MSE('primal', lbd, gamma, xi, alpha, sigma, verbose)
    zeta = xi * gamma
    dual_mse_bias_var[i, :] = MSE('dual', lbd, gamma, zeta, alpha, sigma, verbose)

plt.plot(xi_seq, primal_mse_bias_var[:, 0], label='MSE')
plt.plot(xi_seq, primal_mse_bias_var[:, 1], label='Bias')
plt.plot(xi_seq, primal_mse_bias_var[:, 2], label='Var')
plt.legend()

plt.plot(xi_seq, dual_mse_bias_var[:, 0], label='MSE')
plt.plot(xi_seq, dual_mse_bias_var[:, 1], label='Bias')
plt.plot(xi_seq, dual_mse_bias_var[:, 2], label='Var')
plt.legend()

# bias for different xi, fix lbd=0.7
plt.plot(xi_seq, primal_mse_bias_var[:, 1], label='Primal bias', ls=':')
plt.plot(xi_seq, dual_mse_bias_var[:, 1], label='Dual bias', ls='--')
plt.plot(xi_seq, np.ones(100) * original_bias, label='Original bias')
plt.legend()
plt.xlabel(r'$\xi$')
plt.ylabel('Bias')
plt.grid(linestyle='dotted')
plt.title(r'Bias, $\gamma=${},$\lambda=${}'.format(gamma, lbd))
plt.savefig('bias_gamma={}.png'.format(gamma))

# variance for different xi, fix lbd=0.7
plt.plot(xi_seq, primal_mse_bias_var[:, 2], label='Primal variance', ls=':')
plt.plot(xi_seq, dual_mse_bias_var[:, 2], label='Dual variance', ls='--')
plt.plot(xi_seq, np.ones(100) * original_var, label='Original variance')
plt.legend()
plt.xlabel(r'$\xi$')
plt.ylabel('Variance')
plt.grid(linestyle='dotted')
plt.title(r'Variance, $\gamma=${},$\lambda=${}'.format(gamma, lbd))
plt.savefig('variance_gamma={}.png'.format(gamma))

# MSE for different xi, fix lbd=0.7
plt.plot(xi_seq, primal_mse_bias_var[:, 0], label='Primal MSE', ls=':')
plt.plot(xi_seq, dual_mse_bias_var[:, 0], label='Dual MSE', ls='--')
plt.plot(xi_seq, np.ones(100) * original_mse, label='Original MSE')
plt.legend()
plt.xlabel(r'$\xi$')
plt.ylabel('MSE')
plt.grid(linestyle='dotted')
plt.title(r'MSE, $\gamma=${},$\lambda=${}'.format(gamma, lbd))
plt.savefig('MSE_gamma={}.png'.format(gamma))

# Figure 13
# compare full and primal sketch, fix lambda
alpha = 1
sigma = 1
gamma = 0.7
verbose = 1
lbd = 0.7
primal_mse_bias_var = np.zeros((200, 3))
full_mse_bias_var = np.zeros((200, 3))
for i in range(200):
    xi = xi_seq[i]
    primal_mse_bias_var[i, :] = MSE('primal', lbd, gamma, xi, alpha, sigma, verbose)
    full_mse_bias_var[i, :] = MSE('full', lbd, gamma, xi, alpha, sigma, verbose)

plt.plot(xi_seq, full_mse_bias_var[:, 2], label='Full sketch')
plt.plot(xi_seq, primal_mse_bias_var[:, 2], label='Primal sketch')
plt.legend()

# compare full and primal sketch, optimal lambda
alpha = 1
sigma = 1
gamma = 0.7
xi_seq = np.linspace(0.1, gamma, 200)
verbose = 1
mse_primal_optimal = np.zeros((200, 3))
lbd_primal_optimal = np.zeros(200)
mse_full_optimal= np.zeros((200, 3))
lbd_full_optimal = np.zeros(200)
mse_dual_optimal = np.zeros((200, 3))
lbd_dual_optimal = np.zeros(200)
original_mse, original_bias, original_var = MSE('original', lbd, gamma, 1, alpha, sigma, verbose)
for i in range(200):
    xi = xi_seq[i]

    lbd, _ = optimal_lambda('primal', gamma, lb=0.1, ub=2.5, xi=xi, alpha=alpha, sigma=sigma)
    lbd_primal_optimal[i] = lbd
    mse_primal_optimal[i, :] = MSE('primal', lbd, gamma, xi, alpha, sigma, verbose)

    lbd, _ = optimal_lambda('dual', gamma, lb=0.1, ub=2.5, xi=xi, alpha=alpha, sigma=sigma)
    lbd_dual_optimal[i] = lbd
    mse_dual_optimal[i, :] = MSE('dual', lbd, gamma, xi, alpha, sigma, verbose)

    lbd = gamma * sigma ** 2 / alpha ** 2
    lbd_full_optimal[i] = lbd
    mse_full_optimal[i, :] = MSE('full', lbd, gamma, xi, alpha, sigma, verbose)

# plot mse
plt.plot(xi_seq, mse_primal_optimal[:, 0], label='Primal sketch', linewidth=3)
plt.plot(xi_seq, mse_dual_optimal[:, 0], label='Dual sketch', ls='--', linewidth=3)
plt.plot(xi_seq, mse_full_optimal[:, 0], label='Full sketch', ls='-.', linewidth=3)
plt.plot(xi_seq, np.ones(200) * original_mse, label='Original', ls=':', linewidth=3)
plt.legend(fontsize=14)
plt.xlabel(r'$\xi$', fontsize=14)
plt.ylabel('MSE', fontsize=14)
plt.grid(linestyle='dotted')
plt.title(r'MSE, $\gamma$=0.7', fontsize=14)
plt.savefig('MSE_full_dual_primal.png')
# plot bias
plt.plot(xi_seq, mse_primal_optimal[:, 1], label='Primal sketch', linewidth=3)
plt.plot(xi_seq, mse_dual_optimal[:, 1], label='Dual sketch', ls='--', linewidth=3)
plt.plot(xi_seq, mse_full_optimal[:, 1], label='Full sketch', ls='-.', linewidth=3)
plt.plot(xi_seq, np.ones(200) * original_bias, label='Original', ls=':', linewidth=3)
plt.legend(fontsize=14)
plt.xlabel(r'$\xi$', fontsize=14)
plt.ylabel('bias', fontsize=14)
plt.grid(linestyle='dotted')
plt.title(r'Bias, $\gamma$=0.7', fontsize=14)
plt.savefig('bias_full_dual_primal.png')

plt.plot(xi_seq, mse_primal_optimal[:, 2], label='Primal sketch', linewidth=3)
plt.plot(xi_seq, mse_dual_optimal[:, 2], label='Dual sketch', ls='--', linewidth=3)
plt.plot(xi_seq, mse_full_optimal[:, 2], label='Full sketch', ls='-.', linewidth=3)
plt.plot(xi_seq, np.ones(200) * original_var, label='Original', ls=':', linewidth=3)
plt.legend(fontsize=14)
plt.xlabel(r'$\xi$', fontsize=14)
plt.ylabel('var', fontsize=14)
plt.grid(linestyle='dotted')
plt.title(r'Variance, $\gamma$=0.7', fontsize=14)
plt.savefig('var_full_dual_primal.png')

plt.plot(xi_seq, lbd_primal_optimal, label='Primal sketch', linewidth=3)
plt.plot(xi_seq, lbd_dual_optimal, label='Dual sketch', linewidth=3)
plt.plot(xi_seq, lbd_full_optimal, label='Full sketch', linewidth=3)
plt.legend(fontsize=14)
plt.xlabel(r'$\xi$', fontsize=14)
plt.ylabel(r'$\lambda$', fontsize=14)
plt.grid(linestyle='dotted')
plt.title(r'Optimal $\lambda$, $\gamma$=0.7', fontsize=14)
plt.savefig('lbd_full_dual_primal.png')

