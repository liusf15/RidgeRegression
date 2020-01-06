"""
    Codes for reproducing:
    Figure 3, MSE of primal orthogonal sketching & bias-variance tradeoff
    Figure 4 (left), relative efficiency of marginal regression
    Figure 10, MSE of full sketching
"""

import matplotlib as mpl
import numpy as np

mpl.use('tkAgg')
import matplotlib.pyplot as plt
from numpy import sqrt
from numpy.linalg import norm
from numpy.linalg import inv


def generate_haar_matrix(n, p):
    if n <= p:
        return np.linalg.qr(np.random.randn(p, n))[0].T
    else:
        return np.linalg.qr(np.random.randn(n, p))[0]


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


def residual_original(lbd, gamma, alpha=1, sigma=1, verbose=0):
    the_1 = theta_1(lbd, gamma)
    the_2 = theta_2(lbd, gamma)
    bias = alpha ** 2 * lbd ** 2 * (the_1 - lbd * the_2)
    variance = sigma ** 2 * (1 - 2 * gamma * (1 - lbd * the_1) + gamma * (1 - 2 * lbd * the_1 + lbd ** 2 * the_2))
    if verbose == 0:
        return bias + variance
    else:
        return bias + variance, bias, variance


def primal_orth(lbd, gamma, xi, alpha=1, sigma=1, verbose=0):
    the_1 = 1 / xi * theta_1(lbd / xi, gamma / xi)
    the_2 = 1 / xi ** 2 * theta_2(lbd / xi, gamma / xi)
    bias = alpha ** 2 * ((lbd + xi - 1) ** 2 + gamma * (1 - xi)) * the_2
    variance = gamma * sigma ** 2 * (the_1 - (lbd + xi - 1) * the_2)
    if verbose == 0:
        return bias + variance
    else:
        return bias + variance, bias, variance


def dual_orth(lbd, gamma, zeta, alpha=1, sigma=1, verbose=0):
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


def marginal_orth(lbd, gamma, alpha=1, sigma=1, verbose=0):
    bias = alpha ** 2 * ((lbd - 1) ** 2 + gamma) / lbd ** 2
    variance = gamma * sigma ** 2 / lbd ** 2
    if verbose == 0:
        return bias + variance
    else:
        return bias + variance, bias, variance


def full_orth(lbd, gamma, xi, alpha=1, sigma=1, verbose=0):
    the_1 = theta_1(lbd / xi, gamma / xi) / xi
    the_2 = theta_2(lbd / xi, gamma / xi) / xi ** 2
    bias = alpha ** 2 * lbd ** 2 * the_2
    variance = gamma * sigma ** 2 * (the_1 - lbd * the_2)
    if verbose == 0:
        return bias + variance
    else:
        return bias + variance, bias, variance


def MSE(method, lbd, gamma, xi=1, alpha=1, sigma=1, verbose=0):
    if method == 'original':
        return MSE_original(lbd, gamma, alpha, sigma, verbose)
    if method == 'primal_orth':
        return primal_orth(lbd, gamma, xi, alpha, sigma, verbose)
    if method == 'dual_orth':
        return dual_orth(lbd, gamma, xi, alpha, sigma, verbose)
    if method == 'full_orth':
        return full_orth(lbd, gamma, xi, alpha, sigma, verbose)
    else:
        return 0


def optimal_lambda(method, gamma, xi=1, alpha=1, sigma=1):
    d = np.linspace(0.6, 2.5, 300)
    mse = np.inf
    flag = 0
    for i in range(300):
        lbd = d[i]
        a = MSE(method, lbd, gamma, xi, alpha, sigma)
        if a < mse:
            mse = a
            flag = i
    return d[flag], mse


# primal sketch
n = 300
gamma = 5
p = int(n * gamma)
alpha = 3
sigma = 1
num_steps = 20
xi_seq = np.linspace(0.1, 1, 20)
rep = 10
simu_primal = np.zeros((20, rep))
# lbd_seq = [0.1, 0.3, 0.5]
# lbd = gamma * sigma ** 2 / alpha ** 2
lbd = 1.5
for i in range(20):
    xi = xi_seq[i]
    m = int(n * xi)
    for k in range(rep):
        X = np.random.randn(n, p)
        beta = np.random.randn(p, 1) * alpha / sqrt(p)
        epsilon = np.random.randn(n, 1) * sigma
        Y = X @ beta + epsilon
        L = generate_haar_matrix(m, n)
        beta_primal = inv(X.T @ L.T @ L @ X / n + lbd * np.identity(p)) @ X.T @ Y / n
        beta_ridge = inv(X.T @ X / n + lbd * np.identity(p)) @ X.T @ Y / n
        simu_primal[i, k] = norm(beta_primal - beta) ** 2 / norm(beta_ridge - beta) ** 2

theory_primal = np.zeros(100)
xi_seq_2 = np.linspace(0.1, 1, 100)
for i in range(100):
    xi = xi_seq_2[i]
    # theory_primal[i] = primal_orth(lbd, gamma, xi, alpha, sigma) / alpha ** 2
    theory_primal[i] = primal_orth(lbd, gamma, xi, alpha, sigma) / MSE_original(lbd, gamma, alpha, sigma)

xx = np.mean(simu_primal, 1)
yerr = np.std(simu_primal, 1)
# plt.figure(0, figsize=(12, 8))
plt.plot(xi_seq_2, theory_primal, label='Theory', lw=4, ls='--')
plt.errorbar(xi_seq, xx, yerr, label='Simulation', capsize=3, lw=3)
plt.grid(linestyle='dotted')
plt.xlabel(r'$\xi$', fontsize=15)
plt.ylabel(r'$MSE(\hat\beta_{p})/MSE(\hat\beta)$', fontsize=15)
plt.title(r'Primal sketching MSE', fontsize=15)
plt.legend(fontsize=15)
plt.savefig('./Plots/primal_orth_gamma_{}_lbd_{}_alpha_{}_sigma_{}.png'.format(gamma, lbd, alpha, sigma))

# bias-var tradeoff
primal_bia_var = np.zeros((100, 3))
for i in range(100):
    xi = xi_seq_2[i]
    primal_bia_var[i, :] = primal_orth(lbd, gamma, xi, alpha, sigma, 1)

# Figure 3, MSE of primal sketching
MSE_ridge, bias_ridge, var_ridge = MSE_original(lbd, gamma, alpha, sigma, 1)
plt.plot(xi_seq_2, primal_bia_var[:, 1] / bias_ridge, label='Bias^2', lw=4)
plt.plot(xi_seq_2, primal_bia_var[:, 2] / var_ridge, label='Var', ls='-.', lw=4)
plt.grid(linestyle='dotted')
plt.legend(fontsize=16)
plt.xlabel(r'$\xi$', fontsize=16)
plt.title('Primal sketching bias and variance', fontsize=16)
plt.savefig('./Plots/primal_bia_var_gamma_{}_lbd_{}_alpha_{}_sigma_{}.png'.format(gamma, lbd, alpha, sigma))


# dual sketch
n = 500
gamma = 1.5
p = int(n * gamma)
alpha = 3
sigma = 1
num_steps = 20
zeta_seq = np.linspace(0.1, gamma, 20)
rep = 10
simu_dual = np.zeros((num_steps, rep))
lbd = 1
for i in range(20):
    zeta = zeta_seq[i]
    d = int(n * zeta)
    for k in range(rep):
        X = np.random.randn(n, p)
        beta = np.random.randn(p, 1) * alpha / sqrt(p)
        epsilon = np.random.randn(n, 1) * sigma
        Y = X @ beta + epsilon
        R = generate_haar_matrix(p, d)
        beta_dual = X.T / n @ np.linalg.inv(X @ R @ R.T @ X.T / n + lbd * np.identity(n)) @ Y
        beta_ridge = X.T / n @ np.linalg.inv(X @ X.T / n + lbd * np.identity(n)) @ Y
        simu_dual[i, k] = norm(beta_dual - beta) ** 2 / norm(beta_ridge - beta) ** 2

theory_dual = np.zeros(100)
zeta_seq_2 = np.linspace(0.1, gamma, 100)
for i in range(100):
    zeta = zeta_seq_2[i]
    theory_dual[i] = dual_orth(lbd, gamma, zeta, alpha, sigma) / MSE_original(lbd, gamma, alpha, sigma)

# Figure, MSE of dual sketching
xx = np.mean(simu_dual, 1)
yerr = np.std(simu_dual, 1)
# plt.figure(0, figsize=(12, 8))
plt.plot(zeta_seq_2, theory_dual, label='Theory', lw=4, ls='--')
plt.errorbar(zeta_seq, xx, yerr, label='Simulation', capsize=3, lw=3)
plt.grid(linestyle='dotted')
plt.xlabel(r'$\zeta$', fontsize=14)
plt.ylabel(r'$MSE(\hat\beta_d)/MSE(\hat\beta)$', fontsize=14)
plt.title('Dual sketching MSE', fontsize=14)
plt.legend(fontsize=14)
plt.savefig('./Plots/dual_orth_gamma_{}_alpha_{}_sigma_{}.png'.format(gamma, alpha, sigma))

# bias-var tradeoff
dual_bia_var = np.zeros((100, 3))
for i in range(100):
    zeta = zeta_seq_2[i]
    dual_bia_var[i, :] = dual_orth(lbd, gamma, zeta, alpha, sigma, 1)

MSE_ridge, bias_ridge, var_ridge = MSE_original(lbd, gamma, alpha, sigma, 1)
plt.plot(zeta_seq_2, dual_bia_var[:, 1] / bias_ridge, label='Bias^2', lw=3)
plt.plot(zeta_seq_2, dual_bia_var[:, 2] / var_ridge, label='Var', ls='-.', lw=3)
plt.grid(linestyle='dotted')
plt.legend(fontsize=14)
plt.xlabel(r'$\zeta$', fontsize=14)
plt.title('Dual sketching bias and variance', fontsize=14)
plt.savefig('./Plots/dual_bia_var_gamma_{}_alpha_{}_sigma_{}.png'.format(gamma, alpha, sigma))

# marginal regression
sigma = 1
alpha_seq = np.linspace(0.1, 10, 100)
gamma_seq = [0.7, 1, 3]
marginal_re = np.zeros((3, 100))
for i in range(3):
    gamma = gamma_seq[i]
    for j in range(100):
        alpha = alpha_seq[j]
        lbd = gamma * sigma ** 2 / alpha ** 2 + 1 + gamma
        marginal_re[i, j] = marginal_orth(lbd, gamma, alpha, sigma) / MSE_original(gamma * sigma ** 2 / alpha ** 2, gamma, alpha, sigma)

# Figure 4, MSE of marginal regression
plt.plot(alpha_seq, marginal_re[0, :], label=r'$\gamma={}$'.format(gamma_seq[0]), lw=4)
plt.plot(alpha_seq, marginal_re[1, :], label=r'$\gamma={}$'.format(gamma_seq[1]), lw=4, ls='--')
plt.plot(alpha_seq, marginal_re[2, :], label=r'$\gamma={}$'.format(gamma_seq[2]), lw=4, ls=':')
plt.xlabel(r'SNR', fontsize=14)
plt.ylabel(r'Relative efficiency', fontsize=14)
plt.title('Relative efficiency of marginal regression', fontsize=14)
plt.legend(fontsize=14)
plt.grid(linestyle='dotted')
plt.savefig('./Plots/marginal_re.png')


# full sketch
n = 1000
gamma = 0.1
p = int(n * gamma)
alpha = 1
sigma = 1
c = np.linspace(0.1, 1, 20)
track_full = np.zeros((20, 3))
rep = 10
lbd_seq = [0.1, 0.3, 0.5]
theory_full = np.zeros((100, 3))
for j in range(1):
    lbd = lbd_seq[j]
    for i in range(20):
        xi = c[i]
        m = int(n * xi)
        for k in range(rep):
            X = np.random.randn(n, p)
            beta = np.random.randn(p, 1) * alpha / sqrt(p)
            epsilon = np.random.randn(n, 1) * sigma
            Y = X @ beta + epsilon
            L = generate_haar_matrix(m, n)
            beta_full = np.linalg.inv(X.T @ L.T @ L @ X / n + lbd * np.identity(p)) @ X.T @ L.T @ L @ Y / n
            track_full[i, j] = track_full[i, j] + np.linalg.norm(beta_full - beta) ** 2

xi_seq = np.linspace(0.1, 1, 100)
for j in range(1):
    lbd = lbd_seq[j]
    for i in range(100):
        xi = xi_seq[i]
        theory_full[i, j] = MSE_full(lbd, gamma, xi, alpha, sigma)


for j in [0]:
    lbd = lbd_seq[j]
    plt.scatter(c, track_full[:, j] / rep, label=r'Simulation $\lambda$={}'.format(lbd))
    plt.plot(xi_seq, theory_full[:, j], label=r'Theory $\lambda$={}'.format(lbd))

theory_full = np.zeros((100, 3))
lbd = gamma * sigma ** 2 / alpha ** 2
for i in range(100):
    xi = xi_seq[i]
    theory_full[i, :] = MSE_full(lbd, gamma, xi, alpha, sigma, verbose=1)

# Figure, MSE of full sketching
plt.plot(xi_seq, theory_full[:, 0], label='MSE')
plt.plot(xi_seq, theory_full[:, 1], label='Bias')
plt.plot(xi_seq, theory_full[:, 2], label='Variance')
plt.legend()
plt.legend(fontsize=10)
plt.xlabel(r'$\xi$', fontsize=14)
plt.ylabel(r'$MSE(\hat\beta_p)$', fontsize=14)
plt.title('MSE full sketch', fontsize=14)
plt.grid(linestyle='dotted')
plt.savefig('full_lambda=0.3,0.5.png')
