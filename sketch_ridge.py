import numpy as np
import matplotlib as mpl

mpl.use('tkAgg')
import matplotlib.pyplot as plt
from numpy import sqrt
from scipy.integrate import quad
import os


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
    bias = alpha ** 2 * ((lbd - 1) ** 2 + gamma) / lbd ** 2
    variance = gamma * sigma ** 2 / lbd ** 2
    if verbose == 0:
        return bias + variance
    else:
        return bias + variance, bias, variance


def MSE_full(lbd, gamma, xi, alpha=1, sigma=1, verbose=0):
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
    if method == 'primal':
        return MSE_primal(lbd, gamma, xi, alpha, sigma, verbose)
    if method == 'dual':
        return MSE_dual(lbd, gamma, xi, alpha, sigma, verbose)
    if method == 'full':
        return MSE_full(lbd, gamma, xi, alpha, sigma, verbose)
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


gamma = 0.8
zeta = 0.5
xi = 0.9
alpha = 1
sigma = 1
lbd_seq = np.linspace(0.1, 3, 100)
zeta_seq = np.linspace(0.1, 1, 20)
xi_seq = np.linspace(0.1, 1, 20)
gamma_seq = np.linspace(0.1, 1, 20)
optim_zeta = np.zeros(20)
optim_xi = np.zeros(20)
optim_gamma = np.zeros(20)

for i in range(20):
    gamma = gamma_seq[i]
    mse = np.inf
    for j in range(100):
        lbd = lbd_seq[j]
        a = MSE_primal(gamma, xi, lbd, alpha, sigma)
        if a < mse:
            mse = a
            flag = j
    optim_gamma[i] = lbd_seq[flag]


for i in range(20):
    xi = xi_seq[i]
    mse = np.inf
    for j in range(100):
        lbd = lbd_seq[j]
        a = MSE_primal(gamma, xi, lbd, alpha, sigma)
        if a < mse:
            mse = a
            flag = j
    optim_xi[i] = lbd_seq[flag]


for i in range(20):
    zeta = zeta_seq[i]
    mse = np.inf
    for j in range(100):
        lbd = lbd_seq[j]
        a = MSE_dual(gamma, zeta, lbd, alpha, sigma)
        if a < mse:
            mse = a
            flag = j
        optim_zeta[i] = lbd_seq[flag]

plt.plot(gamma_seq, optim_gamma, label=r'$\gamma$')
plt.plot(xi_seq, optim_xi, label=r'$\xi$')
plt.plot(zeta_seq, optim_zeta, label=r'$\zeta$')
plt.legend()
plt.grid(linestyle='dotted')
plt.ylabel(r'$\lambda^*$')
plt.savefig('optim_lambda.png')
# plt.plot(gamma_seq, 1.42 * gamma_seq + 0.9, '--')

print(np.argmin(mse), lbd_seq[np.argmin(mse)])
optim_lambda = gamma * sigma ** 2 / alpha ** 2
plt.plot(lbd_seq, mse)
plt.plot(optim_lambda, MSE_dual(gamma, xi, optim_lambda, alpha, sigma), 'x')

gamma = 0.5
alpha = 1
sigma = 1.5
d = np.linspace(0.1, 3, 100)
theory = np.zeros(100)
for i in range(100):
    lbd = d[i]
    theory[i] = (alpha ** 2 * lbd ** 2 - lbd * gamma * sigma ** 2) * theta_2(gamma, lbd) + gamma * sigma ** 2 * theta_1(
        gamma, lbd)
plt.plot(d, theory)

i = 2
gamma = 0.2
lbd = 1
quad(MP_moment, (1 - sqrt(gamma)) ** 2, (1 + sqrt(gamma)) ** 2, args=(i, gamma, lbd))

# check the computation of theta_1
for gamma in np.linspace(0.1, 1, 10):
    print(
        quad(MP_moment, (1 - sqrt(gamma)) ** 2, (1 + sqrt(gamma)) ** 2, args=(2, gamma, lbd))[0] - theta_2(gamma, lbd))

# original estimator, vary gamma
n = 700
rep = 20
c = np.linspace(0.5, 2, 20)
alpha = 1
sigma = 1
lbd_seq = [0.2, 1, 2]
track_original = np.zeros((20, 3))
theory_original = np.zeros((100, 3))
for j in range(3):
    print(j)
    lbd = lbd_seq[j]
    for i in range(20):
        gamma = c[i]
        p = int(n * gamma)
        for k in range(rep):
            X = np.random.randn(n, p)
            beta = np.random.randn(p, 1) * alpha / sqrt(p)
            epsilon = np.random.randn(n, 1) * sigma
            Y = X @ beta + epsilon
            beta_full = np.linalg.inv(X.T @ X / n + lbd * np.identity(p)) @ X.T @ Y / n
            track_original[i, j] = track_original[i, j] + np.linalg.norm(beta_full - beta) ** 2

d = np.linspace(0.5, 2, 100)
for j in range(3):
    lbd = lbd_seq[j]
    for i in range(100):
        gamma = d[i]
        theory_original[i, j] = MSE_original(gamma, lbd, alpha, sigma)
        # theta_1 = quad(MP_moment, (1 - sqrt(gamma)) ** 2, (1 + sqrt(gamma)) ** 2, args=(1, gamma, lbd))[0]
        # theta_2 = quad(MP_moment, (1 - sqrt(gamma)) ** 2, (1 + sqrt(gamma)) ** 2, args=(2, gamma, lbd))[0]
        # theory_original[i, j] = alpha ** 2 * lbd ** 2 * theta_2 + gamma * sigma ** 2 * (theta_1 - lbd * theta_2)

# plot for original estimator, vary gamma
for j in range(3):
    lbd = lbd_seq[j]
    plt.scatter(c, track_original[:, j] / rep, label='Simulation, $\lambda$={}'.format(lbd))
    plt.plot(d, theory_original[:, j], label='Theory, $\lambda=${}'.format(lbd))

plt.xlabel(r'$\gamma$', fontsize=14)
plt.ylabel(r'$MSE(\hat\beta)$', fontsize=14)
plt.grid(linestyle='dotted')
plt.title('MSE of ridge regression')
plt.legend(fontsize=10)
plt.savefig('msd_original_vary_gamma.png')

# original estimator, vary lambda
n = 1000
rep = 10
c = np.linspace(0.1, 3, 20)
alpha = 1
sigma = 1
gamma_seq = [0.1, 0.3, 0.6]
MSE_simu = np.zeros((20, 3))
theory_original = np.zeros((100, 3))
for j in range(1):
    print(j)
    gamma = gamma_seq[j]
    for i in range(20):
        lbd = c[i]
        p = int(n * gamma)
        for k in range(rep):
            X = np.random.randn(n, p)
            beta = np.random.randn(p, 1) * alpha / sqrt(p)
            epsilon = np.random.randn(n, 1) * sigma
            Y = X @ beta + epsilon
            beta_full = np.linalg.inv(X.T @ X / n + lbd * np.identity(p)) @ X.T @ Y / n
            MSE_simu[i, j] = MSE_simu[i, j] + np.linalg.norm(beta_full - beta) ** 2

d = np.linspace(0.1, 3, 100)
for j in range(3):
    gamma = gamma_seq[j]
    for i in range(100):
        lbd = d[i]
        theory_original[i, j] = MSE_original(gamma, lbd, alpha, sigma)
    # theta_1 = quad(MP_moment, (1 - sqrt(gamma)) ** 2, (1 + sqrt(gamma)) ** 2, args=(1, gamma, lbd))[0]
    # theta_2 = quad(MP_moment, (1 - sqrt(gamma)) ** 2, (1 + sqrt(gamma)) ** 2, args=(2, gamma, lbd))[0]
    # theory_original[i, j] = alpha ** 2 * lbd ** 2 * theta_2 + gamma * sigma ** 2 * (theta_1 - lbd * theta_2)

# plot for original estimator, vary gamma
for j in range(1):
    gamma = gamma_seq[j]
    plt.scatter(c, MSE_simu[:, j] / rep, label='Simulation, $\gamma$={}'.format(gamma))
    plt.plot(d, theory_original[:, j], label='Theory, $\gamma=${}'.format(gamma))

plt.xlabel(r'$\lambda$', fontsize=14)
plt.ylabel(r'$MSE(\hat\beta)$', fontsize=14)
plt.grid(linestyle='dotted')
plt.title('MSE of ridge regression')
plt.legend(fontsize=10)
plt.savefig('msd_original_vary_lambda.png')




def MSE_full(lbd, gamma, xi, alpha=1, sigma=1, verbose=0):
    the_1 = theta_1(lbd/xi, gamma/xi) / xi
    the_2 = theta_2(lbd/xi, gamma/xi) / xi ** 2
    bias = alpha ** 2 * lbd ** 2 * the_2
    variance = gamma * sigma ** 2 * (the_1 - lbd * the_2)
    if verbose == 0:
        return bias + variance
    else:
        return bias + variance, bias, variance


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

# marginal regression
n = 1000
gamma = 0.2
p = int(n * gamma)
alpha = 1
sigma = 1
lbd_seq = np.linspace(1, 3, 20)
track_marginal = np.zeros((20, 2))
rep = 30
theory_marginal = np.zeros((100, 2))
m = 1
for j in range(20):
    lbd = lbd_seq[j]
    for k in range(rep):
        X = np.random.randn(n, p)
        beta = np.random.randn(p, 1) * alpha / sqrt(p)
        epsilon = np.random.randn(n, 1) * sigma
        Y = X @ beta + epsilon
        L = generate_haar_matrix(m, n)
        beta_marginal = 1 / lbd * X.T @ Y / n
        beta_1 = np.linalg.inv(X.T @ L.T @ L @ X / n + lbd * np.identity(p)) @ X.T @ Y / n
        track_marginal[j, 0] = track_marginal[j, 0] + np.linalg.norm(beta_marginal - beta) ** 2
        track_marginal[j, 1] = track_marginal[j, 1] + np.linalg.norm(beta_1 - beta) ** 2

d = np.linspace(1, 3, 100)
for j in range(100):
    lbd = d[j]
    theory_marginal[j, 0] = MSE_marginal(lbd, gamma, alpha, sigma, verbose=0)
    theory_marginal[j, 1] = MSE_primal(gamma=gamma, xi=m / n, lbd=lbd, alpha=alpha, sigma=sigma)


plt.scatter(lbd_seq, track_marginal[:, 0] / rep)
plt.scatter(lbd_seq, track_marginal[:, 1] / rep)
plt.plot(d, theory_marginal[:, 0])
plt.plot(d, theory_marginal[:, 1])

# residual of ridge regression
n = 1000
p = 100
gamma = p / n
rep = 100
lbd_seq = np.linspace(0.001, 2, 20)
alpha = 1
sigma = 1
res_track = np.zeros(20)
res_theory = np.zeros(100)
for i in range(20):
    lbd = lbd_seq[i]
    for k in range(rep):
        X = np.random.randn(n, p)
        beta = np.random.randn(p, 1) * alpha / sqrt(p)
        epsilon = np.random.randn(n, 1) * sigma
        Y = X @ beta + epsilon
        beta_full = np.linalg.inv(X.T @ X / n + lbd * np.identity(p)) @ X.T @ Y / n
        res_track[i] = res_track[i] + np.linalg.norm(Y - X @ beta_full) ** 2 / n / rep

d = np.linspace(0.001, 2, 100)
for i in range(100):
    lbd = d[i]
    res_theory[i] = residual_original(lbd, gamma, alpha, sigma)

plt.plot(d, res_theory, label='Theory')


plt.errorbar(lbd_seq, np.mean(res_track, 1), np.sqrt(np.var(res_track, 1)), capsize=2, label='CV errorbar')
plt.scatter(lbd_seq, np.mean(res_track, 1))

plt.scatter(lbd_seq, res_track, label='Simulation')
plt.plot(d, res_theory, label='Theory')
plt.title('Residual of ridge regression', fontsize=14)
plt.xlabel(r'$\lambda$', fontsize=14)
plt.ylabel('Residual', fontsize=14)
plt.grid(linestyle='dotted')
plt.legend(fontsize=14)
plt.savefig('residual_ridge.png')


