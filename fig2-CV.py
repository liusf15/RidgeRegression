"""
    Codes for reproducing Figure 2
    K-fold cross-validation on Million Song Dataset and Flight Dataset
"""

import matplotlib as mpl
import numpy as np
import pandas as pd
from numpy import sqrt

mpl.use('tkAgg')
import matplotlib.pyplot as plt

# import data
m = 100000
msd = pd.read_table("/Users/Dropbox/Random Projection/Experiments/old/real_data/YearPredictionMSD.txt",
                    delimiter=',', nrows=100000).as_matrix()
flt = pd.read_csv(
    '/Users/Dropbox/Random Projection/Experiments/old/real_data/nycflight/nycflight.csv').as_matrix()
flt = flt[:, 1:]


# standardize data
def standardize(Y):
    return (Y - np.mean(Y)) / np.std(Y)


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


def MSE_primal(lbd, gamma, xi, alpha=1, sigma=1, verbose=0):
    the_1 = 1 / xi * theta_1(lbd / xi, gamma / xi)
    the_2 = 1 / xi ** 2 * theta_2(lbd / xi, gamma / xi)
    bias = alpha ** 2 * ((lbd + xi - 1) ** 2 + gamma * (1 - xi)) * the_2
    variance = gamma * sigma ** 2 * (the_1 - (lbd + xi - 1) * the_2)
    if verbose == 0:
        return bias + variance
    else:
        return bias + variance, bias, variance


"""
    ridge regression cross validation
"""
# MSD, choose lambda for ridge regression by K-fold cross validation
m = msd.shape[0]
p = msd.shape[1] - 1
for i in range(msd.shape[1]):
    msd[:, i] = standardize(msd[:, i])
np.random.seed(130)
msd = np.random.permutation(msd)
n = 1000
q = int(m / n)
gamma = p / n
K = 5
batch_size = n / K
steps = 40
lbd_seq = np.linspace(0, 0.6, steps)
cv_error = np.zeros((steps, K))


# cross validation
for k in range(q - 10):
    X = msd[n * k: n * (k + 1), 1:]
    Y = msd[n * k: n * (k + 1), 0].reshape(n, 1)
    for i in range(steps):
        lbd = lbd_seq[i]
        for j in range(K):
            test_idx = np.arange(j * batch_size, (j + 1) * batch_size, 1, dtype=int)
            X_test = X[test_idx, :]
            Y_test = Y[test_idx, :]
            train_idx = list(set(np.arange(0, n, 1, dtype=int)) - set(test_idx))
            X_train = X[train_idx, :]
            Y_train = Y[train_idx, :]
            beta_hat = np.linalg.inv(
                X_train.T @ X_train / (n - batch_size) + lbd * np.identity(p)) @ X_train.T @ Y_train / (n - batch_size)
            cv_error[i, j] = cv_error[i, j] + np.linalg.norm(Y_test - X_test @ beta_hat) ** 2 / batch_size / (q - 10)

lbd_cv = lbd_seq[np.argmin(np.mean(cv_error, 1))]
lbd_cv_idx = np.argmin(np.mean(cv_error, 1))
lbd_cv_debiased = lbd_cv * (K - 1) / K
lbd_cv_debiased_idx = 0
for i in range(steps):
    if lbd_seq[i] >= lbd_cv_debiased:
        lbd_cv_debiased_idx = i
        break

# test error
X_train = msd[n * (q - 10): n * (q - 9), 1:]
Y_train = msd[n * (q - 10): n * (q - 9), 0]
X_test = msd[n * (q - 9): n * q, 1:]
Y_test = msd[n * (q - 9): n * q, 0]
test_error = np.zeros(steps)

for i in range(steps):
    lbd = lbd_seq[i]
    beta_ridge = np.linalg.inv(
        X_train.T @ X_train / n + lbd * np.identity(p)) @ X_train.T @ Y_train / n
    test_error[i] = np.linalg.norm(Y_test - X_test @ beta_ridge) ** 2 / (9 * n)

lbd_smallest = lbd_seq[np.argmin(test_error)]
lbd_smallest_idx = np.argmin(test_error)


lbd_theory = gamma
lbd_theory_idx = 0
for i in range(steps):
    if lbd_seq[i] >= lbd_theory:
        lbd_theory_idx = i
        break

lb = np.min(test_error)
ub = np.max(np.mean(cv_error, 1))
plt.errorbar(lbd_seq, np.mean(cv_error, 1), np.sqrt(np.var(cv_error, 1)), capsize=2, label='CV errorbar')
plt.plot(lbd_seq, test_error, label='Test error')
plt.plot(lbd_cv * np.ones(10), np.linspace(lb, ub, 10), ls='--', linewidth=3, label='CV min {:.3f}'.format(test_error[lbd_cv_idx]))
plt.plot(lbd_cv_debiased * np.ones(10), np.linspace(lb, ub, 10), ls='-.', linewidth=3, label='Debiased CV {:.3f}'.format(test_error[lbd_cv_debiased_idx]))
plt.plot(lbd_smallest * np.ones(10), np.linspace(lb, ub, 10), ls=':', label='Test error min {:.3f}'.format(test_error[lbd_smallest_idx]), linewidth=3)
plt.plot(lbd_theory * np.ones(10), np.linspace(lb, ub, 10), ls=':', label='Theory {:.3f}'.format(test_error[lbd_theory_idx]), linewidth=3)
plt.legend(fontsize=13)
plt.grid(linestyle='dotted')
plt.xlabel(r'$\lambda$', fontsize=13)
plt.ylabel('CV test error', fontsize=13)
plt.title('MSD CV', fontsize=13)
plt.savefig("./Plots/CV_MSD.png")

print(test_error[lbd_cv_debiased_idx], test_error[lbd_cv_idx], test_error[lbd_smallest_idx])
print("Debias improve test error by", test_error[lbd_cv_idx] - test_error[lbd_cv_debiased_idx])


# flight dataset
m = flt.shape[0]
p = flt.shape[1] - 1
for i in range(flt.shape[1]):
    flt[:, i] = standardize(flt[:, i])
np.random.seed(130)
flt = np.random.permutation(flt)
n = 300
q = int(m / n)
gamma = p / n
K = 5
batch_size = n / K
steps = 20
cv_primal_flt = np.zeros((steps, K))
xi = 0.5
lbd_seq = np.linspace(0.0001, 0.01, steps)
r = int(n * xi)
cv_error_flt = np.zeros((steps, K))

# cross validation
for k in range(q - 10):
    X = flt[n * k: n * (k + 1), 1:]
    Y = flt[n * k: n * (k + 1), 0].reshape(n, 1)
    for i in range(steps):
        lbd = lbd_seq[i]
        for j in range(K):
            test_idx = np.arange(j * batch_size, (j + 1) * batch_size, 1, dtype=int)
            X_test = X[test_idx, :]
            Y_test = Y[test_idx, :]
            train_idx = list(set(np.arange(0, n, 1, dtype=int)) - set(test_idx))
            X_train = X[train_idx, :]
            Y_train = Y[train_idx, :]
            beta_hat = np.linalg.inv(
                X_train.T @ X_train / (n - batch_size) + lbd * np.identity(p)) @ X_train.T @ Y_train / (n - batch_size)
            cv_error_flt[i, j] = cv_error_flt[i, j] + np.linalg.norm(Y_test - X_test @ beta_hat) ** 2 / batch_size / (q - 10)

lbd_cv = lbd_seq[np.argmin(np.mean(cv_error_flt, 1))]
lbd_cv_idx = np.argmin(np.mean(cv_error_flt, 1))
lbd_cv_debiased = lbd_cv * (K - 1) / K
lbd_cv_debiased_idx = 0
for i in range(steps):
    if lbd_seq[i] >= lbd_cv_debiased:
        lbd_cv_debiased_idx = i
        break

# test error
X_train = flt[n * (q - 10): n * (q - 9), 1:]
Y_train = flt[n * (q - 10): n * (q - 9), 0]
X_test = flt[n * (q - 9): n * q, 1:]
Y_test = flt[n * (q - 9): n * q, 0]
test_error = np.zeros(steps)

for i in range(steps):
    lbd = lbd_seq[i]
    beta_ridge = np.linalg.inv(
        X_train.T @ X_train / n + lbd * np.identity(p)) @ X_train.T @ Y_train / n
    test_error[i] = np.linalg.norm(Y_test - X_test @ beta_ridge) ** 2 / (9 * n)

lbd_smallest = lbd_seq[np.argmin(test_error)]
lbd_smallest_idx = np.argmin(test_error)


lbd_theory = gamma
lbd_theory_idx = 0
for i in range(steps):
    if lbd_seq[i] >= lbd_theory:
        lbd_theory_idx = i
        break


lb = np.min(test_error)
ub = np.max(np.mean(cv_error_flt, 1))
plt.errorbar(lbd_seq, np.mean(cv_error_flt, 1), np.sqrt(np.var(cv_error_flt, 1)), capsize=2, label='CV errorbar')
plt.plot(lbd_seq, test_error, label='Test error')
plt.plot(lbd_cv * np.ones(10), np.linspace(lb, ub, 10), ls='--', linewidth=3, label='CV min {:.3f}'.format(test_error[lbd_cv_idx]))
plt.plot(lbd_cv_debiased * np.ones(10), np.linspace(lb, ub, 10), ls='-.', linewidth=3, label='Debiased CV {:.3f}'.format(test_error[lbd_cv_debiased_idx]))
plt.plot(lbd_smallest * np.ones(10), np.linspace(lb, ub, 10), ls=':', label='Test error min {:.3f}'.format(test_error[lbd_smallest_idx]), linewidth=3)
plt.legend(fontsize=13)
plt.grid(linestyle='dotted')
plt.xlabel(r'$\lambda$', fontsize=13)
plt.ylabel('CV test error', fontsize=13)
plt.title('Flight CV', fontsize=13)
plt.savefig("./Plots/CV_flt.png")

