import numpy as np
import matplotlib as mpl

mpl.use('tkAgg')
import matplotlib.pyplot as plt
from numpy import sqrt
from scipy.integrate import quad
import os

# from sketch_ridge import generate_haar_matrix, theta_1, theta_2, MSE_original, MSE_primal, MSE_dual, MSE_full, \
#     MSE_marginal, MSE, optimal_lambda

n = 1000
gamma = 0.7
p = int(n * gamma)
n_test = 500
alpha = 1
sigma = 1
np.random.seed(45620)
beta = np.random.randn(p, 1) * alpha / np.sqrt(p)
X_train = np.random.randn(n, p)
epsilon_train = np.random.randn(n, 1) * sigma
Y_train = X_train @ beta + epsilon_train

np.random.seed(130)
X_test = np.random.randn(n_test, p)
epsilon_test = np.random.randn(n_test, 1) * sigma
Y_test = X_test @ beta + epsilon_test

K = 5
batch_size = int(n / K)
steps = 20
lbd_seq = np.linspace(0.1, 3, steps)
CV_err = np.zeros((steps, K))
for i in range(steps):
    print(i)
    lbd = lbd_seq[i]
    for j in range(K):
        test_idx = np.arange(j * batch_size, (j + 1) * batch_size, 1, dtype=int)
        X_j = X_train[test_idx, :]
        Y_j = Y_train[test_idx, :]
        train_idx = list(set(np.arange(0, n, 1, dtype=int)) - set(test_idx))
        X = X_train[train_idx, :]
        Y = Y_train[train_idx, :]
        beta_cv = np.linalg.inv(X.T @ X / (n - batch_size) + lbd * np.identity(p)) @ X.T @ Y / (n - batch_size)
        CV_err[i, j] = np.linalg.norm(Y_j - X_j @ beta_cv) ** 2 / batch_size

lbd_cv_idx = np.argmin(np.mean(CV_err, 1))
lbd_cv = lbd_seq[lbd_cv_idx]
lbd_cv_debiased = lbd_cv * (K - 1) / K
lbd_cv_debiased_idx = 0
for i in range(steps):
    if lbd_seq[i] >= lbd_cv_debiased:
        lbd_cv_debiased_idx = i
        break

min_err = np.min(np.mean(CV_err, 1))
one_std_err = min_err + np.sqrt(np.var(CV_err[lbd_cv_idx, :]))
one_std_err_idx = 0
for i in range(steps):
    if np.mean(CV_err, 1)[i] < one_std_err:
        one_std_err_idx = i
        break
one_std_err_lbd = lbd_seq[one_std_err_idx]


X_train_2 = np.random.randn(n, p)
epsilon_train_2 = np.random.randn(n, 1) * sigma
Y_train_2 = X_train_2 @ beta + epsilon_train_2
test_err = np.zeros(steps)
for j in range(steps):
    lbd = lbd_seq[j]
    beta_ridge = np.linalg.inv(X_train_2.T @ X_train_2 / n + lbd * np.identity(p)) @ X_train_2.T @ Y_train_2 / n
    # test_err[i, j] = np.linalg.norm(beta_ridge - beta) ** 2
    test_err[j] = np.linalg.norm(Y_test - X_test @ beta_ridge) ** 2 / n_test

lbd_smallest = lbd_seq[np.argmin(test_err)]
print("Improved test error by debiasing: ", test_err[lbd_cv_idx] - test_err[lbd_cv_debiased_idx])

plt.errorbar(lbd_seq, np.mean(CV_err, 1), np.std(CV_err, 1), capsize=2, label='CV errorbar')
# plt.plot(lbd_seq, np.quantile(CV_err, q=0.95,axis=1))
# plt.plot(lbd_seq, np.quantile(CV_err, q=0.05,axis=1))
# plt.plot(lbd_seq, np.mean(CV_err, axis=1))
plt.plot(lbd_seq, test_err, label='Test error', linewidth=2)
plt.plot(one_std_err_lbd * np.ones(10), np.linspace(1.4, 2.2, 10), ls=':', color='red', linewidth=2, label='One-Std')
# plt.plot(lbd_seq, np.mean(CV_err, 1)[one_std_err_idx] * np.ones(steps), ls=':', color='red', linewidth=2)
plt.plot(lbd_cv * np.ones(10), np.linspace(1.4, 2.2, 10), ls='--', linewidth=3, label='CV min')
plt.plot(lbd_cv_debiased * np.ones(10), np.linspace(1.4, 2.2, 10), ls='-.', linewidth=3, label='Debiased CV min')
plt.plot(lbd_smallest * np.ones(10), np.linspace(1.4, 2.2, 10), ls='-.', label='Test error min', linewidth=3)
plt.grid(linestyle='dotted')
plt.xlabel(r'$\lambda$', fontsize=14)
plt.ylabel('CV test error', fontsize=14)
plt.title(r'CV test error, $\gamma={}$'.format(gamma), fontsize=14)
plt.legend(fontsize=12)
plt.savefig('Plots/CV_test_error_gamma_{}.png'.format(gamma))


plt.plot(lbd_seq, test_err)
plt.plot(lbd_smallest * np.ones(10), np.linspace(1.3, 1.65, 10), ls=':')
plt.plot(one_std_err_lbd * np.ones(10), np.linspace(1.3, 2.5, 10), ls='-.', color='red', linewidth=3)
plt.plot(lbd_cv * np.ones(10), np.linspace(1.3, 2.5, 10), ls='--', linewidth=3)
plt.plot(lbd_cv_debiased * np.ones(10), np.linspace(1.3, 2.5, 10), ls='-.', linewidth=3)

lbd_seq[np.argmin(np.mean(test_err, 0))]
lbd_seq[np.argmin(np.mean(CV_err, 1))]
one_std_err_lbd
