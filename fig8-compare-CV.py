"""
    Codes for reproducing Figure 8
    Compare different ways to do cross validation
"""

import numpy as np
import matplotlib as mpl

mpl.use('tkAgg')
import matplotlib.pyplot as plt
from numpy import sqrt, log
from numpy.linalg import norm
from numpy.linalg import inv
from scipy.integrate import quad
import os
import time
import pandas as pd
os.chdir('/Users/sifanliu/Dropbox/Dual Sketch/experiments')

n = 1000
p = 1500
gamma = p / n
sigma = 1
alpha = 10
X = np.random.randn(n, p)
beta = np.random.randn(p, 1) * alpha / sqrt(p)
epsilon = np.random.randn(n, 1) * sigma
Y = X @ beta + epsilon
lbd = gamma * sigma ** 2 / alpha ** 2
beta_ridge = np.linalg.inv(X.T @ X + n * lbd * np.identity(p)) @ X.T @ Y
beta_ls = np.linalg.lstsq(Y, X)[0].reshape(p, 1)
shuffel = True


class CV:
    def __init__(self, X, Y, K=2, tt_ratio=0.8, shuffel=False):
        self.X = X
        self.Y = Y
        self.n, self.p = X.shape
        self.K = K
        self.tt_ratio = tt_ratio
        if shuffel:
            data_tilde = np.concatenate([self.X, self.Y], axis=1)
            np.random.shuffle(data_tilde)
            self.X = data_tilde[:, :p]
            self.Y = data_tilde[:, p]

    def kfold(self):
        batch_size = np.floor(self.n / self.K)
        lbd_seq = np.linspace(0.0001, 5, 20)
        err_lbd = np.zeros(20)
        beta_lbd = np.zeros((p, 20))
        for i in range(20):
            lbd = lbd_seq[i]
            err_tot = 0
            beta_cv = np.zeros(p)
            for k in range(self.K):
                idx_valid = np.arange(k * batch_size, (k + 1) * batch_size, 1, dtype=int)
                X_valid = self.X[idx_valid, :]
                Y_valid = self.Y[idx_valid]
                idx_train = list(set(np.arange(0, n, 1, dtype=int)) - set(idx_valid))
                X_train = self.X[idx_train, :]
                Y_train = self.Y[idx_train]
                n_train = X_train.shape[0]
                beta_k = np.linalg.inv(X_train.T @ X_train + n_train * lbd * np.identity(self.p)) @ X_train.T @ Y_train
                beta_cv = beta_cv + beta_k
                pred_err_k = norm(Y_valid - X_valid @ beta_k) ** 2
                err_tot = err_tot + pred_err_k
            err_lbd[i] = err_tot / self.K
            beta_lbd[:, i] = beta_cv / self.K
        min_idx = np.argmin(err_lbd)
        lbd_optim = lbd_seq[min_idx]
        beta_optim = beta_lbd[:, min_idx]
        return lbd_optim, beta_optim.reshape(self.p, 1)

    def train_test(self):
        n_train = np.int(self.n * self.tt_ratio)
        n_test = self.n - n_train
        X_train = self.X[:n_train, :]
        Y_train = self.Y[:n_train]
        X_test = self.X[n_train:, :]
        Y_test = self.Y[n_train:]
        lbd_seq = np.linspace(0.0001, 5, 20)
        err_lbd = np.zeros(20)
        beta_lbd = np.zeros((p, 20))
        for i in range(20):
            lbd = lbd_seq[i]
            beta_train = np.linalg.inv(X_train.T @ X_train + n_train * lbd * np.identity(self.p)) @ X_train.T @ Y_train
            err_lbd[i] = norm(Y_test - X_test @ beta_train) ** 2
            beta_lbd[:, i] = beta_train
        min_idx = np.argmin(err_lbd)
        lbd_optim = lbd_seq[min_idx]
        beta_optim = beta_lbd[:, min_idx]
        return lbd_optim, beta_optim.reshape(self.p, 1)

    def leaveOneOut(self):
        lbd_seq = np.linspace(0.001, 5, 20)
        err_lbd = np.zeros(20)
        for i in range(20):
            lbd = lbd_seq[i]
            M_lbd = inv(self.X.T @ self.X + self.n * lbd * np.identity(self.p))
            beta_lbd = M_lbd @ self.X.T @ self.Y.T
            S_lbd = self.X @ M_lbd @ self.X.T
            dd = 1 - np.diag(S_lbd)
            nn = self.Y - self.X @ beta_lbd
            err_lbd[i] = np.mean((nn / dd) ** 2)
        min_idx = np.argmin(err_lbd)
        lbd_optim = lbd_seq[min_idx]
        return lbd_optim


CV1 = CV(X, Y, K=5, shuffel=True)
lbd_tt, beta_tt = CV1.train_test()
lbd_kfold, beta_kfold = CV1.kfold()
lbd_loo = CV1.leaveOneOut()

# test
X_test = np.random.randn(n, p)
epsilon_test = np.random.randn(n, 1)
Y_test = X_test @ beta + epsilon_test
lbd_seq = np.linspace(0.5, 5, 20)
err_test = np.zeros(20)
mse = np.zeros(20)
for i in range(20):
    lbd = lbd_seq[i]
    beta_lbd = inv(X.T @ X + n * lbd * np.identity(p)) @ X.T @ Y
    mse[i] = norm(beta_lbd - beta) ** 2
    err_test[i] = norm(Y_test - X_test @ beta_lbd) ** 2 / n

min_idx = np.argmin(err_test)
lbd_optim = lbd_seq[min_idx]

plt.plot(lbd_seq, err_test)
plt.plot(lbd_seq, mse)


# compare kfold_beta, kfold_refit, kfold_refit_correct, train_test_beta, train_test_refit, train_test_refit_correct, loo_refit
n = 500
gamma = 1.1
p = np.int(n * gamma)
rep = 10
K = 5
alpha = 20
sigma = 1
tt_ratio = 0.8
result = np.zeros((rep, 7))
names = ['kf avg', 'kf refit', 'kf bic refit', 'tt avg', 'tt refit', 'tt bic refit', 'loo']

# test set

for r in range(rep):
    print(r)
    # train set
    X = np.random.randn(n, p)
    beta = np.random.randn(p, 1) * alpha / sqrt(p)
    epsilon = np.random.randn(n, 1) * sigma
    Y = X @ beta + epsilon
    U, D, Vh = np.linalg.svd(X, full_matrices=False)
    gram = X.T @ X
    corr = X.T @ Y
    CV1 = CV(X, Y, K=K, tt_ratio=tt_ratio, shuffel=True)
    X_test = np.random.randn(n, p)
    epsilon_test = np.random.randn(n, 1)
    Y_test = X_test @ beta + epsilon_test

    lbd_kf, beta_kf = CV1.kfold()
    result[r, 0] = norm(Y_test - X_test @ beta_kf) ** 2 / n
    beta_kf_refit = inv(gram + n * lbd_kf * np.identity(p)) @ corr
    result[r, 1] = norm(Y_test - X_test @ beta_kf_refit) ** 2 / n
    lbd_kf_correct = lbd_kf * (K - 1) / K
    beta_kf_refit_correct = inv(gram + n * lbd_kf_correct * np.identity(p)) @ corr
    result[r, 2] = norm(Y_test - X_test @ beta_kf_refit_correct) ** 2 / n

    lbd_tt, beta_tt = CV1.train_test()
    result[r, 3] = norm(Y_test - X_test @ beta_tt) ** 2 / n
    beta_tt_refit = inv(gram + n * lbd_tt * np.identity(p)) @ X.T @ Y
    result[r, 4] = norm(Y_test - X_test @ beta_tt_refit) ** 2 / n
    lbd_tt_correct = lbd_tt * tt_ratio
    beta_tt_refit_correct = inv(gram + n * lbd_tt_correct * np.identity(p)) @ corr
    result[r, 5] = norm(Y_test - X_test @ beta_tt_refit_correct) ** 2 / n

    lbd_loo = CV1.leaveOneOut()
    beta_loo_refit = inv(X.T @ X + n * lbd_loo * np.identity(p)) @ corr
    result[r, 6] = norm(Y_test - X_test @ beta_loo_refit) ** 2 / n

pd.DataFrame(result).to_csv('choose_lambda_50.csv')
xx = np.mean(result, axis=0)
pd.DataFrame(xx).to_csv('choose_lambda.csv')
xx = pd.DataFrame(xx).to_latex()

yerr = np.std(result, axis=0)
pd.DataFrame({'mean': xx, 'std': yerr}).to_csv('compare_cv.csv')
pd.read_csv('compare_cv.csv')

plt.scatter(np.arange(0, 7), xx)
for i, txt in enumerate(names):
    plt.annotate(txt, (i, xx[i]))
# plt.plot(7, norm(Y_test - X_test @ beta_ridge) ** 2 / n, '+')
# plt.annotate('ridge', (7, norm(Y_test - X_test @ beta_ridge) ** 2 / n))
# plt.plot(8, norm(Y_test) ** 2 / n, '*')
# plt.annotate(8, (8, norm(Y_test) ** 2 / n))
plt.grid(linestyle='dotted')
plt.ylabel('Test error')
plt.title(r'$\gamma={}$, $\alpha={}$, $K={}$, $tt ratio={}$'.format(gamma, alpha, K, tt_ratio))
plt.errorbar(np.arange(0, 7), xx, yerr, capsize=3)
plt.savefig('CV_results/gamma_{}_alpha_{}_K_{}_tt_{}.png'.format(gamma, alpha, K, tt_ratio))
# tags = {'names': ['kfold_beta', 'kfold_refit', 'kfold_refit_correct', 'tt_beta', 'tt_refit', 'tt_refit_correct', 'loo_refit'],
#         'formats': [np.float, np.float, np.float, np.float, np.float, np.float, np.float]}
# result.dtype = tags
np.mean(result.astype(np.float), axis=0)
np.std(result, axis=0)
result = result.view(np.float64).reshape(result.shape + (-1,))


