import numpy as np
from numpy import sqrt
from numpy.linalg import inv
from numpy import log
from numpy.linalg import norm
import pandas as pd
import matplotlib.pyplot as plt
import os
# os.chdir('/Users/sifanliu/Dropbox/Dual Sketch/experiments')

n = 300
p = 100
gamma = p / n
alpha = 1
sigma = 0
num_steps = 50
zeta_seq = np.linspace(1e-6, 5, num_steps)
rep = 10
dual = np.zeros((rep, num_steps))
relu = np.zeros((rep, num_steps))
ridge = np.zeros((rep, num_steps))
lbd = 1e-3

# Gaussian dual projection
for i in range(num_steps):
    print(i)
    zeta = zeta_seq[i]
    d = int(n * zeta)
    for k in range(rep):
        X = np.random.randn(n, p)
        # diag = sqrt(np.diag(X @ X.T))
        # X = X / diag.reshape(n, 1) * sqrt(p)
        beta = np.random.randn(p, 1)
        beta = beta / sqrt(p)
        sigma = 0
        epsilon = np.random.randn(n, 1) * sigma
        Y = X @ beta + epsilon
        beta_ridge = X.T / n @ inv(X @ X.T / n + lbd * np.identity(n)) @ Y

        R = np.random.randn(p, d)
        # diag = sqrt(np.diag(R @ R.T))
        # R = R / diag.reshape(p, 1) * sqrt(p)
        # R = generate_haar_matrix(p, d)
        beta_dual = R.T @ X.T / n @ inv(X @ R @ R.T @ X.T / n + lbd * zeta / gamma * np.identity(n)) @ Y

        X_relu = np.maximum(X @ R, np.zeros((n, d)))
        beta_relu = X_relu.T / n @ inv(X_relu @ X_relu.T / n + lbd * zeta / gamma * np.identity(n)) @ Y
        # dual_simu_gaus[k, i] = norm(beta_dual - beta) ** 2

        X_test = np.random.randn(n, p)
        # diag = np.diag(X_test @ X_test.T)
        # X_test = X_test / diag.reshape(n, 1) * sqrt(p)
        epsilon_test = np.random.randn(n, 1) * sigma
        Y_test = X_test @ beta + epsilon_test
        dual[k, i] = norm(Y_test - X_test @ R @ beta_dual) ** 2 / n
        relu[k, i] = norm(Y_test - np.maximum(X_test @ R, np.zeros((n, d))) @ beta_relu) ** 2 / n
        ridge[k, i] = norm(Y_test - X_test @ beta_ridge) ** 2 / n


plt.errorbar(zeta_seq, np.mean(dual, axis=0), np.std(dual, axis=0), capsize=2, lw=2, label='linear')
plt.errorbar(zeta_seq, np.mean(relu, axis=0), np.std(relu, axis=0), capsize=2, lw=2, label='ReLU')
# plt.errorbar(zeta_seq, np.mean(ridge, axis=0), np.std(ridge, axis=0), capsize=2, lw=2, label='ridge')
plt.legend()
plt.grid(linestyle='dotted')
plt.title(r"$\gamma={:.2f},\alpha={},\sigma={}$".format(gamma, alpha, sigma))
plt.xlabel(r'$\zeta$')
plt.ylabel("Test error")
plt.savefig("double_descent_gamma_{:.2f}_alpha_{}_sigma_{}.png".format(gamma, alpha, sigma))


