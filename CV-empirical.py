import matplotlib as mpl
import numpy as np
import pandas as pd
from numpy import sqrt

mpl.use('tkAgg')
import matplotlib.pyplot as plt

# import data
m = 100000
msd = pd.read_table("/Users/sifanliu/Dropbox/Random Projection/Experiments/old/real_data/YearPredictionMSD.txt",
                    delimiter=',', nrows=100000).as_matrix()
flt = pd.read_csv(
    '/Users/sifanliu/Dropbox/Random Projection/Experiments/old/real_data/nycflight/nycflight.csv').as_matrix()
flt = flt[:, 1:]


# standardize data
def standardize(Y):
    return (Y - np.mean(Y)) / np.std(Y)


# msd
m = msd.shape[0]
p = msd.shape[1] - 1
n = 1000
gamma = p / n
for i in range(msd.shape[1]):
    msd[:, i] = standardize(msd[:, i])
np.random.seed(130)
msd = np.random.permutation(msd)

K = 10
batch_size = n / K
steps = 20
lbd_seq = np.linspace(0, 0.6, steps)
cv_error = np.zeros((steps, K))
rep = 100

for k in range(rep):
    print(k)
    idx = np.random.choice(m, n, replace=False)
    X = msd[idx, 1:]
    Y = msd[idx, 0].reshape(n, 1)
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
            cv_error[i, j] = cv_error[i, j] + np.linalg.norm(Y_test - X_test @ beta_hat) ** 2 / batch_size / rep

lbd_cv_idx = np.argmin(np.mean(cv_error, 1))
lbd_cv = lbd_seq[lbd_cv_idx]
lbd_cv_debiased = lbd_cv * (K - 1) / K
lbd_cv_debiased_idx = 0
for i in range(steps):
    if lbd_seq[i] >= lbd_cv_debiased:
        lbd_cv_debiased_idx = i
        break

xx = np.mean(cv_error, 1)
yerr = np.std(cv_error, 1)
plt.errorbar(lbd_seq, xx, yerr)

