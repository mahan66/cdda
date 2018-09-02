import numpy as np
import scipy as sp
from scipy.optimize import minimize
from sklearn import linear_model
import matplotlib.pyplot as plt


def loss(prediction, target):
    return (prediction-target)**2


def g(W, X):
    return np.dot(W[:len(X)], X)+W[len(X)]


def empirical_risk(W, XS1, XS2, YS1, YS2, w):
    N1 = len(XS1)
    N2 = len(XS2)
    term1 = 0
    term2 = 0
    for i in range(N1):
        term1 += loss(g(W, XS1[i]),YS1[i])
    term1 *= w/N1
    for i in range(N2):
        term2 += loss(g(W, XS2[i]),YS2[i])
    term2 *= (1-w)/N2
    return term1 + term2

K = 2
ws = [0.1,0.3,0.5,0.9]

mu_xt, sigma_xt = 0, 1
mu_beta, sigma_beta = 1, 5
mu_rt, sigma_rt = 0, 0.01
mu_xs1, sigma_xs1 = 0.5, 1
mu_rs1, sigma_rs1 = 0, 0.5
mu_xs2, sigma_xs2 = 2, 0.5

source_sizes = []
discrepancies = []

dim = 1

for cnt in range(200,2001,200):
    print('cnt='+str(cnt))
    avg = 0
    source_sizes.append(cnt*2)
    for j in range(30):
        print('\tj='+str(j))
        N1 = N2 = cnt
        XT = [np.random.normal(mu_xt, sigma_xt, dim) for i in range(4000)]
        beta = np.random.normal(mu_beta, sigma_beta, dim)
        RT = np.random.normal(mu_rt, sigma_rt, 1)
        YT = np.dot(XT, beta)+RT
        XS1 = [np.random.normal(mu_xs1, sigma_xs1, dim) for i in range(N1)]
        RS1 = np.random.normal(mu_rs1, sigma_rs1, 1)
        YS1 = np.dot(XS1, beta)+RS1
        XS2 = [np.random.normal(mu_xs2, sigma_xs2, dim) for i in range(N2)]
        YS2 = np.dot(XS2, beta)+RS1

        result = minimize(empirical_risk, np.zeros(dim+1), args=(XS1, XS2, YS1, YS2, ws[2]))
        ES = result.fun
        W = result.x
        ET = 0
        for i in range(len(XT)):
            ET += loss(g(W, XT[i]), YT[i])
        ET /= len(XT)

        avg += np.abs(ES-ET)
        print("Result:", ES, ET, np.abs(ES-ET))

    discrepancies.append(avg/30)

plt.plot(source_sizes, discrepancies)
plt.show()
tmp = 0
