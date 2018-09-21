import numpy as np
import scipy as sp
from scipy.optimize import minimize
from sklearn import linear_model
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyval


def loss(prediction, target):
    return (prediction-target)**2


def g(W, X):
    #res = polyval(X, W)
    #return res
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
discrepancies = {0:[], 1:[], 2:[], 3:[]}

dim = 1

XT = []
YT = []
for i in range(4000):
    XT.append(np.random.normal(mu_xt, sigma_xt, dim))
    betaT = np.random.normal(mu_beta, sigma_beta, dim)
    RT = np.random.normal(mu_rt, sigma_rt, 1)
    YT.append(np.dot(XT[-1], betaT) + RT)

XT = np.divide(XT-np.min(XT), np.max(XT)-np.min(XT))
YT = np.divide(YT-np.min(YT), np.max(YT)-np.min(YT))

for cnt in range(200,2001,200):
    print('cnt='+str(cnt))
    avg={0:0, 1:0, 2:0, 3:0}
    source_sizes.append(cnt*2)
    for j in range(30):
        print('\tj='+str(j))
        N1 = N2 = cnt
        XS1 = []
        YS1 = []
        for i in range(N1):
            betaS1 = np.random.normal(mu_beta, sigma_beta, dim)
            RS1 = np.random.normal(mu_rs1, sigma_rs1, 1)
            XS1.append(np.random.normal(mu_xs1, sigma_xs1, dim))
            YS1.append(np.dot(XS1[-1], betaS1)+RS1)

        XS1 = np.divide(XS1-np.min(XS1), np.max(XS1)-np.min(XS1))
        YS1 = np.divide(YS1-np.min(YS1), np.max(YS1)-np.min(YS1))


        XS2 = []
        YS2 = []
        for i in range(N2):
            betaS2 = np.random.normal(mu_beta, sigma_beta, dim)
            RS2 = np.random.normal(mu_rs1, sigma_rs1, 1)
            XS2.append(np.random.normal(mu_xs2, sigma_xs2, dim))
            YS2.append(np.dot(XS2[-1], betaS2)+RS2)

        XS2 = np.divide(XS2-np.min(XS2), np.max(XS2)-np.min(XS2))
        YS2 = np.divide(YS2-np.min(YS2), np.max(YS2)-np.min(YS2))

        degree = 3
        for kk in range(len(ws)):
            result = minimize(empirical_risk, np.zeros(dim+1), args=(XS1, XS2, YS1, YS2, ws[kk]))
            ES = result.fun
            W = result.x
            #ES = empirical_risk(W, XS1, XS2, YS1, YS2, ws[2])
            ET = 0
            for i in range(len(XT)):
                ET += loss(g(W, XT[i]), YT[i])
            ET /= len(XT)

            # ES1 = 0
            # for i in range(len(XS1)):
            #     ES1 += loss(g(W, XS1[i]), YS1[i])
            # for i in range(len(XS2)):
            #     ES1 += loss(g(W, XS2[i]), YS2[i])
            # ES1 /= (2*len(XS1))

            avg[kk] += np.abs(ES-ET)
            print("Result:", ES, ET, np.abs(ES-ET))
            # plt.scatter(XT, YT, label='T')
            #
            # plt.scatter(XS1, YS1, label='S1')
            # plt.scatter(XS2, YS2, label='S2')
            # xx = np.arange(-4,4,0.01)
            # yy = []
            # for i in range(len(xx)):
            #    yy.append(g(W, [xx[i]]))
            # plt.plot(xx, yy)
            # plt.legend()
            # plt.show()

            temp=0

    for kk in range(len(ws)):
        print("w=",ws[kk],"mean discrepancy: ",avg[kk]/30)
        discrepancies[kk].append(avg[kk]/30)

for kk in range(len(ws)):
    plt.plot(source_sizes, discrepancies[kk], label='W='+str(ws[kk]))
plt.legend()
plt.show()
tmp = 0
