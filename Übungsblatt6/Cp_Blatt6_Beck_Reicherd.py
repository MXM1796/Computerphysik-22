# kein kompilieren notwendig
# python3 CP_Blatt6_Beck_Reicherd.py
# Mit dem obigen Befehl kann das Programm ausgeführt werden. Es werden dann alle Graphen sowohl angezeigt als auch abgespeichert.
# K. Beck, M. Reicherd
import numpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def QR(alpha, beta):  # QR Iteration
    n = len(alpha)
    tau = np.finfo(float).eps  # Maschinengenauigkeit
    epsilon = 10 ** (-3)  # Wert nahe der Maschinengenauigkeit

    while np.abs(beta[n - 2]) > epsilon:  # solange beta_n-2 führe aus:
        sigma = alpha[n - 1]
        x = alpha[0] - sigma
        y = beta[0]

        for i in range(0, n - 2):
            # print(i)
            if np.abs(x) <= tau * np.abs(y):
                w = -y
                c = 0
                s = 1
            else:
                w = np.sqrt(x ** 2 + y ** 2)
                c = x / w
                s = -y / w

            d = alpha[i] - alpha[i + 1]
            z = (2 * c * beta[i] + d * s) * s

            alpha[i] = alpha[i] - z
            alpha[i + 1] = alpha[i + 1] + z
            beta[i] = d * c * s + (c ** 2 - s ** 2) * beta[i]
            x = beta[i]
            if i > 0:
                beta[i - 1] = w
            if i < n - 2:
                y = -s * beta[i + 1]
                beta[i + 1] = c * beta[i + 1]

    return alpha, beta


def iteration(alpha, beta):  # Vorgeschaltete Iteration
    n = len(alpha)
    # tau = np.finfo(float).eps #Maschinengenauigkeiteig = []
    print(alpha)
    print(beta)
    eig = []
    for j in range(n, 1, -1):  # backward iteration bis 2
        # for i in range(j-3,-1,-1): #backward iteration von j-3 bis 0
        #     if np.abs(beta[i]) < tau: #Stopp, wenn Betabetrag kleiner als Maschinengenauigkeit
        #         print("break is activated")
        #         break
        alpha, beta = QR(alpha[:j], beta[:j - 1])
        eig.append(alpha[j - 1])
    print(eig)

    eig.append(alpha[0])

    return eig


# Definition der Konstanten


v0 = 40000000  # [eV]
b = 6  # [fm]
a = 0.7  # [fm]
h = 4.135 * 10 ** (-15)  # [eVs]
hbar = h/(2*np.pi)

l = [0, 1, 2]  # QZ
Plankmasse = 20.7  # [MeV fm**2]
R = 10000 / b
n = 20  # diskretisierungsabschnitte
r = np.linspace(1, R, n)    # dimensionsloses r
V= -np.divide(v0,1+np.exp((r-b)/a))
c= 299792458
Vschlange = np.divide(np.multiply(V,b**2),Plankmasse)
#print(V)
W= np.divide(1,np.multiply(r,r))+Vschlange
print(W)
diag = np.add(2,np.multiply(W,1))
nebenDiag= np.full(len(diag)-1,-1)
print(diag)
print(nebenDiag)
#W = np.divide(l[0] * (l[0] + 1), r ** 2) - np.divide(b ** 2 * v0,
                                                    # 1 + np.exp((r - b) / a) * Plankmasse)  # effektives Potential

# alpha = (2. + h**2 * W) #Überführe effektives Potential in alpha arr
# beta = np.full(n-1,-1) #Überführe nebendiagonalelemente in beta arr

# Test
alpha = [1., 3., 4.,5,6]  # n=3
beta = [-1., -1. , -1 , -1 ]  # n=2

# a,b = iteration(alpha, beta)

print(a, b)


def QR1(alpha, beta, epsillon):
    print(epsillon)
    tau = np.finfo(float).eps
    n = len(alpha)
    while np.abs(beta[n - 2]) > epsillon:
        sigma = alpha[n - 1]


        x = alpha[0] - sigma
        y = beta[0]
        for i in range(n - 1):

            if np.abs(x) <= tau * np.abs(y):
                omega = -y
                c = 0
                s = 1
            else:
                omega = np.sqrt(x ** 2 + y ** 2)
                c = x / omega
                s = -y / omega
            d = alpha[i] - alpha[i + 1]
            z = (2 * c * beta[i] + d * s) * s
            alpha[i] = alpha[i] - z
            alpha[i + 1] = alpha[i + 1] + z
            beta[i] = d * c * s + (c ** 2 - s ** 2) * beta[i]
            x = beta[i]
            if i > 0:
                beta[i - 1] = omega
            if i < n - 2:
                y = -s * beta[i + 1]
                beta[i + 1] = c * beta[i + 1]
    return [alpha, beta]


def iteration1(alpha, beta, epsillon):
    n=len(alpha)
    tau = np.finfo(float).eps
    for j in range(n,0,-1):


        print(QR1(alpha[:j],beta[:j-1],epsillon))

print(iteration1(alpha,beta,10**-3))
