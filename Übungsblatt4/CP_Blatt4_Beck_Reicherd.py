#kein kompilieren notwendig
#python3 CP_Blatt4_Beck_Reicherd.py
#Mit dem obigen Befehl kann das Programm ausgeführt werden. Es werden dann die Graphen angezeigt.
# K. Beck, M. Reicherd

import numpy as np
import numpy.linalg import ql
import matplotlib.pyplot as plt
from math import sin, cos

f = lambda t, x, v: -x -0.1*v -0.03 *x**3 + np.cos(t)
f2= lambda t, x, a: (-x -a-0.03* x**3 + np.cos(t))/0.1

def rungeKutta(f,f2, t0, x0, v0, h, n):
    result = [[t0, x0, v0]]

    for i in range(n):
        a1 = f(result[i][0], result[i][1], result[i][2])
        a2 = 2 * f(result[i][0] + h / 2, result[i][1] + h / 2 * result[i][2], result[i][2] + a1 * h / 2)
        a3 = 2 * f(result[i][0] + h / 2, result[i][1] + h / 2 * result[i][2], result[i][2] + a2 * h / 2)
        a4 = f(result[i][0] + h, result[i][1] + h * result[i][2], result[i][2] + a3 * h)
        a = (a1 + a2 + a3 + a4) / 6

        v1 = f2(result[i][0], result[i][1], a)
        v2 = 2 * f2(result[i][0] + h / 2*v1, result[i][1] + h / 2 *result[i][2], a)
        v3 = 2 * f2(result[i][0] + h / 2*v2, result[i][1] + h / 2 * result[i][2], a )
        v4 = f2(result[i][0] + h, result[i][1] + h*v2, a )
        v = (v1 + v2 + v3 + v4) / 6
        result.append([result[i][0] + h, result[i][1] + h*v, result[i][2] + h * a])

    return np.array(result)


g = rungeKutta(f,f2, 0, 0, 0, 0.01, 100000)
g = g.transpose()


plt.plot(g[0], g[1])

#Phasenverschiebung


plt.show()
