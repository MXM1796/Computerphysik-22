#kein kompilieren notwendig
#python3 CP_Blatt4_Beck_Reicherd.py
#Mit dem obigen Befehl kann das Programm ausgeführt werden. Es werden dann alle Graphen sowohl angezeigt als auch abgespeichert.
# K. Beck, M. Reicherd
import numpy as np
import matplotlib.pyplot as plt
import time


def rungeKutta(f, f2, t0, x0, v0, h, n):
    result = [[t0, x0, v0]]

    for i in range(n):
        a1 = f(result[i][0], result[i][1], result[i][2])
        a2 = 2 * f(result[i][0] + h / 2, result[i][1] + h / 2 * result[i][2], result[i][2] + a1 * h / 2)
        a3 = 2 * f(result[i][0] + h / 2, result[i][1] + h / 2 * result[i][2], result[i][2] + a2 * h / 2)
        a4 = f(result[i][0] + h, result[i][1] + h * result[i][2], result[i][2] + a3 * h)
        a = (a1 + a2 + a3 + a4) / 6 #Berechnung der Beschleunigung

        v1 = f2(result[i][0], result[i][1], a1)
        v2 = 2 * f2(result[i][0] + h / 2 * v1, result[i][1] + h / 2 * result[i][2], a2/2)
        v3 = 2 * f2(result[i][0] + h / 2 * v2, result[i][1] + h / 2 * result[i][2], a3/2)
        v4 = f2(result[i][0] + h, result[i][1] + h * v3, a4)
        v = (v1 + v2 + v3 + v4) / 6#Berechnung der Geschwindigkeit
        result.append([result[i][0] + h, result[i][1] + h * v, result[i][2] + h * a])
        #Hinzufügen der i+1 Werte

    return np.array(result)


def getAmplitude(f, f2, t0, x0, v0, h, n,frequenz): #Berechnet Die Amplitude und die Phasenverschiebung am Ende des Einschwingvorgangs.
    g = rungeKutta(f, f2, t0, x0, v0, h, n)
    T = 2 * np.pi / frequenz
    schritte = int(T / h)
    g = g[n-schritte:n] # Aus dem Array wird genau eine Periode geschnitten
    g = g.transpose()
    c = np.cos(frequenz*g[0])
    maxAmplitude = np.max(g[1])
    phi1 = np.where(c == np.max(c))
    phi2 = np.where(g[1] == maxAmplitude)
    phasenverschiebung= np.abs(g[0][phi1]-g[0][phi2])# Die Phasenverschiebung ergibt sich aus der Differenz der Zeiten bei welchen die Macxima erreicht sind
    return [maxAmplitude,phasenverschiebung]



def getResonanz(b,h,n,d):# Berechnet die Resonanzkurve
    A = []
    P= []
    w1 = np.linspace(0.5, 1.25, 49)# Es Werden mehr Punkte im Resonanzberecih berechnet
    w2 = np.linspace(1.25,1.28,30)
    w3 = np.linspace(1.28,2,30)
    w=np.concatenate((w1,w2,w3))
    if d:# Anderer Bereich für den zweiten Aufgabenteil
        w=np.linspace(0.95,1.3,50)
    x0 = 0
    t0 = 0
    v0 = 0

    for i in w: #für verschiedene Frequenzen A berechen
        f = lambda t, x, v: -x - 0.1 * v - b * x ** 3 + np.cos(i * t) # DGL Nach a umgestellt
        f2 = lambda t, x, a: (-x - a - b * x ** 3 + np.cos(i * t)) / 0.1#DGL Nach v umgestellt

        g = getAmplitude(f, f2, t0, x0, v0, h, n,i)
        A.append(g[0])
        P.append(g[1])
        print(i)


    return [w,A,P]




def plotResFreq(): # Aufgabenteil 2
    res = []

    x = np.linspace(0,0.05,10)
    for b in x:
        a=getResonanz(b,0.02,10000,True)
        m = np.max(a[1])
        i= np.where(a[1] == m)
        print(i)
        res.append(a[0][i[0]])
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$\omega_{max}$")
    plt.plot(x,res)
    plt.savefig("wMaxGegenBeta")
    plt.show()




# Erstellen der Plots
f = lambda t, x, v: -x - 0.1 * v - 0.03 * x ** 3 + np.cos(t)
f2 = lambda t, x, a: (-x - a - 0.03 * x ** 3 + np.cos(t)) / 0.1
b= rungeKutta(f,f2,0,0,0,0.01,20000)
plt.xlabel(r"$t$")
plt.ylabel(r"$x$")
b=b.transpose()
plt.plot(b[0],b[1])
plt.savefig("Bewegungsgleichung")
plt.show()
a=getResonanz(0.03,0.01,20000,False)
plt.xlabel(r"$\omega$")
plt.ylabel(r"$A$")
plt.plot(a[0],a[1])
plt.savefig("ResonanzKurve")
plt.show()
plt.xlabel(r"$\omega$")
plt.ylabel(r"$\Delta\Phi$")
plt.plot(a[0],a[2])
plt.savefig("Phasenverschiebung")
plt.show()
plotResFreq()
