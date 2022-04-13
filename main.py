import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import time






def randomWalk(p, q0, N):                          # Implementierung des randomwalks
    q = []                                         # Erstellen einer leeren Liste
    q.append(q0)                                   # q_0 der Liste hinzufügen


    for n in range(0, N):                          # Schleife von 0 bis N
        i = np.random.randint(0, 8)                # i wird eine zufällige Ganzzahl zugewiesen
        q.append(np.divide(np.add(q[n], p[i]), 3)) # Der Liste wird das Element q_n+1 hinzugefügt

    return q                                       # Die Funktion gibt q_0 bis q_N als Liste zurück


def getN(q, e):                                    # Funktion zum zählen der nicht leeren Raster der Punkte q mit Gitterkonstante e
    raster = np.zeros((int(0.5 / e), int(0.5 / e)))# Das Intervall [0;0]x[0.5;0.5] wird in e*e Quadrate eingeteilt

    for i in q:                                    # Für alle q
        x = int(i[0] / e)                          # Werden die Gitterkoordinaten x und y bestimmt
        y = int(i[1] / e)
        raster[x][y] = 1                           # Das Quadrat in welchem sich der Punkt befindet wird auf 1 gesetzt

    return np.sum(raster)                          # N(e) entspricht dann der Summe der Quadratwerte



def getFractalDim(q, e):                           # Implementierung der Formel (1) mit e_i= 1/2^i
    return np.log(getN(q, e) / getN(q, e / 2)) / (np.log(1 / 2))
t1= time.time()
points = [[0, 0], [0.5, 0], [1, 0], [0, 0.5], [1, 0.5], [0, 1], [0.5, 1], [1, 1]]#Initialisierung der Punkte p_0  bis p_7
q=randomWalk(points,points[0],1000000)#Erzeugen der Punkte q_n

#Erzeugen des Fraktals
a=np.transpose(q)# Zum plotten muss die Matrix transponiert werden
#
plt.figure(figsize=(10, 10))# Bildgröße einstellen
plt.plot(a[0],a[1],"ob" ,ms=1)# Erzeugen des Plots mit Punktgröße ms
plt.xlabel("x")# x-Achse beschriften
plt.ylabel("y")# y-Achse beschriften
#plt.show()     #Zeichnen des Plots

# Erzeugen des Dimensionsplot
i= range(2,9)
e= np.divide(1,np.power(2,i))# Berechnen der Gitterkonstanten von e_2 bis e_8
D=[]
for k in e:# Berchnen der fractalen Dimensionen für vercshiedenen Gitterkonstanten
    D.append(getFractalDim(q,k))


#Erzeugen der analytischen Gerade
plt.figure(200)
x1 = np.linspace(2,8,50);
y1=np.add(x1*0,np.log(8)/np.log(3))
plt.plot(x1,y1,label="anayltisch")

plt.plot(i,D,label="numerisch")#i gegen D auftragen
plt.xlabel("i")# x-Achse beschriften
plt.ylabel("D")# y-Achse berschriften

plt.legend() #Erstellen eienr Legende
t2= time.time()
print(t2-t1)
plt.show() #Zeichnen des Plots


