import numpy as np

points = [[0, 0], [0.5, 0], [1, 0], [0, 0.5], [1, 0.5], [0, 1], [0.5, 1], [1, 1]]#Initialisierung der Punkte p_0  bis p_7


def randomWalk(p, q0, N):                          # Implementierung des randomwalks
    q = []                                         # Erstellen einer leeren Liste
    q.append(q0)                                   # q_0 der Liste hinzufügen


    for n in range(0, N):                          # Schleife von 0 bis N
        i = np.random.randint(0, 8)                # i wird eine zufällige Ganzzahl zugewiesen
        q.append(np.divide(np.add(q[n], p[i]), 3)) # Der Liste wird das Element q_n+1 hinzugefügt

    return q                                       # Die funktion gibt q_0 bis q_N als Liste zurück


def getN(q, e):                                    # Funktion zum zählen der nicht leeren Raster der Punkte q mit Gitterkonstante e
    raster = np.zeros((int(0.5 / e), int(0.5 / e)))# Das Intervall [0;0]x[0.5;0.5] wird in e*e Quadrate eingeteilt

    for i in q:                                    # Für alle q
        x = int(i[0] / e)                          # Werden die Gitterkoordinaten x und y bestimmt
        y = int(i[1] / e)
        raster[x][y] = 1                           # Das Quadrat in welchem sich der Punkt befindet wird auf 1 gesetzt

    return np.sum(raster)                          #N(e) entspricht dann der Summe der Quadratwerte



def getFractalDim(q, e):                            # Implementierung der Formel (1) mit e_i= 1/2^i
    return np.log(getN(q, e) / getN(q, e / 2)) / (np.log(1 / 2))



print (getFractalDim(randomWalk(points,points[0],1000),e=1/128))