import numpy as np
import matplotlib.pyplot as plt


class Ball:

    def __init__(self):  # Initialisierung der Werte in Si-Einheiten
        self.v = np.array([0.0, 0.0, 0.0])
        self.x = np.array([0.5, 0.0, 126.5])
        self.a = np.array([0.0, 0.0, 0.0])
        self.w = np.array([0.0, 3.0, 0.0])
        self.mass = 0.625
        self.density = 1.2
        self.radius = 0.12
        self.gravitation = 9.81
        self.cw = 0.45
        self.area = np.pi * self.radius ** 2

    def updateAcceleration(self):  # Berechnung der Kräfte und Beschleunigung
        force = np.array([0.0, 0.0, 0.0])
        force[2] -= self.mass * self.gravitation  # Gravitationskraft
        vAbs = np.linalg.norm(self.v)
        force += (-self.cw * self.area * self.density * vAbs * self.v) / 2.0  # Luftwiederstandskraft
        force += ((4 * np.pi * self.density * self.radius ** 3) / 3) * (np.cross(self.v, self.w))  # Magnuskraft
        self.a = force / self.mass

    def update(self, time):  # Berechnung der Werte für t_n+1
        self.updateAcceleration()
        self.v += self.a * time
        self.x += self.v * time

    def goToZero(self):  # Berechnung des Nullpunkts
        t = -self.x[2] / self.v[2]
        self.x += t * self.v
        self.v += t * self.a
        return t


def createTrajectory():  # Zeichnung der Trajektorie
    N = 10000  # Anzahl Zeitabschnitte
    dT = 12  # Zeitintervall[0,dT]
    b = Ball()
    xValues = []
    yValues = []
    for t in np.linspace(0, dT, N):
        xValues.append(b.x[0])
        yValues.append(b.x[2])
        b.update(dT / N)  # dT/N = Delta t
        if (b.x[2] <= 0):
            print("Fallzeit t=" + str(t - b.goToZero()))
            break

    print("h_x=" + str(b.x))
    print("v=" + str(b.v) + "    |v|=" + str(np.linalg.norm(b.v)))
    plt.xlabel(r"x[m]")
    plt.ylabel(r"z[m]")
    plt.plot(xValues, yValues)
    plt.savefig("Trajektorie")
    plt.show()


def create_h_x_Graph():  # Zeichung von h_x(w)
    N = 10000
    dT = 12
    xValues = []
    yValues = []
    b = Ball()
    for w in range(0, 30):
        b.__init__()
        b.w[1] = w
        for t in np.linspace(0, dT, N):

            b.update(dT / N)
            if (b.x[2] <= 0):
                b.goToZero()
                xValues.append(w)
                yValues.append(b.x[0])
                break
    plt.xlabel(r"$\omega[1/s]$")
    plt.ylabel(r"h_z[m]")
    plt.plot(xValues, yValues)
    plt.savefig("Aufprallort")
    plt.show()


createTrajectory()
create_h_x_Graph()
