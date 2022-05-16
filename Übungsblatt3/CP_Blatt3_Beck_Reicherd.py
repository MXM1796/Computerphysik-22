import numpy as np
import matplotlib.pyplot as plt
class Ball:

    def __init__(self):
        self.v = np.array([0.0, 0.0, 0.0])
        self.x = np.array([0.0, 126.5, 0.5])
        self.w = np.array([3.0, 0.0, 0.0])
        self.mass = 0.625
        self.density = 1.1
        self.radius = 0.12
        self.gravitation = 9.81
        self.cw = 0.09
        self.area = np.pi * self.radius ** 2




    def getAcceleration(self):
        force = np.array([0.0,0.0,0.0])
        force[1] -= self.mass * self.gravitation # Gravitationskraft
        vAbs= np.linalg.norm(self.v)
        force += (-self.cw * self.area*self.density*vAbs*self.v)/2.0 # Luftwiederstand
        force += ((4*np.pi*self.density*self.radius**3)/3)*(np.cross(self.v,self.w))#Magnus
        return force/self.mass

    def update(self, time):
        self.v += self.getAcceleration() * time
        self.x += self.v * time


b=Ball()
xValues=[]
yValues=[]
for t in np.linspace(0,11,10000):

    b.update(11/10000)
    xValues.append(b.x[2])
    yValues.append(b.x[1])
    if b.x[1]<=0:
        print(t)
        print(b.x[2])
        break

plt.plot(xValues,yValues)
plt.show()