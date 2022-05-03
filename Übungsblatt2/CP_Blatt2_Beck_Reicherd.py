import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

a = pd.read_csv("abweichungdat.sec", sep=" ", encoding='ISO-8859-1')
y1 = a._get_column_array(1)
x1 = a._get_column_array(0)
x_v=[]
y_v=[]

x_m=[]
y_m=[]
for i in range(0,y1.size):
    if x1[i]>10**(-3):
        x_v.append(x1[i])
        y_v.append(y1[i])
    else:
        x_m.append(x1[i])
        y_m.append(y1[i])

x_vLog=np.log(x_v)
y_vLog=np.log(y_v)
average_x= np.average(x_vLog)
average_y= np.average(y_vLog)
average_x2= np.average(np.power(x_vLog,2))
average_xy= np.average(np.multiply(x_vLog,y_vLog))
m= (average_xy-average_x*average_y)/(average_x2-average_x**2)
n= (average_x2*average_y-average_x*average_xy)/(average_x2-average_x**2)
print("m="+str(m)+",  n="+str(n))
x_fit=np.arange(np.min(x_vLog),np.max(x_vLog),1)
y_fit=x_fit*m+n
plt.plot(x_fit,y_fit)
plt.scatter(x_vLog,y_vLog,1)
plt.show()
# Teil 2

ZweiPi = 2 * np.pi
w_arr = np.arange(1, 21)
h = np.linspace(1, 2, 5)
print(h)


# Zwei Punkt Formel
def d_y_approx_2(w, h):
    d_y_a_3 = np.divide(np.cos(ZweiPi * (w + h) * (0.1 / (w + h))) - np.cos(ZweiPi * w * (0.1 / w)), ((w + h) - w))
    return d_y_a


def d_y_exakt(w):
    d_y_e = - (ZweiPi * w) * np.sin(ZweiPi * w * (0.1 / w))
    return d_y_e

# drei Punkt Formel
def d_y_approx_3(w, h):
    d_y_a_3 = np.divide(np.cos(ZweiPi * (w + h) * (0.1 / (w + h))) - np.cos(ZweiPi * (w - h) * (0.1 / (w - h))), 2*((w - h)- w))
    return d_y_a_3


def d_y_exakt(w):
    d_y_e = - (ZweiPi * w) * np.sin(ZweiPi * w * (0.1 / w))
    return d_y_e
