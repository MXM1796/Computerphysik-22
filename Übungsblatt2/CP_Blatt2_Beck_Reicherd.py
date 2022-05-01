import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

a = pd.read_csv("abweichungdat.sec", sep=" ", encoding='ISO-8859-1')
y1 = a._get_column_array(1)
x1 = a._get_column_array(0)
# y=np.log2(y1)

# plt.plot(x1**0.3333, y1)
# plt.show()
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
