import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
a = pd.read_csv("abweichungdat.sec",sep=" ",encoding='ISO-8859-1')
y1=a._get_column_array(1)
x1=a._get_column_array(0)
y=np.log2(y1)

xn = []
yn = []
for i in range(1,2):
    y = np.divide(np.power(x1, i), math.factorial(i))
    yn.append(y)
    xn.append(x1)
    print(xn,yn)
    plt.plot(x1,y)
    plt.show()


# Teil 2

ZweiPi = 2*np.pi
print(ZweiPi)
x_1 = 0.1
testfunktion = np.cos(ZweiPi*x_1)

# ZweiPunktFormel
