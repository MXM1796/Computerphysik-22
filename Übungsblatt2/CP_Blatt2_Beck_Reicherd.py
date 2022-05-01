import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
a = pd.read_csv("abweichungdat.sec",sep=" ",encoding='ISO-8859-1')
y1=a._get_column_array(1)
x1=a._get_column_array(0)
y=np.log2(y1)

for i in range(0,15):
    y = np.divide(np.power(x1, i), math.factorial(i))
    plt.plot(x1,y)
    plt.show()
