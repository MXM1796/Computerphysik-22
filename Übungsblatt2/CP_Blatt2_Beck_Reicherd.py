import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
a = pd.read_csv("abweichungdat.sec",sep=" ",encoding='ISO-8859-1')
y1=a._get_column_array(1)
x1=a._get_column_array(0)
y=np.divide(y1,x1)
plt.plot(x1,y)
plt.show()
print()