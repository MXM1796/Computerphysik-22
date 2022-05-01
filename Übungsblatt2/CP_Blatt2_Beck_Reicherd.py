import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
a = pd.read_csv("abweichungdat.sec",sep=" ",encoding='ISO-8859-1')


plt.plot(a._get_column_array(1),a._get_column_array(0))
plt.show()
print()