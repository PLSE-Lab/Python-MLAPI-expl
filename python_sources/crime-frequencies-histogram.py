import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt


df = pd.read_csv('../input/crimes.csv')
df.groupby('Offense').size().plot(kind='bar')
plt.show()