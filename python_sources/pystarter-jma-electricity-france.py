#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import seaborn as sns
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


nRowsRead = 1000 # specify 'None' if want to read whole file
# free1.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df1 = pd.read_csv('../input/electricity-france/electricity_france.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'electricity_france.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')


# In[ ]:


df1.head(5)


# In[ ]:


ax = sns.scatterplot(x="ActivePower", y="ReactivePower",   data=df1)


# In[ ]:


ax = sns.scatterplot(x="ActivePower", y="Laundry",   data=df1)


# In[ ]:


ax = sns.scatterplot(x="ActivePower", y="Kitchen",   data=df1)
