#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math as mt
import scipy

from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")


# In[ ]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


nRowsRead = 1000 # specify 'None' if want to read whole file
# free1.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df1 = pd.read_csv('../input/pharmacokinetics-of-theophylline/Theoph.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'Theoph.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')


# In[ ]:


df1.head(5)


# In[ ]:


sns.pairplot(df1)


# In[ ]:


ax = sns.scatterplot(x="Time", y="conc", hue="Dose",  data=df1)


# In[ ]:


ax = sns.scatterplot(x="Time", y="conc", hue="Subject",  data=df1)

