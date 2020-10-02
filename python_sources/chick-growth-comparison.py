#!/usr/bin/env python
# coding: utf-8

# In this notebook i will try to compare all the chicks at the beginning (t=0) with the last recorded (t=21)

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
path ='../input/weight-vs-age-of-chicks-on-different-diets/ChickWeight.csv'
data = pd.read_csv(path)


# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


#create histogram for a dataset
import matplotlib.pyplot as plt
data.hist(figsize=(14,14), color='maroon', bins=20)
plt.show()


# In[ ]:


plt.figure(figsize=(12,6))
sns.scatterplot(x='Time',y='weight', hue="Diet", size='Diet', data=data)


# In[ ]:


plt.figure(figsize=(20,14))
sns.scatterplot(x='Time',y='weight', hue="Diet",data=data)


# In[ ]:


plt.figure(figsize=(10,7))
sns.swarmplot(x='Diet',y='Time',data=data)


# In[ ]:


df=data
df0 = df[df['Time'] == 0]
df2 = df[df['Time'] == 2]
df4 = df[df['Time'] == 4]
df6 = df[df['Time'] == 6]
df8 = df[df['Time'] == 8]
df10 = df[df['Time'] == 10]
df12 = df[df['Time'] == 12]
df14 = df[df['Time'] == 14]
df16 = df[df['Time'] == 16]
df18 = df[df['Time'] == 18]
df20 = df[df['Time'] == 20]
df21 = df[df['Time'] == 21]


# In[ ]:


df0.head()


# In[ ]:


df21.head()


# In[ ]:


#left figure is chicken's first weight, the right figure is after receiving 21 day of the diet
fig, ax =plt.subplots(1,2)
sns.swarmplot(x='Diet',y='weight',data=df0, ax=ax[0])
sns.swarmplot(x='Diet',y='weight',data=df21, ax=ax[1])
fig.show()


# from this swarmplot we can see that diet 3 results in highest growth and it's growth overlap other diet, while diet 1 resulted the worst (eventho it have higher weight base, it falls behind compared to other by the end of last recorded)
