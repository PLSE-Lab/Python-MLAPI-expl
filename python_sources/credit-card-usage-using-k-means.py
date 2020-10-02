#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/CreditCardUsage.csv")


# In[ ]:


df.head(5)


# In[ ]:


df.columns


# In[ ]:


df.describe().T


# In[ ]:


df.info()


# In[ ]:


df.isna().sum()


# In[ ]:


df["MINIMUM_PAYMENTS"]


# In[ ]:


df["MINIMUM_PAYMENTS"].median()


# In[ ]:


df["MINIMUM_PAYMENTS"].fillna(df["MINIMUM_PAYMENTS"].median(),inplace=True)


# In[ ]:


df["CREDIT_LIMIT"]


# In[ ]:


df["CREDIT_LIMIT"].median()


# In[ ]:


df["CREDIT_LIMIT"].fillna(df["CREDIT_LIMIT"].median(),inplace=True)


# In[ ]:


df.duplicated().sum()


# In[ ]:


df.drop(columns={'CUST_ID'}, inplace = True)
df.columns


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


sns.heatmap(df.corr())


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)
df_scaled = pd.DataFrame(scaled_df,columns=df.columns)
df_scaled.head()


# K-Means

# In[ ]:


from sklearn.cluster import KMeans
import pylab as pl
import random


# In[ ]:


k_means = KMeans(init = "k-means++", n_clusters = 8, n_init = 12)


# In[ ]:


k_means.fit(scaled_df)


# In[ ]:


k_means.labels_


# In[ ]:


k_means.cluster_centers_


# In[ ]:


Sum_of_squared_distances = []
K = range(1,21)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(df_scaled)
    Sum_of_squared_distances.append(km.inertia_)
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




