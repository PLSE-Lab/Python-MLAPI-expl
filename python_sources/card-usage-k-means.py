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


df["MINIMUM_PAYMENTS"].median()


# In[ ]:


df["MINIMUM_PAYMENTS"].fillna(df["MINIMUM_PAYMENTS"].median(),inplace=True)


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


plt.figure(figsize=(8,5))
plt.title("Balance distribution",fontsize=16)
plt.xlabel ("Balance",fontsize=14)
plt.grid(True)
plt.hist(df["BALANCE"],color='blue',edgecolor='k')
plt.show()


# In[ ]:


plt.figure(figsize=(10,7))
plt.ylabel("Balance",fontsize=18)
plt.xlabel("Purchase Frequency",fontsize=18)
plt.scatter(x=df["PURCHASES_FREQUENCY"],y=df["BALANCE"])


# In[ ]:


sns.boxplot(x=df["CREDIT_LIMIT"],y=df["BALANCE"])


# In[ ]:


sns.lineplot(x=df["CREDIT_LIMIT"],y=df["BALANCE"])


# In[ ]:


df.corr()["BALANCE"]


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
import matplotlib.pyplot as plt


# In[ ]:


k_means = KMeans(init = "k-means++", n_clusters = 8, n_init = 12)


# In[ ]:


k_means.fit(scaled_df)


# In[ ]:


k_means_labels = k_means.labels_
k_means_labels


# In[ ]:


k_means_cluster_centers = k_means.cluster_centers_
k_means_cluster_centers


# In[ ]:


k_values = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in k_values]
kmeans
score = [kmeans[i].fit(scaled_df).score(scaled_df) for i in range(len(kmeans))]
score
pl.plot(k_values,score)
pl.xlabel('No. of Clusters')
pl.ylabel('Score')
pl.title('Elbow Curve')
pl.show()


# In[ ]:


k_values = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in k_values]
kmeans
score = [kmeans[i].fit(scaled_df).inertia_ for i in range(len(kmeans))]
score
pl.plot(k_values,score)
plt.vlines(x=7,ymin=0,ymax=160000,linestyles='-')
pl.xlabel('Number of Clusters')
pl.ylabel('Sum of within sum square')
pl.title('Elbow Curve')
pl.show()


# **We can conclude that k value of 7 will be optimal for this dataset.**

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




