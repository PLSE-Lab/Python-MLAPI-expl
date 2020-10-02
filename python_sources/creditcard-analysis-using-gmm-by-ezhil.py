#!/usr/bin/env python
# coding: utf-8

# ### Created By : Ezhilarasan Kannaiyan
# To analyse the Credit Cards Dataset and use **Gaussian Mixture Model** to cluster the data

# <u>Import appropriate python libraries</u>

# In[ ]:


get_ipython().run_line_magic('reset', '-f')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
import re  #regular expression
from sklearn.preprocessing import StandardScaler
from pandas.plotting import andrews_curves
from mpl_toolkits.mplot3d import Axes3D


# <u>Settings for Display & Running Mode</u>

# In[ ]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[ ]:


pd.options.display.max_rows = 1000
pd.options.display.max_columns = 1000


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# <u>Read the datset</u>

# In[ ]:


cc = pd.read_csv("/kaggle/input/ccdata/CC GENERAL.csv")
cc.shape
cc.head()


# <u>Remove Special characters in Column names </u>

# In[ ]:


cc.columns = [i.lower() for i in cc.columns]
cc.columns


# Drop CustomerID column

# In[ ]:


cc.drop(columns=["cust_id"], inplace=True)


# In[ ]:


cc.head()


# Statistical information

# In[ ]:


cc.describe()


# Check whether any column has null value

# In[ ]:


cc.info()


# In[ ]:


cc.isnull().sum()


# credit_limit and minimum_payments fields have null values. <br/>
# 
# Plot distribution plot for the 2 columns which have null values

# In[ ]:


fig = plt.figure(figsize=(15,5))
ax=plt.subplot(1,2,1)
sns.distplot(cc.credit_limit)
plt.xlim([0,20000])
ax=plt.subplot(1,2,2)
sns.distplot(cc.minimum_payments)
plt.xlim([0,10000])


# Replace null valus with Median of the corresponding column

# In[ ]:


cc.fillna(value = {
                 'minimum_payments' :   cc['minimum_payments'].median(),
                 'credit_limit'               :     cc['credit_limit'].median()
               }, inplace=True)


# In[ ]:


cc.isnull().sum()


# In[ ]:


cc.describe()


# <u>Scaling of the dataset</u>

# In[ ]:


from sklearn.preprocessing import normalize


# In[ ]:


ss =  StandardScaler()
out = ss.fit_transform(cc)
out = normalize(out)
out.shape


# In[ ]:


df_out = pd.DataFrame(out, columns=cc.columns)
df_out.head()


# **Analyse the normalized dataset using seaborn graphs**

# Distribution plots for all the fields in the dataset

# In[ ]:


fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(15,15))
ax = axes.flatten()
fig.tight_layout()
# Do not display 18th, 19th and 20th axes
axes[3,3].set_axis_off()
axes[3,2].set_axis_off()
axes[3,4].set_axis_off()
# Below 'j' is not used.
for i,j in enumerate(df_out.columns):
    sns.distplot(df_out.iloc[:,i], ax = ax[i])


# Distribution plot for the 4 columns : credit_limit, purchases, payments, balance

# In[ ]:


fig = plt.figure(figsize=(15,5))
ax=plt.subplot(2,2,1)
sns.distplot(df_out.credit_limit)
ax=plt.subplot(2,2,2)
sns.distplot(df_out.purchases)
ax=plt.subplot(2,2,3)
sns.distplot(df_out.payments)
ax=plt.subplot(2,2,4)
sns.distplot(df_out.balance)


# Violin plot for the 4 columns : credit_limit, purchases, payments, balance

# In[ ]:


fig = plt.figure(figsize=(15,5))
ax=plt.subplot(2,2,1)
sns.violinplot(df_out.credit_limit)
ax=plt.subplot(2,2,2)
sns.violinplot(df_out.purchases)
ax=plt.subplot(2,2,3)
sns.violinplot(df_out.payments)
ax=plt.subplot(2,2,4)
sns.violinplot(df_out.balance)


# Box plot for the 4 columns : credit_limit, purchases, payments, balance

# In[ ]:


fig = plt.figure(figsize=(15,5))
ax=plt.subplot(1,4,1)
sns.boxplot(y=df_out.credit_limit)
ax=plt.subplot(1,4,2)
sns.boxplot(y=df_out.purchases)
ax=plt.subplot(1,4,3)
sns.boxplot(y=df_out.payments)
ax=plt.subplot(1,4,4)
sns.boxplot(y=df_out.balance)


# Joint plot between credit_limit and purchases

# In[ ]:


sns.jointplot(df_out.credit_limit,df_out.purchases)


# Joint plot between purchases and payments

# In[ ]:


sns.jointplot(df_out.purchases,df_out.payments)


# Pair plots for the 4 columns : credit_limit, purchases, payments, balance

# In[ ]:


sns.pairplot(df_out, vars=["credit_limit","purchases","payments",'balance'])


# Scree Plot to check the number of clusters we could get from the dataset

# In[ ]:


bic = []
aic = []
for i in range(8):
    gm = GaussianMixture(
                     n_components = i+1,
                     n_init = 10,
                     max_iter = 100)
    gm.fit(df_out)
    bic.append(gm.bic(df_out))
    aic.append(gm.aic(df_out))

fig = plt.figure()
plt.plot([1,2,3,4,5,6,7,8], aic)
plt.plot([1,2,3,4,5,6,7,8], bic)
plt.show()


# Based on the scree plot, we can decide number of clusters as 3

# Check the dataset fields and decide which column we can use for visualizing the clusters

# In[ ]:


df_out.columns


# Fit the dataset to GMM model with number of clusters as 3

# In[ ]:


gm = GaussianMixture(n_components = 3,
                     n_init = 10,
                     max_iter = 100)
gm.fit(df_out)


# Draw scatter plot to visualize the GMM output for balance and purchases fields

# In[ ]:


fig = plt.figure()

plt.scatter(df_out.iloc[:, 0], df_out.iloc[:, 2],
            c=gm.predict(df_out),
            s=5)
plt.scatter(gm.means_[:, 0], gm.means_[:, 2],
            marker='v',
            s=10,               # marker size
            linewidths=5,      # linewidth of marker edges
            color='red'
            )
plt.show()


# Draw scatter plot to visualize the GMM output for purchases and payments fields

# In[ ]:


fig = plt.figure()

plt.scatter(df_out.iloc[:, 2], df_out.iloc[:, 13],
            c=gm.predict(df_out),
            s=5)
plt.scatter(gm.means_[:, 2], gm.means_[:, 13],
            marker='v',
            s=10,               # marker size
            linewidths=5,      # linewidth of marker edges
            color='red'
            )
plt.show()


# TSNE Visualization with number of clusters as 3

# In[ ]:


gm = GaussianMixture(
                     n_components = 3,
                     n_init = 10,
                     max_iter = 100)
gm.fit(df_out)

tsne = TSNE(n_components = 2)
tsne_out = tsne.fit_transform(df_out)
plt.scatter(tsne_out[:, 0], tsne_out[:, 1],
            marker='x',
            s=50,              # marker size
            linewidths=5,      # linewidth of marker edges
            c=gm.predict(df_out)   # Colour as per gmm
            )


# Find out Anomalous and Normal points

# In[ ]:


densities = gm.score_samples(df_out)
densities

density_threshold = np.percentile(densities,4)
density_threshold

anomalies = df_out[densities < density_threshold]
anomalies.shape

unanomalies = df_out[densities >= density_threshold]
unanomalies.shape   


# Convert the anomalous and normal points into dataframes 

# In[ ]:


df_anomalies = pd.DataFrame(anomalies)
df_anomalies['type'] = 'anomalous'  
df_normal = pd.DataFrame(unanomalies)
df_normal['type'] = 'unanomalous'

df_anomalies.shape


# In[ ]:


df_anomalies.head()


# In[ ]:


df_normal.head()


# Draw distribution plots for all the columns of df_anomalies and df_normal

# In[ ]:



fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(15,8))
ax = axes.flatten()
fig.tight_layout()
for i in range(15):
    sns.distplot(df_anomalies.iloc[:,i], ax = ax[i],color='b')
    sns.distplot(df_normal.iloc[:,i], ax = ax[i],color='g')


# Get both the dataframes into single one

# In[ ]:


df = pd.concat([df_anomalies,df_normal])
df.head()


# Plot boxplots for the dataframe fields based on the type (anomalous and unanomalous)

# In[ ]:


fig = plt.figure(figsize=(15,5))
ax=plt.subplot(2,2,1)
sns.boxplot(x = df['type'], y = df['balance'])
ax=plt.subplot(2,2,2)
sns.boxplot(x = df['type'], y = df['purchases'])
ax=plt.subplot(2,2,3)
sns.boxplot(x = df['type'], y = df['credit_limit'])
ax=plt.subplot(2,2,4)
sns.boxplot(x = df['type'], y = df['payments'])


# In[ ]:




