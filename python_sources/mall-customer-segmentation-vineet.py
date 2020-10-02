#!/usr/bin/env python
# coding: utf-8

# # Mall Customer Segmentation_Vineet

# ## Mall Customer Segmentation using Gaussian Mixture Model

# ### i)    Read dataset and rename columns appropriately
# ### ii)   Drop customerid column and also transform Gender column to [0,1]
# ### iii)  Use seaborn to understand each feature and relationships among features.
# ### iv)  Use sklearn's StandardScaler() to scale dataset
# ### v)   Perform clustering using Gaussian Mixture Modeling.
# ### vi)  Use aic and bic measures to draw a scree plot and discover ideal number of clusters
# ### viii) Lookup anomalous customers and try to understand their behavior.

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer

import re
import warnings
warnings.filterwarnings("ignore")
import os


# In[ ]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[ ]:


os.chdir('/kaggle/input/customer-segmentation-tutorial-in-python/')


# In[ ]:


os.listdir()


# In[ ]:


df=pd.read_csv("Mall_Customers.csv")


# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


df.shape
df.dtypes


# In[ ]:


df.rename({'CustomerID':'Customer_ID',
           'Annual Income (k$)':'Annual_Income',
           'Spending Score (1-100)':'Spending_Score'},
           axis=1,
           inplace=True)


# In[ ]:


df.columns


# In[ ]:


df.drop(columns={'Customer_ID'}, inplace=True)


# In[ ]:


df.shape
df.columns


# In[ ]:


df.Gender.value_counts()


# In[ ]:


df.Gender[df.Gender == 'Male'] = 1
df.Gender[df.Gender == 'Female'] = 0
# Male=1, Female=0
df.head()
df.describe()


# In[ ]:


df["Age_cat"] = pd.cut(
                       df['Age'],
                       bins = [0,35,50,80],
                       labels= ["y", "m", "s"]
                      )


# In[ ]:


df["Annual_Income_cat"] = pd.cut(
                               df['Annual_Income'],
                               bins = [0,40,80,150],
                               labels= ["l", "m", "h"]
                               )


# In[ ]:


df["Spending_Score_cat"] = pd.cut(
                               df['Spending_Score'],
                               bins = 3,
                               labels= ["Ls", "Ms", "Hs"]
                               )


# In[ ]:


df.sample(n=10)


# In[ ]:


columns = ['Gender', 'Age', 'Annual_Income', 'Spending_Score']
fig = plt.figure(figsize = (10,10))
for i in range(len(columns)):
    plt.subplot(2,2,i+1)
    sns.distplot(df[columns[i]])


# In[ ]:


fig = plt.figure(figsize = (10,8))
sns.barplot(x = 'Gender',
            y = 'Spending_Score',
            hue = 'Age_cat',       # Age-cat wise plots
            estimator = np.mean,
            ci = 68,
            data =df)


# In[ ]:


sns.boxplot(x = 'Age',                 
            y = 'Spending_Score',
            data = df
            )


# In[ ]:


sns.boxplot(x = 'Annual_Income',
            y = 'Age', 
            data = df
            )


# In[ ]:


sns.jointplot(df.Age, df.Spending_Score,kind = "kde")


# In[ ]:


sns.jointplot(df.Age, df.Annual_Income,kind="hex")


# In[ ]:


sns.barplot(x = 'Annual_Income',
            y = 'Spending_Score',
            estimator = np.mean,
            ci = 95,
            data =df
            )


# In[ ]:


df.columns


# In[ ]:


grouped = df.groupby(['Gender', 'Age_cat'])
df_wh = grouped['Spending_Score'].sum().unstack()
df_wh

sns.heatmap(df_wh)


# In[ ]:


grouped = df.groupby(['Gender', 'Age_cat'])
df_wh = grouped['Annual_Income'].sum().unstack()
df_wh

sns.heatmap(df_wh)


# In[ ]:


grouped = df.groupby(['Age_cat','Spending_Score_cat'])
df_wq = grouped['Annual_Income'].sum().unstack()
sns.heatmap(df_wq, cmap = plt.cm.Spectral)


# In[ ]:


sns.catplot(x = 'Spending_Score',
            y = 'Age', 
            row = 'Spending_Score_cat',
            col = 'Age_cat' ,
            kind = 'box',
            estimator = np.sum,
            data = df)


# In[ ]:


sns.relplot(x = 'Annual_Income',
            y = 'Spending_Score', 
            col = 'Age_cat' ,
            kind = 'line',
            estimator = np.sum,
            data = df)


# ### Split Dataset 

# In[ ]:


df.dtypes
df.shape


# In[ ]:


y=df['Spending_Score'].values


# In[ ]:


num1=df.select_dtypes('int64').copy()


# In[ ]:


num1.shape
num1.head()


# In[ ]:


ss=StandardScaler()


# In[ ]:


ss.fit(num1)


# In[ ]:


X=ss.transform(num1)


# In[ ]:


X[:5,]


# ### Perform Clustering using GaussianMixtureModeling

# In[ ]:


gm=GaussianMixture(n_components=3,
                   n_init=10,
                   max_iter=100)


# In[ ]:


gm.fit(X)


# In[ ]:


gm.means_


# In[ ]:


gm.converged_


# In[ ]:


gm.n_iter_


# In[ ]:


gm.predict(X)


# In[ ]:


gm.weights_


# In[ ]:


np.unique(gm.predict(X), return_counts = True)[1]/len(X)


# In[ ]:


gm.sample()


# In[ ]:


fig=plt.figure()


# In[ ]:


plt.scatter(X[:,0],X[:,1],c=gm.predict(X),s=2)


# In[ ]:


plt.scatter(gm.means_[:, 0], gm.means_[:, 1],
            marker='v',
            s=5,               # marker size
            linewidths=5,      # linewidth of marker edges
            color='red'
            )
plt.show()


# In[ ]:


densities=gm.score_samples(X)


# In[ ]:


densities


# In[ ]:


density_threshold=np.percentile(densities,4)


# In[ ]:


density_threshold


# ### Discover ideal number of Clusters - Using aic and bic measures and Scree plot

# In[ ]:


bic = []
aic = []


# In[ ]:


for i in range(8):
    gm = GaussianMixture(
                     n_components = i+1,
                     n_init = 10,
                     max_iter = 100)
    gm.fit(X)
    bic.append(gm.bic(X))
    aic.append(gm.aic(X))


# In[ ]:


fig = plt.figure()
plt.plot([1,2,3,4,5,6,7,8], aic)
plt.plot([1,2,3,4,5,6,7,8], bic)
plt.show()


# In[ ]:


tsne = TSNE(n_components = 2)
tsne_out = tsne.fit_transform(X)
plt.scatter(tsne_out[:, 0], tsne_out[:, 1],
            marker='x',
            s=50,              # marker size
            linewidths=5,      # linewidth of marker edges
            c=gm.predict(X)   # Colour as per gmm
            )


# ### Anomalies Behaviour observation..

# In[ ]:


anomalies=X[densities<density_threshold]


# In[ ]:


anomalies


# In[ ]:


plt.scatter(X[:, 0], X[:, 1], c = gm.predict(X))


# In[ ]:


plt.scatter(anomalies[:, 0], anomalies[:, 1],
            marker='x',
            s=50,               # marker size
            linewidths=5,      # linewidth of marker edges
            color='red'
            )
plt.show()


# In[ ]:


unanomalies = X[densities >= density_threshold]


# In[ ]:


unanomalies.shape


# In[ ]:


df_anomalies = pd.DataFrame(anomalies, columns = ['w','x', 'y'])
df_anomalies['z'] = 'anomalous'   # Create a IIIrd constant column
df_normal = pd.DataFrame(unanomalies, columns = ['w','x','y'])
df_normal['z'] = 'unanomalous'


# In[ ]:


sns.distplot(df_anomalies['w'])
sns.distplot(df_normal['w'])


# In[ ]:


df = pd.concat([df_anomalies,df_normal])


# In[ ]:


sns.boxplot(x = df['z'], y = df['x'])
sns.boxplot(x = df['z'], y = df['w'])


# ## - Thanks...
