#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


pd.options.display.max_columns = None
pd.set_option('display.float_format', lambda x: '%.6f' % x)
from matplotlib import pyplot as plt
plt.style.use('ggplot')
from sklearn.cluster import KMeans
import seaborn as sns
import colorlover as cl
import plotly as py
import plotly.graph_objs as go
py.offline.init_notebook_mode(connected = True)


# In[ ]:


df=pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
df.head(10)


# We can see that there are four variables here. Gender has two sub-categories i.e Male and Female.

# In[ ]:


df=df.rename(columns={'Gender':'gender','Age':'age','Annual Income (K$)':'annual_income','Spending Score (1-100)': 'spending_score'})


# We have changed the column name to avoid the silly mistakes. As gender column has two sub categories, lets replace the Female and Male by 0,1. The conversion is done as other variables are int and gender is in obj format.
# Converting all variables in single format provide ease to the analysis

# In[ ]:


df['gender'].replace(['Female','Male'],[0,1],inplace=True)
df.head()


# In[ ]:


pd.isnull(df).any()


# In[ ]:


df.dtypes


# Scaling data helps to form cluster with more accuracy. Though except gender, all variable are scale lets do the scaling to cope gender with other variables.

# In[ ]:


dfsp = pd.concat([df.mean().to_frame(), df.std().to_frame()], axis=1).transpose()
dfsp.index = ['mean', 'std']
#new dataframe with scaled values
df_scaled = pd.DataFrame()
for c in df.columns:
    if(c=='gender'): df_scaled[c] = df[c]
    else: df_scaled[c] = (df[c] - dfsp.loc['mean', c]) / dfsp.loc['std', c]
df_scaled.head()


# In[ ]:


#the two "intuitive" clusters
dff = df_scaled.loc[df_scaled.gender==0].iloc[:, 1:] #no need of gender column anymore
dfm = df_scaled.loc[df_scaled.gender==1].iloc[:, 1:]


# In[ ]:


def number_of_clusters(df):

    wcss = []
    for i in range(1,20):
        km=KMeans(n_clusters=i, random_state=0)
        km.fit(df)
        wcss.append(km.inertia_)

    df_elbow = pd.DataFrame(wcss)
    df_elbow = df_elbow.reset_index()
    df_elbow.columns= ['n_clusters', 'within_cluster_sum_of_square']
    
    return df_elbow

dfm_elbow = number_of_clusters(dfm)
dff_elbow = number_of_clusters(dff)

fig, ax = plt.subplots(1, 2, figsize=(17,5))

sns.lineplot(data=dff_elbow, x='n_clusters', y='within_cluster_sum_of_square', ax=ax[0])
sns.scatterplot(data=dff_elbow[5:6], x='n_clusters', y='within_cluster_sum_of_square', color='black', ax=ax[0])
ax[0].set(xticks=dff_elbow.index)
ax[0].set_title('Female')

sns.lineplot(data=dfm_elbow, x='n_clusters', y='within_cluster_sum_of_square', ax=ax[1])
sns.scatterplot(data=dfm_elbow[5:6], x='n_clusters', y='within_cluster_sum_of_square', color='black', ax=ax[1])
ax[1].set(xticks=dfm_elbow.index)
ax[1].set_title('Male');


# As we can see that for both Female and Male; n=5 for cluster seems to be appropiate

# In[ ]:


def k_means(n_clusters, df, gender):

    kmf = KMeans(n_clusters=n_clusters, random_state=0) #defining the algorithm
    kmf.fit_predict(df) #fitting and predicting
    centroids = kmf.cluster_centers_ #extracting the clusters' centroids
    cdf = pd.DataFrame(centroids, columns=df.columns) #stocking in dataframe
    cdf['gender'] = gender
    return cdf

df1 = k_means(5, dff, 'female')
df2 = k_means(5, dfm, 'male')
dfc_scaled = pd.concat([df1, df2])
dfc_scaled.head()


# In[ ]:


dfc = pd.DataFrame()
for c in dfc_scaled.columns:
    if(c=='gender'): dfc[c] = dfc_scaled[c]
    else: 
        dfc[c] = (dfc_scaled[c] * dfsp.loc['std', c] + dfsp.loc['mean', c])
        dfc[c] = dfc[c].astype(int)
        
dfc.head()


# In[ ]:


def plot(dfs, names, colors, title):

    data_to_plot = []
    for i, df in enumerate(dfs):
  x = df['spending_score']
        y = df['annual_income']
        z = df['age']
        data = go.Scatter3d(x=x , y=y , z=z , mode='markers', name=names[i], marker = colors[i])
        data_to_plot.append(data)
layout = go.Layout(margin=dict(l=0,r=0,b=0,t=40),
        title= title, scene = dict(xaxis = dict(title  = x.name,), 
        yaxis = dict(title  = y.name), zaxis = dict(title = z.name)))
 figure = go.Figure(data=data_to_plot, layout=layout)
    py.offline.iplot(fig)
dfcf = dfc[dfc.gender=='female']
dfcm = dfc[dfc.gender=='male']
purple = dict(color=cl.scales['9']['seq']['RdPu'][3:8])
blue = dict(color=cl.scales['9']['seq']['Blues'][3:8])

