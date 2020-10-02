#!/usr/bin/env python
# coding: utf-8

# In[338]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import ipywidgets as widgets
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[339]:


df1 = pd.read_csv('../input/super_hero_powers.csv')


# In[340]:


df1['names']=df1['hero_names']
df1=df1.drop(columns='hero_names')
df1.head()


# ****The first basic clustering will use only the heros' powers. This will identifiy which heroes have similar powers, and a recommendation engine will be built.

# In[341]:


# Convert True/False labels to 0,1
df1*=1
print(df1.shape)
df1.head()


# In[342]:


# Optimize number of clusters by KMeans distortion, normalized mutual information score, and adjusted rand score
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans, vq
import scipy.cluster.hierarchy as shc

df = np.array(df1.drop(columns='names').astype(float))

plt.figure(figsize=(20,20))
dend=shc.dendrogram(shc.linkage(df, method='ward'))
# Best separation occurs at k~40


# In[343]:


from sklearn.cluster import AgglomerativeClustering
cluster=AgglomerativeClustering(n_clusters=40, affinity='euclidean', linkage='ward')
cluster.fit_predict(df)


# In[344]:


from sklearn.manifold import TSNE
np.random.seed(777)
tsne=TSNE(n_components=2, verbose=1, perplexity=22, n_iter=10000) # perplexity determined by 667 entries / 40 clusters ~ 22
results=tsne.fit_transform(df)
plt.figure(figsize=(12,12))
plt.scatter(results[:,0], results[:,1], c=cluster.labels_, cmap='cool')


# Working with 40 clusters seems to give good separation.

# In[345]:


df1['clusters'] = cluster.labels_
df1['clusters'].value_counts()


# In[346]:


print(df1.loc[df1['clusters']==36]['names'])
print(df1.loc[df1['clusters']==36][df1==1].dropna(axis=1).columns.values)


# In[347]:


df2 = df1.loc[(df1['clusters'] != 36) & (df1['clusters'] != 37)] # These clusters have < 3 entries
df2['clusters'].value_counts()


# In[348]:


#Run this cell, select an entry from the drop down menu, then run the next cell
import random
a=[36,37]
c=[]
for i in range(5):
    b = random.randint(0,39)
    while b in a: 
        b = random.randint(0,39)
    c.append(b)
    a.append(b)
e=[]
for i in c:
    df_temp=df2.loc[df2['clusters']==i]
    df_temp=df_temp.reset_index(drop=True)
    d = random.randint(0,len(df_temp)-1)
    e.append(df_temp.iloc[d,:]['names'])

print("Select your favorite hero")
dd = widgets.Dropdown(options=e, disabled=False)
dd

    


# In[350]:


df2_temp=df2.loc[df2['names']==dd.value]
#print(dd.value+" has the following powers:")
#print(df2_temp.dropna(axis=1).drop(columns=['names', 'clusters']).columns.values)

v = df2_temp['clusters'].values
df3_temp = df2.loc[df2['clusters'].isin(v)].reset_index(drop=True)
np.random.seed()
aa=[dd.value]
cc=[]
for i in range(2):
    bb = random.randint(0,len(df3_temp)-1)
    while df3_temp.iloc[bb,:]['names'] in aa:
        bb = random.randint(0,len(df3_temp)-1)
    aa.append(df3_temp.iloc[bb,:]['names'])
    cc.append(df3_temp.iloc[bb,:]['names'])

print("If you like", dd.value+", then you may also like", cc[0], "or", cc[1]+".")


# In[ ]:




