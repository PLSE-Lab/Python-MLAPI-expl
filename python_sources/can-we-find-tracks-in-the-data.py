#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("../input/beacon_readings.csv")
data['Time'] = data['Time'].map(lambda l: l.split(' ')[2])
data['Hours'] = data['Time'].map(lambda l: l.split(':')[0])
data['Minutes'] = data['Time'].map(lambda l: l.split(':')[1])
data['Seconds'] = data['Time'].map(lambda l: l.split(':')[2])
data.head(10)


# In[ ]:


sns.factorplot('Date',data=data,kind='count')


# In[ ]:


sns.factorplot('Hours',data=data,kind='count')


# In[ ]:


sns.factorplot('Minutes',data=data,kind='count')


# In[ ]:


data['Time'] = (data['Minutes'].values.astype('float')*60+                data['Seconds'].values.astype('float')) -                 data['Minutes'].values.astype('float')[0]*60
data['Time'] = data['Time'].map(lambda l:int(round(l)))
data['Time'] = data['Time']-data['Time'].values[0]
data.drop(['Hours','Minutes','Seconds','Date'],axis=1,inplace=True)
data['Distance A'] = data['Distance A'].replace(to_replace=0, method='ffill')
data['Distance B'] = data['Distance B'].replace(to_replace=0, method='ffill')
data['Distance C'] = data['Distance C'].replace(to_replace=0, method='ffill')
data = data.set_index(['Time'])


# In[ ]:


data.tail(10)


# In[ ]:


def plot_variable(data,variable_name):
    plt.figure(figsize=(20,20))
    plt.subplot(211)
    plt.plot(data.index.values, data[variable_name].values)
    plt.subplot(212)
    sns.distplot(data[variable_name])


# In[ ]:


plot_variable(data,'Distance A')


# In[ ]:


plot_variable(data,'Distance B')


# In[ ]:


plot_variable(data,'Distance C')


# In[ ]:


plot_variable(data,'Position X')


# In[ ]:


plot_variable(data,'Position Y')


# In[ ]:


from mpl_toolkits.mplot3d import Axes3D

f = plt.figure()
ax = Axes3D(f)
ax.scatter(data.values[:,0],data.values[:,1],data.values[:,2])
ax.set_xlabel('Distance A')
ax.set_ylabel('Distance B')
ax.set_zlabel('Distance C')


# In[ ]:


f = plt.figure()
plt.scatter(data.values[:,3]+np.random.randn(data.values.shape[0]),
            data.values[:,4]+np.random.randn(data.values.shape[0]))
plt.xlabel('Position X')
plt.ylabel('Position Y')


# In[ ]:


from sklearn.cluster import KMeans

position_data = data.values[:,3:]

km = KMeans(5)
km.fit_transform(position_data)
f = plt.figure()
plt.scatter(position_data[:,0]+np.random.randn(position_data.shape[0]),
            position_data[:,1]+np.random.randn(position_data.shape[0]),
            c=km.labels_,cmap="viridis_r")
plt.xlabel('Position X')
plt.ylabel('Position Y')


# In[ ]:


f = plt.figure()
ax = Axes3D(f)
ax.scatter(data.values[:,0],data.values[:,1],data.values[:,2],c=km.labels_,cmap="viridis_r")
ax.set_xlabel('Distance A')
ax.set_ylabel('Distance B')
ax.set_zlabel('Distance C')


# In[ ]:


from sklearn.manifold import TSNE

distance_data = data.values[:,:3]

mf = TSNE()
d = mf.fit_transform(distance_data)
plt.scatter(d[:,0],d[:,1],c=km.labels_,cmap="viridis_r")

