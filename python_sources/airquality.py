#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import sklearn
import keras
import sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import datetime
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.cluster import KMeans


from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import SGD 
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
import itertools
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout

from keras.utils import plot_model

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output


# First we read data and perform some preprocessing on the data

# In[2]:


df = pd.read_csv("../input/dataset.csv")
df.head()


# We drop ID column as this is not helpful in model prediction

# In[3]:


df.set_index('ID', drop=True, inplace=True)
df_loc = df.drop(columns="Datetime")
df_loc.head()


# We replace Daytime with 0, 1 to help in clustering mechanism

# In[4]:


df_loc.Daytime[df.Daytime =="Night"] = 1
df_loc.Daytime[df.Daytime =="Day"] = 0

df.head()


# In[ ]:


df_loc.tail()


# Make some initial clustering based on the KMean

# In[5]:


clusters = KMeans(n_clusters=3).fit(df_loc)
centroids = clusters.cluster_centers_
print(centroids)


# In[ ]:


Scatter plot the clusters


# In[ ]:


df.head()


# In[6]:


plt.figure(figsize=(10,5))
plt.scatter(df_loc['Temperature'], df['Humidity'], c= clusters.labels_.astype(float), cmap='viridis', s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=20)
plt.xlabel('Temp')
plt.ylabel('Humidity')


# In[7]:


plt.figure(figsize=(10,5))
plt.scatter(df_loc['Temperature'], df['Pressure'], c= clusters.labels_.astype(float), cmap='viridis', s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=20)
plt.xlabel('Temp')
plt.ylabel('Pressure')


# In[8]:


plt.figure(figsize=(10,5))
plt.scatter(df_loc['Temperature'], df['PM2.5'], c= clusters.labels_.astype(float), cmap='viridis', s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=20)
plt.xlabel('Temp')
plt.ylabel('PM2.5')


# In[9]:


plt.figure(figsize=(10,5))
plt.scatter(df_loc['Temperature'], df['PM10'], c= clusters.labels_.astype(float), cmap='viridis', s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=20)
plt.xlabel('Temp')
plt.ylabel('PM10')


# In[10]:


plt.figure(figsize=(10,5))
plt.scatter(df_loc['Temperature'], df['Co2 Gas'], c= clusters.labels_.astype(float), cmap='viridis', s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=20)
plt.xlabel('Temp')
plt.ylabel('Co2 Gas')


# In[11]:


plt.figure(figsize=(10,5))
plt.scatter(df_loc['Co2 Gas'], df['PM2.5'], c= clusters.labels_.astype(float), cmap='viridis', s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=20)
plt.xlabel('Co2 Gas')
plt.ylabel('PM2.5')


# In[12]:


plt.figure(figsize=(10,5))
plt.scatter(df_loc['Co2 Gas'], df['PM10'], c= clusters.labels_.astype(float), cmap='viridis', s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=20)
plt.xlabel('Co2 Gas')
plt.ylabel('PM10')


# In[13]:


plt.figure(figsize=(10,5))
plt.scatter(df_loc['PM10'], df['PM2.5'], c= clusters.labels_.astype(float), cmap='viridis', s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=20)
plt.xlabel('PM10')
plt.ylabel('PM2.5')


# In[ ]:





# Normalization of the data

# In[14]:


df_re = df.rename(columns={"PM2.5": "PM25", "Co2 Gas":"CO2Gas"})
df_re.head()


# In[15]:


def normalize_data(df):
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    df['Temperature'] = min_max_scaler.fit_transform(df.Temperature.values.reshape(-1,1))
    df['Humidity'] = min_max_scaler.fit_transform(df.Humidity.values.reshape(-1,1))
    df['Pressure'] = min_max_scaler.fit_transform(df.Pressure.values.reshape(-1,1))
    
    df['PM25'] = min_max_scaler.fit_transform(df.PM25.values.reshape(-1,1))
    df['PM10'] = min_max_scaler.fit_transform(df.PM10.values.reshape(-1,1))
    df['CO2Gas'] = min_max_scaler.fit_transform(df.CO2Gas.values.reshape(-1,1))
    return df


# We will now plot individual values to find patterns in the air quality

# In[16]:


df_re.head()
df_re[['Datetime','PM25']].groupby(["Datetime"]).median().sort_values(by='Datetime',ascending=False).head(10).plot.bar(color='r')
plt.show()


# In[17]:


import seaborn as sns
corr =  df_re.corr()
plt.subplots(figsize=(20,9))
sns.heatmap(corr)


# In[25]:


sns.distplot(df_re['PM25'],hist=False);
fig = plt.figure()
sns.distplot(df_re['PM10'],hist=False);
fig = plt.figure()
sns.distplot(df_re['CO2Gas'],hist=False);
fig = plt.figure()
sns.distplot(df_re['Temperature']);
fig = plt.figure()

normalized = normalize_data(df_re)


# In[ ]:


normalized.head()


# In[27]:


sns.pairplot(df_re);


# In[ ]:




