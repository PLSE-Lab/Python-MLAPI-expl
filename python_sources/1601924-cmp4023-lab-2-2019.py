#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data_path = '/kaggle/input/dwdm-petrol-prices/Petrol Prices.csv'


# In[ ]:


petrol = pd.read_csv(data_path)
petrol


# In[ ]:


petrol.head(7)


# In[ ]:


petrol.tail(4)


# In[ ]:


p=petrol['Date'].value_counts()
p


# In[ ]:


petrol['Date'] = pd.to_datetime(petrol['Date'])


# In[ ]:


print(petrol.loc[143])


# In[ ]:


petrol.iat[143,0]='Aug 18 2016'


# In[ ]:


#petrol['Date'] = pd.to_datetime(petrol['Date'])


# In[ ]:


#petrol['Month'] = petrol['Date'].dt.month


# In[ ]:


petrol['month'] = pd.DatetimeIndex(petrol['Date']).month
petrol


# In[ ]:


test=petrol.groupby('month').count()
test.head(13)


# In[ ]:


test.plot(kind='barh',figsize=(15,15),title='months count')


# In[ ]:


petrol['month'] = petrol['month'].astype(str)


# In[ ]:


i=0

while(i<232):
    if petrol.iat[i,14]=='1.0':
        petrol.iat[i,14]='Jan'
    if petrol.iat[i,14]=='2.0':
        petrol.iat[i,14]='Feb'
    if petrol.iat[i,14]=='3.0':
        petrol.iat[i,14]='Mar'
    if petrol.iat[i,14]=='4.0':
        petrol.iat[i,14]='Apr'
    if petrol.iat[i,14]=='5.0':
        petrol.iat[i,14]='May'
    if petrol.iat[i,14]=='6.0':
        petrol.iat[i,14]='Jun'
    if petrol.iat[i,14]=='7.0':
        petrol.iat[i,14]='Jul'
    if petrol.iat[i,14]=='8.0':
        petrol.iat[i,14]='Aug'
    if petrol.iat[i,14]=='9.0':
        petrol.iat[i,14]='Sep'
    if petrol.iat[i,14]=='10.0':
        petrol.iat[i,14]='Oct'
    if petrol.iat[i,14]=='11.0':
        petrol.iat[i,14]='Nov'
    if petrol.iat[i,14]=='12.0':
        petrol.iat[i,14]='Dec'
    i=i+1


# In[ ]:


petrol['day'] = pd.DatetimeIndex(petrol['Date']).day
petrol


# In[ ]:


petrol['year'] = pd.DatetimeIndex(petrol['Date']).year
petrol


# In[ ]:


petrol = petrol.fillna(0)
petrol['day']=petrol['day'].astype(int)
petrol['year']=petrol['year'].astype(int)
petrol


# In[ ]:


petrol['ts']= pd.to_datetime(petrol['Date']).values.astype(datetime)
petrol


# In[ ]:


petrol['try']=pd.to_datetime(petrol['Date']).values.astype(datetime)
petrol


# In[ ]:


petrol.reset_index
petrol.set_index('ts')
petrol.plot(kind='line',x='ts',y=['Gasolene_87', 'Gasolene_90', 'Auto_Diesel', 'Kerosene', 'Propane','Butane'
                         , 'Asphalt','HFO','ULSD', 'Ex_Refinery'],figsize=(10,5), grid=True)


# In[ ]:


x2 = petrol.copy()

x2 = x2.drop(x2.index[[229,230,231]])

x2.plot(kind='line',x='ts',y=['Gasolene_87', 'Gasolene_90', 'Auto_Diesel', 'Kerosene', 'Propane','Butane'
                         , 'Asphalt','HFO','ULSD', 'Ex_Refinery'],figsize=(10,5), grid=True)


# In[ ]:


interest = petrol['Propane']

interest.pct_change(periods=4)


# In[ ]:


interest.plot()


# In[ ]:


new = {'Propane':petrol.Propane,'ULSD':petrol.ULSD,'month':petrol.month,'day':petrol.day,'year':petrol.year,'timestamp':petrol.ts}

newDatax = pd.DataFrame(new)

newData = newDatax.drop(newDatax.index[[229,230,231]])

newData


# In[ ]:


s={'ULSD': newData.ULSD,'Propane':newData.Propane}
data =pd.DataFrame(s)
data


# In[ ]:


kmeans=KMeans(n_clusters=5, init='k-means++',n_init=10, max_iter=300)
newData['cluster'] = kmeans.fit_predict(data)
newData


# In[ ]:


print(kmeans.labels_)


# In[ ]:


newData.plot(kind='scatter',x='Propane',y='ULSD',c=kmeans.labels_, cmap='rainbow')


# In[ ]:


petrol.groupby('year').mean()


# Autoregression 
# 
# Autoregression is a time series model that uses observations from previous time steps as input to a regression equation to predict the value at the next time step (Brownlee, 2019).
# 
# https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/
