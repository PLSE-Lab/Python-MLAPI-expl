#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data_path = '/kaggle/input/dwdm-petrol-prices/Petrol Prices.csv'
data = pd.read_csv(data_path) 
data.head(7)


# In[ ]:


data.tail(4)


# In[ ]:


data['Date']=data.Date.astype('datetime64')


# In[ ]:


Month = data.Date.dt.month
data['Month'] = Month
result = data.groupby('Month').Date.count()
result


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


data.groupby('Month').Date.count().plot(kind="bar",
    title="Total Record per Month",
    figsize=(12,8)
)
plt.ylabel("Number of Records")


# In[ ]:


from datetime import time


# In[ ]:


Month = data.Date.dt.month_name()
data['Month'] = Month
Day = data.Date.dt.day
data['Day'] = Day
Year = data.Date.dt.year
data['Year'] = Year


# In[ ]:


data['TimeStamp']= data.Date.dt.time


# In[ ]:


data.describe()


# In[ ]:


Interesting_data = data[['Gasolene_87','Gasolene_90','Auto_Diesel','Kerosene','Propane','Butane','HFO','Asphalt','ULSD','87_Change','Ex_Refinery']]
df = pd.DataFrame(Interesting_data)
df
x=df.set_index(data['Date'])
x.plot(kind='line', figsize=(12,8))


# In[ ]:


print(x['Gasolene_90'].pct_change(periods=4))


# In[ ]:


x['Gasolene_90'].pct_change(periods=4).plot(kind='line', figsize=(12,8))


# In[ ]:


data1 = pd.DataFrame({
    'Gasolene_87':data['Gasolene_87'],
    'Gasolene_90':data['Gasolene_90'],
    'Month':data['Month'],
    'Day':data['Day'],
    'Year':data['Year'],
    'Timestamp':data['TimeStamp'],
})
data1


# In[ ]:


from sklearn.cluster import KMeans


# In[ ]:


data_values = data1.iloc[ :, [0,1]].values
data_values


# In[ ]:


wcss = []
for i in range( 1, 15 ):
    kmeans = KMeans(n_clusters=i, init="k-means++", n_init=10, max_iter=300) 
    kmeans.fit_predict( data_values )
    wcss.append( kmeans.inertia_ )
    
plt.plot( wcss, 'ro-', label="WCSS")
plt.title("Computing WCSS for KMeans++")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()


# In[ ]:


data2 = data[['Gasolene_87','Gasolene_90','Auto_Diesel','Kerosene','Propane','Butane','HFO','Asphalt','ULSD','87_Change','Ex_Refinery']]


# In[ ]:


missing_data_results = data2.isnull().sum()
print(missing_data_results)


# In[ ]:


data2 = data2.fillna( data.median() )


# In[ ]:


data_values1 = data2.iloc[ :, :].values
data_values1


# In[ ]:



kmeans = KMeans(n_clusters=4, init="k-means++", n_init=10, max_iter=300) 
data1["cluster"] = kmeans.fit_predict( data_values )
data1.head(25)


# In[ ]:


import seaborn as sns


# In[ ]:


sns.pairplot( data1, hue="cluster")


# In[ ]:


average = data.groupby('Year').mean()
average


# A)Average price for gas before cluster 
# 	    Gasolene_87	Gasolene_90	Auto_Diesel	Kerosene	Propane	     Butane	    HFO	        Asphalt	    ULSD	   87_Change	SCT	        Ex_Refinery	       
# Year													
# 2015	103.617264	105.273243	97.179638	96.638336	35.102489	40.115998	45.031875	66.629389	106.142917	0.069434	23.238175	70.959337
# 2016	99.019338	100.674829	91.794723	83.194646	33.984775	40.704885	40.699758	49.906919	100.427785	-0.123654	29.683023	60.334558
# 2017	111.148185	113.689085	107.640877	91.616954	43.929562	48.724500	63.179696	63.526038	113.380477	-0.430192	35.704369	65.339435	
# 2018	129.460207	132.296107	130.314930	111.805059	49.359339	54.288204	78.827380	83.159994	135.057030	0.018333	37.776100	79.914998	
# 2019	126.284467	129.120367	131.411411	109.790811	44.055822	51.152833	80.628300	93.862844	134.500733	-1.006111	37.776100	77.027961	
#  
# B)Cluster 0 have the third highest gas price while cluster 1 had the highest. 
#  Cluster 3 had the lowest gas prices and cluster 2 had the second highest. 
#  
# C)K-means clustering is a type of unsupervised learning, which is used when you have unlabeled data. The goal of this algorithm is to find groups in the data.
#  
# D)Autoregression Models: According to Brownlee (2019) Autoregression is a time series model that uses observations from previous time steps as input to a regression equation to predict the value at the next time step.
