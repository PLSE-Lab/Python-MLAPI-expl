#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dateutil
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pandasql import sqldf

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data_path='/kaggle/input/dwdm-petrol-prices/Petrol Prices.csv'
data = pd.read_csv(data_path)
data


# In[ ]:


data.head(7)


# In[ ]:


data.tail(4)


# In[ ]:


data.shape # prints a tuple with the (number of rows, number of columns)


# In[ ]:


data.columns


# In[ ]:


data.dtypes


# In[ ]:


# Find count of non-NA across the columns 
data.count(axis = 1) 


# In[ ]:


count = data['Date'].value_counts() 
print(count) 


# In[ ]:


count = data.groupby(['Date']).count() 
print(count) 


# In[ ]:


df_results = sqldf("select count(Date) as Records, Date as Month  " +
                   "from data group by Month")
df_results


# In[ ]:


#data['Date'] = pd.to_datetime(data['Date'])


# change ug 18 2016 to aug 18 2016

# In[ ]:


data.loc[data['Date'] == 'ug 18 2016']


# In[ ]:


data.at[143,'Date']='Aug 18 2016'
data


# In[ ]:


data['Date'] = pd.to_datetime(data['Date'])
data


# In[ ]:


data.loc[data['Date'] == 'Aug 18 2016']


# In[ ]:


data["months"]=pd.DatetimeIndex(data['Date']).month
data


# In[ ]:


do=data.groupby('months').count()
do


# In[ ]:


rng = pd.DataFrame() 
rng['date'] = pd.date_range('1/1/2011', periods = 72, freq ='H') 
  
# Print the dates in dd-mm-yy format 
rng[:5] 
  
# Create features for year, month, day, hour, and minute 
rng['year'] = rng['date'].dt.year 
rng['month'] = rng['date'].dt.month 
rng['day'] = rng['date'].dt.day 

  
# Print the dates divided into features 
rng.head(3) 


# In[ ]:


do.plot(title="Bar graph", kind='bar',
            figsize=(12,8))


# In[ ]:


data["day"]=pd.DatetimeIndex(data['Date']).day
data["year"]=pd.DatetimeIndex(data['Date']).year
data

#data["day"]=0
#data["year"]=0


# In[ ]:


data['Timestamp']= pd.to_datetime(data['Date']).values.astype(datetime)
data


# In[ ]:


# replacing na values in NaN with Null
data["months"].fillna("Null", inplace = True) 
data["day"].fillna("Null", inplace = True)  
data["year"].fillna("Null", inplace = True) 

data 


# In[ ]:


data2 = data[['Gasolene_87', 'Gasolene_90', 'Auto_Diesel', 'Kerosene', 'Propane', 'Butane', 'HFO', 'Asphalt', 'ULSD', 'Ex_Refinery'
]]
data2


# In[ ]:


data3 = data[['Gasolene_87', 'Gasolene_90', 'Auto_Diesel', 'Kerosene', 'Propane', 'Butane', 'HFO', 'Asphalt', 'ULSD', 'Ex_Refinery','Timestamp'
]]
data3


# In[ ]:


data3.plot(kind='line',x='Timestamp', y=['Gasolene_87', 'Gasolene_90', 'Auto_Diesel', 'Kerosene', 'Propane', 'Butane', 'HFO', 'Asphalt', 'ULSD', 'Ex_Refinery'
],
           title="Line",
            figsize=(20,20))


# In[ ]:


data.Gasolene_90[0]


# In[ ]:


help(pd.date_range)


# In[ ]:


intrest_col= data['Gasolene_90'].pct_change(periods=4)
intrest_col


# In[ ]:


intrest_col.head(20)


# In[ ]:


intrest_col.plot(kind='bar', figsize=(20,20))


# In[ ]:


new={'HFO':data.HFO, 'Asphalt':data.Asphalt,'months':data.months,'year':data.year,'day':data.year,'Timestamp':data.Timestamp}
data11=pd.DataFrame(new)
data11


# In[ ]:


data11.info()


# In[ ]:


#drop null
data11=data11.dropna(subset=['HFO'])


# In[ ]:


data11


# In[ ]:


data11['months'].astype(np.int64)


# In[ ]:


kmeans = KMeans(n_clusters=3).fit(data11)
centroids = kmeans.cluster_centers_
print(centroids)


# In[ ]:


kmeans = KMeans(n_clusters=3).fit(data11)


# In[ ]:


new={'HFO':data.HFO, 'Asphalt':data.Asphalt}
data12=pd.DataFrame(new)
data12


# In[ ]:


data12=data12.dropna(subset=['HFO'])


# In[ ]:


data12


# In[ ]:


data11['cluster']=kmeans.fit_predict(data12)
data11


# In[ ]:


data11.plot(kind='scatter',x='HFO',y='Asphalt',c=kmeans.labels_, cmap='rainbow', figsize=(20,20))


# In[ ]:


Year = data[['Gasolene_87', 'Gasolene_90', 'Auto_Diesel', 'Kerosene', 'Propane', 'Butane', 'HFO', 'Asphalt', 'ULSD', 'Ex_Refinery','year'
]]
Year


# In[ ]:


G_year=Year.groupby('year').sum()
G_year


# In[ ]:


G_mean=G_year.groupby('year').mean()
G_mean


# A time series is a sequence of observations taken sequentially in time.
# 
# Moving Average (MA)
# The moving average (MA) method models the **next step in the sequence** as a linear function of the **residual errors** from a mean process at prior time steps.
# 
# The notation for the model involves specifying the order of the model q as a parameter to the MA function, e.g. MA(q). For example, MA(1) is a first-order moving average model.
# 
# The method is suitable for univariate time series without trend and seasonal components.
