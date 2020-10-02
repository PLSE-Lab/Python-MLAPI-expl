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


import folium


# In[ ]:


data=pd.read_csv("../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv")


# In[ ]:


data.head()


# In[ ]:


data['Province/State'].isnull().sum()


# In[ ]:


data.Date.unique()


# In[ ]:


data=data.replace(to_replace ="2020-01-02 23:00:00", 
                 value ="2020-02-01 23:00:00")


# In[ ]:


data.Date.unique()


# In[ ]:


data.dtypes


# In[ ]:


data['Last Update']= pd.to_datetime(data['Last Update']) 


# In[ ]:


#19-nCoV, 31 January 2020**")
from datetime import date


# In[ ]:


data_final=data[(data['Last Update'] >= '2020-2-2 00:00:00') & (data['Last Update'] < '2020-3-2 00:00:00')]


# In[ ]:


data_final=data_final.drop(["Sno"],axis=1)


# In[ ]:


data_final=data_final.reset_index(drop=True)


# In[ ]:


data_final.head()


# In[ ]:


data.Country.unique()


# In[ ]:


data=data.replace(to_replace ="China", 
                 value ="Mainland China")


# In[ ]:


data.Country.unique()


# In[ ]:


print('Total Confirmed Cases:',data_final['Confirmed'].sum())
print('Total Deaths: ',data_final['Deaths'].sum())
print('Total Recovered Cases: ',data_final['Recovered'].sum())


# In[ ]:


data.dtypes


# In[ ]:


from datetime import date


# In[ ]:


data_final.shape


# In[ ]:


#data_final=data_final.drop(["Date"],axis=1)


# In[ ]:


countries=data_final.groupby(["Country"]).sum().reset_index()


# In[ ]:


countries


# Mainland China has maximum confirmed cases of 17220 as of 2nd Feb ,India has 2.0

# In[ ]:


countries.head()


# In[ ]:


top10_countries=countries.nlargest(10,['Confirmed']).reset_index(drop=True)


# In[ ]:


top10_countries.head()


# In[ ]:


top10_countries.plot('Country',['Confirmed'],kind = 'bar',figsize=(20,30))


# In[ ]:


data_china=data_final[data_final['Country']=="Mainland China"]


# In[ ]:


data_china.head()


# In[ ]:


data_china['Province/State'].isnull().sum()


# In[ ]:


data_china=data_china.groupby(["Province/State"]).sum()


# In[ ]:


data_china.reset_index()


# #Hubei has maximum number of cases
# 

# In[ ]:


data_china10=data_china.nlargest(10,['Confirmed']).reset_index()


# In[ ]:


data_china10.head()


# In[ ]:


from matplotlib import pyplot as plt


# In[ ]:


#fig = plt.figure(figsize=(10,))
data_china10.plot('Province/State',['Confirmed'],kind = 'bar',figsize=(20,10))


# In[ ]:



data_china10.plot('Province/State',['Deaths'],kind = 'bar',figsize=(10,10))


# In[ ]:


data.head()


# In[ ]:


data_trend=data[["Date","Confirmed","Deaths","Recovered"]]


# In[ ]:


data_trend.head()


# In[ ]:


data_trend=data_trend.groupby(["Date"]).sum().reset_index()


# In[ ]:


data_trend


# In[ ]:


trend=data[["Date","Confirmed","Deaths","Recovered"]]


# In[ ]:


trend.head()


# In[ ]:


trend=trend.groupby(["Date"]).sum().reset_index()


# In[ ]:


trend.plot('Date',['Confirmed'],figsize=(10,10))


# In[ ]:


trend.plot('Date',['Deaths'],figsize=(10,10))


# In[ ]:


trend.plot('Date',['Recovered'],figsize=(10,10))


# In[ ]:


import pandas as pd
import datetime
import folium
from folium.map import *
from folium import plugins
from folium.plugins import MeasureControl
from folium.plugins import FloatImage


# In[ ]:


data.head()


# In[ ]:


Nonconfirmed_cases=data.loc[data['Confirmed']==0.0] 


# In[ ]:


len(Nonconfirmed_cases)


# In[ ]:


Nonconfirmed_cases


# There are around 19 cases with no confirmation
# 

# 144 records dont have state/province information
# 

# In[ ]:


data.head()


# In[ ]:


data1=data[['Date','Country','Confirmed','Deaths','Recovered']]


# In[ ]:


data1.head()


# In[ ]:


data1.dtypes


# In[ ]:


data1['Date']= pd.to_datetime(data1['Date']) 


# In[ ]:


data1.head(10)


# In[ ]:


data1=data1.groupby(["Date"]).sum().reset_index()


# In[ ]:


data1


# In[ ]:


data1.plot('Date',['Confirmed'],figsize=(10,10))


# In[ ]:


data1=data1.drop(['Deaths','Recovered'],axis=1)


# In[ ]:


data1


# In[ ]:


data1.set_index('Date',inplace=True)


# In[ ]:


data1=data1.diff()


# In[ ]:


data1=data1.fillna(555)
data1.reset_index(inplace=True)


# In[ ]:


data1


# In[ ]:


data1.columns = ['ds', 'y']
data1


# In[ ]:





# I am going to use FB Prophet for this predition

# In[ ]:


from fbprophet import Prophet


# My model is predicting for next 15 days if the condition remains similar and virus is not contained by any means****

# In[ ]:


m = Prophet()
m.fit(data1)
future = m.make_future_dataframe(periods=15)
forecast = m.predict(future)
forecast


# In[ ]:


print('RMSE: %f' % np.sqrt(np.mean((forecast.loc[:1682, 'yhat']-data1['y'])**2)) )


# In[ ]:


future.tail()


# In[ ]:


forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()


# In[ ]:


forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[ ]:


m.plot(forecast,
              uncertainty=True)


# In[ ]:


m.plot_components(forecast)


# In[ ]:


d1=forecast[['ds','yhat']]


# In[ ]:


d1.head()


# In[ ]:


d1['total predicted cases']=d1['yhat'].cumsum()


# In[ ]:


d1=d1.drop(['yhat'],axis=1)


# In[ ]:


d1.head()


# In[ ]:


d1.plot('ds',['total predicted cases'],figsize=(10,10))


# **It shows that the number of confirmed cases will go above 90000 if no measure are taken for containing virus infection **

# Please give any suggestion if you think I could have done differently
# 

# In[ ]:




