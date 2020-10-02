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


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


path = '/kaggle/input/dwdm-petrol-prices/Petrol Prices.csv'
data = pd.read_csv(path)


# In[ ]:


data.head(7)


# In[ ]:


data.tail(4)


# In[ ]:


from pandasql import sqldf # XXXXXXX


# In[ ]:


data['Months'] = pd.DatetimeIndex(data['Date']).month 
data


# #error discovered for in recoord
# #locate error in data set and correct it

# In[ ]:


print(data.loc[143]) #locate record


# In[ ]:


data.iat[143,0] ='Aug 18 2016' #change value


# In[ ]:


print(data.loc[143])#locate cell


# In[ ]:


data['Months'] = pd.DatetimeIndex(data['Date']).month 
data


# In[ ]:


groupy = data.groupby("Months").count()
groupy


# #HFO was not selling for a while....had NAN for multiple columns

# In[ ]:


#import calendar


# In[ ]:


#data['Months'] = calendar.month_name('Months')
#data


# In[ ]:


groupy.plot(kind='bar',title='Data Grouping by Month',figsize=(20,20))


# #creat column and populate wid date

# In[ ]:


data['Day']=pd.DatetimeIndex(data['Date']).day
data['Year']=pd.DatetimeIndex(data['Date']).year
data['Months'].fillna('NULL',inplace = True)
data['Day'].fillna('NULL',inplace = True)
data['Year'].fillna('NULL',inplace = True)
data


# In[ ]:


data['Months'] = data['Months'].astype(str)
data.dtypes


# In[ ]:


data['Timestamp'] = " "


# #create column

# In[ ]:


data['Timestamp']= pd.to_datetime(data['Date'])


# #display date from date column as the int

# In[ ]:


from datetime import datetime


# In[ ]:


data['Timestamp'] = pd.to_datetime(data['Date']).values.astype(datetime)


# In[ ]:


data


# In[ ]:


extract = data[['Gasolene_87', 'Gasolene_90', 'Auto_Diesel', 'Kerosene', 'Propane', 'Butane', 'HFO', 'Asphalt', 'ULSD', 'Ex_Refinery']]
extract


# 9)#plot prices against period(timestamp)

# In[ ]:


extract = data[['Gasolene_87' ,'Gasolene_90', 'Auto_Diesel', 'Kerosene', 'Propane', 'Butane', 'HFO', 'Asphalt', 'ULSD', 'Ex_Refinery','Timestamp']]
extract


# In[ ]:


extract.plot(kind='line',title='Gas prices over the Period',x='Timestamp',
             y=['Gasolene_87' ,'Gasolene_90','Kerosene', 'Propane' ,'Butane' ,'HFO', 'Asphalt' ,'ULSD' ,'Ex_Refinery'],
             figsize=(18,18))


# #10
# #every four time periods is 4 weeks(1 month)

# In[ ]:


percentage = data['Kerosene'].pct_change(periods = 4) #pandas.DataFrame
percentage


# #record 4 is the percentage change for 0, 1, 2, 3....record 8 is the percentage change for 7,6,5 and 4

# In[ ]:


percentage.plot(kind='line', title='percentage change(y) for four time periods(x)',figsize=(18,18))


# #11

# In[ ]:


newdataset = {'Propane' :data.Propane ,'Gasolene_90' :data.Gasolene_90, 'Month' :data.Months,
               'Day' :data.Day, 'Year' :data.Year, 'Timestamp' :data.Timestamp} 
newdataset


# In[ ]:


data2 = pd.DataFrame(newdataset)
data2


# #K-Means

# In[ ]:


data3 = data2.drop(data2.index[[229,230,231]])#error in using null values
data3  #drop last 3 null rows


# In[ ]:


from sklearn.cluster import KMeans


# In[ ]:


interestdataset = {'Propane' :data3.Propane ,'Gasolene_90' :data3.Gasolene_90} #dataset with just two columns for kmmeans
interestdataset


# In[ ]:


data4 = pd.DataFrame(interestdataset) #dataframe with that dataset
data4


# #n_init => Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia

# In[ ]:


Mean=KMeans(n_clusters=10, init='k-means++', n_init=10, max_iter=6)
data3['Cluster#'] = Mean.fit_predict(data4) #assign result from kmeans to the extra row
data3


# In[ ]:


frame2 = frame.drop(frame.index[[229,230,231]])#error in using null values
frame2  #drop last 3 null rows


# #13
# 

# In[ ]:


#suppose to be cluster# column from previous output


# #14

# In[ ]:


import seaborn as sns


# In[ ]:


plt.style.use('dark_background')


# In[ ]:


frame2.dtypes #an error was discovered after the plot ran


# In[ ]:


frame2['Propane']= frame2['Propane'].astype(int)
frame2['Gasolene_90']= frame2['Gasolene_90'].astype(int)
frame2


# In[ ]:


frame2.plot(kind='scatter', x='Propane' , y= 'Gasolene_90', c=Mean.labels_, cmap='rainbow', figsize=(16,16))


# #15

# In[ ]:


#a
grouping = data.groupby('Year')


# In[ ]:


#grouping.mean() gave an error
#changing columns to int
data.dtypes


# In[ ]:


data


# In[ ]:


data['Propane']= data['Propane'].astype(int)
data['Gasolene_90']= data['Gasolene_90'].astype(int)
data['Gasolene_87']= data['Gasolene_87'].astype(int)
data['Auto_Diesel']= data['Auto_Diesel'].astype(int)
data['Kerosene']= data['Kerosene'].astype(int)
data['Butane']= data['Butane'].astype(int)
data['HFO']= data['HFO'].astype(int)
data['Asphalt']= data['Asphalt'].astype(int)
data['ULSD']= data['ULSD'].astype(int)
data['87_Change']= data['87_Change'].astype(int)
data['SCT']= data['SCT'].astype(int)
data['Ex_Refinery']= data['Ex_Refinery'].astype(int)
data


# In[ ]:


grouping.mean()


# #15D
# The Microsoft Time Series algorithm provides multiple algorithms that are optimized for forecasting continuous values, such as product sales, over time
# 
# The simple exponential smoothing (SES) is a short-range forecasting method that assumes a reasonably stable mean in the data with no trend (consistent growth or decline). It is one of the most popular forecasting methods that uses weighted moving average of past data as the basis for a forecast.
# 
