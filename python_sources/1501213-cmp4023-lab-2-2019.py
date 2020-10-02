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


data_path='/kaggle/input/dwdm-petrol-prices/Petrol Prices.csv'
data = pd.read_csv(data_path)
data


# In[ ]:


#data.iloc[:7]
data.head(7)


# In[ ]:


data.tail(4)


# In[ ]:


data.dtypes


# In[ ]:


daf =pd.DataFrame(data)
#daf.groupby('Date').size()

from datetime import datetime

#datetime_object = pd.to_datetime(daf['Date'])


# In[ ]:


#used to find the row that 'ug 18 2016' is in 
df2 = daf['Date'].isin(['ug 18 2016'])
daf[df2]


# In[ ]:


daf.at[143,'Date'] = 'Aug 18 2016'
daf.iloc[143]


# In[ ]:


from datetime import datetime

#daf = pd.to_datetime(daf['Date'])
daf['month'] = pd.DatetimeIndex(daf['Date']).month_name()
#daf.drop(['year','month'])
daf


# In[ ]:



#count_months =daf.pivot_table(index=['month'],aggfunc='size').reset_index('# of Records')
#agg_months = pd.DataFrame(count_months,columns = ['count'])
count_months=daf.groupby('month').size().reset_index(name='# of Records')
count_months


# In[ ]:


daf['Year'] = pd.DatetimeIndex(daf['Date']).year
daf


# In[ ]:


daf['day'] = pd.DatetimeIndex(daf['Date']).day
daf['timestamp'] = pd.DatetimeIndex(daf['Date']).time
daf


# In[ ]:


#7

data2=data[['Date','Gasolene_87', 'Gasolene_90', 'Auto_Diesel', 'Kerosene', 'Propane', 'Butane', 'HFO', 'Asphalt', 'ULSD', 'Ex_Refinery']]
data2['datetime']= pd.to_datetime(data['Date'])
data2 


# In[ ]:


from pandasql import sqldf
q = """Select * from data2 where datetime between '2018-10-01' and '2019-12-30' order by datetime asc;"""
df_results = sqldf(q, globals())
df_results


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

df_results.plot(kind="line",
    x='Date',
  #  y=['Gasolene_87',]
    title="Gas Prices over the last 8 Months ",
    figsize=(12,8))


# In[ ]:


Auto_Diesel = pd.DataFrame(df_results['Auto_Diesel'])
Auto_Diesel.pct_change(periods=4)


# In[ ]:


Auto_Diesel.pct_change(periods=4).plot()


# In[ ]:


KMdata = daf[['day','month','Year','timestamp','Kerosene','Butane']]
KMdata


# In[ ]:


KMdata.plot(kind = 'scatter',x = 'Kerosene', y='Butane')


# In[ ]:


#find missing values
print( KMdata.isnull().sum())


# In[ ]:


#relpace missing values
KMdata.fillna(-1, inplace=True)
KMdata


# In[ ]:


data_values = KMdata[['Kerosene','Butane']]
data_values = data_values.iloc[:,:].values
data_values


# In[ ]:


from sklearn.cluster import KMeans
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


kmeans = KMeans(n_clusters=5, init="k-means++", n_init=10, max_iter=300) 
KMdata["cluster"] = kmeans.fit_predict( data_values )
KMdata


# In[ ]:


KMdata['cluster'].value_counts()


# In[ ]:


temp = pd.DataFrame(KMdata['cluster'].value_counts())

sns.pairplot( KMdata, hue="cluster")


# In[ ]:


KMdata.groupby('Year')['Kerosene','Butane'].mean()


# In[ ]:




