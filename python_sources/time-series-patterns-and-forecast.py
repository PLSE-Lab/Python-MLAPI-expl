#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sb
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Reading csv using pandas
import pandas as pd 

adf = pd.read_csv('../input/avocado.csv')
print(adf.head())


# In[ ]:


adf.info()


# In[ ]:


adf.describe()


# In[ ]:


#First column uncertain: No column name given, repeatin after 0-51 counts! strange!
adf.drop(['Unnamed: 0'],axis=1,inplace=True)
adf.head()


# In[ ]:


#null check 
#adf.isnull() --> only gives binary mask 
adf.isnull().sum()


# In[ ]:



adf_US = adf[adf['region']=='TotalUS']
adf_US_organic = adf_US[adf_US['type']=='organic']
adf_US_organic.head()


# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Distribution of Avg Price")
#average_price_fit = stats.norm.pdf(adf['AveragePrice'],np.mean(adf['AveragePrice']),np.std(adf['AveragePrice']))
plt.xlabel('Average Price')
plt.ylabel('Probability')
#plt.hist(adf['AveragePrice'],bins=40,color='g')
#plt.plot(adf['AveragePrice'],average_price_fit)
sb.distplot(adf_US_organic["AveragePrice"],hist=True,kde=True,rug=True,bins=100, color = 'b')


# ### Observation 1 : 
# Average price is more between 1 to 1.5, but sometimes it went to 3 also.
# 

# In[ ]:


#creating month column
adf['Date'] = pd.to_datetime(adf['Date'], format='%Y-%m-%d')
adf['Month']=adf['Date'].map(lambda x: x.month)
adf = adf.sort_values(by='Date')


# In[ ]:


plt.figure(figsize=(12,3))
sb.lineplot(x="Date", y="AveragePrice",hue='year',data=adf_US_organic,palette='magma')


# In[ ]:


plt.figure(figsize=(12,3))
sb.lineplot(x="Date", y="Total Volume",hue='year',data=adf_US_organic,palette='magma',)


# In[ ]:


plt.figure(figsize=(12,3))
sb.lineplot(x="Date", y="Total Bags",hue='year',data=adf_US_organic,palette='magma')
#sb.lineplot(x="Month", y="Total Volume",hue='year',data=adf_US_organic,palette='copper')


# In[ ]:


plt.figure(figsize=(12,3))
sb.lineplot(x="Date", y="4046",hue='year',data=adf_US_organic,palette='magma',)


# In[ ]:


plt.figure(figsize=(12,3))
sb.lineplot(x="Date", y="4225",hue='year',data=adf_US_organic,palette='magma')


# In[ ]:


plt.figure(figsize=(12,3))
sb.lineplot(x="Date", y="4770",hue='year',data=adf_US_organic,palette='magma')


# ### Observation 2 :
# From month 6 to 11, average prices are increasing in 2017 and 2016.
# And there is definetly a pattern.

# # Trying to train network
# 

# In[ ]:


adf_US_organic = adf_US_organic.sort_values(by='Date')
# Valid = adf[(adf['year'] == 2017) | (adf['year'] == 2018)]
# Train = adf[(adf['year'] != 2017) & (adf['year'] != 2018)]
Train = adf_US_organic.sort_values(by='Date')


# ### Total Volume

# In[ ]:


from fbprophet import Prophet


# In[ ]:


m = Prophet()
date_volume = Train.rename(columns={'Date':'ds', 'Total Volume':'y'})
m.fit(date_volume)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)


# In[ ]:


fig1 = m.plot(forecast)


# In[ ]:


fig2 = m.plot_components(forecast)


# ## Total Bags

# In[ ]:


n = Prophet()
date_bags = Train.rename(columns={'Date':'ds', 'Total Bags':'y'})
n.fit(date_bags)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
fig2 = m.plot(forecast)


# In[ ]:


fig2 = m.plot_components(forecast)


# # Thanks for reading my notebook!

# In[ ]:




