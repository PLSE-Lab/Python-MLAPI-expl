#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()


# In[ ]:


(market_train_df, news_train_df) = env.get_training_data()


# In[ ]:


market_train_df.dtypes


# In[ ]:


market_train_df.tail()


# * Time ==> DateTime
# * AssetCode ==> Object
# * AssetName ==> Category
# * Rest all our float

# In[ ]:


market_train_df.isna().sum()


# In[ ]:


import matplotlib as plt
((market_train_df.isnull().sum()/market_train_df.shape[0])*100).sort_values(ascending=False).plot(kind='bar')


# **Null found , we see that returnsOpenPrevMktres10 and returnsClosePrevMktres10 have most nulls

# Lets check # of unique values

# In[ ]:


market_train_df.nunique()


# #Lets check the count of each month,year

# In[ ]:





# In[ ]:


print("Min date: ",market_train_df['time'].min())
print("Max date: ",market_train_df['time'].max())


# In[ ]:


market_train_df['time'].dt.time.describe()


# Lets check asset code and associated asset name

# In[ ]:


market_train_df[['assetCode','assetName']].head()


# From the above analysis, we can see that assetcodes are more than asset names which means some of the assets codes might have null in them.

# In[ ]:


print("unique asset name", market_train_df['assetName'].nunique())
print("unique asset code", market_train_df['assetCode'].nunique())
print("Difference", abs(market_train_df['assetName'].nunique()-market_train_df['assetCode'].nunique()))


# In[ ]:


market_train_df[market_train_df['assetName']=='Unknown'].head()


# This tells us that for some asset codes, we dont have asset name

#  assetCode with unknown assetName

# In[ ]:


unknown_assetname_codes=market_train_df[market_train_df['assetName']=='Unknown']['assetCode'].unique()
unknown_assetname_codes


# > Check error in data for assetcodes with unknown assetname, 0 ==> no error

# In[ ]:


import numpy as np
for code in unknown_assetname_codes:
   print(np.count_nonzero(market_train_df[market_train_df['assetCode']==code]['assetName']!='Unknown'))


# Check some of the stocks

# In[ ]:


market_train_df[market_train_df['assetCode']=='AAPL.O']


# time==>shows that market is closed on saturdays and sundays, also on official holidays

# In[ ]:


market_train_df[market_train_df['assetCode']=='A.N']


# Lets check Apple Volumes 

# In[ ]:


def volume_trend(assetCode):
    market_train_df[market_train_df['assetCode']==assetCode].sort_values(by='time',ascending=True)[['time','assetCode','volume']].groupby(['time','assetCode']).sum().plot(kind='line',figsize=(25,5))


# In[ ]:


volume_trend('AAPL.O')


# Apple Volumes have increased in recent years

# Lets see top volumes on latest date, top 10

# In[ ]:


top_10_byvolume=market_train_df[(market_train_df['time'].dt.year==2016)&(market_train_df['time'].dt.day==30)&(market_train_df['time'].dt.month==12)].sort_values(by='volume',ascending=False)[['assetCode','volume']].head(10)


# In[ ]:


import matplotlib.pyplot as plt

market_train_df[market_train_df['assetCode'].isin(list(top_10_byvolume['assetCode']))].sort_values(by='time',ascending=True)[['time','assetCode','volume']].groupby(['time','assetCode']).sum().unstack().plot(figsize=(25,10))
#plot(x='time', y='volume')
#market_train_df[market_train_df['assetCode']==assetCode].sort_values(by='time',ascending=True)[['time','assetCode','volume']].groupby(['time','assetCode']).sum().plot(kind='line',figsize=(25,5))


# Bank of America is at top but recently during last 4 to 5 years, volume has decreased

# In[ ]:


#some of the stocks have data for few years
market_train_df[market_train_df['assetCode']=='AMD.O']['time'].dt.year.unique()


# Lets group it monthly and then check

# In[ ]:


market_train_df[market_train_df['assetCode'].isin(list(top_10_byvolume['assetCode']))].sort_values(by='time',ascending=True)[['time','assetCode','volume']].groupby([market_train_df['time'].dt.month,market_train_df['time'].dt.year,'assetCode']).median().unstack().plot(figsize=(25,10),kind='line')


# We also have apple inc on top 10 , lets check that

# In[ ]:


market_train_df[market_train_df['assetCode']=='AAPL.O'].sort_values(by='time',ascending=True)[['time','assetCode','volume']].groupby([market_train_df['time'].dt.month,market_train_df['time'].dt.year,'assetCode']).median().unstack().plot(figsize=(25,10),kind='line')


# In[ ]:


market_train_df[market_train_df['assetCode']=='AAPL.O'].sort_values(by='time',ascending=True)[['time','assetCode','volume']].groupby([market_train_df['time'].dt.month,market_train_df['time'].dt.year,'assetCode']).median().unstack().plot(figsize=(25,10),kind='bar')


# Pattern can easily be detected
# 

# In[ ]:


market_train_df[market_train_df['assetCode']=='BAC.N'].sort_values(by='time',ascending=True)[['time','assetCode','volume']].groupby([market_train_df['time'].dt.month,market_train_df['time'].dt.year,'assetCode']).median().unstack().plot(figsize=(25,10),kind='line')


# In[ ]:


market_train_df[market_train_df['assetCode']=='BAC.N'].sort_values(by='time',ascending=True)[['time','assetCode','volume']].groupby([market_train_df['time'].dt.month,market_train_df['time'].dt.year,'assetCode']).median().unstack().plot(figsize=(25,10),kind='bar')


# Pattern can easily be seen here, time series analysis can be performed here

# Lets check volumes yearly

# In[ ]:


market_train_df[market_train_df['assetCode'].isin(list(top_10_byvolume['assetCode']))].sort_values(by='time',ascending=True)[['time','assetCode','volume']].groupby([market_train_df['time'].dt.year,'assetCode']).mean().unstack().plot(figsize=(25,10),kind='line')


# In[ ]:


market_train_df['volume'].describe()


# Lets now analyze closing prices

# In[ ]:


market_train_df['close'].describe()


# In[ ]:


market_train_df[market_train_df['assetCode']=='BAC.N'].head()


# Check closing price trend of BAC.N

# In[ ]:


market_train_df[market_train_df['assetCode']=='BAC.N'].sort_values(by='time',ascending=True)[['time','close']].groupby('time').median().plot(figsize=(25,10),kind='line')


# Lets check closing price trend month, year basis

# In[ ]:


market_train_df[market_train_df['assetCode']=='BAC.N'].sort_values(by='time',ascending=True)[['time','close']].groupby([market_train_df['time'].dt.month,market_train_df['time'].dt.year]).median().plot(figsize=(25,10),kind='line')


# Perfect trend can be seen, timeseries analysis can easily be performed here

# Lets pick random assets and plot closing price

# In[ ]:


import random
num_to_select = 5                          # set the number to select here.
list_of_random_assets = random.sample(list(set(market_train_df['assetCode'])), num_to_select)
list_of_random_assets 


# In[ ]:


market_train_df[market_train_df['assetCode'].isin(list(list_of_random_assets))].sort_values(by='time',ascending=True)[['time','assetCode','close']].groupby(['time','assetCode']).median().unstack().plot(figsize=(25,10))


# this is random plot every time you make.
# From this random plot I see that there are some companies who only traded for few years. May be it was then acquired by some other company. Here I see one of the company SGI.O which I see was acquired by some other company.
# 
# Stocks of some of the company went up while stocks of some of the company went down may be due to bankruptcy. 
# 
# Some are stable.
# 
# Lets see assetCode='SGI.O''
# 
# You may see other asset codes

# In[ ]:


list(set(market_train_df[market_train_df['assetCode']=='SGI.O']['time'].dt.year))


# We can see that this company only traded in 2011. So there are companies which traded for few years may be due to acquisition, bankruptcy etc 

# In[ ]:


market_train_df[market_train_df['assetCode'].isin(list(list_of_random_assets))].sort_values(by='time',ascending=True)[['time','assetCode','close']].groupby([market_train_df['time'].dt.month,market_train_df['time'].dt.year,'assetCode']).median().unstack().plot(figsize=(25,10))


# In[ ]:


market_train_df[market_train_df['assetCode'].isin(list(list_of_random_assets))].sort_values(by='time',ascending=True)[['time','assetCode','close']].groupby([market_train_df['time'].dt.month,market_train_df['time'].dt.year,'assetCode']).median().unstack().plot(figsize=(25,10),kind='bar')


# Perfect trend can be seen here, this means we can model using time and closing price

# Lets now analyze and corelate both volume and closing price
# 

# Describe each asset for particular year

# In[ ]:


desc_assets=market_train_df[market_train_df['time'].dt.year==2016].groupby('assetCode').describe()


# In[ ]:


desc_assets


# lets see boxplot of closing prices of top 10 volumes for year 2016

# In[ ]:


#list(top_10_byvolume['assetCode'])
desc_assets['close'].transpose()[list(top_10_byvolume['assetCode'])].boxplot(figsize=(25,10))


# Picking one of the assets

# In[ ]:


market_train_df[market_train_df['assetCode']=='BAC.N'][['time','assetCode','volume','close','open']].head()


# As we see right over here, closing of one day is not same as next day opening. This is due to price discovery which I read about. (effected by speculation). Here news data can help us

# In[ ]:


#market_train_df[market_train_df['assetCode']=='BAC.N'][['time','assetCode','volume','close','open']]

