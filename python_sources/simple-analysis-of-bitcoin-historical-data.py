#!/usr/bin/env python
# coding: utf-8

# # SUMMARY
# I will use bitcoin historical data for my first data science experience. Besides basic scatter and line plot examples, this work contains correlation analysis, simple functions of matplotlib, pandas, numpy etc. 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.dates as md
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool
import datetime as dt
import time
import os
print(os.listdir("../input"))


#  **Timestamp Conversion**

# In[ ]:


def parseTimeStamp (timestamp):    
    return dt.datetime.fromtimestamp(np.int64(timestamp))


#  **Read the file**

# In[ ]:


data = pd.read_csv('../input/bitstampUSD_1-min_data_2012-01-01_to_2018-06-27.csv', parse_dates=[0], date_parser=parseTimeStamp)


#  **Rename columns**

# In[ ]:


data = data.rename(columns = {'Volume_(BTC)':'VolumeBTC', 'Volume_(Currency)' : 'VolumeCurrency', 'Weighted_Price' : 'WeightedPrice' })


#  **Months and Years Seperation**

# In[ ]:


data['Year'] = years = [x.year for x in data.Timestamp]
data['Month'] = years = [x.month for x in data.Timestamp]


# In[ ]:


# adjusting VolumeBTC threshold
threshold = sum(data.VolumeBTC)/len(data.VolumeBTC)
print('BTC Volume Threshold is : ', threshold)
data["VolumeLevel"] = ["high" if i > threshold else "low" for i in data.VolumeBTC]
data.loc[:10,["VolumeLevel","VolumeBTC"]]


# In[ ]:


# rows and column counts
data.shape


# In[ ]:


# all data columns
data.columns


# In[ ]:


# info
data.info()


# In[ ]:


data["VolumeLevel"].value_counts(dropna = False)
# There are no NaN values
# Returns nothing because we drop nan values
assert  data['VolumeLevel'].notnull().all() 
assert data.Month.dtypes == np.int


# In[ ]:


data.head(10)


# In[ ]:


data.tail(10)


# In[ ]:


# VolumeLevel frequency
print(data['VolumeLevel'].value_counts(dropna = False))


# In[ ]:


# tidy data
new_data = data.tail(10)
new_data


# In[ ]:


# melted data
melted = pd.melt(frame = new_data, id_vars = 'Timestamp', value_vars= ['WeightedPrice','VolumeLevel'])
melted


# In[ ]:


# pivot example
melted.pivot(index = 'Timestamp', columns = 'variable',values='value')


# In[ ]:


# concatenating example
# axis = 0 : adds dataframes in row - concat row
data1 = data.head()
data2 = data.tail()
conc_data_row = pd.concat([data1, data2], axis = 0, ignore_index = True) 
conc_data_row


# In[ ]:


# axis = 1 : adds dataframes in column - concat column
data1 = data['WeightedPrice'].head(10)
data2 = data['VolumeBTC'].head(10)
conc_data_col = pd.concat([data1, data2], axis = 1) 
conc_data_col


# In[ ]:


data.dtypes


# In[ ]:


data['Year'] = data['Year'].astype('float')
data.dtypes


# In[ ]:


data.describe()


# In[ ]:


data.boxplot(column='VolumeBTC',by = 'Year', figsize=(10,10))


# In[ ]:


data.corr()


# In[ ]:


# correlation map for bitcoin data
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# **Subplots Example**

# In[ ]:


# subplots
data_new = data.loc[:,["Open","Close","High", "Low"]]
data_new.plot(subplots = True, figsize=(10,10))
plt.show()


# In[ ]:


# scatter plot  
data.plot(kind = "scatter",x = "WeightedPrice", y = "VolumeBTC", figsize=(10,10))
plt.show()


# In[ ]:


# histogram
data_new.plot(kind = "hist", y = "High",bins = 50, range= (0, 250), normed = True, figsize=(10,10))


# **Histogram Cumulative Non-Cumulative**

# In[ ]:


# histogram cumulative and non cumulative sample
fig, axes = plt.subplots(nrows = 2 ,ncols = 1)
data_new.plot(kind = "hist", y = "High", bins = 50, range = (0, 250), normed = True, ax = axes[0], figsize=(10,10))
data_new.plot(kind = "hist", y = "High", bins = 50, range = (0, 250), normed = True, ax = axes[1], cumulative = True, figsize=(10,10))
plt.savefig('graph.png')
plt


# In[ ]:


data_head = data.head()
date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]
datetime_object = pd.to_datetime(date_list)
data_head["date"] = datetime_object
# make date as index
data_head = data_head.set_index("date")
data_head 


# In[ ]:


# select according to date index
print(data_head.loc["1993-03-16"])
print(data_head.loc["1992-03-10":"1993-03-16"])


# In[ ]:


# data2 year resample
data_head.resample("A").mean()


# In[ ]:


# data2 month resample
data_head.resample("M").mean()


# In[ ]:


# interpolete from first value
data_head.resample("M").first().interpolate("linear")


# In[ ]:


# interpolate with mean()
data_head.resample("M").mean().interpolate("linear")


# In[ ]:


# indexing examples
data_head = data.head(20)
data_head[["Open","Close"]]
#data_head.Open
#data_head["Open"][1]
#data_head.loc[1,["Open"]]
#data_head.Open[2]


# In[ ]:


# slicing examples
print(type(data["Open"]))     # series
print(type(data[["Open"]]))   # data frames


# In[ ]:


data.loc[1:10,"Open":"Close"] # from column to column
#data.loc[10:1:-1, "Open" : "Close"] # reverse slice
#data.loc[1:10,"Open":] #a column to end


# In[ ]:


# filter examples
first_filter = data.Open > 10000
second_filter = data.VolumeBTC > 400
data[first_filter & second_filter]
#data.Open[data.VolumeBTC > 400]


# In[ ]:


# function example
data_multiplier = data.head()
def multiplier(x):
    return x*2
data_multiplier.Open.apply(multiplier)
#data_multiplier.Open.apply(lambda n : n/2) same results as above


# In[ ]:


# Defining column using other columns
data["openminusclose"] = data.Open - data.Close
data.head()


# In[ ]:


# indexing
print(data.index.name)
data.index.name = "new_index_name"
data.head()


# In[ ]:


data.head()
copied_data = data.head()
# changing index order
copied_data.index = range(100,600,100)
copied_data.head()

# could be
# data= data.set_index("#")
# also you can use 
# data.index = data["#"]


# In[ ]:


# hierarchical indexing
data_reload = pd.read_csv('../input/bitstampUSD_1-min_data_2012-01-01_to_2018-06-27.csv')
data_reload.tail()


# In[ ]:


# setting index
data_reload = data_reload.set_index(["Open","Close"]) 
data_reload.tail(100)
# data1.loc["Fire","Flying"] # howw to use indexes


# In[ ]:


# pivoting
data_reload_head = data_reload.tail()
data_reload_head.pivot(index = "Timestamp", columns = "Volume_(BTC)", values="Weighted_Price")


# In[ ]:


data_reload_head2 = data_reload_head.set_index(["Timestamp","Weighted_Price"])
data_reload_head2


# In[ ]:


# level determines indexes
data_reload_head2.unstack(level = 0)


# In[ ]:


data_reload_head2.unstack(level = 1)


# In[ ]:


# change inner and outer level index position
data_reload_head3 = data_reload_head2.swaplevel(0,1)
data_reload_head3


# In[ ]:


data_reload_head


# In[ ]:


#data_reload_head.pivot(index = "Timestamp", columns = "Volume_(BTC)", values="Weighted_Price")
pd.melt(data_reload_head,id_vars="Timestamp",value_vars=["Volume_(BTC)","Weighted_Price"])


# In[ ]:


# grouping
data_reload_head.groupby("Timestamp").mean()   # mean is aggregation / reduction method


# In[ ]:


data_reload_head.groupby("High").High.max()


# In[ ]:


data_reload_head.groupby("Open")[["High","Low"]].min() 


# In[ ]:


data_reload_head.info()
# convert objects to categorical, because categorical data uses less memory, speed up operations like groupby


#  **WeightedBTCPrice Line Chart**

# In[ ]:


data.WeightedPrice.plot(kind = 'line', color = 'g',label = 'WeightedBTCPrice',linewidth=1,alpha = 1,grid = True,linestyle = '-', figsize=(10,10))
plt.legend(loc='upper right')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('WeightedBTCPrice')
plt.show()


#  **Year - VolumeBTC plot**

# In[ ]:


plt.figure(figsize=(10,10))
plt.plot(data.Year, data.VolumeBTC)
plt.legend(loc='upper right')
plt.xlabel('Year')
plt.ylabel('VolumeBTC')
plt.title('VolumeBTC during years')
plt.show()


# In[ ]:


# filtering the 2018 data
currentYear = dt.datetime(2018, 1, 1, 0, 0, 0, 0)
currentYearData = data[(data['Timestamp'] >= currentYear)]
currentYearData.tail(10)


# **BTC Volume - line chart of 2018**

# In[ ]:


currentYearData.VolumeBTC.plot(kind = 'line', color = 'b',label = 'VolumeBTC',linewidth=1,alpha = 0.5,grid = True,linestyle = ':', figsize=(15,10))
plt.legend(loc='upper right')
plt.xlabel('x axis')             
plt.ylabel('y axis')
plt.title('Line Plot')      
plt.show()


# **Scatter sample for bitcoin highest values by months for 2018**

# In[ ]:


currentYearData.plot(kind='scatter', x='Month', y='High',alpha = 0.5,color = 'red', figsize=(15,10))
plt.xlabel('Month')              
plt.ylabel('Highest Price')
plt.title('Month - Highest Price Scatter Plot') 

