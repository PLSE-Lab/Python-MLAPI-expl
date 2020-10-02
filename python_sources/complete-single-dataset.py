#!/usr/bin/env python
# coding: utf-8

# In this Kernel I am trying to merge all the dataframes into a single dataframe in a good manner because initially I found it difficult to understand data from many different files.You can use this data to get a basic understanding of the problem and even utilize it further to perform various visualizations.
# ### If this helped you,don't forget to upvote! 
# ![](https://cdn.mos.cms.futurecdn.net/QK4BW2pck8CJePSRDeivue-970-80.jpg)
# 
# ### Thanks a lot

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


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re


# In[ ]:


corona_cleaned = pd.read_csv("/kaggle/input/2019-coronavirus-dataset-01212020-01262020/2019_nC0v_20200121_20200126_cleaned.csv")
corona_summary = pd.read_csv("/kaggle/input/2019-coronavirus-dataset-01212020-01262020/2019_nC0v_20200121_20200126 - SUMMARY.csv")





# We are cobining the data except for summary and cleaned data frame into one dataframe.This code should work fine even if updated data unless another summary or cleaned file is uploaded.But if new file ends with cleaned.csv or summary.csv(any alphabet case) it will be no problem.

# In[ ]:


name = os.listdir('/kaggle/input/2019-coronavirus-dataset-01212020-01262020')
address = list()
for x in name:
    if(re.search("CLEANED.csv$|Cleaned.csv$|cleaned.csv$|SUMMARY.csv$|Summary.csv$|summary.csv$",x)== None):
        address.append(x)
address


# In[ ]:



datafull = pd.DataFrame()
for x in address:
    df = pd.read_csv('/kaggle/input/2019-coronavirus-dataset-01212020-01262020/'+x,parse_dates=['Last Update'])
    datafull = pd.concat([datafull,df])


# In[ ]:


datafull


# Now we have combined the data.We will make various operations to make the data usable and clean.

# In[ ]:


#There are duplicates in our data


# In[ ]:


datafull.drop_duplicates(inplace = True)


# In[ ]:


datafull.sort_values(by = ['Province/State','Last Update'],ascending=False,inplace=True)


# In[ ]:


datafull


# In[ ]:


#Making Last update as index in a new dataframe data_series


# In[ ]:


data_series =datafull.set_index('Last Update')


# In[ ]:


#Converting Missing values in provinces into "Unknown"


# In[ ]:


data_series['Province/State'].fillna('Unknown',inplace=True)


# In[ ]:


#Making data into bins of 1 day each instead of having multiple values in each day


# In[ ]:


group = data_series.groupby('Province/State')


# In[ ]:


time_series = pd.DataFrame()
for name in data_series['Province/State'].unique():
    time_series = pd.concat([time_series,group.get_group(name).resample('1D').max()])


# In[ ]:


#Treating Missing Values


# In[ ]:


time_series["Province/State"] = time_series["Province/State"].fillna("Unknown")
cols = ['Confirmed','Suspected','Recovered','Death']
for col in cols:
    time_series[col].fillna(value = 0,inplace = True)


# In[ ]:


time_series.isna().sum()


# In[ ]:


#This missing data left is basically completely empty rows 


# In[ ]:


time_series[time_series["Country/Region"].isna()==True]


# In[ ]:


#We will drop
time_series.dropna(inplace=True)


# In[ ]:


time_series


# # This data is day wise time series of events.We can use this data to perform further analysis.

# In[ ]:





# In[ ]:




