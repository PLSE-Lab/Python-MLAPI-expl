#!/usr/bin/env python
# coding: utf-8

# In this notebook, we look at the spread of coronavirus at the local level, especially in China, for which data is the most rich and complete.

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


data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
data_lead = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv')
data_lead_2 = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv')


# In[ ]:


data.head()
from datetime import datetime
for x in range(0, len(data['ObservationDate'])):
   data['ObservationDate'][x] = datetime.strptime(data['ObservationDate'][x], "%m/%d/%Y")


# In[ ]:


data_china = data[data['Country/Region'] == 'Mainland China'].reset_index()
data_italy = data[data['Country/Region'] == 'Italy'].reset_index()
data_us = data[data['Country/Region'] == 'US'].reset_index()


# In[ ]:


data_china_agg = data_china.groupby(['ObservationDate']).sum()
data_italy_agg = data_italy.groupby(['ObservationDate']).sum()
data_us_agg = data_us.groupby(['ObservationDate']).sum()


# In[ ]:


import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)


# We have seen a lot of data on the national level, such as the below...

# In[ ]:


plt.plot(data_china_agg.index, data_china_agg['Confirmed'], label = 'China')
plt.plot(data_italy_agg.index, data_italy_agg['Confirmed'], label = 'Italy')
plt.plot(data_us_agg.index, data_us_agg['Confirmed'], label = 'United States')


# ... but in this notebook we take a look at how coronavirus spread through China on the local level. We have access to information on a provincial level (equivalent to US states)...

# In[ ]:


data_china['Province/State'].unique()


# We have a time series of how the virus spread provincially...

# In[ ]:


data_china.reset_index().pivot_table(index = 'ObservationDate', columns = 'Province/State', values = 'Confirmed').head()


# We can also take a look from a density and penetration standpoint. It is important to see not only how many cases there were, but how far this penetrated an area's population. Population data taken from Wikipedia...

# In[ ]:


data_china_unstack = data_china.groupby(['ObservationDate', 'Province/State']).sum()['Confirmed'].unstack()
china_population = pd.read_csv("/kaggle/input/china-population3/china population.csv")
china_population = china_population.rename(columns={"Province": "Province/State"})


# In[ ]:


data_china_merged = pd.merge(data_china, china_population)


# In[ ]:


data_china_merged['Penetration'] = data_china_merged['Confirmed']/data_china_merged['Population'].astype(float)
data_china_merged_ordered = data_china_merged.sort_values(by=['Density']).reset_index()
data_china_merged_ordered['Province/State'].unique()
data_china_merged_ordered_unstack = data_china_merged_ordered.groupby(['ObservationDate', 'Province/State']).sum()['Penetration'].unstack()
reordered=['Tibet', 'Qinghai', 'Xinjiang', 'Inner Mongolia', 'Gansu',
       'Heilongjiang', 'Ningxia', 'Yunnan', 'Jilin', 'Sichuan', 'Shaanxi',
       'Guangxi', 'Guizhou', 'Shanxi', 'Hainan', 'Jiangxi', 'Liaoning',
       'Fujian', 'Hubei', 'Hunan', 'Chongqing', 'Hebei', 'Anhui',
       'Zhejiang', 'Henan', 'Guangdong', 'Shandong', 'Jiangsu', 'Tianjin',
       'Beijing', 'Shanghai', 'Hong Kong', 'Macau']

data_china_merged_ordered_unstack = data_china_merged_ordered_unstack[reordered]
data_china_merged_ordered_unstack = data_china_merged_ordered_unstack.drop('Hong Kong', axis=1)
data_china_merged_ordered_unstack = data_china_merged_ordered_unstack.drop('Macau', axis=1)
data_china_merged_ordered_unstack = data_china_merged_ordered_unstack.drop('Hubei', axis=1)


# We look at each province, ordered from first to last by density...

# In[ ]:


fig, ax = plt.subplots(5,7,figsize=(50,50), sharex = True, sharey = True)
num=0

for column in data_china_merged_ordered_unstack:
    num+=1
    plt.ylim(0, 0.00003)
    plt.subplot(6, 6, num)
    plt.title(column)
    data_china_merged_ordered_unstack[column].plot(color = 'blue', linewidth=2.4, alpha=0.9)


# ... within each row we reorder by distance to Hebei (where the first outbreak in Wuhan occurred)...

# In[ ]:


reordered_distance=['Tibet', 'Qinghai', 'Xinjiang','Heilongjiang', 'Inner Mongolia', 'Gansu',
       'Yunnan', 'Jilin','Sichuan','Ningxia', 'Guangxi', 'Shaanxi',
          'Hainan',  'Liaoning','Guizhou','Shanxi','Fujian','Jiangxi',
        'Hebei','Chongqing','Zhejiang','Hunan','Anhui', 'Henan', 
        'Beijing','Jiangsu','Tianjin','Guangdong',  'Shandong', 'Shanghai']
data_china_merged_ordered_unstack = data_china_merged_ordered_unstack[reordered_distance]


# In[ ]:


fig, ax = plt.subplots(5,7,figsize=(50,50), sharex = True, sharey = True)
num=0

for column in data_china_merged_ordered_unstack:
    num+=1
    plt.ylim(0, 0.00003)
    plt.subplot(6, 6, num)
    plt.title(column)
    data_china_merged_ordered_unstack[column].plot(color = 'blue', linewidth=2.4, alpha=0.9)


# If you take a look at Italy, you can see it already has surpassed China's penetration levels...

# In[ ]:


plt.plot(data_italy_agg.index, data_italy_agg['Confirmed']/60480000)


# It is still early days for the US, but from historic examples, it is clear this must be stopped quickly and decisively.

# In[ ]:


data_us_unstack = data_us.groupby(['ObservationDate', 'Province/State']).sum()['Confirmed'].unstack()
data_us_unstack_filtered = data_us_unstack.loc[:, data_us_unstack.max() > 50]
fig, ax = plt.subplots(figsize=(25,15))
data_us_unstack_filtered.plot(ax=ax)


# In[ ]:


plt.plot(data_us_agg.index, data_us_agg['Confirmed']/300000000)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


fig, ax = plt.subplots(figsize=(25,15))
data_china_merged_unstack_nohubei.plot(ax=ax)


# In[ ]:





# In[ ]:


fig, ax = plt.subplots(5,7,figsize=(50,50), sharex = True, sharey = True)
num=0

for column in data_china_merged_ordered_unstack:
    num+=1
    plt.ylim(0, 0.00003)
    plt.subplot(6, 6, num)
    plt.title(column)
    data_china_merged_ordered_unstack[column].plot(color = 'blue', linewidth=2.4, alpha=0.9)


# In[ ]:


data_italy['Penetration'] = data_italy['Confirmed'] / 60480000


# In[ ]:


fig, ax = plt.subplots(figsize=(25,15))
plt.plot(data_italy['ObservationDate'], data_italy['Penetration'])

