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


import matplotlib.pyplot as plt


# In[ ]:


data=pd.read_csv('../input/911.csv')


# In[ ]:


data.head()


# In[ ]:


columns_name=list(data.columns)


# In[ ]:


columns_name


# In[ ]:


def call_type_separator(x):
    x=x.split(':')
    return x[0]


# In[ ]:


data['call_type']=data['title'].apply(call_type_separator)


# In[ ]:


data.head()


# In[ ]:


data.call_type.unique()


# In[ ]:


data.call_type.value_counts()


# ## **DATE AND TIME REGARDING THE CALLS**

# In[ ]:


data.timeStamp=pd.to_datetime(data['timeStamp'], infer_datetime_format=True)


# In[ ]:


data.info()


# In[ ]:


data.head()


# In[ ]:


import datetime as dt


# In[ ]:


data['year']=data['timeStamp'].dt.year
data['month']=data['timeStamp'].dt.month_name()
data['day']=data['timeStamp'].dt.day_name()
data['hour']=data['timeStamp'].dt.hour


# In[ ]:


data.head()


# ## SEPARATING THE EMERGANCY TYPE

# In[ ]:


def emergency_type_separator(x):
    x=x.split(':')
    return x[1]


# In[ ]:


data['emergency_type']=data['title'].apply(emergency_type_separator)


# In[ ]:


data.head()


# In[ ]:


call_types=data.call_type.value_counts()


# In[ ]:


call_types


# In[ ]:


from decimal import Decimal


# In[ ]:


plt.figure(figsize=(15,5))
call_types.plot.bar()


# ## Percentage of different call type in different months

# In[ ]:


calls_data=data.groupby(['month','call_type'])['call_type'].count()


# In[ ]:


calls_data.head()


# In[ ]:


calls_data_percentage = calls_data.groupby(level=0).apply(lambda x: round(100*x/float(x.sum())))


# In[ ]:


calls_data_percentage.head()


# In[ ]:


font = {
    'size': 'x-large',
    'weight': 'bold'
}


# In[ ]:


month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']


# In[ ]:


calls_data_percentage = calls_data_percentage.reindex(month_order, level=0)


# In[ ]:


calls_data_percentage.head()


# In[ ]:


calls_data_percentage = calls_data_percentage.reindex(['EMS', 'Traffic', 'Fire'], level=1)


# In[ ]:


calls_data_percentage.head()


# In[ ]:


import seaborn as sns


# In[ ]:


sns.set(rc={'figure.figsize':(12, 8)})
calls_data_percentage.unstack().plot(kind='bar')
plt.xlabel('Name of the Month', fontdict=font)
plt.ylabel('Percentage of Calls', fontdict=font)
plt.xticks(rotation=0)
plt.title('Calls/Month', fontdict=font)
plt.savefig('Calls per Month.png')


# ## Percentage of different call type at different hours

# In[ ]:


hours_data = data.groupby(['hour', 'call_type'])['call_type'].count()


# In[ ]:


hours_data.head()


# In[ ]:


hours_data_percentage = hours_data.groupby(level=0).apply(lambda x: round(100*x/float(x.sum())))


# In[ ]:


hours_data_percentage.head()


# In[ ]:


hours_data_percentage = hours_data_percentage.reindex(['EMS', 'Traffic', 'Fire'], level=1)


# In[ ]:


hours_data_percentage.head()


# In[ ]:


sns.set(rc={'figure.figsize':(18, 8)})
hours_data_percentage.unstack().plot(kind='bar')
plt.xlabel('Hour of the day', fontdict=font)
plt.ylabel('Percentage of Calls', fontdict=font)
plt.xticks(rotation=0)
plt.title('Calls/Hour', fontdict=font)
plt.savefig('Calls per Hour.png')


# ### EMS-DATA

# In[ ]:


ems_data = data[data['call_type'] == 'EMS']['emergency_type'].value_counts()[:8]


# In[ ]:


plt.pie(ems_data, labels=ems_data.index, autopct="%.2f")
plt.title('EMS-DATA',fontdict=font)


# ## Fire-Data

# In[ ]:


fire_data=data[data['call_type'] == 'Fire']['emergency_type'].value_counts()[:8]


# In[ ]:


plt.pie(fire_data, labels=fire_data.index, autopct="%.2f")
plt.title('FIRE-DATA', fontdict=font)


# ## Traffic-Data

# In[ ]:


traffic_data=data[data['call_type'] == 'Traffic']['emergency_type'].value_counts()[:5]


# In[ ]:


plt.pie(traffic_data, labels=traffic_data.index, autopct="%.2f")
plt.title('TRAFFIC-DATA', fontdict=font)


# In[ ]:




