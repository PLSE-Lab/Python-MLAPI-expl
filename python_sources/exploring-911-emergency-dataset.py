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


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from decimal import Decimal
import datetime as dt


# In[ ]:


data=pd.read_csv("../input/montcoalert/911.csv")


# In[ ]:


sns.set()


# In[ ]:


data.head()


# In[ ]:


data.desc[0]


# In[ ]:


data.shape


# In[ ]:


data.info()


# In[ ]:


data.columns


# In[ ]:


data['title'].head()


# In[ ]:


data['call_type']=data['title'].apply(lambda x :x.split(':')[0])


# In[ ]:


()


# In[ ]:


data['Emergency_type']=data['title'].apply(lambda x :x.split(':')[1])


# In[ ]:


data['call_type'].nunique()


# In[ ]:


data['call_type'].unique()


# In[ ]:


call_types=data['call_type'].value_counts()


# In[ ]:


call_types


# In[ ]:





# In[ ]:





# In[ ]:


plt.figure(figsize=(15, 5))
ax = call_types.plot(kind='bar')
for p in ax.patches:
    ax.annotate(Decimal(str(p.get_height())), (p.get_x(), p.get_height()))
plt.xticks(rotation=0)
plt.savefig("type_of_call.png")


# In[ ]:


data['timeStamp']=pd.to_datetime(data['timeStamp'])


# In[ ]:


data['timeStamp'].head()


# In[ ]:


data.info()


# In[ ]:





# In[ ]:


data['Year']=data['timeStamp'].dt.year


# In[ ]:


data['Month']=data['timeStamp'].dt.month_name()


# In[ ]:


data['Hour']=data['timeStamp'].dt.hour


# In[ ]:


data['day']=data['timeStamp'].dt.day_name()


# In[ ]:


data['emergency_type'] = data['title'].apply(lambda x:x.split(':')[1])


# In[ ]:


calls_data = data.groupby(['Month', 'call_type'])['call_type'].count()


# In[ ]:


calls_data_percentage = calls_data.groupby(level=0).apply(lambda x: round(100*x/float(x.sum())))


# In[ ]:


calls_data_percentage.head()


# In[ ]:


calls_data_percentage = calls_data_percentage.reindex(['EMS', 'Traffic', 'Fire'], level=1)


# In[ ]:


month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']


# In[ ]:


calls_data_percentage = calls_data_percentage.reindex(month_order, level=0)


# In[ ]:


calls_data_percentage.head()


# In[ ]:


font = {
    'size': 'x-large',
    'weight': 'bold'
}


# In[ ]:


sns.set(rc={'figure.figsize':(12, 8)})
calls_data_percentage.unstack().plot(kind='bar')
plt.xlabel('Name of the Month', fontdict=font)
plt.ylabel('Percentage of Calls', fontdict=font)
plt.xticks(rotation=0)
plt.title('Calls/Month', fontdict=font)
plt.savefig('call-month.png')


# In[ ]:


hours_data = data.groupby(['Hour', 'call_type'])['call_type'].count()


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
plt.savefig('call-hour.png')


# In[ ]:


ems_data = data[data['call_type'] == 'EMS']['emergency_type'].value_counts()[:5]


# In[ ]:


fire_data = data[data['call_type'] == 'Fire']['emergency_type'].value_counts()[:5]


# In[ ]:


traffic_data = data[data['call_type'] == 'Traffic']['emergency_type'].value_counts()[:5]


# In[ ]:


plt.pie(ems_data, labels=ems_data.index, autopct="%.2f")
plt.savefig('Ems-calls.png')


# In[ ]:


plt.pie(fire_data, labels=fire_data.index, autopct="%.2f")
plt.savefig('Fire-calls.png')


# In[ ]:


plt.pie(traffic_data, labels=traffic_data.index, autopct="%.2f")
plt.savefig('Traffic-calls.png')


# In[ ]:





# In[ ]:




