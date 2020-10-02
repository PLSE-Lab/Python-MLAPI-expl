#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np


# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[8]:


sns.set()


# In[9]:


data = pd.read_csv('../input/911.csv')


# In[10]:


data.head()


# In[11]:


data.info()


# In[12]:


data.shape


# In[13]:


data.columns


# # 1. How many different types of calls?

# In[14]:


data.title.head()


# In[15]:


def call_type_separator(x):
    x = x.split(':')
    return x[0]


# In[16]:


data['call_type'] = data['title'].apply(call_type_separator)


# In[17]:


data.head(10)


# In[18]:


data['call_type'].unique()


# In[19]:


data['call_type'].value_counts()


# # 2. Extract date and time regarding calls.

# In[20]:


data['timeStamp'] = pd.to_datetime(data['timeStamp'], infer_datetime_format=True)


# In[21]:


data['timeStamp'].head()


# In[22]:


import datetime as dt


# In[23]:


data['year'] = data['timeStamp'].dt.year


# In[24]:


data['month'] = data['timeStamp'].dt.month_name()


# In[25]:


data['day'] = data['timeStamp'].dt.day_name()


# In[26]:


data['hour'] = data['timeStamp'].dt.hour


# In[27]:


data.head()


# In[28]:


def emergency_type_separator(x):
    x = x.split(':')
    x = x[1]
    return x


# # 3. At what time of day/month i can expect which time of call

# In[29]:


data['emergency_type'] = data['title'].apply(emergency_type_separator)


# In[30]:


data.head()


# In[31]:


data.head(2)


# In[32]:


call_types = data['call_type'].value_counts()
call_types


# In[33]:


from decimal import Decimal


# In[34]:


plt.figure(figsize=(15, 5))
ax = call_types.plot.bar()
for p in ax.patches:
    ax.annotate(Decimal(str(p.get_height())), (p.get_x(), p.get_height()))
plt.xticks(rotation=0)


# In[35]:


data.info()


# In[36]:


calls_data = data.groupby(['month', 'call_type'])['call_type'].count()


# In[37]:


calls_data.head(10)


# In[38]:


calls_data_percentage = calls_data.groupby(level=0).apply(lambda x: round(100*x/float(x.sum())))


# In[39]:


calls_data_percentage.head()


# In[40]:


font = {
    'size': 'x-large',
    'weight': 'bold'
}


# In[41]:


month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']


# In[42]:


calls_data_percentage = calls_data_percentage.reindex(month_order, level = 0)


# In[43]:


calls_data_percentage = calls_data_percentage.reindex(['EMS', 'Traffic', 'Fire'], level=1)


# In[44]:


calls_data_percentage.head()


# In[45]:


sns.set(rc={'figure.figsize':(12, 8)})
calls_data_percentage.unstack().plot(kind='bar')
plt.xlabel('Name of the Month', fontdict=font)
plt.ylabel('Percentage of Calls', fontdict=font)
plt.xticks(rotation=30)
plt.title('Calls/Month', fontdict=font)


# In[46]:


hours_data = data.groupby(['hour', 'call_type'])['call_type'].count()


# In[47]:


hours_data.head()


# In[48]:


hours_data_percentage = hours_data.groupby(level=0).apply(lambda x: round(100*x/float(x.sum())))


# In[49]:


hours_data_percentage.head()


# In[50]:


sns.set(rc={'figure.figsize':(18, 8)})
hours_data_percentage.unstack().plot(kind='bar')
plt.xlabel('Hour of the day', fontdict=font)
plt.ylabel('Percentage of Calls', fontdict=font)
plt.xticks(rotation=0)
plt.title('Calls/Hour', fontdict=font)
plt.savefig('hourly.png')


# # 4. Visualize percentage share of emergency type for each call

# In[51]:


data.head()


# In[52]:


ems_data=data[data['call_type']=='EMS']['emergency_type'].value_counts()[:5]


# In[53]:


fire_data=data[data['call_type']=='Fire']['emergency_type'].value_counts()[:5]


# In[54]:


traffic_data=data[data['call_type']=='Traffic']['emergency_type'].value_counts()[:5]


# In[55]:


plt.pie(ems_data,labels=ems_data.index,autopct="%.2f")
plt.savefig('piechart.png')


# In[56]:


plt.pie(fire_data,labels=fire_data.index,autopct="%.2f")
plt.savefig('piefir.png')


# In[57]:


plt.pie(traffic_data,labels=traffic_data.index,autopct="%.2f")
plt.savefig('pietraffic.png')


# In[ ]:




