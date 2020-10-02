#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


sns.set()


# In[4]:


data = pd.read_csv('../input/911.csv')


# In[5]:


data.head()


# In[6]:


data.shape


# In[7]:


data.info()


# In[8]:


columns_names=list(data.columns)


# In[9]:


columns_names


# # 1.How many different types of calls?

# In[10]:


data.title.head()


# In[11]:


def call_type_separator(x):
    x = x.split(':')
    return x[0]


# In[12]:


data['call_type'] = data['title'].apply(call_type_separator)


# In[13]:


data.head()


# In[14]:


data['call_type'].unique()


# In[15]:


data['call_type'].value_counts()


# In[16]:


call_types=data['call_type'].value_counts()


# In[17]:


from decimal import Decimal


# In[18]:


plt.figure(figsize=(20, 5))
ax = call_types.plot(kind='bar')
for p in ax.patches:
    ax.annotate(Decimal(str(p.get_height())), (p.get_x(), p.get_height()))
plt.xticks(rotation=0)
plt.savefig('1.png')


# # 2. Extract date and time regarding calls

# In[19]:


data['timeStamp']=pd.to_datetime(data['timeStamp'])


# In[20]:


data['timeStamp'].head()


# In[21]:


data.info()


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


# # 3. At what time of day/month I can expect which type of call?

# In[28]:


data['emergency_type'] = data['title'].apply(lambda x:x.split(':')[1])


# In[29]:


data.head()


# In[30]:


calls_data = data.groupby(['month', 'call_type'])['call_type'].count()


# In[31]:


calls_data.head()


# In[32]:


calls_data_percentage = calls_data.groupby(level=0).apply(lambda x: round(100*x/float(x.sum())))


# In[33]:


calls_data_percentage.head()


# In[34]:


font = {
    'size': 'x-large',
    'weight': 'bold'
}


# In[35]:


month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']


# In[36]:


calls_data_percentage = calls_data_percentage.reindex(month_order, level=0)


# In[37]:


calls_data_percentage = calls_data_percentage.reindex(['EMS', 'Traffic', 'Fire'], level=1)


# In[38]:


calls_data_percentage.head()


# In[39]:


sns.set(rc={'figure.figsize':(12, 8)})
calls_data_percentage.unstack().plot(kind='bar')
plt.xlabel('Name of the Month', fontdict=font)
plt.ylabel('Percentage of Calls', fontdict=font)
plt.xticks(rotation=0)
plt.title('Calls/Month', fontdict=font)
plt.savefig('2monthly.png')


# In[40]:


hours_data = data.groupby(['hour', 'call_type'])['call_type'].count()


# In[41]:


hours_data.head()


# In[42]:


hours_data_percentage = hours_data.groupby(level=0).apply(lambda x: round(100*x/float(x.sum())))


# In[43]:


hours_data_percentage.head()


# In[44]:


hours_data_percentage = hours_data_percentage.reindex(['EMS', 'Traffic', 'Fire'], level=1)


# In[45]:


sns.set(rc={'figure.figsize':(18, 8)})
hours_data_percentage.unstack().plot(kind='bar')
plt.xlabel('Hour of the day', fontdict=font)
plt.ylabel('Percentage of Calls', fontdict=font)
plt.xticks(rotation=0)
plt.title('Calls/Hour', fontdict=font)
plt.savefig('2hourly.png')


# # 4. Visualize percentage share of emergency type for each call

# In[46]:


ems_data=data[data['call_type']=='EMS']['emergency_type'].value_counts()[:5]


# In[47]:


fire_data=data[data['call_type']=='Fire']['emergency_type'].value_counts()[:5]


# In[48]:


traffic_data=data[data['call_type']=='Traffic']['emergency_type'].value_counts()[:5]


# In[49]:


plt.pie(ems_data,labels=ems_data.index,autopct='%.2f')
plt.savefig('1pie.png')


# In[50]:


plt.pie(fire_data,labels=fire_data.index,autopct='%.2f')
plt.savefig('2pie.png')


# In[51]:


plt.pie(traffic_data,labels=traffic_data.index,autopct='%.2f')
plt.savefig('3pie.png')

