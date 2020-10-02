#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np


# In[11]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[12]:


sns.set()


# In[13]:


data=pd.read_csv('../input/911.csv')


# In[14]:


data.head()


# In[15]:


data.shape


# In[16]:


data.info()


# In[17]:


data.columns


# ### 1.How many different type of calls?

# In[18]:


data['title'].head()


# In[19]:


data['call_type']=data['title'].apply(lambda x :x.split(':')[0])


# In[20]:


data.head()


# In[21]:


data['call_type'].nunique()


# In[22]:


data['call_type'].unique()


# In[23]:


call_types=data['call_type'].value_counts()


# In[24]:


from decimal import Decimal


# In[25]:


plt.figure(figsize=(15, 5))
ax = call_types.plot(kind='bar')
for p in ax.patches:
    ax.annotate(Decimal(str(p.get_height())), (p.get_x(), p.get_height()))
plt.xticks(rotation=0)
plt.savefig("1.png")


# ### 2.Extract Date and Time regarding calls

# In[26]:


data['timeStamp']=pd.to_datetime(data['timeStamp'])


# In[27]:


data['timeStamp'].head()


# In[28]:


data.info()


# In[29]:


import datetime as dt


# In[30]:


data['Year']=data['timeStamp'].dt.year


# In[31]:


data['Month']=data['timeStamp'].dt.month_name()


# In[32]:


data['Hour']=data['timeStamp'].dt.hour


# In[33]:


data['day']=data['timeStamp'].dt.day_name()


# In[34]:


data.head()


# ### 3.At what time of day/month I can expect which type of call

# In[35]:


data['emergency_type'] = data['title'].apply(lambda x:x.split(':')[1])


# In[36]:


data.head()


# In[37]:


calls_data = data.groupby(['Month', 'call_type'])['call_type'].count()


# In[38]:


calls_data_percentage = calls_data.groupby(level=0).apply(lambda x: round(100*x/float(x.sum())))


# In[39]:


calls_data_percentage.head()


# In[40]:


calls_data_percentage = calls_data_percentage.reindex(['EMS', 'Traffic', 'Fire'], level=1)


# In[41]:


month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']


# In[42]:


calls_data_percentage = calls_data_percentage.reindex(month_order, level=0)


# In[43]:


calls_data_percentage.head()


# In[44]:


font = {
    'size': 'x-large',
    'weight': 'bold'
}


# In[45]:


sns.set(rc={'figure.figsize':(12, 8)})
calls_data_percentage.unstack().plot(kind='bar')
plt.xlabel('Name of the Month', fontdict=font)
plt.ylabel('Percentage of Calls', fontdict=font)
plt.xticks(rotation=0)
plt.title('Calls/Month', fontdict=font)
plt.savefig('2monthly.png')


# In[46]:


hours_data = data.groupby(['Hour', 'call_type'])['call_type'].count()


# In[47]:


hours_data.head()


# In[48]:


hours_data_percentage = hours_data.groupby(level=0).apply(lambda x: round(100*x/float(x.sum())))


# In[49]:


hours_data_percentage.head()


# In[50]:


hours_data_percentage = hours_data_percentage.reindex(['EMS', 'Traffic', 'Fire'], level=1)


# In[51]:


hours_data_percentage.head()


# In[52]:


sns.set(rc={'figure.figsize':(18, 8)})
hours_data_percentage.unstack().plot(kind='bar')
plt.xlabel('Hour of the day', fontdict=font)
plt.ylabel('Percentage of Calls', fontdict=font)
plt.xticks(rotation=0)
plt.title('Calls/Hour', fontdict=font)
plt.savefig('2hourly.png')


# ### 4.Visualize percentage share of emergency type for each call

# In[53]:


data.head()


# In[54]:


ems_data = data[data['call_type'] == 'EMS']['emergency_type'].value_counts()[:5]


# In[55]:


fire_data = data[data['call_type'] == 'Fire']['emergency_type'].value_counts()[:5]


# In[56]:


traffic_data = data[data['call_type'] == 'Traffic']['emergency_type'].value_counts()[:5]


# In[57]:


plt.pie(ems_data, labels=ems_data.index, autopct="%.2f")


# In[58]:


plt.pie(fire_data, labels=fire_data.index, autopct="%.2f")


# In[59]:


plt.pie(traffic_data, labels=traffic_data.index, autopct="%.2f")

