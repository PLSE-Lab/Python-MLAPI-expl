#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np


# In[20]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[21]:


sns.set()


# In[22]:


data=pd.read_csv('../input/911.csv')


# In[23]:


data.head()


# In[24]:


data.shape


# In[25]:


data.info()


# In[26]:


data.columns


# ### 1.How many different type of calls?

# In[27]:


data['title'].head()


# In[28]:


data['call_type']=data['title'].apply(lambda x :x.split(':')[0])


# In[29]:


data.head()


# In[30]:


data['call_type'].nunique()


# In[31]:


data['call_type'].unique()


# In[32]:


call_types=data['call_type'].value_counts()


# In[33]:


from decimal import Decimal


# In[34]:


plt.figure(figsize=(15, 5))
ax = call_types.plot(kind='bar')
for p in ax.patches:
    ax.annotate(Decimal(str(p.get_height())), (p.get_x(), p.get_height()))
plt.xticks(rotation=0)
plt.savefig("1.png")


# ### 2.Extract Date and Time regarding calls

# In[35]:


data['timeStamp']=pd.to_datetime(data['timeStamp'])


# In[36]:


data['timeStamp'].head()


# In[37]:


data.info()


# In[38]:


import datetime as dt


# In[39]:


data['Year']=data['timeStamp'].dt.year


# In[40]:


data['Month']=data['timeStamp'].dt.month_name()


# In[41]:


data['Hour']=data['timeStamp'].dt.hour


# In[42]:


data['day']=data['timeStamp'].dt.day_name()


# In[43]:


data.head()


# ### 3.At what time of day/month I can expect which type of call

# In[44]:


data['emergency_type'] = data['title'].apply(lambda x:x.split(':')[1])


# In[45]:


data.head()


# In[46]:


calls_data = data.groupby(['Month', 'call_type'])['call_type'].count()


# In[47]:


calls_data_percentage = calls_data.groupby(level=0).apply(lambda x: round(100*x/float(x.sum())))


# In[48]:


calls_data_percentage.head()


# In[49]:


calls_data_percentage = calls_data_percentage.reindex(['EMS', 'Traffic', 'Fire'], level=1)


# In[50]:


month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']


# In[51]:


calls_data_percentage = calls_data_percentage.reindex(month_order, level=0)


# In[52]:


calls_data_percentage.head()


# In[53]:


font = {
    'size': 'x-large',
    'weight': 'bold'
}


# In[54]:


sns.set(rc={'figure.figsize':(12, 8)})
calls_data_percentage.unstack().plot(kind='bar')
plt.xlabel('Name of the Month', fontdict=font)
plt.ylabel('Percentage of Calls', fontdict=font)
plt.xticks(rotation=0)
plt.title('Calls/Month', fontdict=font)
plt.savefig('2monthly.png')


# In[55]:


hours_data = data.groupby(['Hour', 'call_type'])['call_type'].count()


# In[56]:


hours_data.head()


# In[57]:


hours_data_percentage = hours_data.groupby(level=0).apply(lambda x: round(100*x/float(x.sum())))


# In[58]:


hours_data_percentage.head()


# In[59]:


hours_data_percentage = hours_data_percentage.reindex(['EMS', 'Traffic', 'Fire'], level=1)


# In[60]:


sns.set(rc={'figure.figsize':(18, 8)})
hours_data_percentage.unstack().plot(kind='bar')
plt.xlabel('Hour of the day', fontdict=font)
plt.ylabel('Percentage of Calls', fontdict=font)
plt.xticks(rotation=0)
plt.title('Calls/Hour', fontdict=font)
plt.savefig('2hourly.png')


# ### 4.Visualize percentage share of emergency type for each call

# In[61]:


ems_data=data[data['call_type']=='EMS']['emergency_type'].value_counts()[:5]


# In[62]:


fire_data=data[data['call_type']=='Fire']['emergency_type'].value_counts()[:5]


# In[63]:


traffic_data=data[data['call_type']=='Traffic']['emergency_type'].value_counts()[:5]


# In[64]:


plt.pie(ems_data,labels=ems_data.index,autopct="%.2f")
plt.savefig("3.png")


# In[65]:


plt.pie(fire_data,labels=fire_data.index,autopct="%.2f")
plt.savefig("4.png")


# In[66]:


plt.pie(traffic_data,labels=traffic_data.index,autopct="%.2f")
plt.savefig("5.png")


# In[ ]:




