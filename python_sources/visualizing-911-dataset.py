#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np


# In[12]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[13]:


sns.set()


# In[14]:


data = pd.read_csv('../input/911.csv')


# In[15]:


data.head()


# In[16]:


data.shape


# In[17]:


data.info()


# In[18]:


column_names = list(data.columns)


# In[19]:


column_names


# ## Separating call type from the data

# In[20]:


data.title.head()


# In[21]:


def call_type_separator(x):
    x = x.split(':')
    return x[0]


# In[22]:


data['call_type'] = data['title'].apply(call_type_separator)


# In[23]:


data.head()


# In[24]:


data['call_type'].unique()


# In[25]:


data['call_type'].value_counts()


# ## Converting timestamp to pandas datetime

# In[26]:


data['timeStamp'] = pd.to_datetime(data['timeStamp'], infer_datetime_format=True)


# In[27]:


data['timeStamp'].head()


# ### Extracting year, month_name, day_name, hour of the day from timestamp column

# In[28]:


import datetime as dt


# In[29]:


data['year'] = data['timeStamp'].dt.year


# In[30]:


data['month'] = data['timeStamp'].dt.month_name()


# In[31]:


data['day'] = data['timeStamp'].dt.day_name()


# In[32]:


data['hour'] = data['timeStamp'].dt.hour


# ## Extracting call_detail

# In[33]:


data.head()


# In[34]:


def emergency_type_separator(x):
    x = x.split(':')
    x = x[1]
    return x


# In[35]:


data['emergency_type'] = data['title'].apply(emergency_type_separator)


# In[36]:


data.head()


# ## Visualizing data with different parameters

# In[37]:


data.head(2)


# In[38]:


call_types = data['call_type'].value_counts()
call_types


# In[39]:


from decimal import Decimal


# In[40]:


plt.figure(figsize=(15, 5))
ax = call_types.plot.bar()
for p in ax.patches:
    ax.annotate(Decimal(str(p.get_height())), (p.get_x(), p.get_height()))
plt.xticks(rotation=0)


# In[41]:


data.info()


# In[42]:


calls_data = data.groupby(['month', 'call_type'])['call_type'].count()


# In[43]:


calls_data.head()


# In[44]:


calls_data_percentage = calls_data.groupby(level=0).apply(lambda x: round(100*x/float(x.sum())))


# In[45]:


calls_data_percentage.head()


# In[46]:


font = {
    'size': 'x-large',
    'weight': 'bold'
}


# In[47]:


month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']


# In[ ]:





# In[48]:


calls_data_percentage = calls_data_percentage.reindex(month_order, level=0)


# In[49]:


calls_data_percentage = calls_data_percentage.reindex(['EMS', 'Traffic', 'Fire'], level=1)


# In[50]:


calls_data_percentage.head()


# In[51]:


sns.set(rc={'figure.figsize':(12, 8)})
calls_data_percentage.unstack().plot(kind='bar')
plt.xlabel('Name of the Month', fontdict=font)
plt.ylabel('Percentage of Calls', fontdict=font)
plt.xticks(rotation=0)
plt.title('Calls/Month', fontdict=font)


# ## Hourly Data

# In[52]:


hours_data = data.groupby(['hour', 'call_type'])['call_type'].count()


# In[53]:


hours_data.head()


# In[54]:


hours_data_percentage = hours_data.groupby(level=0).apply(lambda x: round(100*x/float(x.sum())))


# In[55]:


hours_data_percentage.head()


# In[56]:


hours_data_percentage = hours_data_percentage.reindex(['EMS', 'Traffic', 'Fire'], level=1)


# In[57]:


sns.set(rc={'figure.figsize':(18, 8)})
hours_data_percentage.unstack().plot(kind='bar')
plt.xlabel('Hour of the day', fontdict=font)
plt.ylabel('Percentage of Calls', fontdict=font)
plt.xticks(rotation=0)
plt.title('Calls/Hour', fontdict=font)

