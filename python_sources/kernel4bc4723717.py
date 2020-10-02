#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Second Try 


# In[ ]:





# In[ ]:


import pandas as pd


# In[ ]:


data = pd.read_csv("../input/911.csv")


# In[ ]:


data.head(4)


# In[ ]:


def sepa(x):
    val = x.split(":")
    return val[0]


# In[ ]:


data['type'] = data['title'].apply(sepa)


# In[ ]:


data.head(3)


# In[ ]:


data['timeStamp'] = pd.to_datetime(data['timeStamp'] , infer_datetime_format = True)


# In[ ]:


data['timeStamp'].head(2)


# In[ ]:


data.head(2)


# In[ ]:


import datetime as dt
data['year'] = data['timeStamp'].dt.year


# In[ ]:


data['month'] = data['timeStamp'].dt.month_name()


# In[ ]:


data['day'] = data['timeStamp'].dt.day_name()


# In[ ]:


def emergency_type_separator(x):
    x = x.split(':')
    x = x[1]
    return x


# In[ ]:


data['emergency_type'] = data['title'].apply(emergency_type_separator)


# In[ ]:


data.head(2)


# In[ ]:


def emergency_type_separator(x):
    x = x.split(':')
    x = x[1]
    return x


# In[ ]:


data['emergency_type'] = data['title'].apply(emergency_type_separator)


# In[ ]:


call_types = data['type'].value_counts()
call_types


# In[ ]:


calls_data = data.groupby(['month', 'type'])['type'].count()


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


import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


# In[ ]:


sns.set(rc={'figure.figsize':(12, 8)})
calls_data_percentage.unstack().plot(kind='bar')
plt.xlabel('Name of the Month', fontdict=font)
plt.ylabel('Percentage of Calls', fontdict=font)
plt.xticks(rotation=0)
plt.title('Calls/Month', fontdict=font)


# In[ ]:




