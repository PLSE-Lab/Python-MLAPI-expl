#!/usr/bin/env python
# coding: utf-8

# ### import library and read data set

# In[ ]:


import pandas as pd


# In[ ]:


data=pd.read_csv('911.csv')


# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


data.columns


# ###  How many different types of calls.

# In[ ]:


data.title


# In[ ]:


def call_type(x):
    x=x.split(':')
    return x[0]


# In[ ]:


data['call_type']=data['title'].apply(call_type)


# In[ ]:


data['call_type'].unique()


# In[ ]:


call_types=data['call_type'].value_counts()
call_types


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


sns.set()


# In[ ]:


plt.figure(figsize=(12,6))
call_types.plot(kind='bar')
plt.xticks(rotation=30)
font={
    'size':20,
     'weight': 'bold'
}
plt.title("Total_no_of_different_calltypes",fontdict=font)
plt.xlabel("Call_types",fontdict=font)
plt.ylabel("no_of_calls",fontdict=font)


# ### Extract the date and time regarding the calls

# In[ ]:


data.timeStamp.describe()


# In[ ]:


data['timeStamp']=pd.to_datetime(data['timeStamp'],infer_datetime_format=True)


# In[ ]:


data['timeStamp'].head()


# In[ ]:


import datetime as dt


# In[ ]:


data['Year']=data['timeStamp'].dt.year


# In[ ]:


data['Month']=data['timeStamp'].dt.month_name()


# In[ ]:


data['Day']=data['timeStamp'].dt.day_name()


# In[ ]:


data['Hour']=data['timeStamp'].dt.hour


# In[ ]:


data.columns


# In[ ]:


def emergency_type(x):
    x=x.split(':')
    return x[1]


# In[ ]:


data['emergency_type']=data['title'].apply(emergency_type)


# In[ ]:


data['emergency_type'].nunique()


# In[ ]:


data['emergency_type'].value_counts()


# ### At what time of day/month/hour I can expect which type of call and Visualize the percentage share of emergency type for each call.

# In[ ]:


call_data=data.groupby(['Month','call_type'])['call_type'].count()


# In[ ]:


call_data


# In[ ]:


def percentage(x):
    p=round((100*x)/float(x.sum()))
    return p


# In[ ]:


call_data_percentage=call_data.groupby('Month').apply(percentage)


# In[ ]:


call_data_percentage


# In[ ]:


month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']


# In[ ]:


call_data_percentage = call_data_percentage.reindex(month_order,level=0)


# In[ ]:


call_data_percentage


# In[ ]:


call_data_percentage=call_data_percentage.reindex(['EMS','Traffic','Fire'],level=1)


# In[ ]:


call_data_percentage.head()


# In[ ]:


font={
    'size': 'x-large',
    'weight': 'bold'
}


# In[ ]:


plt.figure(figsize=(20, 5))
call_data_percentage.unstack().plot(kind='bar')
plt.xlabel('Name of the Month', fontdict=font)
plt.ylabel('Percentage of Calls', fontdict=font)
plt.xticks(rotation=0)
plt.title('Calls/Month', fontdict=font)


# In[ ]:


call_data=data.groupby(['Day','call_type'])['call_type'].count()


# In[ ]:


call_data


# In[ ]:


def percentage(x):
    p=round(x*100/float(x.sum()))
    return p


# In[ ]:


call_data_percentage=call_data.groupby('Day').apply(percentage)


# In[ ]:


call_data_percentage


# In[ ]:


call_data_percentage=call_data_percentage.reindex(['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'],level=0)


# In[ ]:


call_data_percentage=call_data_percentage.reindex(['EMS','Traffic','Fire'],level=1)
call_data_percentage


# In[ ]:


plt.figure(figsize=(20,5))
call_data_percentage.unstack().plot(kind='bar')
plt.xlabel('Name of the Day', fontdict=font)
plt.ylabel('Percentage of Calls', fontdict=font)
plt.xticks(rotation=0)
plt.title('Calls/Day', fontdict=font)


# In[ ]:


call_data=data.groupby(['Hour','call_type'])['call_type'].count()


# In[ ]:


call_data


# In[ ]:


def percentage(x):
    p=round(x*100/float(x.sum()))
    return p


# In[ ]:


call_data_percentage=call_data.groupby('Hour').apply(percentage)


# In[ ]:


call_data_percentage


# In[ ]:


call_data_percentage=call_data_percentage.reindex(['EMS','Traffic','Fire'],level=1)
call_data_percentage


# In[ ]:


plt.figure(figsize=(20,5))
call_data_percentage.unstack().plot(kind='bar')
plt.xlabel('Name of the hour', fontdict=font)
plt.ylabel('Percentage of Calls', fontdict=font)
plt.xticks(rotation=0)
plt.title('Calls/Hour', fontdict=font)


# In[ ]:




