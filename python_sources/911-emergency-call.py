#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data=pd.read_csv('../input/911.csv')


# In[3]:


data.head()


# In[4]:


data.title.head()


# In[5]:


def title_sep(x):
    x=x.split(':')
    return x[0]


# In[6]:


data['type_of_call']=data['title'].apply(title_sep)


# In[7]:


data.head()


# In[8]:


data['type_of_call'].unique()


# In[9]:


call_type=data['type_of_call'].value_counts()


# In[10]:


import matplotlib.pyplot as plt


# In[11]:


import seaborn as sns


# In[12]:


sns.set()


# In[13]:


font={
    'size':20
}


# In[14]:


from decimal import Decimal


# In[15]:


plt.figure(figsize=(10,5))
px=call_type.plot(kind='bar')
for p in px.patches:
    px.annotate(Decimal(str(p.get_height())), (p.get_x(), p.get_height()))
plt.xticks(rotation=30)
plt.savefig('type_of_call.png')


# ### for extracting time and date

# In[16]:


data.info()


# In[17]:


data['timeStamp']=pd.to_datetime(data['timeStamp'], infer_datetime_format=True)


# In[18]:


data.info()


# In[19]:


import datetime as dt


# In[20]:


data['Year']=data['timeStamp'].dt.year


# In[21]:


data['Month']=data['timeStamp'].dt.month_name()


# In[22]:


data['Day']=data['timeStamp'].dt.day_name()


# In[23]:


data['Hour']=data['timeStamp'].dt.hour


# In[24]:


data.head()


# ### At what time of month I can expect which type of call.

# In[25]:


calls_month = data.groupby(['Month', 'type_of_call'])['type_of_call'].count()


# In[26]:


calls_month


# In[27]:


calls_month_percentage = calls_month.groupby(level=0).apply(lambda x:round(100*x/float(x.sum())))


# In[28]:


calls_month_percentage


# In[29]:


month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']


# In[30]:


calls_month_percentage = calls_month_percentage.reindex(month_order, level=0)


# In[31]:


calls_month_percentage


# In[32]:


calls_month_percentage = calls_month_percentage.reindex(['EMS','Traffic','Fire'], level=1)


# In[33]:


calls_month_percentage


# In[34]:


sns.set(rc={'figure.figsize':(12, 8)})
calls_month_percentage.unstack().plot(kind='bar')
plt.xlabel('Name of the Month', fontdict=font)
plt.ylabel('Percentage of Calls', fontdict=font)
plt.xticks(rotation=0)
plt.title('Calls/Month', fontdict=font)
plt.savefig('call_vs_month.png')


# In[ ]:





# ### At what time of hour I can expect which type of call.

# In[35]:


calls_hour = data.groupby(['Hour', 'type_of_call'])['type_of_call'].count()


# In[36]:


calls_hour


# In[37]:


calls_hour_percentage = calls_hour.groupby(level=0).apply(lambda x:round(100*x/float(x.sum())))


# In[38]:


calls_hour_percentage


# In[39]:


calls_hour_percentage = calls_hour_percentage.reindex(['EMS','Traffic','Fire'], level=1)


# In[40]:


calls_hour_percentage


# In[41]:


sns.set(rc={'figure.figsize':(12, 8)})
calls_hour_percentage.unstack().plot(kind='bar')
plt.xlabel('Hour of the day', fontdict=font)
plt.ylabel('Percentage of Calls', fontdict=font)
plt.xticks(rotation=0)
plt.title('Calls/Month', fontdict=font)
plt.savefig('call-vs-hour.png')


# ### Visualize the percentage share of emergency type for each call.

# In[42]:


data.head()


# In[43]:


def spliter(x):
    x=x.split(':')
    return x[1]


# In[44]:


data['emergency_call']=data['title'].apply(spliter)


# In[45]:


data.head()


# In[46]:


data['emergency_call'].unique()


# In[47]:


emergency_call=data['emergency_call'].value_counts()


# In[48]:


emergency_call


# In[49]:


emergency_call_percentage = emergency_call.groupby(level=0).apply(lambda x:round(100*x/float(emergency_call.sum())))


# AFTER 38 ALL VALUES PERCENTAGE IS 0

# In[50]:


emergency_call_percentage


# In[51]:


emergency_call_percentage=emergency_call_percentage.head(38)


# In[52]:


sns.set(rc={'figure.figsize':(20, 6)})
emergency_call_percentage.plot(kind='bar')
plt.xlabel('type_of_emergency_calls', fontdict=font)
plt.ylabel('Percentage of Calls', fontdict=font)
plt.xticks(rotation=90)
plt.title('Calls/types_of_emergency_calls', fontdict=font)
plt.savefig('call-vs-types_of_emergency.png')


# In[53]:


data.head()


# In[54]:


data1=data[data['type_of_call']=='EMS']


# In[55]:


data1.head()


# In[56]:


data1['emergency_call'].unique()


# In[57]:


ems_data=data1['emergency_call'].value_counts().head()


# In[58]:


data2=data[data['type_of_call']=='Traffic']


# In[59]:


Traffic_data=data2['emergency_call'].value_counts().head()


# In[60]:


data3=data[data['type_of_call']=='Fire']


# In[61]:


Fire_data=data3['emergency_call'].value_counts().head()


# In[62]:


plt.figure(figsize=(10,8))
plt.pie(ems_data.values,labels=ems_data.index,autopct="%.2f")
plt.savefig('EMS_top_5.png')


# In[63]:


plt.figure(figsize=(10,8))
plt.pie(Traffic_data.values,labels=Traffic_data.index,autopct="%.2f")
plt.savefig('Traffic_top_5.png')


# In[64]:


plt.figure(figsize=(10,8))
plt.pie(Fire_data.values,labels=Fire_data.index,autopct="%.2f")
plt.savefig('fire_top_5.png')


# In[ ]:




