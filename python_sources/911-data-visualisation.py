#!/usr/bin/env python
# coding: utf-8

# In[1]:


### 1) Q : max(Ems,Fire,Traffic)
### 2) Q : timestamp max(year),max(month),max(day),max(hours)
### 3) Q : Folium Library ((pie chart:ems,fire,traffic)100 data points)
### 4) Q : 


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns


# In[3]:


data = pd.read_csv("../input/911.csv")
data.head()


# In[4]:


data.shape


# In[5]:


data.columns


# In[6]:


data.info()


# In[7]:


data['timeStamp'] = pd.to_datetime(data['timeStamp'])
data.head(5)


# In[8]:


import re
def type_reason(x):
    x = str(x)
    if (re.search("EMS", x)):
        return "EMS"
    
    elif (re.search("Traffic", x)):
        return "Traffic"
    else:
        return "Fire"


# In[9]:


data["call_type"] = data["title"].apply(type_reason)
data.head()


# ### Which type of call is maximum

# In[10]:


datax_major_type=data["call_type"].value_counts()


# In[11]:


datax_major_type = pd.Series(datax_major_type)
datax_major_type


# In[12]:


font ={
    "size" :20
}
plt.figure(figsize=(8,6))
datax_major_type.plot(kind="bar")
plt.xticks(rotation=30)
plt.xlabel("Types of Calls",fontdict=font)
plt.ylabel("No. of Calls",fontdict=font)
plt.title("Types of Calls Vs No. of Calls",fontdict=font)
plt.savefig("Types-of-Calls-vs-No-of-Calls.png")


# In[13]:


data['Year'] = data['timeStamp'].dt.year
data.head()


# In[14]:


data['Month'] = data['timeStamp'].dt.month_name()
data.head()


# In[15]:


data["Day"] = data['timeStamp'].dt.day_name()
data.head()


# In[16]:


data["Hour"] = data['timeStamp'].dt.hour
data.head()


# In[17]:


def actual_type_call(x):
    x= x.split(':')
    return x[1]


# In[18]:


data["emergency_reason"] = data["title"].apply(actual_type_call)
data.head()


# ### Ems First 7 seven pie data plot

# In[19]:


ems_data = data.copy(deep=True)


# In[20]:


ems_data.query('call_type == "EMS"',inplace = True)
ems_data.head()


# In[21]:


count_ems = ems_data['emergency_reason'].value_counts()


# In[22]:


plt.figsize=(15,10)
plt.pie(count_ems.values[:7],labels=count_ems.index[:7],autopct="%.2f")
plt.savefig("Types-of-EMS-Calls-vs-No-of-Calls.png")


# In[23]:


### Fire Data first 7 pie plot


# In[24]:


fire_data = data.copy(deep = True)


# In[25]:


fire_data.query('call_type == "Fire"',inplace = True)
fire_data.head(2)


# In[26]:


fire_data['emergency_reason'].nunique()


# In[27]:


count_fire = fire_data['emergency_reason'].value_counts()
count_fire.head(2)


# In[28]:


plt.figsize=(15,10)
plt.pie(count_fire.values[:7],labels=count_fire.index[:7],autopct="%.2f")
plt.savefig("Types-of-fire-Calls-vs-No-of-Calls.png")


# In[29]:


### Traffic Pie Chart


# In[30]:


traffic_data = data.copy(deep=True)


# In[31]:


traffic_data.query('call_type == "Traffic"',inplace =True)
traffic_data.head(2)


# In[32]:


traffic_data['emergency_reason'].nunique()


# In[33]:


count_traffic = traffic_data['emergency_reason'].value_counts()
count_traffic.head(7)


# In[34]:


plt.figsize=(15,10)
plt.pie(count_traffic.values[:5],labels=count_traffic[:5].index,autopct="%.2f")
plt.savefig("Types-of-traffic-Calls-vs-No-of-Calls.png")


# ### Plotting using Time stamp

# ## Data Monthly

# In[35]:


calls_data = data.groupby(["Month","call_type"])["call_type"].count()
calls_data.head()


# In[36]:


calls_percentage = calls_data.groupby(level=0).apply(lambda x:round(100*x/x.sum()) )
calls_percentage.head()


# In[37]:


month_order =  ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']


# In[38]:


calls_percentage = calls_percentage.reindex(month_order, level=0)
calls_percentage.head()


# In[39]:


sns.set(rc={'figure.figsize':(12, 8)})
calls_percentage.unstack().plot(kind='bar')
plt.xlabel('Name of the Month', fontdict=font)
plt.ylabel('Percentage of Calls', fontdict=font)
plt.xticks(rotation=0)
plt.title('Calls Per Month', fontdict=font)
plt.savefig("Calls-per-Month.png")


# ## Per Day

# In[40]:


calls_data = data.groupby(["Day","call_type"])["call_type"].count()
calls_data.head()


# In[41]:


day_order= ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']


# In[42]:


calls_percentage = calls_data.groupby(level=0).apply(lambda x:round(100*x/x.sum()) )
calls_percentage.head()


# In[43]:


calls_percentage = calls_percentage.reindex(day_order, level=0)
calls_percentage.head()


# In[44]:


sns.set(rc={'figure.figsize':(12, 8)})
calls_percentage.unstack().plot(kind='bar')
plt.xlabel('Name of the Day', fontdict=font)
plt.ylabel('Percentage of Calls', fontdict=font)
plt.xticks(rotation=0)
plt.title('Calls Per Day', fontdict=font)
plt.savefig("Calls-per-Day.png")


# ## Data Hourly

# In[45]:


calls_data = data.groupby(["Hour","call_type"])["call_type"].count()
calls_data.head()


# In[46]:


calls_percentage = calls_data.groupby(level=0).apply(lambda x:round(100*x/x.sum()) )
calls_percentage.head()


# In[47]:


sns.set(rc={'figure.figsize':(12, 8)})
calls_percentage.unstack().plot(kind='bar')
plt.xlabel('Name of the Hour', fontdict=font)
plt.ylabel('Percentage of Calls', fontdict=font)
plt.xticks(rotation=0)
plt.title('Calls Per Hour', fontdict=font)
plt.savefig("Call-per-Hour.png")


# 

# In[ ]:




