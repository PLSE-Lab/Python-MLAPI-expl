#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[4]:


# The dataset looks like this
calls_data = pd.read_csv("../input/911.csv")
calls_data.head()


# In[5]:


calls_data['zip'].head()


# In[6]:


calls_data['twp'].head()


# In[7]:


calls_data['title'].nunique()


# In[8]:


# Here, we are trying to create a new column, called reasons where we can Identify the reason of calls
# this can be done by splitting the reason from title and adding to 'Reasons'
s = calls_data['title'].apply(lambda x: x.split(':'))
calls_data['Reasons'] = s.apply(lambda x: x[0])
calls_data['Reasons']
calls_data.head()


# In[9]:


# The count shows that most of the calls have been made for EMS(Emergency Medical services)
calls_data['Reasons'].value_counts()


# In[10]:


#Plotting the Reasons
sns.countplot(calls_data['Reasons'])


# In[11]:


# Here, the timeStamp column is a string, so in order to perfrm the analysis we need to convert it into datetime format
calls_data['timeStamp'] = pd.to_datetime(calls_data['timeStamp'])


# In[12]:


# Splitting the timestamp column into hour, month and dayofweek columns
calls_data['hour'] = calls_data['timeStamp'].apply(lambda x: x.hour)
calls_data['hour']

calls_data['month'] = calls_data['timeStamp'].apply(lambda x: x.month)
calls_data['month']

dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
s= calls_data['timeStamp'].apply(lambda x: x.dayofweek)
calls_data['day of week'] = s.map(dmap)


calls_data.head()


# In[13]:


# We can see that, most of the calls recorder are on friday.
bydays = calls_data.groupby('day of week')
bydays.count()


# In[14]:


# By plotting the calls reasons with the days of the week, we can observe that most of the calls are during friday.
sns.countplot(calls_data['day of week'],hue = calls_data['Reasons'])


# In[15]:


# By plotting the calls reasons with the month, we can observe that most of the calls are during the LAst three months
sns.countplot(calls_data['month'],hue = calls_data['Reasons'])


# In[16]:


# By plotting the calls reasons with the hour, we can observe that most of the calls are during evening hours
sns.countplot(calls_data['hour'],hue = calls_data['Reasons'])


# In[17]:


bymonth = calls_data.groupby('month').count()
bymonth


# In[18]:


bymonth['twp'].plot()


# In[19]:


# Linear map to show the by month data of calls, we can see that the plot is scattered.
sns.lmplot(x='month',y='twp',data=bymonth.reset_index())


# In[20]:


calls_data['date'] = calls_data['timeStamp'].apply(lambda x: x.date())
calls_data.head()


# In[21]:


bydate = calls_data.groupby('date').count()['twp'].plot()
plt.tight_layout()


# In[22]:


calls_data[calls_data['Reasons'] == 'EMS'].groupby('date').count()['twp'].plot()


# In[23]:


calls_data[calls_data['Reasons'] == 'Fire'].groupby('date').count()['twp'].plot()


# In[24]:


calls_data[calls_data['Reasons'] == 'Traffic'].groupby('date').count()['twp'].plot()


# In[25]:


dayhour = calls_data.groupby(by=['day of week','hour']).count()['Reasons'].unstack()
dayhour.head()


# In[26]:


sns.heatmap(dayhour)


# In[27]:


sns.clustermap(dayhour)


# In[28]:


daymonth = calls_data.groupby(by=['day of week','month']).count()['Reasons'].unstack()
daymonth.head()


# In[29]:


sns.heatmap(daymonth)


# In[30]:


sns.clustermap(daymonth)


# In[ ]:




