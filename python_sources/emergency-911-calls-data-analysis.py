#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis (EDA)- Python

# ***Emergency 911 Calls - Montgomery County, PA***

# 
# 911 -Emergency call dataset -
# 
# The data- contains the following fields:-- -
# 
#     lat : String variable, Latitude
#     lng: String variable, Longitude
#     desc: String variable, Description of the Emergency Call
#     zip: String variable, Zipcode
#     title: String variable, Title
#     timeStamp: String variable, YYYY-MM-DD HH:MM:SS
#     twp: String variable, Township
#     addr: String variable, Address
#     e: String variable, Dummy variable (always 1)
# 
# 

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/montcoalert/911.csv')


# In[ ]:


df.info()


# In[ ]:


df.head(3)


# ** Top 5 zipcodes for 911 calls **

# In[ ]:


df['zip'].value_counts().head(5)


# In[ ]:


df['twp'].value_counts().head(5)


# In[ ]:


df['title'].nunique()


# In[ ]:


df['Reason'] = df['title'].apply(lambda title: title.split(':')[0])


# ** Most common Reason for a 911 call based off of this new column **

# In[ ]:


df['Reason'].value_counts()


# ** Countplot of 911 calls by Reason. **

# In[ ]:


sns.countplot(x='Reason',data=df)


# ** Converting the timestamp column from strings to DateTime objects using [pd.to_datetime](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.to_datetime.html)  **

# In[ ]:


df['timeStamp'] = pd.to_datetime(df['timeStamp'])


# In[ ]:


df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)
df['Month'] = df['timeStamp'].apply(lambda time: time.month)
df['Day of Week'] = df['timeStamp'].apply(lambda time: time.dayofweek)


# In[ ]:


dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}


# In[ ]:


df['Day of Week'] = df['Day of Week'].map(dmap)


# ** Countplot of the Day of Week column with the hue based off of the Reason column. **

# In[ ]:


sns.countplot(x='Day of Week',data=df,hue='Reason')

# To relocate the legend
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# ** Countplot of the Month column with the hue based off of the Reason column. **

# In[ ]:


sns.countplot(x='Month',data=df,hue='Reason')

# To relocate the legend
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# In[ ]:


byMonth = df.groupby('Month').count()
byMonth.head()


# ** Count of calls per month. **

# In[ ]:


byMonth['twp'].plot()


# ** Creating a linear fit on the number of calls per month **

# In[ ]:


sns.lmplot(x='Month',y='twp',data=byMonth.reset_index())


# ** Plot representing a Fire as the Reason for the 911 call by month**

# In[ ]:


df[df['Reason']=='Fire'].groupby('Month').count()['twp'].plot()
plt.title('Fire')
plt.tight_layout()


# ** Plot representing a EMS as the Reason for the 911 call by month**

# In[ ]:


df[df['Reason']=='EMS'].groupby('Month').count()['twp'].plot()
plt.title('EMS')
plt.tight_layout()


# ** Plot representing a Traffic as the Reason for the 911 call by month**

# In[ ]:


df[df['Reason']=='Traffic'].groupby('Month').count()['twp'].plot()
plt.title('Traffic')
plt.tight_layout()


# In[ ]:


dayHour = df.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()
dayHour.head()


# **Heatmap Day of week by Hour**

# In[ ]:


plt.figure(figsize=(12,6))
sns.heatmap(dayHour)


# **Clustermap Day of week by Hour**

# In[ ]:


sns.clustermap(dayHour)


# In[ ]:


dayMonth = df.groupby(by=['Day of Week','Month']).count()['Reason'].unstack()
dayMonth.head()


# **Heatmap Month by Hour**

# In[ ]:


plt.figure(figsize=(12,6))
sns.heatmap(dayMonth)


# **Clustermap Month by Hour**

# In[ ]:


sns.clustermap(dayMonth)


# In[ ]:




