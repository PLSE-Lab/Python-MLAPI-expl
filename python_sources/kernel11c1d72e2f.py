#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df=pd.read_csv("../input/911csv/911.csv")
df.head()


# In[ ]:


#top 5 zipcode
df['zip'].value_counts().head(5)


# In[ ]:


#Top 5 Township
df['twp'].value_counts().head(5)


# In[ ]:


# ** Take a look at the 'title' column, how many unique title codes are there? **
df['title'].nunique()


# ** In the titles column there are "Reasons/Departments" specified before the title code. These are EMS, Fire, and Traffic. Use .apply() with a custom lambda expression to create a new column called "Reason" that contains this string value.** 
# 
# **For example, if the title column value is EMS: BACK PAINS/INJURY , the Reason column value would be EMS. **

# In[ ]:


df['Reason']=df['title'].apply(lambda title: title.split(':')[0])
df['Reason']


# ** What is the most common Reason for a 911 call based off of this new column? **

# In[ ]:


df['Reason'].value_counts()


# ** Now use seaborn to create a countplot of 911 calls by Reason. **

# In[ ]:


sns.countplot(x='Reason',data=df,palette='viridis')


# ** Now let us begin to focus on time information. What is the data type of the objects in the timeStamp column? **
# 

# In[ ]:


type(df['timeStamp'].iloc()[0])


# In[ ]:


df['timeStamp'] = pd.to_datetime(df['timeStamp'])
df['timeStamp']


# In[ ]:


time = df['timeStamp'].iloc[0]
time.hour


# In[ ]:


df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)
df['Month'] = df['timeStamp'].apply(lambda time: time.month)
df['Day of Week'] = df['timeStamp'].apply(lambda time: time.dayofweek)


# In[ ]:


df.head()


# In[ ]:


dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}


# In[ ]:


df['Day of Week'] = df['Day of Week'].map(dmap)


# In[ ]:


df.head()


# ** Now use seaborn to create a countplot of the Day of Week column with the hue based off of the Reason column. **

# In[ ]:


sns.countplot(x='Day of Week',data=df,hue='Reason',palette='viridis')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# In[ ]:


sns.countplot(x='Month',data=df,hue='Reason',palette='viridis')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# For some missing month

# In[ ]:


byMonth=df.groupby('Month').count()
byMonth.head()


# In[ ]:


byMonth['lat'].plot()


# In[ ]:


sns.lmplot(x='Month',y='twp',data=byMonth.reset_index())


# **Create a new column called 'Date' that contains the date from the timeStamp column. You'll need to use apply along with the .date() method. ** 

# In[ ]:


df['Date']=df['timeStamp'].apply(lambda t: t.date())
df.head()


# ** Now groupby this Date column with the count() aggregate and create a plot of counts of 911 calls.**

# In[ ]:


df.groupby('Date').count()['twp'].plot()
plt.tight_layout()


# In[ ]:


df[df['Reason']=='Traffic'].groupby('Date').count()['twp'].plot()
plt.title('Traffic')
plt.tight_layout()


# In[ ]:


df[df['Reason']=='Fire'].groupby('Date').count()['twp'].plot()
plt.title('Fire')
plt.tight_layout()


# In[ ]:


df[df['Reason']=='EMS'].groupby('Date').count()['twp'].plot()
plt.title('EMS')
plt.tight_layout()


# Heatmap ,combine groupby unstack

# In[ ]:


dayHour = df.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()
dayHour.head()


# 

# In[ ]:


plt.figure(figsize=(12,6))
sns.heatmap(dayHour,cmap='viridis')


# In[ ]:


sns.clustermap(dayHour,cmap='viridis')


# In[ ]:




