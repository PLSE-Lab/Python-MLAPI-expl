#!/usr/bin/env python
# coding: utf-8

# To everyone looking at this Kernel, 
# I am new to Python and this is my first project on Kaggle. I have choosen this dataset in order to make sure that I understand the conepts that I have learnt recently. 

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import os
print(os.listdir("../input"))


# In[2]:


# Reading the given data
df = pd.read_csv('../input/911.csv')
#displaying the head of the data
df.head()


# In[3]:


#checking the data info
df.info()


# Checking the top 5 Townships(twp) that call 911 

# In[4]:


df['twp'].value_counts().head(5)


# Checking the top 5 zip codes that call 911

# In[5]:


df['zip'].value_counts().head(5)


# Top 5 reasons['title'] for calling 911 

# In[6]:


df['title'].value_counts().head(5)


# Number of unique reasons['title'] for calling 911

# In[7]:


#number of unique reasons['title'] for calling 911
df['title'].nunique()


# Each title has a particular reson for which 911 was called and they are:- EMS:, Fire: and traffic:. Lets create a specific column for these resons

# In[8]:


df['reasons']=df['title'].apply(lambda title: title.split(':')[0])


# Now see which reason leads to  the most 911 calls

# In[9]:


df['reasons'].value_counts()


# Lets compare the above results as countplot using seaborn

# In[10]:


sns.countplot(x=df['reasons'],data=df)
plt.show()


# Now lets Focus on the time information

# In[11]:


#checking the data type of the column timeStamps
type(df['timeStamp'].iloc[0])


# In[12]:


#in order to further manipulate the data lets convert the timeStamp form str to DataTime objects
df['timeStamp']=pd.to_datetime(df['timeStamp'])


# Now lets create new columns in order to know the exact hour, month and day of the week of the 911 calls using timeStamp column

# In[13]:


df['Hour']=df['timeStamp'].apply(lambda time: time.hour)
df['Month']=df['timeStamp'].apply(lambda time: time.month)
df['Day of Week']=df['timeStamp'].apply(lambda time: time.dayofweek)


# In[14]:


df['Day of Week'].unique()


# We can see how day of week is in integer data type, 0-6 but we can have string names for the day of the week by creating a dictionary and assigning each value to it

# In[15]:


# creating the dictionary
dmap={0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',4:'Friday',5:'Saturday',6:'Sunday'}
# Changing the day of the week column to have proper strings
df['Day of Week']=df['Day of Week'].map(dmap)


# In[16]:


#making sure that the changes took place
df['Day of Week'].unique()


# Lets use seaborn to create a countplot for the Day of Week based on reasons for the 911 calls

# In[17]:


sns.countplot(x=df['Day of Week'], data=df,hue='reasons')
plt.legend(bbox_to_anchor=(1.05,1),loc=(2),borderaxespad=0.)
plt.show()


# Lets do the same with months instead of day of the week

# In[18]:


sns.countplot(x=df['Month'], data=df,hue='reasons')
plt.legend(bbox_to_anchor=(1.05,1),loc=(2),borderaxespad=0.)
plt.show()


# Lets create a new date column from timeStamp in order to get better perspective of when the 911 calls were made

# In[19]:


df['Date']=df['timeStamp'].apply(lambda t: t.date())


# Now that we have a column for date we use this data in order to represent how different reasons affected the 911 calls on different dates

# 911 calls according to date

# In[20]:


df.groupby('Date').count()['twp'].plot()
plt.tight_layout()


# How traffic affected 911 calls on particular dates

# In[21]:


df[df['reasons']=='Traffic'].groupby('Date').count()['twp'].plot()
plt.title('Traffic')
plt.tight_layout()


# How EMS affected 911 calls on particular dates

# In[22]:


df[df['reasons']=='EMS'].groupby('Date').count()['twp'].plot()
plt.title('EMS')
plt.tight_layout()


# How Fire affected 911 calls on particular dates

# In[23]:


df[df['reasons']=='Fire'].groupby('Date').count()['twp'].plot()
plt.title('Fire')
plt.tight_layout()


# Now lets make Heat Map in order to better represent the given data, but in order to do so we will have to manipulate the data in order to have the index as days of the week and the columns to be the hours. 

# In[24]:


dayHour=df.groupby(by=['Day of Week','Hour']).count()['reasons'].unstack()
dayHour.head()


# Lets see the heatmap for dayHour

# In[25]:


sns.heatmap(dayHour,cmap='YlGnBu').set_title('HeatMap for dayHour')
plt.show()


# Lets see a clustermap for dayHour

# In[26]:


sns.clustermap(dayHour,cmap='mako').fig.suptitle('ClusterMap for dayHour')
plt.show()


# Lets create heatmaps to have months as columns and index as days of the week

# In[27]:


dayMonth=df.groupby(by=['Day of Week','Month']).count()['reasons'].unstack()
dayMonth.head()


# Lets see the heatmap for dayMonth

# In[28]:


sns.heatmap(dayMonth,cmap='mako').set_title('HeatMap for dayMonth')
plt.show()


# Lets see a clustermap for dayMonth

# In[30]:


sns.clustermap(dayMonth,cmap='mako').fig.suptitle('ClusterMap for dayMonth')
plt.show()

