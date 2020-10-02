#!/usr/bin/env python
# coding: utf-8

# This is data of emergency calls from Emergency Calls for Montgomery County, PA.We will try to explore the dataset.This kernel is a work in process.I will be updating the kernel in the coming days.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# **Importing Python Modules**

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
#plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')


# **Importing the data**

# In[ ]:


df=pd.read_csv('../input/montcoalert/911.csv')
df.head()


# So in the dataset we have the place of the call,reason,address etc

# **Summary of Dataset**

# In[ ]:


print('Rows     :',df.shape[0])
print('Columns  :',df.shape[1])
print('\nFeatures :\n     :',df.columns.tolist())
print('\nMissing values    :',df.isnull().values.sum())
print('\nUnique values :  \n',df.nunique())


# **Attributes**

# In[ ]:


df['twp'].values


# In[ ]:


df.index


# In[ ]:


df['lat'].dtype


# **From where the calls come most?**

# In[ ]:


df['zip'].value_counts().head(5).plot.bar();
plt.xlabel('Zip Code')
plt.ylabel('Count')
plt.show()


# Maximum Call comes from Zip Code 19401 which is a place called as Norristown in Pennsylvania, United States.

# **Which are top townships for calls?**

# In[ ]:


df['twp'].value_counts().head(5).plot.bar();
plt.xlabel('Township')
plt.ylabel('Count')
plt.show()


# Lower Merion township has the highest number of calls.

# **How many Unique title?******

# In[ ]:


len(df['title'].unique())


# Or we can use the command

# In[ ]:


df['title'].nunique()


# **Creating a columns with reason:**
# The title column have the general reason for the call with the more detailed reason for the the call.There are three basic category for the call like EMS,Fire and Traffic

# In[ ]:


x=df['title'].iloc[0]


# In[ ]:


x.split(':')[0]


# In[ ]:


df['Reason']=df['title'].apply(lambda x:x.split(':')[0])
df['Reason'].unique()


# With Above Transformations we have managed to create a columns with title reason having the values EMS,Fire and Traffic.

# **What is reason for most calls?**

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
df['Reason'].value_counts().plot.pie(explode=[0,0.1,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Reason for Call')
ax[0].set_ylabel('Count')
sns.countplot('Reason',data=df,ax=ax[1],order=df['Reason'].value_counts().index)
ax[1].set_title('Count of Reason')
plt.show()


# 49% call are for medical emergency followed by Traffic and Fire.

# **Working with Time Data**

# In[ ]:


type(df['timeStamp'].iloc[0])


# Data about time is of time string.We need to convert it into Datetime Format.

# In[ ]:


df['timeStamp']=pd.to_datetime(df['timeStamp'])


# In[ ]:


type(df['timeStamp'].iloc[0])


# In[ ]:


time=df['timeStamp'].iloc[0]
time.hour


# In[ ]:


time.year


# In[ ]:


time.month


# In[ ]:


time.dayofweek


# In[ ]:


df['Hour']=df['timeStamp'].apply(lambda x:x.hour)
df['Month']=df['timeStamp'].apply(lambda x:x.month)
df['DayOfWeek']=df['timeStamp'].apply(lambda x:x.dayofweek)


# In[ ]:


dmap={0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}


# **Calls Per Month**

# In[ ]:


byMonth=df.groupby('Month').count()
byMonth['lat'].plot();


# In[ ]:


sns.lmplot(x='Month',y='twp',data=byMonth.reset_index());


# In[ ]:


mmap={1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}


# In[ ]:


df['Month']=df['Month'].map(mmap)


# In[ ]:


df['DayOfWeek']=df['DayOfWeek'].map(dmap)


# In[ ]:


df.head()


# **Call during the Week?**

# In[ ]:


sns.set_style('darkgrid')
f,ax=plt.subplots(1,2,figsize=(18,8))
k1=sns.countplot(x='DayOfWeek',data=df,ax=ax[0],palette='viridis')
k2=sns.countplot(x='DayOfWeek',data=df,hue='Reason',ax=ax[1],palette='viridis')


# More Emergency calls happen on Friday.EMS call are more.

# **Call during the month?**

# In[ ]:


sns.set_style('darkgrid')
f,ax=plt.subplots(1,2,figsize=(18,8))
k1=sns.countplot(x='Month',data=df,ax=ax[0],palette='viridis')
k2=sns.countplot(x='Month',data=df,hue='Reason',ax=ax[1],palette='viridis')


# We have more Emergency calls in the Months of Jan,Mar and Oct.

# **Creating a Date Column**

# In[ ]:


df['Date']=df['timeStamp'].apply(lambda x:x.date())


# In[ ]:


#df.head()


# In[ ]:


plt.figure(figsize=(20,10))
df.groupby('Date').count()['lat'].plot();
plt.tight_layout()


# **PLotting per day Plot based on Reason******

# In[ ]:


plt.figure(figsize=(20,10))
df[df['Reason']=='Traffic'].groupby('Date').count()['lat'].plot();
plt.title('Calls Per Day for Traffic Issues');
plt.tight_layout()


# In[ ]:


plt.figure(figsize=(20,10))
df[df['Reason']=='Fire'].groupby('Date').count()['lat'].plot();
plt.title('Calls Per Day for Fire Issues');
plt.tight_layout()


# In[ ]:


plt.figure(figsize=(20,10))
df[df['Reason']=='EMS'].groupby('Date').count()['lat'].plot();
plt.title('Calls Per Day for EMS Issues');
plt.tight_layout()


# **Creating Heatmaps based on day**

# In[ ]:


dayHour=df.groupby(by=['DayOfWeek','Hour']).count()['Reason'].unstack()


# In[ ]:


plt.figure(figsize=(12,6))
sns.heatmap(dayHour,cmap='viridis');


# **Cluster Map based on day**

# In[ ]:


plt.figure(figsize=(12,6));
sns.clustermap(dayHour,cmap='viridis');


# We can see from heatmap we can see that we have more calls on Friday and Wenesday between 15-17 Hours.More calls come in the Evening.Very Less calls during the Night time.We have very less 911 calls on weekends.

# **Heat Map Based on Month**

# In[ ]:


dayMonth=df.groupby(by=['DayOfWeek','Month']).count()['Reason'].unstack()


# In[ ]:


plt.figure(figsize=(12,6))
sns.heatmap(dayMonth,cmap='viridis');


# **Cluster Map Based on Month**

# In[ ]:


plt.figure(figsize=(12,6));
sns.clustermap(dayMonth,cmap='coolwarm');


# We have the highest calls in month of March on Friday.
