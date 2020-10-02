#!/usr/bin/env python
# coding: utf-8

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


# > * lat : String variable, Latitude
# > * lng: String variable, Longitude
# > * desc: String variable, Description of the Emergency Call
# > * zip: String variable, Zipcode
# > * title: String variable, Title
# > * timeStamp: String variable, YYYY-MM-DD HH:MM:SS
# > * twp: String variable, Township
# > * addr: String variable, Address
# > * e: String variable, Dummy variable (always 1)

# In[ ]:


df=pd.read_csv('../input/911csv/911.csv')


# In[ ]:


df.dtypes


# In[ ]:


import pandas as pd
df = pd.read_csv("../input/911csv/911.csv")


# In[ ]:


df.head()


# In[ ]:


df.zip.value_counts().head()


# In[ ]:


df.twp.value_counts().head()


#  ## creating new variable

# In[ ]:


df['title'].nunique


# In[ ]:


df['Reason']=df['title'].apply(lambda title:title.split(':')[0]) #I'm creating a new variable from the titles.


# In[ ]:


df.head()


# In[ ]:


df['Reason'].value_counts() #distribution of titles


# In[ ]:


import seaborn as sns
sns.countplot(x="Reason", data=df, palette="viridis");


# In[ ]:


df.timeStamp


# In[ ]:


df['timeStamp']=pd.to_datetime(df['timeStamp']) #convert a timeStamp variable to a datetime


# In[ ]:


time = df['timeStamp'].iloc[0]
time.hour


# In[ ]:


time=df['timeStamp'].iloc[0]
df['Hour']=df['timeStamp'].apply(lambda time:time.hour)
df['Month']=df['timeStamp'].apply(lambda time:time.month)
df['Day']=df['timeStamp'].apply(lambda time:time.dayofweek)

#we create new features from timestamp variable. separating the variable in day,month and hour 


# In[ ]:


df.head()


# In[ ]:


dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}


# In[ ]:


df['Day']=df['Day'].map(dmap)


# In[ ]:


df.head()


# In[ ]:


sns.countplot(x='Day',data=df,hue='Reason',palette='viridis'); #reason on daily basis.


# In[ ]:


sns.countplot(x='Hour',data=df,hue='Reason',palette='viridis'); #reason on hourly basis.


# In[ ]:


sns.countplot(x='Month',data=df,hue='Reason',palette='viridis'); # where is the months of sep,oct,nov?. these observations are not in the dataset


# In[ ]:


Monthnew=df.groupby('Month').count()
Monthnew.head(10)


# In[ ]:


Monthnew['twp'].plot(); # count of calls per month.


# In[ ]:


sns.lmplot(x='Month',y='twp',data=Monthnew.reset_index());
#Now see if we can use seaborn's lmplot() to create a linear fit on the number of calls per month and we reset the index to a column.


# In[ ]:


df['Date']=df['timeStamp'].apply(lambda p:p.date()) #new feature at timestamp column


# In[ ]:


df.head()


# ## visualization

# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


df.groupby('Date').count()['twp'].plot()
plt.tight_layout();
#count of accidents(twp=township) by date.


# In[ ]:


df[df['Reason']=='Traffic'].groupby('Date').count()['twp'].plot()
plt.title('Traffic')
plt.tight_layout()

#If the call reason is traffic accidents.


# In[ ]:


df[df['Reason']=='Fire'].groupby('Date').count()['twp'].plot()
plt.title('Fire')
plt.tight_layout()

#If the call reason is fire accidents.


# In[ ]:


df[df['Reason']=='EMS'].groupby('Date').count()['twp'].plot()
plt.title('EMS')
plt.tight_layout()
#If the call reason is ems(Emergency Medical Services)


# In[ ]:


dayHour = df.groupby(['Day','Hour']).count().Reason.unstack()
dayHour.head(10)
#count of calls per day and hour


# In[ ]:


f,ax = plt.subplots(figsize=(10, 10)) #shape
sns.heatmap(dayHour,cmap='viridis', fmt="d", linewidths = .5);


# In[ ]:


sns.clustermap(dayHour,cmap='viridis', annot=True, fmt="d", linewidths = .5,figsize=(15, 15));


# In[ ]:


dayMonth = df.groupby(by=['Day','Month']).count()['Reason'].unstack()
dayMonth.head(10)


# In[ ]:


f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(dayMonth,cmap='viridis', annot=True, fmt="d",linewidths = .5);


# In[ ]:


sns.clustermap(dayHour,cmap='viridis', annot=True, fmt="d", linewidths = .5,figsize=(15, 15));


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




