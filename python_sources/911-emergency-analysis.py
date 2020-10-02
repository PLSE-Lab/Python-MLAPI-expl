#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/911.csv')


# In[ ]:


df.columns


# In[ ]:


df.isnull().sum()


# In[ ]:


columns = {'lat':'Latitude','lng':'Longitude','desc':'Description of Emergency','zip':'ZIP Code','title':'Title'
,'timeStamp':'Datetime','twp':'Town','addr':'Adress'}


# In[ ]:


df = df.rename(columns,axis=1)


# In[ ]:


df.info()


# In[ ]:


#top 5 zip code
zipcode = df['ZIP Code'].value_counts()


# In[ ]:


zipcode.head()


# In[ ]:


#top 5 Township
df['Town'].value_counts().head()


# In[ ]:


# unique values of title columns
len(df['Title'].unique())


# In[ ]:


df['Title'].nunique()


# In[ ]:


#for getting the particular value like EMS,Fire 


# In[ ]:


x=df['Title'].iloc[0]
x


# In[ ]:


x.split(':')[0]


# In[ ]:


df['Reason'] = df['Title'].apply(lambda Title : Title.split(':')[0])


# In[ ]:


df['Reason'].value_counts()


# In[ ]:


sns.countplot(x='Reason',data = df)


# In[ ]:


df.info()


# In[ ]:


# type of Data and time of the call
type(df['Datetime'].iloc[0])


# In[ ]:


# date of column
df['Datetime']=pd.to_datetime(df['Datetime'])


# In[ ]:


df.shape


# In[ ]:


time = df['Datetime'].iloc[0]


# In[ ]:


time.hour


# In[ ]:


time.month


# In[ ]:


time.day


# In[ ]:


time.second


# In[ ]:


time.minute


# In[ ]:


df['Hour']= df['Datetime'].apply(lambda Datetime : Datetime.hour)


# In[ ]:


df['Hour'].head()


# In[ ]:


type(df['Datetime'].iloc[0].hour)


# In[ ]:


df['Month']=df['Datetime'].apply(lambda Datetime : Datetime.month)
df['DatOFWeek']=df['Datetime'].apply(lambda Datetime : Datetime.dayofweek)


# In[ ]:


dmap ={0:'Sun',1:'Mon',2:'Tue',3:'Wed',4:'Thu',5:'Fri',6:'Sat'}
df['DatOFWeek'] = df['DatOFWeek'].map(dmap)


# In[ ]:


df['DatOFWeek'].head()


# In[ ]:


sns.countplot(x='DatOFWeek', data = df,hue='Reason', palette='viridis').grid()
plt.legend(bbox_to_anchor =(1.05,1),loc=2,borderaxespad= 0.)


# In[ ]:


sns.countplot(x='Month', data= df,hue='Reason', palette='viridis').grid()
plt.legend(bbox_to_anchor = (1.05,1),loc=2)


# In[ ]:


ass=df['Month'].value_counts()
ass


# In[ ]:


byMonth = df.groupby('Month').count()


# In[ ]:


sns.lmplot(x='Month',y='Town', data= byMonth.reset_index())


# In[ ]:


t = df['Datetime'].iloc[0]


# In[ ]:


t


# In[ ]:


df['Date'] = df['Datetime'].apply(lambda t:t.date())
df.head()


# In[ ]:


df.groupby('Date').count()['Latitude'].plot()
plt.tight_layout()


# In[ ]:


df[df['Reason']=='Traffic'].groupby('Date')['Latitude'].count().plot()


# In[ ]:


df[df['Reason']=='Fire'].groupby('Date')['Latitude'].count().plot().grid()


# In[ ]:


df[df['Reason']=='EMS'].groupby('Date')['Latitude'].count().plot().grid()


# In[ ]:


dayHour=df.groupby(['DatOFWeek','Hour']).count()['Reason'].unstack()


# In[ ]:


plt.figure(figsize=(12,6))
sns.heatmap(dayHour, cmap='viridis')


# In[ ]:


sns.clustermap(dayHour,cmap='coolwarm')


# In[ ]:


dayMonth = df.groupby(['DatOFWeek', 'Month']).count()['Reason'].unstack()


# In[ ]:


dayMonth


# In[ ]:


sns.heatmap(dayMonth,cmap = 'coolwarm')


# In[ ]:


sns.clustermap(dayMonth, cmap = 'coolwarm')


# In[ ]:


df['e'].value_counts()


# In[ ]:


df.shape


# In[ ]:




