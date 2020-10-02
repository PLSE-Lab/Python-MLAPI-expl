#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv("../input/911.csv")
df.head()


# In[ ]:


df.info()


# In[ ]:


# We want top five zip codes for 911 cals
df['zip'].value_counts().head()


# In[ ]:


# No of unique zip codes
df['zip'].nunique()


# In[ ]:


#we want top 5 townships for 911 calls
df['twp'].value_counts().head()


# In[ ]:


# No of unique townships 
df['twp'].nunique()


# In[ ]:


#how many unique title are there?
df['title'].nunique()


# In[ ]:


#Create new type and reason column containing reason of title.
df['type'] = df['title'].apply(lambda title : title.split(':')[0])
df['type'].head()


# In[ ]:


df['reason'] = df['title'].apply(lambda title : title.split(':')[1])
df['reason'].head()


# In[ ]:


#Most common type of 911 calls
df['type'].value_counts().head(1)


# In[ ]:


#Countplot of 911 calls by their type
plt.figure(figsize=(10,6))
sns.countplot(x='type',data=df, palette='viridis')
plt.title('Type of Calls')


# Most are Emergency services 911 calls.

# In[ ]:


#Most common reason for 911 calls. We have to remove hyphen or its double counting few reasons.
df['reason'] = df['reason'].apply(lambda t : t.replace('-','').replace(' ',''))
df['reason'].value_counts().head(1)


# In[ ]:


#Countplot of 911 calls by their top 10 reasons
plt.figure(figsize=(12,6))
sns.countplot( y='reason',data=df,order=df['reason'].value_counts().index[:10],hue='type', palette='viridis')
plt.legend(loc=4)
plt.title('Top 10 Reason of call ')


# Thus vehicle accident is primary cause of 911 calls of which most of are traffic related.
# 
# For emergency services 911 calls most reasons are respiratory emergency, fall victim and cardiac emergency. 
# 
# For Fire related 911 calls most are due to fire alarms.

# **Lets Study the Townships distributions.**

# In[ ]:


# Countplot of type of calls based on top 10 township
plt.figure(figsize=(18,6))
sns.countplot( x='twp',data=df,order=df['twp'].value_counts().index[:10], hue='type', palette='viridis')
plt.title('Township wise type of calls')


# Lower Merion township has highest number of traffic and fire related 911 calls and second highest number of emeregency 911 calls.
# 
# Norristown township has highest number of Emergency 911 calls.

# In[ ]:


# We want to see countplot of top 10 Emeregency reasons in Lower Merion township.
plt.figure(figsize=(10,6))
test1=df[df['twp']=='LOWER MERION']
test2=test1[test1['type']=='EMS']
sns.countplot(y='reason',data=test2,order=test2['reason'].value_counts().index[:10],palette='RdYlBu')
plt.title('LOWER MERION EMERGENCY CALLS')


# Lower Merion township has mainly vehicle accidents as main reason behind 911 emergency calls.  

# In[ ]:


# We want to see countplot of top 5 Traffic reasons in Lower Merion township.
plt.figure(figsize=(12,6))
test3=df[df['twp']=='LOWER MERION']
test4=test3[test3['type']=='Traffic']
sns.countplot(y='reason',data=test4, order=test4['reason'].value_counts().index[:5],palette='RdYlBu')
plt.title('LOWER MERION TRAFFIC CALLS')


# In Lower Merion Vehicle Accident is the main contributor for Traffic related 911 calls. So it can be easily concluded that for townships with higher number of vehicle accidents, emergency 911 calls due to vehicle accidents are also high. 
# 
# Lets see for Norristown township which has highest number of Emergency 911 calls but comparaticely very less amount of traffic related 911 calls.

# In[ ]:


# We want to see countplot of top 10 Emeregency reasons in Norristown township.
plt.figure(figsize=(10,6))
test5=df[df['twp']=='NORRISTOWN']
test6=test5[test5['type']=='EMS']
sns.countplot(y='reason',data=test6,order=test6['reason'].value_counts().index[:10],palette='RdYlBu')
plt.title('NORRISTOWN EMERGENCY CALLS')


# In[ ]:


# We want to see countplot of top 5 Traffic reasons in Norristown township.
plt.figure(figsize=(12,6))
test7=df[df['twp']=='NORRISTOWN']
test8=test7[test7['type']=='Traffic']
sns.countplot(y='reason',data=test8, order=test8['reason'].value_counts().index[:5],palette='RdYlBu')
plt.title('NORRISTOWN TRAFFIC CALLS')


# For Norristown we can clearly observe that vehicle accidents are main reason behind traffic related 911 calls, but it's not the primary reason behind 911 emergency calls, rather it's health issue like respiratory and cardiac issues. 
# 
# Thus townships with higher number of vehicle accidents, 911 emergency calls are flooded with them.

# In[ ]:


# We want to see countplot of top 5 Fire reasons in Lower Merion township.
plt.figure(figsize=(12,6))
test9=df[df['twp']=='LOWER MERION']
test10=test9[test9['type']=='Fire']
sns.countplot(y='reason',data=test10, order=test10['reason'].value_counts().index[:5],palette='RdYlBu')
plt.title('LOWER MERION FIRE CALLS')


# In Lower Merion township, Fire related 911 calls is mostly due to Fire alarms and comparatively very less due to vehicle accident. 
# 
# Thus vehicle accidents causes serious injuries to humans than catching fire.

# **Let's study timing distribution.**

# In[ ]:


#We timestamp column is string type. So we will chnage it to datetime object
df['timeStamp']= pd.to_datetime(df['timeStamp'])


# In[ ]:


#Now we will create three separate columns Hour, Month, Day of Week from this datetime object
df['Hour'] = df['timeStamp'].apply(lambda time : time.hour)
df['DayofWeek'] = df['timeStamp'].apply(lambda time : time.dayofweek)
df['Month'] = df['timeStamp'].apply(lambda time : time.month)


# In[ ]:


#Now we will convert the day of week to actual day by following dict
dmap= {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'}
df['DayofWeek']= df['DayofWeek'].map(dmap)
df.head()


# In[ ]:


#We want plot countplot to see daywise 911 calls based on different type
plt.figure(figsize=(12,6))
sns.countplot(x='DayofWeek',data=df,hue='type',palette='viridis')
#As legend is inside. We will relocate it
plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)
plt.title('Daywise distribution of Calls')


# In Sunday Traffic related 911 calls reduces to half of Friday which has maximum.  Whereas Fire and Emergency 911 calls comparatively don't have that much of variation daywise.

# In[ ]:


# We want to see countplot of hourly distribution of traffic 911 calls.
plt.figure(figsize=(12,6))
sns.countplot(x='Hour',data=df[df['type']=='Traffic'], palette='viridis')
plt.title('Hourwise distribution of Traffic Calls')


# Thus most of the traffic related 911 calls are peaked between 3-5pm. As vehicle accidents being prime reason behind, this is a rush hour period.

# In[ ]:


plt.figure(figsize=(12,6))
sns.countplot(x='Hour',data=df[df['reason']=='CARDIACEMERGENCY'], palette='viridis')
plt.title('Hourwise distribution of Cardiac emergency Calls')


# Thus most of the Cardiac emergency 911 calls are done between 9AM and 2PM i.e. mostly during peak hour of the day.

# In[ ]:


plt.figure(figsize=(12,6))
sns.countplot(x='Hour',data=df[df['type']=='Fire'], palette='viridis')
plt.title('Hourwise distribution of Fire Calls')


# Most of the fire related 911 calls are done during daytime, peaking at evening.

# In[ ]:


#We want plot countplot to see monthwise 911 calls based on different type
plt.figure(figsize=(12,6))
sns.countplot(x='Month',data=df,hue='type',palette='viridis')
#As legend is inside. We will relocate it
plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)
plt.title('Monthwise distribution of Calls')


# In[ ]:


#Now we will explore by grouping the monthly datas
bymonth = df.groupby('Month').count()
bymonth.head()


# **As number of calls can be uniquely identified with latitude or longitude. Lets take that as our parameter for number of calls.**

# In[ ]:


bymonth['lat'].plot() 


# Thus 911 calls are more during winter.

# In[ ]:


sns.lmplot(x='Month',y='twp',data=bymonth.reset_index()) 


# We get a linear fit showing a decreasing trend but with a large error.

# In[ ]:


#Now we shall introduce a new column date from timestamp
t=df['timeStamp'].iloc[0]
df['Date'] = df['timeStamp'].apply(lambda t : t.date())


# In[ ]:


#Now we shall group by date and plot any column
plt.figure(figsize=(12,6))
df.groupby('Date').count()['lat'].plot()
plt.tight_layout()
plt.title('Date wise distribution of Calls')


# In[ ]:


df['Date'].value_counts().head(1)


# In[ ]:


df['Date'].value_counts().tail(1)


# The highest number of calls (2187 calls) were done on 2nd March, 2018 and lowest number of calls (41 calls) were done on 8th October, 2017
# 
# Now we shall plot for three different type of calls.

# In[ ]:


plt.figure(figsize=(12,6))
df[df['type']=='Traffic'].groupby('Date').count()['lat'].plot()
plt.tight_layout()
plt.title('Date wise distribution of Traffic Calls')
plt.figure(figsize=(12,6))


# In[ ]:


plt.figure(figsize=(12,6))
df[df['type']=='EMS'].groupby('Date').count()['lat'].plot()
plt.tight_layout()
plt.title('Date wise distribution of Emergency Calls')


# In[ ]:


plt.figure(figsize=(12,6))
df[df['type']=='Fire'].groupby('Date').count()['lat'].plot()
plt.tight_layout()
plt.title('Date wise distribution of Fire Calls')


# **Now lets  summarize the whole picture of 911 calls**

# Now we will restructure the dataframe so that columns are Hours and Rows are day of the week. We will try it via unstack method. So first we will group by multiple columns crating multi level index and then use unstack to form a matrix out of them.

# In[ ]:


dayhour = df.groupby(by=['DayofWeek','Hour']).count()['type'].unstack()


# In[ ]:


#Now lets see it as a heatmap
plt.figure(figsize=(12,6))
sns.heatmap(dayhour,cmap='viridis')
plt.title('Day-hour distribution of 911 Calls')


# This gives a very nice overall picture of 911 calls.

# In[ ]:


#Now lets see a clustermap
plt.figure(figsize=(12,6))
sns.clustermap(dayhour,cmap='coolwarm')
plt.title('Day-hour cluster distribution of 911 Calls')


# In[ ]:


#Now lets do the same thing for months instead of hours
daymonth = df.groupby(by=['DayofWeek','Month']).count()['type'].unstack()


# In[ ]:


plt.figure(figsize=(12,6))
sns.heatmap(daymonth,cmap='viridis')
plt.title('Day-Month distribution of 911 Calls')


# In[ ]:


plt.figure(figsize=(12,6))
sns.clustermap(daymonth,cmap='coolwarm')
plt.title('Day-month cluster distribution of 911 Calls')

