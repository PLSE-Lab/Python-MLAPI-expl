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


# In[ ]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


uber = pd.read_csv("../input/uberdrives/My Uber Drives - 2016.csv")


# In[ ]:


#checking the Data size
uber.shape


# In[ ]:


#checking the content of the data
uber.head()


# In[ ]:


#checking the data types of the variable
uber.dtypes


# ** As you can see from the data types, the start and end date are not in *Datetime* format, so let's convert it to the Datetime format****

# In[ ]:


uber.tail()


# **You can see that there is a invalid data entry in the last row of Startdate, and all the data except miles driven all null, so we can drop this row**

# In[ ]:


uber.drop(uber.tail(1).index, inplace=True)


# In[ ]:


uber.tail()


# In[ ]:


#converting the Datatype of date
uber['START_DATE*']=uber['START_DATE*'].astype('datetime64[ns]')
uber['END_DATE*']=uber['END_DATE*'].astype('datetime64[ns]')


# In[ ]:


#creating new variable called duration to calculate the duration and speed of the drive of the journey
uber['duration']= uber['END_DATE*']-uber['START_DATE*']
uber['duration']=uber['duration'].dt.total_seconds()/60.0
uber['duration']=uber['duration'].astype('float')
uber['speed']=uber['MILES*']/(uber['duration']/60)


# In[ ]:


uber


# In[ ]:


fig,ax=plt.subplots(1,3,figsize=(38,8))
a=uber.groupby(['PURPOSE*'])['duration'].mean().sort_values(ascending=True).plot(kind='barh', ax=ax[0])
a.title.set_text('Mean of duration for each Purpose')
b=uber.groupby(['PURPOSE*'])['duration'].count().sort_values(ascending=True).plot(kind='barh', ax=ax[1])
b.title.set_text('Number of drives per each Purpose')
c=uber.groupby(['PURPOSE*'])['MILES*'].sum().sort_values(ascending=True).plot(kind='barh', ax=ax[2])
c.title.set_text('Total Number of Miles driven per each Purpose')


# ### 1. From the above plots we can infer that the most Number of drives taken by the driver is Meeting, Meal/ Entertainment, which gets reflected in the plot 3 in Total Number of miles.
# 
# ### 2. But the mean of the duration is very high for the COMMUTE, this anomoly is due to the fact that the driver has driven only one Drive for commute which turned out to be a long drive of 180 Miles and and 185 Minutes
# 
# 

# In[ ]:


tab_m=uber.groupby(['PURPOSE*']).mean()
print(tab_m)


# ### 1. From the Above table of Average miles and Average time taken we can see that the AIRPORT/TRAVEL have less mean if Miles and High Duration, this might be because, Since the Airport will be located at the city and the traffic and congestion in the city might be the cuase for the low speed, similarly for travel.
# 

# In[ ]:


uber['speed'].median()


# ### 1.From the Above median we can say that the Median speed of the driver Through out the drive is 21.0 Miles/Hour which is a Average Speed and also a safe speed, so we can concluse that the driver is a safe driver.

# In[ ]:


plt.figure(figsize=(15,8))
sns.countplot(uber['PURPOSE*'], orient="v")


# ### On seeing the below plot it can be said that most booked reason of cabs were Meal/entertainment, errand/supplies and meeting, so it cannot be said that most booked reason for cabs are most  miles driven

# In[ ]:


## Creating Columns for Month and Hour in a Day
uber['month']=uber['START_DATE*'].dt.month
uber['hour']=uber['START_DATE*'].dt.hour


# In[ ]:


fig,ax=plt.subplots(1,2,figsize=(38,8))
i=uber.groupby('month')['MILES*'].sum().sort_values(ascending=True).plot(kind='bar', ax=ax[0], color='orange')
j=uber.groupby('month')['MILES*'].mean().sort_values(ascending=True).plot(kind='bar', ax=ax[1], color='red')
i.title.set_text('Month vs Total Miles')
j.title.set_text('Month vs Average Miles/Month')


# ### 1. From the above graph we can say that the driver drived most miles during the month of October,March and August
# ### 2. But the average miles driven per months is highest in April even though it is not among the highest miles driven month, this means that during april month the driver might have gone for a long trip which drastically increased the average for that month

# In[ ]:


uber['START_DATE*'].dt.month.value_counts().sort_values(ascending=True).plot(kind='bar', color='green')
plt.title('Number of rides/Month')
plt.axhline(y=uber['START_DATE*'].dt.month.value_counts().mean(), xmin=0, xmax=3.5, linestyle="--")


# ### The driver seems to drive mostly above his average Number of rides, only during the months of 1,4,9,5 the number of rides have dropped below average.

# In[ ]:


uber['hour'].value_counts().sort_values(ascending=False).plot(kind='bar')
plt.axhline(y=uber['hour'].value_counts().mean(), xmin=0, xmax=3.5, linestyle="--")
plt.title('Hour in a day/Number of rides')


# ### From the above graph it can bee seen that most of the rides through out the year were recieved from morning 10 am to night 9 pm, during other hours of the day, the ride counts are below average. but still 50 is a good average

# In[ ]:


val=uber["MILES*"]
mil_cat=["<=5","5-10","10-15","15-20",">20"]
dic=dict()
for item in mil_cat:
    dic[item]=0
for i in val.values:
    if i<=5:
        dic["<=5"]+=1
    elif i<=10:
        dic["5-10"]+=1
    elif i<=15:
        dic["10-15"]+=1
    elif i<=20:
        dic["15-20"]+=1
    else:
        dic[">20"]+=1
final=pd.Series(dic)
final.sort_values(inplace=True,ascending=False)


# In[ ]:


sns.barplot(final.index, final.values)


# ### Most of the rides are lesser than 5kms and between 5-10Kms, So we can say that the driver rarely opts for long distance drives

# In[ ]:




