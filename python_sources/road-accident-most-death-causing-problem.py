#!/usr/bin/env python
# coding: utf-8

# #### The dataset contains Road and Rail accidents happened in India from 2001-2014.
# 
# ### These are the main questions that I'm going to find out form this kernel:
# 1. Which state is most prone to Road Accidents
# 2. Does the number of accidents increasing per year
# 3. Which time (morning, evening, early morning etc.) is most deadly or prone to accidents.
# 
# ### These are few more questions that can be explore from this dataset :
# 1. Shift in the number of accidents in each state year wise.
# 2. Shift in the number of accidents in each time-range year wise.
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
print(os.listdir("../input"))


# In[ ]:


df=pd.read_csv("../input/Traffic accidents by time of occurrence 2001-2014.csv")
df.columns=["state",'year','type','latenight_0-3hrs','earlymorning_3-6hrs','morning_6-9hrs',
            'earlynoon_9-12hrs','noon_12-15hrs','evening_15-18hrs','earlynight_18-21hrs',
           'night_21-24hrs','total']
df.head(2)


# #### Function to plot bars

# In[ ]:


def plotbar(df,column,vertical,plottype='bar'):
    count=[]
    for x in df[column].unique():
        c=df[df[column]==x].total.sum()
        count.append(c)
    
    plt.figure(figsize=(10,10))
    if(plottype=='bar'):
        sns.barplot(df[column].unique(),count)
    if(plottype=='line'):
        sns.lineplot(df[column].unique(),count)
    if(vertical):
        plt.xticks(range(df[column].nunique()),rotation='vertical')
    plt.show()


# ## Accidents rate per year

# In[ ]:


plotbar(df,"year",False,'line')


# Accidents are increasing continuously per year with an alarming rate.

# Which states are most prone to accidents

# In[ ]:


plotbar(df,'state',True)


# All the major cities of India have very high accident rate. This may be because of high population density and hence high 
# density of vehicles. 

# ## Which type of accidents are most common ?

# In[ ]:


plotbar(df,"type",False)


# Road accidents are way more common than Rail-road accidents and railway accidents. This signifies how common roadways services are
# and how much casual people are there for road safety.

# ## Which hour is most deadly ?

# In[ ]:


time_column=df.columns[3:-1].values
count=[]
for col in time_column:
    count.append(df[col].sum())
sns.barplot(time_column,count)
plt.xticks(range(len(time_column)),rotation='vertical')
plt.show()


# Statistically `evening` and `early morning` are most prone to accidents. 
# 
# But since these hours has most rush for roads, I think `latenight` and `earlymorning` accidents are also considerably high.

# ## FInding accident trend in each state in each time interval

# In[ ]:


df.groupby("state").sum().iloc[:,1:-1].plot.bar(figsize=(15,10),stacked=True,
                                                title="statewise accidents in each time interval")


# ## Finding accidents trend in each time interval over the years

# In[ ]:


df.groupby('year').sum().iloc[:,0:-1].plot(figsize=(15,10),title="Number of accidents per year in each time interval")
plt.show()


# The earlynight accidents (6pm-9pm) have increased a lot, this suggest the increase in road engagements during
# this hour.
# 
# Latenight(9pm-12pm) accidetns have decrease. Rest all have more or less same trends

# ## Finding trend in number of accidents per year in each state

# In[ ]:


df.head()


# In[ ]:


plt.figure(figsize=(15,10))
sns.lineplot(x="state",y="total",hue="year",data=df)
plt.xticks(rotation='vertical')
plt.show()


# This one is interestsing :
#     1. Number of accidents in state like - andhra pradesh, karnataka, kerala, maharashtra, tamil nadu, gujrat 
#     have increased substantially. These states are also the one with most development rates.
#     2. Overall count of accidents have increased for each state,which is very normal as number of vehicles are increasing overall.

# I explored Indian suicide dataset and found the developing states in India (same AP, KARNATAKA, KERALA, GUJRAT etc.) also have very high
# increase in suicide rates.
# 
# That says development is coming at cost - increase in suicides, accidents and may be others.

# In[ ]:




