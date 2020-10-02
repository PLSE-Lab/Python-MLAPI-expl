#!/usr/bin/env python
# coding: utf-8

# # **Terrorism in India Analysis**
# #### In this example we  will perform exploiratory data analysis on terrorism in india. The goal is to get a visual picture of the scenario in india based on **Global Terrorism Database**

# ## **Data extraction stage**
# 
# In this stage we load essential libraries and the dataset. Then we filter the dataset with respect to india. From here on we will only work with the data relevant to **India**

# In[ ]:


#Importing libraries and the dataset
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
dataset = pd.read_csv('../input/globalterrorismdb_0718dist.csv',encoding='ISO-8859-1')

#Selecting the data with respect to india
dataset_india = dataset[dataset.loc[:,'country_txt'] == "India"]


# Lets have a look at what we got from our original dataset.

# Looks like we have got many null values in the dataset. Lets get rid of as many of them as we can. Here we only keep columns which have null values less than **10000**

# In[ ]:


import numpy as np
#printing count of null values
null_count_array = pd.isna(dataset_india).sum().values
median = np.median(null_count_array)
filter = lambda x : x<10000
filter_vectorise = np.vectorize(filter)
boolean_array = filter_vectorise(null_count_array)

dataset_filtered = dataset_india.iloc[:,boolean_array]
#display(dataset_filtered)


# Looks good now lets create an additional column for dates and do a little bit of cleanup as well

# In[ ]:


#Replacing remaining null with appropriate Datatype

datatype_int = dataset_filtered.select_dtypes(include='int64')
datatype_float = dataset_filtered.select_dtypes(include='float64')
datatype_object  = dataset_filtered.select_dtypes(include='object')

datatype_int = datatype_int.replace(np.nan,0,regex=True)
datatype_float = datatype_float.replace(np.nan,0.0,regex=True)
#datatype_object = datatype_object.replace(np.nan,"",regex=True)
dataset_india_merged = pd.concat([datatype_int,datatype_float,datatype_object],axis = 1)
dataset_india_merged['provstate'].loc[dataset_india_merged['provstate'] == 'Odisha'] = 'Orissa'
#display(dataset_india_merged)


# ## **Time Series Analysis**
# The below graphs shows the number of terrorist attacks on **India** from **1970 to 2017**. Initially it mainly tries to show how terrorist attacks have increased in india in the past few years. 

# In[ ]:


#Time series analysis
import seaborn as sns
from matplotlib import pyplot as plt

sns.set()

date_wise_count = pd.pivot_table(dataset_india_merged,values = ['eventid'],index = ['iyear'],aggfunc='count')
date_wise_count.reset_index(inplace=True)
date_wise_count.rename_axis({"iyear":"year","eventid":'count'},inplace=True)
date_wise_count.columns  = ['year','attack_count']
plt.figure(figsize=(32,9))
plt.title = 'Time series analysis of terrorist attacks'
sns.lineplot(data=date_wise_count,x=date_wise_count.year,y=date_wise_count.attack_count)
plt.fill_between(date_wise_count.year.values, date_wise_count.attack_count.values)
plt.show()


# ## **State wise Analysis**
#  The below graphs show the effect on terrorist activities in different states in india. We start off with a visualisation of terrorist attacks in different states followed by in depth analysis of attack type and target type with respect to individual states. 
#  
#  Finally we have tried to visualise how terrorist activities in individual states grew with respect to time
# 
# ### **State wise attack count**

# In[ ]:


#State wise attack count
plt.figure(figsize=(60,15))
sns.set()
plt.title = "Time series analysis of terrorist attacks"
sns.countplot(data=dataset_india_merged,x='provstate')
plt.show()


# ### Statewise attack type

# In[ ]:


#State wise attack type
plt.figure(figsize=(5,5))
sns.set(style='darkgrid')
plt.title = "Time series analysis of terrorist attacks"
sns.catplot(data=dataset_india_merged,y='attacktype1_txt',kind='count',col='provstate',col_wrap=6)
plt.show()


# ### Statewise Target Type

# In[ ]:


#State wise target type
plt.figure(figsize=(5,5))
sns.set(style='darkgrid')
plt.title = "Time series analysis of terrorist attacks"
sns.catplot(data=dataset_india_merged,y='targtype1_txt',kind='count',col='provstate',col_wrap=6)
plt.show()


# ### Time series analysis of different states of **India** and how terrorism changed across time

# In[ ]:


#Time series combined evolution of state attack with time

states = dataset_india_merged['provstate'].unique()
plt.figure(figsize=(60,12))
plt.title = 'Time series analysis of terrorist attacks'
for x in states:
    temp = dataset_india_merged[dataset_india_merged['provstate'] == x]
    date_wise_count = pd.pivot_table(temp,values = ['eventid'],index = ['iyear'],aggfunc='count')
    date_wise_count.reset_index(inplace=True)
    date_wise_count.rename_axis({"iyear":"year","eventid":'count'},inplace=True)
    date_wise_count.columns  = ['year','attack_count']
    sns.lineplot(data=date_wise_count,x=date_wise_count.year,y=date_wise_count.attack_count,lw=3)
plt.legend(states)
plt.show()


# ## Attack type analysis
# 
# In this section we look at the different types of terrorist attacks **India** has suffered and how these attacks have evolved through time
# 
# The first graph shows us the different types of attacks in india and the frequency of the attacks
# The second graph shows us a plot where we see how these attacks have changed with time

# In[ ]:


#Attack type analysis
plt.figure(figsize=(60,10))
sns.set(style='darkgrid')
sns.countplot(data = dataset_india_merged,y= 'attacktype1_txt')
plt.show()


# In[ ]:


# Time series evolution of attack types

#states = dataset_india_merged['provstate'].unique()
plt.figure(figsize=(60,12))

states = dataset_india_merged['attacktype1_txt'].unique()

for x in states:
    temp = dataset_india_merged[dataset_india_merged['attacktype1_txt'] == x]
    date_wise_count = pd.pivot_table(temp,values = ['eventid'],index = ['iyear'],aggfunc='count')
    date_wise_count.reset_index(inplace=True)
    date_wise_count.rename_axis({"iyear":"year","eventid":'count'},inplace=True)
    date_wise_count.columns  = ['year','attack_count']
    sns.lineplot(data=date_wise_count,x=date_wise_count.year,y=date_wise_count.attack_count,lw=3)
plt.legend(states)
plt.show()


# ## Target type analysis
# 
# In this section we look at the different types of terrorist attacks **India** has suffered and how these targets have changed with time
# 
# The first graph shows us the different types of targeted sites in india and the frequency of the attacks
# The second graph shows us a plot where we see how these targets have changed with time

# In[ ]:


#Target type analysis
plt.figure(figsize=(60,10))
sns.set(style='darkgrid')
sns.countplot(data = dataset_india_merged,y= 'targtype1_txt')
plt.show()


# In[ ]:


# Time series evolution of targets
plt.figure(figsize=(60,12))

states = dataset_india_merged['targtype1_txt'].unique()

for x in states:
    temp = dataset_india_merged[dataset_india_merged['targtype1_txt'] == x]
    date_wise_count = pd.pivot_table(temp,values = ['eventid'],index = ['iyear'],aggfunc='count')
    date_wise_count.reset_index(inplace=True)
    date_wise_count.rename_axis({"iyear":"year","eventid":'count'},inplace=True)
    date_wise_count.columns  = ['year','attack_count']
    sns.lineplot(data=date_wise_count,x=date_wise_count.year,y=date_wise_count.attack_count,lw=3)
plt.legend(states)
plt.show()


# ### Group Type Analysis
# In this section we will focus on the various groups that have been responsible for terrorist activities in **India**

# This graph represents the numbers of attacks done by different groups. The x scale is in the log scale for visual appeal

# In[ ]:


#Group responsible for attacks
plt.figure(figsize=(60,65),dpi=150)
sns.countplot(data= dataset_india_merged,y='gname',log=True)
plt.show()


# As evident from the graphs there are a lot of groups which have caused harm to **India**. So we filter only those groups which have done **10 or more attacks** irrespective of timeframe. Then we plot the activites of these groups across the entire timeline. The timeline is presented in descending order from the most active to the least active group

# In[ ]:


# Top terror groups

terror_groups = pd.pivot_table(dataset_india_merged,index='gname',values='eventid',aggfunc='count')
terror_groups.reset_index(inplace=True)
terror_groups = terror_groups[terror_groups['eventid'] >= 10]
terror_groups.sort_values('eventid',ascending=False,inplace=True)
terror_groups = terror_groups['gname'].head(20).values


# In[ ]:


# Time series analysis
states = terror_groups
for x in states:
    plt.figure(figsize=(60,10))
    plt.figtext = x
    temp = dataset_india_merged[dataset_india_merged['gname'] == x]
    date_wise_count = pd.pivot_table(temp,values = ['eventid'],index = ['iyear'],aggfunc='count')
    date_wise_count.reset_index(inplace=True)
    date_wise_count.rename_axis({"iyear":"year","eventid":'count'},inplace=True)
    date_wise_count.columns  = ['year','attack_count']
    sns.lineplot(data=date_wise_count,x=date_wise_count.year,y=date_wise_count.attack_count)
    plt.fill_between(date_wise_count.year.values, date_wise_count.attack_count.values)
    display(x)
    plt.show()
    


# The above graphs have been plotted to show how the most notorious terrorist groups evolved with time in **India**

# ### **Weapon Type Analysis**
# 
# Based on the top groups we will analyse the most common weapons used by terrorist groups.

# The main weapon type used by the terrorist groups

# In[ ]:


data_subset = dataset_india_merged.loc[dataset_india_merged['gname'].isin(terror_groups)]
sns.catplot(data=data_subset,y='weaptype1_txt',kind='count',col='gname',col_wrap=6,log = True)
plt.show()


# The main weapon subtype used by the terrorist groups

# In[ ]:


data_subset = dataset_india_merged.loc[dataset_india_merged['gname'].isin(terror_groups)]
sns.catplot(data=data_subset,y='weapsubtype1_txt',kind='count',col='gname',col_wrap=6,log = True)
plt.show()


# ## **Conclusion**
# 
# In this notebook we have tried to do a visual analysis of Terrorism in **India**. We have tried to analyse how the country and different states have been affected by terrorism over the years. What type of establishments were targeted and what type of attacks were carried out. We have also focused on the various terror groups that have affected **India**. We have shortlisted the top 20 terror groups and how active they have been over time. Furthermore we also have tried to analyse what weapons these groups have used.
