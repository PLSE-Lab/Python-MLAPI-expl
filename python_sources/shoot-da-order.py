#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


data=pd.read_csv('/kaggle/input/data-police-shootings/fatal-police-shootings-data.csv')


# In[ ]:


data.head()


# In[ ]:


data.info()


# # **Let's check if any missing value present or not ?**

# In[ ]:


sns.heatmap(data.isnull(),cmap='Blues')


# Well having a lot's of missing values :(

# In[ ]:


data.isnull().sum()


# In[ ]:


missing_percent= (data.isnull().sum()/len(data))[(data.isnull().sum()/len(data))>0].sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Percentage':missing_percent*100})
missing_data


# In[ ]:


data.dropna(inplace=True)


# # **Now let's start analyzing the given data :)**

# In[ ]:


data.head()


# 1. **Age**

# In[ ]:


data['age'].value_counts()


# In[ ]:


sns.distplot(data['age'],kde=False)


# Variation of Age vs Race

# In[ ]:


plt.figure(figsize=(8,5))
sns.stripplot(x='race',y='age',data=data)


# Variation of Age vs Race vs Manner_of_death

# In[ ]:


plt.figure(figsize=(8,4))
sns.stripplot(x='race',y='age',data=data,hue='manner_of_death')


# Variation of Age vs Mental illness

# In[ ]:


plt.figure(figsize=(8,5))
sns.violinplot(x='signs_of_mental_illness',y='age',data=data)


# Variation of Flee vs Age vs Gender

# In[ ]:


plt.figure(figsize=(8,5))
sns.violinplot(x='flee',y='age',data=data,hue='gender',split=True)


# 2. **Signs_of_mental_illness**

# In[ ]:


sns.countplot(x=data['signs_of_mental_illness'])


# **Many people with mental illness were also shot which is not a good thing done by the police department :(**

# 3. Armed

# In[ ]:


data['armed'].value_counts()


# In[ ]:


plt.figure(figsize=(16,6))
sns.barplot(y=data['armed'].value_counts()[0:10],x=data['armed'].value_counts()[0:10].index)


# Well a lot people were armed when they were killed 

# 4. **Manner of death**

# In[ ]:


data['manner_of_death'].value_counts()


# In[ ]:


sns.countplot(x=data['manner_of_death'])


# Manner_of_death vs Gender

# In[ ]:


sns.countplot(x=data['manner_of_death'],hue=data['gender'])


# In[ ]:


sns.countplot(hue=data['manner_of_death'],x=data['flee'])


# How  threat level varies vs Manner_of_death
# 

# In[ ]:


sns.countplot(hue=data['threat_level'],x=data['manner_of_death'])


# 5. **Gender**

# In[ ]:


sns.countplot(x=data['gender'])


# **Well men do more crimes than females**

# # **How one thing depend's on other**

# In[ ]:


sns.heatmap(data.corr(),cmap='coolwarm')


# # **States and City**

# 1. State

# In[ ]:


data.head(1)


# In[ ]:


data['state'].value_counts()


# In[ ]:


plt.figure(figsize=(16,6))
sns.barplot(x=data['state'].value_counts().index,y=data['state'].value_counts())


# CA is the state which has highest crime rate :(

# 2. Cities

# In[ ]:


data['city'].value_counts()


# In[ ]:


plt.figure(figsize=(16,6))
sns.barplot(x=data['city'].value_counts()[0:30],y=data['city'].value_counts()[0:30].index)


# Well well los Angeles in the city with the max crime rate :(

# # **Analyzing the data according to the date**

# Spilliting the date column into Year , Month and Date :)

# In[ ]:


data['Year']=[d.split('-')[0] for d in data['date']]
data['Month']=[d.split('-')[1] for d in data['date']]
data['Day']=[d.split('-')[2] for d in data['date']]
data['Year']=data['Year'].astype(int)
data['Month']=data['Month'].astype(int)
data['Day']=data['Day'].astype(int)


# Our final data 

# In[ ]:


data.head()


# Let's Begin our visualisation

#  1. In Which year max people were shoot ???

# In[ ]:


plt.figure(figsize=(8,5))
sns.countplot(x=data['Year'])


# Well in 2015 most of the encounter take place :(

# Well analyze the **Months** in which most encouter take place !

# In[ ]:


plt.figure(figsize=(16,5))
sns.countplot(x=data['Month'],order = data['Month'].value_counts().index)


# Well in the cold months January , February mostly encouter take place !

# Day !!!!

# In[ ]:


plt.figure(figsize=(16,5))
sns.countplot(x=data['Day'],order = data['Day'].value_counts().index)

