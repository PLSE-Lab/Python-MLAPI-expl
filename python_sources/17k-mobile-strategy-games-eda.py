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


# Import some visualization packages
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")
import missingno as msno


# In[ ]:


# bring in the appstore data
df = pd.read_csv('/kaggle/input/17k-apple-app-store-strategy-games/appstore_games.csv')
df.head(100)


# In[ ]:


# use missingno for missing data
msno.matrix(df)


# In[ ]:


fig, ax = plt.subplots(1,1, figsize=(12, 7), dpi=72)
sns.countplot(x='Average User Rating', data=df)


# Most Common Rating is 4.5 and the 4.0

# In[ ]:


fig, ax = plt.subplots(1,1, figsize=(12, 7), dpi=72)
sns.countplot(x='Price', data=df)
plt.xticks(rotation='vertical')


# Most Games are free, followed by a steady but slow increase in price

# In[ ]:


fig, ax = plt.subplots(1,1, figsize=(12, 7), dpi=72)
sns.countplot(x='Age Rating', data=df)


# very few ratings outside of 4+(lowest rating)

# In[ ]:


#lets get the top 100 most rated icons 
#TODO


# In[ ]:


fig, ax = plt.subplots(1,1, figsize=(12, 7), dpi=72)
sns.countplot(y="Developer", data=df, ax=ax, order=df.Developer.value_counts().iloc[:10].index)


# In[ ]:


#let's get average rating per developer top ten
fig, ax = plt.subplots(1,1, figsize=(12, 7), dpi=72)
print(df.Developer.value_counts().iloc[:10].index)
#  'Domyung Kim', 'Tapps Tecnologia da Informa\xe7\xe3o Ltda.' causes issues
top_developers = ['Vikash Patel',
       'Netsummit Marketing, Inc.', 'GabySoft', 'NetSummit Enterprises, Inc.',
       'Andrew Kudrin', 'MmpApps Corp.',  'Amy Prizer',
       'Detention Apps']
for developer in top_developers:
    sns.countplot(y='Average User Rating', data=df.loc[df['Developer'] == developer], ax=ax)


# ### Basically the most posting developers have little to no reviews on their apps
# 
# I checked the opposite but it's just a bunch of developers with one app and no reviews
