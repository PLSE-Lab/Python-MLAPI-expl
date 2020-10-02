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


# # Visualizing some Emergency 911 calls US data
# 
# My first data capstone project for my Udemy Data Science and ML Bootcamp, by Jose Portilla of Udemy.
# 
# https://www.udemy.com/python-for-data-science-and-machine-learning-bootcamp
# 
# Don't expect much, it is just meant to practice my visualization and pandas skills.
# 
# 

# In[ ]:


#Visualization tools
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('../input/911.csv')
df.info()


# In[ ]:


df.keys()


# In[ ]:


topzips=df['zip'].value_counts().head(5)
toptowns=df['twp'].value_counts().head(5)

toptowns, topzips


# **Do toptowns contains such zips? Now we check that 4 zips do**

# In[ ]:


toptownsinfo = pd.concat( [ df[df['twp']==x] for x in list(toptowns.keys())] )

list( set(toptownsinfo['zip']) &  set(topzips.keys()) )


# ## Whats the most common reasons?

# In[ ]:


df['Reason'] = df['title'].apply(lambda title: title.split(':')[0])
sns.countplot(x='Reason',data=df,palette='viridis')


# **Does it varies with time?**

# In[ ]:


df['timeStamp'] = pd.to_datetime(df['timeStamp'])
df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)
df['Month'] = df['timeStamp'].apply(lambda time: time.month)
df['Day of Week'] = df['timeStamp'].apply(lambda time: time.dayofweek)

# formating day fo week
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['Day of Week'] = df['Day of Week'].map(dmap)


# In[ ]:


sns.countplot(x='Day of Week',data=df,hue='Reason',palette='viridis')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) # To relocate the legend


# In[ ]:


sns.countplot(x='Month',data=df,hue='Reason',palette='viridis')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)# To relocate the legend


# **Variation by months/date**

# In[ ]:


byMonth = df.groupby('Month').count()
byMonth['twp'].plot()


# In[ ]:


df['Date']=df['timeStamp'].apply(lambda t: t.date())
df.groupby('Date').count()['twp'].plot()
plt.tight_layout()


# # On the day/hour relation to reports number

# In[ ]:


dayHour = df.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()
dayHour.head()


# In[ ]:


plt.figure(figsize=(12,6))
sns.clustermap(dayHour,cmap='viridis')


# In[ ]:


dayMonth = df.groupby(by=['Day of Week','Month']).count()['Reason'].unstack()
sns.clustermap(dayMonth,cmap='viridis')


# In[ ]:




