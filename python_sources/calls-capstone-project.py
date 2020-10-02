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


# Importing visualization libraries

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('matplotlib', 'inline')


# Reading the csv file

# In[ ]:


df = pd.read_csv('../input/911.csv')


# In[ ]:


df.info()


# Checking the head of df

# In[ ]:


df.head()


# What are the top 5 zipcodes for 911 calls? 

# In[ ]:


df['zip'].value_counts().head()


# What are the top 5 townships (twp) for 911 calls? 

# In[ ]:


df['twp'].value_counts().head()


# Take a look at the 'title' column, how many unique title codes are there?

# In[ ]:


df['title'].nunique()


# create a new column called "Reason" that contains this string value (For example, if the title column value is EMS: BACK PAINS/INJURY , the Reason column value would be EMS.)

# In[ ]:


df['Reason'] = df['title'].apply(lambda title: title.split(':')[0])


# What is the most common Reason for a 911 call based off of this new column?

# In[ ]:


df['Reason'].value_counts()


# Seaborn to create a countplot of 911 calls by Reason.

# In[ ]:


sns.set_style("darkgrid")
sns.countplot(x='Reason',data=df, palette = 'viridis')


# What is the data type of the objects in the timeStamp column?

# In[ ]:


type(df['timeStamp'].iloc[0])


# Use pd.to_datetime to convert the column from strings to DateTime objects

# In[ ]:


df['timeStamp'] = pd.to_datetime(df['timeStamp'])
type(df['timeStamp'].iloc[0])


# use .apply() to create 3 new columns called Hour, Month, and Day of Week. You will create these columns based off of the timeStamp column, reference the solutions if you get stuck on this step.

# In[ ]:


df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)
df['Month'] = df['timeStamp'].apply(lambda time: time.month)
df['Day of Week'] = df['timeStamp'].apply(lambda time: time.dayofweek)


# Use the .map() with this dictionary to map the actual string names to the day of the week

# In[ ]:


dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['Day of Week'] = df['Day of Week'].map(dmap)


# seaborn to create a countplot of the Day of Week column with the hue based off of the Reason column.

# In[ ]:


sns.countplot(x='Day of Week', data=df, hue='Reason', palette='viridis')
#to relocate the legend
plt.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)


# the same for Month:

# In[ ]:


sns.countplot(x='Month', data=df, hue='Reason', palette='viridis')
#to relocate the legend
plt.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)


# create a gropuby object called byMonth, where you group the DataFrame by the month column and use the count() method for aggregation. Use the head() method on this returned DataFrame.

# In[ ]:


byMonth = df.groupby('Month').count()
byMonth.head()


# create a simple plot off of the dataframe indicating the count of calls per month

# In[ ]:


byMonth['lat'].plot()


# use seaborn's lmplot() to create a linear fit on the number of calls per month. 

# In[ ]:


sns.lmplot(x='Month',y='twp',data=byMonth.reset_index())


# Create a new column called 'Date' that contains the date from the timeStamp column. 

# In[ ]:


df['Date'] = df['timeStamp'].apply(lambda t:t.date())


# groupby this Date column with the count() aggregate and create a plot of counts of 911 calls

# In[ ]:


df.groupby('Date').count()['lat'].plot()
plt.tight_layout()


# Now recreate this plot but create 3 separate plots with each plot representing a Reason for the 911 call

# In[ ]:


df[df['Reason']=='Traffic'].groupby('Date').count()['lat'].plot()
plt.title('Traffic')
plt.tight_layout()


# In[ ]:


df[df['Reason']=='Fire'].groupby('Date').count()['lat'].plot()
plt.title('Fire')
plt.tight_layout()


# In[ ]:


df[df['Reason']=='EMS'].groupby('Date').count()['lat'].plot()
plt.title('EMS')
plt.tight_layout()


# restructure the dataframe so that the columns become the Hours and the Index becomes the Day of the Week.

# In[ ]:


dayHour= df.groupby(by=['Day of Week', 'Hour']).count()['Reason'].unstack()


#  create a HeatMap using this new DataFrame

# In[ ]:


plt.figure(figsize=(12,6))
sns.heatmap(dayHour, cmap='viridis')


# create a clustermap using this DataFrame.

# In[ ]:


sns.clustermap(dayHour,cmap='viridis')


# repeat these same plots and operations, for a DataFrame that shows the Month as the column.

# In[ ]:


dayMonth = df.groupby(by=['Day of Week', 'Month']).count()['Reason'].unstack()
dayMonth.head()


# In[ ]:


plt.figure(figsize=(12,6))
sns.heatmap(dayMonth, cmap='viridis')


# In[ ]:


sns.clustermap(dayMonth,cmap='viridis')

