#!/usr/bin/env python
# coding: utf-8

# # Tracy(11/12/2019)_911 Calls

# This dataset contains the following fields:
# * lat : String variable, Latitude
# * lng: String variable, Longitude
# * desc: String variable, Description of the Emergency Call
# * zip: String variable, Zipcode
# * title: String variable, Title
# * timeStamp: String variable, YYYY-MM-DD HH:MM:SS
# * twp: String variable, Township
# * addr: String variable, Address
# * e: String variable, Dummy variable (always 1)

# ## Import Libraries and Data

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('/kaggle/input/montcoalert/911.csv')


# ## Overview

# In[ ]:


df.info()


# In[ ]:


df.head()


# In[ ]:


df['zip'].value_counts().head(5)


# In[ ]:


df['twp'].value_counts().head(5)


# In[ ]:


len(df['title'].unique())


# ## Creating New Features

# In the titles column there are "Reasons/Departments" specified before the title code. These are EMS, Fire, and Traffic. Use .apply() with a custom lambda expression to create a new column called "Reason" that contains this string value.
# **For example, if the title column value is EMS: BACK PAINS/INJURY , the Reason column value would be EMS.**

# In[ ]:


df['Reason'] = df['title'].apply(lambda title:title.split(':')[0])


# In[ ]:


df['Reason'].value_counts()


# In[ ]:


## Use Seaborn to create a countplot of 911 calls by Reason
sns.countplot(x='Reason',data=df,palette='viridis')


# ## Process TimeStamp Column

# In[ ]:


type(df['timeStamp'].iloc[0])


# In[ ]:


# Use pd.to_datetime to convert the column from strings to DateTime objects
df['timeStamp'] = pd.to_datetime(df['timeStamp'])


# In[ ]:


# Create 3 new columns called Hour, Month, and Day of Weel
df['Hour'] = df['timeStamp'].apply(lambda time:time.hour)
df['Month'] = df['timeStamp'].apply(lambda time:time.month)
df['Day of Week'] = df['timeStamp'].apply(lambda time:time.dayofweek)


# In[ ]:


# Use .map() with dictionary to map the actual string names to the day of the week
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['Day of Week'] = df['Day of Week'].map(dmap)


# In[ ]:


# Use seaborn to create a countplot of the Day of Week column with the hue based on the Reason column
sns.countplot(x='Day of Week',data=df,hue='Reason',palette='viridis')
# Relocate the legend
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# In[ ]:


# Use seaborn to create a countplot of the Month column with the hue based on the Reason column
sns.countplot(x='Month',data=df,hue='Reason',palette='viridis')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# ## The Missing Value

# In[ ]:


## There are some missing values, like month 9, 10 and 11.
## We should fill in this information by plotting the information in another way, possibly a simple line plot that fills in the missing months.
## We meed to do some work with pandas.


# In[ ]:


## Create a groupby object called byMonth, where group the DataFrame by the month column.
## Use the count() method for aggregation.
## use the head() method to return DataFrame.
byMonth = df.groupby('Month').count()
byMonth.head()


# In[ ]:


byMonth['twp'].plot()


# In[ ]:


## Use seaborn's lmplot() to create a linear fit on the number of calls per month. Reset the index to a column.
sns.lmplot(x='Month',y='twp',data=byMonth.reset_index())


# In[ ]:


## Create a new column called 'Date' that contains the date from the timeStamp column.
## Apply along with the .date() method.
df['Date'] = df['timeStamp'].apply(lambda t:t.date())
df.groupby('Date').count()['twp'].plot()
plt.tight_layout()


# In[ ]:


## Recreate 3 seperate plot with each plot representing a Reason for the 911 call
df[df['Reason'] == 'Traffic'].groupby('Date').count()['twp'].plot()
plt.title('Traffic')
plt.tight_layout()


# In[ ]:


df[df['Reason'] == 'Fire'].groupby('Date').count()['twp'].plot()
plt.title('Fire')
plt.tight_layout()


# In[ ]:


df[df['Reason'] == 'EMS'].groupby('Date').count()['twp'].plot()
plt.title('EMS')
plt.tight_layout()


# ## Create Heatmaps

# In[ ]:


## Restructure the dataframe: columns become the Hours and Indexs become the Day of the week.
## Combine groupby with an unstack method to realize it.
dayHour = df.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()
dayHour.head()


# In[ ]:


## Create a HeatMap using this new DataFrame
plt.figure(figsize=(12,6))
sns.heatmap(dayHour,cmap='viridis')


# In[ ]:


## Create a clustermap using this DataFrame
sns.clustermap(dayHour,cmap='viridis')


# In[ ]:


## Change the Month as the column
dayMonth = df.groupby(by=['Day of Week','Month']).count()['Reason'].unstack()
dayMonth.head()


# In[ ]:


plt.figure(figsize=(12,6))
sns.heatmap(dayMonth,cmap='viridis')


# In[ ]:


sns.clustermap(dayMonth,cmap='viridis')

