#!/usr/bin/env python
# coding: utf-8

# # Emergency - 911 Calls in Montgomery County, PA Visualization

# This  project uses 911 call data in Montgomery County, PA from a range of dates between 2015 and 2017 from [montocalert.org](https://montcoalert.org/gettingdata). Some more background information can be found at this [link](https://www.kaggle.com/mchirico/montcoalert). 
# 
# 
# For now, lets do some exploratory data analysis using python and its analysis libraries on the dataset and see what we can learn from this dataset.
# 

# In[ ]:



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

#Import our data analysis/visualization libraries along with any other useful libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# Now, lets assign our data to the variable df_calls.

# In[ ]:


#read in data
df_calls = pd.read_csv('../input/911.csv')


# Lets get some information on the dataset and perform some operations from there.

# In[ ]:


df_calls.info()
df_calls.head()


# Also checking for any null values that may affect any further data analysis. Depending on how many entries and the metric that is missing values, we may choose to omit them and delete the entries from the data.

# In[ ]:


df_calls.isnull().sum()


# In[ ]:


#visualize null values
sns.heatmap(df_calls.isnull(),cmap = 'plasma')


# There seems to be quite a few entries (over 35,000) missing from the zip column. For now, we'll ignore this and just do some analysis on the data that we currently have. We may come back to revisit this and see if we can use the information from the desc column to extract any information. Theres a few missing values for townships, but it's not enough to significantly impact the data.

# Lets look at the 10 zip codes and townships that had the most calls for the given data.

# ** Top 10 zipcodes and townships for 911 calls. **

# In[ ]:


#Visualization of number of calls relative to zip code

df_calls['zip'].value_counts().head(10).plot.bar(color = 'green')
plt.xlabel('Zip Codes',labelpad = 20)
plt.ylabel('Number of Calls')
plt.title('Zip Codes with Most Calls')


# In[ ]:


#Visualization of number of calls relative to township

df_calls['twp'].value_counts().head(10).plot.bar(color = 'teal')
plt.xlabel('Townships', labelpad = 20)
plt.ylabel('Number of Calls')
plt.title('Townships with Most Calls')


# Looking at the top values we can see that Lower Merion gets a majority of the calls relative to other townships. As far as zip codes go, we can see that 19401 and 19464 are similar in number of calls. Those areas are associated primarily with Norristown and Pottstown.

# We can do some more analysis on the type of calls that are made to each area by simplifying the data from the title column.

# In[ ]:


df_calls['title'].head(3)


# Looking at the data, it seems like there is a code with the type of call and a description for the emergency associated with the call. Lets divide those entries into two new columns : Type and Emergency Description

# In[ ]:


#New columns that extract call info from title column to use for further analysis
df_calls['Reason'] = df_calls['title'].apply(lambda x: x.split(':')[0])
df_calls['Emergency Description'] = df_calls['title'].apply(lambda x: x.split(':')[1])


# In[ ]:


# Function to remove hyphen at end of values for 'Emergency Description'

def hyph_del(x):
    if x[-1] == '-':
        return x[:-2]
    else:
        return x

df_calls['Emergency Description'] = df_calls['Emergency Description'].apply(hyph_del)


# In[ ]:


#gives count of reason type
df_calls['Reason'].value_counts()


# In[ ]:


#Orders top 30 description calls
df_calls['Emergency Description'].value_counts().head(30)


# ** Plots showing distribution of Reason and Emergency Description **

# In[ ]:


sns.countplot('Reason', data=df_calls, palette='pastel')


# In[ ]:


df_calls['Reason'].value_counts()


# In[ ]:


df_calls['Emergency Description'].value_counts().head(20).plot.bar(color = 'navy')
plt.xlabel('Emergency Description',labelpad = 20)
plt.ylabel('Number of Calls')
plt.title('Top 20 Emergency Description Calls')


# Things to note:
# 
# 
# EMS calls are more common than Traffic calls, making up about 50% of the total call, whereas Traffic calls make up only about 35%.
# 
# Vehicle accidents make up an overwhelming majority of the calls data individually. However, this does include EMS calls that have a vechicle accident description as well, which could explain why this category is so much more common than the others in the graph.
# 

# Now that we've done a bit of analysis on the nature of the calls, lets look at the timestamp data and compare the nature of the calls with when calls were placed to get some more insights. 
# 
# First off, lets seperate the timestamp elements into individual columns and then convert those values from strings to datetime objects.

# In[ ]:


#Converts string in timeStamp to datetime object
df_calls['timeStamp'] = pd.to_datetime(df_calls['timeStamp'])


# In[ ]:


df_calls['Hour'] = df_calls['timeStamp'].apply(lambda time: time.hour)
df_calls['Month'] = df_calls['timeStamp'].apply(lambda time: time.month)
df_calls['Day of Week'] = df_calls['timeStamp'].apply(lambda time: time.dayofweek)


# Looking at the head and info of the new columns, we can see that the day of week is an integer and not a string. We can do some simple operations to fix this and make our analysis a little bit neater.

# In[ ]:


#Change Day of Week column from integer to string by mapping values to string
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

df_calls['Day of Week'] = df_calls['Day of Week'].map(dmap)


# In[ ]:


mmap = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',
       8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}

df_calls['Month'] = df_calls['Month'].map(mmap)


# ** Countplot of calls based on day of week and month, sorted by type of call. **

# In[ ]:


order = ['Sun','Mon', 'Tue', 'Wed','Thu','Fri','Sat']

plt.figure(figsize=(10,5))
sns.countplot('Day of Week', data = df_calls, hue = 'Reason', palette='pastel',order = order )
plt.legend(loc= 'upper right', bbox_to_anchor=(1.15,.8))


# In[ ]:


m_order = ['Jan','Feb', 'Mar', 'Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

plt.figure(figsize=(10,5))
sns.countplot('Month', data = df_calls, hue = 'Reason', palette='Set2',order = m_order)
plt.legend(loc= 'upper right', bbox_to_anchor=(1.15,.8))


# Some things to note based on the above visualizations:
# 
# * Fire and EMS calls tend to stay constant throughout the months and days with some slight increases in December.
# * As expected, Traffic calls tend to be higher on the weekdays than the weekends.
# * Increase of Traffic and EMS calls in the winter months may be attributed to more hazardous driving conditions.

# Now lets work on vizualizing the data by looking at the complete dates.

# In[ ]:


# New column which will use entire date as opposed to seperating the information like before
df_calls['Date'] = df_calls['timeStamp'].apply(lambda time:time.date())


# In[ ]:


plt.figure(figsize=(15,6))
plt.title('Traffic')
plt.ylabel('Number of Calls')
df_calls[df_calls['Reason'] == 'Traffic'].groupby('Date').count()['twp'].plot()
plt.tight_layout


# In[ ]:


plt.figure(figsize=(15,6))
plt.title('Fire')
plt.ylabel('Number of Calls')
df_calls[df_calls['Reason'] == 'Fire'].groupby('Date').count()['lat'].plot(color='green')
plt.tight_layout


# In[ ]:


plt.figure(figsize=(15,6))
plt.title('EMS')
df_calls[df_calls['Reason'] == 'EMS'].groupby('Date').count()['lat'].plot(color='maroon')
plt.tight_layout


# Things to note based on above graphs:
# 
# * Some spikes in calls during different times of the months, where winter months see more Traffic and EMS calls and summer months have more Fire calls.
# * All three graphs show a decline in calls during the same time in 2017, perhaps due to some missing data or other factors.
# * Calls on average tend to float around 150, 60, and 200 for Traffic, Fire, and EMS respectively.

# We can also do some interesting visualizations by looking at heatmaps that compare the relationships between the dates and reasons for the calls.

# 
# ** Heatmap to show relationship of calls between Hour and Day of Week for the data. **

# In[ ]:


DoW =['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri','Sat']

df_heatHour = df_calls.groupby(by = ['Day of Week', 'Hour']).count()['Reason'].unstack()
df_heatHour.index = pd.CategoricalIndex(df_heatHour.index, categories=DoW)
df_heatHour.sort_index(level=0, inplace=True)
df_heatHour.head()


# In[ ]:


plt.figure(figsize=(10,7))
sns.heatmap(df_heatHour, cmap='viridis')
plt.title('Relationship of calls between Hour and DoW')


# * More calls tend to be made during the normal hours that people are awake.
# * Number of calls increase between 3pm-5pm from Monday through Friday (rush hour).
# * Very slight increase in number of calls made after 8pm Friday and Saturday.

# ** Heatmap to show realationship of calls between Month and Day of Week for the data. **

# In[ ]:


# New column for month as an integer
df_calls['Month_Num'] = df_calls['timeStamp'].apply(lambda time: time.month)

df_heatMonth = df_calls.groupby(by = ['Day of Week', 'Month_Num']).count()['Reason'].unstack()
df_heatMonth.index = pd.CategoricalIndex(df_heatMonth.index,categories = DoW)
df_heatMonth.sort_index(level=0, inplace=True)
df_heatMonth.rename(columns = mmap,inplace=True)
df_heatMonth.head()


# In[ ]:


plt.figure(figsize=(10,5))
sns.heatmap(df_heatMonth, cmap='viridis')
plt.xlabel('Month')
plt.title('Relationship of calls between Month and DoW')


# * Large number of calls made on Fridays in December
# * Saturdays and Sundays of any month seem to have the least amount of calls

# There are more inishgts to be gathered from this dataset and I'll explore these at a future date. Thank you for taking the time to read through all of this, if you have any suggestions feedback is always appreciated :) 
