#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#import os
#print(os.listdir("../input"))


# In[ ]:


data = pd.read_csv('../input/911.csv')
data.head()


# In[ ]:


data.info()


# What are the top 5 zipcodes for 911 calls?

# In[ ]:


data['zip'].value_counts().head()


# What are the top 5 townships (twp) for 911 calls?

# In[ ]:


data['twp'].value_counts().head()


# Take a look at the 'title' column, how many unique title codes are there?

# In[ ]:


data['title'].nunique()


# # Creating new features
# In the titles column there are "Reasons/Departments" specified before the title code. These are EMS, Fire, and Traffic. Use .apply() with a custom lambda expression to create a new column called "Reason" that contains this string value.
# 
# For example, if the title column value is EMS: BACK PAINS/INJURY , the Reason column value would be EMS.

# In[ ]:


data['Reason'] = data['title'].apply(lambda title: title.split(':')[0])
data['Reason'].head()


# What is the most common Reason for a 911 call based off of this new column?

# In[ ]:


data['Reason'].value_counts()


# Now use seaborn to create a countplot of 911 calls by Reason.

# In[ ]:


sns.countplot(x='Reason',data=data)


# What is the data type of the objects in the timeStamp column?

# In[ ]:


type(data['timeStamp'].iloc[0])


# Use pd.to_datetime to convert the column from strings to DateTime objects.

# In[ ]:


data['timeStamp'] = pd.to_datetime(data['timeStamp'])
type(data['timeStamp'].iloc[0])


# 
# We can grab specific attributes from a Datetime object like -
# 
# time = data['timeStamp'].iloc[0]
# time.hour
# Use .apply() Method, to create 3 new columns called Hour, Month, and Day of Week. You will create these columns based off the timeStamp column.

# In[ ]:


data['Hour'] = data['timeStamp'].apply(lambda time: time.hour)
data['Month'] = data['timeStamp'].apply(lambda time: time.month)
data['Day of Week'] = data['timeStamp'].apply(lambda time: time.dayofweek)
data.head()


# Day of Week is an integer 0-6. Use the .map() with this dictionary to map the actual string names to the day of the week:
# 
# dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

# In[ ]:


dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
data['Day of Week'] = data['Day of Week'].map(dmap)
data.head()


# Use seaborn to create a countplot of the Day of Week column with the hue based off of the Reason column.

# In[ ]:


sns.countplot(x='Day of Week', data =data, hue='Reason')
# Relocation of the legends outside
plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)


# Now do the same for Month:

# In[ ]:


sns.countplot(x='Month', data =data, hue='Reason')
# Relocation of the legends outside
plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)


# Create a new column called 'Date' that contains the date from the timeStamp column. You'll need to use apply along with the .date() method.

# In[ ]:


data['Date'] = data['timeStamp'].apply(lambda t: t.date())
data.head()


# Now groupby this Date column with the count() aggregate and create a plot of counts of 911 calls.

# In[ ]:


data.groupby('Date').count().head()


# In[ ]:


data.groupby('Date').count()['lat'].plot()
plt.tight_layout()


# Now recreate this plot but create 3 separate plots with each plot representing a Reason for the 911 call

# In[ ]:


data[data['Reason']=='Traffic'].groupby('Date').count()['lat'].plot()
plt.title("Traffic")
plt.tight_layout()


# In[ ]:


data[data['Reason']=='Fire'].groupby('Date').count()['lat'].plot()
plt.title("Fire")
plt.tight_layout()


# In[ ]:


data[data['Reason']=='EMS'].groupby('Date').count()['lat'].plot()
plt.title("EMS")
plt.tight_layout()


# Let's try to create heatmaps with seaborn and our data. But, first we will need to restructure the dataframe so that the columns become the Hours and the Index becomes the Day of the Week. There are lots of ways to do this, but let's try to combine groupby with an unstack method.

# In[ ]:


dayHour = data.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()
dayHour.head(2)


# Now create a HeatMap using this new DataFrame.

# In[ ]:


sns.heatmap(dayHour,cmap='coolwarm')


# Now create a clustermap using this DataFrame.

# In[ ]:


sns.clustermap(dayHour,cmap='coolwarm')


# Now repeat these same plots and operations, for a DataFrame that shows the Month as the column.

# In[ ]:


dayMonth = data.groupby(by=['Day of Week','Month']).count()['Reason'].unstack()
dayMonth.head()


# In[ ]:


sns.heatmap(dayMonth,cmap='coolwarm')


# In[ ]:


sns.clustermap(dayMonth,cmap='coolwarm')


# # Thank You 

# In[ ]:




