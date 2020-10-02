#!/usr/bin/env python
# coding: utf-8

# # Visualizing 911 Calls data
# 

# Welcome in this kernel of visualizing 911 Calls data. We will use tools such as seaborn to perform the activity. But first, we will go through the data to understand it effectively.

# *Let's get started by importing relevant libraries and importing data from kaggle*

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


DataFrame=pd.read_csv('/kaggle/input/montcoalert/911.csv')


# Let's have basic knoudlege about the data

# In[ ]:


DataFrame.info()


# In[ ]:


DataFrame.head(3) 
#First three rows of data


# In[ ]:


DataFrame.tail(2)
#Last two rows of data


# # Getting to Know Basic Questions

# As always, let's ask ourselves some questions to understand our data

# *1. What are the top 5 zipcodes for 911 Calls?*

# In[ ]:


DataFrame['zip'].value_counts().head()


# 2. *What are the top 5 townships(twp) for 911 calls?*

# In[ ]:


DataFrame['twp'].value_counts().head(5)


# *3. How many Unique title codes are in data?*

# In[ ]:


DataFrame['title'].nunique()


# # Creating new features

# In the title column, there are Reasons/Departments specified before code. They are EMS, Fire, and Traffic. We are going to use apply with a custom lambda Expression to create new column called Reason that contains this string value. 
# 
# *For example, if the title column value is EMS: BACK PAINS/INJURY , the Reason column value would be EMS. *

# In[ ]:


DataFrame['Reason']=DataFrame['title'].apply(lambda title:title.split(':')[0])
##title.split(':')[0] divide each string by : and takes the first word


# *What is the most common Reason for a 911 call based off of this new column?*

# In[ ]:


DataFrame['Reason'].value_counts()


# Let's now use Seaborn to create countplot of 911 calls by Reason

# In[ ]:


sns.countplot(x='Reason',data=DataFrame, palette='viridis')


# ** Now let us begin to focus on time information. What is the data type of the objects in the timeStamp column? **

# In[ ]:


type(DataFrame['timeStamp'].iloc[0])


# Since they are strings, let us use pd.to_datetime to convert the column from strings to DateTime objects. 

# In[ ]:


DataFrame['timeStamp'] = pd.to_datetime(DataFrame['timeStamp'])


# In[ ]:


DataFrame['Hour']=DataFrame['timeStamp'].apply(lambda time:time.hour)
DataFrame['Month']=DataFrame['timeStamp'].apply(lambda time:time.month)
DataFrame['Day of Week']=DataFrame['timeStamp'].apply(lambda time:time.dayofweek)


# Notice how the Day of Week is an integer 0-6. Use the .map() with this dictionary to map the actual string names to the day of the week:

# In[ ]:


dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
DataFrame['Day of Week'] = DataFrame['Day of Week'].map(dmap)


# Now use seaborn to create a countplot of the Day of Week column with the hue based off of the Reason column.

# In[ ]:


sns.countplot(x='Day of Week',data=DataFrame,hue='Reason',palette='viridis')

# To relocate the legend
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# *You can see that there are low traffic calls on Sunday due to that people are home in weekend*

# Let's do the same for Month

# In[ ]:


sns.countplot(x='Month',data=DataFrame,hue='Reason',palette='viridis')

# To relocate the legend
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# Let's use seaborn's lmplot() to create a linear fit on the number of calls per month.

# Now let us create a new column called 'Date' that contains the date from the timeStamp column.

# In[ ]:


DataFrame['Date']=DataFrame['timeStamp'].apply(lambda t: t.date())


# Plotting 911 calls by Date

# In[ ]:


plt.figure(figsize=(20,12))
DataFrame.groupby('Date').count()['twp'].plot()
plt.tight_layout()


#  Now let us create this plot but create 3 separate plots with each plot representing a Reason for the 911 call

# In[ ]:



plt.figure(figsize=(20,12))
DataFrame[DataFrame['Reason']=='Traffic'].groupby('Date').count()['twp'].plot()
plt.title('Traffic')
plt.tight_layout()


# In[ ]:


plt.figure(figsize=(20,12))
DataFrame[DataFrame['Reason']=='Fire'].groupby('Date').count()['twp'].plot()
plt.title('Fire')
plt.tight_layout()


# In[ ]:


plt.figure(figsize=(20,12))
DataFrame[DataFrame['Reason']=='EMS'].groupby('Date').count()['twp'].plot()
plt.title('EMS')
plt.tight_layout()


# Now let's move on to creating heatmaps with seaborn and our data. We'll first need to restructure the dataframe so that the columns become the Hours and the Index becomes the Day of the Week. 

# In[ ]:


dayHour = DataFrame.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()
dayHour.head()


# In[ ]:


plt.figure(figsize=(12,6))
sns.heatmap(dayHour,cmap='viridis')


# Let's now create cluster map

# In[ ]:


sns.clustermap(dayHour,cmap='viridis',
    col_colors=None,
    mask=None)


# That's all for 911 calls. As ownership, I used some ideas from from @Pierrana Data Course by Jose Portila, [Python for Data Science and Machine Learning](https://www.udemy.com/course/python-for-data-science-and-machine-learning-bootcamp/)

# In[ ]:




