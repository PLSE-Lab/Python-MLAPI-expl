#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Data and setup
# ___
# ** Importing libraries **

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# ** Read in the csv file as a dataframe called df **

# In[ ]:


df = pd.read_csv('/kaggle/input/montcoalert/911.csv')


# ** Check the info() of the df **

# In[ ]:


df.info()


# ** Checking the head of df **

# In[ ]:


df.head()


# ** What are the top 5 zipcodes for 911 calls? **

# In[ ]:


df['zip'].value_counts().head(5)


# ** What are the top 5 townships (twp) for 911 calls? **

# In[ ]:


df['twp'].value_counts().head(5)


# ** Looking at the 'title' column, how many unique title codes are there? **

# In[ ]:


df['title'].nunique()


# ## Creating new features
# 

# In the titles column there are "Reasons/Departments" specified before the title code. These are EMS, Fire, and Traffic. Use .apply() with a custom lambda expression to create a new column called "Reason" that contains this string value.
# 
# **For example, if the title column value is EMS: BACK PAINS/INJURY , the Reason column value would be EMS. **

# In[ ]:


df['Reason'] = df['title'].apply(lambda title: title.split(':')[0])


# ** What is the most common Reason for a 911 call based off of this new column? **

# In[ ]:


df['Reason'].value_counts()


# ** Now use seaborn to create a countplot of 911 calls by Reason. **

# In[ ]:


sns.countplot(x='Reason',data=df,palette='viridis')


# ___
# ** Now let us begin to focus on time information. What is the data type of the objects in the timeStamp column? **

# In[ ]:


type(df['timeStamp'].iloc[0])


# ** These timestamps are still strings. Use [pd.to_datetime](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.to_datetime.html) to convert the column from strings to DateTime objects. **

# In[ ]:


df['timeStamp'] = pd.to_datetime(df['timeStamp'])


# ** I can now grab specific attributes from a Datetime object by calling them. For example:**
# 
#     time = df['timeStamp'].iloc[0]
#     time.hour
# 
# ** Now that the timestamp column are actually DateTime objects, use .apply() to create 3 new columns called Hour, Month, and Day of Week. **

# In[ ]:


df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)
df['Month'] = df['timeStamp'].apply(lambda time: time.month)
df['Day of Week'] = df['timeStamp'].apply(lambda time: time.dayofweek)


# In[ ]:


dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['Day of Week'] = df['Day of Week'].map(dmap)


# ** Now use seaborn to create a countplot of the Day of Week column with the hue based off of the Reason column. **

# In[ ]:


sns.countplot(x='Day of Week',data=df,hue='Reason',palette='viridis')

# To relocate the legend
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# ** Now do the same for Month:**

# In[ ]:


sns.countplot(x='Month',data=df,hue='Reason',palette='viridis')

# To relocate the legend
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# In[ ]:


byMonth = df.groupby('Month').count()
byMonth.head()


# ** Now create a simple plot off of the dataframe indicating the count of calls per month. **

# In[ ]:


byMonth['twp'].plot()


# **  Use seaborn's lmplot() to create a linear fit on the number of calls per month. Keep in mind you may need to reset the index to a column. **

# In[ ]:


sns.lmplot(x='Month',y='twp',data=byMonth.reset_index())


# **Create a new column called 'Date' that contains the date from the timeStamp column. Need to use apply along with the .date() method. ** 

# In[ ]:


df['Date']=df['timeStamp'].apply(lambda t: t.date())


# ** Now groupby this Date column with the count() aggregate and create a plot of counts of 911 calls.**

# In[ ]:


df.groupby('Date').count()['twp'].plot()
plt.tight_layout()


# ** Now recreate this plot but create 3 separate plots with each plot representing a Reason for the 911 call**

# In[ ]:


df[df['Reason']=='Traffic'].groupby('Date').count()['twp'].plot()
plt.title('Traffic')
plt.tight_layout()


# In[ ]:


df[df['Reason']=='Fire'].groupby('Date').count()['twp'].plot()
plt.title('Fire')
plt.tight_layout()


# In[ ]:


df[df['Reason']=='EMS'].groupby('Date').count()['twp'].plot()
plt.title('EMS')
plt.tight_layout()


# ____
# ** Now let's move on to creating  heatmaps with seaborn and our data. We'll first need to restructure the dataframe so that the columns become the Hours and the Index becomes the Day of the Week. There are lots of ways to do this, but I would recommend trying to combine groupby with an [unstack](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.unstack.html) method. **

# In[ ]:


dayHour = df.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()
dayHour.head()


# ** Now create a HeatMap using this new DataFrame. **

# In[ ]:


plt.figure(figsize=(12,6))
sns.heatmap(dayHour,cmap='viridis')


# ** Now create a clustermap using this DataFrame. **

# In[ ]:


sns.clustermap(dayHour,cmap='viridis')


# ** Now repeat these same plots and operations, for a DataFrame that shows the Month as the column. **

# In[ ]:


dayMonth = df.groupby(by=['Day of Week','Month']).count()['Reason'].unstack()
dayMonth.head()


# In[ ]:


plt.figure(figsize=(12,6))
sns.heatmap(dayMonth,cmap='viridis')


# In[ ]:


sns.clustermap(dayMonth,cmap='viridis')

