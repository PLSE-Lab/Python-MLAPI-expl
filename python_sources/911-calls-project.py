#!/usr/bin/env python
# coding: utf-8

# # 911 Calls Capstone Project

#  The 911 call data contains the following fields:
# 
# * lat : String variable, Latitude
# * lng: String variable, Longitude
# * desc: String variable, Description of the Emergency Call
# * zip: String variable, Zipcode
# * title: String variable, Title
# * timeStamp: String variable, YYYY-MM-DD HH:MM:SS
# * twp: String variable, Township
# * addr: String variable, Address
# * e: String variable, Dummy variable (always 1)
# 

# ## Data and Setup

# ____
# ** Import numpy and pandas **

# In[ ]:


import numpy as np
import pandas as pd


# ** Import visualization libraries and set %matplotlib inline. **

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# ** Read in the csv file as a dataframe called df **

# In[ ]:


df = pd.read_csv("../input/911.csv")


# ** Check the info() of the df **

# In[ ]:


df.info()


# ** Check the head of df **

# In[ ]:


df.head()


# ## Basic Questions

# ** What are the top 5 zipcodes for 911 calls? **

# In[ ]:


df['zip'].value_counts().head()


# ** What are the top 5 townships (twp) for 911 calls? **

# In[ ]:


df['twp'].value_counts().head()


# ** Take a look at the 'title' column, how many unique title codes are there? **

# In[ ]:


df["title"].nunique()


# ## Creating new features

# ** In the titles column there are "Reasons/Departments" specified before the title code. These are EMS, Fire, and Traffic. Use .apply() with a custom lambda expression to create a new column called "Reason" that contains this string value.** 
# 
# **For example, if the title column value is EMS: BACK PAINS/INJURY , the Reason column value would be EMS. **

# In[ ]:


df["Reason"]=df["title"].apply(lambda x: x.split(":")[0])


# In[ ]:


df.head()


# ** What is the most common Reason for a 911 call based off of this new column? **

# In[ ]:


df['Reason'].value_counts()


# ** Now use seaborn to create a countplot of 911 calls by Reason. **

# In[ ]:


sns.countplot(x='Reason',data=df)


# ___
# ** Now let us begin to focus on time information. What is the data type of the objects in the timeStamp column? **

# In[ ]:


type(df['timeStamp'][0])


# ** You should have seen that these timestamps are still strings. Use [pd.to_datetime](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.to_datetime.html) to convert the column from strings to DateTime objects. **
# 

# In[ ]:


df['timeStamp']=pd.to_datetime(df['timeStamp'])


# 
# ** You can now grab specific attributes from a Datetime object by calling them. For example:**
# 
#     time = df['timeStamp'].iloc[0]
#     time.hour
# 
# **You can use Jupyter's tab method to explore the various attributes you can call. Now that the timestamp column are actually DateTime objects, use .apply() to create 3 new columns called Hour, Month, and Day of Week. You will create these columns based off of the timeStamp column, reference the solutions if you get stuck on this step.**

# In[ ]:


df['Hour']=df['timeStamp'].apply(lambda x: x.hour)


# In[ ]:


df['Month']=df['timeStamp'].apply(lambda x: x.month)


# In[ ]:


df['Day of Week']=df['timeStamp'].apply(lambda x: x.dayofweek)


# In[ ]:


df['Year']=df['timeStamp'].apply(lambda x: x.year)


# In[ ]:


df.tail()


# ** Notice how the Day of Week is an integer 0-6. Use the .map() with this dictionary to map the actual string names to the day of the week: *
# 
#     dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

# In[ ]:


dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}


# In[ ]:


df['Day of Week']=df['Day of Week'].map(dmap)


# In[ ]:


df['Day of Week'].head()


# 

# ** Now use seaborn to create a countplot of the Day of Week column with the hue based off of the Reason column. **

# In[ ]:


sns.countplot(x='Day of Week',hue='Reason', data=df)


# **Now do the same for Month:**

# In[ ]:


sns.countplot(x='Month',hue='Reason', data=df[df['Year']==2017])


# In[ ]:


sns.countplot(x='Year',hue='Reason', data=df)


# ____
# ** Now let's move on to creating  heatmaps with seaborn and our data. We'll first need to restructure the dataframe so that the columns become the Hours and the Index becomes the Day of the Week. There are lots of ways to do this, but I would recommend trying to combine groupby with an [unstack](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.unstack.html) method. Reference the solutions if you get stuck on this!**

# In[ ]:


dayHour = df.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()


# ** Now create a HeatMap using this new DataFrame. **

# In[ ]:


plt.figure(figsize=(12,6))
sns.heatmap(dayHour,cmap='viridis')


# ** Now create a clustermap using this DataFrame. **

# In[ ]:


sns.clustermap(dayHour, cmap='viridis')


# ** Now repeat these same plots and operations, for a DataFrame that shows the Month as the column. **

# In[ ]:


dayMonth = df.groupby(by=['Day of Week','Month']).count()['Reason'].unstack()
dayMonth.head()


# In[ ]:


plt.figure(figsize=(12,6))
sns.heatmap(dayMonth,cmap='viridis')


# In[ ]:


sns.clustermap(dayMonth,cmap='viridis')


# 
