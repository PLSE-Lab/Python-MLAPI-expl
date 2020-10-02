#!/usr/bin/env python
# coding: utf-8

# **Importing the necessry libraries for processing**

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')
import os
print(os.listdir("../input"))


# **Emergency - 911 Calls**
# *Montgomery County, PA*
# lets do data for each component 
# 1 > load csv file into panda ,  pandas handle datatype char,int in its dataframe , and more powerful than numpy
# this 911.csv contain dtypes: float64(3), int64(1), object(5)

# In[ ]:


df=pd.read_csv("../input/911.csv")


# **Brief Insights**
# 2> check content and decsription 
# *we will be analyzing some 911 call data *
# head() is giving insights of first five rows
# df.describe()
# this interprets ststistics of data like mean , std deviation,etc 
# * the last column is not needed 'e' so we deleting it.

# In[ ]:


print(df.head(5))
print(df.describe())
del df["e"]


# **3 Creating New Features by  segregating details column [timestamp,title]:**
# 
# **3.1> timestamp column segregation into date /time / day**
#  datatype of timestamp column is string
#  type(df['timeStamp'].iloc[0])
#  -----> str
#  i used** pd.to_datetime ** to convert str dt

# In[ ]:


df['timeStamp'] = pd.to_datetime(df['timeStamp'])
df['Hour'] = df['timeStamp'].apply(lambda t: t.hour)
df['Month'] = df['timeStamp'].apply(lambda t: t.month)
df['Day of Week'] = df['timeStamp'].apply(lambda t: t.dayofweek)
df['Year'] = df['timeStamp'].apply(lambda t: t.year)
df['Date'] = df['timeStamp'].apply(lambda t: t.date)


# to add day we have convert df['Day of Week']  this column into meaningful way using maping
# lets create a dictionary
# d = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
# 

# In[ ]:


d = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['Day of Week'] = df['Day of Week'].map(d)


# **3.2> Create Reason column from title column by splitting at [ : ]**
# In the titles column there are "Reasons/Departments" specified before the title code. These are EMS, Fire, and Traffic. Use .apply() with a custom lambda expression to create a new column called "Reason" that contains this string value.

# In[ ]:


df['Reason'] = df['title'].apply(lambda title: title.split(':')[0])


# **3.3>creating column based on timestamp for DAY/NIGHT**

# In[ ]:


df["day/night"] = df["timeStamp"].apply(lambda x : "night" if int(x.strftime("%H")) > 19 else "day")


# > DATA VISUALISATION ON PROCESSED DATA
# 
# see following graphs , u will get lot of insights about data

# In[ ]:


# Plot for day/night
sns.countplot(x='day/night',data=df)


# In[ ]:


# Plot for Category of reasons:
sns.countplot(x='Reason',data=df)


# In[ ]:


# Calls report Daily: 
sns.countplot(x='Day of Week',data=df,hue='Reason')


# In[ ]:


# Plot for calls recieved monthly combined of all years:
sns.countplot(x='Month',data=df)


# In[ ]:


# Plot for calls recieved yearly:
sns.countplot(x= "Year", data= df)


# In[ ]:


# simple plot of the dataframe indicating the count of calls per month.
d1=df.groupby("Year").count()
d1['Month'].plot()


# >
# 
# HEATMAP 
# 
# 

#  need to restructure the dataframe so that the columns become the Hours and the Index becomes the Day of the Week. 
#      There are lot of method for pivote table search on[ Medium](http://medium.com/) for more details

# In[ ]:


dayHour = df.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()
dayHour.head()


# In[ ]:


plt.figure(figsize=(12,6))
sns.heatmap(dayHour,cmap='viridis')
sns.clustermap(dayHour,cmap='viridis')


# **Day of Week   vs   Month**

# In[ ]:


dayMonth = df.groupby(by=['Day of Week','Month']).count()['Reason'].unstack()
dayMonth.head()
plt.figure(figsize=(12,6))
sns.heatmap(dayMonth,cmap='viridis')


# In[ ]:


sns.clustermap(dayMonth,cmap='viridis')


# **if you have any doubts comment here ,thanks**
# 
