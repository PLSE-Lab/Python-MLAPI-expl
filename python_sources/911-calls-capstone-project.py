#!/usr/bin/env python
# coding: utf-8

# # 911 Calls Capstone Project

# For this capstone project we will be analyzing some 911 call data from [Kaggle](https://www.kaggle.com/mchirico/montcoalert). The data contains the following fields:
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
# Just go along with this notebook and try to complete the instructions or answer the questions in bold using your Python and Data Science skills!

# ## Data and Setup

# ____
# ** Import numpy and pandas **

# In[ ]:


import pandas as pd
import numpy as np


# ** Import visualization libraries and set %matplotlib inline. **

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ** Read in the csv file as a dataframe called df **

# In[ ]:


df = pd.read_csv('../input/911.csv')


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


df['title'].nunique()


# ## Creating new features

# ** In the titles column there are "Reasons/Departments" specified before the title code. These are EMS, Fire, and Traffic. Use .apply() with a custom lambda expression to create a new column called "Reason" that contains this string value.** 
# 
# **For example, if the title column value is EMS: BACK PAINS/INJURY , the Reason column value would be EMS. **

# In[ ]:


df['Reason'] = df['title'].apply(lambda x: x.split(':')[0])
df.head(1)


# ** What is the most common Reason for a 911 call based off of this new column? **

# In[ ]:


df['Reason'].value_counts().head(1)


# ** Now use seaborn to create a countplot of 911 calls by Reason. **

# In[ ]:


sns.set_style('whitegrid')


# In[ ]:


sns.countplot(data=df, x='Reason');


# ___
# ** Now let us begin to focus on time information. What is the data type of the objects in the timeStamp column? **

# In[ ]:


# df['timeStamp'].dtypes

type(df['timeStamp'].iloc[0])


# ** You should have seen that these timestamps are still strings. Use [pd.to_datetime](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.to_datetime.html) to convert the column from strings to DateTime objects. **

# In[ ]:


df['timeStamp'] = pd.to_datetime(df['timeStamp'])


# ** You can now grab specific attributes from a Datetime object by calling them. For example:**
# 
#     time = df['timeStamp'].iloc[0]
#     time.hour
# 
# **You can use Jupyter's tab method to explore the various attributes you can call. Now that the timestamp column are actually DateTime objects, use .apply() to create 3 new columns called Hour, Month, and Day of Week. You will create these columns based off of the timeStamp column, reference the solutions if you get stuck on this step.**

# In[ ]:


df['Hour'] = df['timeStamp'].apply(lambda x: x.hour)
df['Month'] = df['timeStamp'].apply(lambda x: x.month)
df['Day of Week'] = df['timeStamp'].apply(lambda x: x.dayofweek)
df.head(1)


# ** Notice how the Day of Week is an integer 0-6. Use the .map() with this dictionary to map the actual string names to the day of the week: **
# 
#     dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

# In[ ]:


dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}


# In[ ]:


df['Day of Week'] = df['Day of Week'].map(dmap)
df.head(1)


# ** Now use seaborn to create a countplot of the Day of Week column with the hue based off of the Reason column. **

# In[ ]:


sns.countplot(data=df, x='Day of Week', hue='Reason', palette='rainbow')
plt.legend(loc='best', bbox_to_anchor=(1, 0.7));


# **Now do the same for Month:**

# In[ ]:



# mmap = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
# df['Month'] = df['Month'].map(mmap)
# months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']


# In[ ]:


sns.countplot(data=df, x='Month', hue='Reason', palette='viridis')
plt.legend(loc='best', bbox_to_anchor=(1, 0.7));


# ### Does not apply anymore. The dataset was updated
# **Did you notice something strange about the Plot?**
# 
# _____
# 
# ** You should have noticed it was missing some Months, let's see if we can maybe fill in this information by plotting the information in another way, possibly a simple line plot that fills in the missing months, in order to do this, we'll need to do some work with pandas... **

# ** Now create a gropuby object called byMonth, where you group the DataFrame by the month column and use the count() method for aggregation. Use the head() method on this returned DataFrame. **

# In[ ]:


byMonth = df.groupby('Month').count()
# byMonth.apply(months)
byMonth.head(2)


# ** Now create a simple plot off of the dataframe indicating the count of calls per month. **

# In[ ]:


byMonth['twp'].plot();


# ** Now see if you can use seaborn's lmplot() to create a linear fit on the number of calls per month. Keep in mind you may need to reset the index to a column. **

# In[ ]:


sns.lmplot(data=byMonth.reset_index(), x='Month', y='twp');


# **Create a new column called 'Date' that contains the date from the timeStamp column. You'll need to use apply along with the .date() method. ** 

# In[ ]:


df['Date'] = df['timeStamp'].apply(lambda x: x.date());
df.head(2)


# ** Now groupby this Date column with the count() aggregate and create a plot of counts of 911 calls.**

# In[ ]:


byDate = df.groupby('Date').count()
byDate['twp'].plot();
plt.tight_layout()


# ** Now recreate this plot but create 3 separate plots with each plot representing a Reason for the 911 call**

# In[ ]:


df['Reason'].unique()


# In[ ]:


plt.figure(figsize=(18,6));
df[df['Reason']=='EMS'].groupby('Date').count()['twp'].plot();
plt.legend(loc='best', bbox_to_anchor=(1, 1));


# In[ ]:


plt.figure(figsize=(18,6))
df[df['Reason']=='Fire'].groupby('Date').count()['twp'].plot();
plt.legend(loc='best', bbox_to_anchor=(1, 1));


# In[ ]:


plt.figure(figsize=(15,5))
df[df['Reason']=='Traffic'].groupby("Date").count()['twp'].plot()
plt.legend(loc='best', bbox_to_anchor=(1,1));


# ____
# ** Now let's move on to creating  heatmaps with seaborn and our data. We'll first need to restructure the dataframe so that the columns become the Hours and the Index becomes the Day of the Week. There are lots of ways to do this, but I would recommend trying to combine groupby with an [unstack](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.unstack.html) method. Reference the solutions if you get stuck on this!**

# In[ ]:


dayHour = df.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()
dayHour.head()


# ** Now create a HeatMap using this new DataFrame. **

# In[ ]:


sns.heatmap(dayHour);


# ** Now create a clustermap using this DataFrame. **

# In[ ]:


sns.clustermap(dayHour);


# ** Now repeat these same plots and operations, for a DataFrame that shows the Month as the column. **

# In[ ]:


dayMonth = df.groupby(by=['Day of Week','Month']).count()['Reason'].unstack()
dayMonth.head()


# In[ ]:


sns.heatmap(dayMonth);


# In[ ]:


sns.clustermap(dayMonth, cmap='viridis');


# In[ ]:


g = sns.PairGrid(dayMonth)
g.map_diag(plt.hist)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot);


# In[ ]:


sns.pairplot(dayMonth);


# **Continue exploring the Data however you see fit!**
# # Great Job!
