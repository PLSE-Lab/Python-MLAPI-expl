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

# In[1]:


import numpy as np
import pandas as pd


# ** Import visualization libraries and set %matplotlib inline. **

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


# In[ ]:





# ** Read in the csv file as a dataframe called df **

# In[5]:


df= pd.read_csv('../input/911.csv')


# ** Check the info() of the df **

# In[6]:


df.info()


# In[ ]:





# ** Check the head of df **

# In[7]:


df.head()


# In[ ]:





# ## Basic Questions

# ** What are the top 5 zipcodes for 911 calls? **

# In[8]:


df['zip'].value_counts().head(5)


# In[134]:





# ** What are the top 5 townships (twp) for 911 calls? **

# In[9]:


df['twp'].value_counts().head(5)


# In[135]:





# ** Take a look at the 'title' column, how many unique title codes are there? **

# In[10]:


df['title'].nunique()


# In[136]:





# ## Creating new features

# ** In the titles column there are "Reasons/Departments" specified before the title code. These are EMS, Fire, and Traffic. Use .apply() with a custom lambda expression to create a new column called "Reason" that contains this string value.** 
# 
# **For example, if the title column value is EMS: BACK PAINS/INJURY , the Reason column value would be EMS. **

# In[11]:


df['Reason']=df['title'].apply(lambda title: title.split(':')[0])


# In[12]:


df['Reason']


# In[137]:





# ** What is the most common Reason for a 911 call based off of this new column? **

# In[13]:


df['Reason'].value_counts()


# In[138]:





# ** Now use seaborn to create a countplot of 911 calls by Reason. **

# In[14]:


sns.countplot(x='Reason', data=df)


# ___
# ** Now let us begin to focus on time information. What is the data type of the objects in the timeStamp column? **

# In[15]:


type(df['timeStamp'].iloc[2])


# In[140]:





# ** You should have seen that these timestamps are still strings. Use [pd.to_datetime](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.to_datetime.html) to convert the column from strings to DateTime objects. **

# In[16]:


df['timeStamp']=pd.to_datetime(df['timeStamp'])


# 

# In[17]:


df['Month']= df['timeStamp'].apply(lambda time: time.month)
df['Hour']= df['timeStamp'].apply(lambda time: time.hour)
df['day']= df['timeStamp'].apply(lambda time: time.dayofweek)


# In[18]:


df['day'].unique()


# In[142]:





# ** Notice how the Day of Week is an integer 0-6. Use the .map() with this dictionary to map the actual string names to the day of the week: **
# 
#     dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

# In[19]:


dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}


# In[20]:


df['day']=df['day'].map(dmap)


# In[21]:


df['day']


# In[143]:





# In[144]:





# ** Now use seaborn to create a countplot of the Day of Week column with the hue based off of the Reason column. **

# In[22]:


df.info()


# In[23]:


sns.countplot(x='day', data=df, hue='Reason', palette='viridis')
plt.legend(bbox_to_anchor=(1.0,1.0))


# **Now do the same for Month:**

# In[24]:


sns.countplot(x='Month', data=df, hue='Reason', palette='viridis')
plt.legend(bbox_to_anchor=(1.2,1.0))


# it was missing some Months, let's see if we can maybe fill in this information by plotting the information in another way, possibly a simple line plot that fills in the missing months, in order to do this, we'll need to do some work with pandas... **

# ** Now create a gropuby object called byMonth, where you group the DataFrame by the month column and use the count() method for aggregation. Use the head() method on this returned DataFrame. **

# In[26]:


bymonth= df.groupby('Month').count()
bymonth.head()


# In[169]:





# ** Now create a simple plot off of the dataframe indicating the count of calls per month. **

# In[27]:


bymonth['twp'].plot()


# ** Now see if you can use seaborn's lmplot() to create a linear fit on the number of calls per month. Keep in mind you may need to reset the index to a column. **

# In[28]:


sns.lmplot('Month', 'twp', data=bymonth.reset_index())


# **Create a new column called 'Date' that contains the date from the timeStamp column. You'll need to use apply along with the .date() method. ** 

# In[29]:


df['Date']=df['timeStamp'].apply(lambda time: time.date())
df['Date']


# In[193]:





# ** Now groupby this Date column with the count() aggregate and create a plot of counts of 911 calls.**

# In[30]:


df.groupby('Date').count()['twp'].plot()
plt.tight_layout()


# ** Now recreate this plot but create 3 separate plots with each plot representing a Reason for the 911 call**

# In[31]:


df[df['Reason']=='Traffic'].groupby('Date').count()['twp'].plot()
plt.title('Traffic')
plt.tight_layout()


# In[102]:


df[df['Reason']=='Fire'].groupby('Date').count()['twp'].plot()
plt.title('Fire')
plt.tight_layout()


# In[103]:


df[df['Reason']=='EMS'].groupby('Date').count()['twp'].plot()
plt.title('EMS')
plt.tight_layout()


# ____
# ** Now let's move on to creating  heatmaps with seaborn and our data. We'll first need to restructure the dataframe so that the columns become the Hours and the Index becomes the Day of the Week. 

# In[32]:


df.info()


# In[34]:


dayHour = df.groupby(by=['day','Hour']).count()['Reason'].unstack()
dayHour.head()


# In[203]:





# ** Now create a HeatMap using this new DataFrame. **

# In[35]:


plt.figure(figsize=(8,6))
sns.heatmap(dayHour, cmap='viridis')
plt.tight_layout()


# ** Now create a clustermap using this DataFrame. **

# In[36]:


sns.clustermap(dayHour, figsize=(6,4), cmap='viridis')
plt.tight_layout()


# ** Now repeat these same plots and operations, for a DataFrame that shows the Month as the column. **

# In[147]:


datamonth = df.groupby(by=('day', 'Month')).count()['Reason'].unstack()
datamonth.head()


# In[150]:


sns.heatmap(datamonth, cmap='viridis')
plt.figure(figsize=(6,8))


# In[152]:


sns.clustermap(datamonth, cmap='viridis')

