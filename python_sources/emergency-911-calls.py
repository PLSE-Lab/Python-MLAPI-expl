#!/usr/bin/env python
# coding: utf-8

#  The data contains the following fields:
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
# 

# In[ ]:


# Import numpy and pandas

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Import visualization libraries and set %matplotlib inline. 

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Read in the csv file as a dataframe called df 

# In[ ]:


df = pd.read_csv('../input/montcoalert/911.csv')


# # Check the info() of the df

# In[ ]:


df.info()


# # Check the head of df

# In[ ]:


df.head()


# In[ ]:


df.describe()


# ## Basic Questions

# What are the top 5 zipcodes for 911 calls?

# In[ ]:


df['zip'].value_counts().head(5)


# What are the top 5 townships (twp) for 911 calls?

# In[ ]:


df['twp'].value_counts().head(5)


# Take a look at the 'title' column, how many unique title codes are there?

# In[ ]:


df['title'].nunique()


# # Creating new features
# In the titles column there are "Reasons/Departments" specified before the title code. These are EMS, Fire, and Traffic. Use .apply() with a custom lambda expression to create a new column called "Reason" that contains this string value.
# 
# For example, if the title column value is EMS: BACK PAINS/INJURY , the Reason column value would be EMS. 

# In[ ]:


df['Reason']=df['title'].apply(lambda title: title.split(':')[0])


# In[ ]:


df.head()


# What is the most common Reason for a 911 call based off of this new column?

# In[ ]:


df['Reason'].value_counts()


# Now use seaborn to create a countplot of 911 calls by Reason.

# In[ ]:


sns.countplot(x='Reason',data=df,palette='viridis')


# Now let us begin to focus on time information. What is the data type of the objects in the timeStamp column?

# In[ ]:


type(df['timeStamp'].iloc[0])


# You should have seen that these timestamps are still strings. Use pd.to_datetime to convert the column from strings to DateTime objects. 

# In[ ]:


df['timeStamp']= pd.to_datetime(df['timeStamp'])


# The timestamp column are actually DateTime objects, use .apply() to create 3 new columns called Hour, Month, and Day of Week. 

# In[ ]:


df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)
df['Month'] = df['timeStamp'].apply(lambda time: time.month)
df['Day of Week'] = df['timeStamp'].apply(lambda time: time.dayofweek)


# In[ ]:


df.head()


#  Notice how the Day of Week is an integer 0-6. Use the .map() with this dictionary to map the actual string names to the day of the week:
# 
# dmap = {0 :'Mon', 1:'Tue', 2 : 'Wed', 3 : 'Thu',4 : 'Fri', 5 : 'Sat', 6: 'Sun'}

# In[ ]:


dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}


# In[ ]:


df['Day of Week'] = df['Day of Week'].map(dmap)


# In[ ]:


df.head()


# Now use seaborn to create a countplot of the Day of Week column with the hue based off of the Reason column.

# In[ ]:


sns.countplot(x ="Day of Week",data =df,hue = 'Reason',palette='rainbow')

#To relocate the legend
plt.legend(bbox_to_anchor=(1.05,1),loc = 2 ,borderaxespad=0)


# Now do the same for Month:

# In[ ]:


sns.countplot(x = "Month", data = df,hue='Reason',palette="Set1")

#To relocate the legend
plt.legend(bbox_to_anchor=(1.05,1), loc=2,borderaxespad=0.)


#  Now create a gropuby object called byMonth, where we group the DataFrame by the month column and use the count() method for aggregation. Use the head() method on this returned DataFrame.

# In[ ]:


byMonth = df.groupby("Month").count()
byMonth.head()


# Now create a simple plot off of the dataframe indicating the count of calls per month.

# In[ ]:


#Could be any column
byMonth["twp"].plot()


# Now use seaborn's lmplot() to create a linear fit on the number of calls per month. Keep in mind we may need to reset the index to a column.

# In[ ]:


sns.lmplot(x="Month",y= "twp" , data=byMonth.reset_index())


# Create a new column called 'Date' that contains the date from the timeStamp column. You'll need to use apply along with the .date() method.

# In[ ]:


df['Date']=df['timeStamp'].apply(lambda t: t.date())


# Now groupby this Date column with the count() aggregate and create a plot of counts of 911 calls

# In[ ]:


df.groupby('Date').count()['twp'].plot()
plt.tight_layout()


# Now recreate this plot but create 3 separate plots with each plot representing a Reason for the 911 call

# In[ ]:


df[df['Reason']=='Traffic'].groupby('Date').count()['twp'].plot()
plt.title('Traffic')
plt.tight_layout()


# In[ ]:


df[df['Reason']=='Fire'].groupby('Date').count()['twp'].plot()
plt.title('Fire')
plt.tight_layout()


# In[ ]:


df[df["Reason"]=='EMS'].groupby('Date').count()['twp'].plot()
plt.title('EMS')
plt.tight_layout()


# Now let's move on to creating heatmaps with seaborn and our data. We'll first need to restructure the dataframe so that the columns become the Hours and the Index becomes the Day of the Week.

# In[ ]:


#dayHour = df.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack(


dayHour = df.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()
dayHour.head()


# Now create a HeatMap using this new DataFrame.

# In[ ]:


plt.figure(figsize=(12,6))
sns.heatmap(dayHour, cmap='coolwarm')


# Now create a clustermap using this DataFrame.

# In[ ]:


sns.clustermap(dayHour,cmap='coolwarm')


# Now repeat these same plots and operations, for a DataFrame that shows the Month as the column.

# In[ ]:


dayMonth=df.groupby(by=['Day of Week','Month']).count()['Reason'].unstack()
dayMonth.head()


# In[ ]:


plt.figure(figsize=(12,6))
sns.heatmap(dayMonth,cmap='viridis')


# In[ ]:


sns.clustermap(dayMonth,cmap='viridis')


# In[ ]:




