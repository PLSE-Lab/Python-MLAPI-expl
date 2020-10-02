#!/usr/bin/env python
# coding: utf-8

# Emergency (911) Calls: Fire, Traffic, EMS for Montgomery County, PA
# 
# You can get a quick introduction to this Dataset with this kernel: Dataset Walk-through
# 
# Acknowledgements: Data provided by montcoalert.org
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

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv("../input/911.csv")


# In[ ]:


df.head(5)


# In[ ]:


df.shape


# In[ ]:


df.describe()


# In[ ]:


# Top 5 zipcodes for 911 calls:

df['zip'].value_counts().head(5)


# In[ ]:


#top 5 townships for 911 calls? 
df['twp'].value_counts().head(5)


# In[ ]:


df['title'].nunique() #unique title codes


# * ## New features
# 
# 
# 
# In the titles column there are "Reasons/Departments" specified before the title code. These are EMS, Fire, and Traffic. We will use .apply() with a custom lambda expression to create a new column called "Reason" that contains this string value.
# 
# *For example,* if the title column value is EMS: BACK PAINS/INJURY , the Reason column value would be EMS. 

# In[ ]:


df['Reason'] = df['title'].apply(lambda title: title.split(':')[0])


# In[ ]:


df['Reason'].value_counts() #most common Reason for a 911


# In[ ]:


plt.figure(figsize=(15,8))

sns.countplot(x='Reason',data=df,palette='viridis') # countplot of 911 calls by Reason. 


# In[ ]:


type(df['timeStamp'].iloc[0]) # data type of the objects in the timeStamp column


# In[ ]:


# Use pd.to_datetime to convert the column from strings to DateTime objects. **

df['timeStamp'] = pd.to_datetime(df['timeStamp'])


# In[ ]:


#grab specific attributes from a Datetime object by calling them

df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)
df['Month'] = df['timeStamp'].apply(lambda time: time.month)
df['Day of Week'] = df['timeStamp'].apply(lambda time: time.dayofweek)


# Notice how the Day of Week is an integer 0-6. Use the .map() with this dictionary to map the actual string names to the day of the week
# 
# dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

# In[ ]:


dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}


# In[ ]:


df['Day of Week'] = df['Day of Week'].map(dmap)


# In[ ]:


plt.figure(figsize=(15,8))

sns.countplot(x='Day of Week',data=df,hue='Reason',palette='viridis') #ountplot of the Day of Week column with the hue based off of the Reason column.

# To relocate the legend
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


# In[ ]:


plt.figure(figsize=(15,8))

sns.countplot(x='Month',data=df,hue='Reason',palette='viridis') #same for month

# To relocate the legend
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


# In[ ]:


#gropuby object called byMonth, where you group the DataFrame by the month column and use the count() method for aggregation
byMonth = df.groupby('Month').count()
byMonth.head()


# In[ ]:


# Could be any column
plt.figure(figsize=(18,10))
byMonth['twp'].plot()
plt.show()


# In[ ]:


sns.lmplot(x='Month',y='twp',height = 12, data=byMonth.reset_index()) # number of calls per month


# In[ ]:


#new column called 'Date' that contains the date from the timeStamp column
df['Date']=df['timeStamp'].apply(lambda t: t.date()) 


# In[ ]:


plt.figure(figsize=(18,10))
df.groupby('Date').count()['twp'].plot()
plt.tight_layout()


# In[ ]:



plt.figure(figsize=(18,10))
df[df['Reason']=='Traffic'].groupby('Date').count()['twp'].plot()
plt.title('Traffic')
plt.tight_layout()


# In[ ]:


plt.figure(figsize=(18,10))


df[df['Reason']=='Fire'].groupby('Date').count()['twp'].plot()
plt.title('Fire')
plt.tight_layout()


# In[ ]:


plt.figure(figsize=(18,10))

df[df['Reason']=='EMS'].groupby('Date').count()['twp'].plot()
plt.title('EMS')

plt.tight_layout()


# Now let's move on to creating heatmaps with seaborn and our data. We'll first need to restructure the dataframe so that the columns become the Hours and the Index becomes the Day of the Week. There are lots of ways to do this, but I would recommend trying to combine groupby with an unstack method. Reference the solutions if you get stuck on this

# In[ ]:


dayHour = df.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()
dayHour.head()


# In[ ]:


#HeatMap using this new DataFrame
plt.figure(figsize=(18,10))

sns.heatmap(dayHour,cmap='viridis')


# In[ ]:


plt.figure(figsize=(18,10))

sns.clustermap(dayHour,figsize=(16, 17), cmap='viridis') #clustermap


# In[ ]:


# repeat these same plots and operations, for a DataFrame that shows the Month as the column
dayMonth = df.groupby(by=['Day of Week','Month']).count()['Reason'].unstack()
dayMonth.head()


# In[ ]:


plt.figure(figsize=(18,10))
sns.heatmap(dayMonth,cmap='viridis')


# In[ ]:


sns.clustermap(dayMonth,figsize=(16, 17),cmap='viridis')


# In[ ]:




