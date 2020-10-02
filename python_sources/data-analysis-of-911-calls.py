#!/usr/bin/env python
# coding: utf-8

# # Data Analysis of 911 Calls - Capstone Project
# 
# The 911 system was designed to provide a universal, easy-to-remember number for people to reach police, fire or emergency medical assistance from any phone in any location, without having to look up specific phone numbers. Today, people communicate in ways that the designers of the original 911 system could not have envisioned: wireless phones, text and video messages, social media, Internet Protocol (IP)-enabled devices, and more.
# 
# The National 911 Program works with States, technology providers, public safety officials and 911 professionals to ensure a smooth transition to an updated 911 system that takes advantage of new communications technologies. It also creates and shares a variety of resources and tools to help 911 systems.
# 
# Created by Congress in 2004 as the 911 Implementation and Coordination Office (ICO), the National 911 Program is housed within the National Highway Traffic Safety Administration at the U.S. Department of Transportation and is a joint program with the National Telecommunication and Information Administration in the Department of Commerce
# 
# This is a capstone project for the udemy course ["Python for Data Science and Machine Learning Bootcamp"
# ](https://www.udemy.com/course/python-for-data-science-and-machine-learning-bootcamp/) [](http://)
# 
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
# 

# In[ ]:


# Importing numpy and pandas libraries

import numpy as np
import pandas as pd


# In[ ]:


#Importing Visualization libraries

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Read in the csv file from Kaggle and create a dataframe called df

df=pd.read_csv('../input/montcoalert/911.csv')


# In[ ]:


#Check the info() of the df

df.info()


# In[ ]:


#Read in the csv file as a dataframe called df

df.head()


# # Creating new features
# 
# In the titles column there are "Reasons/Departments" specified before the title code. These are EMS, Fire, and Traffic. Now using .apply() with a custom lambda expression we will create a new column called "Reason" that contains this string value.**
# 
# *For example, if the title column value is EMS: BACK PAINS/INJURY , the Reason column value would be EMS. *

# In[ ]:


df['Reason'] = df['title'].apply(lambda title: title.split(':')[0])


# In[ ]:


#What is the most common Reason for a 911 call based off of this new column?

df['Reason'].value_counts()


# In[ ]:


#Now using seaborn to create a countplot of 911 calls by Reason.

sns.countplot(x='Reason',data=df,palette='coolwarm')


# In[ ]:


#Now let us begin to focus on time information. What is the data type of the objects in the timeStamp column?

type(df['timeStamp'].iloc[0])


# In[ ]:


#Use [pd.to_datetime] to convert the column from strings to DateTime objects

df['timeStamp'] = pd.to_datetime(df['timeStamp'])


# In[ ]:


# Since the timestamp column are actually DateTime objects, we will use .apply() to create 3 new columns called Hour, Month, and Day of Week. 

df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)
df['Month'] = df['timeStamp'].apply(lambda time: time.month)
df['Day of Week'] = df['timeStamp'].apply(lambda time: time.dayofweek)


# In[ ]:


#Notice how the Day of Week is an integer 0-6. Use the .map() with this dictionary to map the actual string names to the day of the week:

dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}


# In[ ]:


#Now use seaborn to create a countplot of the Day of Week column with the hue based off of the Reason column.

sns.countplot(x='Day of Week',data=df,hue='Reason',palette='coolwarm')

# To relocate the legend
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# In[ ]:


#Now use seaborn to create a countplot of the Month column.

sns.countplot(x='Month',data=df,hue='Reason',palette='coolwarm')

# To relocate the legend
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# In[ ]:


#Create a new column called 'Date' that contains the date from the timeStamp column. You'll need to use apply along with the .date() method

df['Date']=df['timeStamp'].apply(lambda t: t.date())


# In[ ]:


#Now groupby this Date column with the count() aggregate and create a plot of counts of 911 calls and recreate this plot representing a Reason for the 911 call

df[df['Reason']=='Traffic'].groupby('Date').count()['twp'].plot()
plt.title('Traffic')
plt.tight_layout()


# In[ ]:


#Now recreate this plot but create 3 separate plots with each plot representing a Reason for the 911 call

df[df['Reason']=='Fire'].groupby('Date').count()['twp'].plot()
plt.title('Fire')
plt.tight_layout()


# In[ ]:


#Now recreate this plot but create 3 separate plots with each plot representing a Reason for the 911 call

df[df['Reason']=='EMS'].groupby('Date').count()['twp'].plot()
plt.title('EMS')
plt.tight_layout()


# In[ ]:


# Now let's move on to creating heatmaps with seaborn and our data. We'll first need to restructure the dataframe so that the columns become the Hours and the Index becomes the Day of the Week. 
#There are lots of ways to do this, but I would recommend trying to combine groupby with an unstack method. 

dayHour = df.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()
dayHour.head()


# In[ ]:


#Now create a HeatMap using this new DataFrame.

plt.figure(figsize=(12,6))
sns.heatmap(dayHour,cmap='coolwarm')


# In[ ]:


#Now create a clustermap using this DataFrame

sns.clustermap(dayHour,cmap='coolwarm')


# In[ ]:


#Now repeat these same plots and operations, for a DataFrame that shows the Month as the column

dayMonth = df.groupby(by=['Day of Week','Month']).count()['Reason'].unstack()
dayMonth.head()


# In[ ]:


plt.figure(figsize=(12,6))
sns.heatmap(dayMonth,cmap='coolwarm')


# In[ ]:


sns.clustermap(dayMonth,cmap='coolwarm')

