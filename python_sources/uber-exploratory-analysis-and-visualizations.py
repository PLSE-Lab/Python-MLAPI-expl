#!/usr/bin/env python
# coding: utf-8

# To just explore the Uber dataset through some visualizations. :)

# In[ ]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
import datetime


get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (8.0, 6.0)


# In[ ]:


#read in file
uber2015 = pd.read_csv('../input/uber-raw-data-janjune-15.csv')


# In[ ]:


#browsing dataset
uber2015.head()


# In[ ]:


#number of rows in the dataset
len(uber2015)


# In[ ]:


#Checking the minimum date in the dataset
uber2015['Pickup_date'].min()


# In[ ]:


#Checking the maximum date in the dataset
uber2015['Pickup_date'].max()


# In[ ]:


#converting Pickup_date column to datetime 
uber2015['Pickup_date'] =  pd.to_datetime(uber2015['Pickup_date'], format='%Y-%m-%d %H:%M:%S')
#extracting the Hour from Pickup_date column and putting it into a new column named 'Hour'
uber2015['Hour'] = uber2015.Pickup_date.apply(lambda x: x.hour)
#extracting the Minute from Pickup_date column and putting it into a new column named 'Minute'
uber2015['Minute'] = uber2015.Pickup_date.apply(lambda x: x.minute)
#extracting the Month from Pickup_date column and putting it into a new column named 'Month'
uber2015['Month'] = uber2015.Pickup_date.apply(lambda x: x.month)
#extracting the Day from Pickup_date column and putting it into a new column named 'Day'
uber2015['Day'] = uber2015.Pickup_date.apply(lambda x: x.day)


# In[ ]:


#browse updated dataset
uber2015.head()


# In[ ]:


# Uber pickups by the month in NYC
sns.set_style('whitegrid')
ax = sns.countplot(x="Month", data=uber2015, color="lightsteelblue")
ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.set_xlabel('Month', fontsize = 12)
ax.set_ylabel('Count of Uber Pickups', fontsize = 12)
ax.set_title('Uber Pickups By the Month in NYC (Jan-Jun 2015)', fontsize=16)
ax.tick_params(labelsize = 8)
plt.show()


# We can see that the number of Uber pickup has been steadily increasing throughout the first half of 2015 in NYC.

# In[ ]:


# Uber pickups by the hour in NYC
sns.set_style('whitegrid')
ax = sns.countplot(x="Hour", data=uber2015, color="lightsteelblue")
ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.set_xlabel('Hour of the Day', fontsize = 12)
ax.set_ylabel('Count of Uber Pickups', fontsize = 12)
ax.set_title('Uber Pickups By the Hour in NYC (Jan-Jun 2015)', fontsize=16)
ax.tick_params(labelsize = 8)
plt.show()


# Interestingly, after the morning rush, the number of Uber pickups doesn't dip much throughout the rest of the morning and early afternoon.
# There is significantly more demand in the evening than the daytime.
# Let's investigate to see if there's a difference in hourly pattern for different days of the week.

# In[ ]:


uber2015['Weekday'] = uber2015.Pickup_date.apply(lambda x: x.strftime('%A'))


# In[ ]:


uber2015.head()


# In[ ]:


#group the data by Weekday and hour
summary = uber2015.groupby(['Weekday', 'Hour'])['Pickup_date'].count()


# In[ ]:


#reset index
summary = summary.reset_index()
#convert to dataframe
summary = pd.DataFrame(summary)
#browse data
summary.head()


# In[ ]:


#rename last column
summary=summary.rename(columns = {'Pickup_date':'Counts'})


# In[ ]:


sns.set_style('whitegrid')
ax = sns.pointplot(x="Hour", y="Counts", hue="Weekday", data=summary)
handles,labels = ax.get_legend_handles_labels()
#reordering legend content
handles = [handles[1], handles[5], handles[6], handles[4], handles[0], handles[2], handles[3]]
labels = [labels[1], labels[5], labels[6], labels[4], labels[0], labels[2], labels[3]]
ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.set_xlabel('Hour', fontsize = 12)
ax.set_ylabel('Count of Uber Pickups', fontsize = 12)
ax.set_title('Hourly Uber Pickups By Day of the Week in NYC (Jan-Jun 2015)', fontsize=16)
ax.tick_params(labelsize = 8)
ax.legend(handles,labels,loc=0, title="Legend", prop={'size':8})
ax.get_legend().get_title().set_fontsize('8')
plt.show()


# It's interesting to see that Saturday and Sunday exhibit similar demand throughout the late night/morning/afternoon, but it exhibits opposite trends during the evening. In the evening, Saturday pickups continue to increase throughout the evening, but Sunday pickups takes a downward turn after dinner time.
# 
# We can see that there the weekdays that has the most demand during the late evening is Friday and Saturday, which is expected, but what strikes me is that Thursday nights also exhibits very similar trends as Friday and Saturday nights. 
# 
# It seems like New Yorkers are starting their 'weekends' on Thursday nights. :)
