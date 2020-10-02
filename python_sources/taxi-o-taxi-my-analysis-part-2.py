#!/usr/bin/env python
# coding: utf-8

# Hi guys,
# 
# This is Part 2 of my analysis on Taxi O Taxi !!!
# 
# In Part 1 we explored the **trip_duration** variable (target) and the '**passenger_count**' variable (predictor). In case you missed it, here is the link: [Taxi O Taxi - My Analysis - Part 1][1]
# 
# In this kernel let us analyze some other independent variables (also called predictors or features). We will look at the variables involving time and date (**pickup_datetime** and **dropoff_datetime**)
# 
# So lets get started!! 
# 
#   [1]: https://www.kaggle.com/jeru666/taxi-o-taxi-my-analysis-part-1

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Input data files are available in the "../input/" directory.
path = 'D:/BACKUP/Kaggle/New York City Taxi/Data/'
train_df = pd.read_csv('../input/train.csv')

#--- Let's peek into the data
print (train_df.head())


# Analyzing the **date and time** variables. (ie; **pickup_datetime** and **dropoff_datetime**).
# 
# First let us see the data type of those two columns:

# In[ ]:


print (train_df.dtypes)


# It says that **pickup_datetime** and **dropoff_datetime** are of *object* type.  But what does that mean?
# 
# We will check the datatype of the first element in that column

# In[ ]:


print (train_df['pickup_datetime'][0])
print (type(train_df['pickup_datetime'][0]))


# Voila!! We see that the elements in those columns are of type string.  
# 
# Now we have to convert both those columns to **Timestamp** datatype which is a datetime object

# In[ ]:


train_df['pickup_datetime'] = pd.to_datetime(train_df['pickup_datetime'])
train_df['dropoff_datetime'] = pd.to_datetime(train_df['dropoff_datetime'])

#--- Now let us see the datatype of both those columns ---
print (train_df.dtypes)


# Notice that the datatype of the columns **pickup_datetime** and **dropoff_datetime** have been changed to type **datetime**.
# 
# Now we can split them into day, month, hour and so on....

# In[ ]:


train_df['pickup_month'] = train_df.pickup_datetime.dt.month.astype(np.uint8)
train_df['pickup_day'] = train_df.pickup_datetime.dt.weekday.astype(np.uint8)
train_df['pickup_hour'] = train_df.pickup_datetime.dt.hour.astype(np.uint8)

train_df['dropoff_month'] = train_df.dropoff_datetime.dt.month.astype(np.uint8)
train_df['dropoff_day'] = train_df.dropoff_datetime.dt.weekday.astype(np.uint8)
train_df['dropoff_hour'] = train_df.dropoff_datetime.dt.hour.astype(np.uint8)
print (train_df.head())


# 
# We have split our pick_up and drop_off datetime into separate columns. Now we have an additional 6 columns. We are doing this to try and extract as many features possible.
# 
# Let us check if any correlations exist between any of the newly created columns

# In[ ]:


#--- Correlation between columns 'pickup_month' and 'dropoff_month' ---
print (train_df['pickup_month'].corr(train_df['dropoff_month']))


# In[ ]:


#--- Correlation between columns 'pickup_day' and 'dropoff_day' ---
print (train_df['pickup_day'].corr(train_df['dropoff_day']))


# Both the correlation values are vey high. We can remove any one of them, but for the time being, let's keep them.
# 
# Let's see the relation of these columns with the number of trips made 
# 
# ## Number of trips VS Pickup Months
# 
# 

# In[ ]:


#--- Unique elements in column pickup_month ---
print (train_df['pickup_month'].unique())


# So the data has info for just the 6 months of the year

# In[ ]:


import seaborn as sns   #--- I realized this library helps us in visualizing relations better between columns ---

data = train_df.groupby('pickup_month').aggregate({'id':'count'}).reset_index()
month_list=["Jan","Feb","Mar","Apr","May", "Jun"]
ax = sns.barplot(x='pickup_month', y='id', data=data)
ax.set_xticklabels(month_list)


# We can see an almost normal distribution between the two.
# 
# ## Number of trips VS Pickup Days

# In[ ]:


data = train_df.groupby('pickup_day').aggregate({'id':'count'}).reset_index()
day_list = ["Mon","Tue","Wed","Thu","Fri", "Sat", "Sun"]
ax = sns.barplot(x='pickup_day', y='id', data=data)
ax.set_xticklabels(day_list)


# Mondays and Sundays appear to be the least travelled days
# 
# ## Number of trips VS Pickup Hours

# In[ ]:


data = train_df.groupby('pickup_hour').aggregate({'id':'count'}).reset_index()
sns.barplot(x='pickup_hour', y='id', data=data)


# The time between 6 PM - 10 PM appear to be the favorite time for traveling.
# 
# ## Number of trips VS Drop-off Months
# 

# In[ ]:


data = train_df.groupby('dropoff_month').aggregate({'id':'count'}).reset_index()
ax = sns.barplot(x='dropoff_month', y='id', data=data)
ax.set_xticklabels(month_list)


# ## Number of trips VS Drop-off Days

# In[ ]:


data = train_df.groupby('dropoff_day').aggregate({'id':'count'}).reset_index()
ax = sns.barplot(x='dropoff_day', y='id', data=data)
ax.set_xticklabels(day_list)


# ## Number of trips VS Drop-off Hours

# In[ ]:


data = train_df.groupby('dropoff_hour').aggregate({'id':'count'}).reset_index()
sns.barplot(x='dropoff_hour', y='id', data=data)


# Actually, there was no need for visualizing the **drop_off** date and time because they are similar, if not same as the **pick_up** date and time
# 
# 

# ##Mean trip_duration over pickup_month

# In[ ]:


print('Mean trip_duration over pickup_month')
print(train_df['trip_duration'].groupby(train_df['pickup_month']).mean())
print(' ')
mean_pickup_month = train_df['trip_duration'].groupby(train_df['pickup_month']).mean()
sns.barplot(month_list, mean_pickup_month)


# ##Mean trip_duration over pickup_day

# In[ ]:


print('Mean trip_duration over pickup_day')
print(train_df['trip_duration'].groupby(train_df['pickup_day']).mean())
print(' ')
mean_pickup_day = train_df['trip_duration'].groupby(train_df['pickup_day']).mean()
sns.barplot(day_list, mean_pickup_day)


# ##Mean trip_duration over pickup_hour

# In[ ]:


print('Mean trip_duration over pickup_hour')
print(train_df['trip_duration'].groupby(train_df['pickup_hour']).mean())
mean_pickup_hour = train_df['trip_duration'].groupby(train_df['pickup_hour']).mean()
hour_list = []
for i in range(0, 24):
    hour_list.append(i)
    
sns.barplot(hour_list, mean_pickup_hour)


# Let us do some correlation to find out which column actually influences the trip_duration

# In[30]:


train_df[train_df.columns[1:]].corr()['trip_duration'][:-1]
#train_df.head()


# ##Passenger count vs Trip duration

# In[31]:


sns.barplot(x='passenger_count', y='trip_duration', data = train_df)


# ##Mean Passenger Count vs Trip Duration

# In[48]:


mean_passenger_count = train_df['trip_duration'].groupby(train_df['passenger_count']).mean()
passenger_count_list = []
for i in range(0, 10):
    passenger_count_list.append(i)
    
sns.barplot(passenger_count_list, mean_passenger_count, data = train_df)
print (mean_passenger_count)


# In[50]:


print (train_df['passenger_count'].unique())
print(train_df.groupby('passenger_count').passenger_count.count())


# Part 3 is here : [Taxi O Taxi - My Analysis - Part 3][1]
# 
# Link for Part 1 - [Taxi O Taxi - My Analysis - Part 1][2]
# 
# Modeling : [Modeling Part with the outliers!!][3]
# 
# 
#   [1]: https://www.kaggle.com/jeru666/taxi-o-taxi-my-analysis-part-3
#   [2]: https://www.kaggle.com/jeru666/taxi-o-taxi-my-analysis-part-1
#   [3]: https://www.kaggle.com/jeru666/taxi-o-taxi-modeling-with-the-outlaws
