#!/usr/bin/env python
# coding: utf-8

# Hi all,
# 
# I would like to analyze the data given in this competition and share my approach as well. This is just the first of the many parts that I will be sharing shortly. This is my first kernel in months since joining Kaggle. 
# 
# Hope you all like it. Feel free to comment and share your views
# 
# Enjoy Kaggling !!!!!!
# 
# Here is Part 2: [Taxi O Taxi - My Analysis : Part 2][1]
# 
# Here is Part 3 : [Taxi O Taxi - My Analysis : Part 3][2]
# 
# 
#   [1]: https://www.kaggle.com/jeru666/taxi-o-taxi-my-analysis-part-2/
#   [2]: https://www.kaggle.com/jeru666/taxi-o-taxi-my-analysis-part-3/

# In[1]:




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


# Before analyzing, we have to check whether we have any missing values (Nan)

# In[2]:


#--- Check if there are any Nan values ---
train_df.isnull().values.any()


# Good! We don't have any.
# 
# Analyzing **trip_duration** column (also called the target variable)
# 
# First let us see the values it ranges between

# In[3]:


#--- First see the range of values present ---
print (max(train_df['trip_duration'].values))
print (min(train_df['trip_duration'].values))


# The result shown is in seconds. Let us append another column having trip duration shown in minutes.

# In[4]:


#train_df = train_df.assign(trip_duration_mins = lambda x: int(x.trip_duration/60))
train_df['trip_duration_mins'] = train_df.apply(lambda row: row['trip_duration'] / 60, axis=1)
print (train_df.head())


# Now let us see the maximum and the minimum trip duration in minutes. We will now get a better picture!

# In[5]:


print (max(train_df['trip_duration_mins'].values))
print (min(train_df['trip_duration_mins'].values))


# Seriously!! 58,000 minutes for a cab ride! 
# Wait! What if the person has hired it for the entire week! Lets see if there more such occurrences.
# 

# In[6]:


train_df.plot(x=train_df.index, y='trip_duration_mins')


# From the plot it is clear that there are a handful of rides above the 20k minute mark. Let's see how many of them are there.
# We will count and get the 'id's having trip duration greater than 20,000 minutes.

# In[7]:


long_rides = []
short_rides = []
count_short = 0
count_long = 0

for i in range(0, train_df.shape[0]):
    if train_df['trip_duration_mins'][i] > 20000:
        long_rides.append(train_df['id'][i])
        count_long+=1
    elif train_df['trip_duration_mins'][i] < 1:
        short_rides.append(train_df['id'][i])
        count_short+=1
        
print ("These are {} the ids that had long rides.".format(long_rides))
print (count_long)
print ("These are {} the ids that had short rides.".format(short_rides))
print (count_short)


# That is way too many people who took short taxi rides.
# 
# Now let us analyze the **passenger_count** column.
# 
# Again we will see the distribution of the number of passengers for all rides.

# In[8]:


#--- First let us count the number of unique passenger counts in the data set ---
print ("These are {} unique passenger counts.".format(train_df['passenger_count'].nunique()))

#--- Well what are those counts ? ---
print (train_df['passenger_count'].unique())


# In[9]:


#--- Now let us plot them against the index and see their distribution.

pd.value_counts(train_df['passenger_count']).plot.bar()


# In[10]:


#--- But we want to see the actual count ---
#train_df.groupby(train_df['passenger_count']).count()
train_df['passenger_count'].value_counts()


# Some interesting observations:
#     
# 
#  - There are 60 rides recorded to have been made with no passengers at
#    all. (was the driver dreaming)
#  - Most of the rides had 1 to 6 passengers (with single passenger rides
#    being the highest)
#  - However, rides having 7, 8 & 9 passengers are in single digit
#    collectively!!
# 
# 
# 

# So let's see if any relation exists between the above mention independent variables(predictors) namely **trip_duration_mins** and **passenger_count**. We can do this by finding out the correlation between the two columns.

# In[11]:


print (train_df['trip_duration_mins'].corr(train_df['passenger_count']))


# The correlation between these columns is too low.

# In[12]:


#--- Seeing the rows of data having longest cab rides ---
train_df[train_df.id.isin(long_rides)]


# We do not see any relation between the trip duration and the time covered in any of these 4 rows.
# 
# Continuation of My Analysis on Taxi O Taxi to be continued in Part 2.
# 
# Here is Part 2: [Taxi O Taxi - My Analysis : Part 2][1]
# 
# Here is Part 3: [Taxi O Taxi - My Analysis : Part 3][2]
# 
# 
#   [1]: https://www.kaggle.com/jeru666/taxi-o-taxi-my-analysis-part-2/
#   [2]: https://www.kaggle.com/jeru666/taxi-o-taxi-my-analysis-part-3/
