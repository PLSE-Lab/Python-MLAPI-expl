#!/usr/bin/env python
# coding: utf-8

# Hey guys!
# 
# We are going to continue our analysis. We will see the two remaining columns **vendor_id** and **store_and_fwd_flag**
# 
# 
# Missing Part 3? Here it is: [Taxi O Taxi - My Analysis - Part 3][1]
# 
# and here is Part 1: [Taxi O Taxi - My Analysis - Part 1][2]
# 
# and here is Part 2:[Taxi O Taxi - My Analysis - Part 2][3]
# 
# Let us load the csv as usual.
# 
# 
#   [1]: https://www.kaggle.com/jeru666/taxi-o-taxi-my-analysis-part-3
#   [2]: https://www.kaggle.com/jeru666/taxi-o-taxi-my-analysis-part-1/
#   [3]: https://www.kaggle.com/jeru666/taxi-o-taxi-my-analysis-part-2

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


# Let us also add those additional columns we made in the previous kernels

# In[ ]:


train_df['pickup_datetime'] = pd.to_datetime(train_df['pickup_datetime'])
train_df['dropoff_datetime'] = pd.to_datetime(train_df['dropoff_datetime'])

train_df['pickup_month'] = train_df.pickup_datetime.dt.month.astype(np.uint8)
train_df['pickup_day'] = train_df.pickup_datetime.dt.weekday.astype(np.uint8)
train_df['pickup_hour'] = train_df.pickup_datetime.dt.hour.astype(np.uint8)

train_df['dropoff_month'] = train_df.dropoff_datetime.dt.month.astype(np.uint8)
train_df['dropoff_day'] = train_df.dropoff_datetime.dt.weekday.astype(np.uint8)
train_df['dropoff_hour'] = train_df.dropoff_datetime.dt.hour.astype(np.uint8)
print (train_df.head())


# In[ ]:


from math import radians, cos, sin, asin, sqrt   #--- for the mathematical operations involved in the function ---

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6367 * c
    return km

train_df['Displacement (km)'] = train_df.apply(lambda x: haversine(x['pickup_longitude'], x['pickup_latitude'], x['dropoff_longitude'], x['dropoff_latitude']), axis=1)

print (train_df.head())


# There are other columns left to analyze: **vendor_id** and **store_and_fwd_flag**
# 
# ##Vendor IDs

# In[ ]:


#--- First let us count the number of unique vendor_ids in the data set ---
print ("These are {} unique vendor ids.".format(train_df['vendor_id'].nunique()))

#--- Well what are those counts ? ---
print (train_df['vendor_id'].unique())


# These are probably two different taxi vendors or taxi companies.
# 
# Let us see their distribution in terms of the number of rides each has taken so far.

# In[ ]:


pd.value_counts(train_df['vendor_id']).plot.bar()


# Let us see the actual number of trips made by both the vendors

# In[ ]:


print(train_df['vendor_id'].value_counts())


# Taxi company 2 has been slightly busier.
# 
# Let us see if both the companies have got anything to do with the **trip_duration**

# In[ ]:


s = train_df['trip_duration'].groupby(train_df['vendor_id']).sum()
print (s)


# Clearly, taxi company 2 has more trip durations than taxi company 1.
# 
# But now let us check the mean trip durations for both the vendors:

# In[ ]:


s = train_df['trip_duration'].groupby(train_df['vendor_id']).mean()
print (s)


# Let us find the mean trip duration of both the taxi vendors with respect to their pick-up points.

# In[ ]:


print('Mean pickup_longitude')
print(train_df['pickup_longitude'].groupby(train_df['vendor_id']).mean())
print(' ')
print('Mean pickup_latitude')
print(train_df['pickup_latitude'].groupby(train_df['vendor_id']).mean())


# This doesn't seem to help us much.
# 
# Now let us see if the pickup month, day and hour have any significance on the vendors.

# In[ ]:


print('Mean pickup_month')
print(train_df['pickup_month'].groupby(train_df['vendor_id']).mean())
print(' ')
print('Mean pickup_day')
print(train_df['pickup_day'].groupby(train_df['vendor_id']).mean())
print(' ')
print('Mean pickup_hour')
print(train_df['pickup_hour'].groupby(train_df['vendor_id']).mean())


# Hmmm, unable to infer anything decent from this. But, it is good thing we tried it !!!

# ##Store and fwd flag

# In[ ]:


#--- Now let us count the number of unique store_and_fwd_flags in the data set ---
print ("These are {} unique store_and_fwd_flags.".format(train_df['store_and_fwd_flag'].nunique()))

#--- Well what are those counts ? ---
print (train_df['store_and_fwd_flag'].unique())


# In[ ]:


#--- Let us plot them against the index and see their distribution.

pd.value_counts(train_df['store_and_fwd_flag']).plot.bar()


# Let us get the correct count of each of these flags:

# In[ ]:


print(train_df['store_and_fwd_flag'].value_counts())


# Since this column alone has categorical values, before feeding this info to the model it has to be converted to a numerical valued column.
# 
# Before that we have to convert the datatype of the column to type 'category'. After doing so we can convert it to a column having numerical values.

# In[ ]:



train_df['store_and_fwd_flag'] = train_df['store_and_fwd_flag'].astype('category')
train_df['store_and_fwd_flag'] = train_df['store_and_fwd_flag'].cat.codes

#---Now let us count them again ---
print (train_df['store_and_fwd_flag'].unique())

print(train_df.head())


# If you observe the **'store_and_fwd_flag'** column, they have been changed to N -> 0 and Y -> 1.

# In[ ]:


print('Mean trip duration for each flag')
print(train_df['trip_duration'].groupby(train_df['store_and_fwd_flag']).mean())
print (' ')
print('Mean Displacement in km for each flag')
print(train_df['Displacement (km)'].groupby(train_df['store_and_fwd_flag']).mean())


# In[ ]:


print (train_df['store_and_fwd_flag'].value_counts())


# We notice something interesting here:
# 
#  - The trip duration and the displacement for Y flag (1) is **higher** than N flag (0) **BUT**, the number of trips for the Y flag is **lower** than that of the N flag.
# 
# Maybe THIS can help us at the time of modeling!!!!!

# Now let us see if the pickup month, day and hour have any significance on the respective flags.

# In[ ]:


print('Mean pickup_month')
print(train_df['pickup_month'].groupby(train_df['store_and_fwd_flag']).mean())
print(' ')
print('Mean pickup_day')
print(train_df['pickup_day'].groupby(train_df['store_and_fwd_flag']).mean())
print(' ')
print('Mean pickup_hour')
print(train_df['pickup_hour'].groupby(train_df['store_and_fwd_flag']).mean())


# Flag 'Y' shows a lesser average over the pick-up days compared to flag 'N' Apart from this, the other averages are of not much significance.  

# #There is still a lot to be done!!
# 
# #So STAY TUNED !!!!
