#!/usr/bin/env python
# coding: utf-8

# # All 55 million training data points once and for all
# All the kernels I've seen so far for this challenge are using `nrows=xxxxx` in pandas `read_csv` which would not make a thorough representation of the training dataset. I decided to load the whole dataset in this kernel to get the general information once and for all. Let's make it short and simple.

# In[ ]:


#import libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# ## Reduce memory usage:
# We need to reduce our memory usage as mush as possible:
# * We don't need to load the `key` for data exploration whatsoever. So we only use the columns we would want.
# * One good suggestion from [another kernel](https://www.kaggle.com/szelee/how-to-import-a-csv-file-of-55-million-rows) was to set the data types before loading the data. However, unlike that kernel for the latitude and longitudes, in order not to lose any of our data and to capture approximately 16 decimal points we will need float64 not float32. As @szelee mentioned in the coments section, using float32 would not cause much accuracy loss. Thus, you might use float32 for your modeling purposes.
# 
# Note that this changes **reduced the memory usage from 3.5GB to 2.3GB**! If float32 was used for all spatial georeferences, memory usage would have been 1.5GB.

# In[ ]:


types = {'fare_amount': 'float32',
         'pickup_longitude': 'float64',
         'pickup_latitude': 'float64',
         'dropoff_longitude': 'float64',
         'dropoff_latitude': 'float64',
         'passenger_count': 'uint8'}
cols = ['fare_amount', 'pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']
train_data = pd.read_csv('../input/train.csv', dtype=types, usecols=cols, infer_datetime_format=True, parse_dates=["pickup_datetime"]) # total nrows = 55423855
#test_data = pd.read_csv('../input/test.csv', nrows=0)
train_data.info()


# In[ ]:


train_data.describe()


# ## Inference
# * The minimum fare amount in the entire training dataset is -\$30 and the maximum is \$93,963!! This is why we need to see the whole dataset! the maximum fare I saw on other kernels (with limited number of rows) was about $500. 
# * The maximum passenger_count is 208! Maybe a ship! who knows?
# * Max(pickup_longitude) = 3457.63 degrees! Min(pickup_longitude) = -3492.26 while latitudes range from 0 to 90; longitudes range from 0 to 180. Compass direction North, South, East or West could be represented by `-` and `+`.
# * The same thing goes with other latitudes and longitudes, their maximum and minimums, or essentially any latitude not in `(-90, 90)` and longitude not in `(-180, 180)`,are for sure false values and need to be dropped.

# ## Missing data
# About 0.00068% of the training data is missing. Drop them! It's safe.

# In[ ]:


train_data.isnull().sum()


# ## Histograms

# In[ ]:


counts = train_data[train_data.passenger_count<6].passenger_count.value_counts()
plt.bar(counts.index, counts.values)
plt.xlabel('No. of passengers')
plt.ylabel('Frequency')
plt.xticks(range(0,7))
print(counts)


# In[ ]:


# to capture 75% of the training dataset
train_data[(train_data.fare_amount<125) & (train_data.fare_amount>0)].fare_amount.hist(bins=175, figsize=(15,4))
plt.xlabel('fare $USD')
plt.ylabel('Frequency')
plt.xlim(xmin=0);


# ## Conclusion
# * There might be very harsh/effective outliers and you might miss them if you are exploring data on `nrows=xxxx`.
# * There are not so many missing data points in general. 
# * The last and the most important conclusion: I was curious to have a glance at the entire training dataset at once. If you are too, don't use kaggle's computation power to generate the same results of this kernel again ; )
