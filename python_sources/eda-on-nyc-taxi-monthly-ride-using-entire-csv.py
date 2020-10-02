#!/usr/bin/env python
# coding: utf-8

# # Exploring 55 million rows of NYC Taxi Fare data with efficient memory usage

# As a follow up from my previous [kernel](https://www.kaggle.com/szelee/how-to-import-a-csv-file-of-55-million-rows), I proceed to do EDA on the entire 55 million rows of data. 
# 
# The main ideas is to load the CSV without the pickup and dropoff coordinates. By selectively choosing the columns during CSV file reading, we can reduce the memory usage and loading time.
# 
# I focus on the analysis of monthly taxi ride for this kernel.
# 
# TLDR:
# 1. The test data distribution is quite **different** from training data in terms of monthly total taxi rides. This is not my discovery, I read about it from this [kernel](https://www.kaggle.com/akosciansky/using-ml-for-data-exploration-feat-engineering) by @akosciansky
# 1. There is a sharp increase in taxi fare from **September 2012** onward. This matches the news of [New York Taxis to Start Charging Increased Rates](https://www.nytimes.com/2012/09/04/nyregion/new-york-taxis-to-start-charging-increased-rates.html)
# 1. **\$2.50** is the reasonable minimum cut-off point to remove outliers. 
# 1.  **\$500** is the reasonable maximum cut-off point to remove outliers (further analysis on pickup and dropoff corrdinates should lower this value)
# 1. Most of the outliers of both maximum and minimum monthly fare lie in the  same periods of time.

# # Import libraries

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os


# # Load data csv format
# (This section only needs to be executed once! Subsequently, once feather format data is created, jump straight to **Load data feather format** section below which is much faster!

# In[ ]:


# Set columns to most suitable type to optimize for memory usage, 
# We can also drop passenger_count since it's not used in this kernel, but I just keep it since it uses only uint8
types = {'fare_amount': 'float32',
         'passenger_count': 'uint8'}

# Columns to load for training data
cols_train = ['fare_amount', 'pickup_datetime', 'passenger_count', 'pickup_datetime']

# Columns to load for test data
cols_test = ['pickup_datetime', 'passenger_count']


# In[ ]:


get_ipython().run_cell_magic('time', '', "i = 0\ndf_list = [] # list to hold the batch dataframe\nchunksize = 10_000_000 # 10 million rows at one go. \nfor df_chunk in pd.read_csv('../input/train.csv', usecols=cols_train, dtype=types, chunksize=chunksize):\n    \n    i = i+1\n    print(f'DataFrame Chunk {i}')\n    \n    # Neat trick from https://www.kaggle.com/btyuhas/bayesian-optimization-with-xgboost\n    # Slicing off unnecessary components of the datetime and specifying the date \n    # format results in a MUCH more efficient conversion to a datetime object.\n    df_chunk['pickup_datetime'] = df_chunk['pickup_datetime'].str.slice(0, 16)\n    df_chunk['pickup_datetime'] = pd.to_datetime(df_chunk['pickup_datetime'], utc=True, format='%Y-%m-%d %H:%M')\n    \n    df_list.append(df_chunk) ")


# In[ ]:


# Merge all dataframes into one dataframe
train_df = pd.concat(df_list)

del df_list

train_df.info()


# This has only used less than **700Mb** of memory for the entire **55 million rows** of data.
# 
# (Bear in mind that we have purposely ignore pickup and dropoff coordinates for this analysis, otherwise it would consume about **1.5Gb** of memory instead)
# 
# As comparison, with eat-all-you-can loading such as:
# 
# `train_df =  pd.read_csv('../input/train.csv', nrows = 10_000_000)`
# 
# will result in about **610Mb** just for the first **10 million rows**, and 
# 
# `train_df =  pd.read_csv('../input/train.csv')`
# 
# will consume more than **3Gb** of memory for entire **55 million rows**!

# In[ ]:


# save both training and test data to feather format
os.makedirs('tmp', exist_ok=True)
train_df.to_feather('tmp/taxi-train-no-gps.feather')

test_df = pd.read_csv('../input/test.csv', parse_dates=["pickup_datetime"], usecols=cols_test,
                         infer_datetime_format=True, dtype=types)
test_df.to_feather('tmp/taxi-test-no-gps.feather')


# # Load data feather format
# (begin from here directly once we have the feather format file)

# In[ ]:


get_ipython().run_cell_magic('time', '', "train_df = pd.read_feather('tmp/taxi-train-no-gps.feather')")


# Notice it took only a few seconds to load the entire dataframe.

# In[ ]:


test_df = pd.read_feather('tmp/taxi-test-no-gps.feather')


# In[ ]:


display(train_df.head())
display(train_df.tail())


# In[ ]:


display(test_df.head())
display(test_df.tail())


# # Group and aggregate by year-month

# In[ ]:


# Group all training data rides by month+year combination and aggregate them by count, mean, median, min and max
fare_grouped_df = train_df.fare_amount.groupby([train_df.pickup_datetime.dt.year,train_df.pickup_datetime.dt.month])

fare_count_df = fare_grouped_df.count()
fare_mean_df = fare_grouped_df.mean()
fare_median_df = fare_grouped_df.median()
fare_min_df = fare_grouped_df.min()
fare_max_df = fare_grouped_df.max()


# In[ ]:


# for drawing barchart, so we don't have to retype the following everytime
def draw_barchart(df, title):
    fig = plt.figure(figsize=(20, 4))
    ax = fig.add_subplot(111)
    df.plot(kind='bar')
    ax.set_xlabel("(Year, Month)")
    plt.xticks(rotation=60) 
    plt.title(title)
    plt.show()


# ## Monthly ride count

# This section is derived from this [kernel](https://www.kaggle.com/akosciansky/using-ml-for-data-exploration-feat-engineering) by @akosciansky

# In[ ]:


fare_count_df.describe()


# We also do the aggregate by count for test data, since we are only concern with the **count** of rides here, not the **fare**. We can then compare this with training data.

# In[ ]:


test_group_df = test_df.pickup_datetime.groupby([test_df.pickup_datetime.dt.year,test_df.pickup_datetime.dt.month]).count()


# In[ ]:


draw_barchart(fare_count_df, 'Monthly total rides of training data from January 2009 to June 2015')
draw_barchart(test_group_df, 'Monthly total rides of test data from January 2009 to June 2015')


# While training data barchart has a relative uniform monthly total rides, the test data monthly total rides are not spread evenly, with some random huge spikes. We might need to consider this when we create our validation set.

# ## Monthly mean fare

# In[ ]:


fare_mean_df.describe()


# In[ ]:


draw_barchart(fare_mean_df,'Monthly mean of fare from January 2009 to June 2015')


# From January 2009 until September 2012, the monthly mean fare is quite uniform. 
# 
# There is a gradual increase from January until December before it dropped again on January of following year.
# 
# However, there is a big jump from October 2012 onwards. The monthly mean fare increased by about \$3 during this period. Probably a price increase at this point onwards.
# 
# The pattern where the monthly mean fare drops from December to January remains true during this period.

# ## Monthly median fare

# In[ ]:


fare_median_df.describe()


# In[ ]:


draw_barchart(fare_median_df,'Monthly median of fares from January 2009 to June 2015')


# Here we can see the median is relatively uniform from January 2009 to September 2012
# 
# Not surprisingly, as we've seen in the monthly mean fare bar chart, there is a big jump from October 2012 onwards. 
# 
# The monthly median fare increased permanently by about almost \$2.00 during this period of September 2012 to June 2015.
# 
# A quick Google search verifies this:
# 
# https://www.nytimes.com/2012/09/04/nyregion/new-york-taxis-to-start-charging-increased-rates.html
# 
# One idea is we can create a boolean column based on the month and year to indicate whether it is before or after the price increase. This will help our model to learn better.

# ## Monthly minimum fare

# In[ ]:


fare_min_df.describe()


# The 25% percentile, 50% percentile (median) and 75% percentile all shows \$2.5. Which is probably the real minimum fare.
# 
# Most people just assume the minimum fare of \$0 and drop the negative fares. We need to dig deeper.

# In[ ]:


fare_min_df.mode()


# Mode value is also \$2.50

# In[ ]:


# take note that we are checking the fare amount in the original dataframe, not the aggregated one
print(f"Number of rides below $0.00: \t\t{len(train_df.fare_amount[train_df.fare_amount<0])}")
print(f"Number of rides at $0.00: \t\t{len(train_df.fare_amount[train_df.fare_amount==0])}")
print(f"Number of rides between $0.01 & $2.49:  {len(train_df.fare_amount[train_df.fare_amount.between(0,2.50,inclusive=False)])}")
print()
print(f"Number of rides below $2.50(all above): {len(train_df.fare_amount[train_df.fare_amount<2.5])}")
print()
print(f"Number of rides at $2.50: \t\t{len(train_df.fare_amount[train_df.fare_amount==2.5])}")
print(f"Number of rides of more than $2.50: \t{len(train_df.fare_amount[train_df.fare_amount>2.5])}")


# There are 2,454 cases of negative fare, 1,380 cases of zero fares and 913 cases more than \$0 but below \$2.50. 
# 
# In total, these three categories add up to 4,747 cases.
# 
# There are 224,309 cases at \$2.50 in comparison.
# 
# From above we can see above that \$2.50 would be a more reasonable minimum fare, as can be confirmed below:
# 
# http://www.nyc.gov/html/tlc/html/passenger/taxicab_rate.shtml
# 
# Hence we should remove those below **\$2.50** as outliers, instead of just dropping rows with negative fares.

# In[ ]:


draw_barchart(fare_min_df, 'Monthly minimum of fares from January 2009 to June 2015')


# Many minimum fare outliers fall in simply a few periods, such as February 2010, March 2010, August 2013, and January to June of 2015. 
# 
# Removing all rows with fare below \$.250 will remove those outliers.

# ## Monthly maximum fare

# In[ ]:


fare_max_df.describe()


# There are obviously some extreme values we need to get rid of before the statistics would make any sense.
# 
# The mean value of \$3526 for taxi ride is absurd.
# 
# Unlike minimum fare, we have no way of working out the optimal value for maximum fare.
# 
# Obviously, this is where the dependency with pickup and dropoff coordinates matters more - from where, to where, and how far you travel determines how much is the fare, instead of the date and time when you travel.
# 
# We rely more on visualization help from barchart.

# In[ ]:


draw_barchart(fare_max_df, 'Monthly maximum fares from January 2009 to June 2015')


# Luckily the chart above points shows us very clearly the few outliers that we should remove immediately.
# 
# Quite similar to the monthly minimum fare chart, there are outliers in the same periods, namely February 2010, 
# March 2010, August 2013, and January to June of 2015 (more apparent in February and May).
# 
# We can sort the maximum monthly fare for easier observation.

# In[ ]:


sorted_max_fare = fare_max_df.sort_values(ascending=False)
draw_barchart(sorted_max_fare.head(30), 'Sorted monthly maximum fares')


# Seem that we only need to remove the first 9 elements in the list above. We can confirm by printing out the values.

# In[ ]:


sorted_max_fare.head(30)


# From the sorted values above, the 9th element has value of \$900. Thus we can safely remove those above **\$500.00** as outliers at this point.
# 
# This is better than just picking maximum value by gut feeling.
# 
# The maximum values of \$500.00 for taxi fare is still quite mind-boggling, but this is because we haven't considered the pickup-dropoff coordinates. It's most likely there are some extremely long taxi rides that we need to check.

# # Summary Findings

# 1. By selectively choosing the columns/features during CSV file loading, we can reduce the memory usage of reading the **entire csv files of 55 million rows from 3Gb++ to less than 700Mb**.
# 1. By saving the raw dataframe to feather format, we can reload the entire dataframe in subsequent session much faster from a matter of **minutes to seconds**. There is no need to read the CSV file again.
# 1. Test data distribution is quite **different** from training data in terms of monthly total taxi rides. This is not my discovery, I derived from this [kernel](https://www.kaggle.com/akosciansky/using-ml-for-data-exploration-feat-engineering). This could affect how we create our validation data in order to maximize our test data score.
# 1. There is a jump in taxi fare by about \$2 to \$3 from September 2012. This matches the news report of [New York Taxis to Start Charging Increased Rates](https://www.nytimes.com/2012/09/04/nyregion/new-york-taxis-to-start-charging-increased-rates.html). We can create a boolean column to indicate whether the taxi charge is before or after the rate increase by checking the month and year.
# 1. From monthly minimum fare, there is a high occurence of **\$2.50**, which is the real minimum fare for NYC taxi ride.  Using this value is better than just simply removing rows with negative fare.
# 1. From monthly maximum fare, **\$500** is the reasonable cut-off point to remove outliers. To bring down this maximum value, we need to study the pickup and dropoff coordinates for both training and test data.
# 1. The outliers of maximum and minimum fare almost fall in the same periods of time - February 2010, March 2010, August 2013, and January to June of 2015 (more apparent in February and May).

# # Todo

# 1. Remove outliers based on the findings above. See how much outliers we removed.
# 1. Revisit all the monthly group and aggregates after the previous step, see if we have cleaner data.
# 1. Explore other parts of datetime to see how is the fare affected by week of the year, day of the week and time of the day.
# 1. Consider adding a holiday or calendar events column to see if holiday or other events affect the taxi ride count.
