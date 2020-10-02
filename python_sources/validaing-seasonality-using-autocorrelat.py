#!/usr/bin/env python
# coding: utf-8

# ### Validaing seasonality using Autocorrelation Function (ACF)

# This script provides an easy way to valdiate whether "time" feature in the dataset represents minute.
# The logic of ACF is simple: to see the correlation of a time series between its values at time t and time t-x.
# For example, we expect to see a quarterly series is highly correlated to 4 periods ago of itself.
# ACF is normally used to test stationary in a time series or determine coefficients in ARIMA time series model,
# but it is also extremely useful to determine/validate the seasonal period.
# 
# Thank you for the people who share their insights about "time" is minute,
# so I can validate it qucikly using ACF without further processing and testing.

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
from statsmodels.tsa.stattools import acf


# In[ ]:


df_train = pd.read_csv("../input/train.csv", dtype={"place_id": object})

# assume time is minute
df_train['hour'] = df_train.time // 60
df_train['day'] = df_train.time // (60*24)

# use the place id with the most checkins as an example
checkins_by_place = df_train.place_id.value_counts()
df_sample_place = df_train[df_train.place_id == checkins_by_place.index[0]]


# In[ ]:


# this function plots the ACF when you input frequency series with time index
def check_acf(counts_by_time):

    # fill in the gap with 0 to create a series with fixed intervals
    time_index = np.arange(
        counts_by_time.index.min(),
        counts_by_time.index.max() + 1)
    count_by_time_filled = counts_by_time.reindex(time_index)
    count_by_time_filled.fillna(0, inplace=True)

    # ACF
    acf_raw = acf(count_by_time_filled)
    
    # plot
    sns.barplot(x=np.arange(0, acf_raw.size), y=acf_raw)


# ### Check seasonality of day if we assume feature "time" is minute

# In[ ]:


# plot the ACF of "day"
check_acf(df_sample_place.day.value_counts())


# In the graph above, y is correlation, and peaks can be observed when x equals 7 and its multiples.
# That means the "day" series at time t is higly correlated with its values in 7 periods ago.
# This shows that "day" has a weekly seasonal period of 7,
# so our assumption where "time" is minute is probably correct.
# 
# The correlation is 1 when x=0 since the values at time t is perfectly correlated with itself at t-0.
# The correlation decays when x increases because trend exists in this time series,
# so the older values you used to compare with the base values, the less correlated they are.
# Sometimes the effect of trend may affect the seasonality heavily so that you cannot visualize the seasonal peaks well,
# in those cases, you may want to do differencing of the series to remove trending effect before plotting ACF,
# then a clear seasonal pattern can be observed.
# 
# 
# 
# 

# ### Check seasonality of hour if we assume feature "time" is minute

# In[ ]:


# plot the ACF of hour
check_acf(df_sample_place.hour.value_counts())


# Similarly, y is correlation, and peaks can be observed when x = 24.
# That means the "hour" series at time t is higly correlated with its values in 24 periods ago.
# This shows that "hour" has a hourly seasonal period of 24,
# so our assumption where "time" is minute is probably correct.
