#!/usr/bin/env python
# coding: utf-8

# # Data Measurement Frequency
# 
# It states in the competition: "Bonus points for smaller time slices of the average historical emissions factors, such as one per month for the 12-month period.", so it's important to use (and understand) data that has certain time scales.
# 
# In the provided datasets, what sort of measurement level are we provided with? Is it daily, weekly, yearly? Do we have consistent data or missing data?

# In[ ]:


import os

from datetime import datetime, timedelta

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

data_path = '/kaggle/input/ds4g-environmental-insights-explorer'


# ## Load Data
# 
# There are four provided datasets:
# 
# * Global Power Plant database by WRI
# * Sentinel 5P OFFL NO2 by EU/ESA/Copernicus
# * Global Forecast System 384-Hour Predicted Atmosphere Data by NOAA/NCEP/EMC
# * Global Land Data Assimilation System by NASA
# 

# In[ ]:


os.listdir(data_path + '/eie_data')


# For the GLDAS, GFS and S5P NO2 datasets, we can get the timing from the names.
# 
# They're formatted as: 
# 
# - gldas_20180901_1500.tif
# - gfs_2019030900.tif
# - s5p_no2_20181101T173802_20181107T192402
# 
# For s5p we will take the first date.

# In[ ]:


gldas_files = os.listdir(data_path + '/eie_data/gldas')
gldas_dates = [datetime.strptime(g, 'gldas_%Y%m%d_%H%M.tif') for g in gldas_files]

gfs_files = os.listdir(data_path + '/eie_data/gfs')
gfs_dates = [datetime.strptime(g, 'gfs_%Y%m%d%H.tif') for g in gfs_files]

s5p_files = os.listdir(data_path + '/eie_data/s5p_no2')
s5p_dates = [datetime.strptime(g[:16], 's5p_no2_%Y%m%d') for g in s5p_files]


# The GPPD dataset has just one csv. We can see in the columns that this is yearly data.

# In[ ]:


pd.read_csv(data_path + '/eie_data/gppd/gppd_120_pr.csv').columns


# Get all the dates in one dataframe.

# In[ ]:


all_dates = pd.DataFrame(columns=['dataset', 'datetime']).append(
    pd.DataFrame(gldas_dates, columns=['datetime']) \
        .assign(dataset = 'gldas'), sort=True
).append(
    pd.DataFrame(gfs_dates, columns=['datetime']) \
        .assign(dataset = 'gfs'), sort=True
).append(
    pd.DataFrame(s5p_dates, columns=['datetime']) \
        .assign(dataset = 's5p'), sort=True
).append(
    pd.DataFrame([datetime(y, 1, 1) for y in range(2013, 2018)], columns=['datetime']) \
    .assign(dataset = 'gppd'), sort=True
).assign(date = lambda x: x.datetime.apply(lambda x: x.date()))


# In[ ]:


all_dates


# ##  Analysis

# Some basic info, grouped by date:

# In[ ]:


all_dates.groupby('dataset').date.agg(
    min=min,
    max=max,
    measurement_period=  lambda x: (x.max()-x.min()).days+1,
    measurement_count= 'count',
    measurements_per_day= lambda x: x.count() / ((x.max()-x.min()).days+1)
).transpose()


# So GFS, GLDAS and S5P NO2 are one year's worth of data from July 2018 until end of June 2019, though S5P NO2 appears to be missing one day.
# 
# GPPD is yearly data from 2013 until 2017.
# 
# Straight away we see that the emissions data doesn't align with the activity data. We'll have to do some basic interpolation.
# 
# We also see that GFS and GLDAS appear to have a steady amount of 4 and 8 measurements per day, respectively while S5P NO2 has something more uncommon.

# ## Visualisation
# 
# We can now visualise these:

# In[ ]:


# Get the date index to work with
daily_data = pd.date_range('2018-07-01', '2019-06-30').to_frame()     .merge(all_dates.groupby(['dataset', 'date']).date.count().unstack(level=0),
        left_index=True,
        right_index=True) \
    .drop(columns=[0, 'gppd'], axis=1)

sns.heatmap(daily_data.transpose())
fig = plt.gcf()
fig.set_size_inches(11,3)


# As noticed above, we see that we have steady measurements for GFS and GLDAS, but S5P is erratic, ranging from 0 measurements to 4.

# ## Conclusion
# 
# Some problems we identified:
#  * the activity data (GPPD) doesn't overlap in the same timeframe as the emissions data (others)
#  * the data for S5P NO2 is not evenly spaced, and even has missing dates 
