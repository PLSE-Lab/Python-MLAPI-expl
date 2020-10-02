#!/usr/bin/env python
# coding: utf-8

# # **Sunlight in Austin: Walking through pandas**
# 
# **Objective**
# 
# This is my first contribution to Kaggle Kernels and also one of my first complete project from a raw data set to insights by means of data visualizations. Sunlight in Austin is a case study from the "Pandas Foundations" chapter on [Data Camp](http://www.datacamp.com), a fantastic platform leading Data Science Education and where I found myself devoted on personal studies to aquire my python skills as well as the "data logical thinking". 
# Even consisting in a very simplistic data cleaning steps and basic visualizations, this study case answers pertinent questions regarding the working data sets and is a complete guide for begginers willing to understand how to make proper use and taking advantages of data series using pandas.
# 
# This notebook is structured as follows:
# 
# 1. Loading data
# 2. Time series index
# 2. Exploratory Data Analysis (EDA)
# 3. Statistical EDA
# 4. Data Visualization
# 5. Conclusions
# 
# Work in Progress...
# Estimated date for completition 16/12/2017

# # 1. Loading Data
#         - Dealing with messy data by using the right parameters of pd.read_csv()
#         - Selecting nem column labels using df.columns
#         - Excluding irrelevant attributes with df.drop()
#         - Analyzing data types with df.info()

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Preliminary step reading 2011_Austin_Weather as DataFrame and inspecting df.head()
df = pd.read_csv('../input/noaa-2011-austin-weather/2011_Austin_Weather.txt')
df.head()


# #### Dealing with messy data and selecting new column labels
# By analyzing  pd.head() we immediately notice that there is no header, and thus the columns don't have labels. There is also no obvious index column, since none of the data columns contain a full date or time. 
# 
# We start by indicating header='None' in pd.read_csv() and inspecting pd.head() once again. After this we are using the column_label.txt file with a .split() method to generate a list of strings that will be used as our new columns labels.

# In[ ]:


# Read the 2011_Austin_Weather.txt as a DataFrame attributing no header
df_headers = pd.read_csv('../input/noaa-2011-austin-weather/2011_Austin_Weather.txt', header=None)
df_headers.head()


# In[ ]:


# Open .txt file, read and split it 
# Attribute the splitted list to df.columns and inspect the new df.head()
with open('../input/column-label/column_labels.txt') as file:
    column_labels = file.read()
    column_labels_list = column_labels.split(',')
    df.columns = column_labels_list
df.head()


# #### Excluding irrelevant attributes and looking for data types
# As we can see from the previous df.head(), we are currently working with 44 columns. However for this analysis only a fraction from those are necessary, therefore we drop irrelevant attributes.

# In[ ]:


# Specify the list_to_drop with labels of columns to be dropped
# Drop the columns using df.drop() and inspecting the new df_dropped.head()
list_to_drop = ['sky_conditionFlag',
 'visibilityFlag',
 'wx_and_obst_to_vision',
 'wx_and_obst_to_visionFlag',
 'dry_bulb_farenFlag',
 'dry_bulb_celFlag',
 'wet_bulb_farenFlag',
 'wet_bulb_celFlag',
 'dew_point_farenFlag',
 'dew_point_celFlag',
 'relative_humidityFlag',
 'wind_speedFlag',
 'wind_directionFlag',
 'value_for_wind_character',
 'value_for_wind_characterFlag',
 'station_pressureFlag',
 'pressure_tendencyFlag',
 'pressure_tendency',
 'presschange',
 'presschangeFlag',
 'sea_level_pressureFlag',
 'hourly_precip',
 'hourly_precipFlag',
 'altimeter',
 'record_type',
 'altimeterFlag',
 'junk']
df_dropped = df.drop(list_to_drop, axis='columns')
df_dropped.head()


# #### Getting further .info()
# As shown below we now have a total of 17 columns that are interesting for our analysis. The method df.info() also brings valuable information regarding data types, as you can see both date and time are int64. Since our next step is generating the time series index those attributes will need to be concatenated, which means that we first need to convert their types.

# In[ ]:


# Analyze columns left and data tipes using df_dropped.info()
df_dropped.info()


# ## 2.  Time series index

# To concatenate date and time to work in a time series index we convert both tipes to string first

# In[ ]:


df_dropped['date'] = df_dropped['date'].astype(str)

# Pad leading zeros to the Time column: df_dropped['Time']
df_dropped['Time'] = df_dropped['Time'].apply(lambda x:'{:0>4}'.format(x))

# Checking that both attributes date and Time changed from int64 to object
df_dropped[['date', 'Time']].info()


# In[ ]:


# Concatenate the new date and Time columns: date_string
date_string = df_dropped.date + df_dropped.Time

# Convert the date_string Series to datetime: date_times
date_times = pd.to_datetime(date_string, format='%Y%m%d%H%M')

# Set the index to be the new date_times container: df_clean
df_clean = df_dropped.set_index(date_times)

# Print the output of df_clean.head()
df_clean.head()


# 
