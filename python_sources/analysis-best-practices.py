#!/usr/bin/env python
# coding: utf-8

# # Introduction
# ### We need to ask the right questions for analyze well
# ### And we need to use the pandas library well
# ### This kernel about correct questions, pandas functions and easy plot

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Stanford Open Policing Project

# In[ ]:


data = pd.read_csv("/kaggle/input/police/police.csv")


# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


data.dtypes


# In[ ]:


data.isnull().sum()


# ## Remove the column that only contains missing values
# ### Lessons:
# * Pay attention to default arguments
# * Check your work
# * There is more than one way to do everything in pandas

# In[ ]:


data.drop("county_name",axis = 1,inplace = True)


# In[ ]:


data.columns


# In[ ]:


# alternative method
data.dropna(axis = "columns", how = "all")


# ## Do men or women speed more often?  
# ### Lessons: There is more than one way to understand a question

# In[ ]:


data[data.violation == "Speeding"].driver_gender.value_counts(normalize=True)


# In[ ]:


data[data.driver_gender == "M"].violation.value_counts(normalize = True)


# In[ ]:


data[data.driver_gender == "F"].violation.value_counts(normalize = True)


# In[ ]:


data.groupby("driver_gender").violation.value_counts(normalize = True)


# ## Does gender affect who gets searched during a stop?   
# ### Lenssons:
# * Causation is difficult to conclude, so focus on relationship
# * Include all relevant factors when studying a relationship

# In[ ]:


data.search_conducted.value_counts(normalize = True)


# In[ ]:


data.search_conducted.mean()


# In[ ]:


data.groupby("driver_gender").search_conducted.mean()


# In[ ]:


data.groupby(["violation","driver_gender"]).search_conducted.mean()


# ## Why is search_type missing so often?
# ### Lessons:
# * Verify your assumptions about your data
# * pandas functions ignore missing values by default

# In[ ]:


data.isnull().sum()


# In[ ]:


data.search_conducted.value_counts()


# In[ ]:


data.search_type.value_counts(dropna = False)


# ## During a search, how often is the driver frisked?
# ### Lessons:
# * Use string methods to find partial matches
# * Use the correct denominator when calculating rates
# * pandas calculations ignore missing values
# * Apply the "smell test" to your results

# In[ ]:


data["frisk"] = data.search_type.str.contains("Protective Frisk")


# In[ ]:


data.frisk.value_counts(dropna = False)


# In[ ]:


data.frisk.sum()


# In[ ]:


data.frisk.mean()


# ## Which year had the least number of stops?
# ### Lessons:
# * Consider removing chunks of data that may be biased
# * Use the datetime data type for dates and times

# In[ ]:


data.stop_date.str.slice(0, 4).value_counts()


# In[ ]:


combined =data.stop_date.str.cat(data.stop_time, sep = " ")
data["stop_datetime"] = pd.to_datetime(combined)


# In[ ]:


data.stop_datetime.dt.year.value_counts()


# ## How does drug activity change by time of day?
# ### Lessons:
# * Use plots to help you understand trends
# * Create exploratory plots using pandas one-liners

# In[ ]:


data.drugs_related_stop.dtype


# In[ ]:


# line plot by default (for a Series)
data.groupby(data.stop_datetime.dt.hour).drugs_related_stop.mean().plot()


# In[ ]:


# alternative: count drug-related stops by hour
data.groupby(data.stop_datetime.dt.hour).drugs_related_stop.sum().plot()


# ## Do most stops occur at night?
# ### Lesson:
# *  Be conscious of sorting when plotting

# In[ ]:


data.stop_datetime.dt.hour.value_counts().plot()


# In[ ]:


data.stop_datetime.dt.hour.value_counts().sort_index().plot()


# ## Find the bad data in the stop_duration column and fix it
#  ### Lessons:
# *  Ambiguous data should be marked as missing
# *  Don't ignore the SettingWithCopyWarning
# *  NaN is not a string

# In[ ]:


data.stop_duration.value_counts(dropna = False)


# In[ ]:


data[(data.stop_duration == "1")|(data.stop_duration == "2")].stop_duration = "Nan"


# In[ ]:


# assignment statement did not work
data.stop_duration.value_counts()


# In[ ]:


# solves SettingWithCopyWarning USE .loc
data.loc[(data.stop_duration == '1') | (data.stop_duration == '2'), 'stop_duration'] = 'NaN'


# In[ ]:


# confusing
data.stop_duration.value_counts(dropna=False)


# In[ ]:


# replace 'NaN' string with actual NaN value
import numpy as np
data.loc[data.stop_duration == 'NaN', 'stop_duration'] = np.nan


# In[ ]:


data.stop_duration.value_counts(dropna=False)


# In[ ]:


# alternative method
data.stop_duration.replace(['1', '2'], value=np.nan, inplace=True)


# ## What is the mean stop_duration for each violation_raw?
# ### Lessons:
# * Convert strings to numbers for analysis
# * Approximate when necessary
# * Use count with mean to looking for meaningless means

# In[ ]:


mapping = {'0-15 Min':8, '16-30 Min':23, '30+ Min':45}
data['stop_minutes'] = data.stop_duration.map(mapping)


# In[ ]:


# matches value_counts for stop_duration
data.stop_minutes.value_counts()


# In[ ]:


data.groupby('violation_raw').stop_minutes.mean()


# In[ ]:


data.groupby('violation_raw').stop_minutes.agg(['mean', 'count'])


# ## Plot the results of the first groupby from the previous exercise
# ### Lessons:
# * Don't use a line plot to compare categories
# * Be conscious of sorting and orientation when plotting

# In[ ]:


data.groupby('violation_raw').stop_minutes.mean().plot()


# In[ ]:


data.groupby('violation_raw').stop_minutes.mean().plot(kind='bar')


# In[ ]:


data.groupby('violation_raw').stop_minutes.mean().sort_values().plot(kind='barh')


# ## Compare the age distributions for each violation
# ### Lessons: 
# * Use histograms to show distributions
# * Be conscious of axes when using grouped plots

# In[ ]:


data.groupby('violation').driver_age.describe()


# In[ ]:


data.driver_age.plot(kind='hist')


# In[ ]:


data.driver_age.value_counts().sort_index().plot()


# In[ ]:


data.hist('driver_age', by='violation')
plt.show()


# In[ ]:


data.hist('driver_age', by='violation', sharex=True)
plt.show()


# In[ ]:


# this better then upside
data.hist('driver_age', by='violation', sharex=True, sharey=True)
plt.show()


# ## Pretend you don't have the driver_age column, and create it from driver_age_raw (and call it new_age)
# ### Lessons:
# * Don't assume that the head and tail are representative of the data
# * Columns with missing values may still have bad data (driver_age_raw)
# * Data cleaning sometimes involves guessing (driver_age)
# * Use histograms for a sanity check

# In[ ]:


data['new_age'] = data.stop_datetime.dt.year - data.driver_age_raw
data[['driver_age', 'new_age']].hist()
plt.show()


# In[ ]:


data[['driver_age', 'new_age']].describe()


# In[ ]:


data[(data.new_age < 15) | (data.new_age > 99)].shape


# In[ ]:


data.driver_age_raw.isnull().sum()


# In[ ]:


5621-5327


# In[ ]:


# driver_age_raw NOT MISSING, driver_age MISSING
data[(data.driver_age_raw.notnull()) & (data.driver_age.isnull())].head()


# In[ ]:


# set the ages outside that range as missing
data.loc[(data.new_age < 15) | (data.new_age > 99), 'new_age'] = np.nan
data.new_age.equals(data.driver_age)

