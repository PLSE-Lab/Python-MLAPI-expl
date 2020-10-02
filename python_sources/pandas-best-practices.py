#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Dataset: Stanford Open Policing Project 

# In[ ]:


# ri stands for Rhode Island
ri = pd.read_csv('../input/my-new-dataset/police.csv')


# In[ ]:


# what does each row represent?
ri.head()


# In[ ]:


# what do these numbers mean?
ri.shape


# In[ ]:


# what do these types mean?
ri.dtypes


# In[ ]:


# what are these counts? how does this work?
ri.isnull().sum()


# ## ||1. Remove the column that only contains missing values||

# In[ ]:


# axis=1 also works, inplace is False by default, inplace=True avoids assignment statement
ri.drop('county_name', axis='columns', inplace=True)


# In[ ]:


ri.shape


# In[ ]:


ri.columns


# In[ ]:


# Alternative method
#  ri.dropna(axis='columns', how='all')


# # Lessons:
# 
# * Pay attention to default arguments
# * Check your work
# * There is more than one way to do everything in pandas

# # ||2. Do men or women speed more often?||

# In[ ]:


# when someone is stopped for speeding, how often is it a man or woman?
ri[ri.violation == 'Speeding'].driver_gender.value_counts(normalize=True)


# In[ ]:


# Alternative method
ri.loc[ri.violation == 'Speeding', 'driver_gender'].value_counts(normalize=True)


# In[ ]:


# when a man is pulled over, how often is it for speeding?
ri[ri.driver_gender == 'M'].violation.value_counts(normalize=True)


# In[ ]:


# repeat for women
ri[ri.driver_gender == 'F'].violation.value_counts(normalize=True)


# In[ ]:


# combines the two lines above
ri.groupby('driver_gender').violation.value_counts(normalize=True)


# # What are some relevant facts that we don't know?
# 
# # Lessons:
#  
# * There is more than one way to understand a question

# ## 3. Does gender affect who gets searched during a stop?

# In[ ]:


# ignore gender for the moment
ri.search_conducted.value_counts(normalize=True)


# In[ ]:


# how does this work?
ri.search_conducted.mean()


# In[ ]:


# search rate by gender
ri.groupby('driver_gender').search_conducted.mean()


# In[ ]:


# include a second factor
ri.groupby(['violation', 'driver_gender']).search_conducted.mean()


# # Does this prove causation?
# 
# # Lessons:
# 
# * Causation is difficult to conclude, so focus on relationships
# * Include all relevant factors when studying a relationship

# # 4. Why is search_type missing so often?

# In[ ]:


ri.isnull().sum()


# In[ ]:


# maybe search_type is missing any time search_conducted is False?
ri.search_conducted.value_counts()


# In[ ]:


# test that theory, why is the Series empty?
ri[ri.search_conducted == False].search_type.value_counts()


# In[ ]:


# value_counts ignores missing values by default
ri[ri.search_conducted == False].search_type.value_counts(dropna=False)


# In[ ]:


# when search_conducted is True, search_type is never missing
ri[ri.search_conducted == True].search_type.value_counts(dropna=False)


# In[ ]:


# alternative method
ri[ri.search_conducted == True].search_type.isnull().sum()


# # Lessons:
# 
# * Verify your assumptions about your data
# * pandas functions ignore missing values by default

# # 5. During a search, how often is the driver frisked?

# In[ ]:


# multiple types are separated by commas
ri.search_type.value_counts(dropna=False)


# In[ ]:


# use bracket notation when creating a column
ri['frisk'] = ri.search_type == 'Protective Frisk'


# In[ ]:


ri.frisk.dtype


# In[ ]:


# includes exact matches only
ri.frisk.sum()


# In[ ]:


# is this the answer?
ri.frisk.mean()


# In[ ]:


# uses the wrong denominator (includes stops that didn't involve a search)
ri.frisk.value_counts()


# In[ ]:


# includes partial matches
ri['frisk'] = ri.search_type.str.contains('Protective Frisk')


# In[ ]:


# seems about right
ri.frisk.sum()


# In[ ]:


# frisk rate during a search
ri.frisk.mean()


# In[ ]:


# str.contains preserved missing values from search_type
ri.frisk.value_counts(dropna=False)


# # Lessons:
# 
# * Use string methods to find partial matches
# * Use the correct denominator when calculating rates
# * pandas calculations ignore missing values
# * Apply the "smell test" to your results

# # 6. Which year had the least number of stops?

# In[ ]:


# this works, but there's a better way
ri.stop_date.str.slice(0, 4).value_counts()


# In[ ]:


# make sure you create this column
combined = ri.stop_date.str.cat(ri.stop_time, sep=' ')
ri['stop_datetime'] = pd.to_datetime(combined)


# In[ ]:


ri.dtypes


# In[ ]:


# why is 2005 so much smaller?
ri.stop_datetime.dt.year.value_counts()


# # Lessons:
# 
# * Consider removing chunks of data that may be biased
# * Use the datetime data type for dates and times

# # 7. How does drug activity change by time of day?

# In[ ]:


ri.drugs_related_stop.dtype


# In[ ]:


# baseline rate
ri.drugs_related_stop.mean()


# In[ ]:


# can't groupby 'hour' unless you create it as a column
ri.groupby(ri.stop_datetime.dt.hour).drugs_related_stop.mean()


# In[ ]:


# line plot by default (for a Series)
ri.groupby(ri.stop_datetime.dt.hour).drugs_related_stop.mean().plot()


# In[ ]:


# alternative: count drug-related stops by hour
ri.groupby(ri.stop_datetime.dt.hour).drugs_related_stop.sum().plot()


# # Lessons:
# 
# * Use plots to help you understand trends
# * Create exploratory plots using pandas one-liners

# # 8. Do most stops occur at night?

# In[ ]:


ri.stop_datetime.dt.hour.value_counts()


# In[ ]:


ri.stop_datetime.dt.hour.value_counts().plot()


# In[ ]:


ri.stop_datetime.dt.hour.value_counts().sort_index().plot()


# In[ ]:


# alternative method
ri.groupby(ri.stop_datetime.dt.hour).stop_date.count().plot()


# # Lessons:
# 
# * Be conscious of sorting when plotting

# # 9. Find the bad data in the stop_duration column and fix it

# In[ ]:


# mark bad data as missing
ri.stop_duration.value_counts()


# In[ ]:


# what two things are still wrong with this code?
ri[(ri.stop_duration == '1') | (ri.stop_duration == '2')].stop_duration = 'NaN'


# In[ ]:


# assignment statement did not work
ri.stop_duration.value_counts()


# In[ ]:


# solves Setting With Copy Warning
ri.loc[(ri.stop_duration == '1') | (ri.stop_duration == '2'), 'stop_duration'] = 'NaN'


# In[ ]:


# confusing!
ri.stop_duration.value_counts(dropna=False)


# In[ ]:


# replace 'NaN' string with actual NaN value
import numpy as np
ri.loc[ri.stop_duration == 'NaN', 'stop_duration'] = np.nan


# In[ ]:


ri.stop_duration.value_counts(dropna=False)


# In[ ]:


# alternative method
#ri.stop_duration.replace(['1', '2'], value=np.nan, inplace=True)


# # Lessons:
# 
# * Ambiguous data should be marked as missing
# * Don't ignore the SettingWithCopyWarning
# * NaN is not a string

# # 10. What is the mean stop_duration for each violation_raw?

# In[ ]:


# make sure you create this column
mapping = {'0-15 Min':8, '16-30 Min':23, '30+ Min':45}
ri['stop_minutes'] = ri.stop_duration.map(mapping)


# In[ ]:


# matches value_counts for stop_duration
ri.stop_minutes.value_counts()


# In[ ]:


ri.groupby('violation_raw').stop_minutes.mean()


# In[ ]:


ri.groupby('violation_raw').stop_minutes.agg(['mean', 'count'])


# # Lessons:
# 
# * Convert strings to numbers for analysis
# * Approximate when necessary
# * Use count with mean to looking for meaningless means

#  # 11. Plot the results of the first groupby from the previous exercise

# In[ ]:


# what's wrong with this?
ri.groupby('violation_raw').stop_minutes.mean().plot()


# In[ ]:


# how could this be made better?
ri.groupby('violation_raw').stop_minutes.mean().plot(kind='bar')


# In[ ]:


ri.groupby('violation_raw').stop_minutes.mean().sort_values().plot(kind='barh')


# # Lessons:
# 
# * Don't use a line plot to compare categories
# * Be conscious of sorting and orientation when plotting

# # 12. Compare the age distributions for each violation
# 

# In[ ]:


# good first step
ri.groupby('violation').driver_age.describe()


# In[ ]:


# histograms are excellent for displaying distributions
ri.driver_age.plot(kind='hist')


# In[ ]:


# similar to a histogram
ri.driver_age.value_counts().sort_index().plot()


# In[ ]:


# can't use the plot method
ri.hist('driver_age', by='violation')


# In[ ]:


# what changed? how is this better or worse?
ri.hist('driver_age', by='violation', sharex=True)


# In[ ]:


# what changed? how is this better or worse?
ri.hist('driver_age', by='violation', sharex=True, sharey=True)


# # Lessons:
# 
# * Use histograms to show distributions
# * Be conscious of axes when using grouped plots

# # 13. Pretend you don't have the driver_age column, and create it from driver_age_raw (and call it new_age)

# In[ ]:


ri.head()


# In[ ]:


# appears to be year of stop_date minus driver_age_raw
ri.tail()


# In[ ]:


ri['new_age'] = ri.stop_datetime.dt.year - ri.driver_age_raw


# In[ ]:


# compare the distributions
ri[['driver_age', 'new_age']].hist()


# In[ ]:


# compare the summary statistics (focus on min and max)
ri[['driver_age', 'new_age']].describe()


# In[ ]:


# calculate how many ages are outside that range
ri[(ri.new_age < 15) | (ri.new_age > 99)].shape


# In[ ]:


# raw data given to the researchers
ri.driver_age_raw.isnull().sum()


# In[ ]:


# age computed by the researchers (has more missing values)
ri.driver_age.isnull().sum()


# In[ ]:


# what does this tell us? researchers set driver_age as missing if less than 15 or more than 99
5621-5327


# In[ ]:


# driver_age_raw NOT MISSING, driver_age MISSING
ri[(ri.driver_age_raw.notnull()) & (ri.driver_age.isnull())].head()


# In[ ]:


# set the ages outside that range as missing
ri.loc[(ri.new_age < 15) | (ri.new_age > 99), 'new_age'] = np.nan


# In[ ]:


ri.new_age.equals(ri.driver_age)


# # Lessons:
# 
# * Don't assume that the head and tail are representative of the data
# * Columns with missing values may still have bad data (driver_age_raw)
# * Data cleaning sometimes involves guessing (driver_age)
# * Use histograms for a sanity check
