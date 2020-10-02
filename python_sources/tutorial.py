#!/usr/bin/env python
# coding: utf-8

# # PyCon 2018: Using pandas for Better (and Worse) Data Science
# 
# ### GitHub repository: https://github.com/justmarkham/pycon-2018-tutorial

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
pd.__version__


# ## Dataset: Stanford Open Policing Project  ([video](https://www.youtube.com/watch?v=hl-TGI4550M&list=PL5-da3qGB5IBITZj_dYSFqnd_15JgqwA6&index=1))

# In[2]:


# ri stands for Rhode Island
ri = pd.read_csv('../input/police.csv')


# In[3]:


# what does each row represent?
ri.head()


# In[4]:


# what do these numbers mean?
ri.shape


# In[5]:


# what do these types mean?
ri.dtypes


# - What does NaN mean?
# - Why might a value be missing?
# - Why mark it as NaN? Why not mark it as a 0 or an empty string or a string saying "Unknown"?

# In[6]:


# what are these counts? how does this work?
ri.isnull().sum()


# In[7]:


(True == 1) and (False == 0)


# ## 1. Remove the column that only contains missing values ([video](https://www.youtube.com/watch?v=TW5RqdDBasg&list=PL5-da3qGB5IBITZj_dYSFqnd_15JgqwA6&index=2))

# In[8]:


# axis=1 also works, inplace is False by default, inplace=True avoids assignment statement
ri.drop('county_name', axis='columns', inplace=True)


# In[9]:


ri.shape


# In[10]:


ri.columns


# In[11]:


# alternative method
ri.dropna(axis='columns', how='all').shape


# Lessons:
# 
# - Pay attention to default arguments
# - Check your work
# - There is more than one way to do everything in pandas

# ## 2. Do men or women speed more often? ([video](https://www.youtube.com/watch?v=d0oBRIONOEw&list=PL5-da3qGB5IBITZj_dYSFqnd_15JgqwA6&index=3))

# In[12]:


# when someone is stopped for speeding, how often is it a man or woman?
ri[ri.violation == 'Speeding'].driver_gender.value_counts(normalize=True)


# In[13]:


# alternative
ri.loc[ri.violation == 'Speeding', 'driver_gender'].value_counts(normalize=True)


# In[14]:


# when a man is pulled over, how often is it for speeding?
ri[ri.driver_gender == 'M'].violation.value_counts(normalize=True)


# In[15]:


# repeat for women
ri[ri.driver_gender == 'F'].violation.value_counts(normalize=True)


# In[16]:


# combines the two lines above
ri.groupby('driver_gender').violation.value_counts(normalize=True)


# What are some relevant facts that we don't know?
# 
# Lessons:
# 
# - There is more than one way to understand a question

# ## 3. Does gender affect who gets searched during a stop? ([video](https://www.youtube.com/watch?v=WzpGq1X5U1M&list=PL5-da3qGB5IBITZj_dYSFqnd_15JgqwA6&index=4))

# In[17]:


# ignore gender for the moment
ri.search_conducted.value_counts(normalize=True)


# In[18]:


# how does this work?
ri.search_conducted.mean()


# In[19]:


# search rate by gender
ri.groupby('driver_gender').search_conducted.mean()


# Does this prove that gender affects who gets searched?

# In[20]:


# include a second factor
ri.groupby(['violation', 'driver_gender']).search_conducted.mean()


# Does this prove causation?
# 
# Lessons:
# 
# - Causation is difficult to conclude, so focus on relationships
# - Include all relevant factors when studying a relationship

# ## 4. Why is search_type missing so often? ([video](https://www.youtube.com/watch?v=3D6smaE9c-g&list=PL5-da3qGB5IBITZj_dYSFqnd_15JgqwA6&index=5))

# In[21]:


ri.isnull().sum()


# In[22]:


# maybe search_type is missing any time search_conducted is False?
ri.search_conducted.value_counts()


# In[23]:


# test that theory, why is the Series empty?
ri[ri.search_conducted == False].search_type.value_counts()


# In[24]:


# value_counts ignores missing values by default
ri[ri.search_conducted == False].search_type.value_counts(dropna=False)


# In[25]:


# when search_conducted is True, search_type is never missing
ri[ri.search_conducted == True].search_type.value_counts(dropna=False)


# In[26]:


# alternative
ri[ri.search_conducted == True].search_type.isnull().sum()


# Lessons:
# 
# - Verify your assumptions about your data
# - pandas functions ignore missing values by default

# ## 5. During a search, how often is the driver frisked? ([video](https://www.youtube.com/watch?v=4tTO_xH4aQE&list=PL5-da3qGB5IBITZj_dYSFqnd_15JgqwA6&index=6))

# In[27]:


# multiple types are separated by commas
ri.search_type.value_counts(dropna=False)


# In[28]:


# use bracket notation when creating a column
ri['frisk'] = ri.search_type == 'Protective Frisk'


# In[29]:


ri.frisk.dtype


# In[30]:


# includes exact matches only
ri.frisk.sum()


# In[31]:


# is this the answer?
ri.frisk.mean()


# In[32]:


# uses the wrong denominator (includes stops that didn't involve a search)
ri.frisk.value_counts()


# In[33]:


161 / (91580 + 161)


# In[34]:


# includes partial matches
ri['frisk'] = ri.search_type.str.contains('Protective Frisk')


# In[35]:


# seems about right
ri.frisk.sum()


# In[36]:


# frisk rate during a search
ri.frisk.mean()


# In[37]:


# str.contains preserved missing values from search_type
ri.frisk.value_counts(dropna=False)


# In[38]:


# excludes stops that didn't involve a search
274 / (2922 + 274)


# Lessons:
# 
# - Use string methods to find partial matches
# - Use the correct denominator when calculating rates
# - pandas calculations ignore missing values
# - Apply the "smell test" to your results

# ## 6. Which year had the least number of stops? ([video](https://www.youtube.com/watch?v=W0zGzXQmE7c&list=PL5-da3qGB5IBITZj_dYSFqnd_15JgqwA6&index=7))

# In[39]:


# this works, but there's a better way
ri.stop_date.str.slice(0, 4).value_counts()


# In[40]:


# make sure you create this column
combined = ri.stop_date.str.cat(ri.stop_time, sep=' ')
ri['stop_datetime'] = pd.to_datetime(combined)


# In[41]:


ri.dtypes


# In[42]:


# why is 2005 so much smaller?
ri.stop_datetime.dt.year.value_counts()


# Lessons:
# 
# - Consider removing chunks of data that may be biased
# - Use the datetime data type for dates and times

# ## 7. How does drug activity change by time of day? ([video](https://www.youtube.com/watch?v=jV24N7SPXEU&list=PL5-da3qGB5IBITZj_dYSFqnd_15JgqwA6&index=8))

# In[43]:


ri.drugs_related_stop.dtype


# In[44]:


# baseline rate
ri.drugs_related_stop.mean()


# In[45]:


# can't groupby 'hour' unless you create it as a column
ri.groupby(ri.stop_datetime.dt.hour).drugs_related_stop.mean()


# In[46]:


# line plot by default (for a Series)
ri.groupby(ri.stop_datetime.dt.hour).drugs_related_stop.mean().plot()


# In[47]:


# alternative: count drug-related stops by hour
ri.groupby(ri.stop_datetime.dt.hour).drugs_related_stop.sum().plot()


# Lessons:
# 
# - Use plots to help you understand trends
# - Create exploratory plots using pandas one-liners

# ## 8. Do most stops occur at night? ([video](https://www.youtube.com/watch?v=GsQ6x3pt2w4&list=PL5-da3qGB5IBITZj_dYSFqnd_15JgqwA6&index=9))

# In[48]:


ri.stop_datetime.dt.hour.value_counts()


# In[49]:


ri.stop_datetime.dt.hour.value_counts().plot()


# In[50]:


ri.stop_datetime.dt.hour.value_counts().sort_index().plot()


# In[51]:


# alternative method
ri.groupby(ri.stop_datetime.dt.hour).stop_date.count().plot()


# Lessons:
# 
# - Be conscious of sorting when plotting

# ## 9. Find the bad data in the stop_duration column and fix it ([video](https://www.youtube.com/watch?v=8U8ob9bXakY&list=PL5-da3qGB5IBITZj_dYSFqnd_15JgqwA6&index=10))

# In[52]:


# mark bad data as missing
ri.stop_duration.value_counts()


# In[53]:


# what four things are wrong with this code?
# ri[ri.stop_duration == 1 | ri.stop_duration == 2].stop_duration = 'NaN'


# In[54]:


# what two things are still wrong with this code?
ri[(ri.stop_duration == '1') | (ri.stop_duration == '2')].stop_duration = 'NaN'


# In[55]:


# assignment statement did not work
ri.stop_duration.value_counts()


# In[56]:


# solves SettingWithCopyWarning
ri.loc[(ri.stop_duration == '1') | (ri.stop_duration == '2'), 'stop_duration'] = 'NaN'


# In[57]:


# confusing!
ri.stop_duration.value_counts(dropna=False)


# In[58]:


# replace 'NaN' string with actual NaN value
import numpy as np
ri.loc[ri.stop_duration == 'NaN', 'stop_duration'] = np.nan


# In[59]:


ri.stop_duration.value_counts(dropna=False)


# In[60]:


# alternative method
ri.stop_duration.replace(['1', '2'], value=np.nan, inplace=True)


# Lessons:
# 
# - Ambiguous data should be marked as missing
# - Don't ignore the SettingWithCopyWarning
# - NaN is not a string

# ## 10. What is the mean stop_duration for each violation_raw?

# In[61]:


# make sure you create this column
mapping = {'0-15 Min':8, '16-30 Min':23, '30+ Min':45}
ri['stop_minutes'] = ri.stop_duration.map(mapping)


# In[62]:


# matches value_counts for stop_duration
ri.stop_minutes.value_counts()


# In[63]:


ri.groupby('violation_raw').stop_minutes.mean()


# In[64]:


ri.groupby('violation_raw').stop_minutes.agg(['mean', 'count'])


# Lessons:
# 
# - Convert strings to numbers for analysis
# - Approximate when necessary
# - Use count with mean to looking for meaningless means

# ## 11. Plot the results of the first groupby from the previous exercise

# In[65]:


# what's wrong with this?
ri.groupby('violation_raw').stop_minutes.mean().plot()


# In[66]:


# how could this be made better?
ri.groupby('violation_raw').stop_minutes.mean().plot(kind='bar')


# In[67]:


ri.groupby('violation_raw').stop_minutes.mean().sort_values().plot(kind='barh')


# Lessons:
# 
# - Don't use a line plot to compare categories
# - Be conscious of sorting and orientation when plotting

# ## 12. Compare the age distributions for each violation

# In[68]:


# good first step
ri.groupby('violation').driver_age.describe()


# In[69]:


# histograms are excellent for displaying distributions
ri.driver_age.plot(kind='hist')


# In[70]:


# similar to a histogram
ri.driver_age.value_counts().sort_index().plot()


# In[71]:


# can't use the plot method
ri.hist('driver_age', by='violation')


# In[72]:


# what changed? how is this better or worse?
ri.hist('driver_age', by='violation', sharex=True)


# In[73]:


# what changed? how is this better or worse?
ri.hist('driver_age', by='violation', sharex=True, sharey=True)


# Lessons:
# 
# - Use histograms to show distributions
# - Be conscious of axes when using grouped plots

# ## 13. Pretend you don't have the driver_age column, and create it from driver_age_raw (and call it new_age)

# In[74]:


ri.head()


# In[75]:


# appears to be year of stop_date minus driver_age_raw
ri.tail()


# In[76]:


ri['new_age'] = ri.stop_datetime.dt.year - ri.driver_age_raw


# In[77]:


# compare the distributions
ri[['driver_age', 'new_age']].hist()


# In[78]:


# compare the summary statistics (focus on min and max)
ri[['driver_age', 'new_age']].describe()


# In[79]:


# calculate how many ages are outside that range
ri[(ri.new_age < 15) | (ri.new_age > 99)].shape


# In[80]:


# raw data given to the researchers
ri.driver_age_raw.isnull().sum()


# In[81]:


# age computed by the researchers (has more missing values)
ri.driver_age.isnull().sum()


# In[82]:


# what does this tell us? researchers set driver_age as missing if less than 15 or more than 99
5621-5327


# In[83]:


# driver_age_raw NOT MISSING, driver_age MISSING
ri[(ri.driver_age_raw.notnull()) & (ri.driver_age.isnull())].head()


# In[84]:


# set the ages outside that range as missing
ri.loc[(ri.new_age < 15) | (ri.new_age > 99), 'new_age'] = np.nan


# In[85]:


ri.new_age.equals(ri.driver_age)


# Lessons:
# 
# - Don't assume that the head and tail are representative of the data
# - Columns with missing values may still have bad data (driver_age_raw)
# - Data cleaning sometimes involves guessing (driver_age)
# - Use histograms for a sanity check
