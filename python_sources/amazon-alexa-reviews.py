#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[12]:


alexa_df = pd.read_csv('../input/amazon_alexa.tsv', sep = '\t')
#alexa_df = alexa_df[['date']]
alexa_df.head()


# **Convert to DATETIME and Date, Month, Year, and Quarter Extraction**

# In[13]:


# Convert the column into a date time object
alexa_df['date'] = pd.to_datetime(alexa_df['date'])

# Create a new 'year' feature by extracting the year value from date
alexa_df['year'] = alexa_df.date.dt.year

# Create a new 'month' feature by extracting the month value from date
alexa_df['month'] = alexa_df.date.dt.month

# Create a new 'day' feature by extracting the day value from date
alexa_df['day'] = alexa_df.date.dt.day

# Create a new 'Qtr' feature by extracting the monthly quarter from date
alexa_df['Qtr'] = alexa_df.date.dt.quarter


# In[14]:


np.unique(alexa_df["date"])


# In[15]:


# Examine new data frame
alexa_df.head()


# ## WhenEver You have Dates look for Holiday Season
# ### Pandas Have builtin Calendar Library

# In[16]:


# Create holiday dates
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
cal = calendar()
holidays = cal.holidays()

# Examine holiday dates
cal.holidays()


# In[17]:


# Create New Column 'Holiday'
alexa_df['Holiday'] = alexa_df['date'].isin(holidays)
alexa_df.tail()


# In[19]:


def calculate_xmas(date):
    xmas = pd.to_datetime(pd.Series('2018-12-25'))
    diff = xmas - date
    return diff

alexa_df['days_before_xmas'] = alexa_df.date.apply(calculate_xmas)


# In[20]:


alexa_df.head()


# In[ ]:




