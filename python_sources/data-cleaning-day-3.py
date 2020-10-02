#!/usr/bin/env python
# coding: utf-8

# # 5 Days of Data cleaning challenge.
# 
# ## Day 3 : Parsing dates
# 
# Here's what we're going to do today:
# 
# * [Get our environment set up](#1)
# * [Check the data type of our date column](#2)
# * [Convert our date columns to datetime](#3)
# * [Select just the day of the month from our column](#4)
# * [Plot the day of the month to check the date parsing](#5)
# 
# Let's get started!
# 
# * [Data for Landslides after rainfall](https://www.kaggle.com/nasa/landslide-events)
# * [Data for significant earthquakes](https://www.kaggle.com/usgs/earthquake-database)

# ## Get our environment setup<a id="1"></a>

# In[ ]:


import seaborn as sns
import datetime
import numpy as np # linear algebra
import pandas as pd # data processing

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

np.random.seed(0)


# In[ ]:


# importing data
landslides = pd.read_csv("/kaggle/input/landslide-events/catalog.csv")
earthquakes = pd.read_csv("/kaggle/input/earthquake-database/database.csv")


# In[ ]:


# first look at landslides dataset
print(landslides.columns)
landslides.head()


# In[ ]:


# first look at earthquakes dataset
print(earthquakes.columns)
earthquakes.head()


# ## Check the date type of our date column<a id="2"></a>
# Let's work with the `date` column from the `landslides` dataframe. The very first thing I'm going to do is take a peek at the first few rows to make sure it actually looks like it contains dates.

# In[ ]:


# for landslides

# print the first few rows of the date column
print(landslides['date'].head())


# Yep, those are the dates! But just because I, a human, can tell that these are dates doesn't mean that python knows that they're dates. Notice that the bottom of the output of `head()`, you can see it says that the data type of this column is "object"
# 
# Pandas uses the "object" dtype for storing various types of data types, but most often when you see a column with the dtype "object" it will have strings in it.
# 
# If you check the pandas dtype documentation [here](https://pandas.pydata.org/pandas-docs/stable/getting_started/basics.html), you'll notice that there's also a specific datetime64 dtypes. Because the dtype of our column is object rather than datetime64, we can tell that Python doesn't know that this column contains dates.
# 
# We can also look at just the dtype of your column without printing the first few rows if we like:

# In[ ]:


# check the data type for date column
landslides['date'].dtype


# Now , Let's see the date column from the `earthquakes` dataframe

# In[ ]:


# for earthquakes

# print the first few rows of the date column
print(earthquakes['Date'].head())


# ## Check the date column to datetime<a id="3"></a>
# Now that we know that our date column isn't being recognize as a date, it's time to convert it so that it is recognized as a date. This is called "parsing dates" because we're taking in a string and identifying it's component parts.
# 
# We can use oandas for the format of our dates with a guide called as [strftime directive](https://strftime.org/). The basic idea is that you need to point which parts of the date are where and what punctuation is between them. There are lots of possible paths of date, but the most comman are `%d` for date, `%m` for month, `%y` for a two-digit year and `%Y` for four-digit year.
# 
# Some examples:
# 
# * 1/17/07 has the format "%m/%d/%y"
# * 17-1-2007 has the format "%d-%m-%Y"
# 
# Looking back up at the head of the date column in the `landslides` dataset, we can see that it's in the format "month/day/two-digit year", so we can use the same syntax as the first example to parse in our dates:

# In[ ]:


# for landslides 

# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")


# Now when I check the first few rows of the new column, I can see that the dtype is `datetime64`. I can also see that my dates have been slightly rearranged so that they fit the default order datetime objects (year-month-day).

# In[ ]:


# print the first few rows
landslides['date_parsed'].head()


# Now that our dates are parsed correctly, we can interact with them in useful ways.
# 
# * **What if I run into an error with multiple date formats?** While we're specifying the date format here, sometimes you'll run into an error when there are multiple date formats in a single column. If that happens, you have have pandas try to infer what the right date format should be. You can do that like so:
# 
# landslides['date_parsed'] = pd.to_datetime(landslides['Date'], infer_datetime_format=True)
# 
# * **Why don't you always use infer_datetime_format = True?** There are two big reasons not to always have pandas guess the time format. The first is that pandas won't always been able to figure out the correct date format, especially if someone has gotten creative with data entry. The second is that it's much slower than specifying the exact format of the dates.
# 

# In[ ]:


# for earthquake

# create a new column, date_parsed, with the parsed dates

earthquakes["date_parsed"] = pd.to_datetime(earthquakes["Date"], format="%m/%d/%Y", errors="coerce")

invalid_date_index = earthquakes["date_parsed"][earthquakes["date_parsed"].isnull() == True].index.tolist()


# In[ ]:


# print the first few rows
earthquakes['date_parsed'].head()


# ## Select just the day of the month from our column<a id="4"></a>
# Let's try to get information on the day of the month that a landslide occured on from the original "date" column, which has an "object" dtype:

# In[ ]:


# for landslides

# try to get the day of the month from the date column
day_of_month_landslides = landslides['date'].dt.day


# We got an error! The important part to look at here is the part at the very end that says `AttributeError: Can only use .dt accessor with datetimelike values`. We're getting this error because the dt.day() function doesn't know how to deal with a column with the dtype "object". Even though our dataframe has dates in it, because they haven't been parsed we can't interact with them in a useful way.
# 
# Luckily, we have a column that we parsed earlier , and that lets us get the day of the month out no problem:

# In[ ]:


# for landslides

# get the day of the month from the date_parsed column 
day_of_month_landslides = landslides['date_parsed'].dt.day
day_of_month_landslides.head()


# In[ ]:


# for earthquakes

# get the day of the month from the date_parsed column 
day_of_month_earthquakes = earthquakes['date_parsed'].dt.day
day_of_month_earthquakes.head()


# ## Plot the day of the month to check the date parsing<a id="5"></a>
# One of the biggest dangers in parsing dates is mixing up the months and days. The to_datetime() function does have very helpful error messages, but it doesn't hurt to double-check that the days of the month we've extracted make sense.
# 
# To do this, let's plot a histogram of the days of the month. We expect it to have values between 1 and 31 and, since there's no reason to suppose the landslides are more common on some days of the month than others, a relatively even distribution. (With a dip on 31 because not all months have 31 days.) Let's see if that's the case:

# In[ ]:


# for landslides

# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
sns.distplot(day_of_month_landslides, kde = False, bins = 31)


# Yep, it looks like we did parse our dates correctly & this graph makes good sense to me. Why don't you take a turn checking the dates you parsed earlier?

# In[ ]:


# for earthquakes

#remove na's
day_of_month_earthquakes = day_of_month_earthquakes.dropna()

# plot the day of the month
sns.distplot(day_of_month_earthquakes, kde = False, bins = 31)

