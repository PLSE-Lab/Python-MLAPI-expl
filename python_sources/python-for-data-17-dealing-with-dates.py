#!/usr/bin/env python
# coding: utf-8

# # Python for Data 17: Dealing With Dates
# [back to index](https://www.kaggle.com/hamelg/python-for-data-analysis-index)

# In the last two lessons, we learned a variety of methods to text character and numeric data, but many data sets also contain dates that don't fit nicely into either category. Common date formats contain numbers and sometimes text as well to specify months and days. Getting dates into a friendly format and extracting features of dates like month and year into new variables can be useful preprocessing steps.
# 
# For this lesson I've created and uploaded some dummy date data in a few different formats. Let's read in the date data:

# In[2]:


import numpy as np
import pandas as pd


# In[5]:


dates = pd.read_csv("../input/lesson-16-dates/dates_lesson_16.csv")

dates # Check the dates


# *Note the date data is listed as "lesson 16" instead of 17 because I used the same data set for lesson 16 of my intro to R guide.*
# 
# When you load data with Pandas, dates are typically loaded as strings by default. Let's check the type of data in each column:

# In[6]:


for col in dates:
    print (type(dates[col][1]))


# The output confirms that all the date data is currently in string form. To work with dates, we need to convert them from strings into a data format built for processing dates. The pandas library comes with a Timestamp data object for storing and working with dates. You can instruct pandas to automatically convert a date column in your data into Timestamps when you read your data by adding the "parse_dates" argument to the data reading function with a list of column indices indicated the columns you wish to convert to Timestamps. Let's re-read the data with parse_dates turned on for each column:

# In[14]:


dates = pd.read_csv("../input/lesson-16-dates/dates_lesson_16.csv", 
                    parse_dates=[0,1,2,3]) # Convert cols to Timestamp


# Now let's check the data types again:

# In[15]:


for col in dates:
    print (type(dates[col][1]))


# The output shows that all 4 columns were successfully parsed and translated into Timestamps. The default date parser works on many common date formats. You can also convert date strings to Timestamps using the function pd.to_datetime().
# 
# If you have oddly formatted date time objects, you might have to specify the exact format to get it to convert correctly into a Timestamp. For instance, consider a date format that gives date times of the form hour:minute:second year-day-month:

# In[19]:


odd_date = "12:30:15 2015-29-11"


# The default to_datetime parser will fail to convert this date because it expects dates in the form year-month-day. In cases like this, specify the date's format to convert it to Timestamp:

# In[20]:


pd.to_datetime(odd_date,
               format= "%H:%M:%S %Y-%d-%m") 


# As seen above, date formatting uses special formatting codes for each part of the date. For instance, %H represents hours and %Y represents the four digit year. View a list of formatting codes here.
# 
# Once you have your dates in the Timestamp format, you can extract a variety of properties like the year, month and day. Converting dates into several simpler features can make the data easier to analyze and use in predictive models. Access date properties from a Series of Timestamps with the syntax: Series.dt.property. To illustrate, let's extract some features from the first column of our date data and put them in a new DataFrame:

# In[21]:


column_1 = dates.iloc[:,0]

pd.DataFrame({"year": column_1.dt.year,
              "month": column_1.dt.month,
              "day": column_1.dt.day,
              "hour": column_1.dt.hour,
              "dayofyear": column_1.dt.dayofyear,
              "week": column_1.dt.week,
              "weekofyear": column_1.dt.weekofyear,
              "dayofweek": column_1.dt.dayofweek,
              "weekday": column_1.dt.weekday,
              "quarter": column_1.dt.quarter,
             })


# In addition to extracting date features, you can use the subtraction operator on Timestamp objects to determine the amount of time between two different dates:

# In[22]:


print(dates.iloc[1,0])
print(dates.iloc[3,0])
print(dates.iloc[3,0]-dates.iloc[1,0])


# Pandas includes a variety of more advanced date and time functionality beyond the basics covered in this lesson, particularly for dealing time series data (data consisting of many periodic measurements over time.). Read more about date and time functionality [here](http://pandas.pydata.org/pandas-docs/version/0.17.0/timeseries.html).

# ## Wrap Up

# Pandas makes it easy to convert date data into the Timestamp data format and extract basic date features like day of the year, month and day of week. Simple date features can be powerful predictors because data often exhibit cyclical patterns over different time scales.
# 
# Cleaning and preprocessing numeric, character and date data is sometimes all you need to do before you start a project. In some cases, however, your data may be split across several tables such as different worksheets in an excel file or different tables in a database. In these cases, you might have combine two tables together before proceeding with your project. In the next lesson, we'll explore how to merge data sets.

# ## Next Lesson: [Python for Data 18: Merging Data](https://www.kaggle.com/hamelg/python-for-data-18-merging-data)
# [back to index](https://www.kaggle.com/hamelg/python-for-data-analysis-index)
