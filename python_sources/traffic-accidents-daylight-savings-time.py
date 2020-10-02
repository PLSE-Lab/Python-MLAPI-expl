#!/usr/bin/env python
# coding: utf-8

# The older I get the more the time change into & out of daylight savings time messes with me. I was curious if there were more accidents on the mondays following the two time shifts.

# In[4]:


# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")


# In[57]:


# query to find out the number of accidents
query = """SELECT COUNT(consecutive_number), 
                  EXTRACT(MONTH from timestamp_of_crash) AS MONTH,
                  EXTRACT(DAY from timestamp_of_crash) AS DAY,
                  EXTRACT(DAYOFWEEK from timestamp_of_crash) AS WEEKDAY
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY MONTH, DAY, WEEKDAY
            ORDER BY COUNT(consecutive_number) DESC
                
        """


# In[58]:


# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
accidents_by_day = accidents.query_to_pandas_safe(query)
accidents_by_day.head()


# In[82]:


# library for plotting
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# plot by # of accidents each day of the week
plt.bar(accidents_by_day.WEEKDAY, accidents_by_day.f0_)
plt.title("Number of Accidents by Weekday")


# In[59]:


# in 2015, daylight sacings started on Sunday, March 8 & ended on Sunday, November 1
# here we get the following mondays
accidents_by_day[(accidents_by_day["MONTH"] == 3) & 
                 (accidents_by_day["DAY"] == 9)].\
    append(accidents_by_day[(accidents_by_day["MONTH"] == 11) &
                           (accidents_by_day["DAY"] == 2)])


# In[83]:


# now get all mondays (since day of week is important to #
# of crashes, as we can see above)
all_mondays = accidents_by_day[accidents_by_day["WEEKDAY"] == 2]

# get the # of accidents for middle 50% of mondays
all_mondays.quantile([.25, .75])


# So based on this very small analysis, both Mondays following a time shift for daylight savings time had a pretty hight number of accidents; they were more dangerous than > 75% of all mondays. It's not a big sample so we can't capture the varience well, but it patterns with the findings from [this study](https://www.ncbi.nlm.nih.gov/pubmed/11152980) done in 2001. 
