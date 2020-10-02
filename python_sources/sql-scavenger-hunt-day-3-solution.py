#!/usr/bin/env python
# coding: utf-8

# ## Example: Which day of the week do the most fatal motor accidents happen on?
# ___
# 
# Now we're ready to work through an example. Today, we're going to be using the US Traffic Fatality Records database, which contains information on traffic accidents in the US where at least one person died. (It's definitely a sad topic, but if we can understand this data and the trends in it we can use that information to help prevent additional accidents.)
# 
# First, just like yesterday, we need to get our environment set up. Since you already know how to look at schema information at this point, I'm going to let you do that on your own. 
# 
# > **Important note:** Make sure that you add the BigQuery dataset you're querying to your kernel. Otherwise you'll get 

# In[ ]:


# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")


# We're going to look at which day of the week the most fatal traffic accidents happen on. I'm going to get the count of the unique id's (in this table they're called "consecutive_number") as well as the day of the week for each accident. Then I'm going sort my table so that the days with the most accidents are on returned first.

# In[ ]:


# query to find out the number of accidents which 
# happen on each day of the week
query = """SELECT COUNT(consecutive_number), 
                  EXTRACT(DAYOFWEEK FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(DAYOFWEEK FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """


# Now that our query is ready, let's run it (safely!) and store the results in a dataframe: 

# In[ ]:


# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
accidents_by_day = accidents.query_to_pandas_safe(query)


# And that gives us a dataframe! Let's quickly plot our data to make sure that it's actually been sorted:

# In[ ]:


# library for plotting
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.plot(accidents_by_day.f0_)
plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)")


# Yep, our query was, in fact, returned sorted! Now let's take a quick peek to figure out which days are the most dangerous:

# In[ ]:


print(accidents_by_day)


# To map from the numbers returned for the day of the week (the second column) to the actual day, I consulted [the BigQuery documentation on the DAYOFWEEK function](https://cloud.google.com/bigquery/docs/reference/legacy-sql#dayofweek), which says that it returns "an integer between 1 (Sunday) and 7 (Saturday), inclusively". So we can tell, based on our query, that in 2015 most fatal motor accidents occur on Sunday and Saturday, while the fewest happen on Tuesday.

# # Scavenger hunt
# ___
# 
# Now it's your turn! Here are the questions I would like you to get the data to answer:
# 
# * Which hours of the day do the most accidents occur during?
#     * Return a table that has information on how many accidents occurred in each hour of the day in 2015, sorted by the the number of accidents which occurred each hour. Use either the accident_2015 or accident_2016 table for this, and the timestamp_of_crash column. (Yes, there is an hour_of_crash column, but if you use that one you won't get a chance to practice with dates. :P)
#     * **Hint:** You will probably want to use the [EXTRACT() function](https://cloud.google.com/bigquery/docs/reference/standard-sql/functions-and-operators#extract_1) for this.
# * Which state has the most hit and runs?
#     * Return a table with the number of vehicles registered in each state that were involved in hit-and-run accidents, sorted by the number of hit and runs. Use either the vehicle_2015 or vehicle_2016 table for this, especially the registration_state_name and hit_and_run columns.
# 
# In order to answer these questions, you can fork this notebook by hitting the blue "Fork Notebook" at the very top of this page (you may have to scroll up). "Forking" something is making a copy of it that you can edit on your own without changing the original.

# **Solution starts  here!**

# In[ ]:


# hours of the day with the most accidents

accidentHourQuery = """SELECT EXTRACT(HOUR FROM timestamp_of_crash) AS hour, 
                              COUNT(*) AS crashcount
                       FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
                       GROUP BY hour
                       ORDER BY crashcount DESC
                    """
accidentHours = accidents.query_to_pandas_safe(accidentHourQuery)


# In[ ]:


accidentHours


# If we sort accidentHours by hour, we can plot it:

# In[ ]:


sortedAccidentHours = accidentHours.sort_values("hour")
sortedAccidentHours.plot("hour","crashcount")


# As we can see from the informative graph, the early evening hours have the most accidents: 5-8PM have the most, which looks to correspond with evening rush hour traffic. Interestingly, morning rush hour traffic hours (8-10AM) have some of the lowest numbers of accidents. Something to follow up on!
# 
# Next:

# In[ ]:


# states with the most hit-and-runs

stateHitAndRunQuery = """SELECT registration_state_name AS state,
                                COUNT(hit_and_run) AS hitcount
                         FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
                         GROUP BY state
                         ORDER BY hitcount DESC
                    """

stateHitAndRuns = accidents.query_to_pandas_safe(stateHitAndRunQuery)


# In[ ]:


stateHitAndRuns.head(n = 10)


# Not so surprising that states with more people, for the most part, have more hit-and-runs. We did it! Hooray!
