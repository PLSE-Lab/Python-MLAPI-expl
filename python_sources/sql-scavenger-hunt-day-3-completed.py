#!/usr/bin/env python
# coding: utf-8

# <table>
#     <tr>
#         <td>
#         <center>
#         <font size="+5">SQL Scavenger Hunt: Day 3.</font>
#         </center>
#         </td>
#     </tr>
# </table>

# In[ ]:


# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")


# In[ ]:


accidents.table_schema(table_name="accident_2015")


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

# # Scavenger hunt Questions ?
# 
# 
# * Which hours of the day do the most accidents occur during?
#     * Return a table that has information on how many accidents occurred in each hour of the day in 2015, sorted by the the number of accidents which occurred each hour. Use either the accident_2015 or accident_2016 table for this, and the timestamp_of_crash column. (Yes, there is an hour_of_crash column, but if you use that one you won't get a chance to practice with dates. :P)
#     * **Hint:** You will probably want to use the [EXTRACT() function](https://cloud.google.com/bigquery/docs/reference/standard-sql/functions-and-operators#extract_1) for this.
# * Which state has the most hit and runs?
#     * Return a table with the number of vehicles registered in each state that were involved in hit-and-run accidents, sorted by the number of hit and runs. Use either the vehicle_2015 or vehicle_2016 table for this, especially the registration_state_name and hit_and_run columns.
# 

# In[ ]:


# Your code goes# query to find out the number of accidents which 
# happen on each hour of the day
# Your code goes here :)
hour_query = """SELECT EXTRACT(HOUR FROM timestamp_of_crash) as Hour_of_Accident,
            COUNT(consecutive_number) as Accidents_Count
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY Hour_of_Accident
            ORDER BY Accidents_Count DESC
        """


# In[ ]:


# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
accidents_by_hour = accidents.query_to_pandas_safe(hour_query)


# In[ ]:


print(accidents_by_hour)


# In[ ]:


# library for plotting
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.plot(accidents_by_hour.Hour_of_Accident,accidents_by_hour.Accidents_Count)
plt.title("Number of Accidents by Rank of Hour \n (Most to least dangerous)")

