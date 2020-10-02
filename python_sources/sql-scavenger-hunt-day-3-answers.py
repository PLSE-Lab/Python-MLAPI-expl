#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")


# # Scavenger hunt
# ___
# 
# Now it's your turn! Here are the questions I would like you to get the data to answer:
# 
# * Which hours of the day do the most accidents occur during?
#     * Return a table that has information on how many accidents occurred in each hour of the day in 2015, sorted by the the number of accidents which occurred each day. Use the accident_2015 or accident_2016 table for this, and the timestamp_of_crash column. (Yes, there is an hour_of_crash column, but if you use that one you won't get a chance to practice with dates. :P)
#     * **Hint:** You will probably want to use the [HOUR() function](https://cloud.google.com/bigquery/docs/reference/legacy-sql#hour) for this.
# * Which state has the most hit and runs?
#     * Return a table with the number of vehicles registered in each state that were involved in hit-and-run accidents, sorted by the number of hi Use the vehicle_2015 or vehicle_2016 table for this, especially the registration_state_name and hit_and_run columns.
# 
# In order to answer these questions, you can fork this notebook by hitting the blue "Fork Notebook" at the very top of this page (you may have to scroll up). "Forking" something is making a copy of it that you can edit on your own without changing the original.

# # Question 1
# ___
# This one is straightforward; we can simply use the same query used to get accidents by day but replacing `DAYOFWEEK` with `HOUR`.

# In[ ]:


query = """SELECT COUNT(consecutive_number), 
                  EXTRACT(HOUR FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """


# In[ ]:


accidents_by_hour = accidents.query_to_pandas_safe(query)


# In[ ]:


print(accidents_by_hour)


# It appears that more accidents occur at night than in the morning. Here's a graph visualizing accidents by time of day. 

# In[ ]:


accidents_by_hour.sort_values("f1_").plot.bar("f1_", "f0_")


# # Question 2
# ___
# This one is a little more complicated. We want to get the number of cars (`COUNT(consecutive_number)`) for each state (`GROUP BY registration_state_name`) that has been involved in a hit and run (`WHERE hit_and_run = "Yes"`) and then sort in descending order (`ORDER BY COUNT(consecutive_number) DESC`).

# In[ ]:


query = """SELECT COUNT(consecutive_number), registration_state_name
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
            WHERE hit_and_run = "Yes"
            GROUP BY registration_state_name
            ORDER BY COUNT(consecutive_number) DESC
        """


# In[ ]:


hit_and_run_vehicles = accidents.query_to_pandas_safe(query)


# In[ ]:


print(hit_and_run_vehicles.head())


# There's a lot of unknown registrations, which makes sense since these were vehicles that fled the scene of the accident. The real states with the most hit and runs are also states with the largest populations, so this isn't too surprising either. Here's a terribly ugly bar graph showing the data with the unknowns removed (I might fix this later).

# In[ ]:


hit_and_run_vehicles.tail(54).plot.bar("registration_state_name", "f0_")

