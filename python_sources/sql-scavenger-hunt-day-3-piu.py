#!/usr/bin/env python
# coding: utf-8

# # Importing Packages and Datasets
# 
# 

# In[ ]:


# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")


# In[ ]:


# Listing tables
accidents.list_tables()


# # Scavenger hunt
# 
# ## Question 1 :
# 
# **Which hours of the day do the most accidents occur during?**
#  
#  Return a table that has information on how many accidents occurred in each hour of the day in 2015, sorted by the the number of accidents which occurred each day. Use the accident_2015 or accident_2016 table for this, and the timestamp_of_crash column. (Yes, there is an hour_of_crash column, but if you use that one you won't get a chance to practice with dates. :P)
#  ****
#     * **Hint:** You will probably want to use the [HOUR() function](https://cloud.google.com/bigquery/docs/reference/legacy-sql#hour) for this.

# In[ ]:


# Answer to Question 1

query_1 = """
                SELECT COUNT(consecutive_number) AS No_of_Accidents,
                EXTRACT(HOUR FROM timestamp_of_crash) AS Time_of_Crash
                FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
                GROUP BY Time_of_Crash
                ORDER BY No_of_Accidents DESC
         """

worst_hours = accidents.query_to_pandas_safe(query_1)

worst_hours.head()


# 'Time_of_Crash' indicates a 24-Hour Clock. So, according to the output produced by the above query, most accidents are reported between 17 hrs to 21 hrs (i.e. 5 PM - 9 PM), the highest being reported at 18:00 hrs or 6:00 PM.

# # Scavenger hunt
# 
# ## Question 2 :
# 
# **Which state has the most hit and runs?**
# 
# Return a table with the number of vehicles registered in each state that were involved in hit-and-run accidents, sorted by the number of hi Use the vehicle_2015 or vehicle_2016 table for this, especially the registration_state_name and hit_and_run columns.

# In[ ]:


# Answer to Question 2

query_2 = """
                SELECT registration_state_name, hit_and_run, COUNT(hit_and_run) AS No_of_hit_and_Run
                FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
                GROUP BY registration_state_name, hit_and_run
                HAVING hit_and_run = 'Yes'
                ORDER BY COUNT(hit_and_run) DESC
         """

worst_state = accidents.query_to_pandas_safe(query_2)

worst_state.head()


# Hence, from the above output we can see that the maximum no. of hit and run goes unreported/unknown. Then comes California, followed by Florida, Texas and New York.

# In[ ]:




