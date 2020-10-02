#!/usr/bin/env python
# coding: utf-8

# # Scavenger hunt Day 3 answers
# 
# ## Which hours of the day do the most accidents occur during?
#    - Return a table that has information on how many accidents occurred in each hour of the day in 2015, sorted by the the number of accidents which occurred each day. Use the accident_2015 or accident_2016 table for this, and the timestamp_of_crash column. (Yes, there is an hour_of_crash column, but if you use that one you won't get a chance to practice with dates. :P)
#     - Hint: You will probably want to use the [HOUR() function](https://cloud.google.com/bigquery/docs/reference/legacy-sql#hour) for this.
# 
# 
# 

# In[ ]:


import bq_helper

accident_dat = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")


# In[ ]:


accident_dat.list_tables()


# In[ ]:


test_data = accident_dat.head('accident_2015')


# In[ ]:


test_data.columns


# In[ ]:


q1= """
    SELECT COUNT(consecutive_number), EXTRACT(HOUR FROM timestamp_of_crash)
    FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
    GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
    ORDER BY COUNT(consecutive_number)
    """


# In[ ]:


accident_dat.estimate_query_size(q1)


# In[ ]:


accidents_by_hour = accident_dat.query_to_pandas_safe(q1)


# In[ ]:


accidents_by_hour


# In[ ]:


accidents_by_hour.columns = ["num_accidents","hour_of_day"]


# In[ ]:


accidents_by_hour = accidents_by_hour.sort_values('hour_of_day', axis=0)


# Looks like a trend of a high number of accidents in the evening, highest numbers around evening rush hour -> first hours of darkness. An additional spike in accident number is observed in morning rush hour!

# In[ ]:


import matplotlib.pyplot as plt
plt.bar(accidents_by_hour['hour_of_day'], accidents_by_hour['num_accidents'])


# ## Which state has the most hit and runs?
#  -  Return a table with the number of vehicles registered in each state that were involved in hit-and-run accidents, sorted by the number of hi Use the vehicle_2015 or vehicle_2016 table for this, especially the registration_state_name and hit_and_run columns.

# In[ ]:


v_hickle =  accident_dat.head('vehicle_2015')


# In[ ]:


v_hickle['hit_and_run']


# In[ ]:


list(v_hickle.columns)


# The hit_and_run column is not a YES/NO boolean, italso has Unknowns, so I am going to group by the state and hit_and_run type and then process the results in pandas. Some of the Unknowns may be interesting interesting in their own right so I don't just want to keep the YES data

# In[ ]:


q2= """ SELECT registration_state_name, hit_and_run, COUNT(hit_and_run) 
        FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
        GROUP BY registration_state_name, hit_and_run
        ORDER BY COUNT(hit_and_run) DESC
    """


# In[ ]:


accident_dat.estimate_query_size(q2)


# In[ ]:


hit_and_run_state = accident_dat.query_to_pandas_safe(q2)


# In[ ]:


hit_and_run_state


# Then I just filter out the data for accidents know to not be hit and runs

# In[ ]:


hits_state = hit_and_run_state[hit_and_run_state['hit_and_run']!= "No"]


# Unsuprisingly, the highest number of hit and runs appear to have unknown registrations... that is a sad stat because it likely means the people involved in these hit and runs fled the scene without their car being identified :( 
# 
# The next four highest correspond with what I would reckon are the four largest states by population off the top of my head.

# In[ ]:


hits_state

