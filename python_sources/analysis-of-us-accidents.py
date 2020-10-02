#!/usr/bin/env python
# coding: utf-8

# # Analysis of accidents in USA in 2016
# 
# In this notebook I'm going to analyze the US Traffic Fatality Records database.
# 

# In[ ]:


#First of all we have to import all the libraries that we need and set up our database
import bq_helper
import matplotlib.pyplot as plt #this will help us to plot the datas

accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")


# **Where are the state where the most  accidents have happened in 2016?**
# 
# To answer this question I use the *accidents_2016* table and the *consecutive_number* field that uniquely identifies  each crash.

# In[ ]:


accidents_query =  """
                     SELECT COUNT(consecutive_number) AS crashes, state_name
                     FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016` 
                     GROUP BY state_name
                     ORDER BY crashes DESC                   
                   """
state_with_more_accidents = accidents.query_to_pandas_safe(accidents_query)


# In[ ]:



plt.bar( state_with_more_accidents.state_name.head(), state_with_more_accidents.crashes.head())
plt.ylabel("Number of accidents")
plt.title("States with more accidents in 2016")


# **What can we see? **
# 
# We can see that California, Texas,  and Florida take the top places with an average of accidents around 3000, we can also observe a huge gap with the other states: there's a 1500 difference!! 
# You should also consider that the other states that are not in this histogram have a lower number of accidents, because the query is ordered by decrasing number of accidents.
# 

# **What about the people involved in these accidents?**
# To find information about people involved in accidents we can use the *person_2016*  to get  a glimpse of their role.
#     
# For example, for each role how many people were involved?

# In[ ]:


role_query =  """
                SELECT person_type_name AS role , COUNT(person_type_name) as number
                FROM   `bigquery-public-data.nhtsa_traffic_fatalities.person_2016` 
                GROUP BY role
                ORDER BY number DESC
              """
roles = accidents.query_to_pandas_safe(role_query)


# In[ ]:


roles


# The first two places are taken by the drivers and passengers and it is pretty obvious, but the third place is interesting because a lot of pedestrians are involved. Maybe we should ask ourselves  what should we do to lower this number.
# 
# I'd like to have a deeper  view of this, so **let's see what are the states where more pedestrians are involved in accidents**.
# Remember the *Consecutive Number*  is unique across the tables

# In[ ]:


pedestrians_query= """
                    SELECT a.state_name, COUNT(a.consecutive_number) as people_involved
                    FROM  `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016` AS a
                    INNER JOIN `bigquery-public-data.nhtsa_traffic_fatalities.person_2016` AS p
                          ON a.consecutive_number = p.consecutive_number
                    WHERE p.person_type_name = 'Pedestrian'
                    GROUP BY a.state_name
                    ORDER BY people_involved DESC
                   """
pedestrians_per_state = accidents.query_to_pandas_safe(pedestrians_query)


# In[ ]:


plt.bar(pedestrians_per_state.state_name.head(), pedestrians_per_state.people_involved.head())
plt.ylabel("number of people involved")
plt.title("Pedestrians involved in accidents in 2016")


# We can see that North Carolina is no longer in our histogram and New York takes its place, but the other states are the same as before.
# 
# California and Texas had almost the same number of accidents in 2016 but there's a huge difference in the number of pedestrian involved, as we can see California has almost 200 more pedestrians involved in accidents than Texas.

# **What about  the drivers? **
# Let's take a look at the state of the drivers by looking at the   *distract_2016*  table, it gives us information about the driver's status.

# In[ ]:


status_query = """
                 SELECT driver_distracted_by_name AS status, COUNT(consecutive_number) AS total
                 FROM `bigquery-public-data.nhtsa_traffic_fatalities.distract_2016`
                 GROUP BY status
                 ORDER BY total DESC                 
               """
driver_status = accidents.query_to_pandas_safe(status_query)


# In[ ]:


driver_status.head()


# Most of the drivers were not distracted or didn't know if they were before the accident, but as we can see that in the reporter status the inattetion or distraction is the main cause.
# **Did the driver at least try to avoid the accidents?**
# We can answer  this question by looking at the *maneuver_2016* table

# In[ ]:


maneuver_query = """
                    SELECT driver_maneuvered_to_avoid_name AS maneuvered_to_avoid, 
                           COUNT(consecutive_number) AS number
                    FROM `bigquery-public-data.nhtsa_traffic_fatalities.maneuver_2016`
                    
                    GROUP BY maneuvered_to_avoid
                    ORDER BY number DESC
                 """
maneuvers = accidents.query_to_pandas_safe(maneuver_query)


# In[ ]:


maneuvers.head()


# In most of the cases we have no information, but without watching the *Not Reported* cases we can see that in the majority of cases drivers didn't maneuver to avoid the accidents 

# **What is the month  in which  more accidents happened?**

# In[ ]:


month_query = """
                SELECT month_of_crash, COUNT(consecutive_number) as total
                FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
                GROUP BY month_of_crash
                ORDER BY total  DESC  
             """
accidents_per_month = accidents.query_to_pandas_safe(month_query)


# In[ ]:


plt.rcParams["figure.figsize"] = (20,10)
plt.bar(accidents_per_month.month_of_crash , accidents_per_month.total)
plt.title("accidents per month")


# We can see that the month with the most amount of accidents is october, but there's not a really significant difference between the months. Only in february and january we can see that the accidents decrease but not so much considering a rough average of 2800 accidents per month.

# In[ ]:


accidents_per_month.total.mean()  #Average of accidents per month


# # Conclusion
#   Thanks for you attention, this is  my first experience on Kaggle and with BigQuery.  I made it after the 5 day SQL Scavenger Hunt tutorial.  I hope you enjoyed it and feel free to tell me if you have any suggestions.

# 
