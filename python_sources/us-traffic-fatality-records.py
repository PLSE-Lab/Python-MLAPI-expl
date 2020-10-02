#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")


# Then write the code to answer the questions below

# # Questions
# 
# #### 1) Which hours of the day do the most accidents occur during?
# * Return a table showing how many accidents occurred in each hour of the day in 2015, sorted by the the number of accidents which occurred each hour. Use either the accident_2015 or accident_2016 table for this, and the timestamp_of_crash column (there is an hour_of_crash column, but if you use that one you won't get a chance to practice with dates).
# 
# **Hint:** You will probably want to use the [EXTRACT() function](https://cloud.google.com/bigquery/docs/reference/standard-sql/functions-and-operators#extract_1) for this.
# 

# In[ ]:


accidents.list_tables()


# In[ ]:


accidents.head('accident_2015')


# In[ ]:


query_2015 = """ SELECT  COUNT(consecutive_number),
                         EXTRACT(HOUR FROM timestamp_of_crash)
                 FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
                 GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
                 ORDER BY COUNT(consecutive_number) DESC
             """ 

query_2016 = """ SELECT  COUNT(consecutive_number),
                         EXTRACT(HOUR FROM timestamp_of_crash)
                 FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
                 GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
                 ORDER BY COUNT(consecutive_number) DESC
            """ 


# In[ ]:


accident_hours_2015 = accidents.query_to_pandas_safe(query_2015)
accident_hours_2016 = accidents.query_to_pandas_safe(query_2016)


# In[ ]:


accident_hours_2015.head(7)


# In[ ]:


accident_hours_2016.head(7)


# To plot the dataframes lets first sort the values in f1_

# In[ ]:


sorted_for_plot_2015 = accident_hours_2015.sort_values('f1_')
sorted_for_plot_2016 = accident_hours_2016.sort_values('f1_')


# In[ ]:


import matplotlib.pyplot as plt

x = sorted_for_plot_2015.f1_
y = sorted_for_plot_2015.f0_

xx = sorted_for_plot_2016.f1_
yy = sorted_for_plot_2016.f0_

plt.figure(figsize=(10,5))
plt.plot(x,y, label='2015')
plt.plot(xx,yy, label='2016')
plt.legend()

plt.xlabel('hour of a day')
plt.title('number of accidents for each hour in a day \n in 2015 and 2016')

plt.grid()
plt.show()


# #### 2) Which state has the most hit and runs?
# * Return a table with the number of vehicles registered in each state that were involved in hit-and-run accidents, sorted by the number of hit and runs. Use either the vehicle_2015 or vehicle_2016 table for this, especially the registration_state_name and hit_and_run columns (it may be helpful to view the hit_and_run column to understand its contents).
# 

# In[ ]:


accidents.head('vehicle_2015',3)


# In[ ]:


query = """ SELECT  registration_state_name,
                    COUNT(registration_state_name)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run = 'Yes'
            GROUP BY registration_state_name
            ORDER BY COUNT(registration_state_name) DESC
        """


# In[ ]:


state_vehicle = accidents.query_to_pandas_safe(query)


# In[ ]:


state_vehicle.head(7)


# In[ ]:


import matplotlib.pyplot as plt
plt.figure(figsize=(15,6))
plt.plot(state_vehicle.registration_state_name, state_vehicle.f0_, marker = 'o')
plt.xticks(rotation=90)
plt.title('number of vehicles registered in each state that were involved in hit-and-run accidents')
plt.grid(color='b', linestyle='-', linewidth=0.5)
plt.show()


# In[ ]:


state_vehicle_dropped = state_vehicle.drop([0]) # lets drop unknown state names


# In[ ]:


import matplotlib.pyplot as plt
plt.figure(figsize=(15,6))
plt.plot(state_vehicle_dropped.registration_state_name, state_vehicle_dropped.f0_, marker = 'o')
plt.xticks(rotation=90)
plt.title('number of vehicles registered in each state that were involved in hit-and-run accidents')
plt.grid(color='b', linestyle='-', linewidth=0.5)
plt.show()


# ---
# # Keep Going
# [Click here](https://www.kaggle.com/dansbecker/as-with) to learn how *WITH-AS* clauses  can clean up your code and help you construct more complex queries.
# 
# # Feedback
# Bring any questions or feedback to the [Learn Discussion Forum](kaggle.com/learn-forum).
# 
# ----
# 
# *This tutorial is part of the [SQL Series](https://www.kaggle.com/learn/sql) on Kaggle Learn.*
