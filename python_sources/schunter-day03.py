#!/usr/bin/env python
# coding: utf-8

# Day three of the Scavenger Hunt and I am warming a bit up to Python though I still do not understand why I explicitely have to name both the BigQuery info, the Dataset and The Table in my SQL query when BigQuery info and Dataset is already defined in the handle.
# 
# Apart from that I have tried to acces the individual table descriptions in the .table_schema() call because it would be nice to compress the massive amount text returned from the schema call for better overview. Unfortunately I am not yet able to access the not-iterable/not listable/not dictionaryable SchemaField object and I do not yet seem to be able to get access to the [Information_Schema.Columns](https://cloud.google.com/spanner/docs/information-schema) table through BigQuery so there will be no list of the column contents for the different table. 
# 
# Presume that will change at some stage in the future. Anyways, as usual, my comments are mostly for my personal use even though I try to make them understandable :)
# 
# Thanks to [Rachel Ratman](https://www.kaggle.com/rtatman) for her [tutorial](https://www.kaggle.com/rtatman/sql-scavenger-hunt-day-3) and questions.
# 
# Overview
# 1. Get Python libraries
# 2. Examine the database
# 3. Create query Q1 (accidents and time of day) and check size
# 4. Run query Q1
# 5. Create query Q2 (state with most hit and runs) and check size
# 6. Run query Q2
# 7. Conclusion
# 
# **1. Get Python libraries**
# 

# In[ ]:


import numpy as np
import pandas as pd
from google.cloud import bigquery
from bq_helper import BigQueryHelper
#For plots
import matplotlib.pyplot as plt


# **2. Examine the database**
# 
# First I list the tables in the dataset, then I will look at the two interesint tables ACCIDENT_2016 and VEHICLE_2016

# In[ ]:


tf = BigQueryHelper('bigquery-public-data', 'nhtsa_traffic_fatalities')
tf_tables = tf.list_tables()
print("There are "+str(len(tf_tables))+" tables in the dataset")
print(tf_tables)


# In[ ]:


tf.table_schema('accident_2016')
#Please notice the amount of information returned from schema


# The last field TIMESTAMP_OF_CRASH with data registrered as TIMESTAMP is the field of interest in the ACCIDENT_2016 table and using filed CONSECUTIVE_NUMBER for our COUNT()

# In[ ]:


tf.table_schema('vehicle_2016')
#Please notice the amount of information returned from schema


# In the table VEHICHLE_2016 the interesting fields are HIT_AND_RUN with data registrered as
# 
# "It does not matter whether the hit-and-run vehicle was striking or struck. 0 No 1 Yes -- Not Reported 9 Unknown"
# 
# and REGISTRATION_STATE_NAME with data registrered as STRING

# **3. Create query Q1 (accidents and time of day) and check size**

# In[ ]:


sql1="""SELECT COUNT(consecutive_number), EXTRACT(hour FROM timestamp_of_crash) AS hour
    FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
    GROUP BY hour
    ORDER BY hour
    """
tf.estimate_query_size(sql1)


# ***4. Run query Q1***
# 
# There are no problems with the query size 0.5MB
# 
# We see that most accidents occur between 17-21 hours while there is a low at 4 oclock in the morning and again from 8 to 10. Seems logic as people are moving around a lot when they are off in the evening, and not moving around when sleeping and relaxing when they are at work or the kids have been brought to school.
# 

# In[ ]:


hours = tf.query_to_pandas_safe(sql1)
hours.shape


# In[ ]:


print(hours)
plt.plot(hours.f0_)
plt.xticks(hours.hour)
plt.title("Number of Accidents by hour of Day")


# **5. Create query Q2 (state with most hit and runs) and check size**
# 

# In[ ]:


sql2="""SELECT COUNTIF(hit_and_run!='0') as har, registration_state_name
    FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
    GROUP BY registration_state_name
    ORDER BY har DESC
    """
tf.estimate_query_size(sql2)


# **6. Run query Q2**
# 
# 63 rows are returned which are more than just US state names, reason is bureaucratic entries like "Not Reported", see bottom of list for more examples..
# 
# The top 3 states Texas, California, Florida all have more than twice the number of hit and runs than the fourth state North Carolina. The most safe state seems to be District of Columbia (disregarding Guam).

# In[ ]:


har = tf.query_to_pandas_safe(sql2)
har.shape


# In[ ]:


har.head(5)


# In[ ]:


har.tail(10)


# **7. Conclusion**
# 
# We see that most accidents occur between 17-21 hours while there is a low at 4 oclock in the morning and again from 8 to 10. Seems logic as people are moving around a lot when they are off in the evening, and not moving around when sleeping and relaxing when they are at work or the kids have been brought to school.
# 
# Concerning hit_and_run states 63 rows are returned which are more than just US state names, reason is bureaucratic entries like "Not Reported", see bottom of list for more examples.
# 
# The top 3 states Texas, California, Florida all have more than twice the number of hit and runs than the fourth state North Carolina. The most safe state seems to be District of Columbia (disregarding Guam).
