#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Set up feedback system
from learntools.core import binder
binder.bind(globals())
from learntools.sql.ex5 import *
print("Setup Complete")


# In[ ]:


from google.cloud import bigquery
import pandas as pd

# Create a "Client" object
client = bigquery.Client()

# Construct a reference to the "chicago_taxi_trips" dataset
dataset_ref = client.dataset("chicago_taxi_trips", project="bigquery-public-data")

# API request - fetch the dataset
dataset = client.get_dataset(dataset_ref)


# In[ ]:


table_ref = dataset_ref.table("taxi_trips")
table = client.get_table(table_ref)

client.list_rows(table, max_results=5).to_dataframe()


# In[ ]:


# Kaggle's solution with DATE
date_query = """
               WITH RelevantRides AS
               (
                   SELECT EXTRACT(HOUR FROM trip_start_timestamp) AS hour_of_day, 
                          trip_miles, 
                          trip_seconds
                   FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
                   WHERE trip_start_timestamp > '2017-01-01' AND 
                         trip_start_timestamp < '2017-01-02' AND 
                         trip_seconds > 0 AND 
                         trip_miles > 0
               )
               SELECT hour_of_day, 
                      COUNT(1) AS num_trips, 
                      3600 * SUM(trip_miles) / SUM(trip_seconds) AS avg_mph
               FROM RelevantRides
               GROUP BY hour_of_day
               ORDER BY hour_of_day
               """

# Set up the query (cancel the query if it would use too much of 
# your quota, with the limit set to 1 GB)
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=5*10**9)
date_query_job = client.query(date_query, job_config=safe_config)

# API request - run the query, and return a pandas DataFrame
date_result = date_query_job.to_dataframe()

# Prints sum of 70974
print(date_result["num_trips"].sum())


# In[ ]:


# My solution with timestamp
timestamp_query = """
               WITH RelevantRides AS
               (
                   SELECT EXTRACT(HOUR from trip_start_timestamp) AS hour_of_day, trip_miles, trip_seconds
                   
                   FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
                   WHERE EXTRACT(DATE from trip_start_timestamp) > '2017-01-01' and 
                         EXTRACT(DATE from trip_start_timestamp) < '2017-01-02' 
                   and trip_seconds > 0 and trip_miles > 0
               )
               SELECT hour_of_day, COUNT(hour_of_day) as num_trips, 3600 * SUM(trip_miles) / SUM(trip_seconds) AS avg_mph
               FROM RelevantRides
               GROUP BY hour_of_day
               ORDER BY hour_of_day
               """

# Set up the query (cancel the query if it would use too much of 
# your quota, with the limit set to 1 GB)
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=5*10**9)
timestamp_query_job = client.query(timestamp_query, job_config=safe_config)

# API request - run the query, and return a pandas DataFrame
timestamp_result = timestamp_query_job.to_dataframe()

# Returns Zero
print(timestamp_result["num_trips"].sum())


# The timestamp approach returns empty DataFrame for a single day, it does not count from the first day, so we are loosing 50277 values for the range between 2017-01-01 to 2017-01-02. This is exact the difference for our range of intererst in 2017.

# If results are taken somewhere in the middle of the range, the values obtained by two approach are the same

# In[ ]:


#Query for check, DATE approach

query1 = """SELECT trip_start_timestamp AS time_stamp, EXTRACT(HOUR from trip_start_timestamp) AS hour
            FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
            WHERE trip_seconds > 0 and trip_miles > 0 and
            trip_start_timestamp > '2017-01-01' AND 
            trip_start_timestamp < '2017-02-01' and
            EXTRACT(HOUR from trip_start_timestamp) = 5 and
            EXTRACT(DAY from trip_start_timestamp) = 13
            ORDER BY time_stamp
            """

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=5*10**9)
query1_job = client.query(query1, job_config=safe_config)

query_results = query1_job.to_dataframe()
print(query_results.tail(5))


# In[ ]:


#Query for check, timestamp approach
query2 = """SELECT trip_start_timestamp AS time_stamp, EXTRACT(HOUR from trip_start_timestamp) AS hour
            FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
            WHERE trip_seconds > 0 and trip_miles > 0 and
            EXTRACT(DATE from trip_start_timestamp) > '2017-01-01' and 
            EXTRACT(DATE from trip_start_timestamp) < '2017-02-01' and
            EXTRACT(HOUR from trip_start_timestamp) = 5 and
            EXTRACT(DAY from trip_start_timestamp) = 13
            ORDER BY time_stamp
            """

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=5*10**9)
query2_job = client.query(query2, job_config=safe_config)

query_results = query2_job.to_dataframe()
print(query_results.tail(5))

