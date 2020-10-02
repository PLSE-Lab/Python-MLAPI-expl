#!/usr/bin/env python
# coding: utf-8

# **[Advanced SQL Home Page](https://www.kaggle.com/learn/advanced-sql)**
# 
# ---
# 

# # Introduction
# 
# Here, you'll use window functions to answer questions about the [Chicago Taxi Trips](https://www.kaggle.com/chicago/chicago-taxi-trips-bq) dataset.
# 
# Before you get started, run the code cell below to set everything up.

# In[ ]:


# Set up feedback system
from learntools.core import binder
binder.bind(globals())
from learntools.sql_advanced.ex2 import *
print("Setup Complete")


# The following code cell fetches the `taxi_trips` table from the `chicago_taxi_trips` dataset. We also preview the first five rows of the table.  You'll use the table to answer the questions below.

# In[ ]:


from google.cloud import bigquery

# Create a "Client" object
client = bigquery.Client()

# Construct a reference to the "chicago_taxi_trips" dataset
dataset_ref = client.dataset("chicago_taxi_trips", project="bigquery-public-data")

# API request - fetch the dataset
dataset = client.get_dataset(dataset_ref)

# Construct a reference to the "taxi_trips" table
table_ref = dataset_ref.table("taxi_trips")

# API request - fetch the table
table = client.get_table(table_ref)

# Preview the first five lines of the table
client.list_rows(table, max_results=5).to_dataframe()


# # Exercises
# 
# ### 1) How can you predict the demand for taxis?
# 
# Say you work for a taxi company, and you're interested in predicting the demand for taxis.  Towards this goal, you'd like to create a plot that shows a rolling average of the daily number of taxi trips.  Amend the (partial) query below to return a DataFrame with two columns:
# - `trip_date` - contains one entry for each date from January 1, 2016, to December 31, 2017.
# - `avg_num_trips` - shows the average number of daily trips, calculated over a window including the value for the current date, along with the values for the preceding 15 days and the following 15 days, as long as the days fit within the two-year time frame.  For instance, when calculating the value in this column for January 5, 2016, the window will include the number of trips for the preceding 4 days, the current date, and the following 15 days.
# 
# This query is partially completed for you, and you need only write the part that calculates the `avg_num_trips` column.  Note that this query uses a common table expression (CTE); if you need to review how to use CTEs, you're encouraged to check out [this tutorial](https://www.kaggle.com/dansbecker/as-with) in the [Intro to SQL](https://www.kaggle.com/learn/intro-to-sql) micro-course.

# In[ ]:


# Fill in the blank below
avg_num_trips_query = """
                      WITH trips_by_day AS
                      (
                      SELECT DATE(trip_start_timestamp) AS trip_date,
                          COUNT(*) as num_trips
                      FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
                      WHERE trip_start_timestamp >= '2016-01-01' AND trip_start_timestamp < '2018-01-01'
                      GROUP BY trip_date
                      ORDER BY trip_date
                      )
                      SELECT trip_date,
                          AVG(num_trips)
                          OVER (
                               ORDER BY trip_date
                               ROWS BETWEEN 15 PRECEDING AND 15 FOLLOWING
                               ) AS avg_num_trips
                      FROM trips_by_day
                      """

# Check your answer
q_1.check()


# In[ ]:


# Lines below will give you a hint or solution code
#q_1.hint()
#q_1.solution()


# ### 2) Can you separate and order trips by community area?
# 
# The query below returns a DataFrame with three columns from the table: `pickup_community_area`, `trip_start_timestamp`, and `trip_end_timestamp`.  
# 
# Amend the query to return an additional column called `trip_number` which shows the order in which the trips were taken from their respective community areas.  So, the first trip of the day originating from community area 1 should receive a value of 1; the second trip of the day from the same area should receive a value of 2.  Likewise, the first trip of the day from community area 2 should receive a value of 1, and so on.
# 
# Note that there are many numbering functions that can be used to solve this problem (depending on how you want to deal with trips that started at the same time from the same community area); to answer this question, please use the **RANK()** function.

# In[ ]:


# Amend the query below
trip_number_query = """
                    SELECT pickup_community_area,
                        trip_start_timestamp,
                        trip_end_timestamp,
                        RANK()
                        OVER (
                            PARTITION BY pickup_community_area
                            ORDER BY trip_start_timestamp
                        ) AS trip_number
                    FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
                    WHERE DATE(trip_start_timestamp) = '2017-05-01'
                    
                    """

# Check your answer
q_2.check()


# In[ ]:


# Lines below will give you a hint or solution code
#q_2.hint()
#q_2.solution()


# ### 3) How much time elapses between trips?
# 
# The (partial) query in the code cell below shows, for each trip in the selected time frame, the corresponding `taxi_id`, `trip_start_timestamp`, and `trip_end_timestamp`. 
# 
# Your task in this exercise is to edit the query to include an additional `prev_break` column that shows the length of the break (in minutes) that the driver had before each trip started (this corresponds to the time between `trip_start_timestamp` of the current trip and `trip_end_timestamp` of the previous trip).  Partition the calculation by `taxi_id`, and order the results within each partition by `trip_start_timestamp`.
# 
# Some sample results are shown below, where all rows correspond to the same driver (or `taxi_id`).  Take the time now to make sure that the values in the `prev_break` column make sense to you!
# 
# ![first_commands](https://i.imgur.com/qjvQzg8.png)
# 
# Note that the first trip of the day for each driver should have a value of **NaN** (not a number) in the `prev_break` column.

# In[ ]:


# Fill in the blanks below
break_time_query = """
                   SELECT taxi_id,
                       trip_start_timestamp,
                       trip_end_timestamp,
                       TIMESTAMP_DIFF(
                           trip_start_timestamp, 
                           LAG(trip_end_timestamp)
                               OVER (
                                    PARTITION BY taxi_id
                                    ORDER BY trip_start_timestamp),
                           MINUTE) as prev_break
                   FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
                   WHERE DATE(trip_start_timestamp) = '2017-05-01' 
                   """

# Check your answer
q_3.check()


# In[ ]:


# Lines below will give you a hint or solution code
# q_3.hint()
#q_3.solution()


# # Keep going
# 
# Move on to learn how to query **[nested and repeated data](https://www.kaggle.com/alexisbcook/nested-and-repeated-data)**.

# ---
# **[Advanced SQL Home Page](https://www.kaggle.com/learn/advanced-sql)**
# 
# 
# 
# 
# 
# *Have questions or comments? Visit the [Learn Discussion forum](https://www.kaggle.com/learn-forum) to chat with other Learners.*
