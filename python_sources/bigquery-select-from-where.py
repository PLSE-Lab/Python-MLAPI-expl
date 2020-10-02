#!/usr/bin/env python
# coding: utf-8

# # **[SQL Home Page](https://www.kaggle.com/learn/intro-to-sql)**
# 
# ---
# 

# # Introduction
# 
# Try writing some **SELECT** statements of your own to explore a large dataset of air pollution measurements.
# 
# Run the cell below to set up the feedback system.

# In[ ]:


# Set up feedback system
from learntools.core import binder
binder.bind(globals())
from learntools.sql.ex2 import *
print("Setup Complete")


# The code cell below fetches the `global_air_quality` table from the `openaq` dataset.  We also preview the first five rows of the table.

# In[ ]:


from google.cloud import bigquery

# Create a "Client" object
client = bigquery.Client()

# Construct a reference to the "openaq" dataset
dataset_ref = client.dataset("openaq", project="bigquery-public-data")

# API request - fetch the dataset
dataset = client.get_dataset(dataset_ref)

# Construct a reference to the "global_air_quality" table
table_ref = dataset_ref.table("global_air_quality")

# API request - fetch the table
table = client.get_table(table_ref)

# Preview the first five lines of the "global_air_quality" table
client.list_rows(table, max_results=5).to_dataframe()


# # Exercises
# 
# ### 1) Units of measurement
# 
# Which countries have reported pollution levels in units of "ppm"?  In the code cell below, set `first_query` to an SQL query that pulls the appropriate entries from the `country` column.
# 
# In case it's useful to see an example query, here's some code from the tutorial:
# 
# ```
# query = """
#         SELECT city
#         FROM `bigquery-public-data.openaq.global_air_quality`
#         WHERE country = 'US'
#         """
# ```

# In[ ]:


# Query to select countries with units of "ppm"
first_query = """
        SELECT country
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE unit = 'ppm'
        """

# Set up the query (cancel the query if it would use too much of 
# your quota, with the limit set to 10 GB)
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
first_query_job = client.query(first_query, job_config=safe_config)

# API request - run the query, and return a pandas DataFrame
first_results = first_query_job.to_dataframe()

# View top few rows of results
print(first_results.head())

# Check your answer
q_1.check()


# For the solution, uncomment the line below.

# In[ ]:


#q_1.solution()


# ### 2) High air quality
# 
# Which pollution levels were reported to be exactly 0?  
# - Set `zero_pollution_query` to select **all columns** of the rows where the `value` column is 0.
# - Set `zero_pollution_results` to a pandas DataFrame containing the query results.

# In[ ]:


# Query to select all columns where pollution levels are exactly 0
zero_pollution_query = """
                       SELECT *
                       FROM `bigquery-public-data.openaq.global_air_quality`
                       WHERE value = 0
                       """

# Set up the query
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
query_job = client.query(zero_pollution_query, job_config=safe_config)

# API request - run the query and return a pandas DataFrame
zero_pollution_results = query_job.to_dataframe()

print(zero_pollution_results.head())

# Check your answer
q_2.check()


# For the solution, uncomment the line below.

# In[ ]:


#q_2.solution()


# That query wasn't too complicated, and it got the data you want. But these **SELECT** queries don't organizing data in a way that answers the most interesting questions. For that, we'll need the **GROUP BY** command. 
# 
# If you know how to use [`groupby()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html) in pandas, this is similar. But BigQuery works quickly with far larger datasets.
# 
# Fortunately, that's next.

# # Keep going
# **[GROUP BY](https://www.kaggle.com/dansbecker/group-by-having-count)** clauses and their extensions give you the power to pull interesting statistics out of data, rather than receiving it in just its raw format.

# ---
# **[SQL Home Page](https://www.kaggle.com/learn/intro-to-sql)**
# 
# 
# 
# 
# 
# *Have questions or comments? Visit the [Learn Discussion forum](https://www.kaggle.com/learn-forum) to chat with other Learners.*
