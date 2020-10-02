#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.cloud import bigquery
import pandas as pd

# Create a "Client" object
client = bigquery.Client()


# In[ ]:


# Construct a reference to the "hacker_news" dataset
dataset_ref = client.dataset("openaq", project="bigquery-public-data")

# API request - fetch the dataset
dataset = client.get_dataset(dataset_ref)

# List all the tables in the "hacker_news" dataset
tables = list(client.list_tables(dataset))

# Print names of all tables in the dataset (there are four!)
for table in tables:  
    print(table.table_id)

# Construct a reference to the "full" table
table_ref = dataset_ref.table("global_air_quality")

# API request - fetch the table
table = client.get_table(table_ref)

# Print information on all the columns in the "full" table in the "hacker_news" dataset
table.schema
    


# In[ ]:


# Preview the first five lines of the "full" table
 client.list_rows(table, max_results=5).to_dataframe()


# In[ ]:


# Query to select all the items from the "city" column where the "country" column is 'US'
query = """
        SELECT city
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE country = 'US'
        """


# In[ ]:


# Set up the query
query_job = client.query(query)


# In[ ]:


# API request - run the query, and return a pandas DataFrame
us_cities = query_job.to_dataframe()


# In[ ]:


# What five cities have the most measurements?
us_cities.city.value_counts().head()


# In[ ]:


query = """
        SELECT city, country
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE country = 'US'
        """

