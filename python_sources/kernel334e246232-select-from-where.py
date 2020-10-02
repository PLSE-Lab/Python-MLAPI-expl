#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Set your own project id here
PROJECT_ID = 'your-google-cloud-project'
  
from google.cloud import bigquery
client = bigquery.Client(project=PROJECT_ID, location="US")


# # Introduction
# Now that you know how to access and examine a dataset, you're ready to write your first SQL query! As you'll soon see, SQL queries will help you sort through a massive dataset, to retrieve only the information that you need.
# 
# We'll begin by using the keywords SELECT, FROM, and WHERE to get data from specific columns based on conditions you specify.
# 
# For clarity, we'll work with a small imaginary dataset pet_records which contains just one table, called pets.
# 
# 
# 
# # SELECT ... FROM
# The most basic SQL query selects a single column from a single table. To do this,
# 
# specify the column you want after the word SELECT, and then
# specify the table after the word FROM.
# For instance, to select the Name column (from the pets table in the pet_records database in the bigquery-public-data project), our query would appear as follows:
# 
# 
# 
# Note that when writing an SQL query, the argument we pass to FROM is not in single or double quotation marks (' or "). It is in backticks (`).
# 
# # WHERE ...
# BigQuery datasets are large, so you'll usually want to return only the rows meeting specific conditions. You can do this using the WHERE clause.
# 
# The query below returns the entries from the Name column that are in rows where the Animal column has the text 'Cat'.
# 
# 
# 
# # Example: What are all the U.S. cities in the OpenAQ dataset?
# Now that you've got the basics down, let's work through an example with a real dataset. We'll use an OpenAQ dataset about air quality.
# 
# First, we'll set up everything we need to run queries and take a quick peek at what tables are in our database. (Since you learned how to do this in the previous tutorial, we have hidden the code. But if you'd like to take a peek, you need only click on the "Code" button below.)

# In[ ]:


#from google.cloud import bigquery

# Create a "Client" object
#client = bigquery.Client()

# Construct a reference to the "openaq" dataset
dataset_ref = client.dataset("openaq", project="bigquery-public-data")

# API request - fetch the dataset
dataset = client.get_dataset(dataset_ref)

# List all the tables in the "openaq" dataset
tables = list(client.list_tables(dataset))

# Print names of all tables in the dataset (there's only one!)
for table in tables:  
    print(table.table_id)


# The dataset contains only one table, called global_air_quality. We'll fetch the table and take a peek at the first few rows to see what sort of data it contains. (Again, we have hidden the code. To take a peek, click on the "Code" button below.)

# In[ ]:


# Construct a reference to the "global_air_quality" table
table_ref = dataset_ref.table("global_air_quality")

# API request - fetch the table
table = client.get_table(table_ref)

# Preview the first five lines of the "global_air_quality" table
client.list_rows(table, max_results=5).to_dataframe()


# Everything looks good! So, let's put together a query. Say we want to select all the values from the city column that are in rows where the country column is 'US' (for "United States").

# In[ ]:


# Query to select all the items from the "city" column where the "country" column is 'US'
query = """
        SELECT city
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE country = 'US'
        """


# Take the time now to ensure that this query lines up with what you learned above.
# 
# # Submitting the query to the dataset
# We're ready to use this query to get information from the OpenAQ dataset. As in the previous tutorial, the first step is to create a Client object.

# In[ ]:


# Create a "Client" object
#client = bigquery.Client()
client = bigquery.Client(project=PROJECT_ID, location="US")


# We begin by setting up the query with the query() method. We run the method with the default parameters, but this method also allows us to specify more complicated settings that you can read about in the documentation. We'll revisit this later.
# 
# 

# In[ ]:


# Set up the query
#query_job = client.query(query)

