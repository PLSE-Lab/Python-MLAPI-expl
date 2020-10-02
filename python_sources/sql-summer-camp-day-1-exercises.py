#!/usr/bin/env python
# coding: utf-8

# **[SQL Micro-Course Home Page](https://www.kaggle.com/learn/intro-to-sql)**
# 
# ---
# 

# # Introduction
# 
# The first test of your new data exploration skills uses data describing crime in the city of Chicago.
# 
# Before you get started, run the following cell. It sets up the automated feedback system to review your answers.

# In[ ]:


# Set up feedack system
from learntools.core import binder
binder.bind(globals())
from learntools.sql.ex1 import *
print("Setup Complete")


# Use the next code cell to fetch the dataset.

# In[ ]:


from google.cloud import bigquery

# Create a "Client" object
client = bigquery.Client()

# Construct a reference to the "chicago_crime" dataset
dataset_ref = client.dataset("chicago_crime", project="bigquery-public-data")

# API request - fetch the dataset
dataset = client.get_dataset(dataset_ref)


# # Exercises
# 
# ### 1) Count tables in the dataset
# 
# How many tables are in the Chicago Crime dataset?

# In[ ]:


# Write the code you need here to figure out the answer
tables = list(client.list_tables(dataset))
print(f'no. of tables : {len(tables)}')
print(tables[0].table_id)


# In[ ]:


num_tables = 1  # Store the answer as num_tables and then run this cell

q_1.check()


# For a hint or the solution, uncomment the appropriate line below.

# In[ ]:


#q_1.hint()
#q_1.solution()


# ### 2) Explore the table schema
# 
# How many columns in the `crime` table have `TIMESTAMP` data?

# In[ ]:


# Write the code to figure out the answer
table_ref = dataset_ref.table("crime")
table_data = client.get_table(table_ref)
print(*(i for i in table_data.schema if 'TIMESTAMP'==i.field_type),sep='\n')


# In[ ]:


num_timestamp_fields = 2# Put your answer here

q_2.check()


# For a hint or the solution, uncomment the appropriate line below.

# In[ ]:


#q_2.hint()
#q_2.solution()


# ### 3) Create a crime map
# 
# If you wanted to create a map with a dot at the location of each crime, what are the names of the two fields you likely need to pull out of the `crime` table to plot the crimes on a map?

# In[ ]:


# Write the code here to explore the data so you can find the answer
table_data.schema


# In[ ]:


fields_for_plotting = ['x_coordinate', 'y_coordinate'] # Put your answers here

q_3.check()


# For a hint or the solution, uncomment the appropriate line below.

# In[ ]:


#q_3.hint()
#q_3.solution()


# Thinking about the question above, there are a few columns that appear to have geographic data. Look at a few values (with the `list_rows()` command) to see if you can determine their relationship.  Two columns will still be hard to interpret. But it should be obvious how the `location` column relates to `latitude` and `longitude`.

# In[ ]:


# Scratch space for your code


# # Bunk Time
# Here as part of the SQL Summer Camp?
# 
# Time to head back to the bunk. You'll get another email tomorrow to start your next camp day.

# ---
# **[SQL Micro-Course Home Page](https://www.kaggle.com/learn/intro-to-sql)**
# 
# 
