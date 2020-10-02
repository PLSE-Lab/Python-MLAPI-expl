#!/usr/bin/env python
# coding: utf-8

# **[SQL Home Page](https://www.kaggle.com/learn/intro-to-sql)**
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


tables = list(client.list_tables(dataset))
for table in tables:
    print(table.table_id)
    
# Construct a reference to the "full" table
table_ref = dataset_ref.table("crime")

# API request - fetch the table
table = client.get_table(table_ref)


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


schema = table.schema

for field in table.schema:
    print(field.name)


# In[ ]:


num_timestamp_fields = 2 # Put your answer here

q_2.check()


# For a hint or the solution, uncomment the appropriate line below.

# In[ ]:


#q_2.hint()
#q_2.solution()


# ### 3) Create a crime map
# 
# If you wanted to create a map with a dot at the location of each crime, what are the names of the two fields you likely need to pull out of the `crime` table to plot the crimes on a map?

# In[ ]:


# Preview the first five lines of the "full" table
print(client.list_rows(table, max_results=5).to_dataframe())


# In[ ]:


fields_for_plotting = ''

for field in table.schema:
    if (field.field_type == 'FLOAT'):
        fields_for_plotting = fields_for_plotting + field.name + "\r\n"

print(fields_for_plotting) # Put your answers here

#q_3.check()


# For a hint or the solution, uncomment the appropriate line below.

# In[ ]:


#q_3.hint()
#q_3.solution()


# Thinking about the question above, there are a few columns that appear to have geographic data. Look at a few values (with the `list_rows()` command) to see if you can determine their relationship.  Two columns will still be hard to interpret. But it should be obvious how the `location` column relates to `latitude` and `longitude`.

# In[ ]:


# Scratch space for your code


# # Keep going
# 
# You've looked at the schema, but you haven't yet done anything exciting with the data itself. Things get more interesting when you get to the data, so keep going to **[write your first SQL query](https://www.kaggle.com/dansbecker/select-from-where).**

# ---
# **[SQL Home Page](https://www.kaggle.com/learn/intro-to-sql)**
# 
# 
# 
# 
# 
# *Have questions or comments? Visit the [Learn Discussion forum](https://www.kaggle.com/learn-forum) to chat with other Learners.*
