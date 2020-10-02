#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.cloud import bigquery

# Create a "Client" object
client = bigquery.Client()


# In[ ]:


# Construct a reference to the "hacker_news" dataset
dataset_ref = client.dataset("hacker_news", project = "bigquery-public-data")
# API request - fetch the dataset
dataset = client.get_dataset(dataset_ref)


# In[ ]:


# List all the tables in the "hacker_news" dataset
tables = list(client.list_tables(dataset))

# Print names of all tables in the dataset (there are four!)
for table in tables:
    print(table.table_id)


# In[ ]:


# Construct a reference to the "full" table
table_ref = dataset_ref.table("full")
# API request - fetch the table
table = client.get_table(table_ref)


# In[ ]:


#Prints schema of the table

table.schema


# In[ ]:


client.list_rows(table, max_results=5).to_dataframe()

