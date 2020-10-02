#!/usr/bin/env python
# coding: utf-8

# SQL is the primary programming language used with databases, and it is an important skill for any data scientist.
# 
# BigQuery is a database that lets you use SQL to work with very large datasets.
# 
# 1.   You can **import bq_helper** in your kernel with the command.
# 
# 2 . Create an object  using bq_helper.BigQueryHelper(......)

# In[ ]:


import bq_helper
import pandas as pd
# create a helper object for our bigquery dataset
chicago_crime = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "chicago_crime")


# 

# ## Structure of your Dataset
# Schema: A description of how data is organized within a dataset.
# 
# BigQuery datasets can be very large, and there are some restrictions on how much data you can access.
# 
# #### NOTE : Each Kaggle user can scan 5TB every 30 days for free. If you go over your quota you're going to have to wait for it to reset.
# 

# In[ ]:


chicago_crime.list_tables()


# ## Information about table
# we're looking at the information on the table called "crime". Note that other databases will have different table names, so you will not always use "crime."

# In[ ]:


chicago_crime.table_schema("crime")


# ## Get the First couple of rows
# 

# In[ ]:


chicago_crime.head("crime")


# ## Get the selected columns by number of rows and start with specified index.

# In[ ]:


chicago_crime.head("crime",selected_columns=["unique_key","case_number","date"],num_rows=10,start_index=2)


# ## Run the Query
# 
# <h4>BigQueryHelper.query_to_pandas(query):</h4> This method takes a query and returns a Pandas dataframe.
# 
# <h4>BigQueryHelper.query_to_pandas_safe(query, max_gb_scanned=1):</h4> This method takes a query and returns a Pandas dataframe only if the size of the query is less than the upperSizeLimit (1 gigabyte by default).
# 
# You can do this with the  <h4>BigQueryHelper.estimate_query_size()</h4> method.
# One way to help avoid this is to estimate how big your query will be before you actually execute it.

# In[ ]:


q = """select case_number from `bigquery-public-data.chicago_crime.crime` where district = 4 """
chicago_crime.estimate_query_size(q)
chicago_crime.query_to_pandas(q)


# In[ ]:




