#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import our bq_helper package
import bq_helper 


# In[ ]:


# create a helper object for our bigquery dataset
github_repos = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "github_repos")


# In[ ]:


# print a list of all the tables in the github_repos dataset
github_repos.list_tables()


# In[ ]:


# print information on all the columns in the "commits" table
# in the github_repos dataset
github_repos.table_schema("commits")


# In[ ]:


github_repos.head("commits")


# In[ ]:


github_repos.head("commits", selected_columns="author", num_rows=10)


# In[ ]:


# this query looks in the full table in the github_repos
# dataset, then gets the author column from every row where 
# the commit column has "e9f07afffde8c12c858d8762890a17b0b479e2e1" in it.
query = """SELECT author
            FROM `bigquery-public-data.github_repos.commits`
            WHERE commit = "e9f07afffde8c12c858d8762890a17b0b479e2e1" """

# check how big this query will be
github_repos.estimate_query_size(query)


# - *BigQueryHelper.query_to_pandas(query): *This method takes a query and returns a Pandas dataframe.
# - *BigQueryHelper.query_to_pandas_safe(query, max_gb_scanned=1):* This method takes a query and returns a Pandas dataframe only if the size of the query is less than the upperSizeLimit (1 gigabyte by default).

# In[ ]:


# only run this query if it's less than 100 MB
github_repos.query_to_pandas_safe(query, max_gb_scanned=0.1)

