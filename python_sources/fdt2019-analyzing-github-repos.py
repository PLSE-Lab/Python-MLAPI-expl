#!/usr/bin/env python
# coding: utf-8

# # PLEASE NOTE
# ### This notebook starts as a simple fork of https://www.kaggle.com/poonaml/analyzing-3-million-github-repos-using-bigquery

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# 
# Let's first setup our environment to run BigQueries in Kaggle Kernels.
# 
# ### Importing Kaggle's bq_helper package pointing to the Github Repos public dataset

# In[ ]:


import bq_helper
github_repos = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "github_repos")


# ### Checking the size of a query each time before we run it
# Our Dataset is more than 3TBs so we can easily cross the allowed limits by running few queries. We should always estimate how much data we need to scan for executing this query by **BigQueryHelper.estimate_query_size()** method. We start with a simple query that gives us information on all commits and their authors (pre-aggregated on a user and per month basis with BigQuery-SQL).
# 

# In[ ]:


query = """SELECT EXTRACT(YEAR FROM TIMESTAMP_SECONDS(author.time_sec)) AS year,
                  EXTRACT(MONTH FROM TIMESTAMP_SECONDS(author.time_sec)) AS month,
                  author.name AS author_name, COUNT(*) AS count
            FROM `bigquery-public-data.github_repos.commits`
            GROUP BY year, month, author_name
        """
github_repos.estimate_query_size(query)


# This is the estimated amount of data in GB to be scanned during execution of this query. By default any query scanning more than 1GB of data will get cancelled by kaggle kernel environment.
# 
# ### Running a query
# 
# There are 2 ways to do this:
# 
# 1. BigQueryHelper.query_to_pandas(query): This method takes a query and returns a Pandas dataframe.
# 1. BigQueryHelper.query_to_pandas_safe(query, max_gb_scanned=1): This method takes a query and returns a Pandas dataframe only if the size of the query is less than the given upperSizeLimit (1 gigabyte by default).
# 
# Here's our example of a query with a specified upper limit.

# In[ ]:


commits = github_repos.query_to_pandas_safe(query, max_gb_scanned=5)
commits.head()


# ### Saving the query result as a file (*.csv)
# Here, we simply can use the method to_csv for Pandas dataframes. (Use pd.read_csv('output.csv') to read the dataframe in again!)

# In[ ]:


commits.to_csv('output.csv', index=False)
#commits = pd.read_csv('output.csv')


# ### What are the Numbers of Monthly Commits in Public Github-Repositories
# 
# Using our dataframe we can now compute some further aggregations with Pandas methods. We can e.g. compute the number of commits for all months by using Panda's groupby- and sum-methods. Let's first add a new column for the number of each month (starting with the number 1 for January 1970)!

# In[ ]:


commits['month_nr'] = (commits['year']-1970)*12 + commits['month']
monthly_commits = commits[['month_nr', 'count']].groupby(['month_nr']).sum()
monthly_commits.head()


# Obviously, some commits have a surprisingly "early" date (while others - and we will show this in just a minute - have dates lying in the future!). Let's visualize our results in a simple plot with matplotlib:

# In[ ]:


plt.figure(figsize=(12,6))
plt.plot(monthly_commits)
plt.title("Monthly Commits in Public Github-Repos since 1970")


# ### What are the Numbers of Active Github-Users in Public Repositories
# 
# We can use our dataframe and compute the number of commits again within Pandas:

# In[ ]:


yearly_commits_per_user = commits[['year', 'author_name', 'count']].groupby(['year', 'author_name']).sum()
yearly_commits_per_user.head()


# Again, we see some strange dates (and also strange author names!). However, we can simply count the active users for each year and plot the result to have a simple visualization:

# In[ ]:


yearly_active_users = yearly_commits_per_user.groupby('year').count()
yearly_active_users.head()

plt.figure(figsize=(12,6))
plt.plot(yearly_active_users)
plt.title("Yearly Active Users in Public Github-Repos")


# In[ ]:




