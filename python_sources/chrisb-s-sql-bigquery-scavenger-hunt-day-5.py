#!/usr/bin/env python
# coding: utf-8

# It's the final day of the SQL scavenger hunt. For today, it's joining. For this exercise, we're using the GitHub Repos dataset: 3TB + dataset of nine tables full of explorational opportunities.
# 
# I've done a fair amount of joining in the past, but using dplyr in *R* having already extracted the tables, so let's see if we can do it in SQL directly.
# 
# As per usual, we import the helper functions and create the helper object:

# In[ ]:


# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")


# That done, we'll take a look at the challenge ([after reading Rachael's helpful tutorial of course!][1])
# 
# The day 5 challenge: **How many commits (recorded in the "sample_commits" table) have been made in repos written in the Python programming language?**
# [1]: https://www.kaggle.com/rtatman/sql-scavenger-hunt-day-5/notebook

# In[ ]:


query = ("""
        SELECT COUNT(commit) AS pythonCommits
        FROM `bigquery-public-data.github_repos.sample_commits` AS sc
        INNER JOIN `bigquery-public-data.github_repos.sample_files` AS sf 
            ON sc.repo_name = sf.repo_name
            WHERE sf.path LIKE '%.py'        
        """)


# Now, I'm not really happy with that query; repo_name is common to both tables, but is that sufficient to join on? I was hoping for something more like a unique id or similar, but maybe this will work. That's why we practise and that's how we learn I suppose...

# In[ ]:


countIsPython = github.query_to_pandas_safe(query, max_gb_scanned=6)


# Well, it runs without an error, always a good start...

# In[ ]:


countIsPython


# And we get a number: **there are 31695737 python commits.**
# 
# I look forward to finding out if that's right or not. If the dataset were a bit smaller, it would be nice just to print out a list and count, possible making a tally chart in pencil as I went, but, hey, that's why we do all this big data analysis stuff, right?
# 
# Anyway, that was fun and just nicely mentally stimulating. The bulk of my SQL up until now had been on saved queries (with the occasional modification), or just doing simple requests, exporting more data than I needed, and doing all the processing in *R*so this should be a much more efficient way to do things. Thanks Rachael!
