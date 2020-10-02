#!/usr/bin/env python
# coding: utf-8

# ## Environment setup

# In[ ]:


# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# explore the data set
hacker_news.head("comments")


# **Which comments on the site generated the most replies**

# In[ ]:


query = """
        SELECT parent, COUNT(id) as count
        FROM `bigquery-public-data.hacker_news.comments`
        GROUP BY parent
        HAVING count(id) > 10
        """

popular_stories = hacker_news.query_to_pandas_safe(query)
popular_stories.head()


# ## SQL Scavenger Hunt: Day 2 Task

# **1. How many stories are there of each type in the Full table of the Hacker News?**

# In[ ]:


# exploring the Full table
hacker_news.head("full")


# In[ ]:


query = """
        SELECT type, COUNT(id) as count
        FROM `bigquery-public-data.hacker_news.full`
        GROUP BY type
        HAVING type = 'story'
        """
stories_count = hacker_news.query_to_pandas_safe(query)
stories_count


# **Answer**: we have 2,845,239 stories in the "full" table.

# **2. How many comments have been deleted?**

# In[ ]:


query = """
        SELECT deleted, COUNT(id) as count
        FROM `bigquery-public-data.hacker_news.comments`
        GROUP BY deleted
        HAVING deleted = TRUE
        """
deleted_comments = hacker_news.query_to_pandas_safe(query)
deleted_comments


# **Answer:** we have 227,736 that are deleted in the "comments" table.

# **3. Extra credit: use a different aggregate function**
# 
# *What is the maximum ranking within replies received for each comment logged in the "comments" table ordered in descending order by comment ID?*

# In[ ]:


query = """
        SELECT parent as Comment_ID, max(ranking) as Max_Ranking
        FROM `bigquery-public-data.hacker_news.comments`
        GROUP BY Comment_ID
        ORDER BY Max_Ranking DESC
        """

MaxRanking = hacker_news.query_to_pandas_safe(query)
MaxRanking.head()

