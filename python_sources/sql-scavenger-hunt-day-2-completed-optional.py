#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")


# ## How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?

# **Generating Query:**

# In[ ]:


query1 = """SELECT type, COUNT(id) AS count
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """


# **Casting to a DataFrame:**

# In[ ]:


count_type = hacker_news.query_to_pandas_safe(query1)


# **Final Solution:**

# In[ ]:


count_type


# Let's have a look what we got!

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


sns.barplot(count_type['type'], count_type['count'])


# ## How many comments have been deleted? (If a comment was deleted the "deleted" column in the comments table will have the value "True".)

# **Generating Query:**

# In[ ]:


query2 = """SELECT COUNT(deleted) AS count
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted = True
        """


# **Casting to a DataFrame:**

# In[ ]:


count_deleted = hacker_news.query_to_pandas_safe(query2)


# Let's have a look at our DataFrame.

# In[ ]:


count_deleted


# **Final Solution:**

# In[ ]:


print("Deleted Comments:", count_deleted['count'][0])


# ## **Optional extra credit**: read about [aggregate functions other than COUNT()](https://cloud.google.com/bigquery/docs/reference/standard-sql/functions-and-operators#aggregate-functions) and modify one of the queries you wrote above to use a different aggregate function.

# Alright! Let's get maximum ID assigned to every author from the comments table
# 
# (No practical use, just for fun!)
# 
# **Generating Query:**

# In[ ]:


query3 = """SELECT author, MAX(id) AS MaxID
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY author
        """


# **Casting to a DataFrame:**

# In[ ]:


max_id = hacker_news.query_to_pandas_safe(query3)


# Let's have a look what we got!

# In[ ]:


max_id


# For Output:

# In[ ]:


count_type.to_csv('count_type.csv')
count_deleted.to_csv('deleted.csv')
max_id.to_csv('max.csv')


# In[ ]:




