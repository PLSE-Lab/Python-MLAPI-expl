#!/usr/bin/env python
# coding: utf-8

# **[SQL Home Page](https://www.kaggle.com/learn/intro-to-sql)**
# 
# ---
# 

# # Introduction
# 
# Queries with **GROUP BY** can be powerful. There are many small things that can trip you up (like the order of the clauses), but it will start to feel natural once you've done it a few times. Here, you'll write queries using **GROUP BY** to answer questions from the Hacker News dataset.
# 
# Before you get started, run the following cell to set everything up:

# In[ ]:


# Set up feedback system
from learntools.core import binder
binder.bind(globals())
from learntools.sql.ex3 import *
print("Setup Complete")


# The code cell below fetches the `comments` table from the `hacker_news` dataset.  We also preview the first five rows of the table.

# In[ ]:


from google.cloud import bigquery

# Create a "Client" object
client = bigquery.Client()

# Construct a reference to the "hacker_news" dataset
dataset_ref = client.dataset("hacker_news", project="bigquery-public-data")

# API request - fetch the dataset
dataset = client.get_dataset(dataset_ref)

# Construct a reference to the "comments" table
table_ref = dataset_ref.table("comments")

# API request - fetch the table
table = client.get_table(table_ref)

# Preview the first five lines of the "comments" table
client.list_rows(table, max_results=5).to_dataframe()


# # Exercises
# 
# ### 1) Prolific commenters
# 
# Hacker News would like to send awards to everyone who has written more than 10,000 posts. Write a query that returns all authors with more than 10,000 posts as well as their post counts. Call the column with post counts `NumPosts`.
# 
# In case sample query is helpful, here is a query you saw in the tutorial to answer a similar question:
# ```
# query = """
#         SELECT parent, COUNT(1) AS NumPosts
#         FROM `bigquery-public-data.hacker_news.comments`
#         GROUP BY parent
#         HAVING COUNT(1) > 10
#         """
# ```

# In[ ]:


# Query to select prolific commenters and post counts
prolific_commenters_query =  """
                            SELECT author, COUNT(1) AS NumPosts
                            FROM `bigquery-public-data.hacker_news.comments`
                            GROUP BY author
                            HAVING COUNT(1) > 10000
                            """
# Your code goes here

# Set up the query (cancel the query if it would use too much of 
# your quota, with the limit set to 1 GB)
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
query_job = client.query(prolific_commenters_query, job_config=safe_config)

# API request - run the query, and return a pandas DataFrame
prolific_commenters = query_job.to_dataframe()

# View top few rows of results
print(prolific_commenters.head())

# Check your answer
q_1.check()


# For the solution, uncomment the line below.

# In[ ]:


#q_1.solution()


# ### 2) Deleted comments
# 
# How many comments have been deleted? (If a comment was deleted, the `deleted` column in the comments table will have the value `True`.)

# In[ ]:


# Write your query here and figure out the answer

# Query to select prolific commenters and post counts
prolific_commenters_query =  """
                            SELECT deleted, COUNT(id) as NumDel
                            FROM `bigquery-public-data.hacker_news.comments`
                            GROUP BY deleted
                            """
# Your code goes here

# Set up the query (cancel the query if it would use too much of 
# your quota, with the limit set to 1 GB)
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
query_job = client.query(prolific_commenters_query, job_config=safe_config)

# API request - run the query, and return a pandas DataFrame
prolific_commenters = query_job.to_dataframe()

# View top few rows of results
print(prolific_commenters.head())


# In[ ]:


num_deleted_posts = 227736 # Put your answer here

q_2.check()


# For the solution, uncomment the line below.

# In[ ]:


#q_2.solution()


# # Keep Going
# **[Click here](https://www.kaggle.com/dansbecker/order-by)** to move on and learn about the **ORDER BY** clause.

# ---
# **[SQL Home Page](https://www.kaggle.com/learn/intro-to-sql)**
# 
# 
# 
# 
# 
# *Have questions or comments? Visit the [Learn Discussion forum](https://www.kaggle.com/learn-forum) to chat with other Learners.*

# In[ ]:




