#!/usr/bin/env python
# coding: utf-8

# 

# 

# ## Example: Which Hacker News comments generated the most discussion?
# ___
# 
# Now we're ready to work through an example on a real dataset. Today, we're going to be using the Hacker News dataset, which contains information on stories & comments from the Hacker News social networking site. I want to know which comments on the site generated the most replies.
# 
# First, just like yesterday, we need to get our environment set up. I already know that I want the "comments" table, so I'm going to look at the first couple of rows of that to get started.

# In[ ]:


# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")


# By looking at the documentation, I learned that the "parent" column has information on the comment that each comment was a reply to and the "id" column has the unique id used to identify each comment. So I can group by the "parent" column and count the "id" column in order to figure out the number of comments that were made as responses to a specific comment. 
# 
# Because I'm more interested in popular comments than unpopular comments, I'm also only going to return the groups that have more than ten id's in them. In other words, I'm only going to look at comments that had more than ten comment replies to them.

# In[ ]:


# query to pass to 
query = """SELECT parent, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY parent
            HAVING COUNT(id) > 10
        """


# Now that our query is ready, let's run it (safely!) and store the results in a dataframe: 

# In[ ]:


# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
popular_stories = hacker_news.query_to_pandas_safe(query)


# And, just like yesterday, we have a dataframe we can treat like any other data frame:

# In[ ]:


popular_stories.head()


# Looks good! From here I could do whatever further analysis or visualization I'd like. 
# 
# > **Why is the column with the COUNT() data called f0_**? It's called this because COUNT() is the first (and in our case, only) aggregate function we used in this query. If we'd used a second one, it would be called "f1\_", the third would be called "f2\_", and so on. We'll learn how to name the output of aggregate functions later this week.
# 
# And that should be all you need to get started writing your own kernels with GROUP BY... WHERE and COUNT!

# # Scavenger hunt
# ___
# 
# Now it's your turn! Here's the questions I would like you to get the data to answer:
# 
# * How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?
# * How many comments have been deleted? (If a comment was deleted the "deleted" column in the comments table will have the value "True".)
# * **Optional extra credit**: read about [aggregate functions other than COUNT()](https://cloud.google.com/bigquery/docs/reference/standard-sql/functions-and-operators#aggregate-functions) and modify one of the queries you wrote above to use a different aggregate function.
# 
# In order to answer these questions, you can fork this notebook by hitting the blue "Fork Notebook" at the very top of this page (you may have to scroll up). "Forking" something is making a copy of it that you can edit on your own without changing the original.

# In[ ]:


# Your code goes here :)

# check again what columns the full table consists of
hacker_news.head("full")


# **How many stories of each type are there in the full table?**

# In[ ]:


# the query 

query2 = """SELECT type, COUNT(id)
           FROM `bigquery-public-data.hacker_news.full`
           GROUP BY type
        """


# In[ ]:


# saving the result of the query to a Pandas dataset
stories_by_type = hacker_news.query_to_pandas_safe(query2)

# displaying the result
print(stories_by_type)


# **How many comments have been deleted?**

# In[ ]:


# taking a look at the comments table
hacker_news.head("comments")


# In[ ]:


# the query
query3 = """SELECT deleted, COUNT(id)
           FROM `bigquery-public-data.hacker_news.comments`
           GROUP BY deleted
           HAVING deleted = TRUE
        """


# In[ ]:


# saving the query as a Pandas dataframe
deleted_comments = hacker_news.query_to_pandas_safe(query3)

# returning the result
print(deleted_comments)


# **What are the highest scores per type?**

# In[ ]:


# we use the full table again for the query
query4 = """SELECT type, MAX(score)
           FROM `bigquery-public-data.hacker_news.full`
           GROUP BY type
           ORDER BY MAX(score)
        """


# In[ ]:


# saving the result of the query in a Pandas dataframe
highest_score = hacker_news.query_to_pandas_safe(query4)

# return the highest scores per content type
print(highest_score)


# 

# Please feel free to ask any questions you have in this notebook or in the [Q&A forums](https://www.kaggle.com/questions-and-answers)! 
# 
# Also, if you want to share or get comments on your kernel, remember you need to make it public first! You can change the visibility of your kernel under the "Settings" tab, on the right half of your screen.
