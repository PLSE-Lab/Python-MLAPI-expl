#!/usr/bin/env python
# coding: utf-8

# This is my second bigquery dataset after openAQ dataset as a part of SQL Scavenger hunt.Join me in learning the 
# super awesome series of tutorials and commands.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper


# In[ ]:


hacker=bq_helper.BigQueryHelper(active_project='bigquery-public-data',dataset_name='hacker_news')


# In[ ]:


hacker.list_tables()


# Question 1 :How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?

# In[ ]:


hacker.table_schema('full')


# In[ ]:


hacker.head('full')


# In[ ]:


query=""" SELECT count(*) as count_story,type from `bigquery-public-data.hacker_news.full` GROUP BY type HAVING type= 'story' """


# In[ ]:


hacker.estimate_query_size(query)


# In[ ]:


story=hacker.query_to_pandas_safe(query)


# In[ ]:


story


# From the output we find that the 28M items are present having type as 'story'.Lets go to the second question.

# In[ ]:


hacker.table_schema('comments')


# In[ ]:


hacker.head('comments')


# 

# In[ ]:


query=""" SELECT count(*) as count_deleted,deleted from `bigquery-public-data.hacker_news.comments` group by deleted having deleted is True """ 


# In[ ]:


hacker.estimate_query_size(query)


# In[ ]:


deleted=hacker.query_to_pandas_safe(query)


# In[ ]:


deleted


# We find that there are 2L items that have deleted ='True'

# # Optional Credit 

# In[ ]:


hacker.table_schema('stories')


# In[ ]:


hacker.head('stories')


# In[ ]:


query="""SELECT count(id) as id_count,score from `bigquery-public-data.hacker_news.stories` group by score having score = 1 """


# In[ ]:


hacker.estimate_query_size(query)


# In[ ]:


stories=hacker.query_to_pandas_safe(query)


# In[ ]:


stories


# Thus we find that there are 9L items having score = 1. 

# In[ ]:


query = """ SELECT sum(score) as sum_score,deleted from `bigquery-public-data.hacker_news.stories` group by deleted """


# In[ ]:


hacker.estimate_query_size(query)


# In[ ]:


sum_story=hacker.query_to_pandas_safe(query)


# In[ ]:


sum_story


# From the output we understand that the total score has been provided for contents that have not been deleted.

# This brings us to the end of SQL Scavenger Hunt-Day 2 ...Looking forward for some exciting things in the coming days ...
# **If you like my kernel,pls upvote and encourage**
