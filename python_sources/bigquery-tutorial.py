#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
hacker_news = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "hacker_news")


# In[ ]:


hacker_news.list_tables()


# In[ ]:


hacker_news.table_schema("full")


# In[ ]:


hacker_news.head("full")


# In[ ]:


hacker_news.head("full", selected_columns="by", num_rows=10)


# In[ ]:


query = """SELECT score
            FROM `bigquery-public-data.hacker_news.full`
            WHERE type = "job" """

# check how big this query will be
hacker_news.estimate_query_size(query)


# In[ ]:


job_post_scores = hacker_news.query_to_pandas_safe(query)


# In[ ]:


job_post_scores.score.mean()


# In[ ]:


job_post_scores.to_csv("job_post_scores.csv")

